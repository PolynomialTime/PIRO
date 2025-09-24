import time
import datetime
import numpy as np
import gym
import torch
import copy
import torch.nn.functional as F
from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common import policies
from stable_baselines3.common.evaluation import evaluate_policy
from torch.optim import Adam
from oadam import OAdam
from gym.envs.atari import AtariEnv
from gym.wrappers import FlattenObservation
import torch.utils.tensorboard as tb

from nn_utils import linear_schedule, gradient_penalty
from buffer import ReplayBuffer, ReplayDiscreteBuffer, ReplayAtariBuffer, QReplayBuffer
from arch import Discriminator
from gym_wrappers import ResetWrapper, RewardWrapper, RewardDiscreteWrapper, TremblingHandWrapper, GoalWrapper, AntMazeResetWrapper
from TD3_BC import TD3_BC
import d4rl
import os

class FILTER():
    def __init__(self, env):
        self.env = env

    def sample(
        self,
        env,
        policy,
        trajs,
        no_regret
        ):
        # rollout trajectories using a policy and add to replay buffer
        S_curr = []
        A_curr = []
        total_trajs = 0
        alpha = env.alpha
        env.alpha = 0
        s = 0
        while total_trajs < trajs:
            obs = env.reset()
            done = False
            while not done:
                S_curr.append(obs)
                act = policy.predict(obs)[0]
                A_curr.append(act)
                obs, _, done, _ = env.step(act)
                s += 1
                if done:
                    total_trajs += 1
                    break
        env.alpha = alpha
        if no_regret:
            self.replay_buffer.add(S_curr, A_curr)
        return torch.Tensor(S_curr), torch.Tensor(A_curr), s

    def train(self,
              expert_sa_pairs,
              expert_obs,
              expert_acts,
              a1, a2, a3,
              n_seed=0,
              n_exp=25,
              alpha=0.5,
              no_regret=False,
              expert_policy: policies = None,
              device="cuda:2"):

        def expert_kl(expert, learn, obs, acts, kl_device):
            assert learn is not None
            assert isinstance(expert.policy, policies.ActorCriticPolicy)
            assert isinstance(learn.policy, policies.ActorCriticPolicy)
            obs = np.concatenate(obs, axis=0)
            acts = np.concatenate(acts, axis=0)
            obs_th = torch.as_tensor(obs, device=kl_device)
            acts_th = torch.as_tensor(acts, device=kl_device)
            if acts_th.ndim == 2 and acts_th.shape[1] == 1:
                acts_th = acts_th.squeeze(dim=1)

            expert.policy.to(kl_device)
            learn.policy.to(kl_device)
            input_values, input_log_prob, input_entropy = learn.policy.evaluate_actions(obs_th, acts_th)
            target_values, target_log_prob, target_entropy = expert.policy.evaluate_actions(obs_th, acts_th)

            kl_div = torch.mean(torch.dot(torch.exp(target_log_prob), target_log_prob - input_log_prob))

            return abs(float(kl_div))

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        current_dir = os.path.dirname(__file__)
        log_dir = os.path.join(current_dir, 'logs', f'{self.env}/mm_real/{current_time}')
        global writer
        writer = tb.SummaryWriter(log_dir=log_dir, flush_secs=1)

        if self.env == 'CartPoleBulletEnv-v1' or self.env == 'CartPole-v1':
            num_action = 2
        elif self.env == 'Acrobot-v1':
            num_action = 3
        elif self.env == 'PongNoFrameskip-v4' or self.env == 'QbertNoFrameskip-v4' or self.env == 'SpaceInvadersNoFrameskip-v4':
            num_action = 6
        elif self.env == 'LunarLander-v2' or self.env == 'FrozenLake-v1' or self.env == 'BreakoutNoFrameskip-v4':
            num_action = 4
        elif self.env == 'Meerkat-v3':
            num_action = 16
        else:
            num_action = 0

        if self.env == 'CartPoleBulletEnv-v1' or self.env == 'Acrobot-v1' or self.env == 'LunarLander-v2' or self.env == 'CartPole-v1' or self.env == 'BreakoutNoFrameskip-v4' \
                or self.env == 'PongNoFrameskip-v4' or self.env == 'QbertNoFrameskip-v4' or self.env == 'FrozenLake-v1' or self.env == 'Meerkat-v3'\
                or self.env == 'SpaceInvadersNoFrameskip-v4':
            # 假设前4列是状态，最后1列是动作
            # 1) 拆分
            states = expert_sa_pairs[:, :-1]  # 形状 (N, 4)
            actions = expert_sa_pairs[:, -1].long()  # 形状 (N,); 先转成 long() 保证是整型 0 或 1

            # 2) one-hot 动作 => (N,2)，0=>[1,0], 1=>[0,1]
            actions_oh = F.one_hot(actions, num_classes=num_action).float()

            if 'FrozenLake' in self.env and states.shape[1] == 1:
                n_state = 16   
                states_oh = F.one_hot(states.squeeze(1).long(), num_classes=n_state).float()
            else:
                states_oh = states.float()

            # 3) 拼接 => (N, 6)
            expert_sa_pairs = torch.cat([states_oh, actions_oh], dim=1)

        expert_sa_pairs = expert_sa_pairs.to(device)

        if alpha <= 1e-4:
            name = 'mm_icml'
        elif no_regret:
            name = 'filter_nr_icml'
        else:
            name = 'filter_br_icml'

        learn_rate = 1e-2
        batch_size = 64
        f_steps = 1
        pi_steps = 1000
        num_traj_sample = 1
        outer_steps = 500
        mean_rewards = []
        std_rewards = []
        env_steps = []
        log_interval = 1

        # -----------------------------------------------------
        # 1. 创建环境，并根据需求进行包装
        cur_env = gym.make(self.env)
        if isinstance(cur_env.observation_space, gym.spaces.Discrete):
            cur_env = FlattenObservation(cur_env)
        if 'maze' in self.env:
            cur_env = AntMazeResetWrapper(GoalWrapper(cur_env), a1, a2, a3)
        else:
            cur_env = ResetWrapper(cur_env, a1, a2, a3, expert_obs, expert_acts)

        cur_env.alpha = alpha

        f_net = Discriminator(cur_env).to(device)
        f_opt = OAdam(f_net.parameters(), lr=learn_rate)

        if 'CartPole' in self.env or 'Acrobot' in self.env or 'Lunar' in self.env or 'Human' in self.env or 'Frozen' in self.env \
                or 'Bipedal' in self.env or 'Qbert' in self.env or 'Pong' in self.env or 'Ant' in self.env or 'Meerkat' in self.env\
                or 'Breakout' in self.env or 'Space' in self.env or 'Walker' in self.env or 'Hopper' in self.env or 'Cheetah' in self.env:
            cur_env = RewardDiscreteWrapper(cur_env, f_net)
        else:
            cur_env = RewardWrapper(cur_env, f_net)

        # -----------------------------------------------------
        # 2. 根据环境类型选择不同的算法
        if 'maze' in self.env:
            # ---------------------- Maze 环境的逻辑 ----------------------
            cur_env = TremblingHandWrapper(cur_env, p_tremble=0)
            eval_env = TremblingHandWrapper(GoalWrapper(gym.make(self.env)), p_tremble=0)

            state_dim = cur_env.observation_space.shape[0]
            action_dim = cur_env.action_space.shape[0]
            max_action = float(cur_env.action_space.high[0])

            q_replay_buffer = QReplayBuffer(state_dim, action_dim)
            e = gym.make(self.env)
            dataset = e.get_dataset()
            q_dataset = d4rl.qlearning_dataset(e)
            q_replay_buffer.convert_D4RL(dataset, q_dataset)
            pi_replay_buffer = QReplayBuffer(state_dim, action_dim)

            kwargs = {
                "state_dim": state_dim,
                "action_dim": action_dim,
                "max_action": max_action,
                "discount": 0.99,
                "tau": 0.005,
                # TD3
                "policy_noise": 0.2 * max_action,
                "noise_clip": 0.5 * max_action,
                "policy_freq": 2,
                # TD3 + BC
                "alpha": 2.5,
                "q_replay_buffer": q_replay_buffer,
                "pi_replay_buffer": pi_replay_buffer,
                "env": cur_env,
                "f": f_net,
            }
            pi = TD3_BC(**kwargs)
            for _ in range(1):
                pi.learn(total_timesteps=int(1e4), bc=True)
                mean_reward, std_reward = evaluate_policy(pi, eval_env, n_eval_episodes=25)
                print(100 * mean_reward)

        # elif self.env == 'CartPoleBulletEnv-v1':
        else:
            # ---------------------- CartPoleBulletEnv-v1 用 PPO ----------------------
            # （假设你想沿用 TremblingHandWrapper 等，可以保留或去掉）
            cur_env = TremblingHandWrapper(cur_env, p_tremble=0.1)
            _raw_eval_env = gym.make(self.env)
            if isinstance(_raw_eval_env.observation_space, gym.spaces.Discrete):
                _raw_eval_env = FlattenObservation(_raw_eval_env)
            eval_env = TremblingHandWrapper(_raw_eval_env, p_tremble=0.1)

            # 用 PPO 算法（支持离散动作空间）
            pi = PPO('MlpPolicy', cur_env,
                     verbose=1,
                     policy_kwargs=dict(net_arch=[64, 64]),
                     learning_rate=linear_schedule(7.3e-4),
                     n_steps=2048,  # PPO的常见参数，可自行调整
                     batch_size=64,  # PPO的常见参数，可自行调整
                     gamma=0.98,
                     n_epochs=10,
                     device="cuda:2")

        # else:
        #     # ---------------------- 其它情况（默认 SAC） ----------------------
        #     cur_env = TremblingHandWrapper(cur_env, p_tremble=0.1)
        #     eval_env = TremblingHandWrapper(gym.make(self.env), p_tremble=0.1)
        #     pi = SAC('MlpPolicy', cur_env, verbose=1, policy_kwargs=dict(net_arch=[256, 256]), ent_coef='auto',
        #              learning_rate=linear_schedule(7.3e-4),
        #              train_freq=64,
        #              gradient_steps=64,
        #              gamma=0.98,
        #              tau=0.02,
        #              device=device)
        #     pi.actor.optimizer = OAdam(pi.actor.parameters())
        #     pi.critic.optimizer = OAdam(pi.critic.parameters())

        # -----------------------------------------------------
        # 如果需要 no_regret, 初始化 replay_buffer
        if no_regret:
            if isinstance(cur_env.unwrapped, AtariEnv):
                replay_buffer = ReplayAtariBuffer(obs_shape=cur_env.observation_space.shape)
            elif isinstance(cur_env.action_space, gym.spaces.Discrete):
                # FrozenLake 在上一步已被 FlattenObservation 包装成 Box
                if isinstance(cur_env.observation_space, gym.spaces.Box):
                   obs_dim = cur_env.observation_space.shape[0]
                else:                                # 兜底（极少用到）
                   obs_dim = cur_env.observation_space.n
                replay_buffer = ReplayDiscreteBuffer(obs_space_size=obs_dim)
            else:
                replay_buffer = ReplayBuffer(
                    obs_space_size=cur_env.observation_space.shape[0],
                    action_space_size=cur_env.action_space.shape[0]
                )
            self.replay_buffer = replay_buffer

        # -----------------------------------------------------
        # 3. 外层训练循环
        steps = 0
        for outer in range(outer_steps):
            if not outer == 0:
                learning_rate_used = learn_rate / outer
            else:
                learning_rate_used = learn_rate

            f_opt = OAdam(f_net.parameters(), lr=learning_rate_used)

            # 用选定的 pi 算法训练
            pi.learn(total_timesteps=pi_steps, log_interval=1000)
            steps += pi_steps

            S_curr, A_curr, s = self.sample(cur_env, pi, num_traj_sample, no_regret=no_regret)
            steps += s

            # 采样 learner 的数据
            if no_regret:
                obs_samples, act_samples = self.replay_buffer.sample(batch_size)
                # 转成 PyTorch Tensor
                obs_tensor = torch.tensor(obs_samples, dtype=torch.float32)
                act_tensor = torch.tensor(act_samples)  # 若是离散动作，则后续转换 long

                if isinstance(cur_env.action_space, gym.spaces.Discrete):
                    num_actions = cur_env.action_space.n
                    act_tensor = act_tensor.long().squeeze(-1)

                    act_onehot = F.one_hot(act_tensor, num_classes=num_actions).float()

                    if isinstance(cur_env.unwrapped, AtariEnv):
                        obs_tensor = obs_tensor.view(obs_tensor.size(0), -1)

                    learner_sa_pairs = torch.cat((obs_tensor, act_onehot), dim=1).to(device)

                else:
                    act_tensor = act_tensor.float()
                    learner_sa_pairs = torch.cat((obs_tensor, act_tensor), dim=1).to(device)
            else:
                if isinstance(cur_env.action_space, gym.spaces.Discrete):
                    num_actions = cur_env.action_space.n  # 2, 3, ...
                    A_flat = A_curr.squeeze(-1).long()
                    A_onehot = F.one_hot(A_flat, num_classes=num_actions).float()
                    if isinstance(cur_env.unwrapped, AtariEnv):
                        S_curr = S_curr.view(S_curr.size(0), -1)
                    learner_sa_pairs = torch.cat((S_curr, A_onehot), dim=1).to(device)
                else:
                    learner_sa_pairs = torch.cat((S_curr, A_curr), dim=1).to(device)

            # 训练 f_net (Discriminator)
            for _ in range(f_steps):
                learner_sa = learner_sa_pairs[np.random.choice(len(learner_sa_pairs), batch_size)]
                expert_sa = expert_sa_pairs[np.random.choice(len(expert_sa_pairs), batch_size)]
                f_opt.zero_grad()
                f_learner = f_net(learner_sa.float())
                f_expert = f_net(expert_sa.float())
                gp = gradient_penalty(learner_sa, expert_sa, f_net)
                loss = f_expert.mean() - f_learner.mean() + 10 * gp
                loss.backward()
                f_opt.step()

            kl_div = expert_kl(expert_policy, pi, expert_obs, expert_acts, device)
            mean_rew, _ = evaluate_policy(pi, eval_env)
            writer.add_scalar("Valid/distance", kl_div, outer)
            writer.add_scalar("Valid/reward", mean_rew, outer)
            print(f"kl_div = {kl_div}")
            print(f"mean_rew = {mean_rew}")

            # 每隔 log_interval，评估一次表现
            if outer % log_interval == 0:
                if 'maze' in self.env:
                    mean_reward, std_reward = evaluate_policy(pi, eval_env, n_eval_episodes=25)
                    mean_reward = mean_reward * 100
                    std_reward = std_reward * 100
                else:
                    mean_reward, std_reward = evaluate_policy(pi, eval_env, n_eval_episodes=10)
                mean_rewards.append(mean_reward)
                std_rewards.append(std_reward)
                env_steps.append(steps)
                print("{0} Iteration: {1}".format(int(outer), mean_reward))

            if outer == outer_steps - 1:
                policy_save_path = os.path.join(log_dir, f"policy_filter")
                pi.save(policy_save_path)
                

            # 保存中间结果
            # os.makedirs(os.path.join("learners", self.env), exist_ok=True)
            # np.savez(os.path.join("learners", self.env, "{0}_rewards_{1}_{2}_{3}".format(
            #     name, n_exp, n_seed, outer)),
            #          means=mean_rewards,
            #          stds=std_rewards,
            #          env_steps=env_steps)
