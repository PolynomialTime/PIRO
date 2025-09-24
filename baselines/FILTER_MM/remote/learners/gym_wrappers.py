import numpy as np
import torch
import gym
import os
import sys
from gym.envs.classic_control.meerkat_v3 import MeerkatEnv
from gym.envs.classic_control.acrobot import AcrobotEnv
from gym.envs.classic_control.cartpole import CartPoleEnv
from gym.envs.box2d import BipedalWalker, LunarLander
from gym.envs.atari import AtariEnv
from gym.envs.mujoco.humanoid_v3 import HumanoidEnv
from gym.envs.mujoco.ant_v3 import AntEnv
from gym.envs.mujoco.walker2d_v3 import Walker2dEnv
from gym.envs.mujoco.hopper_v3 import HopperEnv
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv


def _obs_to_vec(obs, obs_space):
    if isinstance(obs_space, gym.spaces.Discrete):
        vec = np.zeros(obs_space.n, dtype=np.float32)
        vec[int(obs)] = 1.0
        return vec
    else:
        return np.asarray(obs, dtype=np.float32).flatten()
    

def set_state(env, base_pos, base_vel, joint_states):
    # 调试代码
    # print("Environment type:", type(env))
    # print("Environment attributes:", dir(env))
    # if hasattr(env, 'env'):
    #     print("Inner environment type:", type(env.env))
    #     print("Inner environment attributes:", dir(env.env))
    p = env.unwrapped._p
    for i in range(p.getNumBodies()):
        p.resetBasePositionAndOrientation(i,*base_pos[i])
        p.resetBaseVelocity(i,*base_vel[i])
        for j in range(p.getNumJoints(i)):
            p.resetJointState(i,j,*joint_states[i][j][:2])

class ResetWrapper(gym.Wrapper):
    def __init__(self, env, P, V, C, expert_obs, expert_acts):
        super().__init__(env)
        self.env = env
        self.alpha = 0.5
        self.P = P
        self.V = V
        self.C = C
        self.expert_obs = expert_obs
        self.expert_acts = expert_acts
        self.t = 0
        self.max_t = 1000

    def reset(self):
        self.env.reset()
        if isinstance(self.env.unwrapped, AcrobotEnv) or isinstance(self.env.unwrapped, BipedalWalker) or isinstance(self.env.unwrapped, CartPoleEnv)\
                or isinstance(self.env.unwrapped, AtariEnv) or isinstance(self.env.unwrapped, LunarLander) or isinstance(self.env.unwrapped, HumanoidEnv)\
                or isinstance(self.env.unwrapped, AntEnv) or isinstance(self.env.unwrapped, MeerkatEnv) or isinstance(self.env.unwrapped, FrozenLakeEnv)\
                or isinstance(self.env.unwrapped, Walker2dEnv) or isinstance(self.env.unwrapped, HopperEnv) or isinstance(self.env.unwrapped, HalfCheetahEnv):
            obs = self.env.reset()
            if np.random.uniform() < self.alpha:
                idx = np.random.choice(len(self.P))
                t = np.random.choice(min(len(self.P[idx]), self.max_t))
                for i in range(t):
                    a = self.expert_acts[idx][i]
                    if isinstance(self.env.unwrapped, AcrobotEnv) or isinstance(self.env.unwrapped, AtariEnv) or isinstance(self.env.unwrapped, CartPoleEnv) \
                            or isinstance(self.env.unwrapped, LunarLander) or isinstance(self.env.unwrapped, MeerkatEnv) or isinstance(self.env.unwrapped, FrozenLakeEnv):
                        a_int = int(a.item())
                        obs, rew, done, info = self.env.step(a_int)
                    else:
                        if isinstance(a, torch.Tensor):
                            a = a.detach().cpu().numpy()
                        obs, rew, done, info = self.env.step(a)
                    if done:
                        # 如果 done，则可以 break 或者重新 reset，看你需求
                        break

                self.t = t
            else:
                self.t = 0

            return obs
        else:
            if np.random.uniform() < self.alpha:
                idx = np.random.choice(len(self.P))
                t = np.random.choice(min(len(self.P[idx]), self.max_t))
                set_state(self.env, self.P[idx][0], self.V[idx][0], self.C[idx][0])
                obs = self.env.env.robot.calc_state()
                for i in range(t):
                    a = self.expert_acts[idx][i]
                    next_obs, rew, done, _ = self.env.step(a)
                    obs = next_obs
                self.t = t
            else:
                self.t = 0
            return self.env.env.robot.calc_state()

    def step(self, action):
        next_obs, rew, done, info = self.env.step(action)
        self.t += 1
        if self.t >= self.max_t:
            done = True
        return next_obs, rew, done, info

class RewardWrapper(gym.Wrapper):
    def __init__(self, env, function):
        super().__init__(env)
        self.env = env
        self.cur_state = None
        self.function = function
        self.low = env.action_space.low
        self.high = env.action_space.high

    def reset(self):
        obs = self.env.reset()
        self.cur_state = _obs_to_vec(obs, self.env.observation_space)
        return obs

    def step(self, action):
        next_state, _, done, info = self.env.step(action)
        #combine action and state
        sa_pair = np.concatenate((self.cur_state, action))
        reward = -(self.function.forward(torch.tensor(sa_pair, dtype=torch.float).to("cuda:2")))
        self.cur_state = _obs_to_vec(next_state, self.env.observation_space)

        return next_state, reward, done, info

class RewardDiscreteWrapper(gym.Wrapper):
    def __init__(self, env, function):
        super().__init__(env)
        self.env = env
        self.cur_state = None
        self.function = function
        self.low = None
        self.high = None

    def reset(self):
        obs = self.env.reset()
        self.cur_state = _obs_to_vec(obs, self.env.observation_space)
        return obs

    def step(self, action):
        # 先执行一步环境
        next_state, _, done, info = self.env.step(action)

        if isinstance(self.env.action_space, gym.spaces.Discrete):
            # 离散动作：转成 one-hot 编码
            n = self.env.action_space.n
            action_vec = np.zeros(n, dtype=np.float32)
            action_vec[action] = 1.0
        elif isinstance(self.env.action_space, gym.spaces.Box):
            # 连续动作：直接转换为 numpy 数组（注意保证数据类型一致）
            action_vec = np.array(action, dtype=np.float32)
        else:
            raise ValueError("Unsupported action space type.")

        if isinstance(self.env.unwrapped, AtariEnv):
            self.cur_state = self.cur_state.flatten()

        # 将当前状态和 one‐hot 动作拼接
        sa_pair = np.concatenate((self.cur_state, action_vec))

        # 用网络计算奖励（此处是负值，只是你示例中的方式）
        # 注意：如果你的网络输入维度与输出维度不匹配，会抛出 mat1 and mat2 shapes error
        reward = -(self.function.forward(torch.tensor(sa_pair, dtype=torch.float).to("cuda:2")))

        self.cur_state = _obs_to_vec(next_state, self.env.observation_space)

        # 这里你可以自行决定返回什么观测；如果想保持 Gym 接口的一致性，
        # 通常返回 next_state，而不是 (状态+动作) 拼起来的东西。
        return next_state, reward, done, info

class TremblingHandWrapper(gym.Wrapper):
    def __init__(self, env, p_tremble=0.01):
        super().__init__(env)
        self.env = env
        self.p_tremble = p_tremble

    def reset(self):
        return self.env.reset()

    def step(self, action):
        if np.random.uniform() < self.p_tremble:
            action = self.env.action_space.sample()
        return self.env.step(action)

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class GoalWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(31,))

    def reset(self):
        with HiddenPrints():
            obs = self.env.reset()
            goal = self.env.target_goal
            return np.concatenate([obs, goal])

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        goal = self.env.target_goal
        return np.concatenate([obs, goal]), rew, done, info

class AntMazeResetWrapper(gym.Wrapper):
    def __init__(self, env, qpos, qvel, G):
        super().__init__(env)
        self.env = env
        self.alpha = 1
        self.qpos = qpos
        self.qvel = qvel
        self.G = G
        self.t = 0
        self.T = 700

    def reset(self):
        obs = self.env.reset()
        if np.random.uniform() < self.alpha:
            idx = np.random.choice(len(self.qpos))
            t = np.random.choice(len(self.qpos[idx]))
            with HiddenPrints():
                self.env.set_target(tuple(self.G[idx][t]))
            self.env.set_state(self.qpos[idx][t], self.qvel[idx][t])
            self.t = t
            obs = self.env.env.wrapped_env._get_obs()
            goal = self.env.target_goal
            obs = np.concatenate([obs, goal])
        else:
            self.t = 0
        return obs

    def step(self, action):
        next_obs, rew, done, info = self.env.step(action)
        self.t += 1
        if self.t >= self.T:
            done = True
        return next_obs, rew, done, info