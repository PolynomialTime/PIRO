'''
Trained and saved reward estimator here.
'''
import sys, os, time
import numpy as np
import torch
import gym
from ruamel.yaml import YAML
from torch.utils.tensorboard import SummaryWriter

from train.models.reward import MLPReward
from common.sac import ReplayBuffer, SAC
from common.robotic_wrapper import make_array_env, make_env_fn

import envs
from utils import system, collect, logger, eval
import datetime
import dateutil.tz
import json, copy
import argparse

def ML_loss(div: str, agent_samples, expert_samples, reward_func, device):
    ''' NOTE: only for ML: E_p[r(tau)] - E_q[r(tau)] w.r.t. r
        agent_samples is numpy array of shape (N, T, d) 
        expert_samples is numpy array of shape (N, T, d) or (N, d)
    '''
    assert div in ['maxentirl']
    sA, _, _ = agent_samples
    _, T, d = sA.shape

    sA_vec = torch.FloatTensor(sA).reshape(-1, d).to(device)
    sE_vec = torch.FloatTensor(expert_samples).reshape(-1, d).to(device)

    t1 = reward_func.r(sA_vec).view(-1) # E_q[r(tau)]
    t2 = reward_func.r(sE_vec).view(-1) # E_p[r(tau)]

    surrogate_objective = t1.mean() - t2.mean() # gradient ascent
    return T * surrogate_objective # same scale

def ML_sa_loss(div: str, agent_samples, expert_samples, reward_func, device):
    ''' NOTE: only for ML_sa: E_p[r(tau)] - E_q[r(tau)] w.r.t. r
        agent_samples is numpy array of shape (N, T, d) 
        expert_samples is numpy array of shape (N, T, d) or (N, d)
    '''
    #assert div in ['maxentirl']
    sA, aA, _ = agent_samples
    print(sA.shape,aA.shape)
    sA=np.concatenate([sA,aA],2)
    _, T, d = sA.shape

    sA_vec = torch.FloatTensor(sA).reshape(-1, d).to(device)
    sE_vec = torch.FloatTensor(expert_samples).reshape(-1, d).to(device)

    t1 = reward_func.r(sA_vec).view(-1) # E_q[r(tau)]
    t2 = reward_func.r(sE_vec).view(-1) # E_p[r(tau)]

    surrogate_objective = t1.mean() - t2.mean() # gradient ascent
    return T * surrogate_objective # same scale


class ConstrainedRewardUpdater:
    """Class to handle constrained reward updates with trust region constraints"""
    
    def __init__(self, v, device, reward_func, sac_agent, env_fn):
        self.v = v
        self.device = device
        self.reward_func = reward_func
        self.sac_agent = sac_agent
        self.env_fn = env_fn
        self.discount = v['reward'].get('discount', 0.99)
        
        # Initialize constraint parameters
        self.target_reward_diff = v['reward'].get('target_reward_diff', 0.1)
        self.target_ratio_upper = v['reward'].get('target_ratio_upper', 1.2)
        self.target_ratio_lower = v['reward'].get('target_ratio_lower', 0.8)
        self.coef_scale_down = v['reward'].get('coef_scale_down', 0.9)
        self.coef_scale_up = v['reward'].get('coef_scale_up', 1.1)
        self.coef_min = v['reward'].get('coef_min', 0.001)
        self.coef_max = v['reward'].get('coef_max', 10.0)
        
        self.target_reward_l2_norm = v['reward'].get('target_reward_l2_norm', 1.0)
        self.l2_coef_scale_up = v['reward'].get('l2_coef_scale_up', 1.1)
        self.l2_coef_scale_down = v['reward'].get('l2_coef_scale_down', 0.9)
        
        # Initialize dynamic coefficients
        self.avg_diff_coef = 1.0
        self.l2_norm_coef = 1.0
        
        # Store old reward network for comparison
        self._old_reward_net = None
        self._global_step = 0
        
    def reward_of_sample_traj_old_policy_cur_reward(self, starting_state, n_timesteps, n_episodes):
        """Estimate discounted return using Monte Carlo sampling"""
        total_return = 0.0
        
        for _ in range(n_episodes):
            env = self.env_fn()
            obs = env.reset()
            if hasattr(starting_state, 'cpu'):
                obs = starting_state.cpu().numpy()
            elif hasattr(starting_state, 'shape'):
                obs = starting_state
            
            episode_return = 0.0
            discount_factor = 1.0
            
            for t in range(n_timesteps):
                # Get action from current policy
                action = self.sac_agent.get_action(obs, deterministic=True)
                
                # Get reward from current reward function
                if self.v['obj'] == 'maxentirl_sa':
                    # For state-action rewards, concatenate state and action
                    obs_action = np.concatenate([obs, action])
                    obs_action_tensor = torch.FloatTensor(obs_action).to(self.device).unsqueeze(0)
                    reward = self.reward_func.get_scalar_reward(obs_action_tensor.cpu().numpy())[0]
                else:
                    # For state-only rewards
                    obs_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
                    reward = self.reward_func.get_scalar_reward(obs_tensor.cpu().numpy())[0]
                
                episode_return += discount_factor * reward
                discount_factor *= self.discount
                
                # Step environment
                obs, _, done, _ = env.step(action)
                if done:
                    break
            
            total_return += episode_return
        
        return torch.tensor(total_return / n_episodes, device=self.device)
    
    def update_reward_constrained(self, samples, expert_samples, expert_samples_sa, reward_optimizer, writer=None):
        """Perform constrained reward update with trust region constraints"""
        
        # Store current reward as old for next iteration
        if self._old_reward_net is None:
            self._old_reward_net = type(self.reward_func)(
                self.reward_func.input_dim,   
                **{k: v for k, v in self.v['reward'].items() if k not in ['lr', 'weight_decay', 'momentum', 'gradient_step']},
                device=self.device
            ).to(self.device)
        
        # Copy current weights to old network
        self._old_reward_net.load_state_dict(self.reward_func.state_dict())
        
        # Get states and actions from samples
        sA, aA, _ = samples
        if self.v['obj'] == 'maxentirl_sa':
            sA = np.concatenate([sA, aA], 2)
        
        _, T, d = sA.shape
        obs = torch.FloatTensor(sA).reshape(-1, d).to(self.device)
        
        # Calculate rewards and differences
        current_rewards = self.reward_func.r(obs).view(-1)
        old_rewards = self._old_reward_net.r(obs).view(-1)
        
        reward_diff = current_rewards - old_rewards
        
        # Calculate discounted returns
        discounts = torch.pow(torch.ones(obs.shape[0], device=self.device) * self.discount,
                            torch.arange(0, obs.shape[0], device=self.device))
        
        # Estimate agent return using Monte Carlo
        # Extract original state (before concatenation with action for SA mode)
        sA_orig, aA_orig, _ = samples
        starting_state = sA_orig[0, 0]  # First state of first trajectory
            
        discounted_agent_return = self.reward_of_sample_traj_old_policy_cur_reward(
            starting_state=starting_state,
            n_timesteps=min(T, sA_orig.shape[1]),  # Use original trajectory length
            n_episodes=self.v['reward'].get('n_episodes', 10)
        )
        
        # Expert return
        if self.v['obj'] == 'maxentirl':
            expert_obs = torch.FloatTensor(expert_samples).reshape(-1, d).to(self.device)
        else:
            expert_obs = torch.FloatTensor(expert_samples_sa).reshape(-1, d).to(self.device)
        
        expert_rewards = self.reward_func.r(expert_obs).view(-1)
        discounted_expert_return = torch.dot(expert_rewards, discounts[:len(expert_rewards)])
        
        # Calculate constraint terms
        avg_reward_diff = torch.mean(reward_diff)
        l2_norm_reward_diff = torch.norm(reward_diff, p=2)
        
        # Adaptive coefficient adjustment
        if avg_reward_diff > self.target_reward_diff * self.target_ratio_upper:
            self.avg_diff_coef = self.avg_diff_coef * self.coef_scale_down
        elif avg_reward_diff < self.target_reward_diff * self.target_ratio_lower:
            self.avg_diff_coef = self.avg_diff_coef * self.coef_scale_up
        
        self.avg_diff_coef = torch.tensor(self.avg_diff_coef)
        self.avg_diff_coef = torch.clamp(self.avg_diff_coef, min=self.coef_min, max=self.coef_max)
        
        if l2_norm_reward_diff > self.target_reward_l2_norm:
            self.l2_norm_coef = self.l2_norm_coef * self.l2_coef_scale_up
        elif l2_norm_reward_diff < self.target_reward_l2_norm:
            self.l2_norm_coef = self.l2_norm_coef * self.l2_coef_scale_down
        
        self.l2_norm_coef = torch.tensor(self.l2_norm_coef)
        self.l2_norm_coef = torch.clamp(self.l2_norm_coef, min=self.coef_min, max=self.coef_max)
        
        # Calculate final loss with constraints
        likelihood = discounted_agent_return - discounted_expert_return
        loss = likelihood  + self.l2_norm_coef * l2_norm_reward_diff
        
        # Perform gradient step
        reward_optimizer.zero_grad()
        loss.backward()
        reward_optimizer.step()
        
        # Log metrics
        if writer is not None:
            writer.add_scalar("Batch/loss", loss.item(), self._global_step)
            writer.add_scalar("Batch/likelihood", likelihood.item(), self._global_step)
            writer.add_scalar("Batch/avg_reward_diff", avg_reward_diff.item(), self._global_step)
            writer.add_scalar("Batch/l2_norm_reward_diff", l2_norm_reward_diff.item(), self._global_step)
        
        self._global_step += 1
        
        return loss, likelihood, avg_reward_diff, l2_norm_reward_diff


class SimpleConstrainedRewardUpdater:
    """Simple constraint addition with adaptive coefficient adjustment"""
    def __init__(self, v, device, reward_func):
        self.v = v
        self.device = device
        self.reward_func = reward_func
        
        # Simple constraint parameters
        self.constraint_weight = v['reward'].get('constraint_weight', 0.01)
        self.max_reward_change = v['reward'].get('max_reward_change', 1.0)
        
        # Adaptive constraint parameters (same as complex version)
        self.target_reward_l2_norm = v['reward'].get('target_reward_l2_norm', 1.0)
        self.l2_coef_scale_up = v['reward'].get('l2_coef_scale_up', 1.1)
        self.l2_coef_scale_down = v['reward'].get('l2_coef_scale_down', 0.9)
        self.coef_min = v['reward'].get('coef_min', 0.001)
        self.coef_max = v['reward'].get('coef_max', 10.0)
        
        # Initialize adaptive coefficient
        self.adaptive_constraint_weight = self.constraint_weight
        
        # Store old reward network for comparison
        self._old_reward_net = None
        
    def update_reward_simple_constrained(self, samples, expert_samples, expert_samples_sa, reward_optimizer, v):
        """Add simple constraints to base loss with adaptive coefficient adjustment"""
        
        # Store current reward as old for next iteration
        if self._old_reward_net is None:
            self._old_reward_net = type(self.reward_func)(
                self.reward_func.input_dim,
                **{k: val for k, val in v['reward'].items() if k not in ['lr', 'weight_decay', 'momentum', 'gradient_step']},
                device=self.device
            ).to(self.device)
        
        # Copy current weights to old network
        self._old_reward_net.load_state_dict(self.reward_func.state_dict())
        
        # Use base ML loss as foundation
        if v['obj'] == 'maxentirl':
            base_loss = ML_loss(v['obj'], samples, expert_samples, self.reward_func, self.device)
        elif v['obj'] == 'maxentirl_sa':
            base_loss = ML_sa_loss(v['obj'], samples, expert_samples_sa, self.reward_func, self.device)
        
        # Add simple constraint: penalize large changes in reward
        constraint_loss = 0.0
        l2_norm_reward_diff = torch.tensor(0.0, device=self.device)
        
        if self._old_reward_net is not None:
            # Get sample observations for constraint calculation
            sA, aA, _ = samples
            if v['obj'] == 'maxentirl_sa':
                sA = np.concatenate([sA, aA], 2)
            
            _, T, d = sA.shape
            obs = torch.FloatTensor(sA).reshape(-1, d).to(self.device)
            
            # Calculate reward difference constraint
            current_rewards = self.reward_func.r(obs).view(-1)
            old_rewards = self._old_reward_net.r(obs).view(-1)
            reward_diff = current_rewards - old_rewards
            
            # Simple L2 penalty on reward changes
            l2_norm_reward_diff = torch.norm(reward_diff, p=2)
            
            # Adaptive coefficient adjustment based on L2 norm
            if l2_norm_reward_diff > self.target_reward_l2_norm*self.l2_coef_scale_up:
                self.adaptive_constraint_weight = self.adaptive_constraint_weight * self.l2_coef_scale_down
            elif l2_norm_reward_diff < self.target_reward_l2_norm/self.l2_coef_scale_up:
                self.adaptive_constraint_weight = self.adaptive_constraint_weight / self.l2_coef_scale_down
            
            # Clamp the adaptive coefficient
            self.adaptive_constraint_weight = max(self.coef_min, min(self.coef_max, self.adaptive_constraint_weight))
            
            constraint_loss = self.adaptive_constraint_weight * l2_norm_reward_diff
        
        # Final loss: original + adaptive simple constraint
        total_loss = base_loss + constraint_loss
        
        # Gradient step
        reward_optimizer.zero_grad()
        total_loss.backward()
        reward_optimizer.step()
        
        return total_loss, base_loss, constraint_loss, l2_norm_reward_diff


def try_evaluate(itr: int, policy_type: str, sac_info):
    assert policy_type in ["Running"]
    update_time = itr * v['reward']['gradient_step']
    env_steps = itr * v['sac']['epochs'] * v['env']['T']
    agent_emp_states = samples[0].copy()
    assert agent_emp_states.shape[0] == v['irl']['training_trajs']

    metrics = eval.KL_summary(expert_samples, agent_emp_states.reshape(-1, agent_emp_states.shape[2]), 
                         env_steps, policy_type)
    # eval real reward
    real_return_det = eval.evaluate_real_return(sac_agent.get_action, env_fn(), 
                                            v['irl']['eval_episodes'], v['env']['T'], True)
    metrics['Real Det Return'] = real_return_det
    print(f"real det return avg: {real_return_det:.2f}")
    logger.record_tabular("Real Det Return", round(real_return_det, 2))

    real_return_sto = eval.evaluate_real_return(sac_agent.get_action, env_fn(), 
                                            v['irl']['eval_episodes'], v['env']['T'], False)
    metrics['Real Sto Return'] = real_return_sto
    print(f"real sto return avg: {real_return_sto:.2f}")
    logger.record_tabular("Real Sto Return", round(real_return_sto, 2))


    logger.record_tabular(f"{policy_type} Update Time", update_time)
    logger.record_tabular(f"{policy_type} Env Steps", env_steps)

    return real_return_det, real_return_sto

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='config yml file')
    parser.add_argument('--l2_coef_scale_up', type=float, default=None, help='override l2_coef_scale_up in yml')
    parser.add_argument('--l2_coef_scale_down', type=float, default=None, help='override l2_coef_scale_down in yml')
    parser.add_argument('--log_dir', type=str, default=None, help='output log dir, e.g. Hopper-v3/x_1.25')
    args = parser.parse_args()

    yaml = YAML()
    v = yaml.load(open(args.config))

    # override l2_coef_scale_up if provided
    if args.l2_coef_scale_up is not None:
        v['reward']['l2_coef_scale_up'] = args.l2_coef_scale_up

    # override l2_coef_scale_down if provided
    if args.l2_coef_scale_down is not None:
        v['reward']['l2_coef_scale_down'] = args.l2_coef_scale_down

    # common parameters
    env_name = v['env']['env_name']
    state_indices = v['env']['state_indices']
    seed = v['seed']
    num_expert_trajs = v['irl']['expert_episodes']

    # system: device, threads, seed, pid
    device = torch.device(f"cuda:{v['cuda']}" if torch.cuda.is_available() and v['cuda'] >= 0 else "cpu")
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    system.reproduce(seed)
    pid=os.getpid()
    
    # assumptions
    assert v['obj'] in ['maxentirl','maxentirl_sa']
    assert v['IS'] == False

    # logs
    if args.log_dir is not None:
        log_folder = args.log_dir
        os.makedirs(log_folder, exist_ok=True)
    else:
        exp_id = f"logs/{env_name}/exp-{num_expert_trajs}/{v['obj']}"
        if not os.path.exists(exp_id):
            os.makedirs(exp_id)
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        log_folder = exp_id + '/' + now.strftime('%Y_%m_%d_%H_%M_%S')
    logger.configure(dir=log_folder)            
    print(f"Logging to directory: {log_folder}")
    os.system(f'cp train/trainPIRO.py {log_folder}')
    os.system(f'cp {args.config} {log_folder}/variant_{pid}.yml')
    with open(os.path.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(v, f, indent=2, sort_keys=True)
    print('pid', pid)
    os.makedirs(os.path.join(log_folder, 'plt'), exist_ok=True)
    os.makedirs(os.path.join(log_folder, 'model'), exist_ok=True)

    # environment
    env_fn = make_env_fn(env_name)
    env_fn = make_array_env(env_fn)
    gym_env = env_fn()
    state_size = gym_env.observation_space.shape[0]
    action_size = gym_env.action_space.shape[0]
    if state_indices == 'all':
        state_indices = list(range(state_size))

    # load expert samples from trained policy
    #expert_trajs = torch.load(f'expert_data/states/{env_name}.pt').numpy()[:, :, state_indices]
    env_name=env_name.split('-')[0]
    expert_trajs = np.load(f'expert_data/{env_name}/states.npy')

    if expert_trajs.dtype == object:
        traj_list = expert_trajs.tolist()
        traj_list = traj_list[:num_expert_trajs]
        try:
            expert_trajs = np.stack(traj_list, axis=0)
        except ValueError:
            min_len = min(t.shape[0] for t in traj_list)
            expert_trajs = np.stack([t[:min_len] for t in traj_list], axis=0)

    expert_trajs = expert_trajs[:num_expert_trajs, :, :] # select first expert_episodes
    expert_samples = expert_trajs.copy().reshape(-1, len(state_indices))
    #expert_a = torch.load(f'expert_data/actions/{env_name}.pt').numpy()[:, :, :]
    expert_a = np.load(f'expert_data/{env_name}/actions.npy')

    if expert_a.dtype == object:
        traj_list = expert_a.tolist()
        traj_list = traj_list[:num_expert_trajs]
        try:
            expert_a = np.stack(traj_list, axis=0)
        except ValueError:
            min_len = min(t.shape[0] for t in traj_list)
            expert_a = np.stack([t[:min_len] for t in traj_list], axis=0)

    expert_a = expert_a[:num_expert_trajs, :, :] # select first expert_episodes
    expert_a_samples = expert_a.copy().reshape(-1, action_size)
    expert_samples_sa=np.concatenate([expert_samples,expert_a_samples],1)
    print(expert_trajs.shape, expert_samples_sa.shape) # ignored starting state

    # Initilialize reward as a neural network
    
    reward_func = MLPReward(len(state_indices), **v['reward'], device=device).to(device)
    sa=False
    if v['obj']=='maxentirl_sa':
        sa=True
        reward_func = MLPReward(len(state_indices)+action_size, **v['reward'], device=device).to(device)
    reward_optimizer = torch.optim.Adam(reward_func.parameters(), lr=v['reward']['lr'], 
        weight_decay=v['reward']['weight_decay'], betas=(v['reward']['momentum'], 0.999))
    
    # Initialize constrained reward updater if enabled
    use_constraints = v['reward'].get('use_constraints', False)
    constraint_type = v['reward'].get('constraint_type', 'simple')
    constrained_updater = None
    simple_constrained_updater = None
    writer = None
    
    if use_constraints:
        if constraint_type == 'complex':
            constrained_updater = ConstrainedRewardUpdater(v, device, reward_func, None, env_fn)
            # Initialize tensorboard writer
            log_dir = logger.get_dir() if hasattr(logger, 'get_dir') and logger.get_dir() else './tensorboard_logs'
            writer = SummaryWriter(log_dir=log_dir)
        elif constraint_type == 'simple':
            simple_constrained_updater = SimpleConstrainedRewardUpdater(v, device, reward_func)
    
    max_real_return_det, max_real_return_sto = -np.inf, -np.inf
    for itr in range(v['irl']['n_itrs']):
        if v['sac']['reinitialize'] or itr == 0:
            # Reset SAC agent with old policy, new environment, and new replay buffer
            print("Reinitializing sac")
            replay_buffer = ReplayBuffer(
                state_size, 
                action_size,
                device=device,
                size=v['sac']['buffer_size'])
                
            sac_agent = SAC(env_fn, replay_buffer,
                steps_per_epoch=v['env']['T'],
                update_after=v['env']['T'] * v['sac']['random_explore_episodes'], 
                max_ep_len=v['env']['T'],
                seed=seed,
                start_steps=v['env']['T'] * v['sac']['random_explore_episodes'],
                reward_state_indices=state_indices,
                device=device,
                sa=sa,
                **v['sac']
            )
        
        sac_agent.reward_function = reward_func.get_scalar_reward # only need to change reward in sac
        
        # Update the sac_agent reference in constrained_updater
        if use_constraints and constraint_type == 'complex' and constrained_updater is not None:
            constrained_updater.sac_agent = sac_agent
            
        sac_info = sac_agent.learn_mujoco(print_out=True)

        start = time.time()
        samples = collect.collect_trajectories_policy_single(gym_env, sac_agent, 
                        n = v['irl']['training_trajs'], state_indices=state_indices)
        # Fit a density model using the samples
        agent_emp_states = samples[0].copy()
        agent_emp_states = agent_emp_states.reshape(-1,agent_emp_states.shape[2]) # n*T states
        print(f'collect trajs {time.time() - start:.0f}s', flush=True)
        # print(agent_emp_states.shape)

        start = time.time()

        # optimization w.r.t. reward
        reward_losses = []
        
        if use_constraints and constraint_type == 'complex' and constrained_updater is not None:
            print("Using complex constrained reward update")
            # Use complex constrained reward update
            for _ in range(v['reward']['gradient_step']):
                if v['irl']['resample_episodes'] > v['irl']['expert_episodes']:
                    expert_res_indices = np.random.choice(expert_trajs.shape[0], v['irl']['resample_episodes'], replace=True)
                    expert_trajs_train = expert_trajs[expert_res_indices].copy()
                elif v['irl']['resample_episodes'] > 0:
                    expert_res_indices = np.random.choice(expert_trajs.shape[0], v['irl']['resample_episodes'], replace=False)
                    expert_trajs_train = expert_trajs[expert_res_indices].copy()
                else:
                    expert_trajs_train = None
                
                loss, likelihood, avg_reward_diff, l2_norm_reward_diff = constrained_updater.update_reward_constrained(
                    samples, expert_samples, expert_samples_sa, reward_optimizer, writer)
                
                reward_losses.append(loss.item())
                print(f"{v['obj']} complex constrained loss: {loss:.4f}, likelihood: {likelihood:.4f}, "
                      f"avg_diff: {avg_reward_diff:.4f}, l2_norm: {l2_norm_reward_diff:.4f}")
                      
                # Log constraint metrics
                if writer is not None:
                    writer.add_scalar("Update_Reward/loss", loss.item(), constrained_updater._global_step)
                    writer.add_scalar("Update_Reward/likelihood", likelihood.item(), constrained_updater._global_step)
                    writer.add_scalar("Update_Reward/avg_reward_diff", avg_reward_diff.item(), constrained_updater._global_step)
                    writer.add_scalar("Update_Reward/l2_norm_reward_diff", l2_norm_reward_diff.item(), constrained_updater._global_step)
                    
        elif use_constraints and constraint_type == 'simple' and simple_constrained_updater is not None:
            print("Using simple constrained reward update")
            # Use simple constrained reward update
            for _ in range(v['reward']['gradient_step']):
                if v['irl']['resample_episodes'] > v['irl']['expert_episodes']:
                    expert_res_indices = np.random.choice(expert_trajs.shape[0], v['irl']['resample_episodes'], replace=True)
                    expert_trajs_train = expert_trajs[expert_res_indices].copy()
                elif v['irl']['resample_episodes'] > 0:
                    expert_res_indices = np.random.choice(expert_trajs.shape[0], v['irl']['resample_episodes'], replace=False)
                    expert_trajs_train = expert_trajs[expert_res_indices].copy()
                else:
                    expert_trajs_train = None
                
                total_loss, base_loss, constraint_loss, l2_norm_reward_diff = simple_constrained_updater.update_reward_simple_constrained(
                    samples, expert_samples, expert_samples_sa, reward_optimizer, v)
                
                reward_losses.append(total_loss.item())
                print(f"{v['obj']} simple constrained loss: {total_loss:.4f} (base: {base_loss:.4f}, constraint: {constraint_loss:.4f}, "
                      f"l2_norm: {l2_norm_reward_diff:.4f}, adaptive_weight: {simple_constrained_updater.adaptive_constraint_weight:.6f})")
                loss = total_loss  # For logging compatibility
        else:
            # Use original reward update
            print("Using original reward update without constraints")
            for _ in range(v['reward']['gradient_step']):
                if v['irl']['resample_episodes'] > v['irl']['expert_episodes']:
                    expert_res_indices = np.random.choice(expert_trajs.shape[0], v['irl']['resample_episodes'], replace=True)
                    expert_trajs_train = expert_trajs[expert_res_indices].copy() # resampling the expert trajectories
                elif v['irl']['resample_episodes'] > 0:
                    expert_res_indices = np.random.choice(expert_trajs.shape[0], v['irl']['resample_episodes'], replace=False)
                    expert_trajs_train = expert_trajs[expert_res_indices].copy()
                else:
                    expert_trajs_train = None # not use expert trajs

                if v['obj'] == 'maxentirl':
                    loss = ML_loss(v['obj'], samples, expert_samples, reward_func, device)
                elif v['obj'] == 'maxentirl_sa':
                    loss = ML_sa_loss(v['obj'], samples, expert_samples_sa, reward_func, device) 
                
                reward_losses.append(loss.item())
                print(f"{v['obj']} loss: {loss}")
                reward_optimizer.zero_grad()
                loss.backward()
                reward_optimizer.step()

        # evaluating the learned reward
        real_return_det, real_return_sto = try_evaluate(itr, "Running", sac_info)
        if real_return_det > max_real_return_det and real_return_sto > max_real_return_sto:
            max_real_return_det, max_real_return_sto = real_return_det, real_return_sto
            torch.save(reward_func.state_dict(), os.path.join(logger.get_dir(), 
                    f"model/reward_model_itr{itr}_det{max_real_return_det:.0f}_sto{max_real_return_sto:.0f}.pkl"))

        logger.record_tabular("Itration", itr)
        logger.record_tabular("Reward Loss", loss.item())
        if v['sac']['automatic_alpha_tuning']:
            logger.record_tabular("alpha", sac_agent.alpha.item())
        
        if v['irl']['save_interval'] > 0 and (itr % v['irl']['save_interval'] == 0 or itr == v['irl']['n_itrs']-1):
            torch.save(reward_func.state_dict(), os.path.join(logger.get_dir(), f"model/reward_model_{itr}.pkl"))

        logger.dump_tabular()
    
    # Cleanup tensorboard writer
    if writer is not None:
        writer.close()