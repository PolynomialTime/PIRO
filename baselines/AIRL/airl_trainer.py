#!/usr/bin/env python3
"""
AIRL (Adversarial Inverse Reinforcement Learning) training script
Based on imitation library implementation
"""

import sys, os, time
import numpy as np
import torch
import gymnasium as gym
from ruamel.yaml import YAML
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import rollout
from imitation.data.types import Trajectory
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
import datetime
import dateutil.tz
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import envs
from utils import system, logger

def load_expert_trajectories(env_name, num_episodes, expert_data_dir="expert_data"):
    """Load expert trajectory data and convert to Trajectory format"""
    
    # Map environment names
    env_mapping = {
        'HalfCheetah-v2': 'HalfCheetah',
        'Ant-v2': 'Ant',
        'Walker2d-v3': 'Walker2d',
        'Hopper-v3': 'Hopper',
        'Humanoid-v3': 'Humanoid'
    }
    
    data_env_name = env_mapping.get(env_name, env_name.split('-')[0])
    data_path = os.path.join(expert_data_dir, data_env_name)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Expert data directory not found: {data_path}")
    
    # Load data
    states = np.load(os.path.join(data_path, "states.npy"))
    actions = np.load(os.path.join(data_path, "actions.npy"))
    rewards = np.load(os.path.join(data_path, "reward.npy"))
    dones = np.load(os.path.join(data_path, "dones.npy"))
    
    print(f"Loaded expert data shapes:")
    print(f"  States: {states.shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  Rewards: {rewards.shape}")
    print(f"  Dones: {dones.shape}")
    
    # Select specified number of episodes
    states = states[:num_episodes]
    actions = actions[:num_episodes]
    rewards = rewards[:num_episodes]
    dones = dones[:num_episodes]
    
    trajectories = []
    
    for ep_idx in range(num_episodes):
        ep_states = states[ep_idx]
        ep_actions = actions[ep_idx]
        ep_rewards = rewards[ep_idx]
        ep_dones = dones[ep_idx]
        
        # Find episode length (when done=True or reach end)
        ep_len = len(ep_states)
        for t in range(len(ep_dones)):
            if ep_dones[t]:
                ep_len = t + 1
                break
        
        # Create trajectory
        obs = ep_states[:ep_len + 1] if ep_len < len(ep_states) else ep_states[:ep_len]
        acts = ep_actions[:ep_len]
        infos = [{} for _ in range(ep_len)]
        
        # Ensure we have the right dimensions
        if len(obs) == len(acts):
            # Add final observation if missing
            obs = np.concatenate([obs, obs[-1:]], axis=0)
        
        trajectory = Trajectory(
            obs=obs,
            acts=acts,
            infos=np.array(infos, dtype=object),
            terminal=True
        )
        trajectories.append(trajectory)
    
    print(f"Created {len(trajectories)} trajectories")
    return trajectories

def make_env_with_gymnasium(env_name):
    """Create environment using gymnasium API"""
    
    try:
        # Try to make the environment directly
        env = gym.make(env_name)
        print(f"Created gymnasium environment: {env_name}")
        return env
    except:
        # If not available, convert from old gym
        print(f"Environment {env_name} not available in gymnasium, using gym conversion")
        
        import gym as old_gym
        old_env = old_gym.make(env_name)
        
        # Convert to gymnasium format manually
        class GymToGymnasiumWrapper(gym.Env):
            def __init__(self, old_env):
                self.old_env = old_env
                
                # Convert observation space
                obs_space = old_env.observation_space
                if hasattr(obs_space, 'low') and hasattr(obs_space, 'high'):
                    self.observation_space = gym.spaces.Box(
                        low=obs_space.low,
                        high=obs_space.high,
                        shape=obs_space.shape,
                        dtype=obs_space.dtype
                    )
                else:
                    self.observation_space = obs_space
                
                # Convert action space
                act_space = old_env.action_space
                if hasattr(act_space, 'low') and hasattr(act_space, 'high'):
                    self.action_space = gym.spaces.Box(
                        low=act_space.low,
                        high=act_space.high,
                        shape=act_space.shape,
                        dtype=act_space.dtype
                    )
                else:
                    self.action_space = act_space
                
                self.metadata = getattr(old_env, 'metadata', {})
                self.render_mode = None
                
            def reset(self, seed=None, options=None):
                if seed is not None:
                    self.old_env.seed(seed)
                obs = self.old_env.reset()
                return obs, {}
                
            def step(self, action):
                obs, reward, done, info = self.old_env.step(action)
                return obs, reward, done, False, info
                
            def render(self):
                return self.old_env.render()
                
            def close(self):
                return self.old_env.close()
        
        wrapped_env = GymToGymnasiumWrapper(old_env)
        return wrapped_env

def main():
    if len(sys.argv) != 2:
        print("Usage: python airl_trainer.py <config_file>")
        sys.exit(1)
    
    # Load configuration
    yaml = YAML()
    v = yaml.load(open(sys.argv[1]))
    
    # Extract parameters
    env_name = v['env']['env_name']
    seed = v['seed']
    
    # Generate random seed if seed is -1
    if seed == -1:
        seed = int(time.time() * 1000) % 100000  # Generate random seed from timestamp
        print(f"Generated random seed: {seed}")
    
    airl_config = v['airl']
    ppo_config = v.get('ppo', {})
    
    # System setup
    device = torch.device(f"cuda:{v['cuda']}" if torch.cuda.is_available() and v['cuda'] >= 0 else "cpu")
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    system.reproduce(seed)
    
    print(f"Using device: {device}")
    print(f"Environment: {env_name}")
    print(f"AIRL Config: {airl_config}")
    print(f"PPO Config: {ppo_config}")
    
    # Create vectorized environment
    rng = np.random.default_rng(seed)
    
    # Create environment using make_vec_env for compatibility
    try:
        # Use custom environment creation for our MuJoCo environments
        def env_fn():
            env = make_env_with_gymnasium(env_name)
            return RolloutInfoWrapper(env)
        
        venv = DummyVecEnv([env_fn for _ in range(airl_config.get('n_envs', 8))])
        print(f"Created vectorized environment with {airl_config.get('n_envs', 8)} environments")
    except Exception as e:
        print(f"Error creating vectorized environment: {e}")
        # Fallback to single environment
        single_env = make_env_with_gymnasium(env_name)
        venv = DummyVecEnv([lambda: RolloutInfoWrapper(single_env)])
        print("Fallback to single environment")
    
    print(f"Environment spaces:")
    print(f"  Observation space: {venv.observation_space}")
    print(f"  Action space: {venv.action_space}")
    
    # Logging setup
    exp_id = f"AIRL/logs/{env_name}/exp-{airl_config['expert_episodes']}"
    if not os.path.exists(exp_id):
        os.makedirs(exp_id, exist_ok=True)
    
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    log_folder = exp_id + '/' + now.strftime('%Y_%m_%d_%H_%M_%S')
    logger.configure(dir=log_folder)
    print(f"Logging to directory: {log_folder}")
    
    # Save config
    os.system(f'cp {sys.argv[1]} {log_folder}/variant.yml')
    with open(os.path.join(log_folder, 'variant.json'), 'w') as f:
        json.dump(v, f, indent=2, sort_keys=True)
    
    os.makedirs(os.path.join(log_folder, 'model'), exist_ok=True)
    
    # Load expert trajectories
    print("Loading expert trajectories...")
    expert_trajectories = load_expert_trajectories(
        env_name, 
        airl_config['expert_episodes']
    )
    
    # Create PPO learner
    print("Creating PPO learner...")
    learner = PPO(
        env=venv,
        policy=MlpPolicy,
        batch_size=ppo_config.get('batch_size', 64),
        ent_coef=ppo_config.get('ent_coef', 0.0),
        learning_rate=ppo_config.get('learning_rate', 0.0005),
        gamma=ppo_config.get('gamma', 0.99),
        clip_range=ppo_config.get('clip_range', 0.2),
        vf_coef=ppo_config.get('vf_coef', 0.5),
        n_epochs=ppo_config.get('n_epochs', 10),
        seed=seed,
        device=device,
        verbose=1
    )
    
    # Create reward network
    print("Creating reward network...")
    reward_net = BasicShapedRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm
    )
    
    # Create AIRL trainer
    print("Creating AIRL trainer...")
    airl_trainer = AIRL(
        demonstrations=expert_trajectories,
        demo_batch_size=airl_config.get('demo_batch_size', 2048),
        gen_replay_buffer_capacity=airl_config.get('gen_replay_buffer_capacity', 512),
        n_disc_updates_per_round=airl_config.get('n_disc_updates_per_round', 16),
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True
    )
    
    # Evaluate before training
    print("Evaluating policy before training...")
    venv.seed(seed)
    learner_rewards_before, _ = evaluate_policy(
        learner, 
        venv, 
        n_eval_episodes=airl_config.get('eval_episodes', 10), 
        return_episode_rewards=True
    )
    mean_reward_before = np.mean(learner_rewards_before)
    std_reward_before = np.std(learner_rewards_before)
    print(f"Mean reward before training: {mean_reward_before:.2f} ± {std_reward_before:.2f}")
    
    # Training with periodic evaluation
    print(f"Starting AIRL training for {airl_config['total_timesteps']} timesteps...")
    start_time = time.time()
    
    # Setup periodic evaluation
    eval_freq = airl_config.get('eval_freq', 50000)  # Evaluate every 50k steps by default
    total_timesteps = airl_config['total_timesteps']
    n_evaluations = total_timesteps // eval_freq
    
    # Create progress CSV
    progress_file = os.path.join(log_folder, 'progress.csv')
    with open(progress_file, 'w') as f:
        f.write("AIRL Round,AIRL Timesteps,AIRL Mean Reward,AIRL Std Reward,AIRL Training Time,AIRL Improvement\n")
    
    # Training with evaluation
    for round_i in range(n_evaluations):
        timesteps_this_round = eval_freq
        if round_i == n_evaluations - 1:
            # Last round might have different timesteps
            timesteps_this_round = total_timesteps - (round_i * eval_freq)
        
        print(f"Training round {round_i + 1}/{n_evaluations} ({timesteps_this_round} timesteps)...")
        airl_trainer.train(total_timesteps=timesteps_this_round)
        
        # Evaluate current policy
        print(f"Evaluating after round {round_i + 1}...")
        venv.seed(seed + round_i)  # Different seed for evaluation
        eval_rewards, _ = evaluate_policy(
            learner, 
            venv, 
            n_eval_episodes=airl_config.get('eval_episodes', 10),
            return_episode_rewards=True
        )
        
        mean_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        improvement = mean_reward - mean_reward_before
        current_time = time.time() - start_time
        
        print(f"Round {round_i + 1} - Mean reward: {mean_reward:.2f} ± {std_reward:.2f}, Improvement: {improvement:.2f}")
        
        # Log to progress CSV
        with open(progress_file, 'a') as f:
            f.write(f"{round_i + 1},{(round_i + 1) * eval_freq},{mean_reward:.6f},{std_reward:.6f},{current_time:.2f},{improvement:.6f}\n")
        
        # Log to tensorboard/logger
        logger.record_tabular("AIRL Round", round_i + 1)
        logger.record_tabular("AIRL Timesteps", (round_i + 1) * eval_freq)
        logger.record_tabular("AIRL Mean Reward", mean_reward)
        logger.record_tabular("AIRL Std Reward", std_reward)
        logger.record_tabular("AIRL Improvement", improvement)
        logger.record_tabular("AIRL Training Time", current_time)
        logger.dump_tabular()
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate after training (final evaluation)
    print("Evaluating policy after training...")
    venv.seed(seed)
    learner_rewards_after, _ = evaluate_policy(
        learner, 
        venv, 
        n_eval_episodes=airl_config.get('eval_episodes', 10) * 2,  # More episodes for final eval
        return_episode_rewards=True
    )
    mean_reward_after = np.mean(learner_rewards_after)
    std_reward_after = np.std(learner_rewards_after)
    print(f"Mean reward after training: {mean_reward_after:.2f} ± {std_reward_after:.2f}")
    
    # Calculate improvement
    improvement = mean_reward_after - mean_reward_before
    print(f"Final improvement: {improvement:.2f}")
    
    # Add final results to progress CSV
    with open(progress_file, 'a') as f:
        f.write(f",,,{airl_config['expert_episodes']},{mean_reward_after:.6f},{len(expert_trajectories)},{std_reward_after:.6f}\n")
    
    # Save models
    learner_path = os.path.join(log_folder, 'model', 'learner_policy.zip')
    learner.save(learner_path)
    print(f"Learner policy saved to: {learner_path}")
    
    reward_net_path = os.path.join(log_folder, 'model', 'reward_net.pt')
    torch.save(reward_net.state_dict(), reward_net_path)
    print(f"Reward network saved to: {reward_net_path}")
    
    # Save results
    results = {
        'mean_reward_before': float(mean_reward_before),
        'std_reward_before': float(std_reward_before),
        'mean_reward_after': float(mean_reward_after),
        'std_reward_after': float(std_reward_after),
        'improvement': float(improvement),
        'training_time': training_time,
        'total_timesteps': airl_config['total_timesteps'],
        'expert_episodes': airl_config['expert_episodes'],
        'expert_trajectories': len(expert_trajectories)
    }
    
    results_path = os.path.join(log_folder, 'airl_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")
    
    # Final logging
    logger.record_tabular("AIRL Mean Reward Before", mean_reward_before)
    logger.record_tabular("AIRL Std Reward Before", std_reward_before)
    logger.record_tabular("AIRL Mean Reward After", mean_reward_after)
    logger.record_tabular("AIRL Std Reward After", std_reward_after)
    logger.record_tabular("AIRL Improvement", improvement)
    logger.record_tabular("AIRL Training Time", training_time)
    logger.record_tabular("AIRL Expert Trajectories", len(expert_trajectories))
    logger.dump_tabular()
    
    print("AIRL training completed!")
    
    # Close environment
    venv.close()

if __name__ == "__main__":
    main()
