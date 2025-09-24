#!/usr/bin/env python3
"""
Behavior Cloning using Imitation library with proper API usage.
Based on official imitation documentation.
"""

import sys, os, time
import numpy as np
import torch
import gymnasium as gym
from ruamel.yaml import YAML
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.types import Trajectory
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
        # Note: observations include one extra state (the final state)
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
    for i, traj in enumerate(trajectories[:3]):  # Show first 3 trajectories
        print(f"  Trajectory {i}: obs.shape={traj.obs.shape}, acts.shape={traj.acts.shape}")
    
    return trajectories

def make_env_with_gymnasium(env_name):
    """Create environment using gymnasium API"""
    
    # Check if the environment exists in gymnasium
    try:
        # Try to make the environment directly
        env = gym.make(env_name)
        print(f"Created gymnasium environment: {env_name}")
        return env
    except:
        # If not available, we need to convert from the old gym
        print(f"Environment {env_name} not available in gymnasium, using gym conversion")
        
        # Import the old gym for environment creation
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
        print(f"Converted gym environment to gymnasium: {env_name}")
        return wrapped_env

def main():
    if len(sys.argv) != 2:
        print("Usage: python bc_imitation_trainer.py <config_file>")
        sys.exit(1)
    
    # Load configuration
    yaml = YAML()
    v = yaml.load(open(sys.argv[1]))
    
    # Extract parameters
    env_name = v['env']['env_name']
    state_indices = v['env']['state_indices']
    seed = v['seed']
    
    # Generate random seed if seed is -1
    if seed == -1:
        seed = int(time.time() * 1000) % 100000  # Generate random seed from timestamp
        print(f"Generated random seed: {seed}")
    
    bc_config = v['bc']
    
    # System setup
    device = torch.device(f"cuda:{v['cuda']}" if torch.cuda.is_available() and v['cuda'] >= 0 else "cpu")
    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    system.reproduce(seed)
    
    print(f"Using device: {device}")
    print(f"Environment: {env_name}")
    print(f"BC Config: {bc_config}")
    
    # Create environment
    rng = np.random.default_rng(seed)
    
    # Create single environment and wrap it manually
    single_env = make_env_with_gymnasium(env_name)
    
    # Create vectorized environment manually using DummyVecEnv
    venv = DummyVecEnv([lambda: make_env_with_gymnasium(env_name)])
    
    print(f"Environment spaces:")
    print(f"  Observation space: {venv.observation_space}")
    print(f"  Action space: {venv.action_space}")
    
    # Logging setup
    exp_id = f"baselines/logs/{env_name}/exp-{bc_config['expert_episodes']}/bc"
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
    expert_data_dir = bc_config.get('expert_data_path', 'expert_data')
    print(f"Expert data directory: {expert_data_dir}")
    trajectories = load_expert_trajectories(
        env_name, 
        bc_config['expert_episodes'],
        expert_data_dir
    )
    
    # Convert trajectories to transitions for BC
    print("Converting trajectories to transitions...")
    transitions = rollout.flatten_trajectories(trajectories)
    print(f"Total transitions: {len(transitions.obs)}")
    print(f"Obs shape: {transitions.obs.shape}")
    print(f"Acts shape: {transitions.acts.shape}")
    
    # Create BC trainer
    print("Creating BC trainer...")
    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        demonstrations=transitions,
        rng=rng,
        batch_size=v.get('bc_network', {}).get('batch_size', 256),
        optimizer_kwargs={
            'lr': v.get('bc_network', {}).get('lr', 1e-3)
        },
        ent_weight=v.get('bc_network', {}).get('ent_weight', 1e-3),
        l2_weight=v.get('bc_network', {}).get('l2_weight', 0.0),
        device=str(device)
    )
    
    # Training loop with evaluation
    print(f"Starting BC training for {bc_config['epochs']} epochs...")
    
    eval_freq = bc_config.get('eval_freq', 100)
    eval_episodes = bc_config.get('eval_episodes', 10)
    
    for epoch in range(0, bc_config['epochs'], eval_freq):
        # Train for eval_freq epochs
        epochs_to_train = min(eval_freq, bc_config['epochs'] - epoch)
        
        print(f"Training epochs {epoch} to {epoch + epochs_to_train}...")
        bc_trainer.train(n_epochs=epochs_to_train)
        
        # Evaluate policy
        print(f"Evaluating at epoch {epoch + epochs_to_train}...")
        try:
            mean_reward, std_reward = evaluate_policy(
                bc_trainer.policy,
                venv,
                n_eval_episodes=eval_episodes,
                deterministic=True
            )
            print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        except Exception as e:
            print(f"  Evaluation failed: {e}")
            mean_reward, std_reward = 0.0, 0.0
        
        # Log results
        logger.record_tabular("BC Epoch", epoch + epochs_to_train)
        logger.record_tabular("BC Mean Reward", mean_reward)
        logger.record_tabular("BC Std Reward", std_reward)
        logger.dump_tabular()
        
        # Save intermediate model
        if (epoch + epochs_to_train) % 1000 == 0:
            model_path = os.path.join(log_folder, 'model', f'bc_policy_epoch_{epoch + epochs_to_train}.zip')
            bc_trainer.policy.save(model_path)
            print(f"Saved intermediate model to: {model_path}")
    
    # Final evaluation
    print("Final evaluation...")
    try:
        final_mean_reward, final_std_reward = evaluate_policy(
            bc_trainer.policy,
            venv,
            n_eval_episodes=eval_episodes * 2,  # More episodes for final eval
            deterministic=True
        )
        print(f"Final BC Performance:")
        print(f"  Mean reward: {final_mean_reward:.2f} ± {final_std_reward:.2f}")
    except Exception as e:
        print(f"Final evaluation failed: {e}")
        final_mean_reward, final_std_reward = 0.0, 0.0
    
    # Save the trained policy
    model_path = os.path.join(log_folder, 'model', 'bc_policy_final.zip')
    bc_trainer.policy.save(model_path)
    print(f"BC policy saved to: {model_path}")
    
    # Save final results
    results_path = os.path.join(log_folder, 'bc_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'final_mean_reward': final_mean_reward,
            'final_std_reward': final_std_reward,
            'num_trajectories': len(trajectories),
            'num_transitions': len(transitions.obs),
            'epochs_trained': bc_config['epochs']
        }, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    
    # Final logging
    logger.record_tabular("BC Final Mean Reward", final_mean_reward)
    logger.record_tabular("BC Final Std Reward", final_std_reward)
    logger.record_tabular("BC Trajectories", len(trajectories))
    logger.record_tabular("BC Transitions", len(transitions.obs))
    logger.dump_tabular()
    
    print("BC training completed!")
    
    # Close environment
    venv.close()

if __name__ == "__main__":
    main()
