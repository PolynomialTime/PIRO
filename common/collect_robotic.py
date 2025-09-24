#!/usr/bin/env python3
import numpy as np
import minari


def download_and_convert(dataset_id="D4RL/pen/expert-v2"):
    # 1. Load and download the dataset
    dataset = minari.load_dataset(dataset_id, download=True)

    # 2. Prepare containers
    states_l = []
    actions_l = []
    rewards_l = []
    next_states_l = []
    dones_l = []
    len_l = []

    # 3. Iterate over episodes
    for episode in dataset.iterate_episodes():
        # Get observations; if it's a dict, use only the 'observation' key
        obs = episode.observations
        if isinstance(obs, dict):
            obs_data = obs.get('observation')
            if obs_data is None:
                raise KeyError("Dictionary observations do not contain key 'observation'.")
        else:
            obs_data = obs

        acts = episode.actions  # ndarray, shape=(T, act_dim)
        rews = episode.rewards  # ndarray, shape=(T,)
        terms = episode.terminations  # ndarray of bool, shape=(T,)

        # Split into (state, action, reward, next_state, done)
        states_l.append(obs_data[:-1])
        next_states_l.append(obs_data[1:])
        actions_l.append(acts)
        rewards_l.append(rews)
        dones_l.append(terms.astype(bool))

        # Record the length of each trajectory
        len_l.append(len(rews))

    # 4. Assemble and save as .npy files
    states_arr = np.array(states_l, dtype=object)
    actions_arr = np.array(actions_l, dtype=object)
    rewards_arr = np.array(rewards_l, dtype=object)
    next_states_arr = np.array(next_states_l, dtype=object)
    dones_arr = np.array(dones_l, dtype=object)
    lengths_arr = np.array(len_l, dtype=object)

    np.save('states.npy', states_arr)
    np.save('actions.npy', actions_arr)
    np.save('rewards.npy', rewards_arr)
    np.save('next_states.npy', next_states_arr)
    np.save('dones.npy', dones_arr)
    np.save('lengths.npy', lengths_arr)

    # 5. Compute and print the average cumulative reward
    # Calculate the cumulative reward for each trajectory
    returns = [np.sum(rew) for rew in rewards_l]
    avg_return = np.mean(returns) if returns else float('nan')
    print(f"Average cumulative reward over {len(returns)} episodes: {avg_return}")

    print(
        "Done! Saved files: states.npy, actions.npy, rewards.npy,\n                 "
        "next_states.npy, dones.npy, lengths.npy")


if __name__ == "__main__":
    # Modify to the desired dataset, e.g., "D4RL/antmaze-umaze-v4/expert-v2"
    download_and_convert(dataset_id="D4RL/pen/human-v2")
