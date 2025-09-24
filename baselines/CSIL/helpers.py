

"""Helpers for experiments."""

import itertools
from typing import Any, Callable, Iterator

import chex
import dm_env
import gym
import jax
import jax.numpy as jnp
import numpy as np
import rlds
import tensorflow as tf
import tree
from acme import specs, types, wrappers
from acme.datasets import tfds
from acme.jax import utils
from sil import learning
import d4rl  # Add d4rl import


ImitationIterator = Callable[
    [int, jax.Array], Iterator[learning.ImitationSample]
]
TransitionIterator = Callable[
    [int, jax.Array], Iterator[types.Transition]
]

# RLDS TFDS file names, see www.tensorflow.org/datasets/catalog/
EXPERT_DATASET_NAMES = {
    'HalfCheetah-v2': 'locomotion/halfcheetah_sac_1M_single_policy_stochastic',
    'Ant-v2': 'locomotion/ant_sac_1M_single_policy_stochastic',
    'Walker2d-v2': 'locomotion/walker2d_sac_1M_single_policy_stochastic',
    'Hopper-v2': 'locomotion/hopper_sac_1M_single_policy_stochastic',
    'Humanoid-v2': 'locomotion/humanoid_sac_15M_single_policy_stochastic',
    'door-v0': 'd4rl_adroit_door/v0-expert',
    'hammer-v0': 'd4rl_adroit_hammer/v0-expert',
    'pen-v0': 'd4rl_adroit_pen/v0-expert',
}

OFFLINE_DATASET_NAMES = {
    'HalfCheetah-v2': 'd4rl_mujoco_halfcheetah/v2-full-replay',
    'Ant-v2': 'd4rl_mujoco_ant/v2-full-replay',
    'Walker2d-v2': 'd4rl_mujoco_walker2d/v2-full-replay',
    'Hopper-v2': 'd4rl_mujoco_hopper/v2-full-replay',
    # No alternative dataset for Humanoid-v2.
    'Humanoid-v2': 'locomotion/humanoid_sac_15M_single_policy_stochastic',
    # Adroit doesn't have suboptimal datasets, so use human demos as alternative
    # the other options are -human or -cloned (50/50 human and agent).
    # Human data is hard for BC to learn from so we compromise with expert.
    'door-v0': 'd4rl_adroit_door/v0-cloned',
    'hammer-v0': 'd4rl_adroit_hammer/v0-cloned',
    'pen-v0': 'd4rl_adroit_pen/v0-cloned',
}


class RandomIterator(Iterator[Any]):

  def __init__(self, dataset: tf.data.Dataset, batch_size: int, seed: int):
    dataset = dataset.shuffle(buffer_size=batch_size * 2, seed=seed)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    self.iterator = itertools.cycle(dataset.as_numpy_iterator())

  def __next__(self) -> Any:
    return self.iterator.__next__()


class OfflineImitationIterator(Iterator[learning.ImitationSample]):
  """ImitationSample iterator for offline IL."""

  def __init__(
      self,
      expert_iterator: Iterator[types.Transition],
      offline_iterator: Iterator[types.Transition],
  ):
    self._expert = expert_iterator
    self._offline = offline_iterator

  def __next__(self) -> learning.ImitationSample:
    """Combine data streams into an ImitationSample iterator."""
    return learning.ImitationSample(
        online_sample=self._offline.__next__(),
        demonstration_sample=self._expert.__next__(),
    )


class MixedIterator(Iterator[types.Transition]):
  """Combine two streams of transitions 50/50."""

  def __init__(
      self,
      first_iterator: Iterator[types.Transition],
      second_iterator: Iterator[types.Transition],
      with_extras: bool = False,
  ):
    self._first = first_iterator
    self._second = second_iterator
    self.with_extras = with_extras

  def __next__(self) -> types.Transition:
    """Combine data streams 50/50 into one, with the equal batch size."""
    combined = tree.map_structure(
        lambda x, y: jnp.concatenate((x, y), axis=0),
        self._first.__next__(),
        self._second.__next__(),
    )
    return tree.map_structure(lambda x: x[::2, ...], combined)


def get_dataset_name(env_name: str, expert: bool = True) -> str:
  """Map environment to an expert or non-expert dataset name."""
  if expert:
    assert (
        env_name in EXPERT_DATASET_NAMES
    ), f"Choose from {', '.join(EXPERT_DATASET_NAMES)}"
    return EXPERT_DATASET_NAMES[env_name]
  else:  # Arbitrary offline data.
    assert (
        env_name in OFFLINE_DATASET_NAMES
    ), f"Choose from {', '.join(OFFLINE_DATASET_NAMES)}"
    return OFFLINE_DATASET_NAMES[env_name]


def add_next_action_extras(
    transitions_iterator: tf.data.Dataset,
) -> tf.data.Dataset:
  """Creates transitions with next-action as extras information."""

  def _add_next_action_extras(
      double_transitions: types.Transition,
  ) -> types.Transition:
    """Creates a new transition containing the next action in extras."""
    # Observations may be dictionary or ndarray.
    get_obs = lambda x: tree.map_structure(lambda y: y[0], x)
    obs = get_obs(double_transitions.observation)
    next_obs = get_obs(double_transitions.next_observation)
    return types.Transition(
        observation=obs,
        action=double_transitions.action[0],
        reward=double_transitions.reward[0],
        discount=double_transitions.discount[0],
        next_observation=next_obs,
        extras={'next_action': double_transitions.action[1]},
    )

  double_transitions = rlds.transformations.batch(
      transitions_iterator, size=2, shift=1, drop_remainder=True
  )
  return double_transitions.map(_add_next_action_extras)


def get_offline_dataset(
    task: str,
    environment_spec: specs.EnvironmentSpec,
    expert_num_demonstration: int,
    offline_num_demonstrations: int,
    expert_offline_data: bool = False,
    use_sarsa: bool = False,
    in_memory: bool = True,
) -> tuple[ImitationIterator, TransitionIterator]:
  """Get the offline dataset for a given task."""
  expert_dataset_name = get_dataset_name(task, expert=True)
  offline_dataset_name = get_dataset_name(task, expert=expert_offline_data)

  # Note: For offline learning we take the key, not a seed.
  def make_offline_dataset(
      batch_size: int, key: jax.Array
  ) -> Iterator[types.Transition]:
    offline_transitions_iterator = tfds.get_tfds_dataset(
        offline_dataset_name,
        offline_num_demonstrations,
        env_spec=environment_spec,
    )
    if use_sarsa:
      offline_transitions_iterator = add_next_action_extras(
          offline_transitions_iterator
      )
    if in_memory:
      return tfds.JaxInMemoryRandomSampleIterator(
          dataset=offline_transitions_iterator, key=key, batch_size=batch_size
      )
    else:
      return RandomIterator(
          offline_transitions_iterator, batch_size, seed=int(key))

  def make_imitation_dataset(
      batch_size: int,
      key: jax.Array,
  ) -> Iterator[learning.ImitationSample]:
    expert_transitions_iterator = tfds.get_tfds_dataset(
        expert_dataset_name, expert_num_demonstration, env_spec=environment_spec
    )
    if use_sarsa:
      expert_transitions_iterator = add_next_action_extras(
          expert_transitions_iterator
      )
    if in_memory:
      expert_iterator = tfds.JaxInMemoryRandomSampleIterator(
          dataset=expert_transitions_iterator, key=key, batch_size=batch_size
      )
    else:
      expert_iterator = RandomIterator(
          expert_transitions_iterator, batch_size, seed=int(key))
    offline_iterator = make_offline_dataset(batch_size, key)
    return OfflineImitationIterator(expert_iterator, offline_iterator)

  return make_imitation_dataset, make_offline_dataset


def get_d4rl_dataset(
    task: str,
    environment_spec: specs.EnvironmentSpec,
    expert_num_demonstration: int,
    offline_num_demonstrations: int,
    expert_offline_data: bool = False,
    use_sarsa: bool = False,
    in_memory: bool = True,
) -> tuple[ImitationIterator, TransitionIterator]:
  """Get the offline dataset using D4RL directly, bypassing envlogger."""
  
  # D4RL environment name mapping
  d4rl_task_map = {
    'HalfCheetah-v2': 'halfcheetah',
    'Ant-v2': 'ant', 
    'Walker2d-v2': 'walker2d',
    'Hopper-v2': 'hopper',
  }
  
  if task not in d4rl_task_map:
    # Fall back to original method for unsupported tasks
    return get_offline_dataset(task, environment_spec, expert_num_demonstration, 
                             offline_num_demonstrations, expert_offline_data, use_sarsa, in_memory)
  
  d4rl_name = d4rl_task_map[task]
  
  def load_d4rl_data(env_name: str, max_samples: int = None):
    """Load data from D4RL environment."""
    env = gym.make(env_name)
    dataset = env.get_dataset()
    
    # Convert to transitions
    observations = dataset['observations']
    actions = dataset['actions'] 
    rewards = dataset['rewards']
    next_observations = dataset['next_observations']
    terminals = dataset['terminals']
    
    if max_samples and len(observations) > max_samples:
      indices = np.random.choice(len(observations), max_samples, replace=False)
      observations = observations[indices]
      actions = actions[indices]
      rewards = rewards[indices] 
      next_observations = next_observations[indices]
      terminals = terminals[indices]
    
    # Create transitions
    transitions = []
    for i in range(len(observations)):
      transition = types.Transition(
        observation=observations[i],
        action=actions[i],
        reward=rewards[i],
        discount=1.0 - float(terminals[i]),
        next_observation=next_observations[i],
        extras={}
      )
      transitions.append(transition)
    
    return transitions

  def make_offline_dataset(batch_size: int, key: jax.Array) -> Iterator[types.Transition]:
    """Create offline dataset iterator using D4RL."""
    if expert_offline_data:
      env_name = f'{d4rl_name}-expert-v2'
    else:
      env_name = f'{d4rl_name}-medium-replay-v2'  # Use medium-replay as alternative
    
    transitions = load_d4rl_data(env_name, offline_num_demonstrations)
    
    # Create iterator that cycles through batches
    def batch_iterator():
      while True:
        # Shuffle transitions
        indices = np.random.permutation(len(transitions))
        for i in range(0, len(transitions), batch_size):
          batch_indices = indices[i:i+batch_size]
          if len(batch_indices) == batch_size:  # Only yield full batches
            yield [transitions[j] for j in batch_indices]
    
    return batch_iterator()

  def make_imitation_dataset(batch_size: int, key: jax.Array) -> Iterator[learning.ImitationSample]:
    """Create imitation dataset iterator using D4RL."""
    # Load expert data
    expert_env_name = f'{d4rl_name}-expert-v2'
    expert_transitions = load_d4rl_data(expert_env_name, expert_num_demonstration)
    
    # Load offline data  
    offline_iterator = make_offline_dataset(batch_size, key)
    
    # Create expert iterator
    def expert_batch_iterator():
      while True:
        indices = np.random.permutation(len(expert_transitions))
        for i in range(0, len(expert_transitions), batch_size):
          batch_indices = indices[i:i+batch_size]
          if len(batch_indices) == batch_size:
            yield [expert_transitions[j] for j in batch_indices]
    
    expert_iterator = expert_batch_iterator()
    
    return OfflineImitationIterator(expert_iterator, offline_iterator)

  return make_imitation_dataset, make_offline_dataset


def get_env_and_demonstrations(
    task: str,
    num_demonstrations: int,
    expert: bool = True,
    use_sarsa: bool = False,
    in_memory: bool = True,
) -> tuple[
    Callable[[], dm_env.Environment],
    specs.EnvironmentSpec,
    Callable[[int, int], Iterator[types.Transition]],
]:
  """Returns environment, spec and expert demonstration iterator."""
  make_env = make_environment(task)

  environment_spec = specs.make_environment_spec(make_env())

  # Create demonstrations function.
  dataset_name = get_dataset_name(task, expert=expert)

  def make_demonstrations(
      batch_size: int, seed: int = 0
  ) -> Iterator[types.Transition]:
    transitions_iterator = tfds.get_tfds_dataset(
        dataset_name, num_demonstrations, env_spec=environment_spec
    )
    if use_sarsa:
      transitions_iterator = add_next_action_extras(transitions_iterator)
    if in_memory:
      return tfds.JaxInMemoryRandomSampleIterator(
          dataset=transitions_iterator,
          key=jax.random.PRNGKey(seed),
          batch_size=batch_size,
      )
    else:
      return RandomIterator(
          transitions_iterator, batch_size, seed=seed)
  return make_env, environment_spec, make_demonstrations


def get_env_and_demonstrations_d4rl(
    task: str,
    num_demonstrations: int,
    expert: bool = True,
    use_sarsa: bool = False,
    in_memory: bool = True,
) -> tuple[
    Callable[[], dm_env.Environment],
    specs.EnvironmentSpec,
    Callable[[int, int], Iterator[types.Transition]],
]:
  """Get environment and demonstrations using D4RL, bypassing envlogger."""
  
  # D4RL environment name mapping
  d4rl_task_map = {
    'HalfCheetah-v2': 'halfcheetah',
    'Ant-v2': 'ant', 
    'Walker2d-v2': 'walker2d',
    'Hopper-v2': 'hopper',
  }
  
  if task not in d4rl_task_map:
    # Fall back to original method for unsupported tasks
    return get_env_and_demonstrations(task, num_demonstrations, expert, use_sarsa, in_memory)
  
  d4rl_name = d4rl_task_map[task]
  
  def make_env() -> dm_env.Environment:
    env = gym.make(task)
    env = wrappers.GymWrapper(env)
    return env

  env = make_env()
  environment_spec = specs.make_environment_spec(env)

  def make_demonstrations(
      batch_size: int, seed: int = 0
  ) -> Iterator[types.Transition]:
    # Use D4RL to load demonstrations
    if expert:
      env_name = f'{d4rl_name}-expert-v2'
    else:
      env_name = f'{d4rl_name}-medium-v2'
    
    print(f"📚 Loading D4RL dataset: {env_name}")
    # Load data from D4RL
    d4rl_env = gym.make(env_name)
    dataset = d4rl_env.get_dataset()
    print(f"✅ Dataset loaded: {len(dataset['observations'])} transitions")
    
    # Convert to transitions
    observations = dataset['observations']
    actions = dataset['actions'] 
    rewards = dataset['rewards']
    next_observations = dataset['next_observations']
    terminals = dataset['terminals']
    
    # Limit to requested number of demonstrations
    if len(observations) > num_demonstrations:
      indices = np.random.choice(len(observations), num_demonstrations, replace=False)
      observations = observations[indices]
      actions = actions[indices]
      rewards = rewards[indices] 
      next_observations = next_observations[indices]
      terminals = terminals[indices]
    
    # Store raw data for faster batch creation
    data = {
      'observations': observations,
      'actions': actions,
      'rewards': rewards,
      'terminals': terminals,
      'next_observations': next_observations
    }
    
    # Create a simple iterator class
    class FastBatchIterator:
      def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.num_samples = len(data['observations'])
      
      def __iter__(self):
        return self
      
      def __next__(self):
        # Get random batch indices (allow replacement if needed)
        replace_needed = self.batch_size > self.num_samples
        indices = np.random.choice(self.num_samples, self.batch_size, replace=replace_needed)
        
        # Create batched transition (not a list of transitions)
        # For SARSA, we need next_action in extras - use current action as approximation
        batch_transition = types.Transition(
          observation=self.data['observations'][indices],
          action=self.data['actions'][indices],
          reward=self.data['rewards'][indices],
          discount=1.0 - self.data['terminals'][indices].astype(float),
          next_observation=self.data['next_observations'][indices],
          extras={'next_action': self.data['actions'][indices]}  # Use current action as next_action approximation
        )
        return batch_transition
    
    return FastBatchIterator(data, batch_size)
    
  return make_env, environment_spec, make_demonstrations


def make_environment(task: str) -> Callable[[], dm_env.Environment]:
  """Makes the requested continuous control environment.

  Args:
    task: Task to load.

  Returns:
    An environment satisfying the dm_env interface expected by Acme agents.
  """

  def make_env():
    env = gym.make(task)
    env = wrappers.GymWrapper(env)
    # Make sure the environment obeys the dm_env.Environment interface.

    # Wrap the environment so the expected continuous action spec is [-1, 1].
    # Note: This is a no-op on 'control' tasks.
    env = wrappers.CanonicalSpecWrapper(env, clip=True)
    env = wrappers.SinglePrecisionWrapper(env)
    return env

  env = make_env()
  # env.render(mode='rgb_array')  # Disabled for headless server environment

  return make_env
