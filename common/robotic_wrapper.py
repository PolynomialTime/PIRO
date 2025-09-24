import gym                  # Original Gym
import gymnasium as gymn    # New Gymnasium
from gymnasium import spaces
from gym import error as gym_error
import gymnasium_robotics   # Retain the robotics registrations you need

# Register robotics envs first
gymn.register_envs(gymnasium_robotics)


class StepReturnWrapper(gym.Wrapper):
    """
    Normalize various reset/step signatures to:
      reset() -> obs
      step(a) -> (obs, reward, done, info)
    And add seed() to ensure the outer .seed() call works.
    """
    def reset(self, *args, **kwargs):
        ret = self.env.reset(*args, **kwargs)
        if isinstance(ret, tuple):
            obs, _ = ret
        else:
            obs = ret
        return obs

    def step(self, action):
        ret = self.env.step(action)
        if len(ret) == 5:
            obs, reward, terminated, truncated, info = ret
            done = terminated or truncated
            return obs, reward, done, info
        elif len(ret) == 4:
            return ret
        else:
            raise RuntimeError(f"Unrecognized step return, got {len(ret)} values")

    def seed(self, seed=None):
        """
        Intercept outer .seed() calls, first try to pass to the base env;
        if the base env is also a Wrapper, unwrap down to the unwrapped env.
        """
        # Find the underlying env that actually has a seed() method
        base = self.env
        while hasattr(base, 'env'):
            base = base.env
        if hasattr(base, 'seed'):
            return base.seed(seed)
        # If none, try the old trick: reset(seed=…)
        try:
            self.reset(seed=seed)
        except TypeError:
            pass


class ArrayObsWrapper(gym.Wrapper):
    """
    Extract only the obs_dict['observation'] part; while supporting:
      - Original Gym:  step -> (obs_dict, reward, done, info)
      - Gymnasium:     step -> (obs_dict, reward, terminated, truncated, info)
    """
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Dict), \
            f"Expected Dict obs space, got {env.observation_space}"
        self.observation_space = env.observation_space.spaces['observation']
        self.action_space = env.action_space

    def reset(self, **kwargs):
        ret = self.env.reset(**kwargs)
        # Gym.reset() -> obs_dict
        # Gymnasium.reset() -> (obs_dict, info)
        if isinstance(ret, tuple) and len(ret) == 2:
            obs_dict, info = ret
        else:
            obs_dict, info = ret, {}
        return obs_dict['observation'], info

    def step(self, action):
        ret = self.env.step(action)
        # Five-tuple: Gymnasium
        if len(ret) == 5:
            obs_dict, r, terminated, truncated, info = ret
            done = terminated or truncated
        # Four-tuple: Old Gym or already unified by an outer Wrapper
        elif len(ret) == 4:
            obs_dict, r, done, info = ret
        else:
            raise RuntimeError(f"Unrecognized step return, got {len(ret)} values")
        return obs_dict['observation'], r, done, info


def make_array_env(env_or_fn):
    """
    - If given a constructor, return a thunk: build and wrap the env on call.
    - If given an env instance:
       1. Wrap dict-observation env with ArrayObsWrapper
       2. Then wrap with StepReturnWrapper (to ensure uniform reset/step signatures)
    """
    if callable(env_or_fn):
        def _thunk(*a, **kw):
            e = env_or_fn(*a, **kw)
            return make_array_env(e)
        return _thunk

    env = env_or_fn
    if isinstance(env.observation_space, spaces.Dict):
        env = ArrayObsWrapper(env)
    env = StepReturnWrapper(env)
    return env


def make_env_fn(env_name: str):
    """
    Zero-argument function: try gym.make first; if NameNotFound, fall back to gymnasium.make
    """
    def _fn():
        try:
            return gym.make(env_name)
        except gym_error.NameNotFound:
            return gymn.make(env_name)
    return _fn
