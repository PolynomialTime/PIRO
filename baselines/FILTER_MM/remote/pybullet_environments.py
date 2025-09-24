import gym
import pybullet_envs  # 这一步会自动注册 PyBullet 相关环境到 Gym

all_envs = [env_spec.id for env_spec in gym.envs.registry.all()
            if "Bullet" in env_spec.id]
for env_id in sorted(all_envs):
    print(env_id)
