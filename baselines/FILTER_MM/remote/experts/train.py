import numpy as np
import gym
import torch
import os
from stable_baselines3 import SAC, PPO
from typing import Callable, Union
from gym.wrappers import FlattenObservation
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

import argparse

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


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func

def make_env_conditional(env_id):
    """
    只有离散观测（如 FrozenLake）时才加 FlattenObservation，
    其余环境原样返回。
    """
    env = gym.make(env_id)
    if isinstance(env.observation_space, gym.spaces.Discrete):
        env = FlattenObservation(env)
    return env

def get_state(env):
    p = env.unwrapped._p
    base_pos = [] # position and orientation of base for each body
    base_vel = [] # velocity of base for each body
    joint_states = [] # joint states for each body
    for i in range(p.getNumBodies()):
        base_pos.append(p.getBasePositionAndOrientation(i))
        base_vel.append(p.getBaseVelocity(i))
        joint_states.append([p.getJointState(i,j) for j in range(p.getNumJoints(i))])
    return base_pos, base_vel, joint_states

def get_acrobot_state(env):
    # env.unwrapped.state: [theta1, theta2, thetaDot1, thetaDot2]
    theta1, theta2, theta_dot1, theta_dot2 = env.unwrapped.state

    # 想要和原先的 p, v, j 结构对应的话，可以自定义：
    # p: position-like (这里可以简单把两段摆的角度当作“位置”)
    # v: velocity-like (这里就是角速度)
    # j: joint states (如果想要更丰富的信息，也可以只把两个角度和速度都当作“关节状态”)

    p = [theta1, theta2]
    v = [theta_dot1, theta_dot2]
    j = None  # 如果你需要的话，也可以把 (theta1, theta2, theta_dot1, theta_dot2) 全部放到 j

    return p, v, j

def get_atari_state(env):

    p = None
    v = None
    j = None  # 如果你需要的话，也可以把 (theta1, theta2, theta_dot1, theta_dot2) 全部放到 j

    return p, v, j

def get_bipidel_state(env):
    unwrapped = env.unwrapped
    hull = unwrapped.hull  # BipedalWalker 的主体对象

    # 将 hull 的位置、角度、速度转为纯 Python 数值
    base_pos = (float(hull.position[0]), float(hull.position[1]), float(hull.angle))
    base_vel = (float(hull.linearVelocity[0]), float(hull.linearVelocity[1]), float(hull.angularVelocity))

    # 对于关节，遍历所有 joints，将角度和角速度提取为浮点数
    joint_states = []
    for joint in unwrapped.joints:
        # 注意：不同版本可能属性名略有不同，如 joint.angle 或 joint.GetJointAngle()
        joint_angle = float(joint.angle)
        joint_speed = float(joint.speed)
        joint_states.append((joint_angle, joint_speed))

    return base_pos, base_vel, joint_states

def get_lunarlander_state(env):
    lander = env.unwrapped.lander
    # 主体
    base_pos = (float(lander.position[0]), float(lander.position[1]), float(lander.angle))
    base_vel = (float(lander.linearVelocity[0]), float(lander.linearVelocity[1]), float(lander.angularVelocity))

    # 两条腿
    leg_states = []
    for leg in env.unwrapped.legs:
        pos = (float(leg.position[0]), float(leg.position[1]), float(leg.angle))
        vel = (float(leg.linearVelocity[0]), float(leg.linearVelocity[1]), float(leg.angularVelocity))
        leg_states.append((pos, vel))

    return base_pos, base_vel, leg_states

def get_cartpole_state(env):
    """
    模拟从 CartPole-v1 中获取类似于 (base_pos, base_vel, joint_states) 的结构。
    """
    x, x_dot, theta, theta_dot = env.unwrapped.state

    base_pos = [
        # (pos, orientation) 也可以再拆成两部分
        # 这里仅示例性给出：位置用 (x, 0.0, 0.0)，朝向用单位四元数 (0,0,0,1) 等
        ((x, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    ]

    # 类似地，base_vel 可以做成 (线速度, 角速度) 的形式
    base_vel = [
        ((x_dot, 0.0, 0.0), (0.0, 0.0, 0.0))
    ]

    joint_states = [
        (theta, theta_dot)
    ]

    return base_pos, base_vel, joint_states


def get_mujoco_state(env):
    # 1) 拿到底层仿真数据对象
    sim_data = env.unwrapped.sim.data  # MuJoCo MjData 对象

    # 2) 解析基座 (torso) 的位姿和速度
    # qpos[:7] = [px, py, pz, qw, qx, qy, qz] (MuJoCo 的 quaternion 顺序可能略有不同)
    base_pos = sim_data.qpos[:7].copy()  # (xyz + 四元数)
    base_vel = sim_data.qvel[:6].copy()  # (vx, vy, vz, wx, wy, wz)

    # 3) 解析各关节的 (角度, 角速度)
    # qpos[7:], qvel[6:] 就是各关节的
    joint_angles = sim_data.qpos[7:].copy()
    joint_vels   = sim_data.qvel[6:].copy()

    # 整理成 (angle, vel) 对
    joint_states = list(zip(joint_angles, joint_vels))

    return base_pos, base_vel, joint_states

def get_frozenlake_state(env):
    """
    FrozenLake 没有真正的物理信息，这里把格子坐标当作 'p'，
    用全 0 的 'v' 占位，'j' 为空数组以保持接口一致。
    """
    idx = env.unwrapped.s              # 当前格子序号: 0 ~ (nrow*ncol-1)
    ncol = env.unwrapped.ncol
    row, col = divmod(idx, ncol)
    p = np.array([row, col], dtype=np.float32)
    v = np.zeros_like(p)
    j = np.array([], dtype=np.float32)
    return p, v, j

# ---------- 2. 把任何 obs / action 变成 1‑D ndarray ----------
def _to_1d(x):
    """兼容 int / list / tuple / ndarray，返回一维 float 数组"""
    if isinstance(x, np.ndarray):
        return x.reshape(-1).astype(np.float32)
    else:                               # 离散整型等
        return np.array([x], dtype=np.float32)

# ---------- 3. 主 rollout ----------
def rollout(pi, env, full_state=False):
    states, actions = [], []

    if full_state:
        body_pos, body_vel, joint_states = [], [], []

    # Gymnasium ≥0.26: reset() -> (obs, info)
    s = env.reset()
    if isinstance(s, tuple):            # 向后兼容
        s, _ = s

    # -------- 3‑A. 第 0 步时的“全状态” --------
    if full_state:
        if isinstance(env.unwrapped, AcrobotEnv):
            p, v, j = get_acrobot_state(env.env)
        elif isinstance(env.unwrapped, BipedalWalker):
            p, v, j = get_bipidel_state(env.env)
        elif isinstance(env.unwrapped, LunarLander):
            p, v, j = get_lunarlander_state(env.env)
        elif isinstance(env.unwrapped, (HumanoidEnv, AntEnv, Walker2dEnv, HopperEnv, HalfCheetahEnv)):
            p, v, j = get_mujoco_state(env.env)
        elif isinstance(env.unwrapped, AtariEnv):
            p, v, j = get_atari_state(env.env)
        elif isinstance(env.unwrapped, CartPoleEnv):
            p, v, j = get_cartpole_state(env.env)
        elif isinstance(env.unwrapped, FrozenLakeEnv):         # <‑‑ NEW
            p, v, j = get_frozenlake_state(env.env)
        else:
            p, v, j = get_state(env.env)

        body_pos.append(p); body_vel.append(v); joint_states.append(j)

    # -------- 3‑B. rollout 主循环 --------
    done, J, traj_length = False, 0.0, 0
    while not done:
        states.append(_to_1d(s))        # 兼容离散 / 连续
        a = pi(s)
        if isinstance(a, tuple):        # 有些策略 network 返回 (action, extra)
            a = a[0]
        actions.append(_to_1d(a))

        # Gymnasium ≥0.26: step() -> (obs, reward, terminated, truncated, info)
        step_ret = env.step(a)
        if len(step_ret) == 5:          # 新版
            s, r, terminated, truncated, _ = step_ret
            done = terminated or truncated
        else:                           # 老版
            s, r, done, _ = step_ret

        if full_state:
            if isinstance(env.unwrapped, AcrobotEnv):
                p, v, j = get_acrobot_state(env.env)
            elif isinstance(env.unwrapped, BipedalWalker):
                p, v, j = get_bipidel_state(env.env)
            elif isinstance(env.unwrapped, LunarLander):
                p, v, j = get_lunarlander_state(env.env)
            elif isinstance(env.unwrapped, (HumanoidEnv, AntEnv, Walker2dEnv, HopperEnv, HalfCheetahEnv)):
                p, v, j = get_mujoco_state(env.env)
            elif isinstance(env.unwrapped, AtariEnv):
                p, v, j = get_atari_state(env.env)
            elif isinstance(env.unwrapped, CartPoleEnv):
                p, v, j = get_cartpole_state(env.env)
            elif isinstance(env.unwrapped, FrozenLakeEnv):     # <‑‑ NEW
                p, v, j = get_frozenlake_state(env.env)
            else:
                p, v, j = get_state(env.env)
            body_pos.append(p); body_vel.append(v); joint_states.append(j)

        J += r
        traj_length += 1
        if traj_length == 128:          # Atari 之类需要截断
            done = True

    # -------- 3‑C. 打包返回 --------
    states  = np.stack(states).astype(np.float32)
    actions = np.stack(actions).astype(np.float32)

    if full_state:
        return states, actions, J, body_pos, body_vel, joint_states
    else:
        return states, actions, J

def train_halfcheetah_expert():
    # No env normalization.
    env = gym.make('HalfCheetahBulletEnv-v0')
    model = SAC('MlpPolicy', env, verbose=1,
                buffer_size=300000, batch_size=256, gamma=0.98, tau=0.02,
                train_freq=64, gradient_steps=64, ent_coef='auto', learning_rate=7.3e-4, 
                learning_starts=10000, policy_kwargs=dict(net_arch=[256, 256], log_std_init=-3),
                use_sde=True)
    model.learn(total_timesteps=1e6)
    model.save("experts/HalfCheetahBulletEnv-v0/expert")

def train_ant_expert():
    # No env normalization.
    env = gym.make('Ant-v3')
    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        verbose=1,
        n_steps=2048,
        batch_size=64,
        gamma=0.98,
        n_epochs=10,
        learning_rate=7.3e-4,
        policy_kwargs=dict(net_arch=[64, 64])
    )
    model.learn(total_timesteps=1e4)
    model.save("Ant-v3/expert")

def train_humanoid_expert():
    # No env normalization.
    env = gym.make('Humanoid-v3')
    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        verbose=1,
        n_steps=2048,
        batch_size=64,
        gamma=0.98,
        n_epochs=10,
        learning_rate=7.3e-4,
        policy_kwargs=dict(net_arch=[64, 64])
    )
    model.learn(total_timesteps=1e4)
    model.policy.to("cpu")
    model.save("Humanoid-v3/expert")

def train_walker_expert():
    # No env normalization.
    env = gym.make('Walker2d-v3')
    model = PPO(
        "MlpPolicy",
        env,
        device="cuda:2",
        verbose=1,
        n_steps=2048,
        batch_size=64,
        gamma=0.98,
        n_epochs=10,
        learning_rate=7.3e-4,
        policy_kwargs=dict(net_arch=[64, 64])
    )
    model.learn(total_timesteps=1e5)
    model.save("Walker2d-v3/expert")

def train_hopper_expert():
    # No env normalization.
    env = gym.make('Hopper-v3')
    model = PPO(
        "MlpPolicy",
        env,
        device="cuda:1",
        verbose=1,
        n_steps=2048,
        batch_size=64,
        gamma=0.98,
        n_epochs=10,
        learning_rate=7.3e-4,
        policy_kwargs=dict(net_arch=[64, 64])
    )
    model.learn(total_timesteps=1e5)
    model.save("Hopper-v3/expert")

def train_cheetah_expert():
    # No env normalization.
    env = gym.make('HalfCheetah-v3')
    model = PPO(
        "MlpPolicy",
        env,
        device="cuda:2",
        verbose=1,
        n_steps=2048,
        batch_size=64,
        gamma=0.98,
        n_epochs=10,
        learning_rate=7.3e-4,
        policy_kwargs=dict(net_arch=[64, 64])
    )
    model.learn(total_timesteps=1e5)
    model.save("HalfCheetah-v3/expert")

def train_acrobot_expert():
    env = gym.make('Acrobot-v1')
    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        verbose=1,
        n_steps=2048,
        batch_size=64,
        gamma=0.98,
        n_epochs=10,
        learning_rate=7.3e-4,
        policy_kwargs=dict(net_arch=[64, 64])
    )
    model.learn(total_timesteps=1e5)
    model.save("Acrobot-v1/expert")

def train_carpole_expert():
    # No env normalization.
    env = gym.make('CartPole-v1')
    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        verbose=1,
        n_steps=2048,
        batch_size=64,
        gamma=0.98,
        n_epochs=10,
        learning_rate=7.3e-4,
        policy_kwargs=dict(net_arch=[64, 64])
    )
    model.learn(total_timesteps=2e5)
    model.save("CartPole-v1/expert")

def train_bipedal_expert():
    env = gym.make('BipedalWalker-v3')
    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        verbose=1,
        n_steps=2048,
        batch_size=64,
        gamma=0.98,
        n_epochs=10,
        learning_rate=7.3e-4,
        policy_kwargs=dict(net_arch=[64, 64])
    )
    model.learn(total_timesteps=1e5)
    model.save("BipedalWalker-v3/expert")

def train_lunarlander_expert():
    env = gym.make('LunarLander-v2')
    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        verbose=1,
        n_steps=2048,
        batch_size=64,
        gamma=0.98,
        n_epochs=10,
        learning_rate=7.3e-4,
        policy_kwargs=dict(net_arch=[64, 64])
    )
    model.learn(total_timesteps=1e5)
    model.save("LunarLander-v2/expert")


def train_frozenlake_expert():
    env = make_env_conditional("FrozenLake-v1")
    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        verbose=1,
        n_steps=2048,
        batch_size=64,
        gamma=0.98,
        n_epochs=10,
        learning_rate=7.3e-4,
        policy_kwargs=dict(net_arch=[64, 64])
    )
    model.learn(total_timesteps=2e4)
    model.save("FrozenLake-v1/expert")


def train_pong_expert():
    env = gym.make('PongNoFrameskip-v4')
    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        verbose=1,
        n_steps=2048,
        batch_size=64,
        gamma=0.98,
        n_epochs=10,
        learning_rate=7.3e-4,
        policy_kwargs=dict(net_arch=[64, 64])
    )

    model.learn(total_timesteps=1e5)
    model.save("PongNoFrameskip-v4/expert")

def train_qbert_expert():
    env = gym.make('QbertNoFrameskip-v4')
    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        verbose=1,
        n_steps=2048,
        batch_size=64,
        gamma=0.98,
        n_epochs=10,
        learning_rate=7.3e-4,
        policy_kwargs=dict(net_arch=[64, 64])
    )
    model.learn(total_timesteps=2e4)
    model.save("QbertNoFrameskip-v4/expert")

def train_breakout_expert():
    env = gym.make('BreakoutNoFrameskip-v4')
    model = PPO(
        "MlpPolicy",
        env,
        device="cuda:1",
        verbose=1,
        n_steps=2048,
        batch_size=64,
        gamma=0.98,
        n_epochs=10,
        learning_rate=7.3e-4,
        policy_kwargs=dict(net_arch=[64, 64])
    )
    model.learn(total_timesteps=1e5)
    model.save("BreakoutNoFrameskip-v4/expert")

def train_space_expert():
    env = gym.make('SpaceInvadersNoFrameskip-v4')
    model = PPO(
        "MlpPolicy",
        env,
        device="cuda:2",
        verbose=1,
        n_steps=2048,
        batch_size=64,
        gamma=0.98,
        n_epochs=10,
        learning_rate=7.3e-4,
        policy_kwargs=dict(net_arch=[64, 64])
    )
    model.learn(total_timesteps=1e5)
    model.save("SpaceInvadersNoFrameskip-v4/expert")

def train_meerkat_expert():
    env = gym.make('Meerkat-v3')
    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        verbose=1,
        n_steps=2048,
        batch_size=64,
        gamma=0.98,
        n_epochs=10,
        learning_rate=7.3e-4,
        policy_kwargs=dict(net_arch=[64, 64])
    )
    model.learn(total_timesteps=1e4)
    model.save("Meerkat-v3/expert")

def generate_demos(env):
    model = PPO.load("{0}/expert".format(env), device="cuda:2")
    def expert(s):
        return model.predict(s, deterministic=True)
    tot = 0
    demo_s = []
    demo_a = []
    P = []
    V = []
    C = []
    demo_env = TremblingHandWrapper(make_env_conditional(env), p_tremble=0.1)
    for i in range(1):
        s_traj, a_traj, J, p, v, c = rollout(expert, demo_env, full_state=True)
        demo_s.append(s_traj)
        demo_a.append(a_traj)
        P.append(p)
        V.append(v)
        C.append(c)
        tot += J
        print(f"{i} traj completed")
    print(tot / 1)
    np.savez("{0}/demos_full_2".format(env), s=demo_s, a=demo_a, P=P, V=V, C=C)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Train expert policies.')
    # parser.add_argument('env', choices=['walker', 'hopper', 'halfcheetah', 'ant'])
    # args = parser.parse_args()
    # if args.env == 'walker':
    #     # train_walker_expert()
    #     generate_demos("Walker2DBulletEnv-v0")
    # elif args.env == 'hopper':
    #     # train_hopper_expert()
    #     generate_demos("HopperBulletEnv-v0")
    # elif args.env == 'halfcheetah':
    #     # train_halfcheetah_expert()
    #     generate_demos("HalfCheetahBulletEnv-v0")
    # elif args.env == 'ant':
    #     train_ant_expert()
    #     generate_demos("AntBulletEnv-v0")
    # else:
    #     print("ERROR: unsupported env.")
    # train_cheetah_expert()
    generate_demos("HalfCheetah-v3")
