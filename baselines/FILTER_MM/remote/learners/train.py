from filt import FILTER
from bc import BehavioralCloning
from data_utils import fetch_demos, fetch_expert
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train expert policies.')
    parser.add_argument('-e', '--env', choices=['walker', 'hopper', 'antmaze', 'halfcheetah', 'ant', 'humanoid', 'cartpole', 'meerkat', 'breakout', 'space',
                                                'acrobot', 'bipedal', 'lunarlander', 'frozenlake', 'pong', 'qbert', 'bullet-all',
                                                'walker', 'hopper', 'cheetah'], default='cheetah')
    parser.add_argument('-a', '--algo', choices=['mm', 'filter-nr', 'filter-br', 'bc'], default='mm')
    parser.add_argument('-s', '--seed', default='3')
    args = parser.parse_args()

    if args.seed is not None and args.seed.isdigit():
        seed = int(args.seed)

    if args.env == 'walker':
        envs = ["Walker2d-v3"]
    elif args.env == "hopper":
        envs = ["Hopper-v3"]
    elif args.env == "ant":
        envs = ["Ant-v3"]
    elif args.env == "humanoid":
        envs = ["Humanoid-v3"]
    elif args.env == "cheetah":
        envs = ["HalfCheetah-v3"]
    elif args.env == 'antmaze':
        envs = ["antmaze-large-diverse-v2", "antmaze-large-play-v2"]
    elif args.env == "cartpole":
        envs = ["CartPole-v1"]
    elif args.env == "meerkat":
        envs = ["Meerkat-v3"]
    elif args.env == "acrobot":
        envs = ["Acrobot-v1"]
    elif args.env == "bipedal":
        envs = ["BipedalWalker-v3"]
    elif args.env == "frozenlake":
        envs = ["FrozenLake-v1"]
    elif args.env == "lunarlander":
        envs = ["LunarLander-v2"]
    elif args.env == "pong":
        envs = ["PongNoFrameskip-v4"]
    elif args.env == "qbert":
        envs = ["QbertNoFrameskip-v4"]
    elif args.env == "breakout":
        envs = ["BreakoutNoFrameskip-v4"]
    elif args.env == "space":
        envs = ["SpaceInvadersNoFrameskip-v4"]
    elif args.env == "bullet-all":
        envs = ["CartPole-v1", "Meerkat-v3", "Acrobot-v1"]

    for env in envs:
        print(env)
        tup = fetch_demos(env)
        expert = fetch_expert(env)
        if args.algo == "mm":
            for i in range(seed):
                print(f"SEED {i}")
                flt = FILTER(env)
                flt.train(*tup, n_seed=i, n_exp=25, alpha=0, no_regret=True, expert_policy=expert, device="cuda:2")
        elif args.algo == "filter-nr":
            for i in range(seed):
                print(f"SEED {i}")
                flt = FILTER(env)
                flt.train(*tup, n_seed=i, n_exp=25, alpha=1, no_regret=True, expert_policy=expert, device="cpu")
        elif args.algo == "filter-br":
            for i in range(seed):
                print(f"SEED {i}")
                flt = FILTER(env)
                flt.train(*tup, n_seed=i, n_exp=25, alpha=1, no_regret=False, expert_policy=expert, device="cuda:2")
        elif args.algo == "bc":
            for i in range(seed):
                print(f"SEED {i}")
                BehavioralCloning(env, *tup[1:3], n_seed=i, device="cpu")
