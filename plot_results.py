import sys
from functools import partial
from os import path
from uuid import uuid4

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym.vector import SyncVectorEnv

from src.algos import UCB1, Random
from src.dataset import generate_probs, hist_from_bandit
from src.envs import BernoulliBandits
from train import DEVICE, Config, Transformer, evaluating


def plot_bars(names, bars, name='exp'):
    assert len(names) == len(bars), \
        f'length of: names={len(names)}, bars={len(bars)}'

    plt.bar(
        names, 
        bars,
        color=list(mcolors.TABLEAU_COLORS.keys())[:len(names)]
    )
    plt.ylabel('Normalized score')
    plt.ylabel('Bandit scores')
    plt.savefig(f"{name}.jpg")
    plt.close()

def get_reward(env, algo, config: Config):
    _, _ = env.reset(seed=config.eval_seed)
    _, cumulative_rewards, _ = hist_from_bandit(
            env, algo, config.num_iterations, config.eval_seed
        )
    return cumulative_rewards[-1]

def get_bars(model, config: Config):
    rng = np.random.default_rng(Config.eval_seed)
    rnd = Random(Config.num_arms)
    names = ['even', 'odd', 'uniform']
    bars = []
    for name in names:
        probs = generate_probs(rng, Config.num_eval_envs, Config.num_arms, type=name)
        get_reward = partial(get_reward, config=Config)

        ucb_rw = np.array([
            get_reward(BernoulliBandits(p), UCB1(Config.num_arms, Config.rho)) for p in probs
        ]).mean()
        rnd_rw = np.array([get_reward(BernoulliBandits(p), rnd) for p in probs]).mean()
        vec_envs = SyncVectorEnv(
                [lambda prob=prob: BernoulliBandits(prob) for prob in probs]
            )
        rw = evaluating(vec_envs, model, config).mean()
        bars.append((rw - rnd_rw) / (ucb_rw - rnd_rw))
    return bars

if __name__ == "__main__":
    PATH = sys.argv[1]
    model = Transformer(config=Config).to(DEVICE)
    model.load_state_dict(torch.load(PATH), weights_only=True)
    model.eval()
    bars = get_bars(model, Config)
    plot_bars(
        ['even', 'odd', 'uniform'],
        bars,
        path.join(Config.data_directory, f'{uuid4()}.jpg')
        )