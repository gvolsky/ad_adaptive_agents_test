import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import wandb


def plot_bars(names, rewards, scales, name='exp'):
    assert len(names) == len(rewards) == len(scales), \
    f'length of: names={len(names)}, rewards={len(rewards)}, scales={len(scales)}'

    plt.bar(
        names, 
        [(r / s).mean() for r, s in zip(rewards, scales)],
        list(mcolors.TABLEAU_COLORS.keys())[:len(names)]
    )
    plt.ylabel('Normalized score')
    plt.ylabel('Bandit scores')
    wandb.log({name: wandb.Image(plt)})
    plt.close()