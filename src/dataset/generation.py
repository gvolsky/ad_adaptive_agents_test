import multiprocessing
import os
import uuid
from collections import defaultdict
from dataclasses import dataclass
from functools import partial

import gym
import numpy as np

from src.algos import UCB1
from src.envs import BernoulliBandits


@dataclass
class Config:
    num_arms: int = 10
    traj_name: str = 'game'
    num_iterations: int = 300
    data_directory: str = ''
    rho: int = 2
    seed: int = 0
    

class Histories:
    def __init__(self):
        self.history = defaultdict(list)

    def add(self, **kwargs):
        for key, value in kwargs.items():
            self.history[key].append(value)

    def dump(self, path: os.PathLike, type_pairs: dict, info: dict = {}):
        for key, type in type_pairs.items():
            self.history[key] = np.array(self.history[key], dtype=type).reshape(-1, 1)
        np.savez(path, **self.history, **info)


def odd_const(type: str):
    return 0.95 if type == 'odd' else 0.05

def generate_probs(rng: np.random.Generator, num_envs: int, num_arms: int, type: str = 'even'):
    assert type in {'even', 'odd', 'uniform'}, f"type must be 'even', 'odd' or 'uniform'"
    if type == 'uniform':
        uniform_data = rng.uniform(low=0.0, high=1.0, size=(num_envs, num_arms))
        return uniform_data / np.sum(uniform_data, axis=1, keepdims=True)
    
    odd_indices = np.arange(1, num_arms, 2)
    even_indices = np.arange(0, num_arms, 2)
    probs = np.zeros((num_envs, num_arms))

    odd_probs = rng.random((num_envs, len(odd_indices)))
    even_probs = rng.random((num_envs, len(even_indices)))
    const = odd_const(type)

    odd_probs = const * odd_probs / np.sum(odd_probs, axis=1, keepdims=True)
    even_probs = (1 - const) * even_probs / np.sum(even_probs, axis=1, keepdims=True)

    probs[:, odd_indices] = odd_probs
    probs[:, even_indices] = even_probs

    return probs

def hist_from_bandit(
    env: gym.Env, algo: UCB1, num_iterations: int, seed: int
):
    hist = Histories()
    obs, _ = env.reset(seed=seed)
    for _ in range(num_iterations):
        arm = algo.select_arm()
        new_obs, reward, term, trunc, info = env.step(arm)
        algo.update(arm, reward)
        hist.add(
            obs=obs,
            act=arm,
            reward=reward,
            term=term,
            trunc=trunc,
            # info=info
        )
        obs = new_obs

    cumulative_rewards = np.cumsum(hist.history['reward'])
    average_rewards = cumulative_rewards / (np.arange(num_iterations) + 1)

    return hist, cumulative_rewards, average_rewards

def process(
    probs: np.ndarray, data_path: os.PathLike, config: Config, seed: int
):
    env = BernoulliBandits(probs)
    algo = UCB1(config.num_arms, config.rho)
    hist, cumulative_rewards, average_rewards = hist_from_bandit(
        env, algo, config.num_iterations, seed
    )
    hist.dump(
        os.path.join(data_path, f'{config.traj_name}_{uuid.uuid4()}'), 
        type_pairs={'act': np.int32, 'reward': np.int32},
        info={
            "cumulative_rewards": cumulative_rewards, 
            "average_rewards": average_rewards, 
            "probs": probs
        }
    )

def generate_dataset(probs: np.ndarray, seed: int, config: Config):
    data_path = os.path.join(config.data_directory, f'traj_{seed}')
    os.makedirs(data_path, exist_ok=True)

    cores = os.cpu_count()
    print(f"Number of CPUs: {cores}")

    worker = partial(process, data_path=data_path, config=config, seed=seed)

    with multiprocessing.Pool(cores) as executor:
        executor.map(worker, probs)

    return data_path
