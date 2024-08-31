import gym
import numpy as np


class BernoulliBandits(gym.Env):
    def __init__(self, probs: np.ndarray):
        self.probs = probs
        self.action_space = gym.spaces.Discrete(probs.shape[0])
        self.observation_space = gym.spaces.Discrete(1)
        self.rng = np.random.default_rng()

    def step(self, act: int):
        assert self.action_space.contains(act), f"step invalid: {(act, self.probs.shape[0])}"
        reward = self.rng.binomial(n=1, p=self.probs[act])
        info = {}

        return 0, reward, False, False, info

    def reset(self, seed: int):
        self.rng = np.random.default_rng(seed)

        return 0, {}

