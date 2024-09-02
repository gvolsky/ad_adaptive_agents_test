import numpy as np


class Random:
    def __init__(self, num_arms: int):
        self.num_arms = num_arms

    def select_arm(self):
        return np.random.randint(low=0, high=self.num_arms)

    def update(self, arm, reward):
        pass
