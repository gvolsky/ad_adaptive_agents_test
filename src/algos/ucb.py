import numpy as np


class UCB1:
    def __init__(self, num_arms: int, rho: float = 2):
        self.num_arms = num_arms
        self.rho = rho

        self.steps_arm = np.zeros(num_arms)
        self.avg_reward = np.zeros(num_arms)
        self.steps = 0

    def select_arm(self):
        for arm in range(self.num_arms):
            if not self.steps_arm[arm]:
                return arm
        ucb_vals = self.avg_reward + np.sqrt(
            self.rho * np.log(self.steps) / self.steps_arm
        )
        return np.argmax(ucb_vals)
    
    def update(self, arm, reward):
        self.steps_arm[arm] += 1
        n, value = self.steps_arm[arm], self.avg_reward[arm]
        self.avg_reward[arm] = ((n - 1) * value + reward) / n
        self.steps += 1