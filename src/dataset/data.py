import os
from collections import defaultdict

import numpy as np
from torch.utils.data import Dataset


def load_data(data_dir: str):
    data = defaultdict(list)
    for file in os.listdir(data_dir):
        load_data = np.load(os.path.join(data_dir, file))
        for item in load_data.files:
            data[item].append(load_data[item])
    return data


class SequenceDataset(Dataset):
    def __init__(self, data_dir: str, seq_len: int = 60):
        data = load_data(data_dir)
        self.seq_len = seq_len

        self.act = np.hstack(data["act"]).T
        self.rewards = np.hstack(data["reward"]).T
        self.cumulative_rewards = np.array([cumulative_rewards[-1] for cumulative_rewards in data["cumulative_rewards"]])
        self.max_prob = np.array([probs.max() for probs in data['probs']])

    def _get_data(self, idx):
        num_hists = self.act.shape[0]
        i1, i2 = idx % num_hists, idx // num_hists

        actions = self.act[i1, i2 : i2 + self.seq_len]
        rewards = self.rewards[i1, i2 : i2 + self.seq_len]
        cumulative_reward = self.cumulative_rewards[i1]
        max_prob = self.max_prob[i1]

        return actions, rewards, cumulative_reward, max_prob

    def __len__(self):
        return self.act.shape[0] * (self.act.shape[1] - self.seq_len + 1)

    def __getitem__(self, idx):
        return self._get_data(idx)