import numpy as np


class ReplayBuffer:
   
    def __init__(self, size, obs_dim, n_actions):
        self.size, self.pos, self.full = size, 0, False
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.next_obs = np.zeros((size, obs_dim), np.float32)
        self.actions = np.zeros(size, np.int64)
        self.rewards = np.zeros(size, np.float32)
        self.dones = np.zeros(size, np.float32)
        self.next_masks = np.zeros((size, n_actions), bool)

    def add(self, obs, action, reward, next_obs, done, next_mask):
        i = self.pos
        self.obs[i], self.next_obs[i] = obs, next_obs
        self.actions[i], self.rewards[i], self.dones[i] = action, reward, float(done)
        self.next_masks[i] = next_mask
        self.pos = (self.pos + 1) % self.size
        self.full = self.full or self.pos == 0

    def __len__(self):
        return self.size if self.full else self.pos

    def sample(self, batch_size, rng):
        idx = rng.integers(0, len(self), batch_size)
        return (self.obs[idx], self.actions[idx], self.rewards[idx],
                self.next_obs[idx], self.dones[idx], self.next_masks[idx])