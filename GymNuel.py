import numpy as np
import gymnasium as gym
from gymnasium import spaces


class NuelOpt(gym.Env):
    def __init__(self, player_accuracies, agent_num, lookup):
        super().__init__()
        self.acc = np.asarray(player_accuracies)
        self.n = len(self.acc)
        self.agent = agent_num
        self.lookup = lookup            
        self.opponents = [i for i in range(self.n) if i != agent_num]

        # action k = "shoot opponent self.opponents[k]"
        self.action_space = spaces.Discrete(self.n - 1)
        self.observation_space = spaces.MultiBinary(self.n)

        self.alive = np.ones(self.n, dtype=bool)

    
    def _obs(self):
        return self.alive.astype(np.int8)

    def _mask(self):
        # valid actions = opponents that are still alive
        return np.array([self.alive[o] for o in self.opponents], dtype=bool)

    def _shoot(self, shooter, target):
        if self.np_random.random() < self.acc[shooter]:
            self.alive[target] = False

    def _opponent_shooter_target(self, shooter):
        alive_idx = tuple(np.where(self.alive)[0])
        game = self.lookup[alive_idx]
        return int(game[2, shooter])

    def _play_until_agent_turn(self):
        while True:
            if not self.alive[self.agent]:
                return True, 0.0
            if self.alive.sum() == 1:
                return True, 1.0

            shooter = self.np_random.choice(np.where(self.alive)[0])
            if shooter == self.agent:
                return False, 0.0          # agent gets a turn.

            target = self._opponent_shooter_target(shooter)
            self._shoot(shooter, target)

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        while True:
            self.alive[:] = True
            terminated, _ = self._play_until_agent_turn()
            if not terminated:             # agent got a turn before dying
                break
        return self._obs(), {"action_mask": self._mask()}

    def step(self, action):
        target = self.opponents[int(action)]

        # Stable Baselines Agents will waste their turn here, as action masking is not supported.
        if self.alive[target]:
            self._shoot(self.agent, target)
        

        terminated, reward = self._play_until_agent_turn()
        info = {"action_mask": self._mask()}
        return self._obs(), reward, terminated, False, info