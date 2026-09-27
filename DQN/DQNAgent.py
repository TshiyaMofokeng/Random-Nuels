import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from DQN.replaybuffer import ReplayBuffer

def make_mlp(in_dim, out_dim, net_arch, activation_fn):
    layers, last = [], in_dim
    for h in net_arch:
        layers += [nn.Linear(last, h), activation_fn()]
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)

class MyDQNAgent:
    def __init__(
        self,
        policy,                       
        env,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=1.0,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10.0,
        double_dqn=True,
        policy_kwargs=None,
        verbose=0,
        seed=None,
        device="cpu",
    ):
        assert policy == "MlpPolicy", "only MlpPolicy is implemented"
        pk = policy_kwargs or {}
        net_arch = pk.get("net_arch", [32, 32, 32])
        activation_fn = pk.get("activation_fn", nn.ReLU)

        self.env = env
        self.n_actions = env.action_space.n
        self.obs_dim = int(np.prod(env.observation_space.shape))
        self.device = torch.device(device)

        self.gamma, self.batch_size = gamma, batch_size
        self.learning_starts, self.train_freq = learning_starts, train_freq
        self.gradient_steps = gradient_steps
        self.target_update_interval = target_update_interval
        self.exploration_fraction = exploration_fraction
        self.eps_start, self.eps_end = exploration_initial_eps, exploration_final_eps
        self.max_grad_norm, self.double_dqn = max_grad_norm, double_dqn
        self.verbose, self.seed = verbose, seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.q = make_mlp(self.obs_dim, self.n_actions, net_arch, activation_fn).to(self.device)
        self.target_q = make_mlp(self.obs_dim, self.n_actions, net_arch, activation_fn).to(self.device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.optim = torch.optim.Adam(self.q.parameters(), lr=learning_rate)

        self.buffer = ReplayBuffer(buffer_size, self.obs_dim, self.n_actions)
        self.num_timesteps = 0
        self._explore_steps = None
        self._last_obs = None
        self._last_mask = None

    @staticmethod
    def get_mask(env, info):
        if "action_mask" in info:
            return np.asarray(info["action_mask"], dtype=bool)
        return np.asarray(env.get_wrapper_attr("action_masks")(), dtype=bool)

    @property
    def exploration_rate(self):
        if self._explore_steps is None:
            return self.eps_end
        frac = min(1.0, self.num_timesteps / self._explore_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def q_values(self, obs):
        # Q-values before action masking
        obs_t = torch.as_tensor(np.asarray(obs, np.float32), device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.q(obs_t).squeeze(0).cpu().numpy()

    def predict(self, observation, state=None, episode_start=None,
                deterministic=True, action_masks=None):
        obs = np.asarray(observation, dtype=np.float32)
        single = obs.ndim == 1
        if single:
            obs = obs[None]
        B = len(obs)

        if action_masks is None:
            masks = np.ones((B, self.n_actions), bool)
        else:
            masks = np.asarray(action_masks, bool).reshape(B, -1)

        with torch.no_grad():
            q = self.q(torch.as_tensor(obs, device=self.device)).cpu().numpy()
        q[~masks] = -np.inf
        actions = q.argmax(axis=1)

        if not deterministic:
            for i in np.where(self.rng.random(B) < self.exploration_rate)[0]:
                actions[i] = self.rng.choice(np.flatnonzero(masks[i]))

        return (actions[0] if single else actions), state

   
    def _train_step(self):
        obs, act, rew, nobs, done, nmask = self.buffer.sample(self.batch_size, self.rng)
        dev = self.device
        obs = torch.as_tensor(obs, device=dev)
        nobs = torch.as_tensor(nobs, device=dev)
        act = torch.as_tensor(act, device=dev)
        rew = torch.as_tensor(rew, device=dev)
        done = torch.as_tensor(done, device=dev)
        nmask = torch.as_tensor(nmask, device=dev)

        with torch.no_grad():
            next_target = self.target_q(nobs)
            if self.double_dqn:
                next_online = self.q(nobs).masked_fill(~nmask, -1e9)
                best = next_online.argmax(dim=1, keepdim=True)
                next_q = next_target.gather(1, best).squeeze(1)
            else:
                next_q = next_target.masked_fill(~nmask, -1e9).max(dim=1).values
            target = rew + self.gamma * (1.0 - done) * next_q

        q_sa = self.q(obs).gather(1, act.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q_sa, target)

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.max_grad_norm)
        self.optim.step()
        return loss.item()

    def learn(self, total_timesteps, log_interval=100):
        env = self.env
        self._explore_steps = max(1, int(self.exploration_fraction * total_timesteps))
        start = self.num_timesteps

        if self._last_obs is None:
            obs, info = env.reset(seed=self.seed)
            mask = self.get_mask(env, info)
        else:
            obs, mask = self._last_obs, self._last_mask

        ep_ret, ep_rets, n_eps, loss = 0.0, deque(maxlen=100), 0, float("nan")

        while self.num_timesteps - start < total_timesteps:
            self.num_timesteps += 1

            # epsilon-greedy over valid actions only
            if self.rng.random() < self.exploration_rate:
                action = int(self.rng.choice(np.flatnonzero(mask)))
            else:
                action = int(self.predict(obs, action_masks=mask, deterministic=True)[0])

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_mask = self.get_mask(env, info)

            self.buffer.add(obs, action, reward, next_obs, terminated, next_mask)
            ep_ret += reward

            if terminated or truncated:
                ep_rets.append(ep_ret)
                ep_ret, n_eps = 0.0, n_eps + 1
                if self.verbose and n_eps % log_interval == 0:
                    print(f"steps {self.num_timesteps:>8} | eps {self.exploration_rate:.3f} | "
                          f"ep_rew_mean {np.mean(ep_rets):.3f} | loss {loss:.4f}")
                next_obs, info = env.reset()
                next_mask = self.get_mask(env, info)

            obs, mask = next_obs, next_mask

            if (self.num_timesteps > self.learning_starts
                    and self.num_timesteps % self.train_freq == 0
                    and len(self.buffer) >= self.batch_size):
                for _ in range(self.gradient_steps):
                    loss = self._train_step()

            if self.num_timesteps % self.target_update_interval == 0:
                self.target_q.load_state_dict(self.q.state_dict())

        self._last_obs, self._last_mask = obs, mask
        return self

    def save(self, path):
        torch.save(self.q.state_dict(), path)

    def load(self, path):
        self.q.load_state_dict(torch.load(path, map_location=self.device))
        self.target_q.load_state_dict(self.q.state_dict())
        return self
