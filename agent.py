"""
DQN avec replay buffer et target network.

Architecture : obs_dim → 128 → 128 → n_actions
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 20_000):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs,      dtype=np.float32),
            np.array(actions,  dtype=np.int64),
            np.array(rewards,  dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones,    dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    def __init__(
        self,
        obs_dim:     int,
        n_actions:   int,
        lr:          float = 3e-4,
        gamma:       float = 0.97,
        batch_size:  int   = 128,
        target_sync: int   = 200,   # steps entre syncs target net
        eps_start:   float = 1.0,
        eps_end:     float = 0.05,
        eps_decay:   int   = 3000,  # épisodes
        buf_capacity: int  = 30_000,
        device:      str   = "cpu",
    ):
        self.n_actions   = n_actions
        self.gamma       = gamma
        self.batch_size  = batch_size
        self.target_sync = target_sync
        self.eps         = eps_start
        self.eps_end     = eps_end
        self.eps_decay   = eps_decay
        self.device      = torch.device(device)

        self.online = QNet(obs_dim, n_actions).to(self.device)
        self.target = QNet(obs_dim, n_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.opt    = optim.Adam(self.online.parameters(), lr=lr)
        self.buf    = ReplayBuffer(buf_capacity)

        self._steps = 0
        self._ep    = 0
        self.loss_history: list[float] = []

    # ------------------------------------------------------------------
    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.online(t).argmax(dim=1).item())

    # ------------------------------------------------------------------
    def push(self, obs, action, reward, next_obs, done):
        self.buf.push(obs, action, reward, next_obs, done)
        self._steps += 1
        if len(self.buf) >= self.batch_size:
            self._learn()
        if self._steps % self.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())

    # ------------------------------------------------------------------
    def _learn(self):
        obs, actions, rewards, next_obs, dones = self.buf.sample(self.batch_size)

        obs      = torch.tensor(obs,     device=self.device)
        actions  = torch.tensor(actions, device=self.device)
        rewards  = torch.tensor(rewards, device=self.device)
        next_obs = torch.tensor(next_obs,device=self.device)
        dones    = torch.tensor(dones,   device=self.device)

        q_val  = self.online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN : online choisit l'action, target évalue
            best_a = self.online(next_obs).argmax(1, keepdim=True)
            q_next = self.target(next_obs).gather(1, best_a).squeeze(1)
            target = rewards + self.gamma * q_next * (1 - dones)

        loss = nn.SmoothL1Loss()(q_val, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()
        self.loss_history.append(float(loss.detach()))

    # ------------------------------------------------------------------
    def end_episode(self):
        self._ep += 1
        frac     = min(self._ep / self.eps_decay, 1.0)
        self.eps = self.eps_end + (1.0 - self.eps_end) * (1.0 - frac)

    # ------------------------------------------------------------------
    def save(self, path: str):
        torch.save({"online": self.online.state_dict(), "eps": self.eps}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["online"])
        self.eps = ckpt.get("eps", self.eps_end)
