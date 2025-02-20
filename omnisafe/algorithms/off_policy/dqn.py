from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn
import torch.optim as optim

import numpy as np

from collections import deque
import random

from omnisafe.algorithms import registry
from omnisafe.envs.core import CMDP, make
from omnisafe.algorithms.base_algo import BaseAlgo
from omnisafe.common.logger import Logger

from omnisafe.models.actor.discretizer_mlp_actor import DiscretizerMLPActor


@registry.register
class DQN(BaseAlgo):
    def _init_env(self) -> None:
        print(self._env_id)
        print(self._cfgs)

        env_cfgs = {}

        if hasattr(self._cfgs, 'env_cfgs') and self._cfgs.env_cfgs is not None:
            env_cfgs = self._cfgs.env_cfgs.todict()

        self._env: CMDP = make(
            self._env_id,
            num_envs=self._cfgs.train_cfgs.vector_env_nums,
            device=self._cfgs.train_cfgs.device,
            **env_cfgs,
        )

        self._env.set_seed(self._cfgs.seed)

        self.discrete_actions = 11
        self.batch_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_episodes = 50
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def _init_model(self) -> None:
        self.q_network = DiscretizerMLPActor(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            hidden_sizes=self._cfgs.model_cfgs.actor.hidden_sizes,
            discrete_actions=self.discrete_actions,
        )
        self.target_network = DiscretizerMLPActor(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            hidden_sizes=self._cfgs.model_cfgs.actor.hidden_sizes,
            discrete_actions=self.discrete_actions,
        )

        self.target_network.net.load_state_dict(self.q_network.net.state_dict())
        self.target_network.net.eval()

    def _init(self) -> None:
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)
        self.memory = deque(maxlen=10000)

    def _init_log(self) -> None: ...

    def learn(self) -> tuple[float, float, float]:
        for episode in range(self.num_episodes):
            obs, _ = self._env.reset()
            total_reward = 0
            done = False

            while not done:
                action_idx = self._select_action(obs, epsilon=self.epsilon)
                action = np.linspace(-2, 2, self.discrete_actions)[action_idx]
                next_state, reward, cost, terminated, truncated, _ = self._env.step([action])
                done = terminated or truncated

                self.memory.append((obs, action_idx, reward, next_state, done))
                self._train()
                state = next_state
                total_reward += reward

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            self.target_network.net.load_state_dict(self.q_network.net.state_dict())
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    def _select_action(self, obs, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.discrete_actions - 1)
        else:
            with torch.no_grad():
                q_values = self.q_network.feed_forward(obs)
            return torch.argmax(q_values).item()

    def _train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.q_network.net.feed_forward(states).gather(1, actions)
        next_q_values = self.target_network.net.feed_forward(next_states).max(1, keepdims=True)[0]
        target_q_values = rewards + 0.99 * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
