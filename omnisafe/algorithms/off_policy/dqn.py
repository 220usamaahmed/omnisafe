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
        self.env_cfgs = {}
        if hasattr(self._cfgs, 'env_cfgs') and self._cfgs.env_cfgs is not None:
            self.env_cfgs = self._cfgs.env_cfgs.todict()

        self.train_cfgs = self._cfgs.train_cfgs
        self.algo_cfgs = self._cfgs.algo_cfgs
        self.model_cfgs = self._cfgs.model_cfgs

        self._env: CMDP = make(
            self._env_id,
            num_envs=self.train_cfgs.vector_env_nums,
            device=self.train_cfgs.device,
            **self.env_cfgs,
        )

        self._env.set_seed(self._cfgs.seed)

        self.discrete_actions = self.model_cfgs.actor.actor_cfgs.discrete_actions
        self.batch_size = self.algo_cfgs.batch_size
        self.device = self.train_cfgs.device
        self.num_episodes = self.train_cfgs.total_episodes
        self.epsilon = self.algo_cfgs.epsilon
        self.epsilon_decay = self.algo_cfgs.epsilon_decay
        self.epsilon_min = self.algo_cfgs.epsilon_min
        self.gamma = self.algo_cfgs.gamma
        self.lr = self.model_cfgs.actor.lr

    def _init_model(self) -> None:
        self.q_network = DiscretizerMLPActor(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            hidden_sizes=self._cfgs.model_cfgs.actor.hidden_sizes,
            weight_initialization_mode=self.model_cfgs.weight_initialization_mode,
            activation=self.model_cfgs.actor.activation,
            discrete_actions=self.discrete_actions,
        )
        self.target_network = DiscretizerMLPActor(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            hidden_sizes=self._cfgs.model_cfgs.actor.hidden_sizes,
            weight_initialization_mode=self.model_cfgs.weight_initialization_mode,
            activation=self.model_cfgs.actor.activation,
            discrete_actions=self.discrete_actions,
        )

        self.target_network.net.load_state_dict(self.q_network.net.state_dict())
        self.target_network.net.eval()

    def _init(self) -> None:
        self.optimizer = optim.Adam(self.q_network.net.parameters(), lr=1e-3)
        self.memory = deque(maxlen=10000)

    def _init_log(self) -> None:
        self._logger: Logger = Logger(
            output_dir=self._cfgs.logger_cfgs.log_dir,
            exp_name=self._cfgs.exp_name,
            seed=self._cfgs.seed,
            use_tensorboard=self._cfgs.logger_cfgs.use_tensorboard,
            use_wandb=self._cfgs.logger_cfgs.use_wandb,
            config=self._cfgs,
        )

        what_to_save: dict[str, Any] = {}
        what_to_save['pi'] = self.q_network

        self._logger.setup_torch_saver(what_to_save)
        self._logger.torch_save()

    def learn(self) -> tuple[float, float, float]:
        for episode in range(self.num_episodes):
            state, _ = self._env.reset()
            total_reward = 0
            done = False

            while not done:
                action_idx = self._select_action(state, epsilon=self.epsilon)
                action = torch.tensor([np.linspace(-2, 2, self.discrete_actions)[action_idx]])

                next_state, reward, _, terminated, truncated, _ = self._env.step(action)
                done = terminated or truncated

                self.memory.append((state.numpy(), action_idx, reward, next_state.numpy(), done))
                self._train()
                state = next_state
                total_reward += reward

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            self.target_network.net.load_state_dict(self.q_network.net.state_dict())

            # TODO: This and other metrics should be logged using the logger
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")

        self._logger.torch_save()

        # TODO: Return the actual ep_ret, ep_cost, ep_len of the last episode
        return 0, 0, 0

    def _select_action(self, obs, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.discrete_actions - 1)
        else:
            with torch.no_grad():
                q_values = self.q_network.net(obs)
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

        q_values = self.q_network.net(states).gather(1, actions)
        next_q_values = self.target_network.net(next_states).max(1, keepdims=True)[0]
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
