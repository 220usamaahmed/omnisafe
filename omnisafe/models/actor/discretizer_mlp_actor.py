from __future__ import annotations

import torch
from torch.distributions import Distribution

import numpy as np

from omnisafe.models.base import Actor
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.model import build_mlp_network


class DiscretizerMLPActor(Actor):

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        output_activation: Activation = 'tanh',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        discrete_actions: int = 11,
    ) -> None:
        self._discrete_actions = discrete_actions

        # TODO: Handle action spaces differnet shapes
        assert (
            act_space.shape[0] == 1
        ), "Currently only env with action space of shape (1,) supported"

        super().__init__(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode)

        self.net: torch.nn.Module = build_mlp_network(
            sizes=[self._obs_dim, *self._hidden_sizes, discrete_actions],
            activation=activation,
            output_activation=output_activation,
            weight_initialization_mode=weight_initialization_mode,
        )

    def predict(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        model_output = self.net(obs)
        action_idx = torch.argmax(model_output).item()

        # TODO: Get limits from action space
        action = np.linspace(-2, 2, self._discrete_actions)[action_idx]

        return torch.tensor(action)

    def feed_forward(self, obs: torch.Tensor) -> torch.Tensor:
        model_output = self.net(obs)
        return model_output

    def _distribution(self, obs: torch.Tensor) -> Distribution:
        raise NotImplementedError

    def forward(self, obs: torch.Tensor) -> Distribution:
        """Forward method implementation.

        Args:
            obs (torch.Tensor): Observation from environments.

        Returns:
            The distribution of the action.
        """
        return self._distribution(obs)

    def log_prob(self, act: torch.Tensor) -> torch.Tensor:
        """Log probability of the action.

        Args:
            act (torch.Tensor): Action from :meth:`predict` or :meth:`forward`  tensor.

        Raises:
            NotImplementedError: The method is not implemented.
        """
        raise NotImplementedError
