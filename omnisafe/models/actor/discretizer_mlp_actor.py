from __future__ import annotations

import torch

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
        action = self.net(obs)
