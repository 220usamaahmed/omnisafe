from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn

from omnisafe.algorithms import registry
from omnisafe.envs.core import CMDP, make
from omnisafe.algorithms.base_algo import BaseAlgo
from omnisafe.common.logger import Logger


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

    def _init_model(self) -> None:
        

    def _init(self) -> None: ...

    def _init_log(self) -> None: ...

    def learn(self) -> tuple[float, float, float]:

