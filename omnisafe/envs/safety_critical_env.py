from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import safety_gymnasium
import torch

from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import DEVICE_CPU, Box

from omnisafe.envs.safety_critical.envs_critical import Glucose, BiGlucose, CSTR


@env_register
class SafetyCriticalEnv(CMDP):

    need_auto_reset_wrapper: bool = False
    need_time_limit_wrapper: bool = False

    _support_envs: ClassVar[list[str]] = [
        'Glucose',
        'BiGlucose',
        'CSTR',
    ]

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        device: torch.device = DEVICE_CPU,
        **kwargs: Any,
    ) -> None:
        super().__init__(env_id)
        self._num_envs = num_envs
        self._device = torch.device(device)

        assert num_envs == 1

        if env_id == "Glucose":
            self._env = Glucose(altered_paras={'n': 0.2, 'p2': 0.005, 'p3': 5e-6})
        elif env_id == "BiGlucose":
            self._env = BiGlucose(
                altered_paras={
                    "D_G": 80,
                    "V_G": 0.18,
                    "k_12": 0.0343,
                    "F_01": 0.0121,
                    "EGP_0": 0.0148,
                    "A_g": 0.8,
                    "t_maxG": 40,
                    "t_maxI": 55,
                    "V_I": 0.12,
                    "k_e": 0.138,
                    "k_a1": 0.0031,
                    "k_a2": 0.0752,
                    "k_a3": 0.0472,
                    "k_b1": 9.114e-06,
                    "k_b2": 6.768e-06,
                    "k_b3": 0.00189,
                    "t_maxN": 32.46,
                    "k_N": 0.62,
                    "V_N": 16.06,
                    "p": 0.016,
                    "S_N": 19600.0,
                    "M_g": 180.16,
                    "BW": 68.5,
                    "N_b": 48.13,
                    "dt": 10,
                }
            )
        elif env_id == "CSTR":
            self._env = CSTR(altered_paras={'alpha': 1.05, 'beta': 1.1})

        assert isinstance(self._env.action_space, Box), 'Only support Box action space.'
        assert isinstance(
            self._env.observation_space,
            Box,
        ), 'Only support Box observation space.'
        self._action_space = self._env.action_space
        self._observation_space = self._env.observation_space

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        obs, reward, cost, terminated, truncated, info = self._env.step(
            action.detach().cpu().numpy(),
        )

        if truncated:
            self._env.reset()

        obs, reward, cost, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (obs, reward, cost, False, truncated)
        )

        return obs, reward, cost, terminated, truncated, info

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        obs, info = self._env.reset(seed=seed)
        return torch.as_tensor(obs, dtype=torch.float32, device=self._device), info

    @property
    def max_episode_steps(self) -> int:
        """The max steps per episode."""
        return 100

    def set_seed(self, seed: int) -> None:
        """Set the seed for the environment.

        Args:
            seed (int): Seed to set.
        """
        self.reset(seed=seed)

    def render(self) -> Any:
        """Compute the render frames as specified by :attr:`render_mode` during the initialization of the environment.

        Returns:
            The render frames: we recommend to use `np.ndarray`
                which could construct video by moviepy.
        """
        return self._env.render()

    def close(self) -> None:
        """Close the environment."""
        self._env.close()
