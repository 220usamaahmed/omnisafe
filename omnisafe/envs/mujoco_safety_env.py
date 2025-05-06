from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import safety_gymnasium
import torch

from omnisafe.envs.core import CMDP, env_register
from omnisafe.envs.safety_mujoco.ant_v4 import AntEnv
from omnisafe.envs.safety_mujoco.reacher_v4 import ReacherEnv
from omnisafe.envs.safety_mujoco.walker2d_v4 import Walker2dEnv
from omnisafe.envs.safety_mujoco.humanoidstandup_v4 import HumanoidStandupEnv
from omnisafe.typing import DEVICE_CPU, Box


@env_register
class MujocoSafetyEnv(CMDP):
    need_auto_reset_wrapper: bool = False
    need_time_limit_wrapper: bool = False

    _support_envs: ClassVar[list[str]] = [
        'MujocoAnt-v4',
        'MujocoWalker2d-v4',
        'MujocoHumanoidStandup-v4',
    ]

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        device: torch.device = DEVICE_CPU,
        **kwargs: Any,
    ) -> None:
        """Initialize an instance of :class:`SafetyGymnasiumEnv`."""
        super().__init__(env_id)
        self._num_envs = num_envs
        self._device = torch.device(device)

        assert num_envs == 1

        if env_id == 'MujocoAnt-v4':
            self._env = AntEnv()
        elif env_id == 'MujocoWalker2d-v4':
            self._env = Walker2dEnv()
        elif env_id == 'MujocoHumanoidStandup-v4':
            self._env = HumanoidStandupEnv()

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
        """Step the environment.

        .. note::
            OmniSafe uses auto reset wrapper to reset the environment when the episode is
            terminated. So the ``obs`` will be the first observation of the next episode. And the
            true ``final_observation`` in ``info`` will be stored in the ``final_observation`` key
            of ``info``.

        Args:
            action (torch.Tensor): Action to take.

        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Some information logged by the environment.
        """
        obs, reward, cost, terminated, truncated, info = self._env.step(
            action.detach().cpu().numpy(),
        )

        obs, reward, cost, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (obs, reward, cost, terminated, truncated)
        )
        if terminated or truncated:
            info['final_observation'] = np.array(
                [array if array is not None else np.zeros(obs.shape[-1]) for array in [None]],
            )
            info['final_observation'] = torch.as_tensor(
                info['final_observation'],
                dtype=torch.float32,
                device=self._device,
            )

        return obs, reward, cost, terminated, truncated, info

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment.

        Args:
            seed (int, optional): The random seed. Defaults to None.
            options (dict[str, Any], optional): The options for the environment. Defaults to None.

        Returns:
            observation: Agent's observation of the current environment.
            info: Some information logged by the environment.
        """
        obs, info = self._env.reset(seed=seed, options=options)
        return torch.as_tensor(obs, dtype=torch.float32, device=self._device), info

    @property
    def max_episode_steps(self) -> int:
        """The max steps per episode."""
        return self._env.spec.max_episode_steps

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
