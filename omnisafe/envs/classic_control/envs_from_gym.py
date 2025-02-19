from __future__ import annotations

from typing import Any, ClassVar
import gymnasium
import torch
import numpy as np
import omnisafe

from omnisafe.envs.core import CMDP, env_register, env_unregister
from omnisafe.typing import DEVICE_CPU


@env_register
class Pendulum(CMDP):
    _support_envs: ClassVar[list[str]] = ["Pendulum-v1"]  # Supported task names

    need_auto_reset_wrapper = True  # Whether `AutoReset` Wrapper is needed
    need_time_limit_wrapper = True  # Whether `TimeLimit` Wrapper is needed

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        device: torch.device = DEVICE_CPU,
        **kwargs: Any,
    ) -> None:
        kwargs = {"render_mode": kwargs.get("render_mode", "rgb_array")}

        super().__init__(env_id)
        self._num_envs = num_envs
        # Instantiate the environment object
        self._env = gymnasium.make(id=env_id, autoreset=True, **kwargs)
        # Specify the action space for initialization by the algorithm layer
        self._action_space = self._env.action_space
        # Specify the observation space for initialization by the algorithm layer
        self._observation_space = self._env.observation_space
        # Optional, for GPU acceleration. Default is CPU
        self._device = device  # 可选项，使用GPU加速。默认为CPU

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # Reset the environment
        obs, info = self._env.reset(seed=seed, options=options)
        # Convert the reset observations to a torch tensor.
        return (
            torch.as_tensor(obs, dtype=torch.float32, device=self._device),
            info,
        )

    @property
    def max_episode_steps(self) -> int | None:
        # Return the maximum number of interaction steps per episode in the environment
        return self._env.env.spec.max_episode_steps

    def set_seed(self, seed: int) -> None:
        # Set the environment's random seed for reproducibility
        self.reset(seed=seed)  # 设定环境的随机种子以实现可复现性

    def render(self) -> Any:
        # Return the image rendered by the environment
        return self._env.render()

    def close(self) -> None:
        # Release the environment instance after training ends
        self._env.close()

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
        # Read the dynamic information after interacting with the environment
        obs, reward, terminated, truncated, info = self._env.step(
            action.detach().cpu().numpy(),
        )
        # Gymnasium does not explicitly include safety constraints; this is just a placeholder.
        cost = np.zeros_like(reward)
        # Convert dynamic information into torch tensor.
        obs, reward, cost, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (obs, reward, cost, terminated, truncated)
        )
        if "final_observation" in info:
            info["final_observation"] = np.array(
                [
                    array if array is not None else np.zeros(obs.shape[-1])
                    for array in info["final_observation"]
                ],
            )
            # Convert the last observation recorded in info into a torch tensor.
            info["final_observation"] = torch.as_tensor(
                info["final_observation"],
                dtype=torch.float32,
                device=self._device,
            )

        return obs, reward, cost, terminated, truncated, info
