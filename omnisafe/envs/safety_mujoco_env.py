from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import safety_gymnasium
import torch

from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import DEVICE_CPU, Box
from omnisafe


@env_register
class SafetyGymnasiumEnv(CMDP):
    """Safety Gymnasium Environment.

    Args:
        env_id (str): Environment id.
        num_envs (int, optional): Number of environments. Defaults to 1.
        device (torch.device, optional): Device to store the data. Defaults to
            ``torch.device('cpu')``.

    Keyword Args:
        render_mode (str, optional): The render mode ranges from 'human' to 'rgb_array' and 'rgb_array_list'.
            Defaults to 'rgb_array'.
        camera_name (str, optional): The camera name.
        camera_id (int, optional): The camera id.
        width (int, optional): The width of the rendered image. Defaults to 256.
        height (int, optional): The height of the rendered image. Defaults to 256.

    Attributes:
        need_auto_reset_wrapper (bool): Whether to use auto reset wrapper.
        need_time_limit_wrapper (bool): Whether to use time limit wrapper.
    """

    need_auto_reset_wrapper: bool = False
    need_time_limit_wrapper: bool = False

    _support_envs: ClassVar[list[str]] = [
        'SafetyPointGoal0-v0',
        'SafetyPointGoal1-v0',
        'SafetyPointGoal2-v0',