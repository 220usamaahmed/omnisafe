from __future__ import annotations

import os
import time
from collections import OrderedDict, deque
from copy import deepcopy
from typing import Any, ClassVar

import gymnasium as gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import xmltodict
from omnisafe.envs.cassie.generator import FootTrajectoryGenerator
from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import DEVICE_CPU
from scipy.spatial.transform import Rotation
import torch

ABS_PATH = os.path.dirname(os.path.abspath(__file__))


class Cassie(CMDP):
    _suppor_envs = ClassVar[list[str]] = ["Cassie-v0"]

    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True

    def __init__(
        self,
        env_id: str,
        device: torch.device = DEVICE_CPU,
        **kwargs: Any,
    ) -> None:
        super().__init__(env_id)

        self._device = device

        # =========== for simulation parameter =========== #
        self.sim_dt = 0.002
        self.contro_freq = 50.0
        self.n_substeps = int(1 / (self.sim_dt * self.contro_freq))
        self.env_dt = self.sim_dt * self.n_substeps
        self.use_fixed_base = kwargs.get("use_fixed_base", False)
        self.gravity = np.array([0, 0, -9.8])
        self.num_legs = 2

        # for init value
        self.init_base_pos = kwargs.get(
            "init_base_pos",
            [0.0, 0.0, 1.0],
        )
        self.init_base_quat = kwargs.get(
            "init_base_quat",
            [1.0, 0.0, 0.0, 0.0],
        )

        # for Kp & Kd of actuator
        # order: abduct, thigh, knee
        self.Kps = np.array([100, 100, 88, 96, 50] * 2)
        self.Kds = np.array([10.0, 10.0, 8.0, 9.6, 5.0] * 2)

        # joint limit
        self.lower_limits = np.array([-15, -22.5, -50, -164, -140] * 2) * np.pi / 180.0
        self.upper_limits = np.array([22.5, 22.5, 80, -37, -30] * 2) * np.pi / 180.0
        self.joint_names = [
            "left_hip_roll",
            "left_hip_yaw",
            "left_hip_pitch",
            "left_knee",
            "left_foot",
            "right_hip_roll",
            "right_hip_yaw",
            "right_hip_pitch",
            "right_knee",
            "right_foot",
        ]

        # for mujoco object
        self.model = self._loadModel(use_fixed_base=self.use_fixed_base)
        self.data = mujoco.MjData(self.model)
        self.viewer = None

        # get sim id
        self.robot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cassie_pelvis")
        self.geom_floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.geom_foot_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_foot")
            for name in ["left", "right"]
        ]

        # joint index offset
        if self.use_fixed_base:
            self.pos_idx_offset = 0
            self.vel_idx_offset = 0
        else:
            self.pos_idx_offset = 7
            self.vel_idx_offset = 6
        # ================================================ #

        # foot step trajectory generator
        self.generator = FootTrajectoryGenerator()
        self.generator.foot_height = 0.0
        self.nominal_joint_targets = self.generator.getJointTargets(0.0)
        nominal_lower_limits = self.lower_limits - self.nominal_joint_targets
        nominal_upper_limits = self.upper_limits - self.nominal_joint_targets
        nominal_action_bounds = np.max(
            np.abs(np.stack([nominal_lower_limits, nominal_upper_limits], axis=0)),
            axis=0,
        ).astype(np.float32)

        # environmental variables
        self.max_episode_length = kwargs.get("max_episode_length", 1000)
        self.is_earlystop = kwargs.get("is_earlystop", False)
        self.cur_step = 0
        self.num_history = 3
        self.joint_pos_history = deque(maxlen=self.num_history)
        self.joint_vel_history = deque(maxlen=self.num_history)
        self.joint_target_history = deque(maxlen=self.num_history)
        self.cmd_lin_vel = np.zeros(3)
        self.cmd_ang_vel = np.zeros(3)
        self.lin_vel_cmd_range = kwargs.get("lin_vel_cmd_range", [-1.0, 1.0])
        self.ang_vel_cmd_range = kwargs.get("ang_vel_cmd_range", [-0.5, 0.5])
        assert self.lin_vel_cmd_range[0] <= self.lin_vel_cmd_range[1]
        assert self.ang_vel_cmd_range[0] <= self.ang_vel_cmd_range[1]
        self.action = np.zeros_like(self.lower_limits)
        self.action_weight = 0.5
        self.power_consumption = 0.0

        # for gym environment
        self.state_keys = [
            "cmd_lin_vel",
            "cmd_ang_vel",
            "gravity_vector",
            "base_lin_vel",
            "base_ang_vel",
            "joint_pos_list",
            "joint_vel_list",
            "phase_list",
            "joint_pos_history",
            "joint_vel_history",
            "joint_target_history",
            "contact_list",
            "base_height",
        ]
        state, info = self.reset()
        raw_state = self._getRawState()
        self.state_dim = state.shape[0]
        self.action_dim = len(self.lower_limits)
        self.reward_dim = len(self._getRewards(raw_state))
        self.cost_dim = len(self._getCosts(raw_state))
        self._observation_space = gym.spaces.Box(
            -np.inf * np.ones(self.state_dim, dtype=np.float32),
            np.inf * np.ones(self.state_dim, dtype=np.float32),
            dtype=np.float32,
        )
        self._action_space = gym.spaces.Box(
            -nominal_action_bounds,
            nominal_action_bounds,
            dtype=np.float32,
        )
        self.reward_space = gym.spaces.Box(
            -np.inf * np.ones(self.reward_dim, dtype=np.float32),
            np.inf * np.ones(self.reward_dim, dtype=np.float32),
            dtype=np.float32,
        )
        self.cost_space = gym.spaces.Box(
            -np.inf * np.ones(self.cost_dim, dtype=np.float32),
            np.inf * np.ones(self.cost_dim, dtype=np.float32),
            dtype=np.float32,
        )

    @property
    def max_episode_steps(self) -> int | None:
        return self.max_episode_length

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        super().reset(seed=seed)

        # reset sim & generator

        mujoco.mj_resetData(self.model, self.data)
        # mujoco.mj_setSeed(seed)
        # mujoco.mj_resetDataRandom(self.model, self.data, seed)

        self.generator.reset()
        self.action = np.zeros_like(self.lower_limits)

        # reset joint pos & vel
        init_qpos_list = [
            -6.05507670e-04,
            -9.46848441e-04,
            6.88587941e-01,
            9.43366294e-01,
            2.79908818e-03,
            2.57290345e-02,
            -3.30741919e-01,
            -1.46884023e00,
            4.99414927e-03,
            1.67887314e00,
            1.56343722e-03,
            -1.55491322e00,
            1.53641059e00,
            -1.65963020e00,
            6.02820781e-04,
            9.20671538e-04,
            6.88739578e-01,
            9.43359484e-01,
            -2.79838679e-03,
            -2.57319768e-02,
            -3.30761119e-01,
            -1.46842817e00,
            4.47926175e-03,
            1.67906987e00,
            1.58396908e-03,
            -1.55491322e00,
            1.53641059e00,
            -1.65963020e00,
        ]
        if not self.use_fixed_base:
            robot_pos = np.concatenate([self.init_base_pos, self.init_base_quat], axis=0)
            self.data.qpos[: self.pos_idx_offset] = robot_pos
            self.data.qvel[: self.vel_idx_offset] = 0.0
        self.data.qpos[self.pos_idx_offset :] = init_qpos_list
        self.data.qvel[self.vel_idx_offset :] = 0.0
        # self.sim.forward()
        mujoco.mj_forward(self.model, self.data)

        # simulate
        joint_targets = self.generator.getJointTargets(0.0)
        joint_targets = np.clip(joint_targets, self.lower_limits, self.upper_limits)
        for _ in range(self.num_history):
            for step_idx in range(self.n_substeps):
                p_error = joint_targets - self._getJointPosList()
                d_error = self._getJointVelList()
                torque = self.Kps * p_error - self.Kds * d_error
                torque[5:7] = -torque[5:7]
                self.data.ctrl[:] = torque
                # self.sim.step()
                mujoco.mj_step(self.model, self.data)
            # reset variables
            self.joint_pos_history.append(p_error)
            self.joint_vel_history.append(d_error)
            self.joint_target_history.append(joint_targets)

        # reset variables
        self.cur_step = 0
        self.is_terminated = False
        self.cmd_lin_vel = np.array([np.random.uniform(*self.lin_vel_cmd_range)] + [0.0, 0.0])
        self.cmd_ang_vel = np.array([0.0, 0.0] + [np.random.uniform(*self.ang_vel_cmd_range)])

        # get state
        raw_state = self._getRawState()
        converted_state = self._convertState(raw_state)
        return (
            torch.as_tensor(converted_state, dtype=torch.float32, device=self._device),
            {},
        )

    def setCommandVel(self, lin_vel, ang_vel):
        self.cmd_lin_vel = np.array([lin_vel, 0.0, 0.0])
        self.cmd_ang_vel = np.array([0.0, 0.0, ang_vel])
        raw_state = self._getRawState()
        return self._convertState(raw_state)

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
        self.cur_step += 1
        if self.is_terminated:
            state = deepcopy(self.terminal_state)
            reward = deepcopy(self.terminal_reward)
            info = deepcopy(self.terminal_info)
        else:
            state, reward, terminate, truncate, info = self._step(action)
            if terminate:
                self.is_terminated = True
                self.terminal_state = deepcopy(state)
                self.terminal_reward = deepcopy(reward)
                self.terminal_info = deepcopy(info)
        terminated = False if not self.is_earlystop else self.is_terminated
        truncated = self.cur_step >= self.max_episode_length
        cost = np.zeros_like(reward)
        state, reward, cost, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (state, reward, cost, terminated, truncated)
        )
        return state, reward, cost, terminated, truncated, info

    def render(self, mode="human", size=(512, 512), **kwargs):
        if mode == "rgb_array":
            renderer = mujoco.Renderer(self.model)
            renderer.update_scene(self.data)

            return renderer.render()
        else:
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.cam.azimuth = 45
                self.viewer.cam.distance = 5.0
                self.viewer.cam.elevation = -20
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        return

    def _loadModel(self, use_fixed_base=False):
        # load xml file
        robot_base_path = f"{ABS_PATH}/models/mjcf.xml"
        with open(robot_base_path) as f:
            robot_base_xml = f.read()
        xml = xmltodict.parse(robot_base_xml)
        body_xml = xml["mujoco"]["worldbody"]["body"]
        if type(body_xml) not in [OrderedDict, dict]:
            body_xml = body_xml[0]

        # for mass
        def getMass(body_xml):
            mass = float(body_xml["inertial"]["@mass"])
            if "body" in body_xml.keys():
                if type(body_xml["body"]) == list:
                    for sub_body in body_xml["body"]:
                        mass += getMass(sub_body)
                else:
                    mass += getMass(body_xml["body"])
            return mass

        self.mass = getMass(body_xml)

        # for time interval
        if type(xml["mujoco"]["option"]) == list:
            for option in xml["mujoco"]["option"]:
                if "@gravity" in option.keys():
                    option["@gravity"] = " ".join([f"{i}" for i in self.gravity])
                if "@timestep" in option.keys():
                    option["@timestep"] = self.sim_dt
        else:
            option = xml["mujoco"]["option"]
            option["@gravity"] = " ".join([f"{i}" for i in self.gravity])
            option["@timestep"] = self.sim_dt

        # for base fix
        if use_fixed_base:
            del body_xml["joint"]
        body_xml["@quat"] = " ".join([f"{i}" for i in self.init_base_quat])
        body_xml["@pos"] = " ".join([f"{i}" for i in self.init_base_pos])

        # convert xml to string & load model
        xml["mujoco"]["compiler"]["@meshdir"] = f"{ABS_PATH}/models/meshes"
        xml_string = xmltodict.unparse(xml)
        model = mujoco.MjModel.from_xml_string(xml_string)
        return model

    def _step(self, action):
        # ====== before simulation step ====== #
        joint_targets = self.generator.getJointTargets(self.cur_step * self.env_dt)
        joint_targets = np.clip(action + joint_targets, self.lower_limits, self.upper_limits)
        self.action = self.action * self.action_weight + joint_targets * (1.0 - self.action_weight)
        # ==================================== #

        # simulate
        self.power_consumption = 0.0
        for _ in range(self.n_substeps):
            p_error = self.action - self._getJointPosList()
            d_error = self._getJointVelList()
            torque = self.Kps * p_error - self.Kds * d_error
            torque[5:7] = -torque[5:7]
            self.data.ctrl[:] = torque
            mujoco.mj_step(self.model, self.data)
            # self.sim.step()
            self.power_consumption += np.mean(
                np.abs(self.data.actuator_force * self._getJointVelList())
            )
        self.power_consumption /= self.n_substeps

        # ====== after simulation step ====== #
        self.joint_pos_history.append(p_error)
        self.joint_vel_history.append(d_error)
        self.joint_target_history.append(self.action)

        raw_state = self._getRawState()
        state = self._convertState(raw_state)
        rewards = self._getRewards(raw_state)
        costs = self._getCosts(raw_state)
        reward = np.concatenate([rewards, costs])

        body_angle = raw_state["gravity_vector"][2] / np.linalg.norm(raw_state["gravity_vector"])
        truncate = self.cur_step >= self.max_episode_length
        terminate = body_angle >= 0

        info = {}
        # =================================== #
        return state, reward, terminate, truncate, info

    def _getJointPosList(self):
        joint_pos_list = np.array(
            [
                self.data.qpos[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_roll")
                ]
                for joint_name in self.joint_names
            ]
        )
        joint_pos_list[5:7] = -joint_pos_list[5:7]
        return joint_pos_list

    def _getJointVelList(self):
        joint_vel_list = np.array(
            [
                self.data.qvel[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_roll")
                ]
                for joint_name in self.joint_names
            ]
        )
        joint_vel_list[5:7] = -joint_vel_list[5:7]
        return joint_vel_list

    def _viewerSetup(self, viewer):
        viewer.cam.trackbodyid = self.robot_id
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 90

    def _getRawState(self):
        state = {}
        state["cmd_lin_vel"] = self.cmd_lin_vel
        state["cmd_ang_vel"] = self.cmd_ang_vel

        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu")

        base_pos = self.data.site_xpos[site_id]
        base_mat = np.reshape(self.data.site_xmat[site_id], (3, 3))

        yaw_angle = Rotation.from_matrix(base_mat).as_euler("zyx")[0]
        rot_mat = Rotation.from_rotvec([0.0, 0.0, yaw_angle]).as_matrix()
        state["base_height"] = base_pos[2:]

        gravity_vector = base_mat @ self.gravity
        state["gravity_vector"] = rot_mat.T @ gravity_vector

        velocities = np.zeros(6)

        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_SITE,
            site_id,
            velocities,
            1,
        )

        base_lin_vel = velocities[0:3]
        base_ang_vel = velocities[3:6]

        state["base_lin_vel"] = base_lin_vel
        state["base_ang_vel"] = base_ang_vel

        state["joint_pos_list"] = self._getJointPosList()
        state["joint_vel_list"] = self._getJointVelList()

        state["phase_list"] = self.generator.getPhaseList()
        state["base_freq"] = np.array([self.generator.default_freq])
        state["freq_list"] = deepcopy(self.generator.freq_list)

        state["joint_pos_history"] = np.concatenate(list(self.joint_pos_history))
        state["joint_vel_history"] = np.concatenate(list(self.joint_vel_history))
        state["joint_target_history"] = np.concatenate(list(self.joint_target_history))

        contact_list = np.zeros(self.num_legs)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            for geom_toe_idx, geom_toe_id in enumerate(self.geom_foot_ids):
                if contact.geom1 == geom_toe_id or contact.geom2 == geom_toe_id:
                    if contact.geom1 != contact.geom2:
                        contact_list[geom_toe_idx] = 1.0
        state["contact_list"] = contact_list

        collision = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 == contact.geom2:
                continue
            condition1 = (contact.geom1 == self.geom_floor_id) and (
                not contact.geom2 in self.geom_foot_ids
            )
            condition2 = (contact.geom2 == self.geom_floor_id) and (
                not contact.geom1 in self.geom_foot_ids
            )
            if condition1 or condition2:
                collision = True
                break
        state["collision"] = collision
        return state

    def _convertState(self, state):
        flatten_state = []
        for key in self.state_keys:
            flatten_state.append(state[key])
        state = np.concatenate(flatten_state)
        return np.array(state, dtype=np.float32)

    def _getRewards(self, state):
        ang_vel_error = (self.cmd_ang_vel[2] - state["base_ang_vel"][2]) ** 2
        lin_vel_error = np.sum(np.square(state["base_lin_vel"][:2] - self.cmd_lin_vel[:2]))
        error = ang_vel_error + lin_vel_error
        power_reward = -1e-3 * self.power_consumption
        reward = 0.1 * (-error + power_reward)
        return np.array([reward])

    def _getCosts(self, state):
        costs = []

        # for body angle constraint
        a = -np.cos(15.0 * (np.pi / 180.0))
        x = state["gravity_vector"][2] / np.linalg.norm(state["gravity_vector"])
        costs.append(1.0 if x > a else 0.0)

        # for height
        a = 0.7
        x = state["base_height"][0]
        costs.append(1.0 if x < a else 0.0)

        # swing timing
        cost = 0.0
        for leg_idx in range(self.num_legs):
            cos_phase, sin_phase = state["phase_list"][2 * leg_idx : 2 * (leg_idx + 1)]
            if sin_phase < 0.0:  # swing phase
                cost += 1.0 if state["contact_list"][leg_idx] else 0.0
            else:  # stance phase
                cost += 0.0 if state["contact_list"][leg_idx] else 1.0
        cost /= self.num_legs
        costs.append(cost)
        return np.array(costs)


if __name__ == "__main__":
    env = Env(use_fixed_base=False)

    for i in range(1):
        env.reset()
        start_t = time.time()
        global_t = 0.0
        elapsed_t = 0.0
        action = np.ones(env.action_space.shape[0])

        # frames = []

        for i in range(100):
            action = env.action_space.sample()
            state, reward, terminate, truncate, info = env.step(action)
            img = env.render("human")
            # frames.append(img)

            global_t += env.env_dt

            elapsed_t = time.time() - start_t
            if elapsed_t < global_t:
                time.sleep(global_t - elapsed_t)

            if terminate or truncate:
                break
