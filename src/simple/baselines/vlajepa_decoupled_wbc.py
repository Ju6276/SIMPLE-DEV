"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

VLA-JEPA adapter for G1 decoupled-WBC evaluation.
"""

from __future__ import annotations

import os
import time

import numpy as np

from simple.agents.sonic_decoupled_wbc_agent import SonicDecoupledWbcAgent
from simple.core.action import ActionCmd
from simple.robots.g1_wholebody import (
    LEFT_ARM_JOINTS,
    LEFT_HAND_JOINTS,
    RIGHT_ARM_JOINTS,
    RIGHT_HAND_JOINTS,
)

from .vlajepa_ws_client import VlajepaWebsocketClient


DEFAULT_G1_HANDOVER_INSTRUCTION = (
    "Hand over cracker box from right hand to left hand and place it on the container."
)


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _build_vlajepa_state(joint_qpos: np.ndarray, height: float) -> np.ndarray:
    left_hand = joint_qpos[29:36]
    right_hand = joint_qpos[36:43]
    left_arm = joint_qpos[15:22]
    right_arm = joint_qpos[22:29]
    torso_rpy = joint_qpos[[13, 14, 12]]
    return np.concatenate(
        [
            left_hand,
            right_hand,
            left_arm,
            right_arm,
            torso_rpy,
            np.array([height], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def _action_to_upper_body_pose(action: np.ndarray) -> dict[str, float]:
    target_upper_body_pose: dict[str, float] = {}
    target_upper_body_pose.update(zip(LEFT_HAND_JOINTS, action[0:7], strict=True))
    target_upper_body_pose.update(zip(RIGHT_HAND_JOINTS, action[7:14], strict=True))
    target_upper_body_pose.update(zip(LEFT_ARM_JOINTS, action[14:21], strict=True))
    target_upper_body_pose.update(zip(RIGHT_ARM_JOINTS, action[21:28], strict=True))
    target_upper_body_pose["waist_roll_joint"] = float(action[28])
    target_upper_body_pose["waist_pitch_joint"] = float(action[29])
    target_upper_body_pose["waist_yaw_joint"] = float(action[30])
    return target_upper_body_pose


def _apply_navigate_deadband(
    navigate_cmd: np.ndarray,
    *,
    enable: bool,
    vx_deadband: float,
    vy_deadband: float,
    vyaw_deadband: float,
    target_yaw_deadband: float,
) -> np.ndarray:
    if not enable:
        return navigate_cmd.astype(np.float32, copy=True)

    filtered = navigate_cmd.astype(np.float32, copy=True)
    thresholds = np.array(
        [vx_deadband, vy_deadband, vyaw_deadband, target_yaw_deadband],
        dtype=np.float32,
    )
    filtered[np.abs(filtered) < thresholds] = 0.0
    return filtered


def _segment_l1_summary(action: np.ndarray, state_32d: np.ndarray) -> dict[str, float]:
    deltas = action[:32] - state_32d
    return {
        "left_hand": float(np.mean(np.abs(deltas[0:7]))),
        "right_hand": float(np.mean(np.abs(deltas[7:14]))),
        "left_arm": float(np.mean(np.abs(deltas[14:21]))),
        "right_arm": float(np.mean(np.abs(deltas[21:28]))),
        "torso_rpyh": float(np.mean(np.abs(deltas[28:32]))),
    }


def _should_log_debug(step_idx: int) -> bool:
    return step_idx < 21 or (step_idx > 0 and step_idx % 70 == 0)


class VlajepaDecoupledWbcAgent(SonicDecoupledWbcAgent):
    def __init__(self, robot, host: str, port: int, upsample_factor: int = 1, **kwargs):
        super().__init__(robot, **kwargs)

        self.server_ip = host
        self.server_port = port
        self.upsample_factor = upsample_factor

        self.client = VlajepaWebsocketClient(host=host, port=port)
        self._global_step_idx = 0
        self._last_base_height_command = 0.74
        self._reset_history = True
        self._enable_navigate_deadband = _env_flag("SIMPLE_VLAJEPA_NAV_DEADBAND", False)
        self._nav_vx_deadband = _env_float("SIMPLE_VLAJEPA_NAV_VX_DEADBAND", 0.10)
        self._nav_vy_deadband = _env_float("SIMPLE_VLAJEPA_NAV_VY_DEADBAND", 0.26)
        self._nav_vyaw_deadband = _env_float("SIMPLE_VLAJEPA_NAV_VYAW_DEADBAND", 0.05)
        self._nav_target_yaw_deadband = _env_float(
            "SIMPLE_VLAJEPA_NAV_TARGET_YAW_DEADBAND", 0.08
        )

        indices = self._dwbc_robot_model.get_joint_group_indices("upper_body")
        self.sonic_upper_joint_names = [
            name
            for name, idx in self._dwbc_robot_model.joint_to_dof_index.items()
            if idx in indices
        ]

    def get_action(
        self,
        observation,
        instruction=None,
        info=None,
        conditions=None,
        **kwargs,
    ):
        self._last_observation = observation
        self._last_qpos = observation["joint_qpos"]

        if len(self._action_queue) == 0:
            state_32d = _build_vlajepa_state(
                observation["joint_qpos"],
                height=self._last_base_height_command,
            )
            payload = {
                "batch_images": [[observation["head_stereo_left"]]],
                "instructions": [
                    instruction or DEFAULT_G1_HANDOVER_INSTRUCTION
                ],
                "state": state_32d[None, None, :],
                "reset": self._reset_history,
            }
            self._reset_history = False

            response = self.client.infer(payload)
            if not response.get("ok", False):
                raise RuntimeError(f"VLA-JEPA server inference failed: {response}")

            pred_action = response["data"]["actions"]
            print(
                f"step {self._global_step_idx}: Received {pred_action.shape[0]} actions from VLA-JEPA server."
            )
            if _should_log_debug(self._global_step_idx) and pred_action.shape[0] > 0:
                first_action = pred_action[0]
                delta_summary = _segment_l1_summary(first_action, state_32d)
                print(
                    "[VLAJEPADebug] "
                    f"step={self._global_step_idx} "
                    f"instruction={payload['instructions'][0]!r} "
                    f"state_rpyh={np.round(state_32d[28:32], 4).tolist()} "
                    f"action_rpyh_nav={np.round(first_action[28:36], 4).tolist()}"
                )
                print(
                    "[VLAJEPADebugUpper] "
                    f"step={self._global_step_idx} "
                    f"delta_l1={{{', '.join(f'{k}: {v:.4f}' for k, v in delta_summary.items())}}} "
                    f"state_l_hand={np.round(state_32d[0:7], 4).tolist()} "
                    f"pred_l_hand={np.round(first_action[0:7], 4).tolist()} "
                    f"state_r_hand={np.round(state_32d[7:14], 4).tolist()} "
                    f"pred_r_hand={np.round(first_action[7:14], 4).tolist()} "
                    f"state_l_arm={np.round(state_32d[14:21], 4).tolist()} "
                    f"pred_l_arm={np.round(first_action[14:21], 4).tolist()} "
                    f"state_r_arm={np.round(state_32d[21:28], 4).tolist()} "
                    f"pred_r_arm={np.round(first_action[21:28], 4).tolist()}"
                )

            for action in pred_action:
                raw_navigate_cmd = action[32:36].astype(np.float32)
                filtered_navigate_cmd = _apply_navigate_deadband(
                    raw_navigate_cmd,
                    enable=self._enable_navigate_deadband,
                    vx_deadband=self._nav_vx_deadband,
                    vy_deadband=self._nav_vy_deadband,
                    vyaw_deadband=self._nav_vyaw_deadband,
                    target_yaw_deadband=self._nav_target_yaw_deadband,
                )
                for _ in range(self.upsample_factor):
                    self.queue_action(
                        ActionCmd(
                            "vla_cmd",
                            target_upper_body_pose=_action_to_upper_body_pose(action),
                            navigate_cmd=filtered_navigate_cmd,
                            base_height_command=action[31:32].astype(np.float32),
                        )
                    )

            if _should_log_debug(self._global_step_idx) and pred_action.shape[0] > 0:
                raw_nav = pred_action[0][32:36].astype(np.float32)
                filtered_nav = _apply_navigate_deadband(
                    raw_nav,
                    enable=self._enable_navigate_deadband,
                    vx_deadband=self._nav_vx_deadband,
                    vy_deadband=self._nav_vy_deadband,
                    vyaw_deadband=self._nav_vyaw_deadband,
                    target_yaw_deadband=self._nav_target_yaw_deadband,
                )
                print(
                    "[VLAJEPADebugDeadband] "
                    f"step={self._global_step_idx} "
                    f"raw_nav={np.round(raw_nav, 4).tolist()} "
                    f"filtered_nav={np.round(filtered_nav, 4).tolist()} "
                    f"enabled={self._enable_navigate_deadband}"
                )

        action_cmd = super().get_action(observation, instruction, **kwargs)
        if action_cmd.type != "vla_cmd":
            raise ValueError(f"Unexpected action type {action_cmd.type} from queue.")

        proprio = self.robot.prepare_obs()
        wbc_obs = self._build_wbc_observation(proprio)
        self._wbc_policy.set_observation(wbc_obs)
        t_now = time.monotonic()
        control_freq = self._control_frequency
        target_time = t_now + 1 / control_freq

        target_upper_body_pose = np.array(
            [action_cmd["target_upper_body_pose"][name] for name in self.sonic_upper_joint_names],
            dtype=np.float32,
        )
        goal = {
            "target_upper_body_pose": target_upper_body_pose,
            "navigate_cmd": action_cmd["navigate_cmd"],
            "base_height_command": action_cmd["base_height_command"],
            "target_time": target_time,
            "interpolation_garbage_collection_time": t_now - 2 / control_freq,
            "timestamp": t_now,
        }
        self._wbc_policy.set_goal(goal)
        wbc_action = self._wbc_policy.get_action(time=t_now)
        self._cached_target_q = self._dwbc_robot_model.get_body_actuated_joints(wbc_action["q"])
        self._cached_left_hand_q = self._dwbc_robot_model.get_hand_actuated_joints(
            wbc_action["q"], side="left"
        )
        self._cached_right_hand_q = self._dwbc_robot_model.get_hand_actuated_joints(
            wbc_action["q"], side="right"
        )

        self._last_base_height_command = float(goal["base_height_command"][0])
        self._last_pred_action = ActionCmd(
            "decoupled_wbc",
            target_q=self._cached_target_q,
            left_hand_q=self._cached_left_hand_q,
            right_hand_q=self._cached_right_hand_q,
        )
        self._global_step_idx += 1
        return self._last_pred_action

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._global_step_idx = 0
        self._last_qpos = None
        self._last_observation = None
        self._last_pred_action = None
        self._last_base_height_command = 0.74
        self._reset_history = True
