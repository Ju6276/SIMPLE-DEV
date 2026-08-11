"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

import json
import os
from pathlib import Path
import time
import numpy as np
from simple.agents.sonic_decoupled_wbc_agent import SonicDecoupledWbcAgent
from simple.core.action import ActionCmd
from openpi_client import websocket_client_policy as _websocket_client_policy

STATE_SLICES = [ # shoule be consistent with scripts/postprocess_psi0.py
    ("left_hand_thumb", 29, 32),
    ("left_hand_middle", 34, 36),
    ("left_hand_index", 32, 34),
    ("right_hand", 36, 43),
    ("left_arm", 15, 22),
    ("right_arm", 22, 29),
]

def from_psi0_upper_joints(psi0_action):
    return np.concatenate([
        psi0_action[14:28],
        psi0_action[0:3], # left thumb
        psi0_action[5:7], # left index
        psi0_action[3:5], # left middle
        psi0_action[7:14], # right hand
    ])


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _float_or_none(value):
    if isinstance(value, (int, float, np.generic)):
        return float(value)
    return None


def _int_or_none(value):
    if isinstance(value, (int, float, np.generic)):
        return int(round(float(value)))
    return None


def _route_type_from_policy_timing(policy_timing):
    route_type = policy_timing.get("route_type")
    if isinstance(route_type, str) and route_type:
        return route_type
    for key in ("is_full_pipeline_round", "used_full_fallback"):
        value = _float_or_none(policy_timing.get(key))
        if value is not None and value >= 0.5:
            return "full"
    full_ms = _float_or_none(policy_timing.get("full_fallback_ms"))
    if full_ms is not None and full_ms > 0.0:
        return "full"
    return "draft"


class Pi05DecoupledWbcAgent(SonicDecoupledWbcAgent):
    def __init__(self, robot, host: str, port: int, upsample_factor=1, **kwargs):
        super().__init__(robot, **kwargs)
        
        self.server_ip = host # if access server host inside docker container
        self.server_port = port
        self.upsample_factor = upsample_factor

        self.client = _websocket_client_policy.WebsocketClientPolicy(
            host=host,
            port=port,
            api_key=None,
        )
        self._global_step_idx = 0
        self._server_query_idx = 0
        self._episode_idx = -1
        self._last_executed_steps = 0
        self._replan_steps = int(os.environ.get("SIMPLE_PI05_REPLAN_STEPS", "12"))
        self._full_exec_steps = int(os.environ.get("SIMPLE_PI05_FULL_EXEC_STEPS", "0"))
        self._use_accepted_prefix = os.environ.get("SIMPLE_PI05_USE_ACCEPTED_PREFIX", "1") != "0"
        self._timing_log_path = Path(
            os.environ.get(
                "SIMPLE_PI05_TIMING_LOG",
                "data/evals_decoupled_wbc/pi05_timing.jsonl",
            )
        )
        self._timing_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._action_log_path = Path(
            os.environ.get(
                "SIMPLE_PI05_ACTION_LOG",
                "data/evals_decoupled_wbc/pi05_executed_actions.jsonl",
            )
        )
        self._action_log_path.parent.mkdir(parents=True, exist_ok=True)

        # last command (high level input to lower policy)
        self._last_cmd_torso_rpyh = np.array([0, 0, 0, 0.74]) # FIXME hardcoded for g1 wholebody, need to be more general in the future
        self._reset_history = True

        indices = self._dwbc_robot_model.get_joint_group_indices("upper_body")
        self.sonic_upper_joint_names = [name for name, idx in self._dwbc_robot_model.joint_to_dof_index.items() if idx in indices]

    def _record_timing(self, response, *, client_roundtrip_ms, reset, accepted_prefix_len=None, exec_len=None):
        server_timing = response.get("server_timing", {})
        if not isinstance(server_timing, dict):
            server_timing = {}
        policy_timing = response.get("policy_timing", {})
        if not isinstance(policy_timing, dict):
            policy_timing = {}

        policy_time_ms = _float_or_none(server_timing.get("policy_time_ms"))
        if policy_time_ms is None:
            policy_time_ms = _float_or_none(server_timing.get("infer_ms"))

        actions = response.get("actions")
        record = {
            "timestamp_s": time.time(),
            "episode_idx": self._episode_idx,
            "global_step_idx": self._global_step_idx,
            "server_query_idx": self._server_query_idx,
            "reset": bool(reset),
            "host": self.server_ip,
            "port": self.server_port,
            "client_roundtrip_ms": float(client_roundtrip_ms),
            "policy_time_ms": policy_time_ms,
            "serve_time_ms": _float_or_none(server_timing.get("serve_time_ms")),
            "ws_unpack_ms": _float_or_none(server_timing.get("ws_unpack_ms")),
            "ws_pack_ms": _float_or_none(server_timing.get("ws_pack_ms")),
            "prev_ws_send_ms": _float_or_none(server_timing.get("prev_ws_send_ms")),
            "prev_total_ms": _float_or_none(server_timing.get("prev_total_ms")),
            "action_shape": list(actions.shape) if hasattr(actions, "shape") else None,
            "accepted_prefix_len": _int_or_none(accepted_prefix_len),
            "exec_len": _int_or_none(exec_len),
            "replan_steps": int(self._replan_steps),
            "full_exec_steps": int(self._full_exec_steps),
            "server_timing": _jsonable(server_timing),
            "policy_timing": _jsonable(policy_timing),
        }
        for key, value in policy_timing.items():
            numeric_value = _float_or_none(value)
            if numeric_value is not None:
                record[f"policy_timing_{key}"] = numeric_value
        with self._timing_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def _record_executed_action(self, action_cmd):
        trace = action_cmd["pi05_trace"] or {}
        raw_action = np.asarray(trace.get("raw_action", []), dtype=np.float32)
        record = {
            "timestamp_s": time.time(),
            "episode_idx": self._episode_idx,
            "global_step_idx": self._global_step_idx,
            "server_query_idx": trace.get("server_query_idx"),
            "chunk_action_idx": trace.get("chunk_action_idx"),
            "queue_repeat_idx": trace.get("queue_repeat_idx"),
            "route_type": trace.get("route_type"),
            "accepted_prefix_len": trace.get("accepted_prefix_len"),
            "exec_len": trace.get("exec_len"),
            "replan_steps": int(self._replan_steps),
            "full_exec_steps": int(self._full_exec_steps),
            "action_dim": int(raw_action.shape[-1]) if raw_action.ndim > 0 else 0,
            "action": raw_action.tolist(),
            "upper_0_28": raw_action[:28].tolist() if raw_action.shape[-1] >= 28 else [],
            "waist_28_31": raw_action[28:31].tolist() if raw_action.shape[-1] >= 31 else [],
            "base_height_31": float(raw_action[31]) if raw_action.shape[-1] >= 32 else None,
            "nav_32_36": raw_action[32:36].tolist() if raw_action.shape[-1] >= 36 else [],
            "target_upper_body_pose": _jsonable(action_cmd["target_upper_body_pose"]),
            "navigate_cmd": _jsonable(action_cmd["navigate_cmd"]),
            "base_height_command": _jsonable(action_cmd["base_height_command"]),
            "policy_timing": _jsonable(trace.get("policy_timing", {})),
        }
        with self._action_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def get_action(
        self, 
        observation, 
        instruction=None, 
        info=None, 
        conditions=None, 
        **kwargs
    ):
        self._last_observation = observation
        self._last_qpos = observation["joint_qpos"]

        while len(self._action_queue) == 0:
            # send query to server

            proprio = observation["joint_qpos"][None]
            waist_rpy = [proprio[:,13:15],proprio[:,12:13]]
            states = np.concatenate(
                [proprio[:, s:e] for _, s, e in STATE_SLICES] + waist_rpy + [
                    self._last_cmd_torso_rpyh[None][:, -1:]
                ],
                axis=1,
            ).astype(np.float32) # (1, 32)
            # state_dict = {"states": states} # np.zeros_like()
            obs = {
                "observation/image": observation["head_stereo_left"],
                "states": states[0, :28], 
                "prompt": instruction or "As a smart robot agent, what to do next?",
                "__executed_steps__": int(self._last_executed_steps),
            }
            
            if self._reset_history:
                obs["__reset_policy_state__"] = True
                self._reset_history = False
            else:
                obs["__reset_policy_state__"] = False

            infer_start = time.perf_counter()
            response = self.client.infer(obs)
            client_roundtrip_ms = (time.perf_counter() - infer_start) * 1000.0
            pred_action = response["actions"]
            server_query_idx = self._server_query_idx
            policy_timing = response.get("policy_timing", {})
            if not isinstance(policy_timing, dict):
                policy_timing = {}
            accepted_prefix_len = response.get("accepted_prefix_len", policy_timing.get("accepted_prefix_len"))
            route_type = _route_type_from_policy_timing(policy_timing)
            accepted_int = None
            if accepted_prefix_len is not None:
                accepted_int = max(0, int(round(float(accepted_prefix_len))))
            if route_type == "full":
                # Speculative serving verifies only the first replan window. Keep full
                # rounds on the same cadence as draft rounds unless explicitly overridden.
                exec_cap = int(self._full_exec_steps) if self._full_exec_steps > 0 else max(1, int(self._replan_steps))
                if accepted_int is not None and self._full_exec_steps <= 0:
                    exec_cap = min(exec_cap, accepted_int)
                exec_len = min(int(pred_action.shape[0]), exec_cap)
                if self._full_exec_steps > 0:
                    exec_len = min(exec_len, int(self._full_exec_steps))
            elif self._use_accepted_prefix and accepted_int is not None:
                exec_len = min(int(pred_action.shape[0]), max(1, int(self._replan_steps)), accepted_int)
            else:
                exec_len = min(int(pred_action.shape[0]), max(1, int(self._replan_steps)))
            self._record_timing(
                response,
                client_roundtrip_ms=client_roundtrip_ms,
                reset=obs["__reset_policy_state__"],
                accepted_prefix_len=accepted_prefix_len,
                exec_len=exec_len,
            )
            self._server_query_idx += 1
            print(
                f"step {self._global_step_idx}: Received {pred_action.shape[0]} actions from server, "
                f"route {route_type}, executing {exec_len}."
            )
            
            """ # DEBUG
            if self._global_step_idx < 180:
                pred_action = pred_action.copy()
                pred_action[:, 31] = 0.72

            if self._global_step_idx <= 180:
                pred_action = pred_action.copy()
                pred_action[:, 32] = 0.2 """

            print(pred_action[:, 31])
            print(pred_action[:, 32])

            if exec_len <= 0:
                self._last_executed_steps = 0
                continue

            for i in range(exec_len):
                for repeat_idx in range(self.upsample_factor): # account for upsampling during training
                    target_qpos = dict(
                        zip(
                            self.robot.joint_names[15:],
                            from_psi0_upper_joints(pred_action[i][:28])
                        )
                    )
                    target_waist_qpos = {
                        "waist_yaw_joint": pred_action[i][30], 
                        "waist_roll_joint": pred_action[i][28], 
                        "waist_pitch_joint": pred_action[i][29] 
                    }
                    trace = {
                        "server_query_idx": int(server_query_idx),
                        "chunk_action_idx": int(i),
                        "queue_repeat_idx": int(repeat_idx),
                        "route_type": str(route_type),
                        "accepted_prefix_len": _int_or_none(accepted_prefix_len),
                        "exec_len": int(exec_len),
                        "raw_action": np.asarray(pred_action[i], dtype=np.float32),
                        "policy_timing": {
                            "route_type": policy_timing.get("route_type", route_type),
                            "accepted_prefix_len": _float_or_none(policy_timing.get("accepted_prefix_len")),
                            "radius_dist": _float_or_none(policy_timing.get("radius_dist")),
                            "scheduled_full_fallback": _float_or_none(policy_timing.get("scheduled_full_fallback")),
                            "used_full_fallback": _float_or_none(policy_timing.get("used_full_fallback")),
                            "simple_stability_guard_applied": _float_or_none(
                                policy_timing.get("simple_stability_guard_applied")
                            ),
                        },
                    }
                    self.queue_action(ActionCmd(
                        "vla_cmd", 
                        target_upper_body_pose={**target_qpos, **target_waist_qpos},  # (31,)
                        navigate_cmd=pred_action[i][32:36],
                        base_height_command=pred_action[i][31:32],
                        pi05_trace=trace,
                    ))
            self._last_executed_steps = int(exec_len)

        action_cmd = super().get_action(observation, instruction, **kwargs)
        if action_cmd.type == "vla_cmd":
            self._record_executed_action(action_cmd)
            proprio = self.robot.prepare_obs()
            wbc_obs = self._build_wbc_observation(proprio)
            self._wbc_policy.set_observation(wbc_obs)
            t_now = time.monotonic()
            control_freq = self._control_frequency
            target_time = t_now + 1 / control_freq

            target_upper_body_pose = np.array([
                action_cmd["target_upper_body_pose"][jName] for jName in self.sonic_upper_joint_names
            ], dtype=np.float32)
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
            self._cached_left_hand_q = self._dwbc_robot_model.get_hand_actuated_joints(wbc_action["q"], side="left")
            self._cached_right_hand_q = self._dwbc_robot_model.get_hand_actuated_joints(wbc_action["q"], side="right")

            # createa a new ActionCmd for the g1_sonic robot
            action_cmd = ActionCmd(
                "decoupled_wbc",
                target_q=self._cached_target_q,
                left_hand_q=self._cached_left_hand_q,
                right_hand_q=self._cached_right_hand_q,
            )
            self._last_cmd_torso_rpyh = np.array([0, 0, 0, goal["base_height_command"][0]])
        else:
            raise ValueError(f"Unexpected action type {action_cmd.type} from queue.")

        self._last_pred_action = action_cmd
        self._global_step_idx += 1
        return action_cmd
    
    def reset(self, **kwargs):
        super().reset(**kwargs)  # clear queue

        self._episode_idx += 1
        self._global_step_idx = 0
        self._server_query_idx = 0
        self._last_qpos = None
        self._last_observation = None
        self._last_pred_action = None
        self._reset_history = True
        self._last_executed_steps = 0
        self._last_cmd_torso_rpyh = np.array([0, 0, 0, 0.74])
