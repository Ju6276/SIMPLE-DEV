"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

VLA-JEPA adapter for G1 decoupled-WBC evaluation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
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


DEFAULT_VLAJEPA_INSTRUCTION = "Complete the robot task shown in the scene."


def _resolve_instruction(instruction: str | None) -> str:
    override = os.environ.get("VLAJEPA_INSTRUCTION_OVERRIDE")
    if override:
        return override
    if instruction:
        normalized = " ".join(instruction.split())
        if normalized:
            return normalized
    return DEFAULT_VLAJEPA_INSTRUCTION


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


def _finite_float_or_none(value):
    value = _float_or_none(value)
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _round_float_or_none(value, ndigits: int = 3):
    value = _finite_float_or_none(value)
    if value is None:
        return None
    return round(value, ndigits)


def _first_finite_float(*values):
    for value in values:
        value = _finite_float_or_none(value)
        if value is not None:
            return value
    return None


def _flag_from_timing(policy_timing: dict, key: str) -> int:
    value = _finite_float_or_none(policy_timing.get(key))
    return int(value is not None and value >= 0.5)


def _env_float_or_none(name: str):
    value = os.environ.get(name)
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _int_or_none(value):
    if isinstance(value, (int, float, np.generic)):
        return int(round(float(value)))
    return None


def _response_data(response: dict) -> dict:
    if response.get("ok", None) is False:
        raise RuntimeError(f"VLA-JEPA server inference failed: {response}")
    data = response.get("data")
    if isinstance(data, dict):
        return data
    return response


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


class VlajepaDecoupledWbcAgent(SonicDecoupledWbcAgent):
    def __init__(self, robot, host: str, port: int, upsample_factor: int = 1, **kwargs):
        super().__init__(robot, **kwargs)

        self.server_ip = host
        self.server_port = port
        self.upsample_factor = upsample_factor

        self.client = VlajepaWebsocketClient(host=host, port=port)
        self._experiment_name = os.environ.get("SIMPLE_JEPA_EXPERIMENT", "flash_jepa")
        self._global_step_idx = 0
        self._server_query_idx = 0
        self._episode_idx = -1
        self._last_executed_steps = 0
        self._replan_steps = int(os.environ.get("SIMPLE_JEPA_REPLAN_STEPS", "20"))
        self._full_exec_steps = int(os.environ.get("SIMPLE_JEPA_FULL_EXEC_STEPS", "0"))
        self._use_accepted_prefix = os.environ.get("SIMPLE_JEPA_USE_ACCEPTED_PREFIX", "1") != "0"
        self._timing_log_path = Path(
            os.environ.get(
                "SIMPLE_JEPA_TIMING_LOG",
                "data/evals_decoupled_wbc/vlajepa_timing.jsonl",
            )
        )
        self._timing_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._timing_verbose = os.environ.get("SIMPLE_JEPA_TIMING_VERBOSE", "0") == "1"
        self._timing_warmup_queries = max(0, int(os.environ.get("SIMPLE_JEPA_TIMING_WARMUP_QUERIES", "1")))
        self._timing_warmup_draft_queries = max(
            0, int(os.environ.get("SIMPLE_JEPA_TIMING_WARMUP_DRAFT_QUERIES", "2"))
        )
        self._timing_warmup_full_queries = max(
            0, int(os.environ.get("SIMPLE_JEPA_TIMING_WARMUP_FULL_QUERIES", "0"))
        )
        self._timing_route_query_counts = {"draft": 0, "full": 0}
        self._full_latency_baseline_ms = _env_float_or_none("SIMPLE_JEPA_FULL_BASELINE_MS")
        self._full_latency_baseline_fixed = self._full_latency_baseline_ms is not None
        self._full_latency_baseline_count = int(self._full_latency_baseline_fixed)
        self._timing_round_count = 0
        self._timing_latency_sum_ms = 0.0
        self._timing_exec_sum = 0
        self._timing_accepted_sum = 0
        self._timing_replan_sum = 0
        self._timing_flash_rounds = 0
        self._timing_flash_latency_sum_ms = 0.0
        self._timing_flash_exec_sum = 0
        self._timing_flash_accepted_sum = 0
        self._timing_flash_replan_sum = 0
        self._timing_full_rounds = 0
        self._timing_fallback_rounds = 0
        self._action_log_path = Path(
            os.environ.get(
                "SIMPLE_JEPA_ACTION_LOG",
                "data/evals_decoupled_wbc/vlajepa_executed_actions.jsonl",
            )
        )
        self._action_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_base_height_command = 0.74
        self._reset_history = True

        indices = self._dwbc_robot_model.get_joint_group_indices("upper_body")
        self.sonic_upper_joint_names = [
            name
            for name, idx in self._dwbc_robot_model.joint_to_dof_index.items()
            if idx in indices
        ]

    def _record_timing(self, response, *, client_roundtrip_ms, reset, accepted_prefix_len=None, exec_len=None):
        data = _response_data(response)
        server_timing = response.get("server_timing", data.get("server_timing", {}))
        if not isinstance(server_timing, dict):
            server_timing = {}
        policy_timing = response.get("policy_timing", data.get("policy_timing", {}))
        if not isinstance(policy_timing, dict):
            policy_timing = {}

        policy_time_ms = _float_or_none(server_timing.get("policy_time_ms"))
        if policy_time_ms is None:
            policy_time_ms = _float_or_none(server_timing.get("infer_ms"))

        route_type = _route_type_from_policy_timing(policy_timing)
        accepted = _int_or_none(accepted_prefix_len)
        executed = _int_or_none(exec_len)
        replan_steps = max(1, int(self._replan_steps))
        latency_ms = _first_finite_float(
            policy_timing.get("sample_actions_ms"),
            policy_timing.get("total_ms"),
            policy_time_ms,
            server_timing.get("infer_ms"),
        )
        is_full_round = int(
            route_type == "full"
            or _flag_from_timing(policy_timing, "is_full_pipeline_round")
            or _flag_from_timing(policy_timing, "used_full_fallback")
        )
        is_flash_round = int(route_type == "draft" and not is_full_round)
        used_fallback = _flag_from_timing(policy_timing, "used_full_fallback")

        route_query_idx = int(self._timing_route_query_counts.get(route_type, 0))
        route_warmup_queries = 0
        if route_type == "draft":
            route_warmup_queries = int(self._timing_warmup_draft_queries)
        elif route_type == "full":
            route_warmup_queries = int(self._timing_warmup_full_queries)
        include_in_summary = (
            self._server_query_idx >= self._timing_warmup_queries
            and route_query_idx >= route_warmup_queries
        )

        if is_full_round and latency_ms is not None and not self._full_latency_baseline_fixed and include_in_summary:
            self._full_latency_baseline_count += 1
            if self._full_latency_baseline_ms is None:
                self._full_latency_baseline_ms = latency_ms
            else:
                alpha = 0.25
                self._full_latency_baseline_ms = (
                    (1.0 - alpha) * self._full_latency_baseline_ms + alpha * latency_ms
                )

        baseline_ms = self._full_latency_baseline_ms
        speedup = None
        saved_ms = None
        if latency_ms is not None and baseline_ms is not None and latency_ms > 0.0:
            speedup = baseline_ms / latency_ms
            saved_ms = baseline_ms - latency_ms

        if include_in_summary:
            self._timing_round_count += 1
            if latency_ms is not None:
                self._timing_latency_sum_ms += latency_ms
            if executed is not None:
                self._timing_exec_sum += max(0, executed)
            if accepted is not None:
                self._timing_accepted_sum += max(0, accepted)
                self._timing_replan_sum += replan_steps
            if is_flash_round:
                self._timing_flash_rounds += 1
                if latency_ms is not None:
                    self._timing_flash_latency_sum_ms += latency_ms
                if executed is not None:
                    self._timing_flash_exec_sum += max(0, executed)
                if accepted is not None:
                    self._timing_flash_accepted_sum += max(0, accepted)
                    self._timing_flash_replan_sum += replan_steps
            self._timing_full_rounds += is_full_round
            self._timing_fallback_rounds += used_fallback

        avg_latency_ms = None
        avg_ms_per_action = None
        avg_speedup = None
        flash_avg_latency_ms = None
        flash_avg_ms_per_action = None
        if self._timing_round_count > 0:
            avg_latency_ms = self._timing_latency_sum_ms / float(self._timing_round_count)
        if self._timing_exec_sum > 0:
            avg_ms_per_action = self._timing_latency_sum_ms / float(self._timing_exec_sum)
        if avg_latency_ms is not None and baseline_ms is not None and avg_latency_ms > 0.0:
            avg_speedup = baseline_ms / avg_latency_ms
        if self._timing_flash_rounds > 0:
            flash_avg_latency_ms = self._timing_flash_latency_sum_ms / float(self._timing_flash_rounds)
        if self._timing_flash_exec_sum > 0:
            flash_avg_ms_per_action = self._timing_flash_latency_sum_ms / float(self._timing_flash_exec_sum)

        accept_ratio = round(float(accepted) / float(replan_steps), 3) if accepted is not None else None
        flash_accept_ratio = accept_ratio if is_flash_round and accepted is not None else None
        cum_flash_accept_ratio = (
            round(float(self._timing_flash_accepted_sum) / float(self._timing_flash_replan_sum), 3)
            if self._timing_flash_replan_sum > 0
            else None
        )
        flash_rate = (
            round(float(self._timing_flash_rounds) / float(self._timing_round_count), 3)
            if self._timing_round_count > 0
            else None
        )

        record = {
            "exp": self._experiment_name,
            "t": _round_float_or_none(time.time(), 3),
            "ep": int(self._episode_idx),
            "step": int(self._global_step_idx),
            "q": int(self._server_query_idx),
            "route_q": route_query_idx,
            "reset": bool(reset),
            "measured": int(include_in_summary),
            "route": route_type,
            "exec": executed,
            "accepted": accepted,
            "replan": replan_steps,
            "accept_ratio": accept_ratio,
            "flash_accept_ratio": flash_accept_ratio,
            "lat_ms": _round_float_or_none(latency_ms),
            "rt_ms": _round_float_or_none(client_roundtrip_ms),
            "ms_per_action": (
                round(latency_ms / float(executed), 3)
                if latency_ms is not None and executed is not None and executed > 0
                else None
            ),
            "full_baseline_ms": _round_float_or_none(baseline_ms),
            "speedup": _round_float_or_none(speedup),
            "saved_ms": _round_float_or_none(saved_ms),
            "enc_ms": _round_float_or_none(policy_timing.get("encoder_ms")),
            "prefill_ms": _round_float_or_none(policy_timing.get("vlm_prefill_ms")),
            "draft_ms": _round_float_or_none(policy_timing.get("draft_ms")),
            "verify_ms": _round_float_or_none(policy_timing.get("action_verify_ms")),
            "denoise_ms": _round_float_or_none(policy_timing.get("full_fallback_ms")),
            "dist": _round_float_or_none(policy_timing.get("radius_dist")),
            "prefix_visual_compiled": _flag_from_timing(policy_timing, "prefix_visual_compiled"),
            "verify_compiled": _flag_from_timing(policy_timing, "triton_runtime_verify_compiled"),
            "shared_prefix_full": _flag_from_timing(policy_timing, "jepa_shared_prefix_full"),
            "full": is_full_round,
            "flash": is_flash_round,
            "fallback": used_fallback,
            "sched_full": _flag_from_timing(policy_timing, "scheduled_full_fallback"),
            "avg_lat_ms": _round_float_or_none(avg_latency_ms),
            "avg_ms_per_action": _round_float_or_none(avg_ms_per_action),
            "avg_speedup": _round_float_or_none(avg_speedup),
            "flash_avg_lat_ms": _round_float_or_none(flash_avg_latency_ms),
            "flash_avg_ms_per_action": _round_float_or_none(flash_avg_ms_per_action),
            "cum_accept_ratio": (
                round(float(self._timing_accepted_sum) / float(self._timing_replan_sum), 3)
                if self._timing_replan_sum > 0
                else None
            ),
            "cum_flash_accept_ratio": cum_flash_accept_ratio,
            "full_rate": (
                round(float(self._timing_full_rounds) / float(self._timing_round_count), 3)
                if self._timing_round_count > 0
                else None
            ),
            "flash_rate": flash_rate,
            "fallback_rate": (
                round(float(self._timing_fallback_rounds) / float(self._timing_round_count), 3)
                if self._timing_round_count > 0
                else None
            ),
        }
        if self._timing_verbose:
            actions = data.get("actions")
            record.update(
                {
                    "host": self.server_ip,
                    "port": self.server_port,
                    "policy_time_ms": _round_float_or_none(policy_time_ms),
                    "serve_time_ms": _round_float_or_none(server_timing.get("serve_time_ms")),
                    "ws_unpack_ms": _round_float_or_none(server_timing.get("ws_unpack_ms")),
                    "ws_pack_ms": _round_float_or_none(server_timing.get("ws_pack_ms")),
                    "action_shape": list(actions.shape) if hasattr(actions, "shape") else None,
                    "prefix_ms": _round_float_or_none(policy_timing.get("triton_runtime_prefix_ms")),
                    "prefix_img_ms": _round_float_or_none(policy_timing.get("prefix_image_ms")),
                    "prefix_embed_ms": _round_float_or_none(policy_timing.get("prefix_embed_ms")),
                    "prefix_static_ms": _round_float_or_none(policy_timing.get("prefix_static_ms")),
                    "prefix_proc_ms": _round_float_or_none(policy_timing.get("prefix_processor_ms")),
                    "prefix_h2d_ms": _round_float_or_none(policy_timing.get("prefix_h2d_ms")),
                    "prefix_visual_ms": _round_float_or_none(policy_timing.get("prefix_visual_ms")),
                    "prefix_scatter_ms": _round_float_or_none(policy_timing.get("prefix_scatter_ms")),
                    "prefix_skip_resize": _flag_from_timing(policy_timing, "prefix_skip_resize"),
                    "prefix_on_device": _flag_from_timing(policy_timing, "prefix_process_on_device"),
                    "cache_hit": _flag_from_timing(policy_timing, "prefix_cache_hit"),
                    "server_timing": _jsonable(server_timing),
                    "policy_timing": _jsonable(policy_timing),
                }
            )
        with self._timing_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self._timing_route_query_counts[route_type] = route_query_idx + 1

    def _record_executed_action(self, action_cmd):
        trace = action_cmd["vlajepa_trace"] or {}
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
        **kwargs,
    ):
        self._last_observation = observation
        self._last_qpos = observation["joint_qpos"]

        while len(self._action_queue) == 0:
            state_32d = _build_vlajepa_state(
                observation["joint_qpos"],
                height=self._last_base_height_command,
            )
            payload = {
                "batch_images": [[observation["head_stereo_left"]]],
                "instructions": [_resolve_instruction(instruction)],
                "state": state_32d[None, None, :],
                "reset": self._reset_history,
                "__reset_policy_state__": self._reset_history,
                "__executed_steps__": int(self._last_executed_steps),
            }
            self._reset_history = False

            infer_start = time.perf_counter()
            response = self.client.infer(payload)
            client_roundtrip_ms = (time.perf_counter() - infer_start) * 1000.0
            data = _response_data(response)
            pred_action = np.asarray(data["actions"], dtype=np.float32)
            server_query_idx = self._server_query_idx
            policy_timing = response.get("policy_timing", data.get("policy_timing", {}))
            if not isinstance(policy_timing, dict):
                policy_timing = {}
            accepted_prefix_len = data.get("accepted_prefix_len", policy_timing.get("accepted_prefix_len"))
            route_type = _route_type_from_policy_timing(policy_timing)
            accepted_int = None
            if accepted_prefix_len is not None:
                accepted_int = max(0, int(round(float(accepted_prefix_len))))

            if route_type == "full":
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
                reset=payload["reset"],
                accepted_prefix_len=accepted_prefix_len,
                exec_len=exec_len,
            )
            self._server_query_idx += 1
            print(
                f"step {self._global_step_idx}: Received {pred_action.shape[0]} actions from VLA-JEPA server, "
                f"route {route_type}, executing {exec_len}."
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

            if exec_len <= 0:
                self._last_executed_steps = 0
                continue

            for i in range(exec_len):
                action = pred_action[i]
                navigate_cmd = action[32:36].astype(np.float32)
                for repeat_idx in range(self.upsample_factor):
                    trace = {
                        "server_query_idx": int(server_query_idx),
                        "chunk_action_idx": int(i),
                        "queue_repeat_idx": int(repeat_idx),
                        "route_type": str(route_type),
                        "accepted_prefix_len": _int_or_none(accepted_prefix_len),
                        "exec_len": int(exec_len),
                        "raw_action": np.asarray(action, dtype=np.float32),
                        "policy_timing": {
                            "route_type": policy_timing.get("route_type", route_type),
                            "accepted_prefix_len": _float_or_none(policy_timing.get("accepted_prefix_len")),
                            "radius_dist": _float_or_none(policy_timing.get("radius_dist")),
                            "scheduled_full_fallback": _float_or_none(policy_timing.get("scheduled_full_fallback")),
                            "used_full_fallback": _float_or_none(policy_timing.get("used_full_fallback")),
                            "simple_stability_guard_applied": _float_or_none(
                                policy_timing.get("simple_stability_guard_applied")
                            ),
                            "triton_runtime_prefix_ms": _float_or_none(policy_timing.get("triton_runtime_prefix_ms")),
                            "triton_runtime_draft_compiled": _float_or_none(
                                policy_timing.get("triton_runtime_draft_compiled")
                            ),
                            "triton_runtime_verify_compiled": _float_or_none(
                                policy_timing.get("triton_runtime_verify_compiled")
                            ),
                        },
                    }
                    self.queue_action(
                        ActionCmd(
                            "vla_cmd",
                            target_upper_body_pose=_action_to_upper_body_pose(action),
                            navigate_cmd=navigate_cmd,
                            base_height_command=action[31:32].astype(np.float32),
                            vlajepa_trace=trace,
                        )
                    )
            self._last_executed_steps = int(exec_len)

        action_cmd = super().get_action(observation, instruction, **kwargs)
        if action_cmd.type != "vla_cmd":
            raise ValueError(f"Unexpected action type {action_cmd.type} from queue.")
        self._record_executed_action(action_cmd)

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
        self._episode_idx += 1
        self._global_step_idx = 0
        self._server_query_idx = 0
        self._last_qpos = None
        self._last_observation = None
        self._last_pred_action = None
        self._last_base_height_command = 0.74
        self._reset_history = True
        self._last_executed_steps = 0
