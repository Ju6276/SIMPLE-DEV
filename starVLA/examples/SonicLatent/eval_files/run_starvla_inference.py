"""
starVLA inference runner for Sonic latent actions.

This script mirrors the deployment flow of `gear_sonic/scripts/run_vla_inference.py`
but talks to the starVLA WebSocket policy server instead of Isaac-GR00T's
PolicyClient server.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import threading
import time

import cv2 as cv
import numpy as np
import tyro
import zmq

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from starVLA.model.tools import read_mode_config

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
from gear_sonic.data.features_sonic_vla import get_g1_robot_model
from gear_sonic.utils.data_collection.keyboard_subscriber import (
    DEFAULT_ZMQ_KEYBOARD_PORT,
    ZMQKeyboardSubscriber,
)
from gear_sonic.utils.data_collection.telemetry import Telemetry
from gear_sonic.utils.data_collection.transforms import compute_projected_gravity
from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber
from gear_sonic.utils.inference.initial_poses import LATENT_INITIAL_MOTION_TOKEN
from gear_sonic.utils.inference.vla_utils import (
    calculate_latency_compensated_index,
    should_trigger_new_inference,
)
from gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver import (
    G1GripperInverseKinematicsSolver,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    pack_pose_message,
)


@dataclass
class InferenceConfig:
    ckpt_path: str
    """Trained starVLA checkpoint path, used for loading normalization stats."""

    host: str = "127.0.0.1"
    """starVLA WebSocket policy server host."""

    port: int = 10093
    """starVLA WebSocket policy server port."""

    action_publish_rate: int = 50
    """Rate at which individual actions are published to the C++ control loop (Hz)."""

    action_horizon: int = 0
    """Deprecated compatibility flag. Actual action chunk size is read from the checkpoint."""

    rate: float = 4
    """Rate at which we run the forward pass of the VLA policy (Hz)."""

    camera_host: str = "localhost"
    camera_port: int = 5555
    state_zmq_host: str = "localhost"
    state_zmq_port: int = 5557
    action_zmq_host: str = "localhost"
    action_zmq_port: int = 5556
    keyboard_zmq_host: str = "localhost"
    keyboard_zmq_port: int = DEFAULT_ZMQ_KEYBOARD_PORT
    prompt: str = "pick up the cylinder and throw it into the trash bin"
    image_size: tuple[int, int] = (224, 224)
    use_ddim: bool = False
    num_ddim_steps: int = 10
    verbose_timing: bool = False
    log_action_stats: bool = False
    latency_compensation: bool = False
    replan_steps: int = int(os.getenv("SONICSTAR_REPLAN_STEPS", "0"))
    """If >0, trigger the next policy request after this many published control steps."""
    timing_log: str | None = None
    action_log: str | None = None


def print_green(x):
    print(f"\033[92m{x}\033[0m")


def pack_latent_action_message(
    motion_token: np.ndarray,
    frame_index: np.ndarray,
    left_hand_joints: np.ndarray = None,
    right_hand_joints: np.ndarray = None,
) -> bytes:
    motion_token = np.asarray(motion_token, dtype=np.float32)
    frame_index = np.asarray(frame_index, dtype=np.int64)

    if frame_index.ndim == 0:
        frame_index = np.array([frame_index], dtype=np.int64)
    elif frame_index.shape[0] != 1:
        frame_index = frame_index[:1]

    if motion_token.ndim == 1:
        motion_token = motion_token.reshape(1, -1)

    pose_data = {
        "token_state": motion_token,
        "frame_index": frame_index,
    }

    if left_hand_joints is not None:
        left_hand_joints = np.asarray(left_hand_joints, dtype=np.float32)
        if left_hand_joints.ndim == 1:
            left_hand_joints = left_hand_joints.reshape(1, 7)
        pose_data["left_hand_joints"] = left_hand_joints

    if right_hand_joints is not None:
        right_hand_joints = np.asarray(right_hand_joints, dtype=np.float32)
        if right_hand_joints.ndim == 1:
            right_hand_joints = right_hand_joints.reshape(1, 7)
        pose_data["right_hand_joints"] = right_hand_joints

    return pack_pose_message(pose_data, topic="pose", version=4)


def _compute_closed_hand_joints(side: str) -> np.ndarray:
    side_str = "left" if side.upper() == "L" else "right"
    solver = G1GripperInverseKinematicsSolver(side=side_str)
    return solver._get_middle_close_q_desired().astype(np.float32)


def _sleep_remaining(t_start: float, loop_period: float):
    elapsed = time.monotonic() - t_start
    remaining = loop_period - elapsed
    if remaining > 0:
        time.sleep(remaining)


class StarVLAPolicyAdapter:
    """Thin client-side adapter around the starVLA WebSocket server."""

    def __init__(self, ckpt_path: str, host: str, port: int, image_size: tuple[int, int]):
        self.client = WebsocketClientPolicy(host=host, port=port)
        self.ckpt_path = Path(ckpt_path)
        self.model_config, self.norm_stats = read_mode_config(self.ckpt_path)
        self.action_stats = self._get_action_stats()
        self.state_stats = self._get_state_stats()
        self.action_chunk_size = self.model_config["framework"]["action_model"]["future_action_window_size"] + 1
        self.image_size = tuple(image_size)

    def _get_action_stats(self) -> dict:
        assert len(self.norm_stats) >= 1, "No normalization stats found in checkpoint config."
        dataset_key = next(iter(self.norm_stats.keys()))
        return self.norm_stats[dataset_key]["action"]

    def _get_state_stats(self) -> dict:
        assert len(self.norm_stats) >= 1, "No normalization stats found in checkpoint config."
        dataset_key = next(iter(self.norm_stats.keys()))
        return self.norm_stats[dataset_key]["state"]

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        state_min = np.asarray(self.state_stats["min"], dtype=np.float32)
        state_max = np.asarray(self.state_stats["max"], dtype=np.float32)
        state = np.asarray(state, dtype=np.float32)
        mask = state_min != state_max
        normalized = np.zeros_like(state, dtype=np.float32)
        normalized[..., mask] = 2.0 * (state[..., mask] - state_min[mask]) / (
            state_max[mask] - state_min[mask]
        ) - 1.0
        return normalized

    def _unnormalize_actions(self, normalized_actions: np.ndarray) -> np.ndarray:
        mask = self.action_stats.get("mask", np.ones_like(self.action_stats["min"], dtype=bool))
        action_high = np.asarray(self.action_stats["max"], dtype=np.float32)
        action_low = np.asarray(self.action_stats["min"], dtype=np.float32)
        normalized_actions = np.clip(normalized_actions, -1, 1)
        return np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        ).astype(np.float32)

    def predict_action(
        self,
        image: np.ndarray,
        state: np.ndarray,
        language_prompt: str,
        use_ddim: bool,
        num_ddim_steps: int,
    ) -> dict:
        resized = cv.resize(image, self.image_size, interpolation=cv.INTER_AREA)
        normalized_state = self._normalize_state(state)
        example = {
            "image": [resized],
            "lang": language_prompt,
            "state": normalized_state[np.newaxis, :].astype(np.float32),
        }
        request_t0 = time.perf_counter()
        response = self.client.predict_action(
            {
                "examples": [example],
                "do_sample": False,
                "use_ddim": use_ddim,
                "num_ddim_steps": num_ddim_steps,
            }
        )
        client_roundtrip_ms = (time.perf_counter() - request_t0) * 1000.0
        normalized_actions = response["data"]["normalized_actions"][0]
        policy_timing = dict(response["data"].get("policy_timing", {}) or {})
        policy_timing.setdefault("client_roundtrip_ms", float(client_roundtrip_ms))
        policy_timing.setdefault("sample_actions_ms", float(client_roundtrip_ms))
        policy_timing.setdefault("total_ms", float(client_roundtrip_ms))
        actions = self._unnormalize_actions(normalized_actions)
        return {
            "motion_token": actions[:, :64],
            "left_hand_joints": actions[:, 64:71],
            "right_hand_joints": actions[:, 71:78],
            "normalized_actions": normalized_actions,
            "policy_timing": policy_timing,
        }


def prepare_observation_from_sensors(
    camera_subscriber,
    state_subscriber,
    robot_model,
    language_prompt: str,
    log_errors: bool = False,
):
    camera_msg = camera_subscriber.read()
    if camera_msg is None:
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for camera msg..", flush=True)
        return None

    state_msg = state_subscriber.get_msg()
    if state_msg is None:
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for state msg..", flush=True)
        return None

    image = camera_msg["images"]["ego_view"]

    left_hand_q = np.asarray(state_msg["left_hand_q"], dtype=np.float32).copy()
    right_hand_q = np.asarray(state_msg["right_hand_q"], dtype=np.float32).copy()
    body_q = np.asarray(state_msg["body_q"], dtype=np.float32)

    # Copy index finger data to middle finger (hardware coupling)
    left_hand_q[5] = left_hand_q[3]
    left_hand_q[6] = left_hand_q[4]

    base_quat = np.asarray(state_msg["base_quat"], dtype=np.float64)
    assert base_quat.shape == (4,), "base_quat must have shape (4,)"
    projected_gravity = compute_projected_gravity(base_quat).astype(np.float32)

    whole_q = robot_model.get_configuration_from_actuated_joints(
        body_actuated_joint_values=body_q,
        left_hand_actuated_joint_values=left_hand_q,
        right_hand_actuated_joint_values=right_hand_q,
    ).astype(np.float32)

    state = np.concatenate([whole_q, projected_gravity], axis=0)
    assert state.shape[0] == 46, f"Expected state dim 46, got {state.shape[0]}"

    return {
        "image": image,
        "state": state,
        "language_prompt": language_prompt,
        "timestamps": camera_msg["timestamps"]["ego_view"],
    }


def _format_range(name: str, value: np.ndarray) -> str:
    value = np.asarray(value, dtype=np.float32)
    return (
        f"{name}: shape={value.shape}, "
        f"min={value.min():.4f}, max={value.max():.4f}, "
        f"first={np.array2string(value[0], precision=3, suppress_small=True)}, "
        f"last={np.array2string(value[-1], precision=3, suppress_small=True)}"
    )


def _format_action_samples(name: str, value: np.ndarray) -> str:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 1:
        return f"{name}[0]={np.array2string(value, precision=3, suppress_small=True)}"
    indices = sorted(set([0, value.shape[0] // 2, value.shape[0] - 1]))
    parts = [
        f"{name}[{idx}]={np.array2string(value[idx], precision=3, suppress_small=True)}"
        for idx in indices
    ]
    return "; ".join(parts)


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _float_or_none(value):
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _int_or_none(value):
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _round_float_or_none(value, digits: int = 3):
    value = _float_or_none(value)
    return None if value is None else round(value, digits)


def _first_finite_float(*values):
    for value in values:
        value = _float_or_none(value)
        if value is not None:
            return value
    return None


def _flag_from_timing(timing: dict, key: str) -> int:
    value = timing.get(key)
    if isinstance(value, str):
        return int(value.strip().lower() in {"1", "true", "yes", "y", "on"})
    return int(bool(value))


def _route_type_from_policy_timing(policy_timing: dict) -> str:
    route = policy_timing.get("route_type")
    if route is not None:
        return str(route)
    if _flag_from_timing(policy_timing, "used_full_fallback"):
        return "full"
    if policy_timing.get("draft_ms") is not None or policy_timing.get("accepted_prefix_len") is not None:
        return "draft"
    return "full"


def _action_summary(action: np.ndarray) -> dict:
    action = np.asarray(action, dtype=np.float32)
    if action.size == 0:
        return {
            "action_shape": list(action.shape),
            "action_min": None,
            "action_max": None,
            "action_mean": None,
            "action_std": None,
        }
    return {
        "action_shape": list(action.shape),
        "action_min": float(np.min(action)),
        "action_max": float(np.max(action)),
        "action_mean": float(np.mean(action)),
        "action_std": float(np.std(action)),
    }


def _append_jsonl(path: Path | None, row: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_json_ready(row), sort_keys=True) + "\n")


def run_policy_inference_and_process(
    policy: StarVLAPolicyAdapter,
    observation: dict,
    use_ddim: bool,
    num_ddim_steps: int,
    log_action_stats: bool = False,
):
    try:
        processed_action = policy.predict_action(
            image=observation["image"],
            state=observation["state"],
            language_prompt=observation["language_prompt"],
            use_ddim=use_ddim,
            num_ddim_steps=num_ddim_steps,
        )
        if np.abs(processed_action["motion_token"]).max() > 1.25:
            print(
                f"[Warning] motion_token max ({np.abs(processed_action['motion_token']).max():.4f}) > 1.25. "
                "Exceeds action bound, skipping."
            )
            return None
        if log_action_stats:
            print_green(_format_range("motion_token", processed_action["motion_token"]))
            print_green(_format_range("left_hand", processed_action["left_hand_joints"]))
            print_green(_format_range("right_hand", processed_action["right_hand_joints"]))
            print_green(_format_action_samples("left_hand", processed_action["left_hand_joints"]))
            print_green(_format_action_samples("right_hand", processed_action["right_hand_joints"]))
        return processed_action
    except Exception as e:
        print(f"Error in inference: {e}")
        import traceback

        traceback.print_exc()
        return None


def _inference_worker_loop(
    inference_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event,
    busy_event: threading.Event,
    prepare_obs_fn,
    inference_fn,
):
    while not stop_event.is_set():
        try:
            try:
                request_token = inference_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            busy_event.set()
            try:
                observation = prepare_obs_fn()
                if observation is None:
                    continue
                inference_start_time = time.monotonic()
                processed_action = inference_fn(observation)
                if processed_action is not None:
                    try:
                        result_queue.put_nowait((request_token, processed_action, inference_start_time))
                    except queue.Full:
                        try:
                            result_queue.get_nowait()
                            result_queue.put_nowait((request_token, processed_action, inference_start_time))
                        except queue.Empty:
                            result_queue.put_nowait((request_token, processed_action, inference_start_time))
            finally:
                busy_event.clear()
        except Exception as e:
            print(f"Error in inference worker thread: {e}")
            import traceback

            traceback.print_exc()


def main(config: InferenceConfig):
    timing_log_raw = config.timing_log or os.getenv("SONICSTAR_TIMING_LOG", "")
    action_log_raw = config.action_log or os.getenv("SONICSTAR_ACTION_LOG", "")
    timing_log_path = Path(timing_log_raw).expanduser() if timing_log_raw else None
    action_log_path = Path(action_log_raw).expanduser() if action_log_raw else None
    action_log_mode = os.getenv("SONICSTAR_ACTION_LOG_MODE", "chunk").strip().lower()
    if action_log_mode not in {"chunk", "executed"}:
        raise ValueError("SONICSTAR_ACTION_LOG_MODE must be 'chunk' or 'executed'")
    experiment_name = os.getenv("SONICSTAR_EXPERIMENT_NAME", "sonicstar")
    episode_idx = int(os.getenv("SONICSTAR_AUTO_EVAL_EPISODE_OFFSET", "0"))
    timing_warmup_queries = max(0, int(os.getenv("SONICSTAR_TIMING_WARMUP_QUERIES", "1")))
    timing_warmup_draft_queries = max(0, int(os.getenv("SONICSTAR_TIMING_WARMUP_DRAFT_QUERIES", "2")))
    timing_warmup_full_queries = max(0, int(os.getenv("SONICSTAR_TIMING_WARMUP_FULL_QUERIES", "0")))
    pause_loop = True
    robot_model = get_g1_robot_model(waist_location="lower_and_upper_body")
    policy = StarVLAPolicyAdapter(
        ckpt_path=config.ckpt_path,
        host=config.host,
        port=config.port,
        image_size=config.image_size,
    )
    print(f"Connected to starVLA policy server at {config.host}:{config.port}")
    print_green(f"Action chunk size from checkpoint: {policy.action_chunk_size}")

    state_subscriber = ZMQStateSubscriber(host=config.state_zmq_host, port=config.state_zmq_port)
    camera_subscriber = ComposedCameraClientSensor(server_ip=config.camera_host, port=config.camera_port)

    zmq_context = zmq.Context()
    zmq_socket = zmq_context.socket(zmq.PUB)
    zmq_socket.bind(f"tcp://{config.action_zmq_host}:{config.action_zmq_port}")
    time.sleep(0.1)
    print_green(f"ZMQ action socket bound to tcp://{config.action_zmq_host}:{config.action_zmq_port}")

    keyboard_listener = ZMQKeyboardSubscriber(port=config.keyboard_zmq_port, host=config.keyboard_zmq_host)
    telemetry = Telemetry(window_size=100)

    loop_period = 1.0 / config.action_publish_rate
    cpp_loop_running = False
    cpp_mode = "OFF"
    initial_pose_left_hand_closed = False
    initial_pose_right_hand_closed = False
    cached_action_chunk = None
    action_chunk_index = 0
    current_replan_steps = (
        int(min(int(config.replan_steps), policy.action_chunk_size))
        if int(config.replan_steps) > 0
        else policy.action_chunk_size
    )
    last_inference_time = 0.0
    inference_interval = 1.0 / config.rate
    zmq_frame_counter = 0
    published_action_steps = 0
    server_query_idx = 0
    reset_for_next_policy_query = True
    timing_route_query_counts: dict[str, int] = {}
    timing_round_count = 0
    timing_latency_sum_ms = 0.0
    timing_exec_sum = 0
    timing_accepted_sum = 0
    timing_replan_sum = 0
    timing_flash_rounds = 0
    timing_flash_latency_sum_ms = 0.0
    timing_flash_exec_sum = 0
    timing_flash_accepted_sum = 0
    timing_flash_replan_sum = 0
    timing_full_rounds = 0
    timing_fallback_rounds = 0
    full_latency_baseline_ms = _float_or_none(os.getenv("SONICSTAR_FULL_BASELINE_MS"))
    full_latency_baseline_fixed = full_latency_baseline_ms is not None
    language_prompt_ref: list[str] = [config.prompt]
    prompt_prefix = "prompt:"

    def publish_initial_pose():
        left_hand = _compute_closed_hand_joints("L") if initial_pose_left_hand_closed else np.zeros(7, dtype=np.float32)
        right_hand = _compute_closed_hand_joints("R") if initial_pose_right_hand_closed else np.zeros(7, dtype=np.float32)
        zmq_message = pack_latent_action_message(
            motion_token=LATENT_INITIAL_MOTION_TOKEN,
            frame_index=np.array([0], dtype=np.int64),
            left_hand_joints=left_hand,
            right_hand_joints=right_hand,
        )
        zmq_socket.send(zmq_message)
        print_green("Sent latent initial pose via ZMQ")
        time.sleep(1.0)

    def send_cpp_control_command(
        start: bool,
        planner: bool = False,
        *,
        repeat: int = 1,
        interval_sec: float = 0.01,
    ):
        nonlocal cpp_loop_running, cpp_mode
        try:
            cmd_msg = build_command_message(start=start, stop=not start, planner=planner)
            for _ in range(max(1, int(repeat))):
                zmq_socket.send(cmd_msg)
                time.sleep(max(0.0, float(interval_sec)))
            cpp_loop_running = start
            cpp_mode = "PLANNER" if (start and planner) else ("POSE" if start else "OFF")
            action_str = "start" if start else "stop"
            mode_str = "planner" if planner else "pose"
            print_green(f"Sent ZMQ command: {action_str} control loop ({mode_str} mode), repeat={max(1, int(repeat))}")
            return True
        except Exception as e:
            print(f"Warning: Failed to send control command: {e}")
            return False

    def check_keyboard_input():
        nonlocal pause_loop, cpp_loop_running, cpp_mode
        nonlocal initial_pose_left_hand_closed, initial_pose_right_hand_closed
        nonlocal cached_action_chunk, action_chunk_index, last_inference_time, zmq_frame_counter
        nonlocal published_action_steps, server_query_idx, reset_for_next_policy_query

        key = keyboard_listener.read_msg()
        if key is None:
            return

        if key.startswith(prompt_prefix):
            new_prompt = key[len(prompt_prefix):]
            if new_prompt:
                old_prompt = language_prompt_ref[0]
                language_prompt_ref[0] = new_prompt
                print_green(f'Inference prompt changed: "{old_prompt}" -> "{new_prompt}"')
            return

        if key == "i":
            zmq_frame_counter = 0
            publish_initial_pose()
            cached_action_chunk = None
            action_chunk_index = 0
            published_action_steps = 0
            server_query_idx = 0
            timing_route_query_counts.clear()
            reset_for_next_policy_query = True
            if cpp_loop_running and cpp_mode == "PLANNER":
                send_cpp_control_command(start=True, planner=False)
        elif key == "p":
            pause_loop = not pause_loop
            print(f"{'Paused' if pause_loop else 'Resumed'} policy loop")
        elif key == "k":
            if cpp_loop_running:
                send_cpp_control_command(start=False, planner=(cpp_mode == "PLANNER"))
            else:
                send_cpp_control_command(start=True, planner=True)
        elif key == "[":
            initial_pose_left_hand_closed = not initial_pose_left_hand_closed
        elif key == "]":
            initial_pose_right_hand_closed = not initial_pose_right_hand_closed

    inference_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)
    inference_stop_event = threading.Event()
    inference_busy_event = threading.Event()

    inference_worker_thread = threading.Thread(
        target=_inference_worker_loop,
        args=(
            inference_queue,
            result_queue,
            inference_stop_event,
            inference_busy_event,
            lambda: prepare_observation_from_sensors(
                camera_subscriber=camera_subscriber,
                state_subscriber=state_subscriber,
                robot_model=robot_model,
                language_prompt=language_prompt_ref[0],
                log_errors=True,
            ),
            lambda obs: run_policy_inference_and_process(
                policy=policy,
                observation=obs,
                use_ddim=config.use_ddim,
                num_ddim_steps=config.num_ddim_steps,
                log_action_stats=config.log_action_stats,
            ),
        ),
        daemon=True,
    )
    inference_worker_thread.start()

    waiting_for_replan_result = False
    replan_request_time = 0.0

    def _queue_policy_request() -> bool:
        nonlocal waiting_for_replan_result, replan_request_time
        if inference_busy_event.is_set():
            return False
        try:
            inference_queue.put_nowait(None)
        except queue.Full:
            return False
        waiting_for_replan_result = True
        replan_request_time = time.monotonic()
        return True

    try:
        while True:
            t_start = time.monotonic()
            check_keyboard_input()

            try:
                result_item = result_queue.get_nowait()
                if len(result_item) == 3:
                    _, processed_action, inference_start_time = result_item
                else:
                    processed_action, inference_start_time = result_item
                inference_delay = time.monotonic() - inference_start_time
                waiting_for_replan_result = False
                action_chunk_index = (
                    calculate_latency_compensated_index(
                        inference_delay, config.action_publish_rate, policy.action_chunk_size
                    )
                    if config.latency_compensation
                    else 0
                )
                cached_action_chunk = processed_action
                policy_timing = dict(processed_action.get("policy_timing", {}) or {})
                accepted_steps_raw = policy_timing.get("accepted_prefix_len", None)
                accepted_steps = None
                if accepted_steps_raw is not None:
                    accepted_steps = int(round(float(accepted_steps_raw)))
                if int(config.replan_steps) > 0:
                    current_replan_steps = int(min(int(config.replan_steps), policy.action_chunk_size))
                    if accepted_steps is not None:
                        current_replan_steps = int(min(current_replan_steps, accepted_steps))
                elif accepted_steps is not None:
                    current_replan_steps = int(min(accepted_steps, policy.action_chunk_size))
                else:
                    current_replan_steps = int(policy.action_chunk_size)
                last_inference_time = time.monotonic()
                chunk_start_step = int(published_action_steps)
                query_idx = int(server_query_idx)
                configured_replan_steps = (
                    int(min(int(config.replan_steps), policy.action_chunk_size))
                    if int(config.replan_steps) > 0
                    else int(policy.action_chunk_size)
                )
                route_type = _route_type_from_policy_timing(policy_timing)
                executed = int(current_replan_steps)
                accepted = _int_or_none(accepted_steps_raw)
                used_fallback = _flag_from_timing(policy_timing, "used_full_fallback")
                is_full_round = int(
                    route_type == "full"
                    or used_fallback
                    or _flag_from_timing(policy_timing, "is_full_pipeline_round")
                )
                if accepted is None and is_full_round:
                    accepted = executed
                is_flash_round = int(route_type == "draft" and not is_full_round)
                latency_ms = _first_finite_float(
                    policy_timing.get("sample_actions_ms"),
                    policy_timing.get("total_ms"),
                    policy_timing.get("client_roundtrip_ms"),
                    inference_delay * 1000.0,
                )
                client_roundtrip_ms = _first_finite_float(
                    policy_timing.get("client_roundtrip_ms"),
                    inference_delay * 1000.0,
                )
                route_query_idx = int(timing_route_query_counts.get(route_type, 0))
                route_warmup_queries = 0
                if route_type == "draft":
                    route_warmup_queries = int(timing_warmup_draft_queries)
                elif route_type == "full":
                    route_warmup_queries = int(timing_warmup_full_queries)
                include_in_summary = (
                    int(server_query_idx) >= int(timing_warmup_queries)
                    and int(route_query_idx) >= int(route_warmup_queries)
                )

                if is_full_round and latency_ms is not None and include_in_summary and not full_latency_baseline_fixed:
                    if full_latency_baseline_ms is None:
                        full_latency_baseline_ms = latency_ms
                    else:
                        full_latency_baseline_ms = 0.75 * float(full_latency_baseline_ms) + 0.25 * latency_ms

                speedup = None
                saved_ms = None
                if latency_ms is not None and full_latency_baseline_ms is not None and latency_ms > 0.0:
                    speedup = float(full_latency_baseline_ms) / latency_ms
                    saved_ms = float(full_latency_baseline_ms) - latency_ms

                if include_in_summary:
                    timing_round_count += 1
                    if latency_ms is not None:
                        timing_latency_sum_ms += latency_ms
                    if executed is not None:
                        timing_exec_sum += max(0, int(executed))
                    if accepted is not None:
                        timing_accepted_sum += max(0, int(accepted))
                        timing_replan_sum += max(1, int(configured_replan_steps))
                    if is_flash_round:
                        timing_flash_rounds += 1
                        if latency_ms is not None:
                            timing_flash_latency_sum_ms += latency_ms
                        if executed is not None:
                            timing_flash_exec_sum += max(0, int(executed))
                        if accepted is not None:
                            timing_flash_accepted_sum += max(0, int(accepted))
                            timing_flash_replan_sum += max(1, int(configured_replan_steps))
                    timing_full_rounds += int(is_full_round)
                    timing_fallback_rounds += int(used_fallback)

                avg_latency_ms = (
                    timing_latency_sum_ms / float(timing_round_count)
                    if timing_round_count > 0
                    else None
                )
                avg_ms_per_action = (
                    timing_latency_sum_ms / float(timing_exec_sum)
                    if timing_exec_sum > 0
                    else None
                )
                avg_speedup = (
                    float(full_latency_baseline_ms) / avg_latency_ms
                    if avg_latency_ms is not None and full_latency_baseline_ms is not None and avg_latency_ms > 0.0
                    else None
                )
                flash_avg_latency_ms = (
                    timing_flash_latency_sum_ms / float(timing_flash_rounds)
                    if timing_flash_rounds > 0
                    else None
                )
                flash_avg_ms_per_action = (
                    timing_flash_latency_sum_ms / float(timing_flash_exec_sum)
                    if timing_flash_exec_sum > 0
                    else None
                )
                accept_ratio = (
                    round(float(accepted) / float(max(1, configured_replan_steps)), 3)
                    if accepted is not None
                    else None
                )
                motion_chunk = np.asarray(processed_action.get("motion_token"), dtype=np.float32)
                left_chunk = np.asarray(processed_action.get("left_hand_joints"), dtype=np.float32)
                right_chunk = np.asarray(processed_action.get("right_hand_joints"), dtype=np.float32)
                action_chunk = np.concatenate([motion_chunk, left_chunk, right_chunk], axis=-1)
                processed_action["_sonic_trace"] = {
                    "experiment": experiment_name,
                    "episode_idx": episode_idx,
                    "global_step_idx": chunk_start_step,
                    "server_query_idx": query_idx,
                    "route_type": route_type,
                    "accepted_prefix_len": accepted,
                    "exec_len": executed,
                    "replan_steps": int(current_replan_steps),
                    "configured_replan_steps": int(configured_replan_steps),
                    "policy_timing": policy_timing,
                }
                timing = dict(policy_timing)
                timing.update(
                    {
                        "event": "new_action_chunk",
                        "kind": "policy_query",
                        "experiment": experiment_name,
                        "exp": experiment_name,
                        "episode": episode_idx,
                        "episode_id": episode_idx,
                        "ep": episode_idx,
                        "episode_step": chunk_start_step,
                        "global_step": chunk_start_step,
                        "step": chunk_start_step,
                        "chunk_start_step": chunk_start_step,
                        "query_idx": query_idx,
                        "q": query_idx,
                        "route_q": route_query_idx,
                        "reset": bool(reset_for_next_policy_query),
                        "measured": int(include_in_summary),
                        "wall_time": time.time(),
                        "t": _round_float_or_none(time.time(), 3),
                        "inference_delay_s": float(inference_delay),
                        "prompt": language_prompt_ref[0],
                        "action_chunk_index": int(action_chunk_index),
                        "route": route_type,
                        "route_type": route_type,
                        "exec": executed,
                        "exec_len": executed,
                        "accepted": accepted,
                        "accepted_prefix_len": accepted,
                        "replan": int(configured_replan_steps),
                        "replan_steps": int(current_replan_steps),
                        "configured_replan_steps": int(configured_replan_steps),
                        "accept_ratio": accept_ratio,
                        "flash_accept_ratio": accept_ratio if is_flash_round and accepted is not None else None,
                        "lat_ms": _round_float_or_none(latency_ms),
                        "rt_ms": _round_float_or_none(client_roundtrip_ms),
                        "client_roundtrip_ms": _round_float_or_none(client_roundtrip_ms),
                        "ms_per_action": (
                            round(float(latency_ms) / float(executed), 3)
                            if latency_ms is not None and executed is not None and executed > 0
                            else None
                        ),
                        "full_baseline_ms": _round_float_or_none(full_latency_baseline_ms),
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
                            round(float(timing_accepted_sum) / float(timing_replan_sum), 3)
                            if timing_replan_sum > 0
                            else None
                        ),
                        "cum_flash_accept_ratio": (
                            round(float(timing_flash_accepted_sum) / float(timing_flash_replan_sum), 3)
                            if timing_flash_replan_sum > 0
                            else None
                        ),
                        "full_rate": (
                            round(float(timing_full_rounds) / float(timing_round_count), 3)
                            if timing_round_count > 0
                            else None
                        ),
                        "flash_rate": (
                            round(float(timing_flash_rounds) / float(timing_round_count), 3)
                            if timing_round_count > 0
                            else None
                        ),
                        "fallback_rate": (
                            round(float(timing_fallback_rounds) / float(timing_round_count), 3)
                            if timing_round_count > 0
                            else None
                        ),
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
                        "chunk_len": int(motion_chunk.shape[0]),
                        "policy_timing": policy_timing,
                    }
                )
                timing.update(_action_summary(action_chunk))
                _append_jsonl(timing_log_path, timing)
                if action_log_mode == "chunk":
                    normalized_action_chunk = np.asarray(processed_action.get("normalized_actions"), dtype=np.float32)
                    action_record = {
                        "event": "new_action_chunk",
                        "kind": "action_chunk",
                        "experiment": experiment_name,
                        "exp": experiment_name,
                        "episode": episode_idx,
                        "episode_id": episode_idx,
                        "ep": episode_idx,
                        "episode_step": chunk_start_step,
                        "global_step": chunk_start_step,
                        "step": chunk_start_step,
                        "chunk_start_step": chunk_start_step,
                        "query_idx": query_idx,
                        "q": query_idx,
                        "wall_time": time.time(),
                        "t": _round_float_or_none(time.time(), 3),
                        "prompt": language_prompt_ref[0],
                        "route": route_type,
                        "route_type": route_type,
                        "accepted": accepted,
                        "accepted_prefix_len": accepted,
                        "exec": executed,
                        "exec_len": executed,
                        "replan": int(configured_replan_steps),
                        "replan_steps": int(current_replan_steps),
                        "configured_replan_steps": int(configured_replan_steps),
                        "motion_token": motion_chunk,
                        "left_hand_joints": left_chunk,
                        "right_hand_joints": right_chunk,
                        "normalized_action_shape": list(normalized_action_chunk.shape),
                        "motion_token_shape": list(motion_chunk.shape),
                        "left_hand_joints_shape": list(left_chunk.shape),
                        "right_hand_joints_shape": list(right_chunk.shape),
                        "policy_timing": policy_timing,
                    }
                    action_record.update(_action_summary(action_chunk))
                    _append_jsonl(action_log_path, action_record)
                timing_route_query_counts[route_type] = route_query_idx + 1
                server_query_idx += 1
                reset_for_next_policy_query = False
                print_green(
                    f'New action chunk (prompt: "{language_prompt_ref[0]}", latency: {inference_delay:.3f}s)'
                )
            except queue.Empty:
                pass

            if cached_action_chunk is None:
                should_start = should_trigger_new_inference(
                    cached_chunk_exists=False,
                    inference_thread_running=inference_busy_event.is_set(),
                    time_since_last_inference=(time.monotonic() - last_inference_time),
                    inference_interval=inference_interval,
                )
            elif int(config.replan_steps) > 0:
                should_start = False
            else:
                should_start = should_trigger_new_inference(
                    cached_chunk_exists=True,
                    inference_thread_running=inference_busy_event.is_set(),
                    time_since_last_inference=(time.monotonic() - last_inference_time),
                    inference_interval=inference_interval,
                )

            if should_start:
                _queue_policy_request()

            if pause_loop:
                time.sleep(0.2)
                continue

            with telemetry.timer("total_loop"):
                if cached_action_chunk is None:
                    _sleep_remaining(t_start, loop_period)
                    continue

                motion_token = np.asarray(cached_action_chunk["motion_token"], dtype=np.float32)
                left_hand_joints = np.asarray(cached_action_chunk["left_hand_joints"], dtype=np.float32)
                right_hand_joints = np.asarray(cached_action_chunk["right_hand_joints"], dtype=np.float32)

                horizon = motion_token.shape[0] if motion_token.ndim == 2 else 1
                if action_chunk_index >= int(current_replan_steps):
                    if not waiting_for_replan_result:
                        _queue_policy_request()
                    elif (
                        not inference_busy_event.is_set()
                        and (time.monotonic() - replan_request_time) > max(1.0, float(inference_interval) * 2.0)
                    ):
                        waiting_for_replan_result = False
                        _queue_policy_request()
                    _sleep_remaining(t_start, loop_period)
                    continue

                current_idx = min(action_chunk_index, horizon - 1)

                if motion_token.ndim == 2:
                    motion_token = motion_token[current_idx]
                if left_hand_joints.ndim == 2:
                    left_hand_joints = left_hand_joints[current_idx]
                if right_hand_joints.ndim == 2:
                    right_hand_joints = right_hand_joints[current_idx]

                frame_index = np.array([zmq_frame_counter], dtype=np.int64)
                zmq_frame_counter += 1

                zmq_message = pack_latent_action_message(
                    motion_token,
                    frame_index,
                    left_hand_joints=left_hand_joints,
                    right_hand_joints=right_hand_joints,
                )
                zmq_socket.send(zmq_message)
                if action_log_mode == "executed":
                    trace = cached_action_chunk.get("_sonic_trace", {}) if isinstance(cached_action_chunk, dict) else {}
                    action_vector = np.concatenate(
                        [
                            np.asarray(motion_token, dtype=np.float32).reshape(-1),
                            np.asarray(left_hand_joints, dtype=np.float32).reshape(-1),
                            np.asarray(right_hand_joints, dtype=np.float32).reshape(-1),
                        ],
                        axis=0,
                    )
                    action_record = {
                        "event": "executed_action",
                        "kind": "executed_action",
                        "experiment": trace.get("experiment", experiment_name),
                        "exp": trace.get("experiment", experiment_name),
                        "timestamp_s": time.time(),
                        "wall_time": time.time(),
                        "t": _round_float_or_none(time.time(), 3),
                        "episode": int(trace.get("episode_idx", episode_idx)),
                        "episode_id": int(trace.get("episode_idx", episode_idx)),
                        "episode_idx": int(trace.get("episode_idx", episode_idx)),
                        "ep": int(trace.get("episode_idx", episode_idx)),
                        "episode_step": int(published_action_steps),
                        "global_step": int(published_action_steps),
                        "global_step_idx": int(published_action_steps),
                        "step": int(published_action_steps),
                        "chunk_start_step": int(trace.get("global_step_idx", published_action_steps)),
                        "query_idx": trace.get("server_query_idx"),
                        "q": trace.get("server_query_idx"),
                        "server_query_idx": trace.get("server_query_idx"),
                        "chunk_action_idx": int(current_idx),
                        "queue_repeat_idx": 0,
                        "route": trace.get("route_type"),
                        "route_type": trace.get("route_type"),
                        "accepted": trace.get("accepted_prefix_len"),
                        "accepted_prefix_len": trace.get("accepted_prefix_len"),
                        "exec": trace.get("exec_len"),
                        "exec_len": trace.get("exec_len"),
                        "replan": trace.get("configured_replan_steps"),
                        "replan_steps": trace.get("replan_steps"),
                        "full_exec_steps": trace.get("configured_replan_steps"),
                        "action_dim": int(action_vector.shape[-1]),
                        "action": action_vector,
                        "motion_token_0_64": action_vector[:64],
                        "left_hand_joints_64_71": action_vector[64:71],
                        "right_hand_joints_71_78": action_vector[71:78],
                        "motion_token": motion_token,
                        "left_hand_joints": left_hand_joints,
                        "right_hand_joints": right_hand_joints,
                        "policy_timing": trace.get("policy_timing", {}),
                    }
                    action_record.update(_action_summary(action_vector))
                    _append_jsonl(action_log_path, action_record)
                published_action_steps += 1
                action_chunk_index = min(action_chunk_index + 1, policy.action_chunk_size - 1)
                if action_chunk_index >= int(current_replan_steps):
                    _queue_policy_request()

            if config.verbose_timing and (time.monotonic() - t_start) > 0:
                telemetry.log_timing_info(context="starVLA Inference Loop", threshold=0.0)

            _sleep_remaining(t_start, loop_period)

    except KeyboardInterrupt:
        print("starVLA inference loop terminated by user")
    finally:
        inference_stop_event.set()
        inference_worker_thread.join(timeout=1.0)
        zmq_socket.close()
        zmq_context.term()
        state_subscriber.close()
        try:
            camera_subscriber.close()
        except Exception:
            pass
        keyboard_listener.close()
        policy.client.close()
        print("Shutdown complete.")


if __name__ == "__main__":
    config = tyro.cli(InferenceConfig)
    main(config)
