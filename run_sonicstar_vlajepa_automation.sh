#!/usr/bin/env bash

# Common launch commands
# Default fixed scene:
#   bash /home/d086/fangbaozhong/Sonicstar/SonicStar/run_sonicstar_vlajepa_automation.sh
#
# Random cup position only:
#   bash /home/d086/fangbaozhong/Sonicstar/SonicStar/run_sonicstar_vlajepa_automation.sh --random-cup
#
# Random table color only:
#   bash /home/d086/fangbaozhong/Sonicstar/SonicStar/run_sonicstar_vlajepa_automation.sh --random-table-color
#
# Random cup color + table color, with fixed cup position:
#   bash /home/d086/fangbaozhong/Sonicstar/SonicStar/run_sonicstar_vlajepa_automation.sh --fixed-cup --random-object-colors
#
# Fixed overhead light:
#   bash /home/d086/fangbaozhong/Sonicstar/SonicStar/run_sonicstar_vlajepa_automation.sh --enable-lights
#
# Random overhead light:
#   bash /home/d086/fangbaozhong/Sonicstar/SonicStar/run_sonicstar_vlajepa_automation.sh --random-lights
#
# Random camera view only:
#   bash /home/d086/fangbaozhong/Sonicstar/SonicStar/run_sonicstar_vlajepa_automation.sh --random-camera-view
#
# Light generalization: random cup + random table color:
#   bash /home/d086/fangbaozhong/Sonicstar/SonicStar/run_sonicstar_vlajepa_automation.sh --random-cup --random-table-color
#
# Medium generalization: random cup + random table color + fixed overhead light + random camera:
#   bash /home/d086/fangbaozhong/Sonicstar/SonicStar/run_sonicstar_vlajepa_automation.sh --random-cup --random-table-color --enable-lights --random-camera-view
#
# Strong generalization: random cup + random table color + random light + random camera:
#   bash /home/d086/fangbaozhong/Sonicstar/SonicStar/run_sonicstar_vlajepa_automation.sh --random-cup --random-table-color --random-lights --random-camera-view
#
# Notes:
#   --enable-lights means fixed overhead light.
#   --random-lights means enable overhead light and randomize its position/intensity.
#   --fixed-lights only disables light randomization; by itself it does not turn lights on.
#   --random-table-color randomizes only the table color.
#   --random-object-colors randomizes both cup and table colors.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SONICSTAR_ROOT="${SONICSTAR_ROOT:-$SCRIPT_DIR}"
STARVLA_ROOT="${SONICSTAR_ROOT}/starVLA"
WBC_ROOT="${SONICSTAR_ROOT}/wbc"
STARVLA_CONDA_ENV="${STARVLA_CONDA_ENV:-VLA_JEPA}"
DEPLOY_CONDA_ENV="${DEPLOY_CONDA_ENV:-g1_deploy}"
TENSORRT_ROOT="${TensorRT_ROOT:-${TENSORRT_ROOT:-/usr}}"
CUDA_TOOLKIT_ROOT="${CUDAToolkit_ROOT:-${CUDA_TOOLKIT_ROOT:-/usr/local/cuda}}"

CKPT_PATH="${CKPT_PATH:-/home/d086/fangbaozhong/Sonicstar/CKPTSONICSTAR/SONICSTAR/checkpoints/steps_90000_pytorch_model.pt}"
POLICY_PROMPT="pick up the cylinder and throw it into the trash bin"
SONICSTAR_BACKEND="${SONICSTAR_BACKEND:-pytorch}"
TRITON_MANIFEST="${TRITON_MANIFEST:-}"
DRAFT_CKPT_PATH="${DRAFT_CKPT_PATH:-}"
SONICSTAR_TIMING_LOG="${SONICSTAR_TIMING_LOG:-}"
SONICSTAR_ACTION_LOG="${SONICSTAR_ACTION_LOG:-}"
SONICSTAR_ACTION_LOG_MODE="${SONICSTAR_ACTION_LOG_MODE:-chunk}"
SONICSTAR_EXPERIMENT_NAME="${SONICSTAR_EXPERIMENT_NAME:-}"
SONICSTAR_TIMING_WARMUP_QUERIES="${SONICSTAR_TIMING_WARMUP_QUERIES:-1}"
SONICSTAR_TIMING_WARMUP_DRAFT_QUERIES="${SONICSTAR_TIMING_WARMUP_DRAFT_QUERIES:-2}"
SONICSTAR_TIMING_WARMUP_FULL_QUERIES="${SONICSTAR_TIMING_WARMUP_FULL_QUERIES:-0}"
SONICSTAR_FULL_BASELINE_MS="${SONICSTAR_FULL_BASELINE_MS:-}"
SONICSTAR_REPLAN_STEPS="${SONICSTAR_REPLAN_STEPS:-0}"
POLICY_RATE="${POLICY_RATE:-1.0}"
MAX_EXEC_STEPS="${MAX_EXEC_STEPS:-20}"
TAU_RADIUS="${TAU_RADIUS:-0.25}"
VERIFY_DIST_DIMS="${VERIFY_DIST_DIMS:-64}"
T_LIST="${T_LIST:-0.1 0.05}"
PERIODIC_FULL_EVERY_N_DRAFT_ROUNDS="${PERIODIC_FULL_EVERY_N_DRAFT_ROUNDS:-1}"
SONICSTAR_DRAFT_START_AFTER_FULL_ROUNDS="${SONICSTAR_DRAFT_START_AFTER_FULL_ROUNDS:-0}"
SONICSTAR_DRAFT_MIN_ACCEPT_STEPS="${SONICSTAR_DRAFT_MIN_ACCEPT_STEPS:-1}"
STARVLA_WS_PING_INTERVAL="${STARVLA_WS_PING_INTERVAL:-0}"
STARVLA_WS_PING_TIMEOUT="${STARVLA_WS_PING_TIMEOUT:-0}"
SONICSTAR_AUTO_EVAL="${SONICSTAR_AUTO_EVAL:-0}"
SONICSTAR_AUTO_EVAL_EPISODES="${SONICSTAR_AUTO_EVAL_EPISODES:-10}"
SONICSTAR_AUTO_EVAL_EPISODE_OFFSET="${SONICSTAR_AUTO_EVAL_EPISODE_OFFSET:-0}"
SONICSTAR_AUTO_EVAL_CHILD="${SONICSTAR_AUTO_EVAL_CHILD:-0}"
SONICSTAR_AUTO_EVAL_MAX_STEPS="${SONICSTAR_AUTO_EVAL_MAX_STEPS:-800}"
SONICSTAR_AUTO_EVAL_RESULT_LOG="${SONICSTAR_AUTO_EVAL_RESULT_LOG:-data/result/auto_eval/episode_results.jsonl}"
SONICSTAR_AUTO_EVAL_SAVE_VIDEO="${SONICSTAR_AUTO_EVAL_SAVE_VIDEO:-0}"
SONICSTAR_AUTO_EVAL_VIDEO_DIR="${SONICSTAR_AUTO_EVAL_VIDEO_DIR:-data/result/auto_eval/videos}"
SONICSTAR_AUTO_EVAL_VIDEO_FPS="${SONICSTAR_AUTO_EVAL_VIDEO_FPS:-30}"
SONICSTAR_AUTO_EVAL_SIM_HOST="${SONICSTAR_AUTO_EVAL_SIM_HOST:-127.0.0.1}"
SONICSTAR_AUTO_EVAL_SIM_PORT="${SONICSTAR_AUTO_EVAL_SIM_PORT:-5561}"
SONICSTAR_AUTO_EVAL_CHECK_INTERVAL_STEPS="${SONICSTAR_AUTO_EVAL_CHECK_INTERVAL_STEPS:-5}"
SONICSTAR_AUTO_EVAL_SUCCESS_HOLD_CHECKS="${SONICSTAR_AUTO_EVAL_SUCCESS_HOLD_CHECKS:-1}"
SONICSTAR_AUTO_EVAL_NO_ACTION_TIMEOUT_SEC="${SONICSTAR_AUTO_EVAL_NO_ACTION_TIMEOUT_SEC:-120}"
SONICSTAR_AUTO_EVAL_PRESS_9_ON_START="${SONICSTAR_AUTO_EVAL_PRESS_9_ON_START:-1}"
SONICSTAR_AUTO_EVAL_POST_9_WAIT_SEC="${SONICSTAR_AUTO_EVAL_POST_9_WAIT_SEC:-3}"
SONICSTAR_AUTO_EVAL_BACKSPACE_RESETS="${SONICSTAR_AUTO_EVAL_BACKSPACE_RESETS:-3}"
SONICSTAR_AUTO_EVAL_BACKSPACE_WAIT_SEC="${SONICSTAR_AUTO_EVAL_BACKSPACE_WAIT_SEC:-2}"
SONICSTAR_AUTO_EVAL_MAX_SECONDS="${SONICSTAR_AUTO_EVAL_MAX_SECONDS:-0}"
DEFAULT_POLICY_WARMUP_IMAGE="${SONICSTAR_ROOT}/heatmap/20260707_115629/step_000000_infer_0009_rgb.png"
SONICSTAR_POLICY_WARMUP_RUNS="${SONICSTAR_POLICY_WARMUP_RUNS:-0}"
SONICSTAR_POLICY_WARMUP_TARGET_MS="${SONICSTAR_POLICY_WARMUP_TARGET_MS:-0}"
SONICSTAR_POLICY_WARMUP_MAX_RUNS="${SONICSTAR_POLICY_WARMUP_MAX_RUNS:-0}"
SONICSTAR_POLICY_WARMUP_IMAGE="${SONICSTAR_POLICY_WARMUP_IMAGE:-$DEFAULT_POLICY_WARMUP_IMAGE}"
SONICSTAR_POLICY_WARMUP_PROMPT="${SONICSTAR_POLICY_WARMUP_PROMPT:-$POLICY_PROMPT}"

POLICY_SERVER_SCRIPT="examples/SonicLatent/eval_files/run_policy_server_vlajepa.sh"
INFERENCE_SCRIPT="starVLA/examples/SonicLatent/eval_files/run_starvla_inference.py"
SIM_LOOP_SCRIPT="gear_sonic/scripts/run_sim_loop.py"
CAMERA_VIEWER_SCRIPT="gear_sonic/scripts/run_camera_viewer.py"
SEND_KEY_SCRIPT="gear_sonic/scripts/send_keyboard_cmd.py"
SEND_MUJOCO_BACK_KEY_SCRIPT="gear_sonic/scripts/send_mujoco_back_key.py"
DEPLOY_SCRIPT="${WBC_ROOT}/gear_sonic_deploy/deploy.sh"
SCENE_POSITION_RANDOMIZER_SCRIPT="gear_sonic/scripts/randomize_scene_43dof_cup.py"
SCENE_COLOR_RANDOMIZER_SCRIPT="gear_sonic/scripts/randomize_scene_43dof_colors.py"
SCENE_LIGHTING_SCRIPT="gear_sonic/scripts/configure_scene_43dof_lighting.py"
BASE_SCENE_XML="${WBC_ROOT}/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"

CAMERA_PORT="${CAMERA_PORT:-5555}"
POLICY_PORT="${POLICY_PORT:-10093}"
STEP1_TO_STEP2_DELAY="${STEP1_TO_STEP2_DELAY:-20}"
STEP2_TO_STEP3_DELAY="${STEP2_TO_STEP3_DELAY:-5}"
STEP3_TO_STEP4_DELAY="${STEP3_TO_STEP4_DELAY:-5}"
STEP4_TO_STEP5_DELAY="${STEP4_TO_STEP5_DELAY:-5}"
STEP5_TO_STEP6_DELAY="${STEP5_TO_STEP6_DELAY:-30}"
STEP5_INIT_TIMEOUT="${STEP5_INIT_TIMEOUT:-300}"
STEP2_READY_TIMEOUT="${STEP2_READY_TIMEOUT:-600}"

SIM_PYTHON="${SIM_PYTHON:-${WBC_ROOT}/.venv_sim/bin/python}"
SESSION_ID="sonicstar_vlajepa_automation_$(date +%s)"
STARTED=0
CLEANED=0
PID_DIR="/tmp/${SESSION_ID}"
SESSION_LOG_DIR=""
RANDOMIZED_POSITION_SCENE_XML=""
RANDOMIZED_COLOR_SCENE_XML=""
OVERHEAD_LIGHTING_SCENE_XML=""
STEP3_SCENE_XML=""
AUTO_EPISODE_VIDEO_DIR=""
AUTO_EPISODE_RECORD_START_FILE=""
AUTO_EPISODE_RECORD_STOP_FILE=""
ENABLE_RANDOM_CUP_POSITION="${ENABLE_RANDOM_CUP_POSITION:-0}"
ENABLE_RANDOM_TABLE_COLOR="${ENABLE_RANDOM_TABLE_COLOR:-0}"
ENABLE_RANDOM_OBJECT_COLORS="${ENABLE_RANDOM_OBJECT_COLORS:-0}"
ENABLE_OVERHEAD_LIGHTING="${ENABLE_OVERHEAD_LIGHTING:-0}"
ENABLE_RANDOM_LIGHTING="${ENABLE_RANDOM_LIGHTING:-0}"
ENABLE_RANDOM_CAMERA_VIEW="${ENABLE_RANDOM_CAMERA_VIEW:-0}"
ENABLE_RANDOM_ROBOT_START="${ENABLE_RANDOM_ROBOT_START:-0}"

CAMERA_POS_JITTER_METERS="${CAMERA_POS_JITTER_METERS:-0.01}"
CAMERA_POS_JITTER_X_METERS="${CAMERA_POS_JITTER_X_METERS:-0.01}"
CAMERA_POS_JITTER_Y_METERS="${CAMERA_POS_JITTER_Y_METERS:-0.01}"
CAMERA_POS_JITTER_Z_METERS="${CAMERA_POS_JITTER_Z_METERS:-0.004}"
CAMERA_ROLL_JITTER_RADIANS="${CAMERA_ROLL_JITTER_RADIANS:-0.0}"
CAMERA_PITCH_JITTER_RADIANS="${CAMERA_PITCH_JITTER_RADIANS:-0.10}"
CAMERA_YAW_JITTER_RADIANS="${CAMERA_YAW_JITTER_RADIANS:-0.10}"
ROBOT_START_MIN_METERS="${ROBOT_START_MIN_METERS:-0.02}"
ROBOT_START_MAX_METERS="${ROBOT_START_MAX_METERS:-0.07}"

# Random cup placement bounds for the red-box region on the table.
CUP_X_MIN="${CUP_X_MIN:-0.56}"
CUP_X_MAX="${CUP_X_MAX:-0.74}"
CUP_Y_MIN="${CUP_Y_MIN:--0.22}"
CUP_Y_MAX="${CUP_Y_MAX:-0.28}"

find_conda_sh() {
  local candidates=(
    "/home/d013/anaconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "/opt/conda/etc/profile.d/conda.sh"
  )
  local path
  for path in "${candidates[@]}"; do
    if [[ -f "$path" ]]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  return 1
}

CONDA_SH="$(find_conda_sh || true)"

require_file() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    printf 'Missing required path: %s\n' "$path" >&2
    exit 1
  fi
}

require_command() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "$cmd" >&2
    exit 1
  fi
}

print_usage() {
  cat <<EOF
Usage: $(basename "$0") [--backend pytorch|triton] [--triton-manifest PATH] [--draft-checkpoint PATH] [--timing-log PATH] [--action-log PATH] [--automation-log-dir PATH] [--auto-eval] [--num-episodes N] [--max-episode-steps N] [--eval-result-log PATH] [--save-video] [--video-dir PATH] [--random-cup] [--fixed-cup] [--random-table-color] [--random-object-colors] [--fixed-table-color] [--enable-lights] [--disable-lights] [--random-lights] [--fixed-lights] [--random-camera-view] [--fixed-camera-view] [--random-robot-start] [--fixed-robot-start] [--help]

Options:
  --backend MODE         Policy server backend: pytorch or triton.
  --triton-manifest PATH Use this SonicStar Triton runtime manifest.
  --draft-checkpoint PATH Enable Sonic draft acceleration with this trained draft checkpoint.
  --timing-log PATH      Append per-policy-call timing JSONL to PATH.
  --action-log PATH      Append predicted action chunks JSONL to PATH.
  --automation-log-dir PATH
                        Write Step1-Step5 terminal logs under PATH.
  --auto-eval            Repeat the manual Step1-Step6 flow automatically; each episode starts a fresh stack.
  --num-episodes N       Number of auto-eval episodes (default: 10).
  --max-episode-steps N  Mark an episode failed after N published action steps (default: 800).
  --eval-result-log PATH Append per-episode success/failure JSONL to PATH.
  --save-video           Save one MP4 per auto-eval episode.
  --no-save-video        Disable auto-eval videos.
  --video-dir PATH       Directory for auto-eval episode MP4s.
  --random-cup          Randomize the cup position within the configured table region.
  --fixed-cup           Use the original fixed cup position from scene_43dof.xml.
  --random-table-color  Randomize only the table color; keep the cup color unchanged.
  --random-object-colors Randomize both the cup color and the table color.
  --fixed-table-color   Use the original table color from scene_43dof.xml.
  --enable-lights       Enable a vertical overhead light directly above the table.
  --disable-lights      Use the original scene lighting from scene_43dof.xml.
  --random-lights       Enable lights and randomize overhead position/intensity.
  --fixed-lights        Use a fixed overhead vertical light above the table center.
  --random-camera-view  Randomize only the MuJoCo head camera pose slightly.
  --fixed-camera-view   Use the original MuJoCo head camera pose.
  --random-robot-start  Randomize robot initial position forward/back by 2-7cm on reset.
  --fixed-robot-start   Use the original robot initial position.
  --help                Show this help message.

Environment:
  ENABLE_RANDOM_CUP_POSITION=1   Same as --random-cup
  ENABLE_RANDOM_TABLE_COLOR=1    Same as --random-table-color
  ENABLE_RANDOM_OBJECT_COLORS=1  Same as --random-object-colors
  ENABLE_OVERHEAD_LIGHTING=1     Same as --enable-lights
  ENABLE_RANDOM_LIGHTING=1       Same as --random-lights
  ENABLE_RANDOM_CAMERA_VIEW=1    Same as --random-camera-view
  ENABLE_RANDOM_ROBOT_START=1    Same as --random-robot-start
  ROBOT_START_MIN_METERS=0.02    Minimum forward/back random offset
  ROBOT_START_MAX_METERS=0.07    Maximum forward/back random offset
  SONICSTAR_BACKEND=triton        Same as --backend triton
  TRITON_MANIFEST=...             Same as --triton-manifest
  DRAFT_CKPT_PATH=...             Same as --draft-checkpoint
  SONICSTAR_TIMING_LOG=...        Same as --timing-log
  SONICSTAR_ACTION_LOG=...        Same as --action-log
  SONICSTAR_ACTION_LOG_MODE=chunk  action log mode: chunk or executed.
  SONICSTAR_EXPERIMENT_NAME=...   exp field for timing/actions logs.
  SONICSTAR_TIMING_WARMUP_QUERIES=1 Skip first N policy queries from rolling averages.
  SONICSTAR_TIMING_WARMUP_DRAFT_QUERIES=2 Skip first N draft-route queries from rolling averages.
  SONICSTAR_TIMING_WARMUP_FULL_QUERIES=0 Skip first N full-route queries from rolling averages.
  SONICSTAR_FULL_BASELINE_MS=...   Fixed full latency baseline for speedup fields.
  SONICSTAR_AUTOMATION_LOG_DIR=... Same as --automation-log-dir
  SONICSTAR_REPLAN_STEPS=20       Trigger a new policy request after 20 published steps.
  SONICSTAR_AUTO_EVAL=1           Same as --auto-eval.
  SONICSTAR_AUTO_EVAL_EPISODES=10 Same as --num-episodes.
  SONICSTAR_AUTO_EVAL_MAX_STEPS=800 Same as --max-episode-steps.
  SONICSTAR_AUTO_EVAL_RESULT_LOG=... Episode result JSONL.
  SONICSTAR_AUTO_EVAL_SAVE_VIDEO=1 Same as --save-video.
  SONICSTAR_AUTO_EVAL_VIDEO_DIR=... Episode MP4 output directory.
  SONICSTAR_AUTO_EVAL_POST_9_WAIT_SEC=3 Seconds to wait after sending MuJoCo key 9.
  SONICSTAR_AUTO_EVAL_BACKSPACE_RESETS=3 Number of MuJoCo BackSpace resets after key 9.
  SONICSTAR_AUTO_EVAL_BACKSPACE_WAIT_SEC=2 Seconds between BackSpace resets.
  SONICSTAR_AUTO_EVAL_MAX_SECONDS=0 Optional wall-time episode timeout; 0 derives from max steps.
  POLICY_RATE=1.0                 Time-based policy request rate when SONICSTAR_REPLAN_STEPS=0.
  JEPA_COMPILE_ACTION_VERIFY=0     Default for SonicStar; set 1 to try compiled verify.
  SONICSTAR_DRAFT_START_AFTER_FULL_ROUNDS=0 Force this many real full rounds before allowing draft.
  SONICSTAR_DRAFT_MIN_ACCEPT_STEPS=1 Reject draft chunks accepted for fewer than this many steps.
  SONICSTAR_POLICY_WARMUP_RUNS=auto Warmup policy before serving; auto means 2 full-only or 4 draft for triton.
  SONICSTAR_POLICY_WARMUP_TARGET_MS=1000 Continue auto warmup until a run is below this latency.
  SONICSTAR_POLICY_WARMUP_MAX_RUNS=10 Maximum auto warmup runs.
  SONICSTAR_POLICY_WARMUP_IMAGE=... Image used for server-side compile warmup.
  STARVLA_WS_PING_INTERVAL=0       Default disables websocket heartbeat during first torch.compile.
  STARVLA_WS_PING_TIMEOUT=0        Default disables websocket heartbeat timeout during first torch.compile.
EOF
}

parse_cli_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --backend)
        if [[ $# -lt 2 ]]; then
          printf 'Missing value for --backend\n' >&2
          exit 1
        fi
        case "$2" in
          pytorch|triton)
            SONICSTAR_BACKEND="$2"
            ;;
          *)
            printf 'Invalid --backend: %s (expected pytorch or triton)\n' "$2" >&2
            exit 1
            ;;
        esac
        shift 2
        ;;
      --triton-manifest)
        if [[ $# -lt 2 ]]; then
          printf 'Missing value for --triton-manifest\n' >&2
          exit 1
        fi
        TRITON_MANIFEST="$2"
        shift 2
        ;;
      --draft-checkpoint)
        if [[ $# -lt 2 ]]; then
          printf 'Missing value for --draft-checkpoint\n' >&2
          exit 1
        fi
        DRAFT_CKPT_PATH="$2"
        shift 2
        ;;
      --timing-log)
        if [[ $# -lt 2 ]]; then
          printf 'Missing value for --timing-log\n' >&2
          exit 1
        fi
        SONICSTAR_TIMING_LOG="$2"
        shift 2
        ;;
      --action-log)
        if [[ $# -lt 2 ]]; then
          printf 'Missing value for --action-log\n' >&2
          exit 1
        fi
        SONICSTAR_ACTION_LOG="$2"
        shift 2
        ;;
      --automation-log-dir)
        if [[ $# -lt 2 ]]; then
          printf 'Missing value for --automation-log-dir\n' >&2
          exit 1
        fi
        SONICSTAR_AUTOMATION_LOG_DIR="$2"
        shift 2
        ;;
      --auto-eval)
        SONICSTAR_AUTO_EVAL="1"
        shift
        ;;
      --manual-eval|--no-auto-eval)
        SONICSTAR_AUTO_EVAL="0"
        shift
        ;;
      --num-episodes)
        if [[ $# -lt 2 ]]; then
          printf 'Missing value for --num-episodes\n' >&2
          exit 1
        fi
        SONICSTAR_AUTO_EVAL_EPISODES="$2"
        shift 2
        ;;
      --max-episode-steps)
        if [[ $# -lt 2 ]]; then
          printf 'Missing value for --max-episode-steps\n' >&2
          exit 1
        fi
        SONICSTAR_AUTO_EVAL_MAX_STEPS="$2"
        shift 2
        ;;
      --eval-result-log)
        if [[ $# -lt 2 ]]; then
          printf 'Missing value for --eval-result-log\n' >&2
          exit 1
        fi
        SONICSTAR_AUTO_EVAL_RESULT_LOG="$2"
        shift 2
        ;;
      --save-video)
        SONICSTAR_AUTO_EVAL_SAVE_VIDEO="1"
        shift
        ;;
      --no-save-video)
        SONICSTAR_AUTO_EVAL_SAVE_VIDEO="0"
        shift
        ;;
      --video-dir)
        if [[ $# -lt 2 ]]; then
          printf 'Missing value for --video-dir\n' >&2
          exit 1
        fi
        SONICSTAR_AUTO_EVAL_VIDEO_DIR="$2"
        shift 2
        ;;
      --random-cup)
        ENABLE_RANDOM_CUP_POSITION="1"
        shift
        ;;
      --fixed-cup)
        ENABLE_RANDOM_CUP_POSITION="0"
        shift
        ;;
      --random-table-color)
        ENABLE_RANDOM_TABLE_COLOR="1"
        ENABLE_RANDOM_OBJECT_COLORS="0"
        shift
        ;;
      --random-object-colors)
        ENABLE_RANDOM_TABLE_COLOR="1"
        ENABLE_RANDOM_OBJECT_COLORS="1"
        shift
        ;;
      --fixed-table-color)
        ENABLE_RANDOM_TABLE_COLOR="0"
        ENABLE_RANDOM_OBJECT_COLORS="0"
        shift
        ;;
      --enable-lights)
        ENABLE_OVERHEAD_LIGHTING="1"
        shift
        ;;
      --disable-lights)
        ENABLE_OVERHEAD_LIGHTING="0"
        shift
        ;;
      --random-lights)
        ENABLE_OVERHEAD_LIGHTING="1"
        ENABLE_RANDOM_LIGHTING="1"
        shift
        ;;
      --fixed-lights)
        ENABLE_RANDOM_LIGHTING="0"
        shift
        ;;
      --random-camera-view)
        ENABLE_RANDOM_CAMERA_VIEW="1"
        shift
        ;;
      --fixed-camera-view)
        ENABLE_RANDOM_CAMERA_VIEW="0"
        shift
        ;;
      --random-robot-start)
        ENABLE_RANDOM_ROBOT_START="1"
        shift
        ;;
      --fixed-robot-start)
        ENABLE_RANDOM_ROBOT_START="0"
        shift
        ;;
      --help|-h)
        print_usage
        exit 0
        ;;
      *)
        printf 'Unknown argument: %s\n\n' "$1" >&2
        print_usage >&2
        exit 1
        ;;
    esac
  done
}

create_randomized_position_scene() {
  local input_scene="$1"
  RANDOMIZED_POSITION_SCENE_XML="$(mktemp -p "$(dirname "$BASE_SCENE_XML")" scene_43dof_position_XXXX.xml)"
  "$SIM_PYTHON" "${WBC_ROOT}/${SCENE_POSITION_RANDOMIZER_SCRIPT}" \
    --input-scene "$input_scene" \
    --output-scene "$RANDOMIZED_POSITION_SCENE_XML" \
    --x-min "$CUP_X_MIN" \
    --x-max "$CUP_X_MAX" \
    --y-min "$CUP_Y_MIN" \
    --y-max "$CUP_Y_MAX"
  printf '[Step 3 Prep] Randomized position scene created: %s\n' "$RANDOMIZED_POSITION_SCENE_XML"
}

create_randomized_color_scene() {
  local input_scene="$1"
  local color_args=()
  RANDOMIZED_COLOR_SCENE_XML="$(mktemp -p "$(dirname "$BASE_SCENE_XML")" scene_43dof_colors_XXXX.xml)"
  if [[ "$ENABLE_RANDOM_OBJECT_COLORS" != "1" ]]; then
    color_args+=(--table-only)
  fi
  "$SIM_PYTHON" "${WBC_ROOT}/${SCENE_COLOR_RANDOMIZER_SCRIPT}" \
    --input-scene "$input_scene" \
    --output-scene "$RANDOMIZED_COLOR_SCENE_XML" \
    "${color_args[@]}"
  if [[ "$ENABLE_RANDOM_OBJECT_COLORS" == "1" ]]; then
    printf '[Step 3 Prep] Randomized cup+table color scene created: %s\n' "$RANDOMIZED_COLOR_SCENE_XML"
  else
    printf '[Step 3 Prep] Randomized table color scene created: %s\n' "$RANDOMIZED_COLOR_SCENE_XML"
  fi
}

create_overhead_lighting_scene() {
  local input_scene="$1"
  local randomize_args=()
  OVERHEAD_LIGHTING_SCENE_XML="$(mktemp -p "$(dirname "$BASE_SCENE_XML")" scene_43dof_lighting_XXXX.xml)"
  if [[ "$ENABLE_RANDOM_LIGHTING" == "1" ]]; then
    randomize_args+=(--randomize)
  fi
  "$SIM_PYTHON" "${WBC_ROOT}/${SCENE_LIGHTING_SCRIPT}" \
    --input-scene "$input_scene" \
    --output-scene "$OVERHEAD_LIGHTING_SCENE_XML" \
    "${randomize_args[@]}"
  printf '[Step 3 Prep] Overhead lighting scene created: %s\n' "$OVERHEAD_LIGHTING_SCENE_XML"
}

resolve_step3_scene() {
  STEP3_SCENE_XML="$BASE_SCENE_XML"

  if [[ "$ENABLE_RANDOM_CUP_POSITION" == "1" ]]; then
    create_randomized_position_scene "$STEP3_SCENE_XML"
    STEP3_SCENE_XML="$RANDOMIZED_POSITION_SCENE_XML"
    printf '[Step 3 Prep] Cup position mode: randomized\n'
  else
    printf '[Step 3 Prep] Cup position mode: fixed original scene\n'
  fi

  if [[ "$ENABLE_RANDOM_TABLE_COLOR" == "1" ]]; then
    create_randomized_color_scene "$STEP3_SCENE_XML"
    STEP3_SCENE_XML="$RANDOMIZED_COLOR_SCENE_XML"
    if [[ "$ENABLE_RANDOM_OBJECT_COLORS" == "1" ]]; then
      printf '[Step 3 Prep] Object color mode: randomized (cup + table colors)\n'
    else
      printf '[Step 3 Prep] Table color mode: randomized (cup color unchanged)\n'
    fi
  else
    printf '[Step 3 Prep] Table color mode: fixed original scene\n'
  fi

  if [[ "$ENABLE_OVERHEAD_LIGHTING" == "1" ]]; then
    create_overhead_lighting_scene "$STEP3_SCENE_XML"
    STEP3_SCENE_XML="$OVERHEAD_LIGHTING_SCENE_XML"
    if [[ "$ENABLE_RANDOM_LIGHTING" == "1" ]]; then
      printf '[Step 3 Prep] Lighting mode: overhead vertical randomized\n'
    else
      printf '[Step 3 Prep] Lighting mode: overhead vertical fixed\n'
    fi
  else
    printf '[Step 3 Prep] Lighting mode: disabled (original scene)\n'
  fi

  printf '[Step 3 Prep] Camera view mode: %s\n' "$([[ "$ENABLE_RANDOM_CAMERA_VIEW" == "1" ]] && printf 'runtime randomized on each reset (MuJoCo head camera only)' || printf 'fixed original scene')"
}

kill_port_processes() {
  local port="$1"
  local pids

  pids="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    printf '[Preflight] Releasing port %s: %s\n' "$port" "$pids"
    kill $pids >/dev/null 2>&1 || true
    sleep 1
    pids="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
    if [[ -n "$pids" ]]; then
      kill -9 $pids >/dev/null 2>&1 || true
    fi
  fi
}

preflight_cleanup() {
  printf '[Preflight] Cleaning up old SONICSTAR VLA-JEPA processes...\n'
  pkill -f "examples/SonicLatent/eval_files/run_policy_server.sh" >/dev/null 2>&1 || true
  pkill -f "examples/SonicLatent/eval_files/run_policy_server_vlajepa.sh" >/dev/null 2>&1 || true
  pkill -f "examples/SonicLatent/eval_files/run_starvla_inference.py" >/dev/null 2>&1 || true
  pkill -f "gear_sonic/scripts/run_sim_loop.py --enable-image-publish --enable-offscreen --camera-port ${CAMERA_PORT}" >/dev/null 2>&1 || true
  pkill -f "gear_sonic/scripts/run_camera_viewer.py --camera-host localhost --camera-port ${CAMERA_PORT}" >/dev/null 2>&1 || true
  pkill -f "bash deploy.sh --input-type zmq_manager sim" >/dev/null 2>&1 || true
  pkill -f "g1_deploy_onnx_ref" >/dev/null 2>&1 || true

  sleep 1

  pkill -9 -f "examples/SonicLatent/eval_files/run_policy_server.sh" >/dev/null 2>&1 || true
  pkill -9 -f "examples/SonicLatent/eval_files/run_policy_server_vlajepa.sh" >/dev/null 2>&1 || true
  pkill -9 -f "examples/SonicLatent/eval_files/run_starvla_inference.py" >/dev/null 2>&1 || true
  pkill -9 -f "gear_sonic/scripts/run_sim_loop.py --enable-image-publish --enable-offscreen --camera-port ${CAMERA_PORT}" >/dev/null 2>&1 || true
  pkill -9 -f "gear_sonic/scripts/run_camera_viewer.py --camera-host localhost --camera-port ${CAMERA_PORT}" >/dev/null 2>&1 || true
  pkill -9 -f "bash deploy.sh --input-type zmq_manager sim" >/dev/null 2>&1 || true
  pkill -9 -f "g1_deploy_onnx_ref" >/dev/null 2>&1 || true

  kill_port_processes "$POLICY_PORT"
  kill_port_processes "$CAMERA_PORT"

  rm -rf /tmp/sonicstar_vla_automation_* >/dev/null 2>&1 || true
  rm -rf /tmp/sonicstar_vlajepa_automation_* >/dev/null 2>&1 || true
  printf '[Preflight] Cleanup complete.\n'
}

cleanup() {
  if [[ "$CLEANED" -eq 1 ]]; then
    return
  fi
  CLEANED=1

  printf '\n[Cleanup] Stopping launched processes...\n'
  if [[ -d "$PID_DIR" ]]; then
    local pid_file
    local pid
    for pid_file in "$PID_DIR"/*.pid; do
      if [[ -f "$pid_file" ]]; then
        pid="$(cat "$pid_file" 2>/dev/null || true)"
        if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
          kill "$pid" >/dev/null 2>&1 || true
        fi
      fi
    done
    sleep 2
    for pid_file in "$PID_DIR"/*.pid; do
      if [[ -f "$pid_file" ]]; then
        pid="$(cat "$pid_file" 2>/dev/null || true)"
        if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
          kill -9 "$pid" >/dev/null 2>&1 || true
        fi
      fi
    done
  fi

  pkill -f "examples/SonicLatent/eval_files/run_policy_server.sh" >/dev/null 2>&1 || true
  pkill -f "examples/SonicLatent/eval_files/run_policy_server_vlajepa.sh" >/dev/null 2>&1 || true
  pkill -f "examples/SonicLatent/eval_files/run_starvla_inference.py" >/dev/null 2>&1 || true
  pkill -f "gear_sonic/scripts/run_sim_loop.py --enable-image-publish --enable-offscreen --camera-port ${CAMERA_PORT}" >/dev/null 2>&1 || true
  pkill -f "gear_sonic/scripts/run_camera_viewer.py --camera-host localhost --camera-port ${CAMERA_PORT}" >/dev/null 2>&1 || true
  pkill -f "bash deploy.sh --input-type zmq_manager sim" >/dev/null 2>&1 || true
  pkill -f "g1_deploy_onnx_ref" >/dev/null 2>&1 || true
  sleep 1
  pkill -9 -f "examples/SonicLatent/eval_files/run_policy_server.sh" >/dev/null 2>&1 || true
  pkill -9 -f "examples/SonicLatent/eval_files/run_policy_server_vlajepa.sh" >/dev/null 2>&1 || true
  pkill -9 -f "examples/SonicLatent/eval_files/run_starvla_inference.py" >/dev/null 2>&1 || true
  pkill -9 -f "gear_sonic/scripts/run_sim_loop.py --enable-image-publish --enable-offscreen --camera-port ${CAMERA_PORT}" >/dev/null 2>&1 || true
  pkill -9 -f "gear_sonic/scripts/run_camera_viewer.py --camera-host localhost --camera-port ${CAMERA_PORT}" >/dev/null 2>&1 || true
  pkill -9 -f "bash deploy.sh --input-type zmq_manager sim" >/dev/null 2>&1 || true
  pkill -9 -f "g1_deploy_onnx_ref" >/dev/null 2>&1 || true

  if [[ -d "$PID_DIR" ]]; then
    rm -rf "$PID_DIR"
  fi
  if [[ -n "$RANDOMIZED_POSITION_SCENE_XML" ]] && [[ -f "$RANDOMIZED_POSITION_SCENE_XML" ]]; then
    rm -f "$RANDOMIZED_POSITION_SCENE_XML"
  fi
  if [[ -n "$RANDOMIZED_COLOR_SCENE_XML" ]] && [[ -f "$RANDOMIZED_COLOR_SCENE_XML" ]]; then
    rm -f "$RANDOMIZED_COLOR_SCENE_XML"
  fi
  if [[ -n "$OVERHEAD_LIGHTING_SCENE_XML" ]] && [[ -f "$OVERHEAD_LIGHTING_SCENE_XML" ]]; then
    rm -f "$OVERHEAD_LIGHTING_SCENE_XML"
  fi
  printf '[Cleanup] Done.\n'
}

on_interrupt() {
  printf '\n[Main] Ctrl+C detected.\n'
  cleanup
  exit 0
}

trap on_interrupt INT TERM

launch_terminal() {
  local title="$1"
  local command="$2"
  local pid_file="$3"
  local hold_on_exit="1"
  local log_name
  local log_file
  local wrapped_command
  if [[ "$SONICSTAR_AUTO_EVAL" == "1" ]]; then
    hold_on_exit="0"
  fi
  log_name="$(printf '%s' "$title" | tr ' /' '__' | tr -cd '[:alnum:]_.-')"
  log_file="${SESSION_LOG_DIR}/${log_name}.log"

  wrapped_command=$(
    cat <<EOF
set -uo pipefail
trap 'exit 0' TERM INT
echo \$\$ > "$pid_file"
mkdir -p "$SESSION_LOG_DIR"
exec > >(tee -a "$log_file") 2>&1
echo "[${title}] Log: $log_file"
if $command; then
  status=0
else
  status=\$?
fi
echo
echo "[${title}] Process exited with code: \$status"
if [[ "$hold_on_exit" == "1" ]]; then
  read -r -p "Press Enter to close this terminal..." _
fi
EOF
  )

  gnome-terminal \
    --title="$title" \
    -- bash -lc "$wrapped_command" >/dev/null 2>&1 &
}

step_log_file() {
  local title="$1"
  local log_name
  log_name="$(printf '%s' "$title" | tr ' /' '__' | tr -cd '[:alnum:]_.-')"
  printf '%s/%s.log\n' "$SESSION_LOG_DIR" "$log_name"
}

wait_for_log_pattern() {
  local log_file="$1"
  local pattern="$2"
  local timeout_sec="$3"
  local waited=0
  while [[ "$waited" -lt "$timeout_sec" ]]; do
    if [[ -f "$log_file" ]] && grep -q "$pattern" "$log_file"; then
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  return 1
}

wait_for_step2_ready() {
  local step2_log
  step2_log="$(step_log_file "SONICSTAR VLA-JEPA Step2 Online Inference")"
  printf '[Wait] Waiting up to %ss for Step2 Online Inference to bind action ZMQ...\n' "$STEP2_READY_TIMEOUT"
  if ! wait_for_log_pattern "$step2_log" "ZMQ action socket bound" "$STEP2_READY_TIMEOUT"; then
    printf '[Wait] Step2 did not become ready. See log: %s\n' "$step2_log" >&2
    return 1
  fi
  printf '[Wait] Step2 Online Inference is ready.\n'
}

send_key() {
  local key="$1"
  printf '[Step 6] Sending key: %s\n' "$key"
  "$SIM_PYTHON" "${WBC_ROOT}/${SEND_KEY_SCRIPT}" "$key"
}

send_window_key() {
  local title_contains="$1"
  local keysym="$2"
  local label="$3"
  printf '[Step 6] Sending %s to window matching "%s"...\n' "$label" "$title_contains"
  "$SIM_PYTHON" "${WBC_ROOT}/${SEND_MUJOCO_BACK_KEY_SCRIPT}" --title-contains "$title_contains" --keysym "$keysym"
}

send_mujoco_9_key() {
  send_window_key "mujoco" "9" "MuJoCo key 9"
}

send_mujoco_back_key() {
  send_window_key "mujoco" "BackSpace" "MuJoCo BackSpace"
}

sonic_abs_path() {
  local raw="$1"
  if [[ "$raw" = /* ]]; then
    printf '%s\n' "$raw"
  else
    printf '%s/%s\n' "$SONICSTAR_ROOT" "$raw"
  fi
}

refresh_session_log_dir() {
  if [[ -n "${SONICSTAR_AUTOMATION_LOG_DIR:-}" ]]; then
    SESSION_LOG_DIR="$(sonic_abs_path "$SONICSTAR_AUTOMATION_LOG_DIR")"
  else
    SESSION_LOG_DIR="${SONICSTAR_ROOT}/data/result/automation_logs/${SESSION_ID}"
  fi
}

refresh_experiment_name() {
  if [[ -n "$SONICSTAR_EXPERIMENT_NAME" ]]; then
    return
  fi
  if [[ "$SONICSTAR_BACKEND" == "triton" ]] && [[ -n "$DRAFT_CKPT_PATH" ]]; then
    SONICSTAR_EXPERIMENT_NAME="sonicstar_flash_triton"
  elif [[ "$SONICSTAR_BACKEND" == "triton" ]]; then
    SONICSTAR_EXPERIMENT_NAME="sonicstar_full_triton"
  else
    SONICSTAR_EXPERIMENT_NAME="sonicstar_baseline_pytorch"
  fi
}

toggle_camera_recording() {
  send_window_key "SONIC Camera Viewer" "r" "Camera Viewer recording toggle R"
}

query_eval_status() {
  "$SIM_PYTHON" -c '
import json
import sys
import zmq

host = sys.argv[1]
port = int(sys.argv[2])
ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.setsockopt(zmq.RCVTIMEO, 2000)
sock.setsockopt(zmq.SNDTIMEO, 2000)
sock.setsockopt(zmq.LINGER, 0)
try:
    sock.connect(f"tcp://{host}:{port}")
    sock.send_json({"command": "status"})
    print(sock.recv_string())
finally:
    sock.close()
    ctx.term()
' "$SONICSTAR_AUTO_EVAL_SIM_HOST" "$SONICSTAR_AUTO_EVAL_SIM_PORT"
}

json_field_bool() {
  "$SIM_PYTHON" -c 'import json, sys; print("1" if json.loads(sys.argv[1]).get(sys.argv[2], False) else "0")' "$1" "$2"
}

auto_episode_timeout_sec() {
  if [[ "$SONICSTAR_AUTO_EVAL_MAX_SECONDS" != "0" ]]; then
    printf '%s\n' "$SONICSTAR_AUTO_EVAL_MAX_SECONDS"
  else
    "$SIM_PYTHON" -c 'import math, sys; print(int(math.ceil(int(sys.argv[1]) / 50.0 + 10.0)))' "$SONICSTAR_AUTO_EVAL_MAX_STEPS"
  fi
}

append_episode_result() {
  local episode="$1"
  local success="$2"
  local reason="$3"
  local elapsed_sec="$4"
  local status_json="$5"
  local video_dir="$6"
  "$SIM_PYTHON" -c '
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
try:
    status = json.loads(sys.argv[6]) if sys.argv[6] else None
except Exception:
    status = {"raw": sys.argv[6]}
row = {
    "event": "episode_result",
    "episode": int(sys.argv[2]),
    "success": sys.argv[3] == "1",
    "reason": sys.argv[4],
    "elapsed_s": float(sys.argv[5]),
    "final_status": status,
    "video_dir": sys.argv[7] or None,
}
with path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, sort_keys=True) + "\n")
' "$SONICSTAR_AUTO_EVAL_RESULT_LOG" "$episode" "$success" "$reason" "$elapsed_sec" "$status_json" "$video_dir"
}

run_manual_step6_sequence() {
  local episode="$1"
  AUTO_EPISODE_VIDEO_DIR=""
  AUTO_EPISODE_RECORD_START_FILE=""
  AUTO_EPISODE_RECORD_STOP_FILE=""

  printf '\n[Auto Eval] Running manual Step6 sequence for episode %s.\n' "$episode"
  printf '[Step 6] Sending k...\n'
  send_key "k"

  printf '\n[Step 6] Sending MuJoCo 9...\n'
  if [[ "$SONICSTAR_AUTO_EVAL_PRESS_9_ON_START" == "1" ]]; then
    send_mujoco_9_key
  fi
  sleep "$SONICSTAR_AUTO_EVAL_POST_9_WAIT_SEC"

  local reset_i
  for ((reset_i = 1; reset_i <= SONICSTAR_AUTO_EVAL_BACKSPACE_RESETS; reset_i++)); do
    printf '\n[Step 6] Sending MuJoCo BackSpace reset %s/%s...\n' "$reset_i" "$SONICSTAR_AUTO_EVAL_BACKSPACE_RESETS"
    send_mujoco_back_key
    sleep "$SONICSTAR_AUTO_EVAL_BACKSPACE_WAIT_SEC"
  done

  if [[ "$SONICSTAR_AUTO_EVAL_SAVE_VIDEO" == "1" ]]; then
    local auto_video_root
    auto_video_root="$(sonic_abs_path "$SONICSTAR_AUTO_EVAL_VIDEO_DIR")"
    printf -v AUTO_EPISODE_VIDEO_DIR "%s/episode_%03d" "$auto_video_root" "$episode"
    AUTO_EPISODE_RECORD_START_FILE="$AUTO_EPISODE_VIDEO_DIR/.record_start"
    AUTO_EPISODE_RECORD_STOP_FILE="$AUTO_EPISODE_VIDEO_DIR/.record_stop"
    mkdir -p "$AUTO_EPISODE_VIDEO_DIR"
    rm -f "$AUTO_EPISODE_RECORD_START_FILE" "$AUTO_EPISODE_RECORD_STOP_FILE"
  fi

  printf '\n[Step 6] Sending i, i, p...\n'
  send_key "i"
  sleep 1
  send_key "i"
  sleep 1
  send_key "p"
  if [[ "$SONICSTAR_AUTO_EVAL_SAVE_VIDEO" == "1" ]]; then
    printf '\n[Step 6] Starting Camera Viewer recording after p: %s\n' "$AUTO_EPISODE_VIDEO_DIR"
    : > "$AUTO_EPISODE_RECORD_START_FILE"
    sleep 0.5
  fi
}

monitor_auto_episode() {
  local episode="$1"
  local video_episode_dir="$2"
  local timeout_sec
  local start_ts
  local now_ts
  local elapsed
  local status_json=""
  local success="0"
  local reason="max_time"
  local hold_count=0

  timeout_sec="$(auto_episode_timeout_sec)"
  start_ts="$(date +%s)"
  printf '[Auto Eval] Monitoring episode %s for success, timeout=%ss.\n' "$episode" "$timeout_sec"

  while true; do
    status_json="$(query_eval_status 2>/dev/null || true)"
    if [[ -n "$status_json" ]] && [[ "$(json_field_bool "$status_json" "success" 2>/dev/null || printf '0')" == "1" ]]; then
      hold_count=$((hold_count + 1))
    else
      hold_count=0
    fi
    if [[ "$hold_count" -ge "$SONICSTAR_AUTO_EVAL_SUCCESS_HOLD_CHECKS" ]]; then
      success="1"
      reason="success"
      break
    fi
    now_ts="$(date +%s)"
    elapsed=$((now_ts - start_ts))
    if [[ "$elapsed" -ge "$timeout_sec" ]]; then
      success="0"
      reason="timeout"
      break
    fi
    sleep 1
  done

  if [[ "$SONICSTAR_AUTO_EVAL_SAVE_VIDEO" == "1" ]]; then
    printf '[Auto Eval] Stopping Camera Viewer recording.\n'
    if [[ -n "$video_episode_dir" ]]; then
      : > "$video_episode_dir/.record_stop"
    else
      toggle_camera_recording || true
    fi
    sleep 1
  fi

  now_ts="$(date +%s)"
  elapsed=$((now_ts - start_ts))
  append_episode_result "$episode" "$success" "$reason" "$elapsed" "$status_json" "$video_episode_dir"
  printf '[Auto Eval] Episode %s result: success=%s reason=%s elapsed=%ss\n' "$episode" "$success" "$reason" "$elapsed"
}

run_auto_eval_restart_per_episode() {
  local total="$SONICSTAR_AUTO_EVAL_EPISODES"
  if ! [[ "$total" =~ ^[0-9]+$ ]] || [[ "$total" -lt 1 ]]; then
    printf 'Invalid --num-episodes for auto-eval mode: %s\n' "$total" >&2
    exit 1
  fi

  mkdir -p "$SESSION_LOG_DIR"
  printf '========================================\n'
  printf 'SONICSTAR VLA-JEPA Per-Episode Restart Auto Eval\n'
  printf 'Outer Session: %s\n' "$SESSION_ID"
  printf 'Episode Count: %s\n' "$total"
  printf 'Outer Step Logs: %s\n' "$SESSION_LOG_DIR"
  printf 'Result Log: %s\n' "$SONICSTAR_AUTO_EVAL_RESULT_LOG"
  if [[ -n "$SONICSTAR_TIMING_LOG" ]]; then
    printf 'Timing Log: %s\n' "$SONICSTAR_TIMING_LOG"
  fi
  if [[ -n "$SONICSTAR_ACTION_LOG" ]]; then
    printf 'Action Log: %s\n' "$SONICSTAR_ACTION_LOG"
  fi
  printf 'Each episode will launch a fresh Step1-Step5 stack and clean it up before the next episode.\n'
  printf '========================================\n'

  local ep
  for ((ep = 0; ep < total; ep++)); do
    local ep_tag
    local ep_log_dir
    printf -v ep_tag "%03d" "$ep"
    ep_log_dir="${SESSION_LOG_DIR}/episode_${ep_tag}"
    printf '\n[Outer Auto Eval] Episode %s/%s: launching fresh stack. Logs: %s\n' "$((ep + 1))" "$total" "$ep_log_dir"
    SONICSTAR_ROOT="$SONICSTAR_ROOT" \
    CKPT_PATH="$CKPT_PATH" \
    SONICSTAR_BACKEND="$SONICSTAR_BACKEND" \
    TRITON_MANIFEST="$TRITON_MANIFEST" \
    DRAFT_CKPT_PATH="$DRAFT_CKPT_PATH" \
    SONICSTAR_TIMING_LOG="$SONICSTAR_TIMING_LOG" \
    SONICSTAR_ACTION_LOG="$SONICSTAR_ACTION_LOG" \
    SONICSTAR_REPLAN_STEPS="$SONICSTAR_REPLAN_STEPS" \
    POLICY_RATE="$POLICY_RATE" \
    MAX_EXEC_STEPS="$MAX_EXEC_STEPS" \
    TAU_RADIUS="$TAU_RADIUS" \
    VERIFY_DIST_DIMS="$VERIFY_DIST_DIMS" \
    T_LIST="$T_LIST" \
    PERIODIC_FULL_EVERY_N_DRAFT_ROUNDS="$PERIODIC_FULL_EVERY_N_DRAFT_ROUNDS" \
    SONICSTAR_DRAFT_START_AFTER_FULL_ROUNDS="$SONICSTAR_DRAFT_START_AFTER_FULL_ROUNDS" \
    SONICSTAR_DRAFT_MIN_ACCEPT_STEPS="$SONICSTAR_DRAFT_MIN_ACCEPT_STEPS" \
    STARVLA_WS_PING_INTERVAL="$STARVLA_WS_PING_INTERVAL" \
    STARVLA_WS_PING_TIMEOUT="$STARVLA_WS_PING_TIMEOUT" \
    CAMERA_PORT="$CAMERA_PORT" \
    POLICY_PORT="$POLICY_PORT" \
    STEP1_TO_STEP2_DELAY="$STEP1_TO_STEP2_DELAY" \
    STEP2_TO_STEP3_DELAY="$STEP2_TO_STEP3_DELAY" \
    STEP3_TO_STEP4_DELAY="$STEP3_TO_STEP4_DELAY" \
    STEP4_TO_STEP5_DELAY="$STEP4_TO_STEP5_DELAY" \
    STEP5_TO_STEP6_DELAY="$STEP5_TO_STEP6_DELAY" \
    STEP5_INIT_TIMEOUT="$STEP5_INIT_TIMEOUT" \
    SONICSTAR_AUTO_EVAL="1" \
    SONICSTAR_AUTO_EVAL_EPISODES="1" \
    SONICSTAR_AUTO_EVAL_EPISODE_OFFSET="$ep" \
    SONICSTAR_AUTO_EVAL_CHILD="1" \
    SONICSTAR_ACTION_LOG_MODE="$SONICSTAR_ACTION_LOG_MODE" \
    SONICSTAR_EXPERIMENT_NAME="$SONICSTAR_EXPERIMENT_NAME" \
    SONICSTAR_TIMING_WARMUP_QUERIES="$SONICSTAR_TIMING_WARMUP_QUERIES" \
    SONICSTAR_TIMING_WARMUP_DRAFT_QUERIES="$SONICSTAR_TIMING_WARMUP_DRAFT_QUERIES" \
    SONICSTAR_TIMING_WARMUP_FULL_QUERIES="$SONICSTAR_TIMING_WARMUP_FULL_QUERIES" \
    SONICSTAR_FULL_BASELINE_MS="$SONICSTAR_FULL_BASELINE_MS" \
    SONICSTAR_AUTO_EVAL_MAX_STEPS="$SONICSTAR_AUTO_EVAL_MAX_STEPS" \
    SONICSTAR_AUTO_EVAL_RESULT_LOG="$SONICSTAR_AUTO_EVAL_RESULT_LOG" \
    SONICSTAR_AUTO_EVAL_SAVE_VIDEO="$SONICSTAR_AUTO_EVAL_SAVE_VIDEO" \
    SONICSTAR_AUTO_EVAL_VIDEO_DIR="$SONICSTAR_AUTO_EVAL_VIDEO_DIR" \
    SONICSTAR_AUTO_EVAL_VIDEO_FPS="$SONICSTAR_AUTO_EVAL_VIDEO_FPS" \
    SONICSTAR_AUTO_EVAL_SIM_HOST="$SONICSTAR_AUTO_EVAL_SIM_HOST" \
    SONICSTAR_AUTO_EVAL_SIM_PORT="$SONICSTAR_AUTO_EVAL_SIM_PORT" \
    SONICSTAR_AUTO_EVAL_CHECK_INTERVAL_STEPS="$SONICSTAR_AUTO_EVAL_CHECK_INTERVAL_STEPS" \
    SONICSTAR_AUTO_EVAL_SUCCESS_HOLD_CHECKS="$SONICSTAR_AUTO_EVAL_SUCCESS_HOLD_CHECKS" \
    SONICSTAR_AUTO_EVAL_NO_ACTION_TIMEOUT_SEC="$SONICSTAR_AUTO_EVAL_NO_ACTION_TIMEOUT_SEC" \
    SONICSTAR_AUTO_EVAL_PRESS_9_ON_START="$SONICSTAR_AUTO_EVAL_PRESS_9_ON_START" \
    SONICSTAR_AUTO_EVAL_POST_9_WAIT_SEC="$SONICSTAR_AUTO_EVAL_POST_9_WAIT_SEC" \
    SONICSTAR_AUTO_EVAL_BACKSPACE_RESETS="$SONICSTAR_AUTO_EVAL_BACKSPACE_RESETS" \
    SONICSTAR_AUTO_EVAL_BACKSPACE_WAIT_SEC="$SONICSTAR_AUTO_EVAL_BACKSPACE_WAIT_SEC" \
    SONICSTAR_AUTO_EVAL_MAX_SECONDS="$SONICSTAR_AUTO_EVAL_MAX_SECONDS" \
    SONICSTAR_AUTOMATION_LOG_DIR="$ep_log_dir" \
    STARVLA_CONDA_ENV="$STARVLA_CONDA_ENV" \
    DEPLOY_CONDA_ENV="$DEPLOY_CONDA_ENV" \
    TENSORRT_ROOT="$TENSORRT_ROOT" \
    TensorRT_ROOT="$TENSORRT_ROOT" \
    CUDA_TOOLKIT_ROOT="$CUDA_TOOLKIT_ROOT" \
    CUDAToolkit_ROOT="$CUDA_TOOLKIT_ROOT" \
    SIM_PYTHON="$SIM_PYTHON" \
    SONICSTAR_POLICY_WARMUP_RUNS="$SONICSTAR_POLICY_WARMUP_RUNS" \
    SONICSTAR_POLICY_WARMUP_TARGET_MS="$SONICSTAR_POLICY_WARMUP_TARGET_MS" \
    SONICSTAR_POLICY_WARMUP_MAX_RUNS="$SONICSTAR_POLICY_WARMUP_MAX_RUNS" \
    SONICSTAR_POLICY_WARMUP_IMAGE="$SONICSTAR_POLICY_WARMUP_IMAGE" \
    SONICSTAR_POLICY_WARMUP_PROMPT="$SONICSTAR_POLICY_WARMUP_PROMPT" \
    ENABLE_RANDOM_CUP_POSITION="$ENABLE_RANDOM_CUP_POSITION" \
    ENABLE_RANDOM_TABLE_COLOR="$ENABLE_RANDOM_TABLE_COLOR" \
    ENABLE_RANDOM_OBJECT_COLORS="$ENABLE_RANDOM_OBJECT_COLORS" \
    ENABLE_OVERHEAD_LIGHTING="$ENABLE_OVERHEAD_LIGHTING" \
    ENABLE_RANDOM_LIGHTING="$ENABLE_RANDOM_LIGHTING" \
    ENABLE_RANDOM_CAMERA_VIEW="$ENABLE_RANDOM_CAMERA_VIEW" \
    ENABLE_RANDOM_ROBOT_START="$ENABLE_RANDOM_ROBOT_START" \
    CUP_X_MIN="$CUP_X_MIN" \
    CUP_X_MAX="$CUP_X_MAX" \
    CUP_Y_MIN="$CUP_Y_MIN" \
    CUP_Y_MAX="$CUP_Y_MAX" \
    CAMERA_POS_JITTER_METERS="$CAMERA_POS_JITTER_METERS" \
    CAMERA_POS_JITTER_X_METERS="$CAMERA_POS_JITTER_X_METERS" \
    CAMERA_POS_JITTER_Y_METERS="$CAMERA_POS_JITTER_Y_METERS" \
    CAMERA_POS_JITTER_Z_METERS="$CAMERA_POS_JITTER_Z_METERS" \
    CAMERA_ROLL_JITTER_RADIANS="$CAMERA_ROLL_JITTER_RADIANS" \
    CAMERA_PITCH_JITTER_RADIANS="$CAMERA_PITCH_JITTER_RADIANS" \
    CAMERA_YAW_JITTER_RADIANS="$CAMERA_YAW_JITTER_RADIANS" \
    ROBOT_START_MIN_METERS="$ROBOT_START_MIN_METERS" \
    ROBOT_START_MAX_METERS="$ROBOT_START_MAX_METERS" \
    bash "$SCRIPT_DIR/run_sonicstar_vlajepa_automation.sh"
    printf '[Outer Auto Eval] Episode %s/%s complete.\n' "$((ep + 1))" "$total"
  done

  printf '\n[Outer Auto Eval] All %s episodes finished with per-episode process restart.\n' "$total"
}

main() {
  parse_cli_args "$@"
  refresh_session_log_dir
  refresh_experiment_name

  if [[ "$SONICSTAR_AUTO_EVAL" == "1" ]] && [[ "$SONICSTAR_AUTO_EVAL_CHILD" != "1" ]]; then
    run_auto_eval_restart_per_episode
    exit 0
  fi

  if [[ "$SONICSTAR_POLICY_WARMUP_RUNS" == "auto" ]]; then
    if [[ "$SONICSTAR_BACKEND" == "triton" ]]; then
      if [[ -n "$DRAFT_CKPT_PATH" ]]; then
        SONICSTAR_POLICY_WARMUP_RUNS="4"
        if [[ "$SONICSTAR_POLICY_WARMUP_MAX_RUNS" == "0" ]]; then
          SONICSTAR_POLICY_WARMUP_MAX_RUNS="10"
        fi
      else
        SONICSTAR_POLICY_WARMUP_RUNS="2"
        if [[ "$SONICSTAR_POLICY_WARMUP_MAX_RUNS" == "0" ]]; then
          SONICSTAR_POLICY_WARMUP_MAX_RUNS="8"
        fi
      fi
      if [[ "$SONICSTAR_POLICY_WARMUP_TARGET_MS" == "0" ]]; then
        SONICSTAR_POLICY_WARMUP_TARGET_MS="1000"
      fi
    else
      SONICSTAR_POLICY_WARMUP_RUNS="0"
    fi
  fi

  require_command gnome-terminal
  require_command lsof
  require_file "$STARVLA_ROOT/$POLICY_SERVER_SCRIPT"
  require_file "$SONICSTAR_ROOT/$INFERENCE_SCRIPT"
  require_file "$WBC_ROOT/$SIM_LOOP_SCRIPT"
  require_file "$WBC_ROOT/$CAMERA_VIEWER_SCRIPT"
  require_file "$WBC_ROOT/$SEND_KEY_SCRIPT"
  require_file "$WBC_ROOT/$SEND_MUJOCO_BACK_KEY_SCRIPT"
  require_file "$WBC_ROOT/$SCENE_POSITION_RANDOMIZER_SCRIPT"
  require_file "$WBC_ROOT/$SCENE_COLOR_RANDOMIZER_SCRIPT"
  require_file "$WBC_ROOT/$SCENE_LIGHTING_SCRIPT"
  require_file "$DEPLOY_SCRIPT"
  require_file "$CKPT_PATH"
  if [[ "$SONICSTAR_BACKEND" == "triton" ]] && [[ -n "$TRITON_MANIFEST" ]]; then
    require_file "$TRITON_MANIFEST"
  fi
  if [[ -n "$DRAFT_CKPT_PATH" ]]; then
    require_file "$DRAFT_CKPT_PATH"
  fi
  if [[ "${SONICSTAR_POLICY_WARMUP_RUNS}" != "0" ]] && [[ -n "$SONICSTAR_POLICY_WARMUP_IMAGE" ]]; then
    require_file "$SONICSTAR_POLICY_WARMUP_IMAGE"
  fi
  require_file "$SIM_PYTHON"
  require_file "$BASE_SCENE_XML"

  if [[ -z "$CONDA_SH" ]]; then
    printf 'Unable to locate conda.sh for activating the starVLA environment.\n' >&2
    exit 1
  fi

  preflight_cleanup

  mkdir -p "$PID_DIR"
  mkdir -p "$SESSION_LOG_DIR"
  resolve_step3_scene

  STARTED=1

  printf '========================================\n'
  printf 'SONICSTAR VLA-JEPA Automation Launcher\n'
  printf 'Session: %s\n' "$SESSION_ID"
  printf 'Step Logs: %s\n' "$SESSION_LOG_DIR"
  printf 'Checkpoint: %s\n' "$CKPT_PATH"
  printf 'Backend: %s\n' "$SONICSTAR_BACKEND"
  if [[ -n "$TRITON_MANIFEST" ]]; then
    printf 'Triton Manifest: %s\n' "$TRITON_MANIFEST"
  fi
  if [[ -n "$DRAFT_CKPT_PATH" ]]; then
    printf 'Draft Checkpoint: %s\n' "$DRAFT_CKPT_PATH"
  else
    printf 'Draft Checkpoint: disabled, baseline full path\n'
  fi
  if [[ -n "$SONICSTAR_TIMING_LOG" ]]; then
    printf 'Timing Log: %s\n' "$SONICSTAR_TIMING_LOG"
  fi
  if [[ -n "$SONICSTAR_ACTION_LOG" ]]; then
    printf 'Action Log: %s\n' "$SONICSTAR_ACTION_LOG"
  fi
  printf 'Action Log Mode: %s\n' "$SONICSTAR_ACTION_LOG_MODE"
  printf 'Experiment Name: %s\n' "$SONICSTAR_EXPERIMENT_NAME"
  printf 'Timing Warmup: queries=%s draft_route=%s full_route=%s\n' \
    "$SONICSTAR_TIMING_WARMUP_QUERIES" \
    "$SONICSTAR_TIMING_WARMUP_DRAFT_QUERIES" \
    "$SONICSTAR_TIMING_WARMUP_FULL_QUERIES"
  printf 'Fixed Full Baseline MS: %s\n' "${SONICSTAR_FULL_BASELINE_MS:-auto}"
  printf 'Policy Warmup Runs: %s\n' "$SONICSTAR_POLICY_WARMUP_RUNS"
  printf 'Draft Start After Full Rounds: %s\n' "$SONICSTAR_DRAFT_START_AFTER_FULL_ROUNDS"
  printf 'Draft Min Accept Steps: %s\n' "$SONICSTAR_DRAFT_MIN_ACCEPT_STEPS"
  printf 'Policy Warmup Target ms: %s\n' "$SONICSTAR_POLICY_WARMUP_TARGET_MS"
  printf 'Policy Warmup Max Runs: %s\n' "$SONICSTAR_POLICY_WARMUP_MAX_RUNS"
  printf 'Auto Eval: %s\n' "$([[ "$SONICSTAR_AUTO_EVAL" == "1" ]] && printf 'enabled' || printf 'disabled')"
  if [[ "$SONICSTAR_AUTO_EVAL" == "1" ]]; then
    printf 'Auto Eval Episodes: %s\n' "$SONICSTAR_AUTO_EVAL_EPISODES"
    printf 'Auto Eval Episode Offset: %s\n' "$SONICSTAR_AUTO_EVAL_EPISODE_OFFSET"
    printf 'Auto Eval Max Steps: %s\n' "$SONICSTAR_AUTO_EVAL_MAX_STEPS"
    printf 'Auto Eval Result Log: %s\n' "$SONICSTAR_AUTO_EVAL_RESULT_LOG"
    printf 'Auto Eval Save Video: %s\n' "$SONICSTAR_AUTO_EVAL_SAVE_VIDEO"
    printf 'Auto Eval Video Dir: %s\n' "$SONICSTAR_AUTO_EVAL_VIDEO_DIR"
    printf 'Auto Eval Sim Endpoint: %s:%s\n' "$SONICSTAR_AUTO_EVAL_SIM_HOST" "$SONICSTAR_AUTO_EVAL_SIM_PORT"
  fi
  if [[ "${SONICSTAR_POLICY_WARMUP_RUNS}" != "0" ]] && [[ -n "$SONICSTAR_POLICY_WARMUP_IMAGE" ]]; then
    printf 'Policy Warmup Image: %s\n' "$SONICSTAR_POLICY_WARMUP_IMAGE"
  fi
  printf 'TensorRT Root: %s\n' "$TENSORRT_ROOT"
  printf 'CUDA Toolkit Root: %s\n' "$CUDA_TOOLKIT_ROOT"
  printf 'Cup Position Mode: %s\n' "$([[ "$ENABLE_RANDOM_CUP_POSITION" == "1" ]] && printf 'randomized' || printf 'fixed')"
  if [[ "$ENABLE_RANDOM_TABLE_COLOR" == "1" ]]; then
    if [[ "$ENABLE_RANDOM_OBJECT_COLORS" == "1" ]]; then
      printf 'Color Mode: cup + table randomized\n'
    else
      printf 'Color Mode: table randomized only\n'
    fi
  else
    printf 'Color Mode: fixed original scene\n'
  fi
  if [[ "$ENABLE_OVERHEAD_LIGHTING" == "1" ]]; then
    if [[ "$ENABLE_RANDOM_LIGHTING" == "1" ]]; then
      printf 'Lighting Mode: overhead vertical randomized\n'
    else
      printf 'Lighting Mode: overhead vertical fixed\n'
    fi
  else
    printf 'Lighting Mode: disabled\n'
  fi
  printf 'Camera View Mode: %s\n' "$([[ "$ENABLE_RANDOM_CAMERA_VIEW" == "1" ]] && printf 'randomized' || printf 'fixed')"
  printf 'Robot Start Mode: %s\n' "$([[ "$ENABLE_RANDOM_ROBOT_START" == "1" ]] && printf 'randomized forward/back %s-%sm' "$ROBOT_START_MIN_METERS" "$ROBOT_START_MAX_METERS" || printf 'fixed')"
  printf '========================================\n'
  printf '\n'
  printf 'This script will launch step1-step5 in separate terminal windows.\n'
  if [[ "$SONICSTAR_AUTO_EVAL" == "1" ]]; then
    printf 'Auto-eval mode follows the manual launch order, auto-confirms deploy, sends k/9/BackSpace/i/i/p, records video, writes results, and exits.\n'
  else
    printf 'You still need to:\n'
    printf '1. Manually enter y in the deploy terminal when prompted.\n'
    printf '2. Manually press 9 in the MuJoCo window.\n'
    printf '3. Press Ctrl+C in this main terminal when you want to stop everything.\n'
  fi
  printf '\n'

  printf '[Launch] Step 1: Policy server\n'
  launch_terminal \
    "SONICSTAR VLA-JEPA Step1 Policy Server" \
    "set -euo pipefail; source '$CONDA_SH'; conda activate '$STARVLA_CONDA_ENV'; cd '$STARVLA_ROOT'; export CKPT_PATH='$CKPT_PATH'; export SONICSTAR_BACKEND='$SONICSTAR_BACKEND'; export TRITON_MANIFEST='$TRITON_MANIFEST'; export DRAFT_CKPT_PATH='$DRAFT_CKPT_PATH'; export MAX_EXEC_STEPS='$MAX_EXEC_STEPS'; export TAU_RADIUS='$TAU_RADIUS'; export VERIFY_DIST_DIMS='$VERIFY_DIST_DIMS'; export T_LIST='$T_LIST'; export PERIODIC_FULL_EVERY_N_DRAFT_ROUNDS='$PERIODIC_FULL_EVERY_N_DRAFT_ROUNDS'; export SONICSTAR_DRAFT_START_AFTER_FULL_ROUNDS='$SONICSTAR_DRAFT_START_AFTER_FULL_ROUNDS'; export SONICSTAR_DRAFT_MIN_ACCEPT_STEPS='$SONICSTAR_DRAFT_MIN_ACCEPT_STEPS'; export SONICSTAR_POLICY_WARMUP_RUNS='$SONICSTAR_POLICY_WARMUP_RUNS'; export SONICSTAR_POLICY_WARMUP_TARGET_MS='$SONICSTAR_POLICY_WARMUP_TARGET_MS'; export SONICSTAR_POLICY_WARMUP_MAX_RUNS='$SONICSTAR_POLICY_WARMUP_MAX_RUNS'; export SONICSTAR_POLICY_WARMUP_IMAGE='$SONICSTAR_POLICY_WARMUP_IMAGE'; export SONICSTAR_POLICY_WARMUP_PROMPT='$SONICSTAR_POLICY_WARMUP_PROMPT'; export STARVLA_WS_PING_INTERVAL='$STARVLA_WS_PING_INTERVAL'; export STARVLA_WS_PING_TIMEOUT='$STARVLA_WS_PING_TIMEOUT'; bash '$POLICY_SERVER_SCRIPT'" \
    "$PID_DIR/step1.pid"

  printf '[Wait] Sleeping %ss before Step 2...\n' "$STEP1_TO_STEP2_DELAY"
  sleep "$STEP1_TO_STEP2_DELAY"

  printf '[Launch] Step 2: Online inference\n'
  local inference_extra_args=()
  if [[ -n "$SONICSTAR_TIMING_LOG" ]]; then
    inference_extra_args+=(--timing-log "$SONICSTAR_TIMING_LOG")
  fi
  if [[ -n "$SONICSTAR_ACTION_LOG" ]]; then
    inference_extra_args+=(--action-log "$SONICSTAR_ACTION_LOG")
  fi
  local inference_launch_command
  inference_launch_command="set -euo pipefail; source '$CONDA_SH'; conda activate '$STARVLA_CONDA_ENV'; cd '$SONICSTAR_ROOT'; export STARVLA_WS_PING_INTERVAL='$STARVLA_WS_PING_INTERVAL'; export STARVLA_WS_PING_TIMEOUT='$STARVLA_WS_PING_TIMEOUT'; export SONICSTAR_ACTION_LOG_MODE='$SONICSTAR_ACTION_LOG_MODE'; export SONICSTAR_EXPERIMENT_NAME='$SONICSTAR_EXPERIMENT_NAME'; export SONICSTAR_TIMING_WARMUP_QUERIES='$SONICSTAR_TIMING_WARMUP_QUERIES'; export SONICSTAR_TIMING_WARMUP_DRAFT_QUERIES='$SONICSTAR_TIMING_WARMUP_DRAFT_QUERIES'; export SONICSTAR_TIMING_WARMUP_FULL_QUERIES='$SONICSTAR_TIMING_WARMUP_FULL_QUERIES'; export SONICSTAR_FULL_BASELINE_MS='$SONICSTAR_FULL_BASELINE_MS'; PYTHONPATH=\$PWD/starVLA:\$PWD/wbc python '$INFERENCE_SCRIPT' --ckpt-path '$CKPT_PATH' --host 127.0.0.1 --port '$POLICY_PORT' --prompt '$POLICY_PROMPT' --rate '$POLICY_RATE' --replan-steps '$SONICSTAR_REPLAN_STEPS' --log-action-stats ${inference_extra_args[*]}"
  launch_terminal \
    "SONICSTAR VLA-JEPA Step2 Online Inference" \
    "$inference_launch_command" \
    "$PID_DIR/step2.pid"

  printf '[Wait] Sleeping %ss before Step 3...\n' "$STEP2_TO_STEP3_DELAY"
  sleep "$STEP2_TO_STEP3_DELAY"

  printf '[Launch] Step 3: MuJoCo sim loop\n'
  local camera_randomization_args=()
  if [[ "$ENABLE_RANDOM_CAMERA_VIEW" == "1" ]]; then
    camera_randomization_args+=(
      "--camera-randomization-enabled"
      "--camera-pos-jitter-x-meters" "$CAMERA_POS_JITTER_X_METERS"
      "--camera-pos-jitter-y-meters" "$CAMERA_POS_JITTER_Y_METERS"
      "--camera-pos-jitter-z-meters" "$CAMERA_POS_JITTER_Z_METERS"
      "--camera-roll-jitter-radians" "$CAMERA_ROLL_JITTER_RADIANS"
      "--camera-pitch-jitter-radians" "$CAMERA_PITCH_JITTER_RADIANS"
      "--camera-yaw-jitter-radians" "$CAMERA_YAW_JITTER_RADIANS"
    )
  fi
  local robot_start_randomization_args=()
  if [[ "$ENABLE_RANDOM_ROBOT_START" == "1" ]]; then
    robot_start_randomization_args+=(
      "--robot-start-randomization-enabled"
      "--robot-start-min-meters" "$ROBOT_START_MIN_METERS"
      "--robot-start-max-meters" "$ROBOT_START_MAX_METERS"
    )
  fi
  local sim_eval_server_args=()
  if [[ "$SONICSTAR_AUTO_EVAL" == "1" ]]; then
    sim_eval_server_args+=(
      "--eval-server-enabled"
      "--eval-server-host" "$SONICSTAR_AUTO_EVAL_SIM_HOST"
      "--eval-server-port" "$SONICSTAR_AUTO_EVAL_SIM_PORT"
    )
  fi
  launch_terminal \
    "SONICSTAR VLA-JEPA Step3 Sim Loop" \
    "set -euo pipefail; cd '$WBC_ROOT'; '$SIM_PYTHON' '$SIM_LOOP_SCRIPT' --robot-scene-override '$STEP3_SCENE_XML' --enable-image-publish --enable-offscreen --camera-port '$CAMERA_PORT' ${camera_randomization_args[*]} ${robot_start_randomization_args[*]} ${sim_eval_server_args[*]}" \
    "$PID_DIR/step3.pid"

  printf '[Wait] Sleeping %ss before Step 4...\n' "$STEP3_TO_STEP4_DELAY"
  sleep "$STEP3_TO_STEP4_DELAY"

  printf '[Launch] Step 4: Camera viewer\n'
  local camera_viewer_extra_args=()
  local camera_viewer_env_exports=""
  if [[ "$SONICSTAR_AUTO_EVAL" == "1" ]] && [[ "$SONICSTAR_AUTO_EVAL_SAVE_VIDEO" == "1" ]]; then
    local video_episode_dir_for_viewer
    local auto_video_root_for_viewer
    local record_start_file_for_viewer
    local record_stop_file_for_viewer
    auto_video_root_for_viewer="$(sonic_abs_path "$SONICSTAR_AUTO_EVAL_VIDEO_DIR")"
    printf -v video_episode_dir_for_viewer "%s/episode_%03d" "$auto_video_root_for_viewer" "$SONICSTAR_AUTO_EVAL_EPISODE_OFFSET"
    record_start_file_for_viewer="$video_episode_dir_for_viewer/.record_start"
    record_stop_file_for_viewer="$video_episode_dir_for_viewer/.record_stop"
    mkdir -p "$video_episode_dir_for_viewer"
    rm -f "$record_start_file_for_viewer" "$record_stop_file_for_viewer"
    camera_viewer_extra_args+=(
      --output-path "$video_episode_dir_for_viewer"
      --record-start-file "$record_start_file_for_viewer"
      --record-stop-file "$record_stop_file_for_viewer"
    )
    camera_viewer_env_exports="export SONICSTAR_CAMERA_OUTPUT_PATH='$video_episode_dir_for_viewer'; export SONICSTAR_CAMERA_RECORD_START_FILE='$record_start_file_for_viewer'; export SONICSTAR_CAMERA_RECORD_STOP_FILE='$record_stop_file_for_viewer';"
  fi
  launch_terminal \
    "SONICSTAR VLA-JEPA Step4 Camera Viewer" \
    "set -euo pipefail; cd '$WBC_ROOT'; $camera_viewer_env_exports '$SIM_PYTHON' '$CAMERA_VIEWER_SCRIPT' --camera-host localhost --camera-port '$CAMERA_PORT' ${camera_viewer_extra_args[*]}" \
    "$PID_DIR/step4.pid"

  printf '[Wait] Sleeping %ss before Step 5...\n' "$STEP4_TO_STEP5_DELAY"
  sleep "$STEP4_TO_STEP5_DELAY"

  printf '[Launch] Step 5: Deploy\n'
  local deploy_launch_command
  if [[ "$SONICSTAR_AUTO_EVAL" == "1" ]]; then
    deploy_launch_command="printf 'y\n' | bash deploy.sh --input-type zmq_manager sim"
  else
    deploy_launch_command="bash deploy.sh --input-type zmq_manager sim"
  fi
  launch_terminal \
    "SONICSTAR VLA-JEPA Step5 Deploy" \
    "set -euo pipefail; source '$CONDA_SH'; conda activate '$DEPLOY_CONDA_ENV'; export TensorRT_ROOT='$TENSORRT_ROOT'; export TENSORRT_ROOT='$TENSORRT_ROOT'; export CUDAToolkit_ROOT='$CUDA_TOOLKIT_ROOT'; export CUDA_HOME='$CUDA_TOOLKIT_ROOT'; export LD_LIBRARY_PATH='$CUDA_TOOLKIT_ROOT/lib64:/usr/lib/x86_64-linux-gnu:/opt/onnxruntime/lib:'\${LD_LIBRARY_PATH:-}; cd '$WBC_ROOT/gear_sonic_deploy'; $deploy_launch_command" \
    "$PID_DIR/step5.pid"

  if [[ "$SONICSTAR_AUTO_EVAL" != "1" ]]; then
    printf '\n'
    printf '[Manual Action Required]\n'
    printf 'A Step5 deploy terminal has been opened.\n'
    printf 'When you see:\n'
    printf '  Proceed with deployment? [Y/n]:\n'
    printf 'please manually type:\n'
    printf '  y\n'
    printf '\n'
  fi
  if [[ "$SONICSTAR_AUTO_EVAL" == "1" ]]; then
    local step5_log
    printf '[Wait] Sleeping %ss before automated Step 6...\n' "$STEP5_TO_STEP6_DELAY"
    sleep "$STEP5_TO_STEP6_DELAY"
    step5_log="$(step_log_file "SONICSTAR VLA-JEPA Step5 Deploy")"
    printf '[Auto Eval] Waiting up to %ss for Step5 Init Done...\n' "$STEP5_INIT_TIMEOUT"
    if ! wait_for_log_pattern "$step5_log" "Init Done" "$STEP5_INIT_TIMEOUT"; then
      printf '[Auto Eval] Step5 did not reach Init Done. See log: %s\n' "$step5_log" >&2
      cleanup
      exit 1
    fi
    if ! wait_for_step2_ready; then
      cleanup
      exit 1
    fi
    printf '\n'
    printf '[Auto Eval] Step5 wait complete. Running automated manual interactions.\n'
    run_manual_step6_sequence "$SONICSTAR_AUTO_EVAL_EPISODE_OFFSET"
    monitor_auto_episode "$SONICSTAR_AUTO_EVAL_EPISODE_OFFSET" "$AUTO_EPISODE_VIDEO_DIR"
    printf '[Auto Eval] Episode finished. Cleaning up launched processes.\n'
    cleanup
    exit 0
  fi

  printf '[Wait] Sleeping %ss before Step 6...\n' "$STEP5_TO_STEP6_DELAY"
  sleep "$STEP5_TO_STEP6_DELAY"

  read -r -p "After Step5 has been running for 30s and reaches 'Init Done', press Enter here to continue Step6..." _

  step5_log="$(step_log_file "SONICSTAR VLA-JEPA Step5 Deploy")"
  printf '[Wait] Confirming Step5 Init Done from log...\n'
  if ! wait_for_log_pattern "$step5_log" "Init Done" "$STEP5_INIT_TIMEOUT"; then
    printf '[Wait] Step5 did not reach Init Done. See log: %s\n' "$step5_log" >&2
    cleanup
    exit 1
  fi
  if ! wait_for_step2_ready; then
    cleanup
    exit 1
  fi

  printf '\n'
  printf '[Step 6] Sending k...\n'
  send_key "k"

  printf '\n'
  printf '[Manual Action Required]\n'
  printf 'Go to the MuJoCo window and press 9.\n'
  printf 'After that, come back here and press Enter.\n'
  printf 'The script will then send one backspace reset automatically.\n'
  printf '\n'
  read -r -p "After you have pressed 9 in MuJoCo, press Enter here to continue..." _

  printf '\n[Step 6] Sending MuJoCo Back key reset...\n'
  send_mujoco_back_key
  sleep 1

  printf '\n[Step 6] Sending i, i, p...\n'
  send_key "i"
  sleep 1
  send_key "i"
  sleep 1
  send_key "p"

  printf '\n'
  printf '========================================\n'
  printf 'All steps have been launched.\n'
  printf 'Main script is now waiting.\n'
  printf 'Press Ctrl+C in this terminal to stop everything.\n'
  printf '========================================\n'
  printf '\n'

  while true; do
    sleep 1
  done
}

main "$@"
