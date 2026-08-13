#!/usr/bin/env bash

set -euo pipefail

SONICSTAR_ROOT="/home/d013/桌面/SonicStar"
STARVLA_ROOT="${SONICSTAR_ROOT}/starVLA"
WBC_ROOT="${SONICSTAR_ROOT}/wbc"
GR00T_ROOT="/home/d013/桌面/GR00T-WholeBodyControl"

CKPT_PATH="/home/d013/桌面/SonicStar/starVLA/starVLA/playground/Checkpoints/sonic_latent_scratch_frozen_vlm/checkpoints/steps_90000_pytorch_model.pt"
POLICY_PROMPT="pick up the cylinder and throw it into the trash bin"

POLICY_SERVER_SCRIPT="examples/SonicLatent/eval_files/run_policy_server.sh"
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

CAMERA_PORT="5555"
POLICY_PORT="10093"
STEP1_TO_STEP2_DELAY=20
STEP2_TO_STEP3_DELAY=5
STEP3_TO_STEP4_DELAY=5
STEP4_TO_STEP5_DELAY=5
STEP5_TO_STEP6_DELAY=30

SIM_PYTHON="${GR00T_ROOT}/.venv_sim/bin/python"
SESSION_ID="sonicstar_vla_automation_$(date +%s)"
STARTED=0
CLEANED=0
PID_DIR="/tmp/${SESSION_ID}"
RANDOMIZED_POSITION_SCENE_XML=""
RANDOMIZED_COLOR_SCENE_XML=""
OVERHEAD_LIGHTING_SCENE_XML=""
STEP3_SCENE_XML=""
ENABLE_RANDOM_CUP_POSITION="${ENABLE_RANDOM_CUP_POSITION:-0}"
ENABLE_RANDOM_TABLE_COLOR="${ENABLE_RANDOM_TABLE_COLOR:-0}"
ENABLE_OVERHEAD_LIGHTING="${ENABLE_OVERHEAD_LIGHTING:-0}"
ENABLE_RANDOM_LIGHTING="${ENABLE_RANDOM_LIGHTING:-0}"
ENABLE_RANDOM_CAMERA_VIEW="${ENABLE_RANDOM_CAMERA_VIEW:-0}"

CAMERA_POS_JITTER_METERS="${CAMERA_POS_JITTER_METERS:-0.01}"
CAMERA_POS_JITTER_X_METERS="${CAMERA_POS_JITTER_X_METERS:-0.01}"
CAMERA_POS_JITTER_Y_METERS="${CAMERA_POS_JITTER_Y_METERS:-0.01}"
CAMERA_POS_JITTER_Z_METERS="${CAMERA_POS_JITTER_Z_METERS:-0.004}"
CAMERA_ROLL_JITTER_RADIANS="${CAMERA_ROLL_JITTER_RADIANS:-0.0}"
CAMERA_PITCH_JITTER_RADIANS="${CAMERA_PITCH_JITTER_RADIANS:-0.10}"
CAMERA_YAW_JITTER_RADIANS="${CAMERA_YAW_JITTER_RADIANS:-0.10}"

# Random cup placement bounds for the red-box region on the table.
CUP_X_MIN="0.56"
CUP_X_MAX="0.74"
CUP_Y_MIN="-0.22"
CUP_Y_MAX="0.28"

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
Usage: $(basename "$0") [--random-cup] [--fixed-cup] [--random-table-color] [--fixed-table-color] [--enable-lights] [--disable-lights] [--random-lights] [--fixed-lights] [--random-camera-view] [--fixed-camera-view] [--help]

Options:
  --random-cup          Randomize the cup position within the configured table region.
  --fixed-cup           Use the original fixed cup position from scene_43dof.xml.
  --random-table-color  Randomize only the table color; keep the cup color unchanged.
  --fixed-table-color   Use the original table color from scene_43dof.xml.
  --enable-lights       Enable a vertical overhead light directly above the table.
  --disable-lights      Use the original scene lighting from scene_43dof.xml.
  --random-lights       Enable lights and randomize overhead position/intensity.
  --fixed-lights        Use a fixed overhead vertical light above the table center.
  --random-camera-view  Randomize only the MuJoCo head camera pose slightly.
  --fixed-camera-view   Use the original MuJoCo head camera pose.
  --help                Show this help message.

Environment:
  ENABLE_RANDOM_CUP_POSITION=1   Same as --random-cup
  ENABLE_RANDOM_TABLE_COLOR=1    Same as --random-table-color
  ENABLE_OVERHEAD_LIGHTING=1     Same as --enable-lights
  ENABLE_RANDOM_LIGHTING=1       Same as --random-lights
  ENABLE_RANDOM_CAMERA_VIEW=1    Same as --random-camera-view
EOF
}

parse_cli_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
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
        shift
        ;;
      --fixed-table-color)
        ENABLE_RANDOM_TABLE_COLOR="0"
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
  RANDOMIZED_COLOR_SCENE_XML="$(mktemp -p "$(dirname "$BASE_SCENE_XML")" scene_43dof_colors_XXXX.xml)"
  "$SIM_PYTHON" "${WBC_ROOT}/${SCENE_COLOR_RANDOMIZER_SCRIPT}" \
    --input-scene "$input_scene" \
    --output-scene "$RANDOMIZED_COLOR_SCENE_XML" \
    --table-only
  printf '[Step 3 Prep] Randomized table color scene created: %s\n' "$RANDOMIZED_COLOR_SCENE_XML"
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
    printf '[Step 3 Prep] Table color mode: randomized (cup color unchanged)\n'
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
  printf '[Preflight] Cleaning up old SONICSTAR processes...\n'
  pkill -f "examples/SonicLatent/eval_files/run_policy_server.sh" >/dev/null 2>&1 || true
  pkill -f "examples/SonicLatent/eval_files/run_starvla_inference.py" >/dev/null 2>&1 || true
  pkill -f "gear_sonic/scripts/run_sim_loop.py --enable-image-publish --enable-offscreen --camera-port ${CAMERA_PORT}" >/dev/null 2>&1 || true
  pkill -f "gear_sonic/scripts/run_camera_viewer.py --camera-host localhost --camera-port ${CAMERA_PORT}" >/dev/null 2>&1 || true
  pkill -f "bash deploy.sh --input-type zmq_manager sim" >/dev/null 2>&1 || true
  pkill -f "g1_deploy_onnx_ref" >/dev/null 2>&1 || true

  sleep 1

  pkill -9 -f "examples/SonicLatent/eval_files/run_policy_server.sh" >/dev/null 2>&1 || true
  pkill -9 -f "examples/SonicLatent/eval_files/run_starvla_inference.py" >/dev/null 2>&1 || true
  pkill -9 -f "gear_sonic/scripts/run_sim_loop.py --enable-image-publish --enable-offscreen --camera-port ${CAMERA_PORT}" >/dev/null 2>&1 || true
  pkill -9 -f "gear_sonic/scripts/run_camera_viewer.py --camera-host localhost --camera-port ${CAMERA_PORT}" >/dev/null 2>&1 || true
  pkill -9 -f "bash deploy.sh --input-type zmq_manager sim" >/dev/null 2>&1 || true
  pkill -9 -f "g1_deploy_onnx_ref" >/dev/null 2>&1 || true

  kill_port_processes "$POLICY_PORT"
  kill_port_processes "$CAMERA_PORT"

  rm -rf /tmp/sonicstar_vla_automation_* >/dev/null 2>&1 || true
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
  pkill -f "examples/SonicLatent/eval_files/run_starvla_inference.py" >/dev/null 2>&1 || true
  pkill -f "gear_sonic/scripts/run_sim_loop.py --enable-image-publish --enable-offscreen --camera-port ${CAMERA_PORT}" >/dev/null 2>&1 || true
  pkill -f "gear_sonic/scripts/run_camera_viewer.py --camera-host localhost --camera-port ${CAMERA_PORT}" >/dev/null 2>&1 || true
  pkill -f "bash deploy.sh --input-type zmq_manager sim" >/dev/null 2>&1 || true
  pkill -f "g1_deploy_onnx_ref" >/dev/null 2>&1 || true
  sleep 1
  pkill -9 -f "examples/SonicLatent/eval_files/run_policy_server.sh" >/dev/null 2>&1 || true
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
  local wrapped_command

  wrapped_command=$(
    cat <<EOF
set -uo pipefail
trap 'exit 0' TERM INT
echo \$\$ > "$pid_file"
if $command; then
  status=0
else
  status=\$?
fi
echo
echo "[${title}] Process exited with code: \$status"
read -r -p "Press Enter to close this terminal..." _
EOF
  )

  gnome-terminal \
    --title="$title" \
    -- bash -lc "$wrapped_command" >/dev/null 2>&1 &
}

send_key() {
  local key="$1"
  printf '[Step 6] Sending key: %s\n' "$key"
  "$SIM_PYTHON" "${WBC_ROOT}/${SEND_KEY_SCRIPT}" "$key"
}

send_mujoco_back_key() {
  printf '[Step 6] Sending real MuJoCo Back key...\n'
  "$SIM_PYTHON" "${WBC_ROOT}/${SEND_MUJOCO_BACK_KEY_SCRIPT}" --title-contains "mujoco"
}

main() {
  parse_cli_args "$@"

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
  require_file "$SIM_PYTHON"
  require_file "$BASE_SCENE_XML"

  if [[ -z "$CONDA_SH" ]]; then
    printf 'Unable to locate conda.sh for activating the starVLA environment.\n' >&2
    exit 1
  fi

  preflight_cleanup

  mkdir -p "$PID_DIR"
  resolve_step3_scene

  STARTED=1

  printf '========================================\n'
  printf 'SONICSTAR VLA Automation Launcher\n'
  printf 'Session: %s\n' "$SESSION_ID"
  printf 'Cup Position Mode: %s\n' "$([[ "$ENABLE_RANDOM_CUP_POSITION" == "1" ]] && printf 'randomized' || printf 'fixed')"
  printf 'Table Color Mode: %s\n' "$([[ "$ENABLE_RANDOM_TABLE_COLOR" == "1" ]] && printf 'randomized' || printf 'fixed')"
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
  printf '========================================\n'
  printf '\n'
  printf 'This script will launch step1-step5 in separate terminal windows.\n'
  printf 'You still need to:\n'
  printf '1. Manually enter y in the deploy terminal when prompted.\n'
  printf '2. Manually press 9 in the MuJoCo window.\n'
  printf '3. Press Ctrl+C in this main terminal when you want to stop everything.\n'
  printf '\n'

  printf '[Launch] Step 1: Policy server\n'
  launch_terminal \
    "SONICSTAR Step1 Policy Server" \
    "set -euo pipefail; source '$CONDA_SH'; conda activate starVLA; cd '$STARVLA_ROOT'; export CKPT_PATH='$CKPT_PATH'; bash '$POLICY_SERVER_SCRIPT'" \
    "$PID_DIR/step1.pid"

  printf '[Wait] Sleeping %ss before Step 2...\n' "$STEP1_TO_STEP2_DELAY"
  sleep "$STEP1_TO_STEP2_DELAY"

  printf '[Launch] Step 2: Online inference\n'
  launch_terminal \
    "SONICSTAR Step2 Online Inference" \
    "set -euo pipefail; source '$CONDA_SH'; conda activate starVLA; cd '$SONICSTAR_ROOT'; PYTHONPATH=\$PWD/starVLA:\$PWD/wbc python '$INFERENCE_SCRIPT' --ckpt-path '$CKPT_PATH' --host 127.0.0.1 --port '$POLICY_PORT' --prompt '$POLICY_PROMPT' --rate 1.0" \
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
  launch_terminal \
    "SONICSTAR Step3 Sim Loop" \
    "set -euo pipefail; source '$GR00T_ROOT/.venv_sim/bin/activate'; cd '$WBC_ROOT'; python '$SIM_LOOP_SCRIPT' --robot-scene-override '$STEP3_SCENE_XML' --enable-image-publish --enable-offscreen --camera-port '$CAMERA_PORT' ${camera_randomization_args[*]}" \
    "$PID_DIR/step3.pid"

  printf '[Wait] Sleeping %ss before Step 4...\n' "$STEP3_TO_STEP4_DELAY"
  sleep "$STEP3_TO_STEP4_DELAY"

  printf '[Launch] Step 4: Camera viewer\n'
  launch_terminal \
    "SONICSTAR Step4 Camera Viewer" \
    "set -euo pipefail; source '$GR00T_ROOT/.venv_sim/bin/activate'; cd '$WBC_ROOT'; python '$CAMERA_VIEWER_SCRIPT' --camera-host localhost --camera-port '$CAMERA_PORT'" \
    "$PID_DIR/step4.pid"

  printf '[Wait] Sleeping %ss before Step 5...\n' "$STEP4_TO_STEP5_DELAY"
  sleep "$STEP4_TO_STEP5_DELAY"

  printf '[Launch] Step 5: C++ deploy\n'
  launch_terminal \
    "SONICSTAR Step5 Deploy" \
    "source '$CONDA_SH'; conda activate g1_deploy; cd '$WBC_ROOT/gear_sonic_deploy'; bash deploy.sh --input-type zmq_manager sim" \
    "$PID_DIR/step5.pid"

  printf '\n'
  printf '[Manual Action Required]\n'
  printf 'A Step5 deploy terminal has been opened.\n'
  printf 'When you see:\n'
  printf '  Proceed with deployment? [Y/n]:\n'
  printf 'please manually type:\n'
  printf '  y\n'
  printf '\n'
  printf '[Wait] Sleeping %ss before Step 6...\n' "$STEP5_TO_STEP6_DELAY"
  sleep "$STEP5_TO_STEP6_DELAY"
  read -r -p "After Step5 has been running for 30s and reaches 'Init Done', press Enter here to continue Step6..." _

  printf '\n[Step 6] Sending k...\n'
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
