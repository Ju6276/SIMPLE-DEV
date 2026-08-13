#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TASK="${TASK:-simple/G1WholebodyOpenTrashCanTeleop-v0}"
POLICY="${POLICY:-psi0_decoupled_wbc}"
SPLIT="${SPLIT:-train}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-22085}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-800}"
DATA_FORMAT="${DATA_FORMAT:-lerobot}"
NUM_EPISODES="${NUM_EPISODES:-10}"
EPISODE_START="${EPISODE_START:-0}"
NUM_WORKERS="${NUM_WORKERS:-1}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
HEADLESS="${HEADLESS:-1}"
SIM_MODE="${SIM_MODE:-mujoco_isaac}"
MUJOCO_GL_VALUE="${MUJOCO_GL:-egl}"
LEVELS="${LEVELS:-level-0 level-1 level-2}"
EVAL_DIR_ROOT="${EVAL_DIR_ROOT:-data/evals_decoupled_wbc}"
CLEAR_OLD_RESULTS="${CLEAR_OLD_RESULTS:-1}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [TASK] [PORT] [NUM_EPISODES]

Examples:
  $(basename "$0")
  $(basename "$0") simple/G1WholebodyOpenTrashCanTeleop-v0
  $(basename "$0") simple/G1WholebodyOpenTrashCanTeleop-v0 22085
  $(basename "$0") simple/G1WholebodyOpenTrashCanTeleop-v0 22085 10

Environment overrides:
  TASK                 default: ${TASK}
  POLICY               default: ${POLICY}
  SPLIT                default: ${SPLIT}
  HOST                 default: ${HOST}
  PORT                 default: ${PORT}
  MAX_EPISODE_STEPS    default: ${MAX_EPISODE_STEPS}
  DATA_FORMAT          default: ${DATA_FORMAT}
  NUM_EPISODES         default: ${NUM_EPISODES}
  EPISODE_START        default: ${EPISODE_START}
  NUM_WORKERS          default: ${NUM_WORKERS}
  SAVE_VIDEO           1 or 0, default: ${SAVE_VIDEO}
  HEADLESS             1 or 0, default: ${HEADLESS}
  SIM_MODE             default: ${SIM_MODE}
  MUJOCO_GL            default: ${MUJOCO_GL_VALUE}
  LEVELS               space-separated list, default: ${LEVELS}
  EVAL_DIR_ROOT        default: ${EVAL_DIR_ROOT}
  CLEAR_OLD_RESULTS    1 or 0, default: ${CLEAR_OLD_RESULTS}
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ $# -ge 1 ]]; then
  TASK="$1"
fi

if [[ $# -ge 2 ]]; then
  PORT="$2"
fi

if [[ $# -ge 3 ]]; then
  NUM_EPISODES="$3"
fi

if [[ $# -gt 3 ]]; then
  echo "[run_eval_psi0_all_levels] too many positional arguments" >&2
  usage >&2
  exit 1
fi

SCRIPT_PATH="$ROOT_DIR/scripts/run_eval_psi0_clean.sh"
if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "[run_eval_psi0_all_levels] missing script: $SCRIPT_PATH" >&2
  exit 1
fi

echo "[run_eval_psi0_all_levels] repo: $ROOT_DIR"
echo "[run_eval_psi0_all_levels] task: $TASK"
echo "[run_eval_psi0_all_levels] policy: $POLICY"
echo "[run_eval_psi0_all_levels] server: ${HOST}:${PORT}"
echo "[run_eval_psi0_all_levels] num episodes per level: $NUM_EPISODES"
echo "[run_eval_psi0_all_levels] levels: $LEVELS"
echo "[run_eval_psi0_all_levels] eval dir root: $EVAL_DIR_ROOT"

task_name="${TASK#simple/}"

if [[ "$CLEAR_OLD_RESULTS" == "1" ]]; then
  for level in $LEVELS; do
    level_eval_dir="$EVAL_DIR_ROOT/$level"
    level_task_dir="$level_eval_dir/$POLICY/$task_name"
    if [[ -d "$level_task_dir" ]]; then
      echo "[run_eval_psi0_all_levels] removing old results: $level_task_dir"
      rm -rf "$level_task_dir"
    fi
  done
fi

for level in $LEVELS; do
  level_eval_dir="$EVAL_DIR_ROOT/$level"
  echo
  echo "========== Running ${TASK} ${level} =========="
  if ! TASK="$TASK" \
    POLICY="$POLICY" \
    SPLIT="$SPLIT" \
    HOST="$HOST" \
    PORT="$PORT" \
    MAX_EPISODE_STEPS="$MAX_EPISODE_STEPS" \
    DATA_FORMAT="$DATA_FORMAT" \
    NUM_EPISODES="$NUM_EPISODES" \
    EPISODE_START="$EPISODE_START" \
    NUM_WORKERS="$NUM_WORKERS" \
    SAVE_VIDEO="$SAVE_VIDEO" \
    HEADLESS="$HEADLESS" \
    SIM_MODE="$SIM_MODE" \
    MUJOCO_GL="$MUJOCO_GL_VALUE" \
    EVAL_DIR="$level_eval_dir" \
    bash "$SCRIPT_PATH" "$level"; then
    echo "[run_eval_psi0_all_levels] failed at ${level}" >&2
    exit 1
  fi
done

echo
echo "[run_eval_psi0_all_levels] finished all levels for $TASK"
for level in $LEVELS; do
  echo "[run_eval_psi0_all_levels] ${level} log: $ROOT_DIR/${EVAL_DIR_ROOT}/${level}/eval_latest.log"
  echo "[run_eval_psi0_all_levels] ${level} stats: $ROOT_DIR/${EVAL_DIR_ROOT}/${level}/eval_stats.txt"
done
