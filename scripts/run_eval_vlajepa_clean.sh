#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TASK="${TASK:-simple/G1WholebodyHandoverTeleop-v0}"
POLICY="${POLICY:-vlajepa_decoupled_wbc}"
SPLIT="${SPLIT:-train}"
LEVEL="${LEVEL:-level-2}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-10090}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-1200}"
DATA_FORMAT="${DATA_FORMAT:-lerobot}"
NUM_EPISODES="${NUM_EPISODES:-}"
EPISODE_START="${EPISODE_START:-0}"
NUM_WORKERS="${NUM_WORKERS:-1}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
HEADLESS="${HEADLESS:-1}"
SIM_MODE="${SIM_MODE:-mujoco_isaac}"
MUJOCO_GL_VALUE="${MUJOCO_GL:-egl}"
EVAL_DIR="${EVAL_DIR:-data/evals_decoupled_wbc}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [level] [port] [-- extra eval-decoupled-wbc args]

Examples:
  $(basename "$0")
  $(basename "$0") level-0
  $(basename "$0") level-1 10090
  $(basename "$0") level-2 10090 -- --num-episodes 4

Environment overrides:
  TASK                 default: ${TASK}
  POLICY               default: ${POLICY}
  SPLIT                default: ${SPLIT}
  LEVEL                default: ${LEVEL}
  HOST                 default: ${HOST}
  PORT                 default: ${PORT}
  MAX_EPISODE_STEPS    default: ${MAX_EPISODE_STEPS}
  DATA_FORMAT          default: ${DATA_FORMAT}
  NUM_EPISODES         optional
  EPISODE_START        default: ${EPISODE_START}
  NUM_WORKERS          default: ${NUM_WORKERS}
  SAVE_VIDEO           1 or 0, default: ${SAVE_VIDEO}
  HEADLESS             1 or 0, default: ${HEADLESS}
  SIM_MODE             default: ${SIM_MODE}
  MUJOCO_GL            default: ${MUJOCO_GL_VALUE}
  EVAL_DIR             default: ${EVAL_DIR}
EOF
}

EXTRA_ARGS=()
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if [[ ${#POSITIONAL[@]} -ge 1 ]]; then
  LEVEL="${POSITIONAL[0]}"
fi

if [[ ${#POSITIONAL[@]} -ge 2 ]]; then
  PORT="${POSITIONAL[1]}"
fi

if [[ ${#POSITIONAL[@]} -gt 2 ]]; then
  echo "[run_eval_vlajepa_clean] too many positional arguments" >&2
  usage >&2
  exit 1
fi

task_name="${TASK#simple/}"
DATA_DIR="${DATA_DIR:-data/evals/simple-eval/${task_name}/${LEVEL}}"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "[run_eval_vlajepa_clean] data dir not found: $DATA_DIR" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[run_eval_vlajepa_clean] missing required command: uv" >&2
  exit 1
fi

if [[ -r /proc/sys/fs/inotify/max_user_watches ]]; then
  current_watches="$(< /proc/sys/fs/inotify/max_user_watches)"
  if [[ "${current_watches}" -lt 262144 ]]; then
    echo "[run_eval_vlajepa_clean] warning: fs.inotify.max_user_watches=${current_watches} is low for Isaac Sim." >&2
    echo "[run_eval_vlajepa_clean] recommendation: sudo sysctl fs.inotify.max_user_watches=524288" >&2
  fi
fi

video_flag="--save-video"
if [[ "$SAVE_VIDEO" == "0" ]]; then
  video_flag="--no-save-video"
fi

headless_flag="--headless"
if [[ "$HEADLESS" == "0" ]]; then
  headless_flag="--no-headless"
fi

CMD=(
  uv run eval-decoupled-wbc
  "$TASK"
  "$POLICY"
  "$SPLIT"
  --data-format "$DATA_FORMAT"
  --data-dir "$DATA_DIR"
  --host "$HOST"
  --port "$PORT"
  "$headless_flag"
  --sim-mode "$SIM_MODE"
  --eval-dir "$EVAL_DIR"
  --max-episode-steps "$MAX_EPISODE_STEPS"
  --episode-start "$EPISODE_START"
  --num-workers "$NUM_WORKERS"
  "$video_flag"
)

if [[ -n "$NUM_EPISODES" ]]; then
  CMD+=(--num-episodes "$NUM_EPISODES")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[run_eval_vlajepa_clean] repo: $ROOT_DIR"
echo "[run_eval_vlajepa_clean] task: $TASK"
echo "[run_eval_vlajepa_clean] policy: $POLICY"
echo "[run_eval_vlajepa_clean] split: $SPLIT"
echo "[run_eval_vlajepa_clean] data dir: $DATA_DIR"
echo "[run_eval_vlajepa_clean] eval dir: $EVAL_DIR"
echo "[run_eval_vlajepa_clean] server: ${HOST}:${PORT}"
echo "[run_eval_vlajepa_clean] mujoco gl: $MUJOCO_GL_VALUE"
echo "[run_eval_vlajepa_clean] clearing CONDA/PYTHON/LD runtime pollution before launch"

MUJOCO_GL="$MUJOCO_GL_VALUE" env \
  -u CONDA_PREFIX \
  -u CONDA_DEFAULT_ENV \
  -u PYTHONPATH \
  -u PYTHONHOME \
  -u LD_PRELOAD \
  -u LD_LIBRARY_PATH \
  "${CMD[@]}"
