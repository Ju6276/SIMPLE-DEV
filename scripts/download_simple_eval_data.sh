#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TASK_NAME="${1:-G1WholebodyLocomotionPickBetweenTablesTeleop-v0}"
shift || true

DEST_DIR="${DEST_DIR:-data/evals}"
REPO_ID="${REPO_ID:-USC-PSI-Lab/psi-data}"
TARGET_ROOT="simple-eval"
ZIP_REL_PATH="${TARGET_ROOT}/${TASK_NAME}.zip"

usage() {
  cat <<EOF
Usage: $(basename "$0") [TASK_NAME] [level...]

Examples:
  $(basename "$0")
  $(basename "$0") G1WholebodyLocomotionPickBetweenTablesTeleop-v0
  $(basename "$0") G1WholebodyLocomotionPickBetweenTablesTeleop-v0 level-0 level-1 level-2

Environment overrides:
  DEST_DIR   default: ${DEST_DIR}
  REPO_ID    default: ${REPO_ID}
EOF
}

if [[ "$TASK_NAME" == "--help" || "$TASK_NAME" == "-h" ]]; then
  usage
  exit 0
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "[download_simple_eval_data] missing required command: hf" >&2
  exit 1
fi

CMD=(
  hf download "$REPO_ID"
  "$ZIP_REL_PATH"
  --repo-type dataset
  --local-dir "$DEST_DIR"
)

echo "[download_simple_eval_data] repo: $REPO_ID"
echo "[download_simple_eval_data] task: $TASK_NAME"
echo "[download_simple_eval_data] dest: $DEST_DIR"
printf '[download_simple_eval_data] command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

ZIP_PATH="${DEST_DIR}/${ZIP_REL_PATH}"
LOCAL_TASK_DIR="${DEST_DIR}/${TARGET_ROOT}/${TASK_NAME}"

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "[download_simple_eval_data] expected zip missing: $ZIP_PATH" >&2
  exit 1
fi

mkdir -p "${DEST_DIR}/${TARGET_ROOT}"
if [[ $# -eq 0 ]]; then
  unzip -o "$ZIP_PATH" -d "${DEST_DIR}/${TARGET_ROOT}" >/dev/null
else
  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "$tmp_dir"' EXIT
  unzip -o "$ZIP_PATH" -d "$tmp_dir" >/dev/null
  mkdir -p "$LOCAL_TASK_DIR"
  for level in "$@"; do
    src_level_dir="${tmp_dir}/${TASK_NAME}/${level}"
    dst_level_dir="${LOCAL_TASK_DIR}/${level}"
    if [[ ! -d "$src_level_dir" ]]; then
      echo "[download_simple_eval_data] level not found in zip: $level" >&2
      exit 1
    fi
    rm -rf "$dst_level_dir"
    mv "$src_level_dir" "$dst_level_dir"
  done
fi

if [[ -d "$LOCAL_TASK_DIR" ]]; then
  echo "[download_simple_eval_data] ready: $LOCAL_TASK_DIR"
else
  echo "[download_simple_eval_data] download finished, but expected dir missing: $LOCAL_TASK_DIR" >&2
  exit 1
fi
