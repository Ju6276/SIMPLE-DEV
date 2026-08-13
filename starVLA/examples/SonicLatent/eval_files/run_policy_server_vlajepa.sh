#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${STARVLA_DIR}"

export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

###########################################################################################
# === Modify these for your environment ===
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
CKPT_PATH="${CKPT_PATH:-/home/d086/fangbaozhong/Sonicstar/CKPTSONICSTAR/SONICSTAR/checkpoints/steps_90000_pytorch_model.pt}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-10093}"
DEVICE="${DEVICE:-cuda:0}"
SONICSTAR_BACKEND="${SONICSTAR_BACKEND:-pytorch}"
TRITON_MANIFEST="${TRITON_MANIFEST:-}"
DRAFT_CKPT_PATH="${DRAFT_CKPT_PATH:-}"
MAX_EXEC_STEPS="${MAX_EXEC_STEPS:-20}"
TAU_RADIUS="${TAU_RADIUS:-0.25}"
VERIFY_DIST_DIMS="${VERIFY_DIST_DIMS:-64}"
T_LIST="${T_LIST:-0.1 0.05}"
PERIODIC_FULL_EVERY_N_DRAFT_ROUNDS="${PERIODIC_FULL_EVERY_N_DRAFT_ROUNDS:-1}"
SONICSTAR_DRAFT_START_AFTER_FULL_ROUNDS="${SONICSTAR_DRAFT_START_AFTER_FULL_ROUNDS:-0}"
SONICSTAR_DRAFT_MIN_ACCEPT_STEPS="${SONICSTAR_DRAFT_MIN_ACCEPT_STEPS:-1}"
JEPA_COMPILE_ACTION_VERIFY="${JEPA_COMPILE_ACTION_VERIFY:-0}"
SONICSTAR_POLICY_WARMUP_RUNS="${SONICSTAR_POLICY_WARMUP_RUNS:-0}"
SONICSTAR_POLICY_WARMUP_TARGET_MS="${SONICSTAR_POLICY_WARMUP_TARGET_MS:-0}"
SONICSTAR_POLICY_WARMUP_MAX_RUNS="${SONICSTAR_POLICY_WARMUP_MAX_RUNS:-0}"
SONICSTAR_POLICY_WARMUP_IMAGE="${SONICSTAR_POLICY_WARMUP_IMAGE:-}"
SONICSTAR_POLICY_WARMUP_PROMPT="${SONICSTAR_POLICY_WARMUP_PROMPT:-pick up the cylinder and throw it into the trash bin}"
###########################################################################################

server_args=(
  --ckpt_path "${CKPT_PATH}"
  --port "${PORT}" \
  --device "${DEVICE}"
  --use_bf16
  --backend "${SONICSTAR_BACKEND}"
  --max_exec_steps "${MAX_EXEC_STEPS}"
  --tau_radius "${TAU_RADIUS}"
  --verify_dist_dims "${VERIFY_DIST_DIMS}"
  --periodic_full_every_n_draft_rounds "${PERIODIC_FULL_EVERY_N_DRAFT_ROUNDS}"
  --draft_start_after_full_rounds "${SONICSTAR_DRAFT_START_AFTER_FULL_ROUNDS}"
  --draft_min_accept_steps "${SONICSTAR_DRAFT_MIN_ACCEPT_STEPS}"
  --warmup_runs "${SONICSTAR_POLICY_WARMUP_RUNS}"
  --warmup_target_ms "${SONICSTAR_POLICY_WARMUP_TARGET_MS}"
  --warmup_max_runs "${SONICSTAR_POLICY_WARMUP_MAX_RUNS}"
  --warmup_prompt "${SONICSTAR_POLICY_WARMUP_PROMPT}"
  --t_list
)

read -r -a t_list_args <<< "${T_LIST}"
server_args+=("${t_list_args[@]}")

if [[ -n "${DRAFT_CKPT_PATH}" ]]; then
  server_args+=(--draft_checkpoint "${DRAFT_CKPT_PATH}")
fi

if [[ -n "${TRITON_MANIFEST}" ]]; then
  server_args+=(--triton_manifest "${TRITON_MANIFEST}")
fi

if [[ -n "${SONICSTAR_POLICY_WARMUP_IMAGE}" ]]; then
  server_args+=(--warmup_image "${SONICSTAR_POLICY_WARMUP_IMAGE}")
fi

export JEPA_COMPILE_ACTION_VERIFY

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${STARVLA_PYTHON}" deployment/model_server/server_policy_vlajepa.py "${server_args[@]}"
