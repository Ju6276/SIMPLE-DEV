#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SIMPLE_ROOT="$ROOT_DIR"

if [[ -f "$ROOT_DIR/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$ROOT_DIR/.venv/bin/activate"
else
    echo "Missing virtual environment activate script at $ROOT_DIR/.venv/bin/activate" >&2
    return 1 2>/dev/null || exit 1
fi

export CUDA_HOME="/home/d013/桌面/cuda-12.8"
export PATH="$CUDA_HOME/bin:$PATH"

TORCH_LIB_DIR="$ROOT_DIR/.venv/lib/python3.10/site-packages/torch/lib"
export LD_LIBRARY_PATH="$TORCH_LIB_DIR:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Optional: borrow a conda-provided Pinocchio stack while keeping SIMPLE on .venv.
SIMPLE_PIN_PREFIX_DEFAULT="$HOME/anaconda3/envs/simple-pin"
export SIMPLE_PIN_PREFIX="${SIMPLE_PIN_PREFIX:-$SIMPLE_PIN_PREFIX_DEFAULT}"
PIN_SITE_PACKAGES="$SIMPLE_PIN_PREFIX/lib/python3.10/site-packages"
if [[ -d "$SIMPLE_PIN_PREFIX/lib" ]]; then
    export LD_LIBRARY_PATH="$SIMPLE_PIN_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ -d "$PIN_SITE_PACKAGES" ]]; then
    export PYTHONPATH="$PIN_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
fi

echo "Activated SIMPLE environment"
echo "  VIRTUAL_ENV=$VIRTUAL_ENV"
echo "  CUDA_HOME=$CUDA_HOME"
if [[ -d "$PIN_SITE_PACKAGES" ]]; then
    echo "  SIMPLE_PIN_PREFIX=$SIMPLE_PIN_PREFIX"
fi
