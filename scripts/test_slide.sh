#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?Usage: bash scripts/test_slide.sh CONFIG CHECKPOINT [GPUS]}
CHECKPOINT=${2:?Usage: bash scripts/test_slide.sh CONFIG CHECKPOINT [GPUS]}
GPUS=${3:-2}
PORT=${PORT:-29501}

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

PORT="${PORT}" bash tools/dist_test.sh "${CONFIG}" "${CHECKPOINT}" "${GPUS}"
