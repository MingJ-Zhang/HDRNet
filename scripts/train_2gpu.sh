#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/hdrnet_segformer_b2_floodnet.py}
GPUS=${GPUS:-2}
PORT=${PORT:-29500}

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

mkdir -p logs
LOG_FILE="logs/train_$(basename ${CONFIG%.py})_$(date +%Y%m%d_%H%M%S).log"

echo "[INFO] CONFIG=${CONFIG}"
echo "[INFO] GPUS=${GPUS}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] Log file: ${LOG_FILE}"

PORT="${PORT}" bash tools/dist_train.sh "${CONFIG}" "${GPUS}" 2>&1 | tee "${LOG_FILE}"
