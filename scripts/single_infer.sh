#!/usr/bin/env bash
set -euo pipefail

IMG=${1:?Usage: bash scripts/single_infer.sh IMG CONFIG CHECKPOINT OUT_DIR}
CONFIG=${2:?Usage: bash scripts/single_infer.sh IMG CONFIG CHECKPOINT OUT_DIR}
CHECKPOINT=${3:?Usage: bash scripts/single_infer.sh IMG CONFIG CHECKPOINT OUT_DIR}
OUT_DIR=${4:-outputs}

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
mkdir -p "${OUT_DIR}"
python demo/image_demo.py "${IMG}" "${CONFIG}" --checkpoint "${CHECKPOINT}" --out-file "${OUT_DIR}/$(basename ${IMG})"
