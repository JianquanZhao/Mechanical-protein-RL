#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${SFT_TRAIN_PYTHON:-/home/jianquanzhao/anaconda24/envs/mprl-vgpt/bin/python}"
DATA_DIR="${SFT_RELEASE_DIR:-/mnt/nas/jianquanzhao/data/mprl/SFT/data/generated/sft_v001/release/sft_actions_v001}"
ESM_MODEL_DIR="${SFT_ESM_MODEL_DIR:-/mnt/data1/home/jianquanzhao/data/models/esm}"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" train_sft.py \
  --data-dir "${DATA_DIR}" \
  --esm-model-dir "${ESM_MODEL_DIR}" \
  --embedding-dim 1280 \
  --hidden-dims 256,256 \
  --epochs 50 \
  --state-batch-size 4 \
  --learning-rate 1e-4 \
  --warmup-ratio 0.05 \
  --enable-tensorboard "$@"

