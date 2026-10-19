#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${SFT_PYTHON:-/home/jianquanzhao/anaconda24/envs/mprl-vgpt/bin/python}"
CONFIG="${SFT_CONFIG:-${PROJECT_ROOT}/sft_data_generation/configs/base.yaml}"
SFT_ROOT="${SFT_ROOT:-/mnt/nas/jianquanzhao/data/mprl/SFT}"
ACTIONS="${SFT_ACTIONS:-${SFT_ROOT}/data/generated/sft_v001/rounds/round_000/depth_000/labels.csv}"
RELEASE_NAME="${SFT_RELEASE_NAME:-sft_actions_round0_base}"
RELEASE_DIR="${SFT_RELEASE_DIR:-${SFT_ROOT}/data/generated/sft_v001/release/${RELEASE_NAME}}"
ESM_MODEL_DIR="${SFT_ESM_MODEL_DIR:-/mnt/data1/home/jianquanzhao/data/models/esm}"
EMBEDDING_DIM="${SFT_EMBEDDING_DIM:-1280}"
EMBEDDING_DTYPE="${SFT_EMBEDDING_DTYPE:-float16}"
EMBEDDING_BATCH_SIZE="${SFT_EMBEDDING_BATCH_SIZE:-4}"
EMBEDDING_DEVICE="${SFT_EMBEDDING_DEVICE:-cuda:0}"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m sft_data_generation.cli --config "${CONFIG}" release-compact \
  --actions "${ACTIONS}" \
  --release-name "${RELEASE_NAME}" \
  --output-dir "${RELEASE_DIR}"

"${PYTHON_BIN}" -m sft_data_generation.cli --config "${CONFIG}" cache-embeddings \
  --release-dir "${RELEASE_DIR}" \
  --embedding-dim "${EMBEDDING_DIM}" \
  --dtype "${EMBEDDING_DTYPE}" \
  --batch-size "${EMBEDDING_BATCH_SIZE}" \
  --device "${EMBEDDING_DEVICE}" \
  --model-dir "${ESM_MODEL_DIR}"

echo "Compact SFT release ready: ${RELEASE_DIR}"
