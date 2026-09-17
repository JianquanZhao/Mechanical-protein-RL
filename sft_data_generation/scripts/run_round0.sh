#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${SFT_PYTHON:-/home/jianquanzhao/anaconda24/envs/mprl-vgpt/bin/python}"
CONFIG="${SFT_CONFIG:-${PROJECT_ROOT}/sft_data_generation/configs/base.yaml}"
SHARDS="${SFT_SHARDS:-64}"
WORKERS="${SFT_WORKERS:-8}"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m sft_data_generation.cli --config "${CONFIG}" build-tasks \
  --round 0 --depth 0 --split train

export PROJECT_ROOT PYTHON_BIN CONFIG SHARDS
seq 0 "$((SHARDS - 1))" | xargs -P "${WORKERS}" -I {} bash -c \
  'cd "$PROJECT_ROOT" && "$PYTHON_BIN" -m sft_data_generation.cli --config "$CONFIG" scan-shard --round 0 --depth 0 --shard-index "$1" --num-shards "$SHARDS"' _ {}

"${PYTHON_BIN}" -m sft_data_generation.cli --config "${CONFIG}" merge-shards --round 0 --depth 0
"${PYTHON_BIN}" -m sft_data_generation.cli --config "${CONFIG}" aggregate --round 0 --depth 0

