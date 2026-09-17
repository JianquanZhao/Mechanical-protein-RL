#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${SFT_PYTHON:-/home/jianquanzhao/anaconda24/envs/mprl-vgpt/bin/python}"
CONFIG="${SFT_CONFIG:-${PROJECT_ROOT}/sft_data_generation/configs/base.yaml}"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m sft_data_generation.cli --config "${CONFIG}" init-layout
"${PYTHON_BIN}" -m sft_data_generation.cli --config "${CONFIG}" prepare-manifest "$@"

