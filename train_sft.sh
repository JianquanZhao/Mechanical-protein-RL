#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

SFT_ROOT="${SFT_ROOT:-/mnt/nas/jianquanzhao/data/mprl/SFT}"
PYTHON_BIN="${SFT_PYTHON:-/home/jianquanzhao/anaconda24/envs/mprl-vgpt/bin/python}"
DATA_DIR="${SFT_DATA_DIR:-${SFT_ROOT}/data/generated/sft_v001/release/sft_actions_round0_base}"
MODEL_ROOT="${SFT_MODEL_ROOT:-${SFT_ROOT}/model}"
LOG_ROOT="${SFT_LOG_ROOT:-${SFT_ROOT}/log}"
TENSORBOARD_ROOT="${SFT_TENSORBOARD_ROOT:-${SFT_ROOT}/tensorboard}"
ESM_MODEL_DIR="${SFT_ESM_MODEL_DIR:-/mnt/data1/home/jianquanzhao/data/models/esm}"

RUN_VERSION="${SFT_RUN_VERSION:-sft_qhead_v001}"
RUN_ID="${SFT_RUN_ID:-${RUN_VERSION}_$(date +%Y%m%d_%H%M%S)}"
VISIBLE_GPUS="${SFT_VISIBLE_GPUS:-0}"
export CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}"
DEVICE="${SFT_DEVICE:-cuda:0}"
EMBEDDING_DIM="${SFT_EMBEDDING_DIM:-1280}"
HIDDEN_DIMS="${SFT_HIDDEN_DIMS:-256,256}"
EPOCHS="${SFT_EPOCHS:-50}"
STATE_BATCH_SIZE="${SFT_STATE_BATCH_SIZE:-4}"
LEARNING_RATE="${SFT_LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${SFT_WEIGHT_DECAY:-1e-5}"
WARMUP_RATIO="${SFT_WARMUP_RATIO:-0.05}"
NUM_WORKERS="${SFT_NUM_WORKERS:-0}"
EXPECTED_TORCH_PREFIX="${SFT_EXPECTED_TORCH_PREFIX:-2.12.0}"
EXPECTED_TORCH_CUDA="${SFT_EXPECTED_TORCH_CUDA:-13.0}"
BACKGROUND="${SFT_BACKGROUND:-1}"

RUN_LOG_DIR="${LOG_ROOT}/${RUN_ID}"
STDOUT_LOG="${RUN_LOG_DIR}/train_stdout.log"
PID_FILE="${RUN_LOG_DIR}/train.pid"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "SFT Python is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${DATA_DIR}/states.csv" || ! -f "${DATA_DIR}/dataset_version.json" ]]; then
  echo "SFT release is incomplete: ${DATA_DIR}" >&2
  echo "Expected states.csv and dataset_version.json. Build the compact release first." >&2
  exit 1
fi

export SFT_PREFLIGHT_DEVICE="${DEVICE}"
export SFT_PREFLIGHT_EMBEDDING_DIM="${EMBEDDING_DIM}"
export SFT_PREFLIGHT_HIDDEN_DIMS="${HIDDEN_DIMS}"
export SFT_PREFLIGHT_EXPECTED_TORCH="${EXPECTED_TORCH_PREFIX}"
export SFT_PREFLIGHT_EXPECTED_CUDA="${EXPECTED_TORCH_CUDA}"
export SFT_PREFLIGHT_DATA_DIR="${DATA_DIR}"
"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path
import torch

from model.sft_module import validate_sft_ddqn_q_network_compatibility

device = os.environ["SFT_PREFLIGHT_DEVICE"]
expected_torch = os.environ["SFT_PREFLIGHT_EXPECTED_TORCH"]
expected_cuda = os.environ["SFT_PREFLIGHT_EXPECTED_CUDA"]
embedding_dim = int(os.environ["SFT_PREFLIGHT_EMBEDDING_DIM"])
hidden_dims = tuple(int(value) for value in os.environ["SFT_PREFLIGHT_HIDDEN_DIMS"].split(","))
data_dir = Path(os.environ["SFT_PREFLIGHT_DATA_DIR"])
version = json.loads((data_dir / "dataset_version.json").read_text(encoding="utf-8"))
if version.get("format_version") != "compact_state_v1":
    raise RuntimeError(
        f"Expected compact_state_v1 release, found {version.get('format_version')!r}."
    )
embedding_cache = version.get("embedding_cache", {})
if not embedding_cache.get("complete", False):
    raise RuntimeError("SFT release embedding cache is incomplete.")
if int(embedding_cache.get("embedding_dim", -1)) != embedding_dim:
    raise RuntimeError(
        "Cached embedding dimension does not match SFT_EMBEDDING_DIM: "
        f"{embedding_cache.get('embedding_dim')} != {embedding_dim}."
    )

if not torch.__version__.startswith(expected_torch):
    raise RuntimeError(f"Expected torch {expected_torch}.*, found {torch.__version__}.")
if torch.version.cuda != expected_cuda:
    raise RuntimeError(f"Expected PyTorch CUDA {expected_cuda}, found {torch.version.cuda}.")
if device.startswith("cuda"):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")
    device_index = torch.device(device).index or 0
    if device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested {device}, but only {torch.cuda.device_count()} CUDA devices are visible."
        )
    probe = torch.ones(1, device=device)
    if not torch.isfinite(probe).all():
        raise RuntimeError("CUDA tensor smoke test produced a non-finite value.")

contract = validate_sft_ddqn_q_network_compatibility(
    embedding_dim=embedding_dim,
    hidden_dims=hidden_dims,
    include_visited_mask=True,
)
print(
    "SFT preflight passed: "
    f"torch={torch.__version__} cuda={torch.version.cuda} device={device} "
    f"layers={contract['layer_dimensions']} parameters={contract['trainable_parameter_count']}"
)
PY

mkdir -p "${MODEL_ROOT}" "${RUN_LOG_DIR}" "${TENSORBOARD_ROOT}/${RUN_ID}"

COMMAND=(
  env PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
  "${PYTHON_BIN}" train_sft.py
  --data-dir "${DATA_DIR}"
  --run-version "${RUN_VERSION}"
  --run-id "${RUN_ID}"
  --model-root "${MODEL_ROOT}"
  --log-root "${LOG_ROOT}"
  --tensorboard-root "${TENSORBOARD_ROOT}"
  --esm-model-dir "${ESM_MODEL_DIR}"
  --embedding-dim "${EMBEDDING_DIM}"
  --hidden-dims "${HIDDEN_DIMS}"
  --include-visited-mask
  --require-cached-embeddings
  --epochs "${EPOCHS}"
  --state-batch-size "${STATE_BATCH_SIZE}"
  --learning-rate "${LEARNING_RATE}"
  --weight-decay "${WEIGHT_DECAY}"
  --warmup-ratio "${WARMUP_RATIO}"
  --num-workers "${NUM_WORKERS}"
  --device "${DEVICE}"
  --enable-tensorboard
)

if [[ "${BACKGROUND}" == "1" ]]; then
  nohup "${COMMAND[@]}" "$@" >"${STDOUT_LOG}" 2>&1 < /dev/null &
  TRAIN_PID=$!
  printf '%s\n' "${TRAIN_PID}" >"${PID_FILE}"
  echo "Started SFT training"
  echo "PID: ${TRAIN_PID}"
  echo "Run ID: ${RUN_ID}"
  echo "Visible physical GPUs: ${VISIBLE_GPUS}"
  echo "Training device: ${DEVICE}"
  echo "Dataset: ${DATA_DIR}"
  echo "Model: ${MODEL_ROOT}/${RUN_ID}"
  echo "Log: ${RUN_LOG_DIR}"
  echo "TensorBoard: ${TENSORBOARD_ROOT}/${RUN_ID}"
else
  "${COMMAND[@]}" "$@" 2>&1 | tee "${STDOUT_LOG}"
fi
