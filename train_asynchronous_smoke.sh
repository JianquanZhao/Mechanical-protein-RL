#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${MPRL_VISIBLE_GPUS:-0,1}"
PYTHON_BIN="${MPRL_PYTHON:-/home/jianquanzhao/anaconda24/envs/mprl-vgpt/bin/python}"
RUN_DIR="${MPRL_RUN_DIR:-/tmp/mprl-async-smoke}"

"${PYTHON_BIN}" asynchronous_training.py \
  --mode asynchronous --device cuda:0 \
  --async-actors 2 --async-inference-gpu-ids 1 \
  --async-inference-batch-size 2 --async-inference-batch-wait-ms 20 \
  --pdb-dir "${MPRL_PDB_DIR:-/mnt/nas/jianquanzhao/data/mprl/pdbs/cath}" \
  --train-index "${MPRL_TRAIN_INDEX:-/mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/indices/train_index.txt}" \
  --val-index "${MPRL_VAL_INDEX:-/mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/indices/val_index.txt}" \
  --output-dir "${RUN_DIR}" --episodes 4 --max-steps 2 \
  --batch-size 4 --replay-warmup-size 4 --replay-capacity 32 \
  --train-frequency 4 --gradient-steps 1 --target-sync-interval 4 \
  --observation-encoder esm2 --embedding-dim 1280 \
  --esm-model-dir /mnt/data1/home/jianquanzhao/data/models/esm \
  --no-minimize --continue-on-update-error --no-terminal-reward \
  --checkpoint-every 0 --replay-checkpoint-every 0 \
  --validate-every 0 --validation-episodes 0 --plot-every-episodes 100 \
  --log-level INFO --log-every-steps 1 --no-save-candidates --no-save-final-replay
