#!/usr/bin/env bash
set -euo pipefail

VISIBLE_GPUS="${MPRL_VISIBLE_GPUS:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}"

PYTHON_BIN="${MPRL_PYTHON:-/home/jianquanzhao/anaconda24/envs/mprl-vgpt/bin/python}"
ACTORS="${MPRL_ASYNC_ACTORS:-12}"
INFERENCE_GPUS="${MPRL_ASYNC_INFERENCE_GPUS:-1,2,3}"
INFERENCE_BATCH_SIZE="${MPRL_ASYNC_INFERENCE_BATCH_SIZE:-8}"
TRAIN_FREQUENCY="${MPRL_TRAIN_FREQUENCY:-8}"
GRADIENT_STEPS="${MPRL_GRADIENT_STEPS:-1}"
RUN_LABEL="${MPRL_RUN_LABEL:-async_a${ACTORS}_esm${INFERENCE_BATCH_SIZE}_f${TRAIN_FREQUENCY}_g${GRADIENT_STEPS}}"
RUN_DIR="${MPRL_RUN_DIR:-/mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/${RUN_LABEL}_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${RUN_DIR}"

nohup env PYTHONUNBUFFERED=1 "${PYTHON_BIN}" asynchronous_training.py \
  --mode asynchronous --device cuda:0 \
  --async-actors "${ACTORS}" \
  --async-inference-gpu-ids "${INFERENCE_GPUS}" \
  --async-inference-batch-size "${INFERENCE_BATCH_SIZE}" \
  --async-inference-batch-wait-ms 10 --async-queue-size 64 \
  --async-policy-sync-interval 100 --async-start-method spawn \
  --pdb-dir /mnt/nas/jianquanzhao/data/mprl/pdbs/cath \
  --train-index /mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/indices/train_index.txt \
  --val-index /mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/indices/val_index.txt \
  --output-dir "${RUN_DIR}" \
  --batch-size 128 --train-frequency "${TRAIN_FREQUENCY}" --gradient-steps "${GRADIENT_STEPS}" \
  --epochs 64 --train-batch-size 8 --max-steps 24 \
  --observation-encoder esm2 --embedding-dim 1280 \
  --esm-model-dir /mnt/data1/home/jianquanzhao/data/models/esm \
  --no-minimize --continue-on-update-error \
  --replay-warmup-size 4096 --replay-capacity 50000 \
  --target-sync-interval 125 --checkpoint-every 240 --replay-checkpoint-every 0 \
  --plot-every-episodes 240 --rolling-window 100 \
  --validate-every 0 --validation-episodes 0 --enable-tensorboard \
  --terminal-reward-artifact params/hbond_random_forest.joblib \
  --terminal-predicted-pdb-dir /mnt/nas/jianquanzhao/data/mprl/outputs/evaluate-mechanical-predictor/pdb-colab/CATH-test \
  --log-level INFO --log-every-steps 100 --no-resume-logs --no-save-candidates \
  --no-save-final-replay \
  > "${RUN_DIR}/train_stdout.log" 2>&1 < /dev/null &

TRAIN_PID=$!
printf '%s\n' "${TRAIN_PID}" > "${RUN_DIR}/train.pid"

echo "Started asynchronous DDQN training"
echo "PID: ${TRAIN_PID}"
echo "Run directory: ${RUN_DIR}"
echo "Learner: cuda:0"
echo "ESM2 inference GPUs: ${INFERENCE_GPUS}"
echo "PyRosetta actors: ${ACTORS}"
