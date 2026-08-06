#!/usr/bin/env bash
set -euo pipefail

VISIBLE_GPUS="${MPRL_VISIBLE_GPUS:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}"

PYTHON_BIN="${MPRL_PYTHON:-/home/jianquanzhao/anaconda24/envs/mprl-vgpt/bin/python}"
GPU_IDS="${MPRL_GPU_IDS:-0,1,2,3}"
TRAIN_FREQUENCY="${MPRL_TRAIN_FREQUENCY:-1}"
GRADIENT_STEPS="${MPRL_GRADIENT_STEPS:-1}"
REPLAY_WARMUP_SIZE="${MPRL_REPLAY_WARMUP_SIZE:-512}"
REPLAY_CAPACITY="${MPRL_REPLAY_CAPACITY:-50000}"
TARGET_SYNC_INTERVAL="${MPRL_TARGET_SYNC_INTERVAL:-500}"
CHECKPOINT_EVERY="${MPRL_CHECKPOINT_EVERY:-240}"
REPLAY_CHECKPOINT_EVERY="${MPRL_REPLAY_CHECKPOINT_EVERY:-2400}"
LOG_EVERY_STEPS="${MPRL_LOG_EVERY_STEPS:-25}"
RUN_LABEL="${MPRL_RUN_LABEL:-full_esm_terminal_multigpu_bs128_f${TRAIN_FREQUENCY}_g${GRADIENT_STEPS}}"
RUN_DIR="${MPRL_RUN_DIR:-/mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/${RUN_LABEL}_$(date +%Y%m%d_%H%M%S)}"

RESUME_ARGS=()
if [[ -n "${MPRL_RESUME_CHECKPOINT_DIR:-}" ]]; then
  RESUME_ARGS+=(--resume-checkpoint-dir "${MPRL_RESUME_CHECKPOINT_DIR}")
fi
if [[ -n "${MPRL_RESUME_NEXT_EPISODE:-}" ]]; then
  RESUME_ARGS+=(--resume-next-episode "${MPRL_RESUME_NEXT_EPISODE}")
fi

FINAL_REPLAY_ARGS=()
if [[ "${MPRL_SAVE_FINAL_REPLAY:-0}" != "1" ]]; then
  FINAL_REPLAY_ARGS+=(--no-save-final-replay)
fi

mkdir -p "${RUN_DIR}"

nohup env PYTHONUNBUFFERED=1 "${PYTHON_BIN}" training.py \
  --pdb-dir /mnt/nas/jianquanzhao/data/mprl/pdbs/cath \
  --train-index /mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/indices/train_index.txt \
  --val-index /mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/indices/val_index.txt \
  --output-dir "${RUN_DIR}" \
  --mode multi --gpu-ids "${GPU_IDS}" \
  --batch-size 128  \
  --train-frequency "${TRAIN_FREQUENCY}" --gradient-steps "${GRADIENT_STEPS}" \
  --epochs 64 --train-batch-size 8 --max-steps 24 \
  --observation-encoder esm2 --embedding-dim 1280 --esm2-device auto \
  --esm-model-dir /mnt/data1/home/jianquanzhao/data/models/esm \
  --no-minimize --continue-on-update-error \
  --replay-warmup-size "${REPLAY_WARMUP_SIZE}" --replay-capacity "${REPLAY_CAPACITY}" \
  --target-sync-interval "${TARGET_SYNC_INTERVAL}" \
  --checkpoint-every "${CHECKPOINT_EVERY}" \
  --replay-checkpoint-every "${REPLAY_CHECKPOINT_EVERY}" \
  --plot-every-episodes 240 --rolling-window 100 \
  --validate-every 240 --validation-episodes 16 --enable-tensorboard \
  --terminal-reward-artifact params/hbond_random_forest.joblib \
  --terminal-predicted-pdb-dir /mnt/nas/jianquanzhao/data/mprl/outputs/evaluate-mechanical-predictor/pdb-colab/CATH-test \
  --log-level INFO --log-every-steps "${LOG_EVERY_STEPS}" --no-resume-logs --no-save-candidates \
  "${RESUME_ARGS[@]}" "${FINAL_REPLAY_ARGS[@]}" \
  > "${RUN_DIR}/train_stdout.log" 2>&1 < /dev/null &

TRAIN_PID=$!
printf '%s\n' "${TRAIN_PID}" > "${RUN_DIR}/train.pid"

echo "Started 4-GPU DDQN training"
echo "PID: ${TRAIN_PID}"
echo "Run directory: ${RUN_DIR}"
echo "Log: ${RUN_DIR}/train_stdout.log"
echo "GPU ids: ${GPU_IDS}"
echo "Update schedule: train_frequency=${TRAIN_FREQUENCY}, gradient_steps=${GRADIENT_STEPS}"
echo "Replay warmup: ${REPLAY_WARMUP_SIZE}"
echo "Replay capacity: ${REPLAY_CAPACITY}"
echo "Target sync interval (optimizer steps): ${TARGET_SYNC_INTERVAL}"
echo "Agent checkpoint interval (episodes): ${CHECKPOINT_EVERY}"
echo "Replay checkpoint interval (episodes): ${REPLAY_CHECKPOINT_EVERY}"
if [[ -n "${MPRL_RESUME_CHECKPOINT_DIR:-}" ]]; then
  echo "Resuming from: ${MPRL_RESUME_CHECKPOINT_DIR}"
fi
