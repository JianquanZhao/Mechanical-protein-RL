#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

VISIBLE_GPUS="${MPRL_VISIBLE_GPUS:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}"

PYTHON_BIN="${MPRL_PYTHON:-/home/jianquanzhao/anaconda24/envs/mprl-vgpt/bin/python}"
ACTORS="${MPRL_ASYNC_ACTORS:-24}"
INFERENCE_GPUS="${MPRL_ASYNC_INFERENCE_GPUS:-1,2,3}"
INFERENCE_BATCH_SIZE="${MPRL_ASYNC_INFERENCE_BATCH_SIZE:-8}"
TRAIN_FREQUENCY="${MPRL_TRAIN_FREQUENCY:-8}"
GRADIENT_STEPS="${MPRL_GRADIENT_STEPS:-1}"
TERMINAL_REWARD_SCALE="${MPRL_TERMINAL_REWARD_SCALE:-8.0}"
STEP_REWARD_SCALE="${MPRL_STEP_REWARD_SCALE:-0.025}"
REPLAY_SAMPLING="${MPRL_REPLAY_SAMPLING:-prioritized}"
POSITIVE_SAMPLE_FRACTION="${MPRL_POSITIVE_SAMPLE_FRACTION:-0.25}"
POSITIVE_REPLAY_RESERVE_FRACTION="${MPRL_POSITIVE_REPLAY_RESERVE_FRACTION:-0.10}"
POSITIVE_REWARD_LOWER_BOUND="${MPRL_POSITIVE_REWARD_LOWER_BOUND:-${MPRL_POSITIVE_REWARD_THRESHOLD:--0.25}}"
N_STEP="${MPRL_N_STEP:-3}"
RUN_LABEL="${MPRL_RUN_LABEL:-async_a${ACTORS}_esm${INFERENCE_BATCH_SIZE}_f${TRAIN_FREQUENCY}_g${GRADIENT_STEPS}_${REPLAY_SAMPLING}_possample${POSITIVE_SAMPLE_FRACTION}_poslcb${POSITIVE_REWARD_LOWER_BOUND}_posreserve${POSITIVE_REPLAY_RESERVE_FRACTION}_n${N_STEP}_stepx${STEP_REWARD_SCALE}_terminalx${TERMINAL_REWARD_SCALE}}"
RUN_DIR="${MPRL_RUN_DIR:-/mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/${RUN_LABEL}_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${RUN_DIR}"

nohup "${PROJECT_ROOT}/scripts/supervise_training.sh" "${RUN_DIR}" \
  env PYTHONUNBUFFERED=1 "${PYTHON_BIN}" asynchronous_training.py \
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
  --prevent-revisit-positions --include-visited-mask-in-observation \
  --replay-warmup-size 4096 --replay-capacity 100000 \
  --replay-sampling "${REPLAY_SAMPLING}" \
  --priority-alpha 0.6 --priority-beta-start 0.4 --priority-beta-end 1.0 \
  --priority-beta-steps 1000000 --priority-epsilon 1e-6 \
  --positive-sample-fraction "${POSITIVE_SAMPLE_FRACTION}" \
  --positive-replay-reserve-fraction "${POSITIVE_REPLAY_RESERVE_FRACTION}" \
  --positive-reward-lower-bound "${POSITIVE_REWARD_LOWER_BOUND}" \
  --n-step "${N_STEP}" \
  --target-sync-interval 125 --checkpoint-every 240 --replay-checkpoint-every 0 \
  --plot-every-episodes 240 --rolling-window 100 \
  --validate-every 240 --validation-episodes 16 \
  --validation-seed 20260819 --validation-bootstrap-samples 1000 \
  --enable-tensorboard \
  --terminal-reward-artifact params/hbond_random_forest.joblib \
  --step-reward-scale "${STEP_REWARD_SCALE}" \
  --terminal-reward-scale "${TERMINAL_REWARD_SCALE}" \
  --log-level INFO --log-every-steps 100 --no-resume-logs --no-save-candidates \
  --no-save-final-replay \
  > "${RUN_DIR}/train_stdout.log" 2>&1 < /dev/null &

SUPERVISOR_PID=$!
for _ in $(seq 1 50); do
  if [[ -s "${RUN_DIR}/train.pid" ]]; then
    break
  fi
  sleep 0.1
done
if [[ ! -s "${RUN_DIR}/train.pid" ]]; then
  echo "Supervisor failed to record learner PID; inspect ${RUN_DIR}/train_stdout.log" >&2
  exit 1
fi
TRAIN_PID="$(<"${RUN_DIR}/train.pid")"

echo "Started asynchronous DDQN training"
echo "Learner PID: ${TRAIN_PID}"
echo "Supervisor PID: ${SUPERVISOR_PID}"
echo "Run directory: ${RUN_DIR}"
echo "Exit status files: learner_exit_code, learner_exit_reason, learner_finished_at"
echo "Learner: cuda:0"
echo "ESM2 inference GPUs: ${INFERENCE_GPUS}"
echo "PyRosetta actors: ${ACTORS}"
echo "Terminal reward: 0.5*delta_z_strength + 0.5*delta_z_toughness"
echo "Terminal reward scale: ${TERMINAL_REWARD_SCALE}"
echo "Step shaping: zero-centered, scale=${STEP_REWARD_SCALE}"
echo "Replay sampling: ${REPLAY_SAMPLING}"
echo "Positive sample fraction: ${POSITIVE_SAMPLE_FRACTION}"
echo "Positive replay reserve fraction: ${POSITIVE_REPLAY_RESERVE_FRACTION}"
echo "Positive reward lower bound: ${POSITIVE_REWARD_LOWER_BOUND}"
echo "N-step return: ${N_STEP}"
echo "Revisit prevention: enabled and included in observation"
