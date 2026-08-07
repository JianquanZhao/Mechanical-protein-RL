#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MPRL_TRAIN_FREQUENCY="${MPRL_TRAIN_FREQUENCY:-4}"
export MPRL_GRADIENT_STEPS="${MPRL_GRADIENT_STEPS:-1}"
export MPRL_REPLAY_WARMUP_SIZE="${MPRL_REPLAY_WARMUP_SIZE:-4096}"
export MPRL_TARGET_SYNC_INTERVAL="${MPRL_TARGET_SYNC_INTERVAL:-125}"
export MPRL_RUN_LABEL="${MPRL_RUN_LABEL:-update_frequency_f4_g1_bs128_h24}"

exec bash "${SCRIPT_DIR}/train_version_terminal_4gpu.sh"
