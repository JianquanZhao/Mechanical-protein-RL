#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash resume_train_update_frequency_f4.sh CHECKPOINT_DIR [NEXT_EPISODE]" >&2
  exit 2
fi

export MPRL_RESUME_CHECKPOINT_DIR="$1"
if [[ $# -eq 2 ]]; then
  export MPRL_RESUME_NEXT_EPISODE="$2"
fi
export MPRL_RUN_LABEL="resume_update_frequency_f4_g1_bs128_h24"

exec bash "${SCRIPT_DIR}/train_update_frequency_f4.sh"
