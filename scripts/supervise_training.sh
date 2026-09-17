#!/usr/bin/env bash
set -uo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: $0 RUN_DIR COMMAND [ARG ...]" >&2
  exit 2
fi

RUN_DIR="$1"
shift
mkdir -p "${RUN_DIR}"

LEARNER_PID=""
FORWARDED_SIGNAL=""
STARTED_AT="$(date --iso-8601=seconds)"

write_atomic() {
  local destination="$1"
  local value="$2"
  local temporary="${destination}.tmp.$$"
  printf '%s\n' "${value}" > "${temporary}"
  mv -f "${temporary}" "${destination}"
}

forward_signal() {
  local signal_name="$1"
  FORWARDED_SIGNAL="${signal_name}"
  if [[ -n "${LEARNER_PID}" ]] && kill -0 "${LEARNER_PID}" 2>/dev/null; then
    kill -"${signal_name}" "${LEARNER_PID}" 2>/dev/null || true
  fi
}

trap 'forward_signal TERM' TERM
trap 'forward_signal INT' INT
trap 'forward_signal HUP' HUP

write_atomic "${RUN_DIR}/supervisor.pid" "$$"
write_atomic "${RUN_DIR}/learner_started_at" "${STARTED_AT}"

"$@" &
LEARNER_PID=$!
write_atomic "${RUN_DIR}/train.pid" "${LEARNER_PID}"

EXIT_CODE=0
while true; do
  wait "${LEARNER_PID}"
  EXIT_CODE=$?
  if ! kill -0 "${LEARNER_PID}" 2>/dev/null; then
    break
  fi
done

FINISHED_AT="$(date --iso-8601=seconds)"
EXIT_REASON="exit"
if (( EXIT_CODE == 0 )); then
  EXIT_REASON="completed"
elif (( EXIT_CODE > 128 )); then
  SIGNAL_NUMBER=$((EXIT_CODE - 128))
  SIGNAL_NAME="$(kill -l "${SIGNAL_NUMBER}" 2>/dev/null || printf 'UNKNOWN')"
  EXIT_REASON="signal:${SIGNAL_NAME}"
fi
if [[ -n "${FORWARDED_SIGNAL}" ]]; then
  EXIT_REASON="forwarded-signal:${FORWARDED_SIGNAL};${EXIT_REASON}"
fi

write_atomic "${RUN_DIR}/learner_exit_code" "${EXIT_CODE}"
write_atomic "${RUN_DIR}/learner_finished_at" "${FINISHED_AT}"
write_atomic "${RUN_DIR}/learner_exit_reason" "${EXIT_REASON}"

exit "${EXIT_CODE}"
