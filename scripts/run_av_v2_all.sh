#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/e/anaconda3/envs/pytorch2.5/python.exe}"
DEVICE="${DEVICE:-cuda}"
ENABLE_FGM="${ENABLE_FGM:-1}"
RUN_ROOT="${RUN_ROOT:-runs/av_v2_parallel}"
LOG_ROOT="${LOG_ROOT:-${RUN_ROOT}/logs}"

mkdir -p "${LOG_ROOT}"

FGM_ARGS=()
RUN_SUFFIX="baseline"
if [[ "${ENABLE_FGM}" == "1" ]]; then
  FGM_ARGS=(--fgm)
  RUN_SUFFIX="fgm"
else
  FGM_ARGS=(--no-fgm)
fi

PIDS=()
NAMES=()

start_run() {
  local dataset="$1"
  local output_dir="${RUN_ROOT}/${dataset}_${RUN_SUFFIX}"
  local log_file="${LOG_ROOT}/${dataset}_${RUN_SUFFIX}.log"

  mkdir -p "${output_dir}"
  echo "Starting ${dataset}: ${log_file}"
  "${PYTHON_BIN}" -m AV_v2.train_video \
    --dataset "${dataset}" \
    --modality av \
    --device "${DEVICE}" \
    --output-dir "${output_dir}" \
    --no-progress \
    "${FGM_ARGS[@]}" \
    "$@" > "${log_file}" 2>&1 &

  PIDS+=("$!")
  NAMES+=("${dataset}")
}

stop_children() {
  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}
trap stop_children INT TERM

start_run "cremad" "$@"
start_run "ks" "$@"
start_run "ave" "$@"

FAILED=0
for index in "${!PIDS[@]}"; do
  pid="${PIDS[$index]}"
  name="${NAMES[$index]}"
  if wait "${pid}"; then
    echo "${name} finished successfully."
  else
    status="$?"
    echo "${name} failed with exit code ${status}. See ${LOG_ROOT}/${name}_${RUN_SUFFIX}.log"
    FAILED=1
  fi
done

if [[ "${FAILED}" -ne 0 ]]; then
  exit 1
fi

echo "All AV_v2 runs finished. Outputs: ${RUN_ROOT}"
