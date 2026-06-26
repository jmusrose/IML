#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
cd "${project_root}"

python_bin="${PYTHON_BIN:-python}"

run_experiment() {
    local name="$1"
    local visual_weight="$2"
    local output_dir="$3"

    mkdir -p "${output_dir}"
    echo "Starting ${name}: visual_loss_weight=${visual_weight}"

    "${python_bin}" -u -m AV_v4.train_ks \
        --visual-loss-weight "${visual_weight}" \
        --output-dir "${output_dir}" \
        2>&1 | tee "${output_dir}/train.log"
}

run_experiment "v1" 1 "runs/av_v4_ks_v1"
run_experiment "v5" 5 "runs/av_v4_ks_v5"

echo "All AV_v4 KineticSound loss-weight experiments finished."
