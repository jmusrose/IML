#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
cd "${project_root}"

python_bin="${PYTHON_BIN:-python}"

run_experiment() {
    local name="$1"
    local audio_weight="$2"
    local visual_weight="$3"
    local output_dir="$4"

    mkdir -p "${output_dir}"
    echo "Starting ${name}: audio_loss_weight=${audio_weight}, visual_loss_weight=${visual_weight}"

    "${python_bin}" -u -m AV_v4.train_ave \
        --epochs 120 \
        --audio-loss-weight "${audio_weight}" \
        --visual-loss-weight "${visual_weight}" \
        --output-dir "${output_dir}" \
        2>&1 | tee "${output_dir}/train.log"
}

run_experiment "av3" 3 3 "runs/av_v4_ave_av3"
run_experiment "av5" 5 5 "runs/av_v4_ave_av5"
run_experiment "v5" 1 5 "runs/av_v4_ave_v5"
run_experiment "v3" 1 3 "runs/av_v4_ave_v3"

echo "All AV_v4 AVE loss-weight experiments finished."
