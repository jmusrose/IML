#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
cd "${project_root}"

python_bin="${PYTHON_BIN:-python}"

run_experiment() {
    local name="$1"
    local batch_size="$2"
    local lr="$3"
    local visual_weight="$4"
    local output_dir="$5"
    local timestamp
    local log_path

    timestamp="$(date +"%Y%m%d-%H%M%S")"
    mkdir -p "${output_dir}/logs"
    log_path="${output_dir}/logs/${name}_${timestamp}.log"

    echo "Starting ${name}: batch_size=${batch_size}, lr=${lr}, visual_loss_weight=${visual_weight}"
    echo "Log: ${log_path}"

    "${python_bin}" -u -m AV_v4.train_ks \
        --batch-size "${batch_size}" \
        --lr "${lr}" \
        --audio-loss-weight 1 \
        --visual-loss-weight "${visual_weight}" \
        --epochs 120 \
        --lr-scheduler multistep \
        --lr-decay-step "[80]" \
        --lr-decay-ratio 0.1 \
        --momentum 0.9 \
        --weight-decay 1e-4 \
        --use-video-frames 3 \
        --image-size 224 \
        --audio-duration 5.0 \
        --n-fft 256 \
        --hop-length 128 \
        --win-length 256 \
        --output-dir "${output_dir}" \
        2>&1 | tee "${log_path}"
}

run_experiment "stable_b20_lr0015_v1" "20" "0.0015" "1" "runs/av_v4_ks_best2/stable_b20_lr0015_v1"
run_experiment "visual_b20_lr002_v5" "20" "0.002" "5" "runs/av_v4_ks_best2/visual_b20_lr002_v5"

echo "All AV_v4 KineticSound best-2 experiments finished."
