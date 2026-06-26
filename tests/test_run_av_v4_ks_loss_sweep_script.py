from pathlib import Path


def test_av_v4_ks_loss_sweep_script_contains_requested_runs():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_av_v4_ks_loss_sweep.sh"
    )
    source = script_path.read_text(encoding="utf-8")

    assert "set -euo pipefail" in source
    assert source.count("run_experiment ") == 2
    assert 'run_experiment "v1" 1 "runs/av_v4_ks_v1"' in source
    assert 'run_experiment "v5" 5 "runs/av_v4_ks_v5"' in source
    assert "--visual-loss-weight" in source
    assert "--audio-loss-weight" not in source
    assert "--epochs" not in source
    assert "--batch-size" not in source
    assert "--lr " not in source
    assert 'tee "${output_dir}/train.log"' in source
