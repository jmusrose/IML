from pathlib import Path


def test_av_v4_ave_loss_sweep_script_contains_requested_runs():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_av_v4_ave_loss_sweep.sh"
    )
    source = script_path.read_text(encoding="utf-8")

    assert "set -euo pipefail" in source
    assert source.count("--epochs 120") == 1
    assert source.count("run_experiment ") == 4
    assert 'run_experiment "av3" 3 3' in source
    assert 'run_experiment "av5" 5 5' in source
    assert 'run_experiment "v5" 1 5' in source
    assert 'run_experiment "v3" 1 3' in source

    for output_dir in (
        "runs/av_v4_ave_av3",
        "runs/av_v4_ave_av5",
        "runs/av_v4_ave_v5",
        "runs/av_v4_ave_v3",
    ):
        assert output_dir in source

    assert 'tee "${output_dir}/train.log"' in source
