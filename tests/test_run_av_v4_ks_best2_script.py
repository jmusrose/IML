from pathlib import Path


def test_run_av_v4_ks_best2_script_contains_two_recommended_configs():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_av_v4_ks_best2.sh"
    source = script.read_text(encoding="utf-8")

    assert "set -euo pipefail" in source
    assert source.count('run_experiment "') == 2
    assert 'run_experiment "stable_b20_lr0015_v1" "20" "0.0015" "1" "runs/av_v4_ks_best2/stable_b20_lr0015_v1"' in source
    assert 'run_experiment "visual_b20_lr002_v5" "20" "0.002" "5" "runs/av_v4_ks_best2/visual_b20_lr002_v5"' in source

    for flag in (
        "--batch-size",
        "--lr",
        "--visual-loss-weight",
        "--audio-loss-weight 1",
        "--epochs 100",
        "--lr-decay-step \"[70]\"",
        "--weight-decay 1e-4",
        "--use-video-frames 3",
        "--audio-duration 5.0",
    ):
        assert flag in source

    assert "logs/${name}_${timestamp}.log" in source
    assert 'tee "${log_path}"' in source
