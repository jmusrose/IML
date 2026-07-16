import math
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


def test_cmi_matches_closed_form_and_is_symmetric():
    from FGM_toy.data import Hb, cmi_A_given_B, cmi_B_given_A

    for s in (0.0, 0.5, 1.0):
        for eta in (0.0, 0.1, 0.3):
            want = s * (1.0 - Hb(eta))
            assert abs(cmi_A_given_B(s, eta) - want) < 1e-6
            assert abs(cmi_B_given_A(s, eta) - want) < 1e-6


def test_dataset_shapes_and_theoretical_metrics():
    from FGM_toy.data import make_dataset, theoretical_joint_acc, theoretical_single_acc

    ds = make_dataset(n=32, s=0.4, eta=0.1, seed=7)

    sample = ds[0]
    assert set(sample) == {"audio", "visual", "label", "zA", "zB"}
    assert tuple(sample["audio"].shape) == (16,)
    assert tuple(sample["visual"].shape) == (16,)
    assert sample["label"].dtype == torch.long
    assert math.isclose(theoretical_single_acc(0.4), 0.8)
    assert math.isclose(theoretical_joint_acc(0.4, 0.1), 0.96)


def test_model_exposes_fgm_compatible_outputs():
    from FGM_toy.model import ToyAVModel

    model = ToyAVModel()
    audio = torch.randn(4, 16)
    visual = torch.randn(4, 16)
    outputs = model.forward_with_modal_logits(audio, visual)

    assert tuple(outputs["logits"].shape) == (4, 2)
    assert tuple(outputs["audio_logits"].shape) == (4, 2)
    assert tuple(outputs["visual_logits"].shape) == (4, 2)
    assert tuple(outputs["audio_feature"].shape) == (4, 32)
    assert tuple(outputs["visual_feature"].shape) == (4, 32)
    assert model.classifier[0].in_features == 64


def test_train_one_epoch_records_fgm_metrics():
    from FGM_toy.data import create_loaders
    from FGM_toy.model import ToyAVModel
    from FGM_toy.train import TrainConfig, train_one_epoch
    from cmi_fgm import CMIFGMState

    train_loader, _ = create_loaders(n_train=64, n_val=32, s=0.5, eta=0.0, batch_size=16, seed=1)
    model = ToyAVModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    state = CMIFGMState(("audio", "visual"), strength=0.5, warmup_steps=0, momentum=0.0)
    metrics = train_one_epoch(
        model,
        train_loader,
        opt,
        torch.device("cpu"),
        TrainConfig(mode="fgm"),
        fgm_state=state,
    )

    assert metrics["loss"] > 0.0
    assert metrics["delta_audio"] >= 0.0
    assert metrics["delta_visual"] >= 0.0
    assert metrics["fgm_coef_audio"] >= 1.0
    assert metrics["fgm_coef_visual"] >= 1.0


def test_acc_baseline_signal_tracks_probe_correctness():
    from FGM_toy.train import _signal_for_mode

    labels = torch.tensor([0, 1])
    fusion_loss = torch.tensor([0.2, 0.2])
    audio_loss = torch.tensor([0.1, 0.1])
    visual_loss = torch.tensor([1.0, 1.0])
    audio_logits = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
    visual_logits = torch.tensor([[0.0, 5.0], [5.0, 0.0]])

    signal = _signal_for_mode(
        "acc_baseline",
        fusion_loss,
        audio_loss,
        visual_loss,
        audio_logits,
        visual_logits,
        labels,
    )

    assert torch.equal(signal[:, 0], torch.ones(2))
    assert torch.equal(signal[:, 1], torch.zeros(2))


def test_train_script_runs_when_called_by_file_path():
    result = subprocess.run(
        [
            sys.executable,
            "FGM_toy/train.py",
            "--mode",
            "fgm",
            "--s",
            "0.5",
            "--eta",
            "0.1",
            "--n-train",
            "16",
            "--n-val",
            "8",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--fgm-warmup-epochs",
            "0",
        ],
        cwd=".",
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert '"true_cmi"' in result.stdout


def test_signal_fidelity_grid_writes_records_and_plots():
    from FGM_toy.run_grid import SignalFidelityConfig, run_signal_fidelity_grid

    with TemporaryDirectory() as tmpdir:
        result = run_signal_fidelity_grid(
            SignalFidelityConfig(
                output_dir=Path(tmpdir),
                s_values=(0.2,),
                eta_values=(0.0, 0.1),
                seeds=(0,),
                n_train=16,
                n_val=8,
                epochs=1,
                batch_size=8,
                fgm_warmup_epochs=0,
            )
        )

        assert len(result["fgm"]) == 2
        assert len(result["no_fgm"]) == 2
        assert (Path(tmpdir) / "signal_fidelity_fgm.json").exists()
        assert (Path(tmpdir) / "signal_fidelity_no_fgm.json").exists()
        assert (Path(tmpdir) / "signal_fidelity_fgm.png").exists()
        assert (Path(tmpdir) / "signal_fidelity_no_fgm.png").exists()
        assert (Path(tmpdir) / "behavior_lines.png").exists()
        behavior = (Path(tmpdir) / "behavior_lines.json").read_text(encoding="utf-8")
        assert "acc_baseline" in behavior
        assert "strength_signal" not in behavior


def test_uniform32_grid_has_32_well_spread_points():
    from FGM_toy.run_grid import build_signal_fidelity_config

    config = build_signal_fidelity_config(
        [
            "--preset",
            "uniform32",
            "--output-dir",
            "unused",
            "--n-train",
            "16",
            "--n-val",
            "8",
            "--epochs",
            "1",
        ]
    )

    pairs = config.points
    true_cmis = [point.true_cmi for point in pairs]
    etas = [point.eta for point in pairs]
    assert len(pairs) == 32
    assert min(true_cmis) >= 0.12
    assert max(true_cmis) >= 0.9
    assert max(etas) <= 0.1


def test_plot_converts_delta_from_nats_to_bits():
    from FGM_toy.plot import delta_bits

    assert math.isclose(delta_bits(math.log(2.0)), 1.0)


def test_loss_gap_baseline_uses_own_modality_loss_gap():
    from FGM_toy.train import _signal_for_mode

    labels = torch.tensor([0])
    fusion_loss = torch.tensor([0.2])
    audio_loss = torch.tensor([0.5])
    visual_loss = torch.tensor([0.8])
    logits = torch.tensor([[1.0, 0.0]])

    signal = _signal_for_mode(
        "loss_gap_baseline",
        fusion_loss,
        audio_loss,
        visual_loss,
        logits,
        logits,
        labels,
    )

    assert torch.allclose(signal, torch.tensor([[0.3, 0.6]]))


def test_no_fgm_does_not_apply_fgm_hooks():
    from FGM_toy.data import create_loaders
    from FGM_toy.model import ToyAVModel
    from FGM_toy.train import TrainConfig, train_one_epoch
    from cmi_fgm import CMIFGMState

    train_loader, _ = create_loaders(n_train=16, n_val=8, s=0.5, eta=0.0, batch_size=8, seed=3)
    model = ToyAVModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    state = CMIFGMState(("audio", "visual"), strength=0.5, warmup_steps=0, momentum=0.0)
    metrics = train_one_epoch(
        model,
        train_loader,
        opt,
        torch.device("cpu"),
        TrainConfig(mode="no_fgm"),
        fgm_state=state,
    )

    assert "fgm_coef_audio" not in metrics
    assert state.num_updates == 0


def test_strict_md_behavior_writes_strength_signal_summary_and_plot():
    from FGM_toy.run_grid import SignalFidelityConfig, run_strict_md_behavior

    with TemporaryDirectory() as tmpdir:
        summary = run_strict_md_behavior(
            SignalFidelityConfig(
                output_dir=Path(tmpdir),
                behavior_s_values=(0.0, 0.5),
                seeds=(0, 1),
                n_train=32,
                n_val=16,
                epochs=1,
                batch_size=16,
                fgm_warmup_epochs=0,
                linear_probe_epochs=1,
            )
        )

        assert len(summary["fgm"]) == 2
        assert len(summary["strength_signal"]) == 2
        assert "probe_acc_audio_mean" in summary["fgm"][0]
        assert "probe_acc_audio_std" in summary["fgm"][0]
        assert (Path(tmpdir) / "behavior_lines_md.json").exists()
        assert (Path(tmpdir) / "behavior_lines_md.png").exists()


def test_strict_md_behavior_defaults_to_three_seeds():
    from FGM_toy.run_grid import build_signal_fidelity_config

    config = build_signal_fidelity_config(["--strict-md-behavior", "--output-dir", "unused"])

    assert config.seeds == (0, 1, 2)
