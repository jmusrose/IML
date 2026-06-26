from argparse import Namespace
import sys
from pathlib import Path

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _NoOpScheduler:
    def step(self):
        return None


def _make_args(output_dir):
    return Namespace(
        output_dir=str(output_dir),
        seed=0,
        deterministic=False,
        device="cpu",
        modality="av",
        epochs=2,
        lr=0.002,
        momentum=0.9,
        weight_decay=1e-4,
        batch_size=4,
        fgm=False,
        no_progress=True,
        audio_loss_weight=1.0,
        visual_loss_weight=5.0,
    )


def _patch_fast_run(monkeypatch, module):
    def fake_create_dataloaders(args):
        args.num_classes = 2
        loader = [object()]
        return loader, loader, loader, {"train": 1, "val": 1, "test": 1}

    def fake_train_one_epoch(*args, **kwargs):
        return {
            "loss": 1.0,
            "fusion_loss": 0.5,
            "audio_loss": 0.25,
            "visual_loss": 0.25,
            "acc": 0.5,
            "macro_f1": 0.5,
            "audio_acc": 0.5,
            "visual_acc": 0.5,
        }

    def fake_evaluate(*args, **kwargs):
        split_name = kwargs.get("split_name")
        acc = 0.75 if split_name == "test" else 0.6
        return {
            "loss": 0.8,
            "fusion_loss": 0.4,
            "audio_loss": 0.2,
            "visual_loss": 0.2,
            "acc": acc,
            "macro_f1": acc,
            "audio_acc": 0.55,
            "visual_acc": 0.65,
        }

    def forbidden_checkpoint(*args, **kwargs):
        raise AssertionError("run_training should not save checkpoint files")

    def forbidden_torch_load(*args, **kwargs):
        raise AssertionError("run_training should not load checkpoint files")

    monkeypatch.setattr(module, "set_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "create_dataloaders", fake_create_dataloaders)
    monkeypatch.setattr(module, "build_model", lambda *args, **kwargs: torch.nn.Linear(1, 2))
    monkeypatch.setattr(module, "build_fgm_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "build_scheduler", lambda *args, **kwargs: _NoOpScheduler())
    monkeypatch.setattr(module, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(module, "evaluate", fake_evaluate)
    monkeypatch.setattr(module, "plot_history", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "save_checkpoint", forbidden_checkpoint, raising=False)
    monkeypatch.setattr(module.torch, "load", forbidden_torch_load)


@pytest.mark.parametrize("module_name", ["train_cremad", "train_ks", "train_ave"])
def test_av_v4_training_runs_write_unique_result_dirs_without_checkpoint_files(
    tmp_path,
    monkeypatch,
    module_name,
):
    module = pytest.importorskip(f"AV_v4.{module_name}")
    _patch_fast_run(monkeypatch, module)

    parent_dir = tmp_path / module_name
    args = _make_args(parent_dir)

    result = module.run_training(args)

    run_dirs = [path for path in parent_dir.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert run_dir != parent_dir
    assert result["test_acc"] == 0.75
    assert (run_dir / "config.json").exists()
    assert (run_dir / "history.jsonl").exists()
    assert (run_dir / "history.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert not (run_dir / "best.pt").exists()
