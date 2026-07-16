from __future__ import annotations

import importlib
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch


ABLATION_DIR = Path(__file__).resolve().parent
ROOT_DIR = ABLATION_DIR.parent
sys.path.insert(0, str(ABLATION_DIR))
sys.path.insert(1, str(ROOT_DIR))


def _reload(name: str):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def _fgm_args(**overrides):
    values = {
        "fgm": True,
        "modality": "av",
        "fgm_lambda": 0.5,
        "fgm_tau": 1.0,
        "fgm_momentum": 0.0,
        "fgm_warmup_steps": 0,
        "fgm_signal": "delta",
        "fgm_use_relative": True,
        "fgm_use_absolute": True,
        "fgm_feature_path": True,
        "fgm_classifier_path": True,
    }
    values.update(overrides)
    return Namespace(**values)


def test_ablation_training_uses_local_models():
    training = _reload("training")

    assert training.AVBaseline.__module__.startswith("models.")


@pytest.mark.parametrize("module_name", ["train_ave", "train_ks", "train_cremad"])
def test_dataset_trainers_default_to_full_method(module_name: str):
    module = _reload(module_name)
    args = module.parse_args([])

    assert args.fgm is True
    assert args.fgm_signal == "delta"
    assert args.fgm_use_relative is True
    assert args.fgm_use_absolute is True
    assert args.fgm_feature_path is True
    assert args.fgm_classifier_path is True


def test_train_video_default_to_full_method():
    train_video = _reload("train_video")
    args = train_video.build_dataset_args(["--dataset", "ave"])

    assert args.fgm is True
    assert args.fgm_signal == "delta"
    assert args.fgm_use_relative is True
    assert args.fgm_use_absolute is True
    assert args.fgm_feature_path is True
    assert args.fgm_classifier_path is True


def test_build_fgm_state_honors_relative_absolute_switches():
    training = _reload("training")
    state = training.build_fgm_state(_fgm_args(fgm_use_relative=False, fgm_use_absolute=False))

    state.update(torch.tensor([[2.0, 0.0]]))
    coeffs = state.coefficients(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)

    assert torch.allclose(coeffs["audio"], torch.tensor([1.25]))
    assert torch.allclose(coeffs["visual"], torch.tensor([1.25]))


def test_build_fgm_state_treats_both_paths_off_as_no_fgm():
    training = _reload("training")

    state = training.build_fgm_state(_fgm_args(fgm_feature_path=False, fgm_classifier_path=False))

    assert state is None


def test_grad_norm_signal_is_reserved_with_clear_error():
    training = _reload("training")

    with pytest.raises(NotImplementedError, match="grad_norm"):
        training.build_fgm_state(_fgm_args(fgm_signal="grad_norm"))


def test_signal_variants_compute_expected_values():
    training = _reload("training")
    fusion = torch.tensor([1.0, 2.0])
    audio = torch.tensor([2.0, 5.0])
    visual = torch.tensor([4.0, 3.0])
    audio_acc = torch.tensor(0.25)
    visual_acc = torch.tensor(0.75)

    assert torch.allclose(
        training.compute_fgm_signal("delta", fusion, audio, visual, audio_acc, visual_acc),
        torch.tensor([[3.0, 1.0], [1.0, 3.0]]),
    )
    assert torch.allclose(
        training.compute_fgm_signal("loss_gap", fusion, audio, visual, audio_acc, visual_acc),
        torch.tensor([[4.0, 2.0], [3.0, 5.0]]),
    )
    assert torch.allclose(
        training.compute_fgm_signal("loss_ratio", fusion, audio, visual, audio_acc, visual_acc),
        torch.tensor([[4.0, 2.0], [1.5, 2.5]]),
    )
    assert torch.allclose(
        training.compute_fgm_signal("acc_gap", fusion, audio, visual, audio_acc, visual_acc),
        torch.tensor([[0.25, 0.75], [0.25, 0.75]]),
    )


def test_forward_backward_smoke_with_ablation_fgm():
    training = _reload("training")
    model = training.AVBaseline(num_classes=3)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    state = training.build_fgm_state(_fgm_args(fgm_use_relative=False, fgm_use_absolute=False))
    audio = torch.randn(2, 1, 64, 80)
    visual = torch.randn(2, 3, 2, 64, 64)
    labels = torch.tensor([0, 1], dtype=torch.long)

    logits, losses, handles = training.forward_and_losses(
        model,
        (audio, visual),
        labels,
        "av",
        criterion,
        fgm_state=state,
    )
    losses["loss"].backward()
    for handle in handles:
        handle.remove()

    assert tuple(logits.shape) == (2, 3)
    assert "fgm_coef_audio" in losses
    assert model.classifier.weight.grad is not None
