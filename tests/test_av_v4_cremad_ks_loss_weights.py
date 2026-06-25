import ast
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


def _encoder_grad_sum(model) -> float:
    return sum(
        parameter.grad.abs().sum().item()
        for parameter in list(model.audio_net.parameters())
        + list(model.visual_net.parameters())
        if parameter.grad is not None
    )


def test_av_v4_cremad_uses_independent_audio_visual_loss_weights():
    from AV_v4.models import AVBaseline
    from AV_v4.train_cremad import forward_and_losses

    model = AVBaseline(num_classes=3)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    audio = torch.randn(2, 1, 64, 80)
    visual = torch.randn(2, 3, 2, 64, 64)
    labels = torch.tensor([0, 1], dtype=torch.long)

    _, losses, handles = forward_and_losses(
        model,
        (audio, visual),
        labels,
        "av",
        criterion,
        audio_loss_weight=1.0,
        visual_loss_weight=3.0,
    )
    for handle in handles:
        handle.remove()

    expected = losses["fusion_loss"] + losses["audio_loss"] + 3.0 * losses["visual_loss"]
    assert torch.allclose(losses["loss"], expected)


def test_av_v4_cremad_keeps_warmup_based_probe_detachment():
    from AV_v4.models import AVBaseline
    from AV_v4.train_cremad import forward_and_losses
    from cmi_fgm import CMIFGMState

    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    audio = torch.randn(2, 1, 64, 80)
    visual = torch.randn(2, 3, 2, 64, 64)
    labels = torch.tensor([0, 1], dtype=torch.long)

    warm_model = AVBaseline(num_classes=3)
    warm_state = CMIFGMState(("audio", "visual"), warmup_steps=2)
    _, warm_losses, warm_handles = forward_and_losses(
        warm_model,
        (audio, visual),
        labels,
        "av",
        criterion,
        fgm_state=warm_state,
        audio_loss_weight=2.0,
        visual_loss_weight=3.0,
    )
    (warm_losses["audio_loss"] + warm_losses["visual_loss"]).backward()
    for handle in warm_handles:
        handle.remove()
    assert _encoder_grad_sum(warm_model) > 0

    detached_model = AVBaseline(num_classes=3)
    detached_state = CMIFGMState(("audio", "visual"), warmup_steps=1)
    detached_state.num_updates = 1
    _, detached_losses, detached_handles = forward_and_losses(
        detached_model,
        (audio, visual),
        labels,
        "av",
        criterion,
        fgm_state=detached_state,
        audio_loss_weight=2.0,
        visual_loss_weight=3.0,
    )
    (detached_losses["audio_loss"] + detached_losses["visual_loss"]).backward()
    for handle in detached_handles:
        handle.remove()
    assert _encoder_grad_sum(detached_model) == 0


def test_av_v4_cremad_and_ks_cli_accept_loss_weights():
    from AV_v4 import train_cremad, train_ks

    cremad_defaults = train_cremad.parse_args(["--device", "cpu"])
    cremad_explicit = train_cremad.parse_args(
        ["--device", "cpu", "--audio-loss-weight", "2", "--visual-loss-weight", "4"]
    )

    with TemporaryDirectory() as tmpdir:
        class_file = Path(tmpdir) / "class.txt"
        class_file.write_text("a,b", encoding="utf-8")
        ks_defaults = train_ks.parse_args(
            ["--device", "cpu", "--class-file", str(class_file)]
        )
        ks_explicit = train_ks.parse_args(
            [
                "--device",
                "cpu",
                "--class-file",
                str(class_file),
                "--audio-loss-weight",
                "3",
                "--visual-loss-weight",
                "5",
            ]
        )

    assert (cremad_defaults.audio_loss_weight, cremad_defaults.visual_loss_weight) == (1.0, 1.0)
    assert (cremad_explicit.audio_loss_weight, cremad_explicit.visual_loss_weight) == (2.0, 4.0)
    assert (ks_defaults.audio_loss_weight, ks_defaults.visual_loss_weight) == (1.0, 1.0)
    assert (ks_explicit.audio_loss_weight, ks_explicit.visual_loss_weight) == (3.0, 5.0)


def test_av_v4_cremad_and_ks_forward_cli_weights_to_all_epochs_and_evaluations():
    root = Path(__file__).resolve().parents[1]
    for relative_path in ("AV_v4/train_cremad.py", "AV_v4/train_ks.py"):
        tree = ast.parse((root / relative_path).read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"train_one_epoch", "evaluate"}
        ]
        assert len(calls) == 3
        for call in calls:
            keywords = {keyword.arg: keyword.value for keyword in call.keywords}
            assert isinstance(keywords["audio_loss_weight"], ast.Attribute)
            assert keywords["audio_loss_weight"].attr == "audio_loss_weight"
            assert isinstance(keywords["visual_loss_weight"], ast.Attribute)
            assert keywords["visual_loss_weight"].attr == "visual_loss_weight"


def test_av_v4_cremad_and_ks_runtime_use_av_v4_modules():
    from AV_v4 import train_cremad, train_ks, train_video

    assert train_cremad.AVBaseline.__module__ == "AV_v4.models.baseline"
    assert train_cremad.CREMADAVDataset.__module__ == "AV_v4.datasets.cremad"
    assert train_ks.train_one_epoch.__module__ == "AV_v4.training"
    assert train_ks.KSDataset.__module__ == "AV_v4.datasets.ks"
    assert train_video.train_cremad.__name__ == "AV_v4.train_cremad"
    assert train_video.train_ks.__name__ == "AV_v4.train_ks"


def test_av_v4_unified_video_entry_accepts_loss_weights():
    from AV_v4.train_video import build_dataset_args

    cremad = build_dataset_args(
        [
            "--dataset",
            "cremad",
            "--audio-loss-weight",
            "2",
            "--visual-loss-weight",
            "4",
        ]
    )

    with TemporaryDirectory() as tmpdir:
        class_file = Path(tmpdir) / "class.txt"
        class_file.write_text("a,b", encoding="utf-8")
        ks = build_dataset_args(
            [
                "--dataset",
                "ks",
                "--class-file",
                str(class_file),
                "--audio-loss-weight",
                "3",
                "--visual-loss-weight",
                "5",
            ]
        )

    assert (cremad.audio_loss_weight, cremad.visual_loss_weight) == (2.0, 4.0)
    assert (ks.audio_loss_weight, ks.visual_loss_weight) == (3.0, 5.0)
