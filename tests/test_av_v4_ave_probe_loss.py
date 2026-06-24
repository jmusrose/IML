import ast
from pathlib import Path

import torch


def test_av_v4_weighted_probe_loss_uses_independent_factors():
    from AV_v4.models import AVBaseline
    from AV_v4.training import forward_and_losses

    torch.manual_seed(0)
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
        visual_loss_weight=5.0,
        detach_probe_features=False,
    )
    for handle in handles:
        handle.remove()

    expected = losses["fusion_loss"] + losses["audio_loss"] + 5.0 * losses["visual_loss"]
    assert torch.allclose(losses["loss"], expected)


def test_av_v4_shared_training_keeps_unit_probe_weight_by_default():
    from AV_v4.models import AVBaseline
    from AV_v4.training import forward_and_losses

    torch.manual_seed(1)
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
    )
    for handle in handles:
        handle.remove()

    expected = losses["fusion_loss"] + losses["audio_loss"] + losses["visual_loss"]
    assert torch.allclose(losses["loss"], expected)


def test_av_v4_probe_losses_reach_encoders_after_fgm_warmup():
    from AV_v4.models import AVBaseline
    from AV_v4.training import forward_and_losses
    from cmi_fgm import CMIFGMState

    model = AVBaseline(num_classes=3)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    state = CMIFGMState(("audio", "visual"), warmup_steps=1)
    state.num_updates = 1
    audio = torch.randn(2, 1, 64, 80)
    visual = torch.randn(2, 3, 2, 64, 64)
    labels = torch.tensor([0, 1], dtype=torch.long)

    _, losses, handles = forward_and_losses(
        model,
        (audio, visual),
        labels,
        "av",
        criterion,
        fgm_state=state,
        audio_loss_weight=5.0,
        visual_loss_weight=5.0,
        detach_probe_features=False,
    )
    (losses["audio_loss"] + losses["visual_loss"]).backward()
    for handle in handles:
        handle.remove()

    audio_grad = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.audio_net.parameters()
        if parameter.grad is not None
    )
    visual_grad = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.visual_net.parameters()
        if parameter.grad is not None
    )
    assert audio_grad > 0
    assert visual_grad > 0


def test_av_v4_ave_entry_point_forwards_independent_probe_loss_weights():
    source_path = Path(__file__).resolve().parents[1] / "AV_v4" / "train_ave.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
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
        assert ast.literal_eval(keywords["detach_probe_features"]) is False


def test_av_v4_ave_runtime_uses_av_v4_training_dataset_and_model():
    import AV_v4.train_ave as train_ave
    import AV_v4.training as training

    assert train_ave.train_one_epoch.__module__ == "AV_v4.training"
    assert train_ave.AVEDataset.__module__ == "AV_v4.datasets.ave"
    assert training.AVBaseline.__module__ == "AV_v4.models.baseline"


def test_av_v4_ave_cli_accepts_independent_probe_loss_weights():
    from AV_v4.train_ave import parse_args

    defaults = parse_args(["--device", "cpu"])
    explicit = parse_args(
        [
            "--device",
            "cpu",
            "--audio-loss-weight",
            "1",
            "--visual-loss-weight",
            "3",
        ]
    )

    assert defaults.audio_loss_weight == 5.0
    assert defaults.visual_loss_weight == 5.0
    assert explicit.audio_loss_weight == 1.0
    assert explicit.visual_loss_weight == 3.0


def test_av_v4_epoch_report_includes_epoch_and_elapsed_time():
    from AV_v4.training import format_epoch_report

    metrics = {"loss": 1.0, "acc": 0.5}
    report = format_epoch_report(
        3,
        metrics,
        metrics,
        epoch_seconds=65.4,
        elapsed_seconds=3665.4,
    )

    assert "Epoch 003 | time 00:01:05 | elapsed 01:01:05" in report
