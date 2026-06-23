import torch


def _probe_encoder_grad_norm(model):
    total = torch.tensor(0.0)
    for param in list(model.audio_net.parameters()) + list(model.visual_net.parameters()):
        if param.grad is not None:
            total = total + param.grad.detach().abs().sum().cpu()
    return float(total.item())


def test_av_v3_probe_features_keep_gradients_during_warmup():
    from AV_v3.models import AVBaseline
    from AV_v3.train_cremad import forward_and_losses
    from cmi_fgm import CMIFGMState

    model = AVBaseline(num_classes=6)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    fgm_state = CMIFGMState(("audio", "visual"), warmup_steps=2)
    audio = torch.randn(2, 1, 64, 80)
    visual = torch.randn(2, 3, 2, 64, 64)
    labels = torch.tensor([0, 1], dtype=torch.long)

    _, losses, handles = forward_and_losses(
        model,
        (audio, visual),
        labels,
        "av",
        criterion,
        fgm_state=fgm_state,
    )
    (losses["audio_loss"] + losses["visual_loss"]).backward()
    for handle in handles:
        handle.remove()

    assert _probe_encoder_grad_norm(model) > 0.0


def test_av_v3_training_builds_av_v3_model():
    from AV_v3 import train_cremad
    from AV_v3.models import AVBaseline

    model = train_cremad.build_model("av", num_classes=6)

    assert isinstance(model, AVBaseline)


def test_av_v3_probe_features_detach_after_warmup():
    from AV_v3.models import AVBaseline
    from AV_v3.train_cremad import forward_and_losses
    from cmi_fgm import CMIFGMState

    model = AVBaseline(num_classes=6)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    fgm_state = CMIFGMState(("audio", "visual"), warmup_steps=1)
    fgm_state.num_updates = 1
    audio = torch.randn(2, 1, 64, 80)
    visual = torch.randn(2, 3, 2, 64, 64)
    labels = torch.tensor([0, 1], dtype=torch.long)

    _, losses, handles = forward_and_losses(
        model,
        (audio, visual),
        labels,
        "av",
        criterion,
        fgm_state=fgm_state,
    )
    (losses["audio_loss"] + losses["visual_loss"]).backward()
    for handle in handles:
        handle.remove()

    assert _probe_encoder_grad_norm(model) == 0.0
