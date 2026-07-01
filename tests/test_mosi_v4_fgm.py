import torch


class TinyTextEncoder(torch.nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.embedding = torch.nn.Embedding(64, hidden_size)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.embedding(input_ids)
        pooled = hidden[:, 0]
        return type("Output", (), {"last_hidden_state": hidden, "pooler_output": pooled})()


def tiny_mosi_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 2, 0], [1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long),
        "vision": torch.randn(2, 3, 47),
        "audio": torch.randn(2, 3, 74),
        "vision_mask": torch.tensor([[True, True, False], [True, True, True]]),
        "audio_mask": torch.tensor([[True, True, False], [True, True, True]]),
        "labels": torch.tensor([1.0, -1.0]),
    }


def tiny_mosi_model():
    from MOSI_v4.models import MOSIRegressionModel

    return MOSIRegressionModel(
        text_encoder=TinyTextEncoder(hidden_size=8),
        text_dim=8,
        vision_dim=47,
        audio_dim=74,
        hidden_sz=10,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    )


def test_mosi_v4_weighted_probe_loss_uses_independent_factors():
    from MOSI_v4.train_mosi import forward_and_losses

    torch.manual_seed(0)
    model = tiny_mosi_model()
    batch = tiny_mosi_batch()
    criterion = torch.nn.MSELoss(reduction="none")

    _, losses, handles = forward_and_losses(
        model,
        batch,
        criterion,
        text_loss_weight=2.0,
        vision_loss_weight=3.0,
        audio_loss_weight=4.0,
        detach_probe_features=False,
    )
    for handle in handles:
        handle.remove()

    expected = (
        losses["fusion_loss"]
        + 2.0 * losses["text_loss"]
        + 3.0 * losses["vision_loss"]
        + 4.0 * losses["audio_loss"]
    )
    assert torch.allclose(losses["loss"], expected)


def test_mosi_v4_probe_losses_can_reach_modality_encoders():
    from MOSI_v4.train_mosi import forward_and_losses

    torch.manual_seed(1)
    model = tiny_mosi_model()
    batch = tiny_mosi_batch()
    criterion = torch.nn.MSELoss(reduction="none")

    _, losses, handles = forward_and_losses(
        model,
        batch,
        criterion,
        detach_probe_features=False,
    )
    (losses["text_loss"] + losses["vision_loss"] + losses["audio_loss"]).backward()
    for handle in handles:
        handle.remove()

    text_grad = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.text_encoder.parameters()
        if parameter.grad is not None
    )
    vision_grad = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.vision_encoder.parameters()
        if parameter.grad is not None
    )
    audio_grad = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.audio_encoder.parameters()
        if parameter.grad is not None
    )
    assert text_grad > 0
    assert vision_grad > 0
    assert audio_grad > 0


def test_mosi_v4_defaults_detach_after_fgm_warmup():
    from MOSI_v4.train_mosi import forward_and_losses
    from cmi_fgm import CMIFGMState

    torch.manual_seed(2)
    model = tiny_mosi_model()
    batch = tiny_mosi_batch()
    criterion = torch.nn.MSELoss(reduction="none")
    state = CMIFGMState(("text", "vision", "audio"), warmup_steps=1)
    state.num_updates = 1

    _, losses, handles = forward_and_losses(
        model,
        batch,
        criterion,
        fgm_state=state,
    )
    (losses["text_loss"] + losses["vision_loss"] + losses["audio_loss"]).backward()
    for handle in handles:
        handle.remove()

    encoder_grad = sum(
        parameter.grad.abs().sum().item()
        for module in (model.text_encoder, model.vision_encoder, model.audio_encoder)
        for parameter in module.parameters()
        if parameter.grad is not None
    )
    assert encoder_grad == 0.0


def test_mosi_v4_cli_accepts_av4_style_probe_options():
    from MOSI_v4.train_mosi import parse_args

    defaults = parse_args([])
    explicit = parse_args(
        [
            "--text-loss-weight",
            "2",
            "--vision-loss-weight",
            "3",
            "--audio-loss-weight",
            "4",
            "--no-detach-probe-features",
        ]
    )

    assert defaults.text_loss_weight == 1.0
    assert defaults.vision_loss_weight == 1.0
    assert defaults.audio_loss_weight == 1.0
    assert defaults.detach_probe_features is True
    assert explicit.text_loss_weight == 2.0
    assert explicit.vision_loss_weight == 3.0
    assert explicit.audio_loss_weight == 4.0
    assert explicit.detach_probe_features is False
