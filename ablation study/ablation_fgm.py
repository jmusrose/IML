from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch


FGM_SIGNAL_CHOICES = ("delta", "loss_gap", "loss_ratio", "acc_gap", "grad_norm")


class AblationFGMState:
    """CMI-FGM state with ablation switches kept local to this folder."""

    def __init__(
        self,
        modalities: Sequence[str],
        strength: float = 0.5,
        temperature: float = 1.0,
        momentum: float = 0.9,
        warmup_steps: int = 0,
        signal_name: str = "delta",
        use_relative: bool = True,
        use_absolute: bool = True,
        feature_path: bool = True,
        classifier_path: bool = True,
        eps: float = 1e-6,
    ) -> None:
        if not modalities:
            raise ValueError("AblationFGMState requires at least one modality.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if not 0 <= momentum <= 1:
            raise ValueError("momentum must be in [0, 1].")
        if signal_name == "grad_norm":
            raise NotImplementedError(
                "fgm_signal='grad_norm' is reserved but not implemented because it "
                "requires an additional gradient collection pass before modulation."
            )
        if signal_name not in FGM_SIGNAL_CHOICES:
            raise ValueError(f"Unsupported fgm signal: {signal_name}")

        self.modalities = tuple(modalities)
        self.strength = float(strength)
        self.temperature = float(temperature)
        self.momentum = float(momentum)
        self.warmup_steps = int(warmup_steps)
        self.signal_name = signal_name
        self.use_relative = bool(use_relative)
        self.use_absolute = bool(use_absolute)
        self.feature_path = bool(feature_path)
        self.classifier_path = bool(classifier_path)
        self.eps = float(eps)
        self.prev_signal: torch.Tensor | None = None
        self.s_bar: torch.Tensor | None = None
        self.num_updates = 0

    def update(self, signal: torch.Tensor) -> None:
        if signal.ndim != 2 or signal.shape[1] != len(self.modalities):
            raise ValueError(
                f"signal must have shape [batch, {len(self.modalities)}], "
                f"got {tuple(signal.shape)}"
            )

        detached = signal.detach()
        positive = detached.clamp_min(0)
        batch_strength = positive.sum(dim=1).mean()
        if self.s_bar is None:
            self.s_bar = batch_strength
        else:
            self.s_bar = self.momentum * self.s_bar.to(batch_strength.device) + (1 - self.momentum) * batch_strength
        self.prev_signal = detached
        self.num_updates += 1

    def coefficients(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        if self.prev_signal is None or self.s_bar is None or self.num_updates <= self.warmup_steps:
            return {
                modality: torch.ones(batch_size, device=device, dtype=dtype)
                for modality in self.modalities
            }

        signal = self.prev_signal.to(device=device, dtype=dtype).clamp_min(0)
        if signal.shape[0] != batch_size:
            signal = signal.mean(dim=0, keepdim=True).expand(batch_size, -1)

        if self.use_relative:
            relative = torch.softmax(signal / self.temperature, dim=1)
        else:
            relative = torch.full_like(signal, 1.0 / len(self.modalities))

        if self.use_absolute:
            absolute = signal.sum(dim=1, keepdim=True)
            s_bar = self.s_bar.to(device=device, dtype=dtype).clamp_min(self.eps)
            normalized = absolute / (s_bar + self.eps)
        else:
            normalized = torch.ones(batch_size, 1, device=device, dtype=dtype)

        coef = 1 + self.strength * relative * normalized
        return {
            modality: coef[:, index].detach()
            for index, modality in enumerate(self.modalities)
        }

    def mean_signal(self) -> dict[str, torch.Tensor]:
        if self.prev_signal is None:
            return {modality: torch.tensor(0.0) for modality in self.modalities}
        signal = self.prev_signal.detach().clamp_min(0)
        return {
            modality: signal[:, index].mean()
            for index, modality in enumerate(self.modalities)
        }


def compute_fgm_signal(
    signal_name: str,
    fusion_per_sample: torch.Tensor,
    audio_per_sample: torch.Tensor,
    visual_per_sample: torch.Tensor,
    audio_acc: torch.Tensor,
    visual_acc: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if signal_name == "delta":
        signal_audio = visual_per_sample.detach() - fusion_per_sample.detach()
        signal_visual = audio_per_sample.detach() - fusion_per_sample.detach()
    elif signal_name == "loss_gap":
        signal_audio = visual_per_sample.detach()
        signal_visual = audio_per_sample.detach()
    elif signal_name == "loss_ratio":
        denom = fusion_per_sample.detach().clamp_min(eps)
        signal_audio = visual_per_sample.detach() / denom
        signal_visual = audio_per_sample.detach() / denom
    elif signal_name == "acc_gap":
        batch_size = fusion_per_sample.shape[0]
        signal_audio = (1 - visual_acc.detach()).expand(batch_size)
        signal_visual = (1 - audio_acc.detach()).expand(batch_size)
    elif signal_name == "grad_norm":
        raise NotImplementedError(
            "fgm_signal='grad_norm' is reserved but not implemented because it "
            "requires feature-gradient norms from an additional backward pass."
        )
    else:
        raise ValueError(f"Unsupported fgm signal: {signal_name}")
    return torch.stack([signal_audio, signal_visual], dim=1)


def add_fgm_arguments(parser: argparse.ArgumentParser, warmup_default: int) -> None:
    parser.add_argument("--fgm", dest="fgm", action="store_true", help="Enable CMI-FGM gradient modulation for AV training.")
    parser.add_argument("--no-fgm", dest="fgm", action="store_false", help="Disable CMI-FGM gradient modulation.")
    parser.set_defaults(fgm=True)
    parser.add_argument("--fgm-lambda", type=float, default=0.5)
    parser.add_argument("--fgm-tau", type=float, default=1.0)
    parser.add_argument("--fgm-momentum", type=float, default=0.9)
    parser.add_argument("--fgm-warmup-steps", type=int, default=warmup_default)
    parser.add_argument("--fgm-signal", choices=FGM_SIGNAL_CHOICES, default="delta")
    parser.add_argument("--fgm-use-relative", dest="fgm_use_relative", action="store_true")
    parser.add_argument("--no-fgm-use-relative", dest="fgm_use_relative", action="store_false")
    parser.set_defaults(fgm_use_relative=True)
    parser.add_argument("--fgm-use-absolute", dest="fgm_use_absolute", action="store_true")
    parser.add_argument("--no-fgm-use-absolute", dest="fgm_use_absolute", action="store_false")
    parser.set_defaults(fgm_use_absolute=True)
    parser.add_argument("--fgm-feature-path", dest="fgm_feature_path", action="store_true")
    parser.add_argument("--no-fgm-feature-path", dest="fgm_feature_path", action="store_false")
    parser.set_defaults(fgm_feature_path=True)
    parser.add_argument("--fgm-classifier-path", dest="fgm_classifier_path", action="store_true")
    parser.add_argument("--no-fgm-classifier-path", dest="fgm_classifier_path", action="store_false")
    parser.set_defaults(fgm_classifier_path=True)
