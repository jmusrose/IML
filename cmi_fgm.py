from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class CMIFGMState:
    """State for previous-step CMI-FGM coefficients."""

    def __init__(
        self,
        modalities: Sequence[str],
        strength: float = 0.5,
        temperature: float = 1.0,
        momentum: float = 0.9,
        warmup_steps: int = 0,
        eps: float = 1e-6,
    ) -> None:
        if not modalities:
            raise ValueError("CMIFGMState requires at least one modality.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if not 0 <= momentum <= 1:
            raise ValueError("momentum must be in [0, 1].")

        self.modalities = tuple(modalities)
        self.strength = float(strength)
        self.temperature = float(temperature)
        self.momentum = float(momentum)
        self.warmup_steps = int(warmup_steps)
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

        relative = torch.softmax(signal / self.temperature, dim=1)
        absolute = signal.sum(dim=1, keepdim=True)
        s_bar = self.s_bar.to(device=device, dtype=dtype).clamp_min(self.eps)
        normalized = absolute / (s_bar + self.eps)
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


def register_feature_gradient_hooks(
    features: dict[str, torch.Tensor],
    coefficients: dict[str, torch.Tensor],
) -> list[torch.utils.hooks.RemovableHandle]:
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for modality, feature in features.items():
        coef = coefficients[modality].to(device=feature.device, dtype=feature.dtype)

        def scale_feature_grad(grad: torch.Tensor, coef: torch.Tensor = coef) -> torch.Tensor:
            shape = (coef.shape[0],) + (1,) * (grad.ndim - 1)
            return grad * coef.view(shape)

        handles.append(feature.register_hook(scale_feature_grad))
    return handles


def register_split_linear_weight_hook(
    linear: nn.Linear,
    split_sizes: Sequence[int],
    modalities: Sequence[str],
    coefficients: dict[str, torch.Tensor],
) -> torch.utils.hooks.RemovableHandle:
    if len(split_sizes) != len(modalities):
        raise ValueError("split_sizes and modalities must have the same length.")
    if sum(split_sizes) != linear.in_features:
        raise ValueError(
            f"split sizes sum to {sum(split_sizes)}, expected {linear.in_features}."
        )

    scale_values = [
        coefficients[modality].detach().mean()
        for modality in modalities
    ]

    def scale_weight_grad(grad: torch.Tensor) -> torch.Tensor:
        scaled = grad.clone()
        start = 0
        for width, scale in zip(split_sizes, scale_values):
            end = start + width
            scaled[:, start:end] *= scale.to(device=grad.device, dtype=grad.dtype)
            start = end
        return scaled

    return linear.weight.register_hook(scale_weight_grad)
