from __future__ import annotations

import torch
import torch.nn as nn


def mlp(in_features: int, hidden_features: int, out_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, hidden_features),
        nn.ReLU(),
        nn.Linear(hidden_features, out_features),
    )


class ToyAVModel(nn.Module):
    def __init__(self, input_dim: int = 16, feature_dim: int = 32, num_classes: int = 2) -> None:
        super().__init__()
        self.audio_encoder = mlp(input_dim, feature_dim, feature_dim)
        self.visual_encoder = mlp(input_dim, feature_dim, feature_dim)
        self.audio_probe = mlp(feature_dim, feature_dim, num_classes)
        self.visual_probe = mlp(feature_dim, feature_dim, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, num_classes),
        )
        self.feature_dim = feature_dim

    def forward(self, audio: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        return self.forward_with_modal_logits(audio, visual)["logits"]

    def forward_with_modal_logits(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
        detach_probe_features: bool = True,
    ) -> dict[str, torch.Tensor]:
        audio_feature = self.audio_encoder(audio)
        visual_feature = self.visual_encoder(visual)
        audio_probe_feature = audio_feature.detach() if detach_probe_features else audio_feature
        visual_probe_feature = visual_feature.detach() if detach_probe_features else visual_feature
        return {
            "logits": self.classifier(torch.cat([audio_feature, visual_feature], dim=1)),
            "audio_logits": self.audio_probe(audio_probe_feature),
            "visual_logits": self.visual_probe(visual_probe_feature),
            "audio_feature": audio_feature,
            "visual_feature": visual_feature,
        }

