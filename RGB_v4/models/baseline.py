from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18_feature_extractor(pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    return model


class RGBBaseline(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        self.rgb_net = build_resnet18_feature_extractor(pretrained=pretrained)
        self.depth_net = build_resnet18_feature_extractor(pretrained=pretrained)
        self.rgb_probe = nn.Linear(512, num_classes)
        self.depth_probe = nn.Linear(512, num_classes)
        self.classifier = nn.Linear(1024, num_classes)

    def extract_rgb_feature(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.rgb_net(rgb)

    def extract_depth_feature(self, depth: torch.Tensor) -> torch.Tensor:
        return self.depth_net(depth)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        rgb_feature = self.extract_rgb_feature(rgb)
        depth_feature = self.extract_depth_feature(depth)
        fusion_feature = torch.cat([rgb_feature, depth_feature], dim=1)
        return self.classifier(fusion_feature)

    def forward_with_modal_logits(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        detach_probe_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        rgb_feature = self.extract_rgb_feature(rgb)
        depth_feature = self.extract_depth_feature(depth)
        fusion_feature = torch.cat([rgb_feature, depth_feature], dim=1)
        rgb_probe_feature = rgb_feature.detach() if detach_probe_features else rgb_feature
        depth_probe_feature = depth_feature.detach() if detach_probe_features else depth_feature
        return {
            "logits": self.classifier(fusion_feature),
            "rgb_logits": self.rgb_probe(rgb_probe_feature),
            "depth_logits": self.depth_probe(depth_probe_feature),
            "rgb_feature": rgb_feature,
            "depth_feature": depth_feature,
        }
