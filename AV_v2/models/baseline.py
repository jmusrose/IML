from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


DATASET_NUM_CLASSES = {
    "CREMAD": 6,
    "KineticSound": 34,
    "VGGSound": 309,
    "AVE": 28,
    "kinect400": 400,
}


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    """2D ResNet18 feature-map backbone without avgpool or fc."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, blocks=2)
        self.layer2 = self._make_layer(BasicBlock, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, blocks=2, stride=2)

        self._init_weights()

    def _make_layer(
        self,
        block: Callable[..., BasicBlock],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def pool_audio_feature(feature_map: torch.Tensor) -> torch.Tensor:
    feature = F.adaptive_avg_pool2d(feature_map, 1)
    return torch.flatten(feature, 1)


def pool_visual_feature(feature_map: torch.Tensor, batch_size: int) -> torch.Tensor:
    _, channels, height, width = feature_map.shape
    feature_map = feature_map.view(batch_size, -1, channels, height, width)
    feature_map = feature_map.permute(0, 2, 1, 3, 4)
    feature = F.adaptive_avg_pool3d(feature_map, 1)
    return torch.flatten(feature, 1)


class AudioBaseline(nn.Module):
    def __init__(self, num_classes: int = DATASET_NUM_CLASSES["CREMAD"]) -> None:
        super().__init__()
        self.encoder = ResNet18(in_channels=1)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        feature_map = self.encoder(audio)
        feature = pool_audio_feature(feature_map)
        return self.classifier(feature)


class VisualBaseline(nn.Module):
    def __init__(self, num_classes: int = DATASET_NUM_CLASSES["CREMAD"]) -> None:
        super().__init__()
        self.encoder = ResNet18(in_channels=3)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, visual: torch.Tensor) -> torch.Tensor:
        batch_size, channels, num_frames, height, width = visual.shape
        visual = visual.permute(0, 2, 1, 3, 4)
        visual = visual.reshape(batch_size * num_frames, channels, height, width)

        feature_map = self.encoder(visual)
        feature = pool_visual_feature(feature_map, batch_size)
        return self.classifier(feature)


class AVBaseline(nn.Module):
    def __init__(
        self,
        num_classes: int = DATASET_NUM_CLASSES["CREMAD"],
        fusion_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.audio_net = ResNet18(in_channels=1)
        self.visual_net = ResNet18(in_channels=3)
        self.audio_probe = nn.Linear(512, num_classes)
        self.visual_probe = nn.Linear(512, num_classes)
        self.fusion_dropout = nn.Dropout(p=fusion_dropout) if fusion_dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(1024, num_classes)

    def extract_audio_feature(self, audio: torch.Tensor) -> torch.Tensor:
        audio_feature_map = self.audio_net(audio)
        return pool_audio_feature(audio_feature_map)

    def extract_visual_feature(self, visual: torch.Tensor) -> torch.Tensor:
        batch_size = visual.shape[0]
        _, channels, num_frames, height, width = visual.shape
        visual = visual.permute(0, 2, 1, 3, 4)
        visual = visual.reshape(batch_size * num_frames, channels, height, width)
        visual_feature_map = self.visual_net(visual)
        return pool_visual_feature(visual_feature_map, batch_size)

    def forward(self, audio: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        audio_feature = self.extract_audio_feature(audio)
        visual_feature = self.extract_visual_feature(visual)
        fusion_feature = torch.cat([audio_feature, visual_feature], dim=1)
        fusion_feature = self.fusion_dropout(fusion_feature)
        return self.classifier(fusion_feature)

    def forward_with_modal_logits(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        audio_feature = self.extract_audio_feature(audio)
        visual_feature = self.extract_visual_feature(visual)
        fusion_feature = torch.cat([audio_feature, visual_feature], dim=1)
        fusion_feature = self.fusion_dropout(fusion_feature)
        audio_logits = self.audio_probe(audio_feature.detach())
        visual_logits = self.visual_probe(visual_feature.detach())
        logits = self.classifier(fusion_feature)
        return {
            "logits": logits,
            "audio_logits": audio_logits,
            "visual_logits": visual_logits,
            "audio_feature": audio_feature,
            "visual_feature": visual_feature,
        }
