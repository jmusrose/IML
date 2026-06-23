from __future__ import annotations

import math

import torch
import torch.nn as nn


class BertTextEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased") -> None:
        super().__init__()
        try:
            from transformers import BertModel
        except ImportError as exc:
            raise ImportError(
                "transformers is required for the default BERT text encoder. "
                "Install transformers or pass a custom text_encoder."
            ) from exc

        self.bert = BertModel.from_pretrained(model_name)
        self.output_dim = self.bert.config.hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask)


class SequenceEncoder(nn.Module):
    """Conv1d followed by a MulT-style Transformer sequence encoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_sz: int = 50,
        num_heads: int = 5,
        num_layers: int = 3,
        conv_kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_sz % num_heads != 0:
            raise ValueError(f"hidden_sz ({hidden_sz}) must be divisible by num_heads ({num_heads}).")

        padding = conv_kernel_size // 2
        self.conv = nn.Conv1d(input_dim, hidden_sz, kernel_size=conv_kernel_size, padding=padding)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_sz,
            nhead=num_heads,
            dim_feedforward=hidden_sz * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_sz)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = inputs.transpose(1, 2)
        hidden = torch.relu(self.conv(hidden)).transpose(1, 2)
        padding_mask = None if mask is None else ~mask.bool()
        hidden = self.transformer(hidden, src_key_padding_mask=padding_mask)
        hidden = self.norm(hidden)
        return masked_mean(hidden, mask)


def masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return values.mean(dim=1)
    weights = mask.to(values.dtype).unsqueeze(-1)
    total = (values * weights).sum(dim=1)
    denom = weights.sum(dim=1).clamp_min(torch.finfo(values.dtype).eps)
    return total / denom


class MOSIRegressionModel(nn.Module):
    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        text_encoder: nn.Module | None = None,
        text_dim: int | None = None,
        vision_dim: int = 47,
        audio_dim: int = 74,
        hidden_sz: int = 50,
        num_heads: int = 5,
        num_layers: int = 3,
        conv_kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder if text_encoder is not None else BertTextEncoder(bert_model_name)
        inferred_text_dim = getattr(self.text_encoder, "output_dim", None)
        if inferred_text_dim is None and hasattr(self.text_encoder, "config"):
            inferred_text_dim = getattr(self.text_encoder.config, "hidden_size", None)
        text_dim = int(text_dim or inferred_text_dim or 768)
        self.text_dim = text_dim
        self.hidden_sz = hidden_sz

        self.vision_encoder = SequenceEncoder(
            vision_dim,
            hidden_sz=hidden_sz,
            num_heads=num_heads,
            num_layers=num_layers,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
        )
        self.audio_encoder = SequenceEncoder(
            audio_dim,
            hidden_sz=hidden_sz,
            num_heads=num_heads,
            num_layers=num_layers,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
        )
        self.text_probe = nn.Sequential(nn.Dropout(dropout), nn.Linear(text_dim, 1))
        self.vision_probe = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_sz, 1))
        self.audio_probe = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_sz, 1))
        self.fusion = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(text_dim + hidden_sz * 2, hidden_sz),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_sz, 1),
        )
        self._init_head()

    def _init_head(self) -> None:
        for module in (self.text_probe, self.vision_probe, self.audio_probe, self.fusion):
            for submodule in module.modules():
                if isinstance(submodule, nn.Linear):
                    nn.init.kaiming_uniform_(submodule.weight, a=math.sqrt(5))
                    if submodule.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(submodule.weight)
                        bound = 1 / math.sqrt(fan_in)
                        nn.init.uniform_(submodule.bias, -bound, bound)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        return pooled

    def encode_modalities(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        vision: torch.Tensor,
        audio: torch.Tensor,
        vision_mask: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_feature = self.encode_text(input_ids, attention_mask)
        vision_feature = self.vision_encoder(vision, vision_mask)
        audio_feature = self.audio_encoder(audio, audio_mask)
        return text_feature, vision_feature, audio_feature

    def forward_with_modal_predictions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        vision: torch.Tensor,
        audio: torch.Tensor,
        vision_mask: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        text_feature, vision_feature, audio_feature = self.encode_modalities(
            input_ids,
            attention_mask,
            vision,
            audio,
            vision_mask,
            audio_mask,
        )
        fused = torch.cat([text_feature, vision_feature, audio_feature], dim=1)
        return {
            "prediction": self.fusion(fused).squeeze(-1),
            "text_prediction": self.text_probe(text_feature.detach()).squeeze(-1),
            "vision_prediction": self.vision_probe(vision_feature.detach()).squeeze(-1),
            "audio_prediction": self.audio_probe(audio_feature.detach()).squeeze(-1),
            "text_feature": text_feature,
            "vision_feature": vision_feature,
            "audio_feature": audio_feature,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        vision: torch.Tensor,
        audio: torch.Tensor,
        vision_mask: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_with_modal_predictions(
            input_ids,
            attention_mask,
            vision,
            audio,
            vision_mask,
            audio_mask,
        )["prediction"]
