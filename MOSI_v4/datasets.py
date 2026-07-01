from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


MOSISample = tuple[tuple[list[str], np.ndarray, np.ndarray], np.ndarray, str]


def load_mosi_splits(path: str | Path) -> dict[str, list[MOSISample]]:
    pkl_path = Path(path)
    with pkl_path.open("rb") as handle:
        data = pickle.load(handle)

    expected = {"train", "dev", "test"}
    missing = expected.difference(data)
    if missing:
        raise ValueError(f"Missing MOSI split(s) in {pkl_path}: {sorted(missing)}")
    return {name: list(data[name]) for name in ("train", "dev", "test")}


class MOSIDataset(Dataset):
    def __init__(self, samples: list[MOSISample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        (words, vision, audio), label, sample_id = self.samples[index]
        return {
            "words": list(words),
            "vision": torch.as_tensor(vision, dtype=torch.float32),
            "audio": torch.as_tensor(audio, dtype=torch.float32),
            "label": torch.as_tensor(label, dtype=torch.float32).view(-1),
            "sample_id": sample_id,
        }


def _pad_modality(sequences: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([sequence.shape[0] for sequence in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True)
    steps = torch.arange(padded.shape[1]).unsqueeze(0)
    mask = steps < lengths.unsqueeze(1)
    return padded.contiguous(), mask


def mosi_collate_fn(
    batch: list[dict[str, Any]],
    tokenizer: Callable[..., dict[str, torch.Tensor]],
    max_text_length: int = 64,
) -> dict[str, Any]:
    texts = [" ".join(item["words"]) for item in batch]
    tokenized = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_text_length,
        return_tensors="pt",
    )
    vision, vision_mask = _pad_modality([item["vision"] for item in batch])
    audio, audio_mask = _pad_modality([item["audio"] for item in batch])
    labels = torch.stack([item["label"][0] for item in batch]).float()
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "vision": vision,
        "audio": audio,
        "vision_mask": vision_mask,
        "audio_mask": audio_mask,
        "labels": labels,
        "sample_ids": [item["sample_id"] for item in batch],
    }

