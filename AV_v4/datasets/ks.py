from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .cremad import CREMADTrainImageTransform, ResizeToTensorNormalize, read_wav_mono, waveform_to_log_spectrogram


@dataclass(frozen=True)
class KSSample:
    sample_id: str
    class_name: str
    label: int
    audio_path: Path
    image_dir: Path


class KSTrainImageTransform(CREMADTrainImageTransform):
    """ICCV/GDL-style KS train transform: random resized crop, flip, normalize."""

    def __init__(
        self,
        size: int = 224,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        horizontal_flip_prob: float = 0.5,
    ) -> None:
        super().__init__(
            size=size,
            mean=mean,
            std=std,
            scale=scale,
            ratio=ratio,
            horizontal_flip_prob=horizontal_flip_prob,
        )


def load_ks_classes(class_file: str | os.PathLike[str]) -> list[str]:
    text = Path(class_file).read_text(encoding="utf-8").strip()
    classes = []
    for name in text.split(","):
        name = name.strip().replace(" ", "_")
        if name:
            classes.append(name)
    return classes


def discover_ks_samples(
    root: str | os.PathLike[str],
    class_file: str | os.PathLike[str],
    mode: str = "train",
    min_frames: int = 3,
) -> tuple[list[KSSample], dict[str, int]]:
    if mode not in {"train", "test"}:
        raise ValueError(f"Unsupported KS mode: {mode}")

    root = Path(root)
    classes = load_ks_classes(class_file)
    class_to_idx = {class_name: index for index, class_name in enumerate(classes)}

    image_split = "train_img" if mode == "train" else "val_img"
    audio_split = "train" if mode == "train" else "test"
    image_root = root / "visual" / image_split / "Image-01-FPS"
    audio_root = root / "audio" / audio_split
    if not image_root.exists():
        raise FileNotFoundError(f"KS image directory not found: {image_root}")
    if not audio_root.exists():
        raise FileNotFoundError(f"KS audio directory not found: {audio_root}")

    samples = []
    for class_name in classes:
        image_class_dir = image_root / class_name
        audio_class_dir = audio_root / class_name
        if not image_class_dir.exists() or not audio_class_dir.exists():
            continue

        for image_dir in sorted(path for path in image_class_dir.iterdir() if path.is_dir()):
            sample_id = image_dir.name
            if len([path for path in image_dir.iterdir() if path.is_file()]) < min_frames:
                continue

            audio_path = audio_class_dir / f"{sample_id}.wav"
            if not audio_path.exists():
                continue

            samples.append(
                KSSample(
                    sample_id=sample_id,
                    class_name=class_name,
                    label=class_to_idx[class_name],
                    audio_path=audio_path,
                    image_dir=image_dir,
                )
            )

    return samples, class_to_idx


def fix_ks_waveform_length(
    waveform: torch.Tensor,
    sample_rate: int,
    duration: float,
    mode: str,
) -> torch.Tensor:
    target_length = int(round(sample_rate * duration))
    if waveform.numel() < target_length:
        repeat_count = int(np.ceil(target_length / max(1, waveform.numel())))
        waveform = waveform.repeat(repeat_count)

    if waveform.numel() == target_length:
        return waveform

    max_start = waveform.numel() - target_length
    if mode == "train":
        start = random.randint(0, max_start)
    else:
        start = max_start // 2
    return waveform[start : start + target_length]


def select_ks_frame_indices(
    frame_count: int,
    use_video_frames: int,
    mode: str,
) -> np.ndarray:
    if frame_count < 1:
        raise ValueError("Cannot select frames from an empty KS frame directory.")
    if use_video_frames < 1:
        raise ValueError(f"use_video_frames must be >= 1, got {use_video_frames}.")

    replace = frame_count < use_video_frames
    if mode == "train":
        indices = np.random.choice(frame_count, size=use_video_frames, replace=replace)
        return np.sort(indices).astype(np.int64)

    if replace:
        return np.linspace(0, frame_count - 1, num=use_video_frames).round().astype(np.int64)
    return np.linspace(0, frame_count - 1, num=use_video_frames, dtype=np.int64)


class KSDataset(Dataset):
    def __init__(
        self,
        samples: list[KSSample],
        modality: str = "av",
        mode: str = "train",
        use_video_frames: int = 3,
        audio_duration: float = 5.0,
        n_fft: int = 256,
        hop_length: int = 128,
        win_length: int | None = None,
        image_size: int = 224,
        image_transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        if modality not in {"av", "audio", "visual"}:
            raise ValueError(f"Unsupported modality: {modality}")
        if mode not in {"train", "test"}:
            raise ValueError(f"Unsupported KS mode: {mode}")

        self.samples = samples
        self.modality = modality
        self.mode = mode
        self.use_video_frames = use_video_frames
        self.audio_duration = audio_duration
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.image_transform = image_transform or ResizeToTensorNormalize(size=image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_audio(self, sample: KSSample) -> torch.Tensor:
        waveform, sample_rate = read_wav_mono(sample.audio_path)
        waveform = fix_ks_waveform_length(waveform, sample_rate, self.audio_duration, self.mode)
        return waveform_to_log_spectrogram(
            waveform,
            sample_rate,
            audio_duration=self.audio_duration,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

    def _load_visual(self, sample: KSSample) -> torch.Tensor:
        frame_paths = sorted(path for path in sample.image_dir.iterdir() if path.is_file())
        indices = select_ks_frame_indices(len(frame_paths), self.use_video_frames, self.mode)
        images = []
        for index in indices:
            image = Image.open(frame_paths[int(index)]).convert("RGB")
            images.append(self.image_transform(image))
        return torch.stack(images, dim=0).permute(1, 0, 2, 3)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        item = {"label": torch.tensor(sample.label, dtype=torch.long)}
        if self.modality in {"av", "audio"}:
            item["audio"] = self._load_audio(sample)
        if self.modality in {"av", "visual"}:
            item["visual"] = self._load_visual(sample)
        return item
