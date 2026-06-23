from __future__ import annotations

import wave
import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path
import random
from typing import Callable, Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


EMOTION_TO_INDEX = {
    "ANG": 0,
    "DIS": 1,
    "FEA": 2,
    "HAP": 3,
    "NEU": 4,
    "SAD": 5,
}


@dataclass(frozen=True)
class CREMADSample:
    sample_id: str
    actor_id: str
    emotion: str
    label: int
    audio_path: Path
    image_dir: Path


def select_frame_indices(
    frame_names: Iterable[str],
    fps: int,
    rng: np.random.Generator | np.random.RandomState | None = None,
) -> np.ndarray:
    """Select frame indices after sorting names, skipping frame 0 when possible."""
    frame_names = sorted(frame_names)
    if not frame_names:
        raise ValueError("Cannot select frames from an empty frame directory.")
    if fps < 1:
        raise ValueError(f"fps must be >= 1, got {fps}.")

    if rng is None:
        rng = np.random

    if len(frame_names) == 1:
        return np.zeros(fps, dtype=np.int64)

    candidate_indices = np.arange(1, len(frame_names))
    replace = len(candidate_indices) < fps
    selected = rng.choice(candidate_indices, size=fps, replace=replace)
    selected = np.sort(selected)
    return selected.astype(np.int64)


def discover_cremad_samples(root: str | os.PathLike[str]) -> list[CREMADSample]:
    root = Path(root)
    audio_root = root / "AudioWAV"
    image_root = root / "Image-01-FPS"
    if not audio_root.exists():
        raise FileNotFoundError(f"CREMA-D AudioWAV directory not found: {audio_root}")
    if not image_root.exists():
        raise FileNotFoundError(f"CREMA-D Image-01-FPS directory not found: {image_root}")

    samples = []
    for image_dir in sorted(image_root.iterdir()):
        if not image_dir.is_dir():
            continue

        sample_id = image_dir.name
        parts = sample_id.split("_")
        if len(parts) < 4:
            continue

        emotion = parts[2]
        if emotion not in EMOTION_TO_INDEX:
            continue

        audio_path = audio_root / f"{sample_id}.wav"
        if not audio_path.exists():
            continue

        samples.append(
            CREMADSample(
                sample_id=sample_id,
                actor_id=parts[0],
                emotion=emotion,
                label=EMOTION_TO_INDEX[emotion],
                audio_path=audio_path,
                image_dir=image_dir,
            )
        )

    return samples


def split_samples_by_actor(
    samples: list[CREMADSample],
    seed: int = 0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, list[CREMADSample]]:
    if not samples:
        raise ValueError("Cannot split an empty sample list.")
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Expected train_ratio > 0, val_ratio >= 0, and train_ratio + val_ratio < 1.")

    actors = sorted({sample.actor_id for sample in samples})
    rng = np.random.default_rng(seed)
    rng.shuffle(actors)

    num_actors = len(actors)
    num_train = max(1, int(round(num_actors * train_ratio)))
    num_val = int(round(num_actors * val_ratio))
    if val_ratio > 0 and num_actors - num_train > 1:
        num_val = max(1, num_val)
    if num_train + num_val >= num_actors:
        num_val = max(0, num_actors - num_train - 1)

    train_actors = set(actors[:num_train])
    val_actors = set(actors[num_train : num_train + num_val])
    test_actors = set(actors[num_train + num_val :])

    return {
        "train": [sample for sample in samples if sample.actor_id in train_actors],
        "val": [sample for sample in samples if sample.actor_id in val_actors],
        "test": [sample for sample in samples if sample.actor_id in test_actors],
    }


def split_samples_random(
    samples: list[CREMADSample],
    seed: int = 0,
    train_ratio: float = 0.8,
) -> dict[str, list[CREMADSample]]:
    if not samples:
        raise ValueError("Cannot split an empty sample list.")
    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("Expected 0 < train_ratio < 1.")

    by_label: dict[int, list[CREMADSample]] = {}
    for sample in samples:
        by_label.setdefault(sample.label, []).append(sample)

    rng = np.random.default_rng(seed)
    split = {"train": [], "test": []}
    for label_samples in by_label.values():
        label_samples = list(label_samples)
        rng.shuffle(label_samples)

        num_samples = len(label_samples)
        num_train = max(1, int(round(num_samples * train_ratio)))
        if num_train >= num_samples:
            num_train = num_samples - 1

        split["train"].extend(label_samples[:num_train])
        split["test"].extend(label_samples[num_train:])

    for part in split.values():
        rng.shuffle(part)

    return split


def split_samples_from_csv(
    samples: list[CREMADSample],
    csv_root: str | os.PathLike[str],
) -> dict[str, list[CREMADSample]]:
    sample_by_id = {sample.sample_id: sample for sample in samples}
    csv_root = Path(csv_root)
    split: dict[str, list[CREMADSample]] = {}

    for split_name in ("train", "test"):
        csv_path = csv_root / f"{split_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CREMA-D split file not found: {csv_path}")

        selected = []
        with csv_path.open(encoding="UTF-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                sample_id = row[0].strip()
                sample = sample_by_id.get(sample_id)
                if sample is not None:
                    selected.append(sample)

        split[split_name] = selected

    return split


def read_wav_mono(path: str | os.PathLike[str]) -> tuple[torch.Tensor, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())

    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported, got sample width {sample_width}.")

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return torch.from_numpy(audio.copy()), sample_rate


def fix_waveform_length(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    if waveform.numel() >= target_length:
        return waveform[:target_length]

    padded = torch.zeros(target_length, dtype=waveform.dtype)
    padded[: waveform.numel()] = waveform
    return padded


def waveform_to_log_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int,
    audio_duration: float = 3.0,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
) -> torch.Tensor:
    target_length = int(round(sample_rate * audio_duration))
    waveform = fix_waveform_length(waveform, target_length)
    window = torch.hann_window(win_length, dtype=waveform.dtype)
    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    spec = spec.abs().pow(2.0)
    return torch.log1p(spec)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.expand_dims(array, axis=-1)
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array)


class ResizeToTensorNormalize:
    def __init__(
        self,
        size: int = 224,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.size = size
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.size, self.size), Image.BILINEAR)
        tensor = pil_to_tensor(image)
        return (tensor - self.mean) / self.std


class CREMADTrainImageTransform(ResizeToTensorNormalize):
    """Train-time random resized crop with optional horizontal flip."""

    def __init__(
        self,
        size: int = 224,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        scale: tuple[float, float] = (0.85, 1.0),
        ratio: tuple[float, float] = (0.95, 1.05),
        horizontal_flip_prob: float = 0.0,
    ) -> None:
        super().__init__(size=size, mean=mean, std=std)
        self.scale = scale
        self.ratio = ratio
        self.horizontal_flip_prob = horizontal_flip_prob

    def _get_random_crop(self, image: Image.Image) -> tuple[int, int, int, int]:
        width, height = image.size
        area = width * height
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))

        for _ in range(10):
            target_area = area * random.uniform(self.scale[0], self.scale[1])
            aspect_ratio = math.exp(random.uniform(log_ratio[0], log_ratio[1]))
            crop_width = int(round(math.sqrt(target_area * aspect_ratio)))
            crop_height = int(round(math.sqrt(target_area / aspect_ratio)))
            if 0 < crop_width <= width and 0 < crop_height <= height:
                left = random.randint(0, width - crop_width)
                top = random.randint(0, height - crop_height)
                return left, top, left + crop_width, top + crop_height

        in_ratio = width / height
        if in_ratio < self.ratio[0]:
            crop_width = width
            crop_height = int(round(crop_width / self.ratio[0]))
        elif in_ratio > self.ratio[1]:
            crop_height = height
            crop_width = int(round(crop_height * self.ratio[1]))
        else:
            crop_width = width
            crop_height = height
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        return left, top, left + crop_width, top + crop_height

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.crop(self._get_random_crop(image))
        image = image.resize((self.size, self.size), Image.BILINEAR)
        if random.random() < self.horizontal_flip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        tensor = pil_to_tensor(image)
        return (tensor - self.mean) / self.std


class CREMADVisualDataset(Dataset):
    """CREMA-D visual dataset with deterministic frame-name ordering.

    Each item returns ``(images, label)`` where ``images`` has shape
    ``[3, fps, H, W]`` by default, so the dataloader produces the model
    input shape ``[B, 3, fps, H, W]``. Set ``return_time_first=True`` for
    the legacy ``[fps, 3, H, W]`` layout.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        fps: int = 1,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        samples: list[tuple[str | os.PathLike[str], int]] | None = None,
        rng: np.random.Generator | np.random.RandomState | None = None,
        return_time_first: bool = False,
    ) -> None:
        self.root = Path(root)
        self.image_root = self.root / "Image-01-FPS"
        self.fps = fps
        self.transform = transform or pil_to_tensor
        self.rng = rng
        self.return_time_first = return_time_first

        if samples is None:
            self.samples = self._discover_samples()
        else:
            self.samples = [(Path(path), int(label)) for path, label in samples]

    def _discover_samples(self) -> list[tuple[Path, int]]:
        if not self.image_root.exists():
            raise FileNotFoundError(f"CREMA-D image directory not found: {self.image_root}")

        samples: list[tuple[Path, int]] = []
        for sample_dir in sorted(self.image_root.iterdir()):
            if not sample_dir.is_dir():
                continue

            parts = sample_dir.name.split("_")
            if len(parts) < 3:
                continue

            emotion = parts[2]
            if emotion not in EMOTION_TO_INDEX:
                continue

            samples.append((sample_dir, EMOTION_TO_INDEX[emotion]))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_dir, label = self.samples[index]
        frame_names = sorted(os.listdir(image_dir))
        selected_indices = select_frame_indices(frame_names, self.fps, self.rng)

        images = []
        for selected_index in selected_indices:
            image_path = image_dir / frame_names[int(selected_index)]
            image = Image.open(image_path).convert("RGB")
            images.append(self.transform(image))

        images_tensor = torch.stack(images, dim=0)
        if not self.return_time_first:
            images_tensor = images_tensor.permute(1, 0, 2, 3)

        return images_tensor, torch.tensor(label, dtype=torch.long)


class CREMADAVDataset(Dataset):
    def __init__(
        self,
        samples: list[CREMADSample],
        modality: str = "av",
        fps: int = 1,
        audio_duration: float = 3.0,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        image_transform: Callable[[Image.Image], torch.Tensor] | None = None,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> None:
        if modality not in {"av", "audio", "visual"}:
            raise ValueError(f"Unsupported modality: {modality}")

        self.samples = samples
        self.modality = modality
        self.fps = fps
        self.audio_duration = audio_duration
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.image_transform = image_transform or ResizeToTensorNormalize()
        self.rng = rng

    def __len__(self) -> int:
        return len(self.samples)

    def _load_audio(self, sample: CREMADSample) -> torch.Tensor:
        waveform, sample_rate = read_wav_mono(sample.audio_path)
        return waveform_to_log_spectrogram(
            waveform,
            sample_rate,
            audio_duration=self.audio_duration,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

    def _load_visual(self, sample: CREMADSample) -> torch.Tensor:
        frame_names = sorted(os.listdir(sample.image_dir))
        selected_indices = select_frame_indices(frame_names, self.fps, self.rng)

        images = []
        for selected_index in selected_indices:
            image_path = sample.image_dir / frame_names[int(selected_index)]
            image = Image.open(image_path).convert("RGB")
            images.append(self.image_transform(image))

        return torch.stack(images, dim=0).permute(1, 0, 2, 3)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        item = {
            "label": torch.tensor(sample.label, dtype=torch.long),
        }

        if self.modality in {"av", "audio"}:
            item["audio"] = self._load_audio(sample)
        if self.modality in {"av", "visual"}:
            item["visual"] = self._load_visual(sample)

        return item
