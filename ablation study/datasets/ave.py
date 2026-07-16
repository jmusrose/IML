from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .cremad import ResizeToTensorNormalize


@dataclass(frozen=True)
class AVESample:
    sample_id: str
    category: str
    label: int
    audio_path: Path
    image_dir: Path


def _parse_ave_line(line: str) -> tuple[str, str] | None:
    """Parse a line from AVE split files.

    Format: ``category&video_id&quality&start&end``
    Example: ``Church bell&c---zaDCTaE&good&0&10``
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split("&")
    if len(parts) < 2:
        return None
    return parts[0].strip(), parts[1].strip()


def _build_class_index(split_files: list[Path]) -> dict[str, int]:
    categories: set[str] = set()
    for path in split_files:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                parsed = _parse_ave_line(line)
                if parsed is not None:
                    categories.add(parsed[0])
    return {cat: idx for idx, cat in enumerate(sorted(categories))}


_SPLIT_FILENAMES = {
    "train": "trainSet.txt",
    "val": "valSet.txt",
    "test": "testSet.txt",
}
_EXPECTED_AUDIO_SHAPE = (257, 1004)


def _read_ave_audio_array(path: Path) -> np.ndarray:
    with path.open("rb") as fh:
        spec = pickle.load(fh)

    array = np.asarray(spec, dtype=np.float32)
    if array.shape != _EXPECTED_AUDIO_SHAPE:
        raise ValueError(
            f"Expected AVE audio spectrogram shape {_EXPECTED_AUDIO_SHAPE}, "
            f"got {tuple(array.shape)} for {path}."
        )
    return array


def _has_valid_ave_audio(path: Path) -> bool:
    try:
        _read_ave_audio_array(path)
    except (EOFError, OSError, pickle.PickleError, TypeError, ValueError):
        return False
    return True


def discover_ave_samples(
    root: str | os.PathLike[str],
    split: str = "train",
    min_frames: int = 1,
) -> tuple[list[AVESample], dict[str, int]]:
    """Discover AVE samples for one split.

    Expected directory layout::

        <root>/
          Audio-1004-SE/
            <video_id>.pkl      # pre-extracted spectrogram, shape (257, 1004)
          Image-01-FPS-SE/
            <video_id>/
              00000.jpg
              00029.jpg
              ...
          trainSet.txt          # format: category&video_id&quality&start&end
          valSet.txt
          testSet.txt

    Returns ``(samples, category_to_idx)``.
    """
    if split not in _SPLIT_FILENAMES:
        raise ValueError(f"Unsupported AVE split: {split!r}. Choose from {list(_SPLIT_FILENAMES)}")

    root = Path(root)
    split_file = root / _SPLIT_FILENAMES[split]
    if not split_file.exists():
        raise FileNotFoundError(f"AVE split file not found: {split_file}")

    audio_root = root / "Audio-1004-SE"
    image_root = root / "Image-01-FPS-SE"
    if not audio_root.exists():
        raise FileNotFoundError(f"AVE audio directory not found: {audio_root}")
    if not image_root.exists():
        raise FileNotFoundError(f"AVE visual directory not found: {image_root}")

    all_split_files = [
        root / fname
        for fname in _SPLIT_FILENAMES.values()
        if (root / fname).exists()
    ]
    category_to_idx = _build_class_index(all_split_files)

    samples: list[AVESample] = []
    with split_file.open(encoding="utf-8") as fh:
        for line in fh:
            parsed = _parse_ave_line(line)
            if parsed is None:
                continue
            category, video_id = parsed

            audio_path = audio_root / f"{video_id}.pkl"
            if not audio_path.exists():
                continue
            if not _has_valid_ave_audio(audio_path):
                continue

            image_dir = image_root / video_id
            if not image_dir.exists() or not image_dir.is_dir():
                continue

            frame_files = [p for p in image_dir.iterdir() if p.is_file()]
            if len(frame_files) < min_frames:
                continue

            label = category_to_idx.get(category)
            if label is None:
                continue

            samples.append(
                AVESample(
                    sample_id=video_id,
                    category=category,
                    label=label,
                    audio_path=audio_path,
                    image_dir=image_dir,
                )
            )

    return samples, category_to_idx


def select_ave_frame_indices(
    frame_count: int,
    use_video_frames: int,
    mode: str,
) -> np.ndarray:
    if frame_count < 1:
        raise ValueError("Cannot select frames from an empty AVE frame directory.")
    if use_video_frames < 1:
        raise ValueError(f"use_video_frames must be >= 1, got {use_video_frames}.")

    replace = frame_count < use_video_frames
    if mode == "train":
        indices = np.random.choice(frame_count, size=use_video_frames, replace=replace)
        return np.sort(indices).astype(np.int64)

    if replace:
        return np.linspace(0, frame_count - 1, num=use_video_frames).round().astype(np.int64)
    return np.linspace(0, frame_count - 1, num=use_video_frames, dtype=np.int64)


class AVEDataset(Dataset):
    """AVE dataset with pre-extracted audio spectrograms (pkl) and video frames.

    Audio pkl files contain numpy arrays of shape ``(257, 1004)`` (freq × time).
    The dataset returns them as ``(1, 257, 1004)`` tensors so they match the
    expected ``(B, 1, H, W)`` input of :class:`AudioBaseline`.
    """

    def __init__(
        self,
        samples: list[AVESample],
        modality: str = "av",
        mode: str = "train",
        use_video_frames: int = 10,
        image_size: int = 224,
        image_transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        if modality not in {"av", "audio", "visual"}:
            raise ValueError(f"Unsupported modality: {modality}")
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported AVE mode: {mode}")

        self.samples = samples
        self.modality = modality
        self.mode = mode
        self.use_video_frames = use_video_frames
        self.image_transform = image_transform or ResizeToTensorNormalize(size=image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_audio(self, sample: AVESample) -> torch.Tensor:
        spec = _read_ave_audio_array(sample.audio_path)
        tensor = torch.from_numpy(np.ascontiguousarray(spec)).clone()
        return tensor.unsqueeze(0)  # (1, freq, time)

    def _load_visual(self, sample: AVESample) -> torch.Tensor:
        frame_paths = sorted(path for path in sample.image_dir.iterdir() if path.is_file())
        indices = select_ave_frame_indices(len(frame_paths), self.use_video_frames, self.mode)
        images = []
        for index in indices:
            image = Image.open(frame_paths[int(index)]).convert("RGB")
            images.append(self.image_transform(image))
        return torch.stack(images, dim=0).permute(1, 0, 2, 3)  # (C, T, H, W)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        item = {"label": torch.tensor(sample.label, dtype=torch.long)}
        if self.modality in {"av", "audio"}:
            item["audio"] = self._load_audio(sample)
        if self.modality in {"av", "visual"}:
            item["visual"] = self._load_visual(sample)
        return item
