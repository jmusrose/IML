from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class RGBSample:
    sample_id: str
    class_name: str
    label: int
    image_path: Path


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.expand_dims(array, axis=-1)
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(array))


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


class RGBTrainImageTransform(ResizeToTensorNormalize):
    def __init__(
        self,
        size: int = 224,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        scale: tuple[float, float] = (0.7, 1.0),
        ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        horizontal_flip_prob: float = 0.5,
        color_jitter: float = 0.2,
    ) -> None:
        super().__init__(size=size, mean=mean, std=std)
        self.scale = scale
        self.ratio = ratio
        self.horizontal_flip_prob = horizontal_flip_prob
        self.color_jitter = color_jitter

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

    def _apply_color_jitter(self, image: Image.Image) -> Image.Image:
        strength = max(0.0, float(self.color_jitter))
        if strength == 0.0:
            return image
        for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
            factor = random.uniform(max(0.0, 1.0 - strength), 1.0 + strength)
            image = enhancer(image).enhance(factor)
        return image

    def _normalize(self, image: Image.Image) -> torch.Tensor:
        tensor = pil_to_tensor(image)
        return (tensor - self.mean) / self.std

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.crop(self._get_random_crop(image))
        image = image.resize((self.size, self.size), Image.BILINEAR)
        if random.random() < self.horizontal_flip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image = self._apply_color_jitter(image)
        return self._normalize(image)

    def paired_call(self, rgb_image: Image.Image, depth_image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        width = min(rgb_image.size[0], depth_image.size[0])
        height = min(rgb_image.size[1], depth_image.size[1])
        if rgb_image.size != (width, height):
            rgb_image = rgb_image.crop((0, 0, width, height))
        if depth_image.size != (width, height):
            depth_image = depth_image.crop((0, 0, width, height))

        crop_box = self._get_random_crop(rgb_image)
        rgb_image = rgb_image.crop(crop_box)
        depth_image = depth_image.crop(crop_box)
        rgb_image = rgb_image.resize((self.size, self.size), Image.BILINEAR)
        depth_image = depth_image.resize((self.size, self.size), Image.BILINEAR)
        if random.random() < self.horizontal_flip_prob:
            rgb_image = rgb_image.transpose(Image.FLIP_LEFT_RIGHT)
            depth_image = depth_image.transpose(Image.FLIP_LEFT_RIGHT)
        rgb_image = self._apply_color_jitter(rgb_image)
        return self._normalize(rgb_image), self._normalize(depth_image)


def _available_splits(root: Path) -> list[str]:
    return [name for name in ("train", "val", "test") if (root / name).is_dir()]


def _build_class_index(root: Path) -> dict[str, int]:
    classes: set[str] = set()
    for split in _available_splits(root):
        split_dir = root / split
        classes.update(path.name for path in split_dir.iterdir() if path.is_dir())
    if not classes:
        raise ValueError(f"No class directories found under {root}.")
    return {class_name: index for index, class_name in enumerate(sorted(classes))}


def discover_rgb_samples(
    root: str | os.PathLike[str],
    split: str = "train",
) -> tuple[list[RGBSample], dict[str, int]]:
    root = Path(root)
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"RGB split directory not found: {split_dir}")
    if not split_dir.is_dir():
        raise NotADirectoryError(f"RGB split path is not a directory: {split_dir}")

    class_to_idx = _build_class_index(root)
    samples: list[RGBSample] = []
    for class_name, label in class_to_idx.items():
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            continue
        for image_path in sorted(path for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS):
            samples.append(
                RGBSample(
                    sample_id=image_path.stem,
                    class_name=class_name,
                    label=label,
                    image_path=image_path,
                )
            )
    return samples, class_to_idx


class RGBImageDataset(Dataset):
    def __init__(
        self,
        samples: list[RGBSample],
        mode: str = "train",
        image_size: int = 224,
        image_transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported RGB dataset mode: {mode}")
        self.samples = samples
        self.mode = mode
        if image_transform is None:
            image_transform = RGBTrainImageTransform(size=image_size) if mode == "train" else ResizeToTensorNormalize(size=image_size)
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.samples)

    def _load_modalities(self, image_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        if width < 2:
            raise ValueError(f"Cannot split image with width < 2: {image_path}")
        midpoint = width // 2
        rgb_image = image.crop((0, 0, midpoint, height))
        depth_image = image.crop((midpoint, 0, width, height))
        if hasattr(self.image_transform, "paired_call"):
            return self.image_transform.paired_call(rgb_image, depth_image)
        return self.image_transform(rgb_image), self.image_transform(depth_image)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        rgb, depth = self._load_modalities(sample.image_path)
        return {
            "rgb": rgb,
            "depth": depth,
            "label": torch.tensor(sample.label, dtype=torch.long),
        }
