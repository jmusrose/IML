import math
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from cremav1.datasets.ks import KSDataset, discover_ks_samples, load_ks_classes


def write_wav(path: Path, frequency: float = 440.0, sample_rate: int = 16000) -> None:
    duration = 0.2
    values = []
    for index in range(int(sample_rate * duration)):
        sample = 0.2 * math.sin(2 * math.pi * frequency * index / sample_rate)
        values.append(int(sample * 32767))

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(np.asarray(values, dtype=np.int16).tobytes())


def make_ks_sample(root: Path, split: str, class_name: str, sample_id: str) -> None:
    image_split = "train_img" if split == "train" else "val_img"
    audio_split = "train" if split == "train" else "test"
    image_dir = root / "visual" / image_split / "Image-01-FPS" / class_name / sample_id
    audio_dir = root / "audio" / audio_split / class_name
    image_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    write_wav(audio_dir / f"{sample_id}.wav")
    for index in range(4):
        image = Image.new("RGB", (16, 16), color=(index * 40, 10, 20))
        image.save(image_dir / f"{index:05d}.jpg")


class KSTrainingTest(unittest.TestCase):
    def test_load_ks_classes_normalizes_spaces(self):
        with TemporaryDirectory() as tmpdir:
            class_file = Path(tmpdir) / "class.txt"
            class_file.write_text("blowing nose, bowling,tying_tie", encoding="utf-8")

            classes = load_ks_classes(class_file)

            self.assertEqual(classes, ["blowing_nose", "bowling", "tying_tie"])

    def test_discover_ks_samples_and_dataset_item(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            class_file = root / "class.txt"
            class_file.write_text("blowing nose, bowling", encoding="utf-8")
            make_ks_sample(root, "train", "blowing_nose", "sample_a")
            make_ks_sample(root, "test", "bowling", "sample_b")

            train_samples, class_to_idx = discover_ks_samples(root, class_file, mode="train")
            test_samples, _ = discover_ks_samples(root, class_file, mode="test")
            dataset = KSDataset(train_samples, modality="av", use_video_frames=2, audio_duration=0.25, image_size=32)

            item = dataset[0]

            self.assertEqual(class_to_idx["blowing_nose"], 0)
            self.assertEqual(class_to_idx["bowling"], 1)
            self.assertEqual(len(train_samples), 1)
            self.assertEqual(len(test_samples), 1)
            self.assertEqual(tuple(item["visual"].shape), (3, 2, 32, 32))
            self.assertEqual(item["label"].item(), 0)


if __name__ == "__main__":
    unittest.main()
