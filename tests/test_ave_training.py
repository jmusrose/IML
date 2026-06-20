import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from AV_v1.datasets.ave import AVEDataset, discover_ave_samples


def write_pkl(path: Path, shape: tuple[int, int] = (257, 1004)) -> None:
    spec = np.random.rand(*shape).astype(np.float32)
    with open(path, "wb") as fh:
        pickle.dump(spec, fh)


def make_ave_root(root: Path) -> None:
    """Create a minimal AVE dataset matching the real directory layout."""
    audio_dir = root / "Audio-1004-SE"
    audio_dir.mkdir(parents=True)

    samples = [
        ("Church bell", "video_001"),
        ("Dog", "video_002"),
        ("Church bell", "video_003"),
    ]
    for _, video_id in samples:
        write_pkl(audio_dir / f"{video_id}.pkl")
        image_dir = root / "Image-01-FPS-SE" / video_id
        image_dir.mkdir(parents=True)
        for i in range(4):
            img = Image.new("RGB", (16, 16), color=(i * 40, 20, 10))
            img.save(image_dir / f"{i:05d}.jpg")

    (root / "trainSet.txt").write_text(
        "Church bell&video_001&good&0&10\nDog&video_002&good&0&10\n", encoding="utf-8"
    )
    (root / "valSet.txt").write_text("Church bell&video_003&good&0&10\n", encoding="utf-8")
    (root / "testSet.txt").write_text("Dog&video_002&good&0&10\n", encoding="utf-8")


class AVEDatasetTest(unittest.TestCase):
    def test_discover_ave_samples(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_ave_root(root)

            train_samples, class_to_idx = discover_ave_samples(root, split="train")

            self.assertEqual(len(train_samples), 2)
            self.assertIn("Church bell", class_to_idx)
            self.assertIn("Dog", class_to_idx)
            self.assertEqual(len(class_to_idx), 2)

    def test_dataset_item_av_shape(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_ave_root(root)

            samples, _ = discover_ave_samples(root, split="train")
            dataset = AVEDataset(
                samples,
                modality="av",
                mode="train",
                use_video_frames=2,
                image_size=32,
            )

            item = dataset[0]

            self.assertIn("audio", item)
            self.assertIn("visual", item)
            self.assertIn("label", item)
            self.assertEqual(tuple(item["audio"].shape), (1, 257, 1004))
            self.assertEqual(tuple(item["visual"].shape), (3, 2, 32, 32))

    def test_dataset_audio_only(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_ave_root(root)

            samples, _ = discover_ave_samples(root, split="train")
            dataset = AVEDataset(samples, modality="audio", mode="test")

            item = dataset[0]

            self.assertIn("audio", item)
            self.assertNotIn("visual", item)
            self.assertEqual(tuple(item["audio"].shape), (1, 257, 1004))

    def test_val_and_test_splits(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_ave_root(root)

            val_samples, _ = discover_ave_samples(root, split="val")
            test_samples, _ = discover_ave_samples(root, split="test")

            self.assertEqual(len(val_samples), 1)
            self.assertEqual(len(test_samples), 1)
            self.assertEqual(val_samples[0].category, "Church bell")
            self.assertEqual(test_samples[0].category, "Dog")

    def test_label_consistency_across_splits(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_ave_root(root)

            _, train_idx = discover_ave_samples(root, split="train")
            _, val_idx = discover_ave_samples(root, split="val")
            _, test_idx = discover_ave_samples(root, split="test")

            self.assertEqual(train_idx, val_idx)
            self.assertEqual(train_idx, test_idx)


if __name__ == "__main__":
    unittest.main()
