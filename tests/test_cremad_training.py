import math
import json
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
import torch

from cremav1.datasets.cremad import (
    CREMADAVDataset,
    ResizeToTensorNormalize,
    discover_cremad_samples,
    split_samples_from_csv,
    split_samples_random,
    split_samples_by_actor,
)
from cremav1.models.baseline import AVBaseline, AudioBaseline
from cremav1.train_cremad import (
    append_epoch_log,
    create_dataloaders,
    evaluate,
    forward_and_losses,
    plot_history,
    train_one_epoch,
    write_history_json,
)


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


def make_sample(root: Path, name: str) -> None:
    (root / "AudioWAV").mkdir(parents=True, exist_ok=True)
    image_dir = root / "Image-01-FPS" / name
    image_dir.mkdir(parents=True, exist_ok=True)

    write_wav(root / "AudioWAV" / f"{name}.wav")
    for index in range(3):
        image = Image.new("RGB", (16, 16), color=(index * 50, 10, 20))
        image.save(image_dir / f"{index:05d}.jpg")


class CremadTrainingTest(unittest.TestCase):
    def test_discover_and_split_samples_by_actor(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_sample(root, "1001_IEO_HAP_LO")
            make_sample(root, "1001_TIE_SAD_XX")
            make_sample(root, "1002_IEO_ANG_XX")
            make_sample(root, "1003_IEO_DIS_XX")

            samples = discover_cremad_samples(root)
            split = split_samples_by_actor(samples, seed=0, train_ratio=0.5, val_ratio=0.25)

            self.assertEqual(len(samples), 4)
            train_actors = {sample.actor_id for sample in split["train"]}
            val_actors = {sample.actor_id for sample in split["val"]}
            test_actors = {sample.actor_id for sample in split["test"]}
            self.assertFalse(train_actors & val_actors)
            self.assertFalse(train_actors & test_actors)
            self.assertFalse(val_actors & test_actors)

    def test_random_sample_split_keeps_class_balance(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for actor in range(1001, 1007):
                make_sample(root, f"{actor}_IEO_HAP_LO")
                make_sample(root, f"{actor}_IEO_ANG_XX")

            samples = discover_cremad_samples(root)
            split = split_samples_random(samples, seed=0, train_ratio=0.5)

            self.assertEqual(sum(len(part) for part in split.values()), len(samples))
            self.assertEqual(set(split), {"train", "test"})
            for part in split.values():
                labels = {sample.emotion for sample in part}
                self.assertEqual(labels, {"ANG", "HAP"})

    def test_csv_split_uses_fixed_train_and_test_files(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_sample(root, "1001_IEO_HAP_LO")
            make_sample(root, "1002_IEO_ANG_XX")
            make_sample(root, "1003_IEO_SAD_XX")
            csv_root = root / "csv"
            csv_root.mkdir()
            (csv_root / "train.csv").write_text("1001_IEO_HAP_LO,HAP\n1002_IEO_ANG_XX,ANG\n", encoding="utf-8")
            (csv_root / "test.csv").write_text("1003_IEO_SAD_XX,SAD\n", encoding="utf-8")

            samples = discover_cremad_samples(root)
            split = split_samples_from_csv(samples, csv_root)

            self.assertEqual([sample.sample_id for sample in split["train"]], ["1001_IEO_HAP_LO", "1002_IEO_ANG_XX"])
            self.assertEqual([sample.sample_id for sample in split["test"]], ["1003_IEO_SAD_XX"])

    def test_create_dataloaders_uses_test_loader_as_validation_loader(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for actor in range(1001, 1007):
                make_sample(root, f"{actor}_IEO_HAP_LO")
                make_sample(root, f"{actor}_IEO_ANG_XX")
            csv_root = root / "csv"
            csv_root.mkdir()
            (csv_root / "train.csv").write_text(
                "\n".join(f"{actor}_IEO_HAP_LO,HAP" for actor in range(1001, 1007)),
                encoding="utf-8",
            )
            (csv_root / "test.csv").write_text(
                "\n".join(f"{actor}_IEO_ANG_XX,ANG" for actor in range(1001, 1007)),
                encoding="utf-8",
            )

            args = type(
                "Args",
                (),
                {
                    "data_root": str(root),
                    "split_csv_root": str(csv_root),
                    "split_seed": 0,
                    "train_ratio": 0.5,
                    "image_size": 32,
                    "modality": "audio",
                    "fps": 1,
                    "audio_duration": 0.25,
                    "n_fft": 512,
                    "hop_length": 160,
                    "win_length": 400,
                    "seed": 0,
                    "batch_size": 2,
                    "num_workers": 0,
                    "pin_memory": False,
                },
            )()

            _, val_loader, test_loader, sizes = create_dataloaders(args)

            self.assertIs(val_loader, test_loader)
            self.assertEqual(sizes["val"], sizes["test"])

    def test_dataset_returns_audio_visual_and_label(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_sample(root, "1001_IEO_HAP_LO")
            samples = discover_cremad_samples(root)
            dataset = CREMADAVDataset(
                samples,
                modality="av",
                fps=1,
                audio_duration=0.25,
                image_transform=ResizeToTensorNormalize(size=32),
                rng=np.random.default_rng(0),
            )

            item = dataset[0]

            self.assertEqual(tuple(item["audio"].shape), (257, 26))
            self.assertEqual(tuple(item["visual"].shape), (3, 1, 32, 32))
            self.assertEqual(item["label"].item(), 3)

    def test_audio_training_step_runs(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_sample(root, "1001_IEO_HAP_LO")
            make_sample(root, "1002_IEO_ANG_XX")
            samples = discover_cremad_samples(root)
            dataset = CREMADAVDataset(samples, modality="audio", audio_duration=0.25)
            loader = torch.utils.data.DataLoader(dataset, batch_size=2)
            model = AudioBaseline(num_classes=6)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
            device = torch.device("cpu")

            train_metrics = train_one_epoch(model, loader, optimizer, device, "audio")
            eval_metrics = evaluate(model, loader, device, "audio")

            self.assertGreater(train_metrics["loss"], 0.0)
            self.assertGreater(train_metrics["audio_loss"], 0.0)
            self.assertGreaterEqual(eval_metrics["acc"], 0.0)
            self.assertGreater(eval_metrics["audio_loss"], 0.0)
            self.assertLessEqual(eval_metrics["acc"], 1.0)

    def test_av_training_metrics_include_modal_losses(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_sample(root, "1001_IEO_HAP_LO")
            make_sample(root, "1002_IEO_ANG_XX")
            samples = discover_cremad_samples(root)
            dataset = CREMADAVDataset(
                samples,
                modality="av",
                fps=1,
                audio_duration=0.25,
                image_transform=ResizeToTensorNormalize(size=32),
            )
            loader = torch.utils.data.DataLoader(dataset, batch_size=2)
            model = AVBaseline(num_classes=6)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
            device = torch.device("cpu")

            train_metrics = train_one_epoch(model, loader, optimizer, device, "av")
            eval_metrics = evaluate(model, loader, device, "av")

            for metrics in (train_metrics, eval_metrics):
                self.assertGreater(metrics["loss"], 0.0)
                self.assertGreater(metrics["fusion_loss"], 0.0)
                self.assertGreater(metrics["audio_loss"], 0.0)
                self.assertGreater(metrics["visual_loss"], 0.0)

    def test_av_loss_backpropagates_fusion_audio_and_visual_terms(self):
        model = AVBaseline(num_classes=6)
        criterion = torch.nn.CrossEntropyLoss()
        audio = torch.randn(2, 1, 64, 80)
        visual = torch.randn(2, 3, 2, 64, 64)
        labels = torch.tensor([0, 1], dtype=torch.long)

        _, losses = forward_and_losses(model, (audio, visual), labels, "av", criterion)

        expected = losses["fusion_loss"] + losses["audio_loss"] + losses["visual_loss"]
        self.assertTrue(torch.allclose(losses["loss"], expected))

    def test_epoch_log_and_curve_files_are_written(self):
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            args = type("Args", (), {"epochs": 2, "lr": 0.1, "modality": "av"})()
            history = [
                {
                    "epoch": 1,
                    "train": {"loss": 3.0, "acc": 0.2},
                    "val": {"loss": 2.0, "acc": 0.3},
                },
                {
                    "epoch": 2,
                    "train": {"loss": 2.5, "acc": 0.4},
                    "val": {"loss": 1.8, "acc": 0.5},
                },
            ]

            append_epoch_log(output_dir / "history.jsonl", history[0], args, {"train": 2, "test": 1, "val": 1})
            append_epoch_log(output_dir / "history.jsonl", history[1], args, {"train": 2, "test": 1, "val": 1})
            write_history_json(output_dir / "history.json", history, args, {"train": 2, "test": 1, "val": 1})
            plot_history(history, output_dir / "curves.png")

            lines = (output_dir / "history.jsonl").read_text(encoding="utf-8").strip().splitlines()
            history_json = json.loads((output_dir / "history.json").read_text(encoding="utf-8"))
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["args"]["lr"], 0.1)
            self.assertEqual(history_json["args"]["modality"], "av")
            self.assertEqual(len(history_json["epochs"]), 2)
            self.assertTrue((output_dir / "curves.png").exists())
            self.assertGreater((output_dir / "curves.png").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
