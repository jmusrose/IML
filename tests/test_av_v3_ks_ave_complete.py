import pickle
import wave
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image


def _write_wav(path: Path, sample_rate: int = 16000) -> None:
    values = np.zeros(int(sample_rate * 0.2), dtype=np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(values.tobytes())


def _make_ks_root(root: Path) -> Path:
    class_file = root / "class.txt"
    class_file.write_text("class one,class two", encoding="utf-8")
    for split, image_split, audio_split, class_name, sample_id in (
        ("train", "train_img", "train", "class_one", "sample_a"),
        ("test", "val_img", "test", "class_two", "sample_b"),
    ):
        del split
        image_dir = root / "visual" / image_split / "Image-01-FPS" / class_name / sample_id
        audio_dir = root / "audio" / audio_split / class_name
        image_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)
        _write_wav(audio_dir / f"{sample_id}.wav")
        for index in range(3):
            Image.new("RGB", (16, 16), color=(index * 30, 10, 20)).save(image_dir / f"{index:05d}.jpg")
    return class_file


def _write_ave_audio(path: Path) -> None:
    spec = np.asfortranarray(np.random.rand(257, 1004).astype(np.float32))
    with path.open("wb") as handle:
        pickle.dump(spec, handle)


def _make_ave_root(root: Path) -> None:
    (root / "Audio-1004-SE").mkdir(parents=True)
    for video_id in ("video_001", "video_002", "video_003"):
        _write_ave_audio(root / "Audio-1004-SE" / f"{video_id}.pkl")
        image_dir = root / "Image-01-FPS-SE" / video_id
        image_dir.mkdir(parents=True)
        for index in range(3):
            Image.new("RGB", (16, 16), color=(index * 30, 20, 10)).save(image_dir / f"{index:05d}.jpg")

    (root / "trainSet.txt").write_text(
        "Church bell&video_001&good&0&10\nDog&video_002&good&0&10\n",
        encoding="utf-8",
    )
    (root / "valSet.txt").write_text("Church bell&video_003&good&0&10\n", encoding="utf-8")
    (root / "testSet.txt").write_text("Dog&video_002&good&0&10\n", encoding="utf-8")


def _minimal_metrics(acc: float = 0.5) -> dict[str, float]:
    return {
        "loss": 1.0,
        "fusion_loss": 0.4,
        "audio_loss": 0.3,
        "visual_loss": 0.3,
        "acc": acc,
        "macro_f1": 0.4,
        "audio_acc": 0.5,
        "visual_acc": 0.5,
    }


def test_av_v3_ks_create_dataloaders_uses_v3_dataset():
    from AV_v3 import train_ks
    from AV_v3.datasets import KSDataset

    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        class_file = _make_ks_root(root)
        args = Namespace(
            data_root=str(root),
            class_file=str(class_file),
            modality="av",
            image_size=32,
            use_video_frames=2,
            audio_duration=0.25,
            n_fft=256,
            hop_length=128,
            win_length=256,
            seed=0,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
        )

        train_loader, val_loader, test_loader, sizes = train_ks.create_dataloaders(args)

        assert isinstance(train_loader.dataset, KSDataset)
        assert isinstance(test_loader.dataset, KSDataset)
        assert val_loader is test_loader
        assert sizes == {"train": 1, "test": 1, "val": 1}
        assert args.num_classes == 2


def test_av_v3_ave_create_dataloaders_uses_v3_dataset():
    from AV_v3 import train_ave
    from AV_v3.datasets import AVEDataset

    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _make_ave_root(root)
        args = Namespace(
            data_root=str(root),
            modality="av",
            image_size=32,
            use_video_frames=2,
            seed=0,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
        )

        train_loader, val_loader, test_loader, sizes = train_ave.create_dataloaders(args)

        assert isinstance(train_loader.dataset, AVEDataset)
        assert isinstance(test_loader.dataset, AVEDataset)
        assert val_loader is test_loader
        assert sizes == {"train": 2, "val": 1, "test": 1}
        assert args.num_classes == 2


def test_av_v3_ks_and_ave_parse_args_accept_argv():
    from AV_v3 import train_ave, train_ks

    with TemporaryDirectory() as tmpdir:
        class_file = Path(tmpdir) / "class.txt"
        class_file.write_text("a,b", encoding="utf-8")

        ks_args = train_ks.parse_args(["--class-file", str(class_file), "--device", "cpu"])
        ave_args = train_ave.parse_args(["--device", "cpu"])

    assert ks_args.device == "cpu"
    assert ks_args.num_classes == 2
    assert ks_args.lr_scheduler == "multistep"
    assert ks_args.lr_decay_step == "[60]"
    assert ks_args.lr_decay_ratio == 0.1
    assert ave_args.device == "cpu"
    assert ave_args.num_classes == 28
    assert ave_args.lr_scheduler == "multistep"
    assert ave_args.lr_decay_step == "[60]"
    assert ave_args.lr_decay_ratio == 0.1


def test_av_v3_ks_and_ave_default_to_multistep_scheduler():
    from AV_v3 import train_ave, train_ks

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.002)

    for module in (train_ks, train_ave):
        args = Namespace(
            epochs=200,
            lr_scheduler="multistep",
            lr_decay_step="[60]",
            lr_decay_ratio=0.1,
        )
        scheduler = module.build_scheduler(optimizer, args)

        assert isinstance(scheduler, torch.optim.lr_scheduler.MultiStepLR)
        assert scheduler.milestones == {60: 1}
        assert scheduler.gamma == 0.1


def test_av_v3_ks_and_ave_use_shared_v3_training_module():
    root = Path(__file__).resolve().parents[1]
    ks_source = (root / "AV_v3" / "train_ks.py").read_text(encoding="utf-8")
    ave_source = (root / "AV_v3" / "train_ave.py").read_text(encoding="utf-8")

    assert "from AV_v3.train_cremad import" not in ks_source
    assert "from AV_v3.train_cremad import" not in ave_source
    assert "from AV_v3.training import" in ks_source
    assert "from AV_v3.training import" in ave_source


def test_av_v3_ks_run_training_writes_artifacts_with_mocked_epoch():
    from AV_v3 import train_ks

    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        args = Namespace(
            seed=0,
            deterministic=False,
            device="cpu",
            modality="av",
            num_classes=2,
            lr=0.01,
            momentum=0.9,
            weight_decay=0.0,
            epochs=1,
            output_dir=str(output_dir),
            no_progress=True,
            fgm=False,
            fgm_lambda=0.5,
            fgm_tau=1.0,
            fgm_momentum=0.9,
            fgm_warmup_steps=0,
            class_file="dummy.txt",
        )
        loader = [object()]
        metrics = _minimal_metrics()

        with patch.object(train_ks, "create_dataloaders", return_value=(loader, loader, loader, {"train": 1, "val": 1, "test": 1})), patch.object(
            train_ks,
            "build_model",
            return_value=torch.nn.Linear(1, 2),
        ), patch.object(
            train_ks,
            "train_one_epoch",
            return_value=metrics,
        ), patch.object(
            train_ks,
            "evaluate",
            return_value=metrics,
        ), patch.object(
            train_ks,
            "load_ks_classes",
            return_value=["a", "b"],
        ):
            result = train_ks.run_training(args)

        assert result["best_epoch"] == 1.0
        assert (output_dir / "best.pt").exists()
        assert (output_dir / "config.json").exists()
        assert (output_dir / "history.json").exists()
        assert (output_dir / "history.jsonl").exists()
        assert (output_dir / "metrics.json").exists()


def test_av_v3_ave_run_training_writes_artifacts_with_mocked_epoch():
    from AV_v3 import train_ave

    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        args = Namespace(
            seed=0,
            deterministic=False,
            device="cpu",
            modality="av",
            num_classes=2,
            lr=0.01,
            momentum=0.9,
            weight_decay=0.0,
            epochs=1,
            output_dir=str(output_dir),
            no_progress=True,
            fgm=False,
            fgm_lambda=0.5,
            fgm_tau=1.0,
            fgm_momentum=0.9,
            fgm_warmup_steps=0,
        )
        loader = [object()]
        metrics = _minimal_metrics()

        with patch.object(train_ave, "create_dataloaders", return_value=(loader, loader, loader, {"train": 1, "val": 1, "test": 1})), patch.object(
            train_ave,
            "build_model",
            return_value=torch.nn.Linear(1, 2),
        ), patch.object(
            train_ave,
            "train_one_epoch",
            return_value=metrics,
        ), patch.object(
            train_ave,
            "evaluate",
            return_value=metrics,
        ):
            result = train_ave.run_training(args)

        assert result["best_epoch"] == 1.0
        assert (output_dir / "best.pt").exists()
        assert (output_dir / "config.json").exists()
        assert (output_dir / "history.json").exists()
        assert (output_dir / "history.jsonl").exists()
        assert (output_dir / "metrics.json").exists()
