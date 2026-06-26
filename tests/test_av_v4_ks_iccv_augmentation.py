import sys
from argparse import Namespace
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_av_v4_ks_train_loader_uses_iccv_visual_augmentation(monkeypatch):
    from AV_v4 import train_ks
    from AV_v4.datasets import KSTrainImageTransform, ResizeToTensorNormalize

    class DummyDataset:
        def __init__(self, samples, *, mode, image_transform, **kwargs):
            self.samples = samples
            self.mode = mode
            self.image_transform = image_transform

        def __len__(self):
            return len(self.samples)

    def fake_discover_ks_samples(data_root, class_file, mode):
        return ([object()], {"class_a": 0})

    monkeypatch.setattr(train_ks, "discover_ks_samples", fake_discover_ks_samples)
    monkeypatch.setattr(train_ks, "KSDataset", DummyDataset)

    args = Namespace(
        data_root="unused",
        class_file="unused",
        modality="av",
        use_video_frames=3,
        audio_duration=5.0,
        n_fft=256,
        hop_length=128,
        win_length=256,
        image_size=224,
        seed=0,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    )

    train_loader, val_loader, test_loader, sizes = train_ks.create_dataloaders(args)

    train_transform = train_loader.dataset.image_transform
    test_transform = test_loader.dataset.image_transform
    assert isinstance(train_transform, KSTrainImageTransform)
    assert train_transform.size == 224
    assert train_transform.scale == (0.08, 1.0)
    assert train_transform.ratio == (3.0 / 4.0, 4.0 / 3.0)
    assert train_transform.horizontal_flip_prob == 0.5
    assert isinstance(test_transform, ResizeToTensorNormalize)
    assert not isinstance(test_transform, KSTrainImageTransform)
    assert val_loader is test_loader
    assert sizes == {"train": 1, "test": 1, "val": 1}
