import argparse
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision.models.resnet import ResNet

from RGB_v4.datasets import RGBImageDataset, RGBTrainImageTransform, discover_rgb_samples
from RGB_v4.evaluate_robustness import (
    apply_view_noise,
    evaluate_conditions,
    format_robustness_table,
)
from RGB_v4.train_nyud2 import create_dataloaders as create_nyud2_dataloaders
from RGB_v4.train_nyud2 import parse_args as parse_nyud2_args
from RGB_v4.train_nyud2 import run_training as run_nyud2_training
from RGB_v4.train_sunrgbd import create_dataloaders as create_sunrgbd_dataloaders
from RGB_v4.training import build_model, evaluate, forward_and_losses, train_one_epoch


def make_rgb_root(root: Path, split_names: tuple[str, ...] = ("train", "val", "test")) -> None:
    for split in split_names:
        for class_name, color in [("kitchen", (200, 10, 10)), ("office", (10, 200, 10))]:
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for index in range(2):
                image = Image.new("RGB", (40, 24), color=color)
                image.save(class_dir / f"{index:03d}.png")


class RGBV4TrainingTest(unittest.TestCase):
    def test_discovers_class_folder_samples_with_stable_labels(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_rgb_root(root)

            samples, class_to_idx = discover_rgb_samples(root, split="train")

            self.assertEqual(class_to_idx, {"kitchen": 0, "office": 1})
            self.assertEqual(len(samples), 4)
            self.assertEqual(samples[0].label, class_to_idx[samples[0].class_name])

    def test_dataset_splits_combined_image_into_two_rgb_modalities(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_rgb_root(root)
            samples, _ = discover_rgb_samples(root, split="train")

            dataset = RGBImageDataset(samples, mode="test", image_size=32)
            item = dataset[0]

            self.assertEqual(tuple(item["rgb"].shape), (3, 32, 32))
            self.assertEqual(tuple(item["depth"].shape), (3, 32, 32))
            self.assertEqual(item["rgb"].dtype, torch.float32)
            self.assertEqual(item["depth"].dtype, torch.float32)
            self.assertEqual(item["label"].dtype, torch.long)

    def test_train_transform_applies_shared_geometry_to_both_views(self):
        image = Image.new("RGB", (40, 24), color=(120, 40, 10))
        transform = RGBTrainImageTransform(
            size=32,
            scale=(0.6, 0.6),
            ratio=(1.0, 1.0),
            horizontal_flip_prob=1.0,
            color_jitter=0.0,
        )

        rgb, depth = transform.paired_call(image, image.copy())

        self.assertTrue(torch.equal(rgb, depth))

    def test_train_dataloader_accepts_tunable_augmentation_args(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_rgb_root(root)
            args = argparse.Namespace(
                data_root=str(root),
                image_size=32,
                batch_size=2,
                num_workers=0,
                pin_memory=False,
                seed=7,
                aug_scale=[0.8, 1.0],
                aug_ratio=[0.9, 1.1],
                aug_hflip_prob=0.25,
                aug_color_jitter=0.15,
            )

            train_loader, _, _ = create_nyud2_dataloaders(args)

            transform = train_loader.dataset.image_transform
            self.assertEqual(transform.scale, (0.8, 1.0))
            self.assertEqual(transform.ratio, (0.9, 1.1))
            self.assertEqual(transform.horizontal_flip_prob, 0.25)
            self.assertEqual(transform.color_jitter, 0.15)

    def test_nyud2_dataloaders_use_train_test_splits(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_rgb_root(root)
            args = argparse.Namespace(
                data_root=str(root),
                image_size=32,
                batch_size=2,
                num_workers=0,
                pin_memory=False,
                seed=7,
            )

            train_loader, test_loader, sizes = create_nyud2_dataloaders(args)

            self.assertEqual(args.num_classes, 2)
            self.assertEqual(sizes, {"train": 4, "test": 4})
            batch = next(iter(train_loader))
            self.assertEqual(tuple(batch["rgb"].shape), (2, 3, 32, 32))
            self.assertEqual(tuple(batch["depth"].shape), (2, 3, 32, 32))
            self.assertEqual(len(test_loader.dataset), 4)

    def test_sunrgbd_dataloaders_use_train_test_splits(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_rgb_root(root, split_names=("train", "test"))
            args = argparse.Namespace(
                data_root=str(root),
                image_size=32,
                batch_size=2,
                num_workers=0,
                pin_memory=False,
                seed=11,
            )

            _, test_loader, sizes = create_sunrgbd_dataloaders(args)

            self.assertEqual(args.num_classes, 2)
            self.assertEqual(sizes, {"train": 4, "test": 4})
            self.assertEqual(len(test_loader.dataset), 4)

    def test_one_epoch_training_and_evaluation_on_cpu(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_rgb_root(root)
            samples, _ = discover_rgb_samples(root, split="train")
            dataset = RGBImageDataset(samples, mode="test", image_size=32)
            loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
            device = torch.device("cpu")
            model = build_model(num_classes=2, pretrained=False).to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

            train_metrics = train_one_epoch(model, loader, optimizer, device, show_progress=False)
            eval_metrics = evaluate(model, loader, device, show_progress=False)

            self.assertIn("loss", train_metrics)
            self.assertIn("fusion_loss", train_metrics)
            self.assertIn("rgb_loss", train_metrics)
            self.assertIn("depth_loss", train_metrics)
            self.assertIn("rgb_acc", train_metrics)
            self.assertIn("depth_acc", train_metrics)
            self.assertIn("acc", train_metrics)
            self.assertIn("macro_f1", eval_metrics)
            self.assertGreaterEqual(eval_metrics["acc"], 0.0)

    def test_model_uses_torchvision_resnet18_backbones(self):
        model = build_model(num_classes=2, pretrained=False)

        self.assertIsInstance(model.rgb_net, ResNet)
        self.assertIsInstance(model.depth_net, ResNet)
        self.assertIsInstance(model.rgb_net.fc, torch.nn.Identity)
        self.assertIsInstance(model.depth_net.fc, torch.nn.Identity)

    def test_probe_losses_update_each_backbone_by_default(self):
        model = build_model(num_classes=2, pretrained=False)
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        rgb = torch.randn(2, 3, 32, 32)
        depth = torch.randn(2, 3, 32, 32)
        labels = torch.tensor([0, 1], dtype=torch.long)

        _, losses = forward_and_losses(
            model,
            (rgb, depth),
            labels,
            criterion,
            rgb_loss_weight=1.0,
            depth_loss_weight=1.0,
        )
        (losses["rgb_loss"] + losses["depth_loss"]).backward()

        self.assertIsNotNone(model.rgb_net.conv1.weight.grad)
        self.assertIsNotNone(model.depth_net.conv1.weight.grad)
        self.assertGreater(float(model.rgb_net.conv1.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.depth_net.conv1.weight.grad.abs().sum()), 0.0)

    def test_training_args_default_to_pretrained_backbone(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_rgb_root(root)

            args = parse_nyud2_args(["--data-root", str(root)])
            disabled = parse_nyud2_args(["--data-root", str(root), "--no-pretrained"])

            self.assertTrue(args.pretrained)
            self.assertFalse(disabled.pretrained)

    def test_apply_view_noise_only_changes_selected_view(self):
        batch = {
            "rgb": torch.zeros(2, 3, 8, 8),
            "depth": torch.zeros(2, 3, 8, 8),
            "label": torch.tensor([0, 1]),
        }

        noised = apply_view_noise(
            batch,
            view="depth",
            noise_type="gaussian",
            epsilon=10.0,
            generator=torch.Generator().manual_seed(0),
        )

        self.assertTrue(torch.equal(noised["rgb"], batch["rgb"]))
        self.assertFalse(torch.equal(noised["depth"], batch["depth"]))
        self.assertTrue(torch.equal(noised["label"], batch["label"]))

    def test_evaluate_conditions_reports_clean_and_noise_metrics(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_rgb_root(root)
            samples, _ = discover_rgb_samples(root, split="train")
            dataset = RGBImageDataset(samples, mode="test", image_size=32)
            loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
            device = torch.device("cpu")
            model = build_model(num_classes=2, pretrained=False).to(device)

            results = evaluate_conditions(
                model,
                loader,
                device,
                noise_view="rgb",
                epsilons=(5.0,),
                seed=123,
                show_progress=False,
            )
            table = format_robustness_table("toy", results)

            self.assertIn("clean", results)
            self.assertIn("gaussian@5", results)
            self.assertIn("salt-pepper@5", results)
            self.assertIn("| toy |", table)
            self.assertIn("Gaussian@5", table)

    def test_run_training_writes_robustness_results_after_clean_test(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "data"
            output_dir = Path(tmpdir) / "runs"
            make_rgb_root(root)
            args = parse_nyud2_args(
                [
                    "--data-root",
                    str(root),
                    "--output-dir",
                    str(output_dir),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "2",
                    "--image-size",
                    "32",
                    "--num-workers",
                    "0",
                    "--device",
                    "cpu",
                    "--no-pin-memory",
                    "--no-progress",
                    "--no-pretrained",
                    "--robustness-epsilons",
                    "5",
                ]
            )

            result = run_nyud2_training(args)

            run_dir = Path(args.output_dir)
            self.assertIn("best_test_acc", result)
            self.assertNotIn("best_val_acc", result)
            self.assertIn("robustness", result)
            self.assertTrue((run_dir / "best_checkpoint.pt").exists())
            self.assertTrue((run_dir / "robustness_metrics.json").exists())
            self.assertTrue((run_dir / "robustness_metrics.md").exists())


if __name__ == "__main__":
    unittest.main()
