import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch


class CMIFGMV2Test(unittest.TestCase):
    def test_fgm_state_uses_previous_signal_for_coefficients(self):
        from cmi_fgm import CMIFGMState

        state = CMIFGMState(
            modalities=("audio", "visual"),
            strength=0.5,
            temperature=1.0,
            momentum=0.0,
            warmup_steps=0,
        )
        initial = state.coefficients(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)

        state.update(torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
        coefs = state.coefficients(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)

        self.assertTrue(torch.allclose(initial["audio"], torch.ones(2)))
        self.assertGreater(coefs["audio"][0].item(), coefs["visual"][0].item())
        self.assertGreater(coefs["visual"][1].item(), coefs["audio"][1].item())
        self.assertGreater(coefs["audio"][0].item(), 1.0)
        self.assertGreater(coefs["visual"][1].item(), 1.0)

    def test_av_v2_probe_heads_do_not_change_fusion_logits(self):
        from AV_v2.models import AVBaseline

        model = AVBaseline(num_classes=6)
        model.eval()
        audio = torch.randn(2, 1, 64, 80)
        visual = torch.randn(2, 3, 2, 64, 64)

        with torch.no_grad():
            before = model.forward_with_modal_logits(audio, visual)["logits"]
            for head in (model.audio_probe, model.visual_probe):
                for param in head.parameters():
                    param.add_(100.0)
            outputs = model.forward_with_modal_logits(audio, visual)

        self.assertEqual(tuple(outputs["logits"].shape), (2, 6))
        self.assertEqual(tuple(outputs["audio_logits"].shape), (2, 6))
        self.assertEqual(tuple(outputs["visual_logits"].shape), (2, 6))
        self.assertTrue(torch.allclose(outputs["logits"], before, atol=1e-5))

    def test_av_v2_fusion_dropout_is_configurable(self):
        from AV_v2 import train_cremad
        from AV_v2.models import AVBaseline

        args = train_cremad.parse_args(["--fusion-dropout", "0.25"])
        model = train_cremad.build_model(
            args.modality,
            num_classes=args.num_classes,
            fusion_dropout=args.fusion_dropout,
        )

        self.assertIsInstance(model, AVBaseline)
        self.assertIsInstance(model.fusion_dropout, torch.nn.Dropout)
        self.assertEqual(model.fusion_dropout.p, 0.25)

    def test_av_v2_training_reports_fgm_metrics_when_enabled(self):
        from AV_v2.models import AVBaseline
        from AV_v2.train_cremad import forward_and_losses
        from cmi_fgm import CMIFGMState

        model = AVBaseline(num_classes=6)
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        fgm_state = CMIFGMState(("audio", "visual"), strength=0.5, warmup_steps=0, momentum=0.0)
        audio = torch.randn(2, 1, 64, 80)
        visual = torch.randn(2, 3, 2, 64, 64)
        labels = torch.tensor([0, 1], dtype=torch.long)

        _, first_losses, first_handles = forward_and_losses(
            model,
            (audio, visual),
            labels,
            "av",
            criterion,
            fgm_state=fgm_state,
        )
        first_losses["loss"].backward()
        for handle in first_handles:
            handle.remove()

        model.zero_grad(set_to_none=True)
        _, second_losses, second_handles = forward_and_losses(
            model,
            (audio, visual),
            labels,
            "av",
            criterion,
            fgm_state=fgm_state,
        )
        second_losses["loss"].backward()
        for handle in second_handles:
            handle.remove()

        self.assertIn("fgm_coef_audio", second_losses)
        self.assertIn("fgm_signal_audio", second_losses)
        self.assertGreaterEqual(second_losses["fgm_coef_audio"].item(), 1.0)
        self.assertGreaterEqual(second_losses["fgm_coef_visual"].item(), 1.0)

    def test_macro_f1_for_multiclass_predictions(self):
        from AV_v2.train_cremad import macro_f1_score

        predictions = torch.tensor([0, 1, 1, 2])
        labels = torch.tensor([0, 1, 2, 2])

        score = macro_f1_score(predictions, labels, num_classes=3)

        self.assertAlmostEqual(score, (1.0 + (2 / 3) + (2 / 3)) / 3)

    def test_av_v2_training_reports_modality_accuracy_and_macro_f1(self):
        from AV_v2.models import AVBaseline
        from AV_v2.train_cremad import train_one_epoch

        batch = {
            "audio": torch.randn(2, 1, 64, 80),
            "visual": torch.randn(2, 3, 2, 64, 64),
            "label": torch.tensor([0, 1], dtype=torch.long),
        }
        loader = torch.utils.data.DataLoader([batch], batch_size=None)
        model = AVBaseline(num_classes=6)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        metrics = train_one_epoch(model, loader, optimizer, torch.device("cpu"), "av")

        self.assertIn("macro_f1", metrics)
        self.assertIn("audio_acc", metrics)
        self.assertIn("visual_acc", metrics)
        self.assertGreaterEqual(metrics["macro_f1"], 0.0)
        self.assertLessEqual(metrics["macro_f1"], 1.0)

    def test_av_v2_run_training_keeps_only_best_checkpoint_and_run_artifacts(self):
        from AV_v2 import train_cremad

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
                epochs=2,
                output_dir=str(output_dir),
                no_progress=True,
                fgm=False,
                fgm_lambda=0.5,
                fgm_tau=1.0,
                fgm_momentum=0.9,
                fgm_warmup_steps=0,
            )
            loader = [object()]
            train_metrics = {
                "loss": 1.0,
                "fusion_loss": 0.4,
                "audio_loss": 0.3,
                "visual_loss": 0.3,
                "acc": 0.5,
                "macro_f1": 0.4,
                "audio_acc": 0.5,
                "visual_acc": 0.5,
            }
            val_metrics = [
                {**train_metrics, "acc": 0.6, "macro_f1": 0.5},
                {**train_metrics, "acc": 0.4, "macro_f1": 0.3},
                {**train_metrics, "acc": 0.7, "macro_f1": 0.6},
            ]

            with patch.object(train_cremad, "create_dataloaders", return_value=(loader, loader, loader, {"train": 1, "val": 1, "test": 1})), patch.object(
                train_cremad,
                "build_model",
                return_value=torch.nn.Linear(1, 2),
            ), patch.object(
                train_cremad,
                "train_one_epoch",
                return_value=train_metrics,
            ), patch.object(
                train_cremad,
                "evaluate",
                side_effect=val_metrics,
            ):
                train_cremad.run_training(args)

            self.assertTrue((output_dir / "best.pt").exists())
            self.assertFalse((output_dir / "last.pt").exists())
            self.assertTrue((output_dir / "config.json").exists())
            self.assertTrue((output_dir / "history.json").exists())
            self.assertTrue((output_dir / "history.jsonl").exists())
            self.assertTrue((output_dir / "metrics.json").exists())

    def test_train_video_cremad_defaults_to_one_visual_frame(self):
        from AV_v2 import train_video

        default_args = train_video.build_dataset_args(["--dataset", "cremad"])
        explicit_args = train_video.build_dataset_args(["--dataset", "cremad", "--use-video-frames", "3"])

        self.assertEqual(default_args.fps, 1)
        self.assertEqual(explicit_args.fps, 3)

    def test_cremad_lr_is_plain_command_line_argument(self):
        from AV_v2 import train_cremad

        visual_args = train_cremad.parse_args(["--modality", "visual"])
        av_args = train_cremad.parse_args([])
        explicit_args = train_cremad.parse_args(["--modality", "visual", "--lr", "0.001"])

        self.assertEqual(visual_args.lr, av_args.lr)
        self.assertEqual(explicit_args.lr, 0.001)

    def test_cremad_scheduler_matches_iccv_multistep_policy(self):
        from AV_v2 import train_cremad

        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.002)
        args = train_cremad.parse_args([])
        scheduler = train_cremad.build_scheduler(optimizer, args)

        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.MultiStepLR)
        self.assertEqual(dict(scheduler.milestones), {70: 1})
        self.assertEqual(scheduler.gamma, 0.1)

    def test_cremad_scheduler_can_be_tuned_from_command_line_args(self):
        from AV_v2 import train_cremad

        multistep_args = train_cremad.parse_args(
            ["--lr-scheduler", "multistep", "--lr-decay-step", "[30,70]", "--lr-decay-ratio", "0.2"]
        )
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.002)

        scheduler = train_cremad.build_scheduler(optimizer, multistep_args)

        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.MultiStepLR)
        self.assertEqual(dict(scheduler.milestones), {30: 1, 70: 1})
        self.assertEqual(scheduler.gamma, 0.2)

        cosine_args = train_cremad.parse_args(["--lr-scheduler", "cosine", "--epochs", "12"])
        cosine_scheduler = train_cremad.build_scheduler(optimizer, cosine_args)

        self.assertIsInstance(cosine_scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        self.assertEqual(cosine_scheduler.T_max, 12)

    def test_av_v2_plots_train_and_val_curves_separately(self):
        from AV_v2.train_cremad import plot_history

        history = [
            {
                "epoch": 1,
                "train": {
                    "loss": 3.0,
                    "fusion_loss": 1.0,
                    "audio_loss": 0.9,
                    "visual_loss": 1.1,
                    "acc": 0.2,
                    "audio_acc": 0.3,
                    "visual_acc": 0.25,
                },
                "val": {
                    "loss": 3.2,
                    "fusion_loss": 1.1,
                    "audio_loss": 1.0,
                    "visual_loss": 1.1,
                    "acc": 0.18,
                    "audio_acc": 0.22,
                    "visual_acc": 0.2,
                },
            },
            {
                "epoch": 2,
                "train": {
                    "loss": 2.5,
                    "fusion_loss": 0.8,
                    "audio_loss": 0.8,
                    "visual_loss": 0.9,
                    "acc": 0.35,
                    "audio_acc": 0.4,
                    "visual_acc": 0.32,
                },
                "val": {
                    "loss": 3.0,
                    "fusion_loss": 1.0,
                    "audio_loss": 0.95,
                    "visual_loss": 1.05,
                    "acc": 0.22,
                    "audio_acc": 0.28,
                    "visual_acc": 0.23,
                },
            },
        ]

        with TemporaryDirectory() as tmpdir:
            plot_history(history, Path(tmpdir) / "curves.png")

            self.assertTrue((Path(tmpdir) / "curves.png").exists())
            self.assertTrue((Path(tmpdir) / "train_loss_curves.png").exists())
            self.assertTrue((Path(tmpdir) / "val_loss_curves.png").exists())
            self.assertTrue((Path(tmpdir) / "train_modality_accuracy.png").exists())
            self.assertTrue((Path(tmpdir) / "val_modality_accuracy.png").exists())
            self.assertFalse((Path(tmpdir) / "loss_curves.png").exists())
            self.assertFalse((Path(tmpdir) / "modality_accuracy.png").exists())

    def test_cremad_splits_use_plain_resize_normalize_without_augmentation(self):
        from AV_v2 import train_cremad
        from AV_v2.datasets import CREMADSample, ResizeToTensorNormalize

        sample = CREMADSample(
            sample_id="1001_IEO_HAP_HI",
            actor_id="1001",
            emotion="HAP",
            label=3,
            audio_path=Path("dummy.wav"),
            image_dir=Path("dummy_images"),
        )
        args = Namespace(
            data_root="dataset/CREMA-D",
            split_csv_root="ICCV2025-GDL-main/dataset/data/CREMAD",
            modality="visual",
            fps=1,
            audio_duration=3.0,
            n_fft=512,
            hop_length=160,
            win_length=400,
            image_size=224,
            seed=0,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
        )

        with patch.object(train_cremad, "discover_cremad_samples", return_value=[sample]), patch.object(
            train_cremad,
            "split_samples_from_csv",
            return_value={"train": [sample], "test": [sample]},
        ):
            train_loader, val_loader, test_loader, _ = train_cremad.create_dataloaders(args)

        self.assertIs(type(train_loader.dataset.image_transform), ResizeToTensorNormalize)
        self.assertIs(type(val_loader.dataset.image_transform), ResizeToTensorNormalize)
        self.assertIs(type(test_loader.dataset.image_transform), ResizeToTensorNormalize)

    def test_cremad_visual_augmentation_preset_applies_to_train_split_only(self):
        from AV_v2 import train_cremad
        from AV_v2.datasets import CREMADSample, CREMADTrainImageTransform, ResizeToTensorNormalize

        sample = CREMADSample(
            sample_id="1001_IEO_HAP_HI",
            actor_id="1001",
            emotion="HAP",
            label=3,
            audio_path=Path("dummy.wav"),
            image_dir=Path("dummy_images"),
        )
        args = Namespace(
            data_root="dataset/CREMA-D",
            split_csv_root="ICCV2025-GDL-main/dataset/data/CREMAD",
            modality="visual",
            fps=1,
            audio_duration=3.0,
            n_fft=512,
            hop_length=160,
            win_length=400,
            image_size=224,
            visual_aug="light",
            aug_scale=None,
            aug_ratio=None,
            aug_hflip_prob=None,
            seed=0,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
        )

        with patch.object(train_cremad, "discover_cremad_samples", return_value=[sample]), patch.object(
            train_cremad,
            "split_samples_from_csv",
            return_value={"train": [sample], "test": [sample]},
        ):
            train_loader, val_loader, test_loader, _ = train_cremad.create_dataloaders(args)

        self.assertIs(type(train_loader.dataset.image_transform), CREMADTrainImageTransform)
        self.assertEqual(train_loader.dataset.image_transform.scale, (0.85, 1.0))
        self.assertEqual(train_loader.dataset.image_transform.ratio, (0.95, 1.05))
        self.assertEqual(train_loader.dataset.image_transform.horizontal_flip_prob, 0.0)
        self.assertIs(type(val_loader.dataset.image_transform), ResizeToTensorNormalize)
        self.assertIs(type(test_loader.dataset.image_transform), ResizeToTensorNormalize)

    def test_cremad_visual_augmentation_strength_is_tunable_from_args(self):
        from AV_v2 import train_cremad
        from AV_v2.datasets import CREMADTrainImageTransform

        args = train_cremad.parse_args(
            [
                "--visual-aug",
                "custom",
                "--aug-scale",
                "0.8,1.0",
                "--aug-ratio",
                "0.9,1.1",
                "--aug-hflip-prob",
                "0.2",
            ]
        )

        transform = train_cremad.build_train_image_transform(args)

        self.assertIs(type(transform), CREMADTrainImageTransform)
        self.assertEqual(transform.scale, (0.8, 1.0))
        self.assertEqual(transform.ratio, (0.9, 1.1))
        self.assertEqual(transform.horizontal_flip_prob, 0.2)


if __name__ == "__main__":
    unittest.main()
