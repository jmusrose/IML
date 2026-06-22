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


if __name__ == "__main__":
    unittest.main()
