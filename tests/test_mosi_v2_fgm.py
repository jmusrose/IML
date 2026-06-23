import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch


class TinyTextEncoder(torch.nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.embedding = torch.nn.Embedding(64, hidden_size)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.embedding(input_ids)
        pooled = hidden[:, 0]
        return type("Output", (), {"last_hidden_state": hidden, "pooler_output": pooled})()


def tiny_mosi_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([[1, 2, 0], [1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long),
        "vision": torch.randn(2, 3, 47),
        "audio": torch.randn(2, 3, 74),
        "vision_mask": torch.tensor([[True, True, False], [True, True, True]]),
        "audio_mask": torch.tensor([[True, True, False], [True, True, True]]),
        "labels": torch.tensor([1.0, -1.0]),
    }


def tiny_mosi_model():
    from MOSI_v2.models import MOSIRegressionModel

    return MOSIRegressionModel(
        text_encoder=TinyTextEncoder(hidden_size=8),
        text_dim=8,
        vision_dim=47,
        audio_dim=74,
        hidden_sz=10,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    )


class MOSIV2FGMTest(unittest.TestCase):
    def test_mosi_v2_probe_heads_do_not_change_fusion_prediction(self):
        model = tiny_mosi_model()
        model.eval()
        batch = tiny_mosi_batch()
        batch.pop("labels")

        with torch.no_grad():
            before = model.forward_with_modal_predictions(**batch)["prediction"]
            for head in (model.text_probe, model.vision_probe, model.audio_probe):
                for param in head.parameters():
                    param.add_(100.0)
            outputs = model.forward_with_modal_predictions(**batch)

        self.assertEqual(tuple(outputs["prediction"].shape), (2,))
        self.assertEqual(tuple(outputs["text_prediction"].shape), (2,))
        self.assertEqual(tuple(outputs["vision_prediction"].shape), (2,))
        self.assertEqual(tuple(outputs["audio_prediction"].shape), (2,))
        self.assertTrue(torch.allclose(outputs["prediction"], before, atol=1e-5))

    def test_mosi_v2_training_reports_fgm_metrics_when_enabled(self):
        from MOSI_v2.train_mosi import forward_and_losses
        from cmi_fgm import CMIFGMState

        model = tiny_mosi_model()
        batch = tiny_mosi_batch()
        criterion = torch.nn.MSELoss(reduction="none")
        fgm_state = CMIFGMState(("text", "vision", "audio"), strength=0.5, warmup_steps=0, momentum=0.0)

        _, first_losses, first_handles = forward_and_losses(model, batch, criterion, fgm_state=fgm_state)
        first_losses["loss"].backward()
        for handle in first_handles:
            handle.remove()

        model.zero_grad(set_to_none=True)
        _, second_losses, second_handles = forward_and_losses(model, batch, criterion, fgm_state=fgm_state)
        second_losses["loss"].backward()
        for handle in second_handles:
            handle.remove()

        self.assertIn("fgm_coef_text", second_losses)
        self.assertIn("fgm_coef_vision", second_losses)
        self.assertIn("fgm_coef_audio", second_losses)
        self.assertIn("fgm_signal_text", second_losses)
        self.assertGreaterEqual(second_losses["fgm_coef_text"].item(), 1.0)
        self.assertGreaterEqual(second_losses["fgm_coef_vision"].item(), 1.0)
        self.assertGreaterEqual(second_losses["fgm_coef_audio"].item(), 1.0)

    def test_mosi_v2_defaults_enable_fgm_and_pin_memory(self):
        from MOSI_v2 import train_mosi

        args = train_mosi.parse_args([])
        disabled = train_mosi.parse_args(["--no-fgm", "--no-pin-memory"])

        self.assertTrue(args.fgm)
        self.assertTrue(args.pin_memory)
        self.assertFalse(disabled.fgm)
        self.assertFalse(disabled.pin_memory)

    def test_mosi_v2_run_training_keeps_only_best_checkpoint_and_run_artifacts(self):
        from MOSI_v2 import train_mosi

        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            args = Namespace(
                seed=0,
                deterministic=False,
                device="cpu",
                bert_model_name="bert-base-uncased",
                vision_dim=47,
                audio_dim=74,
                hidden_sz=10,
                num_heads=2,
                num_layers=1,
                conv_kernel_size=3,
                dropout=0.0,
                lr=0.001,
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
                "text_loss": 0.2,
                "vision_loss": 0.2,
                "audio_loss": 0.2,
                "mae": 0.8,
                "corr": 0.1,
                "acc7": 0.5,
                "binary_acc": 0.5,
                "f1": 0.5,
            }
            val_metrics = [
                {**train_metrics, "mae": 0.7},
                {**train_metrics, "mae": 0.9},
                {**train_metrics, "mae": 0.6},
            ]

            with patch.object(train_mosi, "create_dataloaders", return_value=(loader, loader, loader, {"train": 1, "val": 1, "test": 1})), patch.object(
                train_mosi,
                "MOSIRegressionModel",
                return_value=torch.nn.Linear(1, 1),
            ), patch.object(
                train_mosi,
                "train_one_epoch",
                return_value=train_metrics,
            ), patch.object(
                train_mosi,
                "evaluate",
                side_effect=val_metrics,
            ):
                train_mosi.run_training(args)

            self.assertTrue((output_dir / "best.pt").exists())
            self.assertFalse((output_dir / "last.pt").exists())
            self.assertTrue((output_dir / "config.json").exists())
            self.assertTrue((output_dir / "history.json").exists())
            self.assertTrue((output_dir / "history.jsonl").exists())
            self.assertTrue((output_dir / "metrics.json").exists())

    def test_mosi_v2_plots_train_and_val_loss_curves_separately(self):
        from MOSI_v2.train_mosi import plot_history

        history = [
            {
                "epoch": 1,
                "train": {
                    "loss": 2.0,
                    "fusion_loss": 0.8,
                    "text_loss": 0.4,
                    "vision_loss": 0.4,
                    "audio_loss": 0.4,
                    "mae": 1.0,
                    "corr": 0.1,
                },
                "val": {
                    "loss": 2.2,
                    "fusion_loss": 0.9,
                    "text_loss": 0.45,
                    "vision_loss": 0.43,
                    "audio_loss": 0.42,
                    "mae": 1.1,
                    "corr": 0.05,
                },
            },
            {
                "epoch": 2,
                "train": {
                    "loss": 1.6,
                    "fusion_loss": 0.6,
                    "text_loss": 0.35,
                    "vision_loss": 0.32,
                    "audio_loss": 0.33,
                    "mae": 0.8,
                    "corr": 0.25,
                },
                "val": {
                    "loss": 2.0,
                    "fusion_loss": 0.8,
                    "text_loss": 0.4,
                    "vision_loss": 0.4,
                    "audio_loss": 0.4,
                    "mae": 1.0,
                    "corr": 0.1,
                },
            },
        ]

        with TemporaryDirectory() as tmpdir:
            plot_history(history, Path(tmpdir) / "curves.png")

            self.assertTrue((Path(tmpdir) / "curves.png").exists())
            self.assertTrue((Path(tmpdir) / "train_loss_curves.png").exists())
            self.assertTrue((Path(tmpdir) / "val_loss_curves.png").exists())
            self.assertFalse((Path(tmpdir) / "loss_curves.png").exists())


if __name__ == "__main__":
    unittest.main()
