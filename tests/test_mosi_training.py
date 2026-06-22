import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from MOSI_v1.datasets import MOSIDataset, load_mosi_splits, mosi_collate_fn
from MOSI_v1.models import MOSIRegressionModel
from MOSI_v1.train_mosi import evaluate, regression_metrics, train_one_epoch


class TinyTokenizer:
    def __call__(
        self,
        texts,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    ):
        width = min(max(len(text.split()) for text in texts) + 2, max_length)
        input_ids = torch.zeros(len(texts), width, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, text in enumerate(texts):
            length = min(len(text.split()) + 2, width)
            input_ids[row, :length] = torch.arange(1, length + 1)
            attention_mask[row, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class TinyTextEncoder(torch.nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.embedding = torch.nn.Embedding(64, hidden_size)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.embedding(input_ids)
        pooled = hidden[:, 0]
        return type("Output", (), {"last_hidden_state": hidden, "pooler_output": pooled})()


def write_tiny_mosi(path: Path) -> None:
    sample_a = (
        (
            ["good", "movie"],
            np.ones((2, 47), dtype=np.float32),
            np.ones((2, 74), dtype=np.float32) * 2,
        ),
        np.asarray([[1.5]], dtype=np.float32),
        "a[0]",
    )
    sample_b = (
        (
            ["very", "bad", "film"],
            np.ones((3, 47), dtype=np.float32) * -1,
            np.ones((3, 74), dtype=np.float32) * -2,
        ),
        np.asarray([[-2.0]], dtype=np.float32),
        "b[0]",
    )
    with path.open("wb") as handle:
        pickle.dump({"train": [sample_a, sample_b], "dev": [sample_a], "test": [sample_b]}, handle)


class MOSITrainingTest(unittest.TestCase):
    def test_load_mosi_splits_and_dataset_item(self):
        with TemporaryDirectory() as tmpdir:
            pkl_path = Path(tmpdir) / "mosi.pkl"
            write_tiny_mosi(pkl_path)

            splits = load_mosi_splits(pkl_path)
            dataset = MOSIDataset(splits["train"])
            item = dataset[0]

            self.assertEqual(set(splits), {"train", "dev", "test"})
            self.assertEqual(item["words"], ["good", "movie"])
            self.assertEqual(tuple(item["vision"].shape), (2, 47))
            self.assertEqual(tuple(item["audio"].shape), (2, 74))
            self.assertEqual(tuple(item["label"].shape), (1,))
            self.assertEqual(item["sample_id"], "a[0]")

    def test_mosi_collate_pads_modal_sequences_and_tokenizes_text(self):
        with TemporaryDirectory() as tmpdir:
            pkl_path = Path(tmpdir) / "mosi.pkl"
            write_tiny_mosi(pkl_path)
            dataset = MOSIDataset(load_mosi_splits(pkl_path)["train"])

            batch = mosi_collate_fn([dataset[0], dataset[1]], TinyTokenizer(), max_text_length=8)

            self.assertEqual(tuple(batch["input_ids"].shape), (2, 5))
            self.assertEqual(tuple(batch["attention_mask"].shape), (2, 5))
            self.assertEqual(tuple(batch["vision"].shape), (2, 3, 47))
            self.assertEqual(tuple(batch["audio"].shape), (2, 3, 74))
            self.assertEqual(batch["vision_mask"].tolist(), [[True, True, False], [True, True, True]])
            self.assertEqual(batch["audio_mask"].tolist(), [[True, True, False], [True, True, True]])
            self.assertEqual(tuple(batch["labels"].shape), (2,))

    def test_model_forward_and_training_step_run_on_cpu(self):
        batch = {
            "input_ids": torch.tensor([[1, 2, 0], [1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long),
            "vision": torch.randn(2, 3, 47),
            "audio": torch.randn(2, 3, 74),
            "vision_mask": torch.tensor([[True, True, False], [True, True, True]]),
            "audio_mask": torch.tensor([[True, True, False], [True, True, True]]),
            "labels": torch.tensor([1.0, -1.0]),
        }
        loader = torch.utils.data.DataLoader([batch], batch_size=None)
        model = MOSIRegressionModel(
            text_encoder=TinyTextEncoder(hidden_size=8),
            text_dim=8,
            vision_dim=47,
            audio_dim=74,
            hidden_sz=10,
            num_heads=2,
            num_layers=1,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        device = torch.device("cpu")

        predictions = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["vision"],
            batch["audio"],
            batch["vision_mask"],
            batch["audio_mask"],
        )
        train_metrics = train_one_epoch(model, loader, optimizer, device)
        eval_metrics = evaluate(model, loader, device)

        self.assertEqual(tuple(predictions.shape), (2,))
        self.assertGreater(train_metrics["loss"], 0.0)
        self.assertGreater(eval_metrics["mae"], 0.0)
        self.assertIn("binary_acc", eval_metrics)
        self.assertIn("f1", eval_metrics)

    def test_mosi_fusion_prediction_does_not_depend_on_probe_heads(self):
        model = MOSIRegressionModel(
            text_encoder=TinyTextEncoder(hidden_size=8),
            text_dim=8,
            vision_dim=47,
            audio_dim=74,
            hidden_sz=10,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        )
        model.eval()
        batch = {
            "input_ids": torch.tensor([[1, 2, 0], [1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long),
            "vision": torch.randn(2, 3, 47),
            "audio": torch.randn(2, 3, 74),
            "vision_mask": torch.tensor([[True, True, False], [True, True, True]]),
            "audio_mask": torch.tensor([[True, True, False], [True, True, True]]),
        }

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

    def test_mosi_training_reports_fusion_and_probe_losses(self):
        batch = {
            "input_ids": torch.tensor([[1, 2, 0], [1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long),
            "vision": torch.randn(2, 3, 47),
            "audio": torch.randn(2, 3, 74),
            "vision_mask": torch.tensor([[True, True, False], [True, True, True]]),
            "audio_mask": torch.tensor([[True, True, False], [True, True, True]]),
            "labels": torch.tensor([1.0, -1.0]),
        }
        loader = torch.utils.data.DataLoader([batch], batch_size=None)
        model = MOSIRegressionModel(
            text_encoder=TinyTextEncoder(hidden_size=8),
            text_dim=8,
            vision_dim=47,
            audio_dim=74,
            hidden_sz=10,
            num_heads=2,
            num_layers=1,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        metrics = train_one_epoch(model, loader, optimizer, torch.device("cpu"))

        self.assertGreater(metrics["fusion_loss"], 0.0)
        self.assertGreater(metrics["text_loss"], 0.0)
        self.assertGreater(metrics["vision_loss"], 0.0)
        self.assertGreater(metrics["audio_loss"], 0.0)

    def test_regression_metrics_include_acc7(self):
        predictions = torch.tensor([-2.6, -1.2, 0.2, 1.9, 2.2])
        labels = torch.tensor([-3.0, -1.0, 0.0, 2.0, 3.0])

        metrics = regression_metrics(predictions, labels, loss=0.5)

        self.assertAlmostEqual(metrics["acc7"], 0.8)

    def test_mosi_and_mosei_entry_defaults(self):
        from MOSI_v1 import mosei, mosi

        mosi_args = mosi.parse_args([])
        mosei_args = mosei.parse_args([])

        self.assertEqual(mosi_args.data_path, "dataset/mosi.pkl")
        self.assertEqual(mosi_args.output_dir, "runs/mosi_baseline")
        self.assertEqual(mosi_args.vision_dim, 47)
        self.assertEqual(mosei_args.data_path, "dataset/mosei.pkl")
        self.assertEqual(mosei_args.output_dir, "runs/mosei_baseline")
        self.assertEqual(mosei_args.vision_dim, 35)


if __name__ == "__main__":
    unittest.main()
