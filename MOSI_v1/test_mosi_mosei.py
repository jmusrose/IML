from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from MOSI_v1.datasets import MOSIDataset, load_mosi_splits, mosi_collate_fn
from MOSI_v1.models import MOSIRegressionModel
from MOSI_v1.train_mosi import batch_to_device, forward_batch, format_metrics, regression_metrics


@dataclass(frozen=True)
class DatasetSmokeConfig:
    name: str
    data_path: str | Path
    vision_dim: int


def iter_default_configs(which: str = "both") -> list[DatasetSmokeConfig]:
    configs = [
        DatasetSmokeConfig(name="mosi", data_path="dataset/mosi.pkl", vision_dim=47),
        DatasetSmokeConfig(name="mosei", data_path="dataset/mosei.pkl", vision_dim=35),
    ]
    if which == "both":
        return configs
    return [config for config in configs if config.name == which]


def default_model_factory(
    vision_dim: int,
    bert_model_name: str,
    audio_dim: int,
    hidden_sz: int,
    num_heads: int,
    num_layers: int,
    conv_kernel_size: int,
    dropout: float,
    local_files_only: bool,
) -> MOSIRegressionModel:
    return MOSIRegressionModel(
        bert_model_name=bert_model_name,
        local_files_only=local_files_only,
        vision_dim=vision_dim,
        audio_dim=audio_dim,
        hidden_sz=hidden_sz,
        num_heads=num_heads,
        num_layers=num_layers,
        conv_kernel_size=conv_kernel_size,
        dropout=dropout,
    )


def first_tensor_shapes(batch: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    return {
        key: tuple(value.shape)
        for key, value in batch.items()
        if torch.is_tensor(value)
    }


@torch.no_grad()
def smoke_test_dataset(
    config: DatasetSmokeConfig,
    split: str,
    tokenizer: Callable[..., dict[str, torch.Tensor]],
    model_factory: Callable[[int], nn.Module],
    batch_size: int,
    max_text_length: int,
    max_batches: int,
    device: torch.device,
) -> dict[str, Any]:
    splits = load_mosi_splits(config.data_path)
    if split not in splits:
        raise ValueError(f"Unsupported split {split!r}; available splits: {sorted(splits)}")

    dataset = MOSIDataset(splits[split])
    collate = partial(mosi_collate_fn, tokenizer=tokenizer, max_text_length=max_text_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    model = model_factory(config.vision_dim).to(device)
    model.eval()

    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    total_loss = 0.0
    total_samples = 0
    batch_shapes: dict[str, tuple[int, ...]] | None = None
    criterion = nn.MSELoss()

    for batch_index, raw_batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        if batch_shapes is None:
            batch_shapes = first_tensor_shapes(raw_batch)
        batch = batch_to_device(raw_batch, device)
        predictions = forward_batch(model, batch)
        loss = criterion(predictions, batch["labels"])

        batch_size_actual = batch["labels"].shape[0]
        total_loss += float(loss.item()) * batch_size_actual
        total_samples += batch_size_actual
        all_preds.append(predictions.detach().cpu())
        all_labels.append(batch["labels"].detach().cpu())

    if total_samples == 0:
        raise ValueError(f"No samples were read from {config.name} split {split!r}.")

    metrics = regression_metrics(
        torch.cat(all_preds),
        torch.cat(all_labels),
        total_loss / total_samples,
    )
    split_sizes = {name: len(samples) for name, samples in splits.items()}
    return {
        "name": config.name,
        "data_path": str(config.data_path),
        "split": split,
        "split_sizes": split_sizes,
        "batch_shapes": batch_shapes or {},
        "num_batches": len(all_preds),
        "num_samples": total_samples,
        "metrics": metrics,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(f"[{summary['name']}] data={summary['data_path']} split={summary['split']}")
    print(f"[{summary['name']}] split_sizes={summary['split_sizes']}")
    print(f"[{summary['name']}] first_batch_shapes={summary['batch_shapes']}")
    print(f"[{summary['name']}] batches={summary['num_batches']} samples={summary['num_samples']}")
    print(f"[{summary['name']}] {format_metrics('smoke', summary['metrics'])}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test CMU-MOSI and CMU-MOSEI data/model pipelines in order.")
    parser.add_argument("--dataset", choices=["both", "mosi", "mosei"], default="both")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--bert-model-name", type=str, default="bert-base-uncased")
    parser.add_argument("--allow-download", action="store_false", dest="local_files_only")
    parser.add_argument("--audio-dim", type=int, default=74)
    parser.add_argument("--hidden-sz", type=int, default=50)
    parser.add_argument("--num-heads", type=int, default=5)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--conv-kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-text-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.set_defaults(local_files_only=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    try:
        from transformers import BertTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required for the default tokenizer.") from exc

    tokenizer = BertTokenizer.from_pretrained(args.bert_model_name, local_files_only=args.local_files_only)
    model_factory = partial(
        default_model_factory,
        bert_model_name=args.bert_model_name,
        audio_dim=args.audio_dim,
        hidden_sz=args.hidden_sz,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        conv_kernel_size=args.conv_kernel_size,
        dropout=args.dropout,
        local_files_only=args.local_files_only,
    )

    for config in iter_default_configs(args.dataset):
        summary = smoke_test_dataset(
            config,
            split=args.split,
            tokenizer=tokenizer,
            model_factory=model_factory,
            batch_size=args.batch_size,
            max_text_length=args.max_text_length,
            max_batches=args.max_batches,
            device=device,
        )
        print_summary(summary)


if __name__ == "__main__":
    main()
