from __future__ import annotations

import argparse
import json
import random
import sys
from functools import partial
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from MOSI_v1.datasets import MOSIDataset, load_mosi_splits, mosi_collate_fn
from MOSI_v1.models import MOSIRegressionModel


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    tensor_keys = ("input_ids", "attention_mask", "vision", "audio", "vision_mask", "audio_mask", "labels")
    return {
        key: value.to(device, non_blocking=True) if key in tensor_keys else value
        for key, value in batch.items()
    }


def regression_metrics(predictions: torch.Tensor, labels: torch.Tensor, loss: float) -> dict[str, float]:
    preds = predictions.detach().float().cpu()
    gold = labels.detach().float().cpu()
    mae = torch.mean(torch.abs(preds - gold)).item()
    pred_acc7 = torch.clamp(preds, min=-3.0, max=3.0).round()
    gold_acc7 = torch.clamp(gold, min=-3.0, max=3.0).round()
    acc7 = float((pred_acc7 == gold_acc7).float().mean().item())
    if preds.numel() > 1 and float(preds.std()) > 0.0 and float(gold.std()) > 0.0:
        pearson = float(torch.corrcoef(torch.stack([preds, gold]))[0, 1].item())
    else:
        pearson = 0.0

    nonzero = gold != 0
    if int(nonzero.sum()) == 0:
        binary_acc = 0.0
        f1 = 0.0
    else:
        pred_positive = preds[nonzero] >= 0
        gold_positive = gold[nonzero] >= 0
        binary_acc = float((pred_positive == gold_positive).float().mean().item())
        tp = float((pred_positive & gold_positive).sum().item())
        fp = float((pred_positive & ~gold_positive).sum().item())
        fn = float((~pred_positive & gold_positive).sum().item())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return {
        "loss": float(loss),
        "mae": mae,
        "corr": pearson,
        "acc7": acc7,
        "binary_acc": binary_acc,
        "f1": float(f1),
    }


def forward_batch(model: nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    return model(
        batch["input_ids"],
        batch["attention_mask"],
        batch["vision"],
        batch["audio"],
        batch["vision_mask"],
        batch["audio_mask"],
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int | None = None,
    show_progress: bool = False,
) -> dict[str, float]:
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_samples = 0
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    iterator = loader
    if show_progress:
        desc = "train" if epoch is None else f"train epoch {epoch}"
        iterator = tqdm(loader, desc=desc, leave=False)

    for raw_batch in iterator:
        batch = batch_to_device(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        predictions = forward_batch(model, batch)
        loss = criterion(predictions, batch["labels"])
        loss.backward()
        optimizer.step()

        batch_size = batch["labels"].shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        all_preds.append(predictions.detach().cpu())
        all_labels.append(batch["labels"].detach().cpu())
        if show_progress:
            iterator.set_postfix({"loss": total_loss / max(1, total_samples)})

    return regression_metrics(
        torch.cat(all_preds),
        torch.cat(all_labels),
        total_loss / max(1, total_samples),
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int | None = None,
    split_name: str = "eval",
    show_progress: bool = False,
) -> dict[str, float]:
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_samples = 0
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    iterator = loader
    if show_progress:
        desc = split_name if epoch is None else f"{split_name} epoch {epoch}"
        iterator = tqdm(loader, desc=desc, leave=False)

    for raw_batch in iterator:
        batch = batch_to_device(raw_batch, device)
        predictions = forward_batch(model, batch)
        loss = criterion(predictions, batch["labels"])

        batch_size = batch["labels"].shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        all_preds.append(predictions.detach().cpu())
        all_labels.append(batch["labels"].detach().cpu())
        if show_progress:
            iterator.set_postfix({"loss": total_loss / max(1, total_samples)})

    return regression_metrics(
        torch.cat(all_preds),
        torch.cat(all_labels),
        total_loss / max(1, total_samples),
    )


def create_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    try:
        from transformers import BertTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required to tokenize MOSI text for training.") from exc

    splits = load_mosi_splits(args.data_path)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_name)
    collate = partial(mosi_collate_fn, tokenizer=tokenizer, max_text_length=args.max_text_length)
    datasets = {name: MOSIDataset(samples) for name, samples in splits.items()}

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "worker_init_fn": seed_worker,
        "generator": generator,
        "collate_fn": collate,
    }
    train_loader = DataLoader(datasets["train"], shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(datasets["dev"], shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(datasets["test"], shuffle=False, drop_last=False, **loader_kwargs)
    sizes = {name: len(dataset) for name, dataset in datasets.items()}
    sizes["val"] = sizes["dev"]
    return train_loader, val_loader, test_loader, sizes


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def format_metrics(prefix: str, metrics: dict[str, float]) -> str:
    return (
        f"{prefix}_loss={metrics['loss']:.4f} "
        f"{prefix}_mae={metrics['mae']:.4f} "
        f"{prefix}_corr={metrics['corr']:.4f} "
        f"{prefix}_acc7={metrics['acc7']:.4f} "
        f"{prefix}_acc2={metrics['binary_acc']:.4f} "
        f"{prefix}_f1={metrics['f1']:.4f}"
    )


def append_epoch_log(path: Path, record: dict[str, Any], args: argparse.Namespace, split_sizes: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**record, "args": vars(args), "split_sizes": split_sizes}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_training(args: argparse.Namespace) -> dict[str, float]:
    set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    train_loader, val_loader, test_loader, sizes = create_dataloaders(args)
    model = MOSIRegressionModel(
        bert_model_name=args.bert_model_name,
        vision_dim=args.vision_dim,
        audio_dim=args.audio_dim,
        hidden_sz=args.hidden_sz,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        conv_kernel_size=args.conv_kernel_size,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    history_path = output_dir / "history.jsonl"
    history: list[dict[str, Any]] = []
    best_val_mae = float("inf")
    best_epoch = 0

    print(f"Split sizes: {sizes}")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch=epoch,
            show_progress=not args.no_progress,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            epoch=epoch,
            split_name="val",
            show_progress=not args.no_progress,
        )
        scheduler.step()

        print(f"epoch={epoch:03d} {format_metrics('train', train_metrics)} {format_metrics('val', val_metrics)}")
        record = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        append_epoch_log(history_path, record, args, sizes)
        (output_dir / "history.json").write_text(
            json.dumps({"args": vars(args), "split_sizes": sizes, "epochs": history}, indent=2),
            encoding="utf-8",
        )
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, {"train": train_metrics, "val": val_metrics}, args)

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, {"train": train_metrics, "val": val_metrics}, args)

    best_state = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(best_state["model"])
    test_metrics = evaluate(model, test_loader, device, split_name="test", show_progress=not args.no_progress)
    result = {"best_epoch": float(best_epoch), "best_val_mae": float(best_val_mae), **test_metrics}
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"best_epoch={best_epoch:03d} best_val_mae={best_val_mae:.4f} {format_metrics('test', test_metrics)}")
    return result


def build_arg_parser(
    description: str = "Train CMU-MOSI BERT + visual/audio sequence regression model.",
    default_data_path: str = "dataset/mosi.pkl",
    default_output_dir: str = "runs/mosi_baseline",
    default_vision_dim: int = 47,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-path", type=str, default=default_data_path)
    parser.add_argument("--output-dir", type=str, default=default_output_dir)
    parser.add_argument("--bert-model-name", type=str, default="bert-base-uncased")
    parser.add_argument("--vision-dim", type=int, default=default_vision_dim)
    parser.add_argument("--audio-dim", type=int, default=74)
    parser.add_argument("--hidden-sz", type=int, default=50)
    parser.add_argument("--num-heads", type=int, default=5)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--conv-kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-text-length", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def parse_args(
    argv: list[str] | None = None,
    description: str = "Train CMU-MOSI BERT + visual/audio sequence regression model.",
    default_data_path: str = "dataset/mosi.pkl",
    default_output_dir: str = "runs/mosi_baseline",
    default_vision_dim: int = 47,
) -> argparse.Namespace:
    parser = build_arg_parser(
        description=description,
        default_data_path=default_data_path,
        default_output_dir=default_output_dir,
        default_vision_dim=default_vision_dim,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()
