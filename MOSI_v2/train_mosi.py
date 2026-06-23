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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from cmi_fgm import (
    CMIFGMState,
    register_feature_gradient_hooks,
    register_split_linear_weight_hook,
)
from MOSI_v2.datasets import MOSIDataset, load_mosi_splits, mosi_collate_fn
from MOSI_v2.models import MOSIRegressionModel


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


def build_fgm_state(args: argparse.Namespace) -> CMIFGMState | None:
    if not getattr(args, "fgm", False):
        return None
    return CMIFGMState(
        modalities=("text", "vision", "audio"),
        strength=args.fgm_lambda,
        temperature=args.fgm_tau,
        momentum=args.fgm_momentum,
        warmup_steps=args.fgm_warmup_steps,
    )


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


def forward_and_losses(
    model: nn.Module,
    batch: dict[str, Any],
    criterion: nn.Module,
    fgm_state: CMIFGMState | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[torch.utils.hooks.RemovableHandle]]:
    handles: list[torch.utils.hooks.RemovableHandle] = []
    if hasattr(model, "forward_with_modal_predictions"):
        outputs = model.forward_with_modal_predictions(
            batch["input_ids"],
            batch["attention_mask"],
            batch["vision"],
            batch["audio"],
            batch["vision_mask"],
            batch["audio_mask"],
        )
        predictions = outputs["prediction"]
        fusion_per_sample = criterion(predictions, batch["labels"])
        text_per_sample = criterion(outputs["text_prediction"], batch["labels"])
        vision_per_sample = criterion(outputs["vision_prediction"], batch["labels"])
        audio_per_sample = criterion(outputs["audio_prediction"], batch["labels"])
        fusion_loss = fusion_per_sample.mean()
        text_loss = text_per_sample.mean()
        vision_loss = vision_per_sample.mean()
        audio_loss = audio_per_sample.mean()
        losses = {
            "loss": fusion_loss + text_loss + vision_loss + audio_loss,
            "fusion_loss": fusion_loss,
            "text_loss": text_loss,
            "vision_loss": vision_loss,
            "audio_loss": audio_loss,
        }
        if fgm_state is not None:
            batch_size = batch["labels"].shape[0]
            coefficients = fgm_state.coefficients(batch_size, predictions.device, predictions.dtype)
            handles.extend(
                register_feature_gradient_hooks(
                    {
                        "text": outputs["text_feature"],
                        "vision": outputs["vision_feature"],
                        "audio": outputs["audio_feature"],
                    },
                    coefficients,
                )
            )
            first_fusion_linear = model.fusion[1] if hasattr(model, "fusion") and len(model.fusion) > 1 else None
            if isinstance(first_fusion_linear, nn.Linear):
                handles.append(
                    register_split_linear_weight_hook(
                        first_fusion_linear,
                        split_sizes=(model.text_dim, model.hidden_sz, model.hidden_sz),
                        modalities=("text", "vision", "audio"),
                        coefficients=coefficients,
                    )
                )
            signal = torch.stack(
                [
                    0.5 * (vision_per_sample.detach() + audio_per_sample.detach()) - fusion_per_sample.detach(),
                    0.5 * (text_per_sample.detach() + audio_per_sample.detach()) - fusion_per_sample.detach(),
                    0.5 * (text_per_sample.detach() + vision_per_sample.detach()) - fusion_per_sample.detach(),
                ],
                dim=1,
            )
            fgm_state.update(signal)
            signal_means = fgm_state.mean_signal()
            losses.update(
                {
                    "fgm_coef_text": coefficients["text"].mean(),
                    "fgm_coef_vision": coefficients["vision"].mean(),
                    "fgm_coef_audio": coefficients["audio"].mean(),
                    "fgm_signal_text": signal_means["text"].to(predictions.device),
                    "fgm_signal_vision": signal_means["vision"].to(predictions.device),
                    "fgm_signal_audio": signal_means["audio"].to(predictions.device),
                }
            )
        return predictions, losses, handles

    predictions = forward_batch(model, batch)
    loss = criterion(predictions, batch["labels"]).mean()
    return predictions, {"loss": loss}, handles


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int | None = None,
    show_progress: bool = False,
    fgm_state: CMIFGMState | None = None,
) -> dict[str, float]:
    model.train()
    criterion = nn.MSELoss(reduction="none")
    loss_totals: dict[str, float] = {}
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
        predictions, losses, handles = forward_and_losses(model, batch, criterion, fgm_state=fgm_state)
        loss = losses["loss"]
        loss.backward()
        for handle in handles:
            handle.remove()
        optimizer.step()

        batch_size = batch["labels"].shape[0]
        for name, value in losses.items():
            loss_totals[name] = loss_totals.get(name, 0.0) + float(value.item()) * batch_size
        total_samples += batch_size
        all_preds.append(predictions.detach().cpu())
        all_labels.append(batch["labels"].detach().cpu())
        if show_progress:
            iterator.set_postfix({"loss": loss_totals.get("loss", 0.0) / max(1, total_samples)})

    metrics = regression_metrics(
        torch.cat(all_preds),
        torch.cat(all_labels),
        loss_totals.get("loss", 0.0) / max(1, total_samples),
    )
    metrics.update({name: total / max(1, total_samples) for name, total in loss_totals.items()})
    return metrics


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
    criterion = nn.MSELoss(reduction="none")
    loss_totals: dict[str, float] = {}
    total_samples = 0
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    iterator = loader
    if show_progress:
        desc = split_name if epoch is None else f"{split_name} epoch {epoch}"
        iterator = tqdm(loader, desc=desc, leave=False)

    for raw_batch in iterator:
        batch = batch_to_device(raw_batch, device)
        predictions, losses, handles = forward_and_losses(model, batch, criterion)
        for handle in handles:
            handle.remove()

        batch_size = batch["labels"].shape[0]
        for name, value in losses.items():
            loss_totals[name] = loss_totals.get(name, 0.0) + float(value.item()) * batch_size
        total_samples += batch_size
        all_preds.append(predictions.detach().cpu())
        all_labels.append(batch["labels"].detach().cpu())
        if show_progress:
            iterator.set_postfix({"loss": loss_totals.get("loss", 0.0) / max(1, total_samples)})

    metrics = regression_metrics(
        torch.cat(all_preds),
        torch.cat(all_labels),
        loss_totals.get("loss", 0.0) / max(1, total_samples),
    )
    metrics.update({name: total / max(1, total_samples) for name, total in loss_totals.items()})
    return metrics


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
    parts = [
        f"{prefix}_loss={metrics['loss']:.4f}",
        f"{prefix}_mae={metrics['mae']:.4f}",
        f"{prefix}_corr={metrics['corr']:.4f}",
        f"{prefix}_acc7={metrics['acc7']:.4f}",
        f"{prefix}_acc2={metrics['binary_acc']:.4f}",
        f"{prefix}_f1={metrics['f1']:.4f}",
    ]
    for name in ("fusion_loss", "text_loss", "vision_loss", "audio_loss"):
        if name in metrics:
            parts.append(f"{prefix}_{name}={metrics[name]:.4f}")
    for name in (
        "fgm_coef_text",
        "fgm_coef_vision",
        "fgm_coef_audio",
        "fgm_signal_text",
        "fgm_signal_vision",
        "fgm_signal_audio",
    ):
        if name in metrics:
            parts.append(f"{prefix}_{name}={metrics[name]:.4f}")
    return " ".join(parts)


def format_epoch_report(epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float]) -> str:
    def metric(metrics: dict[str, float], name: str) -> str:
        return f"{metrics[name]:.4f}" if name in metrics else "-"

    lines = [
        f"Epoch {epoch:03d}",
        "  train | "
        f"loss {metric(train_metrics, 'loss')} | "
        f"fusion {metric(train_metrics, 'fusion_loss')} | "
        f"text {metric(train_metrics, 'text_loss')} | "
        f"vision {metric(train_metrics, 'vision_loss')} | "
        f"audio {metric(train_metrics, 'audio_loss')} | "
        f"mae {metric(train_metrics, 'mae')} | "
        f"corr {metric(train_metrics, 'corr')} | "
        f"acc7 {metric(train_metrics, 'acc7')} | "
        f"acc2 {metric(train_metrics, 'binary_acc')} | "
        f"f1 {metric(train_metrics, 'f1')}",
    ]
    if "fgm_coef_text" in train_metrics:
        lines.append(
            "  fgm   | "
            f"coef_t {metric(train_metrics, 'fgm_coef_text')} | "
            f"coef_v {metric(train_metrics, 'fgm_coef_vision')} | "
            f"coef_a {metric(train_metrics, 'fgm_coef_audio')} | "
            f"sig_t {metric(train_metrics, 'fgm_signal_text')} | "
            f"sig_v {metric(train_metrics, 'fgm_signal_vision')} | "
            f"sig_a {metric(train_metrics, 'fgm_signal_audio')}"
        )
    lines.append(
        "  val   | "
        f"loss {metric(val_metrics, 'loss')} | "
        f"fusion {metric(val_metrics, 'fusion_loss')} | "
        f"text {metric(val_metrics, 'text_loss')} | "
        f"vision {metric(val_metrics, 'vision_loss')} | "
        f"audio {metric(val_metrics, 'audio_loss')} | "
        f"mae {metric(val_metrics, 'mae')} | "
        f"corr {metric(val_metrics, 'corr')} | "
        f"acc7 {metric(val_metrics, 'acc7')} | "
        f"acc2 {metric(val_metrics, 'binary_acc')} | "
        f"f1 {metric(val_metrics, 'f1')}"
    )
    return "\n".join(lines)


def append_epoch_log(path: Path, record: dict[str, Any], args: argparse.Namespace, split_sizes: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**record, "args": vars(args), "split_sizes": split_sizes}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_history_json(
    path: Path,
    history: list[dict[str, Any]],
    args: argparse.Namespace,
    split_sizes: dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": vars(args),
        "split_sizes": split_sizes,
        "epochs": history,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_history(history: list[dict[str, Any]], path: Path) -> None:
    if not history:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [item["epoch"] for item in history]
    train_loss = [item["train"]["loss"] for item in history]
    val_loss = [item["val"]["loss"] for item in history]
    train_mae = [item["train"]["mae"] for item in history]
    val_mae = [item["val"]["mae"] for item in history]
    train_corr = [item["train"]["corr"] for item in history]
    val_corr = [item["val"]["corr"] for item in history]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    axes[0, 0].plot(epochs, train_loss, label="train")
    axes[0, 0].set_title("Train Loss")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, train_mae, label="train")
    axes[0, 1].set_title("Train MAE")
    axes[0, 1].set_ylabel("MAE")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(epochs, train_corr, label="train")
    axes[0, 2].set_title("Train Correlation")
    axes[0, 2].set_ylabel("Pearson r")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, val_loss, label="val")
    axes[1, 0].set_title("Val Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, val_mae, label="val")
    axes[1, 1].set_title("Val MAE")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("MAE")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(epochs, val_corr, label="val")
    axes[1, 2].set_title("Val Correlation")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("Pearson r")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    def plot_loss_split(split_name: str, output_path: Path) -> None:
        split_metrics = [item[split_name] for item in history]
        total_loss = [metrics["loss"] for metrics in split_metrics]
        fusion_loss = [metrics.get("fusion_loss") for metrics in split_metrics]
        text_loss = [metrics.get("text_loss") for metrics in split_metrics]
        vision_loss = [metrics.get("vision_loss") for metrics in split_metrics]
        audio_loss = [metrics.get("audio_loss") for metrics in split_metrics]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, total_loss, label="total")
        if all(value is not None for value in fusion_loss):
            ax.plot(epochs, fusion_loss, label="fusion")
        if all(value is not None for value in text_loss):
            ax.plot(epochs, text_loss, label="text")
        if all(value is not None for value in vision_loss):
            ax.plot(epochs, vision_loss, label="vision")
        if all(value is not None for value in audio_loss):
            ax.plot(epochs, audio_loss, label="audio")
        ax.set_title(f"{split_name.title()} Loss Curves")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)

    plot_loss_split("train", path.with_name("train_loss_curves.png"))
    plot_loss_split("val", path.with_name("val_loss_curves.png"))


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
    fgm_state = build_fgm_state(args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    history_path = output_dir / "history.jsonl"
    history_json_path = output_dir / "history.json"
    curve_path = output_dir / "curves.png"
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
            fgm_state=fgm_state,
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

        print(format_epoch_report(epoch, train_metrics, val_metrics))
        record = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        append_epoch_log(history_path, record, args, sizes)
        write_history_json(history_json_path, history, args, sizes)
        plot_history(history, curve_path)

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, {"train": train_metrics, "val": val_metrics}, args)

    best_state = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_state["model"])
    test_metrics = evaluate(model, test_loader, device, split_name="test", show_progress=not args.no_progress)
    result = {
        "best_epoch": float(best_epoch),
        "best_val_mae": float(best_val_mae),
        **{f"test_{name}": float(value) for name, value in test_metrics.items()},
    }
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
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=True)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--fgm", dest="fgm", action="store_true", help="Enable CMI-FGM gradient modulation.")
    parser.add_argument("--no-fgm", dest="fgm", action="store_false", help="Disable CMI-FGM gradient modulation.")
    parser.set_defaults(fgm=True)
    parser.add_argument("--fgm-lambda", type=float, default=0.5)
    parser.add_argument("--fgm-tau", type=float, default=1.0)
    parser.add_argument("--fgm-momentum", type=float, default=0.9)
    parser.add_argument("--fgm-warmup-steps", type=int, default=0)
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
