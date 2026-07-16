from __future__ import annotations

import argparse
import ast
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from RGB_v4.models import RGBBaseline


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


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    return RGBBaseline(num_classes=num_classes, pretrained=pretrained)


def prepare_run_output_dir(args: argparse.Namespace) -> Path:
    parent_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    fields = [
        timestamp,
        f"seed{getattr(args, 'seed', 0)}",
        f"lr{getattr(args, 'lr', 0):g}",
        f"bs{getattr(args, 'batch_size', 0)}",
    ]
    run_dir = parent_dir / "_".join(fields)
    suffix = 1
    while run_dir.exists():
        run_dir = parent_dir / ("_".join(fields) + f"_{suffix:02d}")
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    args.output_dir = str(run_dir)
    return run_dir


def clone_model_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def parse_lr_milestones(value: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        parsed = ast.literal_eval(text)
        if not isinstance(parsed, (list, tuple)):
            raise ValueError(f"Expected lr milestones list, got {value!r}.")
        return [int(item) for item in parsed]
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace | None = None,
) -> torch.optim.lr_scheduler.LRScheduler:
    scheduler_name = getattr(args, "lr_scheduler", "multistep") if args is not None else "multistep"
    if scheduler_name == "cosine":
        epochs = int(getattr(args, "epochs", 100)) if args is not None else 100
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    milestones = parse_lr_milestones(getattr(args, "lr_decay_step", "[60]") if args is not None else "[60]")
    gamma = float(getattr(args, "lr_decay_ratio", 0.1)) if args is not None else 0.1
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    rgb = batch["rgb"].to(device, non_blocking=True)
    depth = batch["depth"].to(device, non_blocking=True)
    label = batch["label"].to(device, non_blocking=True)
    return (rgb, depth), label


def forward_and_losses(
    model: nn.Module,
    inputs: tuple[torch.Tensor, torch.Tensor],
    labels: torch.Tensor,
    criterion: nn.Module,
    rgb_loss_weight: float = 1.0,
    depth_loss_weight: float = 1.0,
    detach_probe_features: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if hasattr(model, "forward_with_modal_logits"):
        outputs = model.forward_with_modal_logits(*inputs, detach_probe_features=detach_probe_features)
        logits = outputs["logits"]
        fusion_per_sample = criterion(logits, labels)
        rgb_per_sample = criterion(outputs["rgb_logits"], labels)
        depth_per_sample = criterion(outputs["depth_logits"], labels)
        fusion_loss = fusion_per_sample.mean()
        rgb_loss = rgb_per_sample.mean()
        depth_loss = depth_per_sample.mean()
        losses = {
            "loss": fusion_loss + rgb_loss_weight * rgb_loss + depth_loss_weight * depth_loss,
            "fusion_loss": fusion_loss,
            "rgb_loss": rgb_loss,
            "depth_loss": depth_loss,
            "rgb_acc": (outputs["rgb_logits"].argmax(dim=1) == labels).float().mean(),
            "depth_acc": (outputs["depth_logits"].argmax(dim=1) == labels).float().mean(),
        }
        return logits, losses

    logits = model(*inputs)
    loss = criterion(logits, labels).mean()
    return logits, {"loss": loss, "fusion_loss": loss}


def macro_f1_score(predictions: torch.Tensor, labels: torch.Tensor, num_classes: int | None = None) -> float:
    predictions = predictions.detach().view(-1).cpu()
    labels = labels.detach().view(-1).cpu()
    if predictions.numel() == 0:
        return 0.0
    if num_classes is None:
        num_classes = int(torch.cat([predictions, labels]).max().item()) + 1

    f1_values = []
    for class_index in range(num_classes):
        pred_positive = predictions == class_index
        label_positive = labels == class_index
        tp = float((pred_positive & label_positive).sum().item())
        fp = float((pred_positive & ~label_positive).sum().item())
        fn = float((~pred_positive & label_positive).sum().item())
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1_values.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(sum(f1_values) / max(len(f1_values), 1))


def update_metric_totals(
    totals: dict[str, float],
    losses: dict[str, torch.Tensor],
    batch_size: int,
) -> None:
    for name, value in losses.items():
        totals[name] = totals.get(name, 0.0) + float(value.item()) * batch_size


def average_metrics(
    totals: dict[str, float],
    total_samples: int,
    total_correct: int,
    predictions: list[torch.Tensor] | None = None,
    labels: list[torch.Tensor] | None = None,
) -> dict[str, float]:
    metrics = {name: value / max(1, total_samples) for name, value in totals.items()}
    metrics["acc"] = total_correct / max(1, total_samples)
    if predictions and labels:
        metrics["macro_f1"] = macro_f1_score(torch.cat(predictions), torch.cat(labels))
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int | None = None,
    show_progress: bool = False,
    rgb_loss_weight: float = 1.0,
    depth_loss_weight: float = 1.0,
    detach_probe_features: bool = False,
) -> dict[str, float]:
    model.train()
    criterion = nn.CrossEntropyLoss(reduction="none")
    totals: dict[str, float] = {}
    total_correct = 0
    total_samples = 0
    all_predictions: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    iterator = loader
    if show_progress:
        desc = "train" if epoch is None else f"train epoch {epoch}"
        iterator = tqdm(loader, desc=desc, leave=False)

    for batch in iterator:
        inputs, labels = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits, losses = forward_and_losses(
            model,
            inputs,
            labels,
            criterion,
            rgb_loss_weight=rgb_loss_weight,
            depth_loss_weight=depth_loss_weight,
            detach_probe_features=detach_probe_features,
        )
        loss = losses["loss"]
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        update_metric_totals(totals, losses, batch_size)
        predictions = logits.detach().argmax(dim=1)
        total_correct += int((predictions == labels).sum().item())
        total_samples += batch_size
        all_predictions.append(predictions.cpu())
        all_labels.append(labels.detach().cpu())

        if show_progress:
            postfix = {"loss": totals["loss"] / max(1, total_samples), "acc": total_correct / max(1, total_samples)}
            if "fusion_loss" in totals:
                postfix["f_loss"] = totals["fusion_loss"] / max(1, total_samples)
            if "rgb_loss" in totals:
                postfix["rgb_loss"] = totals["rgb_loss"] / max(1, total_samples)
            if "depth_loss" in totals:
                postfix["depth_loss"] = totals["depth_loss"] / max(1, total_samples)
            iterator.set_postfix(postfix)

    return average_metrics(totals, total_samples, total_correct, all_predictions, all_labels)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int | None = None,
    split_name: str = "eval",
    show_progress: bool = False,
    rgb_loss_weight: float = 1.0,
    depth_loss_weight: float = 1.0,
    detach_probe_features: bool = False,
) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    totals: dict[str, float] = {}
    total_correct = 0
    total_samples = 0
    all_predictions: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    iterator = loader
    if show_progress:
        desc = split_name if epoch is None else f"{split_name} epoch {epoch}"
        iterator = tqdm(loader, desc=desc, leave=False)

    for batch in iterator:
        inputs, labels = batch_to_device(batch, device)
        logits, losses = forward_and_losses(
            model,
            inputs,
            labels,
            criterion,
            rgb_loss_weight=rgb_loss_weight,
            depth_loss_weight=depth_loss_weight,
            detach_probe_features=detach_probe_features,
        )

        batch_size = labels.size(0)
        update_metric_totals(totals, losses, batch_size)
        predictions = logits.argmax(dim=1)
        total_correct += int((predictions == labels).sum().item())
        total_samples += batch_size
        all_predictions.append(predictions.cpu())
        all_labels.append(labels.detach().cpu())

        if show_progress:
            postfix = {"loss": totals["loss"] / max(1, total_samples), "acc": total_correct / max(1, total_samples)}
            if "fusion_loss" in totals:
                postfix["f_loss"] = totals["fusion_loss"] / max(1, total_samples)
            if "rgb_loss" in totals:
                postfix["rgb_loss"] = totals["rgb_loss"] / max(1, total_samples)
            if "depth_loss" in totals:
                postfix["depth_loss"] = totals["depth_loss"] / max(1, total_samples)
            iterator.set_postfix(postfix)

    return average_metrics(totals, total_samples, total_correct, all_predictions, all_labels)


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
    parts = [f"{prefix}_loss={metrics['loss']:.4f}"]
    if "fusion_loss" in metrics:
        parts.append(f"{prefix}_fusion_loss={metrics['fusion_loss']:.4f}")
    if "rgb_loss" in metrics:
        parts.append(f"{prefix}_rgb_loss={metrics['rgb_loss']:.4f}")
    if "depth_loss" in metrics:
        parts.append(f"{prefix}_depth_loss={metrics['depth_loss']:.4f}")
    if "rgb_acc" in metrics:
        parts.append(f"{prefix}_rgb_acc={metrics['rgb_acc']:.4f}")
    if "depth_acc" in metrics:
        parts.append(f"{prefix}_depth_acc={metrics['depth_acc']:.4f}")
    parts.append(f"{prefix}_acc={metrics['acc']:.4f}")
    if "macro_f1" in metrics:
        parts.append(f"{prefix}_macro_f1={metrics['macro_f1']:.4f}")
    return " ".join(parts)


def format_epoch_report(
    epoch: int,
    train_metrics: dict[str, float],
    eval_metrics: dict[str, float],
    eval_split_name: str = "test",
) -> str:
    def metric(metrics: dict[str, float], name: str) -> str:
        return f"{metrics[name]:.4f}" if name in metrics else "-"

    return "\n".join(
        [
            f"Epoch {epoch:03d}",
            "  train | "
            f"loss {metric(train_metrics, 'loss')} | "
            f"fusion {metric(train_metrics, 'fusion_loss')} | "
            f"rgb {metric(train_metrics, 'rgb_loss')} | "
            f"depth {metric(train_metrics, 'depth_loss')} | "
            f"acc {metric(train_metrics, 'acc')} | "
            f"macroF1 {metric(train_metrics, 'macro_f1')} | "
            f"rgb_acc {metric(train_metrics, 'rgb_acc')} | "
            f"depth_acc {metric(train_metrics, 'depth_acc')}",
            f"  {eval_split_name:<5} | "
            f"loss {metric(eval_metrics, 'loss')} | "
            f"fusion {metric(eval_metrics, 'fusion_loss')} | "
            f"rgb {metric(eval_metrics, 'rgb_loss')} | "
            f"depth {metric(eval_metrics, 'depth_loss')} | "
            f"acc {metric(eval_metrics, 'acc')} | "
            f"macroF1 {metric(eval_metrics, 'macro_f1')} | "
            f"rgb_acc {metric(eval_metrics, 'rgb_acc')} | "
            f"depth_acc {metric(eval_metrics, 'depth_acc')}",
        ]
    )


def args_to_dict(args: argparse.Namespace) -> dict[str, Any]:
    return vars(args)


def append_epoch_log(
    path: Path,
    record: dict[str, Any],
    args: argparse.Namespace,
    split_sizes: dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **record,
        "args": args_to_dict(args),
        "split_sizes": split_sizes,
    }
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
        "args": args_to_dict(args),
        "split_sizes": split_sizes,
        "epochs": history,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_history(history: list[dict[str, Any]], path: Path) -> None:
    if not history:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [item["epoch"] for item in history]
    eval_key = "test" if "test" in history[0] else "val"
    train_loss = [item["train"]["loss"] for item in history]
    eval_loss = [item[eval_key]["loss"] for item in history]
    train_rgb_loss = [item["train"].get("rgb_loss") for item in history]
    eval_rgb_loss = [item[eval_key].get("rgb_loss") for item in history]
    train_depth_loss = [item["train"].get("depth_loss") for item in history]
    eval_depth_loss = [item[eval_key].get("depth_loss") for item in history]
    train_acc = [item["train"]["acc"] for item in history]
    eval_acc = [item[eval_key]["acc"] for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, eval_loss, label=eval_key)
    if all(value is not None for value in train_rgb_loss):
        axes[0].plot(epochs, train_rgb_loss, label="train rgb")
    if all(value is not None for value in eval_rgb_loss):
        axes[0].plot(epochs, eval_rgb_loss, label=f"{eval_key} rgb")
    if all(value is not None for value in train_depth_loss):
        axes[0].plot(epochs, train_depth_loss, label="train depth")
    if all(value is not None for value in eval_depth_loss):
        axes[0].plot(epochs, eval_depth_loss, label=f"{eval_key} depth")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train")
    axes[1].plot(epochs, eval_acc, label=eval_key)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
