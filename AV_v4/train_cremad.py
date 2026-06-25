from __future__ import annotations

import argparse
import ast
import json
import random
import sys
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
from AV_v4.datasets import (
    CREMADAVDataset,
    CREMADTrainImageTransform,
    ResizeToTensorNormalize,
    discover_cremad_samples,
    split_samples_from_csv,
)
from AV_v4.models import AVBaseline, AudioBaseline, VisualBaseline


VISUAL_AUGMENTATION_PRESETS = {
    "light": {
        "scale": (0.85, 1.0),
        "ratio": (0.95, 1.05),
        "horizontal_flip_prob": 0.0,
    },
    "medium": {
        "scale": (0.7, 1.0),
        "ratio": (0.9, 1.1),
        "horizontal_flip_prob": 0.0,
    },
    "strong": {
        "scale": (0.5, 1.0),
        "ratio": (0.8, 1.25),
        "horizontal_flip_prob": 0.2,
    },
}


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


def build_model(modality: str, num_classes: int = 6) -> nn.Module:
    if modality == "av":
        return AVBaseline(num_classes=num_classes)
    if modality == "audio":
        return AudioBaseline(num_classes=num_classes)
    if modality == "visual":
        return VisualBaseline(num_classes=num_classes)
    raise ValueError(f"Unsupported modality: {modality}")


def build_fgm_state(args: argparse.Namespace) -> CMIFGMState | None:
    if not getattr(args, "fgm", False):
        return None
    if args.modality != "av":
        return None
    return CMIFGMState(
        modalities=("audio", "visual"),
        strength=args.fgm_lambda,
        temperature=args.fgm_tau,
        momentum=args.fgm_momentum,
        warmup_steps=args.fgm_warmup_steps,
    )


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

    milestones = parse_lr_milestones(getattr(args, "lr_decay_step", "[70]") if args is not None else "[70]")
    gamma = float(getattr(args, "lr_decay_ratio", 0.1)) if args is not None else 0.1
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=gamma,
    )


def parse_float_pair(value: str | tuple[float, float] | list[float], name: str) -> tuple[float, float]:
    if isinstance(value, (tuple, list)):
        values = [float(item) for item in value]
    else:
        text = str(value).strip()
        if text.startswith("["):
            parsed = ast.literal_eval(text)
            if not isinstance(parsed, (tuple, list)):
                raise ValueError(f"Expected {name} pair, got {value!r}.")
            values = [float(item) for item in parsed]
        else:
            values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != 2:
        raise ValueError(f"Expected {name} to contain exactly two values, got {value!r}.")
    return values[0], values[1]


def build_train_image_transform(args: argparse.Namespace) -> ResizeToTensorNormalize:
    visual_aug = getattr(args, "visual_aug", "none")
    image_size = getattr(args, "image_size", 224)
    if visual_aug == "none":
        return ResizeToTensorNormalize(size=image_size)

    preset = VISUAL_AUGMENTATION_PRESETS.get(visual_aug, VISUAL_AUGMENTATION_PRESETS["light"])
    scale = preset["scale"]
    ratio = preset["ratio"]
    horizontal_flip_prob = preset["horizontal_flip_prob"]

    if getattr(args, "aug_scale", None) is not None:
        scale = parse_float_pair(args.aug_scale, "aug_scale")
    if getattr(args, "aug_ratio", None) is not None:
        ratio = parse_float_pair(args.aug_ratio, "aug_ratio")
    if getattr(args, "aug_hflip_prob", None) is not None:
        horizontal_flip_prob = float(args.aug_hflip_prob)

    return CREMADTrainImageTransform(
        size=image_size,
        scale=scale,
        ratio=ratio,
        horizontal_flip_prob=horizontal_flip_prob,
    )


def batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    modality: str,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    label = batch["label"].to(device, non_blocking=True)

    if modality == "audio":
        audio = batch["audio"].to(device, non_blocking=True)
        if audio.ndim == 3:
            audio = audio.unsqueeze(1)
        return (audio,), label

    if modality == "visual":
        visual = batch["visual"].to(device, non_blocking=True)
        return (visual,), label

    audio = batch["audio"].to(device, non_blocking=True)
    if audio.ndim == 3:
        audio = audio.unsqueeze(1)
    visual = batch["visual"].to(device, non_blocking=True)
    return (audio, visual), label


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> int:
    predictions = logits.argmax(dim=1)
    return int((predictions == labels).sum().item())


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


def forward_and_losses(
    model: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    labels: torch.Tensor,
    modality: str,
    criterion: nn.Module,
    fgm_state: CMIFGMState | None = None,
    audio_loss_weight: float = 1.0,
    visual_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[torch.utils.hooks.RemovableHandle]]:
    handles: list[torch.utils.hooks.RemovableHandle] = []
    if modality == "av" and hasattr(model, "forward_with_modal_logits"):
        detach_probe_features = True
        if fgm_state is not None and fgm_state.num_updates < fgm_state.warmup_steps:
            detach_probe_features = False
        outputs = model.forward_with_modal_logits(*inputs, detach_probe_features=detach_probe_features)
        logits = outputs["logits"]
        fusion_per_sample = criterion(logits, labels)
        audio_per_sample = criterion(outputs["audio_logits"], labels)
        visual_per_sample = criterion(outputs["visual_logits"], labels)
        fusion_loss = fusion_per_sample.mean()
        audio_loss = audio_per_sample.mean()
        visual_loss = visual_per_sample.mean()
        losses = {
            "loss": (
                fusion_loss
                + audio_loss_weight * audio_loss
                + visual_loss_weight * visual_loss
            ),
            "fusion_loss": fusion_loss,
            "audio_loss": audio_loss,
            "visual_loss": visual_loss,
            "audio_acc": (outputs["audio_logits"].argmax(dim=1) == labels).float().mean(),
            "visual_acc": (outputs["visual_logits"].argmax(dim=1) == labels).float().mean(),
        }
        if fgm_state is not None:
            batch_size = labels.size(0)
            coefficients = fgm_state.coefficients(batch_size, logits.device, logits.dtype)
            handles.extend(
                register_feature_gradient_hooks(
                    {
                        "audio": outputs["audio_feature"],
                        "visual": outputs["visual_feature"],
                    },
                    coefficients,
                )
            )
            if isinstance(getattr(model, "classifier", None), nn.Linear):
                handles.append(
                    register_split_linear_weight_hook(
                        model.classifier,
                        split_sizes=(512, 512),
                        modalities=("audio", "visual"),
                        coefficients=coefficients,
                    )
                )
            signal = torch.stack(
                [
                    visual_per_sample.detach() - fusion_per_sample.detach(),
                    audio_per_sample.detach() - fusion_per_sample.detach(),
                ],
                dim=1,
            )
            fgm_state.update(signal)
            signal_means = fgm_state.mean_signal()
            losses.update(
                {
                    "fgm_coef_audio": coefficients["audio"].mean(),
                    "fgm_coef_visual": coefficients["visual"].mean(),
                    "fgm_signal_audio": signal_means["audio"].to(logits.device),
                    "fgm_signal_visual": signal_means["visual"].to(logits.device),
                }
            )
        return logits, losses, handles

    logits = model(*inputs)
    loss = criterion(logits, labels).mean()
    losses = {"loss": loss}
    if modality == "audio":
        losses["audio_loss"] = loss
    elif modality == "visual":
        losses["visual_loss"] = loss
    return logits, losses, handles


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
        all_predictions = torch.cat(predictions)
        all_labels = torch.cat(labels)
        metrics["macro_f1"] = macro_f1_score(all_predictions, all_labels)
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    modality: str,
    epoch: int | None = None,
    show_progress: bool = False,
    fgm_state: CMIFGMState | None = None,
    audio_loss_weight: float = 1.0,
    visual_loss_weight: float = 1.0,
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
        inputs, labels = batch_to_device(batch, device, modality)

        optimizer.zero_grad(set_to_none=True)
        logits, losses, handles = forward_and_losses(
            model,
            inputs,
            labels,
            modality,
            criterion,
            fgm_state=fgm_state,
            audio_loss_weight=audio_loss_weight,
            visual_loss_weight=visual_loss_weight,
        )
        loss = losses["loss"]
        loss.backward()
        for handle in handles:
            handle.remove()
        optimizer.step()

        batch_size = labels.size(0)
        update_metric_totals(totals, losses, batch_size)
        predictions = logits.detach().argmax(dim=1)
        total_correct += int((predictions == labels).sum().item())
        total_samples += batch_size
        all_predictions.append(predictions.cpu())
        all_labels.append(labels.detach().cpu())

        if show_progress:
            postfix = {"loss": totals["loss"] / max(1, total_samples)}
            if "fusion_loss" in totals:
                postfix["f_loss"] = totals["fusion_loss"] / max(1, total_samples)
            if "audio_loss" in totals:
                postfix["a_loss"] = totals["audio_loss"] / max(1, total_samples)
            if "visual_loss" in totals:
                postfix["v_loss"] = totals["visual_loss"] / max(1, total_samples)
            postfix["acc"] = total_correct / max(1, total_samples)
            iterator.set_postfix(postfix)

    return average_metrics(totals, total_samples, total_correct, all_predictions, all_labels)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    modality: str,
    epoch: int | None = None,
    split_name: str = "eval",
    show_progress: bool = False,
    audio_loss_weight: float = 1.0,
    visual_loss_weight: float = 1.0,
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
        inputs, labels = batch_to_device(batch, device, modality)
        logits, losses, handles = forward_and_losses(
            model,
            inputs,
            labels,
            modality,
            criterion,
            audio_loss_weight=audio_loss_weight,
            visual_loss_weight=visual_loss_weight,
        )
        for handle in handles:
            handle.remove()

        batch_size = labels.size(0)
        update_metric_totals(totals, losses, batch_size)
        predictions = logits.argmax(dim=1)
        total_correct += int((predictions == labels).sum().item())
        total_samples += batch_size
        all_predictions.append(predictions.cpu())
        all_labels.append(labels.detach().cpu())

        if show_progress:
            postfix = {"loss": totals["loss"] / max(1, total_samples)}
            if "fusion_loss" in totals:
                postfix["f_loss"] = totals["fusion_loss"] / max(1, total_samples)
            if "audio_loss" in totals:
                postfix["a_loss"] = totals["audio_loss"] / max(1, total_samples)
            if "visual_loss" in totals:
                postfix["v_loss"] = totals["visual_loss"] / max(1, total_samples)
            postfix["acc"] = total_correct / max(1, total_samples)
            iterator.set_postfix(postfix)

    return average_metrics(totals, total_samples, total_correct, all_predictions, all_labels)


def create_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    samples = discover_cremad_samples(args.data_root)
    split = split_samples_from_csv(samples, args.split_csv_root)
    train_image_transform = build_train_image_transform(args)
    eval_image_transform = ResizeToTensorNormalize(size=args.image_size)

    datasets = {
        name: CREMADAVDataset(
            part,
            modality=args.modality,
            fps=args.fps,
            audio_duration=args.audio_duration,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            win_length=args.win_length,
            image_transform=train_image_transform if name == "train" else eval_image_transform,
        )
        for name, part in split.items()
    }

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }

    train_loader = DataLoader(datasets["train"], shuffle=True, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(datasets["test"], shuffle=False, drop_last=False, **loader_kwargs)
    sizes = {name: len(dataset) for name, dataset in datasets.items()}
    sizes["val"] = sizes["test"]
    val_loader = test_loader
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
    ]
    if "fusion_loss" in metrics:
        parts.append(f"{prefix}_fusion_loss={metrics['fusion_loss']:.4f}")
    if "audio_loss" in metrics:
        parts.append(f"{prefix}_audio_loss={metrics['audio_loss']:.4f}")
    if "visual_loss" in metrics:
        parts.append(f"{prefix}_visual_loss={metrics['visual_loss']:.4f}")
    if "audio_acc" in metrics:
        parts.append(f"{prefix}_audio_acc={metrics['audio_acc']:.4f}")
    if "visual_acc" in metrics:
        parts.append(f"{prefix}_visual_acc={metrics['visual_acc']:.4f}")
    if "fgm_coef_audio" in metrics:
        parts.append(f"{prefix}_fgm_coef_audio={metrics['fgm_coef_audio']:.4f}")
    if "fgm_coef_visual" in metrics:
        parts.append(f"{prefix}_fgm_coef_visual={metrics['fgm_coef_visual']:.4f}")
    if "fgm_signal_audio" in metrics:
        parts.append(f"{prefix}_fgm_signal_audio={metrics['fgm_signal_audio']:.4f}")
    if "fgm_signal_visual" in metrics:
        parts.append(f"{prefix}_fgm_signal_visual={metrics['fgm_signal_visual']:.4f}")
    parts.append(f"{prefix}_acc={metrics['acc']:.4f}")
    if "macro_f1" in metrics:
        parts.append(f"{prefix}_macro_f1={metrics['macro_f1']:.4f}")
    return " ".join(parts)


def format_epoch_report(epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float]) -> str:
    def metric(metrics: dict[str, float], name: str) -> str:
        return f"{metrics[name]:.4f}" if name in metrics else "-"

    lines = [
        f"Epoch {epoch:03d}",
        "  train | "
        f"loss {metric(train_metrics, 'loss')} | "
        f"fusion {metric(train_metrics, 'fusion_loss')} | "
        f"audio {metric(train_metrics, 'audio_loss')} | "
        f"visual {metric(train_metrics, 'visual_loss')} | "
        f"acc {metric(train_metrics, 'acc')} | "
        f"macroF1 {metric(train_metrics, 'macro_f1')} | "
        f"a_acc {metric(train_metrics, 'audio_acc')} | "
        f"v_acc {metric(train_metrics, 'visual_acc')}",
    ]
    if "fgm_coef_audio" in train_metrics:
        lines.append(
            "  fgm   | "
            f"coef_a {metric(train_metrics, 'fgm_coef_audio')} | "
            f"coef_v {metric(train_metrics, 'fgm_coef_visual')} | "
            f"sig_a {metric(train_metrics, 'fgm_signal_audio')} | "
            f"sig_v {metric(train_metrics, 'fgm_signal_visual')}"
        )
    lines.append(
        "  val   | "
        f"loss {metric(val_metrics, 'loss')} | "
        f"fusion {metric(val_metrics, 'fusion_loss')} | "
        f"audio {metric(val_metrics, 'audio_loss')} | "
        f"visual {metric(val_metrics, 'visual_loss')} | "
        f"acc {metric(val_metrics, 'acc')} | "
        f"macroF1 {metric(val_metrics, 'macro_f1')} | "
        f"a_acc {metric(val_metrics, 'audio_acc')} | "
        f"v_acc {metric(val_metrics, 'visual_acc')}"
    )
    return "\n".join(lines)


def args_to_dict(args: argparse.Namespace) -> dict[str, Any]:
    if isinstance(args, argparse.Namespace):
        return vars(args)
    return {
        name: getattr(args, name)
        for name in dir(args)
        if not name.startswith("_") and not callable(getattr(args, name))
    }


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
    train_loss = [item["train"]["loss"] for item in history]
    val_loss = [item["val"]["loss"] for item in history]
    train_acc = [item["train"]["acc"] for item in history]
    val_acc = [item["val"]["acc"] for item in history]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    axes[0, 0].plot(epochs, train_loss, label="train total")
    axes[0, 0].set_title("Train Loss")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, train_acc, label="train fusion")
    axes[0, 1].set_title("Train Accuracy")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, val_loss, label="val total")
    axes[1, 0].set_title("Val Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, val_acc, label="val fusion")
    axes[1, 1].set_title("Val Accuracy")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    def plot_loss_split(split_name: str, output_path: Path) -> None:
        split_metrics = [item[split_name] for item in history]
        total_loss = [metrics["loss"] for metrics in split_metrics]
        fusion_loss = [metrics.get("fusion_loss") for metrics in split_metrics]
        audio_loss = [metrics.get("audio_loss") for metrics in split_metrics]
        visual_loss = [metrics.get("visual_loss") for metrics in split_metrics]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, total_loss, label="total")
        if all(value is not None for value in fusion_loss):
            ax.plot(epochs, fusion_loss, label="fusion")
        if all(value is not None for value in audio_loss):
            ax.plot(epochs, audio_loss, label="audio")
        if all(value is not None for value in visual_loss):
            ax.plot(epochs, visual_loss, label="visual")
        ax.set_title(f"{split_name.title()} Loss Curves")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)

    def plot_accuracy_split(split_name: str, output_path: Path) -> None:
        split_metrics = [item[split_name] for item in history]
        fusion_acc = [metrics["acc"] for metrics in split_metrics]
        audio_acc = [metrics.get("audio_acc") for metrics in split_metrics]
        visual_acc = [metrics.get("visual_acc") for metrics in split_metrics]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, fusion_acc, label="fusion")
        if all(value is not None for value in audio_acc):
            ax.plot(epochs, audio_acc, label="audio")
        if all(value is not None for value in visual_acc):
            ax.plot(epochs, visual_acc, label="visual")
        ax.set_title(f"{split_name.title()} Modality Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.0)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)

    plot_loss_split("train", path.with_name("train_loss_curves.png"))
    plot_loss_split("val", path.with_name("val_loss_curves.png"))
    plot_accuracy_split("train", path.with_name("train_modality_accuracy.png"))
    plot_accuracy_split("val", path.with_name("val_modality_accuracy.png"))


def run_training(args: argparse.Namespace) -> dict[str, float]:
    set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    train_loader, val_loader, test_loader, sizes = create_dataloaders(args)
    if sizes["test"] == 0:
        raise ValueError(f"Empty test split: {sizes}")

    model = build_model(args.modality, num_classes=args.num_classes).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    fgm_state = build_fgm_state(args)
    scheduler = build_scheduler(optimizer, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(args_to_dict(args), indent=2), encoding="utf-8")
    history_path = output_dir / "history.jsonl"
    history_json_path = output_dir / "history.json"
    curve_path = output_dir / "curves.png"
    history: list[dict[str, Any]] = []

    best_val_acc = -1.0
    best_epoch = 0
    best_metrics: dict[str, float] = {}

    print(f"Split sizes: {sizes}")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.modality,
            epoch=epoch,
            show_progress=not args.no_progress,
            fgm_state=fgm_state,
            audio_loss_weight=args.audio_loss_weight,
            visual_loss_weight=args.visual_loss_weight,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            args.modality,
            epoch=epoch,
            split_name="val",
            show_progress=not args.no_progress,
            audio_loss_weight=args.audio_loss_weight,
            visual_loss_weight=args.visual_loss_weight,
        )
        scheduler.step()

        print(format_epoch_report(epoch, train_metrics, val_metrics))

        epoch_record = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_record)
        append_epoch_log(history_path, epoch_record, args, sizes)
        write_history_json(history_json_path, history, args, sizes)
        plot_history(history, curve_path)

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            best_metrics = {"train": train_metrics, "val": val_metrics}
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, best_metrics, args)

    best_state = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_state["model"])
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        args.modality,
        split_name="test",
        show_progress=not args.no_progress,
        audio_loss_weight=args.audio_loss_weight,
        visual_loss_weight=args.visual_loss_weight,
    )
    result = {
        "best_epoch": float(best_epoch),
        "best_val_acc": float(best_val_acc),
        **{f"test_{name}": float(value) for name, value in test_metrics.items()},
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"best_epoch={best_epoch:03d} "
        f"best_val_acc={best_val_acc:.4f} "
        f"{format_metrics('test', test_metrics)}"
    )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CREMA-D audio/visual baseline.")
    parser.add_argument("--data-root", type=str, default="dataset/CREMA-D")
    parser.add_argument("--output-dir", type=str, default="runs/cremad_baseline")
    parser.add_argument("--modality", choices=["av", "audio", "visual"], default="av")
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--lr-scheduler", choices=["multistep", "cosine"], default="multistep")
    parser.add_argument("--lr-decay-step", type=str, default="[60]")
    parser.add_argument("--lr-decay-ratio", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--audio-loss-weight", type=float, default=1.0)
    parser.add_argument("--visual-loss-weight", type=float, default=2.0)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--visual-aug", choices=["none", "light", "medium", "strong", "custom"], default="none")
    parser.add_argument("--aug-scale", type=str, default=None)
    parser.add_argument("--aug-ratio", type=str, default=None)
    parser.add_argument("--aug-hflip-prob", type=float, default=None)
    parser.add_argument("--audio-duration", type=float, default=3.0)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--win-length", type=int, default=400)
    parser.add_argument("--split-csv-root", type=str, default="ICCV2025-GDL-main/dataset/data/CREMAD")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=True)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--fgm", dest="fgm", action="store_true", help="Enable CMI-FGM gradient modulation for AV training.")
    parser.add_argument("--no-fgm", dest="fgm", action="store_false", help="Disable CMI-FGM gradient modulation.")
    parser.set_defaults(fgm=True)
    parser.add_argument("--fgm-lambda", type=float, default=0.5)
    parser.add_argument("--fgm-tau", type=float, default=1.0)
    parser.add_argument("--fgm-momentum", type=float, default=0.9)
    parser.add_argument("--fgm-warmup-steps", type=int, default=12)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
