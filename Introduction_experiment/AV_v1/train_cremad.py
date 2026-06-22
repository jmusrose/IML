from __future__ import annotations

import argparse
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
from openpyxl import Workbook

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from AV_v1.datasets import (
    CREMADAVDataset,
    ResizeToTensorNormalize,
    discover_cremad_samples,
    split_samples_from_csv,
)
from AV_v1.models import AVBaseline, AudioBaseline, VisualBaseline


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


def build_model(modality: str, num_classes: int = 6, fusion_head: str = "linear") -> nn.Module:
    if modality == "av":
        return AVBaseline(num_classes=num_classes, fusion_head=fusion_head)
    if modality == "audio":
        return AudioBaseline(num_classes=num_classes)
    if modality == "visual":
        return VisualBaseline(num_classes=num_classes)
    raise ValueError(f"Unsupported modality: {modality}")


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


def compute_loss_gaps(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "fusion_minus_audio": metrics["fusion_loss"] - metrics["audio_loss"],
        "fusion_minus_visual": metrics["fusion_loss"] - metrics["visual_loss"],
    }


def forward_and_losses(
    model: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    labels: torch.Tensor,
    modality: str,
    criterion: nn.Module,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if modality == "av" and hasattr(model, "forward_with_modal_logits"):
        outputs = model.forward_with_modal_logits(*inputs)
        logits = outputs["logits"]
        fusion_loss = criterion(logits, labels)
        audio_loss = criterion(outputs["audio_logits"], labels)
        visual_loss = criterion(outputs["visual_logits"], labels)
        losses = {
            "loss": fusion_loss + audio_loss + visual_loss,
            "fusion_loss": fusion_loss,
            "audio_loss": audio_loss,
            "visual_loss": visual_loss,
        }
        return logits, losses

    logits = model(*inputs)
    loss = criterion(logits, labels)
    losses = {"loss": loss}
    if modality == "audio":
        losses["audio_loss"] = loss
    elif modality == "visual":
        losses["visual_loss"] = loss
    return logits, losses


def update_metric_totals(
    totals: dict[str, float],
    losses: dict[str, torch.Tensor],
    batch_size: int,
) -> None:
    for name, value in losses.items():
        totals[name] = totals.get(name, 0.0) + float(value.item()) * batch_size


def average_metrics(totals: dict[str, float], total_samples: int, total_correct: int) -> dict[str, float]:
    metrics = {name: value / max(1, total_samples) for name, value in totals.items()}
    metrics["acc"] = total_correct / max(1, total_samples)
    if {"fusion_loss", "audio_loss", "visual_loss"}.issubset(metrics):
        metrics.update(compute_loss_gaps(metrics))
    return metrics


def freeze_encoder(model: nn.Module, encoder: str) -> None:
    if not isinstance(model, AVBaseline):
        raise TypeError("Encoder freezing is only supported for AVBaseline.")
    if encoder == "audio":
        target = model.audio_net
    elif encoder == "visual":
        target = model.visual_net
    else:
        raise ValueError(f"Unsupported encoder to freeze: {encoder}")

    target.eval()
    for param in target.parameters():
        param.requires_grad = False


def freeze_all_encoders(model: nn.Module) -> None:
    freeze_encoder(model, "audio")
    freeze_encoder(model, "visual")


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters remain after freezing.")
    return torch.optim.SGD(
        trainable_params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    modality: str,
    epoch: int | None = None,
    show_progress: bool = False,
) -> dict[str, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    totals: dict[str, float] = {}
    total_correct = 0
    total_samples = 0

    iterator = loader
    if show_progress:
        desc = "train" if epoch is None else f"train epoch {epoch}"
        iterator = tqdm(loader, desc=desc, leave=False)

    for batch in iterator:
        if isinstance(model, AVBaseline):
            if not any(param.requires_grad for param in model.audio_net.parameters()):
                model.audio_net.eval()
            if not any(param.requires_grad for param in model.visual_net.parameters()):
                model.visual_net.eval()

        inputs, labels = batch_to_device(batch, device, modality)

        optimizer.zero_grad(set_to_none=True)
        logits, losses = forward_and_losses(
            model,
            inputs,
            labels,
            modality,
            criterion,
        )
        loss = losses["loss"]
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        update_metric_totals(totals, losses, batch_size)
        total_correct += compute_accuracy(logits.detach(), labels)
        total_samples += batch_size

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

    return average_metrics(totals, total_samples, total_correct)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    modality: str,
    epoch: int | None = None,
    split_name: str = "eval",
    show_progress: bool = False,
) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    totals: dict[str, float] = {}
    total_correct = 0
    total_samples = 0

    iterator = loader
    if show_progress:
        desc = split_name if epoch is None else f"{split_name} epoch {epoch}"
        iterator = tqdm(loader, desc=desc, leave=False)

    for batch in iterator:
        inputs, labels = batch_to_device(batch, device, modality)
        logits, losses = forward_and_losses(
            model,
            inputs,
            labels,
            modality,
            criterion,
        )

        batch_size = labels.size(0)
        update_metric_totals(totals, losses, batch_size)
        total_correct += compute_accuracy(logits, labels)
        total_samples += batch_size

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

    return average_metrics(totals, total_samples, total_correct)


def create_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    samples = discover_cremad_samples(args.data_root)
    split = split_samples_from_csv(samples, args.split_csv_root)
    image_transform = ResizeToTensorNormalize(size=args.image_size)

    datasets = {
        name: CREMADAVDataset(
            part,
            modality=args.modality,
            fps=args.fps,
            audio_duration=args.audio_duration,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            win_length=args.win_length,
            image_transform=image_transform,
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
    if "fusion_minus_audio" in metrics:
        parts.append(f"{prefix}_fusion_minus_audio={metrics['fusion_minus_audio']:.4f}")
    if "fusion_minus_visual" in metrics:
        parts.append(f"{prefix}_fusion_minus_visual={metrics['fusion_minus_visual']:.4f}")
    parts.append(f"{prefix}_acc={metrics['acc']:.4f}")
    return " ".join(parts)


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

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="test")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_acc, label="train")
    axes[1].plot(epochs, val_acc, label="test")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_gap_excel(history: list[dict[str, Any]], path: Path, split_name: str = "train") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "loss_gaps"
    sheet.append(["fusion_minus_audio", "fusion_minus_visual"])
    for item in history:
        metrics = item[split_name]
        sheet.append([metrics["fusion_minus_audio"], metrics["fusion_minus_visual"]])
    workbook.save(path)


def plot_gap_history(history: list[dict[str, Any]], path: Path, split_name: str = "train") -> None:
    if not history:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [item["epoch"] for item in history]
    audio_gaps = [item[split_name]["fusion_minus_audio"] for item in history]
    visual_gaps = [item[split_name]["fusion_minus_visual"] for item in history]

    fig, axis = plt.subplots(1, 1, figsize=(7, 4))
    axis.plot(epochs, audio_gaps, marker="o", label="fusion_loss - audio_probe_loss")
    axis.plot(epochs, visual_gaps, marker="o", label="fusion_loss - visual_probe_loss")
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_title("Fusion-Probe Loss Gaps")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss gap")
    axis.legend()
    axis.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def load_shared_av_state(target: AVBaseline, source: AVBaseline) -> None:
    target.audio_net.load_state_dict(source.audio_net.state_dict())
    target.visual_net.load_state_dict(source.visual_net.state_dict())
    target.audio_probe.load_state_dict(source.audio_probe.state_dict())
    target.visual_probe.load_state_dict(source.visual_probe.state_dict())


def run_training(args: argparse.Namespace) -> dict[str, float]:
    set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    train_loader, val_loader, test_loader, sizes = create_dataloaders(args)
    if sizes["test"] == 0:
        raise ValueError(f"Empty test split: {sizes}")

    model = build_model(
        args.modality,
        num_classes=args.num_classes,
        fusion_head=getattr(args, "fusion_head", "linear"),
    ).to(device)
    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(args_to_dict(args), indent=2), encoding="utf-8")
    history_path = output_dir / "history.jsonl"
    history_json_path = output_dir / "history.json"
    curve_path = output_dir / "curves.png"
    gap_excel_path = output_dir / "loss_gaps.xlsx"
    gap_curve_path = output_dir / "loss_gaps.png"
    history: list[dict[str, Any]] = []

    best_val_acc = -1.0
    best_epoch = 0
    best_metrics: dict[str, float] = {}

    print(f"Split sizes: {sizes}")
    for epoch in range(1, args.epochs + 1):
        freeze_after_epochs = getattr(args, "freeze_after_epochs", -1)
        freeze_encoder_name = getattr(args, "freeze_encoder", "")
        if freeze_encoder_name and epoch == freeze_after_epochs + 1:
            freeze_encoder(model, freeze_encoder_name)
            print(f"froze_{freeze_encoder_name}_encoder_after_epoch={freeze_after_epochs}")

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.modality,
            epoch=epoch,
            show_progress=not args.no_progress,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            args.modality,
            epoch=epoch,
            split_name="val",
            show_progress=not args.no_progress,
        )
        scheduler.step()

        print(
            f"epoch={epoch:03d} "
            f"{format_metrics('train', train_metrics)} "
            f"{format_metrics('val', val_metrics)}"
        )

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
        if args.modality == "av":
            write_gap_excel(history, gap_excel_path)
            plot_gap_history(history, gap_curve_path)

        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            {"train": train_metrics, "val": val_metrics},
            args,
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            best_metrics = {"train": train_metrics, "val": val_metrics}
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, best_metrics, args)

    best_state = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(best_state["model"])
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        args.modality,
        split_name="test",
        show_progress=not args.no_progress,
    )
    result = {
        "best_epoch": float(best_epoch),
        "best_val_acc": float(best_val_acc),
        "test_loss": float(test_metrics["loss"]),
        "test_acc": float(test_metrics["acc"]),
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"best_epoch={best_epoch:03d} "
        f"best_val_acc={best_val_acc:.4f} "
        f"{format_metrics('test', test_metrics)}"
    )
    return result


def clone_args(args: argparse.Namespace, **updates: Any) -> argparse.Namespace:
    values = args_to_dict(args).copy()
    values.update(updates)
    return argparse.Namespace(**values)


def train_av_model_for_epochs(
    model: AVBaseline,
    train_loader: DataLoader,
    val_loader: DataLoader,
    sizes: dict[str, int],
    args: argparse.Namespace,
    output_dir: Path,
    epochs: int,
) -> list[dict[str, Any]]:
    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history: list[dict[str, Any]] = []

    history_path = output_dir / "history.jsonl"
    history_json_path = output_dir / "history.json"
    curve_path = output_dir / "curves.png"
    gap_excel_path = output_dir / "loss_gaps.xlsx"
    gap_curve_path = output_dir / "loss_gaps.png"

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            torch.device(args.device),
            "av",
            epoch=epoch,
            show_progress=not args.no_progress,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            torch.device(args.device),
            "av",
            epoch=epoch,
            split_name="val",
            show_progress=not args.no_progress,
        )
        scheduler.step()

        print(
            f"epoch={epoch:03d} "
            f"{format_metrics('train', train_metrics)} "
            f"{format_metrics('val', val_metrics)}"
        )
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
        write_gap_excel(history, gap_excel_path)
        plot_gap_history(history, gap_curve_path)
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            {"train": train_metrics, "val": val_metrics},
            args,
        )

    return history


def run_head_sweep(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    head_names = list(getattr(args, "head_sweep_heads", ["logit_fusion", "linear", "transencoder"]))
    train_loader, val_loader, test_loader, sizes = create_dataloaders(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(args_to_dict(args), indent=2), encoding="utf-8")

    pretrain_epochs = int(getattr(args, "head_pretrain_epochs", 10))
    tune_epochs = int(getattr(args, "head_tune_epochs", args.epochs))
    print(f"Split sizes: {sizes}")
    print(f"pretraining encoders for {pretrain_epochs} epochs")

    pretrain_model = AVBaseline(num_classes=args.num_classes, fusion_head="linear").to(device)
    pretrain_args = clone_args(args, epochs=pretrain_epochs, fusion_head="linear")
    train_av_model_for_epochs(
        pretrain_model,
        train_loader,
        val_loader,
        sizes,
        pretrain_args,
        output_dir / "pretrain_linear_10_epochs",
        pretrain_epochs,
    )

    results: dict[str, dict[str, float]] = {}
    for fusion_head in head_names:
        print(f"running fusion_head={fusion_head} with frozen encoders")
        model = AVBaseline(num_classes=args.num_classes, fusion_head=fusion_head).to(device)
        load_shared_av_state(model, pretrain_model)
        freeze_all_encoders(model)

        head_args = clone_args(args, epochs=tune_epochs, fusion_head=fusion_head)
        head_output_dir = output_dir / fusion_head
        history = train_av_model_for_epochs(
            model,
            train_loader,
            val_loader,
            sizes,
            head_args,
            head_output_dir,
            tune_epochs,
        )
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            "av",
            split_name="test",
            show_progress=not args.no_progress,
        )
        result = {
            "final_val_acc": float(history[-1]["val"]["acc"]) if history else 0.0,
            "test_loss": float(test_metrics["loss"]),
            "test_acc": float(test_metrics["acc"]),
        }
        (head_output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        results[fusion_head] = result
        print(f"fusion_head={fusion_head} {format_metrics('test', test_metrics)}")

    (output_dir / "head_sweep_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def run_experiment_suite(args: argparse.Namespace) -> None:
    root_output = Path(args.output_dir)
    suite = [
        ("experiment1_normal", clone_args(args, modality="av", fusion_head="linear", freeze_encoder="", freeze_after_epochs=-1)),
        (
            "experiment2_freeze_audio_after_5",
            clone_args(args, modality="av", fusion_head="linear", freeze_encoder="audio", freeze_after_epochs=5),
        ),
        (
            "experiment2_freeze_visual_after_5",
            clone_args(args, modality="av", fusion_head="linear", freeze_encoder="visual", freeze_after_epochs=5),
        ),
    ]

    for name, experiment_args in suite:
        experiment_args.output_dir = str(root_output / name)
        print(f"running {name}")
        run_training(experiment_args)

    head_sweep_args = clone_args(args, modality="av")
    head_sweep_args.output_dir = str(root_output / "experiment3_head_sweep")
    print("running experiment3_head_sweep")
    run_head_sweep(head_sweep_args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CREMA-D audio/visual baseline.")
    parser.add_argument("--data-root", type=str, default="dataset/CREMA-D")
    parser.add_argument("--output-dir", type=str, default="runs/cremad_baseline")
    parser.add_argument(
        "--experiment",
        choices=["normal", "freeze_audio", "freeze_visual", "head_sweep", "all"],
        default="normal",
    )
    parser.add_argument("--modality", choices=["av", "audio", "visual"], default="av")
    parser.add_argument("--fusion-head", choices=["linear", "logit_fusion", "transencoder"], default="linear")
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--freeze-after-epochs", type=int, default=-1)
    parser.add_argument("--freeze-encoder", choices=["", "audio", "visual"], default="")
    parser.add_argument("--head-pretrain-epochs", type=int, default=10)
    parser.add_argument("--head-tune-epochs", type=int, default=50)
    parser.add_argument(
        "--head-sweep-heads",
        nargs="+",
        choices=["logit_fusion", "linear", "transencoder"],
        default=["logit_fusion", "linear", "transencoder"],
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--audio-duration", type=float, default=3.0)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--win-length", type=int, default=400)
    parser.add_argument("--split-csv-root", type=str, default="ICCV2025-GDL-main/dataset/data/CREMAD")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.experiment == "all":
        run_experiment_suite(args)
    elif args.experiment == "freeze_audio":
        args.modality = "av"
        args.freeze_encoder = "audio"
        args.freeze_after_epochs = 5
        run_training(args)
    elif args.experiment == "freeze_visual":
        args.modality = "av"
        args.freeze_encoder = "visual"
        args.freeze_after_epochs = 5
        run_training(args)
    elif args.experiment == "head_sweep":
        args.modality = "av"
        run_head_sweep(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
