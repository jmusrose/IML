from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from AV_v2.datasets import AVEDataset, ResizeToTensorNormalize, discover_ave_samples
from AV_v2.train_cremad import batch_to_device as _batch_to_device
from AV_v2.train_cremad import (
    append_epoch_log,
    build_fgm_state,
    build_model,
    evaluate,
    format_metrics,
    plot_history,
    save_checkpoint,
    seed_worker,
    set_seed,
    train_one_epoch,
    write_history_json,
)


def create_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    train_samples, class_to_idx = discover_ave_samples(args.data_root, split="train")
    val_samples, _ = discover_ave_samples(args.data_root, split="val")
    test_samples, _ = discover_ave_samples(args.data_root, split="test")
    args.num_classes = len(class_to_idx)

    image_transform = ResizeToTensorNormalize(size=args.image_size)
    dataset_kwargs: dict[str, Any] = {
        "modality": args.modality,
        "use_video_frames": args.use_video_frames,
        "image_transform": image_transform,
    }
    train_dataset = AVEDataset(train_samples, mode="train", **dataset_kwargs)
    val_dataset = AVEDataset(val_samples, mode="val", **dataset_kwargs)
    test_dataset = AVEDataset(test_samples, mode="test", **dataset_kwargs)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    sizes = {
        "train": len(train_dataset),
        "val": len(val_dataset),
        "test": len(test_dataset),
    }
    return train_loader, val_loader, test_loader, sizes


def run_training(args: argparse.Namespace) -> dict[str, float]:
    set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    train_loader, val_loader, test_loader, sizes = create_dataloaders(args)
    if sizes["test"] == 0:
        raise ValueError(f"Empty AVE test split: {sizes}")

    model = build_model(args.modality, num_classes=args.num_classes).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    fgm_state = build_fgm_state(args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    history_path = output_dir / "history.jsonl"
    history_json_path = output_dir / "history.json"
    curve_path = output_dir / "curves.png"
    history: list[dict[str, Any]] = []

    best_val_acc = -1.0
    best_epoch = 0
    best_metrics: dict[str, float] = {}

    print(f"AVE classes: {args.num_classes}")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AVE audio/visual baseline.")
    parser.add_argument("--data-root", type=str, default="dataset/AVE")
    parser.add_argument("--output-dir", type=str, default="runs/ave_baseline")
    parser.add_argument("--modality", choices=["av", "audio", "visual"], default="av")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--use-video-frames", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--fgm", action="store_true", help="Enable CMI-FGM gradient modulation for AV training.")
    parser.add_argument("--fgm-lambda", type=float, default=0.5)
    parser.add_argument("--fgm-tau", type=float, default=1.0)
    parser.add_argument("--fgm-momentum", type=float, default=0.9)
    parser.add_argument("--fgm-warmup-steps", type=int, default=0)
    args = parser.parse_args()
    args.num_classes = 28
    return args


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
