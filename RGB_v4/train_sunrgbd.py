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

from RGB_v4.datasets import RGBImageDataset, RGBTrainImageTransform, ResizeToTensorNormalize, discover_rgb_samples
from RGB_v4.evaluate_robustness import evaluate_conditions, write_results
from RGB_v4.training import (
    append_epoch_log,
    build_model,
    build_scheduler,
    clone_model_state_dict,
    evaluate,
    format_metrics,
    format_epoch_report,
    plot_history,
    prepare_run_output_dir,
    save_checkpoint,
    seed_worker,
    set_seed,
    train_one_epoch,
    write_history_json,
)


def build_train_transform(args: argparse.Namespace) -> RGBTrainImageTransform:
    return RGBTrainImageTransform(
        size=args.image_size,
        scale=tuple(getattr(args, "aug_scale", [0.7, 1.0])),
        ratio=tuple(getattr(args, "aug_ratio", [3.0 / 4.0, 4.0 / 3.0])),
        horizontal_flip_prob=getattr(args, "aug_hflip_prob", 0.5),
        color_jitter=getattr(args, "aug_color_jitter", 0.2),
    )


def create_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    train_samples, class_to_idx = discover_rgb_samples(args.data_root, split="train")
    test_samples, _ = discover_rgb_samples(args.data_root, split="test")
    args.num_classes = len(class_to_idx)

    train_dataset = RGBImageDataset(
        train_samples,
        mode="train",
        image_transform=build_train_transform(args),
    )
    test_dataset = RGBImageDataset(
        test_samples,
        mode="test",
        image_transform=ResizeToTensorNormalize(size=args.image_size),
    )

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
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    sizes = {"train": len(train_dataset), "test": len(test_dataset)}
    return train_loader, test_loader, sizes


def run_training(args: argparse.Namespace) -> dict[str, float]:
    set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    train_loader, test_loader, sizes = create_dataloaders(args)
    if sizes["train"] == 0 or sizes["test"] == 0:
        raise ValueError(f"Empty SUN RGB-D split: {sizes}")

    model = build_model(num_classes=args.num_classes, pretrained=args.pretrained).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(optimizer, args)

    output_dir = prepare_run_output_dir(args)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    history_path = output_dir / "history.jsonl"
    history_json_path = output_dir / "history.json"
    curve_path = output_dir / "curves.png"
    history: list[dict[str, Any]] = []

    best_test_acc = -1.0
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None

    print(f"SUN RGB-D classes: {args.num_classes}")
    print(f"Split sizes: {sizes}")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch=epoch,
            show_progress=not args.no_progress,
            rgb_loss_weight=args.rgb_loss_weight,
            depth_loss_weight=args.depth_loss_weight,
        )
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            epoch=epoch,
            split_name="test",
            show_progress=not args.no_progress,
            rgb_loss_weight=args.rgb_loss_weight,
            depth_loss_weight=args.depth_loss_weight,
        )
        scheduler.step()

        print(format_epoch_report(epoch, train_metrics, test_metrics, eval_split_name="test"))
        epoch_record = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "test": test_metrics,
        }
        history.append(epoch_record)
        append_epoch_log(history_path, epoch_record, args, sizes)
        write_history_json(history_json_path, history, args, sizes)
        plot_history(history, curve_path)

        if test_metrics["acc"] > best_test_acc:
            best_test_acc = test_metrics["acc"]
            best_epoch = epoch
            best_state_dict = clone_model_state_dict(model)
            save_checkpoint(
                output_dir / "best_checkpoint.pt",
                model,
                optimizer,
                epoch,
                {"train": train_metrics, "test": test_metrics},
                args,
            )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        split_name="test",
        show_progress=not args.no_progress,
        rgb_loss_weight=args.rgb_loss_weight,
        depth_loss_weight=args.depth_loss_weight,
    )
    result = {
        "best_epoch": float(best_epoch),
        "best_test_acc": float(best_test_acc),
        **{f"test_{name}": float(value) for name, value in test_metrics.items()},
    }
    robustness_results = evaluate_conditions(
        model,
        test_loader,
        device,
        noise_view=args.robustness_noise_view,
        epsilons=tuple(args.robustness_epsilons),
        seed=args.seed,
        show_progress=not args.no_progress,
    )
    write_results(
        output_dir / "robustness_metrics.json",
        "SUN RGB-D",
        args.robustness_noise_view,
        robustness_results,
    )
    result["robustness"] = robustness_results
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"best_epoch={best_epoch:03d} best_test_acc={best_test_acc:.4f} {format_metrics('test', test_metrics)}")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RGB_v4 on SUN RGB-D RGB scene images.")
    parser.add_argument("--data-root", type=str, default="dataset/sunrgbd")
    parser.add_argument("--output-dir", type=str, default="runs/rgb_v4_sunrgbd")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr-scheduler", choices=["multistep", "cosine"], default="multistep")
    parser.add_argument("--lr-decay-step", type=str, default="[40,80]")
    parser.add_argument("--lr-decay-ratio", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rgb-loss-weight", type=float, default=1.0)
    parser.add_argument("--depth-loss-weight", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--aug-scale", type=float, nargs=2, default=[0.7, 1.0])
    parser.add_argument("--aug-ratio", type=float, nargs=2, default=[3.0 / 4.0, 4.0 / 3.0])
    parser.add_argument("--aug-hflip-prob", type=float, default=0.5)
    parser.add_argument("--aug-color-jitter", type=float, default=0.2)
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", help="Use ImageNet-pretrained torchvision ResNet18 backbones.")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false", help="Disable ImageNet-pretrained ResNet18 weights.")
    parser.set_defaults(pretrained=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=True)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--robustness-noise-view", choices=["rgb", "depth"], default="depth")
    parser.add_argument("--robustness-epsilons", type=float, nargs="+", default=[5.0, 10.0])
    args = parser.parse_args(argv)
    _, class_to_idx = discover_rgb_samples(args.data_root, split="train")
    args.num_classes = len(class_to_idx)
    return args


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
