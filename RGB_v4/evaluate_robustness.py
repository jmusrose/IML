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
from tqdm.auto import tqdm

from RGB_v4.datasets import RGBImageDataset, ResizeToTensorNormalize, discover_rgb_samples
from RGB_v4.training import average_metrics, batch_to_device, build_model, set_seed, update_metric_totals


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1)


def _normalization_tensors(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    return IMAGENET_MEAN.to(device=device, dtype=dtype), IMAGENET_STD.to(device=device, dtype=dtype)


def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    mean, std = _normalization_tensors(tensor.device, tensor.dtype)
    return (tensor * std + mean).clamp(0.0, 1.0)


def normalize_image(tensor: torch.Tensor) -> torch.Tensor:
    mean, std = _normalization_tensors(tensor.device, tensor.dtype)
    return (tensor - mean) / std


def add_gaussian_noise(
    tensor: torch.Tensor,
    epsilon: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    image = denormalize_image(tensor)
    std = float(epsilon) / 255.0
    noise = torch.randn(
        image.shape,
        generator=generator,
        device=image.device,
        dtype=image.dtype,
    ) * std
    return normalize_image((image + noise).clamp(0.0, 1.0))


def add_salt_pepper_noise(
    tensor: torch.Tensor,
    epsilon: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    image = denormalize_image(tensor)
    probability = max(0.0, min(float(epsilon) / 100.0, 1.0))
    random_values = torch.rand(
        image.shape,
        generator=generator,
        device=image.device,
        dtype=image.dtype,
    )
    salt = random_values < (probability / 2.0)
    pepper = (random_values >= (probability / 2.0)) & (random_values < probability)
    noised = image.clone()
    noised[salt] = 1.0
    noised[pepper] = 0.0
    return normalize_image(noised)


def apply_view_noise(
    batch: dict[str, torch.Tensor],
    view: str,
    noise_type: str,
    epsilon: float,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    if view not in {"rgb", "depth"}:
        raise ValueError(f"Unsupported noise view: {view!r}.")
    if noise_type not in {"gaussian", "salt-pepper"}:
        raise ValueError(f"Unsupported noise type: {noise_type!r}.")

    noised = dict(batch)
    if noise_type == "gaussian":
        noised[view] = add_gaussian_noise(batch[view], epsilon, generator=generator)
    else:
        noised[view] = add_salt_pepper_noise(batch[view], epsilon, generator=generator)
    return noised


@torch.no_grad()
def evaluate_with_optional_noise(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    noise_view: str | None = None,
    noise_type: str | None = None,
    epsilon: float = 0.0,
    seed: int = 0,
    show_progress: bool = False,
    split_name: str = "eval",
) -> dict[str, float]:
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    totals: dict[str, float] = {}
    total_correct = 0
    total_samples = 0
    all_predictions: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    iterator = loader
    if show_progress:
        iterator = tqdm(loader, desc=split_name, leave=False)

    for batch in iterator:
        inputs, labels = batch_to_device(batch, device)
        device_batch = {"rgb": inputs[0], "depth": inputs[1], "label": labels}
        if noise_view is not None and noise_type is not None and epsilon > 0:
            device_batch = apply_view_noise(
                device_batch,
                view=noise_view,
                noise_type=noise_type,
                epsilon=epsilon,
                generator=generator,
            )
        rgb = device_batch["rgb"]
        depth = device_batch["depth"]
        labels = device_batch["label"]

        outputs = model.forward_with_modal_logits(rgb, depth, detach_probe_features=True)
        logits = outputs["logits"]
        losses = {
            "loss": criterion(logits, labels).mean(),
            "fusion_loss": criterion(logits, labels).mean(),
            "rgb_loss": criterion(outputs["rgb_logits"], labels).mean(),
            "depth_loss": criterion(outputs["depth_logits"], labels).mean(),
            "rgb_acc": (outputs["rgb_logits"].argmax(dim=1) == labels).float().mean(),
            "depth_acc": (outputs["depth_logits"].argmax(dim=1) == labels).float().mean(),
        }

        batch_size = labels.size(0)
        update_metric_totals(totals, losses, batch_size)
        predictions = logits.argmax(dim=1)
        total_correct += int((predictions == labels).sum().item())
        total_samples += batch_size
        all_predictions.append(predictions.cpu())
        all_labels.append(labels.detach().cpu())

        if show_progress:
            iterator.set_postfix({"acc": total_correct / max(1, total_samples)})

    return average_metrics(totals, total_samples, total_correct, all_predictions, all_labels)


def evaluate_conditions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    noise_view: str,
    epsilons: tuple[float, ...] = (5.0, 10.0),
    seed: int = 0,
    show_progress: bool = False,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {
        "clean": evaluate_with_optional_noise(
            model,
            loader,
            device,
            seed=seed,
            show_progress=show_progress,
            split_name="clean",
        )
    }
    for noise_type in ("gaussian", "salt-pepper"):
        for epsilon in epsilons:
            label = f"{noise_type}@{int(epsilon) if float(epsilon).is_integer() else epsilon:g}"
            results[label] = evaluate_with_optional_noise(
                model,
                loader,
                device,
                noise_view=noise_view,
                noise_type=noise_type,
                epsilon=epsilon,
                seed=seed,
                show_progress=show_progress,
                split_name=label,
            )
    return results


def format_robustness_table(dataset_name: str, results: dict[str, dict[str, float]]) -> str:
    columns = ["Clean", "Gaussian@5", "Gaussian@10", "Salt-Pepper@5", "Salt-Pepper@10"]
    keys = ["clean", "gaussian@5", "gaussian@10", "salt-pepper@5", "salt-pepper@10"]
    values = []
    for key in keys:
        metrics = results.get(key)
        values.append("-" if metrics is None else f"{metrics['acc'] * 100:.2f}")
    return "\n".join(
        [
            "| Dataset | " + " | ".join(columns) + " |",
            "|---|" + "|".join(["---"] * len(columns)) + "|",
            f"| {dataset_name} | " + " | ".join(values) + " |",
        ]
    )


def write_results(path: Path, dataset_name: str, noise_view: str, results: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset_name,
        "noise_view": noise_view,
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    path.with_suffix(".md").write_text(format_robustness_table(dataset_name, results) + "\n", encoding="utf-8")


def create_eval_loader(args: argparse.Namespace) -> tuple[DataLoader, int, dict[str, int]]:
    eval_split = "test"
    samples, class_to_idx = discover_rgb_samples(args.data_root, split=eval_split)
    dataset = RGBImageDataset(
        samples,
        mode="test",
        image_transform=ResizeToTensorNormalize(size=args.image_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    return loader, len(class_to_idx), {"test": len(dataset)}


def load_checkpoint_model(checkpoint_path: str | Path, num_classes: int, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model = build_model(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(state_dict)
    return model


def run_robustness(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    loader, num_classes, sizes = create_eval_loader(args)
    if sizes["test"] == 0:
        raise ValueError(f"Empty test split: {sizes}")
    model = load_checkpoint_model(args.checkpoint, num_classes=num_classes, device=device)
    results = evaluate_conditions(
        model,
        loader,
        device,
        noise_view=args.noise_view,
        epsilons=tuple(args.epsilons),
        seed=args.seed,
        show_progress=not args.no_progress,
    )
    output_path = Path(args.output)
    write_results(output_path, args.dataset_name, args.noise_view, results)
    print(format_robustness_table(args.dataset_name, results))
    print(f"wrote {output_path} and {output_path.with_suffix('.md')}")
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RGB_v4 single-view noise robustness.")
    parser.add_argument("--dataset", choices=["nyud2", "sunrgbd"], required=True)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="runs/rgb_v4_robustness/metrics.json")
    parser.add_argument("--noise-view", choices=["rgb", "depth"], default="depth")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[5.0, 10.0])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=True)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)

    defaults = {
        "nyud2": ("dataset/nyud2_trainvaltest", "NYU Depth V2"),
        "sunrgbd": ("dataset/sunrgbd", "SUN RGB-D"),
    }
    default_root, dataset_name = defaults[args.dataset]
    if args.data_root is None:
        args.data_root = default_root
    args.dataset_name = dataset_name
    return args


def main() -> None:
    args = parse_args()
    run_robustness(args)


if __name__ == "__main__":
    main()
