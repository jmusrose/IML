from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cmi_fgm import CMIFGMState, register_feature_gradient_hooks, register_split_linear_weight_hook
from FGM_toy.data import cmi_A_given_B, create_loaders
from FGM_toy.model import ToyAVModel


@dataclass
class TrainConfig:
    mode: str = "fgm"
    audio_loss_weight: float = 1.0
    visual_loss_weight: float = 1.0
    detach_probe_features: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _mean(items: list[float]) -> float:
    return float(sum(items) / max(1, len(items)))


def _signal_for_mode(
    mode: str,
    fusion_per_sample: torch.Tensor,
    audio_per_sample: torch.Tensor,
    visual_per_sample: torch.Tensor,
    audio_logits: torch.Tensor,
    visual_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    if mode == "acc_baseline":
        return torch.stack(
            [
                (audio_logits.argmax(dim=1) == labels).float(),
                (visual_logits.argmax(dim=1) == labels).float(),
            ],
            dim=1,
        )
    if mode == "loss_gap_baseline":
        return torch.stack([audio_per_sample - fusion_per_sample, visual_per_sample - fusion_per_sample], dim=1)
    if mode == "strength_signal":
        return torch.stack([torch.exp(-audio_per_sample), torch.exp(-visual_per_sample)], dim=1)
    return torch.stack(
        [
            visual_per_sample - fusion_per_sample,
            audio_per_sample - fusion_per_sample,
        ],
        dim=1,
    )


def forward_and_losses(
    model: ToyAVModel,
    audio: torch.Tensor,
    visual: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    config: TrainConfig,
    fgm_state: CMIFGMState | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[torch.utils.hooks.RemovableHandle]]:
    detach_probe_features = config.detach_probe_features
    if fgm_state is not None and fgm_state.num_updates < fgm_state.warmup_steps:
        detach_probe_features = False
    outputs = model.forward_with_modal_logits(audio, visual, detach_probe_features=detach_probe_features)
    fusion_per_sample = criterion(outputs["logits"], labels)
    audio_per_sample = criterion(outputs["audio_logits"], labels)
    visual_per_sample = criterion(outputs["visual_logits"], labels)
    fusion_loss = fusion_per_sample.mean()
    audio_loss = audio_per_sample.mean()
    visual_loss = visual_per_sample.mean()
    loss = fusion_loss + config.audio_loss_weight * audio_loss + config.visual_loss_weight * visual_loss
    signal = _signal_for_mode(
        config.mode,
        fusion_per_sample.detach(),
        audio_per_sample.detach(),
        visual_per_sample.detach(),
        outputs["audio_logits"].detach(),
        outputs["visual_logits"].detach(),
        labels,
    )
    positive_signal = signal.clamp_min(0)
    losses = {
        "loss": loss,
        "fusion_loss": fusion_loss,
        "audio_loss": audio_loss,
        "visual_loss": visual_loss,
        "delta_audio": positive_signal[:, 0].mean(),
        "delta_visual": positive_signal[:, 1].mean(),
        "audio_acc": (outputs["audio_logits"].argmax(dim=1) == labels).float().mean(),
        "visual_acc": (outputs["visual_logits"].argmax(dim=1) == labels).float().mean(),
    }

    handles: list[torch.utils.hooks.RemovableHandle] = []
    if fgm_state is not None and config.mode in {"fgm", "acc_baseline", "strength_signal"}:
        coefficients = fgm_state.coefficients(labels.size(0), labels.device, outputs["logits"].dtype)
        handles.extend(
            register_feature_gradient_hooks(
                {"audio": outputs["audio_feature"], "visual": outputs["visual_feature"]},
                coefficients,
            )
        )
        handles.append(
            register_split_linear_weight_hook(
                model.classifier[0],
                split_sizes=(model.feature_dim, model.feature_dim),
                modalities=("audio", "visual"),
                coefficients=coefficients,
            )
        )
        fgm_state.update(signal)
        mean_signal = fgm_state.mean_signal()
        losses.update(
            {
                "fgm_coef_audio": coefficients["audio"].mean(),
                "fgm_coef_visual": coefficients["visual"].mean(),
                "r_audio": torch.softmax(signal.clamp_min(0) / fgm_state.temperature, dim=1)[:, 0].mean(),
                "r_visual": torch.softmax(signal.clamp_min(0) / fgm_state.temperature, dim=1)[:, 1].mean(),
                "s_hat": signal.clamp_min(0).sum(dim=1).mean(),
                "fgm_signal_audio": mean_signal["audio"].to(labels.device),
                "fgm_signal_visual": mean_signal["visual"].to(labels.device),
            }
        )
    return outputs["logits"], losses, handles


def train_one_epoch(
    model: ToyAVModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: TrainConfig,
    fgm_state: CMIFGMState | None = None,
) -> dict[str, float]:
    model.train()
    criterion = nn.CrossEntropyLoss(reduction="none")
    totals: dict[str, float] = {}
    correct = 0
    count = 0
    for batch in loader:
        audio = batch["audio"].to(device)
        visual = batch["visual"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, losses, handles = forward_and_losses(model, audio, visual, labels, criterion, config, fgm_state)
        losses["loss"].backward()
        for handle in handles:
            handle.remove()
        optimizer.step()
        batch_size = int(labels.size(0))
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().item()) * batch_size
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        count += batch_size
    metrics = {name: value / max(1, count) for name, value in totals.items()}
    metrics["joint_acc"] = correct / max(1, count)
    return metrics


@torch.no_grad()
def evaluate(model: ToyAVModel, loader: DataLoader, device: torch.device, config: TrainConfig) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    totals: dict[str, float] = {}
    correct = 0
    count = 0
    for batch in loader:
        audio = batch["audio"].to(device)
        visual = batch["visual"].to(device)
        labels = batch["label"].to(device)
        logits, losses, _ = forward_and_losses(model, audio, visual, labels, criterion, config)
        batch_size = int(labels.size(0))
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().item()) * batch_size
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        count += batch_size
    metrics = {name: value / max(1, count) for name, value in totals.items()}
    metrics["joint_acc"] = correct / max(1, count)
    return metrics


@torch.no_grad()
def _collect_features(
    model: ToyAVModel,
    loader: DataLoader,
    device: torch.device,
    modality: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    encoder = model.audio_encoder if modality == "audio" else model.visual_encoder
    key = "audio" if modality == "audio" else "visual"
    for batch in loader:
        x = batch[key].to(device)
        features.append(encoder(x).detach())
        labels.append(batch["label"].to(device))
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def linear_probe_accuracy(
    model: ToyAVModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    modality: str,
    epochs: int,
    lr: float = 0.1,
) -> float:
    train_x, train_y = _collect_features(model, train_loader, device, modality)
    val_x, val_y = _collect_features(model, val_loader, device, modality)
    probe = nn.Linear(train_x.shape[1], 2).to(device)
    optimizer = torch.optim.SGD(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(max(1, epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(probe(train_x), train_y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return float((probe(val_x).argmax(dim=1) == val_y).float().mean().item())


def run_training(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    device = torch.device(args.device)
    train_loader, val_loader = create_loaders(
        n_train=args.n_train,
        n_val=args.n_val,
        s=args.s,
        eta=args.eta,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    model = ToyAVModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    config = TrainConfig(mode=args.mode)
    fgm_state = None
    if args.mode in {"fgm", "acc_baseline", "strength_signal"}:
        warmup_steps = max(0, int(args.fgm_warmup_epochs * len(train_loader)))
        fgm_state = CMIFGMState(
            ("audio", "visual"),
            strength=args.fgm_lambda,
            temperature=args.fgm_tau,
            momentum=args.fgm_momentum,
            warmup_steps=warmup_steps,
        )

    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, config, fgm_state)
        val_metrics = evaluate(model, val_loader, device, config)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

    half = history[len(history) // 2 :] or history
    fgm_tail_keys = ("fgm_coef_audio", "fgm_coef_visual", "r_audio", "r_visual", "s_hat")
    result = {
        "args": vars(args),
        "mode": args.mode,
        "s": args.s,
        "eta": args.eta,
        "seed": args.seed,
        "true_cmi": cmi_A_given_B(args.s, args.eta),
        "delta_audio_tail": _mean([item["train"]["delta_audio"] for item in half]),
        "delta_visual_tail": _mean([item["train"]["delta_visual"] for item in half]),
        "final_val": history[-1]["val"] if history else {},
        "history": history,
    }
    if getattr(args, "linear_probe_epochs", 0) > 0:
        result["probe_acc_audio"] = linear_probe_accuracy(
            model,
            train_loader,
            val_loader,
            device,
            "audio",
            args.linear_probe_epochs,
        )
        result["probe_acc_visual"] = linear_probe_accuracy(
            model,
            train_loader,
            val_loader,
            device,
            "visual",
            args.linear_probe_epochs,
        )
    for key in fgm_tail_keys:
        values = [item["train"][key] for item in half if key in item["train"]]
        if values:
            result[f"{key}_tail"] = _mean(values)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CMI-FGM toy experiment.")
    parser.add_argument(
        "--mode",
        choices=["fgm", "no_fgm", "acc_baseline", "loss_gap_baseline", "strength_signal"],
        default="fgm",
    )
    parser.add_argument("--s", type=float, default=0.5)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=20_000)
    parser.add_argument("--n-val", type=int, default=5_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fgm-lambda", type=float, default=0.5)
    parser.add_argument("--fgm-tau", type=float, default=1.0)
    parser.add_argument("--fgm-momentum", type=float, default=0.9)
    parser.add_argument("--fgm-warmup-epochs", type=int, default=2)
    parser.add_argument("--linear-probe-epochs", type=int, default=0)
    parser.add_argument("--output", default="")
    return parser.parse_args(argv)


def main() -> None:
    result = run_training(parse_args())
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
