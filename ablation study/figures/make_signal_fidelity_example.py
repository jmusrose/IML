from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.75,
        "legend.frameon": False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    }
)


PALETTE = {
    "reference": "#1F2937",
    "delta": "#0072B2",
    "loss_ratio": "#D55E00",
    "accuracy_gap": "#009E73",
    "grad_norm": "#CC79A7",
    "grid": "#D1D5DB",
}


def minmax(values: np.ndarray) -> np.ndarray:
    low = np.percentile(values, 2)
    high = np.percentile(values, 98)
    return np.clip((values - low) / (high - low), 0, 1)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for idx, count in enumerate(counts):
            if count > 1:
                ranks[inverse == idx] = ranks[inverse == idx].mean()
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata(x)
    ry = rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt(np.sum(rx**2) * np.sum(ry**2))
    return float(np.sum(rx * ry) / denom)


def make_synthetic_data(seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    epochs = np.arange(1, 81)

    reference = (
        0.58 * np.exp(-epochs / 42)
        + 0.22 * np.exp(-((epochs - 22) / 15) ** 2)
        + 0.08 * np.sin(epochs / 8)
    )
    reference = minmax(reference)

    delta = minmax(reference + rng.normal(0, 0.035, size=epochs.size))

    lagged_reference = np.roll(reference, 8)
    lagged_reference[:8] = reference[0]
    loss_ratio = minmax(0.76 * lagged_reference + 0.16 * np.sin(epochs / 3.4) + rng.normal(0, 0.09, epochs.size))

    accuracy_gap = minmax(
        0.35 * reference
        + 0.42 * np.exp(-epochs / 14)
        - 0.23 * np.exp(-((epochs - 48) / 9) ** 2)
        + rng.normal(0, 0.11, epochs.size)
    )

    grad_norm = minmax(
        0.55
        - 0.32 * reference
        + 0.25 * np.sin(epochs / 5.5)
        + rng.normal(0, 0.12, epochs.size)
    )

    return {
        "epochs": epochs,
        "reference": reference,
        "delta": delta,
        "loss_ratio": loss_ratio,
        "accuracy_gap": accuracy_gap,
        "grad_norm": grad_norm,
    }


def draw_figure(output_stem: Path) -> None:
    data = make_synthetic_data()
    methods = [
        ("Delta (ours)", "delta", PALETTE["delta"]),
        ("Loss ratio", "loss_ratio", PALETTE["loss_ratio"]),
        ("Accuracy gap", "accuracy_gap", PALETTE["accuracy_gap"]),
        ("Grad-norm ratio", "grad_norm", PALETTE["grad_norm"]),
    ]

    final_acc = {
        "Delta (ours)": 72.4,
        "Loss ratio": 70.9,
        "Accuracy gap": 70.5,
        "Grad-norm ratio": 69.8,
        "Vanilla AV": 69.1,
    }

    fig = plt.figure(figsize=(7.1, 2.65), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.1, 0.72])
    ax_trace = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[0, 2])

    epochs = data["epochs"]
    reference = data["reference"]

    ax_trace.plot(epochs, reference, color=PALETTE["reference"], lw=1.9, label="Reference contribution")
    for label, key, color in methods:
        lw = 1.75 if key == "delta" else 1.1
        alpha = 0.95 if key == "delta" else 0.72
        ax_trace.plot(epochs, data[key], color=color, lw=lw, alpha=alpha, label=label)
    ax_trace.set_title("a  Signal trajectories")
    ax_trace.set_xlabel("Epoch")
    ax_trace.set_ylabel("Normalized signal")
    ax_trace.set_xlim(1, 80)
    ax_trace.set_ylim(-0.04, 1.04)
    ax_trace.grid(axis="y", color=PALETTE["grid"], lw=0.55, alpha=0.8)
    ax_trace.legend(loc="upper right", fontsize=5.8, handlelength=1.6)

    for label, key, color in methods:
        rho = spearman_rho(reference, data[key])
        ax_scatter.scatter(reference, data[key], s=12, color=color, alpha=0.62, edgecolor="none")
        x_line = np.linspace(reference.min(), reference.max(), 100)
        slope, intercept = np.polyfit(reference, data[key], 1)
        ax_scatter.plot(x_line, slope * x_line + intercept, color=color, lw=1.1, label=f"{label}, rho={rho:.2f}")
    ax_scatter.plot([0, 1], [0, 1], color="#9CA3AF", lw=0.8, ls="--", zorder=0)
    ax_scatter.set_title("b  Fidelity to reference")
    ax_scatter.set_xlabel("Reference contribution")
    ax_scatter.set_ylabel("Candidate signal")
    ax_scatter.set_xlim(-0.04, 1.04)
    ax_scatter.set_ylim(-0.04, 1.04)
    ax_scatter.grid(color=PALETTE["grid"], lw=0.45, alpha=0.55)
    ax_scatter.legend(loc="lower right", fontsize=5.5, handlelength=1.3)

    labels = list(final_acc.keys())
    values = np.array([final_acc[item] for item in labels])
    colors = [PALETTE["delta"], PALETTE["loss_ratio"], PALETTE["accuracy_gap"], PALETTE["grad_norm"], "#6B7280"]
    y = np.arange(len(labels))
    ax_bar.barh(y, values, color=colors, height=0.58)
    ax_bar.set_yticks(y, labels)
    ax_bar.invert_yaxis()
    ax_bar.set_title("c  Final accuracy")
    ax_bar.set_xlabel("Accuracy (%)")
    ax_bar.set_xlim(68.4, 73.0)
    ax_bar.grid(axis="x", color=PALETTE["grid"], lw=0.5, alpha=0.65)
    for yi, value in zip(y, values):
        ax_bar.text(value + 0.06, yi, f"{value:.1f}", va="center", ha="left", fontsize=6.2)

    fig.suptitle(
        "Example ablation logic: higher signal fidelity tracks better downstream performance",
        x=0.01,
        y=1.05,
        ha="left",
        fontsize=8.5,
        fontweight="bold",
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    draw_figure(here / "signal_fidelity_example")
