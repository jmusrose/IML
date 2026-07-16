from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt


LN2 = 0.6931471805599453
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "runs" / "fgm_toy" / "strict_md_behavior" / "behavior_lines_md.json"
DEFAULT_OUTPUT = DEFAULT_INPUT.with_name("behavior_single_reference_points")
PURPLE = "#8F6BB3"
ROSE = "#E07A5F"
GREEN = "#3D9970"
GRID_COLOR = "#DDE2E8"
REFERENCE_COLOR = "#A7ADB7"
FGM_LINESTYLE = "none"
LINE_WIDTH = 1.7
BAND_ALPHA = 0.15

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def load_summary(path: str | Path) -> dict[str, list[dict]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("fgm"), list):
        raise ValueError(f"{path} must contain an fgm list")
    return payload


def derive_series(summary: dict[str, list[dict]]) -> dict[str, list[float]]:
    rows = sorted(summary.get("fgm", []), key=lambda row: row["s"])
    if not rows:
        raise ValueError("summary contains no FGM rows")

    error_a = [1.0 - row["probe_acc_audio_mean"] for row in rows]
    error_b = [1.0 - row["probe_acc_visual_mean"] for row in rows]
    error_a_std = [row["probe_acc_audio_std"] for row in rows]
    error_b_std = [row["probe_acc_visual_std"] for row in rows]
    return {
        "s": [row["s"] for row in rows],
        "true_cmi": [row["true_cmi_mean"] for row in rows],
        "true_cmi_std": [row["true_cmi_std"] for row in rows],
        "normalized_strength": [row["s_hat_tail_mean"] / (2.0 * LN2) for row in rows],
        "normalized_strength_std": [row["s_hat_tail_std"] / (2.0 * LN2) for row in rows],
        "mean_error": [(a + b) / 2.0 for a, b in zip(error_a, error_b)],
        "error_low": [
            max(0.0, min(a - a_std, b - b_std))
            for a, b, a_std, b_std in zip(error_a, error_b, error_a_std, error_b_std)
        ],
        "error_high": [
            min(1.0, max(a + a_std, b + b_std))
            for a, b, a_std, b_std in zip(error_a, error_b, error_a_std, error_b_std)
        ],
        "direction_imbalance": [abs(2.0 * row["r_audio_tail_mean"] - 1.0) for row in rows],
        "direction_imbalance_std": [2.0 * row["r_audio_tail_std"] for row in rows],
    }


def _mean_std_band(ax, x, mean, std, color: str) -> None:
    lower = [max(0.0, value - spread) for value, spread in zip(mean, std)]
    upper = [min(1.0, value + spread) for value, spread in zip(mean, std)]
    ax.fill_between(x, lower, upper, color=color, alpha=BAND_ALPHA, linewidth=0, zorder=1)


def plot_summary(
    summary: dict[str, list[dict]], output_base: str | Path
) -> tuple[Path, Path, Path]:
    series = derive_series(summary)
    s = series["s"]

    fig, ax = plt.subplots(figsize=(5.6, 3.8), constrained_layout=True)
    ax.plot(
        s,
        series["true_cmi"],
        color=REFERENCE_COLOR,
        linewidth=1.8,
        label="Analytic CMI target [bit]",
        zorder=2,
    )

    _mean_std_band(
        ax,
        s,
        series["normalized_strength"],
        series["normalized_strength_std"],
        PURPLE,
    )
    ax.plot(
        s,
        series["normalized_strength"],
        color=PURPLE,
        marker="s",
        markersize=6.5,
        linestyle=FGM_LINESTYLE,
        label="FGM strength / 2 (mean ± SD)",
        zorder=4,
    )

    ax.fill_between(
        s,
        series["error_low"],
        series["error_high"],
        color=ROSE,
        alpha=BAND_ALPHA,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        s,
        series["mean_error"],
        color=ROSE,
        marker="^",
        markersize=6.4,
        linewidth=LINE_WIDTH,
        label="Single-modality error (A/B ± SD envelope)",
        zorder=3,
    )

    _mean_std_band(
        ax,
        s,
        series["direction_imbalance"],
        series["direction_imbalance_std"],
        GREEN,
    )
    ax.plot(
        s,
        series["direction_imbalance"],
        color=GREEN,
        marker="D",
        markersize=5.8,
        linestyle=":",
        linewidth=LINE_WIDTH,
        label=r"Direction imbalance $|r_A-r_B|$",
        zorder=3,
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.02, 1.04)
    ax.set_xlabel("Synergy ratio $s$")
    ax.set_ylabel("Response value (0-1)")
    ax.set_title(
        "Measured FGM behavior across synergy levels",
        loc="left",
        fontweight="bold",
        fontsize=10,
    )
    ax.text(
        1.0,
        1.02,
        "Mean ± SD across seeds",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="#6B7280",
        fontsize=7.5,
        style="italic",
    )
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.7)
    ax.legend(loc="upper left", ncol=2, fontsize=7.3, handlelength=2.4)

    output_base = Path(output_base).with_suffix("")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs = tuple(output_base.with_suffix(suffix) for suffix in (".png", ".svg", ".pdf"))
    fig.savefig(outputs[0], dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[1], bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[2], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw one-panel FGM behavior from measured JSON results.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in plot_summary(load_summary(args.input), args.output):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
