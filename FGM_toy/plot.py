from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def delta_bits(delta_nats: float) -> float:
    return float(delta_nats) / 0.6931471805599453


def load_records(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if "records" in payload:
        return list(payload["records"])
    return [payload]


def plot_signal_fidelity(records: list[dict], output: str | Path, title: str) -> None:
    x = [record["true_cmi"] for record in records]
    y = [delta_bits(record["delta_audio_tail"]) for record in records]
    upper = max(x + y + [1e-6])
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.scatter(x, y, s=28)
    ax.plot([0, upper], [0, upper], color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("True CMI I(A;Y|B) [bit]")
    ax.set_ylabel("Empirical E[Delta_A] [bit]")
    ax.set_title(title)
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_behavior_lines(records: dict[str, list[dict]], output: str | Path) -> None:
    fgm = sorted(records.get("fgm", []), key=lambda item: item["s"])
    acc_baseline = sorted(records.get("acc_baseline", []), key=lambda item: item["s"])
    if not fgm:
        return

    s_values = [record["s"] for record in fgm]
    true_cmi = [record["true_cmi"] for record in fgm]
    acc_a = [record["final_val"]["audio_acc"] for record in fgm]
    acc_b = [record["final_val"]["visual_acc"] for record in fgm]
    r_a = [record.get("r_audio_tail", 0.5) for record in fgm]
    r_b = [record.get("r_visual_tail", 0.5) for record in fgm]
    s_hat = [delta_bits(record.get("s_hat_tail", 0.0)) for record in fgm]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(s_values, true_cmi, marker="o", label="True CMI")
    ax.plot(s_values, acc_a, marker="s", label="Probe acc A")
    ax.plot(s_values, acc_b, marker="s", label="Probe acc B")
    ax.plot(s_values, r_a, linewidth=1.5, label="FGM r_A")
    ax.plot(s_values, r_b, linewidth=1.5, label="FGM r_B")
    if acc_baseline:
        ax.plot(
            [record["s"] for record in acc_baseline],
            [record.get("r_audio_tail", 0.5) for record in acc_baseline],
            linestyle="--",
            color="tab:red",
            label="Acc baseline r_A",
        )
    ax.set_xlabel("Synergy ratio s")
    ax.set_ylabel("CMI / accuracy / direction")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(s_values, s_hat, color="tab:purple", marker="^", label="FGM strength")
    ax2.set_ylabel("FGM strength E[Delta_A+Delta_B] [bit]")
    ax2.set_ylim(bottom=0.0)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper center", ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _series(rows: list[dict], key: str) -> list[float]:
    return [float(row.get(key, 0.0)) for row in rows]


def _band(ax, x: list[float], mean: list[float], std: list[float], **kwargs) -> None:
    lower = [m - s for m, s in zip(mean, std)]
    upper = [m + s for m, s in zip(mean, std)]
    ax.fill_between(x, lower, upper, alpha=0.14, linewidth=0, color=kwargs.get("color"))
    ax.plot(x, mean, marker=kwargs.pop("marker", "o"), **kwargs)


def plot_strict_md_behavior(summary: dict[str, list[dict]], output: str | Path) -> None:
    fgm = sorted(summary.get("fgm", []), key=lambda row: row["s"])
    strength = sorted(summary.get("strength_signal", []), key=lambda row: row["s"])
    no_fgm = sorted(summary.get("no_fgm", []), key=lambda row: row["s"])
    if not fgm:
        return

    x = _series(fgm, "s")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), sharex=True)
    ax_cmi, ax_acc, ax_r, ax_strength = axes.ravel()

    _band(
        ax_cmi,
        x,
        _series(fgm, "true_cmi_mean"),
        _series(fgm, "true_cmi_std"),
        color="tab:blue",
        label="True CMI",
    )
    ax_cmi.set_ylabel("CMI [bit]")
    ax_cmi.set_title("Analytic conditional contribution")

    acc_a = _series(fgm, "probe_acc_audio_mean")
    acc_b = _series(fgm, "probe_acc_visual_mean")
    _band(
        ax_acc,
        x,
        acc_a,
        _series(fgm, "probe_acc_audio_std"),
        color="tab:orange",
        label="Frozen linear probe A",
        marker="s",
    )
    _band(
        ax_acc,
        x,
        acc_b,
        _series(fgm, "probe_acc_visual_std"),
        color="tab:green",
        label="Frozen linear probe B",
        marker="s",
    )
    acc_values = acc_a + acc_b
    acc_min = min(acc_values) if acc_values else 0.5
    acc_max = max(acc_values) if acc_values else 1.0
    pad = max(0.02, (acc_max - acc_min) * 0.2)
    ax_acc.set_ylim(max(0.45, acc_min - pad), min(1.01, acc_max + pad))
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Single-modality ability")

    _band(
        ax_r,
        x,
        _series(fgm, "r_audio_tail_mean"),
        _series(fgm, "r_audio_tail_std"),
        color="tab:red",
        label="FGM r_A",
    )
    _band(
        ax_r,
        x,
        _series(fgm, "r_visual_tail_mean"),
        _series(fgm, "r_visual_tail_std"),
        color="tab:purple",
        label="FGM r_B",
    )
    if strength:
        _band(
            ax_r,
            _series(strength, "s"),
            _series(strength, "r_audio_tail_mean"),
            _series(strength, "r_audio_tail_std"),
            color="tab:brown",
            label="Strength-signal r_A",
            marker="^",
        )
    ax_r.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax_r.set_ylim(0.0, 1.0)
    ax_r.set_ylabel("Direction factor")
    ax_r.set_title("Direction allocation")

    _band(
        ax_strength,
        x,
        [delta_bits(value) for value in _series(fgm, "s_hat_tail_mean")],
        [delta_bits(value) for value in _series(fgm, "s_hat_tail_std")],
        color="tab:purple",
        label="FGM strength",
    )
    if no_fgm:
        ax_strength.plot(
            _series(no_fgm, "s"),
            _series(no_fgm, "final_joint_acc_mean"),
            color="tab:gray",
            linestyle=":",
            label="No-FGM joint acc",
        )
    ax_strength.set_ylabel("Strength [bit] / acc")
    ax_strength.set_title("Modulation strength")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Synergy ratio s")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot FGM toy JSON results.")
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Signal fidelity")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    plot_signal_fidelity(load_records(args.input), args.output, args.title)


if __name__ == "__main__":
    main()
