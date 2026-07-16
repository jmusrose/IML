from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "synthetic_behavior_single"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def make_synthetic_records() -> list[dict[str, float]]:
    s_values = [index / 10 for index in range(11)]
    normalized_strength = [0.015, 0.09, 0.21, 0.29, 0.405, 0.515, 0.59, 0.71, 0.79, 0.91, 0.985]
    probe_acc_a = [0.965, 0.94, 0.91, 0.88, 0.84, 0.80, 0.76, 0.72, 0.69, 0.66, 0.63]
    probe_acc_b = [0.925, 0.89, 0.85, 0.81, 0.77, 0.72, 0.68, 0.63, 0.59, 0.54, 0.50]
    r_a = [0.506, 0.508, 0.507, 0.510, 0.509, 0.505, 0.508, 0.507, 0.510, 0.506, 0.509]

    return [
        {
            "s": s,
            "true_cmi": s,
            "probe_acc_a": acc_a,
            "probe_acc_b": acc_b,
            "fgm_r_a": direction_a,
            "fgm_r_b": 1.0 - direction_a,
            "fgm_strength_bits": 2.0 * strength,
        }
        for s, strength, acc_a, acc_b, direction_a in zip(
            s_values, normalized_strength, probe_acc_a, probe_acc_b, r_a
        )
    ]


def derive_series(records: list[dict[str, float]]) -> dict[str, list[float]]:
    rows = sorted(records, key=lambda row: row["s"])
    return {
        "s": [row["s"] for row in rows],
        "true_cmi": [row["true_cmi"] for row in rows],
        "normalized_strength": [row["fgm_strength_bits"] / 2.0 for row in rows],
        "error_a": [1.0 - row["probe_acc_a"] for row in rows],
        "error_b": [1.0 - row["probe_acc_b"] for row in rows],
        "direction_imbalance": [abs(row["fgm_r_a"] - row["fgm_r_b"]) for row in rows],
    }


def write_outputs(output_base: str | Path) -> tuple[Path, Path, Path, Path]:
    records = make_synthetic_records()
    series = derive_series(records)
    output_base = Path(output_base).with_suffix("")
    output_base.parent.mkdir(parents=True, exist_ok=True)

    json_path = output_base.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "kind": "synthetic_illustration",
                "note": "Illustrative values only; not measured training results.",
                "normalization": "normalized_strength = fgm_strength_bits / 2",
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    s = series["s"]
    error_a = series["error_a"]
    error_b = series["error_b"]
    mean_error = [(a + b) / 2.0 for a, b in zip(error_a, error_b)]

    fig, ax = plt.subplots(figsize=(5.6, 3.8), constrained_layout=True)
    ax.plot(s, series["true_cmi"], color="#2F80ED", marker="o", linewidth=2.0, label="True CMI")
    ax.plot(
        s,
        series["normalized_strength"],
        color="#9B51E0",
        marker="s",
        linestyle="--",
        linewidth=1.7,
        label="FGM strength / 2",
    )
    ax.fill_between(s, error_a, error_b, color="#F2994A", alpha=0.18, linewidth=0)
    ax.plot(
        s,
        mean_error,
        color="#F2994A",
        marker="^",
        linewidth=1.7,
        label="Single-modality error (A-B range)",
    )
    ax.plot(
        s,
        series["direction_imbalance"],
        color="#219653",
        marker="D",
        linestyle=":",
        linewidth=1.7,
        label=r"Direction imbalance $|r_A-r_B|$",
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.02, 1.04)
    ax.set_xlabel("Synergy ratio $s$")
    ax.set_ylabel("Normalized response")
    ax.set_title("FGM response tracks conditional contribution", loc="left", fontweight="bold")
    ax.text(
        1.0,
        1.02,
        "Synthetic illustration",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="#6B7280",
        fontsize=7.5,
        style="italic",
    )
    ax.grid(True, color="#DDE2E8", linewidth=0.6, alpha=0.7)
    ax.legend(loc="upper left", ncol=2, fontsize=7.3, handlelength=2.4)

    png_path = output_base.with_suffix(".png")
    svg_path = output_base.with_suffix(".svg")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return json_path, png_path, svg_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw one synthetic FGM behavior figure.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in write_outputs(args.output):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
