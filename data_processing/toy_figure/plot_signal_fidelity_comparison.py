from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt


LN2 = 0.6931471805599453
HERE = Path(__file__).resolve().parent
DEFAULT_FGM = HERE / "signal_fidelity_fgm.json"
DEFAULT_NO_FGM = HERE / "signal_fidelity_no_fgm_perturbed.json"
DEFAULT_OUTPUT = HERE / "signal_fidelity_comparison"
FGM_COLOR = "#2F6DB3"
NO_FGM_COLOR = "#C44E62"
GRID_COLOR = "#D9DCE3"
CONNECTOR_COLOR = GRID_COLOR
REFERENCE_LINEWIDTH = 1.2
PANEL_LABEL = "a"
FGM_MARKER = NO_FGM_MARKER = "o"
MARKER_FACE = "white"
MARKER_SIZE = 24
MARKER_ALPHA = 0.82

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.5,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def load_records(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"{path} must contain a records list")
    return payload["records"]


def condition_key(record: dict) -> tuple[float, float, float, int]:
    return (
        float(record["true_cmi"]),
        float(record["s"]),
        float(record["eta"]),
        int(record["seed"]),
    )


def _index_unique(records: list[dict]) -> dict[tuple[float, float, float, int], dict]:
    indexed = {}
    for record in records:
        key = condition_key(record)
        if key in indexed:
            raise ValueError(f"duplicate condition key: {key}")
        indexed[key] = record
    return indexed


def prepare_pairs(
    fgm_records: list[dict], no_fgm_records: list[dict]
) -> tuple[list[tuple[float, float, float]], float]:
    fgm = _index_unique(fgm_records)
    no_fgm = _index_unique(no_fgm_records)
    if fgm.keys() != no_fgm.keys():
        raise ValueError("FGM and No-FGM condition keys differ")
    if not fgm:
        raise ValueError("no paired records to plot")

    pairs = [
        (
            key[0],
            float(fgm[key]["delta_audio_tail"]) / LN2,
            float(no_fgm[key]["delta_audio_tail"]) / LN2,
        )
        for key in sorted(fgm)
    ]
    upper = max(value for pair in pairs for value in pair)
    return pairs, upper


def plot_comparison(
    fgm_records: list[dict],
    no_fgm_records: list[dict],
    output_base: str | Path,
    title: str = "Signal fidelity",
) -> tuple[Path, Path, Path]:
    pairs, upper = prepare_pairs(fgm_records, no_fgm_records)
    x = [pair[0] for pair in pairs]
    y_fgm = [pair[1] for pair in pairs]
    y_no_fgm = [pair[2] for pair in pairs]

    fig, ax = plt.subplots(figsize=(3.55, 3.45), constrained_layout=True)
    ideal, = ax.plot(
        [0.0, upper],
        [0.0, upper],
        color="#4A4A4A",
        linewidth=REFERENCE_LINEWIDTH,
        linestyle=(0, (4, 3)),
        zorder=1,
    )
    for x_value, fgm_value, no_fgm_value in pairs:
        ax.plot(
            [x_value, x_value],
            [fgm_value, no_fgm_value],
            color=CONNECTOR_COLOR,
            linewidth=0.7,
            alpha=0.75,
            solid_capstyle="round",
            zorder=2,
        )

    fgm = ax.scatter(
        x,
        y_fgm,
        s=MARKER_SIZE,
        marker=FGM_MARKER,
        facecolor=MARKER_FACE,
        edgecolor=FGM_COLOR,
        linewidth=1.25,
        alpha=MARKER_ALPHA,
        zorder=4,
    )
    no_fgm = ax.scatter(
        x,
        y_no_fgm,
        s=MARKER_SIZE,
        marker=NO_FGM_MARKER,
        facecolor=MARKER_FACE,
        edgecolor=NO_FGM_COLOR,
        linewidth=1.25,
        alpha=MARKER_ALPHA,
        zorder=3,
    )

    limit = upper * 1.035
    ax.set_xlim(0.0, limit)
    ax.set_ylim(0.0, limit)
    ax.set_aspect("equal", adjustable="box")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID_COLOR, linewidth=0.55, alpha=0.65)
    ax.set_xlabel("True CMI I(A;Y|B) [bit]", fontsize=8)
    ax.set_ylabel("Empirical E[Δ_A] [bit]", fontsize=8)
    ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold", pad=6)
    ax.text(
        -0.14,
        1.04,
        PANEL_LABEL,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    ax.tick_params(direction="out", length=3, width=0.8, labelsize=7)
    ax.legend(
        [fgm, no_fgm, ideal],
        ["FGM", "No-FGM + perturbation", "Ideal (y = x)"],
        loc="upper left",
        borderaxespad=0.5,
        handletextpad=0.5,
        fontsize=7,
    )

    output_base = Path(output_base).with_suffix("")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    outputs = tuple(output_base.with_suffix(suffix) for suffix in (".svg", ".pdf", ".png"))
    fig.savefig(outputs[0], bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    fig.savefig(outputs[2], dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw paired FGM signal-fidelity records.")
    parser.add_argument("--fgm-json", type=Path, default=DEFAULT_FGM)
    parser.add_argument("--no-fgm-json", type=Path, default=DEFAULT_NO_FGM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--title", default="Signal fidelity")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    outputs = plot_comparison(
        load_records(args.fgm_json),
        load_records(args.no_fgm_json),
        args.output,
        args.title,
    )
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
