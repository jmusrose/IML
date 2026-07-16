from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FGM_toy.data import Hb
from FGM_toy.plot import plot_behavior_lines, plot_signal_fidelity, plot_strict_md_behavior
from FGM_toy.train import parse_args as parse_train_args
from FGM_toy.train import run_training


@dataclass(frozen=True)
class GridPoint:
    s: float
    eta: float
    true_cmi: float


@dataclass(frozen=True)
class SignalFidelityConfig:
    output_dir: Path
    s_values: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)
    eta_values: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.35)
    points: tuple[GridPoint, ...] = ()
    behavior_s_values: tuple[float, ...] = ()
    seeds: tuple[int, ...] = (0,)
    n_train: int = 20_000
    n_val: int = 5_000
    epochs: int = 20
    batch_size: int = 256
    lr: float = 0.05
    device: str = "cpu"
    fgm_warmup_epochs: int = 2
    linear_probe_epochs: int = 0


def _train_args(config: SignalFidelityConfig, mode: str, s: float, eta: float, seed: int) -> argparse.Namespace:
    return parse_train_args(
        [
            "--mode",
            mode,
            "--s",
            str(s),
            "--eta",
            str(eta),
            "--seed",
            str(seed),
            "--n-train",
            str(config.n_train),
            "--n-val",
            str(config.n_val),
            "--epochs",
            str(config.epochs),
            "--batch-size",
            str(config.batch_size),
            "--lr",
            str(config.lr),
            "--device",
            config.device,
            "--fgm-warmup-epochs",
            str(config.fgm_warmup_epochs),
            "--linear-probe-epochs",
            str(config.linear_probe_epochs),
        ]
    )


def _strip_history(record: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in record.items() if key != "history"}


def uniform32_points() -> tuple[GridPoint, ...]:
    targets = [0.12 + i * (0.92 - 0.12) / 31 for i in range(32)]
    eta_choices = (0.10, 0.06, 0.03, 0.0)
    points: list[GridPoint] = []
    for index, target in enumerate(targets):
        valid = [eta for eta in eta_choices if target / (1.0 - Hb(eta)) <= 0.95]
        eta = valid[index % len(valid)] if valid else 0.0
        s = target / (1.0 - Hb(eta))
        points.append(GridPoint(s=s, eta=eta, true_cmi=target))
    return tuple(points)


def _iter_points(config: SignalFidelityConfig) -> tuple[GridPoint, ...]:
    if config.points:
        return config.points
    return tuple(
        GridPoint(s=s, eta=eta, true_cmi=s * (1.0 - Hb(eta)))
        for s in config.s_values
        for eta in config.eta_values
    )


def run_signal_fidelity_grid(config: SignalFidelityConfig) -> dict[str, list[dict[str, object]]]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, list[dict[str, object]]] = {"fgm": [], "no_fgm": []}
    behavior_records: dict[str, list[dict[str, object]]] = {"fgm": [], "acc_baseline": []}
    points = _iter_points(config)
    behavior_s_values = config.behavior_s_values or config.s_values
    total = len(points) * len(config.seeds) * 2 + len(behavior_s_values) * len(config.seeds) * 2
    index = 0
    for mode in ("fgm", "no_fgm"):
        for point in points:
            for seed in config.seeds:
                index += 1
                print(f"[{index}/{total}] mode={mode} s={point.s:g} eta={point.eta:g} seed={seed}", flush=True)
                record = run_training(_train_args(config, mode, point.s, point.eta, seed))
                records[mode].append(_strip_history(record))

    for mode in ("fgm", "acc_baseline"):
        for s in behavior_s_values:
            for seed in config.seeds:
                index += 1
                print(f"[{index}/{total}] behavior mode={mode} s={s:g} eta=0 seed={seed}", flush=True)
                record = run_training(_train_args(config, mode, s, 0.0, seed))
                behavior_records[mode].append(_strip_history(record))

    fgm_json = config.output_dir / "signal_fidelity_fgm.json"
    baseline_json = config.output_dir / "signal_fidelity_no_fgm.json"
    behavior_json = config.output_dir / "behavior_lines.json"
    fgm_png = config.output_dir / "signal_fidelity_fgm.png"
    baseline_png = config.output_dir / "signal_fidelity_no_fgm.png"
    behavior_png = config.output_dir / "behavior_lines.png"
    fgm_json.write_text(json.dumps({"records": records["fgm"]}, indent=2), encoding="utf-8")
    baseline_json.write_text(json.dumps({"records": records["no_fgm"]}, indent=2), encoding="utf-8")
    behavior_json.write_text(json.dumps(behavior_records, indent=2), encoding="utf-8")
    plot_signal_fidelity(records["fgm"], fgm_png, "FGM")
    plot_signal_fidelity(records["no_fgm"], baseline_png, "No-FGM")
    plot_behavior_lines(behavior_records, behavior_png)
    print(f"Wrote {fgm_json}")
    print(f"Wrote {baseline_json}")
    print(f"Wrote {behavior_json}")
    print(f"Wrote {fgm_png}")
    print(f"Wrote {baseline_png}")
    print(f"Wrote {behavior_png}")
    return records


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _summarize_by_s(records: list[dict[str, object]]) -> list[dict[str, float]]:
    keys = (
        "true_cmi",
        "probe_acc_audio",
        "probe_acc_visual",
        "r_audio_tail",
        "r_visual_tail",
        "s_hat_tail",
        "final_joint_acc",
    )
    rows: list[dict[str, float]] = []
    for s in sorted({float(record["s"]) for record in records}):
        group = [record for record in records if float(record["s"]) == s]
        row: dict[str, float] = {"s": s}
        expanded = []
        for record in group:
            item = dict(record)
            item["final_joint_acc"] = record.get("final_val", {}).get("joint_acc", 0.0)
            expanded.append(item)
        for key in keys:
            values = [float(record.get(key, 0.0)) for record in expanded if key in record]
            if values:
                row[f"{key}_mean"] = _mean(values)
                row[f"{key}_std"] = _std(values)
        rows.append(row)
    return rows


def run_strict_md_behavior(config: SignalFidelityConfig) -> dict[str, list[dict[str, float]]]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    behavior_s_values = config.behavior_s_values or tuple(i / 10 for i in range(11))
    records: dict[str, list[dict[str, object]]] = {"fgm": [], "strength_signal": [], "no_fgm": []}
    modes = ("fgm", "no_fgm", "strength_signal")
    total = len(behavior_s_values) * len(config.seeds) * len(modes)
    index = 0
    for mode in modes:
        for s in behavior_s_values:
            for seed in config.seeds:
                index += 1
                print(f"[{index}/{total}] strict-md mode={mode} s={s:g} eta=0 seed={seed}", flush=True)
                record = run_training(_train_args(config, mode, s, 0.0, seed))
                records[mode].append(_strip_history(record))
    summary = {mode: _summarize_by_s(items) for mode, items in records.items()}
    json_path = config.output_dir / "behavior_lines_md.json"
    png_path = config.output_dir / "behavior_lines_md.png"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_strict_md_behavior(summary, png_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {png_path}")
    return summary


def _csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FGM toy grids and write scatter plots.")
    parser.add_argument("--grid", choices=["signal_fidelity"], default="signal_fidelity")
    parser.add_argument("--preset", choices=["spec25", "uniform32"], default="spec25")
    parser.add_argument("--output-dir", default="runs/fgm_toy/signal_fidelity")
    parser.add_argument("--s-values", default="0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--eta-values", default="0.0,0.05,0.1,0.2,0.35")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--n-train", type=int, default=20_000)
    parser.add_argument("--n-val", type=int, default=5_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fgm-warmup-epochs", type=int, default=2)
    parser.add_argument("--linear-probe-epochs", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="Small smoke grid for checking the pipeline.")
    parser.add_argument("--strict-md-behavior", action="store_true", help="Generate strict Figure 1 from the MD spec.")
    return parser.parse_args(argv)


def build_signal_fidelity_config(argv: list[str] | None = None) -> SignalFidelityConfig:
    args = parse_args(argv)
    if args.quick:
        args.preset = "spec25"
        args.s_values = "0.2,0.6"
        args.eta_values = "0.0,0.1"
        args.n_train = 256
        args.n_val = 128
        args.epochs = 2
        args.batch_size = 64
        args.fgm_warmup_epochs = 0
        args.linear_probe_epochs = 1
    elif args.strict_md_behavior and args.seeds == "0":
        args.seeds = "0,1,2"
    points = uniform32_points() if args.preset == "uniform32" else ()
    behavior_s_values = tuple(i / 10 for i in range(11))
    if args.quick:
        behavior_s_values = _csv_floats(args.s_values)
    return SignalFidelityConfig(
        output_dir=Path(args.output_dir),
        s_values=_csv_floats(args.s_values),
        eta_values=_csv_floats(args.eta_values),
        points=points,
        behavior_s_values=behavior_s_values,
        seeds=_csv_ints(args.seeds),
        n_train=args.n_train,
        n_val=args.n_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        fgm_warmup_epochs=args.fgm_warmup_epochs,
        linear_probe_epochs=args.linear_probe_epochs,
    )


def main() -> None:
    args = parse_args()
    config = build_signal_fidelity_config(sys.argv[1:])
    if args.strict_md_behavior:
        if config.linear_probe_epochs <= 0:
            config = SignalFidelityConfig(
                output_dir=config.output_dir,
                s_values=config.s_values,
                eta_values=config.eta_values,
                points=config.points,
                behavior_s_values=config.behavior_s_values,
                seeds=config.seeds,
                n_train=config.n_train,
                n_val=config.n_val,
                epochs=config.epochs,
                batch_size=config.batch_size,
                lr=config.lr,
                device=config.device,
                fgm_warmup_epochs=config.fgm_warmup_epochs,
                linear_probe_epochs=30,
            )
        run_strict_md_behavior(config)
    else:
        run_signal_fidelity_grid(config)


if __name__ == "__main__":
    main()
