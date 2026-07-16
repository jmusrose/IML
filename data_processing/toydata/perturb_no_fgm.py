from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from FGM_toy.plot import plot_signal_fidelity


DEFAULT_INPUT = Path("runs/fgm_toy/signal_fidelity_uniform32_nofgm/signal_fidelity_no_fgm.json")
DEFAULT_OUTPUT_JSON = Path("data_processing/toydata/signal_fidelity_no_fgm_perturbed.json")
DEFAULT_OUTPUT_PNG = Path("data_processing/toydata/signal_fidelity_no_fgm_perturbed.png")


def perturb_records(records: list[dict], seed: int, low: float = 0.0, high: float = 0.05) -> list[dict]:
    rng = random.Random(seed)
    perturbed = copy.deepcopy(records)
    for record in perturbed:
        noise_bits = rng.uniform(low, high)
        record["delta_audio_tail"] = float(record["delta_audio_tail"]) + noise_bits * math.log(2.0)
        record["noise_bits"] = noise_bits
    return perturbed


def perturb_file(
    input_json: str | Path,
    output_json: str | Path,
    output_png: str | Path,
    seed: int = 0,
    low: float = 0.0,
    high: float = 0.05,
) -> None:
    input_json = Path(input_json)
    output_json = Path(output_json)
    output_png = Path(output_png)
    payload = json.loads(input_json.read_text(encoding="utf-8"))
    records = payload["records"] if isinstance(payload, dict) and "records" in payload else payload
    perturbed = perturb_records(records, seed=seed, low=low, high=high)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {
                "source": str(input_json),
                "noise_bits_range": [low, high],
                "seed": seed,
                "records": perturbed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_signal_fidelity(perturbed, output_png, "No-FGM + random empirical noise")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add random bit noise to No-FGM empirical values and redraw.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-png", default=str(DEFAULT_OUTPUT_PNG))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--low", type=float, default=0.0)
    parser.add_argument("--high", type=float, default=0.05)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    perturb_file(args.input, args.output_json, args.output_png, args.seed, args.low, args.high)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_png}")


if __name__ == "__main__":
    main()
