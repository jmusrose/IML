from __future__ import annotations

import argparse
import sys
from copy import copy
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from MOSI_v2.train_mosi import build_arg_parser
from MOSI_v2.train_mosi import run_training


DATASET_DEFAULTS = {
    "mosi": {
        "data_path": "dataset/mosi.pkl",
        "output_dir": "runs/mosi_baseline",
        "vision_dim": 47,
    },
    "mosei": {
        "data_path": "dataset/mosei.pkl",
        "output_dir": "runs/mosei_baseline",
        "vision_dim": 35,
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser(
        description="Train CMU-MOSI and CMU-MOSEI sequentially with shared hyperparameters.",
        default_data_path=DATASET_DEFAULTS["mosi"]["data_path"],
        default_output_dir=DATASET_DEFAULTS["mosi"]["output_dir"],
        default_vision_dim=DATASET_DEFAULTS["mosi"]["vision_dim"],
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_DEFAULTS),
        default=["mosi", "mosei"],
        help="Datasets to train in order.",
    )
    return parser.parse_args(argv)


def build_dataset_args(args: argparse.Namespace) -> list[argparse.Namespace]:
    dataset_args = []
    for dataset_name in args.datasets:
        config = DATASET_DEFAULTS[dataset_name]
        current = copy(args)
        current.dataset_name = dataset_name
        current.data_path = config["data_path"]
        current.output_dir = config["output_dir"]
        current.vision_dim = config["vision_dim"]
        dataset_args.append(current)
    return dataset_args


def main(argv: list[str] | None = None) -> dict[str, dict[str, float]]:
    args = parse_args(argv)
    results = {}
    for current_args in build_dataset_args(args):
        print(f"Running {current_args.dataset_name.upper()} from {current_args.data_path}")
        results[current_args.dataset_name] = run_training(current_args)
    return results


if __name__ == "__main__":
    main()
