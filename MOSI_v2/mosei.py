from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from MOSI_v2.train_mosi import parse_args as _parse_args
from MOSI_v2.train_mosi import run_training


def parse_args(argv: list[str] | None = None):
    return _parse_args(
        argv,
        description="Train CMU-MOSEI BERT + visual/audio sequence regression model.",
        default_data_path="dataset/mosei.pkl",
        default_output_dir="runs/mosei_baseline",
        default_vision_dim=35,
    )


def main(argv: list[str] | None = None) -> None:
    run_training(parse_args(argv))


if __name__ == "__main__":
    main()
