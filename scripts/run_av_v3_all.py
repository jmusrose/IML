from __future__ import annotations

import subprocess
import sys
from pathlib import Path


RUN_MODULES = (
    "AV_v3.train_cremad",
    "AV_v3.train_ks",
    "AV_v3.train_ave",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args:
        print("Usage: python scripts/run_av_v3_all.py", file=sys.stderr)
        return 2

    root = repo_root()
    for module_name in RUN_MODULES:
        command = [sys.executable, "-m", module_name]
        print(f"Running {' '.join(command)}")
        completed = subprocess.run(command, cwd=root)
        if completed.returncode != 0:
            print(f"{module_name} failed with exit code {completed.returncode}", file=sys.stderr)
            return completed.returncode

    print("All AV_v3 training runs finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
