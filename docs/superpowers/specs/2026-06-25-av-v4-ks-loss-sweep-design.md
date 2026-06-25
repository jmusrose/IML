# AV_v4 KineticSound Visual Loss Sweep Design

## Goal

Add one Bash script that runs two AV_v4 KineticSound experiments with visual
auxiliary-loss weights `1` and `5`.

## Interface

Create `scripts/run_av_v4_ks_loss_sweep.sh`.

The script invokes `python -u -m AV_v4.train_ks` twice and passes only:

- `--visual-loss-weight`
- `--output-dir`

All other training parameters, including the audio loss weight, use
`AV_v4.train_ks` defaults.

## Runs

| Visual loss weight | Output directory |
|---|---|
| `1` | `runs/av_v4_ks_v1` |
| `5` | `runs/av_v4_ks_v5` |

Each run writes console output to `<output-dir>/train.log` with `tee`. The
script uses `set -euo pipefail`, resolves the project root from its own path,
and supports overriding Python through `PYTHON_BIN`.

## Verification

Add a focused source-level test that verifies:

- strict Bash settings are enabled;
- exactly two runs are declared;
- the requested visual weights and output directories are present;
- no audio loss weight or other training override is passed;
- each run writes `train.log`.
