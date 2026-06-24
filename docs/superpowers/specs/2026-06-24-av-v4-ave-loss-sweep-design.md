# AV_v4 AVE Loss-Weight Sweep Design

## Goal

Support four AVE audiovisual training experiments in `AV_v4`, each running for 120 epochs:

1. `L_fusion + 3 * L_audio + 3 * L_visual`
2. `L_fusion + 5 * L_audio + 5 * L_visual`
3. `L_fusion + 1 * L_audio + 5 * L_visual`
4. `L_fusion + 1 * L_audio + 3 * L_visual`

Each epoch report must show the epoch duration and cumulative training duration.

## Command-Line Interface

Add two AVE arguments:

```text
--audio-loss-weight
--visual-loss-weight
```

Both default to `5.0` to preserve the current AV_v4 AVE behavior.

The shared AV_v4 training functions accept independent audio and visual probe-loss weights. Their defaults remain `1.0`, preserving CREMA-D and KineticSound behavior.

## Logging

`AV_v4.train_ave.run_training` records a monotonic start time before the epoch loop. For every epoch it calculates:

- `epoch_seconds`: duration of the current training and validation epoch.
- `elapsed_seconds`: cumulative duration since the training loop started.

Both values are:

- included in the console epoch report;
- stored in each history record;
- written to JSON and JSONL history through the existing logging path.

## Git Bash Script

Create `scripts/run_av_v4_ave_loss_sweep.sh`. The script runs the four experiments sequentially with:

- `python -u -m AV_v4.train_ave`;
- `--epochs 120`;
- distinct audio and visual weights;
- distinct output directories;
- stdout and stderr copied to a per-experiment `train.log` using `tee`;
- `set -euo pipefail`, so a failed experiment stops the sweep.

The output directories are:

```text
runs/av_v4_ave_av3
runs/av_v4_ave_av5
runs/av_v4_ave_v5
runs/av_v4_ave_v3
```

## Tests

Tests verify:

1. independent audio and visual weights produce the requested total loss;
2. AVE CLI accepts and defaults the loss weights;
3. AVE forwards both weights to train, validation, and test calls;
4. epoch report formatting includes epoch and elapsed time;
5. the Git Bash script contains all four 120-epoch experiment commands and output directories.
