# FGM Toy

Small synthetic CMI experiment for checking whether FGM follows conditional contribution instead of single-modality strength.

## Sanity Check

```powershell
conda run -n pytorch2.5 python -m FGM_toy.data
```

## Single Run: One Point Only

```powershell
conda run -n pytorch2.5 python -m FGM_toy.train --mode fgm --s 0.5 --eta 0.1 --epochs 20 --output runs/fgm_toy/example_fgm.json
conda run -n pytorch2.5 python -m FGM_toy.train --mode no_fgm --s 0.5 --eta 0.1 --epochs 20 --output runs/fgm_toy/example_no_fgm.json
```

`train.py` runs one `(s, eta, seed, mode)` configuration, so it is one scatter point. Its console JSON is the raw record for that point.

Modes:

- `fgm`: CMI-FGM with feature and split-classifier gradient modulation.
- `no_fgm`: no gradient modulation; used as the Figure 3 scatter baseline.
- `acc_baseline`: same modulation path, but replaces CMI signal with single-modality probe correctness.
- `loss_gap_baseline`: same modulation path, but replaces CMI signal with own-modality loss gaps.
- `strength_signal`: legacy single-modality strength proxy; kept for old commands, not used in the main grid.

## Signal-Fidelity Scatter Grid

Use `run_grid.py` for the real Figure 2/3 scatter plots. It sweeps many `(s, eta)` values, aggregates records, and writes the FGM and No-FGM plots.

Quick smoke run:

```powershell
conda run --live-stream -n pytorch2.5 python FGM_toy/run_grid.py --quick --output-dir runs/fgm_toy/quick_signal_fidelity
```

Spec 25-point grid:

```powershell
conda run --live-stream -n pytorch2.5 python FGM_toy/run_grid.py --output-dir runs/fgm_toy/signal_fidelity --device cuda
```

32-point grid with more even true-CMI spacing and lower difficulty:

```powershell
conda run --live-stream -n pytorch2.5 python FGM_toy/run_grid.py --preset uniform32 --output-dir runs/fgm_toy/signal_fidelity_uniform32 --device cuda
```

Outputs:

- `behavior_lines.json`
- `behavior_lines.png`
- `signal_fidelity_fgm.json`
- `signal_fidelity_no_fgm.json`
- `signal_fidelity_fgm.png`
- `signal_fidelity_no_fgm.png`

The scatter preset controls Figure 2/3. Figure 1 always uses `eta=0` and sweeps `s=0.0,0.1,...,1.0` for the full CLI run.
Figure 1 compares FGM against `acc_baseline`, so the red dashed line is the direction implied by single-modality accuracy.

Strict MD Figure 1:

```powershell
conda run --live-stream -n pytorch2.5 python FGM_toy/run_grid.py --strict-md-behavior --output-dir runs/fgm_toy/strict_md_behavior --device cuda
```

This runs `s=0.0,0.1,...,1.0`, `eta=0`, modes `fgm/no_fgm/strength_signal`, seeds `0,1,2`, and frozen-encoder linear probes. It writes `behavior_lines_md.json` and `behavior_lines_md.png`.

`plot.py` is lower-level. It accepts one JSON record or a JSON object with `records`.

```powershell
conda run -n pytorch2.5 python -m FGM_toy.plot runs/fgm_toy/example_fgm.json --output runs/fgm_toy/example_fgm.png --title "FGM"
```
