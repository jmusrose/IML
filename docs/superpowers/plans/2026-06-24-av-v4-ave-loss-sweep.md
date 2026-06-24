# AV_v4 AVE Loss-Weight Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add independent AVE audio/visual auxiliary-loss weights, epoch timing logs, and a Git Bash script that runs four requested 120-epoch experiments.

**Architecture:** Parameterize AV_v4 shared loss computation with separate audio and visual weights while preserving unit defaults. AVE exposes the weights through CLI arguments and forwards them through training/evaluation. Timing remains in the AVE entry point, and the shell script orchestrates isolated output directories.

**Tech Stack:** Python 3.11, PyTorch 2.5, pytest, Bash.

---

### Task 1: Test independent weights and AVE CLI forwarding

**Files:**
- Modify: `tests/test_av_v4_ave_probe_loss.py`
- Test: `AV_v4/training.py`
- Test: `AV_v4/train_ave.py`

- [ ] Add a failing test calling `forward_and_losses` with `audio_loss_weight=1.0` and `visual_loss_weight=5.0`, asserting `loss == fusion + audio + 5 * visual`.
- [ ] Update the AVE AST test to require `args.audio_loss_weight` and `args.visual_loss_weight` at all three train/evaluate calls.
- [ ] Add a failing CLI test asserting defaults are `5.0` and explicit values are parsed.
- [ ] Run `E:\anaconda3\envs\pytorch2.5\python.exe -m pytest tests/test_av_v4_ave_probe_loss.py -q` and confirm failures are caused by missing interfaces.

### Task 2: Implement independent weights

**Files:**
- Modify: `AV_v4/training.py`
- Modify: `AV_v4/train_ave.py`

- [ ] Replace `probe_loss_weight` with `audio_loss_weight` and `visual_loss_weight`, each defaulting to `1.0`.
- [ ] Calculate `fusion_loss + audio_loss_weight * audio_loss + visual_loss_weight * visual_loss`.
- [ ] Forward both weights through `train_one_epoch` and `evaluate`.
- [ ] Add AVE CLI arguments defaulting to `5.0`.
- [ ] Forward the AVE argument values to train, validation, and test.
- [ ] Run focused tests and confirm they pass.

### Task 3: Add epoch timing

**Files:**
- Modify: `AV_v4/training.py`
- Modify: `AV_v4/train_ave.py`
- Modify: `tests/test_av_v4_ave_probe_loss.py`

- [ ] Add a failing formatting test for `format_epoch_report(..., epoch_seconds=65.4, elapsed_seconds=3665.4)`.
- [ ] Extend `format_epoch_report` with optional timing parameters and append `time 00:01:05 | elapsed 01:01:05` to the epoch heading.
- [ ] Use `time.perf_counter()` around each AVE epoch.
- [ ] Store `epoch_seconds` and `elapsed_seconds` in each history record.
- [ ] Pass timing values to `format_epoch_report`.
- [ ] Run focused tests and confirm they pass.

### Task 4: Create the Git Bash sweep script

**Files:**
- Create: `scripts/run_av_v4_ave_loss_sweep.sh`
- Create: `tests/test_run_av_v4_ave_loss_sweep_script.py`

- [ ] Add a failing test that checks the script has `set -euo pipefail`, four `--epochs 120` invocations, the four requested weight pairs, four distinct output directories, and per-run `tee` logs.
- [ ] Implement a Bash `run_experiment` helper that creates the output directory and invokes `python -u -m AV_v4.train_ave`.
- [ ] Add the four runs in this order: AV3, AV5, V5, V3.
- [ ] Run the script test and confirm it passes.

### Task 5: Verify

**Files:**
- Verify: `AV_v4/training.py`
- Verify: `AV_v4/train_ave.py`
- Verify: `scripts/run_av_v4_ave_loss_sweep.sh`

- [ ] Run both new test files.
- [ ] Compile `AV_v4`.
- [ ] Run `bash -n scripts/run_av_v4_ave_loss_sweep.sh` when Bash is available.
- [ ] Inspect the final diff and confirm AV3, AV4 CREMA-D, and AV4 KS are unchanged.
