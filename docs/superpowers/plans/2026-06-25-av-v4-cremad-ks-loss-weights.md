# AV_v4 CREMA-D and KineticSound Loss Weights Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add independent CLI-configurable audio and visual auxiliary-loss weights to AV_v4 CREMA-D and KineticSound while preserving warmup-based probe detachment.

**Architecture:** KineticSound forwards new CLI values into the existing independently weighted `AV_v4.training` functions. CREMA-D receives equivalent parameters in its local training functions. Both default to unit weights and leave AVE untouched.

**Tech Stack:** Python 3.11, PyTorch 2.5, pytest.

---

### Task 1: Add failing behavior tests

**Files:**
- Create: `tests/test_av_v4_cremad_ks_loss_weights.py`

- [ ] Test CREMA-D `forward_and_losses` with audio weight 1 and visual weight 3.
- [ ] Test CREMA-D and KS CLI defaults of 1 and explicit parameter parsing.
- [ ] AST-test that both entry points forward `args.audio_loss_weight` and `args.visual_loss_weight` to train, validation, and test.
- [ ] Test that CREMA-D probe losses produce encoder gradients during warmup and no encoder gradients after warmup.
- [ ] Run the focused test file and verify expected failures from missing interfaces.

### Task 2: Implement CREMA-D weights

**Files:**
- Modify: `AV_v4/train_cremad.py`

- [ ] Add `audio_loss_weight` and `visual_loss_weight` defaults to local loss, train, and evaluate functions.
- [ ] Compute `fusion + audio_weight * audio + visual_weight * visual`.
- [ ] Add CLI arguments defaulting to 1.
- [ ] Forward values to train, validation, and test calls.
- [ ] Run focused tests.

### Task 3: Implement KineticSound forwarding

**Files:**
- Modify: `AV_v4/train_ks.py`

- [ ] Ensure imports use `AV_v4.datasets` and `AV_v4.training`.
- [ ] Add CLI arguments defaulting to 1.
- [ ] Forward values to train, validation, and test calls.
- [ ] Run focused tests.

### Task 4: Verify

**Files:**
- Verify: `AV_v4/train_cremad.py`
- Verify: `AV_v4/train_ks.py`
- Verify: `AV_v4/train_ave.py`

- [ ] Run the new focused tests and existing AV4 AVE tests.
- [ ] Compile AV_v4.
- [ ] Inspect imports and diffs, confirming AVE still passes independent weights with full-stage no-detach.
