# AV_v4 AVE Full-Stage Probe Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make only AV_v4 AVE audiovisual training and evaluation use `L_fusion + 5 * (L_audio + L_visual)` with probe features attached to both encoders for the full run.

**Architecture:** Add optional probe-loss configuration to the shared AV_v4 training functions while retaining their current defaults. The AVE entry point explicitly supplies its weight and detachment policy; CREMA-D and KineticSound continue using the defaults.

**Tech Stack:** Python 3.11, PyTorch 2.5, pytest.

---

### Task 1: Specify the AVE loss and gradient behavior

**Files:**
- Create: `tests/test_av_v4_ave_probe_loss.py`
- Test: `AV_v4/training.py`
- Test: `AV_v4/models/baseline.py`

- [ ] **Step 1: Write the failing weighted-loss test**

```python
import torch


def test_av_v4_weighted_probe_loss_uses_requested_factor():
    from AV_v4.models import AVBaseline
    from AV_v4.training import forward_and_losses

    torch.manual_seed(0)
    model = AVBaseline(num_classes=3)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    audio = torch.randn(2, 1, 64, 80)
    visual = torch.randn(2, 3, 2, 64, 64)
    labels = torch.tensor([0, 1])

    _, losses, handles = forward_and_losses(
        model,
        (audio, visual),
        labels,
        "av",
        criterion,
        probe_loss_weight=5.0,
        detach_probe_features=False,
    )
    for handle in handles:
        handle.remove()

    expected = losses["fusion_loss"] + 5.0 * (
        losses["audio_loss"] + losses["visual_loss"]
    )
    assert torch.allclose(losses["loss"], expected)
```

- [ ] **Step 2: Write the failing full-stage gradient test**

```python
def test_av_v4_probe_losses_reach_encoders_after_fgm_warmup():
    from AV_v4.models import AVBaseline
    from AV_v4.training import forward_and_losses
    from cmi_fgm import CMIFGMState

    model = AVBaseline(num_classes=3)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    state = CMIFGMState(("audio", "visual"), warmup_steps=1)
    state.num_updates = 1
    audio = torch.randn(2, 1, 64, 80)
    visual = torch.randn(2, 3, 2, 64, 64)
    labels = torch.tensor([0, 1])

    _, losses, handles = forward_and_losses(
        model,
        (audio, visual),
        labels,
        "av",
        criterion,
        fgm_state=state,
        probe_loss_weight=5.0,
        detach_probe_features=False,
    )
    (losses["audio_loss"] + losses["visual_loss"]).backward()
    for handle in handles:
        handle.remove()

    audio_grad = sum(
        p.grad.abs().sum().item()
        for p in model.audio_net.parameters()
        if p.grad is not None
    )
    visual_grad = sum(
        p.grad.abs().sum().item()
        for p in model.visual_net.parameters()
        if p.grad is not None
    )
    assert audio_grad > 0
    assert visual_grad > 0
```

- [ ] **Step 3: Run the tests and verify RED**

Run:

```powershell
E:\anaconda3\envs\pytorch2.5\python.exe -m pytest tests/test_av_v4_ave_probe_loss.py -q
```

Expected: both tests fail because `forward_and_losses` does not accept `probe_loss_weight` or `detach_probe_features`.

- [ ] **Step 4: Commit the failing tests**

```powershell
git add tests/test_av_v4_ave_probe_loss.py
git commit -m "test: specify AV4 AVE probe loss"
```

### Task 2: Parameterize the shared AV_v4 training functions

**Files:**
- Modify: `AV_v4/training.py:145-205`
- Modify: `AV_v4/training.py:249-289`
- Modify: `AV_v4/training.py:313-347`
- Test: `tests/test_av_v4_ave_probe_loss.py`

- [ ] **Step 1: Add optional arguments to `forward_and_losses`**

Add:

```python
    probe_loss_weight: float = 1.0,
    detach_probe_features: bool | None = None,
```

Resolve the detach behavior with:

```python
        resolved_detach = detach_probe_features
        if resolved_detach is None:
            resolved_detach = True
            if fgm_state is not None and fgm_state.num_updates < fgm_state.warmup_steps:
                resolved_detach = False
```

Pass `resolved_detach` to `model.forward_with_modal_logits`.

- [ ] **Step 2: Apply the requested probe-loss weight**

Replace the AV total loss with:

```python
        total_loss = fusion_loss + probe_loss_weight * (audio_loss + visual_loss)
```

and store `"loss": total_loss` in the losses dictionary.

- [ ] **Step 3: Forward the options through `train_one_epoch`**

Add defaults to its signature:

```python
    probe_loss_weight: float = 1.0,
    detach_probe_features: bool | None = None,
```

Pass both arguments to `forward_and_losses`.

- [ ] **Step 4: Forward the options through `evaluate`**

Add the same defaults to its signature and pass both arguments to `forward_and_losses`.

- [ ] **Step 5: Run the focused tests and verify GREEN**

Run:

```powershell
E:\anaconda3\envs\pytorch2.5\python.exe -m pytest tests/test_av_v4_ave_probe_loss.py -q
```

Expected: both tests pass.

- [ ] **Step 6: Commit the shared training change**

```powershell
git add AV_v4/training.py tests/test_av_v4_ave_probe_loss.py
git commit -m "feat: parameterize AV4 probe losses"
```

### Task 3: Enable the AVE-specific policy

**Files:**
- Modify: `AV_v4/train_ave.py:101-120`
- Modify: `AV_v4/train_ave.py:143-151`
- Test: `tests/test_av_v4_ave_probe_loss.py`

- [ ] **Step 1: Add a failing entry-point forwarding test**

Add a source-level assertion that every AVE call to `train_one_epoch` and `evaluate` includes:

```python
probe_loss_weight=5.0
detach_probe_features=False
```

Use Python AST inspection so formatting changes do not break the test.

- [ ] **Step 2: Run the entry-point test and verify RED**

Run:

```powershell
E:\anaconda3\envs\pytorch2.5\python.exe -m pytest tests/test_av_v4_ave_probe_loss.py -q
```

Expected: the forwarding test fails because `AV_v4.train_ave` does not yet pass the options.

- [ ] **Step 3: Pass the AVE policy to training and evaluation**

For the train, validation, and test calls, add:

```python
            probe_loss_weight=5.0,
            detach_probe_features=False,
```

Do not change `AV_v4.train_cremad` or `AV_v4.train_ks`.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```powershell
E:\anaconda3\envs\pytorch2.5\python.exe -m pytest tests/test_av_v4_ave_probe_loss.py -q
```

Expected: all AV4 AVE probe-loss tests pass.

- [ ] **Step 5: Commit the AVE entry-point change**

```powershell
git add AV_v4/train_ave.py tests/test_av_v4_ave_probe_loss.py
git commit -m "feat: keep AV4 AVE probe supervision active"
```

### Task 4: Regression verification

**Files:**
- Verify: `AV_v4/training.py`
- Verify: `AV_v4/train_ave.py`
- Verify: `AV_v4/train_cremad.py`
- Verify: `AV_v4/train_ks.py`

- [ ] **Step 1: Run the focused AV4 tests**

```powershell
E:\anaconda3\envs\pytorch2.5\python.exe -m pytest tests/test_av_v4_ave_probe_loss.py -q
```

- [ ] **Step 2: Run related training tests**

```powershell
E:\anaconda3\envs\pytorch2.5\python.exe -m pytest tests/test_av_v3_probe_warmup.py tests/test_av_v3_ks_ave_complete.py -q
```

- [ ] **Step 3: Compile AV4 modules**

```powershell
E:\anaconda3\envs\pytorch2.5\python.exe -m compileall -q AV_v4
```

- [ ] **Step 4: Inspect the final diff**

```powershell
git diff -- AV_v4/training.py AV_v4/train_ave.py tests/test_av_v4_ave_probe_loss.py
```

Confirm that CREMA-D and KineticSound entry points are unchanged.
