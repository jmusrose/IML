# FGM Masking and Frozen-Probe Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained single-seed pipeline that trains matched No-FGM/FGM audio-visual models on CREMA-D, Kinetics-Sounds, and AVE, then measures fusion dependence with feature masking and representation quality with newly initialized frozen linear probes.

**Architecture:** Copy the complete relevant AV_v4 implementation and root CMI-FGM helper into `FGM_masking_probe`, replace external imports with package-relative imports, and add focused orchestration/analysis modules. Training writes reloadable best checkpoints; post-training analyses operate only on reloaded checkpoints and deterministic analysis loaders.

**Tech Stack:** Python 3.11, PyTorch 2.5, NumPy, matplotlib, pytest, standard-library JSON/CSV/path utilities.

## Global Constraints

- `FGM_masking_probe` must have no runtime import dependency on `AV_v4` or the root `cmi_fgm.py`.
- Copy models, datasets, shared training code, and the three dataset entry points; do not copy `__pycache__` or run artifacts.
- Run exactly one seed per dataset invocation; do not calculate mean/std across seeds.
- No-FGM and FGM retain identical auxiliary probe losses and weights; only the `fgm` Boolean differs.
- Mask at pooled feature nodes, using both zero and each model's deterministic training-feature mean.
- Frozen probes are new `nn.Linear(512, num_classes)` heads trained for 30 epochs with Adam at `1e-3` by default.
- Empirical acceptance-check failures are reported, not raised as program errors.
- Use `E:\anaconda3\envs\pytorch2.5\python.exe` for verification.

---

## File Map

- `FGM_masking_probe/models/`: copied AV ResNet18 encoders, pooling, fusion classifier, and auxiliary heads.
- `FGM_masking_probe/datasets/`: copied CREMA-D, KS, and AVE discovery, transforms, and datasets.
- `FGM_masking_probe/cmi_fgm.py`: copied CMI-FGM state and gradient-hook implementation.
- `FGM_masking_probe/training.py`: copied shared training utilities plus common best-checkpoint writer/loader.
- `FGM_masking_probe/train_cremad.py`, `train_ks.py`, `train_ave.py`: copied dataset-specific training entry points, changed only for local imports and best-checkpoint persistence.
- `FGM_masking_probe/train_video.py`: copied dataset dispatcher with local imports.
- `FGM_masking_probe/configs.py`: dataset adapters, paired-configuration construction, and deterministic analysis loaders.
- `FGM_masking_probe/train_pair.py`: matched No-FGM/FGM training orchestration.
- `FGM_masking_probe/masking.py`: feature means, masked fusion evaluation, drops, asymmetry, dominant/weak labels.
- `FGM_masking_probe/probe.py`: feature caching and newly initialized frozen linear probes.
- `FGM_masking_probe/report.py`: JSON/CSV/Markdown and PNG/PDF output.
- `FGM_masking_probe/run.py`: complete CLI pipeline.
- `tests/test_fgm_masking_probe_copy.py`: isolation and checkpoint tests.
- `tests/test_fgm_masking_probe_analysis.py`: masking and probe tests.
- `tests/test_fgm_masking_probe_report.py`: reporting and CLI smoke tests.

---

### Task 1: Create the Self-Contained Package Copy

**Files:**
- Create: `FGM_masking_probe/__init__.py`
- Create: `FGM_masking_probe/cmi_fgm.py`
- Create: `FGM_masking_probe/training.py`
- Create: `FGM_masking_probe/train_video.py`
- Create: `FGM_masking_probe/train_cremad.py`
- Create: `FGM_masking_probe/train_ks.py`
- Create: `FGM_masking_probe/train_ave.py`
- Create: `FGM_masking_probe/models/__init__.py`
- Create: `FGM_masking_probe/models/baseline.py`
- Create: `FGM_masking_probe/datasets/__init__.py`
- Create: `FGM_masking_probe/datasets/cremad.py`
- Create: `FGM_masking_probe/datasets/ks.py`
- Create: `FGM_masking_probe/datasets/ave.py`
- Test: `tests/test_fgm_masking_probe_copy.py`

**Interfaces:**
- Consumes: source files under `AV_v4/` and root `cmi_fgm.py`.
- Produces: importable `FGM_masking_probe.models`, `FGM_masking_probe.datasets`, and local training entry points.

- [ ] **Step 1: Write the failing isolation test**

```python
import inspect


def test_copied_package_uses_only_local_runtime_modules():
    from FGM_masking_probe import training
    from FGM_masking_probe.models import AVBaseline
    from FGM_masking_probe.train_ave import parse_args

    assert AVBaseline.__module__.startswith("FGM_masking_probe.")
    assert training.AVBaseline.__module__.startswith("FGM_masking_probe.")
    assert parse_args([]).modality == "av"
    source = inspect.getsource(training)
    assert "from AV_v4" not in source
    assert "from cmi_fgm" not in source
```

- [ ] **Step 2: Run the test and confirm the package is absent**

Run:

```powershell
E:\anaconda3\envs\pytorch2.5\python.exe -m pytest tests/test_fgm_masking_probe_copy.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'FGM_masking_probe'`.

- [ ] **Step 3: Copy the source tree mechanically and replace imports**

Copy only `.py` source files. Apply these import mappings everywhere in the copy:

```python
from cmi_fgm import ...
# becomes
from .cmi_fgm import ...

from AV_v4.models import ...
# becomes
from .models import ...

from AV_v4.datasets import ...
# becomes
from .datasets import ...

from AV_v4.training import ...
# becomes
from .training import ...

from AV_v4 import train_ave, train_cremad, train_ks
# becomes
from . import train_ave, train_cremad, train_ks
```

Retain direct-script path bootstrapping where present, but import through `FGM_masking_probe` rather than `AV_v4`.

- [ ] **Step 4: Run the isolation test**

Run the Task 1 pytest command. Expected: PASS.

- [ ] **Step 5: Compile the copied package**

Run:

```powershell
E:\anaconda3\envs\pytorch2.5\python.exe -m compileall -q FGM_masking_probe
```

Expected: exit code 0.

- [ ] **Step 6: Commit**

```powershell
git add FGM_masking_probe tests/test_fgm_masking_probe_copy.py
git commit -m "feat: copy self-contained AV FGM package"
```

---

### Task 2: Save and Reload Best Checkpoints and Build Matched Pairs

**Files:**
- Modify: `FGM_masking_probe/training.py`
- Modify: `FGM_masking_probe/train_cremad.py`
- Modify: `FGM_masking_probe/train_ks.py`
- Modify: `FGM_masking_probe/train_ave.py`
- Create: `FGM_masking_probe/configs.py`
- Create: `FGM_masking_probe/train_pair.py`
- Modify: `tests/test_fgm_masking_probe_copy.py`

**Interfaces:**
- Produces: `save_best_checkpoint(path, model, epoch, metrics, args, dataset, method) -> Path`.
- Produces: `load_best_checkpoint(path, device) -> tuple[nn.Module, dict]`.
- Produces: `build_method_args(dataset, method, output_root, seed, device, epochs=None, num_workers=None) -> argparse.Namespace`.
- Produces: `train_pair(dataset, output_root, seed, device, epochs=None, num_workers=None) -> dict[str, Path]`.

- [ ] **Step 1: Add failing checkpoint and matched-config tests**

```python
from argparse import Namespace
from pathlib import Path

import torch


def test_checkpoint_round_trip(tmp_path):
    from FGM_masking_probe.models import AVBaseline
    from FGM_masking_probe.training import load_best_checkpoint, save_best_checkpoint

    model = AVBaseline(num_classes=3)
    path = save_best_checkpoint(
        tmp_path / "best.pt", model, 4, {"acc": 0.75},
        Namespace(num_classes=3, modality="av"), "ave", "fgm",
    )
    restored, payload = load_best_checkpoint(path, torch.device("cpu"))
    assert payload["epoch"] == 4
    assert payload["dataset"] == "ave"
    assert payload["method"] == "fgm"
    for expected, actual in zip(model.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)


def test_method_configs_differ_only_by_fgm_and_output(tmp_path):
    from FGM_masking_probe.configs import build_method_args

    baseline = build_method_args("ave", "no_fgm", tmp_path, 7, "cpu", epochs=2, num_workers=0)
    fgm = build_method_args("ave", "fgm", tmp_path, 7, "cpu", epochs=2, num_workers=0)
    ignored = {"fgm", "output_dir"}
    assert {k: v for k, v in vars(baseline).items() if k not in ignored} == {
        k: v for k, v in vars(fgm).items() if k not in ignored
    }
    assert baseline.fgm is False and fgm.fgm is True
```

- [ ] **Step 2: Run the focused tests and confirm missing interfaces**

Run:

```powershell
E:\anaconda3\envs\pytorch2.5\python.exe -m pytest tests/test_fgm_masking_probe_copy.py -q
```

Expected: FAIL importing `save_best_checkpoint` or `configs`.

- [ ] **Step 3: Implement checkpoint helpers**

Add to `training.py`:

```python
def save_best_checkpoint(path, model, epoch, metrics, args, dataset, method):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": int(epoch),
        "model": clone_model_state_dict(model),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "args": args_to_dict(args),
        "dataset": str(dataset),
        "method": str(method),
    }, path)
    return path


def load_best_checkpoint(path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    args = payload["args"]
    if args.get("modality", "av") != "av":
        raise ValueError("Masking/probe analysis requires an AV checkpoint.")
    model = build_model("av", num_classes=int(args["num_classes"])).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload
```

- [ ] **Step 4: Save `best.pt` from every copied dataset trainer**

Immediately after loading `best_state_dict`, call:

```python
save_best_checkpoint(
    output_dir / "best.pt",
    model,
    best_epoch,
    best_metrics["val"],
    args,
    dataset="ave",  # use "cremad" or "ks" in the other entry points
    method="fgm" if args.fgm else "no_fgm",
)
```

Ensure `save_best_checkpoint` is imported from the local `.training` module.

- [ ] **Step 5: Implement dataset adapters and matched arguments**

In `configs.py`, map dataset names to local modules:

```python
ADAPTERS = {
    "ave": (train_ave.parse_args, train_ave.create_dataloaders, train_ave.run_training),
    "cremad": (train_cremad.parse_args, train_cremad.create_dataloaders, train_cremad.run_training),
    "ks": (train_ks.parse_args, train_ks.create_dataloaders, train_ks.run_training),
}
```

`build_method_args` parses an empty argument list, assigns the requested seed/device/optional overrides, sets `modality="av"`, sets `fgm = method == "fgm"`, and sets output roots to `<output_root>/<dataset>/<method>`.

- [ ] **Step 6: Implement sequential paired training**

```python
def train_pair(dataset, output_root, seed, device, epochs=None, num_workers=None):
    paths = {}
    for method in ("no_fgm", "fgm"):
        args = build_method_args(dataset, method, output_root, seed, device, epochs, num_workers)
        _, _, run_training = get_adapter(dataset)
        run_training(args)
        checkpoint = Path(args.output_dir) / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Training did not create {checkpoint}")
        paths[method] = checkpoint
    return paths
```

- [ ] **Step 7: Run tests**

Run the Task 2 pytest command. Expected: PASS.

- [ ] **Step 8: Commit**

```powershell
git add FGM_masking_probe tests/test_fgm_masking_probe_copy.py
git commit -m "feat: train matched FGM pairs with checkpoints"
```

---

### Task 3: Add Deterministic Analysis Loaders and Feature Masking

**Files:**
- Modify: `FGM_masking_probe/configs.py`
- Create: `FGM_masking_probe/masking.py`
- Create: `tests/test_fgm_masking_probe_analysis.py`

**Interfaces:**
- Produces: `create_analysis_loaders(dataset, args) -> tuple[DataLoader, DataLoader]` for deterministic train/test feature extraction.
- Produces: `compute_feature_means(model, loader, device) -> dict[str, Tensor]`.
- Produces: `evaluate_masking(model, loader, device, fill, means=None) -> dict[str, float]`.
- Produces: `summarize_masking(accuracies) -> dict[str, float]`.
- Produces: `identify_modalities(baseline_summary) -> dict[str, str]`.

- [ ] **Step 1: Write failing formula tests**

```python
import torch
from torch.utils.data import DataLoader, Dataset


class FeatureDataset(Dataset):
    def __init__(self):
        self.items = [
            {"audio": torch.tensor([2.0, 0.0]), "visual": torch.tensor([0.0, 2.0]), "label": torch.tensor(0)},
            {"audio": torch.tensor([0.0, 2.0]), "visual": torch.tensor([2.0, 0.0]), "label": torch.tensor(1)},
        ]
    def __len__(self): return len(self.items)
    def __getitem__(self, index): return self.items[index]


def test_masking_summary_and_weak_modality():
    from FGM_masking_probe.masking import identify_modalities, summarize_masking

    summary = summarize_masking({"full": 0.9, "mask_audio": 0.3, "mask_visual": 0.8})
    assert summary == {
        "full": 0.9, "mask_audio": 0.3, "mask_visual": 0.8,
        "drop_audio": 0.6, "drop_visual": 0.1, "asymmetry": 0.5,
    }
    assert identify_modalities(summary) == {"dominant": "audio", "weak": "visual"}
```

Use `pytest.approx` for floating-point assertions in the actual test.

- [ ] **Step 2: Run the focused test and confirm `masking` is missing**

Run:

```powershell
E:\anaconda3\envs\pytorch2.5\python.exe -m pytest tests/test_fgm_masking_probe_analysis.py -q
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement deterministic analysis loaders**

Start from each copied trainer's datasets, but build a non-shuffled training loader with evaluation transforms. For AVE and KS, set the copied training dataset's `mode="test"`; for CREMA-D, use `ResizeToTensorNormalize` and a fresh `np.random.default_rng(args.seed)`. Test loaders remain non-shuffled. Use `num_workers=args.num_workers`, `batch_size=args.batch_size`, and `drop_last=False`.

- [ ] **Step 4: Implement feature extraction and means**

```python
@torch.no_grad()
def compute_feature_means(model, loader, device):
    sums = {"audio": None, "visual": None}
    count = 0
    for batch in loader:
        audio, visual, labels = move_av_batch(batch, device)
        features = {
            "audio": model.extract_audio_feature(audio),
            "visual": model.extract_visual_feature(visual),
        }
        for name, feature in features.items():
            value = feature.sum(dim=0)
            sums[name] = value if sums[name] is None else sums[name] + value
        count += labels.numel()
    if count == 0:
        raise ValueError("Cannot compute feature means from an empty loader.")
    return {name: value / count for name, value in sums.items()}
```

- [ ] **Step 5: Implement one-pass masking evaluation**

For each batch, compute real features once. Build fusion inputs for `full`, `mask_audio`, and `mask_visual`; zero-fill with `zeros_like`, mean-fill with `means[name].expand_as(feature)`. Feed concatenated features into `model.classifier`. Validate `fill in {"zero", "train_mean"}` and require means for `train_mean`.

- [ ] **Step 6: Implement exact summaries**

```python
def summarize_masking(values):
    drop_audio = values["full"] - values["mask_audio"]
    drop_visual = values["full"] - values["mask_visual"]
    return {**values, "drop_audio": drop_audio, "drop_visual": drop_visual,
            "asymmetry": abs(drop_audio - drop_visual)}


def identify_modalities(summary):
    dominant = "audio" if summary["drop_audio"] >= summary["drop_visual"] else "visual"
    return {"dominant": dominant, "weak": "visual" if dominant == "audio" else "audio"}
```

- [ ] **Step 7: Run analysis tests**

Run the Task 3 pytest command. Expected: PASS.

- [ ] **Step 8: Commit**

```powershell
git add FGM_masking_probe/configs.py FGM_masking_probe/masking.py tests/test_fgm_masking_probe_analysis.py
git commit -m "feat: evaluate feature-level modality masking"
```

---

### Task 4: Add Newly Initialized Frozen Linear Probes

**Files:**
- Create: `FGM_masking_probe/probe.py`
- Modify: `tests/test_fgm_masking_probe_analysis.py`

**Interfaces:**
- Produces: `cache_modality_features(model, loader, modality, device) -> tuple[Tensor, Tensor]` on CPU.
- Produces: `train_linear_probe(train_features, train_labels, test_features, test_labels, num_classes, device, epochs=30, lr=1e-3, batch_size=256, seed=0) -> dict[str, float]`.
- Produces: `evaluate_frozen_probes(model, train_loader, test_loader, num_classes, device, **probe_kwargs) -> dict[str, dict[str, float]]`.

- [ ] **Step 1: Add failing frozen-probe tests**

```python
def test_cached_features_do_not_update_encoder_and_probe_learns():
    from FGM_masking_probe.probe import cache_modality_features, train_linear_probe

    model = TinyAVModel()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    loader = DataLoader(FeatureDataset(), batch_size=2)
    features, labels = cache_modality_features(model, loader, "audio", torch.device("cpu"))
    result = train_linear_probe(features, labels, features, labels, 2, torch.device("cpu"), epochs=20, lr=0.05, seed=3)
    assert result["accuracy"] >= 0.5
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())
```

The actual `TinyAVModel` exposes `extract_audio_feature` and `extract_visual_feature` and maps the synthetic features without trainable mutation.

- [ ] **Step 2: Run the test and confirm `probe` is missing**

Run the Task 3 pytest command. Expected: FAIL importing `FGM_masking_probe.probe`.

- [ ] **Step 3: Implement CPU feature caching**

Set `model.eval()`, wrap the loop in `torch.no_grad()`, call the selected extraction method, append `feature.detach().cpu()` and `label.detach().cpu()`, and concatenate. Reject unsupported modalities and empty loaders.

- [ ] **Step 4: Implement deterministic linear-probe training**

```python
torch.manual_seed(seed)
head = nn.Linear(train_features.shape[1], num_classes).to(device)
optimizer = torch.optim.Adam(head.parameters(), lr=lr)
dataset = TensorDataset(train_features, train_labels)
generator = torch.Generator().manual_seed(seed)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
for _ in range(epochs):
    head.train()
    for features, labels in loader:
        logits = head(features.to(device))
        loss = F.cross_entropy(logits, labels.to(device))
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite frozen-probe loss.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
```

Evaluate test accuracy with the head in evaluation mode and return `{"accuracy": ..., "epochs": ..., "lr": ...}`.

- [ ] **Step 5: Implement both-modality evaluation**

Cache audio and visual features once each, use the same probe seed and options for both, and return keys `audio` and `visual`.

- [ ] **Step 6: Run analysis tests**

Run the Task 3 pytest command. Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add FGM_masking_probe/probe.py tests/test_fgm_masking_probe_analysis.py
git commit -m "feat: add frozen unimodal linear probes"
```

---

### Task 5: Generate Results, Empirical Checks, and Figures

**Files:**
- Create: `FGM_masking_probe/report.py`
- Create: `tests/test_fgm_masking_probe_report.py`

**Interfaces:**
- Produces: `build_empirical_checks(dataset_result) -> dict[str, bool]`.
- Produces: `write_dataset_outputs(result, output_dir) -> dict[str, Path]`.
- Produces: `write_combined_outputs(results, output_dir) -> dict[str, Path]`.

- [ ] **Step 1: Write failing report tests**

```python
def test_failed_empirical_direction_is_reported_not_raised(tmp_path):
    from FGM_masking_probe.report import build_empirical_checks, write_dataset_outputs

    result = sample_result(fgm_full=0.7, baseline_full=0.8)
    checks = build_empirical_checks(result)
    assert checks["fgm_full_not_worse"] is False
    paths = write_dataset_outputs(result, tmp_path)
    assert paths["json"].is_file()
    assert paths["csv"].is_file()
    assert paths["markdown"].is_file()
    assert "NOT OBSERVED" in paths["markdown"].read_text(encoding="utf-8")
```

- [ ] **Step 2: Run the report test and confirm missing module**

Run:

```powershell
E:\anaconda3\envs\pytorch2.5\python.exe -m pytest tests/test_fgm_masking_probe_report.py -q
```

Expected: FAIL importing `FGM_masking_probe.report`.

- [ ] **Step 3: Implement empirical checks without thresholds invented beyond the spec**

Use exact directional comparisons. Determine weak modality only from baseline zero-mask drops. Check FGM weak probe against baseline, FGM full against baseline, FGM asymmetry against baseline, both FGM drops positive, and whether zero/mean comparisons agree in direction.

- [ ] **Step 4: Implement JSON, CSV, and Markdown writers**

Use `json.dump(..., indent=2)`, `csv.DictWriter`, and UTF-8 text. CSV rows contain dataset, method, fill, full, mask_audio, mask_visual, drop_audio, drop_visual, asymmetry, audio_probe_accuracy, and visual_probe_accuracy. Markdown shows observed checks as `OBSERVED` and failed directions as `NOT OBSERVED`; neither state raises.

- [ ] **Step 5: Implement grouped masking plots**

Create one subplot per dataset, shared y-axis for multiple datasets, conditions `full`, `mask_audio`, `mask_visual`, and adjacent bars for `no_fgm` and `fgm`. Plot the zero-fill primary result and save both `masking.png` and `masking.pdf`. Close the matplotlib figure after saving.

- [ ] **Step 6: Run report tests**

Run the Task 5 pytest command. Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add FGM_masking_probe/report.py tests/test_fgm_masking_probe_report.py
git commit -m "feat: report FGM masking and probe evidence"
```

---

### Task 6: Add the End-to-End CLI Pipeline

**Files:**
- Create: `FGM_masking_probe/run.py`
- Modify: `FGM_masking_probe/__init__.py`
- Modify: `tests/test_fgm_masking_probe_report.py`

**Interfaces:**
- Produces: `run_dataset(dataset, output_root, seed, device, epochs, num_workers, probe_epochs, probe_lr, probe_batch_size) -> dict`.
- Produces: `parse_args(argv=None) -> argparse.Namespace`.
- Produces: `main() -> None`.

- [ ] **Step 1: Add failing CLI tests**

```python
def test_cli_defaults_and_all_dataset_expansion():
    from FGM_masking_probe.run import datasets_for, parse_args

    args = parse_args(["--dataset", "all", "--device", "cpu"])
    assert datasets_for(args.dataset) == ["cremad", "ks", "ave"]
    assert args.probe_epochs == 30
    assert args.probe_lr == 1e-3
    assert args.seed == 0
```

- [ ] **Step 2: Run the report test and confirm `run` is missing**

Run the Task 5 pytest command. Expected: FAIL importing `FGM_masking_probe.run`.

- [ ] **Step 3: Implement the dataset pipeline**

For each dataset:

1. call `train_pair`;
2. rebuild deterministic analysis loaders from the resolved checkpoint args;
3. reload each checkpoint;
4. compute training means;
5. run zero and train-mean masking;
6. train/evaluate audio and visual frozen probes;
7. identify dominant/weak modality from No-FGM zero masking;
8. build checks and write per-dataset outputs.

Return one nested dictionary with `dataset`, `checkpoints`, `masking`, `probes`, `modalities`, and `checks`.

- [ ] **Step 4: Implement CLI parsing and all-dataset reporting**

Expose:

```text
--dataset {cremad,ks,ave,all}
--output-dir runs/fgm_masking_probe
--seed 0
--device cuda
--epochs (optional global override)
--num-workers (optional override)
--probe-epochs 30
--probe-lr 0.001
--probe-batch-size 256
```

Create one timestamped top-level run directory. For `all`, run in the fixed order CREMA-D, KS, AVE and then call `write_combined_outputs`.

- [ ] **Step 5: Run focused tests**

Run:

```powershell
E:\anaconda3\envs\pytorch2.5\python.exe -m pytest tests/test_fgm_masking_probe_copy.py tests/test_fgm_masking_probe_analysis.py tests/test_fgm_masking_probe_report.py -q
```

Expected: PASS.

- [ ] **Step 6: Run CLI help smoke test**

Run:

```powershell
E:\anaconda3\envs\pytorch2.5\python.exe -m FGM_masking_probe.run --help
```

Expected: exit code 0 and all documented options visible.

- [ ] **Step 7: Commit**

```powershell
git add FGM_masking_probe tests/test_fgm_masking_probe_report.py
git commit -m "feat: orchestrate complete masking probe pipeline"
```

---

### Task 7: Final Verification and Documentation

**Files:**
- Create: `FGM_masking_probe/README.md`
- Verify: all files created above.

**Interfaces:**
- Produces: runnable usage documentation and final verification evidence.

- [ ] **Step 1: Write usage documentation**

Document the three single-dataset commands, the `all` command, output tree, No-FGM fairness definition, single-seed limitation, probe defaults, expected GPU/runtime cost, and the rule that `NOT OBSERVED` is an empirical outcome rather than a crash.

- [ ] **Step 2: Scan for forbidden dependencies and copied caches**

Run:

```powershell
rg -n "from AV_v4|import AV_v4|from cmi_fgm" FGM_masking_probe -g "*.py"
Get-ChildItem FGM_masking_probe -Recurse -Directory -Filter __pycache__
```

Expected before compilation cleanup: no forbidden import matches. Remove generated `__pycache__` directories from the deliverable after verification.

- [ ] **Step 3: Run all focused tests**

Run:

```powershell
E:\anaconda3\envs\pytorch2.5\python.exe -m pytest tests/test_fgm_masking_probe_copy.py tests/test_fgm_masking_probe_analysis.py tests/test_fgm_masking_probe_report.py -q
```

Expected: all pass.

- [ ] **Step 4: Run package compilation and CLI smoke tests**

Run:

```powershell
E:\anaconda3\envs\pytorch2.5\python.exe -m compileall -q FGM_masking_probe
E:\anaconda3\envs\pytorch2.5\python.exe -m FGM_masking_probe.run --help
```

Expected: both commands exit 0.

- [ ] **Step 5: Run relevant AV_v4 regression tests**

Run:

```powershell
E:\anaconda3\envs\pytorch2.5\python.exe -m pytest tests/test_av_v4_ave_probe_loss.py tests/test_av_v4_cremad_ks_loss_weights.py tests/test_av_v4_run_output_policy.py -q
```

Expected: all pass, confirming the original package was not changed.

- [ ] **Step 6: Inspect final diff and status**

Run:

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors; only intended new-package/tests/docs changes plus pre-existing unrelated user changes.

- [ ] **Step 7: Commit documentation**

```powershell
git add FGM_masking_probe/README.md
git commit -m "docs: explain masking probe experiment pipeline"
```
