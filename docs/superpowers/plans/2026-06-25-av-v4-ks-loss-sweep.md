# AV_v4 KineticSound Visual Loss Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Bash launcher for AV_v4 KineticSound runs with visual auxiliary-loss weights 1 and 5 while preserving every other training default.

**Architecture:** Follow the existing AVE loss-sweep shell pattern. A source-level pytest test defines the exact two invocations and prevents accidental overrides of audio loss or other training parameters.

**Tech Stack:** Bash, Python, pytest.

---

### Task 1: Specify the KS sweep script

**Files:**
- Create: `tests/test_run_av_v4_ks_loss_sweep_script.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def test_av_v4_ks_loss_sweep_script_contains_requested_runs():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_av_v4_ks_loss_sweep.sh"
    )
    source = script_path.read_text(encoding="utf-8")

    assert "set -euo pipefail" in source
    assert source.count("run_experiment ") == 2
    assert 'run_experiment "v1" 1 "runs/av_v4_ks_v1"' in source
    assert 'run_experiment "v5" 5 "runs/av_v4_ks_v5"' in source
    assert "--visual-loss-weight" in source
    assert "--audio-loss-weight" not in source
    assert "--epochs" not in source
    assert "--batch-size" not in source
    assert "--lr " not in source
    assert 'tee "${output_dir}/train.log"' in source
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
pytest tests/test_run_av_v4_ks_loss_sweep_script.py -q
```

Expected: FAIL with `FileNotFoundError` because the KS script does not exist.

### Task 2: Implement and verify the launcher

**Files:**
- Create: `scripts/run_av_v4_ks_loss_sweep.sh`
- Test: `tests/test_run_av_v4_ks_loss_sweep_script.py`

- [ ] **Step 1: Add the minimal Bash implementation**

```bash
#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
cd "${project_root}"

python_bin="${PYTHON_BIN:-python}"

run_experiment() {
    local name="$1"
    local visual_weight="$2"
    local output_dir="$3"

    mkdir -p "${output_dir}"
    echo "Starting ${name}: visual_loss_weight=${visual_weight}"

    "${python_bin}" -u -m AV_v4.train_ks \
        --visual-loss-weight "${visual_weight}" \
        --output-dir "${output_dir}" \
        2>&1 | tee "${output_dir}/train.log"
}

run_experiment "v1" 1 "runs/av_v4_ks_v1"
run_experiment "v5" 5 "runs/av_v4_ks_v5"

echo "All AV_v4 KineticSound loss-weight experiments finished."
```

- [ ] **Step 2: Run the focused test and verify GREEN**

Run:

```powershell
pytest tests/test_run_av_v4_ks_loss_sweep_script.py -q
```

Expected: `1 passed`.

- [ ] **Step 3: Check Bash syntax**

Run:

```powershell
bash -n scripts/run_av_v4_ks_loss_sweep.sh
```

Expected: exit code 0 and no output.

- [ ] **Step 4: Run related script tests**

Run:

```powershell
pytest tests/test_run_av_v4_ks_loss_sweep_script.py tests/test_run_av_v4_ave_loss_sweep_script.py -q
```

Expected: all tests pass.
