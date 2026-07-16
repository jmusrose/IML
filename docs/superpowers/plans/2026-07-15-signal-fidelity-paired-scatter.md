# Paired Signal-Fidelity Scatter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, publication-ready scatter plot that overlays paired FGM and perturbed No-FGM signal-fidelity records.

**Architecture:** A single Python module loads both JSON files, validates one-to-one experimental-condition pairing, converts `delta_audio_tail` from nats to bits, and renders all vector/raster outputs from one matplotlib figure. A focused pytest module tests the data transformation and pairing independently of visual styling.

**Tech Stack:** Python from conda environment `pytorch2.5`, standard-library `json`/`math`/`pathlib`, matplotlib 3.10, pytest.

## Global Constraints

- Use only `true_cmi` for x and `delta_audio_tail / 0.6931471805599453` for y.
- Pair records by `(true_cmi, s, eta, seed)` and reject mismatches or duplicates.
- Use Python/matplotlib exclusively for drawing, exporting, and visual QA.
- Export editable SVG, PDF, and 300-dpi PNG from one figure.
- Keep x and y limits identical and draw `y=x` from `(0, 0)` to the shared data maximum.

---

### Task 1: Validate, render, and verify the paired scatter

**Files:**
- Create: `data_processing/toy_figure/plot_signal_fidelity_comparison.py`
- Create: `tests/test_toy_figure_plot.py`
- Generate: `data_processing/toy_figure/signal_fidelity_comparison.svg`
- Generate: `data_processing/toy_figure/signal_fidelity_comparison.pdf`
- Generate: `data_processing/toy_figure/signal_fidelity_comparison.png`

**Interfaces:**
- Consumes: two JSON objects containing `records` lists.
- Produces: `condition_key(record)`, `prepare_pairs(fgm_records, no_fgm_records)`, `plot_comparison(...)`, and three figure files.

- [x] **Step 1: Write failing tests**

```python
def test_prepare_pairs_converts_nats_to_bits_and_ignores_accuracy():
    pairs, upper = prepare_pairs([record("fgm", 0.5, math.log(2))], [record("no_fgm", 0.5, 0.5 * math.log(2))])
    assert pairs == [(0.5, 1.0, 0.5)]
    assert upper == 1.0

def test_prepare_pairs_rejects_missing_or_duplicate_conditions():
    with pytest.raises(ValueError, match="condition keys differ"):
        prepare_pairs([record("fgm", 0.5, 0.1)], [])
    duplicate = record("fgm", 0.5, 0.1)
    with pytest.raises(ValueError, match="duplicate"):
        prepare_pairs([duplicate, duplicate], [record("no_fgm", 0.5, 0.1)])
```

- [x] **Step 2: Verify RED**

Run: `conda run -n pytorch2.5 python -m pytest tests/test_toy_figure_plot.py -q`

Expected: collection fails because `data_processing.toy_figure.plot_signal_fidelity_comparison` does not exist.

- [x] **Step 3: Implement the minimum transformation and plotter**

```python
LN2 = 0.6931471805599453

def condition_key(record):
    return tuple(record[name] for name in ("true_cmi", "s", "eta", "seed"))

def prepare_pairs(fgm_records, no_fgm_records):
    fgm = _index_unique(fgm_records)
    no_fgm = _index_unique(no_fgm_records)
    if fgm.keys() != no_fgm.keys():
        raise ValueError("FGM and No-FGM condition keys differ")
    pairs = [(float(key[0]), float(fgm[key]["delta_audio_tail"]) / LN2,
              float(no_fgm[key]["delta_audio_tail"]) / LN2)
             for key in sorted(fgm)]
    return pairs, max(value for pair in pairs for value in pair)
```

Render grey paired segments first, then bright-blue and rose-pink open circles. Draw the dashed reference line only to `upper`, apply identical padded limits and equal aspect, and save SVG/PDF/PNG.

- [x] **Step 4: Verify GREEN**

Run: `conda run -n pytorch2.5 python -m pytest tests/test_toy_figure_plot.py -q`

Expected: all tests pass.

- [x] **Step 5: Generate outputs**

Run: `conda run -n pytorch2.5 python data_processing/toy_figure/plot_signal_fidelity_comparison.py`

Expected: the three `signal_fidelity_comparison` outputs are written beside the JSON inputs.

- [x] **Step 6: Full verification**

Run: `conda run -n pytorch2.5 python -m pytest tests/test_toy_figure_plot.py tests/test_fgm_toy.py -q`

Expected: zero failures. Then inspect the PNG and confirm 32 connector segments, two 32-point series, exact axis labels, equal limits, and unclipped legend/title.
