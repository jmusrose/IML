import math

import pytest

from data_processing.toy_figure import plot_signal_fidelity_comparison as plot_module
from data_processing.toy_figure.plot_signal_fidelity_comparison import prepare_pairs


def record(mode: str, true_cmi: float, delta_audio_tail: float, *, s: float = 0.5) -> dict:
    return {
        "mode": mode,
        "s": s,
        "eta": 0.1,
        "seed": 0,
        "true_cmi": true_cmi,
        "delta_audio_tail": delta_audio_tail,
        "final_val": {"joint_acc": 999.0, "audio_acc": 999.0, "visual_acc": 999.0},
    }


def test_prepare_pairs_converts_nats_to_bits_and_ignores_accuracy():
    pairs, upper = prepare_pairs(
        [record("fgm", 0.5, math.log(2.0))],
        [record("no_fgm", 0.5, 0.5 * math.log(2.0))],
    )

    assert len(pairs) == 1
    assert pairs[0][0] == 0.5
    assert math.isclose(pairs[0][1], 1.0)
    assert math.isclose(pairs[0][2], 0.5)
    assert math.isclose(upper, 1.0)


def test_prepare_pairs_sorts_by_true_cmi():
    pairs, _ = prepare_pairs(
        [record("fgm", 0.8, 0.2), record("fgm", 0.2, 0.1)],
        [record("no_fgm", 0.2, 0.3), record("no_fgm", 0.8, 0.4)],
    )

    assert [pair[0] for pair in pairs] == [0.2, 0.8]


def test_prepare_pairs_rejects_missing_conditions():
    with pytest.raises(ValueError, match="condition keys differ"):
        prepare_pairs([record("fgm", 0.5, 0.1)], [])


def test_prepare_pairs_rejects_duplicate_conditions():
    duplicate = record("fgm", 0.5, 0.1)

    with pytest.raises(ValueError, match="duplicate"):
        prepare_pairs(
            [duplicate, duplicate.copy()],
            [record("no_fgm", 0.5, 0.1)],
        )


def test_series_styles_use_hollow_blue_and_pink_circles():
    assert plot_module.FGM_COLOR == "#2F6DB3"
    assert plot_module.NO_FGM_COLOR == "#C44E62"
    assert plot_module.GRID_COLOR == "#D9DCE3"
    assert plot_module.CONNECTOR_COLOR == plot_module.GRID_COLOR
    assert plot_module.REFERENCE_LINEWIDTH == 1.2
    assert plot_module.PANEL_LABEL == "a"
    assert plot_module.FGM_MARKER == plot_module.NO_FGM_MARKER == "o"
    assert plot_module.MARKER_FACE == "white"
    assert plot_module.MARKER_SIZE == 24
    assert plot_module.MARKER_ALPHA == 0.82
