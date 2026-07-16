import math

import pytest

from data_processing.toydata.plot_behavior_single import (
    BAND_ALPHA,
    DEFAULT_OUTPUT,
    FGM_LINESTYLE,
    GREEN,
    GRID_COLOR,
    LINE_WIDTH,
    PURPLE,
    REFERENCE_COLOR,
    ROSE,
    derive_series,
    plot_summary,
)


def sample_summary() -> dict[str, list[dict]]:
    return {
        "fgm": [
            {
                "s": 0.0,
                "true_cmi_mean": 0.0,
                "true_cmi_std": 0.0,
                "probe_acc_audio_mean": 0.90,
                "probe_acc_audio_std": 0.02,
                "probe_acc_visual_mean": 0.80,
                "probe_acc_visual_std": 0.03,
                "r_audio_tail_mean": 0.51,
                "r_audio_tail_std": 0.01,
                "r_visual_tail_mean": 0.49,
                "r_visual_tail_std": 0.01,
                "s_hat_tail_mean": 0.2 * 2.0 * math.log(2.0),
                "s_hat_tail_std": 0.02 * 2.0 * math.log(2.0),
            },
            {
                "s": 1.0,
                "true_cmi_mean": 1.0,
                "true_cmi_std": 0.0,
                "probe_acc_audio_mean": 0.60,
                "probe_acc_audio_std": 0.02,
                "probe_acc_visual_mean": 0.50,
                "probe_acc_visual_std": 0.03,
                "r_audio_tail_mean": 0.50,
                "r_audio_tail_std": 0.01,
                "r_visual_tail_mean": 0.50,
                "r_visual_tail_std": 0.01,
                "s_hat_tail_mean": 1.0 * 2.0 * math.log(2.0),
                "s_hat_tail_std": 0.02 * 2.0 * math.log(2.0),
            },
        ]
    }


def test_derive_series_uses_measured_means_and_standard_deviations():
    series = derive_series(sample_summary())

    assert series["normalized_strength"] == pytest.approx([0.2, 1.0])
    assert series["normalized_strength_std"] == pytest.approx([0.02, 0.02])
    assert series["mean_error"] == pytest.approx([0.15, 0.45])
    assert series["error_low"] == pytest.approx([0.08, 0.38])
    assert series["error_high"] == pytest.approx([0.23, 0.53])
    assert math.isclose(series["direction_imbalance"][0], 0.02)
    assert series["direction_imbalance_std"] == pytest.approx([0.02, 0.02])


def test_plot_summary_writes_one_figure_in_three_formats(tmp_path):
    outputs = plot_summary(sample_summary(), tmp_path / "behavior_single")

    assert {path.suffix for path in outputs} == {".png", ".svg", ".pdf"}
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)


def test_reference_line_and_unconnected_fgm_points_avoid_visual_overlap():
    assert (PURPLE, ROSE, GREEN) == ("#8F6BB3", "#E07A5F", "#3D9970")
    assert REFERENCE_COLOR == "#A7ADB7"
    assert FGM_LINESTYLE == "none"
    assert GRID_COLOR == "#DDE2E8"
    assert LINE_WIDTH == 1.7
    assert 0.0 < BAND_ALPHA < 0.25
    assert DEFAULT_OUTPUT.name == "behavior_single_reference_points"
