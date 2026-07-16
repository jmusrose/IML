from data_processing.toydata.plot_synthetic_behavior import (
    derive_series,
    make_synthetic_records,
    write_outputs,
)


def test_synthetic_records_encode_the_single_figure_story():
    records = make_synthetic_records()
    series = derive_series(records)

    assert len(records) == 11
    assert series["s"] == sorted(series["s"])
    assert series["true_cmi"][0] < series["true_cmi"][-1]
    assert max(
        abs(strength - cmi)
        for strength, cmi in zip(series["normalized_strength"], series["true_cmi"])
    ) <= 0.03
    assert all(
        error_b > error_a
        for error_a, error_b in zip(series["error_a"], series["error_b"])
    )
    assert max(series["direction_imbalance"]) < 0.04


def test_write_outputs_creates_data_and_one_figure(tmp_path):
    outputs = write_outputs(tmp_path / "synthetic_behavior_single")

    assert {path.suffix for path in outputs} == {".json", ".png", ".svg", ".pdf"}
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)
