import json
import math
from pathlib import Path


def test_perturb_no_fgm_adds_bounded_bit_noise_and_plots(tmp_path):
    from data_processing.toydata.perturb_no_fgm import perturb_file

    source = tmp_path / "signal_fidelity_no_fgm.json"
    output_json = tmp_path / "perturbed.json"
    output_png = tmp_path / "perturbed.png"
    source.write_text(
        json.dumps(
            {
                "records": [
                    {"true_cmi": 0.2, "delta_audio_tail": math.log(2) * 0.1, "mode": "no_fgm"},
                    {"true_cmi": 0.4, "delta_audio_tail": math.log(2) * 0.3, "mode": "no_fgm"},
                ]
            }
        ),
        encoding="utf-8",
    )

    perturb_file(source, output_json, output_png, seed=0)

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert output_png.exists()
    assert len(payload["records"]) == 2
    for before, after in zip(json.loads(source.read_text(encoding="utf-8"))["records"], payload["records"]):
        added_bits = (after["delta_audio_tail"] - before["delta_audio_tail"]) / math.log(2)
        assert 0.0 <= added_bits <= 0.05
        assert math.isclose(after["noise_bits"], added_bits)
