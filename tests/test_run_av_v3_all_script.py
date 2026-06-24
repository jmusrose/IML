import importlib.util
from pathlib import Path
from unittest.mock import patch


def _load_script_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "run_av_v3_all.py"
    spec = importlib.util.spec_from_file_location("run_av_v3_all", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_av_v3_all_runs_three_training_modules_in_order():
    module = _load_script_module()

    with patch.object(module.subprocess, "run") as run:
        run.return_value.returncode = 0
        exit_code = module.main([])

    assert exit_code == 0
    assert [call.args[0][2] for call in run.call_args_list] == [
        "AV_v3.train_cremad",
        "AV_v3.train_ks",
        "AV_v3.train_ave",
    ]


def test_run_av_v3_all_rejects_arguments():
    module = _load_script_module()

    with patch.object(module.subprocess, "run") as run:
        exit_code = module.main(["--device", "cuda"])

    assert exit_code == 2
    run.assert_not_called()
