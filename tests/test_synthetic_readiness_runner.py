import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import benchmark_lorenz_loss_curves as benchmark
from scripts import run_synthetic_readiness as runner


@pytest.mark.parametrize("outcome,expected", [
    ("complete", "completed"), ("hard_timeout", "time_limit"),
    ("soft_timeout", "time_limit"), ("nonfinite", "failed"),
])
def test_runner_retains_partial_history_and_reports_real_status(tmp_path, monkeypatch, outcome, expected):
    folder = tmp_path / "lorenz" / "models" / "smoothing"
    folder.mkdir(parents=True)
    previous = dict(status="ok", model="smoothing", epoch=0, test_rate_mse=2.)
    benchmark.write_history(folder / "history.csv", [previous])
    monkeypatch.setattr(sys, "argv", ["run_synthetic_readiness.py", "--output-dir", str(tmp_path),
                                     "--datasets", "lorenz", "--models", "ndt"])

    def run(command, **kwargs):
        assert kwargs["timeout"] == 600
        current = tmp_path / "lorenz" / "models" / "ndt"
        current.mkdir()
        row = dict(status="ok", model="ndt", epoch=1, train_loss=1.,
                   test_rate_mse=float("nan") if outcome == "nonfinite" else 0.5)
        benchmark.write_history(current / "history.csv", [row])
        if outcome == "hard_timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if outcome == "soft_timeout":
            (current / "metrics.json").write_text(json.dumps({"run_status": "time_budget"}))
        return SimpleNamespace(returncode=0)

    groups = []
    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(benchmark, "write_group_outputs", lambda _, rows, traces: groups.extend(rows))
    runner.main()
    status = json.loads((tmp_path / "status.json").read_text())[0]
    assert status["status"] == expected
    assert status["completed_points"] == 1
    assert {row["model"] for row in groups} == {"ndt", "smoothing"}
