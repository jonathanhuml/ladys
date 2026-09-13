import json

from scripts.report_synthetic_readiness import replace_runs


def test_replacing_diagnostic_preserves_original_and_unrelated_methods(tmp_path):
    root, replacement = tmp_path / "original", tmp_path / "corrected"
    old = root / "lorenz" / "models" / "langevin_flow"
    new = replacement / "lorenz" / "models" / "langevin_flow"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    (old / "history.csv").write_text("old results\n")
    (new / "history.csv").write_text("corrected results\n")
    unchanged = dict(dataset="lorenz", model="mint", status="completed", seconds=2)
    old_status = dict(dataset="lorenz", model="langevin_flow", status="completed", seconds=10)
    new_status = dict(old_status, seconds=11)
    (root / "status.json").write_text(json.dumps([unchanged, old_status]))
    (replacement / "status.json").write_text(json.dumps([new_status]))
    replace_runs(root, replacement)
    assert (old / "history.csv").read_text() == "corrected results\n"
    archived = root / "superseded" / "lorenz" / "langevin_flow" / "history.csv"
    assert archived.read_text() == "old results\n"
    status = json.loads((root / "status.json").read_text())
    assert status[0] == unchanged
    assert status[1]["seconds"] == 11
    assert json.loads((root / "superseded" / "status.json").read_text())[1] == old_status
