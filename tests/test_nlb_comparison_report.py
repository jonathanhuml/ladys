import json

import numpy as np
import pytest

pytest.importorskip("nlb_tools")

from ladys.nlb_eval import nlb_bits_per_spike
from scripts.report_nlb_model_comparison import collect


def make_run(root, rates):
    folder = root / "area2_bump" / "mint"
    folder.mkdir(parents=True)
    spikes = np.array([[[1.0], [0.0], [np.nan]], [[0.0], [1.0], [0.0]]])
    np.savez(folder / "predictions.npz", pred_rates=rates, target_spikes=spikes)
    state = dict(dataset="area2_bump", model="mint", status="complete", epoch=0,
                 epochs=0, best_epoch=0, bin_size_ms=5,
                 best_co_bps=nlb_bits_per_spike(rates, spikes))
    (folder / "status.json").write_text(json.dumps(state))
    return folder, state


def test_report_verifies_official_score_and_selects_by_explicit_directory_order(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    make_run(first, np.full((2, 3, 1), 0.4))
    folder, state = make_run(second, np.full((2, 3, 1), 0.2))
    output = tmp_path / "report"
    rows = collect([first, second], output)
    assert len(rows) == 1
    assert rows[0]["source"] == str(folder)
    assert rows[0]["co_bps"] == pytest.approx(state["best_co_bps"])
    assert rows[0]["official_co_bps"] == pytest.approx(state["best_co_bps"])
    assert rows[0]["double_dt_co_bps"] < rows[0]["co_bps"] - 5
    assert (output / "area2_bump" / "mint" / "predictions.npz").exists()
    assert (output / "results.csv").exists()


def test_report_rejects_stale_or_inconsistent_prediction_metadata(tmp_path):
    root = tmp_path / "run"
    folder, state = make_run(root, np.full((2, 3, 1), 0.4))
    state["best_co_bps"] += 0.1
    (folder / "status.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="Prediction/status mismatch"):
        collect([root], tmp_path / "report")
