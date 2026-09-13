from argparse import Namespace
import json

import h5py
import numpy as np
import pytest

from ladys.config import ExperimentConfig, load_experiment_config
from ladys.datasets import NLBDatasetConfig
from ladys.models.mint import MINTConfig
from ladys.nlb_eval import score_ladys_predictions
from ladys.preprocessing import PreprocessingConfig
from ladys.training import TrainerConfig
from scripts import run_nlb_model_comparison as comparison


@pytest.mark.parametrize("method", comparison.METHODS)
@pytest.mark.parametrize("dataset", comparison.DATASETS)
def test_comparison_selects_native_full_validation_recipe(method, dataset):
    config = load_experiment_config(comparison.config_path(dataset, method))
    assert config.dataset.bin_size_ms == 5
    assert config.dataset.split == "val"
    assert config.dataset.max_trials is None
    assert config.model.name == method


def test_full_library_fit_saves_independently_reproducible_score(tmp_path, monkeypatch):
    path = tmp_path / "area2_bump_val_5ms.h5"
    rng = np.random.default_rng(4)
    with h5py.File(path, "w") as handle:
        for split, trials in (("train", 5), ("eval", 3)):
            for partition, neurons in (("heldin", 3), ("heldout", 2)):
                handle[f"{split}_spikes_{partition}"] = rng.poisson(
                    0.3, (trials, 12, neurons)).astype("float32")
    config = ExperimentConfig(
        dataset=NLBDatasetConfig(name="area2_bump", max_trials=1, bin_size_ms=5),
        model=MINTConfig(dataset="area2_bump", train_source="h5", sigma=1,
                         delta=2, window_length=4, interp=0),
        trainer=TrainerConfig(epochs=0), batch_size=2, preprocessing=PreprocessingConfig(),
    )
    monkeypatch.setattr(comparison, "load_experiment_config", lambda _: config)
    args = Namespace(worker=("mint", "area2_bump"), output_dir=tmp_path / "runs",
                     data_root=tmp_path, device="cpu", seed=0, threads=1,
                     epochs=None, batch_size=None, eval_every=None)
    assert comparison.run_one(args) == 0
    folder = args.output_dir / "area2_bump" / "mint"
    status = json.loads((folder / "status.json").read_text())
    assert status["status"] == "complete"
    assert status["train_trials"] == 5
    assert status["val_trials"] == 3
    assert status["time_bins"] == 12
    assert json.loads((folder / "library_training.json").read_text())["trials"] == 5
    assert status["best_co_bps"] == pytest.approx(
        score_ladys_predictions(folder / "predictions.npz").co_bps)
    reloaded = load_experiment_config(folder / "config.json")
    assert reloaded.dataset.max_trials is None
    assert reloaded.batch_size == 2
    rows = comparison.write_summary(args.output_dir, ["area2_bump"], ["mint", "bgpfa"])
    assert [row["status"] for row in rows] == ["complete", "queued"]
    assert not (folder / "status.tmp").exists()


def test_worker_records_setup_failure(tmp_path, monkeypatch):
    def fail(_):
        raise ValueError("broken recipe")

    monkeypatch.setattr(comparison, "load_experiment_config", fail)
    args = Namespace(worker=("mint", "area2_bump"), output_dir=tmp_path,
                     device="cpu", seed=0)
    assert comparison.run_one(args) == 1
    folder = tmp_path / "area2_bump" / "mint"
    status = json.loads((folder / "status.json").read_text())
    assert status["status"] == "failed"
    assert "broken recipe" in (folder / "error.txt").read_text()
