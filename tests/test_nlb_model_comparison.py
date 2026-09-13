from argparse import Namespace
from copy import deepcopy
import csv
import json
import os
import pickle
import random
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from ladys.config import ExperimentConfig, load_experiment_config
from ladys.datasets import NLBDatasetConfig
from ladys.models.mint import MINTConfig
from ladys.models.ilqr_vae import ILQRVAEConfig
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


def _gradient_run(tmp_path, monkeypatch):
    rng = np.random.default_rng(12)
    with h5py.File(tmp_path / "area2_bump_val_5ms.h5", "w") as handle:
        for split, trials in (("train", 3), ("eval", 2)):
            for partition, neurons in (("heldin", 3), ("heldout", 2)):
                handle[f"{split}_spikes_{partition}"] = rng.poisson(
                    0.4, (trials, 4, neurons)).astype("float32")
    optimization = load_experiment_config(comparison.config_path("area2_bump", "ilqr_vae")).model.optimization
    config = ExperimentConfig(
        dataset=NLBDatasetConfig(name="area2_bump", bin_size_ms=5),
        model=ILQRVAEConfig(latent_dim=4, input_dim=2, max_iter=1, n_posterior_samples=1,
                           optimization=optimization),
        trainer=TrainerConfig(epochs=3), batch_size=2, preprocessing=PreprocessingConfig(),
    )
    monkeypatch.setattr(comparison, "load_experiment_config", lambda _: deepcopy(config))
    return Namespace(worker=("ilqr_vae", "area2_bump"), output_dir=tmp_path / "runs",
                     data_root=tmp_path, device="cpu", seed=2, threads=1,
                     epochs=3, batch_size=None, eval_every=1, resume=False)


@pytest.mark.parametrize("interrupted_function", ["evaluate_model", "_write_predictions"])
def test_resume_restores_optimizer_rng_and_pending_evaluation(tmp_path, monkeypatch, interrupted_function):
    args = _gradient_run(tmp_path, monkeypatch)
    make_strategy = comparison.build_strategy

    def consuming_setup(config):
        strategy = make_strategy(config)
        setup = strategy.setup

        def setup_with_rng(model):
            setup(model)
            torch.rand(1)
            np.random.random()
            random.random()

        strategy.setup = setup_with_rng
        return strategy

    monkeypatch.setattr(comparison, "build_strategy", consuming_setup)
    args.output_dir = tmp_path / "uninterrupted"
    assert comparison.run_one(args) == 0
    reference = args.output_dir / "area2_bump" / "ilqr_vae"

    original = getattr(comparison, interrupted_function)
    monkeypatch.setattr(comparison, interrupted_function, lambda *a, **k: (_ for _ in ()).throw(RuntimeError("interrupted evaluation")))
    args.output_dir = tmp_path / "interrupted"
    assert comparison.run_one(args) == 1
    resumed = args.output_dir / "area2_bump" / "ilqr_vae"
    payload = json.loads((resumed / "config.json").read_text())
    assert comparison.load_resume_checkpoint(resumed, payload)["epoch"] == 1
    assert not (resumed / "best_metrics.json").exists()
    monkeypatch.setattr(comparison, interrupted_function, original)
    monkeypatch.setattr(comparison.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout="python -m pytest tests/test_nlb_model_comparison.py"))
    args.resume = True
    assert comparison.run_one(args) == 0

    expected = comparison.load_resume_checkpoint(reference, payload)
    actual = comparison.load_resume_checkpoint(resumed, payload)
    assert actual["epoch"] == expected["epoch"] == 3
    for key in expected["model"]:
        torch.testing.assert_close(actual["model"][key], expected["model"][key], rtol=0, atol=0)
    torch.testing.assert_close(actual["rng_state"], expected["rng_state"], rtol=0, atol=0)
    assert actual["numpy_rng_state"] == expected["numpy_rng_state"]
    assert actual["python_rng_state"] == expected["python_rng_state"]
    assert actual["strategy"]["scheduler"] == expected["strategy"]["scheduler"]
    assert actual["strategy"]["scheduler"]["last_epoch"] == 6
    actual_lr = actual["strategy"]["optimizer"]["param_groups"][0]["lr"]
    assert isinstance(actual_lr, np.float64)
    assert actual_lr == expected["strategy"]["optimizer"]["param_groups"][0]["lr"]
    with (resumed / "history.csv").open() as handle:
        assert [int(row["epoch"]) for row in csv.DictReader(handle)] == [1, 2, 3]
    rows = json.loads((resumed / "evaluations.json").read_text())
    assert [row["epoch"] for row in rows] == [1, 2, 3]
    assert [row["co_bps"] for row in rows] == pytest.approx(
        [row["co_bps"] for row in json.loads((reference / "evaluations.json").read_text())], abs=1e-12)


@pytest.mark.parametrize("section,key,value", [
    ("model", "latent_dim", 8), ("trainer", "batch_size", 3),
    ("dataset", "seed", 99), ("dataset", "data_path", "/different/data.h5"),
])
def test_resume_rejects_changed_training_configuration(section, key, value):
    saved = {"model": {"latent_dim": 4}, "trainer": {"batch_size": 2, "device": "cuda", "epochs": 10},
             "dataset": {"seed": 0, "data_path": "/data.h5"}, "experiment": {"output_dir": "/old"}}
    allowed = deepcopy(saved)
    allowed["trainer"].update(device="cpu", epochs=25)
    allowed["experiment"]["output_dir"] = "/new"
    comparison.validate_resume_config(saved, allowed)
    allowed[section][key] = value
    with pytest.raises(ValueError, match="configuration differs"):
        comparison.validate_resume_config(saved, allowed)


def test_legacy_checkpoint_pair_and_empty_strategy_rejection(tmp_path):
    payload = {"trainer": {"epochs": 3}, "model": {"name": "ilqr_vae"}}
    (tmp_path / "config.json").write_text(json.dumps(payload))
    comparison.save_checkpoint(tmp_path / "latest_model.pt", {"weight": torch.tensor([2.0])})
    saved = dict(epoch=1, strategy={"optimizer": {"state": {}, "param_groups": [{"lr": np.float64(0.0005)}]}},
                 rng_state=torch.get_rng_state())
    comparison.save_checkpoint(tmp_path / "training_state.pt", saved)
    safe_globals = set(torch.serialization.get_safe_globals())
    with pytest.raises(pickle.UnpicklingError, match="Unsupported global"):
        torch.load(tmp_path / "training_state.pt", weights_only=True)
    loaded = comparison.load_resume_checkpoint(tmp_path, payload)
    torch.testing.assert_close(loaded["model"]["weight"], torch.tensor([2.0]))
    assert loaded["strategy"]["optimizer"]["param_groups"][0]["lr"] == np.float64(0.0005)
    assert set(torch.serialization.get_safe_globals()) == safe_globals
    saved["strategy"] = {}
    comparison.save_checkpoint(tmp_path / "training_state.pt", saved)
    with pytest.raises(ValueError, match="empty strategy state"):
        comparison.load_resume_checkpoint(tmp_path, payload)


def test_default_worker_refuses_to_overwrite_existing_run(tmp_path):
    folder = tmp_path / "area2_bump" / "ilqr_vae"
    folder.mkdir(parents=True)
    original = json.dumps({"status": "complete", "epoch": 3})
    (folder / "status.json").write_text(original)
    args = Namespace(worker=("ilqr_vae", "area2_bump"), output_dir=tmp_path,
                     device="cpu", seed=0, resume=False)
    assert comparison.run_one(args) == 1
    assert (folder / "status.json").read_text() == original


def test_resume_parent_skips_completed_and_rejects_active_worker(tmp_path, monkeypatch):
    folder = tmp_path / "runs" / "area2_bump" / "ilqr_vae"
    folder.mkdir(parents=True)
    comparison.write_status(folder / "status.json", {"dataset": "area2_bump", "model": "ilqr_vae", "status": "complete"})
    arguments = ["--data-root", str(tmp_path), "--output-dir", str(tmp_path / "runs"),
                 "--models", "ilqr_vae", "--datasets", "area2_bump", "--resume", "--device", "cpu"]
    monkeypatch.setattr(comparison.subprocess, "Popen", lambda *a, **k: pytest.fail("Unexpected worker launch"))
    assert comparison.main(arguments) == 0
    assert not (tmp_path / "runs" / "summary.tmp").exists()
    comparison.write_status(folder / "status.json", {"status": "running", "pid": os.getpid()})
    monkeypatch.setattr(comparison.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout="python /repo/scripts/run_nlb_model_comparison.py --worker ilqr_vae area2_bump"))
    with pytest.raises(RuntimeError, match="still active"):
        comparison.main(arguments)
