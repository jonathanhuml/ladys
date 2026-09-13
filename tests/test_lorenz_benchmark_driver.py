import csv
import json

import numpy as np
import pytest
import torch
from torch.utils.data import SequentialSampler

from ladys.datasets import ChaoticRNNDatasetConfig, LorenzDatasetConfig
from ladys.models import BGPFAConfig, ILQRVAEConfig, LFADSConfig, MINTConfig
from ladys.preprocessing import PreprocessedDataset, PreprocessingConfig
from scripts import benchmark_lorenz_loss_curves as benchmark


def _args(dataset, tmp_path):
    return benchmark.parse_args([
        "--dataset", dataset, "--neurons", "3", "--num-inits", "2",
        "--num-trials", "3", "--num-steps", "8", "--burn-steps", "4",
        "--epochs", "3", "--batch-size", "2", "--device", "cpu",
        "--num-rate-traces", "1", "--ilqr-max-iter", "1",
        "--bgpfa-infer-steps", "1", "--bgpfa-infer-mc", "1",
        "--preprocessing-mode", "none", "--output-dir", str(tmp_path),
    ])


@pytest.mark.parametrize("dataset", ["lorenz", "chaotic_rnn"])
@pytest.mark.parametrize("source", ["smoothed_spikes", "true_rates"])
def test_incremental_mint_matches_dataset_aware_fit(dataset, source):
    config = (LorenzDatasetConfig(neurons=3, num_inits=2, num_trials=3, num_steps=8,
                                  burn_steps=4, spike_bin_size=0.25)
              if dataset == "lorenz" else
              ChaoticRNNDatasetConfig(neurons=3, hidden_units=4, num_conditions=2,
                                      num_trials=3, num_steps=8, dt=0.01))
    training, _ = benchmark.make_synthetic_splits(config)
    wrapped = PreprocessedDataset(training, PreprocessingConfig())
    model_config = MINTConfig(dataset=dataset, sigma=0, delta=2, window_length=4,
                              interp=0, lorenz_library_source=source)
    incremental = model_config.build(3, 8)
    benchmark.fit_mint_lorenz_library(incremental, wrapped, config, "cpu")
    standard = model_config.build(3, 8)
    standard.fit_training_data(benchmark.DataLoader(wrapped, batch_size=2), device="cpu")
    assert incremental.Ts == pytest.approx(training.arrays.dt)
    assert incremental.settings.library_rate_source == ("prepared_rates" if source == "true_rates" else "prepared_spikes")
    for actual, expected in zip(incremental.Omega_plus, standard.Omega_plus):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dataset", ["lorenz", "chaotic_rnn"])
@pytest.mark.parametrize("method", ["psth", "smoothing", "mint"])
def test_static_and_library_runs_persist_progress(tmp_path, monkeypatch, dataset, method):
    args = _args(dataset, tmp_path)
    if method == "mint":
        args.mint_window_length = 2
    monkeypatch.setattr(benchmark, "plot_rate_traces", lambda *a, **k: None)
    rows, traces = benchmark.run_case(args, method, tmp_path)
    assert rows and all(row["status"] == "ok" for row in rows)
    assert all(np.isfinite(row["test_rate_mse"]) for row in rows)
    assert traces
    if method in benchmark.STATIC_MODELS:
        assert len(rows) == 1 and rows[0]["epoch"] == 0
        assert rows[0]["training_axis"] == "static"
    else:
        assert [row["epoch"] for row in rows] == [1, 2]
        assert rows[0]["training_axis"] == "trials_per_condition"
    folder = tmp_path / "models" / method
    with (folder / "history.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == len(rows)
    assert (folder / "model.pt").exists()
    assert not list(folder.glob("*.tmp"))


@pytest.mark.parametrize("dataset", ["lorenz", "chaotic_rnn"])
@pytest.mark.parametrize("method", ["ilqr_vae", "lfads"])
def test_gradient_run_uses_dataset_dt_and_saves_before_time_limit(tmp_path, monkeypatch, dataset, method):
    args = _args(dataset, tmp_path)
    args.max_seconds = 1e-12
    config = (ILQRVAEConfig(latent_dim=4, input_dim=2, max_iter=1) if method == "ilqr_vae" else
              LFADSConfig(generator_dim=4, factor_dim=2, g0_encoder_dim=4,
                          controller_encoder_dim=4, controller_dim=4))
    monkeypatch.setattr(benchmark, "build_model_config", lambda *a: config)
    rows, traces = benchmark.run_case(args, method, tmp_path)
    assert len(rows) == 1 and rows[0]["status"] == "ok"
    assert rows[0]["epoch"] == 1
    assert np.isfinite(rows[0]["test_rate_mse"])
    assert traces == []
    folder = tmp_path / "models" / method
    saved = json.loads((folder / "config.json").read_text())
    expected_dt = 1 if dataset == "lorenz" else 0.01
    assert saved["model"]["dt"] == pytest.approx(expected_dt)
    assert json.loads((folder / "metrics.json").read_text())["run_status"] == "time_budget"
    assert (folder / "model.pt").exists()


@pytest.mark.parametrize("dataset", ["lorenz", "chaotic_rnn"])
def test_bgpfa_runner_preserves_training_trial_order(tmp_path, monkeypatch, dataset):
    args = _args(dataset, tmp_path)
    args.max_seconds = 1e-12
    config = BGPFAConfig(latent_dim=2, n_mc_train=1, n_mc_eval=1,
                         optimization={"name": "mgplvm_full_batch_gradient", "n_mc": 1})
    monkeypatch.setattr(benchmark, "build_model_config", lambda *a: config)
    fit = benchmark.Trainer.fit

    def checked_fit(self, model, strategy, loader, *a, **k):
        assert isinstance(loader.sampler, SequentialSampler)
        return fit(self, model, strategy, loader, *a, **k)

    monkeypatch.setattr(benchmark.Trainer, "fit", checked_fit)
    rows, _ = benchmark.run_case(args, "bgpfa", tmp_path)
    assert len(rows) == 1 and rows[0]["status"] == "ok"
    assert np.isfinite(rows[0]["test_rate_mse"])


@pytest.mark.parametrize("plot", [benchmark.plot_test_objective, benchmark.plot_train_test_objective])
def test_objective_grid_has_three_columns_and_correct_axes(tmp_path, monkeypatch, plot):
    figures = []
    monkeypatch.setattr(benchmark, "save_figure", lambda figure, path: figures.append(figure))
    rows = [dict(status="ok", model=model, epoch=1, train_loss=1, test_loss=2, objective="NLL")
            for model in ("mint", "psth", "smoothing", "ilqr_vae")]
    plot(rows, tmp_path / "plot.png")
    assert figures[0].axes[0].get_subplotspec().get_gridspec().ncols == 3
    labels = {axis.get_xlabel() for axis in figures[0].axes}
    assert {"Epoch", "Static Baseline", "Training Trials per Condition"} <= labels
