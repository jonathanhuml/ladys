from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from ladys.config import ExperimentConfig
from ladys.datasets import LorenzDataset, LorenzDatasetConfig, NLBDatasetConfig
from ladys.experiment import Experiment
from ladys.metrics import evaluate_model
from ladys.models.mint import MINTConfig, InterpOptions, fit_poisson_interp
from ladys.preprocessing import PreprocessedDataset, PreprocessingConfig
from torch.utils.data import DataLoader, Subset
from ladys.training import TrainerConfig
from ladys.training.strategies import build_strategy


CUDA_DEVICE = pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
)


def _nlb_experiment(tmp_path, name="mc_maze", source="h5", device="cpu"):
    path = tmp_path / "prepared.h5"
    rng = np.random.default_rng(4)
    with h5py.File(path, "w") as handle:
        for split, n_trials in (("train", 4), ("eval", 2)):
            handle.create_dataset(f"{split}_spikes_heldin", data=rng.poisson(0.3, (n_trials, 12, 3)).astype("float32"))
            handle.create_dataset(f"{split}_spikes_heldout", data=rng.poisson(0.4, (n_trials, 12, 2)).astype("float32"))
        handle.create_dataset("train_cond_idx", data=np.array([[0, 1], [2, 3]]))
    cfg = ExperimentConfig(
        dataset=NLBDatasetConfig(name=name, data_path=str(path), bin_size_ms=5),
        model=MINTConfig(dataset=name, train_source=source, sigma=1, delta=2,
                         window_length=4, interp=0, causal=False, lfads_epochs=1,
                         lfads_generator_dim=4, lfads_factor_dim=3, lfads_encoder_dim=4,
                         lfads_controller_dim=4, lfads_batch_size=2),
        trainer=TrainerConfig(epochs=0, device=device), preprocessing=PreprocessingConfig(),
        output_dir=str(tmp_path / "runs"), batch_size=2,
    )
    experiment = Experiment(cfg)
    model = experiment.build_model()
    return experiment, model


def _fit(experiment, model):
    experiment.trainer.fit(model, build_strategy(model.config.optimization),
                           experiment.data.train_loader(), experiment.data.valid_loader())


@pytest.mark.parametrize("dataset", ["area2_bump", "mc_maze", "dmfc_rsg", "mc_rtt"])
def test_prepared_nlb_fits_dynamic_neuron_counts_and_masks_heldout(tmp_path, dataset):
    experiment, model = _nlb_experiment(tmp_path, dataset)
    _fit(experiment, model)
    assert model.n_heldin == 3 and model.n_heldout == 2
    assert model.Ts == pytest.approx(0.005)
    assert model.library_training["trials"] == 4
    assert len(model.Omega_plus) == (4 if dataset == "mc_rtt" else 2)
    batch = next(iter(experiment.data.valid_loader()))
    first = model(batch["heldin_spikes"])
    contaminated = torch.cat([batch["heldin_spikes"], torch.full_like(batch["heldout_spikes"], 10000)], -1)
    second = model(contaminated)
    torch.testing.assert_close(first.rates, second.rates)
    assert first.rates.shape == (2, 12, 2)
    assert first.extras["full_rates"].shape == (2, 12, 5)
    assert torch.isfinite(first.rates).all()
    assert first.full_rates_unit == first.rates_unit == "counts"
    library = [item.clone() for item in model.Omega_plus]
    _fit(experiment, model)
    for before, after in zip(library, model.Omega_plus):
        torch.testing.assert_close(before, after)


@pytest.mark.parametrize("device", ["cpu", CUDA_DEVICE])
def test_fitted_checkpoint_restores_predictions_without_training_data(tmp_path, device):
    experiment, model = _nlb_experiment(tmp_path)
    _fit(experiment, model)
    query = next(iter(experiment.data.valid_loader()))["heldin_spikes"]
    expected = model(query).extras["full_rates"]
    model.to(device)
    fitted_tensors = model.Omega_plus + model.Phi_plus + [
        model.V, model.first_idx0, model.last_idx0, model.first_tau_prime_idx0,
        model.shifted_idx1, model.shifted_idx2, model.lambda_range, model.rates, model.L,
    ]
    assert all(item.device.type == device for item in fitted_tensors)
    torch.testing.assert_close(model(query.to(device)).extras["full_rates"].cpu(), expected)
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = MINTConfig().build(n_neurons=3, n_time=12).to(device)
    restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    torch.testing.assert_close(restored(query.to(device)).extras["full_rates"].cpu(), expected)
    restored.to("cpu")
    torch.testing.assert_close(restored(query).extras["full_rates"], expected)
    assert restored.library_training == model.library_training


@pytest.mark.parametrize("device", ["cpu", CUDA_DEVICE])
def test_raw_mc_rtt_trains_lfads_and_restores_template_checkpoint(tmp_path, device):
    experiment, model = _nlb_experiment(tmp_path, "mc_rtt", "lfads", device=device)
    _fit(experiment, model)
    assert model.library_training["lfads_epochs"] == 1
    assert np.isfinite(model.library_training["lfads_final_loss"])
    assert model.device.type == device
    assert all(trial.device.type == device for trial in model.Omega_plus)
    query = next(iter(experiment.data.valid_loader()))["heldin_spikes"].to(device)
    restored = model.config.build(3, 12).to(device)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(query).rates, model(query).rates)


def test_standard_synthetic_experiment_fits_and_saves_mint(tmp_path: Path):
    config = ExperimentConfig(
        dataset=LorenzDatasetConfig(neurons=3, num_inits=2, num_trials=3,
                                    num_steps=8, burn_steps=4, train_fraction=0.67,
                                    spike_bin_size=0.25, seed=3),
        model=MINTConfig(sigma=1, window_length=2, delta=1, interp=0),
        trainer=TrainerConfig(epochs=0), preprocessing=PreprocessingConfig(),
        output_dir=str(tmp_path), batch_size=2,
    )
    experiment = Experiment(config)
    result = experiment.run()
    assert result.model_path.exists()
    assert len(experiment.model.Omega_plus) == 2
    assert experiment.model.config.dataset == "lorenz"
    assert experiment.model.Ts == 0.25
    assert np.isfinite(result.metrics["co_bps"])


def test_delta_counts_conversion_preserves_constant_training_intensity(tmp_path):
    experiment, model = _nlb_experiment(tmp_path)
    model.config.sigma = 0
    dataset = experiment.data.train_dataset
    dataset.heldin_spikes.fill_(0.25)
    dataset.raw_spikes.fill_(0.5)
    _fit(experiment, model)
    query = torch.full((1, 12, 3), 0.25)
    torch.testing.assert_close(model(query).rates, torch.full((1, 12, 2), 0.5, dtype=torch.float64))


def test_nlb_experiment_exports_same_counts_as_direct_predictions(tmp_path):
    experiment, _ = _nlb_experiment(tmp_path)
    with pytest.warns(RuntimeWarning, match="co-bps only"):
        result = experiment.run()
    query = next(iter(experiment.data.valid_loader()))["heldin_spikes"]
    expected = experiment.model(query).rates.detach().numpy()
    with h5py.File(result.run_dir / "nlb_submission.h5", "r") as handle:
        actual = handle["mc_maze"]["eval_rates_heldout"][()]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)
    assert result.metrics["nlb_co_bps"] == pytest.approx(result.metrics["co_bps"], abs=1e-6)


def test_cli_routes_synthetic_mint_through_standard_training(tmp_path, monkeypatch):
    from argparse import Namespace
    from ladys.cli import run_command

    config = ExperimentConfig(
        dataset=LorenzDatasetConfig(neurons=2, num_inits=1, num_trials=3,
                                    num_steps=6, burn_steps=3, train_fraction=0.67),
        model=MINTConfig(dataset="lorenz", sigma=1, window_length=2, delta=1, interp=0),
        trainer=TrainerConfig(epochs=0), preprocessing=PreprocessingConfig(),
        output_dir=str(tmp_path), run_name="cli-mint", batch_size=2,
    )
    monkeypatch.setattr("ladys.cli.build_experiment_config", lambda args: config)
    assert run_command(Namespace(resume_from=None)) == 0
    assert (tmp_path / "cli-mint" / "model.pt").exists()


def test_interpolation_does_not_stop_on_negative_newton_step():
    x1 = torch.tensor([0.672034, 1.251349, 5.312476, 4.094878], dtype=torch.float64)
    x2 = torch.tensor([5.102956, 3.884396, 2.468927, 3.123640], dtype=torch.float64)
    spikes = torch.tensor([7, 2, 1, 7], dtype=torch.float64)
    alpha = fit_poisson_interp(spikes, x1, x2, InterpOptions(), 0)
    assert alpha == pytest.approx(1.0)


@pytest.mark.parametrize("interp", [1, 2])
def test_interpolation_with_one_available_library_state(interp):
    model = MINTConfig(dataset="lorenz", sigma=0, delta=1, window_length=4,
                       interp=interp).build(2, 4)
    spikes = torch.ones(2, 4)
    model.fit_library([spikes], [spikes], np.array([0]))
    assert torch.isfinite(model(spikes.T.unsqueeze(0)).rates).all()
    with pytest.raises(ValueError, match="query duration"):
        model(spikes.T[:2].unsqueeze(0))


def test_unsmoothed_legacy_library_preserves_delta_count_units():
    model = MINTConfig(dataset="lorenz", sigma=0, delta=2, window_length=4,
                       interp=0).build(2, 8)
    spikes = torch.full((2, 8), 0.25)
    model.fit_library([spikes], [spikes], np.array([0]))
    expected = torch.full((1, 8, 2), 0.25, dtype=torch.float64)
    torch.testing.assert_close(model(spikes.T.unsqueeze(0)).rates, expected)


def test_synthetic_fit_and_evaluation_ignore_transformed_observations():
    config = LorenzDatasetConfig(neurons=2, num_inits=2, num_trials=3, num_steps=8,
                                 burn_steps=4, train_fraction=0.67, spike_bin_size=0.25)
    training, validation = LorenzDataset.make_splits(config)
    transformed = PreprocessingConfig(observations=[{"name": "anscombe"}])
    raw_model = MINTConfig(dataset="lorenz", sigma=1, window_length=2, interp=0).build(2, 8)
    transformed_model = raw_model.config.build(2, 8)
    raw_model.fit_training_data(DataLoader(training, batch_size=2), device="cpu")
    transformed_model.fit_training_data(DataLoader(PreprocessedDataset(training, transformed), batch_size=2), device="cpu")
    for raw, processed in zip(raw_model.Omega_plus, transformed_model.Omega_plus):
        torch.testing.assert_close(raw, processed)
    expected = evaluate_model(raw_model, DataLoader(validation, batch_size=2))
    actual = evaluate_model(transformed_model, DataLoader(PreprocessedDataset(validation, transformed), batch_size=2))
    np.testing.assert_allclose(actual.predictions["count_rates"], expected.predictions["count_rates"])
    assert actual.metrics["rate_mse"] == expected.metrics["rate_mse"]


@pytest.mark.parametrize("nested", [False, True])
def test_subset_preserves_training_condition_membership(nested):
    config = LorenzDatasetConfig(neurons=2, num_inits=2, num_trials=3, num_steps=8,
                                 burn_steps=4, train_fraction=0.67)
    training = LorenzDataset(config)
    subset = Subset(Subset(training, [3, 1, 0]), [0, 1]) if nested else Subset(training, [1, 3])
    model = MINTConfig(dataset="lorenz", sigma=1, window_length=2, interp=0).build(2, 8)
    model.fit_training_data(DataLoader(subset, batch_size=2), device="cpu")
    assert model._training_conditions.tolist() == [1, 1]
    assert len(model.Omega_plus) == 1
