import math

import pytest
import torch
from torch.utils.data import DataLoader

from ladys.datasets import LorenzDataset, LorenzDatasetConfig
from ladys.models import LFADSConfig
from ladys.types import ModelOutput
from scripts.benchmark_lorenz_loss_curves import (
    collect_rate_trace_rows,
    evaluate_poisson_nll,
    evaluate_rate_mse,
)


class FixedPrediction(torch.nn.Module):
    def __init__(self, rates, unit="counts"):
        super().__init__()
        self.register_buffer("rates", rates)
        self.unit = unit

    def forward(self, x):
        return ModelOutput(rates=self.rates[: x.shape[0]], rates_unit=self.unit)


def test_benchmark_lfads_uses_hz_for_rate_errors_and_counts_for_poisson():
    model = LFADSConfig(
        generator_dim=4, factor_dim=2, g0_encoder_dim=4,
        controller_encoder_dim=4, controller_dim=4, dt=0.005,
        initialize_log_rate_bias=False,
    ).build(n_neurons=2, n_time=3).eval()
    with torch.no_grad():
        model.fc_log_rates.weight.zero_()
        model.fc_log_rates.bias.fill_(math.log(20.0))
    spikes = torch.tensor([[[0.0, 1.0], [1.0, 0.0], [2.0, 0.0]]]).repeat(2, 1, 1)
    dt = torch.tensor([0.005, 0.01])
    batch = {
        "spikes": spikes, "raw_spikes": spikes,
        "rates": torch.full_like(spikes, 20.0), "rates_unit": ["hz", "hz"], "dt": dt,
    }
    assert evaluate_rate_mse(model, [batch], "cpu") == pytest.approx(0.0, abs=1e-9)
    counts = 20.0 * dt[:, None, None]
    expected = (counts - spikes * counts.log() + torch.lgamma(spikes + 1.0)).mean().item()
    assert evaluate_poisson_nll(model, [batch], "cpu") == pytest.approx(expected)


def test_nonunit_lorenz_benchmark_metrics_and_traces_agree():
    dataset = LorenzDataset(LorenzDatasetConfig(
        neurons=2, num_inits=1, num_trials=3, num_steps=8,
        burn_steps=5, spike_bin_size=0.005, seed=2,
    ))
    model = FixedPrediction(dataset.rates * dataset.arrays.dt)
    loader = DataLoader(dataset, batch_size=len(dataset))
    assert evaluate_rate_mse(model, loader, "cpu") == pytest.approx(0.0, abs=1e-9)
    rows = collect_rate_trace_rows(model, dataset, "cpu", "fixed", 2, 0)
    assert all(row["rate_unit"] == "hz" for row in rows)
    for row in rows:
        assert row["true_rate"] == pytest.approx(row["pred_rate"], rel=1e-6)


def test_ctd_count_targets_are_normalized_to_hz_squared_error():
    spikes = torch.ones(2, 3, 1)
    batch = {
        "spikes": spikes, "rates": torch.ones_like(spikes),
        "rates_unit": ["counts", "counts"], "dt": torch.tensor([0.005, 0.01]),
    }
    model = FixedPrediction(torch.full_like(spikes, 2.0))
    expected = ((1.0 / batch["dt"]) ** 2).mean().item()
    assert evaluate_rate_mse(model, [batch], "cpu") == pytest.approx(expected)


def test_benchmark_preserves_monte_carlo_prediction_policy():
    class AveragedPrediction(FixedPrediction):
        prediction_samples = 4

        def predict_rates(self, x):
            return 2.0 * self.rates[: x.shape[0]]

    spikes = torch.ones(1, 3, 1)
    batch = {
        "spikes": spikes, "rates": torch.full_like(spikes, 40.0),
        "rates_unit": "hz", "dt": 0.005,
    }
    model = AveragedPrediction(torch.full_like(spikes, 20.0), unit="hz")
    assert evaluate_rate_mse(model, [batch], "cpu") == 0.0


def test_bgpfa_helper_preserves_inference_options_and_raw_input():
    class PosteriorPrediction(FixedPrediction):
        def infer_latents(self, spikes, **kwargs):
            self.inferred_spikes = spikes
            self.inference_options = kwargs

    batch = {
        "spikes": torch.zeros(1, 3, 1), "raw_spikes": torch.ones(1, 3, 1),
        "rates": torch.ones(1, 3, 1), "rates_unit": "counts", "dt": 0.005,
    }
    model = PosteriorPrediction(torch.ones(2, 3, 1))
    assert evaluate_rate_mse(
        model, [batch, batch], "cpu", use_raw_spikes=True,
        bgpfa_infer_steps=2, bgpfa_infer_mc=3, bgpfa_infer_lr=0.04,
    ) == 0.0
    torch.testing.assert_close(model.inferred_spikes, torch.ones(2, 3, 1))
    assert model.inference_options == {"max_steps": 2, "n_mc": 3, "lrate": 0.04, "burnin": 1}
