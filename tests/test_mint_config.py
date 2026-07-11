from pathlib import Path

import numpy as np
import torch

from ladys.config import load_experiment_config
from ladys.models.base import BaseModelConfig
from ladys.models.mint import MINTConfig


def test_mint_model_is_registered():
    assert BaseModelConfig.registry["mint"] is MINTConfig


def test_mint_nlb_experiment_configs_load():
    paths = [
        Path("configs/experiment/real/area2_bump/mint/mint_area2_bump_nlb_5ms.yaml"),
        Path("configs/experiment/real/mc_maze/mint/mint_mc_maze_nlb_5ms.yaml"),
        Path("configs/experiment/real/mc_rtt/mint/mint_mc_rtt_nlb_5ms.yaml"),
    ]

    configs = [load_experiment_config(path) for path in paths]

    assert [config.model.dataset for config in configs] == ["area2_bump", "mc_maze", "mc_rtt"]
    assert [config.model.train_source for config in configs] == ["nwb", "nwb", "mat"]
    assert all(isinstance(config.model, MINTConfig) for config in configs)
    assert all(config.model.nlb_neural_state_defaults for config in configs)


def test_mint_lorenz_library_defaults_to_smoothed_spikes():
    model = MINTConfig(
        dataset="lorenz",
        sigma=1,
        delta=1,
        window_length=3,
        n_candidates=1,
    ).build(n_neurons=2, n_time=12)
    spikes = [
        torch.zeros(2, 12),
        torch.ones(2, 12),
    ]
    oracle_rates = [
        torch.full((2, 12), 999.0),
        torch.full((2, 12), 999.0),
    ]

    model.fit_library(spikes, oracle_rates, np.asarray([0, 0]))

    assert model.config.lorenz_library_source == "smoothed_spikes"
    assert len(model.Omega_plus) == 1
    assert float(model.Omega_plus[0].max()) < 10.0
