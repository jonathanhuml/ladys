from pathlib import Path

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
