from pathlib import Path

import torch

from ladys.config import load_experiment_config
from ladys.models.ilqr_vae import ILQRVAEConfig


def test_ilqr_vae_nlb_configs_are_self_contained():
    mc_maze = load_experiment_config(
        Path("configs/experiment/real/mc_maze/ilqr_vae/ilqr_vae_mc_maze_nlb_5ms.yaml")
    )
    scratch_paths = [
        Path("configs/experiment/real/area2_bump/ilqr_vae/ilqr_vae_area2_bump_nlb_5ms_train.yaml"),
        Path("configs/experiment/real/dmfc_rsg/ilqr_vae/ilqr_vae_dmfc_rsg_nlb_5ms_train.yaml"),
        Path("configs/experiment/real/mc_rtt/ilqr_vae/ilqr_vae_mc_rtt_nlb_5ms_train.yaml"),
    ]
    scratch_configs = [load_experiment_config(path) for path in scratch_paths]

    assert isinstance(mc_maze.model, ILQRVAEConfig)
    assert mc_maze.model.params_path == "data/real/ilqr_vae/final_params.bin"
    assert "ilqr-vae-tutorial" not in str(mc_maze.model.params_path)
    assert mc_maze.model.initialization == "pretrained"
    assert mc_maze.model.trainable_parameters is False

    for config in scratch_configs:
        assert isinstance(config.model, ILQRVAEConfig)
        assert config.model.params_path is None
        assert config.model.initialization == "checkpoint_transfer"
        assert config.model.template_params_path == "data/real/ilqr_vae/final_params.bin"
        assert "ilqr-vae-tutorial" not in str(config.model.template_params_path)
        assert config.model.random_init_profile == "tutorial_mc_maze"
        assert config.model.readout_bias_initialization == "empirical_rates"
        assert config.model.latent_dim == 90
        assert config.model.input_dim == 15
        assert config.model.objective == "ilqr_vae_elbo"
        assert config.model.trainable_parameters is True
        assert config.model.optimization.kwargs()["lr_scheduler"] == "sqrt_decay"


def test_ilqr_vae_random_build_covers_heldout_output_slice():
    config = ILQRVAEConfig(
        params_path=None,
        initialization="random",
        objective="ilqr_vae_elbo",
        latent_dim=10,
        input_dim=5,
        held_in_neurons=3,
        output_neuron_start=3,
        output_neurons=2,
        trainable_parameters=True,
    )

    model = config.build(n_neurons=3, n_time=4)

    assert model.core.n_neurons == 5


def test_ilqr_vae_empirical_readout_bias_initialization():
    class _Dataset:
        spikes = torch.full((2, 4, 3), 0.02)
        raw_spikes = torch.full((2, 4, 2), 0.04)

    class _Data:
        n_neurons = 3
        n_time = 4
        train_dataset = _Dataset()

    config = ILQRVAEConfig(
        params_path=None,
        initialization="random",
        objective="ilqr_vae_elbo",
        latent_dim=10,
        input_dim=5,
        held_in_neurons=3,
        output_neuron_start=3,
        output_neurons=2,
        trainable_parameters=True,
        readout_bias_initialization="empirical_rates",
        dt=0.005,
    )

    model = config.build_from_data(_Data())

    expected_hz = torch.tensor([4.0, 4.0, 4.0, 8.0, 8.0], dtype=model.core.bias.dtype)
    expected = torch.log(expected_hz - 1.0e-3)
    torch.testing.assert_close(model.core.bias.detach().reshape(-1), expected)
