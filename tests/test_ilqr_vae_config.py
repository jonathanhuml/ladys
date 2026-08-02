from pathlib import Path

import torch

from ladys.config import load_experiment_config
from ladys.models.ilqr_vae import ILQRVAEConfig
from ladys.training.strategies import GradientStrategy


def test_ilqr_vae_nlb_configs_are_self_contained():
    canonical_path = Path(
        "configs/experiment/real/mc_maze/ilqr_vae/ilqr_vae_mc_maze_nlb_5ms.yaml"
    )
    transfer_paths = [
        Path("configs/experiment/real/area2_bump/ilqr_vae/ilqr_vae_area2_bump_nlb_5ms_train.yaml"),
        Path("configs/experiment/real/dmfc_rsg/ilqr_vae/ilqr_vae_dmfc_rsg_nlb_5ms_train.yaml"),
        Path("configs/experiment/real/mc_rtt/ilqr_vae/ilqr_vae_mc_rtt_nlb_5ms_train.yaml"),
    ]
    nlb_paths = sorted(Path("configs/experiment/real").glob("*/ilqr_vae/*.yaml"))

    assert nlb_paths == sorted([canonical_path, *transfer_paths])

    mc_maze = load_experiment_config(canonical_path)
    assert isinstance(mc_maze.model, ILQRVAEConfig)
    assert mc_maze.model.params_path == "data/real/ilqr_vae/final_params.bin"
    assert "ilqr-vae-tutorial" not in str(mc_maze.model.params_path)
    assert mc_maze.model.initialization == "pretrained"
    assert mc_maze.model.trainable_parameters is False

    for path in transfer_paths:
        config = load_experiment_config(path)
        assert isinstance(config.model, ILQRVAEConfig)
        assert config.model.params_path is None
        assert config.model.initialization == "checkpoint_transfer"
        assert config.model.template_params_path == "data/real/ilqr_vae/final_params.bin"
        assert "ilqr-vae-tutorial" not in str(config.model.template_params_path)
        assert config.model.random_init_profile == "tutorial_mc_maze"
        assert config.model.readout_bias_initialization == "empirical_rates"
        assert config.model.latent_dim == 90
        assert config.model.input_dim == 15
        assert config.model.max_iter == 2
        assert config.model.differentiate_controls is False
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


def test_ilqr_vae_control_hessian_mode_propagates_to_core():
    config = ILQRVAEConfig(
        params_path=None,
        initialization="random",
        latent_dim=10,
        input_dim=5,
        control_hessian_mode="fisher",
    )

    model = config.build(n_neurons=3, n_time=4)

    assert model.core.control_hessian_mode == "fisher"


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


def test_ilqr_vae_elbo_training_infers_controls_from_full_observations():
    config = ILQRVAEConfig(
        params_path=None,
        initialization="random",
        objective="ilqr_vae_elbo",
        latent_dim=10,
        input_dim=5,
        solver="ilqr",
        max_iter=0,
        held_in_neurons=3,
        output_neuron_start=3,
        output_neurons=2,
        trainable_parameters=True,
        optimization={"name": "gradient", "optimizer": "Adam", "lr": 1.0e-3},
    )
    model = config.build(n_neurons=3, n_time=4)
    batch = {
        "spikes": torch.full((2, 4, 3), 0.02),
        "heldout_spikes": torch.full((2, 4, 2), 0.04),
    }

    calls = []
    original_infer_controls = model.core.infer_controls

    def record_infer_controls(spikes, *args, **kwargs):
        calls.append((int(spikes.shape[-1]), int(kwargs["held_in_neurons"])))
        return original_infer_controls(spikes, *args, **kwargs)

    model.core.infer_controls = record_infer_controls
    strategy = GradientStrategy(optimizer="Adam", lr=1.0e-3)
    strategy.setup(model)

    result = strategy.step(model, batch, epoch=0)

    assert result.loss == result.loss
    assert calls == [(5, 5), (5, 5)]


def test_ilqr_vae_elbo_training_can_keep_control_gradient_path():
    config = ILQRVAEConfig(
        params_path=None,
        initialization="random",
        objective="ilqr_vae_elbo",
        latent_dim=4,
        input_dim=2,
        solver="ilqr",
        max_iter=1,
        control_hessian_mode="fisher",
        differentiate_controls=True,
        held_in_neurons=3,
        output_neuron_start=3,
        output_neurons=2,
        trainable_parameters=True,
    )
    model = config.build(n_neurons=3, n_time=4)
    x = torch.full((1, 4, 3), 0.02)

    output = model(x)

    assert output.extras["controls"].requires_grad


def test_ilqr_vae_ilqr_failure_falls_back_to_gradient_control_solver():
    config = ILQRVAEConfig(
        params_path=None,
        initialization="random",
        objective="ilqr_vae_elbo",
        latent_dim=10,
        input_dim=5,
        solver="ilqr",
        max_iter=0,
        ilqr_failure_fallback="adam",
        ilqr_fallback_max_iter=1,
        held_in_neurons=3,
        output_neuron_start=3,
        output_neurons=2,
    )
    model = config.build(n_neurons=3, n_time=4)
    original_infer_controls = model.core.infer_controls
    solvers = []

    def fail_ilqr_once(spikes, *args, **kwargs):
        solver = kwargs["solver"]
        solvers.append(solver)
        if solver == "ilqr":
            raise RuntimeError("iLQR backward pass did not find a positive definite Q_uu.")
        return original_infer_controls(spikes, *args, **kwargs)

    model.core.infer_controls = fail_ilqr_once
    x = torch.full((1, 4, 3), 0.02)

    with torch.no_grad():
        output = model(x)

    assert solvers == ["ilqr", "adam"]
    assert output.extras["inference_fallbacks"].item() == 1.0
