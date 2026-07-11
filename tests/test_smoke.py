import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from ladys.datasets import (
    ChaoticRNNDataset,
    ChaoticRNNDatasetConfig,
    LorenzDataset,
    LorenzDatasetConfig,
    NLBDataset,
    NLBDatasetConfig,
)
from ladys.metrics import compute_available_metrics
from ladys.metrics import evaluate_model
from ladys.models import (
    BGPFAConfig,
    CASSMConfig,
    GPFAConfig,
    KalmanConfig,
    LFADSConfig,
    NDTConfig,
    PSTHConfig,
    SmoothingConfig,
)
from ladys.training.strategies import build_strategy


def _assert_all_trainable_parameters_receive_gradients(model, loss):
    model.zero_grad(set_to_none=True)
    loss.backward()
    missing = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is None
    ]
    nonfinite = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad
        and param.grad is not None
        and not bool(param.grad.isfinite().all())
    ]
    assert missing == []
    assert nonfinite == []


def test_model_contracts_smoke():
    config = LorenzDatasetConfig(
        neurons=6,
        num_inits=2,
        num_trials=4,
        num_steps=16,
        burn_steps=20,
        seed=0,
    )
    train_ds, _ = LorenzDataset.make_splits(config)
    batch = next(iter(DataLoader(train_ds, batch_size=2)))
    x = batch["spikes"]

    cassm = CASSMConfig(projection_dim=3).build(n_neurons=x.shape[-1], n_time=x.shape[1])
    cassm_out = cassm(x)
    cassm_loss = cassm.loss(batch, cassm_out)
    assert cassm.predict_rates(x).shape == x.shape
    assert cassm_loss.total.ndim == 0
    _assert_all_trainable_parameters_receive_gradients(cassm, cassm_loss.total)

    kalman = KalmanConfig().build(n_neurons=x.shape[-1], n_time=x.shape[1])
    kalman_out = kalman(x)
    kalman_loss = kalman.loss(batch, kalman_out)
    assert kalman.predict_rates(x).shape == x.shape
    assert kalman_loss.total.ndim == 0
    _assert_all_trainable_parameters_receive_gradients(kalman, kalman_loss.total)

    gpfa_config = GPFAConfig(latent_dim=2)
    gpfa = gpfa_config.build(n_neurons=x.shape[-1], n_time=x.shape[1])
    gpfa_out = gpfa(x)
    gpfa_loss = gpfa.loss(batch, gpfa_out)
    assert gpfa_out.latents.shape[:2] == x.shape[:2]
    assert gpfa_loss.total.ndim == 0
    _assert_all_trainable_parameters_receive_gradients(gpfa, gpfa_loss.total)

    gradient = build_strategy(gpfa_config.optimization)
    gradient.setup(gpfa)
    result = gradient.step(gpfa, batch, epoch=0)
    assert result.batch_size == x.shape[0]

    lfads = LFADSConfig(
        generator_dim=8,
        inferred_input_dim=1,
        factor_dim=4,
        g0_encoder_dim=8,
        controller_encoder_dim=8,
        controller_dim=8,
        keep_prob=1.0,
    ).build(n_neurons=x.shape[-1], n_time=x.shape[1])
    lfads_out = lfads(x)
    lfads_loss = lfads.loss(batch, lfads_out)
    assert lfads_out.rates.shape == x.shape
    assert lfads_out.latents.shape[:2] == x.shape[:2]
    assert lfads.predict_rates(x).shape == x.shape
    assert lfads_loss.total.ndim == 0
    _assert_all_trainable_parameters_receive_gradients(lfads, lfads_loss.total)

    ndt = NDTConfig(
        hidden_size=16,
        num_layers=1,
        embed_dim=2,
        num_heads=2,
        dropout=0.0,
        dropout_rates=0.0,
        dropout_embedding=0.0,
    ).build(n_neurons=x.shape[-1], n_time=x.shape[1])
    ndt_out = ndt(x)
    ndt_loss = ndt.loss(batch, ndt_out)
    assert ndt_out.rates.shape == x.shape
    assert ndt_out.latents.shape[:2] == x.shape[:2]
    assert ndt.predict_rates(x).shape == x.shape
    assert ndt_loss.total.ndim == 0
    _assert_all_trainable_parameters_receive_gradients(ndt, ndt_loss.total)


def test_bgpfa_config_uses_differentiable_full_batch_strategy():
    config = BGPFAConfig(latent_dim=2, n_mc_train=1, n_mc_eval=1)
    assert config.optimization.name == "mgplvm_full_batch_gradient"
    strategy = build_strategy(config.optimization)
    assert strategy.name == "mgplvm_full_batch_gradient"
    assert strategy.steps_per_epoch == 1

    multi_step_strategy = build_strategy(
        BGPFAConfig(
            optimization={
                "name": "mgplvm_full_batch_gradient",
                "steps_per_epoch": 4,
            }
        ).optimization
    )
    assert multi_step_strategy.steps_per_epoch == 4

    with pytest.raises(ValueError, match="does not support optimization.name='em'"):
        BGPFAConfig(optimization={"name": "em"})


@pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None,
    reason="BGPFA vendored mgplvm smoke test requires scikit-learn",
)
def test_bgpfa_vendored_mgplvm_smoke():
    config = LorenzDatasetConfig(
        neurons=4,
        num_inits=1,
        num_trials=2,
        num_steps=6,
        burn_steps=5,
        seed=0,
    )
    train_ds, _ = LorenzDataset.make_splits(config)
    batch = next(iter(DataLoader(train_ds, batch_size=len(train_ds))))
    x = batch["spikes"]

    bgpfa = BGPFAConfig(
        latent_dim=1,
        n_mc_train=1,
        n_mc_eval=1,
        kl_burnin_epochs=0,
    ).build(n_neurons=x.shape[-1], n_time=x.shape[1])
    strategy = build_strategy(BGPFAConfig().optimization)
    strategy.setup(bgpfa)
    result = strategy.step(bgpfa, batch, epoch=0)

    assert result.batch_size == x.shape[0]
    assert result.objective == "negative_elbo"
    assert bgpfa.predict_rates(x).shape == x.shape


def test_chaotic_rnn_dataset_contract():
    config = ChaoticRNNDatasetConfig(
        neurons=5,
        hidden_units=8,
        num_conditions=3,
        num_trials=4,
        num_steps=12,
        seed=0,
    )
    train_ds, valid_ds = ChaoticRNNDataset.make_splits(config)

    assert train_ds.spikes.shape == (9, 12, 5)
    assert valid_ds.spikes.shape == (3, 12, 5)
    assert train_ds.rates.shape == train_ds.spikes.shape
    assert train_ds.latents.shape == (9, 12, 8)

    sample = train_ds[0]
    assert set(sample) == {"spikes", "rates", "latents", "dt"}


def test_nlb_dataset_loads_grouped_20ms_h5(tmp_path: Path):
    path = tmp_path / "nlb.h5"
    heldin = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    heldout = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("mc_rtt_20")
        group.create_dataset("eval_spikes_heldin", data=heldin)
        group.create_dataset("eval_spikes_heldout", data=heldout)

    config = NLBDatasetConfig(name="mc_rtt", data_path=str(path), bin_size_ms=20)
    train_ds, valid_ds = NLBDataset.make_splits(config)

    assert train_ds.spikes.shape == (2, 3, 4)
    assert train_ds.raw_spikes.shape == (2, 3, 2)
    assert valid_ds[0]["dt"].item() == pytest.approx(0.02)
    assert set(train_ds[0]) == {
        "spikes",
        "heldin_spikes",
        "raw_spikes",
        "heldout_spikes",
        "dt",
    }


def test_nlb_dataset_uses_train_tensors_when_available(tmp_path: Path):
    path = tmp_path / "nlb_train_eval.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("train_spikes_heldin", data=np.ones((3, 4, 2), dtype=np.float32))
        handle.create_dataset("train_spikes_heldout", data=np.ones((3, 4, 1), dtype=np.float32) * 2)
        handle.create_dataset("eval_spikes_heldin", data=np.ones((2, 4, 2), dtype=np.float32) * 3)
        handle.create_dataset("eval_spikes_heldout", data=np.ones((2, 4, 1), dtype=np.float32) * 4)

    config = NLBDatasetConfig(name="mc_maze", data_path=str(path), bin_size_ms=5)
    train_ds, valid_ds = NLBDataset.make_splits(config)

    assert train_ds.spikes.shape == (3, 4, 2)
    assert train_ds.raw_spikes.shape == (3, 4, 1)
    assert valid_ds.spikes.shape == (2, 4, 2)
    assert valid_ds.raw_spikes.shape == (2, 4, 1)
    assert train_ds[0]["spikes"].mean().item() == pytest.approx(1.0)
    assert valid_ds[0]["spikes"].mean().item() == pytest.approx(3.0)


def test_gpfa_nlb_adapter_fits_heldout_decoder(tmp_path: Path):
    path = tmp_path / "gpfa_nlb.h5"
    rng = np.random.default_rng(0)
    train_heldin = rng.poisson(0.5, size=(5, 6, 3)).astype(np.float32)
    train_heldout = (train_heldin[..., :1] + 0.1).astype(np.float32)
    eval_heldin = rng.poisson(0.5, size=(2, 6, 3)).astype(np.float32)
    eval_heldout = (eval_heldin[..., :1] + 0.1).astype(np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("train_spikes_heldin", data=train_heldin)
        handle.create_dataset("train_spikes_heldout", data=train_heldout)
        handle.create_dataset("eval_spikes_heldin", data=eval_heldin)
        handle.create_dataset("eval_spikes_heldout", data=eval_heldout)

    config = NLBDatasetConfig(name="mc_maze", data_path=str(path), bin_size_ms=5)
    train_ds, valid_ds = NLBDataset.make_splits(config)
    model = GPFAConfig(
        latent_dim=1,
        init_method="normal",
        init_seed=0,
        learn_kernel_params=False,
    ).build(n_neurons=train_ds.spikes.shape[-1], n_time=train_ds.spikes.shape[1])

    result = evaluate_model(
        model,
        DataLoader(valid_ds, batch_size=2),
        train_loader=DataLoader(train_ds, batch_size=5),
    )

    assert result.predictions["rates"].shape == eval_heldout.shape
    assert result.targets["spikes"].shape == eval_heldout.shape
    assert "co_bps" in result.metrics
    assert np.isfinite(result.metrics["co_bps"])


def test_smoothing_nlb_adapter_fits_heldout_decoder(tmp_path: Path):
    path = tmp_path / "smoothing_nlb.h5"
    rng = np.random.default_rng(0)
    train_heldin = rng.poisson(0.5, size=(5, 8, 3)).astype(np.float32)
    train_heldout = (train_heldin[..., :1] + 0.1).astype(np.float32)
    eval_heldin = rng.poisson(0.5, size=(2, 8, 3)).astype(np.float32)
    eval_heldout = (eval_heldin[..., :1] + 0.1).astype(np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("train_spikes_heldin", data=train_heldin)
        handle.create_dataset("train_spikes_heldout", data=train_heldout)
        handle.create_dataset("eval_spikes_heldin", data=eval_heldin)
        handle.create_dataset("eval_spikes_heldout", data=eval_heldout)

    config = NLBDatasetConfig(name="mc_maze", data_path=str(path), bin_size_ms=5)
    train_ds, valid_ds = NLBDataset.make_splits(config)
    model = SmoothingConfig(
        kern_sd_ms=10.0,
        bin_size_ms=5.0,
        nlb_poisson_max_iter=20,
    ).build(n_neurons=train_ds.spikes.shape[-1], n_time=train_ds.spikes.shape[1])

    result = evaluate_model(
        model,
        DataLoader(valid_ds, batch_size=2),
        train_loader=DataLoader(train_ds, batch_size=5),
    )

    assert result.predictions["rates"].shape == eval_heldout.shape
    assert "co_bps" in result.metrics
    assert np.isfinite(result.metrics["co_bps"])


def test_psth_nlb_adapter_uses_training_condition_psths(tmp_path: Path):
    path = tmp_path / "psth_nlb.h5"
    train_heldin = np.ones((2, 4, 2), dtype=np.float32)
    train_heldout = np.array(
        [
            [[2.0], [2.0], [2.0], [2.0]],
            [[3.0], [3.0], [3.0], [3.0]],
        ],
        dtype=np.float32,
    )
    eval_heldin = np.ones((3, 4, 2), dtype=np.float32)
    eval_heldout = np.array(
        [
            [[2.0], [2.0], [2.0], [2.0]],
            [[3.0], [3.0], [3.0], [3.0]],
            [[2.0], [2.0], [2.0], [2.0]],
        ],
        dtype=np.float32,
    )
    psth = np.zeros((2, 4, 3), dtype=np.float32)
    psth[0, :, 2] = 99.0
    psth[1, :, 2] = 99.0
    with h5py.File(path, "w") as handle:
        handle.create_dataset("train_spikes_heldin", data=train_heldin)
        handle.create_dataset("train_spikes_heldout", data=train_heldout)
        handle.create_dataset("eval_spikes_heldin", data=eval_heldin)
        handle.create_dataset("eval_spikes_heldout", data=eval_heldout)
        handle.create_dataset("psth", data=psth)
        cond_ds = handle.create_dataset("eval_cond_idx", (2,), dtype=h5py.vlen_dtype(np.dtype("int64")))
        cond_ds[0] = np.array([0, 2], dtype=np.int64)
        cond_ds[1] = np.array([1], dtype=np.int64)
        train_cond_ds = handle.create_dataset(
            "train_cond_idx",
            (2,),
            dtype=h5py.vlen_dtype(np.dtype("int64")),
        )
        train_cond_ds[0] = np.array([0], dtype=np.int64)
        train_cond_ds[1] = np.array([1], dtype=np.int64)

    config = NLBDatasetConfig(name="mc_maze", data_path=str(path), bin_size_ms=5)
    train_ds, valid_ds = NLBDataset.make_splits(config)
    model = PSTHConfig(kern_sd_ms=0.0).build(
        n_neurons=train_ds.spikes.shape[-1],
        n_time=train_ds.spikes.shape[1],
    )

    result = evaluate_model(
        model,
        DataLoader(valid_ds, batch_size=2),
        train_loader=DataLoader(train_ds, batch_size=2),
    )

    assert np.allclose(result.predictions["rates"], eval_heldout)
    assert result.metrics["co_bps"] > 0.0


def test_metrics_skip_incompatible_nlb_shapes():
    predictions = {"rates": torch.ones(2, 3, 4)}
    targets = {"spikes": torch.ones(2, 3, 2)}

    assert compute_available_metrics(predictions, targets) == {}


def test_gpfa_warns_when_configured_for_em():
    with pytest.warns(RuntimeWarning, match="legacy full-dataset EM adapter"):
        GPFAConfig(optimization={"name": "em"})


def test_gpfa_initialization_methods_are_configurable():
    config = LorenzDatasetConfig(
        neurons=5,
        num_inits=2,
        num_trials=3,
        num_steps=8,
        burn_steps=5,
        seed=0,
    )
    train_ds, _ = LorenzDataset.make_splits(config)
    x = next(iter(DataLoader(train_ds, batch_size=2)))["spikes"]

    normal = GPFAConfig(
        latent_dim=2,
        init_method="normal",
        init_seed=123,
        learn_kernel_params=False,
    ).build(n_neurons=x.shape[-1], n_time=x.shape[1])
    kaiming = GPFAConfig(
        latent_dim=2,
        init_method="kaiming_normal",
        init_seed=123,
        learn_kernel_params=False,
    ).build(n_neurons=x.shape[-1], n_time=x.shape[1])
    fa = GPFAConfig(
        latent_dim=2,
        init_method="fa",
        init_seed=123,
        fa_max_iters=2,
        learn_kernel_params=False,
    ).build(n_neurons=x.shape[-1], n_time=x.shape[1])

    normal.initialize(x)
    kaiming.initialize(x)
    fa.initialize(x)

    assert normal.initialized
    assert kaiming.initialized
    assert fa.initialized
    assert normal.C.isfinite().all()
    assert kaiming.C.isfinite().all()
    assert fa.C.isfinite().all()
    assert normal._r_diag().isfinite().all()
    assert kaiming._r_diag().isfinite().all()
    assert fa._r_diag().isfinite().all()
    assert not torch.allclose(normal.C, kaiming.C)

    c_before = kaiming.C.detach().clone()
    out = kaiming(x)
    assert "Corth" in out.extras
    assert torch.allclose(kaiming.C, c_before)
