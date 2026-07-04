import pytest

from torch.utils.data import DataLoader

from ladys.datasets import (
    ChaoticRNNDataset,
    ChaoticRNNDatasetConfig,
    LorenzDataset,
    LorenzDatasetConfig,
)
from ladys.models import (
    CASSMConfig,
    GPFAConfig,
    KalmanConfig,
    LFADSConfig,
    NDTConfig,
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


def test_gpfa_warns_when_configured_for_em():
    with pytest.warns(RuntimeWarning, match="legacy full-dataset EM adapter"):
        GPFAConfig(optimization={"name": "em"})
