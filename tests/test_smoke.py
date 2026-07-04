from torch.utils.data import DataLoader

from ladys.datasets import LorenzDataset, LorenzDatasetConfig
from ladys.models import CASSMConfig, GPFAConfig, KalmanConfig, LFADSConfig
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
    em = build_strategy(gpfa_config.optimization)
    em.setup(gpfa)
    result = em.step(gpfa, batch, epoch=0)
    assert result.batch_size == x.shape[0]
    assert gpfa(x).latents.shape[:2] == x.shape[:2]

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
