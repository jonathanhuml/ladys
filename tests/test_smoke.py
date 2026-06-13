from torch.utils.data import DataLoader

from zynamics.datasets import LorenzDataset, LorenzDatasetConfig
from zynamics.models import CASSMConfig, GPFAConfig
from zynamics.training.strategies import build_strategy


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

    gpfa_config = GPFAConfig(latent_dim=2)
    gpfa = gpfa_config.build(n_neurons=x.shape[-1], n_time=x.shape[1])
    em = build_strategy(gpfa_config.optimization)
    em.setup(gpfa)
    result = em.step(gpfa, batch, epoch=0)
    assert result.batch_size == x.shape[0]
    assert gpfa(x).latents.shape[:2] == x.shape[:2]
