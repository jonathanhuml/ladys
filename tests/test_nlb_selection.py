from pathlib import Path

import h5py
import numpy as np
import pytest

from ladys.config import ExperimentConfig
from ladys.datasets.nlb import NLBDatasetConfig
from ladys.experiment import Experiment
from ladys.models import GPFAConfig
from ladys.nlb_eval import prepare_nlb_selection_target
from ladys.preprocessing import PreprocessingConfig
from ladys.training import TrainerConfig
from scripts import run_langevin_flow_nlb_reproduction as flow_runner
from scripts import run_stndt_nlb_reproduction as stndt_runner


def _config(tmp_path, split="val", live_eval_interval=0):
    path = tmp_path / "prepared.h5"
    rng = np.random.default_rng(1)
    with h5py.File(path, "w") as handle:
        for prefix, trials in (("train", 4), ("eval", 2)):
            for partition, neurons in (("heldin", 3), ("heldout", 2)):
                handle[f"{prefix}_spikes_{partition}"] = rng.poisson(
                    0.3, (trials, 4, neurons)
                ).astype("float32")
    return ExperimentConfig(
        dataset=NLBDatasetConfig(data_path=str(path), split=split, max_trials=4),
        model=GPFAConfig(latent_dim=2),
        trainer=TrainerConfig(epochs=1, live_eval_interval=live_eval_interval),
        preprocessing=PreprocessingConfig(),
        batch_size=2, output_dir=str(tmp_path / "runs"),
    )


def test_selection_target_is_copied_from_configured_validation_data(tmp_path):
    config = _config(tmp_path)
    target = prepare_nlb_selection_target(config.dataset, tmp_path / "run")
    with h5py.File(target) as actual, h5py.File(config.dataset.data_path) as source:
        np.testing.assert_array_equal(
            actual["mc_maze/eval_spikes_heldout"], source["eval_spikes_heldout"]
        )
    config.dataset.split = "test"
    with pytest.raises(ValueError, match="split='val'"):
        prepare_nlb_selection_target(config.dataset, tmp_path / "bad")


def test_test_split_never_reaches_training_validation_or_scheduler(tmp_path):
    result = Experiment(_config(tmp_path, split="test")).run()
    assert len(result.history) == 1
    assert result.history[0].valid is None
    assert np.isfinite(result.metrics["co_bps"])


def test_test_split_cannot_select_live_checkpoints(tmp_path):
    experiment = Experiment(_config(tmp_path, split="test", live_eval_interval=1))
    with pytest.raises(ValueError, match="checkpoint selection"):
        experiment.run()


@pytest.mark.parametrize("runner", [flow_runner, stndt_runner])
@pytest.mark.parametrize("split,target,match", [
    ("test", None, "split='val'"),
    ("val", Path("external_test_targets.h5"), "omit --target-h5"),
])
def test_reproduction_selection_rejects_test_or_external_targets(tmp_path, runner, split, target, match):
    kwargs = dict(config=_config(tmp_path, split), target_h5=target,
                  eval_every=1, progress_every=1, patience_evals=1)
    if runner is flow_runner:
        kwargs.update(stop_at_reported=False, skip_validation=False)
    else:
        kwargs.update(checkpoint=None, resume_snapshot=None)
    with pytest.raises(ValueError, match=match):
        runner.run_config(**kwargs)


def test_published_test_scores_cannot_stop_training(tmp_path):
    with pytest.raises(ValueError, match="Published test scores"):
        flow_runner.run_config(
            config=_config(tmp_path), target_h5=None, eval_every=1,
            progress_every=1, patience_evals=1, stop_at_reported=True,
            skip_validation=False,
        )
