from pathlib import Path

from ladys.cli import main
from ladys.config import ExperimentConfig
from ladys.datasets import LorenzDatasetConfig
from ladys.experiment import Experiment
from ladys.models import GPFAConfig
from ladys.preprocessing import PreprocessingConfig
from ladys.training import TrainerConfig


def test_experiment_writes_run_artifacts(tmp_path: Path):
    config = ExperimentConfig(
        dataset=LorenzDatasetConfig(
            neurons=5,
            num_inits=2,
            num_trials=3,
            num_steps=8,
            burn_steps=5,
            seed=0,
        ),
        model=GPFAConfig(
            latent_dim=2,
            learn_kernel_params=False,
            fa_max_iters=3,
            kernel_param_max_iters=1,
        ),
        trainer=TrainerConfig(epochs=1, device="cpu"),
        preprocessing=PreprocessingConfig(),
        batch_size=2,
        output_dir=str(tmp_path),
        run_name="gpfa-smoke",
    )

    result = Experiment(config).run()

    assert result.run_dir == tmp_path / "gpfa-smoke"
    assert result.config_path.exists()
    assert result.history_path.exists()
    assert result.metrics_path.exists()
    assert result.model_path.exists()
    assert result.predictions_path is not None
    assert result.predictions_path.exists()
    assert result.report_path.exists()
    assert "co_bps" in result.metrics
    assert "rate_mse" in result.metrics


def test_cli_run_from_config(tmp_path: Path):
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        """
dataset:
  name: lorenz
  neurons: 5
  num_inits: 2
  num_trials: 3
  num_steps: 8
  burn_steps: 5
  train_fraction: 0.67
  seed: 1
  latent_dt: 0.015
  spike_bin_size: 1.0
  base_rate: 1.0

model:
  name: gpfa
  objective: negative_log_marginal_likelihood
  latent_dim: 2
  bin_width: 20.0
  start_tau: 100.0
  start_eps: 1.0e-3
  min_var_frac: 0.01
  learn_kernel_params: false
  fa_max_iters: 3
  fa_tol: 1.0e-8
  kernel_param_max_iters: 1
  kernel_param_lr: 1.0
  jitter: 1.0e-5
  optimization:
    name: gradient
    optimizer: Adam
    lr: 1.0e-2
    weight_decay: 0.0
    gradient_clip: 100.0

preprocessing:
  observations: null

trainer:
  epochs: 1
  batch_size: 2
  device: cpu
""".lstrip()
    )

    exit_code = main(
        [
            "run",
            "-c",
            str(config_path),
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "cli-run",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "cli-run" / "metrics.json").exists()


def test_cli_lists_models(capsys):
    assert main(["list", "models"]) == 0
    output = capsys.readouterr().out
    assert "gpfa" in output
    assert "kalman" in output
    assert "lfads" in output
    assert "neural_data_transformer" in output


def test_cli_lists_datasets(capsys):
    assert main(["list", "datasets"]) == 0
    output = capsys.readouterr().out
    assert "chaotic_rnn" in output
    assert "lorenz" in output
