"""Experiment config loading."""

from __future__ import annotations

from dataclasses import dataclass

from ladys.datasets import LorenzDatasetConfig
from ladys.models.base import BaseModelConfig
from ladys.preprocessing import PreprocessingConfig
from ladys.training import TrainerConfig
from ladys.utils.yaml import load_yaml


@dataclass
class ExperimentConfig:
    dataset: LorenzDatasetConfig
    model: BaseModelConfig
    trainer: TrainerConfig
    preprocessing: PreprocessingConfig
    batch_size: int = 32
    output_dir: str = "runs"
    run_name: str | None = None
    save_predictions: bool = True


def load_experiment_config(path: str) -> ExperimentConfig:
    """Load dataset, model, and trainer config blocks from YAML."""

    from ladys import models as _models  # noqa: F401

    data = load_yaml(path)
    dataset_name = data["dataset"].get("name")
    if dataset_name != "lorenz":
        raise KeyError(f"Unsupported dataset config '{dataset_name}'.")

    trainer_data = dict(data.get("trainer", {}))
    batch_size = int(trainer_data.pop("batch_size", 32))
    experiment_data = data.get("experiment", {})
    return ExperimentConfig(
        dataset=LorenzDatasetConfig.model_validate(data["dataset"]),
        model=BaseModelConfig.from_dict(data["model"]),
        trainer=TrainerConfig(**trainer_data),
        preprocessing=PreprocessingConfig.model_validate(data.get("preprocessing", {})),
        batch_size=batch_size,
        output_dir=str(experiment_data.get("output_dir", "runs")),
        run_name=experiment_data.get("run_name"),
        save_predictions=bool(experiment_data.get("save_predictions", True)),
    )
