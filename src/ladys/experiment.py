"""Public experiment orchestration API."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch

from ladys.config import ExperimentConfig, load_experiment_config
from ladys.data import DataModule
from ladys.metrics import EvaluationResult, evaluate_model
from ladys.models.base import BaseDynamicsModel
from ladys.training import EpochReport, Trainer
from ladys.training.strategies import build_strategy


@dataclass
class ExperimentResult:
    """Artifacts produced by a completed experiment run."""

    run_dir: Path
    metrics: dict[str, float]
    history: list[EpochReport]
    config_path: Path
    history_path: Path
    metrics_path: Path
    model_path: Path
    report_path: Path
    predictions_path: Path | None = None


class Experiment:
    """Build data, model, training, metrics, and run artifacts from one config."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.data = DataModule(
            config=config.dataset,
            batch_size=config.batch_size,
            preprocessing=config.preprocessing,
        )
        self.model: BaseDynamicsModel | None = None
        self.trainer = Trainer(config.trainer)
        self.result: ExperimentResult | None = None

    @classmethod
    def from_config_path(cls, path: str | Path) -> "Experiment":
        """Create an experiment from a YAML config file."""

        return cls(load_experiment_config(str(path)))

    def build_model(self) -> BaseDynamicsModel:
        """Instantiate the configured model for the prepared data dimensions."""

        if self.data.train_dataset is None:
            self.data.setup()
        self.model = self.config.model.build(
            n_neurons=self.data.n_neurons,
            n_time=self.data.n_time,
        )
        return self.model

    def run(self) -> ExperimentResult:
        """Train the model, evaluate it, and write a self-contained run folder."""

        self._set_seeds()
        self.data.setup()
        model = self.build_model()
        strategy = build_strategy(self.config.model.optimization)
        history = self.trainer.fit(
            model=model,
            strategy=strategy,
            train_loader=self.data.train_loader(),
            valid_loader=self.data.valid_loader(),
        )
        evaluation = evaluate_model(
            model=model,
            loader=self.data.valid_loader(),
            device=self.config.trainer.device,
        )

        run_dir = self._make_run_dir()
        result = self._write_artifacts(run_dir, model, history, evaluation)
        self.result = result
        return result

    def _set_seeds(self) -> None:
        seed = getattr(self.config.dataset, "seed", None)
        if seed is None:
            return
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

    def _make_run_dir(self) -> Path:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        default_name = f"{timestamp}_{self.config.dataset.name}_{self.config.model.name}"
        run_name = self.config.run_name or default_name
        return _create_unique_dir(output_dir / _slugify(run_name))

    def _write_artifacts(
        self,
        run_dir: Path,
        model: BaseDynamicsModel,
        history: list[EpochReport],
        evaluation: EvaluationResult,
    ) -> ExperimentResult:
        config_path = run_dir / "config.json"
        history_path = run_dir / "history.csv"
        metrics_path = run_dir / "metrics.json"
        model_path = run_dir / "model.pt"
        report_path = run_dir / "report.md"
        predictions_path = run_dir / "predictions.npz" if self.config.save_predictions else None

        _write_json(config_path, experiment_config_to_dict(self.config))
        _write_history(history_path, history)
        _write_json(metrics_path, _json_ready(evaluation.metrics))
        torch.save(model.state_dict(), model_path)
        if predictions_path is not None:
            _write_predictions(predictions_path, evaluation)
        _write_report(report_path, self.config, history, evaluation.metrics)

        return ExperimentResult(
            run_dir=run_dir,
            metrics=evaluation.metrics,
            history=history,
            config_path=config_path,
            history_path=history_path,
            metrics_path=metrics_path,
            model_path=model_path,
            report_path=report_path,
            predictions_path=predictions_path,
        )


def experiment_config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Return a JSON-serializable experiment config snapshot."""

    return {
        "dataset": config.dataset.model_dump(mode="json"),
        "model": config.model.model_dump(mode="json"),
        "preprocessing": config.preprocessing.model_dump(mode="json"),
        "trainer": asdict(config.trainer),
        "batch_size": config.batch_size,
        "experiment": {
            "output_dir": config.output_dir,
            "run_name": config.run_name,
            "save_predictions": config.save_predictions,
        },
    }


def _write_history(path: Path, history: list[EpochReport]) -> None:
    fieldnames = [
        "epoch",
        "seconds",
        "train_loss",
        "valid_loss",
        "objective",
        "metrics",
        "train_metrics",
        "valid_metrics",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for report in history:
            writer.writerow(
                {
                    "epoch": report.epoch + 1,
                    "seconds": report.seconds,
                    "train_loss": report.train.loss,
                    "valid_loss": None if report.valid is None else report.valid.loss,
                    "objective": report.train.objective,
                    "metrics": json.dumps(_json_ready(report.metrics), sort_keys=True),
                    "train_metrics": json.dumps(
                        _json_ready(report.train.metrics),
                        sort_keys=True,
                    ),
                    "valid_metrics": json.dumps(
                        _json_ready({} if report.valid is None else report.valid.metrics),
                        sort_keys=True,
                    ),
                }
            )


def _write_predictions(path: Path, evaluation: EvaluationResult) -> None:
    arrays = {}
    arrays.update({f"pred_{key}": value for key, value in evaluation.predictions.items()})
    arrays.update({f"target_{key}": value for key, value in evaluation.targets.items()})
    np.savez_compressed(path, **arrays)


def _write_report(
    path: Path,
    config: ExperimentConfig,
    history: list[EpochReport],
    metrics: dict[str, float],
) -> None:
    final = history[-1] if history else None
    lines = [
        "# LaDyS Experiment Report",
        "",
        f"- Dataset: `{config.dataset.name}`",
        f"- Model: `{config.model.name}`",
        f"- Epochs: `{config.trainer.epochs}`",
        f"- Batch size: `{config.batch_size}`",
        f"- Device: `{config.trainer.device}`",
        "",
        "## Final Training State",
        "",
    ]
    if final is None:
        lines.append("No epochs were run.")
    else:
        valid_loss = "nan" if final.valid is None else f"{final.valid.loss:.6g}"
        lines.extend(
            [
                f"- Train loss: `{final.train.loss:.6g}`",
                f"- Validation loss: `{valid_loss}`",
                f"- Objective: `{final.train.objective}`",
            ]
        )

    lines.extend(["", "## Evaluation Metrics", ""])
    if metrics:
        for key, value in sorted(metrics.items()):
            display = "nan" if not math.isfinite(value) else f"{value:.6g}"
            lines.append(f"- `{key}`: `{display}`")
    else:
        lines.append("No compatible evaluation metrics were available.")

    path.write_text("\n".join(lines) + "\n")


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(_json_ready(data), indent=2, sort_keys=True) + "\n")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return slug or "run"


def _create_unique_dir(path: Path) -> Path:
    if not path.exists():
        path.mkdir(parents=False)
        return path

    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.name}-{index}")
        if not candidate.exists():
            candidate.mkdir(parents=False)
            return candidate
    raise RuntimeError(f"Could not create a unique run directory for {path}.")
