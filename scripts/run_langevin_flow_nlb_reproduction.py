#!/usr/bin/env python3
"""Run LangevinFlow NLB reproductions with upstream hyperparameters.

This runner maps the four public LangevinFlow_CCN training scripts into the
LaDyS experiment stack. It keeps the validation loop active for scheduler
parity, evaluates the full NLB metric suite during training, and exports the
best full-metric checkpoint rather than the final epoch.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.chdir(ROOT)

from ladys.config import ExperimentConfig, load_experiment_config
from ladys.data import build_dataset_config
from ladys.experiment import Experiment, _write_history, _write_json
from ladys.models.base import OptimizationConfig
from ladys.models.langevin_flow import LangevinFlowConfig
from ladys.preprocessing import PreprocessingConfig
from ladys.training import Trainer, TrainerConfig
from ladys.training.strategies import build_strategy
from scripts.run_nlb_classical_table import evaluate_full_nlb, write_full_artifacts


DATASETS = ("area2_bump", "dmfc_rsg", "mc_maze", "mc_rtt")

UPSTREAM_SETTINGS: dict[str, dict[str, Any]] = {
    "area2_bump": {
        "epochs": 2000,
        "batch_size": 32,
        "gamma": 0.55,
        "coordinated_dropout_rate": 0.4,
        "lr": 3.0e-3,
        "weight_decay": 2.0e-5,
    },
    "dmfc_rsg": {
        "epochs": 1000,
        "batch_size": 64,
        "gamma": 0.7,
        "coordinated_dropout_rate": 0.6,
        "lr": 1.0e-3,
        "weight_decay": 1.0e-5,
    },
    "mc_maze": {
        "epochs": 2000,
        "batch_size": 256,
        "gamma": 0.55,
        "coordinated_dropout_rate": 0.5,
        "lr": 3.0e-3,
        "weight_decay": 2.0e-5,
    },
    "mc_rtt": {
        "epochs": 6000,
        "batch_size": 64,
        "gamma": 0.55,
        "coordinated_dropout_rate": 0.3,
        "lr": 3.0e-3,
        "weight_decay": 2.0e-5,
    },
}

REPORTED_CO_BPS = {
    "area2_bump": 0.277,
    "dmfc_rsg": 0.184,
    "mc_maze": 0.362,
    "mc_rtt": 0.190,
}


class StopTraining(Exception):
    """Internal signal used by the epoch callback for early stopping."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument(
        "--config",
        action="append",
        type=Path,
        help="Explicit LangevinFlow experiment YAML. May be passed multiple times.",
    )
    parser.add_argument("--epochs", type=int, help="Override trainer.epochs for every run.")
    parser.add_argument("--batch-size", type=int, help="Override training batch size.")
    parser.add_argument("--bin-size-ms", type=int, choices=(5, 20), default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="runs/langevin_flow_nlb_reproduction")
    parser.add_argument("--target-h5", default="data/real/nlb/eval_data_test.h5")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Train without the per-epoch validation loop, matching run_nlb_classical_table.py.",
    )
    parser.add_argument(
        "--patience-evals",
        type=int,
        default=0,
        help="Stop after this many full-metric evaluations without co-bps improvement.",
    )
    parser.add_argument(
        "--stop-at-reported",
        action="store_true",
        help="Stop a run once co-bps reaches the self-reported LangevinFlow value.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name-suffix", default="upstream_best")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = (
        [
            _load_config(
                path,
                args=args,
                dataset_name=None,
            )
            for path in args.config or []
        ]
        if args.config
        else [
            _build_upstream_config(
                dataset=dataset,
                args=args,
            )
            for dataset in args.datasets
        ]
    )

    rows: list[dict[str, Any]] = []
    for config in configs:
        row = run_config(
            config=config,
            target_h5=Path(args.target_h5),
            eval_every=args.eval_every,
            progress_every=args.progress_every,
            patience_evals=args.patience_evals,
            stop_at_reported=bool(args.stop_at_reported),
            skip_validation=bool(args.skip_validation),
        )
        rows.append(row)
        _write_summary(output_dir, rows)
    _write_summary(output_dir, rows)
    return 0


def _load_config(
    path: Path,
    *,
    args: argparse.Namespace,
    dataset_name: str | None,
) -> ExperimentConfig:
    config = load_experiment_config(str(path))
    if not isinstance(config.model, LangevinFlowConfig):
        raise TypeError(f"{path} is not a LangevinFlow experiment config.")
    dataset = dataset_name or str(config.dataset.name)
    trainer = replace(
        config.trainer,
        device=args.device,
        epochs=int(args.epochs if args.epochs is not None else config.trainer.epochs),
    )
    run_name = config.run_name or f"langevin_flow_{dataset}_nlb_{args.bin_size_ms}ms"
    if args.run_name_suffix:
        run_name = f"{run_name}_{args.run_name_suffix}"
    return replace(
        config,
        trainer=trainer,
        batch_size=int(args.batch_size if args.batch_size is not None else config.batch_size),
        output_dir=str(args.output_dir),
        run_name=run_name,
        save_predictions=True,
    )


def _build_upstream_config(dataset: str, args: argparse.Namespace) -> ExperimentConfig:
    settings = UPSTREAM_SETTINGS[dataset]
    epochs = int(args.epochs if args.epochs is not None else settings["epochs"])
    batch_size = int(args.batch_size if args.batch_size is not None else settings["batch_size"])
    dataset_config = build_dataset_config(
        dataset,
        {
            "name": dataset,
            "split": "test",
            "bin_size_ms": int(args.bin_size_ms),
            "data_path": f"data/real/nlb/{dataset}_test_{int(args.bin_size_ms)}ms.h5",
            "input_mode": "heldin",
            "include_forward": True,
            "seed": int(args.seed),
        },
    )
    model = LangevinFlowConfig(
        hidden_size=280,
        initialization="upstream",
        output_mode="auto",
        fwd_steps=0,
        dropout=0.05,
        gamma=float(settings["gamma"]),
        langevin_step=0.01,
        potential_groups=4,
        potential_kernel_size=3,
        transformer_heads=2,
        transformer_feedforward=512,
        coordinated_dropout_rate=float(settings["coordinated_dropout_rate"]),
        kl_weight=0.1,
        kl_warmup_epochs=500,
        weight_decay_warmup_epochs=500,
        velocity_prior_var=0.1,
        log_rate_min=None,
        log_rate_max=None,
        posterior_logvar_min=None,
        posterior_logvar_max=None,
        sample_train=True,
        sample_eval=True,
        prediction_samples=50,
        optimization=OptimizationConfig(
            name="gradient",
            optimizer="Adam",
            lr=float(settings["lr"]),
            weight_decay=float(settings["weight_decay"]),
            gradient_clip=200.0,
            lr_scheduler="ReduceLROnPlateau",
            scheduler_factor=0.95,
            scheduler_patience=10,
            scheduler_threshold=0.0,
            scheduler_min_lr=1.0e-5,
        ),
    )
    return ExperimentConfig(
        dataset=dataset_config,
        model=model,
        trainer=TrainerConfig(epochs=epochs, device=args.device),
        preprocessing=PreprocessingConfig(observations=None),
        batch_size=batch_size,
        output_dir=str(args.output_dir),
        run_name=f"langevin_flow_{dataset}_nlb_{int(args.bin_size_ms)}ms_{args.run_name_suffix}",
        save_predictions=True,
    )


def run_config(
    *,
    config: ExperimentConfig,
    target_h5: Path,
    eval_every: int,
    progress_every: int,
    patience_evals: int,
    stop_at_reported: bool,
    skip_validation: bool,
) -> dict[str, Any]:
    experiment = Experiment(config)
    experiment._set_seeds()
    experiment.data.setup()
    model = experiment.build_model()
    strategy = build_strategy(config.model.optimization)
    trainer = Trainer(config.trainer)
    device = torch.device(config.trainer.device)
    dataset = str(config.dataset.name)
    run_dir = experiment._make_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "config.json", _config_payload(config))

    best: dict[str, Any] = {
        "epoch": 0,
        "co_bps": float("-inf"),
        "state_dict": None,
        "full": None,
        "metrics": {},
    }
    last_eval: dict[str, Any] | None = None
    evals_since_improvement = 0

    def callback(report) -> None:
        nonlocal evals_since_improvement, last_eval
        epoch = int(report.epoch) + 1
        should_eval = int(eval_every) > 0 and (
            epoch % int(eval_every) == 0 or epoch == int(config.trainer.epochs)
        )
        if progress_every > 0 and epoch % int(progress_every) == 0:
            valid_loss = float("nan") if report.valid is None else report.valid.loss
            print(
                f"langevin_flow {dataset} epoch={epoch} "
                f"train_loss={report.train.loss:.6g} valid_loss={valid_loss:.6g} "
                f"seconds={report.seconds:.3g}",
                flush=True,
            )
        if should_eval:
            full = evaluate_full_nlb(
                model=model,
                data=experiment.data,
                device=device,
                dataset=dataset,
                target_h5=target_h5,
            )
            metrics = full.full_metrics
            report.metrics.update({f"nlb_{key}": float(value) for key, value in metrics.items()})
            co_bps = float(metrics.get("co-bps", float("-inf")))
            last_eval = {
                "epoch": epoch,
                "metrics": dict(metrics),
            }
            print(
                f"langevin_flow {dataset} epoch={epoch} co-bps={co_bps:.6g} "
                f"metrics={metrics}",
                flush=True,
            )
            if co_bps > float(best["co_bps"]):
                best.update(
                    {
                        "epoch": epoch,
                        "co_bps": co_bps,
                        "state_dict": _clone_state_dict_cpu(model.state_dict()),
                        "full": full,
                        "metrics": dict(metrics),
                    }
                )
                torch.save(best["state_dict"], run_dir / "best_model.pt")
                _write_json(
                    run_dir / "best_metrics.json",
                    {
                        "epoch": epoch,
                        "metrics": dict(metrics),
                    },
                )
                write_full_artifacts(run_dir=run_dir, dataset=dataset, full=full)
                evals_since_improvement = 0
            else:
                evals_since_improvement += 1
        _write_history(run_dir / "history.csv", trainer.history)
        _write_epoch_metrics(run_dir / "epoch_nlb_metrics.csv", trainer.history)
        _write_json(
            run_dir / "live_status.json",
            {
                "status": "running",
                "dataset": dataset,
                "epoch": epoch,
                "epochs": int(config.trainer.epochs),
                "run_dir": str(run_dir),
                "last_eval": last_eval,
                "best_eval": {
                    "epoch": int(best["epoch"]),
                    "metrics": dict(best["metrics"]),
                },
            },
        )
        if should_eval and stop_at_reported and co_bps >= REPORTED_CO_BPS.get(dataset, math.inf):
            raise StopTraining
        if should_eval and patience_evals > 0 and evals_since_improvement >= patience_evals:
            raise StopTraining

    try:
        history = trainer.fit(
            model=model,
            strategy=strategy,
            train_loader=experiment.data.train_loader(shuffle=True),
            valid_loader=None if skip_validation else experiment.data.valid_loader(),
            epoch_callback=callback,
        )
    except StopTraining:
        history = trainer.history

    if best["state_dict"] is None:
        full = evaluate_full_nlb(
            model=model,
            data=experiment.data,
            device=device,
            dataset=dataset,
            target_h5=target_h5,
        )
        best.update(
            {
                "epoch": len(history),
                "co_bps": float(full.full_metrics.get("co-bps", float("nan"))),
                "state_dict": _clone_state_dict_cpu(model.state_dict()),
                "full": full,
                "metrics": dict(full.full_metrics),
            }
        )

    model.load_state_dict(best["state_dict"])
    result = experiment._write_artifacts(run_dir, model, history, best["full"].evaluation)
    write_full_artifacts(run_dir=result.run_dir, dataset=dataset, full=best["full"])
    metadata = {
        "dataset": dataset,
        "best_epoch": int(best["epoch"]),
        "best_metrics": dict(best["metrics"]),
        "config_epochs": int(config.trainer.epochs),
        "eval_every": int(eval_every),
        "progress_every": int(progress_every),
        "patience_evals": int(patience_evals),
        "reported_co_bps": REPORTED_CO_BPS.get(dataset),
    }
    _write_json(result.run_dir / "langevin_flow_reproduction.json", metadata)
    _write_json(
        result.run_dir / "live_status.json",
        {
            "status": "complete",
            "dataset": dataset,
            "epoch": len(history),
            "epochs": int(config.trainer.epochs),
            "run_dir": str(result.run_dir),
            "best_eval": {
                "epoch": int(best["epoch"]),
                "metrics": dict(best["metrics"]),
            },
        },
    )
    return {
        "dataset": dataset,
        "run_dir": str(result.run_dir),
        "best_epoch": int(best["epoch"]),
        **{key: float(value) for key, value in best["metrics"].items()},
    }


def _config_payload(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "dataset": config.dataset.model_dump(mode="json"),
        "model": config.model.model_dump(mode="json"),
        "preprocessing": config.preprocessing.model_dump(mode="json"),
        "trainer": {
            "epochs": int(config.trainer.epochs),
            "device": config.trainer.device,
            "live_eval_interval": int(config.trainer.live_eval_interval),
        },
        "batch_size": int(config.batch_size),
        "experiment": {
            "output_dir": config.output_dir,
            "run_name": config.run_name,
            "save_predictions": bool(config.save_predictions),
        },
    }


def _write_epoch_metrics(path: Path, history: list[Any]) -> None:
    keys = sorted({key for report in history for key in report.metrics})
    if not keys:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "seconds", "train_loss", "valid_loss", *keys],
        )
        writer.writeheader()
        for report in history:
            row: dict[str, Any] = {
                "epoch": int(report.epoch) + 1,
                "seconds": float(report.seconds),
                "train_loss": float(report.train.loss),
                "valid_loss": "" if report.valid is None else float(report.valid.loss),
            }
            for key in keys:
                value = report.metrics.get(key, float("nan"))
                row[key] = "" if not math.isfinite(float(value)) else float(value)
            writer.writerow(row)


def _write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = rows
    (output_dir / "summary.json").write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n"
    )
    if not rows:
        return
    keys = [
        "dataset",
        "co-bps",
        "vel R2",
        "tp corr",
        "psth R2",
        "fp-bps",
        "best_epoch",
        "run_dir",
    ]
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _clone_state_dict_cpu(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


if __name__ == "__main__":
    raise SystemExit(main())
