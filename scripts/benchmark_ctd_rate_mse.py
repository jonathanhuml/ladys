"""Benchmark CTD generated-data H5 files with the LaDyS model registry.

The CTD loader exposes ``*_activity`` as true rates, so this script reports
mean squared error between model-predicted rates and those latent simulator
rates. By default it runs the lower-count CTD generated configs:

- NBFF, MultiTask, RandomTarget: 60 neurons
- ChaoticDelayedMatching: 96 neurons

PhaseCodedMemory is also available via ``--datasets all`` or by naming
``ctd_phase_coded_memory`` explicitly, but it is not a lower-count default.

Example:
    PYTHONPATH=src python3 scripts/benchmark_ctd_rate_mse.py \
        --models all --datasets lower --epochs 1 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile
import time
import traceback
from typing import Any

_CACHE_ROOT = Path(tempfile.gettempdir())
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "ladys_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "ladys_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
from torch.utils.data import DataLoader

from ladys.data import build_dataset_config, make_dataset_splits
from ladys.datasets import CTDDatasetConfig
from ladys.metrics import evaluate_model
from ladys.models import (
    BGPFAConfig,
    CASSMConfig,
    GPFAConfig,
    ILQRVAEConfig,
    KalmanConfig,
    LangevinFlowConfig,
    LFADSConfig,
    MINTConfig,
    NDTConfig,
    PSTHConfig,
    SmoothingConfig,
    STNDTConfig,
)
from ladys.models.base import BaseModelConfig
from ladys.preprocessing import PreprocessedDataset, PreprocessingConfig
from ladys.training import Trainer, TrainerConfig
from ladys.training.strategies import build_strategy
from ladys.utils.yaml import load_yaml

from benchmark_lorenz_loss_curves import evaluate_poisson_nll, evaluate_rate_mse


LOWER_CTD_DATASETS = [
    "ctd_nbff",
    "ctd_multitask",
    "ctd_random_target",
    "ctd_chaotic_delayed_matching",
]
ALL_CTD_DATASETS = LOWER_CTD_DATASETS + ["ctd_phase_coded_memory"]
ALL_MODELS = [
    "bgpfa",
    "cassm",
    "gpfa",
    "ilqr_vae",
    "kalman",
    "langevin_flow",
    "lfads",
    "mint",
    "ndt",
    "stndt",
    "psth",
    "smoothing",
]
INFERENCE_ONLY_MODELS = {"psth", "smoothing"}
MODEL_CONFIGS = {
    "bgpfa": BGPFAConfig,
    "cassm": CASSMConfig,
    "gpfa": GPFAConfig,
    "ilqr_vae": ILQRVAEConfig,
    "kalman": KalmanConfig,
    "langevin_flow": LangevinFlowConfig,
    "lfads": LFADSConfig,
    "mint": MINTConfig,
    "ndt": NDTConfig,
    "psth": PSTHConfig,
    "smoothing": SmoothingConfig,
    "stndt": STNDTConfig,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["lower"],
        help="CTD dataset names. Use 'lower' for 60/96-neuron configs or 'all'.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Models to run. Use 'all' for every registered LaDyS method.",
    )
    parser.add_argument("--dataset-config-dir", default="configs/dataset")
    parser.add_argument("--experiment-config-dir", default="configs/experiment")
    parser.add_argument("--output-dir", default="runs/ctd_rate_mse")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--trainer-config-source",
        choices=["cli", "lorenz"],
        default="cli",
        help="Use fixed CLI epochs/batch-size or per-model Lorenz trainer YAML values.",
    )
    parser.add_argument(
        "--model-config-source",
        choices=["ctd", "lorenz"],
        default="ctd",
        help="Use CTD experiment YAMLs when present, or fall back to Lorenz model YAMLs.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Record an error instead of falling back when a CUDA device is unavailable.",
    )
    parser.add_argument(
        "--preprocessing-mode",
        choices=["model", "none"],
        default="model",
        help="Observation preprocessing: 'model' uses experiment YAMLs when present.",
    )
    parser.add_argument(
        "--max-train-trials",
        type=int,
        default=None,
        help="Optional train split cap for quick lower-cost checks.",
    )
    parser.add_argument(
        "--max-valid-trials",
        type=int,
        default=None,
        help="Optional validation split cap for quick lower-cost checks.",
    )
    parser.add_argument("--cassm-projection-dim", type=int, default=None)
    parser.add_argument("--cassm-lr", type=float, default=None)
    parser.add_argument("--gpfa-latent-dim", type=int, default=3)
    parser.add_argument(
        "--gpfa-init-method",
        choices=["fa", "normal", "kaiming", "kaiming_normal", "kaiming_uniform"],
        default="kaiming_normal",
    )
    parser.add_argument("--gpfa-init-seed", type=int, default=None)
    parser.add_argument("--bgpfa-infer-steps", type=int, default=300)
    parser.add_argument("--bgpfa-infer-mc", type=int, default=20)
    parser.add_argument("--bgpfa-infer-lr", type=float, default=1e-1)
    parser.add_argument("--ilqr-max-iter", type=int, default=5)
    parser.add_argument("--mint-n-candidates", type=int, default=None)
    parser.add_argument("--mint-window-length", type=int, default=None)
    parser.add_argument("--mint-delta", type=int, default=None)
    parser.add_argument("--mint-sigma", type=int, default=None)
    parser.add_argument("--mint-causal", action="store_true")
    parser.add_argument(
        "--mint-library-source",
        choices=["smoothed_spikes", "true_rates"],
        default="smoothed_spikes",
        help="'true_rates' is an oracle/debug mode, not a fair comparison.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore existing summary/history CSVs.",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry cases that have existing error rows.",
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        default=None,
        help="Run at most this many pending dataset/model cases.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.datasets = expand_datasets(args.datasets)
    args.models = expand_models(args.models)
    validate_models(args.models)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(output_dir / "run_config.json", args)

    summary_path = output_dir / "summary.csv"
    history_path = output_dir / "history.csv"
    summary_rows = [] if args.overwrite else read_csv(summary_path)
    history_rows = [] if args.overwrite else read_csv(history_path)
    completed = completed_keys(summary_rows, retry_errors=args.retry_errors)

    cases = [
        (dataset_name, model_name)
        for dataset_name in args.datasets
        for model_name in args.models
        if (dataset_name, model_name, int(args.seed)) not in completed
    ]
    if args.stop_after is not None:
        cases = cases[: max(int(args.stop_after), 0)]

    write_outputs(output_dir, summary_rows, history_rows)
    if not cases:
        print(f"No pending CTD benchmark cases. Outputs are current in {output_dir}.")
        return

    print(
        "Running "
        f"{len(cases)} CTD cases on device={args.device}: "
        f"datasets={args.datasets}, models={args.models}, "
        f"trainer_config_source={args.trainer_config_source}"
    )
    run_started = time.perf_counter()
    for index, (dataset_name, model_name) in enumerate(cases, start=1):
        write_heartbeat(
            output_dir,
            {
                "event": "case_start",
                "dataset": dataset_name,
                "model": model_name,
                "case_index": index,
                "total_cases": len(cases),
                "seed": int(args.seed),
                "elapsed_seconds": time.perf_counter() - run_started,
            },
        )
        result = run_case(args, dataset_name, model_name)
        summary_rows = replace_summary(summary_rows, result["summary"])
        history_rows = replace_history(history_rows, result["history"], result["summary"])
        write_outputs(output_dir, summary_rows, history_rows)
        if result["summary"].get("status") == "error":
            write_error_log(output_dir, result["summary"], result.get("traceback", ""))
        print(progress_line(result["summary"], index, len(cases), run_started))
        write_heartbeat(
            output_dir,
            {
                "event": "case_finish",
                "dataset": dataset_name,
                "model": model_name,
                "case_index": index,
                "total_cases": len(cases),
                "seed": int(args.seed),
                "status": result["summary"].get("status"),
                "best_rate_mse": result["summary"].get("best_rate_mse"),
                "elapsed_seconds": time.perf_counter() - run_started,
            },
        )

    write_outputs(output_dir, summary_rows, history_rows)
    print(f"Wrote {summary_path}")
    print(f"Wrote {history_path}")
    print(f"Wrote {output_dir / 'summary.md'}")


def run_case(args: argparse.Namespace, dataset_name: str, model_name: str) -> dict[str, Any]:
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    try:
        if args.require_cuda and str(args.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was required but torch.cuda.is_available() is false.")

        dataset_config = load_ctd_dataset_config(args, dataset_name)
        train_ds, valid_ds = make_dataset_splits(dataset_config)
        n_time, n_neurons = train_ds.spikes.shape[1:]
        case_trainer = resolve_trainer_config(
            args=args,
            dataset_name=dataset_name,
            model_name=model_name,
            n_neurons=int(n_neurons),
        )
        preprocessing = build_preprocessing_config(
            model_name=model_name,
            config_dir=args.experiment_config_dir,
            preprocessing_mode=args.preprocessing_mode,
            dataset_name=dataset_name,
            n_neurons=int(n_neurons),
            fallback_dataset=fallback_experiment_dataset(args),
        )
        train_ds = PreprocessedDataset(train_ds, preprocessing)
        valid_ds = PreprocessedDataset(valid_ds, preprocessing)
        train_loader = DataLoader(train_ds, batch_size=case_trainer["batch_size"], shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=case_trainer["batch_size"], shuffle=False)

        model_config = build_model_config(args, model_name, dataset_name, int(n_neurons))
        model = model_config.build(n_neurons=int(n_neurons), n_time=int(n_time))

        if model_name == "mint":
            history = run_mint_case(
                args=args,
                model=model,
                model_name=model_name,
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                train_ds=train_ds,
                train_loader=train_loader,
                valid_loader=valid_loader,
                epochs=case_trainer["epochs"],
                output_dir=Path(args.output_dir),
            )
        elif model_name in INFERENCE_ONLY_MODELS or int(case_trainer["epochs"]) <= 0:
            history = run_inference_only_case(
                args=args,
                model=model,
                model_name=model_name,
                dataset_name=dataset_name,
                train_loader=train_loader,
                valid_loader=valid_loader,
                epochs=case_trainer["epochs"],
            )
        else:
            history = run_trained_case(
                args=args,
                model=model,
                model_name=model_name,
                dataset_name=dataset_name,
                model_config=model_config,
                train_loader=train_loader,
                valid_loader=valid_loader,
                epochs=case_trainer["epochs"],
                output_dir=Path(args.output_dir),
            )

        extra_metrics = evaluate_extra_metrics(model, valid_loader, train_loader, args.device)
        summary = summarize_case(
            args=args,
            dataset_name=dataset_name,
            model_name=model_name,
            status="ok",
            history_rows=history,
            dataset_config=dataset_config,
            train_trials=len(train_ds),
            valid_trials=len(valid_ds),
            n_neurons=int(n_neurons),
            n_time=int(n_time),
            epochs=case_trainer["epochs"],
            batch_size=case_trainer["batch_size"],
            trainer_config_path=case_trainer["path"],
            extra_metrics=extra_metrics,
        )
        cleanup_cuda()
        return {"summary": summary, "history": history, "traceback": ""}
    except Exception as exc:
        cleanup_cuda()
        return error_result(args, dataset_name, model_name, exc)


def load_ctd_dataset_config(args: argparse.Namespace, dataset_name: str) -> CTDDatasetConfig:
    path = Path(args.dataset_config_dir) / f"{dataset_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing CTD dataset config: {path}")
    data = dict(load_yaml(path))
    data.pop("name", None)
    if args.max_train_trials is not None:
        data["max_train_trials"] = int(args.max_train_trials)
    if args.max_valid_trials is not None:
        data["max_valid_trials"] = int(args.max_valid_trials)
    config = build_dataset_config(dataset_name, data)
    if not isinstance(config, CTDDatasetConfig):
        raise TypeError(f"{dataset_name} did not build a CTDDatasetConfig.")
    return config


def build_model_config(
    args: argparse.Namespace,
    model_name: str,
    dataset_name: str,
    n_neurons: int,
) -> BaseModelConfig:
    model_data = load_model_data(
        config_dir=args.experiment_config_dir,
        dataset_name=dataset_name,
        model_name=model_name,
        n_neurons=n_neurons,
        fallback_dataset=fallback_experiment_dataset(args),
    )

    if model_name == "psth":
        return BaseModelConfig.from_dict(model_data) if model_data else PSTHConfig()
    if model_name == "smoothing":
        return BaseModelConfig.from_dict(model_data) if model_data else SmoothingConfig()
    if model_name == "cassm":
        projection_dim = args.cassm_projection_dim
        if projection_dim is None:
            configured = 20 if model_data is None else int(model_data.get("projection_dim", 20))
            projection_dim = choose_divisor_at_most(n_neurons, min(configured, n_neurons))
        if n_neurons % int(projection_dim) != 0:
            raise ValueError(
                "CASSM sparse projection requires neurons to be divisible by "
                f"projection_dim; got neurons={n_neurons}, projection_dim={projection_dim}."
            )
        if model_data is None:
            if args.cassm_lr is None:
                return CASSMConfig(projection_dim=int(projection_dim))
            return CASSMConfig(
                projection_dim=int(projection_dim),
                optimization={
                    "name": "gradient",
                    "optimizer": "Adam",
                    "lr": float(args.cassm_lr),
                    "weight_decay": 0.0,
                    "gradient_clip": 300.0,
                },
            )
        model_data["projection_dim"] = int(projection_dim)
        if args.cassm_lr is not None:
            optimization = dict(model_data.get("optimization", {}))
            optimization["lr"] = float(args.cassm_lr)
            model_data["optimization"] = optimization
        return BaseModelConfig.from_dict(model_data)
    if model_name == "gpfa":
        init_seed = args.seed if args.gpfa_init_seed is None else args.gpfa_init_seed
        if model_data is None:
            return GPFAConfig(
                latent_dim=int(args.gpfa_latent_dim),
                init_method=args.gpfa_init_method,
                init_seed=int(init_seed),
            )
        model_data["latent_dim"] = int(args.gpfa_latent_dim)
        model_data["init_method"] = args.gpfa_init_method
        model_data["init_seed"] = int(init_seed)
        return BaseModelConfig.from_dict(model_data)
    if model_name == "kalman":
        return BaseModelConfig.from_dict(model_data) if model_data else KalmanConfig()
    if model_name == "bgpfa":
        return BaseModelConfig.from_dict(model_data) if model_data else BGPFAConfig()
    if model_name == "ilqr_vae":
        if model_data is not None:
            model_data["held_in_neurons"] = n_neurons
            model_data["output_neuron_start"] = 0
            model_data["output_neurons"] = n_neurons
            model_data["max_iter"] = int(args.ilqr_max_iter)
            return BaseModelConfig.from_dict(model_data)
        return ILQRVAEConfig(
            objective="ilqr_vae_elbo",
            params_path=None,
            initialization="random",
            trainable_parameters=True,
            max_iter=int(args.ilqr_max_iter),
            held_in_neurons=n_neurons,
            output_neuron_start=0,
            output_neurons=n_neurons,
            dt=1.0,
            optimization={
                "name": "gradient",
                "optimizer": "Adam",
                "lr": 4e-3,
                "weight_decay": 0.0,
                "gradient_clip": 200.0,
            },
        )
    if model_name == "mint":
        # MINT has no CTD-specific preset. The Lorenz synthetic path is the
        # generic repeated-trajectory library builder, and the CTD rates still
        # remain the held-out evaluation target.
        model_data = {"name": "mint", "dataset": "lorenz"} if model_data is None else model_data
        model_data["dataset"] = "lorenz"
        model_data["lorenz_library_source"] = args.mint_library_source
        if args.mint_n_candidates is not None:
            model_data["n_candidates"] = int(args.mint_n_candidates)
        if args.mint_window_length is not None:
            model_data["window_length"] = int(args.mint_window_length)
        if args.mint_delta is not None:
            model_data["delta"] = int(args.mint_delta)
        if args.mint_sigma is not None:
            model_data["sigma"] = int(args.mint_sigma)
        if args.mint_causal:
            model_data["causal"] = True
        return BaseModelConfig.from_dict(model_data)
    if model_name == "lfads":
        return BaseModelConfig.from_dict(model_data) if model_data else LFADSConfig()
    if model_name in {"langevin_flow", "ndt", "stndt"}:
        return BaseModelConfig.from_dict(model_data) if model_data else MODEL_CONFIGS[model_name]()
    raise KeyError(model_name)


def run_inference_only_case(
    args: argparse.Namespace,
    model,
    model_name: str,
    dataset_name: str,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    epochs: int,
) -> list[dict[str, Any]]:
    sync_device(args.device)
    started = time.perf_counter()
    evaluation = evaluate_model(
        model=model,
        loader=valid_loader,
        device=args.device,
        train_loader=train_loader,
    )
    train_loss = evaluate_poisson_nll(model, train_loader, args.device)
    test_loss = evaluate_poisson_nll(model, valid_loader, args.device)
    sync_device(args.device)
    elapsed = time.perf_counter() - started
    n_time, n_neurons = dataset_shape(train_loader)
    return [
        {
            "status": "ok",
            "dataset": dataset_name,
            "model": model_name,
            "neurons": n_neurons,
            "trials": len(train_loader.dataset),
            "num_steps": n_time,
            "seed": int(args.seed),
            "epoch": max(1, int(epochs)),
            "optimizer_seconds": elapsed,
            "cumulative_optimizer_seconds": elapsed,
            "wall_seconds": elapsed,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_rate_mse": evaluation.metrics.get("rate_mse", np.nan),
            "objective": getattr(model, "objective", "inference_only"),
            "error": "",
        }
    ]


def run_trained_case(
    args: argparse.Namespace,
    model,
    model_name: str,
    dataset_name: str,
    model_config: BaseModelConfig,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    epochs: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    strategy = build_strategy(model_config.optimization)
    trainer = Trainer(TrainerConfig(epochs=int(epochs), device=args.device))
    metric_fns = {
        "test_rate_mse": lambda current_model: evaluate_rate_mse(
            current_model,
            valid_loader,
            args.device,
            bgpfa_infer_steps=int(args.bgpfa_infer_steps),
            bgpfa_infer_mc=int(args.bgpfa_infer_mc),
            bgpfa_infer_lr=float(args.bgpfa_infer_lr),
        )
    }

    sync_device(args.device)
    started = time.perf_counter()
    validation_loader = None if model_name == "bgpfa" else valid_loader
    reports = trainer.fit(
        model,
        strategy,
        train_loader,
        validation_loader,
        metric_fns,
        epoch_callback=lambda report: write_heartbeat(
            output_dir,
            {
                "event": "epoch_finish",
                "dataset": dataset_name,
                "model": model_name,
                "seed": int(args.seed),
                "epoch": int(report.epoch) + 1,
                "epochs": int(epochs),
                "train_loss": report.train.loss,
                "test_loss": np.nan if report.valid is None else report.valid.loss,
                "test_rate_mse": report.metrics.get("test_rate_mse", np.nan),
                "elapsed_seconds": time.perf_counter() - started,
            },
        ),
    )
    sync_device(args.device)
    wall_seconds = time.perf_counter() - started

    n_time, n_neurons = dataset_shape(train_loader)
    rows = []
    cumulative_optimizer_seconds = 0.0
    for report in reports:
        cumulative_optimizer_seconds += float(report.seconds)
        rows.append(
            {
                "status": "ok",
                "dataset": dataset_name,
                "model": model_name,
                "neurons": n_neurons,
                "trials": len(train_loader.dataset),
                "num_steps": n_time,
                "seed": int(args.seed),
                "epoch": report.epoch + 1,
                "optimizer_seconds": float(report.seconds),
                "cumulative_optimizer_seconds": cumulative_optimizer_seconds,
                "wall_seconds": wall_seconds,
                "train_loss": report.train.loss,
                "test_loss": np.nan if report.valid is None else report.valid.loss,
                "test_rate_mse": report.metrics.get("test_rate_mse", np.nan),
                "objective": report.train.objective,
                "error": "",
            }
        )
    return rows


def run_mint_case(
    args: argparse.Namespace,
    model,
    model_name: str,
    dataset_name: str,
    dataset_config: CTDDatasetConfig,
    train_ds: PreprocessedDataset,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    epochs: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    rows = []
    repeat_counts = mint_epoch_trial_counts(train_ds, int(epochs))
    cumulative_optimizer_seconds = 0.0
    sync_device(args.device)
    started = time.perf_counter()
    for epoch, max_trials in enumerate(repeat_counts, start=1):
        epoch_started = time.perf_counter()
        fit_mint_ctd_library(
            model=model,
            dataset=train_ds,
            dataset_config=dataset_config,
            device=args.device,
            max_trials=max_trials,
        )
        sync_device(args.device)
        optimizer_seconds = time.perf_counter() - epoch_started
        cumulative_optimizer_seconds += optimizer_seconds
        train_loss = evaluate_poisson_nll(model, train_loader, args.device, use_raw_spikes=True)
        test_loss = evaluate_poisson_nll(model, valid_loader, args.device, use_raw_spikes=True)
        test_rate_mse = evaluate_rate_mse(
            model,
            valid_loader,
            args.device,
            use_raw_spikes=True,
            bgpfa_infer_steps=int(args.bgpfa_infer_steps),
            bgpfa_infer_mc=int(args.bgpfa_infer_mc),
            bgpfa_infer_lr=float(args.bgpfa_infer_lr),
        )
        n_time, n_neurons = dataset_shape(train_loader)
        rows.append(
            {
                "status": "ok",
                "dataset": dataset_name,
                "model": model_name,
                "neurons": n_neurons,
                "trials": len(train_loader.dataset),
                "num_steps": n_time,
                "seed": int(args.seed),
                "epoch": epoch,
                "optimizer_seconds": optimizer_seconds,
                "cumulative_optimizer_seconds": cumulative_optimizer_seconds,
                "wall_seconds": time.perf_counter() - started,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "test_rate_mse": test_rate_mse,
                "objective": "mint_poisson_nll",
                "error": "",
            }
        )
        write_heartbeat(
            output_dir,
            {
                "event": "epoch_finish",
                "dataset": dataset_name,
                "model": model_name,
                "seed": int(args.seed),
                "epoch": epoch,
                "epochs": int(epochs),
                "train_loss": train_loss,
                "test_loss": test_loss,
                "test_rate_mse": test_rate_mse,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
    return rows


def fit_mint_ctd_library(
    model,
    dataset: PreprocessedDataset,
    dataset_config: CTDDatasetConfig,
    device: str,
    max_trials: int | None = None,
) -> None:
    if not hasattr(model, "fit_library"):
        raise TypeError(f"{type(model).__name__} does not expose fit_library().")

    torch_device = torch.device(device)
    model.to(torch_device)
    spikes = getattr(dataset, "raw_spikes", dataset.spikes)
    rates = getattr(dataset, "rates", None)
    latents = getattr(dataset, "latents", None)
    n_trials, n_time, _ = spikes.shape
    if max_trials is not None:
        n_trials = min(int(max_trials), int(n_trials))
        spikes = spikes[:n_trials]
        if rates is not None:
            rates = rates[:n_trials]
        if latents is not None:
            latents = latents[:n_trials]
    if n_trials <= 0:
        raise ValueError("MINT CTD fitting received an empty training split.")

    library_source = getattr(getattr(model, "config", None), "lorenz_library_source", "smoothed_spikes")
    if library_source == "true_rates":
        if rates is None:
            raise AttributeError("MINT CTD true_rates fitting requires CTD train rates.")
        z_source = rates
    elif latents is not None:
        z_source = latents
    else:
        z_source = spikes

    if hasattr(model, "settings"):
        model.settings.Ts = float(dataset_config.dt)
        model.settings.trial_alignment = range(0, int(n_time))
        model.settings.test_alignment = range(0, int(n_time))
    if hasattr(model, "hyperparams"):
        model.hyperparams.trajectories_alignment = range(0, int(n_time))
        n_candidates = int(getattr(model.hyperparams, "n_candidates", 1))
        model.hyperparams.n_candidates = min(max(n_candidates, 1), int(n_trials))
        if model.hyperparams.n_candidates < 2:
            model.hyperparams.interp = 1

    spike_trials = [spikes[i].T.contiguous().to(torch_device) for i in range(n_trials)]
    z_trials = [
        z_source[i].T.contiguous().to(device=torch_device, dtype=torch.float64)
        for i in range(n_trials)
    ]
    # CTD generated data has no repeated condition labels in the exported H5.
    # One trial per library condition gives MINT a non-oracle template library.
    condition = np.arange(n_trials, dtype=np.int64)
    model.fit_library(spike_trials, z_trials, condition)


def mint_epoch_trial_counts(dataset: PreprocessedDataset, requested_epochs: int) -> list[int]:
    total_trials = len(dataset)
    if total_trials <= 0:
        raise ValueError("MINT CTD fitting received an empty training split.")
    n_epochs = max(1, int(requested_epochs))
    if n_epochs == 1:
        return [total_trials]
    return np.unique(np.linspace(1, total_trials, n_epochs, dtype=np.int64)).tolist()


def build_preprocessing_config(
    model_name: str,
    config_dir: str,
    preprocessing_mode: str,
    dataset_name: str,
    n_neurons: int | None = None,
    fallback_dataset: str | None = None,
) -> PreprocessingConfig:
    if preprocessing_mode == "none":
        return PreprocessingConfig()
    path = find_experiment_config_path(
        config_dir,
        dataset_name,
        model_name,
        n_neurons,
        fallback_dataset=fallback_dataset,
    )
    if path is None:
        return PreprocessingConfig()
    data = load_yaml(path)
    return PreprocessingConfig.model_validate(data.get("preprocessing", {}))


def load_model_data(
    config_dir: str,
    dataset_name: str,
    model_name: str,
    n_neurons: int | None = None,
    fallback_dataset: str | None = None,
) -> dict[str, Any] | None:
    path = find_experiment_config_path(
        config_dir,
        dataset_name,
        model_name,
        n_neurons,
        fallback_dataset=fallback_dataset,
    )
    if path is None:
        return None
    data = load_yaml(path)
    return dict(data["model"] if "model" in data else data)


def resolve_trainer_config(
    args: argparse.Namespace,
    dataset_name: str,
    model_name: str,
    n_neurons: int | None = None,
) -> dict[str, Any]:
    if args.trainer_config_source == "cli":
        return {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "path": "",
        }
    path = find_experiment_config_path(
        args.experiment_config_dir,
        dataset_name,
        model_name,
        n_neurons,
        fallback_dataset="lorenz",
    )
    if path is None:
        return {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "path": "",
        }
    data = load_yaml(path)
    trainer = dict(data.get("trainer", {}))
    return {
        "epochs": int(trainer.get("epochs", args.epochs)),
        "batch_size": int(trainer.get("batch_size", args.batch_size)),
        "path": str(path),
    }


def fallback_experiment_dataset(args: argparse.Namespace) -> str | None:
    if args.model_config_source == "lorenz":
        return "lorenz"
    return None


def experiment_config_path(
    config_dir: str,
    dataset_name: str,
    model_name: str,
    n_neurons: int | None = None,
) -> Path:
    candidates = experiment_config_candidates(config_dir, dataset_name, model_name, n_neurons)
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def find_experiment_config_path(
    config_dir: str,
    dataset_name: str,
    model_name: str,
    n_neurons: int | None = None,
    fallback_dataset: str | None = None,
) -> Path | None:
    datasets = [dataset_name]
    if fallback_dataset is not None and fallback_dataset not in datasets:
        datasets.append(fallback_dataset)
    for candidate_dataset in datasets:
        for path in experiment_config_candidates(
            config_dir,
            candidate_dataset,
            model_name,
            n_neurons,
        ):
            if path.exists():
                return path
    return None


def experiment_config_candidates(
    config_dir: str,
    dataset_name: str,
    model_name: str,
    n_neurons: int | None = None,
) -> list[Path]:
    root = Path(config_dir)
    candidates = []
    if n_neurons is not None:
        candidates.append(
            root
            / "synthetic"
            / dataset_name
            / model_name
            / f"{model_name}_{dataset_name}_{n_neurons}.yaml"
        )
    candidates.extend(
        [
            root / "synthetic" / dataset_name / model_name / f"{model_name}_{dataset_name}.yaml",
            root / "synthetic" / dataset_name / model_name / f"{model_name}_{dataset_name}_100.yaml",
            root / dataset_name / model_name / f"{model_name}_{dataset_name}.yaml",
            root / dataset_name / model_name / f"{model_name}_{dataset_name}_100.yaml",
            root / f"{model_name}_{dataset_name}.yaml",
            root / f"{model_name}_{dataset_name}_100.yaml",
        ]
    )
    return candidates


def evaluate_extra_metrics(
    model,
    valid_loader: DataLoader,
    train_loader: DataLoader,
    device: str,
) -> dict[str, float]:
    try:
        return evaluate_model(
            model=model,
            loader=valid_loader,
            device=device,
            train_loader=train_loader,
        ).metrics
    except Exception:
        return {}


def summarize_case(
    args: argparse.Namespace,
    dataset_name: str,
    model_name: str,
    status: str,
    history_rows: list[dict[str, Any]],
    dataset_config: CTDDatasetConfig | None,
    train_trials: int,
    valid_trials: int,
    n_neurons: int,
    n_time: int,
    epochs: int | None = None,
    batch_size: int | None = None,
    trainer_config_path: str = "",
    extra_metrics: dict[str, float] | None = None,
    error: str = "",
) -> dict[str, Any]:
    extra_metrics = extra_metrics or {}
    ok_rows = [row for row in history_rows if row.get("status") == "ok"]
    final = ok_rows[-1] if ok_rows else {}
    best = best_mse_row(ok_rows)
    metadata = dataset_config.metadata if dataset_config is not None else {}
    total_neurons = metadata.get("total_neurons", n_neurons)
    total_trials = metadata.get("total_trials", train_trials + valid_trials)
    return {
        "status": status,
        "dataset": dataset_name,
        "task": "" if dataset_config is None else dataset_config.task,
        "model": model_name,
        "neurons": n_neurons,
        "configured_total_neurons": total_neurons,
        "configured_total_trials": total_trials,
        "train_trials": train_trials,
        "valid_trials": valid_trials,
        "num_steps": n_time,
        "seed": int(args.seed),
        "epochs": int(args.epochs if epochs is None else epochs),
        "batch_size": int(args.batch_size if batch_size is None else batch_size),
        "trainer_config_path": trainer_config_path,
        "best_epoch": best.get("epoch", ""),
        "final_epoch": final.get("epoch", ""),
        "final_rate_mse": final.get("test_rate_mse", np.nan),
        "best_rate_mse": best.get("test_rate_mse", np.nan),
        "convergence_seconds": best.get("cumulative_optimizer_seconds", np.nan),
        "total_optimizer_seconds": final.get("cumulative_optimizer_seconds", np.nan),
        "wall_seconds": final.get("wall_seconds", np.nan),
        "train_loss": final.get("train_loss", np.nan),
        "test_loss": final.get("test_loss", np.nan),
        "rate_r2": extra_metrics.get("rate_r2", np.nan),
        "co_bps": extra_metrics.get("co_bps", np.nan),
        "poisson_nll": extra_metrics.get("poisson_nll", np.nan),
        "latent_linear_r2": extra_metrics.get("latent_linear_r2", np.nan),
        "error": error,
    }


def error_result(
    args: argparse.Namespace,
    dataset_name: str,
    model_name: str,
    exc: BaseException,
) -> dict[str, Any]:
    tb = traceback.format_exc()
    error = f"{type(exc).__name__}: {exc}"
    history = [
        {
            "status": "error",
            "dataset": dataset_name,
            "model": model_name,
            "neurons": 0,
            "trials": 0,
            "num_steps": 0,
            "seed": int(args.seed),
            "epoch": -1,
            "optimizer_seconds": np.nan,
            "cumulative_optimizer_seconds": np.nan,
            "wall_seconds": np.nan,
            "train_loss": np.nan,
            "test_loss": np.nan,
            "test_rate_mse": np.nan,
            "objective": "",
            "error": error,
        }
    ]
    summary = summarize_case(
        args=args,
        dataset_name=dataset_name,
        model_name=model_name,
        status="error",
        history_rows=history,
        dataset_config=None,
        train_trials=0,
        valid_trials=0,
        n_neurons=0,
        n_time=0,
        extra_metrics={},
        error=error,
    )
    return {"summary": summary, "history": history, "traceback": tb}


def expand_datasets(values: list[str]) -> list[str]:
    expanded = []
    for value in values:
        if value == "lower":
            expanded.extend(LOWER_CTD_DATASETS)
        elif value == "all":
            expanded.extend(ALL_CTD_DATASETS)
        else:
            expanded.append(value)
    return list(dict.fromkeys(expanded))


def expand_models(values: list[str]) -> list[str]:
    expanded = []
    for value in values:
        if value == "all":
            expanded.extend(ALL_MODELS)
        else:
            expanded.append(value)
    return list(dict.fromkeys(expanded))


def validate_models(models: list[str]) -> None:
    unknown = sorted(set(models).difference(MODEL_CONFIGS))
    if unknown:
        raise KeyError(f"Unknown model(s): {unknown}. Choices: {sorted(MODEL_CONFIGS)}")


def choose_divisor_at_most(n: int, requested: int) -> int:
    n = int(n)
    requested = max(1, min(int(requested), n))
    for value in range(requested, 0, -1):
        if n % value == 0:
            return value
    return 1


def dataset_shape(loader: DataLoader) -> tuple[int, int]:
    dataset = loader.dataset
    spikes = getattr(dataset, "spikes")
    return int(spikes.shape[1]), int(spikes.shape[2])


def write_outputs(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    history_rows: list[dict[str, Any]],
) -> None:
    write_csv(output_dir / "summary.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(output_dir / "history.csv", history_rows, HISTORY_FIELDS)
    write_summary_md(output_dir / "summary.md", summary_rows)


def write_summary_md(path: Path, rows: list[dict[str, Any]]) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    lines = [
        "# CTD Rate-MSE Benchmark",
        "",
        "Metric: MSE between predicted rates and CTD `*_activity` true rates.",
        "",
    ]
    if not ok_rows:
        lines.append("No completed cases yet.")
        path.write_text("\n".join(lines) + "\n")
        return
    datasets = sorted({str(row.get("dataset", "")) for row in ok_rows})
    models = sorted({str(row.get("model", "")) for row in ok_rows})
    for dataset in datasets:
        lines.extend([f"## {dataset}", ""])
        lines.append("| model | neurons | train trials | valid trials | best MSE | final MSE | seconds |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for model in models:
            candidates = [
                row for row in ok_rows
                if row.get("dataset") == dataset and row.get("model") == model
            ]
            if not candidates:
                continue
            row = min(candidates, key=lambda item: to_float(item.get("best_rate_mse")))
            lines.append(
                "| "
                + " | ".join(
                    [
                        model,
                        str(row.get("neurons", "")),
                        str(row.get("train_trials", "")),
                        str(row.get("valid_trials", "")),
                        fmt_number(row.get("best_rate_mse")),
                        fmt_number(row.get("final_rate_mse")),
                        fmt_number(row.get("wall_seconds")),
                    ]
                )
                + " |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


SUMMARY_FIELDS = [
    "status",
    "dataset",
    "task",
    "model",
    "neurons",
    "configured_total_neurons",
    "configured_total_trials",
    "train_trials",
    "valid_trials",
    "num_steps",
    "seed",
    "epochs",
    "batch_size",
    "trainer_config_path",
    "best_epoch",
    "final_epoch",
    "final_rate_mse",
    "best_rate_mse",
    "convergence_seconds",
    "total_optimizer_seconds",
    "wall_seconds",
    "train_loss",
    "test_loss",
    "rate_r2",
    "co_bps",
    "poisson_nll",
    "latent_linear_r2",
    "error",
]
HISTORY_FIELDS = [
    "status",
    "dataset",
    "model",
    "neurons",
    "trials",
    "num_steps",
    "seed",
    "epoch",
    "optimizer_seconds",
    "cumulative_optimizer_seconds",
    "wall_seconds",
    "train_loss",
    "test_loss",
    "test_rate_mse",
    "objective",
    "error",
]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    if not rows:
        path.write_text(",".join(fieldnames) + "\n")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=sort_key):
            writer.writerow(row)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def completed_keys(rows: list[dict[str, Any]], retry_errors: bool) -> set[tuple[str, str, int]]:
    keys = set()
    for row in rows:
        if retry_errors and row.get("status") == "error":
            continue
        keys.add((str(row.get("dataset")), str(row.get("model")), to_int(row.get("seed"))))
    return keys


def replace_summary(
    rows: list[dict[str, Any]],
    row: dict[str, Any],
) -> list[dict[str, Any]]:
    key = (row.get("dataset"), row.get("model"), to_int(row.get("seed")))
    out = [
        item for item in rows
        if (item.get("dataset"), item.get("model"), to_int(item.get("seed"))) != key
    ]
    out.append(row)
    return out


def replace_history(
    rows: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    key = (summary.get("dataset"), summary.get("model"), to_int(summary.get("seed")))
    out = [
        item for item in rows
        if (item.get("dataset"), item.get("model"), to_int(item.get("seed"))) != key
    ]
    out.extend(case_rows)
    return out


def write_error_log(output_dir: Path, summary: dict[str, Any], tb: str) -> None:
    error_dir = output_dir / "errors"
    error_dir.mkdir(parents=True, exist_ok=True)
    path = error_dir / f"{summary.get('dataset')}_{summary.get('model')}_seed{summary.get('seed')}.log"
    path.write_text(str(summary.get("error", "")) + "\n\n" + tb)


def write_run_config(path: Path, args: argparse.Namespace) -> None:
    data = vars(args).copy()
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_heartbeat(output_dir: Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["timestamp"] = time.time()
    payload["timestamp_local"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    path = output_dir / "heartbeat.json"
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def best_mse_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    finite = [row for row in rows if np.isfinite(to_float(row.get("test_rate_mse")))]
    if not finite:
        return rows[-1] if rows else {}
    return min(finite, key=lambda row: to_float(row.get("test_rate_mse")))


def progress_line(
    summary: dict[str, Any],
    index: int,
    total: int,
    started: float,
) -> str:
    elapsed = time.perf_counter() - started
    status = str(summary.get("status"))
    dataset = str(summary.get("dataset"))
    model = str(summary.get("model"))
    mse = fmt_number(summary.get("best_rate_mse"))
    return (
        f"[{index}/{total}] {status} dataset={dataset} model={model} "
        f"best_rate_mse={mse} elapsed={elapsed:.1f}s"
    )


def sort_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row.get("dataset", "")), str(row.get("model", "")), to_int(row.get("seed")))


def to_int(value: Any) -> int:
    try:
        if value in {"", None}:
            return 0
    except TypeError:
        pass
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def to_float(value: Any) -> float:
    try:
        if value in {"", None}:
            return float("nan")
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fmt_number(value: Any) -> str:
    number = to_float(value)
    if not np.isfinite(number):
        return "nan"
    if abs(number) >= 1.0e4 or (abs(number) < 1.0e-3 and number != 0.0):
        return f"{number:.3e}"
    return f"{number:.6g}"


def sync_device(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def cleanup_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
