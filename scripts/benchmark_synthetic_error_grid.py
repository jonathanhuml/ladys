"""Run synthetic rate-MSE heat-map sweeps over synthetic dataset size grids.

This script turns the appendix placeholder heat map into a resumable benchmark:
each cell is one synthetic dataset size, with neurons on the y-axis and trials
on the x-axis by default. It can also fix neurons and vary time points on the
y-axis. After every case, it rewrites tabular outputs and heat maps so a long
HAL job leaves usable partial results.

Examples:
    PYTHONPATH=src python3 scripts/benchmark_synthetic_error_grid.py \
        --models psth smoothing --device cuda --workers 2

    PYTHONPATH=src python3 scripts/benchmark_synthetic_error_grid.py \
        --models all --epochs 20 --device cuda --workers 1
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

_CACHE_ROOT = Path(tempfile.gettempdir())
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "ladys_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "ladys_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm, Normalize
from torch.utils.data import DataLoader

from ladys.datasets import (
    ChaoticRNNDataset,
    ChaoticRNNDatasetConfig,
    LorenzDataset,
    LorenzDatasetConfig,
)
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
from ladys.plotting import model_color, plot_context, save_figure, style_axis
from ladys.preprocessing import PreprocessedDataset, PreprocessingConfig
from ladys.training import Trainer, TrainerConfig
from ladys.training.strategies import build_strategy
from ladys.utils.yaml import load_yaml

from benchmark_lorenz_loss_curves import (
    evaluate_poisson_nll,
    evaluate_rate_mse,
    fit_mint_lorenz_library,
    mint_lorenz_epoch_trial_counts,
)


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
DEFAULT_MODELS = ["psth", "smoothing"]
BASELINE_MODELS = {"psth", "smoothing"}
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
MODEL_LABELS = {
    "bgpfa": "bGPFA",
    "cassm": "CASSM",
    "gpfa": "GPFA",
    "ilqr_vae": "iLQR-VAE",
    "kalman": "Kalman",
    "langevin_flow": "LangevinFlow",
    "lfads": "LFADS",
    "mint": "MINT",
    "ndt": "NDT",
    "stndt": "STNDT",
    "psth": "PSTH",
    "smoothing": "Smoothing",
}


@dataclass(frozen=True)
class GridCase:
    model: str
    neurons: int
    trials: int
    num_steps: int
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Models to run. Use 'all' for every registered LADYS method.",
    )
    parser.add_argument(
        "--panel-models",
        nargs="+",
        default=["all"],
        help="Panels shown in heat maps. Defaults to all LADYS methods.",
    )
    parser.add_argument(
        "--dataset",
        choices=["lorenz", "chaotic_rnn"],
        default="lorenz",
        help="Synthetic dataset to benchmark.",
    )
    parser.add_argument(
        "--grid-y",
        choices=["neurons", "time"],
        default="neurons",
        help="Quantity to place on the heat-map y-axis.",
    )
    parser.add_argument("--neurons", nargs="+", type=int, help="Explicit neuron grid.")
    parser.add_argument("--trials", nargs="+", type=int, help="Explicit trial grid.")
    parser.add_argument("--time-points", nargs="+", type=int, help="Explicit time-point grid.")
    parser.add_argument(
        "--fixed-neurons",
        type=int,
        default=500,
        help="Neuron count used when --grid-y=time.",
    )
    parser.add_argument("--min-neurons", type=int, default=10)
    parser.add_argument("--max-neurons", type=int, default=10_000)
    parser.add_argument("--num-neuron-points", type=int, default=15)
    parser.add_argument("--min-trials", type=int, default=10)
    parser.add_argument("--max-trials", type=int, default=1_000)
    parser.add_argument("--num-trial-points", type=int, default=10)
    parser.add_argument("--min-time-points", type=int, default=10)
    parser.add_argument("--max-time-points", type=int, default=1_000)
    parser.add_argument("--num-time-points", type=int, default=15)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--num-inits",
        type=int,
        default=1,
        help=(
            "Synthetic initial conditions. For Lorenz, total generated trials are "
            "num_inits * trials; default 1 keeps the x-axis equal to total trials."
        ),
    )
    parser.add_argument(
        "--num-conditions",
        type=int,
        default=None,
        help="Chaotic-RNN conditions. Defaults to --num-inits.",
    )
    parser.add_argument(
        "--hidden-units",
        type=int,
        default=None,
        help="Chaotic-RNN hidden units. Defaults to max(neurons, 50).",
    )
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--burn-steps", type=int, default=1000)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--max-dataset-gb",
        type=float,
        default=None,
        help=(
            "Optional preflight memory budget for one synthetic dataset case. "
            "Cases estimated above this budget are recorded as errors before "
            "allocating arrays."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Record an error instead of falling back if CUDA is unavailable.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", default="runs/synthetic_error_grid")
    parser.add_argument(
        "--paper-figure-path",
        help="Optional extra PDF path to mirror the MSE heat map into the paper repo.",
    )
    parser.add_argument("--experiment-config-dir", default="configs/experiment")
    parser.add_argument(
        "--preprocessing-mode",
        choices=["model", "none"],
        default="model",
        help="Observation preprocessing: 'model' uses experiment YAMLs; 'none' uses raw spikes.",
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
    parser.add_argument(
        "--mint-lorenz-library-source",
        choices=["smoothed_spikes", "true_rates"],
        default=None,
    )
    parser.add_argument("--mint-causal", action="store_true")
    parser.add_argument(
        "--heatmap-metric",
        choices=["final_rate_mse", "best_rate_mse"],
        default="final_rate_mse",
    )
    parser.add_argument(
        "--timing-metric",
        choices=["convergence_seconds", "wall_seconds", "total_optimizer_seconds"],
        default="convergence_seconds",
    )
    parser.add_argument(
        "--linear-color",
        action="store_true",
        help="Use linear color scaling for heat maps instead of log color scaling.",
    )
    parser.add_argument(
        "--heatmap-title",
        default="Error as a function of trials and neurons",
        help="Title for the primary MSE heat-map figure.",
    )
    parser.add_argument(
        "--heatmap-height-scale",
        type=float,
        default=2.0,
        help="Height multiplier for multi-panel heat maps.",
    )
    parser.add_argument(
        "--heatmap-width-scale",
        type=float,
        default=1.08,
        help="Width multiplier for multi-panel heat maps.",
    )
    parser.add_argument(
        "--plot-every",
        type=int,
        default=1,
        help=(
            "Regenerate heat maps and Markdown grids every N completed cases. "
            "CSV and NumPy data are still saved after every case. Use 0 for "
            "final-only plot/report refreshes."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry existing error rows when resuming.",
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        default=None,
        help="Run at most this many pending cases, useful for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.models = expand_models(args.models)
    args.panel_models = expand_models(args.panel_models)
    validate_models(args.models)
    validate_models(args.panel_models)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    (output_dir / "errors").mkdir(parents=True, exist_ok=True)

    neuron_grid = resolve_neuron_grid(args)
    trial_grid = resolve_grid(
        explicit=args.trials,
        minimum=args.min_trials,
        maximum=args.max_trials,
        points=args.num_trial_points,
    )
    time_grid = resolve_time_grid(args)
    y_grid = time_grid if args.grid_y == "time" else neuron_grid
    write_run_config(output_dir / "run_config.json", args, neuron_grid, trial_grid, time_grid, y_grid)

    summary_path = output_dir / "summary.csv"
    history_path = output_dir / "error_by_time.csv"
    summary_rows = [] if args.overwrite else read_csv(summary_path)
    history_rows = [] if args.overwrite else read_csv(history_path)
    completed = completed_keys(summary_rows, retry_errors=args.retry_errors)
    cases = [
        case
        for case in build_cases(args.models, neuron_grid, trial_grid, time_grid, args.seeds, args.grid_y)
        if case_key(case) not in completed
    ]
    if args.stop_after is not None:
        cases = cases[: max(int(args.stop_after), 0)]

    write_outputs(output_dir, summary_rows, history_rows, args, y_grid, trial_grid)
    if not cases:
        print(f"No pending cases. Outputs are current in {output_dir}.")
        return

    print(
        "Running "
        f"{len(cases)} cases on device={args.device} with workers={args.workers}: "
        f"models={args.models}, y_axis={args.grid_y}, y_grid={y_grid}, "
        f"neurons={neuron_grid}, trials={trial_grid}, time_points={time_grid}, seeds={args.seeds}"
    )
    run_started = time.perf_counter()
    completed_pending = 0
    for result in run_cases(args, cases):
        completed_pending += 1
        summary_row = result["summary"]
        case_history = result["history"]
        summary_rows = replace_summary_row(summary_rows, summary_row)
        history_rows = replace_history_rows(history_rows, case_history, summary_row)
        write_outputs(
            output_dir,
            summary_rows,
            history_rows,
            args,
            y_grid,
            trial_grid,
            include_reports=should_refresh_reports(completed_pending, args.plot_every),
        )
        if summary_row.get("status") == "error":
            write_error_log(output_dir, summary_row, result.get("traceback", ""))
        print(progress_line(summary_row, completed_pending, len(cases), run_started))

    write_outputs(output_dir, summary_rows, history_rows, args, y_grid, trial_grid)
    print(f"Wrote {summary_path}")
    print(f"Wrote {history_path}")
    print(f"Wrote {output_dir / 'plots' / 'synthetic_mse_heatmaps.png'}")


def run_cases(args: argparse.Namespace, cases: list[GridCase]):
    if args.workers <= 1:
        for case in cases:
            yield run_grid_case(vars(args), case)
        return

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
        futures = {
            executor.submit(run_grid_case, vars(args), case): case
            for case in cases
        }
        for future in as_completed(futures):
            case = futures[future]
            try:
                yield future.result()
            except Exception as exc:
                error_args = argparse.Namespace(**vars(args))
                error_args.neurons = case.neurons
                error_args.num_trials = case.trials
                error_args.num_steps = case.num_steps
                error_args.seed = case.seed
                yield error_result_from_exception(error_args, case, exc)


def run_grid_case(args_dict: dict[str, Any], case: GridCase) -> dict[str, Any]:
    args = argparse.Namespace(**args_dict)
    args.neurons = case.neurons
    args.num_trials = case.trials
    args.num_steps = case.num_steps
    args.seed = case.seed
    torch.manual_seed(case.seed)
    np.random.seed(case.seed)

    try:
        if args.require_cuda and str(args.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was required but torch.cuda.is_available() is false.")

        enforce_dataset_memory_budget(args)
        dataset_config = build_synthetic_dataset_config(args)
        train_ds, test_ds = make_synthetic_splits(dataset_config)
        preprocessing = build_preprocessing_config(
            model_name=case.model,
            config_dir=args.experiment_config_dir,
            preprocessing_mode=args.preprocessing_mode,
            dataset_name=args.dataset,
            n_neurons=case.neurons,
        )
        train_ds = PreprocessedDataset(train_ds, preprocessing)
        test_ds = PreprocessedDataset(test_ds, preprocessing)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        n_time, n_neurons = train_ds.spikes.shape[1:]

        model_config = build_model_config(args, case.model, n_neurons)
        model = model_config.build(n_neurons=n_neurons, n_time=n_time)

        if case.model == "mint":
            case_history = run_mint_case(
                args=args,
                model=model,
                model_name=case.model,
                train_ds=train_ds,
                dataset_config=dataset_config,
                train_loader=train_loader,
                test_loader=test_loader,
            )
        elif case.model in INFERENCE_ONLY_MODELS or int(args.epochs) <= 0:
            case_history = run_inference_only_case(
                args=args,
                model=model,
                model_name=case.model,
                train_loader=train_loader,
                test_loader=test_loader,
            )
        else:
            case_history = run_trained_case(
                args=args,
                model=model,
                model_name=case.model,
                model_config=model_config,
                train_loader=train_loader,
                test_loader=test_loader,
            )

        extra_metrics = evaluate_extra_metrics(
            model=model,
            loader=test_loader,
            train_loader=train_loader,
            device=args.device,
        )
        summary = summarize_case(
            args=args,
            case=case,
            status="ok",
            history_rows=case_history,
            train_trials=len(train_ds),
            valid_trials=len(test_ds),
            extra_metrics=extra_metrics,
        )
        cleanup_cuda()
        return {"summary": summary, "history": case_history, "traceback": ""}
    except Exception as exc:
        cleanup_cuda()
        return error_result_from_exception(args, case, exc)


def run_inference_only_case(
    args: argparse.Namespace,
    model,
    model_name: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
) -> list[dict[str, Any]]:
    sync_device(args.device)
    started = time.perf_counter()
    evaluation = evaluate_model(
        model=model,
        loader=test_loader,
        device=args.device,
        train_loader=train_loader,
    )
    train_loss = evaluate_poisson_nll(model, train_loader, args.device)
    test_loss = evaluate_poisson_nll(model, test_loader, args.device)
    sync_device(args.device)
    elapsed = time.perf_counter() - started
    return [
        {
            "status": "ok",
            "dataset": args.dataset,
            "model": model_name,
            "neurons": args.neurons,
            "trials": args.num_trials,
            "num_steps": args.num_steps,
            "seed": args.seed,
            "epoch": 1,
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
    model_config: BaseModelConfig,
    train_loader: DataLoader,
    test_loader: DataLoader,
) -> list[dict[str, Any]]:
    strategy = build_strategy(model_config.optimization)
    trainer = Trainer(TrainerConfig(epochs=int(args.epochs), device=args.device))
    metric_fns = {
        "test_rate_mse": lambda current_model: evaluate_rate_mse(
            current_model,
            test_loader,
            args.device,
            bgpfa_infer_steps=args.bgpfa_infer_steps,
            bgpfa_infer_mc=args.bgpfa_infer_mc,
            bgpfa_infer_lr=args.bgpfa_infer_lr,
        )
    }

    sync_device(args.device)
    started = time.perf_counter()
    valid_loader = None if model_name == "bgpfa" else test_loader
    history = trainer.fit(model, strategy, train_loader, valid_loader, metric_fns)
    sync_device(args.device)
    wall_seconds = time.perf_counter() - started

    rows = []
    cumulative_optimizer_seconds = 0.0
    for report in history:
        cumulative_optimizer_seconds += float(report.seconds)
        rows.append(
            {
                "status": "ok",
                "dataset": args.dataset,
                "model": model_name,
                "neurons": args.neurons,
                "trials": args.num_trials,
                "num_steps": args.num_steps,
                "seed": args.seed,
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
    train_ds: PreprocessedDataset,
    dataset_config: LorenzDatasetConfig | ChaoticRNNDatasetConfig,
    train_loader: DataLoader,
    test_loader: DataLoader,
) -> list[dict[str, Any]]:
    rows = []
    cumulative_optimizer_seconds = 0.0
    repeat_counts = mint_lorenz_epoch_trial_counts(
        train_ds,
        dataset_config,
        requested_epochs=args.epochs,
    )
    sync_device(args.device)
    started = time.perf_counter()
    for epoch, repeat_count in enumerate(repeat_counts, start=1):
        epoch_started = time.perf_counter()
        try:
            fit_mint_lorenz_library(
                model,
                train_ds,
                dataset_config,
                args.device,
                max_repeats_per_condition=repeat_count,
            )
        except TypeError as exc:
            if "max_repeats_per_condition" not in str(exc):
                raise
            fit_mint_lorenz_library(
                model,
                train_ds,
                dataset_config,
                args.device,
                max_trials=repeat_count,
            )
        sync_device(args.device)
        optimizer_seconds = time.perf_counter() - epoch_started
        cumulative_optimizer_seconds += optimizer_seconds
        train_loss = evaluate_poisson_nll(model, train_loader, args.device, use_raw_spikes=True)
        test_loss = evaluate_poisson_nll(model, test_loader, args.device, use_raw_spikes=True)
        test_rate_mse = evaluate_rate_mse(
            model,
            test_loader,
            args.device,
            use_raw_spikes=True,
            bgpfa_infer_steps=args.bgpfa_infer_steps,
            bgpfa_infer_mc=args.bgpfa_infer_mc,
            bgpfa_infer_lr=args.bgpfa_infer_lr,
        )
        rows.append(
            {
                "status": "ok",
                "dataset": args.dataset,
                "model": model_name,
                "neurons": args.neurons,
                "trials": args.num_trials,
                "num_steps": args.num_steps,
                "seed": args.seed,
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
    return rows


def evaluate_extra_metrics(
    model,
    loader: DataLoader,
    train_loader: DataLoader,
    device: str,
) -> dict[str, float]:
    try:
        evaluation = evaluate_model(
            model=model,
            loader=loader,
            device=device,
            train_loader=train_loader,
        )
        return evaluation.metrics
    except Exception:
        return {}


def build_synthetic_dataset_config(
    args: argparse.Namespace,
) -> LorenzDatasetConfig | ChaoticRNNDatasetConfig:
    if args.dataset == "lorenz":
        return LorenzDatasetConfig(
            neurons=int(args.neurons),
            num_inits=int(args.num_inits),
            num_trials=int(args.num_trials),
            num_steps=int(args.num_steps),
            burn_steps=int(args.burn_steps),
            train_fraction=float(args.train_fraction),
            seed=int(args.seed),
        )
    if args.dataset == "chaotic_rnn":
        num_conditions = args.num_inits if args.num_conditions is None else args.num_conditions
        hidden_units = args.hidden_units if args.hidden_units is not None else max(args.neurons, 50)
        return ChaoticRNNDatasetConfig(
            neurons=int(args.neurons),
            hidden_units=max(int(hidden_units), int(args.neurons)),
            num_conditions=int(num_conditions),
            num_trials=int(args.num_trials),
            num_steps=int(args.num_steps),
            train_fraction=float(args.train_fraction),
            seed=int(args.seed),
        )
    raise KeyError(args.dataset)


def enforce_dataset_memory_budget(args: argparse.Namespace) -> None:
    if args.max_dataset_gb is None:
        return
    estimated_bytes = estimate_synthetic_dataset_bytes(args)
    max_bytes = float(args.max_dataset_gb) * (1024.0 ** 3)
    if estimated_bytes > max_bytes:
        raise MemoryError(
            "Estimated synthetic dataset footprint exceeds --max-dataset-gb: "
            f"{estimated_bytes / (1024.0 ** 3):.2f} GiB > {float(args.max_dataset_gb):.2f} GiB."
        )


def estimate_synthetic_dataset_bytes(args: argparse.Namespace) -> float:
    bytes_per_float = 4.0
    neurons = int(args.neurons)
    trials = int(args.num_trials)
    steps = int(args.num_steps)
    if args.dataset == "chaotic_rnn":
        num_conditions = args.num_inits if args.num_conditions is None else args.num_conditions
        total_trials = int(num_conditions) * trials
        hidden_units = args.hidden_units if args.hidden_units is not None else max(neurons, 50)
        hidden_units = max(int(hidden_units), neurons)
        recurrent_bytes = hidden_units * hidden_units * bytes_per_float
        hidden_bytes = total_trials * steps * hidden_units * bytes_per_float
        observed_bytes = total_trials * steps * neurons * bytes_per_float
        condition_bytes = int(num_conditions) * hidden_units * bytes_per_float
        # Generation holds several observed copies transiently: sampled,
        # normalized, mean counts, spikes, and train/valid tensors.
        return recurrent_bytes + hidden_bytes + 5.0 * observed_bytes + condition_bytes
    if args.dataset == "lorenz":
        total_trials = int(args.num_inits) * trials
        observed_bytes = total_trials * steps * neurons * bytes_per_float
        latent_bytes = total_trials * steps * 3 * bytes_per_float
        return 4.0 * observed_bytes + 2.0 * latent_bytes
    raise KeyError(args.dataset)


def make_synthetic_splits(
    config: LorenzDatasetConfig | ChaoticRNNDatasetConfig,
) -> tuple[LorenzDataset | ChaoticRNNDataset, LorenzDataset | ChaoticRNNDataset]:
    if isinstance(config, LorenzDatasetConfig):
        return LorenzDataset.make_splits(config)
    if isinstance(config, ChaoticRNNDatasetConfig):
        return ChaoticRNNDataset.make_splits(config)
    raise TypeError(f"Unsupported synthetic config {type(config).__name__}.")


def build_model_config(
    args: argparse.Namespace,
    model_name: str,
    n_neurons: int,
) -> BaseModelConfig:
    model_data = load_model_data(
        config_dir=args.experiment_config_dir,
        dataset_name=args.dataset,
        model_name=model_name,
        n_neurons=n_neurons,
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
                    "lr": args.cassm_lr,
                    "weight_decay": 0.0,
                    "gradient_clip": 300.0,
                },
            )
        model_data["projection_dim"] = int(projection_dim)
        if args.cassm_lr is not None:
            optimization = dict(model_data.get("optimization", {}))
            optimization["lr"] = args.cassm_lr
            model_data["optimization"] = optimization
        return BaseModelConfig.from_dict(model_data)
    if model_name == "gpfa":
        init_seed = args.seed if args.gpfa_init_seed is None else args.gpfa_init_seed
        if model_data is None:
            return GPFAConfig(
                latent_dim=args.gpfa_latent_dim,
                init_method=args.gpfa_init_method,
                init_seed=init_seed,
            )
        model_data["latent_dim"] = args.gpfa_latent_dim
        model_data["init_method"] = args.gpfa_init_method
        model_data["init_seed"] = init_seed
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
            model_data["max_iter"] = args.ilqr_max_iter
            return BaseModelConfig.from_dict(model_data)
        return ILQRVAEConfig(
            objective="ilqr_vae_elbo",
            params_path=None,
            initialization="random",
            trainable_parameters=True,
            max_iter=args.ilqr_max_iter,
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
        if model_data is None:
            model_data = {"name": "mint", "dataset": args.dataset}
        model_data["dataset"] = args.dataset
        if args.mint_n_candidates is not None:
            model_data["n_candidates"] = args.mint_n_candidates
        if args.mint_window_length is not None:
            model_data["window_length"] = args.mint_window_length
        if args.mint_delta is not None:
            model_data["delta"] = args.mint_delta
        if args.mint_lorenz_library_source is not None:
            model_data["lorenz_library_source"] = args.mint_lorenz_library_source
        if args.mint_causal:
            model_data["causal"] = True
        return BaseModelConfig.from_dict(model_data)
    if model_name == "lfads":
        if model_data is not None:
            return BaseModelConfig.from_dict(model_data)
        if args.dataset == "chaotic_rnn":
            return LFADSConfig(dt=0.01)
        return LFADSConfig()
    if model_name in {"langevin_flow", "ndt", "stndt"}:
        if model_data is not None:
            return BaseModelConfig.from_dict(model_data)
        return MODEL_CONFIGS[model_name]()
    raise KeyError(model_name)


def build_preprocessing_config(
    model_name: str,
    config_dir: str,
    preprocessing_mode: str = "model",
    dataset_name: str = "lorenz",
    n_neurons: int | None = None,
) -> PreprocessingConfig:
    if preprocessing_mode == "none":
        return PreprocessingConfig()
    path = synthetic_experiment_config_path(config_dir, dataset_name, model_name, n_neurons)
    if not path.exists():
        return PreprocessingConfig()
    data = load_yaml(path)
    return PreprocessingConfig.model_validate(data.get("preprocessing", {}))


def load_model_data(
    config_dir: str,
    dataset_name: str,
    model_name: str,
    n_neurons: int | None = None,
) -> dict[str, Any] | None:
    path = synthetic_experiment_config_path(config_dir, dataset_name, model_name, n_neurons)
    if not path.exists():
        return None
    data = load_yaml(path)
    return dict(data["model"] if "model" in data else data)


def synthetic_experiment_config_path(
    config_dir: str,
    dataset_name: str,
    model_name: str,
    n_neurons: int | None = None,
) -> Path:
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
            root / dataset_name / model_name / f"{model_name}_{dataset_name}.yaml",
            root / f"{model_name}_{dataset_name}.yaml",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def choose_divisor_at_most(n: int, requested: int) -> int:
    n = int(n)
    requested = max(1, min(int(requested), n))
    for value in range(requested, 0, -1):
        if n % value == 0:
            return value
    return 1


def summarize_case(
    args: argparse.Namespace,
    case: GridCase,
    status: str,
    history_rows: list[dict[str, Any]],
    train_trials: int,
    valid_trials: int,
    extra_metrics: dict[str, float] | None = None,
    error: str = "",
) -> dict[str, Any]:
    extra_metrics = extra_metrics or {}
    ok_rows = [row for row in history_rows if row.get("status") == "ok"]
    final = ok_rows[-1] if ok_rows else {}
    best = best_mse_row(ok_rows)
    return {
        "status": status,
        "dataset": args.dataset,
        "model": case.model,
        "neurons": case.neurons,
        "trials": case.trials,
        "seed": case.seed,
        "num_inits": args.num_inits,
        "total_train_trials": train_trials,
        "total_valid_trials": valid_trials,
        "num_steps": case.num_steps,
        "epochs": args.epochs,
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


def error_result_from_exception(
    args: argparse.Namespace,
    case: GridCase,
    exc: BaseException,
) -> dict[str, Any]:
    tb = traceback.format_exc()
    error = f"{type(exc).__name__}: {exc}"
    history = [
        {
            "status": "error",
            "dataset": getattr(args, "dataset", ""),
            "model": case.model,
            "neurons": case.neurons,
            "trials": case.trials,
            "num_steps": case.num_steps,
            "seed": case.seed,
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
        case=case,
        status="error",
        history_rows=history,
        train_trials=0,
        valid_trials=0,
        extra_metrics={},
        error=error,
    )
    return {"summary": summary, "history": history, "traceback": tb}


def write_outputs(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    history_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    y_grid: list[int],
    trial_grid: list[int],
    include_reports: bool = True,
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "summary.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(output_dir / "error_by_time.csv", history_rows, HISTORY_FIELDS)
    write_csv(output_dir / "timing.csv", summary_rows, SUMMARY_FIELDS)
    write_numpy(output_dir / "summary.npy", summary_rows)
    if not include_reports:
        return
    y_axis_label = grid_y_axis_label(args)
    y_axis_table_label = grid_y_table_label(args)
    write_summary_md(output_dir / "summary.md", summary_rows, args)
    write_grid_md(
        output_dir / "error_grid.md",
        summary_rows,
        args.panel_models,
        y_grid,
        trial_grid,
        metric=args.heatmap_metric,
        title="Synthetic Rate-MSE Grid",
        grid_y=args.grid_y,
        y_axis_table_label=y_axis_table_label,
    )
    write_grid_md(
        output_dir / "timing_grid.md",
        summary_rows,
        args.panel_models,
        y_grid,
        trial_grid,
        metric=args.timing_metric,
        title="Synthetic Timing Grid",
        grid_y=args.grid_y,
        y_axis_table_label=y_axis_table_label,
    )
    plot_heatmaps(
        summary_rows,
        args.panel_models,
        y_grid,
        trial_grid,
        metric=args.heatmap_metric,
        path=plots_dir / "synthetic_mse_heatmaps.png",
        title=args.heatmap_title,
        colorbar_label="MSE",
        grid_y=args.grid_y,
        y_axis_label=y_axis_label,
        log_color=not args.linear_color,
        height_scale=args.heatmap_height_scale,
        width_scale=args.heatmap_width_scale,
    )
    plot_heatmaps(
        summary_rows,
        args.panel_models,
        y_grid,
        trial_grid,
        metric=args.heatmap_metric,
        path=plots_dir / "synthetic_mse_heatmaps.pdf",
        title=args.heatmap_title,
        colorbar_label="MSE",
        grid_y=args.grid_y,
        y_axis_label=y_axis_label,
        log_color=not args.linear_color,
        height_scale=args.heatmap_height_scale,
        width_scale=args.heatmap_width_scale,
    )
    plot_heatmaps(
        summary_rows,
        args.panel_models,
        y_grid,
        trial_grid,
        metric=args.timing_metric,
        path=plots_dir / "synthetic_timing_heatmaps.png",
        title=f"{args.dataset} timing",
        colorbar_label="seconds",
        grid_y=args.grid_y,
        y_axis_label=y_axis_label,
        log_color=not args.linear_color,
        height_scale=args.heatmap_height_scale,
        width_scale=args.heatmap_width_scale,
    )
    plot_error_time(summary_rows, history_rows, plots_dir / "error_vs_time.png")
    if args.paper_figure_path:
        mirror_path = Path(args.paper_figure_path)
        mirror_path.parent.mkdir(parents=True, exist_ok=True)
        plot_heatmaps(
            summary_rows,
            args.panel_models,
            y_grid,
            trial_grid,
            metric=args.heatmap_metric,
            path=mirror_path,
            title=args.heatmap_title,
            colorbar_label="MSE",
            grid_y=args.grid_y,
            y_axis_label=y_axis_label,
            log_color=not args.linear_color,
            height_scale=args.heatmap_height_scale,
            width_scale=args.heatmap_width_scale,
        )


SUMMARY_FIELDS = [
    "status",
    "dataset",
    "model",
    "neurons",
    "trials",
    "seed",
    "num_inits",
    "total_train_trials",
    "total_valid_trials",
    "num_steps",
    "epochs",
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
    rows = sorted(rows, key=sort_key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_numpy(path: Path, rows: list[dict[str, Any]]) -> None:
    dtype = [
        ("status", "U16"),
        ("dataset", "U32"),
        ("model", "U32"),
        ("neurons", "i8"),
        ("trials", "i8"),
        ("seed", "i8"),
        ("num_inits", "i8"),
        ("total_train_trials", "i8"),
        ("total_valid_trials", "i8"),
        ("num_steps", "i8"),
        ("epochs", "i8"),
        ("best_epoch", "i8"),
        ("final_epoch", "i8"),
        ("final_rate_mse", "f8"),
        ("best_rate_mse", "f8"),
        ("convergence_seconds", "f8"),
        ("total_optimizer_seconds", "f8"),
        ("wall_seconds", "f8"),
        ("train_loss", "f8"),
        ("test_loss", "f8"),
        ("rate_r2", "f8"),
        ("co_bps", "f8"),
        ("poisson_nll", "f8"),
        ("latent_linear_r2", "f8"),
        ("error", "U512"),
    ]
    arr = np.empty(len(rows), dtype=dtype)
    for idx, row in enumerate(sorted(rows, key=sort_key)):
        values = []
        for name, dtype_name in dtype:
            value = row.get(name, "")
            if dtype_name.startswith("f"):
                value = to_float(value)
            elif dtype_name.startswith("i"):
                value = to_int(value)
            values.append(value)
        arr[idx] = tuple(values)
    np.save(path, arr)


def write_summary_md(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    y_axis_table_label = grid_y_table_label(args)
    y_axis_summary_label = "time points" if args.grid_y == "time" else "neurons"
    num_steps_label = "varied by y-axis" if args.grid_y == "time" else str(args.num_steps)
    fixed_neuron_lines = [f"- Fixed neurons: `{args.fixed_neurons}`"] if args.grid_y == "time" else []
    lines = [
        "# Synthetic Error Grid",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Device: `{args.device}`",
        f"- Models requested: `{', '.join(args.models)}`",
        f"- Y-axis: `{y_axis_summary_label}`",
        *fixed_neuron_lines,
        f"- Heat-map title: `{args.heatmap_title}`",
        f"- Heat-map metric: `{args.heatmap_metric}`",
        f"- Timing metric: `{args.timing_metric}`",
        f"- Num inits: `{args.num_inits}`",
        f"- Num steps: `{num_steps_label}`",
        f"- Epochs: `{args.epochs}`",
        f"- Batch size: `{args.batch_size}`",
        f"- Plot/report refresh: every `{args.plot_every}` completed cases",
        "",
        "## Completed Cases",
        "",
        (
            "| model | ok | errors | best MSE | median convergence seconds | "
            f"max {y_axis_summary_label} | max trials |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in args.panel_models:
        model_rows = [row for row in rows if row.get("model") == model]
        ok_rows = [row for row in model_rows if row.get("status") == "ok"]
        err_rows = [row for row in model_rows if row.get("status") == "error"]
        mse_values = [to_float(row.get("best_rate_mse")) for row in ok_rows]
        sec_values = [to_float(row.get("convergence_seconds")) for row in ok_rows]
        y_values = [case_y_value(row, args.grid_y) for row in ok_rows]
        lines.append(
            "| "
            + " | ".join(
                [
                    model_label(model),
                    str(len(ok_rows)),
                    str(len(err_rows)),
                    fmt_number(np.nanmin(mse_values) if finite_values(mse_values) else np.nan),
                    fmt_number(np.nanmedian(sec_values) if finite_values(sec_values) else np.nan),
                    str(max(y_values, default="")),
                    str(max([to_int(row.get("trials")) for row in ok_rows], default="")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `summary.csv`: one row per model/neuron/trial/time/seed case.",
            "- `error_by_time.csv`: per-epoch MSE with cumulative elapsed time.",
            "- `timing.csv`: same case table, retained for timing-focused downstream scripts.",
            f"- `error_grid.md`: rate-MSE tables with rows as {y_axis_table_label} and columns as trials.",
            f"- `timing_grid.md`: convergence-time tables with rows as {y_axis_table_label} and columns as trials.",
            "- `plots/synthetic_mse_heatmaps.png` and `.pdf`: appendix-style MSE panels.",
            "- `plots/synthetic_timing_heatmaps.png`: timing panels.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def write_grid_md(
    path: Path,
    rows: list[dict[str, Any]],
    models: list[str],
    y_grid: list[int],
    trial_grid: list[int],
    metric: str,
    title: str,
    grid_y: str,
    y_axis_table_label: str,
) -> None:
    lines = [f"# {title}", ""]
    for model in models:
        lines.extend(
            [
                f"## {model_label(model)}",
                "",
                f"| {y_axis_table_label} \\ trials | " + " | ".join(str(t) for t in trial_grid) + " |",
                "| ---: | " + " | ".join("---:" for _ in trial_grid) + " |",
            ]
        )
        for y_value in y_grid:
            values = []
            for trials in trial_grid:
                value = aggregate_metric(rows, model, y_value, trials, metric, grid_y)
                values.append(fmt_number(value))
            lines.append(f"| {y_value} | " + " | ".join(values) + " |")
        lines.append("")
    path.write_text("\n".join(lines))


def plot_heatmaps(
    rows: list[dict[str, Any]],
    models: list[str],
    y_grid: list[int],
    trial_grid: list[int],
    metric: str,
    path: Path,
    title: str,
    colorbar_label: str,
    grid_y: str,
    y_axis_label: str,
    log_color: bool = True,
    height_scale: float = 2.0,
    width_scale: float = 1.08,
) -> None:
    if not rows:
        return

    matrices = {
        model: metric_matrix(rows, model, y_grid, trial_grid, metric, grid_y)
        for model in models
    }
    values = np.concatenate(
        [
            matrix[np.isfinite(matrix)]
            for matrix in matrices.values()
            if np.isfinite(matrix).any()
        ]
    ) if any(np.isfinite(matrix).any() for matrix in matrices.values()) else np.array([])
    if values.size == 0:
        return

    positive_values = values[values > 0.0]
    if log_color and positive_values.size > 0:
        vmin = max(float(np.nanmin(positive_values)), 1e-12)
        vmax = float(np.nanmax(positive_values))
        if vmax <= vmin:
            vmax = vmin * 10.0
        norm = LogNorm(
            vmin=vmin,
            vmax=vmax,
        )
    else:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        if vmax <= vmin:
            pad = max(abs(vmin) * 0.1, 1.0)
            vmin -= pad
            vmax += pad
        norm = Normalize(vmin=vmin, vmax=vmax)

    ncols = 4
    nrows = int(math.ceil(len(models) / ncols))
    x_edges = log_edges(trial_grid)
    y_edges = log_edges(y_grid)

    x_labels = tick_labels_for_grid(trial_grid, max_labels=6)
    y_labels = tick_labels_for_grid(y_grid, max_labels=8)

    with plot_context(
        nrows=nrows,
        ncols=ncols,
        rel_width=1.0,
        width_scale=width_scale,
        height_scale=height_scale,
    ):
        fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False)
        last_mesh = None
        for ax, model in zip(axes.ravel(), models):
            matrix = np.ma.masked_invalid(matrices[model])
            if matrix.count() > 0:
                last_mesh = ax.pcolormesh(
                    x_edges,
                    y_edges,
                    matrix,
                    cmap="magma",
                    norm=norm,
                    shading="auto",
                )
            else:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(model_label(model), color=model_color(model), fontweight="bold", pad=5)
            ax.set_xticks(trial_grid)
            ax.set_yticks(y_grid)
            ax.set_xticklabels(x_labels, rotation=35, ha="right")
            ax.set_yticklabels(y_labels)
            ax.tick_params(axis="both", labelsize=6)
            style_axis(ax, which="both")
        for ax in axes.ravel()[len(models) :]:
            ax.axis("off")
        fig.supxlabel("Number of trials")
        fig.supylabel(y_axis_label)
        fig.suptitle(title)
        if last_mesh is not None:
            fig.colorbar(last_mesh, ax=axes, shrink=0.72, pad=0.025, label=colorbar_label)
        save_figure(fig, path)
        plt.close(fig)


def plot_error_time(
    summary_rows: list[dict[str, Any]],
    history_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    ok_rows = [row for row in history_rows if row.get("status") == "ok"]
    if not ok_rows:
        return
    top_rows = select_error_time_rows(summary_rows, ok_rows)
    if not top_rows:
        return
    models = sorted({str(row["model"]) for row in top_rows}, key=ALL_MODELS.index)
    with plot_context(nrows=1, ncols=1, rel_width=0.9, height_scale=1.35):
        fig, ax = plt.subplots()
        for model in models:
            rows = sorted(
                [row for row in top_rows if row["model"] == model],
                key=lambda row: (
                    to_int(row.get("neurons")),
                    to_int(row.get("trials")),
                    to_int(row.get("num_steps")),
                    to_int(row.get("seed")),
                    to_int(row.get("epoch")),
                ),
            )
            ax.plot(
                [to_float(row["cumulative_optimizer_seconds"]) for row in rows],
                [to_float(row["test_rate_mse"]) for row in rows],
                color=model_color(model),
                label=model_label(model),
            )
        ax.set_xlabel("Cumulative optimizer seconds")
        ax.set_ylabel("Rate MSE")
        ax.set_yscale("log")
        ax.set_title("Error by Time for Largest Completed Cell")
        style_axis(ax, which="both")
        ax.legend()
        save_figure(fig, path)
        plt.close(fig)


def select_error_time_rows(
    summary_rows: list[dict[str, Any]],
    history_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = []
    for model in sorted({str(row["model"]) for row in summary_rows}):
        ok_summaries = [
            row for row in summary_rows
            if row.get("model") == model and row.get("status") == "ok"
        ]
        if not ok_summaries:
            continue
        largest = max(
            ok_summaries,
            key=lambda row: (
                to_int(row.get("neurons"))
                * to_int(row.get("trials"))
                * max(to_int(row.get("num_steps")), 1)
            ),
        )
        selected.extend(
            row for row in history_rows
            if (
                row.get("model") == model
                and to_int(row.get("neurons")) == to_int(largest.get("neurons"))
                and to_int(row.get("trials")) == to_int(largest.get("trials"))
                and to_int(row.get("num_steps")) == to_int(largest.get("num_steps"))
                and to_int(row.get("seed")) == to_int(largest.get("seed"))
                and row.get("status") == "ok"
            )
        )
    return selected


def replace_summary_row(rows: list[dict[str, Any]], new_row: dict[str, Any]) -> list[dict[str, Any]]:
    new_key = summary_key(new_row)
    kept = [row for row in rows if summary_key(row) != new_key]
    kept.append(new_row)
    return kept


def replace_history_rows(
    rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    summary_row: dict[str, Any],
) -> list[dict[str, Any]]:
    new_key = (
        summary_row.get("model"),
        to_int(summary_row.get("neurons")),
        to_int(summary_row.get("trials")),
        to_int(summary_row.get("num_steps")),
        to_int(summary_row.get("seed")),
    )
    kept = [
        row for row in rows
        if (
            row.get("model"),
            to_int(row.get("neurons")),
            to_int(row.get("trials")),
            to_int(row.get("num_steps")),
            to_int(row.get("seed")),
        )
        != new_key
    ]
    kept.extend(new_rows)
    return kept


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def completed_keys(rows: list[dict[str, Any]], retry_errors: bool) -> set[tuple[str, int, int, int, int]]:
    completed_statuses = {"ok", "skipped", "error"}
    if retry_errors:
        completed_statuses.remove("error")
    return {
        summary_key(row)
        for row in rows
        if str(row.get("status")) in completed_statuses
    }


def build_cases(
    models: list[str],
    neuron_grid: list[int],
    trial_grid: list[int],
    time_grid: list[int],
    seeds: list[int],
    grid_y: str,
) -> list[GridCase]:
    if grid_y == "time":
        cases = [
            GridCase(model=model, neurons=neuron_grid[0], trials=trials, num_steps=num_steps, seed=seed)
            for model in models
            for num_steps in time_grid
            for trials in trial_grid
            for seed in seeds
        ]
    else:
        cases = [
            GridCase(model=model, neurons=neurons, trials=trials, num_steps=time_grid[0], seed=seed)
            for model in models
            for neurons in neuron_grid
            for trials in trial_grid
            for seed in seeds
        ]
    return sorted(
        cases,
        key=lambda item: (
            item.neurons * item.trials * item.num_steps,
            item.num_steps,
            item.neurons,
            item.trials,
            item.model,
            item.seed,
        ),
    )


def summary_key(row: dict[str, Any]) -> tuple[str, int, int, int, int]:
    return (
        str(row.get("model")),
        to_int(row.get("neurons")),
        to_int(row.get("trials")),
        to_int(row.get("num_steps")),
        to_int(row.get("seed")),
    )


def case_key(case: GridCase) -> tuple[str, int, int, int, int]:
    return (case.model, int(case.neurons), int(case.trials), int(case.num_steps), int(case.seed))


def sort_key(row: dict[str, Any]) -> tuple[int, int, int, int, int, str, int]:
    return (
        to_int(row.get("neurons")) * to_int(row.get("trials")) * max(to_int(row.get("num_steps")), 1),
        to_int(row.get("neurons")),
        to_int(row.get("trials")),
        to_int(row.get("num_steps")),
        to_int(row.get("seed")),
        str(row.get("model")),
        to_int(row.get("epoch")),
    )


def expand_models(values: list[str]) -> list[str]:
    expanded = []
    for value in values:
        if value == "all":
            expanded.extend(ALL_MODELS)
        else:
            expanded.append(value)
    seen = set()
    out = []
    for model in expanded:
        if model not in seen:
            out.append(model)
            seen.add(model)
    return out


def validate_models(models: list[str]) -> None:
    unknown = sorted(set(models) - set(ALL_MODELS))
    if unknown:
        known = ", ".join(ALL_MODELS)
        raise KeyError(f"Unknown model(s) {unknown}. Known models: {known}.")


def resolve_grid(
    explicit: list[int] | None,
    minimum: int,
    maximum: int,
    points: int,
) -> list[int]:
    if explicit:
        values = sorted({int(value) for value in explicit})
    else:
        if points < 1:
            raise ValueError("grid point counts must be positive.")
        values = sorted({int(round(value)) for value in np.geomspace(minimum, maximum, points)})
        values[0] = int(minimum)
        values[-1] = int(maximum)
    return [value for value in values if value > 0]


def resolve_neuron_grid(args: argparse.Namespace) -> list[int]:
    if args.grid_y == "time":
        if args.neurons:
            values = sorted({int(value) for value in args.neurons if int(value) > 0})
            if not values:
                raise ValueError("--neurons must contain positive values.")
            return [values[0]]
        if int(args.fixed_neurons) <= 0:
            raise ValueError("--fixed-neurons must be positive.")
        return [int(args.fixed_neurons)]
    return resolve_grid(
        explicit=args.neurons,
        minimum=args.min_neurons,
        maximum=args.max_neurons,
        points=args.num_neuron_points,
    )


def resolve_time_grid(args: argparse.Namespace) -> list[int]:
    if args.grid_y == "time":
        return resolve_grid(
            explicit=args.time_points,
            minimum=args.min_time_points,
            maximum=args.max_time_points,
            points=args.num_time_points,
        )
    if int(args.num_steps) <= 0:
        raise ValueError("--num-steps must be positive.")
    return [int(args.num_steps)]


def write_run_config(
    path: Path,
    args: argparse.Namespace,
    neuron_grid: list[int],
    trial_grid: list[int],
    time_grid: list[int],
    y_grid: list[int],
) -> None:
    payload = {
        "args": json_ready(vars(args)),
        "grid_y": args.grid_y,
        "y_grid": y_grid,
        "neuron_grid": neuron_grid,
        "trial_grid": trial_grid,
        "time_grid": time_grid,
        "model_order": ALL_MODELS,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def case_y_value(row: dict[str, Any], grid_y: str) -> int:
    if grid_y == "time":
        return to_int(row.get("num_steps"))
    return to_int(row.get("neurons"))


def grid_y_axis_label(args: argparse.Namespace) -> str:
    if args.grid_y == "time":
        return "Number of time points"
    return "Number of neurons"


def grid_y_table_label(args: argparse.Namespace) -> str:
    if args.grid_y == "time":
        return "time points"
    return "neurons"


def aggregate_metric(
    rows: list[dict[str, Any]],
    model: str,
    y_value: int,
    trials: int,
    metric: str,
    grid_y: str,
) -> float:
    values = [
        to_float(row.get(metric))
        for row in rows
        if (
            row.get("status") == "ok"
            and row.get("model") == model
            and case_y_value(row, grid_y) == int(y_value)
            and to_int(row.get("trials")) == int(trials)
        )
    ]
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return float("nan")
    return float(np.mean(values))


def metric_matrix(
    rows: list[dict[str, Any]],
    model: str,
    y_grid: list[int],
    trial_grid: list[int],
    metric: str,
    grid_y: str,
) -> np.ndarray:
    matrix = np.full((len(y_grid), len(trial_grid)), np.nan, dtype=float)
    for i, y_value in enumerate(y_grid):
        for j, trials in enumerate(trial_grid):
            matrix[i, j] = aggregate_metric(rows, model, y_value, trials, metric, grid_y)
    return matrix


def log_edges(values: list[int]) -> np.ndarray:
    centers = np.asarray(values, dtype=float)
    if centers.size == 1:
        return np.array([centers[0] / math.sqrt(10.0), centers[0] * math.sqrt(10.0)])
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = np.sqrt(centers[:-1] * centers[1:])
    edges[0] = centers[0] ** 2 / edges[1]
    edges[-1] = centers[-1] ** 2 / edges[-2]
    return edges


def best_mse_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    finite = [row for row in rows if math.isfinite(to_float(row.get("test_rate_mse")))]
    if not finite:
        return rows[-1] if rows else {}
    return min(finite, key=lambda row: to_float(row.get("test_rate_mse")))


def sync_device(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_error_log(output_dir: Path, summary_row: dict[str, Any], tb: str) -> None:
    if not tb:
        return
    filename = (
        f"{summary_row.get('model')}_n{summary_row.get('neurons')}"
        f"_tr{summary_row.get('trials')}_steps{summary_row.get('num_steps')}"
        f"_s{summary_row.get('seed')}.txt"
    )
    (output_dir / "errors" / filename).write_text(tb)


def should_refresh_reports(completed_pending: int, plot_every: int) -> bool:
    plot_every = max(int(plot_every), 0)
    return plot_every > 0 and completed_pending % plot_every == 0


def progress_line(
    row: dict[str, Any],
    n_rows: int,
    n_cases: int,
    started: float,
) -> str:
    elapsed = time.perf_counter() - started
    metric = fmt_number(row.get("final_rate_mse"))
    seconds = fmt_number(row.get("wall_seconds"))
    return (
        f"[{n_rows}/{n_cases} rows, elapsed {elapsed:.1f}s] "
        f"{row.get('status')} model={row.get('model')} "
        f"neurons={row.get('neurons')} trials={row.get('trials')} "
        f"steps={row.get('num_steps')} seed={row.get('seed')} "
        f"mse={metric} wall_s={seconds}"
    )


def finite_values(values: list[float]) -> bool:
    return bool([value for value in values if math.isfinite(value)])


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


def fmt_number(value: Any) -> str:
    value = to_float(value)
    if not math.isfinite(value):
        return ""
    if abs(value) >= 1000.0 or (0 < abs(value) < 1e-3):
        return f"{value:.4e}"
    return f"{value:.6g}"


def format_tick(value: int) -> str:
    if value >= 1000 and value % 1000 == 0:
        return f"{value // 1000}k"
    return str(value)


def tick_labels_for_grid(values: list[int], max_labels: int) -> list[str]:
    if len(values) <= max_labels:
        return [format_tick(value) for value in values]

    stride = int(math.ceil((len(values) - 1) / max(max_labels - 1, 1)))
    keep = set(range(0, len(values), stride))
    keep.add(0)
    keep.add(len(values) - 1)
    return [format_tick(value) if idx in keep else "" for idx, value in enumerate(values)]


def model_label(model: str) -> str:
    return MODEL_LABELS.get(str(model), str(model))


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    main()
