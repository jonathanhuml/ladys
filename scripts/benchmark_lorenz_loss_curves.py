"""Benchmark Lorenz train/test loss curves over multiple optimizer epochs.

This script is intended to become a PR artifact generator. It trains one or
more methods on the same Lorenz split through the shared trainer/strategy
contract, then writes:

- `lorenz_loss_history.csv`
- `test_rate_mse_curves.png`
- `test_objective_curves.png`
- `train_test_objective_curves.png`
- `{model}_rate_traces.png`
- `{model}_rate_traces.csv`

Example:
    PYTHONPATH=src python3 scripts/benchmark_lorenz_loss_curves.py \
        --models cassm gpfa kalman --neurons 100 --epochs 30

    PYTHONPATH=src python3 scripts/benchmark_lorenz_loss_curves.py \
        --models cassm gpfa kalman --neurons 90 --epochs 30 \
        --cassm-projection-dim 10 --gpfa-latent-dim 10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ladys_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/ladys_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from ladys.datasets import LorenzDataset, LorenzDatasetConfig
from ladys.models import (
    BGPFAConfig,
    CASSMConfig,
    GPFAConfig,
    ILQRVAEConfig,
    KalmanConfig,
    LFADSConfig,
    MINTConfig,
    NDTConfig,
)
from ladys.models.base import BaseModelConfig
from ladys.plotting import (
    legend_outside,
    model_color,
    model_label,
    model_marker,
    plot_context,
    save_figure,
    style_axis,
)
from ladys.preprocessing import PreprocessedDataset, PreprocessingConfig
from ladys.training import Trainer, TrainerConfig
from ladys.training.strategies import build_strategy
from ladys.utils.yaml import load_yaml


MODEL_CONFIGS = {
    "bgpfa": BGPFAConfig,
    "cassm": CASSMConfig,
    "gpfa": GPFAConfig,
    "ilqr_vae": ILQRVAEConfig,
    "kalman": KalmanConfig,
    "lfads": LFADSConfig,
    "mint": MINTConfig,
    "ndt": NDTConfig,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["bgpfa", "cassm", "gpfa", "ilqr_vae", "kalman", "lfads", "mint", "ndt"],
    )
    parser.add_argument("--neurons", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num-inits", type=int, default=10)
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--burn-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="runs/lorenz_loss_curves")
    parser.add_argument("--num-rate-traces", type=int, default=10)
    parser.add_argument("--trace-sample-index", type=int, default=0)
    parser.add_argument("--experiment-config-dir", default="configs/experiment")
    parser.add_argument(
        "--preprocessing-mode",
        choices=["model", "none"],
        default="model",
        help=(
            "Observation preprocessing: 'model' uses each experiment config, "
            "and 'none' uses raw spikes for all models."
        ),
    )
    parser.add_argument(
        "--cassm-projection-dim",
        type=int,
        default=None,
        help="CASSM sparse projection dimension. Defaults to min(20, neurons).",
    )
    parser.add_argument(
        "--gpfa-latent-dim",
        type=int,
        default=3,
        help="GPFA latent dimensionality.",
    )
    parser.add_argument(
        "--gpfa-init-method",
        choices=["fa", "normal", "kaiming", "kaiming_normal", "kaiming_uniform"],
        default="kaiming_normal",
        help="GPFA observation loading initialization.",
    )
    parser.add_argument(
        "--gpfa-init-seed",
        type=int,
        default=None,
        help="GPFA initialization seed. Defaults to --seed when omitted.",
    )
    parser.add_argument(
        "--bgpfa-infer-steps",
        type=int,
        default=300,
        help="Held-out latent inference steps for BGPFA evaluation.",
    )
    parser.add_argument(
        "--bgpfa-infer-mc",
        type=int,
        default=20,
        help="Monte Carlo samples for BGPFA held-out latent inference.",
    )
    parser.add_argument(
        "--bgpfa-infer-lr",
        type=float,
        default=1e-1,
        help="Learning rate for BGPFA held-out latent inference.",
    )
    parser.add_argument(
        "--mint-n-candidates",
        type=int,
        default=None,
        help="MINT interpolation candidates for the Lorenz library.",
    )
    parser.add_argument(
        "--mint-window-length",
        type=int,
        default=None,
        help="MINT likelihood window length in Lorenz time steps.",
    )
    parser.add_argument(
        "--mint-delta",
        type=int,
        default=None,
        help="MINT binning stride in Lorenz time steps.",
    )
    parser.add_argument(
        "--mint-lorenz-library-source",
        choices=["smoothed_spikes", "true_rates"],
        default=None,
        help=(
            "Source for Lorenz MINT trajectory-library rates. "
            "'smoothed_spikes' is the fair default; 'true_rates' is an oracle sanity check."
        ),
    )
    parser.add_argument(
        "--mint-causal",
        action="store_true",
        help="Use causal MINT inference on Lorenz. The default Lorenz preset is acausal.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    trace_rows = []
    for model_name in args.models:
        if model_name not in MODEL_CONFIGS:
            raise KeyError(f"Unknown model '{model_name}'. Choices: {sorted(MODEL_CONFIGS)}")

        print(
            f"Running model={model_name}, neurons={args.neurons}, "
            f"seed={args.seed}, epochs={args.epochs}"
        )
        case_rows, case_trace_rows = run_case(args, model_name, output_dir)
        rows.extend(case_rows)
        trace_rows.extend(case_trace_rows)
        write_group_outputs(output_dir, rows, trace_rows)

    write_group_outputs(output_dir, rows, trace_rows)
    print(f"Wrote {output_dir / 'summary.csv'}")
    print(f"Wrote {plots_dir / 'test_rate_mse_curves.png'}")
    print(f"Wrote {plots_dir / 'test_rate_mse_curves_log.png'}")
    print(f"Wrote {plots_dir / 'test_objective_curves.png'}")
    print(f"Wrote {plots_dir / 'train_test_objective_curves.png'}")
    print(f"Wrote {plots_dir / 'rate_traces_all_models.png'}")


def run_case(
    args: argparse.Namespace,
    model_name: str,
    output_dir: Path,
) -> tuple[list[dict[str, str | int | float]], list[dict[str, str | int | float]]]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model_dir = output_dir / "models" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    dataset_config = LorenzDatasetConfig(
        neurons=args.neurons,
        num_inits=args.num_inits,
        num_trials=args.num_trials,
        num_steps=args.num_steps,
        burn_steps=args.burn_steps,
        seed=args.seed,
    )
    train_ds, test_ds = LorenzDataset.make_splits(dataset_config)
    preprocessing = build_preprocessing_config(
        model_name,
        args.experiment_config_dir,
        args.preprocessing_mode,
        args.neurons,
    )
    train_ds = PreprocessedDataset(train_ds, preprocessing)
    test_ds = PreprocessedDataset(test_ds, preprocessing)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    n_time, n_neurons = train_ds.spikes.shape[1:]

    model_config = build_model_config(args, model_name, n_neurons)
    model = model_config.build(n_neurons=n_neurons, n_time=n_time)
    if model_name == "mint":
        started = time.perf_counter()
        try:
            fit_mint_lorenz_library(model, train_ds, dataset_config, args.device)
            rows = rows_for_inference_only_model(
                args=args,
                model=model,
                model_name=model_name,
                test_loader=test_loader,
                started=started,
            )
            trace_rows = collect_rate_trace_rows(
                model=model,
                dataset=test_ds,
                device=args.device,
                model_name=model_name,
                num_neurons=args.num_rate_traces,
                sample_index=args.trace_sample_index,
                bgpfa_infer_steps=args.bgpfa_infer_steps,
                bgpfa_infer_mc=args.bgpfa_infer_mc,
                bgpfa_infer_lr=args.bgpfa_infer_lr,
            )
            write_rate_traces(model_dir / "rate_traces.csv", trace_rows)
            plot_rate_traces(
                trace_rows,
                model_dir / "rate_traces.png",
                title=f"{model_name} held-out firing-rate traces",
            )
            write_model_artifacts(
                model_dir,
                args,
                model_name,
                model_config,
                dataset_config,
                rows,
                trace_rows,
                model,
            )
            return rows, trace_rows
        except Exception as exc:
            elapsed = time.perf_counter() - started
            print(f"Error for model={model_name}: {exc}")
            rows = [
                {
                    "status": "error",
                    "model": model_name,
                    "neurons": args.neurons,
                    "seed": args.seed,
                    "epoch": -1,
                    "optimizer_seconds": elapsed,
                    "wall_seconds": elapsed,
                    "train_loss": np.nan,
                    "test_loss": np.nan,
                    "test_rate_mse": np.nan,
                    "objective": "",
                    "error": str(exc),
                }
            ]
            write_model_artifacts(model_dir, args, model_name, model_config, dataset_config, rows, [], None)
            return rows, []

    strategy = build_strategy(model_config.optimization)
    trainer = Trainer(TrainerConfig(epochs=args.epochs, device=args.device))
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

    started = time.perf_counter()
    try:
        valid_loader = None if model_name == "bgpfa" else test_loader
        history = trainer.fit(model, strategy, train_loader, valid_loader, metric_fns)
    except Exception as exc:
        elapsed = time.perf_counter() - started
        print(f"Error for model={model_name}: {exc}")
        rows = [
            {
                "status": "error",
                "model": model_name,
                "neurons": args.neurons,
                "seed": args.seed,
                "epoch": -1,
                "optimizer_seconds": elapsed,
                "wall_seconds": elapsed,
                "train_loss": np.nan,
                "test_loss": np.nan,
                "test_rate_mse": np.nan,
                "objective": "",
                "error": str(exc),
            }
        ]
        write_model_artifacts(model_dir, args, model_name, model_config, dataset_config, rows, [], None)
        return rows, []

    trace_rows = collect_rate_trace_rows(
        model=model,
        dataset=test_ds,
        device=args.device,
        model_name=model_name,
        num_neurons=args.num_rate_traces,
        sample_index=args.trace_sample_index,
        bgpfa_infer_steps=args.bgpfa_infer_steps,
        bgpfa_infer_mc=args.bgpfa_infer_mc,
        bgpfa_infer_lr=args.bgpfa_infer_lr,
    )
    write_rate_traces(model_dir / "rate_traces.csv", trace_rows)
    plot_rate_traces(
        trace_rows,
        model_dir / "rate_traces.png",
        title=f"{model_name} held-out firing-rate traces",
    )

    wall_seconds = time.perf_counter() - started
    rows = []
    cumulative_optimizer_seconds = 0.0
    for report in history:
        cumulative_optimizer_seconds += report.seconds
        rows.append(
            {
                "status": "ok",
                "model": model_name,
                "neurons": args.neurons,
                "seed": args.seed,
                "epoch": report.epoch + 1,
                "optimizer_seconds": report.seconds,
                "cumulative_optimizer_seconds": cumulative_optimizer_seconds,
                "wall_seconds": wall_seconds,
                "train_loss": report.train.loss,
                "test_loss": np.nan if report.valid is None else report.valid.loss,
                "test_rate_mse": report.metrics.get("test_rate_mse", np.nan),
                "objective": report.train.objective,
                "error": "",
            }
        )
    write_model_artifacts(model_dir, args, model_name, model_config, dataset_config, rows, trace_rows, model)
    return rows, trace_rows


def fit_mint_lorenz_library(
    model,
    dataset: PreprocessedDataset,
    dataset_config: LorenzDatasetConfig,
    device: str,
) -> None:
    if not hasattr(model, "fit_library"):
        raise TypeError(f"{type(model).__name__} does not expose fit_library().")

    torch_device = torch.device(device)
    model.to(torch_device)
    spikes = getattr(dataset, "raw_spikes", dataset.spikes)
    library_source = getattr(getattr(model, "config", None), "lorenz_library_source", "smoothed_spikes")
    rates = getattr(dataset, "rates", None)
    latents = getattr(dataset, "latents", None)
    if library_source == "true_rates" and rates is None:
        raise AttributeError("MINT Lorenz fitting requires true training rates.")
    z_source = rates if library_source == "true_rates" else latents
    if z_source is None:
        z_source = spikes
    if spikes.ndim != 3 or z_source.ndim != 3:
        raise ValueError("MINT Lorenz fitting expects trial x time x feature tensors.")

    n_trials, n_time, _ = spikes.shape
    if n_trials == 0:
        raise ValueError("MINT Lorenz fitting received an empty training split.")

    if hasattr(model, "settings"):
        model.settings.trial_alignment = range(0, n_time)
        model.settings.test_alignment = range(0, n_time)
    if hasattr(model, "hyperparams"):
        model.hyperparams.trajectories_alignment = range(0, n_time)

    n_conditions = min(max(int(dataset_config.num_inits), 1), int(n_trials))
    condition = np.arange(n_trials, dtype=np.int64) % n_conditions
    if hasattr(model, "hyperparams"):
        model.hyperparams.n_candidates = min(int(model.hyperparams.n_candidates), n_conditions)
        if model.hyperparams.n_candidates < 2:
            model.hyperparams.interp = 1

    spike_trials = [spikes[i].T.contiguous().to(torch_device) for i in range(n_trials)]
    z_trials = [
        z_source[i].T.contiguous().to(device=torch_device, dtype=torch.float64)
        for i in range(n_trials)
    ]
    model.fit_library(spike_trials, z_trials, condition)


def rows_for_inference_only_model(
    args: argparse.Namespace,
    model,
    model_name: str,
    test_loader: DataLoader,
    started: float,
) -> list[dict[str, str | int | float]]:
    test_rate_mse = evaluate_rate_mse(
        model,
        test_loader,
        args.device,
        bgpfa_infer_steps=args.bgpfa_infer_steps,
        bgpfa_infer_mc=args.bgpfa_infer_mc,
        bgpfa_infer_lr=args.bgpfa_infer_lr,
    )
    wall_seconds = time.perf_counter() - started
    rows = []
    for epoch in range(args.epochs):
        optimizer_seconds = wall_seconds if epoch == 0 else 0.0
        rows.append(
            {
                "status": "ok",
                "model": model_name,
                "neurons": args.neurons,
                "seed": args.seed,
                "epoch": epoch + 1,
                "optimizer_seconds": optimizer_seconds,
                "cumulative_optimizer_seconds": wall_seconds,
                "wall_seconds": wall_seconds,
                "train_loss": 0.0,
                "test_loss": 0.0,
                "test_rate_mse": test_rate_mse,
                "objective": getattr(model, "objective", "inference_only"),
                "error": "",
            }
        )
    return rows


def build_model_config(
    args: argparse.Namespace,
    model_name: str,
    n_neurons: int,
) -> BaseModelConfig:
    path = _lorenz_experiment_config_path(args.experiment_config_dir, model_name, n_neurons)
    model_data = None
    if path.exists():
        model_data = dict(load_yaml(path)["model"])

    if model_name == "cassm":
        projection_dim = args.cassm_projection_dim
        if projection_dim is None:
            configured_projection_dim = 20
            if model_data is not None:
                configured_projection_dim = int(
                    model_data.get("projection_dim", configured_projection_dim)
                )
            projection_dim = min(configured_projection_dim, n_neurons)
        if n_neurons % projection_dim != 0:
            raise ValueError(
                "CASSM sparse projection requires neurons to be divisible by "
                f"projection_dim; got neurons={n_neurons}, "
                f"projection_dim={projection_dim}."
            )
        if model_data is None:
            return CASSMConfig(projection_dim=projection_dim)
        model_data["projection_dim"] = projection_dim
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
        if model_data is not None:
            return BaseModelConfig.from_dict(model_data)
        return KalmanConfig()
    if model_name == "bgpfa":
        if model_data is not None:
            return BaseModelConfig.from_dict(model_data)
        return BGPFAConfig()
    if model_name == "ilqr_vae":
        if model_data is not None:
            model_data["held_in_neurons"] = n_neurons
            model_data["output_neuron_start"] = 0
            model_data["output_neurons"] = n_neurons
            return BaseModelConfig.from_dict(model_data)
        return ILQRVAEConfig(
            objective="ilqr_vae_elbo",
            params_path=None,
            initialization="random",
            trainable_parameters=True,
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
            model_data = {"name": "mint", "dataset": "lorenz"}
        model_data["dataset"] = "lorenz"
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
    if model_name in {"lfads", "ndt"}:
        if model_data is not None:
            return BaseModelConfig.from_dict(model_data)
        return MODEL_CONFIGS[model_name]()
    raise KeyError(model_name)


def build_preprocessing_config(
    model_name: str,
    config_dir: str,
    preprocessing_mode: str = "model",
    n_neurons: int | None = None,
) -> PreprocessingConfig:
    if preprocessing_mode == "none":
        return PreprocessingConfig()

    path = _lorenz_experiment_config_path(config_dir, model_name, n_neurons)
    if not path.exists():
        return PreprocessingConfig()
    data = load_yaml(path)
    return PreprocessingConfig.model_validate(data.get("preprocessing", {}))


def _lorenz_experiment_config_path(
    config_dir: str,
    model_name: str,
    n_neurons: int | None = None,
) -> Path:
    root = Path(config_dir)
    candidates = []
    if n_neurons is not None:
        candidates.append(
            root
            / "synthetic"
            / "lorenz"
            / model_name
            / f"{model_name}_lorenz_{n_neurons}.yaml"
        )
    candidates.extend(
        [
            root / "synthetic" / "lorenz" / model_name / f"{model_name}_lorenz.yaml",
        root / "lorenz" / model_name / f"{model_name}_lorenz.yaml",
        root / f"{model_name}_lorenz.yaml",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def evaluate_rate_mse(
    model,
    loader: DataLoader,
    device: str,
    bgpfa_infer_steps: int = 300,
    bgpfa_infer_mc: int = 20,
    bgpfa_infer_lr: float = 1e-1,
) -> float:
    model.eval()
    if hasattr(model, "infer_latents"):
        return evaluate_bgpfa_rate_mse(
            model,
            loader,
            device,
            bgpfa_infer_steps=bgpfa_infer_steps,
            bgpfa_infer_mc=bgpfa_infer_mc,
            bgpfa_infer_lr=bgpfa_infer_lr,
        )

    losses = []
    weights = []
    torch_device = torch.device(device)
    with torch.no_grad():
        for batch in loader:
            spikes = batch["spikes"].to(torch_device)
            rates = batch["rates"].to(torch_device)
            pred = model.predict_rates(spikes)
            loss = torch.mean((pred - rates) ** 2)
            losses.append(float(loss.detach().cpu()))
            weights.append(int(spikes.shape[0]))
    if not losses:
        return float("nan")
    return float(np.average(losses, weights=weights))


def evaluate_bgpfa_rate_mse(
    model,
    loader: DataLoader,
    device: str,
    bgpfa_infer_steps: int,
    bgpfa_infer_mc: int,
    bgpfa_infer_lr: float,
) -> float:
    torch_device = torch.device(device)
    batches = []
    rates = []
    for batch in loader:
        batches.append(batch["spikes"].to(torch_device))
        rates.append(batch["rates"].to(torch_device))
    if not batches:
        return float("nan")

    spikes = torch.cat(batches, dim=0)
    true_rates = torch.cat(rates, dim=0)
    model.infer_latents(
        spikes,
        max_steps=bgpfa_infer_steps,
        n_mc=bgpfa_infer_mc,
        lrate=bgpfa_infer_lr,
        burnin=1,
    )
    with torch.no_grad():
        pred = model.predict_rates(spikes)
        loss = torch.mean((pred - true_rates.to(pred.device, pred.dtype)) ** 2)
    return float(loss.detach().cpu())


def collect_rate_trace_rows(
    model,
    dataset: LorenzDataset,
    device: str,
    model_name: str,
    num_neurons: int,
    sample_index: int,
    bgpfa_infer_steps: int = 300,
    bgpfa_infer_mc: int = 20,
    bgpfa_infer_lr: float = 1e-1,
) -> list[dict[str, str | int | float]]:
    if len(dataset) == 0:
        return []

    sample_index = min(max(sample_index, 0), len(dataset) - 1)
    sample = dataset[sample_index]
    spikes = sample["spikes"].unsqueeze(0).to(device)
    true_rates = sample["rates"].cpu()
    model_input = sample["spikes"].cpu()
    observed = sample.get("raw_spikes", sample["spikes"]).cpu()

    model.eval()
    if hasattr(model, "infer_latents"):
        model.infer_latents(
            spikes,
            max_steps=bgpfa_infer_steps,
            n_mc=bgpfa_infer_mc,
            lrate=bgpfa_infer_lr,
            burnin=1,
        )
    with torch.no_grad():
        pred_rates = model.predict_rates(spikes).squeeze(0).detach().cpu()

    n_neurons = min(num_neurons, true_rates.shape[-1], pred_rates.shape[-1])
    rows = []
    for neuron in range(n_neurons):
        for timestep in range(true_rates.shape[0]):
            rows.append(
                {
                    "model": model_name,
                    "sample_index": sample_index,
                    "neuron": neuron,
                    "time": timestep,
                    "true_rate": float(true_rates[timestep, neuron]),
                    "pred_rate": float(pred_rates[timestep, neuron]),
                    "model_input": float(model_input[timestep, neuron]),
                    "observed_spikes": float(observed[timestep, neuron]),
                }
            )
    return rows


def write_rate_traces(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = [
        "model",
        "sample_index",
        "neuron",
        "time",
        "true_rate",
        "pred_rate",
        "model_input",
        "observed_spikes",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_rate_traces(rows: list[dict], path: Path, title: str) -> None:
    if not rows:
        return

    neurons = sorted({int(row["neuron"]) for row in rows})
    ncols = 2
    nrows = int(np.ceil(len(neurons) / ncols))
    with plot_context(nrows=nrows, ncols=ncols):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            sharex=True,
            squeeze=False,
        )

        for ax, neuron in zip(axes.ravel(), neurons):
            neuron_rows = sorted(
                [row for row in rows if int(row["neuron"]) == neuron],
                key=lambda row: int(row["time"]),
            )
            times = [int(row["time"]) for row in neuron_rows]
            true_rates = [float(row["true_rate"]) for row in neuron_rows]
            pred_rates = [float(row["pred_rate"]) for row in neuron_rows]
            ax.plot(
                times,
                true_rates,
                color=model_color("true"),
                linewidth=1.4,
                label="true",
            )
            ax.plot(
                times,
                pred_rates,
                color=model_color("prediction"),
                linewidth=1.2,
                label="predicted",
            )
            ax.set_title(f"neuron {neuron}")
            style_axis(ax)

        for ax in axes.ravel()[len(neurons) :]:
            ax.axis("off")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.suptitle(title)
        fig.supxlabel("Time")
        fig.supylabel("Firing rate")
        save_figure(fig, path)
        plt.close(fig)


def write_history(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = [
        "status",
        "model",
        "neurons",
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
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_test_rate_mse(rows: list[dict], path: Path, log_y: bool = False) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return

    models = sorted({str(row["model"]) for row in ok_rows})
    with plot_context(nrows=1, ncols=1, rel_width=0.86, height_scale=1.45):
        fig, ax = plt.subplots()

        for model in models:
            model_rows = sorted(
                [row for row in ok_rows if row["model"] == model],
                key=lambda row: int(row["epoch"]),
            )
            epochs = [int(row["epoch"]) for row in model_rows]
            test_rate_mse = [float(row["test_rate_mse"]) for row in model_rows]
            ax.plot(
                epochs,
                test_rate_mse,
                color=model_color(model),
                linewidth=1.4,
                label=model_label(model),
            )

        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Held-out firing-rate MSE")
        ax.set_title("Held-out Firing-Rate MSE")
        style_axis(ax)
        legend_outside(ax)
        save_figure(fig, path)
        plt.close(fig)


def plot_test_objective(rows: list[dict], path: Path) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return

    models = sorted({str(row["model"]) for row in ok_rows})
    with plot_context(nrows=len(models), ncols=1):
        fig, axes = plt.subplots(
            len(models),
            1,
            sharex=True,
            squeeze=False,
        )

        for ax, model in zip(axes[:, 0], models):
            model_rows = sorted(
                [row for row in ok_rows if row["model"] == model],
                key=lambda row: int(row["epoch"]),
            )
            epochs = [int(row["epoch"]) for row in model_rows]
            test_loss = [float(row["test_loss"]) for row in model_rows]
            objective = str(model_rows[0]["objective"])
            ax.plot(
                epochs,
                test_loss,
                marker=model_marker(model),
                color=model_color(model),
                label="test objective",
            )
            ax.set_ylabel("Objective")
            ax.set_title(f"{model_label(model)} ({objective})")
            style_axis(ax)
            ax.legend()

        axes[-1, 0].set_xlabel("Epoch")
        save_figure(fig, path)
        plt.close(fig)


def plot_train_test_objective(rows: list[dict], path: Path) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return

    models = sorted({str(row["model"]) for row in ok_rows})
    with plot_context(nrows=len(models), ncols=1):
        fig, axes = plt.subplots(
            len(models),
            1,
            sharex=True,
            squeeze=False,
        )

        for ax, model in zip(axes[:, 0], models):
            model_rows = sorted(
                [row for row in ok_rows if row["model"] == model],
                key=lambda row: int(row["epoch"]),
            )
            epochs = [int(row["epoch"]) for row in model_rows]
            train_loss = [float(row["train_loss"]) for row in model_rows]
            test_loss = [float(row["test_loss"]) for row in model_rows]
            objective = str(model_rows[0]["objective"])
            color = model_color(model)
            ax.plot(epochs, train_loss, color=color, linestyle="--", label="train")
            ax.plot(epochs, test_loss, color=color, label="test")
            ax.set_ylabel("Loss")
            ax.set_title(f"{model_label(model)} ({objective})")
            style_axis(ax)
            ax.legend()

        axes[-1, 0].set_xlabel("Epoch")
        save_figure(fig, path)
        plt.close(fig)


def write_group_outputs(output_dir: Path, rows: list[dict], trace_rows: list[dict]) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    write_history(output_dir / "summary.csv", rows)
    write_history(output_dir / "lorenz_loss_history.csv", rows)
    write_best_metrics(output_dir / "best_metrics.csv", rows)
    write_group_summary(output_dir / "summary.md", rows)
    plot_test_rate_mse(rows, plots_dir / "test_rate_mse_curves.png")
    plot_test_rate_mse(rows, plots_dir / "test_rate_mse_curves_log.png", log_y=True)
    plot_test_objective(rows, plots_dir / "test_objective_curves.png")
    plot_train_test_objective(rows, plots_dir / "train_test_objective_curves.png")
    plot_combined_rate_traces(trace_rows, plots_dir / "rate_traces_all_models.png")


def write_model_artifacts(
    model_dir: Path,
    args: argparse.Namespace,
    model_name: str,
    model_config: BaseModelConfig,
    dataset_config: LorenzDatasetConfig,
    rows: list[dict],
    trace_rows: list[dict],
    model,
) -> None:
    write_history(model_dir / "history.csv", rows)
    if trace_rows:
        write_rate_traces(model_dir / "rate_traces.csv", trace_rows)
    final = rows[-1] if rows else {}
    best = _best_mse_row(rows)
    metrics = {
        "status": final.get("status"),
        "model": model_name,
        "neurons": args.neurons,
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch": best.get("epoch"),
        "best_test_rate_mse": best.get("test_rate_mse"),
        "best_train_loss": best.get("train_loss"),
        "best_test_loss": best.get("test_loss"),
        "final_epoch": final.get("epoch"),
        "final_train_loss": final.get("train_loss"),
        "final_test_loss": final.get("test_loss"),
        "final_test_rate_mse": final.get("test_rate_mse"),
        "wall_seconds": final.get("wall_seconds"),
        "error": final.get("error", ""),
    }
    (model_dir / "metrics.json").write_text(json.dumps(_json_ready(metrics), indent=2, sort_keys=True) + "\n")
    config = {
        "dataset": dataset_config.model_dump(mode="json"),
        "model": model_config.model_dump(mode="json"),
        "trainer": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "device": args.device,
        },
        "benchmark_args": _json_ready(vars(args)),
    }
    (model_dir / "config.json").write_text(json.dumps(_json_ready(config), indent=2, sort_keys=True) + "\n")
    (model_dir / "report.md").write_text(_model_report_text(metrics) + "\n")
    if model is not None:
        torch.save(model.state_dict(), model_dir / "model.pt")


def write_group_summary(path: Path, rows: list[dict]) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    lines = [
        "# Lorenz Loss-Curve Run Group",
        "",
        (
            "| model | status | best epoch | best test rate MSE | final epoch | "
            "final test rate MSE | final train loss | final test loss | wall seconds |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in _model_names_by_best(rows):
        model_rows = sorted(
            [row for row in rows if row["model"] == model],
            key=lambda row: int(row["epoch"]),
        )
        final = model_rows[-1]
        best = _best_mse_row(model_rows)
        lines.append(
            "| "
            + " | ".join(
                [
                    model_label(model),
                    str(final.get("status", "")),
                    str(best.get("epoch", "")),
                    _fmt(best.get("test_rate_mse")),
                    str(final.get("epoch", "")),
                    _fmt(final.get("test_rate_mse")),
                    _fmt(final.get("train_loss")),
                    _fmt(final.get("test_loss")),
                    _fmt(final.get("wall_seconds")),
                ]
            )
            + " |"
        )
    if ok_rows:
        lines.extend(
            [
                "",
                "## Plots",
                "",
                "- `plots/test_rate_mse_curves.png`",
                "- `plots/test_rate_mse_curves_log.png`",
                "- `plots/test_objective_curves.png`",
                "- `plots/train_test_objective_curves.png`",
                "- `plots/rate_traces_all_models.png`",
            ]
        )
    if any(str(row.get("model")) == "mint" for row in rows):
        lines.extend(
            [
                "",
                "## Notes",
                "",
                (
                    "- MINT is inference-only here: fitting builds a trajectory library once, "
                    "so its metric curve is constant over epochs."
                ),
                (
                    "- Lorenz MINT defaults to spike-derived smoothed trajectory libraries. "
                    "Runs configured with `lorenz_library_source=true_rates` are oracle "
                    "sanity checks only."
                ),
                (
                    "- The repeated-trial Lorenz split measures denoising of seen "
                    "initial-condition trajectories, not generalization to unseen "
                    "trajectories."
                ),
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def write_best_metrics(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = [
        "model",
        "status",
        "best_epoch",
        "best_test_rate_mse",
        "final_epoch",
        "final_test_rate_mse",
        "final_train_loss",
        "final_test_loss",
        "wall_seconds",
        "error",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for model in _model_names_by_best(rows):
            model_rows = sorted(
                [row for row in rows if row["model"] == model],
                key=lambda row: int(row["epoch"]),
            )
            final = model_rows[-1]
            best = _best_mse_row(model_rows)
            writer.writerow(
                {
                    "model": model,
                    "status": final.get("status", ""),
                    "best_epoch": best.get("epoch", ""),
                    "best_test_rate_mse": best.get("test_rate_mse", ""),
                    "final_epoch": final.get("epoch", ""),
                    "final_test_rate_mse": final.get("test_rate_mse", ""),
                    "final_train_loss": final.get("train_loss", ""),
                    "final_test_loss": final.get("test_loss", ""),
                    "wall_seconds": final.get("wall_seconds", ""),
                    "error": final.get("error", ""),
                }
            )


def plot_combined_rate_traces(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    neurons = sorted({int(row["neuron"]) for row in rows})
    models = sorted({str(row["model"]) for row in rows})
    ncols = 2
    nrows = int(np.ceil(len(neurons) / ncols))
    with plot_context(nrows=nrows, ncols=ncols):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            sharex=True,
            squeeze=False,
        )

        for ax, neuron in zip(axes.ravel(), neurons):
            neuron_rows = [row for row in rows if int(row["neuron"]) == neuron]
            first_model = models[0]
            truth_rows = sorted(
                [row for row in neuron_rows if str(row["model"]) == first_model],
                key=lambda row: int(row["time"]),
            )
            if truth_rows:
                ax.plot(
                    [int(row["time"]) for row in truth_rows],
                    [float(row["true_rate"]) for row in truth_rows],
                    color=model_color("true"),
                    linewidth=1.5,
                    label="true",
                )
            for model in models:
                model_rows = sorted(
                    [row for row in neuron_rows if str(row["model"]) == model],
                    key=lambda row: int(row["time"]),
                )
                ax.plot(
                    [int(row["time"]) for row in model_rows],
                    [float(row["pred_rate"]) for row in model_rows],
                    color=model_color(model),
                    linewidth=1.1,
                    label=model_label(model),
                )
            ax.set_title(f"neuron {neuron}")
            style_axis(ax)

        for ax in axes.ravel()[len(neurons) :]:
            ax.axis("off")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc="upper right")
        fig.suptitle("Held-out Firing-Rate Traces by Model")
        fig.supxlabel("Time")
        fig.supylabel("Firing rate")
        save_figure(fig, path)
        plt.close(fig)


def _model_report_text(metrics: dict) -> str:
    return "\n".join(
        [
            "# LaDyS Benchmark Model Run",
            "",
            f"- Model: `{model_label(str(metrics.get('model')))}`",
            f"- Status: `{metrics.get('status')}`",
            f"- Best epoch: `{_fmt(metrics.get('best_epoch'))}`",
            f"- Best test rate MSE: `{_fmt(metrics.get('best_test_rate_mse'))}`",
            f"- Final train loss: `{_fmt(metrics.get('final_train_loss'))}`",
            f"- Final test loss: `{_fmt(metrics.get('final_test_loss'))}`",
            f"- Final test rate MSE: `{_fmt(metrics.get('final_test_rate_mse'))}`",
            f"- Wall seconds: `{_fmt(metrics.get('wall_seconds'))}`",
        ]
    )


def _best_mse_row(rows: list[dict]) -> dict:
    candidates = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        try:
            value = float(row.get("test_rate_mse"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            candidates.append((value, row))
    if not candidates:
        return rows[-1] if rows else {}
    return min(candidates, key=lambda item: item[0])[1]


def _model_names_by_best(rows: list[dict]) -> list[str]:
    return sorted(
        {str(row["model"]) for row in rows},
        key=lambda name: _sort_value(
            _best_mse_row([row for row in rows if row["model"] == name])
        ),
    )


def _sort_value(row: dict) -> float:
    try:
        value = float(row.get("test_rate_mse"))
    except (TypeError, ValueError):
        return float("inf")
    return value if np.isfinite(value) else float("inf")


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    return value


def _fmt(value) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(val):
        return ""
    return f"{val:.6g}"


if __name__ == "__main__":
    main()
