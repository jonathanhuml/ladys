"""Benchmark training-time scaling on Lorenz datasets with increasing neurons.

This script is intended to become a PR artifact generator. It dynamically
creates Lorenz datasets at fixed trial count and increasing neuron count, trains
one or more methods through the shared trainer/strategy contract, then writes:

- `lorenz_scaling_results.csv`
- `lorenz_scaling_results.npy`
- `time_vs_neurons.png`

Example:
    PYTHONPATH=src python3 scripts/benchmark_lorenz_scaling.py \
        --models cassm gpfa kalman lfads --neurons 10 100 1000 --seeds 1 2 3 4 5

    PYTHONPATH=src python3 scripts/benchmark_lorenz_scaling.py \
        --models cassm gpfa kalman lfads --neurons 90 900 --cassm-projection-dim 10 \
        --gpfa-latent-dim 10
"""

from __future__ import annotations

import argparse
import csv
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
from ladys.metrics import evaluate_model
from ladys.models import (
    CASSMConfig,
    GPFAConfig,
    KalmanConfig,
    LFADSConfig,
    NDTConfig,
)
from ladys.models.base import BaseModelConfig
from ladys.plotting import (
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
    "cassm": CASSMConfig,
    "gpfa": GPFAConfig,
    "kalman": KalmanConfig,
    "lfads": LFADSConfig,
    "ndt": NDTConfig,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["cassm", "gpfa", "kalman", "lfads", "ndt"],
    )
    parser.add_argument("--neurons", nargs="+", type=int, default=[10, 100, 1000])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--lfads-epochs",
        type=int,
        default=None,
        help="Optional LFADS-specific epoch count for accuracy runs.",
    )
    parser.add_argument("--num-inits", type=int, default=10)
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--burn-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="runs/lorenz_scaling")
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
    parser.add_argument("--lfads-generator-dim", type=int, default=64)
    parser.add_argument("--lfads-factor-dim", type=int, default=20)
    parser.add_argument("--lfads-inferred-input-dim", type=int, default=2)
    parser.add_argument("--lfads-encoder-dim", type=int, default=64)
    parser.add_argument("--lfads-controller-dim", type=int, default=64)
    parser.add_argument("--lfads-lr", type=float, default=1e-3)
    parser.add_argument("--lfads-keep-prob", type=float, default=0.95)
    parser.add_argument("--ndt-hidden-size", type=int, default=128)
    parser.add_argument("--ndt-num-layers", type=int, default=6)
    parser.add_argument("--ndt-num-heads", type=int, default=2)
    parser.add_argument("--ndt-embed-dim", type=int, default=2)
    parser.add_argument("--ndt-mask-ratio", type=float, default=0.25)
    parser.add_argument("--ndt-lr", type=float, default=1e-3)
    parser.add_argument("--ndt-weight-decay", type=float, default=0.0)
    parser.add_argument("--ndt-dropout", type=float, default=0.1)
    parser.add_argument("--ndt-dropout-rates", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"

    existing = [] if args.overwrite else _read_existing(csv_path)
    completed = {
        (row["model"], int(row["neurons"]), int(row["seed"]))
        for row in existing
        if row.get("status") == "ok"
    }
    rows = list(existing)

    for model_name in args.models:
        if model_name not in MODEL_CONFIGS:
            raise KeyError(f"Unknown model '{model_name}'. Choices: {sorted(MODEL_CONFIGS)}")

        for neurons in args.neurons:
            for seed in args.seeds:
                key = (model_name, neurons, seed)
                if key in completed:
                    print(f"Skipping model={model_name}, neurons={neurons}, seed={seed}")
                    continue

                print(f"Running model={model_name}, neurons={neurons}, seed={seed}")
                row = run_case(args, model_name, neurons, seed)
                rows.append(row)
                _write_csv(csv_path, rows)
                _write_numpy(output_dir / "summary.npy", rows)
                plot_results(rows, plots_dir / "time_vs_neurons.png")
                write_summary(rows, output_dir / "summary.md")

    _write_csv(csv_path, rows)
    _write_numpy(output_dir / "summary.npy", rows)
    plot_results(rows, plots_dir / "time_vs_neurons.png")
    write_summary(rows, output_dir / "summary.md")
    print(f"Wrote {csv_path}")
    print(f"Wrote {plots_dir / 'time_vs_neurons.png'}")


def run_case(
    args: argparse.Namespace,
    model_name: str,
    neurons: int,
    seed: int,
) -> dict[str, str | int | float]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_config = LorenzDatasetConfig(
        neurons=neurons,
        num_inits=args.num_inits,
        num_trials=args.num_trials,
        num_steps=args.num_steps,
        burn_steps=args.burn_steps,
        seed=seed,
    )
    train_ds, valid_ds = LorenzDataset.make_splits(dataset_config)
    preprocessing = build_preprocessing_config(
        model_name,
        args.experiment_config_dir,
        args.preprocessing_mode,
    )
    train_ds = PreprocessedDataset(train_ds, preprocessing)
    valid_ds = PreprocessedDataset(valid_ds, preprocessing)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)
    n_time, n_neurons = train_ds.spikes.shape[1:]

    model_config = build_model_config(args, model_name, n_neurons)
    model = model_config.build(n_neurons=n_neurons, n_time=n_time)
    strategy = build_strategy(model_config.optimization)
    epochs = epochs_for_model(args, model_name)
    trainer = Trainer(TrainerConfig(epochs=epochs, device=args.device))

    started = time.perf_counter()
    try:
        history = trainer.fit(model, strategy, train_loader, valid_loader)
        wall_seconds = time.perf_counter() - started
        optimizer_seconds = sum(report.seconds for report in history)
        final = history[-1]
        metrics = evaluate_model(model, valid_loader, args.device).metrics
        return {
            "status": "ok",
            "model": model_name,
            "neurons": neurons,
            "seed": seed,
            "epochs": epochs,
            "seconds": optimizer_seconds,
            "seconds_per_epoch": optimizer_seconds / max(epochs, 1),
            "wall_seconds": wall_seconds,
            "train_loss": final.train.loss,
            "valid_loss": np.nan if final.valid is None else final.valid.loss,
            "rate_mse": metrics.get("rate_mse", np.nan),
            "rate_r2": metrics.get("rate_r2", np.nan),
            "co_bps": metrics.get("co_bps", np.nan),
            "poisson_nll": metrics.get("poisson_nll", np.nan),
            "latent_linear_r2": metrics.get("latent_linear_r2", np.nan),
            "error": "",
        }
    except Exception as exc:
        elapsed = time.perf_counter() - started
        print(f"Error for model={model_name}, neurons={neurons}, seed={seed}: {exc}")
        return {
            "status": "error",
            "model": model_name,
            "neurons": neurons,
            "seed": seed,
            "epochs": epochs,
            "seconds": elapsed,
            "seconds_per_epoch": elapsed / max(epochs, 1),
            "wall_seconds": elapsed,
            "train_loss": np.nan,
            "valid_loss": np.nan,
            "rate_mse": np.nan,
            "rate_r2": np.nan,
            "co_bps": np.nan,
            "poisson_nll": np.nan,
            "latent_linear_r2": np.nan,
            "error": str(exc),
        }


def build_model_config(
    args: argparse.Namespace,
    model_name: str,
    n_neurons: int,
) -> BaseModelConfig:
    if model_name == "cassm":
        projection_dim = args.cassm_projection_dim
        if projection_dim is None:
            projection_dim = min(20, n_neurons)
        if n_neurons % projection_dim != 0:
            raise ValueError(
                "CASSM sparse projection requires neurons to be divisible by "
                f"projection_dim; got neurons={n_neurons}, "
                f"projection_dim={projection_dim}."
            )
        return CASSMConfig(projection_dim=projection_dim)
    if model_name == "gpfa":
        return GPFAConfig(latent_dim=args.gpfa_latent_dim)
    if model_name == "kalman":
        return KalmanConfig()
    if model_name == "lfads":
        return LFADSConfig(
            generator_dim=args.lfads_generator_dim,
            factor_dim=args.lfads_factor_dim,
            inferred_input_dim=args.lfads_inferred_input_dim,
            g0_encoder_dim=args.lfads_encoder_dim,
            controller_encoder_dim=args.lfads_encoder_dim,
            controller_dim=args.lfads_controller_dim,
            keep_prob=args.lfads_keep_prob,
            optimization={
                "name": "gradient",
                "optimizer": "Adam",
                "lr": args.lfads_lr,
                "weight_decay": 0.0,
                "gradient_clip": 200.0,
            },
        )
    if model_name == "ndt":
        return NDTConfig(
            hidden_size=getattr(args, "ndt_hidden_size", 128),
            num_layers=getattr(args, "ndt_num_layers", 6),
            num_heads=getattr(args, "ndt_num_heads", 2),
            embed_dim=getattr(args, "ndt_embed_dim", 2),
            mask_ratio=getattr(args, "ndt_mask_ratio", 0.25),
            dropout=getattr(args, "ndt_dropout", 0.1),
            dropout_rates=getattr(args, "ndt_dropout_rates", 0.2),
            optimization={
                "name": "gradient",
                "optimizer": "Adam",
                "lr": getattr(args, "ndt_lr", 1e-3),
                "weight_decay": getattr(args, "ndt_weight_decay", 0.0),
                "gradient_clip": 200.0,
            },
        )
    raise KeyError(model_name)


def epochs_for_model(args: argparse.Namespace, model_name: str) -> int:
    if model_name == "lfads" and args.lfads_epochs is not None:
        return int(args.lfads_epochs)
    return int(args.epochs)


def build_preprocessing_config(
    model_name: str,
    config_dir: str,
    preprocessing_mode: str = "model",
) -> PreprocessingConfig:
    if preprocessing_mode == "none":
        return PreprocessingConfig()

    path = _lorenz_experiment_config_path(config_dir, model_name)
    if not path.exists():
        return PreprocessingConfig()
    data = load_yaml(path)
    return PreprocessingConfig.model_validate(data.get("preprocessing", {}))


def _lorenz_experiment_config_path(config_dir: str, model_name: str) -> Path:
    root = Path(config_dir)
    candidates = [
        root / "synthetic" / "lorenz" / model_name / f"{model_name}_lorenz.yaml",
        root / "lorenz" / model_name / f"{model_name}_lorenz.yaml",
        root / f"{model_name}_lorenz.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def plot_results(rows: list[dict], path: Path) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return

    models = sorted({str(row["model"]) for row in ok_rows})
    with plot_context(nrows=1, ncols=1):
        fig, ax = plt.subplots()
        for model in models:
            model_rows = [row for row in ok_rows if row["model"] == model]
            neurons = sorted({int(row["neurons"]) for row in model_rows})
            means = []
            lows = []
            highs = []
            for n in neurons:
                vals = np.array(
                    [
                        float(row["seconds_per_epoch"])
                        for row in model_rows
                        if int(row["neurons"]) == n
                    ],
                    dtype=float,
                )
                means.append(float(np.mean(vals)))
                lows.append(float(np.min(vals)))
                highs.append(float(np.max(vals)))
            means_arr = np.array(means)
            color = model_color(model)
            ax.plot(
                neurons,
                means_arr,
                marker=model_marker(model),
                color=color,
                label=model_label(model),
            )
            if any(np.array(highs) > np.array(lows)):
                ax.fill_between(
                    neurons,
                    lows,
                    highs,
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                )

        ax.set_xscale("log")
        ax.set_xlabel("Number of neurons")
        ax.set_ylabel("Seconds per epoch")
        ax.set_title("Lorenz Neuron Scaling")
        style_axis(ax, which="both")
        ax.legend()
        save_figure(fig, path)
        plt.close(fig)


def write_summary(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    models = sorted({str(row["model"]) for row in rows})
    neurons = sorted({int(row["neurons"]) for row in rows})
    by_key = {(str(row["model"]), int(row["neurons"]), int(row["seed"])): row for row in rows}
    seeds = sorted({int(row["seed"]) for row in rows})
    lines = [
        "# Lorenz Neuron-Scaling Run Group",
        "",
        "| neurons | " + " | ".join(f"{model_label(model)} s/epoch" for model in models) + " |",
        "| ---: | " + " | ".join("---:" for _ in models) + " |",
    ]
    for n in neurons:
        values = []
        for model in models:
            vals = [
                float(by_key[(model, n, seed)]["seconds_per_epoch"])
                for seed in seeds
                if (model, n, seed) in by_key and by_key[(model, n, seed)].get("status") == "ok"
            ]
            values.append("" if not vals else f"{float(np.mean(vals)):.6g}")
        lines.append(f"| {n} | " + " | ".join(values) + " |")
    lines.extend(["", "## Plots", "", "- `plots/time_vs_neurons.png`"])
    path.write_text("\n".join(lines) + "\n")


def _read_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = [
        "status",
        "model",
        "neurons",
        "seed",
        "epochs",
        "seconds",
        "seconds_per_epoch",
        "wall_seconds",
        "train_loss",
        "valid_loss",
        "rate_mse",
        "rate_r2",
        "co_bps",
        "poisson_nll",
        "latent_linear_r2",
        "error",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_numpy(path: Path, rows: list[dict]) -> None:
    dtype = [
        ("status", "U16"),
        ("model", "U32"),
        ("neurons", "i8"),
        ("seed", "i8"),
        ("epochs", "i8"),
        ("seconds", "f8"),
        ("seconds_per_epoch", "f8"),
        ("wall_seconds", "f8"),
        ("train_loss", "f8"),
        ("valid_loss", "f8"),
        ("rate_mse", "f8"),
        ("rate_r2", "f8"),
        ("co_bps", "f8"),
        ("poisson_nll", "f8"),
        ("latent_linear_r2", "f8"),
        ("error", "U256"),
    ]
    arr = np.empty(len(rows), dtype=dtype)
    for idx, row in enumerate(rows):
        values = []
        for name, dtype_name in dtype:
            value = row.get(name, "")
            if value == "" and dtype_name.startswith("f"):
                value = np.nan
            elif value == "" and dtype_name.startswith("i"):
                value = 0
            values.append(value)
        arr[idx] = tuple(values)
    np.save(path, arr)


if __name__ == "__main__":
    main()
