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
        --models cassm gpfa --neurons 100 --epochs 30

    PYTHONPATH=src python3 scripts/benchmark_lorenz_loss_curves.py \
        --models cassm gpfa --neurons 90 --epochs 30 \
        --cassm-projection-dim 10 --gpfa-latent-dim 10
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
from ladys.models import CASSMConfig, GPFAConfig
from ladys.models.base import BaseModelConfig
from ladys.preprocessing import PreprocessedDataset, PreprocessingConfig
from ladys.training import Trainer, TrainerConfig
from ladys.training.strategies import build_strategy
from ladys.utils.yaml import load_yaml


MODEL_CONFIGS = {
    "cassm": CASSMConfig,
    "gpfa": GPFAConfig,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["cassm", "gpfa"])
    parser.add_argument("--neurons", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num-inits", type=int, default=10)
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--burn-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="artifacts/lorenz_loss_curves")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name in args.models:
        if model_name not in MODEL_CONFIGS:
            raise KeyError(f"Unknown model '{model_name}'. Choices: {sorted(MODEL_CONFIGS)}")

        print(
            f"Running model={model_name}, neurons={args.neurons}, "
            f"seed={args.seed}, epochs={args.epochs}"
        )
        rows.extend(run_case(args, model_name, output_dir))
        write_history(output_dir / "lorenz_loss_history.csv", rows)
        plot_test_rate_mse(rows, output_dir / "test_rate_mse_curves.png")
        plot_test_objective(rows, output_dir / "test_objective_curves.png")
        plot_train_test_objective(rows, output_dir / "train_test_objective_curves.png")

    write_history(output_dir / "lorenz_loss_history.csv", rows)
    plot_test_rate_mse(rows, output_dir / "test_rate_mse_curves.png")
    plot_test_objective(rows, output_dir / "test_objective_curves.png")
    plot_train_test_objective(rows, output_dir / "train_test_objective_curves.png")
    print(f"Wrote {output_dir / 'lorenz_loss_history.csv'}")
    print(f"Wrote {output_dir / 'test_rate_mse_curves.png'}")
    print(f"Wrote {output_dir / 'test_objective_curves.png'}")
    print(f"Wrote {output_dir / 'train_test_objective_curves.png'}")


def run_case(
    args: argparse.Namespace,
    model_name: str,
    output_dir: Path,
) -> list[dict[str, str | int | float]]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
    )
    train_ds = PreprocessedDataset(train_ds, preprocessing)
    test_ds = PreprocessedDataset(test_ds, preprocessing)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    n_time, n_neurons = train_ds.spikes.shape[1:]

    model_config = build_model_config(args, model_name, n_neurons)
    model = model_config.build(n_neurons=n_neurons, n_time=n_time)
    strategy = build_strategy(model_config.optimization)
    trainer = Trainer(TrainerConfig(epochs=args.epochs, device=args.device))
    metric_fns = {
        "test_rate_mse": lambda current_model: evaluate_rate_mse(
            current_model,
            test_loader,
            args.device,
        )
    }

    started = time.perf_counter()
    try:
        history = trainer.fit(model, strategy, train_loader, test_loader, metric_fns)
    except Exception as exc:
        elapsed = time.perf_counter() - started
        print(f"Error for model={model_name}: {exc}")
        return [
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

    trace_rows = collect_rate_trace_rows(
        model=model,
        dataset=test_ds,
        device=args.device,
        model_name=model_name,
        num_neurons=args.num_rate_traces,
        sample_index=args.trace_sample_index,
    )
    write_rate_traces(output_dir / f"{model_name}_rate_traces.csv", trace_rows)
    plot_rate_traces(
        trace_rows,
        output_dir / f"{model_name}_rate_traces.png",
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
    return rows


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
    raise KeyError(model_name)


def build_preprocessing_config(
    model_name: str,
    config_dir: str,
    preprocessing_mode: str = "model",
) -> PreprocessingConfig:
    if preprocessing_mode == "none":
        return PreprocessingConfig()

    path = Path(config_dir) / f"{model_name}_lorenz.yaml"
    if not path.exists():
        return PreprocessingConfig()
    data = load_yaml(path)
    return PreprocessingConfig.model_validate(data.get("preprocessing", {}))


def evaluate_rate_mse(model, loader: DataLoader, device: str) -> float:
    model.eval()
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


def collect_rate_trace_rows(
    model,
    dataset: LorenzDataset,
    device: str,
    model_name: str,
    num_neurons: int,
    sample_index: int,
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
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(10, max(3, 2.0 * nrows)),
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
        ax.plot(times, true_rates, color="black", linewidth=1.4, label="true")
        ax.plot(times, pred_rates, color="#1f77b4", linewidth=1.2, label="pred")
        ax.set_title(f"neuron {neuron}")
        ax.grid(True, alpha=0.2)

    for ax in axes.ravel()[len(neurons) :]:
        ax.axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title)
    fig.supxlabel("Time")
    fig.supylabel("Firing rate")
    fig.tight_layout(rect=(0, 0, 0.96, 0.96))
    fig.savefig(path, dpi=200)
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


def plot_test_rate_mse(rows: list[dict], path: Path) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return

    models = sorted({str(row["model"]) for row in ok_rows})
    fig, axes = plt.subplots(
        len(models),
        1,
        figsize=(7, max(3, 2.8 * len(models))),
        sharex=True,
        squeeze=False,
    )

    for ax, model in zip(axes[:, 0], models):
        model_rows = sorted(
            [row for row in ok_rows if row["model"] == model],
            key=lambda row: int(row["epoch"]),
        )
        epochs = [int(row["epoch"]) for row in model_rows]
        test_rate_mse = [float(row["test_rate_mse"]) for row in model_rows]
        ax.plot(epochs, test_rate_mse, marker="o", markersize=3, label="rate MSE")
        ax.set_ylabel("MSE")
        ax.set_title(f"{model} held-out firing-rate MSE")
        ax.grid(True, alpha=0.25)
        ax.legend()

    axes[-1, 0].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_test_objective(rows: list[dict], path: Path) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return

    models = sorted({str(row["model"]) for row in ok_rows})
    fig, axes = plt.subplots(
        len(models),
        1,
        figsize=(7, max(3, 2.8 * len(models))),
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
        ax.plot(epochs, test_loss, marker="o", markersize=3, label="test objective")
        ax.set_ylabel("Objective")
        ax.set_title(f"{model} ({objective})")
        ax.grid(True, alpha=0.25)
        ax.legend()

    axes[-1, 0].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_train_test_objective(rows: list[dict], path: Path) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return

    models = sorted({str(row["model"]) for row in ok_rows})
    fig, axes = plt.subplots(
        len(models),
        1,
        figsize=(7, max(3, 2.8 * len(models))),
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
        ax.plot(epochs, train_loss, linestyle="--", label="train")
        ax.plot(epochs, test_loss, label="test")
        ax.set_ylabel("Loss")
        ax.set_title(f"{model} ({objective})")
        ax.grid(True, alpha=0.25)
        ax.legend()

    axes[-1, 0].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
