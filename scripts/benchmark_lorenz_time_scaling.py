"""Benchmark training-time scaling over Lorenz sequence length.

This script keeps neuron count fixed and varies `num_steps`, which is the
time-axis length used by GPFA's dense posterior covariance operations.

Example:
    PYTHONPATH=src python3 scripts/benchmark_lorenz_time_scaling.py \
        --models cassm gpfa kalman lfads --time-steps 10 100 1000 10000 \
        --neurons 100 --cassm-projection-dim 5 --gpfa-latent-dim 5
"""

from __future__ import annotations

import argparse
import csv
import os
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

from benchmark_lorenz_scaling import MODEL_CONFIGS, run_case
from ladys.plotting import (
    model_color,
    model_label,
    model_marker,
    plot_context,
    save_figure,
    style_axis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["cassm", "gpfa", "kalman", "lfads"])
    parser.add_argument("--time-steps", nargs="+", type=int, default=[10, 100, 1000])
    parser.add_argument("--neurons", type=int, default=100)
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
    parser.add_argument("--burn-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="runs/lorenz_time_scaling")
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
    parser.add_argument(
        "--max-gpfa-dense-elements",
        type=int,
        default=100_000_000,
        help=(
            "Skip GPFA cases whose dense posterior matrix would exceed this "
            "many elements. Use --allow-large-gpfa to disable."
        ),
    )
    parser.add_argument("--allow-large-gpfa", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"

    existing = [] if args.overwrite else _read_existing(csv_path)
    completed = {
        (row["model"], int(row["time_steps"]), int(row["seed"]))
        for row in existing
        if row.get("status") in {"ok", "skipped"}
    }
    rows = list(existing)

    for model_name in args.models:
        if model_name not in MODEL_CONFIGS:
            raise KeyError(f"Unknown model '{model_name}'. Choices: {sorted(MODEL_CONFIGS)}")

        for time_steps in args.time_steps:
            for seed in args.seeds:
                key = (model_name, time_steps, seed)
                if key in completed:
                    print(
                        f"Skipping model={model_name}, time_steps={time_steps}, seed={seed}"
                    )
                    continue

                print(f"Running model={model_name}, time_steps={time_steps}, seed={seed}")
                if model_name == "gpfa" and not args.allow_large_gpfa:
                    skip_row = _maybe_skip_large_gpfa(args, time_steps, seed)
                    if skip_row is not None:
                        rows.append(skip_row)
                        _write_outputs(output_dir, rows)
                        continue

                case_args = argparse.Namespace(**vars(args))
                case_args.num_steps = time_steps
                row = run_case(case_args, model_name, args.neurons, seed)
                row["time_steps"] = time_steps
                rows.append(row)
                _write_outputs(output_dir, rows)

    _write_outputs(output_dir, rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {output_dir / 'plots' / 'time_vs_steps.png'}")


def _maybe_skip_large_gpfa(
    args: argparse.Namespace,
    time_steps: int,
    seed: int,
) -> dict[str, str | int | float] | None:
    dense_dim = int(args.gpfa_latent_dim) * int(time_steps)
    dense_elements = dense_dim * dense_dim
    if dense_elements <= int(args.max_gpfa_dense_elements):
        return None

    error = (
        "skipped: GPFA dense posterior matrix would have "
        f"{dense_elements} elements ({dense_dim}x{dense_dim}); "
        f"limit is {args.max_gpfa_dense_elements}. "
        "Use --allow-large-gpfa to run anyway."
    )
    print(error)
    return {
        "status": "skipped",
        "model": "gpfa",
        "neurons": args.neurons,
        "time_steps": time_steps,
        "seed": seed,
        "epochs": args.epochs,
        "seconds": np.nan,
        "seconds_per_epoch": np.nan,
        "wall_seconds": np.nan,
        "train_loss": np.nan,
        "valid_loss": np.nan,
        "rate_mse": np.nan,
        "rate_r2": np.nan,
        "co_bps": np.nan,
        "poisson_nll": np.nan,
        "latent_linear_r2": np.nan,
        "error": error,
    }


def _write_outputs(output_dir: Path, rows: list[dict]) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "summary.csv", rows)
    plot_results(rows, plots_dir / "time_vs_steps.png")
    write_summary(rows, output_dir / "time_scaling_summary.md")


def plot_results(rows: list[dict], path: Path) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return

    with plot_context(nrows=1, ncols=1):
        fig, ax = plt.subplots()
        for model in sorted({str(row["model"]) for row in ok_rows}):
            model_rows = sorted(
                [row for row in ok_rows if row["model"] == model],
                key=lambda row: int(row["time_steps"]),
            )
            ax.plot(
                [int(row["time_steps"]) for row in model_rows],
                [float(row["seconds_per_epoch"]) for row in model_rows],
                marker=model_marker(model),
                color=model_color(model),
                label=model_label(model),
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Time bins per trial")
        ax.set_ylabel("Seconds per epoch")
        ax.set_title("Lorenz Time-Length Scaling")
        style_axis(ax, which="both")
        ax.legend()
        save_figure(fig, path)
        plt.close(fig)


def write_summary(rows: list[dict], path: Path) -> None:
    models = _ordered_models(rows)
    include_ratio = {"cassm", "gpfa"}.issubset(models)
    lines = [
        "# Lorenz Time-Length Scaling Summary",
        "",
        "| time bins | "
        + " | ".join(f"{model_label(model)} s/epoch" for model in models)
        + (" | CASSM/GPFA time ratio" if include_ratio else "")
        + " | "
        + " | ".join(f"{model_label(model)} status" for model in models)
        + " |",
        "| ---: | "
        + " | ".join("---:" for _ in models)
        + (" | ---:" if include_ratio else "")
        + " | "
        + " | ".join("---" for _ in models)
        + " |",
    ]
    by_key = {(str(row["model"]), int(row["time_steps"])): row for row in rows}
    for time_steps in sorted({int(row["time_steps"]) for row in rows}):
        model_rows = [by_key.get((model, time_steps)) for model in models]
        ratio = ""
        cassm = by_key.get(("cassm", time_steps))
        gpfa = by_key.get(("gpfa", time_steps))
        if cassm and gpfa and cassm.get("status") == "ok" and gpfa.get("status") == "ok":
            ratio = f"{float(cassm['seconds_per_epoch']) / float(gpfa['seconds_per_epoch']):.2f}x"
        lines.append(
            "| "
            f"{time_steps} | "
            + " | ".join(_format_seconds(row) for row in model_rows)
            + (f" | {ratio}" if include_ratio else "")
            + " | "
            + " | ".join("" if row is None else str(row.get("status", "")) for row in model_rows)
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def _ordered_models(rows: list[dict]) -> list[str]:
    known_order = ["cassm", "gpfa", "kalman", "lfads"]
    present = {str(row["model"]) for row in rows}
    ordered = [model for model in known_order if model in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _format_seconds(row: dict | None) -> str:
    if row is None or row.get("status") != "ok":
        return ""
    return f"{float(row['seconds_per_epoch']):.6g}"


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
        "time_steps",
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


if __name__ == "__main__":
    main()
