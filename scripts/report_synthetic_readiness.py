"""Render comparable synthetic learning curves and inspectable final rates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

from scripts.benchmark_lorenz_loss_curves import (
    plt, np, plot_context, save_figure, style_axis, model_label,
    read_existing_history, read_existing_rate_traces, write_group_outputs,
)


DATASETS = {"lorenz": ("Lorenz", "#007C83"), "chaotic_rnn": ("Chaotic RNN", "#C44E52")}
METHODS = ["mint", "ilqr_vae", "bgpfa", "cassm", "gpfa", "kalman",
           "lfads", "langevin_flow", "ndt", "stndt", "psth", "smoothing"]


def read_rows(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream))


def replace_runs(root, replacement):
    status = json.loads((root / "status.json").read_text())
    incoming = json.loads((replacement / "status.json").read_text())
    archive = root / "superseded"
    archive.mkdir(exist_ok=True)
    if not (archive / "status.json").exists():
        shutil.copy2(root / "status.json", archive / "status.json")
    by_key = {(row["dataset"], row["model"]): row for row in incoming}
    for (dataset, method), row in by_key.items():
        if dataset not in DATASETS or method not in METHODS or row["status"] != "completed":
            raise ValueError("Replacement must be a completed known method/dataset run")
        source = replacement / dataset / "models" / method
        target = root / dataset / "models" / method
        archived = archive / dataset / method
        if target.exists() and not archived.exists():
            shutil.copytree(target, archived)
        shutil.copytree(source, target, dirs_exist_ok=True)
        row["source_run"] = str(replacement)
    merged = [by_key.get((row["dataset"], row["model"]), row) for row in status]
    temporary = root / "status.tmp"
    temporary.write_text(json.dumps(merged, indent=2) + "\n")
    temporary.replace(root / "status.json")


def render(root):
    histories = {(dataset, method): read_rows(root / dataset / "models" / method / "history.csv")
                 for dataset in DATASETS for method in METHODS}
    for dataset in DATASETS:
        folder = root / dataset
        rows = []
        for history in sorted((folder / "models").glob("*/history.csv")):
            rows.extend(read_existing_history(history, set()))
        if rows:
            write_group_outputs(folder, rows, read_existing_rate_traces(folder, set()))
    with plot_context(nrows=4, ncols=3):
        fig, axes = plt.subplots(4, 3, figsize=(12, 10), layout="constrained")
        for ax, method in zip(axes.flat, METHODS):
            for dataset, (label, color) in DATASETS.items():
                reference = histories[dataset, "smoothing"]
                rows = [row for row in histories[dataset, method] if row["status"] == "ok"]
                if not rows or not reference:
                    continue
                baseline = float(reference[0]["test_rate_mse"])
                values = [float(row["test_rate_mse"]) / baseline for row in rows]
                ax.plot([int(row["epoch"]) for row in rows], values, color=color,
                        marker="o" if len(rows) <= 4 else None, markersize=3, label=label)
            ax.axhline(1, color="0.55", linestyle=":", linewidth=1)
            ax.set_yscale("log")
            ax.set_title(model_label(method))
            ax.set_xlabel("Training trials per condition" if method == "mint" else "Epoch")
            if method in {"psth", "smoothing"}:
                ax.set_xlabel("Fixed baseline")
                ax.set_xticks([])
            style_axis(ax)
        axes[0, 0].legend(fontsize=9)
        fig.supylabel("Held-out rate MSE / smoothing MSE (lower is better)")
        fig.suptitle("Small synthetic checks: 12 neurons, 3 conditions, 6 trials per condition")
        save_figure(fig, root / "learning_curves.png")
        plt.close(fig)

    for dataset, (label, color) in DATASETS.items():
        with plot_context(nrows=4, ncols=3):
            fig, axes = plt.subplots(4, 3, figsize=(12, 10), layout="constrained")
            for ax, method in zip(axes.flat, METHODS):
                rows = read_rows(root / dataset / "models" / method / "rate_traces.csv")
                rows = [row for row in rows if int(row["neuron"]) == 0]
                if rows:
                    time = [float(row["time"]) for row in rows]
                    ax.plot(time, [float(row["true_rate"]) for row in rows], "k--",
                            linewidth=1.5, label="True rate")
                    ax.plot(time, [float(row["pred_rate"]) for row in rows], color=color,
                            linewidth=1.2, label="Predicted rate")
                else:
                    ax.text(.5, .5, "No final trace", transform=ax.transAxes, ha="center")
                ax.set_title(model_label(method))
                style_axis(ax)
            axes[0, 0].legend(fontsize=9)
            fig.supxlabel("Time (dataset exposure units)")
            fig.supylabel("Firing rate (Hz)")
            fig.suptitle(f"{label}: held-out trial 0, neuron 0, final fitted model")
            save_figure(fig, root / f"{dataset}_rate_traces.png")
            plt.close(fig)

    status = json.loads((root / "status.json").read_text())
    lines = ["# Synthetic Readiness Results", "",
             "12 neurons, 3 conditions, 6 trials per condition, 60 bins; seed 1. "
             "Four repeats per condition train the model and two independent spike repeats validate it. "
             "Both splits share underlying condition trajectories: this is not unseen-condition generalization.", "",
             "Up to 40 optimizer epochs, a 600-second hard cap per method/dataset, "
             "and 100 inference steps / 5 Monte Carlo samples for bGPFA validation. "
             "MINT instead fits libraries with 1, 2, 3, then 4 training repeats per condition. "
             "PSTH and smoothing are noniterative baselines.", "",
             "LangevinFlow uses the explicit `current` encoder alignment, matching "
             "[paper Algorithm 1](https://arxiv.org/html/2507.11531v2). "
             "The released code's lagged-input behavior remains available as `upstream_lagged`. "
             "Earlier diagnostic fits with that lagged indexing are retained under `superseded/`. "
             "This is a documented paper/code discrepancy, not a claim of exact released-code parity.", "",
             "## Figures", "",
             "![Learning curves](learning_curves.png)", "",
             "Teal is Lorenz; red is chaotic RNN. The dotted line is smoothing (ratio 1). "
             "Lower is better. Each dataset has its own smoothing normalization. "
             "For MINT the horizontal axis counts training repeats, not optimizer epochs.", "",
             "## Actual Rate Errors", "",
             "Rate MSE is in Hz squared. First means after the first completed epoch/library fit, "
             "not untrained initialization. Best values are descriptive validation minima, not independent test scores.", "",
             "| Dataset | Method | Status | Points | First MSE | Final MSE | Best MSE | Seconds |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in status:
        lines.append(f"| {row['dataset']} | {row['model']} | {row['status']} | "
                     f"{row.get('completed_points', 0)} | {row.get('first_rate_mse', np.nan):.4g} | "
                     f"{row.get('final_rate_mse', np.nan):.4g} | {row.get('best_rate_mse', np.nan):.4g} | "
                     f"{row.get('seconds', 0):.1f} |")
    lines.extend(["", "## Quality Flags", "",
                  "Completed means the requested fits finished with finite outputs, not that "
                  "they passed a denoising-quality threshold.", ""])
    for row in status:
        first, final, best = (row.get(key, np.nan) for key in
                              ["first_rate_mse", "final_rate_mse", "best_rate_mse"])
        if final > first:
            lines.append(f"- {row['dataset']}/{row['model']}: rate MSE worsened "
                         f"from {first:.4g} to {final:.4g}.")
        elif final > 1.25 * best:
            lines.append(f"- {row['dataset']}/{row['model']}: final MSE {final:.4g} "
                         f"is worse than its earlier best {best:.4g}.")
    lines.extend(["", "The paper-current LangevinFlow chaotic-RNN fit was also checked "
                  "independently against its saved checkpoint: count/Hz conversion and "
                  "grad-enabled versus no-grad predictions agree. Poor denoising is not "
                  "explained by a metric-unit mismatch. The earlier released-code-lagged "
                  "fit remains available for comparison, not substituted into the final plot."])
    lines.extend(["", "## Recovered Rates", "",
                  "Black dashed curves are known truth; solid colored curves are final predictions. "
                  "The same held-out trial and neuron are shown for every method. Axes autoscale per panel.", "",
                  "![Lorenz rates](lorenz_rate_traces.png)", "",
                  "![Chaotic RNN rates](chaotic_rnn_rate_traces.png)", "",
                  "## Native Objectives", "",
                  "Per-dataset `plots/train_test_objective_curves.png` uses dashed training and solid "
                  "validation losses. Objectives have different scales and regularization schedules; "
                  "do not compare their numerical levels across methods. bGPFA has no native validation "
                  "ELBO curve here, but does have held-out rate errors. LFADS/LangevinFlow/bGPFA warmups "
                  "may be unfinished after 40 epochs. Finite losses and decreasing curves do not establish "
                  "convergence or published-method parity.", ""])
    (root / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--replace-from", type=Path)
    args = parser.parse_args()
    if args.replace_from is not None:
        replace_runs(args.output_dir, args.replace_from)
    render(args.output_dir)
