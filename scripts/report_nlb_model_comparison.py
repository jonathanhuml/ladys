#!/usr/bin/env python3
"""Collect full-data NLB validation results and verify saved prediction scores."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from ladys.nlb_eval import nlb_bits_per_spike
from nlb_tools.evaluation import bits_per_spike


def collect(run_dirs, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = {}
    # Explicit directory order selects replacement runs, never their scores.
    for root in run_dirs:
        for path in sorted(root.glob("*/*/status.json")):
            state = json.loads(path.read_text())
            selected[state["dataset"], state["model"]] = (path.parent, state)
    rows = []
    for (dataset, method), (folder, state) in sorted(selected.items()):
        row = dict(dataset=dataset, model=method, status=state["status"],
                   completed_epochs=state.get("epoch", 0), budget=state.get("epochs"),
                   best_epoch=state.get("best_epoch"), co_bps=None, official_co_bps=None,
                   final_co_bps=state.get("co_bps"), double_dt_co_bps=None,
                   train_trials=state.get("train_trials"), val_trials=state.get("val_trials"),
                   minutes=state.get("seconds", 0) / 60, source=str(folder.resolve()))
        predictions = folder / "predictions.npz"
        if predictions.exists():
            with np.load(predictions) as data:
                rates = data["pred_rates"].astype(np.float64)
                spikes = data["target_spikes"].astype(np.float64)
            row["co_bps"] = nlb_bits_per_spike(rates, spikes)
            row["official_co_bps"] = float(bits_per_spike(rates.copy(), spikes))
            if not np.isclose(row["co_bps"], row["official_co_bps"], rtol=0, atol=1e-9):
                raise ValueError(f"Official NLB score mismatch: {folder}")
            if not np.isclose(row["co_bps"], state["best_co_bps"], rtol=0, atol=1e-7):
                raise ValueError(f"Prediction/status mismatch (checkpoint may be updating): {folder}")
            dt = state["bin_size_ms"] / 1000.0
            row["double_dt_co_bps"] = nlb_bits_per_spike(rates * dt, spikes)
            snapshot = output_dir / dataset / method
            snapshot.mkdir(parents=True, exist_ok=True)
            for name in ("status.json", "config.json", "history.csv", "evaluations.json",
                         "best_metrics.json", "library_training.json", "predictions.npz"):
                source = folder / name
                if source.exists():
                    shutil.copy2(source, snapshot / name)
        rows.append(row)
    (output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    if rows:
        with (output_dir / "results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# Full-Data NLB 5 ms Validation", "",
             "One seed; all training and validation trials. Best validation-selected checkpoints, not test scores.",
             "Running jobs are preliminary. Epochs denote complete passes, except BGPFA where each epoch is one full-batch update.",
             "MINT epoch zero includes the complete library fit (and its configured LFADS training for RTT).", "",
             "| Dataset | Model | Status | Epoch / Budget | Best Epoch | co-BPS | Minutes |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        value = "pending" if row["co_bps"] is None else f"{row['co_bps']:.5f}"
        lines.append(f"| {row['dataset']} | {row['model']} | {row['status']} | "
                     f"{row['completed_epochs']} / {row['budget']} | {row['best_epoch']} | "
                     f"{value} | {row['minutes']:.1f} |")
    lines += ["", "Every available prediction score was independently reproduced with `nlb_tools.evaluation.bits_per_spike`.",
              "`double_dt_co_bps` in the CSV/JSON shows the erroneous score from multiplying count predictions by the bin width again.",
              "Prediction and metadata snapshots accompany this report; model and optimizer checkpoints remain in the source run directories."]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True,
                        help="Later directories replace earlier runs for the same dataset/model.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    collect(args.run_dirs, args.output_dir)
    print((args.output_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
