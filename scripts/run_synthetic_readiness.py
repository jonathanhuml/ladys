"""Run bounded, isolated learning-curve checks on both synthetic datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=["lorenz", "chaotic_rnn"],
                        default=["lorenz", "chaotic_rnn"])
    parser.add_argument("--models", nargs="+", default=[
        "psth", "smoothing", "mint", "gpfa", "kalman", "cassm", "ndt", "stndt",
        "lfads", "langevin_flow", "bgpfa", "ilqr_vae"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-models", nargs="*", default=[
        "psth", "smoothing", "mint", "gpfa", "kalman", "cassm", "bgpfa", "ilqr_vae"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--neurons", type=int, default=12)
    parser.add_argument("--num-inits", type=int, default=3)
    parser.add_argument("--num-trials", type=int, default=6)
    parser.add_argument("--num-steps", type=int, default=60)
    parser.add_argument("--bgpfa-infer-steps", type=int, default=100)
    parser.add_argument("--bgpfa-infer-mc", type=int, default=5)
    args = parser.parse_args()
    if args.max_seconds <= 0:
        parser.error("max-seconds must be positive")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []

    def save():
        temporary = output / "status.tmp"
        temporary.write_text(json.dumps(records, indent=2) + "\n")
        temporary.replace(output / "status.json")

    for dataset in args.datasets:
        folder = output / dataset
        folder.mkdir(parents=True, exist_ok=True)
        for method in args.models:
            device = "cpu" if method in args.cpu_models else args.device
            command = [sys.executable, "scripts/benchmark_lorenz_loss_curves.py",
                       "--dataset", dataset, "--models", method,
                       "--device", device, "--output-dir", str(folder),
                       "--append-existing", "--epochs", str(args.epochs),
                       "--neurons", str(args.neurons), "--num-inits", str(args.num_inits),
                       "--num-trials", str(args.num_trials), "--num-steps", str(args.num_steps),
                       "--batch-size", "4", "--num-rate-traces", "3",
                       "--cassm-projection-dim", str(max(
                           dim for dim in range(1, min(4, args.neurons) + 1)
                           if args.neurons % dim == 0)),
                       "--bgpfa-infer-steps", str(args.bgpfa_infer_steps),
                       "--bgpfa-infer-mc", str(args.bgpfa_infer_mc),
                       "--max-seconds", str(max(1, args.max_seconds - 30))]
            record = dict(dataset=dataset, model=method, status="running", command=command)
            records.append(record)
            save()
            print(f"Starting {dataset}/{method} ({device})", flush=True)
            started = time.perf_counter()
            with (folder / f"{method}.log").open("w") as log:
                try:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=args.max_seconds, check=False)
                    record.update(status="completed" if result.returncode == 0 else "failed",
                                  returncode=result.returncode)
                except subprocess.TimeoutExpired:
                    record["status"] = "time_limit"
            record["seconds"] = time.perf_counter() - started
            history = folder / "models" / method / "history.csv"
            if history.exists():
                with history.open() as stream:
                    rows = list(csv.DictReader(stream))
                good = [row for row in rows if row["status"] == "ok"]
                record["completed_points"] = len(good)
                if good:
                    record["first_rate_mse"] = float(good[0]["test_rate_mse"])
                    record["final_rate_mse"] = float(good[-1]["test_rate_mse"])
                    record["best_rate_mse"] = min(float(row["test_rate_mse"]) for row in good)
                if any(row["status"] == "error" for row in rows):
                    record.update(status="failed", error=rows[-1].get("error", ""))
                elif any(not math.isfinite(float(row[key])) for row in good
                         for key in ["train_loss", "test_rate_mse"]):
                    record.update(status="failed", error="Nonfinite loss or rate error")
            metrics = folder / "models" / method / "metrics.json"
            if record["status"] == "completed" and metrics.exists():
                if json.loads(metrics.read_text()).get("run_status") == "time_budget":
                    record["status"] = "time_limit"
            save()
            print(json.dumps(record), flush=True)

        # Recover completed epoch artifacts even when a child reaches its hard limit.
        from scripts.benchmark_lorenz_loss_curves import (
            read_existing_history, read_existing_rate_traces, write_group_outputs,
        )
        rows = []
        for history in sorted((folder / "models").glob("*/history.csv")):
            rows.extend(read_existing_history(history, set()))
        if rows:
            write_group_outputs(folder, rows, read_existing_rate_traces(folder, set()))


if __name__ == "__main__":
    main()
