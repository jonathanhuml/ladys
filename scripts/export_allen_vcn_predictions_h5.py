#!/usr/bin/env python3
"""Export Allen VCN held-out predictions from LaDyS runs to H5 files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="summary.csv from run_allen_vcn_nlb_style.py")
    parser.add_argument("--output-dir", required=True, help="directory for per-method H5 exports")
    parser.add_argument(
        "--include-targets",
        action="store_true",
        help="also include eval_spikes_heldout copied from predictions.npz for self-contained scoring",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        help="optional method subset to export; defaults to every successful method in the summary",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = set(args.methods or [])

    rows = _read_rows(summary_path)
    exported: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        method = row["method"]
        if methods and method not in methods:
            continue
        run_dir = Path(row["run_dir"])
        pred_path = run_dir / "predictions.npz"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing predictions for {method} {row['dataset']}: {pred_path}")
        rates, targets = _load_prediction_arrays(pred_path)
        h5_path = output_dir / f"{method}_allen_vcn_eval_rates.h5"
        group_name = row["dataset"]
        with h5py.File(h5_path, "a") as handle:
            if group_name in handle:
                del handle[group_name]
            group = handle.create_group(group_name)
            group.create_dataset("eval_rates_heldout", data=rates, compression="gzip")
            if args.include_targets:
                group.create_dataset("eval_spikes_heldout", data=targets, compression="gzip")
            group.attrs["method"] = method
            group.attrs["source_run_dir"] = str(run_dir)
            group.attrs["source_predictions"] = str(pred_path)
            group.attrs["co_bps"] = _maybe_float(row.get("co_bps"))
            group.attrs["poisson_nll"] = _maybe_float(row.get("poisson_nll"))
            group.attrs["config"] = row.get("config", "")
        exported.setdefault(method, []).append(
            {
                "dataset": group_name,
                "h5": str(h5_path),
                "rates_shape": list(rates.shape),
                "targets_shape": list(targets.shape),
                "co_bps": _maybe_float(row.get("co_bps")),
                "run_dir": str(run_dir),
                "config": row.get("config", ""),
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(exported, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(output_dir), "manifest": str(manifest_path)}, indent=2))
    return 0


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _load_prediction_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        if "pred_rates" in data and "target_spikes" in data:
            rates = data["pred_rates"]
            targets = data["target_spikes"]
        elif "rates" in data and "spikes" in data:
            rates = data["rates"]
            targets = data["spikes"]
        else:
            keys = ", ".join(data.files)
            raise KeyError(f"{path} must contain pred_rates/target_spikes. Found: {keys}")
    rates = np.asarray(rates, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if rates.shape != targets.shape:
        raise ValueError(f"{path} rates and targets differ: {rates.shape} != {targets.shape}")
    return rates, targets


def _maybe_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
