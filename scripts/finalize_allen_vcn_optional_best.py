#!/usr/bin/env python3
"""Merge Allen VCN optional sweeps, export H5 predictions, and validate shapes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import h5py


METHOD_ORDER = [
    "psth",
    "smoothing",
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
]

SUMMARY_FIELDS = [
    "group",
    "dataset",
    "method",
    "run_name",
    "status",
    "seconds",
    "co_bps",
    "poisson_nll",
    "baseline_target",
    "baseline_margin",
    "beats_baseline",
    "run_dir",
    "config",
    "log",
    "error",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-output-dir", required=True)
    parser.add_argument(
        "--summaries",
        nargs="+",
        default=[
            "summary_with_all_and_cassm_ensemble_top4.csv",
            "summary_with_optional_fallback.csv",
            "summary_with_bgpfa_ordered_gpfa_like_fallback.csv",
        ],
    )
    parser.add_argument("--output-summary", default="summary_with_all_optional_best_ordered.csv")
    parser.add_argument(
        "--report-name",
        default="all_methods_margin_report_all_optional_best_ordered.txt",
    )
    parser.add_argument(
        "--manifest-json",
        default="selected_config_manifest_all_optional_best_ordered.json",
    )
    parser.add_argument(
        "--manifest-csv",
        default="selected_config_manifest_all_optional_best_ordered.csv",
    )
    parser.add_argument(
        "--h5-output-name",
        default="h5_predictions_all_optional_best_ordered",
    )
    parser.add_argument("--wait-pid-files", nargs="*", default=[])
    parser.add_argument("--sleep-seconds", type=float, default=300.0)
    parser.add_argument("--python", default="/home/jon/torch-gpu/bin/python")
    args = parser.parse_args()

    base = Path(args.base_output_dir)
    log_path = base / "finalize_all_optional_best_ordered.log"
    for pid_file in args.wait_pid_files:
        _wait_for_pid_file(Path(pid_file), args.sleep_seconds, log_path)

    rows = _merge_summaries(base, args.summaries)
    output_summary = base / args.output_summary
    _write_summary(output_summary, rows)
    _write_margin_report(rows, base / args.report_name)
    _write_manifest(rows, base / args.manifest_json, base / args.manifest_csv)

    h5_dir = base / args.h5_output_name
    subprocess.run(
        [
            args.python,
            "scripts/export_allen_vcn_predictions_h5.py",
            "--summary",
            str(output_summary),
            "--output-dir",
            str(h5_dir),
            "--include-targets",
        ],
        cwd=Path.cwd(),
        check=True,
    )
    validation = _validate_h5_manifest(h5_dir / "manifest.json")
    validation_path = h5_dir / "validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    if validation["issues"]:
        raise RuntimeError(f"H5 validation found issues: {validation['issues']}")

    result = {
        "summary": str(output_summary),
        "report": str(base / args.report_name),
        "manifest_json": str(base / args.manifest_json),
        "h5_output_dir": str(h5_dir),
        "validation": str(validation_path),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _wait_for_pid_file(path: Path, sleep_seconds: float, log_path: Path) -> None:
    if not path.exists():
        _append_log(log_path, f"missing_pid_file={path} at={_timestamp()}")
        return
    text = path.read_text().strip()
    if not text:
        _append_log(log_path, f"empty_pid_file={path} at={_timestamp()}")
        return
    pid = int(text.split()[0])
    while _pid_is_alive(pid):
        _append_log(log_path, f"waiting_pid={pid} source={path} at={_timestamp()}")
        time.sleep(sleep_seconds)
    _append_log(log_path, f"pid_finished={pid} source={path} at={_timestamp()}")


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _merge_summaries(base: Path, names: list[str]) -> list[dict[str, str]]:
    merged: dict[tuple[str, str], dict[str, str]] = {}
    for name in names:
        path = base / name
        if not path.exists():
            continue
        for row in _read_rows(path):
            if row.get("status") != "ok":
                continue
            group = row.get("group", "")
            method = row.get("method", "")
            if not group or not method:
                continue
            key = (group, method)
            if key not in merged or _score(row) > _score(merged[key]):
                merged[key] = dict(row)

    rows = list(merged.values())
    _recompute_baseline_fields(rows)
    return sorted(rows, key=lambda row: (row.get("group", ""), _method_order(row.get("method", ""))))


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def _write_margin_report(rows: list[dict[str, str]], path: Path) -> None:
    lines: list[str] = []
    for group in sorted({row.get("group", "") for row in rows if row.get("group")}):
        baseline = _baseline(rows, group)
        lines.append(f"[{group}] smoothing={baseline:.9f}")
        group_rows = [row for row in rows if row.get("group") == group]
        for row in sorted(group_rows, key=lambda item: _sort_score(item)):
            score = _score(row)
            margin = score - baseline if math.isfinite(score) and math.isfinite(baseline) else math.nan
            if not math.isfinite(margin):
                status = "missing"
            elif margin >= 0.0:
                status = "beats"
            elif margin >= -0.02:
                status = "slight_miss"
            else:
                status = "miss"
            lines.append(
                f"  {row.get('method', ''):<14} co_bps={score: .9f} "
                f"margin={margin: .9f} {status}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_manifest(rows: list[dict[str, str]], json_path: Path, csv_path: Path) -> None:
    selected = [
        {
            "group": row.get("group", ""),
            "dataset": row.get("dataset", ""),
            "method": row.get("method", ""),
            "co_bps": row.get("co_bps", ""),
            "run_dir": row.get("run_dir", ""),
            "config": row.get("config", ""),
            "run_name": row.get("run_name", ""),
            "error": row.get("error", ""),
        }
        for row in rows
    ]
    payload = {"generated_at": _timestamp(), "rows": selected}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    fields = list(selected[0].keys()) if selected else []
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)


def _validate_h5_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    issues: list[Any] = []
    for method, entries in manifest.items():
        datasets = {entry.get("dataset") for entry in entries}
        if len(entries) != 4:
            issues.append({"method": method, "entries": len(entries)})
        if len(datasets) != len(entries):
            issues.append({"method": method, "duplicate_datasets": sorted(datasets)})
        for entry in entries:
            with h5py.File(entry["h5"], "r") as handle:
                group = handle[entry["dataset"]]
                rates = group["eval_rates_heldout"]
                targets = group.get("eval_spikes_heldout")
                if targets is None:
                    issues.append({"method": method, "dataset": entry["dataset"], "issue": "missing_targets"})
                elif rates.shape != targets.shape:
                    issues.append(
                        {
                            "method": method,
                            "dataset": entry["dataset"],
                            "rates_shape": list(rates.shape),
                            "targets_shape": list(targets.shape),
                        }
                    )
    return {"manifest": str(path), "methods": sorted(manifest), "issues": issues}


def _recompute_baseline_fields(rows: list[dict[str, str]]) -> None:
    for row in rows:
        baseline = _baseline(rows, row.get("group", ""))
        score = _score(row)
        row["baseline_target"] = "" if not math.isfinite(baseline) else f"{baseline:.10g}"
        if math.isfinite(score) and math.isfinite(baseline):
            margin = score - baseline
            row["baseline_margin"] = f"{margin:.10g}"
            row["beats_baseline"] = str(margin > 0.0).lower()
        else:
            row["baseline_margin"] = ""
            row["beats_baseline"] = "false"


def _baseline(rows: list[dict[str, str]], group: str) -> float:
    values = [
        _score(row)
        for row in rows
        if row.get("group") == group and row.get("method") in {"psth", "smoothing"}
    ]
    values = [value for value in values if math.isfinite(value)]
    return max(values) if values else math.nan


def _score(row: dict[str, str]) -> float:
    try:
        return float(row.get("co_bps", ""))
    except (TypeError, ValueError):
        return -math.inf


def _sort_score(row: dict[str, str]) -> tuple[bool, float, str]:
    score = _score(row)
    return not math.isfinite(score), -score if math.isfinite(score) else 0.0, row.get("method", "")


def _method_order(method: str) -> int:
    try:
        return METHOD_ORDER.index(method)
    except ValueError:
        return len(METHOD_ORDER)


def _append_log(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


if __name__ == "__main__":
    raise SystemExit(main())
