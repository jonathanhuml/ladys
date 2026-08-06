#!/usr/bin/env python3
"""Run focused NDT fallback sweeps for Allen VCN groups below smoothing."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_allen_vcn_nlb_style import (  # noqa: E402
    _metric,
    _read_existing_summary,
    _read_json,
    _run_dir_from_stdout,
    _upsert_row,
    _write_summary,
    build_config,
    read_shape_info,
    row_from_run,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-output-dir", required=True)
    parser.add_argument("--data-path", default="data/real/allen_vcn/allen_vcn_low_trial_20ms.h5")
    parser.add_argument("--groups", nargs="+")
    parser.add_argument("--python", default="/home/jon/torch-gpu/bin/python")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--sleep-seconds", type=float, default=300.0)
    parser.add_argument("--margin-threshold", type=float, default=-0.01)
    parser.add_argument("--export-h5", action="store_true")
    args = parser.parse_args()

    base_output = Path(args.base_output_dir)
    sweep_dir = base_output / "ndt_fallback_sweep"
    config_dir = sweep_dir / "configs"
    log_dir = sweep_dir / "_logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = sweep_dir / "fallback.log"
    _append_log(log_path, f"ndt_fallback_start={_timestamp()}")

    if args.wait_pid is not None:
        _wait_for_pid(args.wait_pid, args.sleep_seconds, log_path)

    summary_path = _best_available_summary(base_output)
    rows = _read_existing_summary(summary_path)
    if not rows:
        raise RuntimeError(f"No rows found in {summary_path}")

    row_groups = sorted({r["group"] for r in rows if r.get("group")})
    shape_info = read_shape_info(Path(args.data_path), row_groups)
    groups = args.groups or _groups_needing_fallback(rows, args.margin_threshold)
    _append_log(log_path, f"summary={summary_path}")
    _append_log(log_path, f"groups={groups}")

    selected_rows = list(rows)
    variant_rows: list[dict[str, str]] = []
    for group in groups:
        baseline = _baseline(rows, group)
        best_row = _existing_ndt_row(rows, group)
        best_score = _float_or_nan(best_row.get("co_bps") if best_row else "")
        for variant_name, patch in _variants(shape_info[group]):
            row = _run_variant(
                group=group,
                variant_name=variant_name,
                patch=patch,
                shape=shape_info[group],
                data_path=args.data_path,
                base_output=base_output,
                config_dir=config_dir,
                log_dir=log_dir,
                python_bin=args.python,
                device=args.device,
                baseline_rows=rows,
            )
            variant_rows.append(row)
            score = _float_or_nan(row.get("co_bps"))
            _append_log(
                log_path,
                f"variant={variant_name} group={group} status={row.get('status')} "
                f"co_bps={row.get('co_bps')} margin={row.get('baseline_margin')}",
            )
            if not math.isnan(score) and (math.isnan(best_score) or score > best_score):
                best_score = score
                best_row = _as_selected_ndt_row(row)
            if not math.isnan(score) and not math.isnan(baseline) and score - baseline >= args.margin_threshold:
                _append_log(log_path, f"early_stop group={group} variant={variant_name} co_bps={score} baseline={baseline}")
                break
        if best_row is not None:
            selected_rows = _upsert_row(selected_rows, best_row)
            _append_log(log_path, f"selected group={group} co_bps={best_row.get('co_bps')}")

    _write_variant_summary(sweep_dir / "variant_summary.csv", variant_rows)
    selected_summary = base_output / "summary_with_ndt_fallback.csv"
    _write_summary(selected_summary, selected_rows)
    _write_margin_report(selected_rows, base_output / "overnight_margin_report.txt")

    if args.export_h5:
        subprocess.run(
            [
                args.python,
                "scripts/export_allen_vcn_predictions_h5.py",
                "--summary",
                str(selected_summary),
                "--output-dir",
                str(base_output / "h5_predictions"),
                "--include-targets",
            ],
            cwd=Path.cwd(),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    _append_log(log_path, f"ndt_fallback_done={_timestamp()}")
    print(json.dumps({"selected_summary": str(selected_summary), "variant_summary": str(sweep_dir / "variant_summary.csv")}, indent=2))
    return 0


def _wait_for_pid(pid: int, sleep_seconds: float, log_path: Path) -> None:
    while subprocess.run(["kill", "-0", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        _append_log(log_path, f"waiting={_timestamp()} pid={pid}")
        time.sleep(sleep_seconds)
    _append_log(log_path, f"wait_done={_timestamp()} pid={pid}")


def _best_available_summary(base_output: Path) -> Path:
    for name in ("summary_with_classical_fallback.csv", "summary_with_stndt_fallback.csv", "summary.csv"):
        path = base_output / name
        if path.exists():
            return path
    return base_output / "summary.csv"


def _groups_needing_fallback(rows: list[dict[str, str]], threshold: float) -> list[str]:
    groups = sorted({row.get("group", "") for row in rows if row.get("group")})
    result = []
    for group in groups:
        baseline = _baseline(rows, group)
        ndt = _existing_ndt_row(rows, group)
        score = _float_or_nan(ndt.get("co_bps") if ndt else "")
        if math.isnan(score) or score - baseline < threshold:
            result.append(group)
    return result


def _existing_ndt_row(rows: list[dict[str, str]], group: str) -> dict[str, str] | None:
    for row in rows:
        if row.get("group") == group and row.get("method") == "ndt" and row.get("status") == "ok":
            return row
    return None


def _baseline(rows: list[dict[str, str]], group: str) -> float:
    values = [
        _float_or_nan(row.get("co_bps", ""))
        for row in rows
        if row.get("group") == group and row.get("method") in {"psth", "smoothing"}
    ]
    values = [value for value in values if not math.isnan(value)]
    return max(values) if values else math.nan


def _run_variant(
    *,
    group: str,
    variant_name: str,
    patch: dict[str, Any],
    shape: dict[str, int],
    data_path: str,
    base_output: Path,
    config_dir: Path,
    log_dir: Path,
    python_bin: str,
    device: str,
    baseline_rows: list[dict[str, str]],
) -> dict[str, str]:
    dataset = f"allen_vcn_{group}"
    run_name = f"ndt_{dataset}_20ms_nlb_style_fallback_{variant_name}"
    config = build_config(
        group=group,
        method="ndt",
        data_path=data_path,
        output_dir=str(base_output / "ndt_fallback_sweep"),
        run_name=run_name,
        device=device,
        shape=shape,
    )
    _deep_update(config, patch)
    config_path = config_dir / f"{run_name}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    run_dir = base_output / "ndt_fallback_sweep" / run_name
    metrics_path = run_dir / "metrics.json"
    log_path = log_dir / f"{run_name}.log"
    if metrics_path.exists():
        seconds = 0.0
        status = "ok"
        error = ""
    else:
        started = time.perf_counter()
        proc = subprocess.run(
            [python_bin, "-m", "ladys.cli", "run", "-c", str(config_path)],
            cwd=Path.cwd(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        seconds = time.perf_counter() - started
        log_path.write_text(proc.stdout)
        status = "ok" if proc.returncode == 0 else f"exit_{proc.returncode}"
        error = "" if proc.returncode == 0 else proc.stdout[-1600:].replace("\n", " | ")
        run_dir = _run_dir_from_stdout(proc.stdout, default=run_dir)

    row = row_from_run(
        group=group,
        method=f"ndt_fallback_{variant_name}",
        dataset=dataset,
        run_name=run_name,
        config_path=config_path,
        log_path=log_path,
        run_dir=run_dir,
        seconds=seconds,
        status=status,
        error=error,
        rows=baseline_rows,
    )
    row["fallback_variant"] = variant_name
    metrics = _read_json(run_dir / "metrics.json")
    row["co_bps"] = _metric(metrics, "co_bps")
    row["poisson_nll"] = _metric(metrics, "poisson_nll")
    return row


def _variants(shape: dict[str, int]) -> list[tuple[str, dict[str, Any]]]:
    full = int(shape["full"])
    padded_full = full + (full % 2)
    time_bins = int(shape["time"])
    long_context = min(100, max(time_bins - 1, 1))
    short_context = min(64, max(time_bins - 1, 1))
    return [
        (
            "linear_timestep",
            {"model": {"output_neurons": padded_full, "num_heads": 2, "mask_mode": "timestep"}},
        ),
        (
            "linear_timestep_3000",
            {
                "model": {
                    "output_neurons": padded_full,
                    "num_heads": 2,
                    "mask_mode": "timestep",
                    "optimization": {"lr": 5.0e-4, "warmup_steps": 300, "total_steps": 3000},
                },
                "trainer": {"epochs": 3000, "live_eval_interval": 500},
            },
        ),
        (
            "linear_full_3000",
            {
                "model": {
                    "output_neurons": padded_full,
                    "num_heads": 2,
                    "mask_mode": "full",
                    "optimization": {"lr": 5.0e-4, "warmup_steps": 300, "total_steps": 3000},
                },
                "trainer": {"epochs": 3000, "live_eval_interval": 500},
            },
        ),
        (
            "linear_timestep_high_dropout",
            {
                "model": {
                    "output_neurons": padded_full,
                    "num_heads": 2,
                    "mask_mode": "timestep",
                    "dropout": 0.5,
                    "dropout_rates": 0.5,
                    "dropout_embedding": 0.5,
                }
            },
        ),
        (
            "canonical_heads1",
            {
                "model": {
                    "output_neurons": full,
                    "num_heads": 1,
                    "context_forward": short_context,
                    "context_backward": short_context,
                    "dropout": 0.5,
                    "dropout_rates": 0.5,
                    "dropout_embedding": 0.5,
                    "linear_embedder": False,
                    "embed_dim": 0,
                    "mask_mode": "timestep",
                }
            },
        ),
        (
            "heads1_linear_no_pad",
            {
                "model": {
                    "output_neurons": full,
                    "num_heads": 1,
                    "linear_embedder": True,
                    "embed_dim": 1,
                    "mask_mode": "full",
                }
            },
        ),
        (
            "mcrtt_3000",
            {
                "model": {
                    "output_neurons": padded_full,
                    "num_heads": 2,
                    "context_forward": long_context,
                    "context_backward": long_context,
                    "dropout": 0.3,
                    "dropout_rates": 0.3,
                    "dropout_embedding": 0.3,
                    "mask_mode": "timestep",
                    "optimization": {"lr": 5.0e-4, "warmup_steps": 300, "total_steps": 3000},
                },
                "trainer": {"epochs": 3000, "live_eval_interval": 500},
            },
        ),
    ]


def _as_selected_ndt_row(row: dict[str, str]) -> dict[str, str]:
    selected = dict(row)
    selected["method"] = "ndt"
    selected["error"] = f"selected_from_fallback_variant={row.get('fallback_variant', '')}"
    return selected


def _write_variant_summary(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_margin_report(rows: list[dict[str, str]], path: Path) -> None:
    groups = sorted({row.get("group", "") for row in rows if row.get("group")})
    lines: list[str] = []
    for group in groups:
        smoothing = _baseline(rows, group)
        lines.append(f"[{group}] smoothing={smoothing:.9f}")
        group_rows = [row for row in rows if row.get("group") == group and row.get("status") == "ok"]
        for row in sorted(group_rows, key=lambda r: (math.isnan(_float_or_nan(r.get("co_bps", ""))), -_float_or_nan(r.get("co_bps", "")), r.get("method", ""))):
            score = _float_or_nan(row.get("co_bps", ""))
            margin = score - smoothing if not math.isnan(score) and not math.isnan(smoothing) else math.nan
            if math.isnan(margin):
                status = "missing"
            elif margin >= 0:
                status = "beats"
            elif margin >= -0.02:
                status = "slight_miss"
            else:
                status = "miss"
            lines.append(f"  {row.get('method',''):<14} co_bps={score: .9f} margin={margin: .9f} {status}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _deep_update(target: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)


def _float_or_nan(value: str | float | None) -> float:
    try:
        return float(value) if value not in {None, ""} else math.nan
    except (TypeError, ValueError):
        return math.nan


def _append_log(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


if __name__ == "__main__":
    raise SystemExit(main())
