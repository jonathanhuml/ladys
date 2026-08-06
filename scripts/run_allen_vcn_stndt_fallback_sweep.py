#!/usr/bin/env python3
"""Run focused STNDT fallback sweeps for Allen VCN groups below smoothing."""

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

from scripts.run_allen_vcn_nlb_style import (
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
    parser.add_argument("--main-pid", type=int)
    parser.add_argument("--sleep-seconds", type=float, default=300.0)
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=-0.01,
        help="rerun groups whose STNDT margin versus smoothing is below this value",
    )
    parser.add_argument("--export-h5", action="store_true")
    args = parser.parse_args()

    base_output = Path(args.base_output_dir)
    fallback_dir = base_output / "stndt_fallback_sweep"
    config_dir = fallback_dir / "configs"
    log_dir = fallback_dir / "_logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = fallback_dir / "fallback.log"
    _append_log(log_path, f"fallback_start={_timestamp()}")
    _append_log(log_path, f"base_output={base_output}")

    if args.main_pid is not None:
        _wait_for_pid(args.main_pid, args.sleep_seconds, log_path)

    summary_path = base_output / "summary.csv"
    rows = _read_existing_summary(summary_path)
    if not rows:
        raise RuntimeError(f"No rows found in {summary_path}")

    shape_info = read_shape_info(Path(args.data_path), sorted({r["group"] for r in rows if r.get("group")}))
    groups = args.groups or _groups_needing_fallback(rows, args.margin_threshold)
    _append_log(log_path, f"groups={groups}")

    all_variant_rows: list[dict[str, str]] = []
    selected_rows = list(rows)
    for group in groups:
        variants = _stndt_variants(group, shape_info[group])
        baseline = _baseline(rows, group)
        best_row = _existing_stndt_row(rows, group)
        best_score = _float_or_nan(best_row.get("co_bps") if best_row else "")
        _append_log(log_path, f"group={group} existing_stndt={best_score}")

        for variant_name, patch in variants:
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
            all_variant_rows.append(row)
            score = _float_or_nan(row.get("co_bps"))
            _append_log(
                log_path,
                f"variant={variant_name} group={group} status={row.get('status')} "
                f"co_bps={row.get('co_bps')} margin={row.get('baseline_margin')}",
            )
            if not math.isnan(score) and (math.isnan(best_score) or score > best_score):
                best_score = score
                best_row = _as_selected_stndt_row(row)
            if not math.isnan(score) and not math.isnan(baseline) and score >= baseline:
                _append_log(log_path, f"early_stop group={group} variant={variant_name} co_bps={score} baseline={baseline}")
                break

        if best_row is not None:
            selected_rows = _upsert_row(selected_rows, best_row)
            _append_log(log_path, f"selected group={group} co_bps={best_row.get('co_bps')} run={best_row.get('run_name')}")

    _write_variant_summary(fallback_dir / "variant_summary.csv", all_variant_rows)
    selected_summary = base_output / "summary_with_stndt_fallback.csv"
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

    _append_log(log_path, f"fallback_done={_timestamp()}")
    print(json.dumps({"selected_summary": str(selected_summary), "variant_summary": str(fallback_dir / "variant_summary.csv")}, indent=2))
    return 0


def _wait_for_pid(pid: int, sleep_seconds: float, log_path: Path) -> None:
    while _pid_is_running(pid):
        _append_log(log_path, f"waiting_for_main={_timestamp()} pid={pid}")
        time.sleep(sleep_seconds)
    _append_log(log_path, f"main_finished={_timestamp()} pid={pid}")


def _pid_is_running(pid: int) -> bool:
    return subprocess.run(["kill", "-0", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def _groups_needing_fallback(rows: list[dict[str, str]], threshold: float) -> list[str]:
    groups = sorted({row.get("group", "") for row in rows if row.get("group")})
    result = []
    for group in groups:
        target = _baseline(rows, group)
        stndt = _existing_stndt_row(rows, group)
        score = _float_or_nan(stndt.get("co_bps") if stndt else "")
        if math.isnan(score) or score - target < threshold:
            result.append(group)
    return result


def _existing_stndt_row(rows: list[dict[str, str]], group: str) -> dict[str, str] | None:
    for row in rows:
        if row.get("group") == group and row.get("method") == "stndt" and row.get("status") == "ok":
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
    run_name = f"stndt_{dataset}_20ms_nlb_style_fallback_{variant_name}"
    config = build_config(
        group=group,
        method="stndt",
        data_path=data_path,
        output_dir=str(base_output / "stndt_fallback_sweep"),
        run_name=run_name,
        device=device,
        shape=shape,
    )
    _deep_update(config, patch)
    config_path = config_dir / f"{run_name}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    run_dir = base_output / "stndt_fallback_sweep" / run_name
    metrics_path = run_dir / "metrics.json"
    log_path = log_dir / f"{run_name}.log"
    started = time.perf_counter()
    if metrics_path.exists():
        seconds = 0.0
        status = "ok"
        error = ""
    else:
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
        method=f"stndt_fallback_{variant_name}",
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


def _stndt_variants(group: str, shape: dict[str, int]) -> list[tuple[str, dict[str, Any]]]:
    time_bins = int(shape["time"])
    long_forward = max(time_bins - 1, 1)
    long_backward = min(32, max(time_bins - 1, 1))
    base_epochs = 3000
    return [
        (
            "mc_rtt_contrast",
            {
                "model": {
                    "dropout": 0.3,
                    "dropout_rates": 0.3,
                    "dropout_embedding": 0.4,
                    "mask_ratio": 0.25,
                    "mask_token_ratio": 0.85,
                    "mask_random_ratio": 0.85,
                    "mask_max_span": 5,
                    "do_contrast": True,
                    "contrast_mask_ratio": 0.15,
                    "contrast_mask_token_ratio": 0.8,
                    "contrast_mask_random_ratio": 0.8,
                    "contrast_lambda": 0.5,
                    "optimization": {"lr": 0.008, "weight_decay": 0.0003, "warmup_steps": 1000, "total_steps": base_epochs},
                },
                "trainer": {"epochs": base_epochs, "live_eval_interval": 500},
            },
        ),
        (
            "area2_contrast",
            {
                "model": {
                    "context_forward": long_forward,
                    "context_backward": long_backward,
                    "dropout": 0.15,
                    "dropout_rates": 0.15,
                    "dropout_embedding": 0.15,
                    "mask_ratio": 0.3,
                    "mask_token_ratio": 0.8,
                    "mask_random_ratio": 0.95,
                    "mask_max_span": 1,
                    "mask_span_ramp_start": 600,
                    "mask_span_ramp_end": 1200,
                    "do_contrast": True,
                    "contrast_mask_ratio": 0.1,
                    "contrast_mask_token_ratio": 0.8,
                    "contrast_mask_random_ratio": 0.7,
                    "contrast_lambda": 0.3,
                    "optimization": {"lr": 0.0015, "weight_decay": 0.0001, "warmup_steps": 800, "total_steps": base_epochs},
                },
                "trainer": {"epochs": base_epochs, "live_eval_interval": 500},
            },
        ),
        (
            "low_dropout_no_contrast",
            {
                "model": {
                    "dropout": 0.15,
                    "dropout_rates": 0.15,
                    "dropout_embedding": 0.15,
                    "mask_ratio": 0.3,
                    "mask_token_ratio": 0.8,
                    "mask_random_ratio": 0.95,
                    "mask_max_span": 1,
                    "do_contrast": False,
                    "optimization": {"lr": 0.0015, "weight_decay": 0.0001, "warmup_steps": 800, "total_steps": base_epochs},
                },
                "trainer": {"epochs": base_epochs, "live_eval_interval": 500},
            },
        ),
        (
            "ndt_like",
            {
                "model": {
                    "dropout": 0.2,
                    "dropout_rates": 0.2,
                    "dropout_embedding": 0.2,
                    "mask_ratio": 0.25,
                    "mask_token_ratio": 0.8,
                    "mask_random_ratio": 0.8,
                    "mask_max_span": 1,
                    "do_contrast": False,
                    "optimization": {"lr": 0.001, "weight_decay": 5.0e-5, "warmup_steps": 250, "total_steps": 1500},
                },
                "trainer": {"epochs": 1500, "live_eval_interval": 250},
            },
        ),
        (
            "dmfc_contrast",
            {
                "model": {
                    "context_forward": min(4, max(time_bins - 1, 1)),
                    "context_backward": min(8, max(time_bins - 1, 1)),
                    "num_layers": 6,
                    "dropout": 0.1,
                    "dropout_rates": 0.2,
                    "dropout_embedding": 0.2,
                    "mask_ratio": 0.25,
                    "mask_token_ratio": 1.0,
                    "mask_random_ratio": 0.5,
                    "mask_max_span": 1,
                    "do_contrast": True,
                    "contrast_mask_ratio": 0.05,
                    "contrast_mask_token_ratio": 0.5,
                    "contrast_mask_random_ratio": 0.5,
                    "contrast_lambda": 0.1,
                    "optimization": {"lr": 0.001, "weight_decay": 5.0e-5, "warmup_steps": 1000, "total_steps": base_epochs},
                },
                "trainer": {"epochs": base_epochs, "live_eval_interval": 500},
            },
        ),
        (
            "full_context",
            {
                "model": {
                    "context_forward": long_forward,
                    "context_backward": long_backward,
                    "full_context": True,
                    "dropout": 0.2,
                    "dropout_rates": 0.2,
                    "dropout_embedding": 0.2,
                    "mask_ratio": 0.25,
                    "mask_token_ratio": 0.8,
                    "mask_random_ratio": 0.8,
                    "mask_max_span": 1,
                    "do_contrast": False,
                    "optimization": {"lr": 0.001, "weight_decay": 5.0e-5, "warmup_steps": 250, "total_steps": 1500},
                },
                "trainer": {"epochs": 1500, "live_eval_interval": 250},
            },
        ),
    ]


def _as_selected_stndt_row(row: dict[str, str]) -> dict[str, str]:
    selected = dict(row)
    selected["method"] = "stndt"
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
