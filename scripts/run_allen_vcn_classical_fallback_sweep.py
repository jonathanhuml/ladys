#!/usr/bin/env python3
"""Run focused CASSM/Kalman fallback sweeps for Allen VCN groups below smoothing."""

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


METHODS = ("cassm", "kalman")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-output-dir", required=True)
    parser.add_argument("--data-path", default="data/real/allen_vcn/allen_vcn_low_trial_20ms.h5")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--groups", nargs="+")
    parser.add_argument("--python", default="/home/jon/torch-gpu/bin/python")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--main-pid", type=int)
    parser.add_argument("--sleep-seconds", type=float, default=300.0)
    parser.add_argument("--margin-threshold", type=float, default=-0.01)
    parser.add_argument("--export-h5", action="store_true")
    args = parser.parse_args()

    base_output = Path(args.base_output_dir)
    sweep_dir = base_output / "classical_fallback_sweep"
    config_dir = sweep_dir / "configs"
    log_dir = sweep_dir / "_logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = sweep_dir / "fallback.log"
    _append_log(log_path, f"classical_fallback_start={_timestamp()}")
    if args.main_pid is not None:
        _wait_for_pid(args.main_pid, args.sleep_seconds, log_path)

    summary_path = _best_available_summary(base_output)
    rows = _read_existing_summary(summary_path)
    if not rows:
        raise RuntimeError(f"No rows found in {summary_path}")

    row_groups = sorted({r["group"] for r in rows if r.get("group")})
    shape_info = read_shape_info(Path(args.data_path), row_groups)
    if args.groups:
        targets = [(method, group) for group in args.groups for method in args.methods]
    else:
        targets = _targets_needing_fallback(rows, args.methods, args.margin_threshold)
    _append_log(log_path, f"summary={summary_path}")
    _append_log(log_path, f"targets={targets}")

    selected_rows = list(rows)
    variant_rows: list[dict[str, str]] = []
    for method, group in targets:
        baseline = _baseline(rows, group)
        best_row = _existing_row(rows, group, method)
        best_score = _float_or_nan(best_row.get("co_bps") if best_row else "")
        for variant_name, patch in _variants(method):
            row = _run_variant(
                method=method,
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
                f"variant={variant_name} method={method} group={group} status={row.get('status')} "
                f"co_bps={row.get('co_bps')} margin={row.get('baseline_margin')}",
            )
            if not math.isnan(score) and (math.isnan(best_score) or score > best_score):
                best_score = score
                best_row = _as_selected_row(row, method)
            if not math.isnan(score) and not math.isnan(baseline) and score - baseline >= args.margin_threshold:
                _append_log(
                    log_path,
                    f"early_stop method={method} group={group} variant={variant_name} "
                    f"co_bps={score} baseline={baseline}",
                )
                break
        if best_row is not None:
            selected_rows = _upsert_row(selected_rows, best_row)
            _append_log(log_path, f"selected method={method} group={group} co_bps={best_row.get('co_bps')}")

    _write_variant_summary(sweep_dir / "variant_summary.csv", variant_rows)
    selected_summary = base_output / "summary_with_classical_fallback.csv"
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

    _append_log(log_path, f"classical_fallback_done={_timestamp()}")
    print(json.dumps({"selected_summary": str(selected_summary), "variant_summary": str(sweep_dir / "variant_summary.csv")}, indent=2))
    return 0


def _wait_for_pid(pid: int, sleep_seconds: float, log_path: Path) -> None:
    while subprocess.run(["kill", "-0", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        _append_log(log_path, f"waiting_for_main={_timestamp()} pid={pid}")
        time.sleep(sleep_seconds)
    _append_log(log_path, f"main_finished={_timestamp()} pid={pid}")


def _best_available_summary(base_output: Path) -> Path:
    for name in (
        "summary_with_classical_fallback.csv",
        "summary_with_ndt_fallback.csv",
        "summary_with_stndt_fallback.csv",
        "summary.csv",
    ):
        path = base_output / name
        if path.exists():
            return path
    return base_output / "summary.csv"


def _targets_needing_fallback(rows: list[dict[str, str]], methods: list[str], threshold: float) -> list[tuple[str, str]]:
    groups = sorted({row.get("group", "") for row in rows if row.get("group")})
    targets: list[tuple[str, str]] = []
    for group in groups:
        baseline = _baseline(rows, group)
        for method in methods:
            row = _existing_row(rows, group, method)
            score = _float_or_nan(row.get("co_bps") if row else "")
            if math.isnan(score) or score - baseline < threshold:
                targets.append((method, group))
    return targets


def _existing_row(rows: list[dict[str, str]], group: str, method: str) -> dict[str, str] | None:
    for row in rows:
        if row.get("group") == group and row.get("method") == method and row.get("status") == "ok":
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
    method: str,
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
    run_name = f"{method}_{dataset}_20ms_nlb_style_fallback_{variant_name}"
    config = build_config(
        group=group,
        method=method,
        data_path=data_path,
        output_dir=str(base_output / "classical_fallback_sweep"),
        run_name=run_name,
        device=device,
        shape=shape,
    )
    _deep_update(config, patch)
    config_path = config_dir / f"{run_name}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    run_dir = base_output / "classical_fallback_sweep" / run_name
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
        method=f"{method}_fallback_{variant_name}",
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


def _variants(method: str) -> list[tuple[str, dict[str, Any]]]:
    if method == "cassm":
        return [
            ("pred_alpha50", {"model": {"nlb_feature_source": "predict_rates", "nlb_decoder": "ridge", "nlb_ridge_alpha": 50.0}}),
            ("pred_alpha100", {"model": {"nlb_feature_source": "predict_rates", "nlb_decoder": "ridge", "nlb_ridge_alpha": 100.0}}),
            ("pred_alpha1000", {"model": {"nlb_feature_source": "predict_rates", "nlb_decoder": "ridge", "nlb_ridge_alpha": 1000.0}}),
            ("pred_alpha2000", {"model": {"nlb_feature_source": "predict_rates", "nlb_decoder": "ridge", "nlb_ridge_alpha": 2000.0}}),
            ("pred_alpha5000", {"model": {"nlb_feature_source": "predict_rates", "nlb_decoder": "ridge", "nlb_ridge_alpha": 5000.0}}),
            ("smooth25_pred_alpha500", {"preprocessing": {"observations": {"kern_sd_ms": 25.0}}, "model": {"nlb_feature_source": "predict_rates", "nlb_decoder": "ridge", "nlb_ridge_alpha": 500.0}}),
            ("smooth75_pred_alpha500", {"preprocessing": {"observations": {"kern_sd_ms": 75.0}}, "model": {"nlb_feature_source": "predict_rates", "nlb_decoder": "ridge", "nlb_ridge_alpha": 500.0}}),
            ("pred_proj64_alpha500", {"model": {"projection_dim": 64, "nlb_feature_source": "predict_rates", "nlb_decoder": "ridge", "nlb_ridge_alpha": 500.0}}),
            ("pred_proj128_alpha500", {"model": {"projection_dim": 128, "nlb_feature_source": "predict_rates", "nlb_decoder": "ridge", "nlb_ridge_alpha": 500.0}}),
            ("pred_proj256_alpha500", {"model": {"projection_dim": 256, "nlb_feature_source": "predict_rates", "nlb_decoder": "ridge", "nlb_ridge_alpha": 500.0}}),
            ("pred_proj512_alpha500", {"model": {"projection_dim": 512, "nlb_feature_source": "predict_rates", "nlb_decoder": "ridge", "nlb_ridge_alpha": 500.0}}),
            ("rates_alpha500", {"model": {"nlb_feature_source": "rates", "nlb_decoder": "ridge", "nlb_ridge_alpha": 500.0}}),
            ("latents_alpha500", {"model": {"nlb_feature_source": "latents", "nlb_decoder": "ridge", "nlb_ridge_alpha": 500.0}}),
            ("pred_poisson", {"model": {"nlb_feature_source": "predict_rates", "nlb_decoder": "poisson"}}),
            ("pred_alpha500", {"model": {"nlb_feature_source": "predict_rates", "nlb_decoder": "ridge", "nlb_ridge_alpha": 500.0}}),
            ("pred_alpha750", {"model": {"nlb_feature_source": "predict_rates", "nlb_decoder": "ridge", "nlb_ridge_alpha": 750.0}}),
            ("pred_alpha1500", {"model": {"nlb_feature_source": "predict_rates", "nlb_decoder": "ridge", "nlb_ridge_alpha": 1500.0}}),
            ("pred_alpha3000", {"model": {"nlb_feature_source": "predict_rates", "nlb_decoder": "ridge", "nlb_ridge_alpha": 3000.0}}),
            ("pred_proj80_alpha2000", {"model": {"projection_dim": 80, "nlb_feature_source": "predict_rates", "nlb_decoder": "ridge", "nlb_ridge_alpha": 2000.0}}),
            ("pred_proj112_alpha2000", {"model": {"projection_dim": 112, "nlb_feature_source": "predict_rates", "nlb_decoder": "ridge", "nlb_ridge_alpha": 2000.0}}),
            ("pred_proj160_alpha2000", {"model": {"projection_dim": 160, "nlb_feature_source": "predict_rates", "nlb_decoder": "ridge", "nlb_ridge_alpha": 2000.0}}),
            ("smooth35_pred_alpha2000", {"preprocessing": {"observations": {"kern_sd_ms": 35.0}}, "model": {"nlb_feature_source": "predict_rates", "nlb_decoder": "ridge", "nlb_ridge_alpha": 2000.0}}),
            ("smooth65_pred_alpha2000", {"preprocessing": {"observations": {"kern_sd_ms": 65.0}}, "model": {"nlb_feature_source": "predict_rates", "nlb_decoder": "ridge", "nlb_ridge_alpha": 2000.0}}),
            ("pred_alpha2000_e200", {"model": {"nlb_feature_source": "predict_rates", "nlb_decoder": "ridge", "nlb_ridge_alpha": 2000.0}, "trainer": {"epochs": 200}}),
            (
                "pred_alpha2000_lr002_e200",
                {
                    "model": {
                        "nlb_feature_source": "predict_rates",
                        "nlb_decoder": "ridge",
                        "nlb_ridge_alpha": 2000.0,
                        "optimization": {"lr": 2.0e-2},
                    },
                    "trainer": {"epochs": 200},
                },
            ),
        ]
    if method == "kalman":
        return [
            ("alpha50", {"model": {"nlb_ridge_alpha": 50.0}}),
            ("alpha100", {"model": {"nlb_ridge_alpha": 100.0}}),
            ("alpha1000", {"model": {"nlb_ridge_alpha": 1000.0}}),
            ("alpha2000", {"model": {"nlb_ridge_alpha": 2000.0}}),
            ("alpha5000", {"model": {"nlb_ridge_alpha": 5000.0}}),
            ("smooth25_alpha500", {"preprocessing": {"observations": {"kern_sd_ms": 25.0}}}),
            ("smooth75_alpha500", {"preprocessing": {"observations": {"kern_sd_ms": 75.0}}}),
        ]
    raise ValueError(method)


def _as_selected_row(row: dict[str, str], method: str) -> dict[str, str]:
    selected = dict(row)
    selected["method"] = method
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
