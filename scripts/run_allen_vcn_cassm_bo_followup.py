#!/usr/bin/env python3
"""Run targeted CASSM follow-up variants for Allen BO drifting one TF."""

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

from scripts.export_allen_vcn_predictions_h5 import main as export_h5_main  # noqa: E402
from scripts.run_allen_vcn_nlb_style import (  # noqa: E402
    _read_existing_summary,
    _run_dir_from_stdout,
    _upsert_row,
    _write_summary,
    build_config,
    read_shape_info,
    row_from_run,
)


DEFAULT_GROUP = "bo_drifting_one_tf"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-output-dir", required=True)
    parser.add_argument("--data-path", default="data/real/allen_vcn/allen_vcn_low_trial_20ms.h5")
    parser.add_argument("--group", default=DEFAULT_GROUP)
    parser.add_argument("--python", default="/home/jon/torch-gpu/bin/python")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--summary-name", default=None)
    parser.add_argument("--stop-on-beat", action="store_true")
    parser.add_argument("--export-h5", action="store_true")
    args = parser.parse_args()

    base_output = Path(args.base_output_dir)
    sweep_dir = base_output / "cassm_bo_followup_sweep"
    config_dir = sweep_dir / "configs"
    log_dir = sweep_dir / "_logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    status_path = sweep_dir / "status.json"
    log_path = sweep_dir / "driver.log"
    _append_log(log_path, f"followup_start={_timestamp()}")

    summary_path = _best_available_summary(base_output, args.summary_name)
    rows = _read_existing_summary(summary_path)
    if not rows:
        raise RuntimeError(f"No rows found in {summary_path}")
    shape = read_shape_info(Path(args.data_path), [args.group])[args.group]
    dataset = f"allen_vcn_{args.group}"
    baseline = _baseline(rows, args.group)
    best_row = _existing_row(rows, args.group, "cassm")
    best_score = _float_or_nan(best_row.get("co_bps") if best_row else "")
    variants = _variants()

    manifest = {
        "base_output_dir": str(base_output),
        "summary": str(summary_path),
        "data_path": args.data_path,
        "group": args.group,
        "dataset": dataset,
        "baseline": baseline,
        "initial_best_cassm": best_score,
        "shape": shape,
        "variants": [{"name": variant["name"], "patch": variant["patch"]} for variant in variants],
    }
    (sweep_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    selected_rows = list(rows)
    variant_rows = _read_existing_variant_rows(sweep_dir / "variant_summary.csv")
    selected_summary = base_output / "summary_with_cassm_followup.csv"
    for index, variant in enumerate(variants, start=1):
        _write_status(
            status_path,
            status="running",
            variant=variant["name"],
            index=index,
            total=len(variants),
            best_score=best_score,
            baseline=baseline,
        )
        row = _run_variant(
            variant=variant,
            group=args.group,
            dataset=dataset,
            shape=shape,
            data_path=args.data_path,
            base_output=base_output,
            config_dir=config_dir,
            log_dir=log_dir,
            python_bin=args.python,
            device=args.device,
            baseline_rows=rows,
        )
        variant_rows = _upsert_variant_row(variant_rows, row)
        _write_variant_summary(sweep_dir / "variant_summary.csv", variant_rows)

        score = _float_or_nan(row.get("co_bps"))
        margin = score - baseline if not math.isnan(score) and not math.isnan(baseline) else math.nan
        _append_log(
            log_path,
            f"variant={variant['name']} status={row.get('status')} "
            f"co_bps={row.get('co_bps')} margin={margin:.10g}",
        )
        if not math.isnan(score) and (math.isnan(best_score) or score > best_score):
            best_score = score
            best_row = _as_selected_row(row)
            selected_rows = _upsert_row(selected_rows, best_row)
            _write_summary(selected_summary, selected_rows)
            _write_margin_report(selected_rows, base_output / "overnight_margin_report.txt")
            _write_selected_manifest(selected_rows, base_output / "selected_config_manifest.json", base_output / "selected_config_manifest.csv")

        if args.stop_on_beat and not math.isnan(margin) and margin > 0.0:
            _append_log(log_path, f"stop_on_beat variant={variant['name']} co_bps={score} baseline={baseline}")
            break

    if best_row is not None:
        selected_rows = _upsert_row(selected_rows, best_row)
    _write_summary(selected_summary, selected_rows)
    _write_margin_report(selected_rows, base_output / "overnight_margin_report.txt")
    _write_selected_manifest(selected_rows, base_output / "selected_config_manifest.json", base_output / "selected_config_manifest.csv")

    if args.export_h5:
        _export_h5(selected_summary, base_output / "h5_predictions_cassm_followup")

    _write_status(
        status_path,
        status="complete",
        variant="",
        index=len(variant_rows),
        total=len(variants),
        best_score=best_score,
        baseline=baseline,
    )
    _append_log(log_path, f"followup_done={_timestamp()} best_score={best_score:.10g} baseline={baseline:.10g}")
    print(json.dumps({"selected_summary": str(selected_summary), "variant_summary": str(sweep_dir / "variant_summary.csv")}, indent=2))
    return 0


def _variants() -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for seed in (7, 1, 2, 3, 5, 11, 17, 23, 31, 43):
        variants.append(
            {
                "name": f"seed{seed}_pred_alpha2000_proj96_e100",
                "patch": {
                    "dataset": {"seed": seed},
                    "model": {
                        "projection_dim": 96,
                        "use_dense_projection": True,
                        "nlb_feature_source": "predict_rates",
                        "nlb_decoder": "ridge",
                        "nlb_ridge_alpha": 2000.0,
                    },
                    "trainer": {"epochs": 100},
                },
            }
        )

    variants.extend(
        [
            {
                "name": "seed7_mcmaze_latents_alpha200_proj20_e10",
                "patch": {
                    "dataset": {"seed": 7},
                    "model": {
                        "projection_dim": 20,
                        "use_dense_projection": True,
                        "nlb_feature_source": "latents",
                        "nlb_decoder": "ridge",
                        "nlb_ridge_alpha": 200.0,
                    },
                    "trainer": {"epochs": 10},
                },
            },
            {
                "name": "seed7_mcmaze_latents_alpha200_proj20_e100",
                "patch": {
                    "dataset": {"seed": 7},
                    "model": {
                        "projection_dim": 20,
                        "use_dense_projection": True,
                        "nlb_feature_source": "latents",
                        "nlb_decoder": "ridge",
                        "nlb_ridge_alpha": 200.0,
                    },
                    "trainer": {"epochs": 100},
                },
            },
            {
                "name": "seed7_latents_alpha500_proj20_e100",
                "patch": {
                    "dataset": {"seed": 7},
                    "model": {
                        "projection_dim": 20,
                        "use_dense_projection": True,
                        "nlb_feature_source": "latents",
                        "nlb_decoder": "ridge",
                        "nlb_ridge_alpha": 500.0,
                    },
                    "trainer": {"epochs": 100},
                },
            },
            {
                "name": "seed7_latents_alpha2000_proj20_e100",
                "patch": {
                    "dataset": {"seed": 7},
                    "model": {
                        "projection_dim": 20,
                        "use_dense_projection": True,
                        "nlb_feature_source": "latents",
                        "nlb_decoder": "ridge",
                        "nlb_ridge_alpha": 2000.0,
                    },
                    "trainer": {"epochs": 100},
                },
            },
            {
                "name": "seed7_rates_alpha2000_proj96_e100",
                "patch": {
                    "dataset": {"seed": 7},
                    "model": {
                        "projection_dim": 96,
                        "use_dense_projection": True,
                        "nlb_feature_source": "rates",
                        "nlb_decoder": "ridge",
                        "nlb_ridge_alpha": 2000.0,
                    },
                    "trainer": {"epochs": 100},
                },
            },
            {
                "name": "seed7_pred_alpha2000_proj192_e100",
                "patch": {
                    "dataset": {"seed": 7},
                    "model": {
                        "projection_dim": 192,
                        "use_dense_projection": True,
                        "nlb_feature_source": "predict_rates",
                        "nlb_decoder": "ridge",
                        "nlb_ridge_alpha": 2000.0,
                    },
                    "trainer": {"epochs": 100},
                },
            },
            {
                "name": "seed7_pred_alpha2000_proj384_e100",
                "patch": {
                    "dataset": {"seed": 7},
                    "model": {
                        "projection_dim": 384,
                        "use_dense_projection": True,
                        "nlb_feature_source": "predict_rates",
                        "nlb_decoder": "ridge",
                        "nlb_ridge_alpha": 2000.0,
                    },
                    "trainer": {"epochs": 100},
                },
            },
            {
                "name": "seed7_pred_alpha2000_sparse_proj96_e100",
                "patch": {
                    "dataset": {"seed": 7},
                    "model": {
                        "projection_dim": 96,
                        "use_dense_projection": False,
                        "nlb_feature_source": "predict_rates",
                        "nlb_decoder": "ridge",
                        "nlb_ridge_alpha": 2000.0,
                    },
                    "trainer": {"epochs": 100},
                },
            },
            {
                "name": "seed7_smooth45_pred_alpha2000_proj96_e100",
                "patch": {
                    "dataset": {"seed": 7},
                    "preprocessing": {"observations": {"kern_sd_ms": 45.0}},
                    "model": {
                        "projection_dim": 96,
                        "use_dense_projection": True,
                        "nlb_feature_source": "predict_rates",
                        "nlb_decoder": "ridge",
                        "nlb_ridge_alpha": 2000.0,
                    },
                    "trainer": {"epochs": 100},
                },
            },
            {
                "name": "seed7_smooth55_pred_alpha2000_proj96_e100",
                "patch": {
                    "dataset": {"seed": 7},
                    "preprocessing": {"observations": {"kern_sd_ms": 55.0}},
                    "model": {
                        "projection_dim": 96,
                        "use_dense_projection": True,
                        "nlb_feature_source": "predict_rates",
                        "nlb_decoder": "ridge",
                        "nlb_ridge_alpha": 2000.0,
                    },
                    "trainer": {"epochs": 100},
                },
            },
            {
                "name": "seed7_pred_alpha2500_proj96_e100",
                "patch": {
                    "dataset": {"seed": 7},
                    "model": {
                        "projection_dim": 96,
                        "use_dense_projection": True,
                        "nlb_feature_source": "predict_rates",
                        "nlb_decoder": "ridge",
                        "nlb_ridge_alpha": 2500.0,
                    },
                    "trainer": {"epochs": 100},
                },
            },
            {
                "name": "seed7_pred_alpha2000_proj96_lr008_e150",
                "patch": {
                    "dataset": {"seed": 7},
                    "model": {
                        "projection_dim": 96,
                        "use_dense_projection": True,
                        "nlb_feature_source": "predict_rates",
                        "nlb_decoder": "ridge",
                        "nlb_ridge_alpha": 2000.0,
                        "optimization": {"lr": 8.0e-2},
                    },
                    "trainer": {"epochs": 150},
                },
            },
            {
                "name": "seed7_pred_poisson_proj96_e100",
                "patch": {
                    "dataset": {"seed": 7},
                    "model": {
                        "projection_dim": 96,
                        "use_dense_projection": True,
                        "nlb_feature_source": "predict_rates",
                        "nlb_decoder": "poisson",
                    },
                    "trainer": {"epochs": 100},
                },
            },
        ]
    )
    return variants


def _run_variant(
    *,
    variant: dict[str, Any],
    group: str,
    dataset: str,
    shape: dict[str, int],
    data_path: str,
    base_output: Path,
    config_dir: Path,
    log_dir: Path,
    python_bin: str,
    device: str,
    baseline_rows: list[dict[str, str]],
) -> dict[str, str]:
    variant_name = str(variant["name"])
    run_name = f"cassm_{dataset}_20ms_nlb_style_followup_{variant_name}"
    config = build_config(
        group=group,
        method="cassm",
        data_path=data_path,
        output_dir=str(base_output / "cassm_bo_followup_sweep"),
        run_name=run_name,
        device=device,
        shape=shape,
    )
    _deep_update(config, variant["patch"])
    config_path = config_dir / f"{run_name}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    default_run_dir = base_output / "cassm_bo_followup_sweep" / run_name
    log_path = log_dir / f"{run_name}.log"
    existing_run_dir = _find_existing_run_dir(default_run_dir)
    if existing_run_dir is not None:
        run_dir = existing_run_dir
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
        run_dir = _run_dir_from_stdout(proc.stdout, default=default_run_dir)

    row = row_from_run(
        group=group,
        method=f"cassm_followup_{variant_name}",
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
    row["method_selected_as"] = "cassm"
    return row


def _find_existing_run_dir(default_run_dir: Path) -> Path | None:
    candidates = [default_run_dir]
    candidates.extend(sorted(default_run_dir.parent.glob(f"{default_run_dir.name}-*")))
    for candidate in candidates:
        if (candidate / "metrics.json").exists():
            return candidate
    return None


def _best_available_summary(base_output: Path, explicit_name: str | None) -> Path:
    if explicit_name is not None:
        return base_output / explicit_name
    for name in (
        "summary_with_cassm_followup.csv",
        "summary_with_classical_fallback.csv",
        "summary_with_ndt_fallback.csv",
        "summary_with_stndt_fallback.csv",
        "summary.csv",
    ):
        path = base_output / name
        if path.exists():
            return path
    return base_output / "summary.csv"


def _baseline(rows: list[dict[str, str]], group: str) -> float:
    values = [
        _float_or_nan(row.get("co_bps", ""))
        for row in rows
        if row.get("group") == group and row.get("method") in {"psth", "smoothing"}
    ]
    values = [value for value in values if not math.isnan(value)]
    return max(values) if values else math.nan


def _existing_row(rows: list[dict[str, str]], group: str, method: str) -> dict[str, str] | None:
    for row in rows:
        if row.get("group") == group and row.get("method") == method and row.get("status") == "ok":
            return row
    return None


def _as_selected_row(row: dict[str, str]) -> dict[str, str]:
    selected = dict(row)
    selected["method"] = "cassm"
    selected["error"] = f"selected_from_followup_variant={row.get('fallback_variant', '')}"
    return selected


def _read_existing_variant_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _upsert_variant_row(rows: list[dict[str, str]], row: dict[str, str]) -> list[dict[str, str]]:
    key = row.get("fallback_variant", "")
    out = [existing for existing in rows if existing.get("fallback_variant", "") != key]
    out.append(row)
    return out


def _write_variant_summary(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
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


def _write_selected_manifest(rows: list[dict[str, str]], json_path: Path, csv_path: Path) -> None:
    selected = [row for row in rows if row.get("status") == "ok"]
    payload = {
        "generated_at": _timestamp(),
        "rows": [
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
            for row in selected
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    fields = ["group", "dataset", "method", "co_bps", "run_dir", "config", "run_name", "error"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in payload["rows"]:
            writer.writerow(row)


def _export_h5(summary_path: Path, output_dir: Path) -> None:
    argv = sys.argv
    try:
        sys.argv = [
            "export_allen_vcn_predictions_h5.py",
            "--summary",
            str(summary_path),
            "--output-dir",
            str(output_dir),
            "--include-targets",
        ]
        export_h5_main()
    finally:
        sys.argv = argv


def _write_status(
    path: Path,
    *,
    status: str,
    variant: str,
    index: int,
    total: int,
    best_score: float,
    baseline: float,
) -> None:
    payload = {
        "status": status,
        "variant": variant,
        "index": index,
        "total": total,
        "best_score": best_score,
        "baseline": baseline,
        "best_margin": best_score - baseline if not math.isnan(best_score) and not math.isnan(baseline) else math.nan,
        "updated_at": _timestamp(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


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
