#!/usr/bin/env python3
"""Run focused optional-method fallbacks for Allen VCN NLB-style splits."""

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
    DEFAULT_GROUPS,
    _read_existing_summary,
    _run_dir_from_stdout,
    _upsert_row,
    _write_summary,
    build_config,
    read_shape_info,
    row_from_run,
)


METHODS = ("bgpfa", "ilqr_vae", "mint")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-output-dir", required=True)
    parser.add_argument("--summary-name", default="summary_with_all_and_cassm_ensemble_top4.csv")
    parser.add_argument("--data-path", default="data/real/allen_vcn/allen_vcn_low_trial_20ms.h5")
    parser.add_argument("--groups", nargs="+", default=list(DEFAULT_GROUPS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--python", default="/home/jon/torch-gpu/bin/python")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sweep-name", default="optional_fallback_sweep")
    parser.add_argument("--selected-summary-name", default="summary_with_optional_fallback.csv")
    parser.add_argument("--report-name", default="all_methods_margin_report_optional_fallback.txt")
    parser.add_argument("--h5-output-name", default="h5_predictions_optional_fallback")
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--sleep-seconds", type=float, default=300.0)
    parser.add_argument("--export-h5", action="store_true")
    parser.add_argument(
        "--variant-prefixes",
        nargs="*",
        default=[],
        help="Only run variants whose names start with one of these prefixes.",
    )
    args = parser.parse_args()

    base_output = Path(args.base_output_dir)
    sweep_dir = base_output / args.sweep_name
    config_dir = sweep_dir / "configs"
    log_dir = sweep_dir / "_logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    driver_log = sweep_dir / "driver.log"
    status_path = sweep_dir / "status.json"
    _append_log(driver_log, f"optional_fallback_start={_timestamp()}")
    if args.wait_pid is not None:
        _wait_for_pid(args.wait_pid, args.sleep_seconds, driver_log)

    summary_path = base_output / args.summary_name
    rows = _read_existing_summary(summary_path)
    if not rows:
        raise RuntimeError(f"No rows found in {summary_path}")
    shape_info = read_shape_info(Path(args.data_path), args.groups)
    selected_rows = list(rows)
    variant_rows = _read_existing_variant_rows(sweep_dir / "variant_summary.csv")
    targets = _targets_needing_fallback(rows, args.methods)
    variants_by_method = {
        method: _filter_variants(_variants(method), args.variant_prefixes)
        for method in args.methods
    }

    manifest = {
        "base_output_dir": str(base_output),
        "summary": str(summary_path),
        "data_path": args.data_path,
        "groups": args.groups,
        "methods": args.methods,
        "targets": [{"method": method, "group": group} for method, group in targets],
        "variant_prefixes": args.variant_prefixes,
        "variants": {
            method: [{"name": name, "patch": patch} for name, patch in variants_by_method[method]]
            for method in args.methods
        },
    }
    (sweep_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    total = sum(len(variants_by_method[method]) for method, _group in targets)
    completed = 0
    for method, group in targets:
        baseline = _baseline(rows, group)
        best_row = _existing_row(selected_rows, group, method)
        best_score = _float_or_nan(best_row.get("co_bps") if best_row else "")
        for variant_name, patch in variants_by_method[method]:
            completed += 1
            _write_status(
                status_path,
                status="running",
                method=method,
                group=group,
                variant=variant_name,
                index=completed,
                total=total,
                best_score=best_score,
                baseline=baseline,
            )
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
            variant_rows = _upsert_variant_row(variant_rows, row)
            _write_variant_summary(sweep_dir / "variant_summary.csv", variant_rows)

            score = _float_or_nan(row.get("co_bps"))
            margin = score - baseline if not math.isnan(score) and not math.isnan(baseline) else math.nan
            _append_log(
                driver_log,
                f"variant={variant_name} method={method} group={group} status={row.get('status')} "
                f"co_bps={row.get('co_bps')} margin={margin:.10g}",
            )
            if not math.isnan(score) and (math.isnan(best_score) or score > best_score):
                best_score = score
                best_row = _as_selected_row(row, method)
                selected_rows = _upsert_row(selected_rows, best_row)
                _write_outputs(base_output, selected_rows, args=args)
            if not math.isnan(margin) and margin > 0.0:
                _append_log(
                    driver_log,
                    f"target_cleared method={method} group={group} variant={variant_name} "
                    f"co_bps={score} baseline={baseline}",
                )
                break
        if best_row is not None:
            selected_rows = _upsert_row(selected_rows, best_row)
            _write_outputs(base_output, selected_rows, args=args)

    _write_outputs(base_output, selected_rows, args=args)
    if args.export_h5:
        subprocess.run(
            [
                args.python,
                "scripts/export_allen_vcn_predictions_h5.py",
                "--summary",
                str(base_output / args.selected_summary_name),
                "--output-dir",
                str(base_output / args.h5_output_name),
                "--include-targets",
            ],
            cwd=Path.cwd(),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    _write_status(
        status_path,
        status="complete",
        method="",
        group="",
        variant="",
        index=completed,
        total=total,
        best_score=math.nan,
        baseline=math.nan,
    )
    _append_log(driver_log, f"optional_fallback_done={_timestamp()}")
    print(json.dumps({"summary": str(base_output / args.selected_summary_name)}, indent=2))
    return 0


def _targets_needing_fallback(rows: list[dict[str, str]], methods: list[str]) -> list[tuple[str, str]]:
    groups = sorted({row.get("group", "") for row in rows if row.get("group")})
    targets: list[tuple[str, str]] = []
    for group in groups:
        baseline = _baseline(rows, group)
        for method in methods:
            row = _existing_row(rows, group, method)
            score = _float_or_nan(row.get("co_bps") if row else "")
            if math.isnan(score) or score <= baseline:
                targets.append((method, group))
    return targets


def _variants(method: str) -> list[tuple[str, dict[str, Any]]]:
    if method == "bgpfa":
        variants = [
            (
                "infer300_poisson_latent12_lr01_e250",
                {
                    "model": {
                        "likelihood": "poisson",
                        "latent_dim": 12,
                        "learn_scale": False,
                        "nlb_latent_infer_steps": 300,
                        "nlb_latent_infer_n_mc": 20,
                        "nlb_latent_infer_lr": 1.0e-1,
                        "nlb_ridge_alpha": 500.0,
                        "optimization": {"lr": 1.0e-2, "burnin": 100, "steps_per_epoch": 1},
                    },
                    "trainer": {"epochs": 250, "live_eval_interval": 0},
                },
            ),
            (
                "infer300_poisson_latent24_lr01_e250",
                {
                    "model": {
                        "likelihood": "poisson",
                        "latent_dim": 24,
                        "learn_scale": False,
                        "nlb_latent_infer_steps": 300,
                        "nlb_latent_infer_n_mc": 20,
                        "nlb_latent_infer_lr": 1.0e-1,
                        "nlb_ridge_alpha": 500.0,
                        "optimization": {"lr": 1.0e-2, "burnin": 100, "steps_per_epoch": 1},
                    },
                    "trainer": {"epochs": 250, "live_eval_interval": 0},
                },
            ),
            (
                "infer500_poisson_latent24_lr005_e400",
                {
                    "model": {
                        "likelihood": "poisson",
                        "latent_dim": 24,
                        "learn_scale": False,
                        "nlb_latent_infer_steps": 500,
                        "nlb_latent_infer_n_mc": 20,
                        "nlb_latent_infer_lr": 5.0e-2,
                        "nlb_ridge_alpha": 500.0,
                        "optimization": {"lr": 5.0e-3, "burnin": 150, "steps_per_epoch": 1},
                    },
                    "trainer": {"epochs": 400, "live_eval_interval": 0},
                },
            ),
            (
                "infer300_gaussian_latent12_lr02_e150_alpha500",
                {
                    "model": {
                        "likelihood": "gaussian",
                        "latent_dim": 12,
                        "learn_scale": False,
                        "nlb_latent_infer_steps": 300,
                        "nlb_latent_infer_n_mc": 20,
                        "nlb_latent_infer_lr": 1.0e-1,
                        "nlb_ridge_alpha": 500.0,
                        "optimization": {"lr": 2.0e-2, "burnin": 50, "steps_per_epoch": 1},
                    },
                    "trainer": {"epochs": 150, "live_eval_interval": 0},
                },
            ),
            (
                "infer500_gaussian_latent24_lr01_e250_alpha500",
                {
                    "model": {
                        "likelihood": "gaussian",
                        "latent_dim": 24,
                        "learn_scale": False,
                        "nlb_latent_infer_steps": 500,
                        "nlb_latent_infer_n_mc": 20,
                        "nlb_latent_infer_lr": 5.0e-2,
                        "nlb_ridge_alpha": 500.0,
                        "optimization": {"lr": 1.0e-2, "burnin": 100, "steps_per_epoch": 1},
                    },
                    "trainer": {"epochs": 250, "live_eval_interval": 0},
                },
            ),
            (
                "infer500_gaussian_latent24_lr01_e250_alpha100",
                {
                    "model": {
                        "likelihood": "gaussian",
                        "latent_dim": 24,
                        "learn_scale": False,
                        "nlb_latent_infer_steps": 500,
                        "nlb_latent_infer_n_mc": 20,
                        "nlb_latent_infer_lr": 5.0e-2,
                        "nlb_ridge_alpha": 100.0,
                        "optimization": {"lr": 1.0e-2, "burnin": 100, "steps_per_epoch": 1},
                    },
                    "trainer": {"epochs": 250, "live_eval_interval": 0},
                },
            ),
            (
                "infer500_gaussian_latent24_lr01_e250_alpha1000",
                {
                    "model": {
                        "likelihood": "gaussian",
                        "latent_dim": 24,
                        "learn_scale": False,
                        "nlb_latent_infer_steps": 500,
                        "nlb_latent_infer_n_mc": 20,
                        "nlb_latent_infer_lr": 5.0e-2,
                        "nlb_ridge_alpha": 1000.0,
                        "optimization": {"lr": 1.0e-2, "burnin": 100, "steps_per_epoch": 1},
                    },
                    "trainer": {"epochs": 250, "live_eval_interval": 0},
                },
            ),
            (
                "refkl_gaussian_latent24_lr02_e300_mc8_steps2",
                {
                    "model": {
                        "likelihood": "gaussian",
                        "latent_dim": 24,
                        "learn_scale": False,
                        "n_mc_train": 3,
                        "n_mc_eval": 8,
                        "nlb_latent_infer_steps": 700,
                        "nlb_latent_infer_n_mc": 32,
                        "nlb_latent_infer_lr": 5.0e-2,
                        "nlb_poisson_max_iter": 120,
                        "optimization": {
                            "lr": 2.0e-2,
                            "burnin": 150,
                            "n_mc": 8,
                            "batch_mc": 2,
                            "steps_per_epoch": 2,
                            "analytic_kl": False,
                        },
                    },
                    "trainer": {"epochs": 300, "live_eval_interval": 0},
                },
            ),
            (
                "refkl_gaussian_latent32_lr01_e300_mc8_steps2",
                {
                    "model": {
                        "likelihood": "gaussian",
                        "latent_dim": 32,
                        "learn_scale": False,
                        "n_mc_train": 3,
                        "n_mc_eval": 8,
                        "nlb_latent_infer_steps": 700,
                        "nlb_latent_infer_n_mc": 32,
                        "nlb_latent_infer_lr": 5.0e-2,
                        "nlb_poisson_max_iter": 120,
                        "optimization": {
                            "lr": 1.0e-2,
                            "burnin": 150,
                            "n_mc": 8,
                            "batch_mc": 2,
                            "steps_per_epoch": 2,
                            "analytic_kl": False,
                        },
                    },
                    "trainer": {"epochs": 300, "live_eval_interval": 0},
                },
            ),
            (
                "refkl_gaussian_latent48_lr005_e250_mc8_steps2",
                {
                    "model": {
                        "likelihood": "gaussian",
                        "latent_dim": 48,
                        "learn_scale": False,
                        "n_mc_train": 3,
                        "n_mc_eval": 8,
                        "nlb_latent_infer_steps": 700,
                        "nlb_latent_infer_n_mc": 32,
                        "nlb_latent_infer_lr": 5.0e-2,
                        "nlb_poisson_max_iter": 120,
                        "optimization": {
                            "lr": 5.0e-3,
                            "burnin": 150,
                            "n_mc": 8,
                            "batch_mc": 2,
                            "steps_per_epoch": 2,
                            "analytic_kl": False,
                        },
                    },
                    "trainer": {"epochs": 250, "live_eval_interval": 0},
                },
            ),
            (
                "refkl_gaussian_latent32_lr01_e200_mc8_steps3",
                {
                    "model": {
                        "likelihood": "gaussian",
                        "latent_dim": 32,
                        "learn_scale": False,
                        "n_mc_train": 3,
                        "n_mc_eval": 8,
                        "nlb_latent_infer_steps": 900,
                        "nlb_latent_infer_n_mc": 32,
                        "nlb_latent_infer_lr": 2.0e-2,
                        "nlb_poisson_max_iter": 120,
                        "optimization": {
                            "lr": 1.0e-2,
                            "burnin": 200,
                            "n_mc": 8,
                            "batch_mc": 2,
                            "steps_per_epoch": 3,
                            "analytic_kl": False,
                        },
                    },
                    "trainer": {"epochs": 200, "live_eval_interval": 0},
                },
            ),
            (
                "faobs_gaussian_latent24_lr02_e200_mc4_refkl",
                {
                    "model": {
                        "likelihood": "gaussian",
                        "latent_dim": 24,
                        "learn_scale": False,
                        "observation_init": "fa",
                        "n_mc_train": 3,
                        "n_mc_eval": 8,
                        "nlb_latent_infer_steps": 700,
                        "nlb_latent_infer_n_mc": 32,
                        "nlb_latent_infer_lr": 5.0e-2,
                        "nlb_poisson_max_iter": 120,
                        "optimization": {
                            "lr": 2.0e-2,
                            "burnin": 150,
                            "n_mc": 4,
                            "batch_mc": 2,
                            "steps_per_epoch": 1,
                            "analytic_kl": False,
                        },
                    },
                    "trainer": {"epochs": 200, "live_eval_interval": 0},
                },
            ),
            (
                "faobs_gaussian_latent32_lr01_e250_mc4_refkl",
                {
                    "model": {
                        "likelihood": "gaussian",
                        "latent_dim": 32,
                        "learn_scale": False,
                        "observation_init": "fa",
                        "n_mc_train": 3,
                        "n_mc_eval": 8,
                        "nlb_latent_infer_steps": 700,
                        "nlb_latent_infer_n_mc": 32,
                        "nlb_latent_infer_lr": 5.0e-2,
                        "nlb_poisson_max_iter": 120,
                        "optimization": {
                            "lr": 1.0e-2,
                            "burnin": 150,
                            "n_mc": 4,
                            "batch_mc": 2,
                            "steps_per_epoch": 1,
                            "analytic_kl": False,
                        },
                    },
                    "trainer": {"epochs": 250, "live_eval_interval": 0},
                },
            ),
            (
                "faobs_gaussian_latent48_lr005_e200_mc4_refkl",
                {
                    "model": {
                        "likelihood": "gaussian",
                        "latent_dim": 48,
                        "learn_scale": False,
                        "observation_init": "fa",
                        "n_mc_train": 3,
                        "n_mc_eval": 8,
                        "nlb_latent_infer_steps": 700,
                        "nlb_latent_infer_n_mc": 32,
                        "nlb_latent_infer_lr": 5.0e-2,
                        "nlb_poisson_max_iter": 120,
                        "optimization": {
                            "lr": 5.0e-3,
                            "burnin": 150,
                            "n_mc": 4,
                            "batch_mc": 2,
                            "steps_per_epoch": 1,
                            "analytic_kl": False,
                        },
                    },
                    "trainer": {"epochs": 200, "live_eval_interval": 0},
                },
            ),
            (
                "faobs1_gaussian_latent24_lr02_e250_refkl",
                {
                    "model": {
                        "likelihood": "gaussian",
                        "latent_dim": 24,
                        "learn_scale": False,
                        "observation_init": "fa",
                        "n_mc_train": 1,
                        "n_mc_eval": 8,
                        "nlb_latent_infer_steps": 700,
                        "nlb_latent_infer_n_mc": 20,
                        "nlb_latent_infer_lr": 5.0e-2,
                        "nlb_poisson_max_iter": 120,
                        "optimization": {
                            "lr": 2.0e-2,
                            "burnin": 150,
                            "n_mc": 1,
                            "steps_per_epoch": 1,
                            "analytic_kl": False,
                        },
                    },
                    "trainer": {"epochs": 250, "live_eval_interval": 0},
                },
            ),
            (
                "faobs1_gaussian_latent32_lr01_e300_refkl",
                {
                    "model": {
                        "likelihood": "gaussian",
                        "latent_dim": 32,
                        "learn_scale": False,
                        "observation_init": "fa",
                        "n_mc_train": 1,
                        "n_mc_eval": 8,
                        "nlb_latent_infer_steps": 700,
                        "nlb_latent_infer_n_mc": 20,
                        "nlb_latent_infer_lr": 5.0e-2,
                        "nlb_poisson_max_iter": 120,
                        "optimization": {
                            "lr": 1.0e-2,
                            "burnin": 150,
                            "n_mc": 1,
                            "steps_per_epoch": 1,
                            "analytic_kl": False,
                        },
                    },
                    "trainer": {"epochs": 300, "live_eval_interval": 0},
                },
            ),
            (
                "faobs1_gaussian_latent48_lr005_e250_refkl",
                {
                    "model": {
                        "likelihood": "gaussian",
                        "latent_dim": 48,
                        "learn_scale": False,
                        "observation_init": "fa",
                        "n_mc_train": 1,
                        "n_mc_eval": 8,
                        "nlb_latent_infer_steps": 700,
                        "nlb_latent_infer_n_mc": 20,
                        "nlb_latent_infer_lr": 5.0e-2,
                        "nlb_poisson_max_iter": 120,
                        "optimization": {
                            "lr": 5.0e-3,
                            "burnin": 150,
                            "n_mc": 1,
                            "steps_per_epoch": 1,
                            "analytic_kl": False,
                        },
                    },
                    "trainer": {"epochs": 250, "live_eval_interval": 0},
                },
            ),
            (
                "faobs1tune_gaussian_latent16_lr02_e350_refkl_infer1000",
                {
                    "model": {
                        "likelihood": "gaussian",
                        "latent_dim": 16,
                        "learn_scale": False,
                        "observation_init": "fa",
                        "n_mc_train": 1,
                        "n_mc_eval": 8,
                        "nlb_latent_infer_steps": 1000,
                        "nlb_latent_infer_n_mc": 20,
                        "nlb_latent_infer_lr": 2.0e-2,
                        "nlb_poisson_max_iter": 160,
                        "optimization": {
                            "lr": 2.0e-2,
                            "burnin": 150,
                            "n_mc": 1,
                            "steps_per_epoch": 1,
                            "analytic_kl": False,
                        },
                    },
                    "trainer": {"epochs": 350, "live_eval_interval": 0},
                },
            ),
            (
                "faobs1tune_gaussian_latent20_lr02_e350_refkl_infer1000",
                {
                    "model": {
                        "likelihood": "gaussian",
                        "latent_dim": 20,
                        "learn_scale": False,
                        "observation_init": "fa",
                        "n_mc_train": 1,
                        "n_mc_eval": 8,
                        "nlb_latent_infer_steps": 1000,
                        "nlb_latent_infer_n_mc": 20,
                        "nlb_latent_infer_lr": 2.0e-2,
                        "nlb_poisson_max_iter": 160,
                        "optimization": {
                            "lr": 2.0e-2,
                            "burnin": 150,
                            "n_mc": 1,
                            "steps_per_epoch": 1,
                            "analytic_kl": False,
                        },
                    },
                    "trainer": {"epochs": 350, "live_eval_interval": 0},
                },
            ),
            (
                "faobs1tune_gaussian_latent24_lr015_e400_refkl_infer1000",
                {
                    "model": {
                        "likelihood": "gaussian",
                        "latent_dim": 24,
                        "learn_scale": False,
                        "observation_init": "fa",
                        "n_mc_train": 1,
                        "n_mc_eval": 8,
                        "nlb_latent_infer_steps": 1000,
                        "nlb_latent_infer_n_mc": 20,
                        "nlb_latent_infer_lr": 2.0e-2,
                        "nlb_poisson_max_iter": 160,
                        "optimization": {
                            "lr": 1.5e-2,
                            "burnin": 150,
                            "n_mc": 1,
                            "steps_per_epoch": 1,
                            "analytic_kl": False,
                        },
                    },
                    "trainer": {"epochs": 400, "live_eval_interval": 0},
                },
            ),
            (
                "faobs1rates_gaussian_latent24_lr02_e250_refkl",
                {
                    "model": {
                        "likelihood": "gaussian",
                        "latent_dim": 24,
                        "learn_scale": False,
                        "observation_init": "fa",
                        "n_mc_train": 1,
                        "n_mc_eval": 8,
                        "nlb_feature_source": "rates",
                        "nlb_latent_infer_steps": 700,
                        "nlb_latent_infer_n_mc": 20,
                        "nlb_latent_infer_lr": 5.0e-2,
                        "nlb_poisson_max_iter": 160,
                        "optimization": {
                            "lr": 2.0e-2,
                            "burnin": 150,
                            "n_mc": 1,
                            "steps_per_epoch": 1,
                            "analytic_kl": False,
                        },
                    },
                    "trainer": {"epochs": 250, "live_eval_interval": 0},
                },
            ),
            (
                "faobs1smooth_gaussian_latent24_lr02_e250_refkl",
                {
                    "model": {
                        "likelihood": "gaussian",
                        "latent_dim": 24,
                        "learn_scale": False,
                        "observation_init": "fa",
                        "n_mc_train": 1,
                        "n_mc_eval": 8,
                        "nlb_latent_infer_steps": 700,
                        "nlb_latent_infer_n_mc": 20,
                        "nlb_latent_infer_lr": 5.0e-2,
                        "nlb_poisson_max_iter": 160,
                        "optimization": {
                            "lr": 2.0e-2,
                            "burnin": 150,
                            "n_mc": 1,
                            "steps_per_epoch": 1,
                            "analytic_kl": False,
                        },
                    },
                    "preprocessing": {
                        "observations": {
                            "name": "smooth_firing_rate",
                            "sampling_precision": 20.0,
                            "kern_sd_ms": 50.0,
                        }
                    },
                    "trainer": {"epochs": 250, "live_eval_interval": 0},
                },
            ),
            (
                "infer500_poisson_latent32_lr005_e400_alpha1000",
                {
                    "model": {
                        "likelihood": "poisson",
                        "latent_dim": 32,
                        "learn_scale": False,
                        "nlb_latent_infer_steps": 500,
                        "nlb_latent_infer_n_mc": 20,
                        "nlb_latent_infer_lr": 5.0e-2,
                        "nlb_ridge_alpha": 1000.0,
                        "optimization": {"lr": 5.0e-3, "burnin": 150, "steps_per_epoch": 1},
                    },
                    "trainer": {"epochs": 400, "live_eval_interval": 0},
                },
            ),
            (
                "infer300_poisson_rates_latent12_lr01_e250_alpha500",
                {
                    "model": {
                        "likelihood": "poisson",
                        "latent_dim": 12,
                        "learn_scale": False,
                        "nlb_feature_source": "rates",
                        "nlb_decoder": "ridge",
                        "nlb_latent_infer_steps": 300,
                        "nlb_latent_infer_n_mc": 20,
                        "nlb_latent_infer_lr": 1.0e-1,
                        "nlb_ridge_alpha": 500.0,
                        "optimization": {"lr": 1.0e-2, "burnin": 100, "steps_per_epoch": 1},
                    },
                    "trainer": {"epochs": 250, "live_eval_interval": 0},
                },
            ),
            (
                "infer300_poisson_recon_latent12_lr01_e250_alpha500",
                {
                    "model": {
                        "likelihood": "poisson",
                        "latent_dim": 12,
                        "learn_scale": False,
                        "nlb_feature_source": "reconstruction",
                        "nlb_decoder": "ridge",
                        "nlb_latent_infer_steps": 300,
                        "nlb_latent_infer_n_mc": 20,
                        "nlb_latent_infer_lr": 1.0e-1,
                        "nlb_ridge_alpha": 500.0,
                        "optimization": {"lr": 1.0e-2, "burnin": 100, "steps_per_epoch": 1},
                    },
                    "trainer": {"epochs": 250, "live_eval_interval": 0},
                },
            ),
            (
                "infer300_poisson_latents_glm_latent12_lr01_e250",
                {
                    "model": {
                        "likelihood": "poisson",
                        "latent_dim": 12,
                        "learn_scale": False,
                        "nlb_feature_source": "latents",
                        "nlb_decoder": "poisson",
                        "nlb_poisson_max_iter": 50,
                        "nlb_latent_infer_steps": 300,
                        "nlb_latent_infer_n_mc": 20,
                        "nlb_latent_infer_lr": 1.0e-1,
                        "optimization": {"lr": 1.0e-2, "burnin": 100, "steps_per_epoch": 1},
                    },
                    "trainer": {"epochs": 250, "live_eval_interval": 0},
                },
            ),
        ]
        return _bgpfa_poisson_decoder_variants(variants)
    if method == "ilqr_vae":
        return [
            (
                "e50_maxiter2_lr002_seed0",
                {
                    "model": {
                        "init_seed": 0,
                        "max_iter": 2,
                        "optimization": {"lr": 2.0e-3, "sqrt_decay_scale": 25.0},
                    },
                    "trainer": {"epochs": 50},
                },
            ),
            (
                "e50_maxiter5_lr002_seed0",
                {
                    "model": {
                        "init_seed": 0,
                        "max_iter": 5,
                        "optimization": {"lr": 2.0e-3, "sqrt_decay_scale": 25.0},
                    },
                    "trainer": {"epochs": 50},
                },
            ),
            (
                "e75_maxiter5_lr001_seed1",
                {
                    "model": {
                        "init_seed": 1,
                        "max_iter": 5,
                        "optimization": {"lr": 1.0e-3, "sqrt_decay_scale": 50.0},
                    },
                    "trainer": {"epochs": 75},
                },
            ),
        ]
    if method == "mint":
        return [
            ("cand16_win4_sigma2", {"model": {"n_candidates": 16, "window_length": 4, "delta": 1, "sigma": 2}}),
            ("cand32_win4_sigma2", {"model": {"n_candidates": 32, "window_length": 4, "delta": 1, "sigma": 2}}),
            ("cand32_win6_sigma4", {"model": {"n_candidates": 32, "window_length": 6, "delta": 1, "sigma": 4}}),
            ("cand64_win6_sigma4", {"model": {"n_candidates": 64, "window_length": 6, "delta": 1, "sigma": 4}}),
            (
                "condition_interp2_cand2_win6_sigma2",
                {"model": {"interp": 2, "n_candidates": 2, "window_length": 6, "delta": 1, "sigma": 2}},
            ),
            (
                "trial_interp0_win6_sigma2",
                {
                    "model": {
                        "allen_condition_mode": "trial_index",
                        "interp": 0,
                        "n_candidates": 2,
                        "window_length": 6,
                        "delta": 1,
                        "sigma": 2,
                    }
                },
            ),
            (
                "trial_interp1_win6_sigma2",
                {
                    "model": {
                        "allen_condition_mode": "trial_index",
                        "interp": 1,
                        "n_candidates": 2,
                        "window_length": 6,
                        "delta": 1,
                        "sigma": 2,
                    }
                },
            ),
            (
                "trial_interp2_cand2_win6_sigma2",
                {
                    "model": {
                        "allen_condition_mode": "trial_index",
                        "interp": 2,
                        "n_candidates": 2,
                        "window_length": 6,
                        "delta": 1,
                        "sigma": 2,
                    }
                },
            ),
            (
                "trial_interp2_cand4_win6_sigma2",
                {
                    "model": {
                        "allen_condition_mode": "trial_index",
                        "interp": 2,
                        "n_candidates": 4,
                        "window_length": 6,
                        "delta": 1,
                        "sigma": 2,
                    }
                },
            ),
            (
                "lowrank64_condition_interp2_cand2_win6_sigma2",
                {
                    "model": {
                        "interp": 2,
                        "n_candidates": 2,
                        "window_length": 6,
                        "delta": 1,
                        "sigma": 2,
                        "n_neural_dims": 64,
                    }
                },
            ),
            (
                "lowrank64_trial_interp2_cand2_win6_sigma2",
                {
                    "model": {
                        "allen_condition_mode": "trial_index",
                        "interp": 2,
                        "n_candidates": 2,
                        "window_length": 6,
                        "delta": 1,
                        "sigma": 2,
                        "n_neural_dims": 64,
                    }
                },
            ),
            (
                "lowrank128_trial_interp2_cand2_win6_sigma2",
                {
                    "model": {
                        "allen_condition_mode": "trial_index",
                        "interp": 2,
                        "n_candidates": 2,
                        "window_length": 6,
                        "delta": 1,
                        "sigma": 2,
                        "n_neural_dims": 128,
                    }
                },
            ),
            (
                "lfadslib_condition_interp1_cand2_win6_sigma2",
                {
                    "model": {
                        "allen_library_source": "lfads_checkpoint",
                        "interp": 1,
                        "n_candidates": 2,
                        "window_length": 6,
                        "delta": 1,
                        "sigma": 2,
                    }
                },
            ),
            (
                "lfadslib_condition_interp2_cand2_win6_sigma2",
                {
                    "model": {
                        "allen_library_source": "lfads_checkpoint",
                        "interp": 2,
                        "n_candidates": 2,
                        "window_length": 6,
                        "delta": 1,
                        "sigma": 2,
                    }
                },
            ),
            (
                "lfadslib_trial_interp2_cand2_win6_sigma2",
                {
                    "model": {
                        "allen_condition_mode": "trial_index",
                        "allen_library_source": "lfads_checkpoint",
                        "interp": 2,
                        "n_candidates": 2,
                        "window_length": 6,
                        "delta": 1,
                        "sigma": 2,
                    }
                },
            ),
        ]
    raise ValueError(method)


def _filter_variants(
    variants: list[tuple[str, dict[str, Any]]],
    prefixes: list[str],
) -> list[tuple[str, dict[str, Any]]]:
    if not prefixes:
        return variants
    return [
        (name, patch)
        for name, patch in variants
        if any(name.startswith(prefix) for prefix in prefixes)
    ]


def _bgpfa_poisson_decoder_variants(
    variants: list[tuple[str, dict[str, Any]]],
) -> list[tuple[str, dict[str, Any]]]:
    normalized: list[tuple[str, dict[str, Any]]] = []
    for name, patch in variants:
        patch = deepcopy(patch)
        model_patch = patch.setdefault("model", {})
        model_patch["latent_init"] = "fa"
        model_patch.setdefault("nlb_feature_source", "latents")
        model_patch["nlb_decoder"] = "poisson"
        model_patch["nlb_ridge_alpha"] = 1.0e-2
        model_patch.setdefault("nlb_poisson_max_iter", 80)
        normalized.append((f"{name}_fainit_poisdec", patch))
    return normalized


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
    run_name = f"{method}_{dataset}_20ms_nlb_style_optional_fallback_{variant_name}"
    config = build_config(
        group=group,
        method=method,
        data_path=data_path,
        output_dir=str(base_output / config_dir.parent.name),
        run_name=run_name,
        device=device,
        shape=shape,
    )
    patch = deepcopy(patch)
    model_patch = patch.setdefault("model", {})
    if method == "mint" and model_patch.get("allen_library_source") == "lfads_checkpoint":
        model_patch.setdefault("allen_lfads_run_dir", str(base_output / f"lfads_{dataset}_20ms_nlb_style"))
    _deep_update(config, patch)
    config_path = config_dir / f"{run_name}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    default_run_dir = base_output / config_dir.parent.name / run_name
    run_dir = _find_existing_run_dir(default_run_dir)
    log_path = log_dir / f"{run_name}.log"
    if run_dir is not None:
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
        method=f"{method}_optional_fallback_{variant_name}",
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
    row["method_selected_as"] = method
    return row


def _write_outputs(base_output: Path, rows: list[dict[str, str]], *, args: argparse.Namespace) -> None:
    _write_summary(base_output / args.selected_summary_name, rows)
    _write_margin_report(rows, base_output / args.report_name)
    _write_selected_manifest(
        rows,
        base_output / f"selected_config_manifest_{args.sweep_name}.json",
        base_output / f"selected_config_manifest_{args.sweep_name}.csv",
    )


def _wait_for_pid(pid: int, sleep_seconds: float, log_path: Path) -> None:
    while subprocess.run(["kill", "-0", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        _append_log(log_path, f"waiting_for_pid={pid} at={_timestamp()}")
        time.sleep(sleep_seconds)
    _append_log(log_path, f"wait_pid_finished={pid} at={_timestamp()}")


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


def _as_selected_row(row: dict[str, str], method: str) -> dict[str, str]:
    selected = dict(row)
    selected["method"] = method
    selected["error"] = f"selected_from_optional_fallback_variant={row.get('fallback_variant', '')}"
    return selected


def _find_existing_run_dir(default_run_dir: Path) -> Path | None:
    candidates = [default_run_dir]
    candidates.extend(sorted(default_run_dir.parent.glob(f"{default_run_dir.name}-*")))
    for candidate in candidates:
        if (candidate / "metrics.json").exists():
            return candidate
    return None


def _read_existing_variant_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _upsert_variant_row(rows: list[dict[str, str]], row: dict[str, str]) -> list[dict[str, str]]:
    key = (row.get("group", ""), row.get("method", ""), row.get("fallback_variant", ""))
    out = [
        existing
        for existing in rows
        if (existing.get("group", ""), existing.get("method", ""), existing.get("fallback_variant", "")) != key
    ]
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
        for row in sorted(group_rows, key=lambda r: _sort_score(r)):
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
            method = row.get("method", "")
            lines.append(f"  {method:<14} co_bps={score: .9f} margin={margin: .9f} {status}")
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


def _write_status(
    path: Path,
    *,
    status: str,
    method: str,
    group: str,
    variant: str,
    index: int,
    total: int,
    best_score: float,
    baseline: float,
) -> None:
    payload = {
        "status": status,
        "method": method,
        "group": group,
        "variant": variant,
        "index": index,
        "total": total,
        "best_score": best_score,
        "baseline": baseline,
        "best_margin": best_score - baseline if not math.isnan(best_score) and not math.isnan(baseline) else math.nan,
        "updated_at": _timestamp(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sort_score(row: dict[str, str]) -> tuple[bool, float, str]:
    score = _float_or_nan(row.get("co_bps", ""))
    return math.isnan(score), -score if not math.isnan(score) else 0.0, row.get("method", "")


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
