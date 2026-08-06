#!/usr/bin/env python3
"""Build a reproducible CASSM prediction ensemble for one Allen VCN group."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ladys.metrics import bits_per_spike, poisson_negative_log_likelihood  # noqa: E402
from scripts.run_allen_vcn_nlb_style import (  # noqa: E402
    _read_existing_summary,
    _upsert_row,
    _write_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-output-dir", required=True)
    parser.add_argument("--summary-name", default="summary_with_classical_fallback.csv")
    parser.add_argument("--candidate-dir", default="classical_fallback_sweep")
    parser.add_argument("--group", default="bo_drifting_one_tf")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--method", default="cassm")
    parser.add_argument("--ensemble-name", default=None)
    args = parser.parse_args()

    base_output = Path(args.base_output_dir)
    rows = _read_existing_summary(base_output / args.summary_name)
    if not rows:
        raise RuntimeError(f"No rows found in {base_output / args.summary_name}")

    dataset = f"allen_vcn_{args.group}"
    candidates = _collect_candidates(
        base_output / args.candidate_dir,
        group=args.group,
        top_k=args.top_k,
    )
    if len(candidates) < args.top_k:
        raise RuntimeError(f"Found only {len(candidates)} candidates for top_k={args.top_k}")

    rates, targets = _average_predictions(candidates)
    co_bps = bits_per_spike(torch.from_numpy(rates), torch.from_numpy(targets))
    valid = torch.isfinite(torch.from_numpy(targets))
    poisson_nll = float(
        poisson_negative_log_likelihood(
            torch.from_numpy(rates)[valid],
            torch.from_numpy(targets)[valid],
        )
        .mean()
        .detach()
        .cpu()
    )

    ensemble_name = args.ensemble_name or f"cassm_{dataset}_20ms_nlb_style_ensemble_top{args.top_k}"
    run_dir = _unique_dir(base_output / "cassm_ensembles" / ensemble_name)
    run_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(run_dir / "predictions.npz", pred_rates=rates.astype(np.float32), target_spikes=targets.astype(np.float32))

    config_payload = {
        "kind": "prediction_ensemble",
        "method": args.method,
        "dataset": dataset,
        "group": args.group,
        "top_k": args.top_k,
        "combine": "arithmetic_mean_pred_rates",
        "candidate_dir": str(base_output / args.candidate_dir),
        "sources": [_source_record(candidate) for candidate in candidates],
    }
    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2, sort_keys=True) + "\n")
    (run_dir / "metrics.json").write_text(
        json.dumps({"co_bps": co_bps, "poisson_nll": poisson_nll}, indent=2, sort_keys=True) + "\n"
    )
    (run_dir / "report.md").write_text(_report(config_payload, co_bps, poisson_nll), encoding="utf-8")

    selected_rows = _upsert_row(
        rows,
        {
            "group": args.group,
            "dataset": dataset,
            "method": args.method,
            "run_name": run_dir.name,
            "status": "ok",
            "seconds": "0.000",
            "co_bps": f"{co_bps:.10g}",
            "poisson_nll": f"{poisson_nll:.10g}",
            "baseline_target": _baseline_target(rows, args.group),
            "baseline_margin": _margin(rows, args.group, co_bps),
            "beats_baseline": str(_beats(rows, args.group, co_bps)).lower(),
            "run_dir": str(run_dir),
            "config": str(run_dir / "config.json"),
            "log": "",
            "error": f"selected_from_cassm_ensemble_top{args.top_k}",
        },
    )
    summary_path = base_output / f"summary_with_cassm_ensemble_top{args.top_k}.csv"
    _write_summary(summary_path, selected_rows)
    _write_margin_report(selected_rows, base_output / f"margin_report_cassm_ensemble_top{args.top_k}.txt")
    _write_selected_manifest(
        selected_rows,
        base_output / f"selected_config_manifest_cassm_ensemble_top{args.top_k}.json",
        base_output / f"selected_config_manifest_cassm_ensemble_top{args.top_k}.csv",
    )

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "summary": str(summary_path),
                "co_bps": co_bps,
                "poisson_nll": poisson_nll,
                "sources": [_source_record(candidate) for candidate in candidates],
            },
            indent=2,
        )
    )
    return 0


def _collect_candidates(base: Path, *, group: str, top_k: int) -> list[tuple[float, str, Path]]:
    rows: list[tuple[float, str, Path]] = []
    prefix = f"cassm_allen_vcn_{group}_20ms_nlb_style_fallback_"
    for metrics_path in base.glob(f"{prefix}*/metrics.json"):
        run_dir = metrics_path.parent
        pred_path = run_dir / "predictions.npz"
        if not pred_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        co_bps = metrics.get("co_bps")
        if co_bps is None:
            continue
        variant = run_dir.name.split("_fallback_", 1)[1]
        rows.append((float(co_bps), variant, run_dir))
    rows.sort(key=lambda item: item[0], reverse=True)
    return rows[:top_k]


def _average_predictions(candidates: list[tuple[float, str, Path]]) -> tuple[np.ndarray, np.ndarray]:
    rates: list[np.ndarray] = []
    targets: np.ndarray | None = None
    for _score, _variant, run_dir in candidates:
        with np.load(run_dir / "predictions.npz") as data:
            rates.append(np.asarray(data["pred_rates"], dtype=np.float64))
            candidate_targets = np.asarray(data["target_spikes"], dtype=np.float64)
            if targets is None:
                targets = candidate_targets
            elif targets.shape != candidate_targets.shape or not np.array_equal(targets, candidate_targets):
                raise ValueError(f"Targets differ for ensemble candidate {run_dir}")
    if targets is None:
        raise RuntimeError("No prediction targets found.")
    return np.mean(rates, axis=0), targets


def _source_record(candidate: tuple[float, str, Path]) -> dict[str, Any]:
    score, variant, run_dir = candidate
    return {
        "variant": variant,
        "co_bps": score,
        "run_dir": str(run_dir),
        "config": str(run_dir / "config.json"),
        "predictions": str(run_dir / "predictions.npz"),
    }


def _baseline_target(rows: list[dict[str, str]], group: str) -> str:
    values = []
    for row in rows:
        if row.get("group") == group and row.get("method") in {"psth", "smoothing"} and row.get("co_bps"):
            values.append(float(row["co_bps"]))
    return "" if not values else f"{max(values):.10g}"


def _margin(rows: list[dict[str, str]], group: str, score: float) -> str:
    target = _baseline_target(rows, group)
    return "" if target == "" else f"{score - float(target):.10g}"


def _beats(rows: list[dict[str, str]], group: str, score: float) -> bool:
    target = _baseline_target(rows, group)
    return target != "" and score > float(target)


def _write_margin_report(rows: list[dict[str, str]], path: Path) -> None:
    groups = sorted({row.get("group", "") for row in rows if row.get("group")})
    lines: list[str] = []
    for group in groups:
        baseline = float(_baseline_target(rows, group))
        lines.append(f"[{group}] smoothing={baseline:.9f}")
        group_rows = [row for row in rows if row.get("group") == group and row.get("status") == "ok"]
        for row in sorted(group_rows, key=lambda r: _sort_score(r)):
            score = _float_or_nan(row.get("co_bps", ""))
            margin = score - baseline if not math.isnan(score) else math.nan
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
        "generated_at": datetime.now().isoformat(timespec="seconds"),
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


def _sort_score(row: dict[str, str]) -> tuple[bool, float, str]:
    score = _float_or_nan(row.get("co_bps", ""))
    return math.isnan(score), -score if not math.isnan(score) else 0.0, row.get("method", "")


def _float_or_nan(value: str | float | None) -> float:
    try:
        return float(value) if value not in {None, ""} else math.nan
    except (TypeError, ValueError):
        return math.nan


def _report(config: dict[str, Any], co_bps: float, poisson_nll: float) -> str:
    lines = [
        "# CASSM Ensemble",
        "",
        f"- dataset: {config['dataset']}",
        f"- combine: {config['combine']}",
        f"- top_k: {config['top_k']}",
        f"- co_bps: {co_bps:.10f}",
        f"- poisson_nll: {poisson_nll:.10f}",
        "",
        "## Sources",
    ]
    for source in config["sources"]:
        lines.append(f"- {source['variant']}: {source['co_bps']:.10f} ({source['run_dir']})")
    return "\n".join(lines) + "\n"


def _unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    suffix = 1
    while True:
        candidate = path.with_name(f"{path.name}-{suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


if __name__ == "__main__":
    raise SystemExit(main())
