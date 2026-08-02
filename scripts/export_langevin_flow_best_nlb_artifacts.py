#!/usr/bin/env python3
"""Export full NLB artifacts for saved LangevinFlow best checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.chdir(ROOT)

from ladys.config import load_experiment_config
from ladys.experiment import Experiment
from scripts.run_nlb_classical_table import evaluate_full_nlb, write_full_artifacts


DATASETS = ("area2_bump", "dmfc_rsg", "mc_maze", "mc_rtt")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-h5", default="data/real/nlb/eval_data_test.h5")
    parser.add_argument(
        "--stable-base",
        default="runs/langevin_flow_stable_best_20260802",
        help="Directory containing langevin_flow_<dataset>_nlb_5ms_stable_best runs.",
    )
    parser.add_argument("--config", action="append", type=Path)
    parser.add_argument("--checkpoint", action="append", type=Path)
    parser.add_argument("--run-dir", action="append", type=Path)
    args = parser.parse_args()

    jobs = _explicit_jobs(args) if args.config or args.checkpoint or args.run_dir else _stable_jobs(args)
    rows = []
    for job in jobs:
        rows.append(export_job(job=job, device=torch.device(args.device), target_h5=Path(args.target_h5)))
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


def _explicit_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    configs = list(args.config or [])
    checkpoints = list(args.checkpoint or [])
    run_dirs = list(args.run_dir or [])
    if not (len(configs) == len(checkpoints) == len(run_dirs)):
        raise ValueError("--config, --checkpoint, and --run-dir must be passed the same number of times.")
    jobs = []
    for config, checkpoint, run_dir in zip(configs, checkpoints, run_dirs, strict=True):
        dataset = _dataset_from_config(config)
        jobs.append(
            {
                "dataset": dataset,
                "config": config,
                "checkpoint": checkpoint,
                "run_dir": run_dir,
            }
        )
    return jobs


def _stable_jobs(args: argparse.Namespace) -> list[dict[str, Any]]:
    stable_base = Path(args.stable_base)
    jobs = []
    for dataset in args.datasets:
        run_dir = stable_base / f"langevin_flow_{dataset}_nlb_5ms_stable_best"
        jobs.append(
            {
                "dataset": dataset,
                "config": Path(
                    f"configs/experiment/real/{dataset}/langevin_flow/"
                    f"langevin_flow_{dataset}_nlb_5ms.yaml"
                ),
                "checkpoint": run_dir / "best_model.pt",
                "run_dir": run_dir,
            }
        )
    return jobs


def _dataset_from_config(path: Path) -> str:
    config = load_experiment_config(str(path))
    return str(config.dataset.name)


def export_job(*, job: dict[str, Any], device: torch.device, target_h5: Path) -> dict[str, Any]:
    dataset = str(job["dataset"])
    config_path = Path(job["config"])
    checkpoint = Path(job["checkpoint"])
    run_dir = Path(job["run_dir"])

    config = load_experiment_config(str(config_path))
    experiment = Experiment(config)
    experiment._set_seeds()
    experiment.data.setup()
    model = experiment.build_model()
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)

    full = evaluate_full_nlb(
        model=model,
        data=experiment.data,
        device=device,
        dataset=dataset,
        target_h5=target_h5,
    )
    write_full_artifacts(run_dir=run_dir, dataset=dataset, full=full)
    payload = {
        "dataset": dataset,
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "run_dir": str(run_dir),
        "metrics": full.full_metrics,
    }
    (run_dir / "checkpoint_full_nlb_artifacts.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, sort_keys=True), flush=True)
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
