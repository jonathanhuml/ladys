"""Merge sharded full-rate NLB exports and run the official evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ladys.config import load_experiment_config
from ladys.experiment import Experiment
from ladys.nlb_eval import (
    _flatten_nlb_result,
    _nlb_group_name,
    _write_grouped_target_h5,
    _write_rate_parts,
    evaluate_nlb_submission,
    score_to_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Merge train/eval shard .npz files produced by "
            "export_nlb_full_rates_shard.py, write an EvalAI-style NLB H5 "
            "submission, and optionally run nlb_tools.evaluation.evaluate."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--pattern",
        default="*.npz",
        help="glob pattern for shard files inside --shard-dir",
    )
    parser.add_argument(
        "--skip-evaluate",
        action="store_true",
        help="write H5 artifacts without running nlb_tools evaluation",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    experiment = Experiment(config)
    experiment.data.setup()
    if experiment.data.train_dataset is None or experiment.data.valid_dataset is None:
        raise RuntimeError("Experiment data was not initialized.")

    shard_dir = Path(args.shard_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = sorted(shard_dir.glob(args.pattern))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files matched {shard_dir / args.pattern}")

    expected_totals = {
        "train": len(experiment.data.train_dataset),
        "eval": len(experiment.data.valid_dataset),
    }
    shards = [_load_shard(path) for path in shard_paths]
    train_parts = _merge_split(shards, split="train", expected_total=expected_totals["train"])
    eval_parts = _merge_split(shards, split="eval", expected_total=expected_totals["eval"])

    group_name = _nlb_group_name(config.dataset)
    target_path = output_dir / "nlb_eval_target.h5"
    submission_path = output_dir / "nlb_submission.h5"
    _write_grouped_target_h5(config.dataset, target_path, group_name)

    with h5py.File(submission_path, "w") as handle:
        group = handle.create_group(group_name)
        _write_rate_parts(group, "train", train_parts)
        _write_rate_parts(group, "eval", eval_parts)

    payload: dict[str, Any] = {
        "config": args.config,
        "shard_dir": str(shard_dir),
        "submission_path": str(submission_path),
        "target_path": str(target_path),
        "metrics": {},
        "raw_result": None,
        "shards": [
            {
                "path": str(shard["path"]),
                "split": shard["split"],
                "start": shard["start"],
                "stop": shard["stop"],
                "total": shard["total"],
            }
            for shard in shards
        ],
    }

    if not args.skip_evaluate:
        raw_result = evaluate_nlb_submission(target_path, submission_path)
        payload["raw_result"] = raw_result
        payload["metrics"] = _flatten_nlb_result(raw_result)
        (output_dir / "nlb_full_metrics.json").write_text(
            score_to_json(raw_result) + "\n"
        )

    metrics_path = output_dir / "sharded_full_metrics.json"
    metrics_path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n")
    print(json.dumps(_json_ready(payload), indent=2, sort_keys=True), flush=True)
    return 0


def _load_shard(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        split = str(data["split"])
        shard = {
            "path": path,
            "split": split,
            "start": int(data["start"]),
            "stop": int(data["stop"]),
            "total": int(data["total"]),
            "parts": {
                key: data[key].astype(np.float32, copy=False)
                for key in data.files
                if key.startswith("rates_")
            },
        }
    if split not in {"train", "eval"}:
        raise ValueError(f"{path}: invalid split {split!r}")
    if shard["stop"] < shard["start"]:
        raise ValueError(f"{path}: stop precedes start")
    if not shard["parts"]:
        raise ValueError(f"{path}: no rate arrays found")
    return shard


def _merge_split(
    shards: list[dict[str, Any]],
    *,
    split: str,
    expected_total: int,
) -> dict[str, np.ndarray]:
    selected = sorted(
        [shard for shard in shards if shard["split"] == split],
        key=lambda shard: shard["start"],
    )
    if not selected:
        raise ValueError(f"No {split} shards were provided.")

    cursor = 0
    part_keys = set(selected[0]["parts"])
    arrays_by_key: dict[str, list[np.ndarray]] = {key: [] for key in part_keys}
    for shard in selected:
        if shard["total"] != expected_total:
            raise ValueError(
                f"{shard['path']}: expected total {expected_total}, found {shard['total']}"
            )
        if shard["start"] != cursor:
            raise ValueError(
                f"{split} shards are not contiguous: expected start {cursor}, "
                f"found {shard['start']} in {shard['path']}"
            )
        if set(shard["parts"]) != part_keys:
            raise ValueError(f"{shard['path']}: rate part keys differ across shards")
        shard_rows = shard["stop"] - shard["start"]
        for key, array in shard["parts"].items():
            if array.shape[0] != shard_rows:
                raise ValueError(
                    f"{shard['path']}:{key} has {array.shape[0]} rows for "
                    f"range length {shard_rows}"
                )
            arrays_by_key[key].append(array)
        cursor = shard["stop"]

    if cursor != expected_total:
        raise ValueError(
            f"{split} shards cover {cursor} trials, expected {expected_total}"
        )

    return {
        key: np.concatenate(values, axis=0).astype(np.float32, copy=False)
        for key, values in arrays_by_key.items()
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
