#!/usr/bin/env python3
"""Run STNDT NLB reproductions with official full-metric checkpoint selection.

The default dataset set is the three non-MC-Maze NLB reproductions. MC Maze is
available as a single staged YAML with mask-only pretraining followed by
contrastive fine-tuning.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import fields, replace
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ladys.config import ExperimentConfig, load_experiment_config
from ladys.experiment import Experiment, _write_json
from ladys.metrics import EvaluationResult, compute_available_metrics
from ladys.mint_nlb import _score_full_nlb_metrics
from ladys.models.stndt import STNDTConfig
from ladys.training import Trainer, TrainerConfig
from ladys.training.strategies import build_strategy
from ladys.utils.yaml import load_yaml
from scripts.run_nlb_classical_table import FullNLBResult, evaluate_full_nlb, write_full_artifacts


DEFAULT_CONFIGS = {
    "area2_bump": Path(
        "configs/experiment/real/area2_bump/stndt/stndt_area2_bump_nlb_5ms.yaml"
    ),
    "dmfc_rsg": Path("configs/experiment/real/dmfc_rsg/stndt/stndt_dmfc_rsg_nlb_5ms.yaml"),
    "mc_maze": Path("configs/experiment/real/mc_maze/stndt/stndt_mc_maze_nlb_5ms.yaml"),
    "mc_rtt": Path("configs/experiment/real/mc_rtt/stndt/stndt_mc_rtt_nlb_5ms.yaml"),
}


class StopTraining(Exception):
    """Internal signal used by the epoch callback for early stopping."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DEFAULT_CONFIGS),
        default=["area2_bump", "dmfc_rsg", "mc_rtt"],
    )
    parser.add_argument(
        "--config",
        action="append",
        type=Path,
        help="Explicit experiment YAML. May be passed multiple times.",
    )
    parser.add_argument(
        "--config-list",
        type=Path,
        help="Text file containing one experiment YAML path per line.",
    )
    parser.add_argument("--epochs", type=int, help="Override trainer.epochs for every config.")
    parser.add_argument("--batch-size", type=int, help="Override training batch size.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="runs/stndt_nlb_reproduction")
    parser.add_argument("--target-h5", default="data/real/nlb/eval_data_test.h5")
    parser.add_argument(
        "--eval-every",
        type=int,
        default=20,
        help="Evaluate full NLB metrics every N epochs and at the final epoch.",
    )
    parser.add_argument(
        "--patience-evals",
        type=int,
        default=0,
        help="Stop after this many full-metric evaluations without co-bps improvement.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print train-loss progress every N epochs without running full NLB scoring.",
    )
    parser.add_argument(
        "--ensemble-max-size",
        type=int,
        default=2,
        help="Evaluate top-N artifact ensembles per dataset after all candidate runs. Use 0 to disable.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional LaDyS model state_dict to load before training one config.",
    )
    parser.add_argument(
        "--resume-snapshot",
        type=Path,
        help=(
            "Resume one config from a snapshot written by --snapshot-every. "
            "The snapshot stores model weights, best NLB metrics so far, and the global epoch."
        ),
    )
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help="Write a resumable model snapshot every N epochs. Use 0 to disable.",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        help="Directory for --snapshot-every outputs. Defaults to <output-dir>/_snapshots.",
    )
    parser.add_argument("--run-name-suffix", default="")
    args = parser.parse_args()

    paths = []
    if args.config_list is not None:
        paths.extend(_read_config_list(args.config_list))
    if args.config:
        paths.extend(args.config)
    if not paths:
        paths = [DEFAULT_CONFIGS[dataset] for dataset in args.datasets]
    if args.checkpoint is not None and len(paths) != 1:
        raise ValueError("--checkpoint can only be used with exactly one config.")
    if args.resume_snapshot is not None and len(paths) != 1:
        raise ValueError("--resume-snapshot can only be used with exactly one config.")
    if args.checkpoint is not None and args.resume_snapshot is not None:
        raise ValueError("--checkpoint and --resume-snapshot are mutually exclusive.")

    rows: list[dict[str, Any]] = []
    for path in paths:
        stage_configs = _load_configs(
            path=path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            output_dir=args.output_dir,
            run_name_suffix=args.run_name_suffix,
        )
        checkpoint_state: dict[str, torch.Tensor] | None = None
        for stage_index, (stage_name, config) in enumerate(stage_configs):
            result = run_config(
                config=config,
                target_h5=Path(args.target_h5),
                eval_every=args.eval_every,
                patience_evals=args.patience_evals,
                progress_every=args.progress_every,
                checkpoint=args.checkpoint if stage_index == 0 else None,
                resume_snapshot=args.resume_snapshot if stage_index == 0 else None,
                snapshot_every=args.snapshot_every,
                snapshot_dir=args.snapshot_dir,
                checkpoint_state=checkpoint_state,
                stage_name=stage_name,
            )
            checkpoint_state = result.get("_best_state_dict")
            rows.append(result)
            _write_summary(Path(args.output_dir), rows)
    if int(args.ensemble_max_size) > 0:
        ensemble_rows = _run_artifact_ensembles(
            rows=rows,
            output_dir=Path(args.output_dir),
            target_h5=Path(args.target_h5),
            max_size=int(args.ensemble_max_size),
        )
        _write_ensemble_summary(Path(args.output_dir), ensemble_rows)
    return 0


def _read_config_list(path: Path) -> list[Path]:
    return [
        Path(line.strip())
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _load_configs(
    *,
    path: Path,
    epochs: int | None,
    batch_size: int | None,
    device: str,
    output_dir: str,
    run_name_suffix: str,
) -> list[tuple[str | None, ExperimentConfig]]:
    config = load_experiment_config(str(path))
    if not isinstance(config.model, STNDTConfig):
        raise TypeError(f"{path} is not an STNDT experiment config.")

    config = _apply_runtime_overrides(
        config,
        batch_size=batch_size if batch_size is not None else config.batch_size,
        device=device,
        epochs=None,
        output_dir=output_dir,
        run_name_suffix=run_name_suffix,
    )

    raw = load_yaml(path)
    stage_data = raw.get("stndt_reproduction", {}).get("stages", {})
    if not stage_data:
        config = _apply_runtime_overrides(
            config,
            batch_size=config.batch_size,
            device=device,
            epochs=epochs,
            output_dir=output_dir,
            run_name_suffix="",
        )
        return [(None, config)]

    stages: list[tuple[str, ExperimentConfig]] = []
    for name, overrides in stage_data.items():
        if not isinstance(overrides, dict):
            raise TypeError(f"Stage '{name}' in {path} must be a mapping.")
        stage_config = _apply_stage_overrides(config, name=str(name), overrides=overrides)
        stage_config = _apply_runtime_overrides(
            stage_config,
            batch_size=batch_size if batch_size is not None else stage_config.batch_size,
            device=device,
            epochs=epochs,
            output_dir=output_dir,
            run_name_suffix="",
        )
        stages.append((str(name), stage_config))
    return stages


def _apply_runtime_overrides(
    config: ExperimentConfig,
    *,
    batch_size: int,
    device: str,
    epochs: int | None,
    output_dir: str,
    run_name_suffix: str,
) -> ExperimentConfig:
    trainer = replace(config.trainer, device=device)
    if epochs is not None:
        trainer = replace(trainer, epochs=int(epochs))
    run_name = config.run_name
    if run_name_suffix:
        run_name = f"{run_name}_{run_name_suffix}" if run_name else run_name_suffix
    return replace(
        config,
        trainer=trainer,
        batch_size=batch_size,
        output_dir=output_dir,
        run_name=run_name,
        save_predictions=True,
    )


def _apply_stage_overrides(
    config: ExperimentConfig,
    *,
    name: str,
    overrides: dict[str, Any],
) -> ExperimentConfig:
    model = config.model
    model_overrides = overrides.get("model", {})
    if model_overrides:
        if not isinstance(model_overrides, dict):
            raise TypeError(f"Stage '{name}' model override must be a mapping.")
        model_data = _deep_update(model.model_dump(mode="python"), model_overrides)
        model = type(model).model_validate(model_data)

    trainer = config.trainer
    stage_batch_size = config.batch_size
    trainer_overrides = dict(overrides.get("trainer", {}))
    if trainer_overrides:
        allowed = {field.name for field in fields(TrainerConfig)}
        if "batch_size" in trainer_overrides:
            stage_batch_size = int(trainer_overrides.pop("batch_size"))
        unknown = sorted(set(trainer_overrides) - allowed)
        if unknown:
            raise KeyError(f"Unknown trainer override(s) for stage '{name}': {unknown}")
        trainer = replace(trainer, **trainer_overrides)

    run_name = config.run_name
    suffix = str(overrides.get("run_name_suffix", name))
    if suffix:
        run_name = f"{run_name}_{suffix}" if run_name else suffix

    return replace(
        config,
        model=model,
        trainer=trainer,
        batch_size=stage_batch_size,
        run_name=run_name,
    )


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def run_config(
    *,
    config: ExperimentConfig,
    target_h5: Path,
    eval_every: int,
    patience_evals: int,
    progress_every: int,
    checkpoint: Path | None,
    resume_snapshot: Path | None,
    snapshot_every: int = 0,
    snapshot_dir: Path | None = None,
    checkpoint_state: dict[str, torch.Tensor] | None = None,
    stage_name: str | None = None,
) -> dict[str, Any]:
    experiment = Experiment(config)
    experiment._set_seeds()
    experiment.data.setup()
    model = experiment.build_model()
    dataset = str(config.dataset.name)
    start_epoch = 0
    resume_data: dict[str, Any] | None = None
    if checkpoint_state is not None:
        model.load_state_dict(checkpoint_state)
    elif resume_snapshot is not None:
        if resume_snapshot.exists():
            resume_data = torch.load(resume_snapshot, map_location="cpu")
            state_dict = resume_data.get("model_state_dict", resume_data)
            model.load_state_dict(state_dict)
            start_epoch = int(resume_data.get("epoch", 0))
            print(f"{dataset} resume_snapshot={resume_snapshot} start_epoch={start_epoch}", flush=True)
        else:
            print(f"{dataset} resume_snapshot_missing={resume_snapshot} starting_fresh", flush=True)
    elif checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    strategy = build_strategy(config.model.optimization)
    trainer = Trainer(config.trainer)
    device = torch.device(config.trainer.device)
    best: dict[str, Any] = {
        "epoch": 0,
        "co_bps": float("-inf"),
        "state_dict": None,
        "full": None,
        "metrics": {},
    }
    if resume_data is not None:
        best.update(
            {
                "epoch": int(resume_data.get("best_epoch", 0)),
                "co_bps": float(resume_data.get("best_co_bps", float("-inf"))),
                "state_dict": resume_data.get("best_state_dict"),
                "metrics": dict(resume_data.get("best_metrics", {})),
            }
        )
    evals_since_improvement = 0
    snapshot_path = _snapshot_path(config, snapshot_dir or (Path(config.output_dir) / "_snapshots"))

    def callback(report) -> None:
        nonlocal evals_since_improvement
        epoch = int(report.epoch) + 1
        should_eval = eval_every > 0 and (
            epoch % int(eval_every) == 0 or epoch == int(config.trainer.epochs)
        )
        if progress_every > 0 and epoch % int(progress_every) == 0:
            ramp = _stndt_ramp_probability(model)
            print(
                f"{dataset} epoch={epoch} train_loss={report.train.loss:.6g} "
                f"seconds={report.seconds:.3g} mask_span_expand_prob={ramp:.6g}",
                flush=True,
            )
        if snapshot_every > 0 and epoch % int(snapshot_every) == 0:
            _write_training_snapshot(
                path=snapshot_path,
                epoch=epoch,
                model=model,
                strategy=strategy,
                best=best,
                config=config,
                target_h5=target_h5,
                eval_every=eval_every,
                progress_every=progress_every,
                checkpoint=checkpoint,
                stage_name=stage_name,
            )
            print(f"{dataset} epoch={epoch} snapshot={snapshot_path}", flush=True)
        if not should_eval:
            return
        full = evaluate_full_nlb(
            model=model,
            data=experiment.data,
            device=device,
            dataset=dataset,
            target_h5=target_h5,
        )
        metrics = full.full_metrics
        report.metrics.update({f"nlb_{key}": float(value) for key, value in metrics.items()})
        co_bps = float(metrics.get("co-bps", float("-inf")))
        print(
            f"{dataset} epoch={epoch} co-bps={co_bps:.6g} metrics={metrics}",
            flush=True,
        )
        if co_bps > float(best["co_bps"]):
            best.update(
                {
                    "epoch": epoch,
                    "co_bps": co_bps,
                    "state_dict": _clone_state_dict_cpu(model.state_dict()),
                    "full": full,
                    "metrics": dict(metrics),
                }
            )
            evals_since_improvement = 0
        else:
            evals_since_improvement += 1
        if patience_evals > 0 and evals_since_improvement >= patience_evals:
            raise StopTraining

    try:
        history = trainer.fit(
            model=model,
            strategy=strategy,
            train_loader=experiment.data.train_loader(shuffle=True),
            valid_loader=None,
            epoch_callback=callback,
            start_epoch=start_epoch,
            strategy_state=None if resume_data is None else resume_data.get("strategy_state"),
        )
    except StopTraining:
        history = trainer.history

    if best["state_dict"] is None:
        full = evaluate_full_nlb(
            model=model,
            data=experiment.data,
            device=device,
            dataset=dataset,
            target_h5=target_h5,
        )
        best.update(
            {
                "epoch": len(history),
                "co_bps": float(full.full_metrics.get("co-bps", float("nan"))),
                "state_dict": _clone_state_dict_cpu(model.state_dict()),
                "full": full,
                "metrics": dict(full.full_metrics),
            }
        )
    elif best["full"] is None:
        model.load_state_dict(best["state_dict"])
        full = evaluate_full_nlb(
            model=model,
            data=experiment.data,
            device=device,
            dataset=dataset,
            target_h5=target_h5,
        )
        best.update(
            {
                "co_bps": float(full.full_metrics.get("co-bps", float("nan"))),
                "full": full,
                "metrics": dict(full.full_metrics),
            }
        )

    model.load_state_dict(best["state_dict"])
    run_dir = experiment._make_run_dir()
    result = experiment._write_artifacts(run_dir, model, history, best["full"].evaluation)
    write_full_artifacts(run_dir=result.run_dir, dataset=dataset, full=best["full"])
    metadata = {
        "dataset": dataset,
        "best_epoch": int(best["epoch"]),
        "best_metrics": best["metrics"],
        "config_epochs": int(config.trainer.epochs),
        "eval_every": int(eval_every),
        "progress_every": int(progress_every),
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "resume_snapshot": None if resume_snapshot is None else str(resume_snapshot),
        "start_epoch": int(start_epoch),
    }
    if stage_name is not None:
        metadata["stage"] = stage_name
    _write_json(result.run_dir / "stndt_reproduction.json", metadata)
    return {
        "dataset": dataset,
        "run_dir": str(result.run_dir),
        "_best_state_dict": best["state_dict"],
        **metadata,
    }


def _snapshot_path(config: ExperimentConfig, snapshot_dir: Path) -> Path:
    run_name = config.run_name or f"{config.dataset.name}_{config.model.name}"
    safe = "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in run_name)
    return snapshot_dir / f"{safe}_latest.pt"


def _write_training_snapshot(
    *,
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    strategy: Any,
    best: dict[str, Any],
    config: ExperimentConfig,
    target_h5: Path,
    eval_every: int,
    progress_every: int,
    checkpoint: Path | None,
    stage_name: str | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": _clone_state_dict_cpu(model.state_dict()),
            "best_epoch": int(best.get("epoch", 0)),
            "best_co_bps": float(best.get("co_bps", float("-inf"))),
            "best_state_dict": best.get("state_dict"),
            "best_metrics": dict(best.get("metrics", {})),
            "strategy_state": strategy.state_dict() if hasattr(strategy, "state_dict") else {},
            "dataset": str(config.dataset.name),
            "run_name": config.run_name,
            "target_h5": str(target_h5),
            "eval_every": int(eval_every),
            "progress_every": int(progress_every),
            "checkpoint": None if checkpoint is None else str(checkpoint),
            "stage": stage_name,
        },
        tmp,
    )
    tmp.replace(path)


def _write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    (output_dir / "summary.json").write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n"
    )


def _run_artifact_ensembles(
    *,
    rows: list[dict[str, Any]],
    output_dir: Path,
    target_h5: Path,
    max_size: int,
) -> list[dict[str, Any]]:
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        stage = row.get("stage")
        if stage not in (None, "contrast"):
            continue
        run_dir = Path(str(row["run_dir"]))
        if not (run_dir / "full_predictions.npz").exists():
            continue
        by_dataset.setdefault(str(row["dataset"]), []).append(row)

    ensemble_rows: list[dict[str, Any]] = []
    for dataset, candidates in sorted(by_dataset.items()):
        candidates = sorted(
            candidates,
            key=lambda row: float(row.get("best_metrics", {}).get("co-bps", float("-inf"))),
            reverse=True,
        )
        for ensemble_size in range(1, min(max_size, len(candidates)) + 1):
            selected = candidates[:ensemble_size]
            payload = _average_prediction_payloads(
                [Path(str(row["run_dir"])) / "full_predictions.npz" for row in selected]
            )
            metrics = _score_ensemble_payload(dataset=dataset, target_h5=target_h5, payload=payload)
            evaluation = EvaluationResult(
                metrics={"co_bps": float(metrics["co-bps"])},
                predictions={"rates": payload["eval_rates_heldout"].astype(np.float32)},
                targets={"spikes": payload["target_spikes"].astype(np.float32)},
            )
            full = FullNLBResult(
                evaluation=evaluation,
                full_metrics=metrics,
                train_rates_heldin=payload["train_rates_heldin"].astype(np.float32),
                train_rates_heldout=payload["train_rates_heldout"].astype(np.float32),
                eval_rates_heldin=payload["eval_rates_heldin"].astype(np.float32),
                eval_rates_heldout=payload["eval_rates_heldout"].astype(np.float32),
                eval_rates_heldin_forward=payload.get("eval_rates_heldin_forward"),
                eval_rates_heldout_forward=payload.get("eval_rates_heldout_forward"),
            )
            run_dir = _make_unique_dir(output_dir / f"stndt_{dataset}_ensemble_top{ensemble_size}")
            write_full_artifacts(run_dir=run_dir, dataset=dataset, full=full)
            _write_json(
                run_dir / "stndt_ensemble.json",
                {
                    "dataset": dataset,
                    "ensemble_size": ensemble_size,
                    "metrics": metrics,
                    "selected_run_dirs": [str(row["run_dir"]) for row in selected],
                    "selected_co_bps": [
                        float(row.get("best_metrics", {}).get("co-bps", float("nan")))
                        for row in selected
                    ],
                },
            )
            ensemble_rows.append(
                {
                    "dataset": dataset,
                    "ensemble_size": ensemble_size,
                    "run_dir": str(run_dir),
                    "metrics": metrics,
                    "selected_run_dirs": [str(row["run_dir"]) for row in selected],
                }
            )
            print(
                f"{dataset} ensemble_size={ensemble_size} metrics={metrics}",
                flush=True,
            )
    return ensemble_rows


def _average_prediction_payloads(paths: list[Path]) -> dict[str, np.ndarray]:
    required = [
        "train_rates_heldin",
        "train_rates_heldout",
        "eval_rates_heldin",
        "eval_rates_heldout",
        "target_spikes",
    ]
    payloads = []
    for path in paths:
        with np.load(path) as loaded:
            payload = {key: loaded[key] for key in loaded.files}
        missing = sorted(set(required) - set(payload))
        if missing:
            raise KeyError(f"{path} is missing prediction keys required for ensembling: {missing}")
        payloads.append(payload)

    averaged: dict[str, np.ndarray] = {}
    for key in required:
        if key == "target_spikes":
            averaged[key] = payloads[0][key].astype(np.float32, copy=False)
            continue
        arrays = [payload[key].astype(np.float64, copy=False) for payload in payloads]
        _validate_same_shape(key, arrays)
        averaged[key] = np.mean(np.stack(arrays, axis=0), axis=0).astype(np.float32)

    for key in ("eval_rates_heldin_forward", "eval_rates_heldout_forward"):
        if all(key in payload for payload in payloads):
            arrays = [payload[key].astype(np.float64, copy=False) for payload in payloads]
            _validate_same_shape(key, arrays)
            averaged[key] = np.mean(np.stack(arrays, axis=0), axis=0).astype(np.float32)
    return averaged


def _validate_same_shape(key: str, arrays: list[np.ndarray]) -> None:
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"Cannot ensemble key '{key}' with mismatched shapes: {sorted(shapes)}")


def _score_ensemble_payload(
    *,
    dataset: str,
    target_h5: Path,
    payload: dict[str, np.ndarray],
) -> dict[str, float]:
    output_dict = {
        dataset: {
            "train_rates_heldin": payload["train_rates_heldin"],
            "train_rates_heldout": payload["train_rates_heldout"],
            "eval_rates_heldin": payload["eval_rates_heldin"],
            "eval_rates_heldout": payload["eval_rates_heldout"],
        }
    }
    if "eval_rates_heldin_forward" in payload and "eval_rates_heldout_forward" in payload:
        output_dict[dataset]["eval_rates_heldin_forward"] = payload["eval_rates_heldin_forward"]
        output_dict[dataset]["eval_rates_heldout_forward"] = payload["eval_rates_heldout_forward"]
    try:
        from nlb_tools.evaluation import evaluate
    except ImportError:
        evaluate = None

    if evaluate is not None:
        for item in evaluate(str(target_h5), output_dict):
            key = f"{dataset}_split"
            if key in item:
                return {metric: float(value) for metric, value in item[key].items()}

    direct = compute_available_metrics(
        {"rates": torch.as_tensor(payload["eval_rates_heldout"])},
        {"spikes": torch.as_tensor(payload["target_spikes"])},
    )
    return _score_full_nlb_metrics(
        target_path=target_h5,
        dataset=dataset,
        eval_rates_heldout=payload["eval_rates_heldout"],
        eval_rates_heldin=payload["eval_rates_heldin"],
        train_rates_heldout=payload["train_rates_heldout"],
        train_rates_heldin=payload["train_rates_heldin"],
        co_bps=float(direct["co_bps"]),
        eval_rates_heldin_forward=payload.get("eval_rates_heldin_forward"),
        eval_rates_heldout_forward=payload.get("eval_rates_heldout_forward"),
    )


def _make_unique_dir(base: Path) -> Path:
    if not base.exists():
        base.mkdir(parents=True)
        return base
    for index in range(1, 1000):
        candidate = base.with_name(f"{base.name}_{index:03d}")
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
    raise RuntimeError(f"Could not create unique run directory near {base}")


def _write_ensemble_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    serializable = rows
    (output_dir / "ensemble_summary.json").write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n"
    )


def _clone_state_dict_cpu(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def _stndt_ramp_probability(model: torch.nn.Module) -> float:
    ramp = getattr(model, "_current_mask_span_expand_prob", None)
    if callable(ramp):
        return float(ramp(contrast=False))
    return float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
