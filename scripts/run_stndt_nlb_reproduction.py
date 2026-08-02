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

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ladys.config import ExperimentConfig, load_experiment_config
from ladys.experiment import Experiment, _write_json
from ladys.models.stndt import STNDTConfig
from ladys.training import Trainer, TrainerConfig
from ladys.training.strategies import build_strategy
from ladys.utils.yaml import load_yaml
from scripts.run_nlb_classical_table import evaluate_full_nlb, write_full_artifacts


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
        "--checkpoint",
        type=Path,
        help="Optional LaDyS model state_dict to load before training one config.",
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
                checkpoint_state=checkpoint_state,
                stage_name=stage_name,
            )
            checkpoint_state = result.get("_best_state_dict")
            rows.append(result)
            _write_summary(Path(args.output_dir), rows)
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
    checkpoint_state: dict[str, torch.Tensor] | None = None,
    stage_name: str | None = None,
) -> dict[str, Any]:
    experiment = Experiment(config)
    experiment._set_seeds()
    experiment.data.setup()
    model = experiment.build_model()
    if checkpoint_state is not None:
        model.load_state_dict(checkpoint_state)
    elif checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    strategy = build_strategy(config.model.optimization)
    trainer = Trainer(config.trainer)
    device = torch.device(config.trainer.device)
    dataset = str(config.dataset.name)
    best: dict[str, Any] = {
        "epoch": 0,
        "co_bps": float("-inf"),
        "state_dict": None,
        "full": None,
        "metrics": {},
    }
    evals_since_improvement = 0

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


def _write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
    (output_dir / "summary.json").write_text(
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
