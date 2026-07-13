"""Run LaDyS NLB baselines and export table-ready metrics.

This script is intentionally scoped to the Appendix NLB reproduction table:
GPFA, Kalman, NDT, PSTH, and Smoothing on the four 5 ms public NLB test splits.
It writes normal LaDyS run artifacts plus an EvalAI-style H5 containing
held-in and held-out rates so the official NLB secondary metrics can be
computed locally.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable

import h5py
import numpy as np
import torch
from torch import Tensor

from ladys.config import ExperimentConfig
from ladys.data import DataModule, build_dataset_config
from ladys.experiment import Experiment, _write_json
from ladys.metrics import EvaluationResult, NLBCoSmoothingAdapter, compute_available_metrics
from ladys.mint_nlb import _score_full_nlb_metrics
from ladys.models.base import BaseDynamicsModel, OptimizationConfig
from ladys.models.baselines import PSTH, PSTHConfig, SmoothingConfig
from ladys.models.gpfa import GPFAConfig
from ladys.models.kalman import KalmanConfig
from ladys.models.ndt import NDTConfig
from ladys.preprocessing import PreprocessingConfig
from ladys.training import Trainer, TrainerConfig
from ladys.training.strategies import build_strategy
from ladys.types import move_batch_to_device, observations_from_batch


DATASETS = ("area2_bump", "dmfc_rsg", "mc_maze", "mc_rtt")
METHODS = ("gpfa", "kalman", "ndt", "psth", "smoothing")

GPFA_LATENT_DIMS = {
    "area2_bump": 22,
    "dmfc_rsg": 32,
    "mc_maze": 52,
    "mc_rtt": 36,
}

NDT_BASE_DATASET_OVERRIDES = {
    "area2_bump": {"embed_dim": 2, "num_layers": 4},
    "dmfc_rsg": {"embed_dim": 0, "num_layers": 6},
    "mc_maze": {"embed_dim": 0, "num_layers": 4},
    "mc_rtt": {"embed_dim": 0, "num_layers": 4},
}

NDT_SWEEP_DATASET_OVERRIDES = {
    "area2_bump": {
        "embed_dim": 2,
        "num_layers": 4,
        "dropout": 0.4,
        "dropout_rates": 0.4,
        "dropout_embedding": 0.4,
        "context_forward": 64,
        "context_backward": 64,
        "mask_random_ratio": 0.95,
        "mask_token_ratio": 0.75,
        "mask_max_span": 5,
    },
    "dmfc_rsg": {
        "embed_dim": 0,
        "num_layers": 6,
        "dropout": 0.5,
        "dropout_rates": 0.5,
        "dropout_embedding": 0.5,
        "context_forward": 120,
        "context_backward": 120,
        "mask_random_ratio": 0.95,
        "mask_token_ratio": 0.75,
        "mask_max_span": 7,
    },
    "mc_maze": {
        "embed_dim": 0,
        "num_layers": 4,
        "dropout": 0.5,
        "dropout_rates": 0.5,
        "dropout_embedding": 0.5,
        "context_forward": 64,
        "context_backward": 64,
        "mask_random_ratio": 0.9,
        "mask_token_ratio": 0.75,
        "mask_max_span": 7,
    },
    "mc_rtt": {
        "embed_dim": 0,
        "num_layers": 4,
        "dropout": 0.5,
        "dropout_rates": 0.5,
        "dropout_embedding": 0.5,
        "context_forward": 64,
        "context_backward": 64,
        "mask_random_ratio": 0.9,
        "mask_token_ratio": 0.75,
        "mask_max_span": 7,
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--target-h5", default="data/real/nlb/eval_data_test.h5")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--ndt-profile",
        choices=("base", "sweep"),
        default="sweep",
        help="NDT hyperparameter profile: official launch defaults or deterministic sweep-derived settings.",
    )
    parser.add_argument(
        "--train-classical",
        action="store_true",
        help=(
            "Use GPFA EM and Kalman gradient updates for --epochs. Without this, "
            "the repository's inference-only NLB baseline semantics are used."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip dataset/method pairs whose run directory already has full metrics.",
    )
    parser.add_argument(
        "--epoch-eval-every",
        type=int,
        default=0,
        help=(
            "If positive, compute full NLB metrics during training every N epochs "
            "and save them in each run directory's epoch_nlb_metrics.csv."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir or f"runs/nlb_classical_{args.epochs}epoch")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        for method in args.methods:
            run_name = f"{method}_{dataset}_nlb_5ms_{args.epochs}epoch"
            run_dir = output_dir / run_name
            metrics_path = run_dir / "full_nlb_metrics_official.json"
            if args.resume and metrics_path.exists():
                rows.append(_row_from_existing(dataset, method, run_dir))
                continue

            start = time.perf_counter()
            config = build_config(
                dataset=dataset,
                method=method,
                epochs=args.epochs,
                output_dir=output_dir,
                run_name=run_name,
                device=args.device,
                train_classical=args.train_classical,
                ndt_profile=args.ndt_profile,
            )
            print(
                f"Running {method} on {dataset}: epochs={config.trainer.epochs}, "
                f"optimization={config.model.optimization.name}",
                flush=True,
            )
            result = run_and_score(
                config=config,
                target_h5=Path(args.target_h5),
                epoch_eval_every=args.epoch_eval_every,
            )
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "run_dir": str(result["run_dir"]),
                    "seconds": elapsed,
                    **result["metrics"],
                }
            )
            write_summary(output_dir, rows)

    write_summary(output_dir, rows)
    return 0


def build_config(
    *,
    dataset: str,
    method: str,
    epochs: int,
    output_dir: Path,
    run_name: str,
    device: str,
    train_classical: bool,
    ndt_profile: str = "sweep",
) -> ExperimentConfig:
    dataset_config = build_dataset_config(
        dataset,
        {
            "name": dataset,
            "data_path": f"data/real/nlb/{dataset}_test_5ms.h5",
            "split": "test",
            "bin_size_ms": 5,
            "max_trials": None,
        },
    )

    if method == "gpfa":
        optimization = OptimizationConfig(name="em") if train_classical else OptimizationConfig(name="inference_only")
        model = GPFAConfig(
            latent_dim=GPFA_LATENT_DIMS[dataset],
            bin_width=5.0,
            start_tau=100.0,
            start_eps=1.0e-3,
            min_var_frac=0.01,
            init_method="fa",
            init_seed=0,
            learn_kernel_params=False,
            fa_max_iters=500,
            fa_tol=1.0e-8,
            kernel_param_max_iters=8,
            kernel_param_lr=1.0,
            jitter=1.0e-5,
            optimization=optimization,
        )
        preprocessing = PreprocessingConfig(observations=None)
        batch_size = 4096
    elif method == "kalman":
        optimization = (
            OptimizationConfig(
                name="gradient",
                optimizer="Adam",
                lr=1.0e-2,
                weight_decay=0.0,
                gradient_clip=300.0,
            )
            if train_classical
            else OptimizationConfig(name="inference_only")
        )
        model = KalmanConfig(
            dt=0.005,
            dataset_name=dataset,
            save_model=False,
            nlb_ridge_alpha=500.0,
            optimization=optimization,
        )
        preprocessing = PreprocessingConfig(
            observations={
                "name": "smooth_firing_rate",
                "sampling_precision": 5.0,
                "kern_sd_ms": 50.0,
            }
        )
        batch_size = 64
    elif method == "psth":
        model = PSTHConfig(
            kern_sd_ms=70.0,
            bin_size_ms=5.0,
            prediction_floor=1.0e-9,
            optimization=OptimizationConfig(name="inference_only"),
        )
        preprocessing = PreprocessingConfig(observations=None)
        batch_size = 4096
    elif method == "ndt":
        overrides = (
            NDT_BASE_DATASET_OVERRIDES[dataset]
            if ndt_profile == "base"
            else NDT_SWEEP_DATASET_OVERRIDES[dataset]
        )
        warmup_steps = max(1, int(round(0.1 * epochs)))
        ramp_start = max(0, int(round(0.16 * epochs)))
        ramp_end = max(ramp_start + 1, int(round(0.24 * epochs)))
        model = NDTConfig(
            output_mode="auto",
            context_forward=overrides.get("context_forward", 4),
            context_backward=overrides.get("context_backward", 8),
            full_context=False,
            hidden_size=128,
            dropout=overrides.get("dropout", 0.1),
            dropout_rates=overrides.get("dropout_rates", 0.2),
            dropout_embedding=overrides.get("dropout_embedding", 0.2),
            num_heads=2,
            num_layers=overrides["num_layers"],
            linear_embedder=False,
            embed_dim=overrides["embed_dim"],
            learnable_position=True,
            lograte=True,
            fixup_init=True,
            pre_norm=True,
            position_offset=True,
            mask_ratio=0.25,
            mask_mode="timestep",
            mask_token_ratio=overrides.get("mask_token_ratio", 1.0),
            mask_random_ratio=overrides.get("mask_random_ratio", 0.5),
            mask_max_span=overrides.get("mask_max_span", 1),
            mask_span_ramp_start=ramp_start if ndt_profile == "sweep" else 0,
            mask_span_ramp_end=ramp_end if ndt_profile == "sweep" else 0,
            nlb_decoder="direct",
            optimization=OptimizationConfig(
                name="gradient",
                optimizer="AdamW" if ndt_profile == "sweep" else "Adam",
                lr=1.0e-3,
                weight_decay=5.0e-5,
                gradient_clip=200.0,
                lr_scheduler="warmup_cosine" if ndt_profile == "sweep" else None,
                warmup_steps=warmup_steps,
                total_steps=int(epochs),
                scheduler_step="epoch",
            ),
        )
        preprocessing = PreprocessingConfig(observations=None)
        batch_size = 64
    elif method == "smoothing":
        model = SmoothingConfig(
            kern_sd_ms=50.0,
            bin_size_ms=5.0,
            log_offset=1.0e-4,
            nlb_decoder_alpha=0.01,
            nlb_poisson_max_iter=500,
            prediction_floor=1.0e-9,
            optimization=OptimizationConfig(name="inference_only"),
        )
        preprocessing = PreprocessingConfig(observations=None)
        batch_size = 4096
    else:
        raise ValueError(f"Unknown method: {method}")

    return ExperimentConfig(
        dataset=dataset_config,
        model=model,
        trainer=TrainerConfig(epochs=int(epochs), device=device),
        preprocessing=preprocessing,
        batch_size=batch_size,
        output_dir=str(output_dir),
        run_name=run_name,
        save_predictions=True,
    )


def run_and_score(
    *,
    config: ExperimentConfig,
    target_h5: Path,
    epoch_eval_every: int = 0,
) -> dict[str, Any]:
    experiment = Experiment(config)
    experiment._set_seeds()
    experiment.data.setup()
    model = experiment.build_model()
    strategy = build_strategy(config.model.optimization)
    trainer = Trainer(config.trainer)
    train_loader = experiment.data.train_loader(shuffle=True)
    epoch_metrics = build_epoch_nlb_metric_fns(
        data=experiment.data,
        device=torch.device(config.trainer.device),
        dataset=str(config.dataset.name),
        target_h5=target_h5,
        every=epoch_eval_every,
        total_epochs=int(config.trainer.epochs),
    )
    history = trainer.fit(
        model=model,
        strategy=strategy,
        train_loader=train_loader,
        valid_loader=None,
        epoch_metrics=epoch_metrics,
    )

    full = evaluate_full_nlb(
        model=model,
        data=experiment.data,
        device=torch.device(config.trainer.device),
        dataset=str(config.dataset.name),
        target_h5=target_h5,
    )
    run_dir = experiment._make_run_dir()
    result = experiment._write_artifacts(run_dir, model, history, full.evaluation)
    write_epoch_metric_artifact(result.run_dir, history)
    write_full_artifacts(
        run_dir=result.run_dir,
        dataset=str(config.dataset.name),
        full=full,
    )
    return {"run_dir": result.run_dir, "metrics": full.full_metrics}


class PeriodicNLBMetricTracker:
    def __init__(
        self,
        *,
        data: DataModule,
        device: torch.device,
        dataset: str,
        target_h5: Path,
        every: int,
        total_epochs: int,
    ) -> None:
        self.data = data
        self.device = device
        self.dataset = dataset
        self.target_h5 = target_h5
        self.every = int(every)
        self.total_epochs = int(total_epochs)
        self.epoch = 0
        self.last_eval_epoch = 0
        self.last_metrics: dict[str, float] = {}

    def metric(self, model: BaseDynamicsModel, key: str, *, advance: bool = False) -> float:
        if advance:
            self.epoch += 1
        if self.every <= 0:
            return float("nan")
        if self.epoch % self.every != 0 and self.epoch != self.total_epochs:
            return float("nan")
        if self.last_eval_epoch != self.epoch:
            full = evaluate_full_nlb(
                model=model,
                data=self.data,
                device=self.device,
                dataset=self.dataset,
                target_h5=self.target_h5,
            )
            self.last_metrics = full.full_metrics
            self.last_eval_epoch = self.epoch
        return float(self.last_metrics.get(key, float("nan")))


def build_epoch_nlb_metric_fns(
    *,
    data: DataModule,
    device: torch.device,
    dataset: str,
    target_h5: Path,
    every: int,
    total_epochs: int,
) -> dict[str, Any] | None:
    if int(every) <= 0:
        return None
    tracker = PeriodicNLBMetricTracker(
        data=data,
        device=device,
        dataset=dataset,
        target_h5=target_h5,
        every=int(every),
        total_epochs=int(total_epochs),
    )
    return {
        "nlb_co_bps": lambda model: tracker.metric(model, "co-bps", advance=True),
        "nlb_vel_R2": lambda model: tracker.metric(model, "vel R2"),
        "nlb_tp_corr": lambda model: tracker.metric(model, "tp corr"),
        "nlb_psth_R2": lambda model: tracker.metric(model, "psth R2"),
    }


class FullNLBResult:
    def __init__(
        self,
        *,
        evaluation: EvaluationResult,
        full_metrics: dict[str, float],
        train_rates_heldin: np.ndarray,
        train_rates_heldout: np.ndarray,
        eval_rates_heldin: np.ndarray,
        eval_rates_heldout: np.ndarray,
        eval_rates_heldin_forward: np.ndarray | None = None,
        eval_rates_heldout_forward: np.ndarray | None = None,
    ) -> None:
        self.evaluation = evaluation
        self.full_metrics = full_metrics
        self.train_rates_heldin = train_rates_heldin
        self.train_rates_heldout = train_rates_heldout
        self.eval_rates_heldin = eval_rates_heldin
        self.eval_rates_heldout = eval_rates_heldout
        self.eval_rates_heldin_forward = eval_rates_heldin_forward
        self.eval_rates_heldout_forward = eval_rates_heldout_forward


def evaluate_full_nlb(
    *,
    model: BaseDynamicsModel,
    data: DataModule,
    device: torch.device,
    dataset: str,
    target_h5: Path,
) -> FullNLBResult:
    model.to(device)
    model.eval()
    train_loader = data.train_loader(shuffle=False)
    valid_loader = data.valid_loader()

    if isinstance(model, PSTH):
        train_rates_heldin, train_rates_heldout = psth_full_rates(model, data, split="train")
        eval_rates_heldin, eval_rates_heldout = psth_full_rates(model, data, split="eval")
        eval_rates_heldin_forward = None
        eval_rates_heldout_forward = None
        eval_targets = collect_heldout_targets(valid_loader, device).cpu().numpy()
        metrics = compute_available_metrics(
            {"rates": torch.as_tensor(eval_rates_heldout)},
            {"spikes": torch.as_tensor(eval_targets)},
        )
        evaluation = EvaluationResult(
            metrics=metrics,
            predictions={"rates": eval_rates_heldout.astype(np.float32)},
            targets={"spikes": eval_targets.astype(np.float32)},
        )
    else:
        adapter = model.evaluation_adapter("nlb")
        if adapter is None:
            adapter = NLBCoSmoothingAdapter()
        adapter.fit(model, train_loader, device)
        train_eval = adapter.evaluate(model, train_loader, device)
        evaluation = adapter.evaluate(model, valid_loader, device)
        train_rates_heldout = train_eval.predictions["rates"]
        eval_rates_heldout = evaluation.predictions["rates"]
        train_rates_heldin = collect_heldin_rates(model, train_loader, device)
        eval_rates_heldin = collect_heldin_rates(model, valid_loader, device)
        eval_rates_heldin_forward, eval_rates_heldout_forward = collect_forward_rates(
            model,
            valid_loader,
            device,
        )

    full_metrics = _score_full_nlb_metrics(
        target_path=target_h5,
        dataset=dataset,
        eval_rates_heldout=eval_rates_heldout,
        eval_rates_heldin=eval_rates_heldin,
        train_rates_heldout=train_rates_heldout,
        train_rates_heldin=train_rates_heldin,
        co_bps=float(evaluation.metrics["co_bps"]),
        eval_rates_heldin_forward=eval_rates_heldin_forward,
        eval_rates_heldout_forward=eval_rates_heldout_forward,
    )
    return FullNLBResult(
        evaluation=evaluation,
        full_metrics=full_metrics,
        train_rates_heldin=train_rates_heldin,
        train_rates_heldout=train_rates_heldout,
        eval_rates_heldin=eval_rates_heldin,
        eval_rates_heldout=eval_rates_heldout,
        eval_rates_heldin_forward=eval_rates_heldin_forward,
        eval_rates_heldout_forward=eval_rates_heldout_forward,
    )


def collect_heldout_targets(loader: Iterable, device: torch.device) -> Tensor:
    targets = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        targets.append(batch["heldout_spikes"].detach().cpu())
    return torch.cat(targets, dim=0)


def collect_heldin_rates(
    model: BaseDynamicsModel,
    loader: Iterable,
    device: torch.device,
) -> np.ndarray:
    rates = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            x = observations_from_batch(batch)
            prediction = model.predict_rates(x)
            rates.append(prediction[:, : x.shape[1], : x.shape[-1]].detach().cpu())
    return torch.cat(rates, dim=0).numpy().astype(np.float32)


def collect_forward_rates(
    model: BaseDynamicsModel,
    loader: Iterable,
    device: torch.device,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    heldin_rates = []
    heldout_rates = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            if (
                not isinstance(batch, dict)
                or "heldin_forward_spikes" not in batch
                or "heldout_forward_spikes" not in batch
            ):
                return None, None
            x = observations_from_batch(batch)
            prediction = model.predict_rates(x)
            n_time = int(x.shape[1])
            n_heldin = int(x.shape[-1])
            n_forward = int(batch["heldin_forward_spikes"].shape[1])
            n_heldout = int(batch["heldout_forward_spikes"].shape[-1])
            if prediction.shape[1] < n_time + n_forward:
                return None, None
            if prediction.shape[-1] < n_heldin + n_heldout:
                return None, None
            heldin_rates.append(
                prediction[:, n_time : n_time + n_forward, :n_heldin].detach().cpu()
            )
            heldout_rates.append(
                prediction[
                    :,
                    n_time : n_time + n_forward,
                    n_heldin : n_heldin + n_heldout,
                ].detach().cpu()
            )
    if not heldin_rates:
        return None, None
    return (
        torch.cat(heldin_rates, dim=0).numpy().astype(np.float32),
        torch.cat(heldout_rates, dim=0).numpy().astype(np.float32),
    )


def psth_full_rates(model: PSTH, data: DataModule, *, split: str) -> tuple[np.ndarray, np.ndarray]:
    dataset = unwrap_dataset(data.train_dataset if split == "train" else data.valid_dataset)
    train_dataset = unwrap_dataset(data.train_dataset)
    if dataset is None or train_dataset is None:
        raise RuntimeError("PSTH full metrics require NLB datasets.")

    train_arrays = train_dataset.arrays
    target_trials = int(dataset.spikes.shape[0])
    heldin = condition_psth_rates(
        train_source=train_arrays.train_heldin_spikes,
        train_dataset=train_dataset,
        target_dataset=dataset,
        target_trials=target_trials,
        kernel=model.kernel,
        prediction_floor=model.prediction_floor,
        split=split,
    )
    heldout = condition_psth_rates(
        train_source=train_arrays.train_heldout_spikes,
        train_dataset=train_dataset,
        target_dataset=dataset,
        target_trials=target_trials,
        kernel=model.kernel,
        prediction_floor=model.prediction_floor,
        split=split,
    )
    return heldin.numpy().astype(np.float32), heldout.numpy().astype(np.float32)


def condition_psth_rates(
    *,
    train_source: Tensor,
    train_dataset: Any,
    target_dataset: Any,
    target_trials: int,
    kernel: Tensor,
    prediction_floor: float,
    split: str,
) -> Tensor:
    train_smoothed = smooth_trials(train_source.float(), kernel)
    fallback = train_smoothed.mean(dim=0).clamp_min(prediction_floor)
    rates = fallback.unsqueeze(0).expand(target_trials, -1, -1).clone()

    config = getattr(train_dataset, "config", None)
    if config is None or not hasattr(config, "resolved_data_path"):
        return rates
    path = Path(config.resolved_data_path)
    if not path.exists():
        return rates

    with h5py.File(path, "r") as handle:
        group_name = getattr(config, "resolved_group", None)
        group = handle[group_name] if isinstance(group_name, str) and group_name in handle else handle
        if "train_cond_idx" not in group:
            return rates
        train_cond_idx = group["train_cond_idx"][()]
        target_key = "train_cond_idx" if split == "train" else "eval_cond_idx"
        if target_key not in group:
            return rates
        target_cond_idx = group[target_key][()]

    for condition, trial_indices in enumerate(target_cond_idx):
        if condition >= len(train_cond_idx) or len(trial_indices) == 0:
            continue
        train_idx = torch.as_tensor(train_cond_idx[condition], dtype=torch.long)
        train_valid = train_idx[(train_idx >= 0) & (train_idx < train_smoothed.shape[0])]
        condition_rate = fallback if train_valid.numel() == 0 else train_smoothed.index_select(0, train_valid).mean(dim=0)
        idx = torch.as_tensor(trial_indices, dtype=torch.long)
        valid = idx[(idx >= 0) & (idx < target_trials)]
        if valid.numel() > 0:
            rates[valid] = condition_rate.clamp_min(prediction_floor)
    return torch.nan_to_num(rates, nan=prediction_floor).clamp_min(prediction_floor)


def smooth_trials(x: Tensor, kernel: Tensor) -> Tensor:
    import torch.nn.functional as F

    kernel = kernel.to(device=x.device, dtype=x.dtype)
    if kernel.numel() <= 1:
        return x
    batch, time_steps, neurons = x.shape
    weight = kernel.flip(0).view(1, 1, -1).repeat(neurons, 1, 1)
    padded = F.pad(x.transpose(1, 2), (kernel.numel() - 1, kernel.numel() - 1))
    full = F.conv1d(padded, weight, groups=neurons)
    start = (kernel.numel() - 1) // 2
    return full[..., start : start + time_steps].transpose(1, 2).reshape(batch, time_steps, neurons)


def unwrap_dataset(dataset: Any) -> Any:
    while dataset is not None and hasattr(dataset, "dataset"):
        dataset = getattr(dataset, "dataset")
    return dataset


def write_full_artifacts(*, run_dir: Path, dataset: str, full: FullNLBResult) -> None:
    submission_path = run_dir / "eval_rates.h5"
    with h5py.File(submission_path, "w") as handle:
        group = handle.create_group(dataset)
        group.create_dataset("train_rates_heldin", data=full.train_rates_heldin, compression="gzip")
        group.create_dataset("train_rates_heldout", data=full.train_rates_heldout, compression="gzip")
        group.create_dataset("eval_rates_heldin", data=full.eval_rates_heldin, compression="gzip")
        group.create_dataset("eval_rates_heldout", data=full.eval_rates_heldout, compression="gzip")
        if full.eval_rates_heldin_forward is not None:
            group.create_dataset(
                "eval_rates_heldin_forward",
                data=full.eval_rates_heldin_forward,
                compression="gzip",
            )
        if full.eval_rates_heldout_forward is not None:
            group.create_dataset(
                "eval_rates_heldout_forward",
                data=full.eval_rates_heldout_forward,
                compression="gzip",
            )

    prediction_payload: dict[str, Any] = {
        "train_rates_heldin": full.train_rates_heldin,
        "train_rates_heldout": full.train_rates_heldout,
        "eval_rates_heldin": full.eval_rates_heldin,
        "eval_rates_heldout": full.eval_rates_heldout,
        "pred_rates": full.eval_rates_heldout,
        "target_spikes": full.evaluation.targets["spikes"],
    }
    if full.eval_rates_heldin_forward is not None:
        prediction_payload["eval_rates_heldin_forward"] = full.eval_rates_heldin_forward
    if full.eval_rates_heldout_forward is not None:
        prediction_payload["eval_rates_heldout_forward"] = full.eval_rates_heldout_forward
    np.savez_compressed(
        run_dir / "full_predictions.npz",
        **prediction_payload,
    )
    payload = [{f"{dataset}_split": full.full_metrics}]
    (run_dir / "full_nlb_metrics_official.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
    else:
        metrics = {}
    metrics.update(full.full_metrics)
    _write_json(metrics_path, metrics)


def write_epoch_metric_artifact(run_dir: Path, history: list[Any]) -> None:
    keys = sorted({key for report in history for key in report.metrics})
    if not keys:
        return
    path = run_dir / "epoch_nlb_metrics.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "seconds", *keys])
        writer.writeheader()
        for report in history:
            row: dict[str, Any] = {
                "epoch": int(report.epoch) + 1,
                "seconds": float(report.seconds),
            }
            for key in keys:
                value = report.metrics.get(key, float("nan"))
                row[key] = "" if not math.isfinite(float(value)) else float(value)
            writer.writerow(row)


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = [
        "dataset",
        "method",
        "co-bps",
        "vel R2",
        "tp corr",
        "psth R2",
        "seconds",
        "run_dir",
    ]
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    (output_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n"
    )


def _row_from_existing(dataset: str, method: str, run_dir: Path) -> dict[str, Any]:
    data = json.loads((run_dir / "full_nlb_metrics_official.json").read_text())
    metrics = data[0][f"{dataset}_split"]
    return {
        "dataset": dataset,
        "method": method,
        "run_dir": str(run_dir),
        "seconds": float("nan"),
        **metrics,
    }


if __name__ == "__main__":
    raise SystemExit(main())
