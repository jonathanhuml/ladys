"""MINT NLB hidden-test runner integrated with LaDyS configs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any

import h5py
import numpy as np
import torch

from ladys.config import ExperimentConfig, load_experiment_config
from ladys.datasets.nlb import DATASET_TO_DANDISET, DATASET_TO_TEST_NWB
from ladys.experiment import experiment_config_to_dict
from ladys.models.mint import MINT, MINTConfig
from ladys.models.mint_core.co_bps import heldout_count, observed_neuron_mask
from ladys.models.mint_core.nwb_data import default_train_nwb_path, get_nwb_trial_data
from ladys.models.mint_core.tasks import get_trial_data
from ladys.models.mint_core.utils import TORCH_DTYPE, bin_data
from ladys.nlb_eval import score_count_predictions


@dataclass(frozen=True)
class MINTNLBResult:
    """Artifacts from a MINT NLB run."""

    run_dir: Path
    co_bps: float
    metrics_path: Path
    predictions_path: Path
    submission_path: Path
    csv_path: Path


def run_mint_nlb_config(
    config_path: Path | str,
    *,
    output_dir: Path | str | None = None,
    run_name: str | None = None,
    device: str | None = None,
) -> MINTNLBResult:
    """Run one MINT NLB hidden-test config."""

    config = load_experiment_config(str(config_path))
    if not isinstance(config.model, MINTConfig):
        raise TypeError(f"{config_path} model must be `mint`, got {type(config.model).__name__}.")
    if output_dir is not None:
        config.output_dir = str(output_dir)
    if run_name is not None:
        config.run_name = run_name
    if device is not None:
        config.trainer.device = device
    return run_mint_nlb(config)


def run_mint_nlb(config: ExperimentConfig) -> MINTNLBResult:
    """Run MINT on a public NLB hidden-test split from a LaDyS config."""

    if not isinstance(config.model, MINTConfig):
        raise TypeError(f"Expected MINTConfig, got {type(config.model).__name__}.")

    cfg = config.model
    dataset = cfg.dataset
    device = torch.device(config.trainer.device)
    model = MINT(cfg).to(device)
    _apply_mint_overrides(dataset, model.hyperparams, cfg)
    model.settings.data_path = Path(cfg.mat_data_root) / f"{dataset}.mat"

    train_split = cfg.train_split
    if train_split == "auto":
        train_split = "trainval" if cfg.nlb_neural_state_defaults else ("train" if dataset == "mc_rtt" else "trainval")

    train_S, train_Z, condition = _load_training_data(model, train_split, cfg, device)
    core = model.make_core().fit(train_S, train_Z, condition)
    model.core = core

    test_nwb = _test_nwb_path(dataset, Path(cfg.nwb_root))
    heldin = _load_buffered_heldin(dataset, test_nwb, model.settings, model.hyperparams)
    _, keep = _buffered_alignment(model.settings, model.hyperparams)
    S = _heldin_to_mint_spikes(heldin, dataset, device)
    mask = observed_neuron_mask(dataset, S[0].shape[0], device)
    x_hat, _ = core.predict(S, likelihood_neuron_mask=mask)
    x_eval = [item[:, keep] for item in x_hat]

    n_heldout = heldout_count(dataset)
    eval_rates_heldout = _binned_counts_from_state(
        x_eval,
        0,
        n_heldout,
        cfg.eval_bin_size_ms,
        core.Delta,
    )
    eval_rates_heldin = _binned_counts_from_state(
        x_eval,
        n_heldout,
        x_eval[0].shape[0],
        cfg.eval_bin_size_ms,
        core.Delta,
    )
    target = _read_target_spikes(_resolve_target_h5(cfg.target_h5), dataset)
    if eval_rates_heldout.shape != target.shape:
        raise ValueError(f"{dataset}: predicted {eval_rates_heldout.shape}, target {target.shape}")
    if np.isnan(eval_rates_heldout).any():
        raise ValueError(f"{dataset}: NaNs found in predicted held-out rates.")

    score = score_count_predictions(eval_rates_heldout.astype(float), target.astype(float))
    run_dir = _make_run_dir(config, dataset)
    return _write_run_artifacts(
        run_dir=run_dir,
        config=config,
        dataset=dataset,
        train_split=train_split,
        n_train_units=len(train_S),
        n_eval_trials=heldin.shape[0],
        eval_rates_heldout=eval_rates_heldout,
        eval_rates_heldin=eval_rates_heldin,
        target=target,
        score=score.co_bps,
    )


def _apply_mint_overrides(dataset: str, hyperparams, cfg: MINTConfig) -> None:
    if cfg.nlb_neural_state_defaults:
        hyperparams.causal = False
        hyperparams.window_length = 500
        hyperparams.n_candidates = 5 if dataset == "mc_rtt" else 2
        hyperparams.min_rate = 0.1
    if cfg.causal is not None:
        hyperparams.causal = bool(cfg.causal)
    if cfg.n_candidates is not None:
        hyperparams.n_candidates = int(cfg.n_candidates)
    if cfg.window_length is not None:
        hyperparams.window_length = int(cfg.window_length)
    if cfg.delta is not None:
        hyperparams.Delta = int(cfg.delta)
    if cfg.sigma is not None:
        hyperparams.sigma = int(cfg.sigma)
    if cfg.min_rate is not None:
        hyperparams.min_rate = float(cfg.min_rate)


def _load_training_data(model: MINT, train_split: str, cfg: MINTConfig, device: torch.device):
    dataset = cfg.dataset
    if cfg.train_source == "mat":
        S, Z, condition, _ = get_trial_data(model.settings, train_split, None, device)
        return S, Z, condition
    if dataset == "mc_rtt":
        raise ValueError("Direct MC_RTT NWB training needs AutoLFADS rates; use train_source: mat.")
    train_nwb = default_train_nwb_path(dataset, Path(cfg.nwb_root))
    S, Z, condition, _ = get_nwb_trial_data(model.settings, train_split, train_nwb, None, device)
    return S, Z, condition


def _buffered_alignment(settings, hyperparams) -> tuple[np.ndarray, np.ndarray]:
    test_buffer = [-hyperparams.window_length + 1, 0]
    if not hyperparams.causal:
        adjustment = round((hyperparams.window_length + hyperparams.Delta) / 2)
        test_buffer = [test_buffer[0] + adjustment, test_buffer[1] + adjustment]
    test_alignment = np.asarray(list(settings.test_alignment), dtype=np.int64)
    buffered = np.arange(settings.test_alignment.start + test_buffer[0], settings.test_alignment.stop + test_buffer[1])
    keep = np.isin(buffered, test_alignment)
    return buffered, keep


def _load_buffered_heldin(dataset: str, nwb_path: Path, settings, hyperparams) -> np.ndarray:
    from nlb_tools.make_tensors import PARAMS
    from nlb_tools.nwb_interface import NWBDataset

    buffered, _ = _buffered_alignment(settings, hyperparams)
    make_params = PARAMS[dataset]["make_params"].copy()
    make_params["align_range"] = (int(buffered[0]), int(buffered[-1] + 1))
    make_params["allow_overlap"] = True
    make_params["allow_nans"] = True
    ds = NWBDataset(nwb_path)
    trial_mask = ds.trial_info["split"] == "test"
    trial_data = ds.make_trial_data(ignored_trials=~trial_mask, **make_params)
    grouped = dict(tuple(trial_data.groupby("trial_id", sort=False)))

    n_trials = int(trial_mask.sum())
    n_time = len(buffered)
    n_heldin = ds.data[PARAMS[dataset]["spk_field"]].shape[1]
    heldin = np.full((n_trials, n_time, n_heldin), np.nan, dtype=np.float32)
    offset_lookup = {int(t): i for i, t in enumerate(buffered)}
    for out_idx, trial_id in enumerate(ds.trial_info.loc[trial_mask, "trial_id"]):
        trial = grouped.get(trial_id)
        if trial is None:
            continue
        align_ms = (trial[("align_time", "")].dt.total_seconds().to_numpy() * 1000).round().astype(int)
        keep = np.asarray([t in offset_lookup for t in align_ms], dtype=bool)
        if not np.any(keep):
            continue
        dest = np.asarray([offset_lookup[int(t)] for t in align_ms[keep]], dtype=np.int64)
        heldin[out_idx, dest] = trial[PARAMS[dataset]["spk_field"]].to_numpy(dtype=np.float32)[keep]
    return heldin


def _heldin_to_mint_spikes(heldin: np.ndarray, dataset: str, device: torch.device) -> list[torch.Tensor]:
    n_heldout = heldout_count(dataset)
    out = []
    for trial in heldin:
        heldin_t = torch.as_tensor(trial.T, dtype=TORCH_DTYPE, device=device)
        heldout_placeholder = torch.zeros((n_heldout, heldin_t.shape[1]), dtype=TORCH_DTYPE, device=device)
        out.append(torch.cat([heldout_placeholder, heldin_t], dim=0))
    return out


def _binned_counts_from_state(
    x_hat: list[torch.Tensor],
    start: int,
    stop: int,
    eval_bin_size: int,
    delta: int,
) -> np.ndarray:
    pieces = []
    for item in x_hat:
        counts = bin_data(item[start:stop], eval_bin_size, "mean") * (eval_bin_size / delta)
        pieces.append(counts.cpu().numpy().T)
    return np.stack(pieces, axis=0)


def _test_nwb_path(dataset: str, nwb_root: Path) -> Path:
    return nwb_root / DATASET_TO_DANDISET[dataset] / DATASET_TO_TEST_NWB[dataset]


def _resolve_target_h5(path: str | None) -> Path:
    if path is not None:
        target = Path(path)
        if target.exists():
            return target
        raise FileNotFoundError(f"NLB target H5 not found: {target}")
    candidates = [
        Path("data/real/nlb/eval_data_test.h5"),
        Path("data/real/eval_data_test.h5"),
        Path("data/eval_data_test.h5"),
        Path("../mint/data/eval_data_test.h5"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find NLB public test target H5.")


def _read_target_spikes(path: Path, dataset: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return handle[dataset]["eval_spikes_heldout"][()].astype(float)


def _make_run_dir(config: ExperimentConfig, dataset: str) -> Path:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = config.run_name or f"{timestamp}_{dataset}_mint"
    return _create_unique_dir(output_dir / _slugify(run_name))


def _write_run_artifacts(
    *,
    run_dir: Path,
    config: ExperimentConfig,
    dataset: str,
    train_split: str,
    n_train_units: int,
    n_eval_trials: int,
    eval_rates_heldout: np.ndarray,
    eval_rates_heldin: np.ndarray,
    target: np.ndarray,
    score: float,
) -> MINTNLBResult:
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    predictions_path = run_dir / "predictions.npz"
    submission_path = run_dir / "mint_eval_rates.h5"
    csv_path = run_dir / "hidden_test_co_bps.csv"
    report_path = run_dir / "report.md"

    metrics = {
        "co_bps": float(score),
        "dataset": dataset,
        "train_split": train_split,
        "train_source": config.model.train_source if isinstance(config.model, MINTConfig) else None,
        "n_train_units": int(n_train_units),
        "n_eval_trials": int(n_eval_trials),
    }
    _write_json(config_path, experiment_config_to_dict(config))
    _write_json(metrics_path, metrics)
    np.savez_compressed(
        predictions_path,
        pred_rates=eval_rates_heldout.astype(np.float32),
        target_spikes=target.astype(np.float32),
        pred_rates_heldin=eval_rates_heldin.astype(np.float32),
    )
    with h5py.File(submission_path, "w") as handle:
        group = handle.create_group(dataset)
        group.create_dataset("eval_rates_heldin", data=eval_rates_heldin.astype(np.float32), compression="gzip")
        group.create_dataset("eval_rates_heldout", data=eval_rates_heldout.astype(np.float32), compression="gzip")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "train_split", "train_source", "n_train_units", "n_eval_trials", "co_bps"],
        )
        writer.writeheader()
        writer.writerow(metrics)
    report_path.write_text(
        "\n".join(
            [
                "# LaDyS MINT NLB Report",
                "",
                f"- Dataset: `{dataset}`",
                f"- Train split: `{train_split}`",
                f"- Train source: `{metrics['train_source']}`",
                f"- co-BPS: `{score:.12g}`",
                f"- Predictions: `{predictions_path.name}`",
                f"- EvalAI rates: `{submission_path.name}`",
            ]
        )
        + "\n"
    )
    return MINTNLBResult(
        run_dir=run_dir,
        co_bps=float(score),
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        submission_path=submission_path,
        csv_path=csv_path,
    )


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(_json_ready(data), indent=2, sort_keys=True) + "\n")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    return value


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return slug or "run"


def _create_unique_dir(path: Path) -> Path:
    if not path.exists():
        path.mkdir(parents=False)
        return path
    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.name}-{index}")
        if not candidate.exists():
            candidate.mkdir(parents=False)
            return candidate
    raise RuntimeError(f"Could not create a unique run directory for {path}.")
