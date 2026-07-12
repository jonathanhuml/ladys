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
from ladys.models.mint import (
    MINT,
    MINTConfig,
    MintMatFile,
    TORCH_DTYPE,
    _spikes_tensor,
    bin_data,
    default_train_nwb_path,
    find_time_index,
    get_nwb_trial_data,
    get_trial_data,
    heldout_count,
    observed_neuron_mask,
)
from ladys.models.lfads import LFADSConfig
from ladys.nlb_eval import score_count_predictions


@dataclass(frozen=True)
class MINTNLBResult:
    """Artifacts from a MINT NLB run."""

    run_dir: Path
    co_bps: float
    metrics: dict[str, float]
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
    _apply_source_overrides(model, cfg)
    _apply_mint_overrides(dataset, model.hyperparams, cfg)
    model.settings.data_path = Path(cfg.mat_data_root) / f"{dataset}.mat"

    train_split = cfg.train_split
    if train_split == "auto":
        train_split = "trainval" if cfg.nlb_neural_state_defaults else ("train" if dataset == "mc_rtt" else "trainval")

    train_S, train_Z, condition = _load_training_data(model, train_split, config, device)
    model.fit_library(train_S, train_Z, condition)
    target_path = _resolve_target_h5(cfg.target_h5)

    train_rates_heldout = None
    train_rates_heldin = None
    if dataset == "mc_rtt" and cfg.train_source == "mat":
        train_rates_heldout, train_rates_heldin = _load_mc_rtt_mat_train_rate_windows(
            model,
            train_split,
            cfg.eval_bin_size_ms,
        )
    elif dataset != "dmfc_rsg":
        train_keep = _training_eval_keep(model, train_S)
        train_rates_heldout, train_rates_heldin = _predict_binned_rate_parts(
            model,
            train_S,
            dataset,
            keep=train_keep,
            eval_bin_size_ms=cfg.eval_bin_size_ms,
        )

    if dataset == "dmfc_rsg" and cfg.train_source in {"h5", "lfads"}:
        heldin = _load_h5_eval_heldin(_experiment_h5_path(config), dataset)
        keep = np.ones(heldin.shape[1], dtype=bool)
    elif dataset == "dmfc_rsg":
        test_nwb = _test_nwb_path(dataset, Path(cfg.nwb_root))
        heldin = _load_dmfc_eval_heldin_nwb(test_nwb)
        keep = np.ones(heldin.shape[1], dtype=bool)
    else:
        test_nwb = _test_nwb_path(dataset, Path(cfg.nwb_root))
        heldin = _load_buffered_heldin(dataset, test_nwb, model.settings, model.hyperparams)
        _, keep = _buffered_alignment(model.settings, model.hyperparams)
    S = _heldin_to_mint_spikes(heldin, dataset, device)
    mask = observed_neuron_mask(dataset, S[0].shape[0], device)
    x_hat, _ = model.predict_spike_trials(S, likelihood_neuron_mask=mask, verbose=False)
    x_eval = [item[:, keep] for item in x_hat]

    n_heldout = heldout_count(dataset)
    eval_rates_heldout = _binned_counts_from_state(
        x_eval,
        0,
        n_heldout,
        cfg.eval_bin_size_ms,
        model.Delta,
        model.Ts * 1000.0,
    )
    eval_rates_heldin = _binned_counts_from_state(
        x_eval,
        n_heldout,
        x_eval[0].shape[0],
        cfg.eval_bin_size_ms,
        model.Delta,
        model.Ts * 1000.0,
    )
    target = _read_target_spikes(target_path, dataset)
    if eval_rates_heldout.shape != target.shape:
        raise ValueError(f"{dataset}: predicted {eval_rates_heldout.shape}, target {target.shape}")
    if np.isnan(eval_rates_heldout).any():
        raise ValueError(f"{dataset}: NaNs found in predicted held-out rates.")

    score = score_count_predictions(eval_rates_heldout.astype(float), target.astype(float))
    full_metrics = _score_full_nlb_metrics(
        target_path=target_path,
        dataset=dataset,
        eval_rates_heldout=eval_rates_heldout,
        eval_rates_heldin=eval_rates_heldin,
        train_rates_heldout=train_rates_heldout,
        train_rates_heldin=train_rates_heldin,
        co_bps=score.co_bps,
    )
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
        train_rates_heldout=train_rates_heldout,
        train_rates_heldin=train_rates_heldin,
        target=target,
        score=score.co_bps,
        full_metrics=full_metrics,
    )


def _apply_mint_overrides(dataset: str, hyperparams, cfg: MINTConfig) -> None:
    if cfg.nlb_neural_state_defaults:
        hyperparams.causal = False
        if dataset == "dmfc_rsg" and cfg.train_source == "nwb":
            hyperparams.window_length = 1500
            hyperparams.Delta = 20
        else:
            hyperparams.window_length = 500
        if dataset == "dmfc_rsg" and cfg.train_source in {"h5", "lfads"}:
            hyperparams.window_length = 100
            hyperparams.Delta = 4
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


def _apply_source_overrides(model: MINT, cfg: MINTConfig) -> None:
    if cfg.dataset == "dmfc_rsg" and cfg.train_source in {"h5", "lfads"}:
        model.settings.Ts = 0.005
        model.settings.trial_alignment = range(0, 300)
        model.settings.test_alignment = range(0, 300)
        model.hyperparams.trajectories_alignment = range(0, 300)
        model.hyperparams.sigma = 14
        model.hyperparams.Delta = 4
        model.hyperparams.window_length = 100
    if cfg.dataset == "dmfc_rsg" and cfg.train_source == "lfads":
        model.settings.library_rate_source = "lfads"


def _load_training_data(model: MINT, train_split: str, config: ExperimentConfig, device: torch.device):
    cfg = config.model
    if not isinstance(cfg, MINTConfig):
        raise TypeError(f"Expected MINTConfig, got {type(cfg).__name__}.")
    dataset = cfg.dataset
    if cfg.train_source == "h5":
        return _load_h5_training_data(_experiment_h5_path(config), dataset, device)
    if cfg.train_source == "lfads":
        return _load_lfads_training_data(model, train_split, config, device)
    if cfg.train_source == "mat":
        S, Z, condition, _ = get_trial_data(model.settings, train_split, None, device)
        return S, Z, condition
    if dataset == "mc_rtt":
        raise ValueError("Direct MC_RTT NWB training needs AutoLFADS rates; use train_source: mat.")
    train_nwb = default_train_nwb_path(dataset, Path(cfg.nwb_root))
    S, Z, condition, _ = get_nwb_trial_data(model.settings, train_split, train_nwb, None, device)
    return S, Z, condition


def _load_lfads_training_data(
    model: MINT,
    train_split: str,
    config: ExperimentConfig,
    device: torch.device,
):
    cfg = config.model
    if not isinstance(cfg, MINTConfig):
        raise TypeError(f"Expected MINTConfig, got {type(cfg).__name__}.")
    if cfg.dataset != "dmfc_rsg":
        raise ValueError("MINT train_source='lfads' is currently implemented for dmfc_rsg only.")

    del train_split
    raw_S, Z, condition = _load_h5_training_data(_experiment_h5_path(config), cfg.dataset, device)
    rate_S = _fit_lfads_rate_trials(raw_S, model, cfg, device)
    return rate_S, Z, condition


def _fit_lfads_rate_trials(
    S: list[torch.Tensor],
    mint_model: MINT,
    cfg: MINTConfig,
    device: torch.device,
) -> list[torch.Tensor]:
    if not S:
        raise ValueError("LFADS rate estimation received no MINT training trials.")

    n_neurons, n_time = S[0].shape
    train_bin_size = max(1, int(cfg.lfads_train_bin_size))
    if train_bin_size > 1:
        binned_trials = [bin_data(trial, train_bin_size, "sum").T for trial in S]
        lfads_dt = mint_model.settings.Ts * train_bin_size
    else:
        binned_trials = [trial.T for trial in S]
        lfads_dt = mint_model.settings.Ts
    spikes = torch.stack([trial.to(dtype=torch.float32) for trial in binned_trials], dim=0)
    spikes = torch.nan_to_num(spikes, nan=0.0, posinf=0.0, neginf=0.0).to(device)

    torch.manual_seed(0)
    lfads_cfg = LFADSConfig(
        generator_dim=cfg.lfads_generator_dim,
        factor_dim=cfg.lfads_factor_dim,
        inferred_input_dim=cfg.lfads_inferred_input_dim,
        g0_encoder_dim=cfg.lfads_encoder_dim,
        controller_encoder_dim=cfg.lfads_encoder_dim,
        controller_dim=cfg.lfads_controller_dim,
        keep_prob=cfg.lfads_keep_prob,
        dt=lfads_dt,
        optimization={"name": "gradient", "optimizer": "Adam", "lr": cfg.lfads_lr},
    )
    lfads = lfads_cfg.build(n_neurons=n_neurons, n_time=spikes.shape[1]).to(device)
    optimizer = torch.optim.Adam(lfads.parameters(), lr=cfg.lfads_lr)
    batch_size = max(1, int(cfg.lfads_batch_size))
    dt = torch.as_tensor(lfads_dt, dtype=torch.float32, device=device)

    for epoch in range(max(0, int(cfg.lfads_epochs))):
        lfads.train()
        permutation = torch.randperm(spikes.shape[0], device=device)
        epoch_loss = 0.0
        epoch_items = 0
        for start in range(0, spikes.shape[0], batch_size):
            idx = permutation[start : start + batch_size]
            batch_x = spikes.index_select(0, idx)
            batch = {
                "spikes": batch_x,
                "dt": dt.expand(batch_x.shape[0]),
            }
            output = lfads(batch_x)
            loss = lfads.loss(batch, output, epoch=epoch)
            optimizer.zero_grad(set_to_none=True)
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(lfads.parameters(), 200.0)
            optimizer.step()
            epoch_loss += float(loss.total.detach().cpu()) * int(batch_x.shape[0])
            epoch_items += int(batch_x.shape[0])
        if epoch_items:
            mean_loss = epoch_loss / epoch_items
            print(f"LFADS library epoch {epoch + 1}/{cfg.lfads_epochs}: loss={mean_loss:.6g}", flush=True)

    lfads.eval()
    rate_batches = []
    with torch.no_grad():
        for start in range(0, spikes.shape[0], batch_size):
            batch_x = spikes[start : start + batch_size]
            rate_batches.append(lfads.predict_rates(batch_x).detach().cpu())
    rates_hz = torch.cat(rate_batches, dim=0)
    expected_counts = rates_hz * float(mint_model.settings.Ts * mint_model.hyperparams.Delta)
    if train_bin_size > 1:
        expected_counts = expected_counts.repeat_interleave(train_bin_size, dim=1)
        expected_counts = expected_counts[:, :n_time]
    return [
        expected_counts[trial].T.to(dtype=TORCH_DTYPE, device=device).contiguous()
        for trial in range(expected_counts.shape[0])
    ]


def _experiment_h5_path(config: ExperimentConfig) -> Path:
    data_config = config.dataset
    if hasattr(data_config, "resolved_data_path"):
        return Path(data_config.resolved_data_path)
    data_path = getattr(data_config, "data_path", None)
    if data_path is None:
        raise ValueError("H5-backed MINT runs require dataset.data_path.")
    return Path(data_path)


def _select_nlb_h5_group(handle: h5py.File, dataset: str):
    if dataset in handle:
        return handle[dataset]
    return handle


def _load_h5_training_data(path: Path, dataset: str, device: torch.device):
    with h5py.File(path, "r") as handle:
        group = _select_nlb_h5_group(handle, dataset)
        heldin = np.asarray(group["train_spikes_heldin"], dtype=np.float64)
        heldout = np.asarray(group["train_spikes_heldout"], dtype=np.float64)
        behavior = np.asarray(group.get("train_behavior", np.zeros((heldin.shape[0], 1))), dtype=np.float64)
        if "train_cond_idx" in group:
            cond_trials = [np.asarray(item, dtype=np.int64) for item in group["train_cond_idx"][()]]
        else:
            cond_trials = _condition_trials_from_behavior(behavior)

    if heldin.shape[:2] != heldout.shape[:2]:
        raise ValueError(f"{path}: train held-in {heldin.shape} and held-out {heldout.shape} are incompatible.")

    n_trials, n_time, _ = heldin.shape
    S, Z, condition = [], [], []
    for cond, trials in enumerate(cond_trials):
        for trial in trials:
            trial_idx = int(trial)
            if trial_idx < 0 or trial_idx >= n_trials:
                continue
            spikes = np.concatenate([heldout[trial_idx], heldin[trial_idx]], axis=1).T
            values = np.nan_to_num(behavior[trial_idx], nan=0.0, posinf=0.0, neginf=0.0)
            z = np.repeat(values[:, None], n_time, axis=1)
            S.append(_spikes_tensor(spikes, device))
            Z.append(torch.as_tensor(z, dtype=TORCH_DTYPE, device=device))
            condition.append(cond)

    if not S:
        raise ValueError(f"{path}: no condition-indexed training trials found for {dataset}.")
    return S, Z, np.asarray(condition, dtype=np.int64)


def _condition_trials_from_behavior(behavior: np.ndarray) -> list[np.ndarray]:
    cond_fields = behavior[:, : min(4, behavior.shape[1])]
    finite = np.all(np.isfinite(cond_fields), axis=1)
    cond_list = np.unique(cond_fields[finite], axis=0)
    trials = []
    for row in cond_list:
        trials.append(np.flatnonzero(finite & np.all(cond_fields == row, axis=1)).astype(np.int64))
    return trials


def _load_h5_eval_heldin(path: Path, dataset: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        group = _select_nlb_h5_group(handle, dataset)
        return np.asarray(group["eval_spikes_heldin"], dtype=np.float32)


def _load_dmfc_eval_heldin_nwb(nwb_path: Path) -> np.ndarray:
    from nlb_tools.make_tensors import make_eval_input_tensors
    from nlb_tools.nwb_interface import NWBDataset

    ds = NWBDataset(nwb_path)
    data = make_eval_input_tensors(
        ds,
        dataset_name="dmfc_rsg",
        trial_split="test",
        save_file=False,
        return_dict=True,
        seed=0,
    )
    heldin = np.asarray(data["eval_spikes_heldin"], dtype=np.float32)
    if heldin.shape[1] != 1500:
        raise ValueError(f"dmfc_rsg test NWB produced {heldin.shape[1]} eval samples, expected 1500.")
    return heldin


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


def _training_eval_keep(model: MINT, train_trials: list[torch.Tensor]) -> np.ndarray:
    if not train_trials:
        raise ValueError("Cannot produce train rates without training trials.")
    n_time = int(train_trials[0].shape[1])
    trial_alignment = np.asarray(list(model.settings.trial_alignment), dtype=np.int64)
    eval_alignment = np.asarray(list(model.settings.test_alignment), dtype=np.int64)
    if trial_alignment.size == n_time:
        keep = np.isin(trial_alignment, eval_alignment)
        if int(keep.sum()) != int(eval_alignment.size):
            raise ValueError(
                "Training alignment does not contain every eval-aligned time point "
                f"({keep.sum()} of {eval_alignment.size})."
            )
        return keep
    if eval_alignment.size == n_time:
        return np.ones(n_time, dtype=bool)
    raise ValueError(
        "Cannot align MINT train rates to the NLB eval window: "
        f"train trial has {n_time} samples, trial alignment has {trial_alignment.size}, "
        f"eval alignment has {eval_alignment.size}."
    )


def _predict_binned_rate_parts(
    model: MINT,
    spikes: list[torch.Tensor],
    dataset: str,
    *,
    keep: np.ndarray | None,
    eval_bin_size_ms: int,
    likelihood_neuron_mask: torch.Tensor | None = None,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    n_heldout = heldout_count(dataset)
    heldout_batches: list[np.ndarray] = []
    heldin_batches: list[np.ndarray] = []
    for start in range(0, len(spikes), batch_size):
        batch = spikes[start : start + batch_size]
        x_hat, _ = model.predict_spike_trials(
            batch,
            likelihood_neuron_mask=likelihood_neuron_mask,
            verbose=False,
        )
        x_eval = [item[:, keep] for item in x_hat] if keep is not None else x_hat
        heldout_batches.append(
            _binned_counts_from_state(
                x_eval,
                0,
                n_heldout,
                eval_bin_size_ms,
                model.Delta,
                model.Ts * 1000.0,
            ).astype(np.float32, copy=False)
        )
        heldin_batches.append(
            _binned_counts_from_state(
                x_eval,
                n_heldout,
                x_eval[0].shape[0],
                eval_bin_size_ms,
                model.Delta,
                model.Ts * 1000.0,
            ).astype(np.float32, copy=False)
        )
    return np.concatenate(heldout_batches, axis=0), np.concatenate(heldin_batches, axis=0)


def _load_mc_rtt_mat_train_rate_windows(
    model: MINT,
    train_split: str,
    eval_bin_size_ms: int,
) -> tuple[np.ndarray, np.ndarray]:
    T, TrialInfo = MintMatFile(model.settings.data_path, "mc_rtt").load()
    if train_split == "train":
        split_labels = {"train"}
    elif train_split == "trainval":
        split_labels = {"train", "val"}
    else:
        raise ValueError(f"MC_RTT MAT train rates do not support split '{train_split}'.")

    sample_period_ms = model.Ts * 1000.0
    bin_samples_float = float(eval_bin_size_ms) / float(sample_period_ms)
    bin_samples = int(round(bin_samples_float))
    if bin_samples <= 0 or not np.isclose(bin_samples, bin_samples_float):
        raise ValueError(
            f"eval_bin_size_ms={eval_bin_size_ms} is not an integer number of "
            f"samples at {sample_period_ms:g} ms/sample."
        )

    alignment = np.asarray(list(model.settings.test_alignment), dtype=np.int64)
    n_heldout = heldout_count("mc_rtt")
    pieces: list[np.ndarray] = []
    for trial_idx in range(TrialInfo.n_trials):
        if str(TrialInfo["split"][trial_idx]) not in split_labels:
            continue
        start_idx = find_time_index(T["time"], float(TrialInfo["start_time"][trial_idx]))
        sample_idx = start_idx + alignment
        rates_hz = np.nan_to_num(
            T["autolfads_rates"][:, sample_idx],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        counts = bin_data(torch.as_tensor(rates_hz, dtype=TORCH_DTYPE), bin_samples, "mean")
        counts = counts.cpu().numpy().T * (float(eval_bin_size_ms) / 1000.0)
        pieces.append(counts.astype(np.float32, copy=False))

    if not pieces:
        raise ValueError(f"MC_RTT MAT train-rate export found no trials for split '{train_split}'.")
    rates = np.stack(pieces, axis=0)
    return rates[:, :, :n_heldout], rates[:, :, n_heldout:]


def _binned_counts_from_state(
    x_hat: list[torch.Tensor],
    start: int,
    stop: int,
    eval_bin_size_ms: int,
    delta_samples: int,
    sample_period_ms: float,
) -> np.ndarray:
    bin_samples_float = float(eval_bin_size_ms) / float(sample_period_ms)
    bin_samples = int(round(bin_samples_float))
    if bin_samples <= 0 or not np.isclose(bin_samples, bin_samples_float):
        raise ValueError(
            f"eval_bin_size_ms={eval_bin_size_ms} is not an integer number of "
            f"samples at {sample_period_ms:g} ms/sample."
        )
    scale = float(eval_bin_size_ms) / (float(delta_samples) * float(sample_period_ms))
    pieces = []
    for item in x_hat:
        counts = bin_data(item[start:stop], bin_samples, "mean") * scale
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
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find NLB public test target H5.")


def _read_target_spikes(path: Path, dataset: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return handle[dataset]["eval_spikes_heldout"][()].astype(float)


def _score_full_nlb_metrics(
    *,
    target_path: Path,
    dataset: str,
    eval_rates_heldout: np.ndarray,
    eval_rates_heldin: np.ndarray,
    train_rates_heldout: np.ndarray | None,
    train_rates_heldin: np.ndarray | None,
    co_bps: float,
) -> dict[str, float]:
    metrics = {"co-bps": float(co_bps)}
    try:
        from nlb_tools.evaluation import eval_psth, speed_tp_correlation, velocity_decoding
    except ImportError:
        return metrics

    with h5py.File(target_path, "r") as handle:
        if dataset not in handle:
            return metrics
        group = handle[dataset]
        eval_rates = np.concatenate([eval_rates_heldin, eval_rates_heldout], axis=-1).astype(float, copy=False)
        if dataset == "dmfc_rsg" and "eval_behavior" in group:
            metrics["tp corr"] = float(
                speed_tp_correlation(
                    group["eval_spikes_heldout"][()].astype(float),
                    eval_rates,
                    group["eval_behavior"][()].astype(float),
                )
            )
        elif (
            train_rates_heldout is not None
            and train_rates_heldin is not None
            and "train_behavior" in group
            and "eval_behavior" in group
        ):
            train_rates = np.concatenate([train_rates_heldin, train_rates_heldout], axis=-1).astype(float, copy=False)
            if "train_decode_mask" in group:
                train_decode_mask = group["train_decode_mask"][()]
                eval_decode_mask = group["eval_decode_mask"][()]
            else:
                train_decode_mask = np.full(train_rates.shape[0], True)[:, None]
                eval_decode_mask = np.full(eval_rates.shape[0], True)[:, None]
            metrics["vel R2"] = float(
                velocity_decoding(
                    train_rates,
                    group["train_behavior"][()].astype(float),
                    train_decode_mask,
                    eval_rates,
                    group["eval_behavior"][()].astype(float),
                    eval_decode_mask,
                )
            )
        if "psth" in group and "eval_cond_idx" in group:
            jitter = group["eval_jitter"][()] if "eval_jitter" in group else None
            metrics["psth R2"] = float(
                eval_psth(
                    group["psth"][()].astype(float),
                    eval_rates,
                    group["eval_cond_idx"][()],
                    jitter=jitter,
                )
            )
    return metrics


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
    train_rates_heldout: np.ndarray | None,
    train_rates_heldin: np.ndarray | None,
    target: np.ndarray,
    score: float,
    full_metrics: dict[str, float],
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
    metrics.update({key: float(value) for key, value in full_metrics.items()})
    if isinstance(config.model, MINTConfig) and config.model.train_source == "lfads":
        metrics.update(
            {
                "lfads_epochs": int(config.model.lfads_epochs),
                "lfads_batch_size": int(config.model.lfads_batch_size),
                "lfads_train_bin_size": int(config.model.lfads_train_bin_size),
                "lfads_lr": float(config.model.lfads_lr),
                "lfads_generator_dim": int(config.model.lfads_generator_dim),
                "lfads_factor_dim": int(config.model.lfads_factor_dim),
                "lfads_keep_prob": float(config.model.lfads_keep_prob),
            }
        )
    _write_json(config_path, experiment_config_to_dict(config))
    _write_json(metrics_path, metrics)
    prediction_arrays = {
        "pred_rates": eval_rates_heldout.astype(np.float32),
        "target_spikes": target.astype(np.float32),
        "pred_rates_heldin": eval_rates_heldin.astype(np.float32),
    }
    if train_rates_heldout is not None and train_rates_heldin is not None:
        prediction_arrays["train_rates_heldout"] = train_rates_heldout.astype(np.float32)
        prediction_arrays["train_rates_heldin"] = train_rates_heldin.astype(np.float32)
    np.savez_compressed(predictions_path, **prediction_arrays)
    with h5py.File(submission_path, "w") as handle:
        group = handle.create_group(dataset)
        group.create_dataset("eval_rates_heldin", data=eval_rates_heldin.astype(np.float32), compression="gzip")
        group.create_dataset("eval_rates_heldout", data=eval_rates_heldout.astype(np.float32), compression="gzip")
        if train_rates_heldout is not None and train_rates_heldin is not None:
            group.create_dataset("train_rates_heldin", data=train_rates_heldin.astype(np.float32), compression="gzip")
            group.create_dataset("train_rates_heldout", data=train_rates_heldout.astype(np.float32), compression="gzip")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "train_split", "train_source", "n_train_units", "n_eval_trials", "co_bps"],
            extrasaction="ignore",
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
                *[
                    f"- {key}: `{value:.12g}`"
                    for key, value in full_metrics.items()
                    if key != "co-bps"
                ],
                f"- Predictions: `{predictions_path.name}`",
                f"- EvalAI rates: `{submission_path.name}`",
            ]
        )
        + "\n"
    )
    return MINTNLBResult(
        run_dir=run_dir,
        co_bps=float(score),
        metrics={key: float(value) for key, value in full_metrics.items()},
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
