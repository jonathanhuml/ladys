from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .config import get_config
from .core import MINT
from .tasks import get_trial_data, preprocess_behavior
from .utils import bin_data_np


def compute_r2(behavior, behavior_estimate, eval_bin_size: int = 5) -> np.ndarray:
    z = [bin_data_np(item, eval_bin_size, "mean") for item in behavior]
    z_hat = [bin_data_np(item, eval_bin_size, "mean") for item in behavior_estimate]
    z_cat = np.concatenate(z, axis=1)
    z_hat_cat = np.concatenate(z_hat, axis=1)
    nan_mask = np.any(np.isnan(z_hat_cat), axis=0)
    z_cat = z_cat[:, ~nan_mask]
    z_hat_cat = z_hat_cat[:, ~nan_mask]
    ss_res = np.sum((z_cat - z_hat_cat) ** 2, axis=1)
    ss_tot = np.sum((z_cat - np.mean(z_cat, axis=1, keepdims=True)) ** 2, axis=1)
    return 1.0 - ss_res / ss_tot


def _object_array(items):
    arr = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        arr[i] = item
    return arr


def _concat(items):
    return np.concatenate(items, axis=1)


def save_npz(path: Path, behavior, behavior_estimate, neural_state_estimate, r2, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        behavior=_object_array(behavior),
        behavior_estimate=_object_array(behavior_estimate),
        neural_state_estimate=_object_array(neural_state_estimate),
        behavior_concat=_concat(behavior),
        behavior_estimate_concat=_concat(behavior_estimate),
        neural_state_estimate_concat=_concat(neural_state_estimate),
        R2=r2,
        behavior_labels=np.asarray(labels, dtype=object),
    )


def run_decoder(
    dataset: str,
    output: Optional[Path] = None,
    device: str = "cpu",
    max_train_trials: Optional[int] = None,
    max_test_trials: Optional[int] = None,
):
    torch_device = torch.device(device)
    settings, hyperparams = get_config(dataset)
    train_S, train_Z, condition, cond_info = get_trial_data(settings, "train", max_train_trials, torch_device)
    settings.CondInfo = cond_info
    model = MINT(settings, hyperparams, torch_device).fit(train_S, train_Z, condition)

    test_S, test_Z, _, _ = get_trial_data(settings, "test", max_test_trials, torch_device)
    test_Z, _ = preprocess_behavior(test_Z, settings)

    test_buffer = [-hyperparams.window_length + 1, 0]
    if not hyperparams.causal:
        adjustment = round((hyperparams.window_length + hyperparams.Delta) / 2)
        test_buffer = [test_buffer[0] + adjustment, test_buffer[1] + adjustment]

    trial_alignment = np.asarray(list(settings.trial_alignment), dtype=np.int64)
    test_alignment = np.asarray(list(settings.test_alignment), dtype=np.int64)
    buffered_alignment = np.arange(settings.test_alignment.start + test_buffer[0], settings.test_alignment.stop - 1 + test_buffer[1] + 1)
    t_mask = np.isin(trial_alignment, buffered_alignment)
    test_S = [item[:, t_mask] for item in test_S]
    test_Z = [item[:, t_mask] for item in test_Z]

    x_hat, z_hat = model.predict(test_S)
    not_buff_mask = np.isin(buffered_alignment, test_alignment)
    behavior = [item[:, not_buff_mask].cpu().numpy() for item in test_Z]
    behavior_estimate = [item[:, not_buff_mask].cpu().numpy() for item in z_hat]
    neural_state_estimate = [(item[:, not_buff_mask] / (model.Delta * model.Ts)).cpu().numpy() for item in x_hat]
    r2 = compute_r2(behavior, behavior_estimate)

    if output is None:
        output = settings.results_path / f"{settings.task}_decode.npz"
    save_npz(output, behavior, behavior_estimate, neural_state_estimate, r2, model.behavior_labels)
    return {
        "model": model,
        "behavior": behavior,
        "behavior_estimate": behavior_estimate,
        "neural_state_estimate": neural_state_estimate,
        "R2": r2,
        "output": output,
    }
