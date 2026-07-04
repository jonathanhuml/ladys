from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from .mat_data import MatTable, MintMatFile, find_time_index, stack_rows, take_time_window
from .utils import TORCH_DTYPE, as_tensor, gauss_filt, smooth_average


TrialData = Tuple[List[torch.Tensor], List[torch.Tensor], np.ndarray, object]


def _alignment_array(values: range) -> np.ndarray:
    return np.asarray(list(values), dtype=np.int64)


def _mask_by_alignment(source: range, target: range) -> np.ndarray:
    source_arr = _alignment_array(source)
    return np.isin(source_arr, _alignment_array(target))


def _limit(items: Sequence, max_items: Optional[int]):
    return list(items) if max_items is None else list(items)[:max_items]


def _spikes_tensor(spikes: np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
    spikes = np.asarray(spikes)
    if not np.isnan(spikes).any() and np.nanmin(spikes) >= 0 and np.nanmax(spikes) <= 255:
        return torch.as_tensor(spikes, dtype=torch.uint8, device=device)
    return as_tensor(spikes, device)


def _split_labels(split) -> list[str]:
    if isinstance(split, (list, tuple, set)):
        labels = []
        for item in split:
            labels.extend(_split_labels(item))
        return list(dict.fromkeys(labels))
    if split == "train":
        return ["train"]
    if split == "val":
        return ["val"]
    if split == "test":
        return ["val"]
    if split == "trainval":
        return ["train", "val"]
    raise ValueError(f"Unsupported split: {split}")


def get_trial_data(settings, split: str, max_trials: Optional[int] = None, device: Optional[torch.device] = None) -> TrialData:
    loader = MintMatFile(settings.data_path, settings.task)
    T, TrialInfo = loader.load()
    if settings.task == "area2_bump":
        return _area2_get_trial_data(T, TrialInfo, settings, split, max_trials, device)
    if settings.task == "mc_maze":
        return _mc_maze_get_trial_data(T, TrialInfo, settings, split, max_trials, device)
    if settings.task == "mc_rtt":
        return _mc_rtt_get_trial_data(T, TrialInfo, settings, split, max_trials, device)
    raise ValueError(f"Unknown task: {settings.task}")


def _area2_get_trial_data(T, TrialInfo, settings, split, max_trials, device):
    good = TrialInfo["result"] == "R"
    TrialInfo = TrialInfo.subset_trials(good)
    cond_mat = np.column_stack([TrialInfo["cond_dir"], TrialInfo["ctr_hold_bump"].astype(np.float64)])
    cond_list = np.unique(cond_mat, axis=0)

    S, Z, condition = [], [], []
    alignment = _alignment_array(settings.trial_alignment)
    time = T["time"]
    for tr in range(TrialInfo.n_trials):
        move_idx = find_time_index(time, TrialInfo["move_onset_time"][tr])
        idx = move_idx + alignment
        spikes = stack_rows([take_time_window(T["heldout_spikes"], idx), take_time_window(T["spikes"], idx)])
        behavior = stack_rows(
            [
                take_time_window(T["hand_pos"], idx),
                take_time_window(T["hand_vel"], idx),
                take_time_window(T["force"], idx),
                take_time_window(T["joint_ang"], idx),
                take_time_window(T["joint_vel"], idx),
                take_time_window(T["muscle_len"], idx),
                take_time_window(T["muscle_vel"], idx),
            ]
        )
        if np.isnan(behavior).any():
            raise ValueError("Data boundaries exceeded.")
        cond_row = np.asarray([TrialInfo["cond_dir"][tr], float(TrialInfo["ctr_hold_bump"][tr])])
        cond = int(np.flatnonzero(np.all(cond_list == cond_row, axis=1))[0])
        S.append(_spikes_tensor(spikes, device))
        Z.append(as_tensor(behavior, device))
        condition.append(cond)

    split_mask = np.isin(TrialInfo["split"], _split_labels(split))
    idx = np.flatnonzero(split_mask)
    idx = idx[:max_trials] if max_trials is not None else idx
    return [S[i] for i in idx], [Z[i] for i in idx], np.asarray([condition[i] for i in idx]), cond_list


def _mc_maze_get_trial_data(T, TrialInfo, settings, split, max_trials, device):
    cond_mat = np.column_stack([TrialInfo["trial_type"], TrialInfo["trial_version"]])
    cond_list = np.unique(cond_mat, axis=0)

    S, Z, condition = [], [], []
    alignment = _alignment_array(settings.trial_alignment)
    time = T["time"]
    for tr in range(TrialInfo.n_trials):
        move_idx = find_time_index(time, TrialInfo["move_onset_time"][tr])
        idx = move_idx + alignment
        spikes = stack_rows([take_time_window(T["heldout_spikes"], idx), take_time_window(T["spikes"], idx)])
        behavior = stack_rows([take_time_window(T["hand_pos"], idx), take_time_window(T["hand_vel"], idx)])
        if np.isnan(behavior).any():
            raise ValueError("Data boundaries exceeded.")
        cond_row = np.asarray([TrialInfo["trial_type"][tr], TrialInfo["trial_version"][tr]])
        cond = int(np.flatnonzero(np.all(cond_list == cond_row, axis=1))[0])
        S.append(_spikes_tensor(spikes, device))
        Z.append(as_tensor(behavior, device))
        condition.append(cond)

    split_mask = np.isin(TrialInfo["split"], _split_labels(split))
    idx = np.flatnonzero(split_mask)
    idx = idx[:max_trials] if max_trials is not None else idx
    return [S[i] for i in idx], [Z[i] for i in idx], np.asarray([condition[i] for i in idx]), cond_list


def _mc_rtt_get_trial_data(T, TrialInfo, settings, split, max_trials, device):
    if split == "train":
        train_idx = np.flatnonzero(TrialInfo["split"] == "train")
        last_train = int(train_idx[-1])
        end_time = TrialInfo["end_time"][last_train]
        time_mask = T["time"] <= end_time
        T = T.subset_time(time_mask)
        TrialInfo = TrialInfo.subset_trials(np.arange(TrialInfo.n_trials) <= last_train)
        return _mc_rtt_get_continuous_data(T, max_trials, device)

    if split == "trainval":
        public_idx = np.flatnonzero(np.isin(TrialInfo["split"], ["train", "val"]))
        last_public = int(public_idx[-1])
        end_time = TrialInfo["end_time"][last_public]
        time_mask = T["time"] <= end_time
        T = T.subset_time(time_mask)
        return _mc_rtt_get_continuous_data(T, max_trials, device)

    if split == "val":
        first_val = int(np.flatnonzero(TrialInfo["split"] == "val")[0])
        start_time = TrialInfo["start_time"][first_val]
        time_mask = T["time"] >= start_time
        T = T.subset_time(time_mask)
        TrialInfo = TrialInfo.subset_trials(np.arange(TrialInfo.n_trials) >= first_val)
        return _mc_rtt_get_trialized_data(T, TrialInfo, settings, max_trials, device)

    val_idx = np.flatnonzero(TrialInfo["split"] == "val")
    first_val = int(val_idx[0])
    start_trial = first_val + 2
    start_time = TrialInfo["start_time"][start_trial]
    time_mask = T["time"] >= start_time
    T = T.subset_time(time_mask)
    TrialInfo = TrialInfo.subset_trials(np.arange(TrialInfo.n_trials) >= start_trial)
    return _mc_rtt_get_trialized_data(T, TrialInfo, settings, max_trials, device)


def _mc_rtt_get_continuous_data(T: MatTable, max_trials, device) -> TrialData:
    gap_mask = np.isnan(T["finger_pos"][0]) | np.isnan(T["autolfads_rates"][0])
    diff = np.diff(gap_mask.astype(np.int8))
    starts = np.concatenate([[0], np.flatnonzero(diff == -1) + 1])
    ends = np.concatenate([np.flatnonzero(diff == 1), [len(gap_mask) - 1]])
    S, Z = [], []
    for start, end in zip(starts, ends):
        idx = np.arange(start, end + 1)
        spikes = stack_rows([T["heldout_spikes"][:, idx], T["spikes"][:, idx]])
        behavior = stack_rows([T["finger_pos"][:2, idx], T["finger_vel"][:, idx], T["autolfads_rates"][:, idx]])
        S.append(_spikes_tensor(spikes, device))
        Z.append(as_tensor(behavior, device))
    if max_trials is not None:
        S = S[:max_trials]
        Z = Z[:max_trials]
    condition = np.arange(len(S), dtype=np.int64)
    return S, Z, condition, condition[:, None]


def _mc_rtt_get_trialized_data(T: MatTable, TrialInfo: MatTable, settings, max_trials, device) -> TrialData:
    alignment = _alignment_array(settings.trial_alignment)
    n_trials = TrialInfo.n_trials if max_trials is None else min(TrialInfo.n_trials, max_trials)
    S, Z = [], []
    for tr in range(n_trials):
        start_idx = find_time_index(T["time"], TrialInfo["start_time"][tr])
        idx = start_idx + alignment
        spikes = stack_rows([take_time_window(T["heldout_spikes"], idx), take_time_window(T["spikes"], idx)])
        behavior = stack_rows(
            [
                take_time_window(T["finger_pos"][:2], idx),
                take_time_window(T["finger_vel"], idx),
                take_time_window(T["autolfads_rates"], idx),
            ]
        )
        S.append(_spikes_tensor(spikes, device))
        Z.append(as_tensor(behavior, device))
    condition = np.arange(len(S), dtype=np.int64)
    return S, Z, condition, condition[:, None]


def preprocess_behavior(Z: Sequence[torch.Tensor], settings):
    if settings.task == "area2_bump":
        zero_idx = list(settings.trial_alignment).index(0)
        out = []
        for item in Z:
            pos = item[:2] - item[:2, zero_idx : zero_idx + 1]
            out.append(torch.cat([pos, item[2:]], dim=0))
        labels = (
            ["xpos", "ypos", "xvel", "yvel"]
            + [f"force_{i}" for i in range(1, 7)]
            + [f"joint_ang_{i}" for i in range(1, 8)]
            + [f"joint_vel_{i}" for i in range(1, 8)]
            + [f"muscle_len_{i}" for i in range(1, 40)]
            + [f"muscle_vel_{i}" for i in range(1, 40)]
        )
        return out, labels
    if settings.task == "mc_maze":
        zero_idx = list(settings.trial_alignment).index(0)
        out = []
        for item in Z:
            pos = item[:2] - item[:2, zero_idx : zero_idx + 1]
            out.append(torch.cat([pos, item[2:]], dim=0))
        return out, ["xpos", "ypos", "xvel", "yvel"]
    if settings.task == "mc_rtt":
        return [item[2:4] for item in Z], ["xvel", "yvel"]
    raise ValueError(f"Unknown task: {settings.task}")


def fit_trajectories(S, Z, condition, settings, hyperparams):
    if settings.task in {"area2_bump", "mc_maze"}:
        S_smooth = [as_tensor(gauss_filt(spikes.cpu().numpy(), hyperparams.sigma, hyperparams.Delta), spikes.device) for spikes in S]
        Z_proc, labels = preprocess_behavior(Z, settings)
        t_mask = _mask_by_alignment(settings.trial_alignment, hyperparams.trajectories_alignment)
        S_smooth = [item[:, t_mask] for item in S_smooth]
        Z_proc = [item[:, t_mask] for item in Z_proc]

        cond_list = np.unique(condition)
        grouped_x, grouped_z = [], []
        for cond in cond_list:
            trial_idx = np.flatnonzero(condition == cond)
            grouped_x.append([S_smooth[i] for i in trial_idx])
            grouped_z.append([Z_proc[i] for i in trial_idx])
        z_bar = [torch.mean(torch.stack(group, dim=2), dim=2) for group in grouped_z]
        x_bar = smooth_average(grouped_x, hyperparams, settings.Ts)
        return x_bar, z_bar, labels

    if settings.task == "mc_rtt":
        vel, labels = preprocess_behavior(Z, settings)
        rates = [item[4:] * settings.Ts * hyperparams.Delta for item in Z]
        return rates, vel, labels

    raise ValueError(f"Unknown task: {settings.task}")
