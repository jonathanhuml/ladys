from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from .mat_data import stack_rows, take_time_window
from .tasks import TrialData
from .utils import TORCH_DTYPE, as_tensor


TRAIN_NWB_REL = {
    "area2_bump": Path("000127/sub-Han/sub-Han_desc-train_behavior+ecephys.nwb"),
    "mc_maze": Path("000128/sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"),
    "mc_rtt": Path("000129/sub-Indy/sub-Indy_desc-train_behavior+ecephys.nwb"),
}


def default_train_nwb_path(dataset: str, nwb_dir: Path = Path("data/dandi")) -> Path:
    return Path(nwb_dir) / TRAIN_NWB_REL[dataset]


def _alignment_array(values: range) -> np.ndarray:
    return np.asarray(list(values), dtype=np.int64)


def _limit_indices(idx: np.ndarray, max_trials: Optional[int]) -> np.ndarray:
    return idx if max_trials is None else idx[:max_trials]


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


def _spikes_tensor(spikes: np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
    spikes = np.asarray(spikes)
    finite = spikes[np.isfinite(spikes)]
    if finite.size and finite.min() >= 0 and finite.max() <= 255:
        return torch.as_tensor(spikes, dtype=torch.uint8, device=device)
    return as_tensor(spikes, device)


def _to_ms(value) -> int:
    if hasattr(value, "total_seconds"):
        return int(round(value.total_seconds() * 1000.0))
    return int(round(float(value) * 1000.0))


def _field_matrix(ds, field: str, dtype=np.float64) -> np.ndarray:
    return ds.data[field].to_numpy(dtype=dtype).T


def _condition_index(cond_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cond_list = np.unique(cond_mat, axis=0)
    condition = []
    for row in cond_mat:
        condition.append(int(np.flatnonzero(np.all(cond_list == row, axis=1))[0]))
    return np.asarray(condition, dtype=np.int64), cond_list


def get_nwb_trial_data(
    settings,
    split: str,
    nwb_path: Path,
    max_trials: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> TrialData:
    from nlb_tools.nwb_interface import NWBDataset

    nwb_path = Path(nwb_path)
    ds = NWBDataset(nwb_path)
    if settings.task == "area2_bump":
        return _area2_nwb_trial_data(ds, settings, split, max_trials, device)
    if settings.task == "mc_maze":
        return _mc_maze_nwb_trial_data(ds, settings, split, max_trials, device)
    raise ValueError(f"Direct trial NWB loading is not implemented for {settings.task}.")


def _area2_nwb_trial_data(ds, settings, split, max_trials, device) -> TrialData:
    trial_info = ds.trial_info
    split_mask = trial_info["split"].isin(_split_labels(split)).to_numpy()
    good_mask = (trial_info["result"] == "R").to_numpy()
    idx = _limit_indices(np.flatnonzero(split_mask & good_mask), max_trials)

    cond_mat = np.column_stack(
        [
            trial_info.iloc[idx]["cond_dir"].to_numpy(dtype=np.float64),
            trial_info.iloc[idx]["ctr_hold_bump"].to_numpy(dtype=np.float64),
        ]
    )
    condition, cond_list = _condition_index(cond_mat)

    heldout = _field_matrix(ds, "heldout_spikes")
    spikes = _field_matrix(ds, "spikes")
    behavior_fields = [
        _field_matrix(ds, "hand_pos"),
        _field_matrix(ds, "hand_vel"),
        _field_matrix(ds, "force"),
        _field_matrix(ds, "joint_ang"),
        _field_matrix(ds, "joint_vel"),
        _field_matrix(ds, "muscle_len"),
        _field_matrix(ds, "muscle_vel"),
    ]
    alignment = _alignment_array(settings.trial_alignment)

    S, Z = [], []
    for tr in idx:
        move_idx = _to_ms(trial_info.iloc[tr]["move_onset_time"])
        time_idx = move_idx + alignment
        spikes_trial = stack_rows([take_time_window(heldout, time_idx), take_time_window(spikes, time_idx)])
        behavior = stack_rows(take_time_window(field, time_idx) for field in behavior_fields)
        if np.isnan(behavior).any():
            raise ValueError(f"area2_bump trial {tr}: behavior window contains NaNs.")
        S.append(_spikes_tensor(spikes_trial, device))
        Z.append(as_tensor(behavior, device))
    return S, Z, condition, cond_list


def _mc_maze_nwb_trial_data(ds, settings, split, max_trials, device) -> TrialData:
    trial_info = ds.trial_info
    split_mask = trial_info["split"].isin(_split_labels(split)).to_numpy()
    idx = _limit_indices(np.flatnonzero(split_mask), max_trials)

    cond_mat = np.column_stack(
        [
            trial_info.iloc[idx]["trial_type"].to_numpy(dtype=np.float64),
            trial_info.iloc[idx]["trial_version"].to_numpy(dtype=np.float64),
        ]
    )
    condition, cond_list = _condition_index(cond_mat)

    heldout = _field_matrix(ds, "heldout_spikes")
    spikes = _field_matrix(ds, "spikes")
    hand_pos = _field_matrix(ds, "hand_pos")
    hand_vel = _field_matrix(ds, "hand_vel")
    alignment = _alignment_array(settings.trial_alignment)

    S, Z = [], []
    for tr in idx:
        move_idx = _to_ms(trial_info.iloc[tr]["move_onset_time"])
        time_idx = move_idx + alignment
        spikes_trial = stack_rows([take_time_window(heldout, time_idx), take_time_window(spikes, time_idx)])
        behavior = stack_rows([take_time_window(hand_pos, time_idx), take_time_window(hand_vel, time_idx)])
        if np.isnan(behavior).any():
            raise ValueError(f"mc_maze trial {tr}: behavior window contains NaNs.")
        S.append(_spikes_tensor(spikes_trial, device))
        Z.append(as_tensor(behavior, device))
    return S, Z, condition, cond_list


def get_mc_rtt_lfads_trial_data(
    settings,
    split: str,
    nwb_path: Path,
    rates_npz: Path,
    max_trials: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> TrialData:
    from nlb_tools.nwb_interface import NWBDataset

    if split not in {"train", "trainval"}:
        raise ValueError("MC_RTT LFADS NWB training data currently supports only train or trainval splits.")

    ds = NWBDataset(Path(nwb_path))
    rates_file = np.load(Path(rates_npz))
    rates = np.asarray(rates_file["rates_1ms"], dtype=np.float64)
    if rates.ndim != 2:
        raise ValueError(f"Expected rates_1ms to have shape (time, neurons), got {rates.shape}.")

    trial_info = ds.trial_info
    labels = _split_labels(split)
    public_idx = np.flatnonzero(trial_info["split"].isin(labels).to_numpy())
    if public_idx.size == 0:
        raise ValueError(f"No MC_RTT trials found for split {split}.")
    end_ms = _to_ms(trial_info.iloc[int(public_idx[-1])]["end_time"])
    end_ms = min(end_ms, len(ds.data), rates.shape[0])

    heldout = _field_matrix(ds, "heldout_spikes")[:, :end_ms]
    spikes = _field_matrix(ds, "spikes")[:, :end_ms]
    finger_pos = _field_matrix(ds, "finger_pos")[:2, :end_ms]
    finger_vel = _field_matrix(ds, "finger_vel")[:, :end_ms]
    rates_t = rates[:end_ms].T
    valid = np.isfinite(rates_t[0]) & np.isfinite(finger_pos[0])
    if "valid_1ms" in rates_file:
        valid &= np.asarray(rates_file["valid_1ms"][:end_ms], dtype=bool)

    S, Z = [], []
    for start, stop in _contiguous_true_runs(valid):
        idx = np.arange(start, stop, dtype=np.int64)
        if idx.size == 0:
            continue
        spikes_trial = stack_rows([heldout[:, idx], spikes[:, idx]])
        behavior = stack_rows([finger_pos[:, idx], finger_vel[:, idx], rates_t[:, idx]])
        S.append(_spikes_tensor(spikes_trial, device))
        Z.append(torch.as_tensor(behavior, dtype=TORCH_DTYPE, device=device))

    if max_trials is not None:
        S = S[:max_trials]
        Z = Z[:max_trials]
    condition = np.arange(len(S), dtype=np.int64)
    return S, Z, condition, condition[:, None]


def _contiguous_true_runs(mask: Sequence[bool]) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    stops = np.flatnonzero(diff == -1)
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]
