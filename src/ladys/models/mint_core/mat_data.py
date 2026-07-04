from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import h5py
import numpy as np


DATASET_FIELDS: Mapping[str, Mapping[str, Mapping[str, str]]] = {
    "mc_maze": {
        "T": {
            "time": "d",
            "hand_pos": "e",
            "hand_vel": "f",
            "spikes": "g",
            "heldout_spikes": "h",
        },
        "TrialInfo": {
            "start_time": "B",
            "move_onset_time": "C",
            "end_time": "D",
            "trial_type": "E",
            "trial_version": "F",
            "maze_id": "G",
            "success": "H",
            "split": "I",
        },
    },
    "mc_rtt": {
        "T": {
            "time": "d",
            "spikes": "e",
            "heldout_spikes": "f",
            "finger_pos": "g",
            "finger_vel": "h",
            "target_pos": "i",
            "autolfads_rates": "j",
        },
        "TrialInfo": {
            "start_time": "H",
            "end_time": "I",
            "split": "J",
        },
    },
    "area2_bump": {
        "T": {
            "time": "d",
            "spikes": "e",
            "heldout_spikes": "f",
            "hand_pos": "g",
            "hand_vel": "h",
            "force": "i",
            "joint_ang": "j",
            "joint_vel": "k",
            "muscle_len": "l",
            "muscle_vel": "m",
        },
        "TrialInfo": {
            "start_time": "Q",
            "end_time": "R",
            "result": "S",
            "ctr_hold_bump": "do",
            "bump_dir": "eo",
            "target_dir": "fo",
            "move_onset_time": "go",
            "cond_dir": "ho",
            "split": "io",
        },
    },
}


def _decode_char(dataset: h5py.Dataset) -> str:
    arr = np.asarray(dataset[()])
    return "".join(chr(int(x)) for x in arr.ravel() if int(x) != 0)


def _decode_cellstr(file: h5py.File, dataset: h5py.Dataset) -> np.ndarray:
    values: List[str] = []
    for ref in np.asarray(dataset[()]).ravel():
        values.append(_decode_char(file[ref]))
    return np.asarray(values, dtype=object)


@dataclass
class MatTable:
    fields: Dict[str, np.ndarray]

    def __getitem__(self, key: str) -> np.ndarray:
        return self.fields[key]

    def subset_time(self, mask: np.ndarray) -> "MatTable":
        out = dict(self.fields)
        for key, value in out.items():
            if key == "time":
                out[key] = value[mask]
            elif value.ndim == 2 and value.shape[1] == mask.shape[0]:
                out[key] = value[:, mask]
        return MatTable(out)

    def subset_trials(self, mask: np.ndarray) -> "MatTable":
        out = {}
        for key, value in self.fields.items():
            out[key] = value[mask]
        return MatTable(out)

    @property
    def n_time(self) -> int:
        return int(self.fields["time"].shape[0])

    @property
    def n_trials(self) -> int:
        first = next(iter(self.fields.values()))
        return int(first.shape[0])


class MintMatFile:
    def __init__(self, path: Path, dataset: str):
        self.path = Path(path)
        self.dataset = dataset

    def load(self) -> tuple[MatTable, MatTable]:
        if self.dataset not in DATASET_FIELDS:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        with h5py.File(self.path, "r") as file:
            t_fields = self._load_group(file, DATASET_FIELDS[self.dataset]["T"])
            trial_fields = self._load_group(file, DATASET_FIELDS[self.dataset]["TrialInfo"])
        return MatTable(t_fields), MatTable(trial_fields)

    @staticmethod
    def _load_group(file: h5py.File, mapping: Mapping[str, str]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        refs = file["#refs#"]
        for field, ref_name in mapping.items():
            dataset = refs[ref_name]
            matlab_class = dataset.attrs.get("MATLAB_class")
            if matlab_class == b"cell":
                out[field] = _decode_cellstr(file, dataset)
            else:
                arr = np.asarray(dataset[()])
                if arr.ndim == 2 and arr.shape[0] == 1:
                    arr = arr.ravel()
                out[field] = arr
        return out


def find_time_index(time: np.ndarray, value: float) -> int:
    idx = np.flatnonzero(time == value)
    if idx.size == 0:
        idx = np.flatnonzero(np.isclose(time, value, rtol=0.0, atol=1e-12))
    if idx.size == 0:
        raise ValueError(f"Could not find time value {value}")
    return int(idx[0])


def take_time_window(matrix: np.ndarray, indices: Sequence[int], pad_value: float = np.nan) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    valid = (indices >= 0) & (indices < matrix.shape[1])
    out = np.full((matrix.shape[0], len(indices)), pad_value, dtype=np.float64)
    out[:, valid] = matrix[:, indices[valid]]
    return out


def stack_rows(parts: Iterable[np.ndarray]) -> np.ndarray:
    return np.vstack([np.asarray(part, dtype=np.float64) for part in parts])
