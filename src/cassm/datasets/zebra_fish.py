from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import scipy.io as sio
import torch


def load_zebra_fish_data(
    mat_path: Union[str, Path],
    h5_path: Union[str, Path],
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, dict]:
    """Load zebra fish recordings and anatomical coordinates from local files.

    Parameters
    ----------
    mat_path:
        Path to the MATLAB metadata file containing ``data.CellXYZ_norm`` and
        ``data.stim_full``.
    h5_path:
        Path to the HDF5 time-series file containing ``CellRespZ`` and
        ``absIX``.

    Returns
    -------
    dataset:
        Fluorescence time series with shape ``(T, N)``.
    coords:
        Anatomical coordinates aligned to ``dataset`` with shape ``(N, 3)``.
    idxs:
        One-indexed neuron indices loaded from the HDF5 file.
    mat:
        Parsed MATLAB metadata dictionary.
    """
    mat_path = Path(mat_path).expanduser()
    h5_path = Path(h5_path).expanduser()

    import h5py

    mat = sio.loadmat(mat_path, simplify_cells=True)
    with h5py.File(h5_path, "r") as f:
        if "CellRespZ" not in f:
            raise KeyError(f"'CellRespZ' not found. Available keys: {list(f.keys())}")
        dset = f["CellRespZ"]
        idxs = np.array(f["absIX"], dtype=np.int32).ravel()
        cellresp_z = np.array(dset)

    dataset = torch.from_numpy(cellresp_z).float()
    coords = torch.from_numpy(mat["data"]["CellXYZ_norm"][idxs - 1].astype(np.float32))
    return dataset, coords, idxs, mat


def subsample_neurons(
    dataset: torch.Tensor,
    coords: torch.Tensor,
    n: int,
    seed: Optional[int] = None,
):
    """
    Subsample n random neurons from dataset and coords, keeping alignment.

    Args:
        dataset (torch.Tensor): [T, N] fluorescence time series.
        coords  (torch.Tensor): [N, 3] anatomical coordinates.
        n       (int): number of neurons to sample.
        seed    (int, optional): random seed for reproducibility.

    Returns:
        dataset_sub (torch.Tensor): [T, n] subset of neurons.
        coords_sub  (torch.Tensor): [n, 3] corresponding coordinates.
        idxs        (torch.Tensor): [n] indices of selected neurons.
    """
    assert (
        dataset.shape[1] == coords.shape[0]
    ), "dataset and coords must have same neuron dimension"

    if seed is not None:
        torch.manual_seed(seed)

    N = dataset.shape[1]
    idxs = torch.randperm(N)[:n]  # random neuron indices
    dataset_sub = dataset[:, idxs]
    coords_sub = coords[idxs, :]

    return dataset_sub, coords_sub, idxs


def find_change_indices(arr):
    """
    Parameters
    ----------
    arr : 1D array-like
        Stimulus vector (length T).

    Returns
    -------
    change_idxs : np.ndarray (int)
        Indices i where arr[i] != arr[i-1].
    change_labels : np.ndarray (same dtype as arr)
        Labels immediately before each change (i.e., arr[i-1]).
    """
    a = np.asarray(arr).ravel()
    if a.size < 2:
        return np.array([], dtype=int), np.array([], dtype=a.dtype)

    change_idxs = np.flatnonzero(a[1:] != a[:-1]) + 1
    change_labels = a[change_idxs - 1]
    return change_idxs, change_labels


def zebra_fish_data(
    mat_path: Union[str, Path],
    h5_path: Union[str, Path],
    neurons: int = 100,
    seed: int = 0,
):
    dataset, coords, _, mat = load_zebra_fish_data(mat_path, h5_path)
    dataset_sub, coords_sub, idxs = subsample_neurons(
        dataset, coords, n=neurons, seed=seed
    )
    find_change_indices(mat["data"]["stim_full"])

    PERIOD = 30
    T_total, n_sub = dataset_sub.shape
    n_trials = T_total // PERIOD
    T_sampled = n_trials * PERIOD

    trialized_dataset = dataset_sub[:T_sampled].view(n_trials, PERIOD, n_sub)

    return trialized_dataset, coords_sub, idxs


def _prompt_for_path(label: str) -> Path:
    path = input(f"Enter path to {label}: ").strip()
    if not path:
        raise ValueError(f"{label} path is required")
    return Path(path).expanduser()


if __name__ == "__main__":
    mat_path = _prompt_for_path("zebra fish MATLAB metadata file")
    h5_path = _prompt_for_path("zebra fish HDF5 time-series file")
    dataset, coords, idxs, _ = load_zebra_fish_data(mat_path, h5_path)
    print(f"Loaded dataset shape: {tuple(dataset.shape)}")
    print(f"Loaded coords shape: {tuple(coords.shape)}")
    print(f"Loaded {len(idxs)} neuron indices")
