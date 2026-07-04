"""Neural Latents Benchmark dataset wrappers and preparation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Iterable, Literal, Optional

import h5py
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor
from torch.utils.data import Dataset


NLBCoreDataset = Literal["mc_maze", "mc_rtt", "area2_bump", "dmfc_rsg"]
NLBSplit = Literal["val", "test"]
NLBBinSize = Literal[5, 20]

NLB_DATASETS: tuple[str, ...] = ("area2_bump", "mc_maze", "mc_rtt", "dmfc_rsg")
NLB_BIN_SIZES_MS: tuple[int, ...] = (5, 20)

DATASET_TO_DANDISET: dict[str, str] = {
    "area2_bump": "000127",
    "mc_maze": "000128",
    "mc_rtt": "000129",
    "dmfc_rsg": "000130",
}

DATASET_TO_TEST_NWB: dict[str, Path] = {
    "area2_bump": Path("sub-Han/sub-Han_desc-test_ecephys.nwb"),
    "mc_maze": Path("sub-Jenkins/sub-Jenkins_ses-full_desc-test_ecephys.nwb"),
    "mc_rtt": Path("sub-Indy/sub-Indy_desc-test_ecephys.nwb"),
    "dmfc_rsg": Path("sub-Haydn/sub-Haydn_desc-test_ecephys.nwb"),
}

DATASET_TO_TRAIN_NWB: dict[str, Path] = {
    "area2_bump": Path("sub-Han/sub-Han_desc-train_behavior+ecephys.nwb"),
    "mc_maze": Path("sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"),
    "mc_rtt": Path("sub-Indy/sub-Indy_desc-train_behavior+ecephys.nwb"),
    "dmfc_rsg": Path("sub-Haydn/sub-Haydn_desc-train_ecephys.nwb"),
}


class NLBDatasetConfig(BaseModel):
    """Config for an NLB co-smoothing evaluation H5."""

    model_config = ConfigDict(extra="forbid")

    name: NLBCoreDataset = "mc_maze"
    data_path: Optional[str] = None
    split: NLBSplit = "test"
    bin_size_ms: NLBBinSize = 5
    bin_size: Optional[float] = Field(
        default=None,
        description="Legacy seconds-valued bin size. Prefer bin_size_ms.",
    )
    group: Optional[str] = None
    max_trials: Optional[int] = None
    input_key: str = "eval_spikes_heldin"
    target_key: str = "eval_spikes_heldout"

    @model_validator(mode="after")
    def _sync_legacy_bin_size(self) -> "NLBDatasetConfig":
        if self.bin_size is None:
            self.bin_size = float(self.bin_size_ms) / 1000.0
            return self
        bin_ms = int(round(float(self.bin_size) * 1000.0))
        if bin_ms not in NLB_BIN_SIZES_MS:
            raise ValueError("bin_size must correspond to 5 ms or 20 ms.")
        self.bin_size_ms = bin_ms  # type: ignore[assignment]
        return self

    @property
    def resolved_data_path(self) -> Path:
        if self.data_path is not None:
            return Path(self.data_path)
        return default_nlb_h5_path(self.name, self.split, self.bin_size_ms)

    @property
    def resolved_group(self) -> str:
        if self.group is not None:
            return self.group
        return nlb_group_name(self.name, self.bin_size_ms)


@dataclass
class NLBArrays:
    """Loaded held-in input and held-out target arrays."""

    heldin_spikes: Tensor
    heldout_spikes: Tensor
    dt: float


def nlb_group_name(dataset: str, bin_size_ms: int) -> str:
    """Return the standard nlb_tools H5 group for a dataset/bin-size pair."""

    if bin_size_ms == 5:
        return dataset
    return f"{dataset}_{bin_size_ms}"


def default_nlb_h5_path(dataset: str, split: str, bin_size_ms: int) -> Path:
    """Return the repository-local LaDyS NLB H5 path."""

    return Path("data") / "real" / "nlb" / f"{dataset}_{split}_{bin_size_ms}ms.h5"


def load_nlb_h5(config: NLBDatasetConfig) -> NLBArrays:
    """Load held-in and held-out spikes from a LaDyS/NLB H5 file."""

    path = config.resolved_data_path
    if not path.exists():
        raise FileNotFoundError(
            f"NLB H5 not found: {path}. Run `ladys prepare-nlb --datasets "
            f"{config.name} --splits {config.split} --bin-sizes-ms {config.bin_size_ms}` first."
        )

    with h5py.File(path, "r") as handle:
        group = _select_h5_group(handle, config.resolved_group)
        heldin = np.asarray(group[config.input_key])
        heldout = np.asarray(group[config.target_key])

    if heldin.shape[:2] != heldout.shape[:2]:
        raise ValueError(
            f"{path}: held-in shape {heldin.shape} is incompatible with held-out {heldout.shape}."
        )

    if config.max_trials is not None:
        heldin = heldin[: config.max_trials]
        heldout = heldout[: config.max_trials]

    return NLBArrays(
        heldin_spikes=torch.from_numpy(heldin.copy()).float(),
        heldout_spikes=torch.from_numpy(heldout.copy()).float(),
        dt=float(config.bin_size_ms) / 1000.0,
    )


class NLBDataset(Dataset):
    """PyTorch Dataset for NLB held-in to held-out co-smoothing."""

    def __init__(
        self,
        config: Optional[NLBDatasetConfig] = None,
        split: Literal["train", "valid"] = "train",
        arrays: Optional[NLBArrays] = None,
    ) -> None:
        self.config = config or NLBDatasetConfig()
        self.split = split
        self.arrays = arrays or load_nlb_h5(self.config)
        self.spikes = self.arrays.heldin_spikes
        self.raw_spikes = self.arrays.heldout_spikes

    @classmethod
    def make_splits(
        cls,
        config: Optional[NLBDatasetConfig] = None,
    ) -> tuple["NLBDataset", "NLBDataset"]:
        config = config or NLBDatasetConfig()
        arrays = load_nlb_h5(config)
        return cls(config, "train", arrays), cls(config, "valid", arrays)

    def __len__(self) -> int:
        return int(self.spikes.shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "spikes": self.spikes[index],
            "heldin_spikes": self.spikes[index],
            "raw_spikes": self.raw_spikes[index],
            "heldout_spikes": self.raw_spikes[index],
            "dt": torch.tensor(self.arrays.dt, dtype=torch.float32),
        }


@dataclass(frozen=True)
class PreparedNLBFile:
    """Summary of one prepared NLB file."""

    dataset: str
    split: str
    bin_size_ms: int
    path: Path
    heldin_shape: tuple[int, ...]
    heldout_shape: tuple[int, ...]


def prepare_nlb_data(
    datasets: Iterable[str] = NLB_DATASETS,
    splits: Iterable[str] = ("test",),
    bin_sizes_ms: Iterable[int] = NLB_BIN_SIZES_MS,
    output_dir: Path | str = Path("data/real/nlb"),
    target_h5: Path | str | None = None,
    nwb_root: Path | str = Path("data/real/nlb/dandi"),
    search_roots: Iterable[Path | str] | None = None,
    download: bool = False,
    overwrite: bool = False,
    include_psth: bool = False,
    train_trial_split: str = "train",
    eval_trial_split: str | None = None,
) -> list[PreparedNLBFile]:
    """Prepare LaDyS-ready NLB H5 files from DANDI NWB and NLB target tensors."""

    output_dir = Path(output_dir)
    nwb_root = Path(nwb_root)
    target_path = _resolve_target_h5(target_h5)
    roots = _default_nwb_search_roots(nwb_root)
    if search_roots is not None:
        roots.extend(Path(p) for p in search_roots)
    roots = list(dict.fromkeys(roots))
    prepared: list[PreparedNLBFile] = []

    for dataset in datasets:
        _validate_dataset(dataset)
        for split in splits:
            if split not in {"val", "test"}:
                raise ValueError(f"Unsupported NLB split '{split}'. Expected 'val' or 'test'.")
            for bin_size_ms in bin_sizes_ms:
                if int(bin_size_ms) not in NLB_BIN_SIZES_MS:
                    raise ValueError("Only 5 ms and 20 ms NLB tensors are supported.")
                output = output_dir / f"{dataset}_{split}_{int(bin_size_ms)}ms.h5"
                if output.exists() and not overwrite:
                    prepared.append(_validate_prepared_h5(output, dataset, split, int(bin_size_ms)))
                    continue
                output.parent.mkdir(parents=True, exist_ok=True)
                if split == "test":
                    result = _prepare_test_h5(
                        dataset=dataset,
                        bin_size_ms=int(bin_size_ms),
                        output=output,
                        target_h5=target_path,
                        nwb_root=nwb_root,
                        search_roots=roots,
                        download=download,
                    )
                else:
                    result = _prepare_validation_h5(
                        dataset=dataset,
                        bin_size_ms=int(bin_size_ms),
                        output=output,
                        nwb_root=nwb_root,
                        search_roots=roots,
                        download=download,
                        include_psth=include_psth,
                        train_trial_split=train_trial_split,
                        eval_trial_split=eval_trial_split or "val",
                    )
                prepared.append(result)
    return prepared


def _select_h5_group(handle: h5py.File, group_name: str) -> h5py.Group | h5py.File:
    if group_name in handle:
        return handle[group_name]
    if "eval_spikes_heldin" in handle and "eval_spikes_heldout" in handle:
        return handle
    keys = ", ".join(handle.keys())
    raise KeyError(f"Could not find group '{group_name}' or root eval tensors. H5 keys: {keys}")


def _validate_dataset(dataset: str) -> None:
    if dataset not in NLB_DATASETS:
        known = ", ".join(NLB_DATASETS)
        raise ValueError(f"Unknown NLB dataset '{dataset}'. Expected one of: {known}.")


def _resolve_target_h5(path: Path | str | None) -> Path:
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
        Path("../nlb_tools/data/eval_data_test.h5"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find NLB public test target H5. Pass --target-h5 pointing to "
        "nlb_tools/data/eval_data_test.h5."
    )


def _default_nwb_search_roots(nwb_root: Path) -> list[Path]:
    return [
        nwb_root,
        Path("data/real/dandi"),
        Path("../mint/data/dandi"),
        Path("../STNDT/data"),
    ]


def _prepare_test_h5(
    dataset: str,
    bin_size_ms: int,
    output: Path,
    target_h5: Path,
    nwb_root: Path,
    search_roots: list[Path],
    download: bool,
) -> PreparedNLBFile:
    heldin = _build_eval_heldin(
        dataset=dataset,
        split="test",
        bin_size_ms=bin_size_ms,
        nwb_root=nwb_root,
        search_roots=search_roots,
        download=download,
    )
    group_name = nlb_group_name(dataset, bin_size_ms)
    target_dict = _read_group_dict(target_h5, group_name)
    target_dict["eval_spikes_heldin"] = heldin.astype(np.float32, copy=False)
    _write_flat_h5(output, target_dict)
    return _validate_prepared_h5(output, dataset, "test", bin_size_ms)


def _prepare_validation_h5(
    dataset: str,
    bin_size_ms: int,
    output: Path,
    nwb_root: Path,
    search_roots: list[Path],
    download: bool,
    include_psth: bool,
    train_trial_split: str,
    eval_trial_split: str,
) -> PreparedNLBFile:
    nwb_path = _resolve_nwb_path(dataset, "train", nwb_root, search_roots, download)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        input_h5 = tmp_dir / "eval_input.h5"
        target_h5 = tmp_dir / "target.h5"
        try:
            from nlb_tools.make_tensors import (
                make_eval_input_tensors,
                make_eval_target_tensors,
            )
            from nlb_tools.nwb_interface import NWBDataset
        except ImportError as exc:
            raise RuntimeError("nlb_tools is required to build NLB tensors from NWB.") from exc

        dataset_obj = NWBDataset(nwb_path)
        dataset_obj.resample(bin_size_ms)
        make_eval_input_tensors(
            dataset_obj,
            dataset_name=dataset,
            trial_split=eval_trial_split,
            save_file=True,
            save_path=str(input_h5),
        )
        make_eval_target_tensors(
            dataset_obj,
            dataset_name=dataset,
            train_trial_split=train_trial_split,
            eval_trial_split=eval_trial_split,
            include_psth=include_psth,
            save_file=True,
            save_path=str(target_h5),
        )

        data = _read_group_or_root_dict(input_h5, nlb_group_name(dataset, bin_size_ms))
        data.update(_read_group_dict(target_h5, nlb_group_name(dataset, bin_size_ms)))
        _write_flat_h5(output, data)
    return _validate_prepared_h5(output, dataset, "val", bin_size_ms)


def _build_eval_heldin(
    dataset: str,
    split: str,
    bin_size_ms: int,
    nwb_root: Path,
    search_roots: list[Path],
    download: bool,
) -> np.ndarray:
    nwb_path = _resolve_nwb_path(dataset, split, nwb_root, search_roots, download)
    with tempfile.TemporaryDirectory() as tmp:
        input_h5 = Path(tmp) / "eval_input.h5"
        try:
            from nlb_tools.make_tensors import make_eval_input_tensors
            from nlb_tools.nwb_interface import NWBDataset
        except ImportError as exc:
            raise RuntimeError("nlb_tools is required to build NLB tensors from NWB.") from exc

        dataset_obj = NWBDataset(nwb_path)
        dataset_obj.resample(bin_size_ms)
        make_eval_input_tensors(
            dataset_obj,
            dataset_name=dataset,
            trial_split=split,
            save_file=True,
            save_path=str(input_h5),
        )
        data = _read_group_or_root_dict(input_h5, nlb_group_name(dataset, bin_size_ms))
    return np.asarray(data["eval_spikes_heldin"])


def _resolve_nwb_path(
    dataset: str,
    split: str,
    nwb_root: Path,
    search_roots: list[Path],
    download: bool,
) -> Path:
    rel = DATASET_TO_TEST_NWB[dataset] if split == "test" else DATASET_TO_TRAIN_NWB[dataset]
    dandiset = DATASET_TO_DANDISET[dataset]
    candidates = []
    for root in search_roots:
        candidates.append(root / dandiset / rel)
        candidates.append(root / rel)
    for candidate in candidates:
        if candidate.exists():
            return candidate

    if download:
        _download_dandiset(dataset, nwb_root)
        downloaded = nwb_root / dandiset / rel
        if downloaded.exists():
            return downloaded
        alternate = nwb_root / rel
        if alternate.exists():
            return alternate

    formatted = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find {dataset} {split} NWB. Checked:\n  {formatted}\n"
        "Rerun with --download or pass --nwb-root/--search-root."
    )


def _download_dandiset(dataset: str, nwb_root: Path) -> None:
    dandiset = DATASET_TO_DANDISET[dataset]
    nwb_root.mkdir(parents=True, exist_ok=True)
    if shutil.which("dandi") is None:
        raise RuntimeError(
            "dandi CLI not found. Install with `pip install -e .[data]` or pass local NWB files."
        )
    subprocess.run(
        [
            "dandi",
            "download",
            f"https://dandiarchive.org/dandiset/{dandiset}",
            "-o",
            str(nwb_root),
        ],
        check=True,
    )


def _read_group_dict(path: Path, group_name: str) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        if group_name not in handle:
            keys = ", ".join(handle.keys())
            raise KeyError(f"{path} does not contain group '{group_name}'. H5 keys: {keys}")
        return {
            key: np.asarray(value)
            for key, value in handle[group_name].items()
            if isinstance(value, h5py.Dataset)
        }


def _read_group_or_root_dict(path: Path, group_name: str) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        group = handle[group_name] if group_name in handle else handle
        return {
            key: np.asarray(value)
            for key, value in group.items()
            if isinstance(value, h5py.Dataset)
        }


def _write_flat_h5(path: Path, data: dict[str, np.ndarray]) -> None:
    with h5py.File(path, "w") as handle:
        for key, value in data.items():
            handle.create_dataset(key, data=value, compression="gzip")


def _validate_prepared_h5(
    path: Path,
    dataset: str,
    split: str,
    bin_size_ms: int,
) -> PreparedNLBFile:
    with h5py.File(path, "r") as handle:
        group = _select_h5_group(handle, nlb_group_name(dataset, bin_size_ms))
        heldin_shape = tuple(group["eval_spikes_heldin"].shape)
        heldout_shape = tuple(group["eval_spikes_heldout"].shape)
    if heldin_shape[:2] != heldout_shape[:2]:
        raise ValueError(f"{path}: held-in {heldin_shape} and held-out {heldout_shape} mismatch.")
    return PreparedNLBFile(
        dataset=dataset,
        split=split,
        bin_size_ms=bin_size_ms,
        path=path,
        heldin_shape=heldin_shape,
        heldout_shape=heldout_shape,
    )
