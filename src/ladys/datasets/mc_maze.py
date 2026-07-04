"""NLB MC_Maze H5 dataset wrapper.

The validation H5 follows the Neural Latents Benchmark co-smoothing convention:
held-in spikes are model inputs and held-out spikes are evaluation targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import h5py
import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from torch.utils.data import Dataset


class MCMazeDatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "mc_maze"
    data_path: str = "data/real/mc_maze_val.h5"
    max_trials: Optional[int] = None
    bin_size: float = 5e-3


@dataclass
class MCMazeArrays:
    heldin_spikes: Tensor
    heldout_spikes: Tensor
    dt: float


def load_mc_maze_h5(config: MCMazeDatasetConfig) -> MCMazeArrays:
    """Load held-in and held-out spikes from an NLB-style H5 file."""

    path = Path(config.data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"MC_Maze H5 not found: {path}. Run scripts/prepare_mc_maze_data.py first."
        )

    with h5py.File(path, "r") as handle:
        group = handle["mc_maze"] if "mc_maze" in handle else handle
        heldin = group["eval_spikes_heldin"][:]
        heldout = group["eval_spikes_heldout"][:]

    if config.max_trials is not None:
        heldin = heldin[: config.max_trials]
        heldout = heldout[: config.max_trials]

    return MCMazeArrays(
        heldin_spikes=torch.from_numpy(heldin.copy()).float(),
        heldout_spikes=torch.from_numpy(heldout.copy()).float(),
        dt=float(config.bin_size),
    )


class MCMazeDataset(Dataset):
    """PyTorch Dataset for NLB MC_Maze co-smoothing evaluation."""

    def __init__(
        self,
        config: Optional[MCMazeDatasetConfig] = None,
        split: Literal["train", "valid"] = "train",
        arrays: Optional[MCMazeArrays] = None,
    ) -> None:
        self.config = config or MCMazeDatasetConfig()
        self.split = split
        self.arrays = arrays or load_mc_maze_h5(self.config)
        self.spikes = self.arrays.heldin_spikes
        self.raw_spikes = self.arrays.heldout_spikes

    @classmethod
    def make_splits(
        cls,
        config: Optional[MCMazeDatasetConfig] = None,
    ) -> tuple["MCMazeDataset", "MCMazeDataset"]:
        config = config or MCMazeDatasetConfig()
        arrays = load_mc_maze_h5(config)
        return cls(config, "train", arrays), cls(config, "valid", arrays)

    def __len__(self) -> int:
        return int(self.spikes.shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "spikes": self.spikes[index],
            "raw_spikes": self.raw_spikes[index],
            "heldout_spikes": self.raw_spikes[index],
            "dt": torch.tensor(self.arrays.dt, dtype=torch.float32),
        }
