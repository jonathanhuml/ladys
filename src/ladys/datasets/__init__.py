"""Dataset registry."""

from ladys.datasets.chaotic_rnn import (
    ChaoticRNNDataset,
    ChaoticRNNDatasetConfig,
    generate_chaotic_rnn_data,
)
from ladys.datasets.lorenz import LorenzDataset, LorenzDatasetConfig, generate_lorenz_data
from ladys.datasets.mc_maze import MCMazeDataset, MCMazeDatasetConfig

__all__ = [
    "ChaoticRNNDataset",
    "ChaoticRNNDatasetConfig",
    "LorenzDataset",
    "LorenzDatasetConfig",
    "MCMazeDataset",
    "MCMazeDatasetConfig",
    "generate_chaotic_rnn_data",
    "generate_lorenz_data",
]
