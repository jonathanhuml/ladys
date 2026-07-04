"""Dataset registry."""

from ladys.datasets.chaotic_rnn import (
    ChaoticRNNDataset,
    ChaoticRNNDatasetConfig,
    generate_chaotic_rnn_data,
)
from ladys.datasets.lorenz import LorenzDataset, LorenzDatasetConfig, generate_lorenz_data
from ladys.datasets.mc_maze import MCMazeDataset, MCMazeDatasetConfig
from ladys.datasets.nlb import (
    NLB_DATASETS,
    NLBDataset,
    NLBDatasetConfig,
    prepare_nlb_data,
)

__all__ = [
    "ChaoticRNNDataset",
    "ChaoticRNNDatasetConfig",
    "LorenzDataset",
    "LorenzDatasetConfig",
    "MCMazeDataset",
    "MCMazeDatasetConfig",
    "NLB_DATASETS",
    "NLBDataset",
    "NLBDatasetConfig",
    "generate_chaotic_rnn_data",
    "generate_lorenz_data",
    "prepare_nlb_data",
]
