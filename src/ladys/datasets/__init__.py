"""Dataset registry."""

from ladys.datasets.chaotic_rnn import (
    ChaoticRNNDataset,
    ChaoticRNNDatasetConfig,
    generate_chaotic_rnn_data,
)
from ladys.datasets.ctd import CTDDataset, CTDDatasetConfig, load_ctd_h5
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
    "CTDDataset",
    "CTDDatasetConfig",
    "LorenzDataset",
    "LorenzDatasetConfig",
    "MCMazeDataset",
    "MCMazeDatasetConfig",
    "NLB_DATASETS",
    "NLBDataset",
    "NLBDatasetConfig",
    "generate_chaotic_rnn_data",
    "generate_lorenz_data",
    "load_ctd_h5",
    "prepare_nlb_data",
]
