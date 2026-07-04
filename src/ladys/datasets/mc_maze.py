"""Backward-compatible MC_Maze aliases for the generic NLB dataset."""

from ladys.datasets.nlb import NLBArrays as MCMazeArrays
from ladys.datasets.nlb import NLBDataset as MCMazeDataset
from ladys.datasets.nlb import NLBDatasetConfig as MCMazeDatasetConfig
from ladys.datasets.nlb import load_nlb_h5 as load_mc_maze_h5

__all__ = [
    "MCMazeArrays",
    "MCMazeDataset",
    "MCMazeDatasetConfig",
    "load_mc_maze_h5",
]
