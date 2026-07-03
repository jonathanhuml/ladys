from pathlib import Path
import torch
from typing import Optional

_THIS_DIR = Path(__file__).resolve().parent
_DATA_DIR = _THIS_DIR / "real_dataset_tensors"

if not _DATA_DIR.is_dir():
    raise FileNotFoundError(
        f"'real_dataset_tensors' directory not found at {_DATA_DIR}"
    )

class NeuralLatentsBenchmark:
    _FILES = {
        "mc_maze_large": {
            None:  ("mc_maze_large_train.pt",     "mc_maze_large_test.pt"),
            "val": ("val_mc_maze_large_train.pt", "val_mc_maze_large_test.pt"),
        },
        "mc_maze_medium": {
            None: ("mc_maze_medium_train.pt", "mc_maze_medium_test.pt"),
        },
        "mc_maze_small": {
            None: ("mc_maze_small_train.pt", "mc_maze_small_test.pt"),
        },
        "mc_maze": {
            None: ("mc_maze_train.pt", "mc_maze_test.pt"),
        },

        "area2_bump": {
            None: ("area2_bump_train.pt", "area2_bump_test.pt"),
        },
        "dmfc_rsg": {
            None: ("dmfc_rsg_train.pt", "dmfc_rsg_test.pt"),
            "val": ("val_dmfc_rsg_train.pt", "val_dmfc_rsg_test.pt"),
        },
        "mc_rtt": {
            None: ("mc_rtt_train.pt", "mc_rtt_test.pt"),
            "val": ("val_mc_rtt_train.pt", "val_mc_rtt_test.pt"),
            
        },
    }

    def __init__(self, dataset_name: str, phase: Optional[str] = None):
        if dataset_name not in self._FILES:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Choose from {list(self._FILES)}."
            )
        if phase not in self._FILES[dataset_name]:
            raise ValueError(
                f"Unknown phase '{phase}' for dataset '{dataset_name}'. "
                f"Choose from {list(self._FILES[dataset_name])}."
            )

        train_file, eval_file = self._FILES[dataset_name][phase]
        self.train = self._load_tensor(train_file)
        self.eval  = self._load_tensor(eval_file)

        if phase == "val":
            target_file = f"val_{dataset_name}_target.pt"
            self.target = self._load_tensor(target_file)
        else:
            self.target = None

    def _load_tensor(self, filename: str):
        path = _DATA_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Expected data file not found: {path}")
        return torch.load(path, map_location="cpu", weights_only=False)

if __name__ == "__main__":
    bench = NeuralLatentsBenchmark("mc_maze")
    print("eval held-in shape:", bench.eval["eval_spikes_heldin"].shape)
