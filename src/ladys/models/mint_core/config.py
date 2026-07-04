from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class Settings:
    task: str
    data_path: Path
    results_path: Path
    Ts: float = 0.001
    trial_alignment: range = range(0)
    test_alignment: range = range(0)
    CondInfo: Optional[object] = None


@dataclass
class HyperParams:
    soft_norm: float = 5.0
    min_prob: float = 1e-6
    min_lambda: float = 1.0
    min_rate: float = 0.0
    interp: int = 2
    n_candidates: int = 2
    interp_within_trajectories: bool = False
    min_k_dist: int = 1000
    causal: bool = True
    Delta: int = 20
    window_length: int = 0
    trajectories_alignment: range = range(0)
    sigma: int = 0
    n_neural_dims: Optional[int] = None
    n_cond_dims: Optional[int] = None
    n_trial_dims: Optional[int] = 1


def _generic(task: str, repo_root: Path) -> Tuple[Settings, HyperParams]:
    settings = Settings(
        task=task,
        data_path=repo_root / "data" / f"{task}.mat",
        results_path=repo_root / "results_pytorch",
    )
    hp = HyperParams()
    return settings, hp


def get_config(dataset: str, repo_root: Optional[Path] = None) -> Tuple[Settings, HyperParams]:
    repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    dataset = dataset.lower()
    settings, hp = _generic(dataset, repo_root)

    if dataset == "area2_bump":
        settings.trial_alignment = range(-700, 851)
        settings.test_alignment = range(-100, 501)
        hp.trajectories_alignment = range(-350, 751)
        hp.sigma = 25
        hp.n_neural_dims = None
        hp.n_cond_dims = None
        hp.n_trial_dims = 1
        hp.causal = True
        hp.Delta = 20
        hp.window_length = 240 if hp.causal else 560
        hp.n_candidates = 2
        hp.interp_within_trajectories = False
    elif dataset == "mc_maze":
        settings.trial_alignment = range(-800, 901)
        settings.test_alignment = range(-250, 451)
        hp.trajectories_alignment = range(-500, 701)
        hp.sigma = 30
        hp.n_neural_dims = None
        hp.n_cond_dims = 21
        hp.n_trial_dims = 1
        hp.causal = True
        hp.Delta = 20
        hp.window_length = 300 if hp.causal else 580
        hp.n_candidates = 2
        hp.interp_within_trajectories = False
    elif dataset == "mc_rtt":
        settings.trial_alignment = range(-600, 1201)
        settings.test_alignment = range(0, 600)
        hp.causal = True
        hp.Delta = 20
        hp.window_length = 480 if hp.causal else 920
        hp.n_candidates = 6
        hp.interp_within_trajectories = True
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return settings, hp
