"""MINT as a native LaDyS model.

This file intentionally keeps the PyTorch MINT implementation self-contained
instead of importing a copied helper package. MINT is an inference-only library
method: fitting builds condition/trajectory libraries, and prediction runs the
Poisson likelihood recursion plus interpolation against those libraries.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from pydantic import Field
from torch import Tensor

from ladys.models.base import BaseDynamicsModel, BaseModelConfig, OptimizationConfig
from ladys.types import LossOutput, ModelOutput


TORCH_DTYPE = torch.float64

HELDOUT_COUNTS = {
    "area2_bump": 16,
    "mc_maze": 45,
    "mc_rtt": 32,
}

TRAIN_NWB_REL = {
    "area2_bump": Path("000127/sub-Han/sub-Han_desc-train_behavior+ecephys.nwb"),
    "mc_maze": Path("000128/sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"),
    "mc_rtt": Path("000129/sub-Indy/sub-Indy_desc-train_behavior+ecephys.nwb"),
}


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


@dataclass
class InterpOptions:
    max_iters: int = 10
    step_tol: float = 0.01


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
        return MatTable({key: value[mask] for key, value in self.fields.items()})

    @property
    def n_trials(self) -> int:
        first = next(iter(self.fields.values()))
        return int(first.shape[0])


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


@BaseModelConfig.register
class MINTConfig(BaseModelConfig):
    """Config for the MINT NLB co-smoothing port."""

    name: Literal["mint"] = "mint"
    objective: str = "mint_likelihood_recursion"
    dataset: Literal["area2_bump", "mc_maze", "mc_rtt"] = "mc_maze"
    train_source: Literal["mat", "nwb"] = "nwb"
    train_split: Literal["auto", "train", "trainval"] = "trainval"
    nlb_neural_state_defaults: bool = True
    nwb_root: str = "data/real/nlb/dandi"
    mat_data_root: str = "data/mint"
    target_h5: Optional[str] = None
    eval_bin_size_ms: int = 5
    n_candidates: Optional[int] = None
    window_length: Optional[int] = None
    delta: Optional[int] = None
    sigma: Optional[int] = None
    min_rate: Optional[float] = None
    causal: Optional[bool] = None
    optimization: OptimizationConfig = Field(
        default_factory=lambda: OptimizationConfig(name="inference_only")
    )

    def build(self, n_neurons: int, n_time: int) -> "MINT":
        del n_neurons, n_time
        return MINT(self)


class MINT(BaseDynamicsModel):
    """MINT library model following the LaDyS ``BaseDynamicsModel`` API."""

    def __init__(self, config: MINTConfig) -> None:
        super().__init__()
        self.config = config
        self.objective = config.objective
        self.settings, self.hyperparams = get_mint_config(config.dataset)
        self.settings.data_path = Path(config.mat_data_root) / f"{config.dataset}.mat"

        self.Ts = self.settings.Ts
        self.Delta = self.hyperparams.Delta
        self.dt = self.Delta * self.Ts
        self.window_length = self.hyperparams.window_length
        self.tau_prime = round(self.window_length / self.Delta) - 1
        self.causal = self.hyperparams.causal
        self.interp = self.hyperparams.interp

        self.min_prob = self.hyperparams.min_prob
        self.min_lambda = self.hyperparams.min_lambda
        self.n_rates = 2000
        self.min_rate = 0.0
        self.max_spikes = 0
        self.InterpOptions = InterpOptions()

        self.Omega_plus: List[Tensor] = []
        self.Phi_plus: List[Tensor] = []
        self.behavior_labels: List[str] = []
        self.V: Optional[Tensor] = None
        self.first_idx0: Optional[Tensor] = None
        self.last_idx0: Optional[Tensor] = None
        self.first_tau_prime_idx0: Optional[Tensor] = None
        self.shifted_idx1: Optional[Tensor] = None
        self.shifted_idx2: Optional[Tensor] = None

        self.register_buffer("_device_anchor", torch.empty(0))
        self._refresh_runtime_params()

    def _refresh_runtime_params(self) -> None:
        """Refresh derived tensors after config/hyperparameter overrides."""

        self.Ts = self.settings.Ts
        self.Delta = self.hyperparams.Delta
        self.dt = self.Delta * self.Ts
        self.window_length = self.hyperparams.window_length
        self.tau_prime = round(self.window_length / self.Delta) - 1
        self.causal = self.hyperparams.causal
        self.interp = self.hyperparams.interp
        self.lambda_range = (
            torch.as_tensor([self.min_lambda, 500.0], dtype=TORCH_DTYPE, device=self.device)
            * self.dt
        )
        self.rates = torch.linspace(
            float(self.lambda_range[0]),
            float(self.lambda_range[1]),
            self.n_rates,
            dtype=TORCH_DTYPE,
            device=self.device,
        )
        self.max_spikes = round(self.dt * 1000)
        self.L = self._build_likelihood_table()
        self.min_rate = self.hyperparams.min_rate * self.dt

    def _build_likelihood_table(self) -> Tensor:
        counts = torch.arange(self.max_spikes + 1, dtype=TORCH_DTYPE, device=self.device).reshape(1, -1)
        rates = self.rates.reshape(-1, 1)
        logp = counts * torch.log(rates) - rates - torch.lgamma(counts + 1.0)
        p = torch.exp(logp)
        invalid = p <= self.min_prob
        p_nan = p.masked_fill(invalid, float("nan"))
        nan_count = invalid.sum(dim=1, keepdim=True).to(TORCH_DTYPE)
        row_sum = torch.nansum(p_nan, dim=1, keepdim=True)
        scale = (1.0 - self.min_prob * nan_count) / row_sum
        p_scaled = p_nan * scale
        p_scaled = torch.where(torch.isnan(p_scaled), torch.full_like(p_scaled, self.min_prob), p_scaled)
        return torch.log(p_scaled)

    def fit_library(self, S: Sequence[Tensor], Z: Sequence[Tensor], condition: Sequence[int]) -> "MINT":
        """Build MINT trajectory libraries from spike and behavior/rate trials."""

        self._refresh_runtime_params()
        self.Omega_plus, self.Phi_plus, self.behavior_labels = fit_trajectories(
            S, Z, condition, self.settings, self.hyperparams
        )
        lambdas = [bin_data(omega, self.Delta, "mean") for omega in self.Omega_plus]
        v_cells = [get_rate_indices(lam, self.lambda_range, self.n_rates) for lam in lambdas]

        lengths = [int(v.shape[1]) for v in v_cells]
        starts = []
        total = 0
        for length in lengths:
            starts.append(total)
            total += length
        self.first_idx0 = torch.as_tensor(starts, dtype=torch.long, device=self.device)
        self.last_idx0 = torch.as_tensor(
            [start + length - 1 for start, length in zip(starts, lengths)],
            dtype=torch.long,
            device=self.device,
        )
        self.first_tau_prime_idx0 = torch.cat(
            [start + torch.arange(self.tau_prime, dtype=torch.long, device=self.device) for start in self.first_idx0]
        ).sort().values
        self.V = torch.cat([v.T for v in v_cells], dim=0).to(torch.long)
        self._build_shifted_indices()
        return self

    def _build_shifted_indices(self) -> None:
        idx1, idx2 = [], []
        assert self.first_idx0 is not None and self.last_idx0 is not None
        for start, last in zip(self.first_idx0.tolist(), self.last_idx0.tolist()):
            if last - self.tau_prime - 1 >= start:
                idx1.append(torch.arange(start, last - self.tau_prime, dtype=torch.long, device=self.device))
                idx2.append(torch.arange(start + self.tau_prime + 1, last + 1, dtype=torch.long, device=self.device))
        if idx1:
            self.shifted_idx1 = torch.cat(idx1)
            self.shifted_idx2 = torch.cat(idx2)
        else:
            self.shifted_idx1 = torch.empty(0, dtype=torch.long, device=self.device)
            self.shifted_idx2 = torch.empty(0, dtype=torch.long, device=self.device)

    def forward(self, x: Tensor) -> ModelOutput:
        if x.ndim != 3:
            raise ValueError("MINT expects batched spikes with shape (batch, time, neurons).")
        spikes = [trial.T.contiguous() for trial in x]
        rates, _ = self.predict_spike_trials(spikes)
        return ModelOutput(rates=torch.stack([item.T for item in rates], dim=0))

    def loss(
        self,
        batch: Tensor | dict[str, Tensor],
        output: ModelOutput,
        epoch: int = 0,
    ) -> LossOutput:
        del batch, output, epoch
        return LossOutput(
            total=torch.zeros((), dtype=torch.float32, device=self.device),
            objective=self.objective,
        )

    def predict_spike_trials(
        self,
        S: Sequence[Tensor],
        return_aux: bool = False,
        likelihood_neuron_mask: Optional[Tensor] = None,
    ):
        """Predict neural states for trials shaped ``(neurons, time)``."""

        if self.V is None:
            raise RuntimeError("MINT must be fit with fit_library() before predict().")

        S_bar = []
        for spikes in S:
            binned = bin_data(spikes, self.Delta, "sum")
            binned = torch.nan_to_num(binned, nan=0.0, posinf=float(self.max_spikes), neginf=0.0)
            S_bar.append(torch.clamp(binned, 0, self.max_spikes).to(torch.uint8))

        n_trials = len(S)
        X_hat, Z_hat, C_hat, K_hat, Alpha_hat = [], [], [], [], []
        n_early_samples = self.Delta * (self.tau_prime + 1) - 1
        if likelihood_neuron_mask is not None:
            likelihood_neuron_mask = likelihood_neuron_mask.to(device=self.device, dtype=torch.bool)

        for tr in range(n_trials):
            T = S[tr].shape[1]
            T_prime = S_bar[tr].shape[1]
            Q = torch.zeros(self.V.shape[0], dtype=TORCH_DTYPE, device=self.device)
            x_hat = torch.zeros((self.Omega_plus[0].shape[0], T), dtype=TORCH_DTYPE, device=self.device)
            z_hat = torch.zeros((self.Phi_plus[0].shape[0], T), dtype=TORCH_DTYPE, device=self.device)
            if self.interp == 2:
                c_hat = torch.zeros((2, T), dtype=TORCH_DTYPE, device=self.device)
                k_hat = torch.zeros((4, T), dtype=TORCH_DTYPE, device=self.device)
                alpha_hat = torch.zeros((3, T), dtype=TORCH_DTYPE, device=self.device)
            else:
                c_hat = torch.zeros((1, T), dtype=TORCH_DTYPE, device=self.device)
                k_hat = torch.zeros((1, T), dtype=TORCH_DTYPE, device=self.device)
                alpha_hat = torch.zeros((1, T), dtype=TORCH_DTYPE, device=self.device)

            for t0 in range(T_prime):
                t_prime_one = t0 + 1
                s_new = S_bar[tr][:, t0]
                if t_prime_one > self.tau_prime + 1:
                    s_old = S_bar[tr][:, t0 - self.tau_prime - 1]
                else:
                    s_old = torch.zeros_like(s_new)
                Q = self._recursion(Q, s_new, s_old, t_prime_one, likelihood_neuron_mask)

                if t_prime_one > self.tau_prime:
                    t_idx0, f = get_time_indices(t_prime_one, T_prime, T, self.Delta, self.tau_prime, self.causal)
                    t_idx0 = t_idx0.to(self.device)
                    s_curr = S_bar[tr][:, t0 - self.tau_prime : t0 + 1].to(TORCH_DTYPE)
                    x, z, c, k, a = self._estimate_states(Q, s_curr, f, likelihood_neuron_mask)
                    x_hat[:, t_idx0] = x
                    z_hat[:, t_idx0] = z
                    c_hat[:, t_idx0] = c
                    k_hat[:, t_idx0] = k
                    alpha_hat[:, t_idx0] = a

            x_hat = torch.clamp(x_hat, min=self.min_rate)
            if self.causal:
                x_hat[:, :n_early_samples] = float("nan")
                z_hat[:, :n_early_samples] = float("nan")

            X_hat.append(x_hat)
            Z_hat.append(z_hat)
            C_hat.append(c_hat)
            K_hat.append(k_hat)
            Alpha_hat.append(alpha_hat)
            print(f"Completed trial {tr + 1}")

        if return_aux:
            return X_hat, Z_hat, C_hat, K_hat, Alpha_hat
        return X_hat, Z_hat

    def _recursion(
        self,
        Q: Tensor,
        s_new: Tensor,
        s_old: Tensor,
        t_prime_one: int,
        likelihood_neuron_mask: Optional[Tensor] = None,
    ) -> Tensor:
        assert self.V is not None and self.first_idx0 is not None
        assert self.shifted_idx1 is not None and self.shifted_idx2 is not None
        updated = torch.empty_like(Q)
        updated[0] = 0.0
        updated[1:] = Q[:-1]
        updated[self.first_idx0] = 0.0
        Q = updated

        if likelihood_neuron_mask is not None:
            V = self.V[:, likelihood_neuron_mask]
            s_new = s_new[likelihood_neuron_mask]
            s_old = s_old[likelihood_neuron_mask]
        else:
            V = self.V

        s_new = torch.clamp(s_new.to(torch.long), 0, self.max_spikes)
        gathered = self.L[V, s_new.reshape(1, -1).expand_as(V)]
        Q = Q + gathered.sum(dim=1)
        if t_prime_one > self.tau_prime + 1:
            s_old = torch.clamp(s_old.to(torch.long), 0, self.max_spikes)
            old_v = V[self.shifted_idx1]
            old_gathered = self.L[old_v, s_old.reshape(1, -1).expand_as(old_v)]
            Q[self.shifted_idx2] = Q[self.shifted_idx2] - old_gathered.sum(dim=1)
        return Q

    def _estimate_states(
        self,
        Q: Tensor,
        S_curr: Tensor,
        f: Callable[[int], Tensor],
        likelihood_neuron_mask: Optional[Tensor] = None,
    ):
        K_lengths = [phi.shape[1] for phi in self.Phi_plus]
        if self.interp == 0:
            c0, k_hats = self._maximum_likelihood(Q, restricted_conds=[])
            k_idx = get_state_indices(k_hats, f, K_lengths[c0]).to(self.device)
            x = self.Omega_plus[c0][:, k_idx[0]]
            z = self.Phi_plus[c0][:, k_idx[0]]
            return (
                x,
                z,
                torch.full((1, z.shape[1]), c0 + 1, dtype=TORCH_DTYPE, device=self.device),
                (k_idx[0] + 1).reshape(1, -1).to(TORCH_DTYPE),
                torch.full((1, z.shape[1]), float("nan"), dtype=TORCH_DTYPE, device=self.device),
            )

        if self.interp == 1:
            c0, k_hats = self._maximum_likelihood(Q, restricted_conds=[])
            k_idx = get_state_indices(k_hats, f, K_lengths[c0]).to(self.device)
            x, z, _, alpha = self._interp_adjacent_states(S_curr, c0, k_hats, k_idx, likelihood_neuron_mask)
            return (
                x,
                z,
                torch.full((1, z.shape[1]), float("nan"), dtype=TORCH_DTYPE, device=self.device),
                torch.full((1, z.shape[1]), float("nan"), dtype=TORCH_DTYPE, device=self.device),
                torch.full((1, z.shape[1]), alpha, dtype=TORCH_DTYPE, device=self.device),
            )

        candidates = []
        states_to_exclude = []
        conds_to_exclude: List[int] = []
        min_k_prime_dist = self.hyperparams.min_k_dist / self.Delta
        for _ in range(self.hyperparams.n_candidates):
            if self.hyperparams.interp_within_trajectories:
                c0, k_hats = self._maximum_likelihood(
                    Q, states_to_exclude=states_to_exclude, min_k_prime_dist=min_k_prime_dist
                )
                states_to_exclude.append((c0, k_hats[0]))
            else:
                c0, k_hats = self._maximum_likelihood(Q, restricted_conds=conds_to_exclude)
                conds_to_exclude.append(c0)
            k_idx = get_state_indices(k_hats, f, K_lengths[c0]).to(self.device)
            candidates.append(
                (*self._interp_adjacent_states(S_curr, c0, k_hats, k_idx, likelihood_neuron_mask), c0, k_idx)
            )

        interps = []
        for a, b in combinations(range(len(candidates)), 2):
            x_a, z_a, lam_a, alpha_a, c_a, k_a = candidates[a]
            x_b, z_b, lam_b, alpha_b, c_b, k_b = candidates[b]
            beta = fit_poisson_interp(
                _masked_rows(S_curr, likelihood_neuron_mask),
                _masked_rows(lam_a, likelihood_neuron_mask),
                _masked_rows(lam_b, likelihood_neuron_mask),
                self.InterpOptions,
                0.0,
            )
            lam = (1.0 - beta) * lam_a + beta * lam_b
            x = (1.0 - beta) * x_a + beta * x_b
            z = (1.0 - beta) * z_a + beta * z_b
            interps.append((x, z, lam, beta, alpha_a, alpha_b, c_a, c_b, k_a, k_b))

        x, z, _, beta, alpha_a, alpha_b, c_a, c_b, k_a, k_b = use_best_interp(
            S_curr, interps, likelihood_neuron_mask
        )
        return (
            x,
            z,
            torch.as_tensor([c_a + 1, c_b + 1], dtype=TORCH_DTYPE, device=self.device).reshape(2, 1).expand(2, z.shape[1]),
            torch.cat([(k_a + 1).to(TORCH_DTYPE), (k_b + 1).to(TORCH_DTYPE)], dim=0),
            torch.as_tensor([beta, alpha_a, alpha_b], dtype=TORCH_DTYPE, device=self.device).reshape(3, 1).expand(3, z.shape[1]),
        )

    def _maximum_likelihood(
        self,
        Q: Tensor,
        restricted_conds: Optional[Sequence[int]] = None,
        states_to_exclude: Optional[Sequence[Tuple[int, int]]] = None,
        min_k_prime_dist: Optional[float] = None,
    ) -> Tuple[int, List[int]]:
        assert self.first_idx0 is not None and self.first_tau_prime_idx0 is not None
        q = Q.clone()
        q[self.first_tau_prime_idx0] = float("nan")
        lengths = (torch.cat([self.first_idx0[1:], torch.as_tensor([len(Q)], device=self.device)]) - self.first_idx0).tolist()
        if restricted_conds:
            for c0 in restricted_conds:
                q[self.first_idx0[c0] : self.first_idx0[c0] + lengths[c0]] = float("nan")
        if states_to_exclude:
            assert min_k_prime_dist is not None
            for c0, k_one in states_to_exclude:
                start = int(self.first_idx0[c0])
                center = start + k_one - 1
                exclude_start = max(int(center - min_k_prime_dist), start)
                exclude_end = min(int(center + min_k_prime_dist), start + lengths[c0] - 1)
                q[exclude_start : exclude_end + 1] = float("nan")

        idx0 = int(torch.argmax(torch.nan_to_num(q, nan=-torch.inf)).item())
        c0, k1 = ind2ck(idx0, self.first_idx0)
        q_c = q[self.first_idx0[c0] : self.first_idx0[c0] + lengths[c0]]
        if k1 > self.tau_prime + 1 and k1 < lengths[c0]:
            if q_c[k1 - 2] > q_c[k1]:
                k2 = k1 - 1
            else:
                k2 = k1 + 1
        elif k1 > self.tau_prime + 1:
            k2 = k1 - 1
        else:
            k2 = k1 + 1
        return c0, [k1, k2]

    def _interp_adjacent_states(
        self,
        S_curr: Tensor,
        c0: int,
        k_hats: Sequence[int],
        k_idx: Tensor,
        likelihood_neuron_mask: Optional[Tensor] = None,
    ):
        assert self.first_idx0 is not None and self.V is not None
        offsets = torch.arange(-self.tau_prime, 1, dtype=torch.long, device=self.device)
        idx1 = ck2ind(c0, torch.as_tensor(k_hats[0], device=self.device) + offsets, self.first_idx0)
        idx2 = ck2ind(c0, torch.as_tensor(k_hats[1], device=self.device) + offsets, self.first_idx0)
        lambda1 = self.rates[self.V[idx1]].T
        lambda2 = self.rates[self.V[idx2]].T
        alpha = fit_poisson_interp(
            _masked_rows(S_curr, likelihood_neuron_mask),
            _masked_rows(lambda1, likelihood_neuron_mask),
            _masked_rows(lambda2, likelihood_neuron_mask),
            self.InterpOptions,
            0.0,
        )
        lam = (1.0 - alpha) * lambda1 + alpha * lambda2
        x = (1.0 - alpha) * self.Omega_plus[c0][:, k_idx[0]] + alpha * self.Omega_plus[c0][:, k_idx[1]]
        z = (1.0 - alpha) * self.Phi_plus[c0][:, k_idx[0]] + alpha * self.Phi_plus[c0][:, k_idx[1]]
        return x, z, lam, alpha


def get_mint_config(dataset: str) -> Tuple[Settings, HyperParams]:
    settings = Settings(
        task=dataset,
        data_path=Path("data") / f"{dataset}.mat",
        results_path=Path("results"),
    )
    hp = HyperParams()
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
        hp.window_length = 240
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
        hp.window_length = 300
        hp.n_candidates = 2
        hp.interp_within_trajectories = False
    elif dataset == "mc_rtt":
        settings.trial_alignment = range(-600, 1201)
        settings.test_alignment = range(0, 600)
        hp.causal = True
        hp.Delta = 20
        hp.window_length = 480
        hp.n_candidates = 6
        hp.interp_within_trajectories = True
    else:
        raise ValueError(f"Unknown MINT dataset: {dataset}")
    return settings, hp


def as_tensor(array: np.ndarray, device: Optional[torch.device] = None) -> Tensor:
    return torch.as_tensor(array, dtype=TORCH_DTYPE, device=device)


def bin_data(data: Tensor, bin_size: int, method: str) -> Tensor:
    data = data.to(TORCH_DTYPE)
    n_bins = data.shape[1] // bin_size
    trimmed = data[:, : n_bins * bin_size]
    reshaped = trimmed.reshape(data.shape[0], n_bins, bin_size)
    if method == "mean":
        return torch.nanmean(reshaped, dim=2)
    if method == "sum":
        return torch.sum(reshaped, dim=2)
    raise ValueError(f"Unrecognized binning method: {method}")


def gaussian_window(length: int, sigma: int) -> np.ndarray:
    center = (length - 1) / 2.0
    n = np.arange(length, dtype=np.float64) - center
    return np.exp(-0.5 * (n / sigma) ** 2)


def gauss_filt(spikes: np.ndarray, sigma: int, bin_size: int) -> np.ndarray:
    spikes = np.asarray(spikes, dtype=np.float64)
    nan_mask = np.any(np.isnan(spikes), axis=0)
    had_nan = bool(np.any(nan_mask))
    prepend_nan = False
    if had_nan:
        nan_idx = np.flatnonzero(nan_mask)
        if not np.all(np.diff(nan_idx) == 1):
            raise ValueError("Non-consecutive NaNs encountered while filtering.")
        prepend_nan = bool(nan_mask[0])
        if not prepend_nan and not bool(nan_mask[-1]):
            raise ValueError("Time series broken up by a stretch of NaNs.")
        spikes_work = spikes[:, ~nan_mask]
    else:
        spikes_work = spikes

    width = 4
    pad = width * sigma
    length = 2 * pad + 1
    kernel = gaussian_window(length, sigma)
    kernel = kernel / kernel.sum() * bin_size

    pre = np.repeat(np.mean(spikes_work[:, :sigma], axis=1, dtype=np.float64)[:, None], pad, axis=1)
    post = np.repeat(np.mean(spikes_work[:, -sigma:], axis=1, dtype=np.float64)[:, None], pad, axis=1)
    padded = np.concatenate([pre, spikes_work, post], axis=1)

    filtered = np.zeros_like(padded, dtype=np.float64)
    for n in range(padded.shape[0]):
        conv = np.convolve(padded[n], kernel)
        filtered[n] = conv[pad : conv.shape[0] - pad]

    filtered = filtered[:, pad : filtered.shape[1] - pad]
    if had_nan:
        nan_block = np.full((filtered.shape[0], int(nan_mask.sum())), np.nan, dtype=np.float64)
        filtered = np.concatenate([nan_block, filtered], axis=1) if prepend_nan else np.concatenate([filtered, nan_block], axis=1)
    return filtered


def _pca_coeff(data: Tensor, n_components: int) -> Tensor:
    centered = data - torch.mean(data, dim=0, keepdim=True)
    if centered.shape[0] <= 1:
        cov = centered.T @ centered
    else:
        cov = centered.T @ centered / (centered.shape[0] - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    return eigvecs[:, order][:, :n_components]


def smooth_average(grouped_trials: Sequence[Sequence[Tensor]], hyperparams, Ts: float) -> List[Tensor]:
    x_avg = [torch.nanmean(torch.stack(list(group), dim=2), dim=2) for group in grouped_trials]
    all_avg = torch.cat(x_avg, dim=1)
    soft_norm = hyperparams.soft_norm * hyperparams.Delta * Ts
    mu = torch.mean(all_avg, dim=1, keepdim=True)
    norm_factor = 1.0 / (soft_norm + torch.max(all_avg, dim=1, keepdim=True).values)

    normalized: List[List[Tensor]] = []
    for group in grouped_trials:
        normalized.append([(trial - mu) * norm_factor for trial in group])

    if hyperparams.n_trial_dims is not None:
        for c, group in enumerate(normalized):
            rows = [trial.T.reshape(1, -1) for trial in group]
            x_nt = torch.cat(rows, dim=0).T
            coeff = _pca_coeff(x_nt, int(hyperparams.n_trial_dims))
            projected = coeff @ coeff.T @ x_nt.T
            n_neurons, n_times = group[0].shape
            normalized[c] = [projected[i].reshape(n_times, n_neurons).T for i in range(len(group))]

    x_bar = [torch.nanmean(torch.stack(group, dim=2), dim=2) for group in normalized]

    if hyperparams.n_neural_dims is not None:
        data = torch.cat(x_bar, dim=1).T
        coeff = _pca_coeff(data, int(hyperparams.n_neural_dims))
        x_bar = [coeff @ coeff.T @ item for item in x_bar]

    if hyperparams.n_cond_dims is not None:
        n_conds = len(x_bar)
        n_neurons, n_times = x_bar[0].shape
        x_bar_nt = torch.cat([item.T.reshape(1, -1) for item in x_bar], dim=0).T
        coeff = _pca_coeff(x_bar_nt, int(hyperparams.n_cond_dims))
        projected = coeff @ coeff.T @ x_bar_nt.T
        x_bar = [projected[i].reshape(n_times, n_neurons).T for i in range(n_conds)]

    return [torch.clamp(item / norm_factor + mu, min=0.0) for item in x_bar]


def get_rate_indices(lambda_values: Tensor, lambda_range: Tensor, n_rates: int) -> Tensor:
    lam_min = lambda_range[0]
    lam_max = lambda_range[1]
    clipped = torch.clamp(lambda_values, min=float(lam_min), max=float(lam_max))
    scaled = (clipped - lam_min) / (lam_max - lam_min) * (n_rates - 1) + 1.0
    matlab_uint = torch.floor(scaled + 0.5).to(torch.long)
    return torch.clamp(matlab_uint - 1, 0, n_rates - 1)


def ck2ind(c0: int, k_one_based: Tensor, first_idx0: Tensor) -> Tensor:
    return first_idx0[c0] + k_one_based.to(torch.long) - 1


def ind2ck(index0: int, first_idx0: Tensor) -> Tuple[int, int]:
    starts = first_idx0.cpu().numpy()
    c0 = int(np.searchsorted(starts, index0, side="right") - 1)
    return c0, int(index0 - starts[c0] + 1)


def get_time_indices(
    t_prime_one: int,
    T_prime: int,
    T: int,
    Delta: int,
    tau_prime: int,
    causal: bool,
) -> Tuple[Tensor, Callable[[int], Tensor]]:
    t = t_prime_one * Delta
    if not causal:
        tau = (tau_prime + 1) * Delta - 1
        adjustment = round((tau + 1 + Delta) / 2)
        t = t - adjustment
    t_idx = list(range(t, t + Delta))
    if t_prime_one == tau_prime + 1:
        t_idx = list(range(1, t_idx[0])) + t_idx
    if t_prime_one == T_prime and t_idx[-1] < T:
        t_idx = t_idx + list(range(t_idx[-1] + 1, T + 1))
    t_idx = [idx for idx in t_idx if idx <= T]
    t_idx0 = torch.as_tensor([idx - 1 for idx in t_idx], dtype=torch.long)

    def f(k_prime_one: int) -> Tensor:
        return (k_prime_one - t_prime_one) * Delta + t_idx0

    return t_idx0, f


def get_state_indices(k_prime_hats: Sequence[int], f: Callable[[int], Tensor], K: int) -> Tensor:
    out = torch.stack([f(k_prime_hats[0]), f(k_prime_hats[1])], dim=0)
    return torch.clamp(out, 0, K - 1)


def _masked_rows(tensor: Tensor, mask: Optional[Tensor]) -> Tensor:
    return tensor if mask is None else tensor[mask]


def fit_poisson_interp(S: Tensor, X1: Tensor, X2: Tensor, options: InterpOptions, default_alpha: float) -> float:
    x2_minus_x1 = X2 - X1
    x2_minus_x1_sum = torch.sum(x2_minus_x1)
    alpha = 0.5
    i = 0
    while i < options.max_iters:
        denom = X1 + alpha * x2_minus_x1
        fraction = x2_minus_x1 / denom
        deriv1 = torch.sum(S * fraction) - x2_minus_x1_sum
        deriv2 = -torch.sum(S * (fraction**2))
        alpha_step = float((deriv1 / deriv2).item())
        alpha = alpha - alpha_step
        if alpha_step < options.step_tol or alpha < 0.0 or alpha > 1.0:
            alpha = max(min(alpha, 1.0), 0.0)
            break
        i += 1
    if alpha != alpha:
        return default_alpha
    return alpha


def use_best_interp(S_curr: Tensor, interps, likelihood_neuron_mask: Optional[Tensor] = None):
    S_curr = _masked_rows(S_curr, likelihood_neuron_mask)
    scores = []
    for interp in interps:
        lam = _masked_rows(interp[2], likelihood_neuron_mask)
        scores.append(torch.sum(S_curr * torch.log(lam) - lam))
    idx = int(torch.argmax(torch.stack(scores)).item())
    return interps[idx]


def _decode_char(dataset: h5py.Dataset) -> str:
    arr = np.asarray(dataset[()])
    return "".join(chr(int(x)) for x in arr.ravel() if int(x) != 0)


def _decode_cellstr(file: h5py.File, dataset: h5py.Dataset) -> np.ndarray:
    values: List[str] = []
    for ref in np.asarray(dataset[()]).ravel():
        values.append(_decode_char(file[ref]))
    return np.asarray(values, dtype=object)


class MintMatFile:
    def __init__(self, path: Path, dataset: str):
        self.path = Path(path)
        self.dataset = dataset

    def load(self) -> Tuple[MatTable, MatTable]:
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


def _alignment_array(values: range) -> np.ndarray:
    return np.asarray(list(values), dtype=np.int64)


def _mask_by_alignment(source: range, target: range) -> np.ndarray:
    return np.isin(_alignment_array(source), _alignment_array(target))


def _split_labels(split) -> List[str]:
    if isinstance(split, (list, tuple, set)):
        labels = []
        for item in split:
            labels.extend(_split_labels(item))
        return list(dict.fromkeys(labels))
    if split == "train":
        return ["train"]
    if split == "val":
        return ["val"]
    if split == "test":
        return ["val"]
    if split == "trainval":
        return ["train", "val"]
    raise ValueError(f"Unsupported split: {split}")


def _spikes_tensor(spikes: np.ndarray, device: Optional[torch.device] = None) -> Tensor:
    spikes = np.asarray(spikes)
    finite = spikes[np.isfinite(spikes)]
    if finite.size and finite.min() >= 0 and finite.max() <= 255:
        return torch.as_tensor(spikes, dtype=torch.uint8, device=device)
    return as_tensor(spikes, device)


def get_trial_data(settings, split: str, max_trials: Optional[int] = None, device: Optional[torch.device] = None):
    T, TrialInfo = MintMatFile(settings.data_path, settings.task).load()
    if settings.task == "area2_bump":
        return _area2_get_trial_data(T, TrialInfo, settings, split, max_trials, device)
    if settings.task == "mc_maze":
        return _mc_maze_get_trial_data(T, TrialInfo, settings, split, max_trials, device)
    if settings.task == "mc_rtt":
        return _mc_rtt_get_trial_data(T, TrialInfo, settings, split, max_trials, device)
    raise ValueError(f"Unknown task: {settings.task}")


def _area2_get_trial_data(T, TrialInfo, settings, split, max_trials, device):
    good = TrialInfo["result"] == "R"
    TrialInfo = TrialInfo.subset_trials(good)
    cond_mat = np.column_stack([TrialInfo["cond_dir"], TrialInfo["ctr_hold_bump"].astype(np.float64)])
    cond_list = np.unique(cond_mat, axis=0)
    S, Z, condition = [], [], []
    alignment = _alignment_array(settings.trial_alignment)
    time = T["time"]
    for tr in range(TrialInfo.n_trials):
        move_idx = find_time_index(time, TrialInfo["move_onset_time"][tr])
        idx = move_idx + alignment
        spikes = stack_rows([take_time_window(T["heldout_spikes"], idx), take_time_window(T["spikes"], idx)])
        behavior = stack_rows(
            [
                take_time_window(T["hand_pos"], idx),
                take_time_window(T["hand_vel"], idx),
                take_time_window(T["force"], idx),
                take_time_window(T["joint_ang"], idx),
                take_time_window(T["joint_vel"], idx),
                take_time_window(T["muscle_len"], idx),
                take_time_window(T["muscle_vel"], idx),
            ]
        )
        cond_row = np.asarray([TrialInfo["cond_dir"][tr], float(TrialInfo["ctr_hold_bump"][tr])])
        cond = int(np.flatnonzero(np.all(cond_list == cond_row, axis=1))[0])
        S.append(_spikes_tensor(spikes, device))
        Z.append(as_tensor(behavior, device))
        condition.append(cond)
    idx = np.flatnonzero(np.isin(TrialInfo["split"], _split_labels(split)))
    idx = idx[:max_trials] if max_trials is not None else idx
    return [S[i] for i in idx], [Z[i] for i in idx], np.asarray([condition[i] for i in idx]), cond_list


def _mc_maze_get_trial_data(T, TrialInfo, settings, split, max_trials, device):
    cond_mat = np.column_stack([TrialInfo["trial_type"], TrialInfo["trial_version"]])
    cond_list = np.unique(cond_mat, axis=0)
    S, Z, condition = [], [], []
    alignment = _alignment_array(settings.trial_alignment)
    time = T["time"]
    for tr in range(TrialInfo.n_trials):
        move_idx = find_time_index(time, TrialInfo["move_onset_time"][tr])
        idx = move_idx + alignment
        spikes = stack_rows([take_time_window(T["heldout_spikes"], idx), take_time_window(T["spikes"], idx)])
        behavior = stack_rows([take_time_window(T["hand_pos"], idx), take_time_window(T["hand_vel"], idx)])
        cond_row = np.asarray([TrialInfo["trial_type"][tr], TrialInfo["trial_version"][tr]])
        cond = int(np.flatnonzero(np.all(cond_list == cond_row, axis=1))[0])
        S.append(_spikes_tensor(spikes, device))
        Z.append(as_tensor(behavior, device))
        condition.append(cond)
    idx = np.flatnonzero(np.isin(TrialInfo["split"], _split_labels(split)))
    idx = idx[:max_trials] if max_trials is not None else idx
    return [S[i] for i in idx], [Z[i] for i in idx], np.asarray([condition[i] for i in idx]), cond_list


def _mc_rtt_get_trial_data(T, TrialInfo, settings, split, max_trials, device):
    if split == "train":
        train_idx = np.flatnonzero(TrialInfo["split"] == "train")
        last_train = int(train_idx[-1])
        end_time = TrialInfo["end_time"][last_train]
        T = T.subset_time(T["time"] <= end_time)
        return _mc_rtt_get_continuous_data(T, max_trials, device)
    if split == "trainval":
        public_idx = np.flatnonzero(np.isin(TrialInfo["split"], ["train", "val"]))
        last_public = int(public_idx[-1])
        end_time = TrialInfo["end_time"][last_public]
        T = T.subset_time(T["time"] <= end_time)
        return _mc_rtt_get_continuous_data(T, max_trials, device)
    raise ValueError("MC_RTT MINT NLB training supports train or trainval splits.")


def _mc_rtt_get_continuous_data(T: MatTable, max_trials, device):
    gap_mask = np.isnan(T["finger_pos"][0]) | np.isnan(T["autolfads_rates"][0])
    diff = np.diff(gap_mask.astype(np.int8))
    starts = np.concatenate([[0], np.flatnonzero(diff == -1) + 1])
    ends = np.concatenate([np.flatnonzero(diff == 1), [len(gap_mask) - 1]])
    S, Z = [], []
    for start, end in zip(starts, ends):
        idx = np.arange(start, end + 1)
        spikes = stack_rows([T["heldout_spikes"][:, idx], T["spikes"][:, idx]])
        behavior = stack_rows([T["finger_pos"][:2, idx], T["finger_vel"][:, idx], T["autolfads_rates"][:, idx]])
        S.append(_spikes_tensor(spikes, device))
        Z.append(as_tensor(behavior, device))
    if max_trials is not None:
        S = S[:max_trials]
        Z = Z[:max_trials]
    condition = np.arange(len(S), dtype=np.int64)
    return S, Z, condition, condition[:, None]


def preprocess_behavior(Z: Sequence[Tensor], settings):
    if settings.task == "area2_bump":
        zero_idx = list(settings.trial_alignment).index(0)
        out = []
        for item in Z:
            pos = item[:2] - item[:2, zero_idx : zero_idx + 1]
            out.append(torch.cat([pos, item[2:]], dim=0))
        labels = (
            ["xpos", "ypos", "xvel", "yvel"]
            + [f"force_{i}" for i in range(1, 7)]
            + [f"joint_ang_{i}" for i in range(1, 8)]
            + [f"joint_vel_{i}" for i in range(1, 8)]
            + [f"muscle_len_{i}" for i in range(1, 40)]
            + [f"muscle_vel_{i}" for i in range(1, 40)]
        )
        return out, labels
    if settings.task == "mc_maze":
        zero_idx = list(settings.trial_alignment).index(0)
        out = []
        for item in Z:
            pos = item[:2] - item[:2, zero_idx : zero_idx + 1]
            out.append(torch.cat([pos, item[2:]], dim=0))
        return out, ["xpos", "ypos", "xvel", "yvel"]
    if settings.task == "mc_rtt":
        return [item[2:4] for item in Z], ["xvel", "yvel"]
    raise ValueError(f"Unknown task: {settings.task}")


def fit_trajectories(S, Z, condition, settings, hyperparams):
    if settings.task in {"area2_bump", "mc_maze"}:
        S_smooth = [as_tensor(gauss_filt(spikes.cpu().numpy(), hyperparams.sigma, hyperparams.Delta), spikes.device) for spikes in S]
        Z_proc, labels = preprocess_behavior(Z, settings)
        t_mask = _mask_by_alignment(settings.trial_alignment, hyperparams.trajectories_alignment)
        S_smooth = [item[:, t_mask] for item in S_smooth]
        Z_proc = [item[:, t_mask] for item in Z_proc]
        cond_list = np.unique(condition)
        grouped_x, grouped_z = [], []
        for cond in cond_list:
            trial_idx = np.flatnonzero(condition == cond)
            grouped_x.append([S_smooth[i] for i in trial_idx])
            grouped_z.append([Z_proc[i] for i in trial_idx])
        z_bar = [torch.mean(torch.stack(group, dim=2), dim=2) for group in grouped_z]
        x_bar = smooth_average(grouped_x, hyperparams, settings.Ts)
        return x_bar, z_bar, labels
    if settings.task == "mc_rtt":
        vel, labels = preprocess_behavior(Z, settings)
        rates = [item[4:] * settings.Ts * hyperparams.Delta for item in Z]
        return rates, vel, labels
    raise ValueError(f"Unknown task: {settings.task}")


def default_train_nwb_path(dataset: str, nwb_dir: Path = Path("data/real/nlb/dandi")) -> Path:
    return Path(nwb_dir) / TRAIN_NWB_REL[dataset]


def _to_ms(value) -> int:
    if hasattr(value, "total_seconds"):
        return int(round(value.total_seconds() * 1000.0))
    return int(round(float(value) * 1000.0))


def _field_matrix(ds, field: str, dtype=np.float64) -> np.ndarray:
    return ds.data[field].to_numpy(dtype=dtype).T


def _condition_index(cond_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cond_list = np.unique(cond_mat, axis=0)
    condition = []
    for row in cond_mat:
        condition.append(int(np.flatnonzero(np.all(cond_list == row, axis=1))[0]))
    return np.asarray(condition, dtype=np.int64), cond_list


def get_nwb_trial_data(
    settings,
    split: str,
    nwb_path: Path,
    max_trials: Optional[int] = None,
    device: Optional[torch.device] = None,
):
    from nlb_tools.nwb_interface import NWBDataset

    ds = NWBDataset(Path(nwb_path))
    if settings.task == "area2_bump":
        return _area2_nwb_trial_data(ds, settings, split, max_trials, device)
    if settings.task == "mc_maze":
        return _mc_maze_nwb_trial_data(ds, settings, split, max_trials, device)
    raise ValueError(f"Direct trial NWB loading is not implemented for {settings.task}.")


def _area2_nwb_trial_data(ds, settings, split, max_trials, device):
    trial_info = ds.trial_info
    split_mask = trial_info["split"].isin(_split_labels(split)).to_numpy()
    good_mask = (trial_info["result"] == "R").to_numpy()
    idx = np.flatnonzero(split_mask & good_mask)
    idx = idx if max_trials is None else idx[:max_trials]
    cond_mat = np.column_stack(
        [
            trial_info.iloc[idx]["cond_dir"].to_numpy(dtype=np.float64),
            trial_info.iloc[idx]["ctr_hold_bump"].to_numpy(dtype=np.float64),
        ]
    )
    condition, cond_list = _condition_index(cond_mat)
    heldout = _field_matrix(ds, "heldout_spikes")
    spikes = _field_matrix(ds, "spikes")
    behavior_fields = [
        _field_matrix(ds, "hand_pos"),
        _field_matrix(ds, "hand_vel"),
        _field_matrix(ds, "force"),
        _field_matrix(ds, "joint_ang"),
        _field_matrix(ds, "joint_vel"),
        _field_matrix(ds, "muscle_len"),
        _field_matrix(ds, "muscle_vel"),
    ]
    alignment = _alignment_array(settings.trial_alignment)
    S, Z = [], []
    for tr in idx:
        time_idx = _to_ms(trial_info.iloc[tr]["move_onset_time"]) + alignment
        spikes_trial = stack_rows([take_time_window(heldout, time_idx), take_time_window(spikes, time_idx)])
        behavior = stack_rows(take_time_window(field, time_idx) for field in behavior_fields)
        if np.isnan(behavior).any():
            raise ValueError(f"area2_bump trial {tr}: behavior window contains NaNs.")
        S.append(_spikes_tensor(spikes_trial, device))
        Z.append(as_tensor(behavior, device))
    return S, Z, condition, cond_list


def _mc_maze_nwb_trial_data(ds, settings, split, max_trials, device):
    trial_info = ds.trial_info
    split_mask = trial_info["split"].isin(_split_labels(split)).to_numpy()
    idx = np.flatnonzero(split_mask)
    idx = idx if max_trials is None else idx[:max_trials]
    cond_mat = np.column_stack(
        [
            trial_info.iloc[idx]["trial_type"].to_numpy(dtype=np.float64),
            trial_info.iloc[idx]["trial_version"].to_numpy(dtype=np.float64),
        ]
    )
    condition, cond_list = _condition_index(cond_mat)
    heldout = _field_matrix(ds, "heldout_spikes")
    spikes = _field_matrix(ds, "spikes")
    hand_pos = _field_matrix(ds, "hand_pos")
    hand_vel = _field_matrix(ds, "hand_vel")
    alignment = _alignment_array(settings.trial_alignment)
    S, Z = [], []
    for tr in idx:
        time_idx = _to_ms(trial_info.iloc[tr]["move_onset_time"]) + alignment
        spikes_trial = stack_rows([take_time_window(heldout, time_idx), take_time_window(spikes, time_idx)])
        behavior = stack_rows([take_time_window(hand_pos, time_idx), take_time_window(hand_vel, time_idx)])
        if np.isnan(behavior).any():
            raise ValueError(f"mc_maze trial {tr}: behavior window contains NaNs.")
        S.append(_spikes_tensor(spikes_trial, device))
        Z.append(as_tensor(behavior, device))
    return S, Z, condition, cond_list


def heldout_count(dataset: str) -> int:
    return HELDOUT_COUNTS[dataset]


def observed_neuron_mask(dataset: str, n_neurons: int, device=None) -> Tensor:
    n_heldout = heldout_count(dataset)
    mask = torch.ones(n_neurons, dtype=torch.bool, device=device)
    mask[:n_heldout] = False
    return mask
