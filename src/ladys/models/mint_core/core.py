from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional, Sequence, Tuple

import torch

from .tasks import fit_trajectories
from .utils import (
    TORCH_DTYPE,
    bin_data,
    ck2ind,
    get_rate_indices,
    get_state_indices,
    get_time_indices,
    ind2ck,
)


@dataclass
class InterpOptions:
    max_iters: int = 10
    step_tol: float = 0.01


class MINT:
    def __init__(self, settings, hyperparams, device: Optional[torch.device] = None):
        self.Settings = settings
        self.HyperParams = hyperparams
        self.device = device or torch.device("cpu")

        self.Ts = settings.Ts
        self.Delta = hyperparams.Delta
        self.dt = self.Delta * self.Ts
        self.window_length = hyperparams.window_length
        self.tau_prime = round(self.window_length / self.Delta) - 1
        self.causal = hyperparams.causal
        self.interp = hyperparams.interp

        self.min_prob = hyperparams.min_prob
        self.min_lambda = hyperparams.min_lambda
        self.n_rates = 2000
        self.lambda_range = torch.as_tensor([self.min_lambda, 500.0], dtype=TORCH_DTYPE, device=self.device) * self.dt
        self.rates = torch.linspace(
            float(self.lambda_range[0]),
            float(self.lambda_range[1]),
            self.n_rates,
            dtype=TORCH_DTYPE,
            device=self.device,
        )
        self.max_spikes = round(self.dt * 1000)
        self.L = self._build_likelihood_table()
        self.min_rate = hyperparams.min_rate * self.dt
        self.InterpOptions = InterpOptions()

        self.Omega_plus: List[torch.Tensor] = []
        self.Phi_plus: List[torch.Tensor] = []
        self.behavior_labels: List[str] = []
        self.V: Optional[torch.Tensor] = None
        self.first_idx0: Optional[torch.Tensor] = None
        self.last_idx0: Optional[torch.Tensor] = None
        self.first_tau_prime_idx0: Optional[torch.Tensor] = None
        self.shifted_idx1: Optional[torch.Tensor] = None
        self.shifted_idx2: Optional[torch.Tensor] = None

    def _build_likelihood_table(self) -> torch.Tensor:
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

    def fit(self, S: Sequence[torch.Tensor], Z: Sequence[torch.Tensor], condition):
        self.Omega_plus, self.Phi_plus, self.behavior_labels = fit_trajectories(
            S, Z, condition, self.Settings, self.HyperParams
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
        self.last_idx0 = torch.as_tensor([start + length - 1 for start, length in zip(starts, lengths)], dtype=torch.long, device=self.device)
        self.first_tau_prime_idx0 = torch.cat(
            [start + torch.arange(self.tau_prime, dtype=torch.long, device=self.device) for start in self.first_idx0]
        ).sort().values
        self.V = torch.cat([v.T for v in v_cells], dim=0).to(torch.long)
        self._build_shifted_indices()
        return self

    def _build_shifted_indices(self):
        idx1, idx2 = [], []
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

    def _recursion(
        self,
        Q: torch.Tensor,
        s_new: torch.Tensor,
        s_old: torch.Tensor,
        t_prime_one: int,
        likelihood_neuron_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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

    def predict(
        self,
        S: Sequence[torch.Tensor],
        return_aux: bool = False,
        likelihood_neuron_mask: Optional[torch.Tensor] = None,
    ):
        if self.V is None:
            raise RuntimeError("Model must be fit before predict().")

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

    def _estimate_states(
        self,
        Q: torch.Tensor,
        S_curr: torch.Tensor,
        f,
        likelihood_neuron_mask: Optional[torch.Tensor] = None,
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
        min_k_prime_dist = self.HyperParams.min_k_dist / self.Delta
        for _ in range(self.HyperParams.n_candidates):
            if self.HyperParams.interp_within_trajectories:
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
            beta = fit_poisson_interp(_masked_rows(S_curr, likelihood_neuron_mask), _masked_rows(lam_a, likelihood_neuron_mask), _masked_rows(lam_b, likelihood_neuron_mask), self.InterpOptions, 0.0)
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
        Q: torch.Tensor,
        restricted_conds: Optional[Sequence[int]] = None,
        states_to_exclude: Optional[Sequence[Tuple[int, int]]] = None,
        min_k_prime_dist: Optional[float] = None,
    ) -> tuple[int, List[int]]:
        q = Q.clone()
        q[self.first_tau_prime_idx0] = float("nan")
        lengths = (torch.cat([self.first_idx0[1:], torch.as_tensor([len(Q)], device=self.device)]) - self.first_idx0).tolist()
        if restricted_conds:
            for c0 in restricted_conds:
                q[self.first_idx0[c0] : self.first_idx0[c0] + lengths[c0]] = float("nan")
        if states_to_exclude:
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
        S_curr,
        c0: int,
        k_hats: Sequence[int],
        k_idx: torch.Tensor,
        likelihood_neuron_mask: Optional[torch.Tensor] = None,
    ):
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


def _masked_rows(tensor: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    return tensor if mask is None else tensor[mask]


def fit_poisson_interp(S: torch.Tensor, X1: torch.Tensor, X2: torch.Tensor, options: InterpOptions, default_alpha: float) -> float:
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


def use_best_interp(S_curr: torch.Tensor, interps, likelihood_neuron_mask: Optional[torch.Tensor] = None):
    S_curr = _masked_rows(S_curr, likelihood_neuron_mask)
    scores = []
    for interp in interps:
        lam = _masked_rows(interp[2], likelihood_neuron_mask)
        scores.append(torch.sum(S_curr * torch.log(lam) - lam))
    idx = int(torch.argmax(torch.stack(scores)).item())
    return interps[idx]
