from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
import torch


TORCH_DTYPE = torch.float64


def as_tensor(array: np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
    return torch.as_tensor(array, dtype=TORCH_DTYPE, device=device)


def bin_data(data: torch.Tensor, bin_size: int, method: str) -> torch.Tensor:
    data = data.to(TORCH_DTYPE)
    n_bins = data.shape[1] // bin_size
    trimmed = data[:, : n_bins * bin_size]
    reshaped = trimmed.reshape(data.shape[0], n_bins, bin_size)
    if method == "mean":
        return torch.nanmean(reshaped, dim=2)
    if method == "sum":
        return torch.sum(reshaped, dim=2)
    raise ValueError(f"Unrecognized binning method: {method}")


def bin_data_np(data: np.ndarray, bin_size: int, method: str) -> np.ndarray:
    tensor = torch.as_tensor(data, dtype=TORCH_DTYPE)
    return bin_data(tensor, bin_size, method).cpu().numpy()


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


def _pca_coeff(data: torch.Tensor, n_components: int) -> torch.Tensor:
    centered = data - torch.mean(data, dim=0, keepdim=True)
    if centered.shape[0] <= 1:
        cov = centered.T @ centered
    else:
        cov = centered.T @ centered / (centered.shape[0] - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    eigvecs = eigvecs[:, order]
    return eigvecs[:, :n_components]


def smooth_average(
    grouped_trials: Sequence[Sequence[torch.Tensor]],
    hyperparams,
    Ts: float,
) -> List[torch.Tensor]:
    x_avg = [torch.nanmean(torch.stack(list(group), dim=2), dim=2) for group in grouped_trials]
    all_avg = torch.cat(x_avg, dim=1)
    soft_norm = hyperparams.soft_norm * hyperparams.Delta * Ts
    mu = torch.mean(all_avg, dim=1, keepdim=True)
    norm_factor = 1.0 / (soft_norm + torch.max(all_avg, dim=1, keepdim=True).values)

    normalized: List[List[torch.Tensor]] = []
    for group in grouped_trials:
        normalized.append([(trial - mu) * norm_factor for trial in group])

    if hyperparams.n_trial_dims is not None:
        for c, group in enumerate(normalized):
            rows = [trial.T.reshape(1, -1) for trial in group]
            x_nt = torch.cat(rows, dim=0).T
            coeff = _pca_coeff(x_nt, int(hyperparams.n_trial_dims))
            projected = coeff @ coeff.T @ x_nt.T
            n_trials = len(group)
            n_neurons, n_times = group[0].shape
            normalized[c] = [projected[i].reshape(n_times, n_neurons).T for i in range(n_trials)]

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

    x_bar = [torch.clamp(item / norm_factor + mu, min=0.0) for item in x_bar]
    return x_bar


def get_rate_indices(lambda_values: torch.Tensor, lambda_range: torch.Tensor, n_rates: int) -> torch.Tensor:
    lam_min = lambda_range[0]
    lam_max = lambda_range[1]
    clipped = torch.clamp(lambda_values, min=float(lam_min), max=float(lam_max))
    scaled = (clipped - lam_min) / (lam_max - lam_min) * (n_rates - 1) + 1.0
    matlab_uint = torch.floor(scaled + 0.5).to(torch.long)
    return torch.clamp(matlab_uint - 1, 0, n_rates - 1)


def ck2ind(c0: int, k_one_based: torch.Tensor, first_idx0: torch.Tensor) -> torch.Tensor:
    return first_idx0[c0] + k_one_based.to(torch.long) - 1


def ind2ck(index0: int, first_idx0: torch.Tensor) -> tuple[int, int]:
    starts = first_idx0.cpu().numpy()
    c0 = int(np.searchsorted(starts, index0, side="right") - 1)
    k_one_based = int(index0 - starts[c0] + 1)
    return c0, k_one_based


def get_time_indices(
    t_prime_one: int,
    T_prime: int,
    T: int,
    Delta: int,
    tau_prime: int,
    causal: bool,
) -> tuple[torch.Tensor, Callable[[int], torch.Tensor]]:
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

    def f(k_prime_one: int) -> torch.Tensor:
        return (k_prime_one - t_prime_one) * Delta + t_idx0

    return t_idx0, f


def get_state_indices(k_prime_hats: Sequence[int], f: Callable[[int], torch.Tensor], K: int) -> torch.Tensor:
    rows = [f(k_prime_hats[0]), f(k_prime_hats[1])]
    out = torch.stack(rows, dim=0)
    return torch.clamp(out, 0, K - 1)
