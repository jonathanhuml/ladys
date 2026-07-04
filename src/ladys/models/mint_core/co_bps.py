from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from .utils import TORCH_DTYPE, bin_data


HELDOUT_COUNTS = {
    "area2_bump": 16,
    "mc_maze": 45,
    "mc_rtt": 32,
}


def heldout_count(dataset: str) -> int:
    return HELDOUT_COUNTS[dataset]


def observed_neuron_mask(dataset: str, n_neurons: int, device=None) -> torch.Tensor:
    n_heldout = heldout_count(dataset)
    mask = torch.ones(n_neurons, dtype=torch.bool, device=device)
    mask[:n_heldout] = False
    return mask


def null_counts_per_bin(train_spikes: Sequence[torch.Tensor], dataset: str, eval_bin_size: int) -> torch.Tensor:
    n_heldout = heldout_count(dataset)
    pieces = [spikes[:n_heldout].to(TORCH_DTYPE) for spikes in train_spikes]
    concat = torch.cat(pieces, dim=1)
    return torch.nanmean(concat, dim=1, keepdim=True) * eval_bin_size


def counts_for_alignment(spikes: Sequence[torch.Tensor], dataset: str, eval_bin_size: int) -> list[torch.Tensor]:
    n_heldout = heldout_count(dataset)
    return [bin_data(item[:n_heldout], eval_bin_size, "sum") for item in spikes]


def predicted_counts_per_bin(
    x_hat: Sequence[torch.Tensor],
    dataset: str,
    eval_bin_size: int,
    decoder_delta: int,
) -> list[torch.Tensor]:
    n_heldout = heldout_count(dataset)
    return [bin_data(item[:n_heldout], eval_bin_size, "mean") * (eval_bin_size / decoder_delta) for item in x_hat]


def poisson_bits_per_spike(
    heldout_counts: Sequence[torch.Tensor],
    predicted_counts: Sequence[torch.Tensor],
    null_counts: torch.Tensor | None = None,
    eps: float = 1e-9,
) -> float:
    """NLB/EvalAI-style bits per spike.

    If ``null_counts`` is omitted, the null model is each heldout neuron's
    mean count over the evaluation set, matching ``nlb_tools.evaluation``.
    """
    y = torch.cat([item.reshape(item.shape[0], -1) for item in heldout_counts], dim=1).to(TORCH_DTYPE)
    lam = torch.cat([item.reshape(item.shape[0], -1) for item in predicted_counts], dim=1).to(TORCH_DTYPE)
    if null_counts is None:
        null = torch.nanmean(y, dim=1, keepdim=True).expand_as(lam)
    else:
        null = null_counts.to(device=lam.device, dtype=TORCH_DTYPE).expand_as(lam)

    finite = torch.isfinite(y) & torch.isfinite(lam) & torch.isfinite(null)
    y = y[finite]
    lam = torch.clamp(lam[finite], min=eps)
    null = torch.clamp(null[finite], min=eps)
    spike_count = torch.sum(y)
    if spike_count <= 0:
        return float("nan")
    ll_model = torch.sum(y * torch.log(lam) - lam)
    ll_null = torch.sum(y * torch.log(null) - null)
    return float(((ll_model - ll_null) / (spike_count * np.log(2.0))).item())
