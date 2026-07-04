"""Evaluation metrics for LaDyS model outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable

import numpy as np
import torch
from torch import Tensor

from ladys.models.base import BaseDynamicsModel
from ladys.types import move_batch_to_device, observations_from_batch


EPS = 1e-8


@dataclass
class EvaluationResult:
    """Metrics and arrays collected from model evaluation."""

    metrics: dict[str, float]
    predictions: dict[str, np.ndarray] = field(default_factory=dict)
    targets: dict[str, np.ndarray] = field(default_factory=dict)


def poisson_negative_log_likelihood(
    rates: Tensor,
    spikes: Tensor,
    eps: float = EPS,
) -> Tensor:
    """Poisson negative log likelihood without the spike factorial constant."""

    rates = rates.clamp_min(eps)
    return rates - spikes * torch.log(rates)


def bits_per_spike(
    rates: Tensor,
    spikes: Tensor,
    eps: float = EPS,
) -> float:
    """NLB-style improvement over a per-neuron mean-rate Poisson baseline."""

    rates = rates.clamp_min(eps)
    spikes = spikes.float()
    reduce_dims = tuple(range(max(spikes.ndim - 1, 0)))
    null_rates = spikes.mean(dim=reduce_dims, keepdim=True).clamp_min(eps)
    model_nll = poisson_negative_log_likelihood(rates, spikes, eps=eps).sum()
    null_nll = poisson_negative_log_likelihood(null_rates, spikes, eps=eps).sum()
    total_spikes = spikes.sum()
    if float(total_spikes) <= 0.0:
        return float("nan")
    return float(((null_nll - model_nll) / total_spikes / math.log(2.0)).detach().cpu())


def r2_score(prediction: Tensor, target: Tensor, eps: float = EPS) -> float:
    """Coefficient of determination over all samples and timesteps."""

    target_mean = target.mean(dim=tuple(range(max(target.ndim - 1, 0))), keepdim=True)
    ss_res = torch.sum((target - prediction) ** 2)
    ss_tot = torch.sum((target - target_mean) ** 2)
    if float(ss_tot) <= eps:
        return float("nan")
    return float((1.0 - ss_res / ss_tot).detach().cpu())


def linear_r2_score(prediction: Tensor, target: Tensor, eps: float = EPS) -> float:
    """R2 after fitting the best affine map from prediction to target."""

    if prediction.shape[:-1] != target.shape[:-1]:
        return float("nan")

    x = prediction.reshape(-1, prediction.shape[-1]).double()
    y = target.reshape(-1, target.shape[-1]).double()
    ones = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
    design = torch.cat([x, ones], dim=1)
    solution = torch.linalg.lstsq(design, y).solution
    fitted = design @ solution
    return r2_score(fitted, y, eps=eps)


def evaluate_model(
    model: BaseDynamicsModel,
    loader: Iterable,
    device: torch.device | str = "cpu",
) -> EvaluationResult:
    """Evaluate a trained model on a dataloader using available outputs."""

    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()

    rate_predictions: list[Tensor] = []
    latent_predictions: list[Tensor] = []
    observed_spikes: list[Tensor] = []
    true_rates: list[Tensor] = []
    true_latents: list[Tensor] = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, torch_device)
            x = observations_from_batch(batch)
            output = model(x)
            rates = output.rates if output.rates is not None else output.reconstruction
            if rates is None:
                rates = model.predict_rates(x)
            rate_predictions.append(rates.detach().cpu())

            if output.latents is not None:
                latent_predictions.append(output.latents.detach().cpu())
            if isinstance(batch, dict):
                observed = batch.get("raw_spikes", batch.get("spikes"))
                if observed is not None:
                    observed_spikes.append(observed.detach().cpu())
                if "rates" in batch:
                    true_rates.append(batch["rates"].detach().cpu())
                if "latents" in batch:
                    true_latents.append(batch["latents"].detach().cpu())

    predictions: dict[str, Tensor] = {}
    targets: dict[str, Tensor] = {}
    if rate_predictions:
        predictions["rates"] = torch.cat(rate_predictions, dim=0)
    if latent_predictions:
        predictions["latents"] = torch.cat(latent_predictions, dim=0)
    if observed_spikes:
        targets["spikes"] = torch.cat(observed_spikes, dim=0)
    if true_rates:
        targets["rates"] = torch.cat(true_rates, dim=0)
    if true_latents:
        targets["latents"] = torch.cat(true_latents, dim=0)

    metrics = compute_available_metrics(predictions, targets)
    return EvaluationResult(
        metrics=metrics,
        predictions={key: value.numpy() for key, value in predictions.items()},
        targets={key: value.numpy() for key, value in targets.items()},
    )


def compute_available_metrics(
    predictions: dict[str, Tensor],
    targets: dict[str, Tensor],
) -> dict[str, float]:
    """Compute metrics supported by the returned predictions and targets."""

    metrics: dict[str, float] = {}
    pred_rates = predictions.get("rates")
    spikes = targets.get("spikes")
    rates = targets.get("rates")
    pred_latents = predictions.get("latents")
    latents = targets.get("latents")

    if pred_rates is not None and spikes is not None and pred_rates.shape == spikes.shape:
        metrics["co_bps"] = bits_per_spike(pred_rates, spikes)
        nll = poisson_negative_log_likelihood(pred_rates, spikes).mean()
        metrics["poisson_nll"] = float(nll.detach().cpu())
    if pred_rates is not None and rates is not None and pred_rates.shape == rates.shape:
        metrics["rate_mse"] = float(torch.mean((pred_rates - rates) ** 2).detach().cpu())
        metrics["rate_r2"] = r2_score(pred_rates, rates)
    if pred_latents is not None and latents is not None and pred_latents.shape[:-1] == latents.shape[:-1]:
        metrics["latent_linear_r2"] = linear_r2_score(pred_latents, latents)

    return metrics
