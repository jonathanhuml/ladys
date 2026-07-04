"""Neural Latents Benchmark scoring helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.special import gammaln


@dataclass
class NLBScore:
    """NLB co-smoothing score for held-out spike-count predictions."""

    co_bps: float
    spike_count: float
    prediction_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "co_bps": self.co_bps,
            "spike_count": self.spike_count,
            "prediction_shape": list(self.prediction_shape),
            "target_shape": list(self.target_shape),
        }
        data.update(self.metrics)
        return data


def nlb_negative_log_likelihood(
    rates: np.ndarray,
    spikes: np.ndarray,
    *,
    zero_floor: float = 1e-9,
) -> float:
    """NLB Poisson negative log-likelihood for spike-count rates."""

    rates = np.asarray(rates, dtype=np.float64).copy()
    spikes = np.asarray(spikes, dtype=np.float64)
    if rates.shape != spikes.shape:
        raise ValueError(f"rates and spikes shapes differ: {rates.shape} != {spikes.shape}")

    if np.any(np.isnan(spikes)):
        mask = ~np.isnan(spikes)
        rates = rates[mask]
        spikes = spikes[mask]

    if np.any(np.isnan(rates)):
        raise ValueError("NaN rate predictions found")
    if np.any(rates < 0):
        raise ValueError("negative rate predictions found")
    rates[rates == 0] = zero_floor
    return float(np.sum(rates - spikes * np.log(rates) + gammaln(spikes + 1.0)))


def nlb_bits_per_spike(rates: np.ndarray, spikes: np.ndarray) -> float:
    """NLB/EvalAI co-smoothing bits per spike for count-valued predictions."""

    rates = np.asarray(rates, dtype=np.float64)
    spikes = np.asarray(spikes, dtype=np.float64)
    if rates.shape != spikes.shape:
        raise ValueError(f"rates and spikes shapes differ: {rates.shape} != {spikes.shape}")

    nll_model = nlb_negative_log_likelihood(rates, spikes)
    null_rates = np.tile(
        np.nanmean(spikes, axis=tuple(range(spikes.ndim - 1)), keepdims=True),
        spikes.shape[:-1] + (1,),
    )
    nll_null = nlb_negative_log_likelihood(null_rates, spikes)
    spike_count = float(np.nansum(spikes))
    if spike_count <= 0.0:
        raise ValueError("cannot compute bits/spike with zero held-out spikes")
    return float((nll_null - nll_model) / spike_count / np.log(2.0))


def score_count_predictions(rates: np.ndarray, spikes: np.ndarray) -> NLBScore:
    """Score held-out count predictions against held-out spikes."""

    rates = np.asarray(rates)
    spikes = np.asarray(spikes)
    score = nlb_bits_per_spike(rates, spikes)
    return NLBScore(
        co_bps=score,
        spike_count=float(np.nansum(spikes)),
        prediction_shape=tuple(rates.shape),
        target_shape=tuple(spikes.shape),
        metrics={"co-bps": score},
    )


def score_ladys_predictions(path: Path | str) -> NLBScore:
    """Score a LaDyS ``predictions.npz`` artifact with the NLB co-bps metric."""

    path = Path(path)
    with np.load(path) as data:
        if "pred_rates" not in data or "target_spikes" not in data:
            keys = ", ".join(data.files)
            raise KeyError(
                f"{path} must contain pred_rates and target_spikes arrays. Found: {keys}"
            )
        return score_count_predictions(data["pred_rates"], data["target_spikes"])


def score_run_dir(path: Path | str) -> NLBScore:
    """Score ``predictions.npz`` inside a LaDyS run directory."""

    return score_ladys_predictions(Path(path) / "predictions.npz")


def evaluate_nlb_submission(
    target_h5: Path | str,
    submission_h5: Path | str,
) -> list[dict[str, dict[str, float]]]:
    """Run the full nlb_tools EvalAI-style evaluator on an H5 submission."""

    try:
        from nlb_tools.evaluation import evaluate
    except ImportError as exc:
        raise RuntimeError("nlb_tools is required for full NLB H5 evaluation.") from exc
    return evaluate(str(target_h5), str(submission_h5))


def read_submission_co_bps(
    target_h5: Path | str,
    submission_h5: Path | str,
    dataset: str,
    bin_size_ms: int = 5,
) -> NLBScore:
    """Score one EvalAI-style H5 group when only co-bps is required."""

    group_name = dataset if bin_size_ms == 5 else f"{dataset}_{bin_size_ms}"
    with h5py.File(target_h5, "r") as target, h5py.File(submission_h5, "r") as pred:
        spikes = target[group_name]["eval_spikes_heldout"][()]
        rates = pred[group_name]["eval_rates_heldout"][()]
    return score_count_predictions(rates, spikes)


def score_to_json(score: NLBScore | list[dict[str, dict[str, float]]]) -> str:
    """Serialize NLB scorer output for CLI use."""

    if isinstance(score, NLBScore):
        payload: Any = score.to_dict()
    else:
        payload = score
    return json.dumps(_json_ready(payload), indent=2, sort_keys=True)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    return value
