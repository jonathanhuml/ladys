"""Negative log-likelihood."""

from __future__ import annotations

from typing import TYPE_CHECKING
from jaxtyping import Float


if TYPE_CHECKING:
    from torch import distributions, Tensor


def negative_log_likelihood(
    preds: Float[
        distributions.MultivariateNormal, "batch *condition timepoint observation"
    ],
    targets: Float[Tensor, "batch *condition timepoint observation"],
) -> Float[Tensor, ""]:
    """Negative log-likelihood.

    :param preds:   Model predictions.
    :param targets: Targets.
    """
    return -preds.log_prob(targets).mean()
