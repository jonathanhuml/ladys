"""Mean squared error."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from jaxtyping import Float

if TYPE_CHECKING:
    from torch import Tensor


def mean_squared_error(
    preds: Float[Tensor, "batch *condition timepoint observation"],
    targets: Float[Tensor, "batch *condition timepoint observation"],
) -> Float[Tensor, ""]:
    """Mean squared error.

    :param preds:   Model predictions.
    :param targets: Targets.
    """
    diff = targets - preds
    return torch.square(diff).mean()


def root_mean_squared_error(
    preds: Float[Tensor, "batch *condition timepoint observation"],
    targets: Float[Tensor, "batch *condition timepoint observation"],
) -> Float[Tensor, ""]:
    """Root mean squared error.

    :param preds:   Model predictions.
    :param targets: Targets.
    """
    return torch.sqrt(mean_squared_error(preds=preds, targets=targets))
