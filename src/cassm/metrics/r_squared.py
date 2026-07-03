"""R-squared score."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from jaxtyping import Float

if TYPE_CHECKING:
    from torch import Tensor


def r_squared_score(
    preds: Float[Tensor, "batch num_data"], targets: Float[Tensor, "batch num_data"]
) -> Float[Tensor, "batch"]:
    """R-squared score.

    :param preds:   Model predictions.
    :param targets: Targets.
    """
    # TODO: ensure the type hint dimensions are correct and the calculation
    sum_squared_residuals = torch.sum(targets - preds, dim=-1)
    total_sum_squares = torch.sum(targets - targets.mean(dim=-1), dim=-1)
    return 1 - (sum_squared_residuals / total_sum_squares)
