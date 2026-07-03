"""Sparse block-diagonal linear operator used by CA-SSM projections."""

from typing import Union

import torch
from jaxtyping import Float
from linear_operator.operators import AddedDiagLinearOperator, DiagLinearOperator
from linear_operator.operators._linear_operator import LinearOperator
from torch import Tensor


class BlockDiagonalSparseLinearOperator(LinearOperator):
    """Linear operator with sparse rows that become block diagonal after reordering.

    The operator stores the non-zero entries for each output row in ``blocks`` and
    the corresponding input indices in ``non_zero_idcs``. It currently supports
    equally sized ``1 x NNZ`` blocks, which is the projection structure used by
    the computation-aware filter.
    """

    def __init__(
        self,
        non_zero_idcs: Float[torch.Tensor, "M NNZ"],
        blocks: Float[torch.Tensor, "M NNZ"],
        size_input_dim: int,
    ):
        super().__init__(non_zero_idcs, blocks, size_input_dim=size_input_dim)
        self.non_zero_idcs = torch.atleast_2d(non_zero_idcs)
        self.non_zero_idcs.requires_grad = False
        self.blocks = torch.atleast_2d(blocks)
        self.size_input_dim = size_input_dim

    def _matmul(
        self: Float[LinearOperator, "*batch M N"],
        rhs: Float[torch.Tensor, "*batch2 N C"],
    ) -> Union[Float[torch.Tensor, "... M C"], Float[torch.Tensor, "... M"]]:
        if isinstance(rhs, AddedDiagLinearOperator):
            return self._matmul(rhs._linear_op) + self._matmul(rhs._diag_tensor)

        if isinstance(rhs, DiagLinearOperator):
            return BlockDiagonalSparseLinearOperator(
                non_zero_idcs=self.non_zero_idcs,
                blocks=rhs.diag()[self.non_zero_idcs] * self.blocks,
                size_input_dim=self.size_input_dim,
            ).to_dense()

        rhs_non_zero = rhs[..., self.non_zero_idcs, :]

        if rhs.ndim == 2 and rhs.shape[-1] == 1:
            return (self.blocks.unsqueeze(-1) * rhs_non_zero).sum(dim=-2)

        return (self.blocks.unsqueeze(-2) @ rhs_non_zero).squeeze(-2)

    def _size(self) -> torch.Size:
        return torch.Size((self.non_zero_idcs.shape[0], self.size_input_dim))

    def to_dense(self: LinearOperator) -> Tensor:
        if self.size() == self.blocks.shape:
            return self.blocks
        return torch.zeros(
            (self.blocks.shape[0], self.size_input_dim),
            dtype=self.blocks.dtype,
            device=self.blocks.device,
        ).scatter_(src=self.blocks, index=self.non_zero_idcs, dim=1)
