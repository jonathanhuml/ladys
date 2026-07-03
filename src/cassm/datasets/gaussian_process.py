"""Dataset generated from a Gaussian process."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Optional
from jaxtyping import Float, Int
import torch
import gpytorch
import linear_operator

if TYPE_CHECKING:
    from torch import Tensor


class GaussianProcess(torch.utils.data.Dataset):
    """Gaussian process dataset.

    Time-series data generated from a Gaussian process
    with the specified kernel and Gaussian observation noise.
    The data will have shape [*batch_dims num_timepoints observation_dim].

    :param timepoints: Timepoints for the observations.
    :param batch_dims: Number of batch dimensions. Could for example correspond to the
        number of experiment trials.
    :param observation_dim: Dimensionality of the observed data. Corresponds to
        using a vector-valued Gaussian process as a latent function.
    :param mean_fn: Mean function of the latent Gaussian process.
    :param cov_fn: Covariance function of the latent Gaussian process.
    :param lengthscale: Lengthscale of the Gaussian process.
    :param noise_scale: Scale of the observation noise (i.e. standard deviation).
    """

    def __init__(
        self,
        timepoints: Float[Tensor, "timepoint"],
        batch_dims: Optional[Tuple[int]] = None,
        observation_dim: int = 1,
        mean_fn: gpytorch.means.Mean = gpytorch.means.ZeroMean(),
        cov_fn: gpytorch.kernels.Kernel = gpytorch.kernels.MaternKernel(nu=1.5),
        lengthscale: float = 0.1,
        noise_scale: float = 0.3,
    ):

        with torch.no_grad():
            self.timepoints = timepoints
            self.observation_dim = observation_dim

            try:
                iter(batch_dims)
                self.batch_dims = batch_dims
            except TypeError:
                if batch_dims is None:
                    self.batch_dims = ()
                else:
                    self.batch_dims = (batch_dims,)

            # Gaussian process
            if observation_dim == 1:
                self.mean_fn = mean_fn
                self.cov_fn = cov_fn
                self.cov_fn.lengthscale = lengthscale
            else:
                self.mean_fn = gpytorch.means.MultitaskMean(
                    base_means=mean_fn, num_tasks=self.observation_dim
                )
                cov_fn.lengthscale = lengthscale
                self.cov_fn = gpytorch.kernels.MultitaskKernel(
                    data_covar_module=cov_fn, num_tasks=self.observation_dim
                )

            # Sample data from GP
            kernel_matrix = self.cov_fn(self.timepoints).to_dense()
            kernel_matrix_chol_factor = (
                linear_operator.utils.cholesky.psd_safe_cholesky(kernel_matrix)
            )
            gp_prior_draw = (
                self.mean_fn(self.timepoints).reshape(-1, 1)
                + kernel_matrix_chol_factor
                @ torch.randn((len(self.timepoints), self.observation_dim)).reshape(
                    -1, 1
                )
            ).reshape((len(self.timepoints), self.observation_dim))
            self.observations = gp_prior_draw + noise_scale * torch.randn(
                self.batch_dims + (len(self.timepoints), self.observation_dim)
            )

    def __getitem__(self, index) -> Tuple[
        Float[Tensor, "timepoint"],
        Int[Tensor, "*batch timepoint observation"],
    ]:
        if self.batch_dims == ():
            return (self.timepoints[index], self.observations[index])
        return (
            self.timepoints,
            self.observations[index],
        )

    def __len__(self) -> int:
        return self.observations.shape[0]
