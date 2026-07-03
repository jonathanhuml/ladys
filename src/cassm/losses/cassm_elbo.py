"""Evidence lower bound loss for computation-aware state-space models."""

import math

import torch
import torch.nn as nn


class CASSMElboLoss(nn.Module):
    """Compute the per-step CA-SSM evidence lower bound objective.

    The computation-aware filter observes a low-dimensional projection of the
    full observation vector at each timestep. This objective combines two terms:

    - an expected full-observation negative log likelihood under the current
      posterior state covariance, and
    - a KL correction between the projected Kalman update and the projected
      observation noise distribution.

    The module is parameter-free. It exists to keep the filter recursion focused
    on state propagation while making the objective easy to test, document, and
    replace with future loss variants.
    """

    def forward(
        self,
        prior_predictive_residual: torch.Tensor,
        prior_state_covariance,
        obs_noise: torch.Tensor,
        cakf_mean_message: torch.Tensor,
        mean_update_term: torch.Tensor,
        projected_noise: torch.Tensor,
        innovation_cholesky: torch.Tensor,
        use_dense_projection: bool,
    ) -> torch.Tensor:
        """Return one timestep of the normalized CA-SSM ELBO numerator.

        Parameters
        ----------
        prior_predictive_residual:
            Full residual ``y_t - H m_t^-`` with shape ``(batch, dim, 1)``.
        prior_state_covariance:
            State covariance linear operator after the timestep prediction. The
            diagonal position entries are used in the expectation term.
        obs_noise:
            Positive full observation-noise variances with shape ``(dim,)``.
        cakf_mean_message:
            Projected dual mean message ``H' S^{-1} r``.
        mean_update_term:
            State-space mean update ``P H' S^{-1} r``.
        projected_noise:
            Observation noise in projected coordinates. Dense projections pass a
            matrix; block projections pass a vector of diagonal entries.
        innovation_cholesky:
            Lower Cholesky factor of the projected innovation matrix.
        use_dense_projection:
            Whether ``projected_noise`` is dense or diagonal.

        Returns
        -------
        torch.Tensor
            Scalar loss contribution for this timestep. The caller is
            responsible for averaging across timesteps and observed dimensions.
        """
        pos_diag = prior_state_covariance.diagonal(dim1=-1, dim2=-2)[..., 0::2]
        inv_obs_noise = obs_noise.pow(-1)

        trace_cov = (inv_obs_noise * pos_diag).sum(-1)
        residual_quadratic = torch.mean(
            (prior_predictive_residual.mT * inv_obs_noise) @ prior_predictive_residual
        )
        normalizer = obs_noise.numel() * torch.log(
            2 * torch.tensor(math.pi, device=obs_noise.device, dtype=obs_noise.dtype)
        )

        expectation_term = 0.5 * (
            obs_noise.log().sum()
            + residual_quadratic
            + torch.mean(trace_cov)
            + normalizer
        )

        if use_dense_projection:
            projected_noise_matrix = projected_noise
        else:
            projected_noise_matrix = torch.diag_embed(projected_noise)
        cov_scaled_innovation = torch.cholesky_solve(
            projected_noise_matrix, innovation_cholesky, upper=False
        )

        trace_term = cov_scaled_innovation.diagonal(dim1=-2, dim2=-1).sum(-1)
        kl_term = 0.5 * (
            torch.mean(cakf_mean_message.mT @ mean_update_term)
            - torch.mean(trace_term)
            - torch.mean(torch.logdet(cov_scaled_innovation))
        )

        return kl_term + expectation_term
