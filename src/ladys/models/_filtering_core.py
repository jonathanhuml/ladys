"""Internal filtering cores adapted from the CASSM implementation."""

from __future__ import annotations

import math
import os
from random import randint
import warnings

import gpytorch
from linear_operator import operators
from linear_operator.operators import (
    AddedDiagLinearOperator,
    DiagLinearOperator,
    IdentityLinearOperator,
    KroneckerProductLinearOperator,
)
from linear_operator.operators._linear_operator import LinearOperator
import torch
from torch import Tensor, nn


class CASSMElboLoss(nn.Module):
    """Per-step CA-SSM evidence lower bound objective."""

    def forward(
        self,
        prior_predictive_residual: Tensor,
        prior_state_covariance,
        obs_noise: Tensor,
        cakf_mean_message: Tensor,
        mean_update_term: Tensor,
        projected_noise: Tensor,
        innovation_cholesky: Tensor,
        use_dense_projection: bool,
    ) -> Tensor:
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

        projected_noise_matrix = (
            projected_noise if use_dense_projection else torch.diag_embed(projected_noise)
        )
        cov_scaled_innovation = torch.cholesky_solve(
            projected_noise_matrix,
            innovation_cholesky,
            upper=False,
        )

        trace_term = cov_scaled_innovation.diagonal(dim1=-2, dim2=-1).sum(-1)
        kl_term = 0.5 * (
            torch.mean(cakf_mean_message.mT @ mean_update_term)
            - torch.mean(trace_term)
            - torch.mean(torch.logdet(cov_scaled_innovation))
        )

        return kl_term + expectation_term


class BlockDiagonalSparseLinearOperator(LinearOperator):
    """Sparse projection operator used by the computation-aware filter."""

    def __init__(
        self,
        non_zero_idcs: Tensor,
        blocks: Tensor,
        size_input_dim: int,
    ) -> None:
        super().__init__(non_zero_idcs, blocks, size_input_dim=size_input_dim)
        self.non_zero_idcs = torch.atleast_2d(non_zero_idcs)
        self.non_zero_idcs.requires_grad = False
        self.blocks = torch.atleast_2d(blocks)
        self.size_input_dim = size_input_dim

    def _matmul(self, rhs):
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

    def to_dense(self) -> Tensor:
        if self.size() == self.blocks.shape:
            return self.blocks
        return torch.zeros(
            (self.blocks.shape[0], self.size_input_dim),
            dtype=self.blocks.dtype,
            device=self.blocks.device,
        ).scatter_(src=self.blocks, index=self.non_zero_idcs, dim=1)


def _svd_f_and_t_inv(s: Tensor) -> tuple[Tensor, Tensor]:
    s2 = s * s
    s2 = torch.where(s2 < 1e-30, torch.zeros_like(s2), s2)
    s_i = s2[..., :, None]
    s_j = s2[..., None, :]
    denom = s_j - s_i
    both_zero = (s_i == 0) & (s_j == 0)
    eq_mask = denom == 0

    inv = torch.where(eq_mask | both_zero, torch.zeros_like(denom), 1.0 / denom)
    f_matrix = torch.where(torch.isfinite(inv), inv, torch.zeros_like(inv))
    logi = f_matrix.abs() > 1e30
    f_matrix = torch.where(logi, torch.zeros_like(f_matrix), f_matrix)

    k = s.shape[-1]
    diag_mask = torch.eye(k, dtype=torch.bool, device=s.device)
    eq_offdiag = eq_mask & (~diag_mask)

    inv_sj = (1.0 / s)[..., None, :]
    t_matrix = torch.zeros_like(f_matrix)
    t_matrix = torch.where(eq_offdiag & ~both_zero, inv_sj, t_matrix)
    t_matrix = torch.where(logi, inv_sj, t_matrix)

    return f_matrix, t_matrix


class _SVDInv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        u, s_vec, vh = torch.linalg.svd(x, full_matrices=False)
        v = vh.mH
        ctx.save_for_backward(x, u, s_vec, v)
        return u, s_vec, v

    @staticmethod
    def backward(ctx, dl_du, dl_ds_vec, dl_dv):
        _, u, s_vec, v = ctx.saved_tensors
        dtype = u.dtype
        s_inv_vec = torch.where(s_vec > 0, 1.0 / s_vec, torch.zeros_like(s_vec))

        utdu = u.mH @ dl_du
        vtdv = v.mH @ dl_dv
        f_matrix, t_matrix = _svd_f_and_t_inv(s_vec)
        f_matrix = f_matrix.to(utdu.dtype)
        t_matrix = t_matrix.to(utdu.dtype)
        fmat_u = f_matrix * (utdu - utdu.mH)
        fmat_v = f_matrix * (vtdv - vtdv.mH)
        c_u1 = (fmat_u * s_vec[..., None, :]) + (t_matrix * utdu)
        c_u1 = u @ c_u1

        dl_du_sinv = dl_du * s_inv_vec[..., None, :]
        c_u2 = dl_du_sinv - u @ (u.mH @ dl_du_sinv)

        c_u = (c_u1 + c_u2) @ v.mH
        c_s = (u * dl_ds_vec.to(dtype)[..., None, :]) @ v.mH

        c_v1 = (s_vec[..., :, None] * fmat_v) @ v.mH
        tmp = (s_inv_vec[..., :, None] * dl_dv.mH)
        c_v2 = tmp - tmp @ v @ v.mH

        return c_u + c_s + (u @ (c_v1 + c_v2))


def _matern32_time_process_cov(
    delta_t: Tensor,
    sigma_f2: Tensor,
    ell: Tensor,
) -> Tensor:
    lam = torch.sqrt(torch.tensor(3.0, device=delta_t.device)) / ell
    rho2 = torch.exp(-2.0 * lam * delta_t)
    u = lam * delta_t

    q11 = 3.0 * sigma_f2 * (1.0 - rho2 * (1.0 + 2.0 * u + 2.0 * u**2))
    q22 = 3.0 * sigma_f2 * lam**2 * (1.0 - rho2 * (1.0 - 2.0 * u + 2.0 * u**2))
    q12 = 6.0 * sigma_f2 * lam**3 * delta_t**2 * rho2

    return torch.stack(
        [torch.stack([q11, q12], -1), torch.stack([q12, q22], -1)],
        dim=-2,
    )


def _matern32_transition_matrix(delta_t: Tensor, ell: Tensor) -> Tensor:
    lam = torch.sqrt(torch.tensor(3.0, device=delta_t.device)) / ell
    zero = torch.zeros_like(lam)
    one = torch.ones_like(lam)
    f_time = torch.stack(
        [torch.stack([zero, one]), torch.stack([-(lam**2), -2.0 * lam])],
    ).squeeze(-1)
    return torch.matrix_exp(f_time * delta_t)


def _matern32_time_stationary_cov(sigma_f2: Tensor, ell: Tensor) -> Tensor:
    lam = torch.sqrt(torch.tensor(3.0, device=sigma_f2.device)) / ell
    return torch.stack(
        [
            torch.stack([sigma_f2, torch.zeros_like(sigma_f2)], -1),
            torch.stack([torch.zeros_like(sigma_f2), lam**2 * sigma_f2], -1),
        ],
        -2,
    )


def _log_marginal_likelihood(residual: Tensor, y_cholesky: Tensor) -> Tensor:
    n_neurons = residual.shape[1]
    loss1 = torch.mean(
        0.5
        * residual.transpose(1, 2)
        @ torch.cholesky_solve(input=residual, input2=y_cholesky, upper=False)
    )
    cholesky_diags = torch.diagonal(y_cholesky, offset=0, dim1=1, dim2=2)
    loss2 = torch.mean(torch.sum(torch.log(cholesky_diags), dim=1))
    loss3 = 0.5 * torch.log(2 * torch.tensor(math.pi, device=residual.device)) * n_neurons
    return loss1 + loss2 + loss3


def _random_run_id(length: int = 6) -> str:
    return "".join(str(randint(0, 9)) for _ in range(length))


class ComputationAwareFilterSmoother(nn.Module):
    """Sparse computation-aware state-space filter used by LaDyS CASSM."""

    def __init__(
        self,
        projection_dim: int,
        nneurons: int,
        timesteps: int,
        device: torch.device,
        dt: float = 1.0,
        dataset_name: str | None = None,
        spatial_prior: Tensor | None = None,
        save_model: bool = False,
        use_dense_projection: bool = False,
    ) -> None:
        super().__init__()

        self.dim = nneurons
        self.projection_dim = projection_dim
        self.remainder = self.dim % self.projection_dim
        if self.remainder != 0 and not use_dense_projection:
            warnings.warn(
                "Number of neurons is not divisible by projection dimension. "
                f"Throwing away {self.remainder} neurons.",
                stacklevel=2,
            )
            self.dim -= self.remainder

        self.state_dim = 2 * self.dim
        self.t = timesteps
        self.device = device
        self.save_model = save_model
        self.dt = torch.tensor(float(dt), device=device)

        self.raw_sigma_f = nn.Parameter(1e-1 * torch.ones(1, device=device))
        self.raw_ell = nn.Parameter(1e-1 * torch.ones(1, device=device))
        self.softplus = nn.Softplus()
        self.loss_fn = CASSMElboLoss()

        if spatial_prior is not None:
            self.latent_locations = spatial_prior.float().to(device)
        else:
            self.latent_locations = nn.Parameter(
                torch.randn(self.dim, 3, device=device)
                / torch.sqrt(torch.tensor(float(self.dim), device=device))
            )

        self.spatial_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=3)
        )
        self.spatial_kernel(self.latent_locations)

        self.use_dense_projection = use_dense_projection
        self.obs_noise_values = nn.Parameter(1e-2 * torch.ones(self.dim, device=device))
        self.belief_initial_state = nn.Parameter(torch.empty(self.state_dim, 1, device=device))

        if self.use_dense_projection:
            self.dense_projection = nn.Parameter(
                torch.empty(self.projection_dim, self.dim, device=device)
            )
            nn.init.orthogonal_(self.dense_projection)
        else:
            self.projection = nn.Parameter(
                torch.ones(
                    (self.projection_dim, self.dim // self.projection_dim),
                    device=device,
                )
            )
            self.projection_indices = torch.arange(device=device, end=self.dim).reshape(
                self.projection_dim,
                self.dim // self.projection_dim,
            )

        self.observation_matrix = KroneckerProductLinearOperator(
            IdentityLinearOperator(self.dim, device=device),
            torch.tensor([[1.0, 0.0]], device=device),
        )

        if self.save_model:
            if dataset_name is None:
                raise ValueError("dataset_name is required when save_model=True.")
            self.save_path = f"./cassm_runs/{dataset_name}/run_id_{_random_run_id()}/"
            os.makedirs(self.save_path, exist_ok=True)

    def _build_dynamics(self):
        ell = self.softplus(self.raw_ell)
        sigma_f2 = self.softplus(self.raw_sigma_f)

        a_t = _matern32_transition_matrix(self.dt, ell)
        transition_matrix = KroneckerProductLinearOperator(
            IdentityLinearOperator(self.dim, device=self.device),
            a_t,
        )

        sigma_inf_t = _matern32_time_stationary_cov(sigma_f2, ell)
        sigma_inf_op = KroneckerProductLinearOperator(
            self.spatial_kernel(self.latent_locations),
            sigma_inf_t,
        )

        return transition_matrix, sigma_inf_op

    def _build_projected_obs(self):
        if self.use_dense_projection:
            h_proj = (self.dense_projection @ self.observation_matrix).unsqueeze(0)
            r_proj = (
                self.dense_projection
                * self.softplus(self.obs_noise_values)
                @ self.dense_projection.mT
            )
        else:
            projection = BlockDiagonalSparseLinearOperator(
                non_zero_idcs=self.projection_indices,
                blocks=self.projection,
                size_input_dim=self.dim,
            )
            h_proj = (projection @ self.observation_matrix).to_dense().unsqueeze(0)
            r_proj = (
                self.projection.pow(2)
                * self.softplus(self.obs_noise_values)[self.projection_indices]
            ).sum(dim=1)

        return h_proj, r_proj

    def _truncate_downdate(self, matrix: Tensor) -> Tensor:
        u, s, _ = _SVDInv.apply(matrix)
        return u[..., : self.projection_dim] * s[..., : self.projection_dim].unsqueeze(-2)

    def filter(self, data: Tensor, return_type: str = "forward"):
        num_trials, time_steps = data.shape[:2]

        if self.remainder != 0:
            data = data[:, :, : self.dim]

        if return_type == "prediction":
            updated_belief_state_means = torch.empty(
                (num_trials, time_steps, self.state_dim),
                device=self.device,
            )
            updated_belief_obs_vars = torch.empty(
                (num_trials, time_steps, self.dim),
                device=self.device,
            )

        prior_belief_state_mean = self.belief_initial_state.unsqueeze(0).expand(
            num_trials,
            -1,
            -1,
        )

        downdate_sqrt = torch.zeros(
            size=(1, self.state_dim, self.projection_dim),
            device=self.device,
        )
        loss = torch.tensor(0.0, device=self.device)

        transition_matrix, sigma_inf_op = self._build_dynamics()
        h_proj, r_proj = self._build_projected_obs()

        prior_belief_state_cov_op = sigma_inf_op - operators.RootLinearOperator(
            downdate_sqrt
        )

        for t in range(time_steps):
            tmp = prior_belief_state_cov_op.matmul(h_proj.mT)
            innovation_matrix = h_proj @ tmp
            if self.use_dense_projection:
                innovation_matrix = innovation_matrix + r_proj.unsqueeze(0)
            else:
                innovation_matrix.diagonal(dim1=-2, dim2=-1).add_(r_proj)

            prior_predictive_residual = (
                data[:, t, :].unsqueeze(-1)
                - self.observation_matrix @ prior_belief_state_mean
            )

            if self.use_dense_projection:
                projected_residual = self.dense_projection @ prior_predictive_residual
            else:
                projected_residual = (
                    (
                        self.projection
                        * prior_predictive_residual.squeeze(-1)[
                            ...,
                            self.projection_indices,
                        ]
                    )
                    .sum(-1)
                    .unsqueeze(-1)
                )

            cholesky = torch.linalg.cholesky(innovation_matrix, upper=False)

            cakf_mean_message = h_proj.mT @ torch.cholesky_solve(
                projected_residual,
                cholesky,
                upper=False,
            )
            cakf_cov_message = torch.linalg.solve_triangular(
                cholesky,
                h_proj,
                upper=False,
            ).mT

            mean_update_term = prior_belief_state_cov_op.matmul(cakf_mean_message)
            updated_belief_state_mean = prior_belief_state_mean + mean_update_term
            scaled_cov = prior_belief_state_cov_op.matmul(cakf_cov_message)

            m_trunc = self._truncate_downdate(torch.cat([downdate_sqrt, scaled_cov], dim=-1))
            prior_belief_state_mean = transition_matrix @ updated_belief_state_mean
            downdate_sqrt = transition_matrix @ m_trunc
            prior_belief_state_cov_op = (
                prior_belief_state_cov_op.linear_ops[0]
                - operators.RootLinearOperator(downdate_sqrt)
            )

            loss = loss + self.loss_fn(
                prior_predictive_residual=prior_predictive_residual,
                prior_state_covariance=prior_belief_state_cov_op,
                obs_noise=self.softplus(self.obs_noise_values),
                cakf_mean_message=cakf_mean_message,
                mean_update_term=mean_update_term,
                projected_noise=r_proj,
                innovation_cholesky=cholesky,
                use_dense_projection=self.use_dense_projection,
            )

            if return_type == "prediction":
                pos_diag = prior_belief_state_cov_op.diagonal(dim1=-1, dim2=-2)[..., 0::2]
                updated_belief_obs_vars[:, t, :] = pos_diag + self.softplus(
                    self.obs_noise_values
                )
                updated_belief_state_means[:, t, :] = updated_belief_state_mean[:, :, 0]

        loss = loss * (1 / time_steps) * (1 / self.dim)

        if return_type == "forward":
            return loss
        if return_type == "prediction":
            return updated_belief_state_means, updated_belief_obs_vars
        raise ValueError(
            f"Unknown return_type '{return_type}'. Expected: forward | prediction"
        )

    def forward(self, data: Tensor) -> Tensor:
        return self.filter(data, return_type="forward")


class DenseKalmanFilterSmoother(nn.Module):
    """Dense Kalman filter baseline used by LaDyS Kalman."""

    def __init__(
        self,
        nneurons: int,
        timesteps: int,
        device: torch.device,
        dt: float = 1.0,
        dataset_name: str | None = None,
        save_model: bool = False,
    ) -> None:
        super().__init__()

        self.dim = nneurons
        self.latent_dim = self.dim
        self.state_dim = 2 * self.dim
        self.t = timesteps
        self.device = device
        self.save_model = save_model
        self.dt = torch.tensor(float(dt), device=device)

        self.raw_sigma_f = nn.Parameter(1e-1 * torch.ones(1, device=device))
        self.raw_ell = nn.Parameter(1e-1 * torch.ones(1, device=device))
        self.softplus = nn.Softplus()

        self.latent_locations = nn.Parameter(
            torch.arange(self.dim, device=device).float().unsqueeze(-1)
        )
        self.spatial_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5)
        )
        self.obs_noise_values = nn.Parameter(1e-1 * torch.ones(self.dim, device=device))

        if self.save_model:
            if dataset_name is None:
                raise ValueError("dataset_name is required when save_model=True.")
            self.save_path = f"./kalman_runs/{dataset_name}/run_id_{_random_run_id()}/"
            os.makedirs(self.save_path, exist_ok=True)

    def build_matern_observation_matrix(self) -> Tensor:
        eye = torch.eye(self.dim, device=self.device)
        h_time = torch.tensor([[1.0, 0.0]], device=self.device)
        return torch.kron(eye, h_time)

    def spatial_cov(self) -> Tensor:
        return self.spatial_kernel(self.latent_locations).to_dense()

    def filter(
        self,
        data: Tensor,
        return_type: str = "for_prediction",
        holdout: bool = False,
    ):
        num_trials, time_steps = data.shape[:2]

        updated_belief_state_means = torch.zeros(
            size=(num_trials, time_steps, self.state_dim),
            device=self.device,
        )
        updated_belief_state_covs = torch.zeros(
            size=(1, time_steps, self.state_dim, self.state_dim),
            device=self.device,
        )

        prior_belief_state_mean = torch.zeros(
            size=(num_trials, self.state_dim, 1),
            device=self.device,
        )
        prior_belief_state_cov = IdentityLinearOperator(
            self.state_dim,
            batch_shape=(1,),
            device=self.device,
        )
        loss = torch.tensor(0.0, device=self.device)

        ell = self.softplus(self.raw_ell)
        sigma_f2 = self.softplus(self.raw_sigma_f)
        obs_noise = self.softplus(self.obs_noise_values)

        a_t = _matern32_transition_matrix(self.dt, ell)
        transition_matrix = torch.kron(
            torch.eye(self.latent_dim, device=self.device),
            a_t,
        ).unsqueeze(0)

        q_t = _matern32_time_process_cov(self.dt, sigma_f2, ell)
        process_noise = torch.kron(self.spatial_cov(), q_t)

        if holdout:
            n_held_in = data.shape[2]
            truncated = torch.eye(self.dim, device=self.device)[:n_held_in, :]
            h_time = torch.tensor([[1.0, 0.0]], device=self.device)
            observation_matrix = torch.kron(truncated, h_time).unsqueeze(0)
            obs_noise = obs_noise[:n_held_in]
        else:
            observation_matrix = self.build_matern_observation_matrix().unsqueeze(0)

        for t in range(time_steps):
            innovation_matrix = (
                observation_matrix
                @ prior_belief_state_cov
                @ observation_matrix.mT
            )
            innovation_matrix.diagonal(dim1=-2, dim2=-1).add_(obs_noise)

            prior_predictive_residual = (
                data[:, t, :].unsqueeze(-1)
                - torch.matmul(observation_matrix, prior_belief_state_mean)
            )

            innovation_cholesky = torch.linalg.cholesky(innovation_matrix, upper=False)
            innovation_inverse_obs_matrix = torch.cholesky_solve(
                input=observation_matrix,
                input2=innovation_cholesky,
                upper=False,
            )
            kalman_gain = (
                prior_belief_state_cov
                @ innovation_inverse_obs_matrix.transpose(1, 2)
            )

            updated_belief_state_mean = prior_belief_state_mean + torch.matmul(
                kalman_gain,
                prior_predictive_residual,
            )
            joseph_gain = kalman_gain @ observation_matrix
            joseph_gain.diagonal(dim1=-2, dim2=-1).sub_(1.0)
            updated_belief_state_cov = (
                joseph_gain
                @ prior_belief_state_cov
                @ joseph_gain.transpose(1, 2)
                + kalman_gain * obs_noise @ kalman_gain.transpose(1, 2)
            )

            prior_belief_state_mean = torch.matmul(
                transition_matrix,
                updated_belief_state_mean,
            )
            prior_belief_state_cov = (
                transition_matrix
                @ updated_belief_state_cov
                @ transition_matrix.transpose(1, 2)
                + process_noise
            )

            loss = loss + _log_marginal_likelihood(
                prior_predictive_residual,
                innovation_cholesky,
            )

            updated_belief_state_means[:, t, :] = updated_belief_state_mean[:, :, 0]
            updated_belief_state_covs[:, t, :, :] = updated_belief_state_cov

        loss = loss / (time_steps * self.dim)

        if return_type == "for_forward":
            return loss
        return updated_belief_state_means, updated_belief_state_covs

    def forward(self, data: Tensor) -> Tensor:
        return self.filter(data, return_type="for_forward")
