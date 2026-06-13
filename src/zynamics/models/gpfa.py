"""Gaussian Process Factor Analysis with an EM training adapter.

This ports the core shape of the local GPFA-MATLAB implementation:

- `gpfaEngine.m` initializes observation parameters using factor analysis.
- `exactInferenceWithLL.m` performs posterior inference and data likelihood.
- `em.m` alternates posterior inference with closed-form updates for `C`, `d`,
  and diagonal `R`.
- `learnGPparams.m` updates RBF GP timescales from E-step sufficient statistics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from pydantic import Field
from torch import Tensor

from zynamics.models.base import BaseDynamicsModel, BaseModelConfig, OptimizationConfig
from zynamics.types import LossOutput, ModelOutput, observations_from_batch


@BaseModelConfig.register
class GPFAConfig(BaseModelConfig):
    """Config for Gaussian-observation GPFA."""

    name: Literal["gpfa"] = "gpfa"
    objective: str = "negative_log_marginal_likelihood"
    latent_dim: int = 3
    bin_width: float = 20.0
    start_tau: float = 100.0
    start_eps: float = 1e-3
    min_var_frac: float = 0.01
    learn_kernel_params: bool = True
    fa_max_iters: int = 500
    fa_tol: float = 1e-8
    kernel_param_max_iters: int = 8
    kernel_param_lr: float = 1.0
    jitter: float = 1e-5
    optimization: OptimizationConfig = Field(
        default_factory=lambda: OptimizationConfig(name="em")
    )

    def build(self, n_neurons: int, n_time: int) -> "GPFA":
        return GPFA(
            n_neurons=n_neurons,
            n_time=n_time,
            latent_dim=self.latent_dim,
            bin_width=self.bin_width,
            start_tau=self.start_tau,
            start_eps=self.start_eps,
            min_var_frac=self.min_var_frac,
            learn_kernel_params=self.learn_kernel_params,
            fa_max_iters=self.fa_max_iters,
            fa_tol=self.fa_tol,
            kernel_param_max_iters=self.kernel_param_max_iters,
            kernel_param_lr=self.kernel_param_lr,
            jitter=self.jitter,
            objective=self.objective,
        )


@dataclass
class GPFAPosterior:
    latents: Tensor
    cov_t: Tensor
    cov_gp: Tensor
    cov_big: Tensor
    log_likelihood: Tensor


class GPFA(BaseDynamicsModel):
    """Gaussian Process Factor Analysis with diagonal observation noise."""

    def __init__(
        self,
        n_neurons: int,
        n_time: int,
        latent_dim: int = 3,
        bin_width: float = 20.0,
        start_tau: float = 100.0,
        start_eps: float = 1e-3,
        min_var_frac: float = 0.01,
        learn_kernel_params: bool = True,
        fa_max_iters: int = 500,
        fa_tol: float = 1e-8,
        kernel_param_max_iters: int = 8,
        kernel_param_lr: float = 1.0,
        jitter: float = 1e-5,
        objective: str = "negative_log_marginal_likelihood",
    ) -> None:
        super().__init__()
        self.n_neurons = n_neurons
        self.n_time = n_time
        self.latent_dim = latent_dim
        self.bin_width = float(bin_width)
        self.min_var_frac = float(min_var_frac)
        self.learn_kernel_params = learn_kernel_params
        self.fa_max_iters = int(fa_max_iters)
        self.fa_tol = float(fa_tol)
        self.kernel_param_max_iters = int(kernel_param_max_iters)
        self.kernel_param_lr = float(kernel_param_lr)
        self.jitter = float(jitter)
        self.objective = objective

        gamma = (bin_width / start_tau) ** 2
        self.register_buffer("gamma", torch.full((latent_dim,), float(gamma)))
        self.register_buffer("eps", torch.full((latent_dim,), float(start_eps)))
        self.register_buffer("C", torch.zeros(n_neurons, latent_dim))
        self.register_buffer("d", torch.zeros(n_neurons))
        self.register_buffer("R_diag", torch.ones(n_neurons))
        self.register_buffer("_initialized", torch.tensor(False))

    @property
    def initialized(self) -> bool:
        return bool(self._initialized.item())

    def forward(self, x: Tensor) -> ModelOutput:
        if not self.initialized:
            self.initialize(x)
        posterior = self._e_step(x.float(), get_ll=True)
        reconstruction = self._decode(posterior.latents)
        return ModelOutput(
            rates=reconstruction.clamp_min(0.0),
            latents=posterior.latents,
            reconstruction=reconstruction,
            extras={"log_likelihood": posterior.log_likelihood},
        )

    def loss(
        self,
        batch: Tensor | dict[str, Tensor],
        output: ModelOutput,
        epoch: int = 0,
    ) -> LossOutput:
        x = observations_from_batch(batch)
        total = -output.extras["log_likelihood"] / x.numel()
        return LossOutput(
            total=total,
            named_terms={"log_likelihood": output.extras["log_likelihood"]},
            objective=self.objective,
        )

    def fit_em_epoch(self, x: Tensor, epoch: int = 0) -> LossOutput:
        """Run one full E/M update and report normalized negative LL."""

        x = x.float()
        with torch.no_grad():
            if not self.initialized:
                self.initialize(x)
            posterior = self._e_step(x, get_ll=True)
            self._m_step(x, posterior)

        if self.learn_kernel_params:
            self._learn_gp_params(posterior)

        with torch.no_grad():
            total = -posterior.log_likelihood / x.numel()
            return LossOutput(
                total=total,
                named_terms={"log_likelihood": posterior.log_likelihood},
                objective=self.objective,
            )

    @torch.no_grad()
    def initialize(self, x: Tensor) -> None:
        """Initialize observation parameters with GPFA-MATLAB's fast FA EM."""

        x = x.float()
        flat = x.reshape(-1, self.n_neurons)
        n_points = flat.shape[0]
        mean = flat.mean(dim=0)
        centered = flat - mean
        c_x = centered.T @ centered / max(n_points, 1)
        c_x = 0.5 * (c_x + c_x.T)
        diag_cx = torch.clamp(torch.diagonal(c_x), min=self.jitter)
        var_floor = torch.clamp(self.min_var_frac * diag_cx, min=self.jitter)

        scale = self._factor_analysis_scale(c_x)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0)
        loadings = torch.randn(
            self.n_neurons,
            self.latent_dim,
            generator=generator,
            dtype=x.dtype,
        ).to(x.device)
        loadings = loadings * math.sqrt(scale / max(self.latent_dim, 1))
        private_var = diag_cx.clone()

        eye_z = torch.eye(self.latent_dim, device=x.device, dtype=x.dtype)
        ll_base: Tensor | None = None
        ll_old = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        for iteration in range(max(self.fa_max_iters, 1)):
            inv_private = 1.0 / torch.clamp(private_var, min=self.jitter)
            inv_private_loadings = inv_private[:, None] * loadings
            inner = eye_z + loadings.T @ inv_private_loadings
            chol_inner = torch.linalg.cholesky(
                0.5 * (inner + inner.T)
                + self.jitter * torch.eye(self.latent_dim, device=x.device, dtype=x.dtype)
            )

            beta_rhs = loadings.T * inv_private.unsqueeze(0)
            beta = torch.cholesky_solve(beta_rhs, chol_inner)
            c_x_beta = c_x @ beta.T
            expected_zz = eye_z - beta @ loadings + beta @ c_x_beta
            expected_zz = 0.5 * (expected_zz + expected_zz.T)

            logdet_sigma = torch.log(torch.clamp(private_var, min=self.jitter)).sum()
            logdet_sigma = logdet_sigma + 2.0 * torch.log(
                torch.diagonal(chol_inner)
            ).sum()
            middle = inv_private_loadings.T @ c_x @ inv_private_loadings
            trace_term = (diag_cx * inv_private).sum()
            trace_term = trace_term - torch.trace(
                torch.cholesky_solve(middle, chol_inner)
            )
            ll_current = (
                n_points * (-self.n_neurons / 2.0 * math.log(2 * math.pi))
                - 0.5 * n_points * (logdet_sigma + trace_term)
            )

            loadings_new = torch.linalg.solve(expected_zz.T, c_x_beta.T).T
            private_new = diag_cx - (c_x_beta * loadings_new).sum(dim=1)
            private_new = torch.maximum(var_floor, private_new)

            if iteration <= 1:
                ll_base = ll_current
            elif ll_base is not None:
                previous_gain = ll_old - ll_base
                current_gain = ll_current - ll_base
                if (
                    ll_current >= ll_old
                    and current_gain < (1.0 + self.fa_tol) * previous_gain
                ):
                    loadings = loadings_new
                    private_var = private_new
                    break

            loadings = loadings_new
            private_var = private_new
            ll_old = ll_current

        self.C.copy_(loadings)
        self.d.copy_(mean)
        self.R_diag.copy_(private_var + self.jitter)
        self._initialized.copy_(torch.tensor(True, device=x.device))

    def _decode(self, latents: Tensor) -> Tensor:
        return torch.einsum("btd,nd->btn", latents, self.C) + self.d

    def _e_step(self, x: Tensor, get_ll: bool) -> GPFAPosterior:
        batch_size, n_time, n_neurons = x.shape
        if n_neurons != self.n_neurons:
            raise ValueError(f"Expected {self.n_neurons} neurons, got {n_neurons}.")

        k_big, k_inv, logdet_k = self._make_k_big(n_time, x.device, x.dtype)
        r_inv = 1.0 / torch.clamp(self.R_diag.to(x.device, x.dtype), min=self.jitter)
        c = self.C.to(x.device, x.dtype)
        d = self.d.to(x.device, x.dtype)

        crinv = c.T * r_inv.unsqueeze(0)
        crinvc = crinv @ c
        block_crinvc = torch.kron(
            torch.eye(n_time, device=x.device, dtype=x.dtype),
            crinvc,
        )
        precision = k_inv + block_crinvc
        precision = 0.5 * (precision + precision.T)
        precision = precision + self.jitter * torch.eye(
            self.latent_dim * n_time, device=x.device, dtype=x.dtype
        )
        chol_precision = torch.linalg.cholesky(precision)
        cov_big = torch.cholesky_inverse(chol_precision)
        logdet_precision = 2.0 * torch.log(torch.diagonal(chol_precision)).sum()

        dif = x - d
        messages = torch.einsum("dn,btn->btd", crinv, dif)
        term1 = messages.reshape(batch_size, n_time * self.latent_dim).T
        xsm = (cov_big @ term1).T.reshape(batch_size, n_time, self.latent_dim)

        cov_t = torch.empty(
            n_time,
            self.latent_dim,
            self.latent_dim,
            device=x.device,
            dtype=x.dtype,
        )
        for t in range(n_time):
            idx = slice(t * self.latent_dim, (t + 1) * self.latent_dim)
            cov_t[t] = cov_big[idx, idx]

        cov_gp = torch.empty(
            self.latent_dim,
            n_time,
            n_time,
            device=x.device,
            dtype=x.dtype,
        )
        for dim in range(self.latent_dim):
            idx = torch.arange(dim, self.latent_dim * n_time, self.latent_dim, device=x.device)
            cov_gp[dim] = cov_big[idx[:, None], idx[None, :]]

        if get_ll:
            logdet_r = torch.log(torch.clamp(self.R_diag, min=self.jitter)).sum().to(
                x.device, x.dtype
            )
            val = (
                -n_time * logdet_r
                - logdet_k
                - logdet_precision
                - n_neurons * n_time * math.log(2 * math.pi)
            )
            dif_quad = (dif.pow(2) * r_inv).sum()
            quad = ((term1.T @ cov_big) * term1.T).sum()
            ll = 0.5 * (batch_size * val - dif_quad + quad)
        else:
            ll = torch.tensor(float("nan"), device=x.device, dtype=x.dtype)

        return GPFAPosterior(
            latents=xsm,
            cov_t=cov_t,
            cov_gp=cov_gp,
            cov_big=cov_big,
            log_likelihood=ll,
        )

    def _m_step(self, x: Tensor, posterior: GPFAPosterior) -> None:
        batch_size, n_time, _ = x.shape
        total_steps = batch_size * n_time
        xsm = posterior.latents

        sum_pauto = batch_size * posterior.cov_t.sum(dim=0)
        sum_pauto = sum_pauto + torch.einsum("btd,bte->de", xsm, xsm)

        y_mat = x.reshape(total_steps, self.n_neurons).T
        x_mat = xsm.reshape(total_steps, self.latent_dim).T
        sum_yxtrans = y_mat @ x_mat.T
        sum_xall = x_mat.sum(dim=1)
        sum_yall = y_mat.sum(dim=1)

        lhs = torch.zeros(
            self.latent_dim + 1,
            self.latent_dim + 1,
            device=x.device,
            dtype=x.dtype,
        )
        lhs[: self.latent_dim, : self.latent_dim] = sum_pauto
        lhs[: self.latent_dim, -1] = sum_xall
        lhs[-1, : self.latent_dim] = sum_xall
        lhs[-1, -1] = total_steps
        lhs = lhs + self.jitter * torch.eye(lhs.shape[0], device=x.device, dtype=x.dtype)

        rhs = torch.cat([sum_yxtrans, sum_yall[:, None]], dim=1)
        cd = torch.linalg.solve(lhs.T, rhs.T).T
        C_new = cd[:, : self.latent_dim]
        d_new = cd[:, -1]

        var_floor = torch.clamp(
            self.min_var_frac * torch.var(y_mat, dim=1, unbiased=False),
            min=self.jitter,
        )
        sum_yytrans = (y_mat * y_mat).sum(dim=1)
        yd = sum_yall * d_new
        correction = ((sum_yxtrans - d_new[:, None] * sum_xall[None, :]) * C_new).sum(
            dim=1
        )
        r = d_new.pow(2) + (sum_yytrans - 2.0 * yd - correction) / total_steps
        r = torch.clamp(r, min=var_floor)

        self.C.copy_(C_new)
        self.d.copy_(d_new)
        self.R_diag.copy_(r + self.jitter)

    def _learn_gp_params(self, posterior: GPFAPosterior) -> None:
        """Port of learnGPparams.m for RBF gamma with fixed GP noise."""

        batch_size, n_time, _ = posterior.latents.shape
        new_gamma = self.gamma.detach().clone()

        for dim in range(self.latent_dim):
            x_dim = posterior.latents[:, :, dim]
            pauto_sum = batch_size * posterior.cov_gp[dim] + x_dim.T @ x_dim
            pauto_sum = pauto_sum.detach()
            log_gamma = torch.nn.Parameter(
                torch.log(torch.clamp(self.gamma[dim].detach(), min=self.jitter))
                .to(device=pauto_sum.device, dtype=pauto_sum.dtype)
                .clone()
            )
            optimizer = torch.optim.LBFGS(
                [log_gamma],
                lr=self.kernel_param_lr,
                max_iter=max(self.kernel_param_max_iters, 1),
                line_search_fn="strong_wolfe",
            )

            def closure() -> Tensor:
                optimizer.zero_grad(set_to_none=True)
                objective = self._kernel_objective(
                    log_gamma=log_gamma,
                    pauto_sum=pauto_sum,
                    n_trials=batch_size,
                    eps=self.eps[dim].to(pauto_sum.device, pauto_sum.dtype),
                    jitter=self.jitter,
                )
                objective.backward()
                return objective

            try:
                optimizer.step(closure)
            except RuntimeError:
                continue

            updated = torch.exp(log_gamma.detach()).clamp(min=self.jitter, max=1e6)
            new_gamma[dim] = updated.to(new_gamma.device, new_gamma.dtype)

        self.gamma.copy_(new_gamma)

    def _make_k_big(
        self,
        n_time: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor]:
        total_dim = self.latent_dim * n_time
        k_big = torch.zeros(total_dim, total_dim, device=device, dtype=dtype)
        k_inv = torch.zeros_like(k_big)
        times = torch.arange(n_time, device=device, dtype=dtype)
        time_dif = times[:, None] - times[None, :]
        eye_t = torch.eye(n_time, device=device, dtype=dtype)
        logdet_k = torch.tensor(0.0, device=device, dtype=dtype)

        gamma = self.gamma.to(device=device, dtype=dtype)
        eps = self.eps.to(device=device, dtype=dtype)
        for dim in range(self.latent_dim):
            k = (1.0 - eps[dim]) * torch.exp(-0.5 * gamma[dim] * time_dif.pow(2))
            k = k + eps[dim] * eye_t + self.jitter * eye_t
            chol = torch.linalg.cholesky(k)
            k_dim_inv = torch.cholesky_inverse(chol)
            logdet_k = logdet_k + 2.0 * torch.log(torch.diagonal(chol)).sum()

            idx = torch.arange(dim, total_dim, self.latent_dim, device=device)
            k_big[idx[:, None], idx[None, :]] = k
            k_inv[idx[:, None], idx[None, :]] = k_dim_inv

        return k_big, k_inv, logdet_k

    def _factor_analysis_scale(self, covariance: Tensor) -> float:
        eye = torch.eye(covariance.shape[0], device=covariance.device, dtype=covariance.dtype)
        try:
            chol = torch.linalg.cholesky(covariance)
            scale = torch.exp(2.0 * torch.log(torch.diagonal(chol)).sum() / covariance.shape[0])
        except RuntimeError:
            evals = torch.linalg.eigvalsh(covariance)
            positive = evals[evals > self.jitter]
            if positive.numel() == 0:
                scale = torch.diagonal(covariance).mean()
            else:
                scale = torch.exp(torch.log(positive).mean())
        scale = torch.clamp(scale, min=self.jitter)
        if not torch.isfinite(scale):
            scale = torch.clamp(torch.diagonal(covariance + self.jitter * eye).mean(), min=self.jitter)
        return float(scale.detach().cpu())

    @staticmethod
    def _kernel_objective(
        log_gamma: Tensor,
        pauto_sum: Tensor,
        n_trials: int,
        eps: Tensor,
        jitter: float,
    ) -> Tensor:
        n_time = pauto_sum.shape[0]
        times = torch.arange(n_time, device=pauto_sum.device, dtype=pauto_sum.dtype)
        dif_sq = (times[:, None] - times[None, :]).pow(2)
        eye = torch.eye(n_time, device=pauto_sum.device, dtype=pauto_sum.dtype)
        gamma = torch.exp(log_gamma)
        kernel = (1.0 - eps) * torch.exp(-0.5 * gamma * dif_sq)
        kernel = kernel + eps * eye + jitter * eye
        chol = torch.linalg.cholesky(kernel)
        logdet = 2.0 * torch.log(torch.diagonal(chol)).sum()
        solve = torch.cholesky_solve(pauto_sum, chol)
        return 0.5 * n_trials * logdet + 0.5 * torch.trace(solve)
