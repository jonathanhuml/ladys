"""LFADS adapter for raw-count neural population modeling."""

from __future__ import annotations

import math
from typing import Literal

import torch
from pydantic import Field
from torch import Tensor, nn
import torch.nn.functional as F

from ladys.models.base import BaseDynamicsModel, BaseModelConfig, OptimizationConfig
from ladys.types import LossOutput, ModelOutput, observations_from_batch


def _kl_diag_gaussian(
    posterior_mean: Tensor,
    posterior_logvar: Tensor,
    prior_mean: Tensor,
    prior_logvar: Tensor,
) -> Tensor:
    """KL divergence from posterior diagonal Gaussian to prior diagonal Gaussian."""

    return 0.5 * (
        prior_logvar
        - posterior_logvar
        + torch.exp(posterior_logvar - prior_logvar)
        + (posterior_mean - prior_mean).pow(2) / torch.exp(prior_logvar)
        - 1.0
    ).sum(dim=-1)


@BaseModelConfig.register
class LFADSConfig(BaseModelConfig):
    """Config for LFADS on raw spike-count observations."""

    name: Literal["lfads"] = "lfads"
    objective: str = "lfads_elbo"
    generator_dim: int = 64
    inferred_input_dim: int = 2
    factor_dim: int = 20
    g0_encoder_dim: int = 64
    controller_encoder_dim: int = 64
    controller_dim: int = 64
    g0_prior_kappa: float = 0.1
    inferred_input_prior_kappa: float = 0.1
    keep_prob: float = 0.95
    clip_val: float = 5.0
    dt: float = 1.0
    log_rate_min: float = -8.0
    log_rate_max: float = 8.0
    posterior_logvar_min: float = math.log(1e-4)
    posterior_logvar_max: float = 5.0
    use_log1p_encoder_inputs: bool = True
    initialize_log_rate_bias: bool = True
    prediction_samples: int = 1
    kl_weight_schedule_start: int = 0
    kl_weight_schedule_dur: int = 2_000
    l2_weight_schedule_start: int = 0
    l2_weight_schedule_dur: int = 2_000
    l2_gen_scale: float = 0.0
    l2_con_scale: float = 0.0
    optimization: OptimizationConfig = Field(
        default_factory=lambda: OptimizationConfig(
            name="gradient",
            optimizer="Adam",
            lr=1e-3,
            weight_decay=0.0,
            gradient_clip=200.0,
        )
    )

    def build(self, n_neurons: int, n_time: int) -> "LFADS":
        return LFADS(
            n_neurons=n_neurons,
            n_time=n_time,
            generator_dim=self.generator_dim,
            inferred_input_dim=self.inferred_input_dim,
            factor_dim=self.factor_dim,
            g0_encoder_dim=self.g0_encoder_dim,
            controller_encoder_dim=self.controller_encoder_dim,
            controller_dim=self.controller_dim,
            g0_prior_kappa=self.g0_prior_kappa,
            inferred_input_prior_kappa=self.inferred_input_prior_kappa,
            keep_prob=self.keep_prob,
            clip_val=self.clip_val,
            dt=self.dt,
            log_rate_min=self.log_rate_min,
            log_rate_max=self.log_rate_max,
            posterior_logvar_min=self.posterior_logvar_min,
            posterior_logvar_max=self.posterior_logvar_max,
            use_log1p_encoder_inputs=self.use_log1p_encoder_inputs,
            initialize_log_rate_bias=self.initialize_log_rate_bias,
            prediction_samples=self.prediction_samples,
            kl_weight_schedule_start=self.kl_weight_schedule_start,
            kl_weight_schedule_dur=self.kl_weight_schedule_dur,
            l2_weight_schedule_start=self.l2_weight_schedule_start,
            l2_weight_schedule_dur=self.l2_weight_schedule_dur,
            l2_gen_scale=self.l2_gen_scale,
            l2_con_scale=self.l2_con_scale,
            objective=self.objective,
        )


class LFADS(BaseDynamicsModel):
    """Latent Factor Analysis via Dynamical Systems for binned spike counts.

    ## When to use

    Use LFADS as a nonlinear variational sequence model for neural population
    spike counts. This implementation adapts the LFADS demo architecture into
    the LaDyS model contract: bidirectional encoders infer a generator initial
    condition and controller context, a generator GRU produces latent factors,
    and an exponential readout returns Poisson firing rates.

    ## Assumptions

    LFADS expects raw nonnegative spike counts. Dataset-level smoothing should
    be disabled for this model. The optional `log1p` encoder transform only
    changes the recognition network input; the reconstruction loss still uses
    raw counts and the returned `rates` remain in the generated Lorenz rate
    space. The default readout bias initialization uses observed count means,
    not generated ground-truth rates.

    ## Outputs

    `forward` returns nonnegative rate predictions, latent factor trajectories,
    and variational diagnostics in `extras`. `loss` computes a Poisson
    reconstruction objective plus scheduled KL and optional recurrent L2 terms.
    """

    def __init__(
        self,
        n_neurons: int,
        n_time: int,
        generator_dim: int = 64,
        inferred_input_dim: int = 2,
        factor_dim: int = 20,
        g0_encoder_dim: int = 64,
        controller_encoder_dim: int = 64,
        controller_dim: int = 64,
        g0_prior_kappa: float = 0.1,
        inferred_input_prior_kappa: float = 0.1,
        keep_prob: float = 0.95,
        clip_val: float = 5.0,
        dt: float = 1.0,
        log_rate_min: float = -8.0,
        log_rate_max: float = 8.0,
        posterior_logvar_min: float = math.log(1e-4),
        posterior_logvar_max: float = 5.0,
        use_log1p_encoder_inputs: bool = True,
        initialize_log_rate_bias: bool = True,
        prediction_samples: int = 1,
        kl_weight_schedule_start: int = 0,
        kl_weight_schedule_dur: int = 2_000,
        l2_weight_schedule_start: int = 0,
        l2_weight_schedule_dur: int = 2_000,
        l2_gen_scale: float = 0.0,
        l2_con_scale: float = 0.0,
        objective: str = "lfads_elbo",
    ) -> None:
        super().__init__()
        self.n_neurons = int(n_neurons)
        self.n_time = int(n_time)
        self.generator_dim = int(generator_dim)
        self.inferred_input_dim = int(inferred_input_dim)
        self.factor_dim = int(factor_dim)
        self.g0_encoder_dim = int(g0_encoder_dim)
        self.controller_encoder_dim = int(controller_encoder_dim)
        self.controller_dim = int(controller_dim)
        self.keep_prob = float(keep_prob)
        self.clip_val = float(clip_val)
        self.dt = float(dt)
        self.log_rate_min = float(log_rate_min)
        self.log_rate_max = float(log_rate_max)
        self.posterior_logvar_min = float(posterior_logvar_min)
        self.posterior_logvar_max = float(posterior_logvar_max)
        self.use_log1p_encoder_inputs = bool(use_log1p_encoder_inputs)
        self.initialize_log_rate_bias = bool(initialize_log_rate_bias)
        self.prediction_samples = int(prediction_samples)
        self.kl_weight_schedule_start = int(kl_weight_schedule_start)
        self.kl_weight_schedule_dur = int(kl_weight_schedule_dur)
        self.l2_weight_schedule_start = int(l2_weight_schedule_start)
        self.l2_weight_schedule_dur = int(l2_weight_schedule_dur)
        self.l2_gen_scale = float(l2_gen_scale)
        self.l2_con_scale = float(l2_con_scale)
        self.objective = objective

        if not 0.0 < self.keep_prob <= 1.0:
            raise ValueError("keep_prob must be in (0, 1].")
        if self.inferred_input_dim < 1:
            raise ValueError("inferred_input_dim must be positive.")

        self.g0_encoder = nn.GRU(
            input_size=self.n_neurons,
            hidden_size=self.g0_encoder_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.controller_encoder = nn.GRU(
            input_size=self.n_neurons,
            hidden_size=self.controller_encoder_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.controller = nn.GRUCell(
            input_size=2 * self.controller_encoder_dim + self.factor_dim,
            hidden_size=self.controller_dim,
        )
        self.generator = nn.GRUCell(
            input_size=self.inferred_input_dim,
            hidden_size=self.generator_dim,
        )

        self.fc_g0_mean = nn.Linear(2 * self.g0_encoder_dim, self.generator_dim)
        self.fc_g0_logvar = nn.Linear(2 * self.g0_encoder_dim, self.generator_dim)
        self.fc_u_mean = nn.Linear(self.controller_dim, self.inferred_input_dim)
        self.fc_u_logvar = nn.Linear(self.controller_dim, self.inferred_input_dim)
        self.fc_factors = nn.Linear(self.generator_dim, self.factor_dim)
        self.fc_log_rates = nn.Linear(self.factor_dim, self.n_neurons)
        self.dropout = nn.Dropout(1.0 - self.keep_prob)

        self.g0_prior_mean = nn.Parameter(torch.tensor(0.0))
        self.u_prior_mean = nn.Parameter(torch.tensor(0.0))
        self.g0_prior_logvar = nn.Parameter(torch.tensor(math.log(g0_prior_kappa)))
        self.u_prior_logvar = nn.Parameter(torch.tensor(math.log(inferred_input_prior_kappa)))
        self.register_buffer("_train_step", torch.zeros((), dtype=torch.long))
        self.register_buffer("_rate_bias_initialized", torch.tensor(False))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.GRU):
                for name, parameter in module.named_parameters():
                    if "weight_ih" in name:
                        parameter.data.normal_(std=parameter.shape[1] ** -0.5)
                    elif "weight_hh" in name:
                        parameter.data.normal_(std=parameter.shape[1] ** -0.5)
                    elif "bias" in name:
                        parameter.data.zero_()
            elif isinstance(module, nn.GRUCell):
                module.weight_ih.data.normal_(std=module.weight_ih.shape[1] ** -0.5)
                module.weight_hh.data.normal_(std=module.weight_hh.shape[1] ** -0.5)
                module.bias_ih.data.zero_()
                module.bias_hh.data.zero_()
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(std=module.in_features ** -0.5)
                module.bias.data.zero_()
        self.fc_factors.weight.data = F.normalize(self.fc_factors.weight.data, dim=1)

    def forward(self, x: Tensor) -> ModelOutput:
        return self._forward(x, sample=self.training)

    def loss(
        self,
        batch: Tensor | dict[str, Tensor],
        output: ModelOutput,
        epoch: int = 0,
    ) -> LossOutput:
        target = observations_from_batch(batch).to(device=self.device, dtype=output.rates.dtype)
        dt = self._batch_dt(batch, target).to(device=self.device, dtype=output.rates.dtype)
        spike_means = (output.rates * dt).clamp_min(1e-8)
        recon_nll = (spike_means - target * torch.log(spike_means)).sum() / target.shape[0]

        kl = output.extras["kl_g0"] + output.extras["kl_u"]
        kl_weight = self._scheduled_weight(
            int(self._train_step.item()),
            self.kl_weight_schedule_start,
            self.kl_weight_schedule_dur,
        )
        l2 = self._l2_penalty()
        l2_weight = self._scheduled_weight(
            int(self._train_step.item()),
            self.l2_weight_schedule_start,
            self.l2_weight_schedule_dur,
        )
        total = recon_nll + kl_weight * kl + l2_weight * l2

        if self.training:
            self._train_step.add_(1)

        return LossOutput(
            total=total,
            named_terms={
                "reconstruction_nll": recon_nll,
                "kl": kl,
                "kl_weight": kl_weight,
                "l2": l2,
                "l2_weight": l2_weight,
            },
            objective=self.objective,
        )

    @torch.no_grad()
    def predict_rates(self, x: Tensor) -> Tensor:
        if self.prediction_samples <= 1:
            return self._forward(x, sample=False).rates

        rates = []
        was_training = self.training
        self.eval()
        try:
            for _ in range(self.prediction_samples):
                rates.append(self._forward(x, sample=True).rates)
        finally:
            self.train(was_training)
        return torch.stack(rates, dim=0).mean(dim=0)

    def _forward(self, x: Tensor, sample: bool) -> ModelOutput:
        if x.ndim != 3:
            raise ValueError("LFADS expects input shape (batch, time, neurons).")
        if x.shape[1] != self.n_time:
            raise ValueError(f"Expected {self.n_time} time bins, got {x.shape[1]}.")
        if x.shape[-1] != self.n_neurons:
            raise ValueError(f"Expected {self.n_neurons} neurons, got {x.shape[-1]}.")
        if torch.any(x < 0):
            raise ValueError("LFADS expects nonnegative spike-count observations.")

        x = x.to(device=self.device, dtype=self.fc_log_rates.weight.dtype)
        self._maybe_initialize_rate_bias(x)
        encoder_x = torch.log1p(x) if self.use_log1p_encoder_inputs else x
        if self.keep_prob < 1.0:
            encoder_x = self.dropout(encoder_x)

        _, g0_hidden = self.g0_encoder(encoder_x)
        g0_context = torch.cat((g0_hidden[0], g0_hidden[1]), dim=-1)
        if self.keep_prob < 1.0:
            g0_context = self.dropout(g0_context)

        g0_mean = self.fc_g0_mean(g0_context)
        g0_logvar = self._clamp_logvar(self.fc_g0_logvar(g0_context))
        generator_state = self._reparameterize(g0_mean, g0_logvar, sample=sample)

        controller_context, _ = self.controller_encoder(encoder_x)
        controller_state = torch.zeros(
            x.shape[0],
            self.controller_dim,
            device=x.device,
            dtype=x.dtype,
        )
        factors_t = self._factors_from_generator(generator_state)

        rates = []
        factors = []
        u_means = []
        u_logvars = []
        kl_u_terms = []

        prior_u_mean = self.u_prior_mean.expand(x.shape[0], self.inferred_input_dim)
        prior_u_logvar = self.u_prior_logvar.expand(x.shape[0], self.inferred_input_dim)

        for t in range(self.n_time):
            controller_input = torch.cat((controller_context[:, t], factors_t), dim=-1)
            if self.keep_prob < 1.0:
                controller_input = self.dropout(controller_input)

            controller_state = self.controller(controller_input, controller_state)
            controller_state = torch.clamp(controller_state, min=0.0, max=self.clip_val)

            u_mean = self.fc_u_mean(controller_state)
            u_logvar = self._clamp_logvar(self.fc_u_logvar(controller_state))
            inferred_input = self._reparameterize(u_mean, u_logvar, sample=sample)
            kl_u_terms.append(
                _kl_diag_gaussian(
                    u_mean,
                    u_logvar,
                    prior_u_mean,
                    prior_u_logvar,
                ).mean()
            )

            generator_state = self.generator(inferred_input, generator_state)
            generator_state = torch.clamp(generator_state, min=0.0, max=self.clip_val)
            if self.keep_prob < 1.0:
                generator_state = self.dropout(generator_state)

            factors_t = self._factors_from_generator(generator_state)
            log_rates_t = self.fc_log_rates(factors_t).clamp(
                min=self.log_rate_min,
                max=self.log_rate_max,
            )
            rates.append(torch.exp(log_rates_t))
            factors.append(factors_t)
            u_means.append(u_mean)
            u_logvars.append(u_logvar)

        prior_g0_mean = self.g0_prior_mean.expand_as(g0_mean)
        prior_g0_logvar = self.g0_prior_logvar.expand_as(g0_logvar)
        kl_g0 = _kl_diag_gaussian(g0_mean, g0_logvar, prior_g0_mean, prior_g0_logvar).mean()
        kl_u = torch.stack(kl_u_terms).sum()
        rates_tensor = torch.stack(rates, dim=1)
        factors_tensor = torch.stack(factors, dim=1)

        return ModelOutput(
            rates=rates_tensor,
            latents=factors_tensor,
            reconstruction=rates_tensor,
            extras={
                "kl_g0": kl_g0,
                "kl_u": kl_u,
                "g0_mean": g0_mean,
                "g0_logvar": g0_logvar,
                "u_mean": torch.stack(u_means, dim=1),
                "u_logvar": torch.stack(u_logvars, dim=1),
            },
        )

    def _factors_from_generator(self, generator_state: Tensor) -> Tensor:
        weight = F.normalize(self.fc_factors.weight, dim=1)
        return F.linear(generator_state, weight, self.fc_factors.bias)

    def _clamp_logvar(self, logvar: Tensor) -> Tensor:
        return logvar.clamp(
            min=self.posterior_logvar_min,
            max=self.posterior_logvar_max,
        )

    @staticmethod
    def _reparameterize(mean: Tensor, logvar: Tensor, sample: bool) -> Tensor:
        if not sample:
            return mean
        noise = torch.randn_like(mean)
        return mean + noise * torch.exp(0.5 * logvar)

    def _l2_penalty(self) -> Tensor:
        gen = self.generator.weight_hh.norm(2) / self.generator.weight_hh.numel()
        con = self.controller.weight_hh.norm(2) / self.controller.weight_hh.numel()
        return self.l2_gen_scale * gen + self.l2_con_scale * con

    def _maybe_initialize_rate_bias(self, x: Tensor) -> None:
        if not self.initialize_log_rate_bias or bool(self._rate_bias_initialized.item()):
            return
        with torch.no_grad():
            mean_rate = x.mean(dim=(0, 1)) / max(self.dt, 1e-8)
            mean_rate = mean_rate.clamp(
                min=math.exp(self.log_rate_min),
                max=math.exp(self.log_rate_max),
            )
            self.fc_log_rates.bias.copy_(torch.log(mean_rate))
            self._rate_bias_initialized.copy_(
                torch.tensor(True, device=self._rate_bias_initialized.device)
            )

    @staticmethod
    def _scheduled_weight(step: int, start: int, duration: int) -> float:
        if duration <= 0:
            return 1.0 if step >= start else 0.0
        return float(min(max(step - start, 0) / duration, 1.0))

    def _batch_dt(self, batch: Tensor | dict[str, Tensor], target: Tensor) -> Tensor:
        if isinstance(batch, dict) and "dt" in batch:
            dt = batch["dt"]
            while dt.ndim < target.ndim:
                dt = dt.unsqueeze(-1)
            return dt
        return torch.as_tensor(self.dt, device=target.device, dtype=target.dtype)
