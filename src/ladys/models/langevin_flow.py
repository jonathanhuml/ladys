"""LangevinFlow adapter for neural spike-count sequence modeling."""

from __future__ import annotations

import math
from typing import Any, Literal, Optional

import torch
from pydantic import Field, model_validator
from torch import Tensor, nn
import torch.nn.functional as F

from ladys.metrics import EvaluationAdapter, EvaluationResult, NLBCoSmoothingAdapter
from ladys.metrics import compute_available_metrics
from ladys.models.base import BaseDynamicsModel, BaseModelConfig, OptimizationConfig
from ladys.types import LossOutput, ModelOutput, move_batch_to_device, observations_from_batch


class FixupTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """One-layer Transformer decoder block with the upstream T-Fixup scaling."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.fixup_initialization()

    def fixup_initialization(self) -> None:
        scale = 0.67 * (3.0 ** (-0.25))
        with torch.no_grad():
            self.linear1.weight.mul_(scale)
            self.linear2.weight.mul_(scale)
            self.self_attn.out_proj.weight.mul_(scale)


class CoupledOscillatorPotential(nn.Module):
    """Locally coupled oscillator potential over grouped latent coordinates."""

    def __init__(self, hidden_size: int, groups: int = 4, kernel_size: int = 3) -> None:
        super().__init__()
        if groups < 1:
            raise ValueError("groups must be positive.")
        if hidden_size % groups != 0:
            raise ValueError("hidden_size must be divisible by potential groups.")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("potential_kernel_size must be a positive odd integer.")

        self.hidden_size = int(hidden_size)
        self.groups = int(groups)
        self.kernel_size = int(kernel_size)
        self.channels_per_group = self.hidden_size // self.groups
        self.conv_z = nn.Parameter(torch.zeros(self.groups, self.groups, self.kernel_size))

    def forward(self, z: Tensor) -> Tensor:
        grouped = z.reshape(z.shape[0], self.groups, self.channels_per_group)
        weight = F.normalize(self.conv_z, p=2, dim=2, eps=1e-8)
        coupled = F.conv1d(
            grouped,
            weight,
            bias=None,
            stride=1,
            padding=self.kernel_size // 2,
        )
        interaction = coupled.bmm(grouped.transpose(1, 2))
        return interaction.sum(dim=(1, 2))


@BaseModelConfig.register
class LangevinFlowConfig(BaseModelConfig):
    """Config for the LangevinFlow sequential VAE."""

    name: Literal["langevin_flow"] = "langevin_flow"
    objective: str = "langevin_flow_elbo"
    hidden_size: int = 64
    output_neurons: Optional[int] = None
    output_mode: Literal["auto", "heldin", "heldin_heldout"] = "auto"
    fwd_steps: int = 0
    dropout: float = 0.05
    gamma: float = 0.55
    langevin_step: float = 0.01
    potential_groups: int = 4
    potential_kernel_size: int = 3
    transformer_heads: int = 2
    transformer_feedforward: int = 512
    coordinated_dropout_rate: float = 0.5
    kl_weight: float = 0.1
    kl_warmup_epochs: int = 500
    velocity_prior_var: float = 0.1
    log_rate_min: float = -8.0
    log_rate_max: float = 8.0
    posterior_logvar_min: float = math.log(1e-4)
    posterior_logvar_max: float = 5.0
    sample_train: bool = True
    sample_eval: bool = False
    prediction_samples: int = 1
    optimization: OptimizationConfig = Field(
        default_factory=lambda: OptimizationConfig(
            name="gradient",
            optimizer="Adam",
            lr=3.0e-3,
            weight_decay=2.0e-5,
            gradient_clip=200.0,
        )
    )

    @model_validator(mode="after")
    def validate_dimensions(self) -> "LangevinFlowConfig":
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive.")
        if self.hidden_size % self.potential_groups != 0:
            raise ValueError("hidden_size must be divisible by potential_groups.")
        if self.transformer_heads < 1:
            raise ValueError("transformer_heads must be positive.")
        if (3 * self.hidden_size) % self.transformer_heads != 0:
            raise ValueError("3 * hidden_size must be divisible by transformer_heads.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must be in [0, 1).")
        if self.langevin_step <= 0.0:
            raise ValueError("langevin_step must be positive.")
        if self.potential_kernel_size < 1 or self.potential_kernel_size % 2 == 0:
            raise ValueError("potential_kernel_size must be a positive odd integer.")
        if not 0.0 < self.coordinated_dropout_rate <= 1.0:
            raise ValueError("coordinated_dropout_rate must be in (0, 1].")
        if self.kl_weight < 0.0:
            raise ValueError("kl_weight must be nonnegative.")
        if self.kl_warmup_epochs < 0:
            raise ValueError("kl_warmup_epochs must be nonnegative.")
        if self.velocity_prior_var <= 0.0:
            raise ValueError("velocity_prior_var must be positive.")
        if self.prediction_samples < 1:
            raise ValueError("prediction_samples must be positive.")
        if self.output_neurons is not None and self.output_neurons < 1:
            raise ValueError("output_neurons must be positive when provided.")
        if self.fwd_steps < 0:
            raise ValueError("fwd_steps must be nonnegative.")
        return self

    def build(self, n_neurons: int, n_time: int) -> "LangevinFlow":
        output_neurons = self.output_neurons or n_neurons
        return self._build(n_neurons=n_neurons, n_time=n_time, output_neurons=output_neurons)

    def build_from_data(self, data: Any) -> "LangevinFlow":
        n_neurons = int(data.n_neurons)
        n_time = int(data.n_time)
        output_neurons = self.output_neurons
        if output_neurons is None:
            output_neurons = n_neurons
            if self.output_mode != "heldin":
                train_dataset = data.train_dataset
                if train_dataset is None:
                    raise RuntimeError("DataModule.setup() must run before build_from_data().")
                heldout = getattr(train_dataset, "raw_spikes", None)
                if heldout is not None:
                    output_neurons = n_neurons + int(heldout.shape[-1])
                elif self.output_mode == "heldin_heldout":
                    raise ValueError(
                        "output_mode='heldin_heldout' requires a dataset with raw_spikes."
                    )
        return self._build(
            n_neurons=n_neurons,
            n_time=n_time,
            output_neurons=output_neurons,
        )

    def _build(self, n_neurons: int, n_time: int, output_neurons: int) -> "LangevinFlow":
        return LangevinFlow(
            n_neurons=n_neurons,
            n_time=n_time,
            output_neurons=output_neurons,
            hidden_size=self.hidden_size,
            fwd_steps=self.fwd_steps,
            dropout=self.dropout,
            gamma=self.gamma,
            langevin_step=self.langevin_step,
            potential_groups=self.potential_groups,
            potential_kernel_size=self.potential_kernel_size,
            transformer_heads=self.transformer_heads,
            transformer_feedforward=self.transformer_feedforward,
            coordinated_dropout_rate=self.coordinated_dropout_rate,
            kl_weight=self.kl_weight,
            kl_warmup_epochs=self.kl_warmup_epochs,
            velocity_prior_var=self.velocity_prior_var,
            log_rate_min=self.log_rate_min,
            log_rate_max=self.log_rate_max,
            posterior_logvar_min=self.posterior_logvar_min,
            posterior_logvar_max=self.posterior_logvar_max,
            sample_train=self.sample_train,
            sample_eval=self.sample_eval,
            prediction_samples=self.prediction_samples,
            objective=self.objective,
        )


class LangevinFlow(BaseDynamicsModel):
    """LangevinFlow sequential VAE for binned neural spike counts.

    ## When to use

    Use LangevinFlow as a nonlinear latent dynamics model for raw spike-count
    sequences. A GRU encoder updates short-range hidden state, latent position
    and velocity variables evolve through an underdamped Langevin step with a
    locally coupled oscillator potential, and a one-layer Transformer decoder
    reads the whole latent sequence into Poisson firing rates.

    ## Assumptions

    LangevinFlow expects nonnegative spike counts. On synthetic datasets the
    readout reconstructs the observed neurons. When built by `Experiment` on an
    NLB dataset, `output_mode: auto` sizes the readout to reconstruct held-in
    plus held-out training neurons and evaluates the held-out output slice.

    ## Outputs

    `forward` returns natural-space firing rates, concatenated
    `[position, velocity, hidden]` latent trajectories, and ELBO diagnostics in
    `extras`. `loss` computes Poisson reconstruction with a scheduled
    Langevin KL penalty and coordinated-dropout gradient masking.
    """

    def __init__(
        self,
        n_neurons: int,
        n_time: int,
        output_neurons: int,
        hidden_size: int = 64,
        fwd_steps: int = 0,
        dropout: float = 0.05,
        gamma: float = 0.55,
        langevin_step: float = 0.01,
        potential_groups: int = 4,
        potential_kernel_size: int = 3,
        transformer_heads: int = 2,
        transformer_feedforward: int = 512,
        coordinated_dropout_rate: float = 0.5,
        kl_weight: float = 0.1,
        kl_warmup_epochs: int = 500,
        velocity_prior_var: float = 0.1,
        log_rate_min: float = -8.0,
        log_rate_max: float = 8.0,
        posterior_logvar_min: float = math.log(1e-4),
        posterior_logvar_max: float = 5.0,
        sample_train: bool = True,
        sample_eval: bool = False,
        prediction_samples: int = 1,
        objective: str = "langevin_flow_elbo",
    ) -> None:
        super().__init__()
        self.n_neurons = int(n_neurons)
        self.n_time = int(n_time)
        self.output_neurons = int(output_neurons)
        self.hidden_size = int(hidden_size)
        self.fwd_steps = int(fwd_steps)
        self.dropout_rate = float(dropout)
        self.gamma = float(gamma)
        self.langevin_step = float(langevin_step)
        self.coordinated_dropout_rate = float(coordinated_dropout_rate)
        self.kl_weight = float(kl_weight)
        self.kl_warmup_epochs = int(kl_warmup_epochs)
        self.velocity_prior_var = float(velocity_prior_var)
        self.log_rate_min = float(log_rate_min)
        self.log_rate_max = float(log_rate_max)
        self.posterior_logvar_min = float(posterior_logvar_min)
        self.posterior_logvar_max = float(posterior_logvar_max)
        self.sample_train = bool(sample_train)
        self.sample_eval = bool(sample_eval)
        self.prediction_samples = int(prediction_samples)
        self.objective = objective

        self.encoder = nn.GRUCell(input_size=self.n_neurons, hidden_size=self.hidden_size)
        self.linear_z_means = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_z_logvar = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_v_means = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_v_logvar = nn.Linear(self.hidden_size, self.hidden_size)
        self.decoder = FixupTransformerEncoderLayer(
            d_model=3 * self.hidden_size,
            nhead=int(transformer_heads),
            dim_feedforward=int(transformer_feedforward),
            dropout=self.dropout_rate,
        )
        self.potential = CoupledOscillatorPotential(
            hidden_size=self.hidden_size,
            groups=int(potential_groups),
            kernel_size=int(potential_kernel_size),
        )
        self.readout = nn.Linear(3 * self.hidden_size, self.output_neurons)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.register_buffer("_train_step", torch.zeros((), dtype=torch.long))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.GRUCell):
                module.weight_ih.data.normal_(std=module.weight_ih.shape[1] ** -0.5)
                module.weight_hh.data.normal_(std=module.weight_hh.shape[1] ** -0.5)
                module.bias_ih.data.zero_()
                module.bias_hh.data.zero_()
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(std=module.in_features ** -0.5)
                module.bias.data.zero_()
        self.decoder.fixup_initialization()
        self.potential.conv_z.data.zero_()

    def forward(self, x: Tensor) -> ModelOutput:
        sample = self.sample_train if self.training else self.sample_eval
        return self._forward(x, sample=sample)

    def loss(
        self,
        batch: Tensor | dict[str, Tensor],
        output: ModelOutput,
        epoch: int = 0,
    ) -> LossOutput:
        if output.rates is None:
            raise RuntimeError("LangevinFlow forward output is missing rates.")
        log_rates = output.extras["log_rates"]
        target = self._reconstruction_target(batch, log_rates)
        target = target.to(device=self.device, dtype=log_rates.dtype)
        log_rates = log_rates[:, : target.shape[1], : target.shape[2]]
        recon = F.poisson_nll_loss(log_rates, target, log_input=True, reduction="none")
        cd_mask = output.extras.get("coordinated_dropout_mask")
        if self.training and cd_mask is not None:
            cd_mask = self._pad_cd_mask(cd_mask, recon)
            recon = recon * cd_mask + (recon * (1.0 - cd_mask)).detach()
        recon_nll = recon.mean()

        kl = output.extras["kl"]
        kl_weight = self._kl_weight(epoch)
        total = recon_nll + kl_weight * kl
        if self.training:
            self._train_step.add_(1)

        return LossOutput(
            total=total,
            named_terms={
                "reconstruction_nll": recon_nll,
                "kl": kl,
                "kl_weight": kl_weight,
            },
            objective=self.objective,
        )

    @torch.no_grad()
    def predict_rates(self, x: Tensor) -> Tensor:
        was_training = self.training
        self.eval()
        try:
            if self.prediction_samples <= 1:
                return self._forward(x, sample=self.sample_eval).rates
            rates = [
                self._forward(x, sample=True).rates
                for _ in range(self.prediction_samples)
            ]
        finally:
            self.train(was_training)
        return torch.stack(rates, dim=0).mean(dim=0)

    def evaluation_adapter(self, task: str) -> EvaluationAdapter | None:
        if task != "nlb":
            return None
        if self.output_neurons > self.n_neurons:
            return LangevinFlowNLBAdapter()
        return NLBCoSmoothingAdapter(feature_source="latents")

    def _forward(self, x: Tensor, sample: bool) -> ModelOutput:
        if x.ndim != 3:
            raise ValueError("LangevinFlow expects input shape (batch, time, neurons).")
        if x.shape[1] != self.n_time:
            raise ValueError(f"Expected {self.n_time} time bins, got {x.shape[1]}.")
        if x.shape[-1] != self.n_neurons:
            raise ValueError(f"Expected {self.n_neurons} neurons, got {x.shape[-1]}.")
        if torch.any(x < 0):
            raise ValueError("LangevinFlow expects nonnegative spike-count observations.")

        observ, cd_mask = self._coordinated_dropout(x)
        total_steps = self.n_time + self.fwd_steps
        hidden = self.dropout(self.encoder(observ[:, 0]))
        z_mu = self.linear_z_means(hidden)
        z_logvar = self._clamp_logvar(self.linear_z_logvar(hidden))
        v_mu = self.linear_v_means(hidden)
        v_logvar = self._clamp_logvar(self.linear_v_logvar(hidden))
        z = self._reparameterize(z_mu, z_logvar, sample=sample)
        v = self._reparameterize(v_mu, v_logvar, sample=sample)
        kl = self._kl_diag_gaussian(z_mu, z_logvar, prior_var=1.0)
        kl = kl + self._kl_diag_gaussian(v_mu, v_logvar, prior_var=1.0)

        latent_steps = [torch.cat([z, v, hidden], dim=1)]
        noise_var = max(2.0 * self.gamma, 1e-8)
        noise_std = math.sqrt(noise_var)
        for t in range(1, total_steps):
            if t < self.n_time:
                hidden_input = observ[:, t - 1]
            else:
                hidden_input = observ[:, -1]
            hidden = self.dropout(self.encoder(hidden_input, hidden))
            z, v, step_kl = self._langevin_step(z, v, noise_std, sample)
            kl = kl + step_kl
            latent_steps.append(torch.cat([z, v, hidden], dim=1))

        latents = torch.stack(latent_steps, dim=1)
        decoded = self.decoder(latents)
        log_rates = self.readout(decoded).clamp(self.log_rate_min, self.log_rate_max)
        rates = torch.exp(log_rates)
        return ModelOutput(
            rates=rates,
            latents=latents,
            extras={
                "log_rates": log_rates,
                "kl": kl,
                "coordinated_dropout_mask": cd_mask,
            },
        )

    def _langevin_step(
        self,
        z: Tensor,
        v: Tensor,
        noise_std: float,
        sample: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        with torch.enable_grad():
            z_for_grad = z.clone().requires_grad_(True)
            energy = self.potential(z_for_grad)
            force = torch.autograd.grad(
                energy.sum(),
                z_for_grad,
                create_graph=self.training,
                retain_graph=self.training,
            )[0]

        z_next = z_for_grad + self.langevin_step * v
        v_half = v - self.langevin_step * force
        v_mean = (1.0 - self.gamma) * v_half
        if sample:
            v_next = v_mean + torch.randn_like(v_mean) * noise_std
        else:
            v_next = v_mean
        step_kl = self._kl_diag_gaussian(
            v_next,
            torch.full_like(v_next, 2.0 * self.gamma),
            prior_var=self.velocity_prior_var,
        )
        return z_next, v_next, step_kl

    def _coordinated_dropout(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        if not self.training or self.coordinated_dropout_rate >= 1.0:
            return x, None
        keep_prob = self.coordinated_dropout_rate
        keep = torch.bernoulli(torch.full_like(x, keep_prob))
        pass_mask = torch.bernoulli(torch.full_like(x, 1.0 - keep_prob))
        grad_mask = torch.logical_or(keep == 0.0, pass_mask == 1.0).to(dtype=x.dtype)
        return x * keep / keep_prob, grad_mask

    def _reconstruction_target(
        self,
        batch: Tensor | dict[str, Tensor],
        log_rates: Tensor,
    ) -> Tensor:
        observed = observations_from_batch(batch)
        if isinstance(batch, dict) and "heldout_spikes" in batch:
            heldout = batch["heldout_spikes"]
            total_neurons = observed.shape[-1] + heldout.shape[-1]
            if log_rates.shape[-1] >= total_neurons:
                return torch.cat([observed, heldout], dim=-1)
        return observed

    @staticmethod
    def _pad_cd_mask(mask: Tensor, loss: Tensor) -> Tensor:
        time_pad = loss.shape[1] - mask.shape[1]
        neuron_pad = loss.shape[2] - mask.shape[2]
        if time_pad < 0 or neuron_pad < 0:
            return mask[:, : loss.shape[1], : loss.shape[2]]
        return F.pad(mask, (0, neuron_pad, 0, time_pad), value=1.0)

    def _kl_weight(self, epoch: int) -> float:
        if self.kl_warmup_epochs <= 0:
            return self.kl_weight
        progress = max(float(epoch) / float(self.kl_warmup_epochs), 0.0)
        return self.kl_weight * min(progress, 1.0)

    def _clamp_logvar(self, logvar: Tensor) -> Tensor:
        return logvar.clamp(self.posterior_logvar_min, self.posterior_logvar_max)

    @staticmethod
    def _reparameterize(mean: Tensor, logvar: Tensor, sample: bool) -> Tensor:
        if not sample:
            return mean
        std = torch.exp(0.5 * logvar)
        return mean + torch.randn_like(std) * std

    @staticmethod
    def _kl_diag_gaussian(mean: Tensor, logvar: Tensor, prior_var: float) -> Tensor:
        prior_logvar = math.log(prior_var)
        var = torch.exp(logvar)
        kl = 0.5 * (prior_logvar - logvar - 1.0 + (mean.pow(2) + var) / prior_var)
        return kl.sum() / mean.shape[0]


class LangevinFlowNLBAdapter(EvaluationAdapter):
    """Direct held-out NLB scorer for LangevinFlow full readouts."""

    task = "nlb"

    def evaluate(
        self,
        model: BaseDynamicsModel,
        loader: Any,
        device: torch.device,
    ) -> EvaluationResult:
        predictions: list[Tensor] = []
        targets: list[Tensor] = []

        with torch.no_grad():
            for batch in loader:
                batch = move_batch_to_device(batch, device)
                x = observations_from_batch(batch)
                if not isinstance(batch, dict) or "heldout_spikes" not in batch:
                    raise TypeError("NLB evaluation requires heldout_spikes in dict batches.")
                rates = model.predict_rates(x)
                target = batch["heldout_spikes"]
                n_heldin = x.shape[-1]
                n_heldout = target.shape[-1]
                pred = rates[:, : target.shape[1], n_heldin : n_heldin + n_heldout]
                if pred.shape != target.shape:
                    raise ValueError(
                        "LangevinFlow direct NLB predictions have shape "
                        f"{tuple(pred.shape)}, expected {tuple(target.shape)}."
                    )
                predictions.append(pred.detach().cpu())
                targets.append(target.detach().cpu())

        pred_dict: dict[str, Tensor] = {"rates": torch.cat(predictions, dim=0)}
        target_dict: dict[str, Tensor] = {"spikes": torch.cat(targets, dim=0)}
        metrics = compute_available_metrics(pred_dict, target_dict)
        return EvaluationResult(
            metrics=metrics,
            predictions={key: value.numpy() for key, value in pred_dict.items()},
            targets={key: value.numpy() for key, value in target_dict.items()},
        )
