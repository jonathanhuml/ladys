"""iLQR-VAE adapter for fixed-parameter posterior inference."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import torch
from pydantic import Field
from torch import Tensor

from ladys.models.base import BaseDynamicsModel, BaseModelConfig, OptimizationConfig
from ladys.models.ilqr_vae_core import ILQRVAE as TutorialILQRVAE
from ladys.models.ilqr_vae_core import load_tutorial_params
from ladys.models.ilqr_vae_core import make_random_params
from ladys.metrics import poisson_negative_log_likelihood
from ladys.types import LossOutput, ModelOutput, observations_from_batch


@BaseModelConfig.register
class ILQRVAEConfig(BaseModelConfig):
    """Config for the tutorial iLQR-VAE posterior-inference adapter."""

    name: Literal["ilqr_vae"] = "ilqr_vae"
    objective: Literal["posterior_control", "ilqr_vae_elbo"] = "posterior_control"
    params_path: Optional[str] = "data/real/ilqr_vae/final_params.bin"
    initialization: Literal["pretrained", "random"] = "pretrained"
    latent_dim: int = 20
    input_dim: int = 5
    init_seed: int = 0
    solver: Literal["ilqr", "lbfgs", "adam"] = "ilqr"
    max_iter: int = 100
    lr: Optional[float] = None
    trainable_parameters: bool = False
    n_posterior_samples: int = 1
    include_elbo_constants: bool = True
    dynamics_regularizer: float = 0.0
    held_in_neurons: Optional[int] = None
    output_neuron_start: Optional[int] = None
    output_neurons: Optional[int] = None
    rate_mode: Literal["likelihood", "pre_sample"] = "likelihood"
    dt: float = 5e-3
    optimization: OptimizationConfig = Field(
        default_factory=lambda: OptimizationConfig(name="inference_only")
    )

    def build(self, n_neurons: int, n_time: int) -> "ILQRVAE":
        return ILQRVAE(
            n_neurons=n_neurons,
            n_time=n_time,
            params_path=self.params_path,
            initialization=self.initialization,
            latent_dim=self.latent_dim,
            input_dim=self.input_dim,
            init_seed=self.init_seed,
            solver=self.solver,
            max_iter=self.max_iter,
            lr=self.lr,
            trainable_parameters=self.trainable_parameters,
            n_posterior_samples=self.n_posterior_samples,
            include_elbo_constants=self.include_elbo_constants,
            dynamics_regularizer=self.dynamics_regularizer,
            held_in_neurons=self.held_in_neurons,
            output_neuron_start=self.output_neuron_start,
            output_neurons=self.output_neurons,
            rate_mode=self.rate_mode,
            dt=self.dt,
            objective=self.objective,
        )


class ILQRVAE(BaseDynamicsModel):
    """Fixed-parameter iLQR-VAE posterior inference.

    This adapter wraps the PyTorch port of the iLQR-VAE tutorial model. It does
    not train generative parameters inside LaDyS; each forward pass solves for
    posterior controls from the input spikes and decodes held-out expected spike
    counts for benchmark metrics.
    """

    def __init__(
        self,
        n_neurons: int,
        n_time: int,
        params_path: Optional[str],
        initialization: str = "pretrained",
        latent_dim: int = 20,
        input_dim: int = 5,
        init_seed: int = 0,
        solver: str = "ilqr",
        max_iter: int = 100,
        lr: Optional[float] = None,
        trainable_parameters: bool = False,
        n_posterior_samples: int = 1,
        include_elbo_constants: bool = True,
        dynamics_regularizer: float = 0.0,
        held_in_neurons: Optional[int] = None,
        output_neuron_start: Optional[int] = None,
        output_neurons: Optional[int] = None,
        rate_mode: str = "likelihood",
        dt: float = 5e-3,
        objective: str = "posterior_control",
    ) -> None:
        super().__init__()
        self.n_neurons = int(n_neurons)
        self.n_time = int(n_time)
        self.params_path = None if params_path is None else str(params_path)
        self.initialization = initialization
        self.latent_dim = int(latent_dim)
        self.input_dim = int(input_dim)
        self.init_seed = int(init_seed)
        self.solver = solver
        self.max_iter = int(max_iter)
        self.lr = lr
        self.trainable_parameters = bool(trainable_parameters)
        self.n_posterior_samples = int(n_posterior_samples)
        self.include_elbo_constants = bool(include_elbo_constants)
        self.dynamics_regularizer = float(dynamics_regularizer)
        self.held_in_neurons = held_in_neurons
        self.output_neuron_start = output_neuron_start
        self.output_neurons = output_neurons
        self.rate_mode = rate_mode
        self.dt = float(dt)
        self.objective = objective

        if initialization == "pretrained":
            if params_path is None:
                raise ValueError("params_path is required for pretrained iLQR-VAE initialization.")
            params = load_tutorial_params(Path(params_path))
        elif initialization == "random":
            params = make_random_params(
                latent_dim=self.latent_dim,
                input_dim=self.input_dim,
                n_neurons=self.n_neurons,
                n_time=self.n_time,
                seed=self.init_seed,
            )
        else:
            raise ValueError(f"unknown iLQR-VAE initialization {initialization!r}")
        self.core = TutorialILQRVAE(params, dt=self.dt, trainable=self.trainable_parameters)

    def forward(self, x: Tensor) -> ModelOutput:
        if x.ndim != 3:
            raise ValueError(f"expected input shape batch x time x neurons, got {tuple(x.shape)}")

        held_in = self.held_in_neurons or int(x.shape[-1])
        output_start = self.output_neuron_start if self.output_neuron_start is not None else 0
        output_stop = (
            self.core.n_neurons
            if self.output_neurons is None
            else output_start + int(self.output_neurons)
        )

        rates = []
        latents = []
        controls = []
        eval_counts = []
        objectives = []
        for trial in x:
            result = self.core.infer_controls(
                trial.detach(),
                held_in_neurons=held_in,
                solver=self.solver,
                max_iter=self.max_iter,
                lr=self.lr,
            )
            observed_latents = self.core.observation_latents(
                result.latents,
                n_observed_steps=int(trial.shape[0]),
            )
            rates_hz = self.core.firing_rates(observed_latents, mode=self.rate_mode)
            rates.append((self.dt * rates_hz[:, output_start:output_stop]).to(x.dtype))
            latents.append(observed_latents.to(x.dtype))
            controls.append(result.controls)
            eval_counts.append(len(result.loss_history))
            objectives.append(result.loss_history[-1] if result.loss_history else float("nan"))

        return ModelOutput(
            rates=torch.stack(rates, dim=0),
            latents=torch.stack(latents, dim=0),
            extras={
                "controls": torch.stack(controls, dim=0),
                "ilqr_evaluations": torch.tensor(eval_counts, dtype=torch.float32, device=x.device),
                "posterior_objective": torch.tensor(objectives, dtype=torch.float32, device=x.device),
            },
        )

    def loss(
        self,
        batch: Tensor | dict[str, Tensor],
        output: ModelOutput,
        epoch: int = 0,
    ) -> LossOutput:
        del epoch
        x = observations_from_batch(batch)
        if self.objective == "ilqr_vae_elbo":
            return self._elbo_loss(x, output)

        target = batch.get("raw_spikes", x) if isinstance(batch, dict) else x
        if output.rates is None:
            raise RuntimeError("ILQRVAE.forward did not return rates.")
        total = poisson_negative_log_likelihood(output.rates, target).mean()
        return LossOutput(
            total=total,
            named_terms={
                "poisson_nll": total,
                "mean_ilqr_evaluations": output.extras["ilqr_evaluations"].mean(),
                "mean_posterior_objective": output.extras["posterior_objective"].mean(),
            },
            objective=self.objective,
        )

    def _elbo_loss(self, x: Tensor, output: ModelOutput) -> LossOutput:
        controls = output.extras.get("controls")
        if not isinstance(controls, Tensor):
            raise RuntimeError("ILQR-VAE ELBO requires posterior controls from forward().")
        held_in = self.held_in_neurons or int(x.shape[-1])
        losses = []
        elbos = []
        for trial, trial_controls in zip(x, controls):
            trial = trial.to(dtype=self.core.c.dtype, device=self.core.c.device)
            trial_controls = trial_controls.detach().to(
                dtype=self.core.c.dtype,
                device=self.core.c.device,
            )
            elbo = self.core.elbo_from_controls(
                trial_controls,
                trial,
                held_in_neurons=held_in,
                n_posterior_samples=self.n_posterior_samples,
                include_constants=self.include_elbo_constants,
            )
            normalizer = max(int(trial[:, :held_in].numel()), 1)
            elbos.append(elbo)
            losses.append(-elbo / float(normalizer))

        negative_elbo = torch.stack(losses).mean()
        regularizer = self._dynamics_regularizer()
        total = negative_elbo + regularizer
        return LossOutput(
            total=total,
            named_terms={
                "negative_elbo": negative_elbo,
                "elbo": torch.stack(elbos).mean(),
                "dynamics_regularizer": regularizer,
                "mean_ilqr_evaluations": output.extras["ilqr_evaluations"].mean(),
                "mean_posterior_objective": output.extras["posterior_objective"].mean(),
            },
            objective=self.objective,
        )

    def _dynamics_regularizer(self) -> Tensor:
        if self.dynamics_regularizer <= 0.0:
            return self.core.c.new_zeros(())
        scale = self.dynamics_regularizer / float(self.core.n_latent * self.core.n_latent)
        return scale * (torch.sum(self.core.uh**2) + torch.sum(self.core.uf**2))

    def project_parameters(self) -> None:
        self.core.project_parameters()
