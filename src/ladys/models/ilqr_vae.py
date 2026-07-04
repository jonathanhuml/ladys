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
from ladys.metrics import poisson_negative_log_likelihood
from ladys.types import LossOutput, ModelOutput, observations_from_batch


@BaseModelConfig.register
class ILQRVAEConfig(BaseModelConfig):
    """Config for the tutorial iLQR-VAE posterior-inference adapter."""

    name: Literal["ilqr_vae"] = "ilqr_vae"
    objective: str = "posterior_control"
    params_path: str = "data/real/ilqr_vae/final_params.bin"
    solver: Literal["ilqr", "lbfgs", "adam"] = "ilqr"
    max_iter: int = 100
    lr: Optional[float] = None
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
            solver=self.solver,
            max_iter=self.max_iter,
            lr=self.lr,
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
        params_path: str,
        solver: str = "ilqr",
        max_iter: int = 100,
        lr: Optional[float] = None,
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
        self.params_path = str(params_path)
        self.solver = solver
        self.max_iter = int(max_iter)
        self.lr = lr
        self.held_in_neurons = held_in_neurons
        self.output_neuron_start = output_neuron_start
        self.output_neurons = output_neurons
        self.rate_mode = rate_mode
        self.dt = float(dt)
        self.objective = objective

        params = load_tutorial_params(Path(params_path))
        self.core = TutorialILQRVAE(params, dt=self.dt)

    def forward(self, x: Tensor) -> ModelOutput:
        if x.ndim != 3:
            raise ValueError(f"expected input shape batch x time x neurons, got {tuple(x.shape)}")

        held_in = self.held_in_neurons or int(x.shape[-1])
        output_start = self.output_neuron_start if self.output_neuron_start is not None else held_in
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
            controls.append(result.controls.to(x.dtype))
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
