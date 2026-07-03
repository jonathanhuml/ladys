"""Lorenz system."""

from __future__ import annotations

from jaxtyping import Float, Int
from torch import Tensor
from scipy.integrate import solve_ivp
import torch
import numpy as np
from typing import Tuple
from cassm.utils import preprocessing


class LorenzSystem(torch.utils.data.Dataset):
    r"""Lorenz system.

    Synthetic neural spiking data generated from latent dynamics given by the Lorenz system, 
    a non-linear three-dimensional dynamical system governed by the equations:

    .. math::

        \dot{x}_1(t) &= \sigma (x_2(t) - x_1(t)) \\
        \dot{x}_2(t) &= x_1(t) (\rho - x_3(t)) - x_2(t) \\
        \dot{x}_3(t) &= x_1(t) x_2(t) - \beta x_3(t)

    The solution of the ODE system is computed via numerical integration.

    See also Section 4.1 of http://arxiv.org/abs/1608.06315 for a more detailed description
    and also https://github.com/catniplab/vlgp/blob/accc969055db623f92a96bef8f0d583bd3360c55/vlgp/simulation.py#L11.

    :param num_neurons: Number of observed neurons.
    :param num_conditions: Number of conditions. Corresponds
        to starting the Lorenz system with a random initial state vector for each condition.
    :param num_trials_per_condition: Number of trials per condition. Corresponds to
        the number of times an experiment was repeated per condition.
    :param sigma: :math:`\sigma` parameter of the Lorenz system.
    :param rho: :math:`\rho` parameter of the Lorenz system.
    :param beta: :math:`\beta` parameter of the Lorenz system.
    :param time_step_size: Step size of the numerical integrator used to solve the dynamical system
        which defines the ground truth latent dynamics.
    """

    def __init__(
        self,
        num_neurons: int,
        num_conditions: int = 65,
        num_trials_per_condition: int = 20,
        sigma: float = 10,
        rho: float = 28,
        beta: float = 8 / 3,
        time_step_size: float = 0.006,
        t_final: float = 20,
        normalize_latent_dynamics: bool = True,
        smoothed_firing_rates: bool = True,
    ) -> None:

        super().__init__()

        with torch.no_grad():
            # Lorenz system parameters
            self.latent_dim = 3
            self.sigma = sigma
            self.rho = rho
            self.beta = beta

            self.time_step_size = time_step_size
            self.t_final = t_final
            self.timepoints = torch.arange(0, self.t_final, step=self.time_step_size)
            self.latent_dynamics_init = 10 * (
                torch.randn((num_conditions, self.latent_dim))
            )

            # Neural data parameters
            self.num_neurons = num_neurons
            self.num_conditions = num_conditions
            self.num_trials_per_condition = num_trials_per_condition

            # Dataset
            self.latent_dynamics = self._generate_latent_dynamics(
                timepoints=self.timepoints,
                latent_dynamics_init=self.latent_dynamics_init,
                normalize=normalize_latent_dynamics,
            )
            self.firing_rates = self._generate_firing_rates(
                latent_dynamics=self.latent_dynamics
            )
            self.spike_trains = self._generate_spikes(firing_rates=self.firing_rates)
            self.smoothed_firing_rates = smoothed_firing_rates

            if self.smoothed_firing_rates:
                self.smoothed_rates = preprocessing.estimate_firing_rate(
                    self.spike_trains, self.timepoints, self.t_final
                )

                self.smoothed_rates = preprocessing.anscombe_transform(
                    torch.Tensor(self.smoothed_rates)
                )

    def _lorenz_ode(
        self,
        t: Float[np.array, "timepoint"],
        x: Float[np.array, "latent"],
    ) -> Float[np.array, "latent timepoint"]:
        """Lorenz ODE system defining the latent dynamics."""
        return np.stack(
            (
                self.sigma * (x[1] - x[0]),
                x[0] * (self.rho - x[2]) - x[1],
                x[0] * x[1] - self.beta * x[2],
            ),
            axis=0,
        )

    def _generate_latent_dynamics(
        self,
        timepoints: Float[Tensor, "time"],
        latent_dynamics_init: Float[Tensor, "condition latent"],
        normalize: bool,
    ) -> Float[Tensor, "condition timepoint latent"]:
        """Generate latent dynamics from Lorenz system."""

        # Simulate latent dynamics for different conditions
        latent_dynamics = torch.empty(
            (self.num_conditions, len(timepoints), self.latent_dim),
            requires_grad=False,
        )

        # Solve Lorenz system for different initial conditions
        for idx_condition in range(self.num_conditions):

            ivp_solution = solve_ivp(
                fun=self._lorenz_ode,
                y0=latent_dynamics_init[idx_condition].detach().cpu().numpy(),
                t_span=(timepoints[0], timepoints[-1]),
                t_eval=timepoints.detach().cpu().numpy(),
                method="RK45",
            )

            latent_dynamics[idx_condition] = torch.as_tensor(ivp_solution.y).mT

        if normalize:
            latent_dynamics = (
                latent_dynamics - torch.mean(latent_dynamics, dim=-2, keepdim=True)
            ) / torch.std(latent_dynamics, dim=-2, keepdim=True)

        return latent_dynamics

    def _generate_firing_rates(
        self, latent_dynamics: Float[Tensor, "condition timepoint latent"]
    ) -> Float[Tensor, "trial condition timepoint neuron"]:
        """Generate firing rates based on latent dynamics."""
        # Sample linear readout matrices (one per trial)
        self.readout_tensor = torch.randn(
            (
                self.num_trials_per_condition,
                self.num_conditions,
                self.num_neurons,
                self.latent_dim,
            )
        )

        # Firing rates (num_trials_per_condition x num_conditions x timepoint x num_neurons)
        # TODO: should we add a bias or additive noise to the readout?
        firing_rates = torch.exp(
            torch.einsum("BCND,CTD->BCTN", self.readout_tensor, latent_dynamics)
        )

        return firing_rates

    def _generate_spikes(
        self,
        firing_rates: Float[Tensor, "trial condition timepoint neuron"],
    ) -> Int[Tensor, "trial condition timepoint neuron"]:
        """Generate spike trains for all neurons based on firing rates."""

        # Generate spikes
        spike_trains = torch.poisson(firing_rates * self.time_step_size).clip(
            0, 1
        )  # TODO: Should we really clip here?

        return spike_trains

    def __getitem__(self, index) -> Tuple[
        Float[Tensor, "timepoint"],
        Int[Tensor, "condition timepoint neuron"],
        Float[Tensor, "condition timepoint latent"],
    ]:
        if self.smoothed_firing_rates:
            self.spike_trains = self.smoothed_rates
        return (
            self.timepoints,
            self.spike_trains[
                index
            ],  # self.smoothed_rates[index] if spikes no longer needed
            self.latent_dynamics,
        )  # NOTE: Index selects the trial

    def __len__(self) -> int:
        return self.spike_trains.shape[0]
