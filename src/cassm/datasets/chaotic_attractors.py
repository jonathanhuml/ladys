from __future__ import annotations

from jaxtyping import Float, Int
from torch import Tensor
from scipy.integrate import solve_ivp
import torch
import numpy as np
from typing import Tuple
from cassm.utils import preprocessing

from dysts.flows import Lorenz
import dysts


class ChaoticSystem(torch.utils.data.Dataset):

    def __init__(
        self,
        num_neurons: int,
        model = dysts.flows.Lorenz(),
        num_conditions: int = 65,
        num_trials_per_condition: int = 20,
        time_step_size: float = 0.006,
        t_final: float = 20,
        normalize_latent_dynamics: bool = True,
        smoothed_firing_rates: bool = True,
    ) -> None:

        super().__init__()

        with torch.no_grad():
            # Lorenz system parameters
            self.model = model
            sol = self.model.make_trajectory(10, standardize=True)
            self.latent_dim = sol.shape[-1]

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

        initial_conditions = np.random.normal(loc=0, scale=5, size=(self.num_conditions, self.latent_dim))

        # solutions = []

        # Solve the Lorenz system for each initial condition
        for idx in range(len(initial_conditions)):
            self.model.ic = initial_conditions[idx]
            sol = self.model.make_trajectory(len(self.timepoints), resample=True) 
            # solutions.append(sol)
            latent_dynamics[idx] = torch.Tensor(sol)

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

if __name__ == "__main__":
    model = dysts.flows.Lorenz()
    # model.ic = np.array([4, 2, 8])
    t = 1000
    sol = model.make_trajectory(t)
    latent_dim = sol.shape[-1]

    print(sol[0])
    
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3-D)

    # sol is shape (T, 3): columns = x, y, z
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(sol[:, 0], sol[:, 1], sol[:, 2], lw=1.5)   # the trajectory
    ax.scatter(sol[0, 0], sol[0, 1], sol[0, 2], c="green", label="start")
    ax.scatter(sol[-1, 0], sol[-1, 1], sol[-1, 2], c="red", label="end")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Trajectory in 3-D")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()
    


