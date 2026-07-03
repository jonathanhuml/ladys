"""Linear, Gaussian System."""

from __future__ import annotations

from jaxtyping import Float, Int
from torch import Tensor
from scipy.stats import ortho_group
import torch
from typing import Tuple


class LinearSystem(torch.utils.data.Dataset):
    """
    Check out https://ipnpr.jpl.nasa.gov/progress_report/42-233/42-233A.pdf (Chen 2023) for mathematical notation
    """

    def __init__(
            self,
            num_neurons: int,
            num_conditions: int = 65,
            num_trials: int = 4,
            timesteps: int = 100,
            transition_noise: float = 1e-2,
            measurement_noise: float = 1e-2
    ) -> None:
        super().__init__()

        with torch.no_grad():
            # Neural data parameters
            self.num_neurons = num_neurons
            self.num_conditions = num_conditions
            self.num_trials = num_trials
            self.timesteps = timesteps

            self.timepoints = torch.arange(self.timesteps)

            # This dataset has normal, uncorrelated noise processes
            # Therefore, the noise "matrices" are diagonal (i.e. vectors)
            self.transition_noise = transition_noise * torch.ones(self.num_neurons)
            self.measurement_noise = measurement_noise * torch.ones(self.num_neurons)

            # Transition and measurement functions are linear, orthonormal
            self.transition_dynamics = torch.Tensor(ortho_group.rvs(dim=self.num_neurons))
            self.measurement_dynamics = torch.Tensor(ortho_group.rvs(dim=self.num_neurons))

            # Dataset
            self.latent_dynamics, self.observations = self.generate_data()

    def linear_system(self, latent_init):
        latent_noisy = torch.zeros(self.timesteps, self.num_neurons)
        observation_noisy = torch.zeros(self.timesteps, self.num_neurons)
        latent_true = torch.zeros(self.timesteps, self.num_neurons)
        observation_true = torch.zeros(self.timesteps, self.num_neurons)

        latent_noisy[0] = latent_init
        latent_true[0] = latent_init

        for t in range(self.timesteps):
            transition_noise_t = torch.normal(mean=torch.zeros(self.num_neurons), std=torch.sqrt(self.transition_noise))
            measurement_noise_t = torch.normal(mean=torch.zeros(self.num_neurons), std=torch.sqrt(self.measurement_noise))

            observation_noisy[t] = self.measurement_dynamics @ latent_noisy[t] + measurement_noise_t
            observation_true[t] = self.measurement_dynamics @ latent_true[t]

            # If there's a next timestep, push through transition function to next state
            if t < (self.timesteps - 1):
                latent_noisy[t + 1] = self.transition_dynamics @ latent_noisy[t] + transition_noise_t
                latent_true[t + 1] = self.transition_dynamics @ latent_true[t]

        return latent_true, observation_noisy

    def generate_data(self):
        observations = torch.zeros(size=(self.num_trials, self.num_conditions, self.timesteps, self.num_neurons))
        latent_truth = torch.zeros(size=(self.num_trials, self.num_conditions, self.timesteps, self.num_neurons))
        # make [trials per condition, number of conditions, neurons, timepoints]

        for condition in range(self.num_conditions):
            latent_init = torch.normal(mean=torch.zeros(self.num_neurons), std=torch.ones(self.num_neurons))
            for trial in range(self.num_trials):
                latent_true, observation_noisy = self.linear_system(latent_init)

                observations[trial, condition, :, :] = observation_noisy
                latent_truth[trial, condition, :, :] = latent_true

        return latent_truth, observations

    def __getitem__(self, index) -> Tuple[
        Float[Tensor, "num_timepoints"],
        Float[Tensor, "num_conditions num_timepoints num_neurons"],
        Float[Tensor, "num_conditions num_timepoints num_neurons"],
    ]:
        return (
            self.timepoints,
            self.observations[index],
            self.latent_dynamics[index],
        )  # NOTE: Index selects the trial

    def __len__(self):
        return self.num_trials
