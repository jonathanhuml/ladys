"""Switching linear dynamical system baseline in the LaDyS model API."""

from __future__ import annotations

import math
from typing import Iterable, Literal, Optional

import torch
from pydantic import Field
from torch import Tensor, nn
import torch.nn.functional as F

from ladys.metrics import EvaluationAdapter, EvaluationResult, compute_available_metrics
from ladys.models.base import BaseDynamicsModel, BaseModelConfig, OptimizationConfig
from ladys.types import LossOutput, ModelOutput, move_batch_to_device, observations_from_batch


@BaseModelConfig.register
class SLDSConfig(BaseModelConfig):
    """Config for a regular Poisson switching linear dynamical system."""

    name: Literal["slds"] = "slds"
    objective: str = "slds_variational_em"
    states: int = 4
    latent_dim: int = 8
    transition_stickiness: float = 0.95
    dynamics_l2_A: float = 1e-4
    dynamics_l2_b: float = 1e-6
    dynamics_var_floor: float = 1e-4
    emission_l2: float = 1e-5
    emission_lr: float = 1.0
    emission_mstep_steps: int = 20
    emission_batch_size: int = 16384
    emission_max_points: Optional[int] = 200_000
    em_alpha: float = 0.2
    local_inference_iters: int = 2
    continuous_steps: int = 25
    continuous_lr: float = 1.0
    continuous_tolerance: float = 1e-4
    latent_l2: float = 0.0
    latent_clip: Optional[float] = None
    init_seed: Optional[int] = 0
    init_max_points: int = 50_000
    init_em_iters: int = 50
    prediction_floor: float = 1e-9
    optimization: OptimizationConfig = Field(
        default_factory=lambda: OptimizationConfig(name="em")
    )

    def build(self, n_neurons: int, n_time: int) -> "SLDS":
        return SLDS(
            n_neurons=n_neurons,
            n_time=n_time,
            states=self.states,
            latent_dim=self.latent_dim,
            transition_stickiness=self.transition_stickiness,
            dynamics_l2_A=self.dynamics_l2_A,
            dynamics_l2_b=self.dynamics_l2_b,
            dynamics_var_floor=self.dynamics_var_floor,
            emission_l2=self.emission_l2,
            emission_lr=self.emission_lr,
            emission_mstep_steps=self.emission_mstep_steps,
            emission_batch_size=self.emission_batch_size,
            emission_max_points=self.emission_max_points,
            em_alpha=self.em_alpha,
            local_inference_iters=self.local_inference_iters,
            continuous_steps=self.continuous_steps,
            continuous_lr=self.continuous_lr,
            continuous_tolerance=self.continuous_tolerance,
            latent_l2=self.latent_l2,
            latent_clip=self.latent_clip,
            init_seed=self.init_seed,
            init_max_points=self.init_max_points,
            init_em_iters=self.init_em_iters,
            prediction_floor=self.prediction_floor,
            objective=self.objective,
        )


class SLDS(BaseDynamicsModel):
    """Regular switching linear dynamical system with Poisson emissions.

    ## When to use

    Use SLDS as the regular switching linear dynamical-system baseline from
    the Neural Latents Benchmark family. The model has a stationary Markov
    chain over discrete states, state-specific linear Gaussian dynamics over a
    continuous latent trajectory, and a shared linear softplus-Poisson emission
    model for binned spike counts.

    ## Assumptions

    Inputs are nonnegative binned spike counts. The current port intentionally
    targets the regular NLB SLDS path rather than recurrent transitions or
    black-box variational inference. EM is chunked through the dataloader so
    NLB training can use held-in and held-out training neurons jointly while
    NLB evaluation hides held-out eval neurons behind a mask.

    ## Outputs

    `forward` runs approximate posterior inference for the provided observed
    neurons and returns smoothed firing-rate predictions plus continuous
    latent trajectories. On NLB datasets, `evaluation_adapter("nlb")` evaluates
    held-out neurons directly by building the same masked observation problem
    used by the public SLDS baseline.
    """

    def __init__(
        self,
        n_neurons: int,
        n_time: int,
        states: int = 4,
        latent_dim: int = 8,
        transition_stickiness: float = 0.95,
        dynamics_l2_A: float = 1e-4,
        dynamics_l2_b: float = 1e-6,
        dynamics_var_floor: float = 1e-4,
        emission_l2: float = 1e-5,
        emission_lr: float = 1.0,
        emission_mstep_steps: int = 20,
        emission_batch_size: int = 16384,
        emission_max_points: int | None = 200_000,
        em_alpha: float = 0.2,
        local_inference_iters: int = 2,
        continuous_steps: int = 25,
        continuous_lr: float = 1.0,
        continuous_tolerance: float = 1e-4,
        latent_l2: float = 0.0,
        latent_clip: float | None = None,
        init_seed: int | None = 0,
        init_max_points: int = 50_000,
        init_em_iters: int = 50,
        prediction_floor: float = 1e-9,
        objective: str = "slds_variational_em",
    ) -> None:
        super().__init__()
        if states < 1:
            raise ValueError("states must be >= 1.")
        if latent_dim < 1:
            raise ValueError("latent_dim must be >= 1.")
        self.n_time = int(n_time)
        self.states = int(states)
        self.latent_dim = int(latent_dim)
        self.observation_dim = int(n_neurons)
        self.transition_stickiness = float(transition_stickiness)
        self.dynamics_l2_A = float(dynamics_l2_A)
        self.dynamics_l2_b = float(dynamics_l2_b)
        self.dynamics_var_floor = float(dynamics_var_floor)
        self.emission_l2 = float(emission_l2)
        self.emission_lr = float(emission_lr)
        self.emission_mstep_steps = int(emission_mstep_steps)
        self.emission_batch_size = int(emission_batch_size)
        self.emission_max_points = emission_max_points
        self.em_alpha = float(em_alpha)
        self.local_inference_iters = int(local_inference_iters)
        self.continuous_steps = int(continuous_steps)
        self.continuous_lr = float(continuous_lr)
        self.continuous_tolerance = float(continuous_tolerance)
        self.latent_l2 = float(latent_l2)
        self.latent_clip = None if latent_clip is None else float(latent_clip)
        self.init_seed = init_seed
        self.init_max_points = int(init_max_points)
        self.init_em_iters = int(init_em_iters)
        self.prediction_floor = float(prediction_floor)
        self.objective = objective

        self.register_buffer("_initialized", torch.tensor(False))
        self._make_parameters(self.observation_dim)

    @property
    def initialized(self) -> bool:
        return bool(self._initialized.item())

    def forward(self, x: Tensor) -> ModelOutput:
        if x.ndim != 3:
            raise ValueError("SLDS expects input shape (batch, time, neurons).")
        full, mask, observed_dim = self._observed_to_full(x.float())
        if not self.initialized:
            self._initialize_from_observations(full, mask)
        with torch.enable_grad():
            rates, latents, gamma, _, loss = self._infer_batch(full, mask, optimize=True)
        return ModelOutput(
            rates=rates[..., :observed_dim].clamp_min(self.prediction_floor),
            latents=latents,
            reconstruction=rates[..., :observed_dim],
            extras={"state_probs": gamma, "slds_objective": loss},
        )

    def loss(
        self,
        batch: Tensor | dict[str, Tensor],
        output: ModelOutput,
        epoch: int = 0,
    ) -> LossOutput:
        del epoch
        x = observations_from_batch(batch).to(device=output.rates.device, dtype=output.rates.dtype)
        total = _poisson_nll(output.rates, x, self.prediction_floor).mean()
        return LossOutput(
            total=total,
            named_terms={"poisson_nll": total},
            objective=self.objective,
        )

    def evaluation_adapter(self, task: str):
        if task == "nlb":
            return SLDSNLBEvaluationAdapter()
        return None

    def fit_em_epoch_from_loader(
        self,
        loader: Iterable,
        device: torch.device | str,
        epoch: int = 0,
    ) -> LossOutput:
        del epoch
        torch_device = torch.device(device)
        self.to(torch_device)
        if not self.initialized:
            self._initialize_from_loader(loader, torch_device)

        stats = self._empty_stats(torch_device)
        emission_xs: list[Tensor] = []
        emission_ys: list[Tensor] = []
        emission_masks: list[Tensor] = []
        losses: list[Tensor] = []
        emission_rows = 0

        for batch in loader:
            batch = move_batch_to_device(batch, torch_device)
            y, mask = self._full_observations_from_batch(batch)
            rates, latents, gamma, xi, loss, covariances, cross_moments = self._infer_batch(
                y,
                mask,
                optimize=True,
                return_moments=True,
            )
            del rates
            losses.append(loss.detach())
            self._accumulate_stats(
                stats,
                latents.detach(),
                gamma.detach(),
                xi.detach(),
                covariances.detach(),
                cross_moments.detach(),
            )
            emission_rows = self._append_emission_rows(
                emission_xs,
                emission_ys,
                emission_masks,
                latents.detach(),
                y.detach(),
                mask.detach(),
                emission_rows,
            )

        if emission_xs:
            self._m_step(stats, emission_xs, emission_ys, emission_masks)

        if losses:
            total = torch.stack(losses).mean()
        else:
            total = torch.tensor(float("nan"), device=torch_device)
        return LossOutput(
            total=total,
            named_terms={"slds_em_objective": total},
            objective=self.objective,
        )

    def _append_emission_rows(
        self,
        emission_xs: list[Tensor],
        emission_ys: list[Tensor],
        emission_masks: list[Tensor],
        x: Tensor,
        y: Tensor,
        mask: Tensor,
        rows_seen: int,
    ) -> int:
        x_rows = x.reshape(-1, x.shape[-1])
        y_rows = y.reshape(-1, y.shape[-1])
        mask_rows = mask.reshape(-1, mask.shape[-1])
        if self.emission_max_points is None:
            emission_xs.append(x_rows.detach().cpu())
            if torch.is_floating_point(y_rows) and torch.allclose(y_rows, y_rows.round()):
                y_rows = y_rows.round().to(dtype=torch.int16)
            emission_ys.append(y_rows.detach().cpu())
            if bool(mask_rows.all().item()):
                emission_masks.append(torch.empty(0, dtype=torch.bool))
            else:
                emission_masks.append(mask_rows.detach().cpu())
            return rows_seen + x_rows.shape[0]
        remaining = int(self.emission_max_points) - rows_seen
        if remaining <= 0:
            return rows_seen
        take = min(remaining, x_rows.shape[0])
        if take < x_rows.shape[0]:
            idx = torch.randperm(x_rows.shape[0], device=x_rows.device)[:take]
            x_rows = x_rows.index_select(0, idx)
            y_rows = y_rows.index_select(0, idx)
            mask_rows = mask_rows.index_select(0, idx)
        emission_xs.append(x_rows.detach().cpu())
        if torch.is_floating_point(y_rows) and torch.allclose(y_rows, y_rows.round()):
            y_rows = y_rows.round().to(dtype=torch.int16)
        emission_ys.append(y_rows.detach().cpu())
        if bool(mask_rows.all().item()):
            emission_masks.append(torch.empty(0, dtype=torch.bool))
        else:
            emission_masks.append(mask_rows.detach().cpu())
        return rows_seen + take

    def predict_nlb_heldout(self, heldin: Tensor, n_heldout: int) -> Tensor:
        if not self.initialized:
            raise RuntimeError("SLDS must be fitted before NLB evaluation.")
        heldin = heldin.float()
        batch, time, n_heldin = heldin.shape
        if self.observation_dim < n_heldin + n_heldout:
            self._ensure_observation_dim(n_heldin + n_heldout)
        full = torch.zeros(
            batch,
            time,
            self.observation_dim,
            device=heldin.device,
            dtype=heldin.dtype,
        )
        mask = torch.zeros_like(full, dtype=torch.bool)
        full[..., :n_heldin] = heldin
        mask[..., :n_heldin] = True
        with torch.enable_grad():
            rates, *_ = self._infer_batch(full, mask, optimize=True)
        return rates[..., n_heldin : n_heldin + n_heldout].clamp_min(self.prediction_floor)

    def _make_parameters(self, observation_dim: int) -> None:
        k = self.states
        d = self.latent_dim
        n = int(observation_dim)
        dtype = torch.get_default_dtype()

        pi = torch.full((k,), 1.0 / k, dtype=dtype)
        sticky = self.transition_stickiness
        transition = (1.0 - sticky) / max(k - 1, 1) * torch.ones(k, k, dtype=dtype)
        transition.fill_diagonal_(sticky)
        if k == 1:
            transition.fill_(1.0)

        eye = torch.eye(d, dtype=dtype)
        A = eye.unsqueeze(0).repeat(k, 1, 1)
        A = A + 0.03 * torch.randn(k, d, d)
        C = 0.05 * torch.randn(n, d, dtype=dtype)

        self.log_initial_probs = nn.Parameter(torch.log(pi))
        self.log_transition = nn.Parameter(torch.log(transition))
        self.A = nn.Parameter(A.to(dtype=dtype))
        self.b = nn.Parameter(torch.zeros(k, d, dtype=dtype))
        self.sqrt_Sigmas = nn.Parameter(0.1 * torch.randn(k, d, d, dtype=dtype))
        self.init_mean = nn.Parameter(torch.zeros(k, d, dtype=dtype), requires_grad=False)
        self.sqrt_Sigmas_init = nn.Parameter(
            eye.unsqueeze(0).repeat(k, 1, 1),
            requires_grad=False,
        )
        self.C = nn.Parameter(C)
        self.d = nn.Parameter(torch.zeros(n, dtype=dtype))

    def _ensure_observation_dim(self, observation_dim: int) -> None:
        observation_dim = int(observation_dim)
        if observation_dim == self.observation_dim:
            return
        if observation_dim < 1:
            raise ValueError("observation_dim must be positive.")
        old_C = self.C.detach()
        old_d = self.d.detach()
        device = old_C.device
        dtype = old_C.dtype
        new_C = 0.05 * torch.randn(
            observation_dim,
            self.latent_dim,
            device=device,
            dtype=dtype,
        )
        new_d = torch.zeros(observation_dim, device=device, dtype=dtype)
        rows = min(self.observation_dim, observation_dim)
        new_C[:rows] = old_C[:rows]
        new_d[:rows] = old_d[:rows]
        self.observation_dim = observation_dim
        self.C = nn.Parameter(new_C)
        self.d = nn.Parameter(new_d)

    def _observed_to_full(self, x: Tensor) -> tuple[Tensor, Tensor, int]:
        observed_dim = int(x.shape[-1])
        if observed_dim > self.observation_dim:
            self._ensure_observation_dim(observed_dim)
        if observed_dim == self.observation_dim:
            return x, torch.ones_like(x, dtype=torch.bool), observed_dim
        full = torch.zeros(
            *x.shape[:-1],
            self.observation_dim,
            device=x.device,
            dtype=x.dtype,
        )
        mask = torch.zeros_like(full, dtype=torch.bool)
        full[..., :observed_dim] = x
        mask[..., :observed_dim] = True
        return full, mask, observed_dim

    def _full_observations_from_batch(self, batch: Tensor | dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        x = observations_from_batch(batch).float()
        if isinstance(batch, dict) and "heldout_spikes" in batch:
            y = torch.cat([x, batch["heldout_spikes"].float()], dim=-1)
        else:
            y = x
        self._ensure_observation_dim(int(y.shape[-1]))
        mask = torch.ones_like(y, dtype=torch.bool)
        return y, mask

    def _initialize_from_loader(self, loader: Iterable, device: torch.device) -> None:
        if self.init_seed is not None:
            torch.manual_seed(int(self.init_seed))
        sum_rates: Tensor | None = None
        counts: Tensor | None = None
        flat_rows: list[Tensor] = []
        flat_row_count = 0
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            y, mask = self._full_observations_from_batch(batch)
            y = y.to(device=self.C.device, dtype=self.C.dtype)
            mask = mask.to(device=y.device)
            batch_sum = (y * mask).sum(dim=(0, 1))
            batch_counts = mask.sum(dim=(0, 1)).to(dtype=y.dtype)
            sum_rates = batch_sum if sum_rates is None else sum_rates + batch_sum
            counts = batch_counts if counts is None else counts + batch_counts

            link = _inv_softplus(y.clamp_min(0.1))
            flat = link.reshape(-1, link.shape[-1])
            flat_mask = mask.reshape(-1, mask.shape[-1])
            observed_rows = flat_mask.all(dim=1)
            if observed_rows.any():
                flat_rows.append(flat[observed_rows].detach().cpu())
                flat_row_count += int(observed_rows.sum().item())
                if flat_row_count > 2 * self.init_max_points:
                    flat = torch.cat(flat_rows, dim=0)
                    perm = torch.randperm(flat.shape[0])[: self.init_max_points]
                    flat_rows = [flat.index_select(0, perm)]
                    flat_row_count = flat_rows[0].shape[0]

        if sum_rates is None or counts is None:
            return
        self._initialize_emissions_from_summary(sum_rates, counts, flat_rows)

        latents: list[Tensor] = []
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            y, mask = self._full_observations_from_batch(batch)
            y = y.to(device=self.C.device, dtype=self.C.dtype)
            mask = mask.to(device=y.device)
            latents.append(self._initial_latents(y, mask).detach())
        if latents:
            self._initialize_arhmm_from_latents(torch.cat(latents, dim=0))
            self._initialized.copy_(torch.tensor(True, device=self._initialized.device))

    @torch.no_grad()
    def _initialize_from_observations(self, y: Tensor, mask: Tensor) -> None:
        self._ensure_observation_dim(int(y.shape[-1]))
        y = y.to(device=self.C.device, dtype=self.C.dtype)
        mask = mask.to(device=y.device)
        link = _inv_softplus(y.clamp_min(0.1))
        counts = mask.sum(dim=(0, 1)).clamp_min(1)
        mean_rates = (y * mask).sum(dim=(0, 1)) / counts
        bias = _inv_softplus(mean_rates.clamp_min(self.prediction_floor))
        flat = link.reshape(-1, link.shape[-1])
        flat_mask = mask.reshape(-1, mask.shape[-1])
        observed_rows = flat_mask.all(dim=1)
        if observed_rows.any():
            flat = flat[observed_rows]
        if flat.shape[0] > self.init_max_points:
            perm = torch.randperm(flat.shape[0], device=flat.device)[: self.init_max_points]
            flat = flat.index_select(0, perm)
        bias = bias.to(device=flat.device, dtype=flat.dtype)
        centered = flat - bias
        q = min(self.latent_dim, centered.shape[0], centered.shape[1])
        if q > 0:
            _, _, v = torch.pca_lowrank(centered.float(), q=q, center=False)
            C = torch.zeros_like(self.C)
            C[:, :q] = v[:, :q].to(device=C.device, dtype=C.dtype)
            self.C.copy_(C)
        self.d.copy_(bias.to(device=self.d.device, dtype=self.d.dtype))

        latents = self._initial_latents(y, mask)
        self._initialize_arhmm_from_latents(latents)
        self._initialized.copy_(torch.tensor(True, device=self._initialized.device))

    @torch.no_grad()
    def _initialize_emissions_from_summary(
        self,
        sum_rates: Tensor,
        counts: Tensor,
        flat_rows: list[Tensor],
    ) -> None:
        counts = counts.to(device=self.C.device, dtype=self.C.dtype).clamp_min(1)
        sum_rates = sum_rates.to(device=self.C.device, dtype=self.C.dtype)
        mean_rates = sum_rates / counts
        bias = _inv_softplus(mean_rates.clamp_min(self.prediction_floor))
        if flat_rows:
            flat = torch.cat(flat_rows, dim=0).to(device=self.C.device, dtype=self.C.dtype)
            if flat.shape[0] > self.init_max_points:
                perm = torch.randperm(flat.shape[0], device=flat.device)[: self.init_max_points]
                flat = flat.index_select(0, perm)
            centered = flat - bias
            q = min(self.latent_dim, centered.shape[0], centered.shape[1])
            if q > 0:
                _, _, v = torch.pca_lowrank(centered.float(), q=q, center=False)
                C = torch.zeros_like(self.C)
                C[:, :q] = v[:, :q].to(device=C.device, dtype=C.dtype)
                self.C.copy_(C)
        self.d.copy_(bias.to(device=self.d.device, dtype=self.d.dtype))

    @torch.no_grad()
    def _initialize_arhmm_from_latents(self, latents: Tensor) -> None:
        latents = latents.to(device=self.A.device, dtype=self.A.dtype)
        labels = self._initial_state_labels(latents)
        stats = self._empty_stats(latents.device)
        for start, stop in self._trial_chunks(latents.shape[0]):
            gamma = F.one_hot(labels[start:stop], num_classes=self.states).to(dtype=latents.dtype)
            xi = self._hard_transition_expectations(
                labels[start:stop],
                latents.device,
                latents.dtype,
            )
            self._accumulate_stats(stats, latents[start:stop], gamma, xi)
        self._m_step_markov_and_dynamics(stats)

        for _ in range(max(self.init_em_iters, 0)):
            stats = self._empty_stats(latents.device)
            for start, stop in self._trial_chunks(latents.shape[0]):
                chunk = latents[start:stop]
                gamma, xi, _ = self._posterior_discrete(chunk)
                self._accumulate_stats(stats, chunk, gamma, xi)
            self._m_step_markov_and_dynamics(stats)

    def _trial_chunks(self, n_trials: int) -> Iterable[tuple[int, int]]:
        chunk_size = 8 if self.latent_dim >= 32 else 32
        for start in range(0, n_trials, chunk_size):
            yield start, min(start + chunk_size, n_trials)

    def _initial_state_labels(self, latents: Tensor) -> Tensor:
        batch, time, dim = latents.shape
        if self.states == 1:
            return torch.zeros(batch, time, device=latents.device, dtype=torch.long)
        flat = latents.reshape(-1, dim)
        if flat.shape[0] > self.init_max_points:
            perm = torch.randperm(flat.shape[0], device=flat.device)[: self.init_max_points]
            sample = flat.index_select(0, perm)
        else:
            sample = flat
        try:
            from sklearn.cluster import KMeans

            kmeans = KMeans(
                n_clusters=self.states,
                n_init=5,
                random_state=self.init_seed,
            )
            kmeans.fit(sample.detach().cpu().numpy())
            centers = torch.as_tensor(
                kmeans.cluster_centers_,
                device=latents.device,
                dtype=latents.dtype,
            )
            distances = torch.cdist(flat, centers)
            labels = distances.argmin(dim=1)
        except Exception:
            labels = torch.randint(
                self.states,
                (flat.shape[0],),
                device=latents.device,
            )
        return labels.reshape(batch, time)

    def _hard_transition_expectations(
        self,
        labels: Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if labels.shape[1] <= 1:
            return torch.empty(
                labels.shape[0],
                0,
                self.states,
                self.states,
                device=device,
                dtype=dtype,
            )
        prev = F.one_hot(labels[:, :-1], num_classes=self.states).to(dtype=dtype)
        nxt = F.one_hot(labels[:, 1:], num_classes=self.states).to(dtype=dtype)
        return prev[..., :, None] * nxt[..., None, :]

    def _infer_batch(
        self,
        y: Tensor,
        mask: Tensor,
        optimize: bool,
        return_moments: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        y = y.to(device=self.C.device, dtype=self.C.dtype)
        mask = mask.to(device=y.device)
        x = self._initial_latents(y, mask)
        gamma = None
        xi = None
        for _ in range(max(self.local_inference_iters, 1)):
            with torch.no_grad():
                gamma, xi, _ = self._posterior_discrete(x)
            if optimize and self.continuous_steps > 0:
                x = self._newton_continuous(y, mask, x, gamma)
        with torch.no_grad():
            gamma, xi, _ = self._posterior_discrete(x)
            if optimize:
                x, covariances, cross_moments = self._laplace_moments(x, y, mask, gamma)
                gamma, xi, _ = self._posterior_discrete(x)
            else:
                covariances = torch.empty(0, device=y.device, dtype=y.dtype)
                cross_moments = torch.empty(0, device=y.device, dtype=y.dtype)
            rates = self._emission_rates(x)
            loss = self._complete_objective(x, y, mask, gamma)
        if return_moments:
            return rates, x, gamma, xi, loss, covariances, cross_moments
        return rates, x, gamma, xi, loss

    def _initial_latents(self, y: Tensor, mask: Tensor) -> Tensor:
        link = _inv_softplus(y.clamp_min(0.1))
        obs_dims = mask.any(dim=(0, 1))
        if not obs_dims.any():
            return torch.zeros(
                y.shape[0],
                y.shape[1],
                self.latent_dim,
                device=y.device,
                dtype=y.dtype,
            )
        C_obs = self.C[obs_dims].to(device=y.device, dtype=y.dtype)
        d_obs = self.d[obs_dims].to(device=y.device, dtype=y.dtype)
        centered = link[..., obs_dims] - d_obs
        projection = torch.linalg.pinv(C_obs.T.float()).to(device=y.device, dtype=y.dtype)
        latents = centered.reshape(-1, centered.shape[-1]) @ projection
        return latents.reshape(y.shape[0], y.shape[1], self.latent_dim)

    def _newton_continuous(self, y: Tensor, mask: Tensor, x: Tensor, gamma: Tensor) -> Tensor:
        x_current = x.detach().clone()
        scale = float(max(x_current.shape[1] * x_current.shape[2], 1))
        for _ in range(max(self.continuous_steps, 1)):
            x_var = x_current.detach().clone().requires_grad_(True)
            objective = self._continuous_objective_per_trial(x_var, y, mask, gamma) / scale
            (grad,) = torch.autograd.grad(objective.sum(), x_var)
            diag, lower = self._laplace_hessian_blocks(x_var.detach(), y, mask, gamma)
            diag = diag / scale
            lower = lower / scale
            step_direction = -_solve_symm_block_tridiag(diag, lower, grad)
            lambda_sq = torch.sum(grad * -step_direction, dim=(1, 2))
            if torch.all(lambda_sq / 2.0 <= self.continuous_tolerance):
                break
            step = self._newton_line_search(
                x_current,
                step_direction,
                grad,
                objective.detach(),
                y,
                mask,
                gamma,
                scale,
            )
            x_current = x_current + step[:, None, None] * step_direction
            if self.latent_clip is not None:
                x_current = x_current.clamp(
                    min=-self.latent_clip,
                    max=self.latent_clip,
                )
            if not torch.isfinite(x_current).all():
                return x.detach()
        return x_current.detach()

    def _newton_line_search(
        self,
        x: Tensor,
        direction: Tensor,
        grad: Tensor,
        objective: Tensor,
        y: Tensor,
        mask: Tensor,
        gamma: Tensor,
        scale: float,
    ) -> Tensor:
        alpha = 0.2
        beta = 0.7
        min_step = 1e-8
        step = torch.full(
            (x.shape[0],),
            max(self.continuous_lr, min_step),
            device=x.device,
            dtype=x.dtype,
        )
        grad_term = alpha * torch.sum(grad * direction, dim=(1, 2))
        with torch.no_grad():
            while torch.any(step > min_step):
                candidate = x + step[:, None, None] * direction
                candidate_obj = (
                    self._continuous_objective_per_trial(candidate, y, mask, gamma) / scale
                )
                reject = (
                    ~torch.isfinite(candidate_obj)
                    | (candidate_obj > objective + step * grad_term)
                ) & (step > min_step)
                if not torch.any(reject):
                    break
                step = torch.where(reject, step * beta, step)
        return step.clamp_min(min_step)

    def _posterior_discrete(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        log_likes = self._dynamics_log_likelihoods(x)
        log_pi0 = F.log_softmax(self.log_initial_probs, dim=0)
        log_P = F.log_softmax(self.log_transition, dim=1)
        return _hmm_expected_states(log_pi0, log_P, log_likes)

    def _dynamics_log_likelihoods(self, x: Tensor) -> Tensor:
        batch, time, _ = x.shape
        dtype = x.dtype
        device = x.device
        log_two_pi = math.log(2.0 * math.pi)

        init_cov = self._initial_covariances(device=device, dtype=dtype)
        init_precision, init_logdet = _precision_and_logdet(init_cov)
        init_mean = self.init_mean.to(device=device, dtype=dtype)
        init_diff = x[:, 0, None, :] - init_mean[None, :, :]
        init_quad = torch.einsum("bkd,kde,bke->bk", init_diff, init_precision, init_diff)
        init_ll = -0.5 * (init_quad + init_logdet[None, :] + self.latent_dim * log_two_pi)

        if time == 1:
            return init_ll[:, None, :]

        A = self.A.to(device=device, dtype=dtype)
        b = self.b.to(device=device, dtype=dtype)
        dyn_cov = self._dynamics_covariances(device=device, dtype=dtype)
        dyn_precision, dyn_logdet = _precision_and_logdet(dyn_cov)
        pred = torch.einsum("btd,ked->btke", x[:, :-1], A) + b[None, None, :, :]
        diff = x[:, 1:, None, :] - pred
        ar_quad = torch.einsum("btkd,kde,btke->btk", diff, dyn_precision, diff)
        ar_ll = -0.5 * (
            ar_quad + dyn_logdet[None, None, :] + self.latent_dim * log_two_pi
        )
        return torch.cat([init_ll[:, None, :], ar_ll], dim=1)

    def _dynamics_covariances(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        root = self.sqrt_Sigmas.to(device=device, dtype=dtype)
        cov = root @ root.transpose(-1, -2)
        eye = torch.eye(self.latent_dim, device=device, dtype=dtype)
        return _symmetrize(cov) + self.dynamics_var_floor * eye[None]

    def _initial_covariances(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        root = self.sqrt_Sigmas_init.to(device=device, dtype=dtype)
        cov = root @ root.transpose(-1, -2)
        eye = torch.eye(self.latent_dim, device=device, dtype=dtype)
        return _symmetrize(cov) + self.dynamics_var_floor * eye[None]

    def _emission_rates(self, x: Tensor) -> Tensor:
        eta = torch.einsum("btd,nd->btn", x, self.C.to(device=x.device, dtype=x.dtype))
        eta = eta + self.d.to(device=x.device, dtype=x.dtype)
        return F.softplus(eta).clamp_min(self.prediction_floor)

    def _continuous_objective_per_trial(
        self,
        x: Tensor,
        y: Tensor,
        mask: Tensor,
        gamma: Tensor,
    ) -> Tensor:
        rates = self._emission_rates(x)
        emission = (_poisson_nll(rates, y, self.prediction_floor) * mask).sum(dim=(1, 2))
        dynamics_ll = self._dynamics_log_likelihoods(x)
        dynamics = -(gamma * dynamics_ll).sum(dim=(1, 2))
        latent_penalty = self.latent_l2 * x.pow(2).sum(dim=(1, 2))
        return emission + dynamics + latent_penalty

    def _complete_objective(
        self,
        x: Tensor,
        y: Tensor,
        mask: Tensor,
        gamma: Tensor,
    ) -> Tensor:
        scale = mask.sum().clamp_min(1).to(dtype=y.dtype)
        return self._continuous_objective_per_trial(x, y, mask, gamma).sum() / scale

    def _laplace_hessian_blocks(
        self,
        x: Tensor,
        y: Tensor,
        mask: Tensor,
        gamma: Tensor,
    ) -> tuple[Tensor, Tensor]:
        J_ini, J_dyn_11, J_dyn_21, J_dyn_22 = self._dynamics_hessian_params(gamma)
        J_obs = self._emission_hessian_params(x, y, mask)
        batch, time, dim = x.shape
        diag = J_obs.clone()
        diag[:, 0] = diag[:, 0] + J_ini
        if time > 1:
            diag[:, :-1] = diag[:, :-1] + J_dyn_11
            diag[:, 1:] = diag[:, 1:] + J_dyn_22
            lower = J_dyn_21
        else:
            lower = torch.empty(batch, 0, dim, dim, device=x.device, dtype=x.dtype)
        if self.latent_l2 > 0.0:
            eye = torch.eye(dim, device=x.device, dtype=x.dtype)
            diag = diag + (2.0 * self.latent_l2) * eye[None, None]
        return _symmetrize(diag), lower

    def _laplace_moments(
        self,
        x: Tensor,
        y: Tensor,
        mask: Tensor,
        gamma: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        J_ini, J_dyn_11, J_dyn_21, J_dyn_22 = self._dynamics_hessian_params(gamma)
        J_obs = self._emission_hessian_params(x, y, mask)
        h_ini, h_dyn_1, h_dyn_2, h_obs = self._hessian_params_to_hs(
            x,
            J_ini,
            J_dyn_11,
            J_dyn_21,
            J_dyn_22,
            J_obs,
        )
        means, covariances, cross_moments = _kalman_info_smoother(
            J_ini,
            h_ini,
            J_dyn_11,
            J_dyn_21,
            J_dyn_22,
            h_dyn_1,
            h_dyn_2,
            J_obs,
            h_obs,
        )
        return means, covariances, cross_moments

    def _dynamics_hessian_params(self, gamma: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        device = gamma.device
        dtype = gamma.dtype
        init_cov = self._initial_covariances(device=device, dtype=dtype)
        init_precision, _ = _precision_and_logdet(init_cov)
        J_ini = torch.sum(gamma[:, 0, :, None, None] * init_precision[None], dim=1)
        if gamma.shape[1] <= 1:
            empty = torch.empty(
                gamma.shape[0],
                0,
                self.latent_dim,
                self.latent_dim,
                device=device,
                dtype=dtype,
            )
            return J_ini, empty, empty, empty
        dyn_cov = self._dynamics_covariances(device=device, dtype=dtype)
        dyn_precision, _ = _precision_and_logdet(dyn_cov)
        A = self.A.to(device=device, dtype=dtype)
        dyn_terms = A.transpose(-1, -2) @ dyn_precision @ A
        off_diag_terms = dyn_precision @ A
        weights = gamma[:, 1:]
        J_dyn_11 = torch.sum(weights[..., None, None] * dyn_terms[None, None], dim=2)
        J_dyn_22 = torch.sum(weights[..., None, None] * dyn_precision[None, None], dim=2)
        J_dyn_21 = -torch.sum(weights[..., None, None] * off_diag_terms[None, None], dim=2)
        return J_ini, _symmetrize(J_dyn_11), J_dyn_21, _symmetrize(J_dyn_22)

    def _emission_hessian_params(self, x: Tensor, y: Tensor, mask: Tensor) -> Tensor:
        del y
        rates = self._emission_rates(x)
        weights = rates * mask.to(dtype=x.dtype)
        C = self.C.to(device=x.device, dtype=x.dtype)
        return torch.einsum("btn,ni,nj->btij", weights, C, C)

    def _hessian_params_to_hs(
        self,
        x: Tensor,
        J_ini: Tensor,
        J_dyn_11: Tensor,
        J_dyn_21: Tensor,
        J_dyn_22: Tensor,
        J_obs: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        h_ini = torch.einsum("bij,bj->bi", J_ini, x[:, 0])
        h_obs = torch.einsum("btij,btj->bti", J_obs, x)
        if x.shape[1] <= 1:
            empty = torch.empty(
                x.shape[0],
                0,
                x.shape[2],
                device=x.device,
                dtype=x.dtype,
            )
            return h_ini, empty, empty, h_obs
        h_dyn_1 = torch.einsum("btij,btj->bti", J_dyn_11, x[:, :-1])
        h_dyn_1 = h_dyn_1 + torch.einsum(
            "btji,btj->bti",
            J_dyn_21,
            x[:, 1:],
        )
        h_dyn_2 = torch.einsum("btij,btj->bti", J_dyn_22, x[:, 1:])
        h_dyn_2 = h_dyn_2 + torch.einsum(
            "btij,btj->bti",
            J_dyn_21,
            x[:, :-1],
        )
        return h_ini, h_dyn_1, h_dyn_2, h_obs

    def _empty_stats(self, device: torch.device) -> dict[str, Tensor]:
        dtype = self.C.dtype
        k = self.states
        d = self.latent_dim
        return {
            "pi": torch.zeros(k, device=device, dtype=dtype),
            "trans": torch.zeros(k, k, device=device, dtype=dtype),
            "init_w": torch.zeros(k, device=device, dtype=dtype),
            "init_sum": torch.zeros(k, d, device=device, dtype=dtype),
            "init_sq": torch.zeros(k, d, d, device=device, dtype=dtype),
            "dyn_w": torch.zeros(k, device=device, dtype=dtype),
            "dyn_xx": torch.zeros(k, d + 1, d + 1, device=device, dtype=dtype),
            "dyn_xy": torch.zeros(k, d + 1, d, device=device, dtype=dtype),
            "dyn_yy": torch.zeros(k, d, d, device=device, dtype=dtype),
        }

    def _accumulate_stats(
        self,
        stats: dict[str, Tensor],
        x: Tensor,
        gamma: Tensor,
        xi: Tensor,
        covariances: Tensor | None = None,
        cross_moments: Tensor | None = None,
    ) -> None:
        stats["pi"] += gamma[:, 0].sum(dim=0)
        if xi.numel() > 0:
            stats["trans"] += xi.sum(dim=(0, 1))
        stats["init_w"] += gamma[:, 0].sum(dim=0)
        stats["init_sum"] += torch.einsum("bk,bd->kd", gamma[:, 0], x[:, 0])
        if covariances is None or covariances.numel() == 0:
            ExxT = torch.einsum("bti,btj->btij", x, x)
        else:
            ExxT = covariances + torch.einsum("bti,btj->btij", x, x)
        stats["init_sq"] += torch.einsum("bk,bij->kij", gamma[:, 0], ExxT[:, 0])

        if x.shape[1] <= 1:
            return
        weights = gamma[:, 1:]
        prev = x[:, :-1]
        nxt = x[:, 1:]
        if cross_moments is None or cross_moments.numel() == 0:
            ExxnT = torch.einsum("bti,btj->btij", prev, nxt)
        else:
            ExxnT = cross_moments

        stats["dyn_w"] += weights.sum(dim=(0, 1))
        stats["dyn_xx"][:, : self.latent_dim, : self.latent_dim] += torch.einsum(
            "btk,btij->kij",
            weights,
            ExxT[:, :-1],
        )
        prev_sum = torch.einsum("btk,bti->ki", weights, prev)
        stats["dyn_xx"][:, : self.latent_dim, -1] += prev_sum
        stats["dyn_xx"][:, -1, : self.latent_dim] += prev_sum
        stats["dyn_xx"][:, -1, -1] += weights.sum(dim=(0, 1))
        stats["dyn_xy"][:, : self.latent_dim, :] += torch.einsum(
            "btk,btij->kij",
            weights,
            ExxnT,
        )
        stats["dyn_xy"][:, -1, :] += torch.einsum("btk,bti->ki", weights, nxt)
        stats["dyn_yy"] += torch.einsum("btk,btij->kij", weights, ExxT[:, 1:])

    def _m_step(
        self,
        stats: dict[str, Tensor],
        emission_xs: list[Tensor],
        emission_ys: list[Tensor],
        emission_masks: list[Tensor],
    ) -> None:
        self._m_step_markov_and_dynamics(stats)
        self._fit_emissions(emission_xs, emission_ys, emission_masks)

    def _m_step_markov_and_dynamics(self, stats: dict[str, Tensor]) -> None:
        with torch.no_grad():
            eps = torch.finfo(self.C.dtype).eps
            pi = (stats["pi"] + eps) / (stats["pi"].sum() + eps * self.states)
            trans = stats["trans"] + eps
            trans = trans / trans.sum(dim=1, keepdim=True).clamp_min(eps)
            self.log_initial_probs.copy_(
                self.em_alpha * self.log_initial_probs
                + (1.0 - self.em_alpha) * torch.log(pi)
            )
            self.log_transition.copy_(
                self.em_alpha * self.log_transition
                + (1.0 - self.em_alpha) * torch.log(trans)
            )

            J0_diag = torch.zeros(
                self.latent_dim + 1,
                device=self.C.device,
                dtype=self.C.dtype,
            )
            J0_diag[: self.latent_dim] = self.dynamics_l2_A
            J0_diag[-1] = self.dynamics_l2_b
            J0 = torch.diag(J0_diag)
            h0 = torch.zeros(
                self.latent_dim + 1,
                self.latent_dim,
                device=self.C.device,
                dtype=self.C.dtype,
            )
            h0[: self.latent_dim] = self.dynamics_l2_A * torch.eye(
                self.latent_dim,
                device=self.C.device,
                dtype=self.C.dtype,
            )
            eye = torch.eye(self.latent_dim, device=self.C.device, dtype=self.C.dtype)
            new_A = torch.empty_like(self.A)
            new_b = torch.empty_like(self.b)
            new_cov = torch.empty(
                self.states,
                self.latent_dim,
                self.latent_dim,
                device=self.C.device,
                dtype=self.C.dtype,
            )
            for state in range(self.states):
                lhs = stats["dyn_xx"][state] + J0
                rhs = stats["dyn_xy"][state] + h0
                solution = _safe_solve(lhs, rhs)
                W = solution.T
                new_A[state] = W[:, : self.latent_dim]
                new_b[state] = W[:, -1]

                EWxyT = W @ stats["dyn_xy"][state]
                sqerr = (
                    stats["dyn_yy"][state]
                    - EWxyT.T
                    - EWxyT
                    + W @ stats["dyn_xx"][state] @ W.T
                )
                nu = 1e-4 + stats["dyn_w"][state]
                cov = (sqerr + self.dynamics_var_floor * eye) / (
                    nu + self.latent_dim + 1.0
                )
                new_cov[state] = _symmetrize(cov) + self.dynamics_var_floor * eye

            used = torch.where(stats["dyn_w"] > 1.0)[0]
            unused = torch.where(stats["dyn_w"] <= 1.0)[0]
            if used.numel() == 0:
                used = torch.arange(self.states, device=self.C.device)
            for state in unused.tolist():
                source = int(used[state % used.numel()].item())
                new_A[state] = new_A[source] + 0.01 * torch.randn_like(new_A[source])
                new_b[state] = new_b[source] + 0.01 * torch.randn_like(new_b[source])
                new_cov[state] = new_cov[source]

            self.A.copy_(new_A)
            self.b.copy_(new_b)
            self.sqrt_Sigmas.copy_(_safe_cholesky(new_cov, self.dynamics_var_floor))

    def _fit_emissions(
        self,
        emission_xs: list[Tensor],
        emission_ys: list[Tensor],
        emission_masks: list[Tensor],
    ) -> None:
        if self.emission_mstep_steps <= 0 or not emission_xs:
            return
        x = torch.cat([item.reshape(-1, item.shape[-1]) for item in emission_xs], dim=0)
        x = x.to(device=self.C.device, dtype=self.C.dtype)
        y = torch.cat([item.reshape(-1, item.shape[-1]) for item in emission_ys], dim=0)
        y = y.to(device=self.C.device)
        if all(item.numel() == 0 for item in emission_masks):
            mask = None
        else:
            mask = torch.cat(
                [item.reshape(-1, item.shape[-1]) for item in emission_masks],
                dim=0,
            ).to(device=self.C.device)
        if self.emission_max_points is not None and x.shape[0] > self.emission_max_points:
            perm = torch.randperm(x.shape[0], device=x.device)[: self.emission_max_points]
            x = x.index_select(0, perm)
            y = y.index_select(0, perm)
            if mask is not None:
                mask = mask.index_select(0, perm)

        old_C = self.C.detach().clone()
        old_d = self.d.detach().clone()
        optimizer = torch.optim.LBFGS(
            [self.C, self.d],
            lr=self.emission_lr,
            max_iter=self.emission_mstep_steps,
            line_search_fn="strong_wolfe",
        )
        rows = x.shape[0]
        batch_size = max(min(self.emission_batch_size, rows), 1)
        if mask is None:
            denom = torch.tensor(y.numel(), device=x.device, dtype=x.dtype).clamp_min(1)
        else:
            denom = mask.sum().clamp_min(1).to(dtype=x.dtype)

        def closure() -> Tensor:
            optimizer.zero_grad(set_to_none=True)
            total = torch.zeros((), device=x.device, dtype=x.dtype)
            for start in range(0, rows, batch_size):
                xb = x[start : start + batch_size]
                yb = y[start : start + batch_size].to(dtype=x.dtype)
                mb = None if mask is None else mask[start : start + batch_size]
                eta = xb @ self.C.T + self.d
                rates = F.softplus(eta).clamp_min(self.prediction_floor)
                loss_terms = _poisson_nll(rates, yb, self.prediction_floor)
                if mb is not None:
                    loss_terms = loss_terms * mb
                loss = loss_terms.sum() / denom
                loss.backward()
                total = total + loss.detach()
            if self.emission_l2 > 0.0:
                reg = self.emission_l2 * self.C.pow(2).mean()
                reg.backward()
                total = total + reg.detach()
            return total

        optimizer.step(closure)
        with torch.no_grad():
            self.C.copy_(self.em_alpha * old_C + (1.0 - self.em_alpha) * self.C)
            self.d.copy_(self.em_alpha * old_d + (1.0 - self.em_alpha) * self.d)


class SLDSNLBEvaluationAdapter(EvaluationAdapter):
    """NLB co-smoothing adapter for masked SLDS inference."""

    task = "nlb"

    def fit(
        self,
        model: BaseDynamicsModel,
        loader: Iterable | None,
        device: torch.device,
    ) -> None:
        del loader, device
        if not isinstance(model, SLDS):
            raise TypeError("SLDSNLBEvaluationAdapter requires an SLDS model.")

    def evaluate(
        self,
        model: BaseDynamicsModel,
        loader: Iterable,
        device: torch.device,
    ) -> EvaluationResult:
        if not isinstance(model, SLDS):
            raise TypeError("SLDSNLBEvaluationAdapter requires an SLDS model.")
        predictions: list[Tensor] = []
        targets: list[Tensor] = []
        with torch.no_grad():
            for batch in loader:
                batch = move_batch_to_device(batch, device)
                if not isinstance(batch, dict) or "heldout_spikes" not in batch:
                    raise TypeError("SLDS NLB evaluation requires heldout_spikes batches.")
                heldin = observations_from_batch(batch).float()
                target = batch["heldout_spikes"].float()
                prediction = model.predict_nlb_heldout(heldin, target.shape[-1])
                predictions.append(prediction.detach().cpu())
                targets.append(target.detach().cpu())

        pred_dict = {"rates": torch.cat(predictions, dim=0)}
        target_dict = {"spikes": torch.cat(targets, dim=0)}
        metrics = compute_available_metrics(pred_dict, target_dict)
        return EvaluationResult(
            metrics=metrics,
            predictions={key: value.numpy() for key, value in pred_dict.items()},
            targets={key: value.numpy() for key, value in target_dict.items()},
        )


def _symmetrize(matrix: Tensor) -> Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def _dtype_jitter(tensor: Tensor, jitter: float) -> float:
    if tensor.dtype in (torch.float16, torch.bfloat16, torch.float32):
        return max(jitter, 1e-5)
    return jitter


def _safe_solve(matrix: Tensor, rhs: Tensor, jitter: float = 1e-7) -> Tensor:
    jitter = _dtype_jitter(matrix, jitter)
    squeeze = rhs.ndim == matrix.ndim - 1
    if squeeze:
        rhs = rhs.unsqueeze(-1)
    eye = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
    matrix = _symmetrize(matrix)
    for scale in (0.0, jitter, 10.0 * jitter, 100.0 * jitter, 1000.0 * jitter):
        try:
            adjusted = matrix + scale * eye
            result = torch.linalg.solve(adjusted, rhs)
            return result.squeeze(-1) if squeeze else result
        except RuntimeError:
            continue
    result = torch.linalg.pinv(matrix + 1000.0 * jitter * eye) @ rhs
    return result.squeeze(-1) if squeeze else result


def _safe_inverse(matrix: Tensor, jitter: float = 1e-7) -> Tensor:
    eye = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
    rhs = eye.expand(*matrix.shape[:-2], matrix.shape[-1], matrix.shape[-1])
    return _safe_solve(matrix, rhs, jitter=jitter)


def _safe_cholesky(covariance: Tensor, jitter: float) -> Tensor:
    jitter = _dtype_jitter(covariance, jitter)
    eye = torch.eye(
        covariance.shape[-1],
        device=covariance.device,
        dtype=covariance.dtype,
    )
    covariance = _symmetrize(covariance)
    for scale in (0.0, jitter, 10.0 * jitter, 100.0 * jitter, 1000.0 * jitter):
        try:
            return torch.linalg.cholesky(covariance + scale * eye)
        except RuntimeError:
            continue
    eigvals, eigvecs = torch.linalg.eigh(covariance)
    clipped = eigvals.clamp_min(max(jitter, 1e-7))
    repaired = eigvecs @ torch.diag_embed(clipped) @ eigvecs.transpose(-1, -2)
    return torch.linalg.cholesky(_symmetrize(repaired) + jitter * eye)


def _precision_and_logdet(covariance: Tensor) -> tuple[Tensor, Tensor]:
    covariance = _symmetrize(covariance)
    chol = _safe_cholesky(covariance, jitter=1e-7)
    precision = torch.cholesky_inverse(chol)
    logdet = 2.0 * torch.log(
        torch.diagonal(chol, dim1=-2, dim2=-1).clamp_min(1e-12)
    ).sum(dim=-1)
    return _symmetrize(precision), logdet


def _solve_symm_block_tridiag(diag: Tensor, lower: Tensor, rhs: Tensor) -> Tensor:
    batch, time, dim = rhs.shape
    if time == 1:
        return _safe_solve(diag[:, 0], rhs[:, 0])[:, None]

    schur_diags: list[Tensor] = [_symmetrize(diag[:, 0])]
    schur_rhs: list[Tensor] = [rhs[:, 0]]
    for step in range(1, time):
        lower_block = lower[:, step - 1]
        solve_prev_lower_t = _safe_solve(
            schur_diags[-1],
            lower_block.transpose(-1, -2),
        )
        solve_prev_rhs = _safe_solve(schur_diags[-1], schur_rhs[-1])
        current_diag = diag[:, step] - lower_block @ solve_prev_lower_t
        current_rhs = rhs[:, step] - torch.einsum(
            "bij,bj->bi",
            lower_block,
            solve_prev_rhs,
        )
        schur_diags.append(_symmetrize(current_diag))
        schur_rhs.append(current_rhs)

    solution = torch.empty_like(rhs)
    solution[:, -1] = _safe_solve(schur_diags[-1], schur_rhs[-1])
    for step in range(time - 2, -1, -1):
        lower_block = lower[:, step]
        adjusted_rhs = schur_rhs[step] - torch.einsum(
            "bji,bj->bi",
            lower_block,
            solution[:, step + 1],
        )
        solution[:, step] = _safe_solve(schur_diags[step], adjusted_rhs)
    return solution


def _kalman_info_smoother(
    J_ini: Tensor,
    h_ini: Tensor,
    J_dyn_11: Tensor,
    J_dyn_21: Tensor,
    J_dyn_22: Tensor,
    h_dyn_1: Tensor,
    h_dyn_2: Tensor,
    J_obs: Tensor,
    h_obs: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    batch, time, dim = h_obs.shape
    filtered_Js = torch.zeros(batch, time, dim, dim, device=h_obs.device, dtype=h_obs.dtype)
    filtered_hs = torch.zeros(batch, time, dim, device=h_obs.device, dtype=h_obs.dtype)
    predicted_Js = torch.zeros_like(filtered_Js)
    predicted_hs = torch.zeros_like(filtered_hs)

    predicted_Js[:, 0] = J_ini
    predicted_hs[:, 0] = h_ini
    for step in range(time - 1):
        filtered_Js[:, step] = _symmetrize(predicted_Js[:, step] + J_obs[:, step])
        filtered_hs[:, step] = predicted_hs[:, step] + h_obs[:, step]
        tmp_J = _symmetrize(filtered_Js[:, step] + J_dyn_11[:, step])
        tmp_h = filtered_hs[:, step] + h_dyn_1[:, step]
        predicted_Js[:, step + 1] = _symmetrize(
            J_dyn_22[:, step]
            - J_dyn_21[:, step]
            @ _safe_solve(tmp_J, J_dyn_21[:, step].transpose(-1, -2))
        )
        predicted_hs[:, step + 1] = h_dyn_2[:, step] - torch.einsum(
            "bij,bj->bi",
            J_dyn_21[:, step],
            _safe_solve(tmp_J, tmp_h),
        )

    filtered_Js[:, -1] = _symmetrize(predicted_Js[:, -1] + J_obs[:, -1])
    filtered_hs[:, -1] = predicted_hs[:, -1] + h_obs[:, -1]

    smoothed_Js = torch.zeros_like(filtered_Js)
    smoothed_hs = torch.zeros_like(filtered_hs)
    smoothed_means = torch.zeros_like(filtered_hs)
    smoothed_covariances = torch.zeros_like(filtered_Js)
    cross_moments = torch.zeros(
        batch,
        max(time - 1, 0),
        dim,
        dim,
        device=h_obs.device,
        dtype=h_obs.dtype,
    )

    smoothed_Js[:, -1] = filtered_Js[:, -1]
    smoothed_hs[:, -1] = filtered_hs[:, -1]
    smoothed_covariances[:, -1] = _safe_inverse(smoothed_Js[:, -1])
    smoothed_means[:, -1] = torch.einsum(
        "bij,bj->bi",
        smoothed_covariances[:, -1],
        smoothed_hs[:, -1],
    )

    for step in range(time - 2, -1, -1):
        J_inner = _symmetrize(
            smoothed_Js[:, step + 1] - predicted_Js[:, step + 1] + J_dyn_22[:, step]
        )
        h_inner = smoothed_hs[:, step + 1] - predicted_hs[:, step + 1] + h_dyn_2[:, step]
        smoothed_Js[:, step] = _symmetrize(
            filtered_Js[:, step]
            + J_dyn_11[:, step]
            - J_dyn_21[:, step].transpose(-1, -2)
            @ _safe_solve(J_inner, J_dyn_21[:, step])
        )
        smoothed_hs[:, step] = filtered_hs[:, step] + h_dyn_1[:, step] - torch.einsum(
            "bji,bj->bi",
            J_dyn_21[:, step],
            _safe_solve(J_inner, h_inner),
        )
        smoothed_covariances[:, step] = _safe_inverse(smoothed_Js[:, step])
        smoothed_means[:, step] = torch.einsum(
            "bij,bj->bi",
            smoothed_covariances[:, step],
            smoothed_hs[:, step],
        )
        pair_precision = _symmetrize(filtered_Js[:, step] + J_dyn_11[:, step])
        cross_moments[:, step] = -_safe_solve(
            pair_precision,
            J_dyn_21[:, step].transpose(-1, -2) @ smoothed_covariances[:, step + 1],
        )
        cross_moments[:, step] = cross_moments[:, step] + torch.einsum(
            "bi,bj->bij",
            smoothed_means[:, step],
            smoothed_means[:, step + 1],
        )

    return smoothed_means, _symmetrize(smoothed_covariances), cross_moments


def _hmm_expected_states(
    log_pi0: Tensor,
    log_transition: Tensor,
    log_likes: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    batch, time, states = log_likes.shape
    log_alpha = torch.empty_like(log_likes)
    log_alpha[:, 0] = log_pi0[None, :] + log_likes[:, 0]
    for step in range(1, time):
        log_alpha[:, step] = log_likes[:, step] + torch.logsumexp(
            log_alpha[:, step - 1, :, None] + log_transition[None, :, :],
            dim=1,
        )
    log_norm = torch.logsumexp(log_alpha[:, -1], dim=1)

    log_beta = torch.zeros_like(log_likes)
    for step in range(time - 2, -1, -1):
        log_beta[:, step] = torch.logsumexp(
            log_transition[None, :, :]
            + log_likes[:, step + 1, None, :]
            + log_beta[:, step + 1, None, :],
            dim=2,
        )

    gamma = torch.exp(log_alpha + log_beta - log_norm[:, None, None])
    if time <= 1:
        xi = torch.empty(batch, 0, states, states, device=log_likes.device, dtype=log_likes.dtype)
        return gamma, xi, log_norm
    log_xi = (
        log_alpha[:, :-1, :, None]
        + log_transition[None, None, :, :]
        + log_likes[:, 1:, None, :]
        + log_beta[:, 1:, None, :]
        - log_norm[:, None, None, None]
    )
    xi = torch.exp(log_xi)
    return gamma, xi, log_norm


def _inv_softplus(value: Tensor) -> Tensor:
    value = value.clamp_min(1e-6)
    return value + torch.log(-torch.expm1(-value))


def _poisson_nll(rates: Tensor, spikes: Tensor, eps: float) -> Tensor:
    rates = rates.clamp_min(eps)
    return rates - spikes * torch.log(rates)
