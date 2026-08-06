"""Bayesian GPFA adapter backed by mgplvm-pytorch."""

from __future__ import annotations

import importlib
import math
import numpy as np
from typing import Any, Iterable, Literal, Optional

import torch
from pydantic import Field, model_validator
from torch import Tensor

from ladys.metrics import EvaluationResult, NLBCoSmoothingAdapter
from ladys.models.base import BaseDynamicsModel, BaseModelConfig, OptimizationConfig
from ladys.types import LossOutput, ModelOutput, move_batch_to_device, observations_from_batch


@BaseModelConfig.register
class BGPFAConfig(BaseModelConfig):
    """Config for variational Bayesian GPFA."""

    name: Literal["bgpfa"] = "bgpfa"
    objective: str = "negative_elbo"
    latent_dim: int = 3
    binsize: float = 25.0
    ell0: Optional[float] = None
    rho: float = 2.0
    n_mc_train: int = 3
    n_mc_eval: int = 5
    kl_burnin_epochs: int = 1
    latent_scale_init: float = 1.0
    likelihood: Literal["gaussian", "poisson"] = "gaussian"
    learn_scale: bool = False
    ard: bool = True
    dtype: Literal["float64", "float32"] = "float64"
    latent_init: Literal["gp_prior", "fa"] = "gp_prior"
    observation_init: Literal["mgplvm", "fa"] = "mgplvm"
    nlb_feature_source: Literal["latents", "rates", "reconstruction"] = "latents"
    nlb_decoder: Literal["ridge", "poisson"] = "poisson"
    nlb_ridge_alpha: float = 1.0e-2
    nlb_poisson_max_iter: int = 80
    nlb_latent_infer_steps: int = 300
    nlb_latent_infer_n_mc: int = 20
    nlb_latent_infer_lr: float = 1e-1
    nlb_latent_infer_burnin: int = 1
    optimization: OptimizationConfig = Field(
        default_factory=lambda: OptimizationConfig(
            name="mgplvm_full_batch_gradient",
            optimizer="Adam",
            lr=1e-1,
            steps_per_epoch=1,
            burnin=150,
            n_mc=3,
            weight_decay=0.0,
        )
    )

    @model_validator(mode="after")
    def reject_em_optimization(self) -> "BGPFAConfig":
        if self.optimization.name == "em":
            raise ValueError(
                "BGPFA uses a differentiable variational ELBO and does not support "
                "optimization.name='em'."
            )
        return self

    def build(self, n_neurons: int, n_time: int) -> "BGPFA":
        return BGPFA(
            n_neurons=n_neurons,
            n_time=n_time,
            latent_dim=self.latent_dim,
            binsize=self.binsize,
            ell0=self.ell0,
            rho=self.rho,
            n_mc_train=self.n_mc_train,
            n_mc_eval=self.n_mc_eval,
            kl_burnin_epochs=self.kl_burnin_epochs,
            latent_scale_init=self.latent_scale_init,
            likelihood=self.likelihood,
            learn_scale=self.learn_scale,
            ard=self.ard,
            dtype=self.dtype,
            latent_init=self.latent_init,
            observation_init=self.observation_init,
            nlb_feature_source=self.nlb_feature_source,
            nlb_decoder=self.nlb_decoder,
            nlb_ridge_alpha=self.nlb_ridge_alpha,
            nlb_poisson_max_iter=self.nlb_poisson_max_iter,
            nlb_latent_infer_steps=self.nlb_latent_infer_steps,
            nlb_latent_infer_n_mc=self.nlb_latent_infer_n_mc,
            nlb_latent_infer_lr=self.nlb_latent_infer_lr,
            nlb_latent_infer_burnin=self.nlb_latent_infer_burnin,
            objective=self.objective,
        )


class BGPFA(BaseDynamicsModel):
    """Variational Bayesian GPFA with ARD and differentiable ELBO training.

    ## When to use

    Use BGPFA when you want the Bayesian GPFA objective from
    `tachukao/mgplvm-pytorch` inside the LaDyS trainer contract. Unlike the
    classical GPFA EM baseline, this adapter optimizes a Monte Carlo variational
    negative ELBO with standard PyTorch backpropagation.

    ## Assumptions

    Observations are passed as `(batch, time, neurons)` tensors and internally
    transposed to mgplvm's `(trials, neurons, time)` convention. The latent
    posterior has per-trial variational parameters, so the default optimization
    strategy is `mgplvm_full_batch_gradient`. One LaDyS epoch can run multiple
    mgplvm optimizer updates via `optimization.steps_per_epoch`; this is useful
    when matching reference bGPFA scripts that report fixed optimizer-step
    budgets.

    ## Outputs

    `forward` returns predictive rates/reconstructions, variational latent
    means, and ELBO terms in `extras`. The core mgplvm implementation is
    vendored in `src/mgplvm`; this class only adapts it to the LaDyS model,
    loss, and trainer contracts.
    """

    def __init__(
        self,
        n_neurons: int,
        n_time: int,
        latent_dim: int = 3,
        binsize: float = 25.0,
        ell0: float | None = None,
        rho: float = 2.0,
        n_mc_train: int = 3,
        n_mc_eval: int = 5,
        kl_burnin_epochs: int = 1,
        latent_scale_init: float = 1.0,
        likelihood: Literal["gaussian", "poisson"] = "gaussian",
        learn_scale: bool = False,
        ard: bool = True,
        dtype: Literal["float64", "float32"] = "float64",
        latent_init: str = "gp_prior",
        observation_init: str = "mgplvm",
        nlb_feature_source: str = "latents",
        nlb_decoder: str = "poisson",
        nlb_ridge_alpha: float = 1.0e-2,
        nlb_poisson_max_iter: int = 80,
        nlb_latent_infer_steps: int = 300,
        nlb_latent_infer_n_mc: int = 20,
        nlb_latent_infer_lr: float = 1e-1,
        nlb_latent_infer_burnin: int = 1,
        objective: str = "negative_elbo",
    ) -> None:
        super().__init__()
        self.n_neurons = int(n_neurons)
        self.n_time = int(n_time)
        self.latent_dim = int(latent_dim)
        self.binsize = float(binsize)
        self.ell0 = float(ell0) if ell0 is not None else 200.0 / self.binsize
        self.rho = float(rho)
        self.n_mc_train = int(n_mc_train)
        self.n_mc_eval = int(n_mc_eval)
        self.kl_burnin_epochs = int(kl_burnin_epochs)
        self.latent_scale_init = float(latent_scale_init)
        self.likelihood = likelihood
        self.learn_scale = bool(learn_scale)
        self.ard = bool(ard)
        self.dtype = dtype
        self.latent_init = str(latent_init)
        self.observation_init = str(observation_init)
        self.nlb_feature_source = str(nlb_feature_source)
        self.nlb_decoder = str(nlb_decoder)
        self.nlb_ridge_alpha = float(nlb_ridge_alpha)
        self.nlb_poisson_max_iter = int(nlb_poisson_max_iter)
        self.nlb_latent_infer_steps = int(nlb_latent_infer_steps)
        self.nlb_latent_infer_n_mc = int(nlb_latent_infer_n_mc)
        self.nlb_latent_infer_lr = float(nlb_latent_infer_lr)
        self.nlb_latent_infer_burnin = int(nlb_latent_infer_burnin)
        self.objective = objective

        fit_ts = torch.arange(self.n_time, dtype=self._torch_dtype())[None, None, :]
        self.register_buffer("fit_ts", fit_ts)
        self._train_n_trials: int | None = None
        self._train_mod: torch.nn.Module | None = None
        self._eval_cache: dict[int, torch.nn.Module] = {}

    def forward(self, x: Tensor) -> ModelOutput:
        x = self._coerce_observations(x)
        self._validate_input(x)
        mod = self._model_for(x)
        y = self._to_mgplvm_observations(x)

        n_mc = self.n_mc_train if self.training else max(1, self.n_mc_eval)
        svgp_elbo, latent_kl = mod(
            y,
            n_mc,
            analytic_kl="GP" in mod.lat_dist.name,
        )
        latent_kl = latent_kl.mean() if latent_kl.ndim > 0 else latent_kl
        reconstruction = self._predict_from_latent_mean(mod)
        latents = mod.lat_dist.lat_mu.to(x.device, x.dtype)

        return ModelOutput(
            rates=reconstruction.clamp_min(0.0),
            latents=latents,
            reconstruction=reconstruction,
            extras={
                "svgp_elbo": svgp_elbo,
                "latent_kl": latent_kl,
                "mgplvm_model": mod,
            },
        )

    def loss(
        self,
        batch: Tensor | dict[str, Tensor],
        output: ModelOutput,
        epoch: int = 0,
    ) -> LossOutput:
        x = observations_from_batch(batch)
        svgp_elbo = output.extras["svgp_elbo"]
        latent_kl = output.extras["latent_kl"]
        kl_weight = self._kl_weight(epoch)
        total = (-svgp_elbo + kl_weight * latent_kl) / x.numel()
        return LossOutput(
            total=total,
            named_terms={
                "negative_elbo": total,
                "svgp_elbo": svgp_elbo.detach(),
                "latent_kl": latent_kl.detach(),
                "kl_weight": kl_weight,
            },
            objective=self.objective,
        )

    def evaluation_adapter(self, task: str):
        if task == "nlb":
            return _BGPFANLBCoSmoothingAdapter(
                feature_source=self.nlb_feature_source,
                decoder=self.nlb_decoder,
                ridge_alpha=self.nlb_ridge_alpha,
                poisson_max_iter=self.nlb_poisson_max_iter,
                latent_infer_steps=self.nlb_latent_infer_steps,
                latent_infer_n_mc=self.nlb_latent_infer_n_mc,
                latent_infer_lr=self.nlb_latent_infer_lr,
                latent_infer_burnin=self.nlb_latent_infer_burnin,
            )
        return None

    def _model_for(self, x: Tensor) -> torch.nn.Module:
        x = self._coerce_observations(x)
        if self.training:
            if self._train_mod is None:
                self._train_n_trials = int(x.shape[0])
                self._train_mod = self._build_mgplvm_model(x)
            elif int(x.shape[0]) != self._train_n_trials:
                raise ValueError(
                    "BGPFA training requires a stable full-batch trial count. "
                    f"Expected {self._train_n_trials}, got {int(x.shape[0])}."
                )
            return self._train_mod

        n_trials = int(x.shape[0])
        if self._train_mod is not None and n_trials == self._train_n_trials:
            return self._train_mod
        if n_trials not in self._eval_cache:
            mod = self._build_mgplvm_model(x)
            if self._train_mod is not None:
                self._copy_observation_state(target=mod, source=self._train_mod)
            self._eval_cache[n_trials] = mod
        return self._eval_cache[n_trials]

    def mgplvm_training_model(self, x: Tensor) -> torch.nn.Module:
        x = self._coerce_observations(x)
        self._validate_input(x)
        if self._train_mod is None:
            self._train_n_trials = int(x.shape[0])
            self._train_mod = self._build_mgplvm_model(x)
        elif int(x.shape[0]) != self._train_n_trials:
            raise ValueError(
                "BGPFA training requires a stable full-batch trial count. "
                f"Expected {self._train_n_trials}, got {int(x.shape[0])}."
            )
        return self._train_mod

    def mgplvm_observations(self, x: Tensor) -> Tensor:
        return self._to_mgplvm_observations(self._coerce_observations(x))

    def infer_latents(
        self,
        x: Tensor,
        max_steps: int = 300,
        n_mc: int = 20,
        lrate: float = 1e-1,
        burnin: int = 1,
    ) -> torch.nn.Module:
        if self._train_mod is None:
            raise RuntimeError("BGPFA must be trained before held-out latent inference.")

        mgp = _require_mgplvm()
        x = self._coerce_observations(x)
        self._validate_input(x)
        mod = self._build_mgplvm_model(x)
        self._copy_observation_state(target=mod, source=self._train_mod)
        for param in mod.parameters():
            param.requires_grad = False
        for param in mod.lat_dist.parameters():
            if param.is_floating_point() or param.is_complex():
                param.requires_grad = True

        params = mgp.crossval.training_params(
            max_steps=max_steps,
            n_mc=n_mc,
            lrate=lrate,
            print_every=np.nan,
            burnin=burnin,
            mask_Ts=lambda value: value * 1,
        )
        mgp.crossval.train_model(mod, self._to_mgplvm_observations(x), params)
        self._eval_cache[int(x.shape[0])] = mod
        return mod

    def _build_mgplvm_model(self, x: Tensor) -> torch.nn.Module:
        mgp = _require_mgplvm()
        y_np = self._to_mgplvm_observations(x).detach().cpu().numpy()
        n_trials = int(x.shape[0])
        manif = mgp.manifolds.Euclid(self.n_time, self.latent_dim)
        lat_dist = mgp.rdist.GP_circ(
            manif,
            self.n_time,
            n_trials,
            self.fit_ts.to(x.device, x.dtype),
            _scale=self.latent_scale_init,
            ell=self.ell0,
        )
        lprior = mgp.lpriors.Null(manif)
        likelihood = self._build_likelihood(mgp, x, y_np)
        mod = mgp.models.Lvgplvm(
            self.n_neurons,
            self.n_time,
            self.latent_dim,
            n_trials,
            lat_dist,
            lprior,
            likelihood,
            Y=y_np,
            learn_scale=self.learn_scale,
            ard=self.ard,
            rel_scale=self.rho,
        )
        mod = mod.to(device=x.device, dtype=x.dtype)
        if self.latent_init not in {"gp_prior", "fa"}:
            raise ValueError(f"Unsupported bGPFA latent_init '{self.latent_init}'.")
        if self.observation_init not in {"mgplvm", "fa"}:
            raise ValueError(f"Unsupported bGPFA observation_init '{self.observation_init}'.")
        if self.latent_init == "fa" or self.observation_init == "fa":
            self._initialize_from_fa(mod, y_np, x.device, x.dtype)
        return mod

    def _initialize_from_fa(
        self,
        mod: torch.nn.Module,
        y_np: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        from sklearn import decomposition

        n_trials, n_neurons, n_time = y_np.shape
        flat = np.asarray(y_np, dtype=np.float64).transpose(0, 2, 1).reshape(
            n_trials * n_time,
            n_neurons,
        )
        components = min(self.latent_dim, n_neurons, flat.shape[0])
        if components < 1:
            return
        fa = decomposition.FactorAnalysis(n_components=components)
        scores = fa.fit_transform(flat)
        loadings = np.asarray(fa.components_.T, dtype=np.float64)
        scale = np.std(scores, axis=0, keepdims=True)
        score_scale = 0.5 / np.maximum(scale, 1.0e-6)
        scores = scores * score_scale
        loadings = loadings / score_scale
        if components < self.latent_dim:
            padded = np.zeros((scores.shape[0], self.latent_dim), dtype=np.float64)
            padded[:, :components] = scores
            scores = padded
            padded_loadings = np.zeros((n_neurons, self.latent_dim), dtype=np.float64)
            padded_loadings[:, :components] = loadings
            loadings = padded_loadings
        latents = scores.reshape(n_trials, n_time, self.latent_dim)
        nu = torch.as_tensor(
            latents.transpose(0, 2, 1),
            dtype=dtype,
            device=device,
        )
        with torch.no_grad():
            if self.latent_init == "fa":
                mod.lat_dist._nu.copy_(nu)
            if self.observation_init == "fa" and hasattr(mod.obs, "_q_mu"):
                obs = mod.obs
                effective_scale = (
                    obs.neuron_scale.detach()
                    * obs.scale.detach().reshape(1, 1)
                    * obs.dim_scale.detach().reshape(1, -1)
                )
                q_mu = torch.as_tensor(loadings, dtype=dtype, device=device) / effective_scale.clamp_min(
                    1.0e-8
                )
                obs._q_mu.copy_(q_mu.unsqueeze(0))

    def _build_likelihood(self, mgp: Any, x: Tensor, y_np: Any) -> torch.nn.Module:
        if self.likelihood == "gaussian":
            sigma = 0.1 * torch.ones(self.n_neurons, device=x.device, dtype=x.dtype)
            return mgp.likelihoods.Gaussian(
                self.n_neurons,
                Y=y_np,
                sigma=sigma,
            )
        if self.likelihood == "poisson":
            return mgp.likelihoods.Poisson(
                self.n_neurons,
                binsize=self.binsize,
            )
        raise ValueError(f"Unsupported BGPFA likelihood '{self.likelihood}'.")

    @staticmethod
    def _copy_observation_state(
        target: torch.nn.Module,
        source: torch.nn.Module,
    ) -> None:
        target.obs.load_state_dict(source.obs.state_dict())

    def _predict_from_latent_mean(self, mod: torch.nn.Module) -> Tensor:
        with torch.no_grad():
            query = mod.lat_dist.lat_mu.detach().transpose(-1, -2)
            samples = mod.svgp.sample(
                query,
                n_mc=max(1, self.n_mc_eval),
                noise=False,
            )
            return samples.mean(dim=0).permute(0, 2, 1)

    def _kl_weight(self, epoch: int) -> float:
        if self.kl_burnin_epochs <= 0:
            return 1.0
        return float(1.0 - math.exp(-float(epoch + 1) / (3.0 * self.kl_burnin_epochs)))

    def _to_mgplvm_observations(self, x: Tensor) -> Tensor:
        return x.permute(0, 2, 1).contiguous()

    def _coerce_observations(self, x: Tensor) -> Tensor:
        return x.to(dtype=self._torch_dtype())

    def _torch_dtype(self) -> torch.dtype:
        if self.dtype == "float64":
            return torch.float64
        if self.dtype == "float32":
            return torch.float32
        raise ValueError(f"Unsupported BGPFA dtype '{self.dtype}'.")

    def _validate_input(self, x: Tensor) -> None:
        if x.ndim != 3:
            raise ValueError("BGPFA expects observations shaped (batch, time, neurons).")
        if int(x.shape[1]) != self.n_time:
            raise ValueError(f"Expected {self.n_time} time bins, got {int(x.shape[1])}.")
        if int(x.shape[2]) != self.n_neurons:
            raise ValueError(f"Expected {self.n_neurons} neurons, got {int(x.shape[2])}.")


class _BGPFANLBCoSmoothingAdapter(NLBCoSmoothingAdapter):
    """NLB adapter that infers eval-trial bGPFA latents before decoding."""

    def __init__(
        self,
        *,
        feature_source: str,
        decoder: str,
        ridge_alpha: float,
        poisson_max_iter: int,
        latent_infer_steps: int,
        latent_infer_n_mc: int,
        latent_infer_lr: float,
        latent_infer_burnin: int,
    ) -> None:
        super().__init__(
            feature_source=feature_source,
            decoder=decoder,
            ridge_alpha=ridge_alpha,
            poisson_max_iter=poisson_max_iter,
        )
        self.latent_infer_steps = int(latent_infer_steps)
        self.latent_infer_n_mc = int(latent_infer_n_mc)
        self.latent_infer_lr = float(latent_infer_lr)
        self.latent_infer_burnin = int(latent_infer_burnin)

    def evaluate(
        self,
        model: BaseDynamicsModel,
        loader: Iterable,
        device: torch.device,
    ) -> EvaluationResult:
        if isinstance(model, BGPFA):
            self._infer_eval_latents(model, loader, device)
        return super().evaluate(model, loader, device)

    def _infer_eval_latents(
        self,
        model: "BGPFA",
        loader: Iterable,
        device: torch.device,
    ) -> None:
        model.to(device)
        model.eval()
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            x = observations_from_batch(batch)
            model.infer_latents(
                x,
                max_steps=self.latent_infer_steps,
                n_mc=self.latent_infer_n_mc,
                lrate=self.latent_infer_lr,
                burnin=self.latent_infer_burnin,
            )


def _require_mgplvm() -> Any:
    try:
        return importlib.import_module("mgplvm")
    except ImportError as exc:
        raise ImportError(
            "BGPFA requires the vendored `mgplvm` package and its runtime "
            "dependencies. Reinstall LaDyS after this change, and make sure "
            "`scikit-learn` is available in the active environment."
        ) from exc
