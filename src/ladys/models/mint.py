"""MINT adapter for LaDyS.

MINT is an inference-only library method rather than a gradient-trained
``forward``/``backward`` model. The registered config keeps it discoverable in
the LaDyS model registry, while the NLB-specific runner in ``ladys.mint_nlb``
handles the required trajectory-library construction and buffered test
evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import torch
from pydantic import Field
from torch import Tensor

from ladys.models.base import BaseDynamicsModel, BaseModelConfig, OptimizationConfig
from ladys.models.mint_core.config import get_config
from ladys.models.mint_core.core import MINT as CoreMINT
from ladys.types import LossOutput, ModelOutput


@BaseModelConfig.register
class MINTConfig(BaseModelConfig):
    """Config for the MINT NLB co-smoothing port."""

    name: Literal["mint"] = "mint"
    objective: str = "mint_likelihood_recursion"
    dataset: Literal["area2_bump", "mc_maze", "mc_rtt"] = "mc_maze"
    train_source: Literal["mat", "nwb"] = "nwb"
    train_split: Literal["auto", "train", "trainval"] = "trainval"
    nlb_neural_state_defaults: bool = True
    nwb_root: str = "data/real/nlb/dandi"
    mat_data_root: str = "data/mint"
    target_h5: Optional[str] = None
    eval_bin_size_ms: int = 5
    n_candidates: Optional[int] = None
    window_length: Optional[int] = None
    delta: Optional[int] = None
    sigma: Optional[int] = None
    min_rate: Optional[float] = None
    causal: Optional[bool] = None
    optimization: OptimizationConfig = Field(
        default_factory=lambda: OptimizationConfig(name="inference_only")
    )

    def build(self, n_neurons: int, n_time: int) -> "MINT":
        del n_neurons, n_time
        return MINT(self)


class MINT(BaseDynamicsModel):
    """Thin LaDyS module wrapper around the validated PyTorch MINT core."""

    def __init__(self, config: MINTConfig) -> None:
        super().__init__()
        self.config = config
        self.objective = config.objective
        self.settings, self.hyperparams = get_config(config.dataset, repo_root=Path("."))
        self.settings.data_path = Path(config.mat_data_root) / f"{config.dataset}.mat"
        self.core: CoreMINT | None = None
        self.register_buffer("_device_anchor", torch.empty(0))

    def make_core(self) -> CoreMINT:
        """Construct an unfitted core MINT model from the current config."""

        return CoreMINT(self.settings, self.hyperparams, self.device)

    def forward(self, x: Tensor) -> ModelOutput:
        if self.core is None:
            raise RuntimeError("MINT must be fit with trajectory-library data before inference.")
        if x.ndim != 3:
            raise ValueError("MINT expects batched spikes with shape (batch, time, neurons).")
        spikes = [trial.T.contiguous() for trial in x]
        rates, _ = self.core.predict(spikes)
        return ModelOutput(rates=torch.stack([item.T for item in rates], dim=0))

    def loss(
        self,
        batch: Tensor | dict[str, Tensor],
        output: ModelOutput,
        epoch: int = 0,
    ) -> LossOutput:
        del batch, output, epoch
        return LossOutput(
            total=torch.zeros((), dtype=torch.float32, device=self.device),
            objective=self.objective,
        )
