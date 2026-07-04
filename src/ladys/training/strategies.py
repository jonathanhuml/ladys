"""Optimization strategy abstraction.

The benchmark trainer owns epochs, logging, timing, and dataloaders. Strategies
own the parameter update procedure so gradient, variational, and EM-style
methods can share the same outer training/reporting contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import torch
from torch import Tensor

from ladys.models.base import BaseDynamicsModel, OptimizationConfig
from ladys.types import StepResult, move_batch_to_device, observations_from_batch


class OptimizationStrategy(ABC):
    name: str

    def setup(self, model: BaseDynamicsModel) -> None:
        """Initialize optimizer state."""

    def on_epoch_start(self, model: BaseDynamicsModel, epoch: int) -> None:
        """Hook before an epoch starts."""

    def on_epoch_end(self, model: BaseDynamicsModel, epoch: int) -> None:
        """Hook after an epoch ends."""

    def train_epoch(
        self,
        model: BaseDynamicsModel,
        loader: Iterable,
        epoch: int,
        device: torch.device | str,
    ) -> list[StepResult]:
        results = []
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            results.append(self.step(model, batch, epoch))
        return results

    @abstractmethod
    def step(
        self,
        model: BaseDynamicsModel,
        batch: Tensor | dict[str, Tensor],
        epoch: int,
    ) -> StepResult:
        """Run one training update."""

    def validation_step(
        self,
        model: BaseDynamicsModel,
        batch: Tensor | dict[str, Tensor],
        epoch: int,
    ) -> StepResult:
        with torch.no_grad():
            x = observations_from_batch(batch)
            output = model(x)
            loss = model.loss(batch, output, epoch=epoch)
            return StepResult.from_loss(loss, batch_size=int(x.shape[0]))


class GradientStrategy(OptimizationStrategy):
    name = "gradient"

    def __init__(
        self,
        optimizer: str = "AdamW",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        gradient_clip: float | None = None,
    ) -> None:
        self.optimizer_name = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.optimizer: torch.optim.Optimizer | None = None

    def setup(self, model: BaseDynamicsModel) -> None:
        optimizer_cls = getattr(torch.optim, self.optimizer_name)
        self.optimizer = optimizer_cls(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def step(
        self,
        model: BaseDynamicsModel,
        batch: Tensor | dict[str, Tensor],
        epoch: int,
    ) -> StepResult:
        if self.optimizer is None:
            raise RuntimeError("GradientStrategy.setup() must be called before training.")

        model.train()
        x = observations_from_batch(batch)
        output = model(x)
        loss = model.loss(batch, output, epoch=epoch)

        self.optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        if self.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
        self.optimizer.step()

        return StepResult.from_loss(loss, batch_size=int(x.shape[0]))


class FullBatchGradientStrategy(OptimizationStrategy):
    """Gradient strategy for models with dataset-shaped variational parameters."""

    name = "full_batch_gradient"

    def __init__(
        self,
        optimizer: str = "AdamW",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        gradient_clip: float | None = None,
    ) -> None:
        self.optimizer_name = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.optimizer: torch.optim.Optimizer | None = None

    def setup(self, model: BaseDynamicsModel) -> None:
        self.optimizer = None

    def train_epoch(
        self,
        model: BaseDynamicsModel,
        loader: Iterable,
        epoch: int,
        device: torch.device | str,
    ) -> list[StepResult]:
        batches = [move_batch_to_device(batch, device) for batch in loader]
        if not batches:
            return []

        batch = _concat_batches(batches)
        return [self.step(model, batch, epoch)]

    def step(
        self,
        model: BaseDynamicsModel,
        batch: Tensor | dict[str, Tensor],
        epoch: int,
    ) -> StepResult:
        model.train()
        x = observations_from_batch(batch)
        output = model(x)
        loss = model.loss(batch, output, epoch=epoch)

        if self.optimizer is None:
            optimizer_cls = getattr(torch.optim, self.optimizer_name)
            params = [param for param in model.parameters() if param.requires_grad]
            if not params:
                raise RuntimeError(
                    f"{type(model).__name__} has no trainable parameters after forward()."
                )
            self.optimizer = optimizer_cls(
                params,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

        self.optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        if self.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
        self.optimizer.step()

        return StepResult.from_loss(loss, batch_size=int(x.shape[0]))


class EMStrategy(OptimizationStrategy):
    """Full-dataset EM strategy.

    One benchmark epoch corresponds to one call to `model.fit_em_epoch(x)`.
    """

    name = "em"

    def setup(self, model: BaseDynamicsModel) -> None:
        if not hasattr(model, "fit_em_epoch"):
            raise TypeError(f"{type(model).__name__} does not implement fit_em_epoch().")

    def train_epoch(
        self,
        model: BaseDynamicsModel,
        loader: Iterable,
        epoch: int,
        device: torch.device | str,
    ) -> list[StepResult]:
        batches = []
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            batches.append(observations_from_batch(batch))
        x = torch.cat(batches, dim=0)
        loss = model.fit_em_epoch(x, epoch=epoch)
        return [StepResult.from_loss(loss, batch_size=int(x.shape[0]))]

    def step(
        self,
        model: BaseDynamicsModel,
        batch: Tensor | dict[str, Tensor],
        epoch: int,
    ) -> StepResult:
        x = observations_from_batch(batch)
        loss = model.fit_em_epoch(x, epoch=epoch)
        return StepResult.from_loss(loss, batch_size=int(x.shape[0]))


def build_strategy(config: OptimizationConfig) -> OptimizationStrategy:
    kwargs = config.kwargs()
    if config.name == "gradient":
        return GradientStrategy(**kwargs)
    if config.name == "full_batch_gradient":
        return FullBatchGradientStrategy(**kwargs)
    if config.name == "em":
        return EMStrategy(**kwargs)
    raise KeyError(f"Unknown optimization strategy '{config.name}'.")


def _concat_batches(batches: list[Tensor | dict[str, Tensor]]) -> Tensor | dict[str, Tensor]:
    first = batches[0]
    if isinstance(first, Tensor):
        return torch.cat([batch for batch in batches if isinstance(batch, Tensor)], dim=0)
    if isinstance(first, dict):
        out: dict[str, Tensor] = {}
        for key, value in first.items():
            if isinstance(value, Tensor):
                out[key] = torch.cat([batch[key] for batch in batches], dim=0)
        return out
    raise TypeError(f"Unsupported batch type {type(first).__name__}.")
