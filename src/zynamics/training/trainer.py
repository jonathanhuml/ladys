"""Minimal trainer that enforces a common benchmark lifecycle."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

import torch

from zynamics.models.base import BaseDynamicsModel
from zynamics.training.strategies import OptimizationStrategy
from zynamics.types import StepResult, move_batch_to_device


@dataclass
class TrainerConfig:
    epochs: int = 10
    device: str = "cpu"


@dataclass
class EpochReport:
    epoch: int
    train: StepResult
    valid: StepResult | None
    seconds: float


class Trainer:
    def __init__(self, config: TrainerConfig | None = None) -> None:
        self.config = config or TrainerConfig()
        self.history: list[EpochReport] = []

    def fit(
        self,
        model: BaseDynamicsModel,
        strategy: OptimizationStrategy,
        train_loader: Iterable,
        valid_loader: Iterable | None = None,
    ) -> list[EpochReport]:
        device = torch.device(self.config.device)
        model.to(device)
        strategy.setup(model)

        for epoch in range(self.config.epochs):
            start = time.perf_counter()
            strategy.on_epoch_start(model, epoch)
            train_results = strategy.train_epoch(model, train_loader, epoch, device)
            strategy.on_epoch_end(model, epoch)
            seconds = time.perf_counter() - start
            valid_result = self.validate(model, strategy, valid_loader, epoch, device)

            report = EpochReport(
                epoch=epoch,
                train=_aggregate_results(train_results),
                valid=valid_result,
                seconds=seconds,
            )
            self.history.append(report)

        return self.history

    def validate(
        self,
        model: BaseDynamicsModel,
        strategy: OptimizationStrategy,
        loader: Iterable | None,
        epoch: int,
        device: torch.device,
    ) -> StepResult | None:
        if loader is None:
            return None

        model.eval()
        results = []
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            results.append(strategy.validation_step(model, batch, epoch))
        return _aggregate_results(results)


def _aggregate_results(results: list[StepResult]) -> StepResult:
    if not results:
        return StepResult(loss=float("nan"), batch_size=0)

    total_batch = sum(max(result.batch_size, 1) for result in results)
    loss = sum(result.loss * max(result.batch_size, 1) for result in results) / total_batch
    metrics: dict[str, float] = {}
    keys = set().union(*(result.metrics.keys() for result in results))
    for key in keys:
        metrics[key] = (
            sum(result.metrics.get(key, 0.0) * max(result.batch_size, 1) for result in results)
            / total_batch
        )

    return StepResult(
        loss=loss,
        metrics=metrics,
        batch_size=total_batch,
        objective=results[0].objective,
    )
