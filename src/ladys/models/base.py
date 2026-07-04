"""Base model and config registry for latent dynamics methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from ladys.types import LossOutput, ModelOutput
from ladys.utils.yaml import load_yaml


class OptimizationConfig(BaseModel):
    """Config block that selects an optimization strategy."""

    model_config = ConfigDict(extra="allow")

    name: str = "gradient"

    def kwargs(self) -> dict[str, Any]:
        data = self.model_dump()
        data.pop("name", None)
        return data


class BaseModelConfig(BaseModel, ABC):
    """Pydantic model config that builds a PyTorch module."""

    model_config = ConfigDict(extra="forbid")

    registry: ClassVar[dict[str, type["BaseModelConfig"]]] = {}

    name: str
    objective: str = "negative_log_likelihood"
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)

    @abstractmethod
    def build(self, n_neurons: int, n_time: int) -> "BaseDynamicsModel":
        """Construct the model for runtime data dimensions."""

    @classmethod
    def register(cls, config_cls: type["BaseModelConfig"]) -> type["BaseModelConfig"]:
        name = config_cls.model_fields["name"].default
        if not isinstance(name, str):
            raise ValueError(f"{config_cls.__name__} must define a string default name.")
        cls.registry[name] = config_cls
        return config_cls

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseModelConfig":
        name = data.get("name")
        if name not in cls.registry:
            known = ", ".join(sorted(cls.registry)) or "<none>"
            raise KeyError(f"Unknown model config '{name}'. Registered models: {known}.")
        return cls.registry[name].model_validate(data)


class BaseDynamicsModel(nn.Module, ABC):
    """Base class for models taking `(batch, time, neurons)` tensors."""

    objective: str

    @abstractmethod
    def forward(self, x: Tensor) -> ModelOutput:
        """Run model inference on observations."""

    @abstractmethod
    def loss(
        self,
        batch: Tensor | dict[str, Tensor],
        output: ModelOutput,
        epoch: int = 0,
    ) -> LossOutput:
        """Compute the model objective for a batch and forward output."""

    def predict_rates(self, x: Tensor) -> Tensor:
        output = self.forward(x)
        if output.rates is not None:
            return output.rates
        if output.reconstruction is not None:
            return output.reconstruction
        raise RuntimeError(f"{type(self).__name__} did not return rates or reconstruction.")

    def evaluation_adapter(self, task: str) -> Any | None:
        """Return a model-specific evaluation adapter for a task, if needed."""

        del task
        return None

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            try:
                return next(self.buffers()).device
            except StopIteration:
                return torch.device("cpu")


def load_model_config(path: str) -> BaseModelConfig:
    """Load a model config from YAML."""

    from ladys import models as _models  # noqa: F401

    data = load_yaml(path)
    model_data = data["model"] if "model" in data else data
    return BaseModelConfig.from_dict(model_data)
