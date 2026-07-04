"""NDT adapter for masked spike-count modeling."""

from __future__ import annotations

import math
from typing import Literal

import torch
from pydantic import Field, model_validator
from torch import Tensor, nn
import torch.nn.functional as F

from ladys.models.base import BaseDynamicsModel, BaseModelConfig, OptimizationConfig
from ladys.types import LossOutput, ModelOutput, observations_from_batch


UNMASKED_LABEL = -100.0


@BaseModelConfig.register
class NDTConfig(BaseModelConfig):
    """Config for the masked-count NeuralDataTransformer (NDT) adapter."""

    name: Literal["ndt"] = "ndt"
    objective: str = "masked_poisson_nll"
    context_forward: int = 4
    context_backward: int = 8
    context_wrap_initial: bool = False
    full_context: bool = False
    hidden_size: int = 128
    dropout: float = 0.1
    dropout_rates: float = 0.2
    dropout_embedding: float = 0.2
    num_heads: int = 2
    num_layers: int = 6
    activation: Literal["relu", "gelu"] = "relu"
    linear_embedder: bool = False
    embed_dim: int = 2
    learnable_position: bool = True
    max_spike_count: int = 20
    lograte: bool = True
    log_rate_min: float = -8.0
    log_rate_max: float = 8.0
    spike_log_init: bool = False
    fixup_init: bool = True
    pre_norm: bool = True
    scale_norm: bool = False
    decoder_layers: int = 1
    position_offset: bool = True
    mask_ratio: float = 0.25
    mask_mode: Literal["full", "timestep", "neuron", "timestep_only"] = "timestep"
    mask_token_ratio: float = 1.0
    mask_random_ratio: float = 0.5
    mask_max_span: int = 1
    mask_span_expand_prob: float = 0.0
    use_zero_mask: bool = True
    topk_loss_fraction: float = 1.0
    optimization: OptimizationConfig = Field(
        default_factory=lambda: OptimizationConfig(
            name="gradient",
            optimizer="Adam",
            lr=1e-3,
            weight_decay=0.0,
            gradient_clip=200.0,
        )
    )

    @model_validator(mode="after")
    def validate_dimensions(self) -> "NDTConfig":
        if self.embed_dim < 0:
            raise ValueError("embed_dim must be nonnegative.")
        if self.num_heads < 1:
            raise ValueError("num_heads must be positive.")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive.")
        if self.decoder_layers < 1:
            raise ValueError("decoder_layers must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if not 0.0 <= self.dropout_rates < 1.0:
            raise ValueError("dropout_rates must be in [0, 1).")
        if not 0.0 <= self.dropout_embedding < 1.0:
            raise ValueError("dropout_embedding must be in [0, 1).")
        if not 0.0 <= self.mask_ratio <= 1.0:
            raise ValueError("mask_ratio must be in [0, 1].")
        if not 0.0 <= self.mask_token_ratio <= 1.0:
            raise ValueError("mask_token_ratio must be in [0, 1].")
        if not 0.0 <= self.mask_random_ratio <= 1.0:
            raise ValueError("mask_random_ratio must be in [0, 1].")
        if self.mask_max_span < 1:
            raise ValueError("mask_max_span must be positive.")
        if not 0.0 <= self.mask_span_expand_prob <= 1.0:
            raise ValueError("mask_span_expand_prob must be in [0, 1].")
        if not 0.0 < self.topk_loss_fraction <= 1.0:
            raise ValueError("topk_loss_fraction must be in (0, 1].")
        return self

    def build(self, n_neurons: int, n_time: int) -> "NDT":
        return NDT(
            n_neurons=n_neurons,
            n_time=n_time,
            context_forward=self.context_forward,
            context_backward=self.context_backward,
            context_wrap_initial=self.context_wrap_initial,
            full_context=self.full_context,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            dropout_rates=self.dropout_rates,
            dropout_embedding=self.dropout_embedding,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            activation=self.activation,
            linear_embedder=self.linear_embedder,
            embed_dim=self.embed_dim,
            learnable_position=self.learnable_position,
            max_spike_count=self.max_spike_count,
            lograte=self.lograte,
            log_rate_min=self.log_rate_min,
            log_rate_max=self.log_rate_max,
            spike_log_init=self.spike_log_init,
            fixup_init=self.fixup_init,
            pre_norm=self.pre_norm,
            scale_norm=self.scale_norm,
            decoder_layers=self.decoder_layers,
            position_offset=self.position_offset,
            mask_ratio=self.mask_ratio,
            mask_mode=self.mask_mode,
            mask_token_ratio=self.mask_token_ratio,
            mask_random_ratio=self.mask_random_ratio,
            mask_max_span=self.mask_max_span,
            mask_span_expand_prob=self.mask_span_expand_prob,
            use_zero_mask=self.use_zero_mask,
            topk_loss_fraction=self.topk_loss_fraction,
            objective=self.objective,
        )


class NDT(BaseDynamicsModel):
    """Transformer encoder trained with a masked Poisson spike objective.

    ## When to use

    Use NeuralDataTransformer (NDT) as a self-supervised sequence baseline for
    binned spike counts. This adapter follows the Lorenz NDT configuration from
    `snel-repo/neural-data-transformers`: per-neuron spike-count embeddings,
    optional local temporal attention, learnable positions, pre-norm transformer
    layers, and a Poisson decoder trained on randomly masked observations.

    ## Assumptions

    NeuralDataTransformer (NDT) expects raw nonnegative spike counts.
    Dataset-level smoothing should be disabled for this model. The model
    returns natural-space rates for metrics; internally, the default decoder
    predicts log rates for stable Poisson loss.

    ## Outputs

    `forward` returns nonnegative rate predictions, transformer factor
    trajectories in `latents`, and masking diagnostics in `extras`. During
    training, `loss` uses only masked entries. During evaluation, it uses all
    entries so validation reports a full reconstruction objective.
    """

    def __init__(
        self,
        n_neurons: int,
        n_time: int,
        context_forward: int = 4,
        context_backward: int = 8,
        context_wrap_initial: bool = False,
        full_context: bool = False,
        hidden_size: int = 128,
        dropout: float = 0.1,
        dropout_rates: float = 0.2,
        dropout_embedding: float = 0.2,
        num_heads: int = 2,
        num_layers: int = 6,
        activation: str = "relu",
        linear_embedder: bool = False,
        embed_dim: int = 2,
        learnable_position: bool = True,
        max_spike_count: int = 20,
        lograte: bool = True,
        log_rate_min: float = -8.0,
        log_rate_max: float = 8.0,
        spike_log_init: bool = False,
        fixup_init: bool = True,
        pre_norm: bool = True,
        scale_norm: bool = False,
        decoder_layers: int = 1,
        position_offset: bool = True,
        mask_ratio: float = 0.25,
        mask_mode: str = "timestep",
        mask_token_ratio: float = 1.0,
        mask_random_ratio: float = 0.5,
        mask_max_span: int = 1,
        mask_span_expand_prob: float = 0.0,
        use_zero_mask: bool = True,
        topk_loss_fraction: float = 1.0,
        objective: str = "masked_poisson_nll",
    ) -> None:
        super().__init__()
        self.n_neurons = int(n_neurons)
        self.n_time = int(n_time)
        self.context_forward = int(context_forward)
        self.context_backward = int(context_backward)
        self.context_wrap_initial = bool(context_wrap_initial)
        self.full_context = bool(full_context)
        self.hidden_size = int(hidden_size)
        self.dropout = float(dropout)
        self.dropout_rates = float(dropout_rates)
        self.dropout_embedding = float(dropout_embedding)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.activation = str(activation)
        self.linear_embedder = bool(linear_embedder)
        self.embed_dim = int(embed_dim)
        self.learnable_position = bool(learnable_position)
        self.max_spike_count = int(max_spike_count)
        self.lograte = bool(lograte)
        self.log_rate_min = float(log_rate_min)
        self.log_rate_max = float(log_rate_max)
        self.spike_log_init = bool(spike_log_init)
        self.fixup_init = bool(fixup_init)
        self.pre_norm = bool(pre_norm)
        self.scale_norm = bool(scale_norm)
        self.decoder_layers = int(decoder_layers)
        self.position_offset = bool(position_offset)
        self.mask_ratio = float(mask_ratio)
        self.mask_mode = str(mask_mode)
        self.mask_token_ratio = float(mask_token_ratio)
        self.mask_random_ratio = float(mask_random_ratio)
        self.mask_max_span = int(mask_max_span)
        self.mask_span_expand_prob = float(mask_span_expand_prob)
        self.use_zero_mask = bool(use_zero_mask)
        self.topk_loss_fraction = float(topk_loss_fraction)
        self.objective = objective

        if self.n_neurons < 1:
            raise ValueError("n_neurons must be positive.")
        if self.n_time < 1:
            raise ValueError("n_time must be positive.")
        if self.embed_dim < 0:
            raise ValueError("embed_dim must be nonnegative.")
        if self.linear_embedder:
            self.model_dim = self.n_neurons if self.embed_dim == 0 else self.n_neurons * self.embed_dim
            self.embedder: nn.Module = nn.Linear(self.n_neurons, self.model_dim)
        elif self.embed_dim == 0:
            self.model_dim = self.n_neurons
            self.embedder = nn.Identity()
        else:
            self.model_dim = self.n_neurons * self.embed_dim
            self.embedder = nn.Embedding(self.max_spike_count + 2, self.embed_dim)

        if self.model_dim % self.num_heads != 0:
            raise ValueError(
                "NDT model dimension must be divisible by num_heads; got "
                f"model_dim={self.model_dim}, num_heads={self.num_heads}."
            )

        self.input_scale = math.sqrt(self.model_dim)
        self.position = PositionalEncoding(
            n_time=self.n_time,
            model_dim=self.model_dim,
            dropout=self.dropout_embedding,
            learnable=self.learnable_position,
            offset=self.position_offset,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_size,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
            norm_first=self.pre_norm,
        )
        norm: nn.Module
        if self.scale_norm:
            norm = ScaleNorm(self.model_dim**0.5)
        else:
            norm = nn.LayerNorm(self.model_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
            norm=norm,
            enable_nested_tensor=False,
        )
        self.rate_dropout = nn.Dropout(self.dropout_rates)
        self.decoder = self._build_decoder()
        self._attn_masks: dict[str, Tensor | None] = {}

        self.init_weights()
        if self.fixup_init:
            self.fixup_initialization()

    def forward(self, x: Tensor) -> ModelOutput:
        return self._forward(x, should_mask=self.training)

    def loss(
        self,
        batch: Tensor | dict[str, Tensor],
        output: ModelOutput,
        epoch: int = 0,
    ) -> LossOutput:
        target = observations_from_batch(batch).to(device=self.device, dtype=output.rates.dtype)
        mask = output.extras.get("loss_mask")
        if mask is None:
            mask = torch.ones_like(target, dtype=torch.bool, device=target.device)
        else:
            mask = mask.to(device=target.device, dtype=torch.bool)

        if self.lograte:
            log_rates = output.extras["log_rates"].to(device=target.device, dtype=target.dtype)
            per_entry = F.poisson_nll_loss(
                log_rates,
                target,
                log_input=True,
                full=False,
                reduction="none",
            )
        else:
            rates = output.rates.to(device=target.device, dtype=target.dtype).clamp_min(1e-8)
            per_entry = rates - target * torch.log(rates)

        masked = per_entry[mask]
        if masked.numel() == 0:
            masked = per_entry.reshape(-1)
        if self.topk_loss_fraction < 1.0 and masked.numel() > 1:
            k = max(1, int(masked.numel() * self.topk_loss_fraction))
            masked = torch.topk(masked, k=k).values

        total = masked.mean()
        return LossOutput(
            total=total,
            named_terms={
                "masked_poisson_nll": total,
                "mask_fraction": mask.float().mean(),
            },
            objective=self.objective,
        )

    @torch.no_grad()
    def predict_rates(self, x: Tensor) -> Tensor:
        was_training = self.training
        self.eval()
        try:
            return self._forward(x, should_mask=False).rates
        finally:
            self.train(was_training)

    def _forward(self, x: Tensor, should_mask: bool) -> ModelOutput:
        if x.ndim != 3:
            raise ValueError("NDT expects input shape (batch, time, neurons).")
        if x.shape[1] != self.n_time:
            raise ValueError(f"Expected {self.n_time} time bins, got {x.shape[1]}.")
        if x.shape[-1] != self.n_neurons:
            raise ValueError(f"Expected {self.n_neurons} neurons, got {x.shape[-1]}.")
        if torch.any(x < 0):
            raise ValueError("NDT expects nonnegative spike-count observations.")

        x = x.to(device=self.device, dtype=torch.float32)
        masked_x, labels, loss_mask = self._mask_observations(x, should_mask=should_mask)
        embedded = self._embed(masked_x) * self.input_scale
        embedded = self.position(embedded)
        attn_mask = self._get_or_generate_context_mask(embedded)
        factors = self.transformer_encoder(embedded, mask=attn_mask)
        decoded = self.decoder(self.rate_dropout(factors))
        if self.lograte:
            log_rates = decoded.clamp(min=self.log_rate_min, max=self.log_rate_max)
            rates = torch.exp(log_rates)
        else:
            rates = decoded.clamp_min(0.0)
            log_rates = torch.log(rates.clamp_min(1e-8))

        return ModelOutput(
            rates=rates,
            latents=factors,
            extras={
                "log_rates": log_rates,
                "mask_labels": labels,
                "loss_mask": loss_mask,
            },
        )

    def _build_decoder(self) -> nn.Module:
        if self.decoder_layers == 1:
            layers: list[nn.Module] = [nn.Linear(self.model_dim, self.n_neurons)]
        else:
            layers = [nn.Linear(self.model_dim, 16), nn.ReLU()]
            for _ in range(self.decoder_layers - 2):
                layers.extend([nn.Linear(16, 16), nn.ReLU()])
            layers.append(nn.Linear(16, self.n_neurons))
        if not self.lograte:
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def _embed(self, x: Tensor) -> Tensor:
        if self.linear_embedder:
            return self.embedder(x.float())
        if self.embed_dim == 0:
            return x.float()
        tokens = x.round().long().clamp(min=0, max=self.max_spike_count + 1)
        embedded = self.embedder(tokens)
        return embedded.flatten(start_dim=-2)

    def _mask_observations(self, x: Tensor, should_mask: bool) -> tuple[Tensor, Tensor, Tensor]:
        labels = x.clone()
        if not should_mask or self.mask_ratio <= 0.0:
            return x, labels, torch.ones_like(x, dtype=torch.bool)

        mask = self._sample_loss_mask(x)
        labels[~mask] = UNMASKED_LABEL
        masked_x = x.clone()

        replace_mask = (
            torch.rand_like(x, dtype=torch.float32) < self.mask_token_ratio
        ) & mask
        if self.use_zero_mask:
            masked_x[replace_mask] = 0.0
        else:
            masked_x[replace_mask] = float(self.max_spike_count + 1)

        random_mask = (
            torch.rand_like(x, dtype=torch.float32) < self.mask_random_ratio
        ) & mask & ~replace_mask
        if random_mask.any():
            max_count = min(max(int(torch.ceil(x.max()).item()), 1), self.max_spike_count)
            random_spikes = torch.randint(
                high=max_count + 1,
                size=x.shape,
                device=x.device,
                dtype=torch.long,
            ).float()
            masked_x[random_mask] = random_spikes[random_mask]

        return masked_x, labels, mask

    def _sample_loss_mask(self, x: Tensor) -> Tensor:
        batch, time, neurons = x.shape
        ratio = self.mask_ratio
        width = 1
        should_expand = (
            self.mask_max_span > 1
            and self.mask_span_expand_prob > 0.0
            and torch.rand((), device=x.device).item() < self.mask_span_expand_prob
        )
        if should_expand:
            width = int(torch.randint(1, self.mask_max_span + 1, (), device=x.device).item())
            ratio = ratio / max(width, 1)

        if self.mask_mode == "full":
            mask = torch.rand((batch, time, neurons), device=x.device) < ratio
        elif self.mask_mode == "timestep":
            mask_2d = torch.rand((batch, time), device=x.device) < ratio
            if width > 1:
                mask_2d = _expand_time_mask(mask_2d, width)
            mask = mask_2d.unsqueeze(-1).expand(batch, time, neurons)
        elif self.mask_mode == "neuron":
            mask_2d = torch.rand((batch, neurons), device=x.device) < ratio
            mask = mask_2d.unsqueeze(1).expand(batch, time, neurons)
        elif self.mask_mode == "timestep_only":
            mask_1d = torch.rand((time,), device=x.device) < ratio
            if width > 1:
                mask_1d = _expand_time_mask(mask_1d.unsqueeze(0), width).squeeze(0)
            mask = mask_1d.view(1, time, 1).expand(batch, time, neurons)
        else:
            raise KeyError(f"Unknown NDT mask mode '{self.mask_mode}'.")

        if not bool(mask.any()):
            flat_index = torch.randint(mask.numel(), (), device=x.device)
            mask = mask.reshape(-1)
            mask[flat_index] = True
            mask = mask.reshape(batch, time, neurons)
        return mask

    def _get_or_generate_context_mask(self, src: Tensor) -> Tensor | None:
        if self.full_context:
            return None
        key = f"{src.device}:{src.shape[1]}:{src.dtype}"
        if key in self._attn_masks:
            return self._attn_masks[key]

        size = src.shape[1]
        context_forward = self.context_forward if self.context_forward >= 0 else size
        allowed = (
            torch.triu(torch.ones(size, size, device=src.device), diagonal=-context_forward) == 1
        ).transpose(0, 1)
        if self.context_backward > 0:
            back_allowed = (
                torch.triu(torch.ones(size, size, device=src.device), diagonal=-self.context_backward)
                == 1
            )
            allowed = allowed & back_allowed
        if self.context_wrap_initial and self.context_backward > 0:
            initial = min(self.context_backward, size)
            initial_mask = torch.triu(torch.ones(initial, initial, device=src.device)).bool()
            allowed[:initial, :initial] |= initial_mask

        attn_mask = torch.zeros(size, size, device=src.device, dtype=src.dtype)
        attn_mask = attn_mask.masked_fill(~allowed, float("-inf"))
        self._attn_masks[key] = attn_mask
        return attn_mask

    def init_weights(self) -> None:
        initrange = 0.1
        if isinstance(self.embedder, nn.Embedding):
            if self.spike_log_init:
                max_spikes = self.embedder.num_embeddings + 1
                log_scale = torch.arange(
                    1,
                    max_spikes,
                    device=self.embedder.weight.device,
                    dtype=self.embedder.weight.dtype,
                ).log()
                log_scale = (log_scale - log_scale.mean()) / (log_scale[-1] - log_scale[0])
                log_scale = log_scale[: self.embedder.num_embeddings] * initrange
                self.embedder.weight.data.uniform_(-initrange / 10.0, initrange / 10.0)
                self.embedder.weight.data += log_scale.unsqueeze(1).expand_as(
                    self.embedder.weight.data
                )
            else:
                self.embedder.weight.data.uniform_(-initrange, initrange)

        for module in self.decoder.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.uniform_(-initrange, initrange)
                module.bias.data.zero_()

    def fixup_initialization(self) -> None:
        scale = 0.67 * (self.num_layers ** (-0.25))
        for module in self.transformer_encoder.layers:
            module.linear1.weight.data.mul_(scale)
            module.linear2.weight.data.mul_(scale)
            module.self_attn.out_proj.weight.data.mul_(scale)
            module.self_attn.in_proj_weight.data.mul_(scale)


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        n_time: int,
        model_dim: int,
        dropout: float,
        learnable: bool,
        offset: bool = True,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.learnable = bool(learnable)
        if self.learnable:
            self.register_buffer("position_ids", torch.arange(n_time, dtype=torch.long))
            self.embedding = nn.Embedding(n_time, model_dim)
        else:
            position = torch.arange(0, n_time, dtype=torch.float32).unsqueeze(1)
            if offset:
                position = position + 1
            pe = torch.zeros(n_time, model_dim, dtype=torch.float32)
            div_term = torch.exp(
                torch.arange(0, model_dim, 2, dtype=torch.float32)
                * (-math.log(10000.0) / model_dim)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            if model_dim > 1:
                pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
            self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        if self.learnable:
            positions = self.position_ids[: x.shape[1]].to(x.device)
            x = x + self.embedding(positions).unsqueeze(0)
        else:
            x = x + self.pe[:, : x.shape[1]].to(device=x.device, dtype=x.dtype)
        return self.dropout(x)


class ScaleNorm(nn.Module):
    def __init__(self, scale: float, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.eps = float(eps)

    def forward(self, x: Tensor) -> Tensor:
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm


def _expand_time_mask(mask: Tensor, width: int) -> Tensor:
    kernel = torch.ones(width, device=mask.device, dtype=mask.dtype).view(1, 1, -1)
    expanded = F.conv1d(mask.float().unsqueeze(1), kernel.float(), padding=width // 2)
    if width % 2 == 0:
        expanded = expanded[..., :-1]
    return expanded.squeeze(1).clamp_(0, 1).bool()
