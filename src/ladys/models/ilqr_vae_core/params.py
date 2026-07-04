"""Load OCaml tutorial parameters into named NumPy arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .ocaml_marshal import Bigarray, Block, load_marshal


@dataclass(frozen=True)
class TutorialParams:
    spatial_stds: np.ndarray
    nu: float
    first_step: np.ndarray
    uf: np.ndarray
    wh: np.ndarray
    uh: np.ndarray
    bh: np.ndarray
    b: np.ndarray
    c: np.ndarray
    bias: np.ndarray
    gain: np.ndarray
    space_cov_d: np.ndarray
    space_cov_t: np.ndarray
    time_cov_d: np.ndarray
    time_cov_t: np.ndarray


def load_tutorial_params(path: str | Path) -> TutorialParams:
    """Load ``final_params.bin`` or ``progress_*.params.bin``.

    The expected parameter layout is the model defined in ``demo.ipynb``:
    Student prior, Mini_GRU_IO dynamics, Poisson likelihood, and iLQR
    recognition covariance.
    """

    root = _block(load_marshal(path), tag=0, size=2, label="model")
    generative = _block(root.fields[0], tag=0, label="generative")
    recognition = _block(root.fields[1], tag=0, label="recognition")
    if len(generative.fields) not in {3, 4}:
        raise ValueError(f"generative: expected 3 or 4 fields, got {len(generative.fields)}")

    prior = _block(generative.fields[0], tag=0, size=3, label="prior")
    dynamics = _block(generative.fields[1], tag=0, size=5, label="dynamics")
    likelihood = _block(generative.fields[2], tag=0, size=4, label="likelihood")

    if len(recognition.fields) == 2:
        space_cov = _block(recognition.fields[0], tag=0, size=2, label="space_cov")
        time_cov = _block(recognition.fields[1], tag=0, size=2, label="time_cov")
    elif len(recognition.fields) == 4:
        space_cov = _block(recognition.fields[1], tag=0, size=2, label="space_cov")
        time_cov = _block(recognition.fields[2], tag=0, size=2, label="time_cov")
    else:
        raise ValueError(f"recognition: expected 2 or 4 fields, got {len(recognition.fields)}")

    c_mask = likelihood.fields[1]
    if c_mask != 0:
        raise ValueError("masked likelihood readouts are not supported")

    b = _option_param(dynamics.fields[4], "dynamics.b")
    if b is None:
        raise ValueError("tutorial Mini_GRU_IO parameters should include dynamics.b")

    return TutorialParams(
        spatial_stds=_param_array(prior.fields[0], "prior.spatial_stds"),
        nu=float(_param_array(prior.fields[1], "prior.nu")),
        first_step=_param_array(prior.fields[2], "prior.first_step"),
        uf=_param_array(dynamics.fields[0], "dynamics.uf"),
        wh=_param_array(dynamics.fields[1], "dynamics.wh"),
        uh=_param_array(dynamics.fields[2], "dynamics.uh"),
        bh=_param_array(dynamics.fields[3], "dynamics.bh"),
        b=b,
        c=_param_array(likelihood.fields[0], "likelihood.c"),
        bias=_param_array(likelihood.fields[2], "likelihood.bias"),
        gain=_param_array(likelihood.fields[3], "likelihood.gain"),
        space_cov_d=_param_array(space_cov.fields[0], "recognition.space_cov.d"),
        space_cov_t=_param_array(space_cov.fields[1], "recognition.space_cov.t"),
        time_cov_d=_param_array(time_cov.fields[0], "recognition.time_cov.d"),
        time_cov_t=_param_array(time_cov.fields[1], "recognition.time_cov.t"),
    )


def _param_array(value: Any, label: str) -> np.ndarray:
    """Extract an ``Owl_parameters.tag`` value."""

    block = _block(value, label=label)
    if block.tag in {0, 1}:  # Pinned | Learned
        _check_size(block, 1, label)
        return _ad_value(block.fields[0], label)
    if block.tag == 2:  # Learned_bounded
        _check_size(block, 3, label)
        return _ad_value(block.fields[0], label)
    raise ValueError(f"{label}: unsupported Owl_parameters tag {block.tag}")


def _option_param(value: Any, label: str) -> np.ndarray | None:
    if value == 0:
        return None
    some = _block(value, tag=0, size=1, label=label)
    return _param_array(some.fields[0], label)


def _ad_value(value: Any, label: str) -> np.ndarray:
    block = _block(value, label=label)
    if block.tag == 0:
        _check_size(block, 1, label)
        return np.asarray(block.fields[0], dtype=np.float64)
    if block.tag == 1:
        _check_size(block, 1, label)
        bigarray = block.fields[0]
        if not isinstance(bigarray, Bigarray):
            raise TypeError(f"{label}: expected Bigarray, got {type(bigarray).__name__}")
        return bigarray.data.copy()
    raise ValueError(f"{label}: unsupported Owl Algodiff value tag {block.tag}")


def _block(value: Any, *, tag: int | None = None, size: int | None = None, label: str) -> Block:
    if not isinstance(value, Block):
        raise TypeError(f"{label}: expected OCaml block, got {type(value).__name__}")
    if tag is not None and value.tag != tag:
        raise ValueError(f"{label}: expected tag {tag}, got {value.tag}")
    if size is not None:
        _check_size(value, size, label)
    return value


def _check_size(block: Block, size: int, label: str) -> None:
    if len(block.fields) != size:
        raise ValueError(f"{label}: expected {size} fields, got {len(block.fields)}")
