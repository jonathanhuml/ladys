#!/usr/bin/env python3
"""Generate native LaDyS STNDT sweep configs from upstream STNDT search spaces."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ladys.utils.yaml import load_yaml


BASE_CONFIGS = {
    "area2_bump": Path(
        "configs/experiment/real/area2_bump/stndt/stndt_area2_bump_nlb_5ms.yaml"
    ),
    "dmfc_rsg": Path("configs/experiment/real/dmfc_rsg/stndt/stndt_dmfc_rsg_nlb_5ms.yaml"),
    "mc_rtt": Path("configs/experiment/real/mc_rtt/stndt/stndt_mc_rtt_nlb_5ms.yaml"),
}


SEARCH_SPACES: dict[str, dict[str, dict[str, Any]]] = {
    "area2_bump": {
        "MODEL.DROPOUT": {"low": 0.0, "high": 0.6},
        "MODEL.DROPOUT_RATES": {"low": 0.0, "high": 0.6},
        "MODEL.DROPOUT_EMBEDDING": {"low": 0.0, "high": 0.6},
        "MODEL.CONTEXT_FORWARD": {"low": 4, "high": 100, "sample_fn": "randint"},
        "MODEL.CONTEXT_BACKWARD": {"low": 4, "high": 100, "sample_fn": "randint"},
        "TRAIN.LR.INIT": {"low": 1e-5, "high": 1e-2},
        "TRAIN.LR.WARMUP": {"low": 0, "high": 7000, "sample_fn": "randint"},
        "TRAIN.MASK_RANDOM_RATIO": {"low": 0.9, "high": 1.0},
        "TRAIN.MASK_TOKEN_RATIO": {"low": 0.5, "high": 1.0, "sample_fn": "loguniform"},
        "TRAIN.MASK_RATIO": {"low": 0.1, "high": 0.6},
        "TRAIN.CONTRAST_MASK_RATIO": {"low": 0.05, "high": 0.2},
        "TRAIN.CONTRAST_MASK_RANDOM_RATIO": {"low": 0.3, "high": 1.0},
        "TRAIN.CONTRAST_MASK_TOKEN_RATIO": {
            "low": 0.5,
            "high": 1.0,
            "sample_fn": "loguniform",
        },
        "MODEL.LAMBDA": {"low": 0.01, "high": 1.0},
    },
    "dmfc_rsg": {
        "MODEL.DROPOUT": {"low": 0.3, "high": 0.7},
        "MODEL.DROPOUT_RATES": {"low": 0.3, "high": 0.7},
        "MODEL.DROPOUT_EMBEDDING": {"low": 0.3, "high": 0.7},
        "MODEL.CONTEXT_FORWARD": {"low": 40, "high": 240, "sample_fn": "randint"},
        "MODEL.CONTEXT_BACKWARD": {"low": 40, "high": 240, "sample_fn": "randint"},
        "TRAIN.LR.INIT": {"low": 1e-5, "high": 5e-3},
        "TRAIN.LR.WARMUP": {"low": 0, "high": 2000, "sample_fn": "randint"},
        "TRAIN.MASK_RANDOM_RATIO": {"low": 0.9, "high": 1.0},
        "TRAIN.MASK_TOKEN_RATIO": {"low": 0.5, "high": 1.0, "sample_fn": "loguniform"},
        "TRAIN.MASK_MAX_SPAN": {"low": 1, "high": 7, "sample_fn": "randint"},
        "TRAIN.CONTRAST_MASK_RATIO": {"low": 0.05, "high": 0.2},
        "TRAIN.CONTRAST_MASK_RANDOM_RATIO": {"low": 0.3, "high": 1.0},
        "TRAIN.CONTRAST_MASK_TOKEN_RATIO": {
            "low": 0.5,
            "high": 1.0,
            "sample_fn": "loguniform",
        },
        "MODEL.LAMBDA": {"low": 0.01, "high": 1.0},
    },
    "mc_rtt": {
        "MODEL.DROPOUT": {"low": 0.0, "high": 0.4},
        "MODEL.DROPOUT_RATES": {"low": 0.0, "high": 0.6},
        "MODEL.DROPOUT_EMBEDDING": {"low": 0.0, "high": 0.5},
        "MODEL.CONTEXT_FORWARD": {"low": 1, "high": 100, "sample_fn": "randint"},
        "MODEL.CONTEXT_BACKWARD": {"low": 1, "high": 100, "sample_fn": "randint"},
        "TRAIN.LR.INIT": {"low": 1e-4, "high": 1e-1},
        "TRAIN.LR.WARMUP": {"low": 0, "high": 7000, "sample_fn": "randint"},
        "TRAIN.MASK_RANDOM_RATIO": {"low": 0.6, "high": 1.0},
        "TRAIN.MASK_TOKEN_RATIO": {"low": 0.5, "high": 1.0, "sample_fn": "loguniform"},
        "TRAIN.MASK_MAX_SPAN": {"low": 1, "high": 7, "sample_fn": "randint"},
        "TRAIN.CONTRAST_MASK_RATIO": {"low": 0.05, "high": 0.2},
        "TRAIN.CONTRAST_MASK_RANDOM_RATIO": {"low": 0.3, "high": 1.0},
        "TRAIN.CONTRAST_MASK_TOKEN_RATIO": {
            "low": 0.5,
            "high": 1.0,
            "sample_fn": "loguniform",
        },
        "MODEL.LAMBDA": {"low": 0.01, "high": 1.0},
    },
}


DEFAULT_HP_SPACE = {
    "TRAIN.WEIGHT_DECAY": {"low": 1e-8, "high": 1e-3, "sample_fn": "loguniform"},
    "TRAIN.MASK_RATIO": {"low": 0.1, "high": 0.4},
}


FIELD_MAP = {
    "MODEL.DROPOUT": ("model", "dropout"),
    "MODEL.DROPOUT_RATES": ("model", "dropout_rates"),
    "MODEL.DROPOUT_EMBEDDING": ("model", "dropout_embedding"),
    "MODEL.CONTEXT_FORWARD": ("model", "context_forward"),
    "MODEL.CONTEXT_BACKWARD": ("model", "context_backward"),
    "MODEL.LAMBDA": ("model", "contrast_lambda"),
    "TRAIN.LR.INIT": ("model", "optimization", "lr"),
    "TRAIN.LR.WARMUP": ("model", "optimization", "warmup_steps"),
    "TRAIN.WEIGHT_DECAY": ("model", "optimization", "weight_decay"),
    "TRAIN.MASK_RATIO": ("model", "mask_ratio"),
    "TRAIN.MASK_RANDOM_RATIO": ("model", "mask_random_ratio"),
    "TRAIN.MASK_TOKEN_RATIO": ("model", "mask_token_ratio"),
    "TRAIN.MASK_MAX_SPAN": ("model", "mask_max_span"),
    "TRAIN.CONTRAST_MASK_RATIO": ("model", "contrast_mask_ratio"),
    "TRAIN.CONTRAST_MASK_RANDOM_RATIO": ("model", "contrast_mask_random_ratio"),
    "TRAIN.CONTRAST_MASK_TOKEN_RATIO": ("model", "contrast_mask_token_ratio"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(BASE_CONFIGS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-random", type=int, default=8)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument(
        "--input-mode",
        choices=["heldin", "full_observed"],
        help="Override dataset.input_mode in generated configs.",
    )
    parser.add_argument(
        "--include-hand-points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include a few deterministic boundary/focused candidates.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_paths: list[Path] = []
    manifest: list[dict[str, Any]] = []
    for dataset in args.datasets:
        rng = random.Random(args.seed + stable_offset(dataset))
        base = load_yaml(ROOT / BASE_CONFIGS[dataset])
        candidates: list[tuple[str, dict[str, Any]]] = []
        if args.include_hand_points:
            candidates.extend(hand_points(dataset))
        for index in range(args.num_random):
            candidates.append((f"random{index:02d}", sample_candidate(dataset, rng)))

        for index, (label, flat) in enumerate(candidates):
            config = build_config(
                base,
                dataset=dataset,
                label=label,
                flat=flat,
                input_mode=args.input_mode,
            )
            path = args.output_dir / f"stndt_{dataset}_{index:02d}_{label}.yaml"
            write_yaml(path, config)
            all_paths.append(path)
            manifest.append({"dataset": dataset, "label": label, "path": str(path), "flat": flat})

    list_path = args.output_dir / "config_list.txt"
    list_path.write_text("".join(f"{path}\n" for path in all_paths))
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(list_path)
    return 0


def stable_offset(dataset: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(dataset))


def sample_candidate(dataset: str, rng: random.Random) -> dict[str, Any]:
    space = dict(DEFAULT_HP_SPACE)
    space.update(SEARCH_SPACES[dataset])
    return {key: sample_value(spec, rng) for key, spec in space.items()}


def sample_value(spec: dict[str, Any], rng: random.Random) -> Any:
    low = spec["low"]
    high = spec["high"]
    sample_fn = spec.get("sample_fn", "uniform")
    if sample_fn == "randint":
        return int(rng.randrange(int(low), int(high)))
    if sample_fn == "loguniform":
        return math.exp(rng.uniform(math.log(float(low)), math.log(float(high))))
    if sample_fn == "uniform":
        return rng.uniform(float(low), float(high))
    raise KeyError(f"Unsupported sample_fn: {sample_fn}")


def hand_points(dataset: str) -> list[tuple[str, dict[str, Any]]]:
    if dataset == "area2_bump":
        return [
            (
                "short_ctx_fast",
                {
                    "MODEL.CONTEXT_FORWARD": 16,
                    "MODEL.CONTEXT_BACKWARD": 16,
                    "MODEL.DROPOUT": 0.1,
                    "MODEL.DROPOUT_RATES": 0.2,
                    "MODEL.DROPOUT_EMBEDDING": 0.2,
                    "TRAIN.LR.INIT": 3.0e-3,
                    "TRAIN.LR.WARMUP": 1200,
                    "TRAIN.WEIGHT_DECAY": 5.0e-5,
                    "TRAIN.MASK_RATIO": 0.25,
                    "TRAIN.MASK_TOKEN_RATIO": 1.0,
                    "TRAIN.MASK_RANDOM_RATIO": 0.95,
                    "TRAIN.CONTRAST_MASK_RATIO": 0.05,
                    "TRAIN.CONTRAST_MASK_TOKEN_RATIO": 0.5,
                    "TRAIN.CONTRAST_MASK_RANDOM_RATIO": 0.5,
                    "MODEL.LAMBDA": 0.1,
                },
            ),
            (
                "wide_ctx_low_drop",
                {
                    "MODEL.CONTEXT_FORWARD": 80,
                    "MODEL.CONTEXT_BACKWARD": 32,
                    "MODEL.DROPOUT": 0.15,
                    "MODEL.DROPOUT_RATES": 0.15,
                    "MODEL.DROPOUT_EMBEDDING": 0.15,
                    "TRAIN.LR.INIT": 1.5e-3,
                    "TRAIN.LR.WARMUP": 2500,
                    "TRAIN.WEIGHT_DECAY": 1.0e-4,
                    "TRAIN.MASK_RATIO": 0.3,
                    "TRAIN.MASK_TOKEN_RATIO": 0.8,
                    "TRAIN.MASK_RANDOM_RATIO": 0.95,
                    "TRAIN.CONTRAST_MASK_RATIO": 0.1,
                    "TRAIN.CONTRAST_MASK_TOKEN_RATIO": 0.8,
                    "TRAIN.CONTRAST_MASK_RANDOM_RATIO": 0.7,
                    "MODEL.LAMBDA": 0.3,
                },
            ),
        ]
    if dataset == "dmfc_rsg":
        return [
            (
                "long_ctx_mid_drop",
                {
                    "MODEL.CONTEXT_FORWARD": 120,
                    "MODEL.CONTEXT_BACKWARD": 120,
                    "MODEL.DROPOUT": 0.5,
                    "MODEL.DROPOUT_RATES": 0.5,
                    "MODEL.DROPOUT_EMBEDDING": 0.5,
                    "TRAIN.LR.INIT": 1.0e-3,
                    "TRAIN.LR.WARMUP": 1000,
                    "TRAIN.WEIGHT_DECAY": 5.0e-5,
                    "TRAIN.MASK_RATIO": 0.25,
                    "TRAIN.MASK_TOKEN_RATIO": 1.0,
                    "TRAIN.MASK_RANDOM_RATIO": 0.95,
                    "TRAIN.MASK_MAX_SPAN": 3,
                    "TRAIN.CONTRAST_MASK_RATIO": 0.05,
                    "TRAIN.CONTRAST_MASK_TOKEN_RATIO": 0.5,
                    "TRAIN.CONTRAST_MASK_RANDOM_RATIO": 0.5,
                    "MODEL.LAMBDA": 0.1,
                },
            ),
        ]
    if dataset == "mc_rtt":
        return [
            (
                "mc_maze_like",
                {
                    "MODEL.CONTEXT_FORWARD": 46,
                    "MODEL.CONTEXT_BACKWARD": 7,
                    "MODEL.DROPOUT": 0.3,
                    "MODEL.DROPOUT_RATES": 0.3,
                    "MODEL.DROPOUT_EMBEDDING": 0.4,
                    "TRAIN.LR.INIT": 8.0e-3,
                    "TRAIN.LR.WARMUP": 1000,
                    "TRAIN.WEIGHT_DECAY": 3.0e-4,
                    "TRAIN.MASK_RATIO": 0.25,
                    "TRAIN.MASK_TOKEN_RATIO": 0.85,
                    "TRAIN.MASK_RANDOM_RATIO": 0.85,
                    "TRAIN.MASK_MAX_SPAN": 5,
                    "TRAIN.CONTRAST_MASK_RATIO": 0.15,
                    "TRAIN.CONTRAST_MASK_TOKEN_RATIO": 0.8,
                    "TRAIN.CONTRAST_MASK_RANDOM_RATIO": 0.8,
                    "MODEL.LAMBDA": 0.5,
                },
            ),
        ]
    raise KeyError(dataset)


def build_config(
    base: dict[str, Any],
    *,
    dataset: str,
    label: str,
    flat: dict[str, Any],
    input_mode: str | None,
) -> dict[str, Any]:
    config = deepcopy(base)
    for source_key, value in flat.items():
        target = FIELD_MAP.get(source_key)
        if target is None:
            raise KeyError(source_key)
        set_nested(config, target, value)
    suffix = label if input_mode is None else f"{label}_{input_mode}"
    if input_mode is not None:
        config["dataset"]["input_mode"] = input_mode
    config["experiment"]["run_name"] = f"stndt_{dataset}_nlb_5ms_sweep_{suffix}"
    return config


def set_nested(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor = config
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to write sweep configs.") from exc
    path.write_text(yaml.safe_dump(data, sort_keys=False))


if __name__ == "__main__":
    raise SystemExit(main())
