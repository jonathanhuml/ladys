"""Evaluate a saved LaDyS checkpoint with the full NLB metric suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ladys.config import load_experiment_config
from ladys.experiment import Experiment
from ladys.metrics import evaluate_model
from ladys.nlb_eval import evaluate_model_nlb_submission


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--skip-direct-metrics",
        action="store_true",
        help="skip the lightweight held-out evaluation before full NLB scoring",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="print full NLB export progress by dataloader batch",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    experiment = Experiment(config)
    experiment.data.setup()
    model = experiment.build_model()
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    if not args.skip_direct_metrics:
        evaluation = evaluate_model(
            model=model,
            loader=experiment.data.valid_loader(),
            device=args.device,
            train_loader=experiment.data.train_loader(shuffle=False),
        )
        metrics.update(evaluation.metrics)
    full = evaluate_model_nlb_submission(
        model=model,
        train_loader=experiment.data.train_loader(shuffle=False),
        valid_loader=experiment.data.valid_loader(),
        dataset_config=config.dataset,
        device=args.device,
        output_dir=output_dir,
        progress=args.progress,
    )
    if full is not None:
        metrics.update(full.metrics)

    payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "metrics": metrics,
    }
    (output_dir / "checkpoint_full_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
