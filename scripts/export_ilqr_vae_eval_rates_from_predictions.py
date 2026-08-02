"""Export full eval rates for iLQR-VAE from saved evaluation latents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ladys.config import load_experiment_config
from ladys.experiment import Experiment


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert pred_latents from a saved iLQR-VAE predictions.npz file into "
            "EvalAI-style eval_rates_heldin/eval_rates_heldout shard arrays. This "
            "avoids rerunning the expensive iLQR posterior solve for eval trials."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prediction-floor", type=float, default=1e-9)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    experiment = Experiment(config)
    experiment.data.setup()
    model = experiment.build_model()
    state = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)

    if experiment.data.valid_dataset is None:
        raise RuntimeError("Experiment validation data was not initialized.")
    valid_dataset = experiment.data.valid_dataset
    total = len(valid_dataset)
    heldin_neurons = int(getattr(valid_dataset, "heldin_spikes").shape[-1])
    heldout_neurons = int(getattr(valid_dataset, "raw_spikes").shape[-1])
    main_steps = int(getattr(valid_dataset, "raw_spikes").shape[1])
    heldout_start = int(
        getattr(model, "output_neuron_start", heldin_neurons)
        or heldin_neurons
    )
    heldout_stop = heldout_start + heldout_neurons

    with np.load(args.predictions) as data:
        if "pred_latents" not in data:
            raise KeyError(f"{args.predictions} does not contain pred_latents")
        latents = torch.from_numpy(np.asarray(data["pred_latents"])).to(
            device=torch.device(args.device),
            dtype=model.core.c.dtype,
        )
        saved_heldout = data["pred_rates"] if "pred_rates" in data else None

    if latents.shape[0] != total:
        raise ValueError(
            f"pred_latents contains {latents.shape[0]} trials but eval split has {total}"
        )
    if latents.shape[1] < main_steps:
        raise ValueError(
            f"pred_latents contains {latents.shape[1]} time steps but {main_steps} are required"
        )

    model.to(args.device)
    model.eval()
    dt = float(
        getattr(config.dataset, "bin_size", None)
        or float(getattr(config.dataset, "bin_size_ms")) / 1000.0
    )
    with torch.no_grad():
        full_rates_hz = model.core.firing_rates(latents[:, :main_steps], mode=model.rate_mode)
        full_counts = (full_rates_hz * dt).clamp_min(float(args.prediction_floor))
        if full_counts.shape[-1] < heldout_stop:
            raise ValueError(
                f"model decoded {full_counts.shape[-1]} neurons but {heldout_stop} are required"
            )
        rates_heldin = full_counts[:, :, :heldin_neurons].detach().cpu().numpy()
        rates_heldout = (
            full_counts[:, :, heldout_start:heldout_stop].detach().cpu().numpy()
        )

    max_abs_saved_diff = None
    if saved_heldout is not None:
        max_abs_saved_diff = float(np.max(np.abs(rates_heldout - saved_heldout)))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        split="eval",
        start=0,
        stop=total,
        total=total,
        config=args.config,
        checkpoint=args.checkpoint,
        predictions=args.predictions,
        rates_heldin=rates_heldin.astype(np.float32, copy=False),
        rates_heldout=rates_heldout.astype(np.float32, copy=False),
    )

    summary = {
        "output": str(output),
        "split": "eval",
        "start": 0,
        "stop": total,
        "total": total,
        "max_abs_saved_heldout_diff": max_abs_saved_diff,
        "shapes": {
            "rates_heldin": list(rates_heldin.shape),
            "rates_heldout": list(rates_heldout.shape),
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
