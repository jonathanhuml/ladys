"""Export one shard of full-rate NLB predictions from a saved checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ladys.config import load_experiment_config
from ladys.experiment import Experiment
from ladys.types import move_batch_to_device, observations_from_batch


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export train/eval held-in and held-out full-rate tensors for a trial "
            "range. Rates are written as spike-count rates per bin, matching the "
            "NLB EvalAI submission format."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "eval"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prediction-floor", type=float, default=1e-9)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    experiment = Experiment(config)
    experiment.data.setup()
    model = experiment.build_model()
    state = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)

    dataset = (
        experiment.data.train_dataset
        if args.split == "train"
        else experiment.data.valid_dataset
    )
    if dataset is None:
        raise RuntimeError("Experiment data was not initialized.")
    total_trials = len(dataset)
    start = int(args.start)
    stop = total_trials if args.stop is None else int(args.stop)
    if start < 0 or stop < start or stop > total_trials:
        raise ValueError(
            f"invalid shard range [{start}, {stop}) for {args.split} split with "
            f"{total_trials} trials"
        )

    batch_size = int(args.batch_size or experiment.data.batch_size)
    subset = Subset(dataset, range(start, stop))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    device = torch.device(args.device)
    model.to(device)
    model.eval()
    dt = float(
        getattr(config.dataset, "bin_size", None)
        or float(getattr(config.dataset, "bin_size_ms")) / 1000.0
    )
    parts = collect_full_rate_parts(
        model=model,
        loader=loader,
        device=device,
        dt=dt,
        prediction_floor=float(args.prediction_floor),
        progress_label=args.split if args.progress else None,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "split": args.split,
        "start": start,
        "stop": stop,
        "total": total_trials,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "batch_size": batch_size,
    }
    payload.update(parts)
    np.savez(output, **payload)

    summary = {
        "output": str(output),
        "split": args.split,
        "start": start,
        "stop": stop,
        "total": total_trials,
        "shapes": {key: list(value.shape) for key, value in parts.items()},
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


def collect_full_rate_parts(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    dt: float,
    prediction_floor: float,
    progress_label: str | None = None,
) -> dict[str, np.ndarray]:
    heldin_rates: list[np.ndarray] = []
    heldout_rates: list[np.ndarray] = []
    heldin_forward_rates: list[np.ndarray] = []
    heldout_forward_rates: list[np.ndarray] = []
    any_forward = False

    total_batches = len(loader)
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            if not isinstance(batch, dict):
                raise TypeError("NLB full-rate export expects dictionary batches.")
            if progress_label is not None:
                print(
                    f"{progress_label} batch {batch_index}/{total_batches}",
                    flush=True,
                )
            batch = move_batch_to_device(batch, device)
            output = model(observations_from_batch(batch))
            extras = getattr(output, "extras", {})
            full_rates = extras.get("full_rates") if isinstance(extras, dict) else None
            if full_rates is None:
                full_rates = output.rates if getattr(output, "rates", None) is not None else None
            if full_rates is None:
                raise RuntimeError(
                    f"{type(model).__name__} did not return rates or extras['full_rates']."
                )

            full_counts = (full_rates * dt).clamp_min(prediction_floor)
            heldin = batch.get("heldin_spikes", batch.get("spikes"))
            heldout = batch.get("heldout_spikes", batch.get("raw_spikes"))
            if heldin is None or heldout is None:
                raise KeyError("NLB batch is missing held-in or held-out spike tensors.")

            main_steps = int(heldout.shape[1])
            heldin_neurons = int(heldin.shape[-1])
            heldout_neurons = int(heldout.shape[-1])
            heldout_start = int(
                getattr(model, "output_neuron_start", heldin_neurons)
                or heldin_neurons
            )
            heldout_stop = heldout_start + heldout_neurons
            if full_counts.shape[1] < main_steps:
                raise ValueError(
                    f"model returned {full_counts.shape[1]} time steps but "
                    f"{main_steps} are required"
                )
            if full_counts.shape[-1] < heldout_stop:
                raise ValueError(
                    f"model returned {full_counts.shape[-1]} neurons but "
                    f"{heldout_stop} are required"
                )

            heldin_rates.append(
                full_counts[:, :main_steps, :heldin_neurons].detach().cpu().numpy()
            )
            heldout_rates.append(
                full_counts[:, :main_steps, heldout_start:heldout_stop]
                .detach()
                .cpu()
                .numpy()
            )

            heldin_forward = batch.get("heldin_forward_spikes")
            heldout_forward = batch.get("heldout_forward_spikes")
            if heldin_forward is not None and heldout_forward is not None:
                forward_steps = int(heldout_forward.shape[1])
                if full_counts.shape[1] >= main_steps + forward_steps:
                    any_forward = True
                    heldin_forward_rates.append(
                        full_counts[
                            :,
                            main_steps : main_steps + forward_steps,
                            :heldin_neurons,
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    heldout_forward_rates.append(
                        full_counts[
                            :,
                            main_steps : main_steps + forward_steps,
                            heldout_start:heldout_stop,
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                    )

    if not heldin_rates or not heldout_rates:
        raise RuntimeError("No batches were exported.")

    parts = {
        "rates_heldin": np.concatenate(heldin_rates, axis=0).astype(np.float32, copy=False),
        "rates_heldout": np.concatenate(heldout_rates, axis=0).astype(np.float32, copy=False),
    }
    if any_forward and heldin_forward_rates and heldout_forward_rates:
        parts["rates_heldin_forward"] = np.concatenate(
            heldin_forward_rates,
            axis=0,
        ).astype(np.float32, copy=False)
        parts["rates_heldout_forward"] = np.concatenate(
            heldout_forward_rates,
            axis=0,
        ).astype(np.float32, copy=False)
    return parts


if __name__ == "__main__":
    raise SystemExit(main())
