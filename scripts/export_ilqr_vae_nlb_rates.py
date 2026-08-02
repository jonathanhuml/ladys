"""Export iLQR-VAE NLB held-in and held-out rates for a split shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ladys.config import load_experiment_config
from ladys.experiment import Experiment
from ladys.types import move_batch_to_device, observations_from_batch


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run iLQR-VAE posterior inference over a train/eval NLB split shard "
            "and save full held-in plus held-out count-rate predictions."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["train", "eval"], default="eval")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prediction-floor", type=float, default=1e-9)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    experiment = Experiment(config)
    experiment.data.setup()
    model = experiment.build_model()
    state = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(args.device)
    model.eval()

    dataset = (
        experiment.data.train_dataset
        if args.split == "train"
        else experiment.data.valid_dataset
    )
    if dataset is None:
        raise RuntimeError("Experiment data was not initialized.")

    total = len(dataset)
    start = max(int(args.start), 0)
    stop = total if args.stop is None else min(int(args.stop), total)
    if stop <= start:
        raise ValueError(f"empty shard range: start={start}, stop={stop}, total={total}")

    heldin = getattr(dataset, "heldin_spikes")
    heldout = getattr(dataset, "raw_spikes")
    heldin_neurons = int(heldin.shape[-1])
    heldout_neurons = int(heldout.shape[-1])
    main_steps = int(heldout.shape[1])
    heldout_start = int(getattr(model, "output_neuron_start", heldin_neurons) or heldin_neurons)
    heldout_stop = heldout_start + heldout_neurons
    dt = float(
        getattr(config.dataset, "bin_size", None)
        or float(getattr(config.dataset, "bin_size_ms")) / 1000.0
    )

    subset = Subset(dataset, range(start, stop))
    loader = DataLoader(
        subset,
        batch_size=int(args.batch_size or config.batch_size),
        shuffle=False,
    )

    rates_heldin: list[np.ndarray] = []
    rates_heldout: list[np.ndarray] = []
    seen = start
    next_progress = start + max(int(args.progress_every), 1)
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, torch.device(args.device))
            x = observations_from_batch(batch)
            output = model(x)
            full_rates_hz = output.extras["full_rates"][:, :main_steps]
            full_counts = (full_rates_hz * dt).clamp_min(float(args.prediction_floor))
            if full_counts.shape[-1] < heldout_stop:
                raise ValueError(
                    f"model decoded {full_counts.shape[-1]} neurons but "
                    f"{heldout_stop} are required."
                )
            rates_heldin.append(
                full_counts[:, :, :heldin_neurons]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            rates_heldout.append(
                full_counts[:, :, heldout_start:heldout_stop]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            seen += int(x.shape[0])
            if (
                args.progress_every
                and (
                    seen == start + int(x.shape[0])
                    or seen >= next_progress
                    or seen >= stop
                )
            ):
                print(
                    json.dumps(
                        {
                            "split": args.split,
                            "start": start,
                            "stop": stop,
                            "seen": seen,
                            "elapsed_s": time.time() - t0,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                next_progress += int(args.progress_every)

    rates_heldin_array = np.concatenate(rates_heldin, axis=0)
    rates_heldout_array = np.concatenate(rates_heldout, axis=0)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        split=args.split,
        start=start,
        stop=stop,
        total=total,
        config=args.config,
        checkpoint=args.checkpoint,
        rates_heldin=rates_heldin_array,
        rates_heldout=rates_heldout_array,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "split": args.split,
                "start": start,
                "stop": stop,
                "total": total,
                "elapsed_s": time.time() - t0,
                "shapes": {
                    "rates_heldin": list(rates_heldin_array.shape),
                    "rates_heldout": list(rates_heldout_array.shape),
                },
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
