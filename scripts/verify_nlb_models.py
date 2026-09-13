#!/usr/bin/env python3
"""Bounded, real-data 5 ms training and checkpoint checks, not a benchmark run.

Run from a repository checkout with PYTHONPATH=src:. and prepared validation H5s.
The source H5s are read-only; small diagnostic subsets and reports go to output-dir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from ladys.config import load_experiment_config
from ladys.experiment import Experiment, experiment_config_to_dict
from ladys.metrics import evaluate_model
from ladys.nlb_eval import _collect_full_rate_parts
from ladys.training.strategies import build_strategy
from ladys.types import observations_from_batch


def subset(source: Path, target: Path, dataset: str, steps: int) -> int:
    with h5py.File(source, "r") as src, h5py.File(target, "w") as dst:
        group = src[dataset] if dataset in src else src
        # DMFC scores only an event-defined interval. Use mask metadata, never
        # spike values or model scores, to select a scorable diagnostic window.
        valid = np.isfinite(group["eval_spikes_heldout"][:2]).all(axis=(0, 2))
        starts = np.flatnonzero(np.convolve(valid.astype(int), np.ones(steps, dtype=int), "valid") == steps)
        if not len(starts):
            raise ValueError(f"No {steps}-bin fully scored interval in {source}.")
        start = int(starts[0])
        for prefix, trials in (("train", 4), ("eval", 2)):
            for partition in ("heldin", "heldout"):
                key = f"{prefix}_spikes_{partition}"
                dst[key] = group[key][:trials, start:start + steps]
        return start


def verify(dataset: str, method: str, args) -> dict:
    start = time.perf_counter()
    folder = args.output_dir / dataset / method
    folder.mkdir(parents=True, exist_ok=True)
    source = args.data_root / f"{dataset}_val_5ms.h5"
    data_path = folder / "diagnostic_val_5ms.h5"
    start_bin = subset(source, data_path, dataset, args.steps)
    paths = sorted((Path("configs/experiment/real") / dataset / method).glob("*nlb_5ms*.yaml"))
    config = load_experiment_config(paths[0])
    config.dataset.data_path = str(data_path)
    config.dataset.split = "val"
    config.dataset.max_trials = 4
    config.dataset.include_forward = False
    config.batch_size = 2
    config.trainer.device = args.device
    config.trainer.epochs = 2
    config.output_dir = str(folder)
    if method == "mint":
        config.trainer.epochs = 0
        overrides = dict(sigma=1, delta=2, window_length=4, interp=0, lfads_epochs=1,
                         lfads_generator_dim=4, lfads_factor_dim=3, lfads_encoder_dim=4,
                         lfads_controller_dim=4, lfads_batch_size=2)
    elif method == "ilqr_vae":
        overrides = dict(latent_dim=4, input_dim=2, max_iter=2, n_posterior_samples=1,
                         ilqr_fallback_max_iter=2)
    else:
        overrides = dict(latent_dim=2, n_mc_train=1, n_mc_eval=1,
                         nlb_latent_infer_steps=3, nlb_latent_infer_n_mc=1,
                         nlb_decoder="ridge")
    for key, value in overrides.items():
        setattr(config.model, key, value)
    if method == "bgpfa":
        config.model.optimization.n_mc = 1
    payload = experiment_config_to_dict(config)
    payload["trainer"]["batch_size"] = payload.pop("batch_size")
    (folder / "config.json").write_text(json.dumps(payload, indent=2) + "\n")

    torch.manual_seed(42)
    np.random.seed(42)
    experiment = Experiment(config)
    model = experiment.build_model()
    strategy = build_strategy(config.model.optimization)
    train = experiment.data.train_loader(shuffle=False)
    valid = experiment.data.valid_loader()
    history = experiment.trainer.fit(model, strategy, train, valid)
    losses = [report.train.loss for report in history]
    assert all(np.isfinite(losses)), losses
    result = evaluate_model(model, valid, device=args.device, train_loader=train)
    assert np.isfinite(result.metrics["co_bps"]), result.metrics
    assert np.isfinite(result.predictions["rates"]).all()
    if method == "mint":
        assert model.Omega_plus and model.library_training["trials"] == 4
        if dataset == "mc_rtt":
            assert model.library_training["lfads_epochs"] == 1
            assert np.isfinite(model.library_training["lfads_final_loss"])

    checkpoint = folder / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    if hasattr(config.model, "build_from_data"):
        restored = config.model.build_from_data(experiment.data)
    else:
        restored = config.model.build(experiment.data.n_neurons, experiment.data.n_time)
    restored.to(args.device)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=args.device))
    query = observations_from_batch(next(iter(valid))).to(args.device)
    outputs = []
    for current in (model, restored):
        current.to(args.device).eval()
        torch.manual_seed(7)
        with torch.no_grad():
            outputs.append(current(query).count_rates(0.005))
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-5, atol=1e-7)
    if method in ("mint", "ilqr_vae"):
        parts = _collect_full_rate_parts(model=model, loader=valid,
                                        device=torch.device(args.device), dt=0.005,
                                        prediction_floor=1e-9)
        np.testing.assert_allclose(parts["rates_heldout"], result.predictions["rates"],
                                   rtol=1e-5, atol=1e-7)
    row = dict(dataset=dataset, model=method, device=args.device, source=str(source),
               steps=args.steps, start_bin=start_bin, train_trials=4, eval_trials=2, train_losses=losses,
               metrics=result.metrics, checkpoint_roundtrip=True,
               seconds=time.perf_counter() - start)
    (folder / "verification.json").write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row), flush=True)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--datasets", nargs="+", default=["mc_maze", "area2_bump", "mc_rtt", "dmfc_rsg"])
    parser.add_argument("--models", nargs="+", choices=["mint", "ilqr_vae", "bgpfa"], default=["mint", "ilqr_vae", "bgpfa"])
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [verify(dataset, method, args) for dataset in args.datasets for method in args.models]
    (args.output_dir / "verification.json").write_text(json.dumps(rows, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
