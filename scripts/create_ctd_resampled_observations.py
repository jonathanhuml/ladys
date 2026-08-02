#!/usr/bin/env python3
"""Create resampled-neuron CTD-style benchmark H5 files.

This derives synthetic observation/rate matrices from canonical CTD latent
trajectories by sampling a new latent-to-rate readout from the canonical readout
distribution, then drawing Poisson spike counts from those rates. It preserves
task latents, inputs, extra fields, dt, and the CTD key names used by LaDyS.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import yaml


LOWER = [
    "ctd_nbff",
    "ctd_multitask",
    "ctd_random_target",
    "ctd_chaotic_delayed_matching",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config-dir", default="configs/dataset")
    parser.add_argument("--output-config-dir", required=True)
    parser.add_argument("--output-data-dir", required=True)
    parser.add_argument("--datasets", nargs="+", default=["lower"])
    parser.add_argument("--total-trials", type=int, required=True)
    parser.add_argument("--train-trials", type=int, required=True)
    parser.add_argument("--valid-trials", type=int, required=True)
    parser.add_argument("--neurons", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--chunk-trials", type=int, default=4)
    parser.add_argument("--readout-noise-scale", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def expand_datasets(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if value == "lower":
            out.extend(LOWER)
        else:
            out.append(value)
    return out


def rel_to_repo(path: Path, repo: Path) -> str:
    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


def copy_if_present(src: h5py.File, dst: h5py.File, key: str, sl=None) -> None:
    if key not in src:
        return
    data = src[key][sl] if sl is not None else src[key][()]
    dst.create_dataset(key, data=data)


def make_readout(
    old_readout: np.ndarray,
    neurons: int,
    rng: np.random.Generator,
    noise_scale: float,
) -> np.ndarray:
    latent_dim, old_neurons = old_readout.shape
    idx = rng.integers(0, old_neurons, size=neurons)
    new = old_readout[:, idx].astype(np.float32, copy=True)
    per_latent_std = old_readout.std(axis=1, keepdims=True).astype(np.float32)
    new += rng.normal(0.0, noise_scale, size=(latent_dim, neurons)).astype(np.float32) * per_latent_std
    return new


def write_rates_and_spikes(
    src: h5py.File,
    dst: h5py.File,
    split: str,
    trials: int,
    readout: np.ndarray,
    rng: np.random.Generator,
    chunk_trials: int,
) -> None:
    lat_src = src[f"{split}_latents"]
    if trials > lat_src.shape[0]:
        raise ValueError(f"requested {trials} {split} trials but source has {lat_src.shape[0]}")

    time = int(lat_src.shape[1])
    neurons = int(readout.shape[1])
    chunk_neurons = min(512, neurons)
    activity = dst.create_dataset(
        f"{split}_activity",
        shape=(trials, time, neurons),
        dtype="float32",
        chunks=(1, time, chunk_neurons),
    )
    recon = dst.create_dataset(
        f"{split}_recon_data",
        shape=(trials, time, neurons),
        dtype="float32",
        chunks=(1, time, chunk_neurons),
    )
    for suffix in ["latents", "inputs", "extra", "inds"]:
        copy_if_present(src, dst, f"{split}_{suffix}", slice(0, trials))

    for start in range(0, trials, chunk_trials):
        stop = min(start + chunk_trials, trials)
        lat = lat_src[start:stop].astype(np.float32, copy=False)
        flat = lat.reshape(-1, lat.shape[-1])
        rates = np.exp(np.clip(flat @ readout, -20.0, 20.0))
        rates = rates.reshape(stop - start, time, neurons).astype(np.float32)
        spikes = rng.poisson(rates).astype(np.float32)
        activity[start:stop] = rates
        recon[start:stop] = spikes


def write_activity_resampled_rates_and_spikes(
    src: h5py.File,
    dst: h5py.File,
    split: str,
    trials: int,
    source_neurons: np.ndarray,
    rng: np.random.Generator,
) -> None:
    activity_src = src[f"{split}_activity"]
    if trials > activity_src.shape[0]:
        raise ValueError(f"requested {trials} {split} trials but source has {activity_src.shape[0]}")

    rates = activity_src[:trials][..., source_neurons].astype(np.float32, copy=False)
    spikes = rng.poisson(rates).astype(np.float32)
    dst.create_dataset(f"{split}_activity", data=rates)
    dst.create_dataset(f"{split}_recon_data", data=spikes)
    for suffix in ["latents", "inputs", "extra", "inds"]:
        copy_if_present(src, dst, f"{split}_{suffix}", slice(0, trials))


def main() -> None:
    args = parse_args()
    if args.train_trials + args.valid_trials != args.total_trials:
        raise SystemExit("train_trials + valid_trials must equal total_trials")
    if args.total_trials < 1 or args.neurons < 1:
        raise SystemExit("total_trials and neurons must be positive")

    repo = Path.cwd()
    source_config_dir = Path(args.source_config_dir)
    output_config_dir = Path(args.output_config_dir)
    output_data_dir = Path(args.output_data_dir)
    output_config_dir.mkdir(parents=True, exist_ok=True)
    output_data_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.total_trials}trials_{args.neurons}neurons"

    for ds_index, dataset_name in enumerate(expand_datasets(args.datasets)):
        src_config_path = source_config_dir / f"{dataset_name}.yaml"
        if not src_config_path.exists():
            raise FileNotFoundError(src_config_path)
        cfg = yaml.safe_load(src_config_path.read_text())
        src_h5 = repo / cfg["data_path"]
        task = dataset_name.removeprefix("ctd_")
        out_h5 = output_data_dir / f"{task}_{tag}.h5"

        if out_h5.exists() and not args.overwrite:
            print(f"exists: {out_h5}")
        else:
            if out_h5.exists():
                out_h5.unlink()
            rng = np.random.default_rng(args.seed + 1009 * ds_index)
            with h5py.File(src_h5, "r") as src, h5py.File(out_h5, "w") as dst:
                source_neurons = rng.integers(0, src["readout"].shape[1], size=args.neurons)
                if dataset_name == "ctd_chaotic_delayed_matching":
                    # CDM's canonical activity is stored on the firing-rate scale, but
                    # its readout does not directly parameterize exp(latents @ readout).
                    # Resampling activity channels preserves the original rate units.
                    readout = src["readout"][()][:, source_neurons].astype(np.float32, copy=True)
                    dst.create_dataset("source_neurons", data=source_neurons.astype(np.int64))
                    dst.create_dataset("readout", data=readout.astype(np.float32))
                    dst.create_dataset("perm_neurons", data=np.arange(args.neurons, dtype=np.int64))
                    copy_if_present(src, dst, "orig_mean")
                    copy_if_present(src, dst, "orig_std")
                    write_activity_resampled_rates_and_spikes(
                        src, dst, "train", args.train_trials, source_neurons, rng
                    )
                    write_activity_resampled_rates_and_spikes(
                        src, dst, "valid", args.valid_trials, source_neurons, rng
                    )
                else:
                    readout = make_readout(
                        src["readout"][()],
                        args.neurons,
                        rng,
                        args.readout_noise_scale,
                    )
                    dst.create_dataset("readout", data=readout.astype(np.float32))
                    dst.create_dataset("perm_neurons", data=np.arange(args.neurons, dtype=np.int64))
                    copy_if_present(src, dst, "orig_mean")
                    copy_if_present(src, dst, "orig_std")
                    write_rates_and_spikes(src, dst, "train", args.train_trials, readout, rng, args.chunk_trials)
                    write_rates_and_spikes(src, dst, "valid", args.valid_trials, readout, rng, args.chunk_trials)
            print(f"wrote: {out_h5}")

        old_meta = dict(cfg.get("metadata", {}))
        old_total = int(old_meta.get("total_neurons", args.neurons))
        old_heldin = int(old_meta.get("n_neurons_heldin", old_total))
        heldin = max(1, min(args.neurons - 1, round(args.neurons * old_heldin / max(old_total, 1))))
        if dataset_name == "ctd_chaotic_delayed_matching":
            generation_note = (
                f"{args.total_trials}-trial/{args.neurons}-neuron derived CTD dataset; "
                "rates sampled with replacement from canonical activity channels, "
                "spikes~Poisson(rates)"
            )
        else:
            generation_note = (
                f"{args.total_trials}-trial/{args.neurons}-neuron derived CTD dataset; "
                "rates=exp(latents @ sampled_readout), spikes~Poisson(rates)"
            )
        out_cfg = {
            "name": dataset_name,
            "task": cfg.get("task", task),
            "data_path": rel_to_repo(out_h5, repo),
            "dt": cfg.get("dt", 1.0),
            "metadata": {
                "n_neurons_heldin": int(heldin),
                "n_neurons_heldout": int(args.neurons - heldin),
                "total_neurons": int(args.neurons),
                "total_trials": int(args.total_trials),
                "train_trials": int(args.train_trials),
                "valid_trials": int(args.valid_trials),
                "num_steps": int(old_meta.get("num_steps", 0)),
                "latent_dim": int(old_meta.get("latent_dim", 0)),
                "source_data_path": cfg["data_path"],
                "generation_note": generation_note,
            },
        }
        out_cfg_path = output_config_dir / f"{dataset_name}.yaml"
        out_cfg_path.write_text(yaml.safe_dump(out_cfg, sort_keys=False))
        print(f"wrote: {out_cfg_path}")


if __name__ == "__main__":
    main()
