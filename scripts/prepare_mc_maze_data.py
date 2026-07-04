#!/usr/bin/env python3
"""Prepare local MC_Maze validation data for LaDyS.

The fast path copies an existing NLB-style H5 file. If that is unavailable,
the script can build the validation tensors from NWB files using nlb_tools.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import h5py


DANDI_MC_MAZE = "https://dandiarchive.org/dandiset/000128"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/real/mc_maze_val.h5"))
    parser.add_argument(
        "--local-source",
        type=Path,
        default=Path("/Users/jonathanhuml/Desktop/STNDT/data/mc_maze_val.h5"),
        help="Existing NLB-style validation H5 to copy if present.",
    )
    parser.add_argument(
        "--nwb-dir",
        type=Path,
        default=Path("data/real/dandi/000128/sub-Jenkins"),
        help="Directory containing downloaded MC_Maze NWB files.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Use the dandi CLI to download MC_Maze NWB data if --nwb-dir is missing.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        _validate_h5(args.output)
        print(f"exists: {args.output}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.local_source.exists():
        shutil.copy2(args.local_source, args.output)
        _validate_h5(args.output)
        print(f"copied: {args.local_source} -> {args.output}")
        return 0

    if args.download and not args.nwb_dir.exists():
        _download_dandi(args.nwb_dir)

    if args.nwb_dir.exists():
        _build_from_nwb(args.nwb_dir, args.output)
        _validate_h5(args.output)
        print(f"built from NWB: {args.output}")
        return 0

    print(
        "MC_Maze data not found. Provide --local-source, place NWB files at "
        f"{args.nwb_dir}, or rerun with --download.",
        file=sys.stderr,
    )
    return 1


def _download_dandi(nwb_dir: Path) -> None:
    root = nwb_dir.parents[1]
    root.mkdir(parents=True, exist_ok=True)
    if shutil.which("dandi") is None:
        raise RuntimeError("dandi CLI not found. Install dependencies from requirements.txt.")
    subprocess.run(["dandi", "download", DANDI_MC_MAZE, "-o", str(root)], check=True)


def _build_from_nwb(nwb_dir: Path, output: Path) -> None:
    try:
        from nlb_tools.make_tensors import (
            combine_h5,
            make_eval_input_tensors,
            make_eval_target_tensors,
        )
        from nlb_tools.nwb_interface import NWBDataset
    except ImportError as exc:
        raise RuntimeError("nlb_tools is required to build MC_Maze tensors from NWB.") from exc

    dataset = NWBDataset(str(nwb_dir))
    dataset.resample(5)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        input_h5 = tmp_dir / "mc_maze_val_input.h5"
        target_h5 = tmp_dir / "mc_maze_val_target.h5"
        make_eval_input_tensors(
            dataset,
            dataset_name="mc_maze",
            trial_split="val",
            save_file=True,
            save_path=str(input_h5),
        )
        make_eval_target_tensors(
            dataset,
            dataset_name="mc_maze",
            train_trial_split="train",
            eval_trial_split="val",
            include_psth=True,
            save_file=True,
            save_path=str(target_h5),
        )
        combine_h5([str(input_h5), str(target_h5)], save_path=str(output))


def _validate_h5(path: Path) -> None:
    with h5py.File(path, "r") as handle:
        group = handle["mc_maze"] if "mc_maze" in handle else handle
        heldin = group["eval_spikes_heldin"]
        heldout = group["eval_spikes_heldout"]
        if heldin.shape[:2] != heldout.shape[:2]:
            raise ValueError(
                f"held-in shape {heldin.shape} is incompatible with held-out {heldout.shape}"
            )
        print(f"eval_spikes_heldin: {heldin.shape}")
        print(f"eval_spikes_heldout: {heldout.shape}")


if __name__ == "__main__":
    raise SystemExit(main())
