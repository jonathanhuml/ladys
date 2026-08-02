#!/usr/bin/env python3
"""Prepare low-trial Allen Visual Coding Neuropixels H5 datasets.

The seven benchmark configs share three raw NWB sessions. This script can
download those NWBs directly from Allen's public WellKnownFile endpoint and
write a compact LaDyS-ready H5 with held-in/held-out neuron splits.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Iterable, Literal
import urllib.parse
import urllib.request

import h5py
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO


ALLEN_API = "http://api.brain-map.org"
DEFAULT_RAW_DIR = Path("data") / "real" / "allen_vcn" / "raw"
DEFAULT_OUTPUT = Path("data") / "real" / "allen_vcn" / "allen_vcn_low_trial_20ms.h5"


@dataclass(frozen=True)
class SessionSpec:
    session_id: int
    well_known_file_id: int
    session_type: str
    default_units: int
    sdk_relaxed_units: int
    raw_api_units: int

    @property
    def nwb_name(self) -> str:
        return f"ecephys_session_{self.session_id}.nwb"

    @property
    def download_url(self) -> str:
        return f"{ALLEN_API}/api/v2/well_known_file_download/{self.well_known_file_id}"


@dataclass(frozen=True)
class ConfigSpec:
    name: str
    session_id: int
    stimulus_name: str
    block: str
    conditions: str
    repeats: str
    expected_trials: int
    selection: Literal["flashes", "drifting_one_tf", "movie_repeats"]
    window_start_s: float = 0.0
    window_stop_s: float | None = None
    movie_clip_repeats: int | None = None


SESSIONS: dict[int, SessionSpec] = {
    794812542: SessionSpec(
        session_id=794812542,
        well_known_file_id=1026124759,
        session_type="functional_connectivity",
        default_units=1005,
        sdk_relaxed_units=2137,
        raw_api_units=2680,
    ),
    757216464: SessionSpec(
        session_id=757216464,
        well_known_file_id=1026124603,
        session_type="brain_observatory_1.1",
        default_units=959,
        sdk_relaxed_units=2111,
        raw_api_units=2800,
    ),
    771160300: SessionSpec(
        session_id=771160300,
        well_known_file_id=1026124918,
        session_type="functional_connectivity",
        default_units=930,
        sdk_relaxed_units=2222,
        raw_api_units=2890,
    ),
}

CONFIGS: tuple[ConfigSpec, ...] = (
    ConfigSpec(
        name="flash_fc_top",
        session_id=794812542,
        stimulus_name="flashes",
        block="flashes",
        conditions="2 flash colors",
        repeats="75 per color",
        expected_trials=150,
        selection="flashes",
        window_stop_s=0.5,
    ),
    ConfigSpec(
        name="flash_bo_top",
        session_id=757216464,
        stimulus_name="flashes",
        block="flashes",
        conditions="2 flash colors",
        repeats="75 per color",
        expected_trials=150,
        selection="flashes",
        window_stop_s=0.5,
    ),
    ConfigSpec(
        name="flash_fc_second",
        session_id=771160300,
        stimulus_name="flashes",
        block="flashes",
        conditions="2 flash colors",
        repeats="75 per color",
        expected_trials=150,
        selection="flashes",
        window_stop_s=0.5,
    ),
    ConfigSpec(
        name="bo_drifting_one_tf",
        session_id=757216464,
        stimulus_name="drifting_gratings",
        block="drifting gratings, one temporal-frequency slice",
        conditions="8 directions",
        repeats="15 per direction",
        expected_trials=120,
        selection="drifting_one_tf",
        window_stop_s=2.0,
    ),
    ConfigSpec(
        name="fc_movie_one",
        session_id=794812542,
        stimulus_name="natural_movie_one",
        block="natural movie one",
        conditions="1 movie clip",
        repeats="60 clip repeats",
        expected_trials=60,
        selection="movie_repeats",
        movie_clip_repeats=60,
    ),
    ConfigSpec(
        name="bo_movie_one",
        session_id=757216464,
        stimulus_name="natural_movie_one",
        block="natural movie one",
        conditions="1 movie clip",
        repeats="20 clip repeats",
        expected_trials=20,
        selection="movie_repeats",
        movie_clip_repeats=20,
    ),
    ConfigSpec(
        name="bo_movie_three",
        session_id=757216464,
        stimulus_name="natural_movie_three",
        block="natural movie three",
        conditions="1 movie clip",
        repeats="10 clip repeats",
        expected_trials=10,
        selection="movie_repeats",
        movie_clip_repeats=10,
    ),
)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw_dir = Path(args.raw_dir)
    output = Path(args.output)

    if args.download:
        download_sessions(raw_dir, overwrite=bool(args.overwrite_raw))
    if args.prepare:
        prepare_configs(
            raw_dir=raw_dir,
            output=output,
            config_names=args.configs,
            bin_size_ms=args.bin_size_ms,
            train_fraction=args.train_fraction,
            heldin_fraction=args.heldin_fraction,
            seed=args.seed,
            max_units=args.max_units,
            overwrite=bool(args.overwrite),
        )
    if not args.download and not args.prepare:
        print("Nothing to do. Pass --download, --prepare, or both.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[spec.name for spec in CONFIGS],
        choices=[spec.name for spec in CONFIGS],
        help="prepared config groups to write",
    )
    parser.add_argument("--download", action="store_true", help="download missing raw NWB sessions")
    parser.add_argument("--prepare", action="store_true", help="write prepared H5 groups")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing H5 groups")
    parser.add_argument("--overwrite-raw", action="store_true", help="restart raw NWB downloads")
    parser.add_argument("--bin-size-ms", type=float, default=20.0)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--heldin-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-units", type=int, help="optional cap on default-filtered units")
    return parser


def download_sessions(raw_dir: Path, overwrite: bool = False) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for spec in SESSIONS.values():
        path = raw_dir / spec.nwb_name
        if path.exists() and not overwrite:
            print(f"raw exists: {path}")
            continue
        if overwrite and path.exists():
            path.unlink()
        print(f"downloading {spec.session_id} -> {path}")
        subprocess.run(
            [
                "curl",
                "-L",
                "-C",
                "-",
                "--fail",
                "--retry",
                "5",
                "--retry-delay",
                "5",
                "-o",
                str(path),
                spec.download_url,
            ],
            check=True,
        )


def prepare_configs(
    *,
    raw_dir: Path,
    output: Path,
    config_names: list[str],
    bin_size_ms: float,
    train_fraction: float,
    heldin_fraction: float,
    seed: int,
    max_units: int | None,
    overwrite: bool,
) -> None:
    _validate_fraction(train_fraction, "train_fraction")
    _validate_fraction(heldin_fraction, "heldin_fraction")
    selected = [spec for spec in CONFIGS if spec.name in set(config_names)]
    if not selected:
        raise ValueError("No configs selected.")

    output.parent.mkdir(parents=True, exist_ok=True)
    sessions_by_id: dict[int, object] = {}
    handles: list[NWBHDF5IO] = []
    try:
        with h5py.File(output, "a") as h5:
            for spec in selected:
                if spec.name in h5 and not overwrite:
                    print(f"exists: {output}:{spec.name}")
                    continue
                if spec.name in h5:
                    del h5[spec.name]

                nwb, io_handle = _get_session(raw_dir, spec.session_id, sessions_by_id, handles)
                if io_handle is not None:
                    handles.append(io_handle)
                print(f"preparing {spec.name} from session {spec.session_id}")
                units = _default_filtered_units(nwb, max_units=max_units)
                stimulus_table = _stimulus_table(nwb)
                trial_windows, condition_ids, selection_metadata = _select_trials(
                    stimulus_table,
                    spec,
                    bin_size_ms=bin_size_ms,
                )
                spikes = _bin_spikes(units["spike_times"], trial_windows, bin_size_ms / 1000.0)
                _write_group(
                    h5,
                    spec,
                    session=SESSIONS[spec.session_id],
                    spikes=spikes,
                    condition_ids=condition_ids,
                    unit_ids=units["unit_ids"],
                    trial_windows=trial_windows,
                    bin_size_ms=bin_size_ms,
                    train_fraction=train_fraction,
                    heldin_fraction=heldin_fraction,
                    seed=seed,
                    metadata=selection_metadata,
                )
                print(
                    f"wrote {spec.name}: spikes={spikes.shape} "
                    f"heldin={int(round(spikes.shape[-1] * heldin_fraction))}"
                )
    finally:
        for io_handle in handles:
            io_handle.close()


def _get_session(
    raw_dir: Path,
    session_id: int,
    sessions_by_id: dict[int, object],
    handles: list[NWBHDF5IO],
) -> tuple[object, NWBHDF5IO | None]:
    del handles
    if session_id in sessions_by_id:
        return sessions_by_id[session_id], None
    path = raw_dir / SESSIONS[session_id].nwb_name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing raw NWB for session {session_id}: {path}. "
            "Run this script with --download first."
        )
    io_handle = NWBHDF5IO(str(path), "r", load_namespaces=True)
    nwb = io_handle.read()
    sessions_by_id[session_id] = nwb
    return nwb, io_handle


def _default_filtered_units(nwb, max_units: int | None = None) -> dict[str, list]:
    if nwb.units is None:
        raise ValueError("NWB file has no units table.")
    units = nwb.units.to_dataframe()
    required = ["amplitude_cutoff", "presence_ratio", "isi_violations", "quality", "spike_times"]
    missing = [key for key in required if key not in units.columns]
    if missing:
        raise KeyError(f"Units table is missing required columns: {missing}")

    mask = (
        (units["amplitude_cutoff"] <= 0.1)
        & (units["presence_ratio"] >= 0.95)
        & (units["isi_violations"] <= 0.5)
        & (units["quality"] == "good")
    )
    filtered = units.loc[mask].copy()
    if "peak_channel_id" in filtered.columns and nwb.electrodes is not None:
        electrodes = nwb.electrodes.to_dataframe()
        valid_peak_channel = []
        for peak_channel_id in filtered["peak_channel_id"]:
            if peak_channel_id not in electrodes.index:
                valid_peak_channel.append(False)
                continue
            valid_peak_channel.append(bool(electrodes.loc[peak_channel_id].get("valid_data", True)))
        filtered = filtered.loc[np.asarray(valid_peak_channel, dtype=bool)]
    if "ecephys_structure_id" in filtered.columns:
        filtered = filtered.loc[~filtered["ecephys_structure_id"].isna()]
    filtered = filtered.sort_index()
    if max_units is not None:
        filtered = filtered.iloc[:max_units]
    if len(filtered) == 0:
        raise ValueError("No units passed default Allen VCN filters.")
    return {
        "unit_ids": [int(item) for item in filtered.index.to_numpy()],
        "spike_times": [np.asarray(item, dtype=np.float64) for item in filtered["spike_times"]],
    }


def _stimulus_table(nwb):
    if "stimulus_presentations" in nwb.intervals:
        table = nwb.intervals["stimulus_presentations"].to_dataframe()
        table = table.copy()
        table["stimulus_name_raw"] = table["stimulus_name"]
        table["stimulus_name"] = table["stimulus_name"].map(_canonical_stimulus_name)
        return table
    if "trials" in nwb.intervals:
        table = nwb.intervals["trials"].to_dataframe()
        if "stimulus_name" in table.columns:
            table = table.copy()
            table["stimulus_name_raw"] = table["stimulus_name"]
            table["stimulus_name"] = table["stimulus_name"].map(_canonical_stimulus_name)
            return table
    frames = []
    for interval_name, interval in nwb.intervals.items():
        if not interval_name.endswith("_presentations"):
            continue
        frame = interval.to_dataframe().copy()
        if "stimulus_name" not in frame.columns:
            frame["stimulus_name"] = interval_name.removesuffix("_presentations")
        frame["stimulus_interval_name"] = interval_name
        frame["stimulus_name_raw"] = frame["stimulus_name"]
        frame["stimulus_name"] = frame["stimulus_name"].map(_canonical_stimulus_name)
        frames.append(frame)
    if frames:
        return pd.concat(frames, axis=0, ignore_index=True, sort=False)
    available = ", ".join(nwb.intervals.keys())
    raise KeyError(f"No stimulus presentation interval tables found. Intervals: {available}")


def _canonical_stimulus_name(name: object) -> str:
    text = str(name)
    aliases = {
        "natural_movie_one_more_repeats": "natural_movie_one",
        "natural_movie_1_more_repeats": "natural_movie_one",
        "natural_movie_1": "natural_movie_one",
        "natural_movie_3": "natural_movie_three",
        "Natural Images": "natural_scenes",
        "flash": "flashes",
    }
    return aliases.get(text, text)


def _select_trials(table, spec: ConfigSpec, bin_size_ms: float):
    stim = table.loc[table["stimulus_name"] == spec.stimulus_name].copy()
    if len(stim) == 0 and spec.selection == "flashes":
        stim = table.loc[table["stimulus_name"].isin(["flashes", "flash"])].copy()
    if len(stim) == 0:
        names = ", ".join(str(item) for item in sorted(table["stimulus_name"].dropna().unique()))
        raise ValueError(
            f"Session {spec.session_id} has no rows for '{spec.stimulus_name}'. "
            f"Available stimulus names: {names}"
        )

    metadata: dict[str, object] = {}
    if spec.selection == "flashes":
        rows = stim.sort_values("start_time")
        condition_ids = _condition_ids(rows, preferred_columns=["color"])
        trial_windows = _fixed_windows(rows, spec.window_start_s, spec.window_stop_s)
        metadata["observed_trials"] = int(len(rows))
        return trial_windows, condition_ids, metadata

    if spec.selection == "drifting_one_tf":
        if "temporal_frequency" not in stim.columns:
            raise KeyError("drifting grating stimulus table is missing temporal_frequency")
        rows = stim.loc[stim["temporal_frequency"].astype(str) != "null"].copy()
        rows = rows.loc[rows["temporal_frequency"].notna()]
        tf_counts = rows["temporal_frequency"].value_counts()
        if len(tf_counts) == 0:
            raise ValueError("No non-null temporal frequencies in drifting gratings.")
        top_count = tf_counts.max()
        top_values = sorted(tf_counts.loc[tf_counts == top_count].index.tolist())
        selected_tf = top_values[0]
        rows = rows.loc[rows["temporal_frequency"] == selected_tf].sort_values("start_time")
        condition_ids = _condition_ids(rows, preferred_columns=["orientation"])
        trial_windows = _fixed_windows(rows, spec.window_start_s, spec.window_stop_s)
        metadata["selected_temporal_frequency"] = float(selected_tf)
        metadata["observed_trials"] = int(len(rows))
        return trial_windows, condition_ids, metadata

    if spec.selection == "movie_repeats":
        rows = stim.sort_values("start_time")
        repeat_windows = _movie_repeat_windows(rows)
        if spec.movie_clip_repeats is not None and len(repeat_windows) > spec.movie_clip_repeats:
            repeat_windows = repeat_windows[: spec.movie_clip_repeats]
        condition_ids = np.zeros(len(repeat_windows), dtype=np.int64)
        metadata["observed_trials"] = int(len(repeat_windows))
        return repeat_windows, condition_ids, metadata

    raise ValueError(f"Unknown selection mode {spec.selection}")


def _fixed_windows(rows, start_offset_s: float, stop_offset_s: float | None) -> np.ndarray:
    starts = rows["start_time"].to_numpy(dtype=np.float64) + float(start_offset_s)
    if stop_offset_s is None:
        stops = rows["stop_time"].to_numpy(dtype=np.float64)
    else:
        stops = rows["start_time"].to_numpy(dtype=np.float64) + float(stop_offset_s)
    windows = np.stack([starts, stops], axis=1)
    if np.any(windows[:, 1] <= windows[:, 0]):
        raise ValueError("Encountered non-positive trial window duration.")
    return windows


def _movie_repeat_windows(rows) -> np.ndarray:
    if "frame" not in rows.columns:
        return _single_or_gap_windows(rows)
    frames = rows["frame"].to_numpy()
    starts = rows["start_time"].to_numpy(dtype=np.float64)
    stops = rows["stop_time"].to_numpy(dtype=np.float64)
    breakpoints = [0]
    for index in range(1, len(rows)):
        frame_reset = frames[index] <= frames[index - 1]
        gap = starts[index] - stops[index - 1] > 1.0
        if bool(frame_reset) or bool(gap):
            breakpoints.append(index)
    breakpoints.append(len(rows))
    windows = []
    for left, right in zip(breakpoints[:-1], breakpoints[1:]):
        if right <= left:
            continue
        windows.append((starts[left], stops[right - 1]))
    return np.asarray(windows, dtype=np.float64)


def _single_or_gap_windows(rows) -> np.ndarray:
    starts = rows["start_time"].to_numpy(dtype=np.float64)
    stops = rows["stop_time"].to_numpy(dtype=np.float64)
    if len(rows) == 1:
        return np.asarray([[starts[0], stops[0]]], dtype=np.float64)
    gaps = np.flatnonzero(starts[1:] - stops[:-1] > 1.0) + 1
    edges = np.concatenate([[0], gaps, [len(rows)]])
    return np.asarray([(starts[left], stops[right - 1]) for left, right in zip(edges[:-1], edges[1:])])


def _condition_ids(rows, preferred_columns: list[str]) -> np.ndarray:
    columns = [column for column in preferred_columns if column in rows.columns]
    if not columns:
        return np.zeros(len(rows), dtype=np.int64)
    values = rows[columns].astype(str).agg("|".join, axis=1)
    labels = {value: index for index, value in enumerate(sorted(values.unique()))}
    return values.map(labels).to_numpy(dtype=np.int64)


def _bin_spikes(
    spike_times_by_unit: list[np.ndarray],
    trial_windows: np.ndarray,
    bin_size_s: float,
) -> np.ndarray:
    duration = float(np.min(trial_windows[:, 1] - trial_windows[:, 0]))
    n_bins = max(int(np.floor(duration / bin_size_s)), 1)
    relative_edges = np.arange(n_bins + 1, dtype=np.float64) * bin_size_s
    spikes = np.zeros((len(trial_windows), n_bins, len(spike_times_by_unit)), dtype=np.uint16)
    for unit_index, spike_times in enumerate(spike_times_by_unit):
        spike_times = np.asarray(spike_times, dtype=np.float64)
        for trial_index, (start, _stop) in enumerate(trial_windows):
            edges = start + relative_edges
            counts = np.diff(np.searchsorted(spike_times, edges, side="left"))
            spikes[trial_index, :, unit_index] = np.minimum(counts, np.iinfo(np.uint16).max)
    return spikes


def _write_group(
    h5: h5py.File,
    spec: ConfigSpec,
    *,
    session: SessionSpec,
    spikes: np.ndarray,
    condition_ids: np.ndarray,
    unit_ids: list[int],
    trial_windows: np.ndarray,
    bin_size_ms: float,
    train_fraction: float,
    heldin_fraction: float,
    seed: int,
    metadata: dict[str, object],
) -> None:
    rng = np.random.default_rng(seed)
    trial_order = rng.permutation(spikes.shape[0])
    n_train = max(1, min(spikes.shape[0] - 1, int(round(spikes.shape[0] * train_fraction))))
    train_trials = np.sort(trial_order[:n_train])
    eval_trials = np.sort(trial_order[n_train:])
    if len(eval_trials) == 0:
        raise ValueError(f"{spec.name} has no eval trials after train/eval split.")

    neuron_order = rng.permutation(spikes.shape[-1])
    n_heldin = max(1, min(spikes.shape[-1] - 1, int(round(spikes.shape[-1] * heldin_fraction))))
    heldin_idx = np.sort(neuron_order[:n_heldin])
    heldout_idx = np.sort(neuron_order[n_heldin:])

    group = h5.create_group(spec.name)
    group.create_dataset("train_spikes_full", data=spikes[train_trials], compression="gzip")
    group.create_dataset("eval_spikes_full", data=spikes[eval_trials], compression="gzip")
    group.create_dataset("train_spikes_heldin", data=spikes[train_trials][:, :, heldin_idx], compression="gzip")
    group.create_dataset("train_spikes_heldout", data=spikes[train_trials][:, :, heldout_idx], compression="gzip")
    group.create_dataset("eval_spikes_heldin", data=spikes[eval_trials][:, :, heldin_idx], compression="gzip")
    group.create_dataset("eval_spikes_heldout", data=spikes[eval_trials][:, :, heldout_idx], compression="gzip")
    group.create_dataset("train_condition_ids", data=condition_ids[train_trials], compression="gzip")
    group.create_dataset("eval_condition_ids", data=condition_ids[eval_trials], compression="gzip")
    group.create_dataset("unit_ids", data=np.asarray(unit_ids, dtype=np.int64))
    group.create_dataset("heldin_unit_indices", data=heldin_idx.astype(np.int64))
    group.create_dataset("heldout_unit_indices", data=heldout_idx.astype(np.int64))
    group.create_dataset("trial_windows", data=trial_windows)
    group.create_dataset("train_trial_indices", data=train_trials.astype(np.int64))
    group.create_dataset("eval_trial_indices", data=eval_trials.astype(np.int64))

    attrs = {
        "config": spec.name,
        "session_id": spec.session_id,
        "session_type": session.session_type,
        "stimulus_name": spec.stimulus_name,
        "block": spec.block,
        "conditions": spec.conditions,
        "repeats": spec.repeats,
        "expected_trials": spec.expected_trials,
        "default_units_metadata": session.default_units,
        "sdk_relaxed_units_metadata": session.sdk_relaxed_units,
        "raw_api_units_metadata": session.raw_api_units,
        "n_units_prepared": spikes.shape[-1],
        "n_trials_prepared": spikes.shape[0],
        "bin_size_ms": float(bin_size_ms),
        "bin_size_s": float(bin_size_ms) / 1000.0,
        "train_fraction": float(train_fraction),
        "heldin_fraction": float(heldin_fraction),
        "seed": int(seed),
        "selection_metadata_json": json.dumps(metadata, sort_keys=True),
    }
    for key, value in attrs.items():
        group.attrs[key] = value


def _validate_fraction(value: float, name: str) -> None:
    if not (0.0 < value < 1.0):
        raise ValueError(f"{name} must be in (0, 1).")


def query_download_links() -> dict[int, str]:
    """Return current Allen download URLs from the public API."""

    links = {}
    for session_id in SESSIONS:
        criteria = (
            "model::WellKnownFile,"
            "rma::criteria"
            "[attachable_type$eq'EcephysSession']"
            f"[attachable_id$eq{session_id}],"
            "well_known_file_type,"
            "rma::options[num_rows$eqall]"
        )
        safe_chars = ",:$[]='()"
        encoded_criteria = urllib.parse.quote(criteria, safe=safe_chars)
        url = f"{ALLEN_API}/api/v2/data/query.json?criteria={encoded_criteria}"
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not payload.get("success") or not payload["msg"]:
            raise RuntimeError(f"No WellKnownFile returned for {session_id}: {payload}")
        links[session_id] = ALLEN_API + payload["msg"][0]["download_link"]
    return links


if __name__ == "__main__":
    raise SystemExit(main())
