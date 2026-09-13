#!/usr/bin/env python3
"""Run full-data, from-scratch 5 ms NLB validation comparisons."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

from ladys.config import load_experiment_config
from ladys.experiment import (
    Experiment, _write_history, _write_json, _write_predictions,
    experiment_config_to_dict,
)
from ladys.metrics import evaluate_model
from ladys.nlb_eval import nlb_bits_per_spike
from ladys.training.strategies import build_strategy
from ladys.training import EpochReport
from ladys.types import StepResult


METHODS = ("mint", "bgpfa", "ilqr_vae")
DATASETS = ("area2_bump", "mc_maze", "dmfc_rsg", "mc_rtt")


class ProgressLoader:
    def __init__(self, loader, callback, phase, before_first_iteration=None):
        self.loader, self.callback, self.phase = loader, callback, phase
        self.before_first_iteration = before_first_iteration

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        if self.before_first_iteration is not None:
            restore = self.before_first_iteration
            self.before_first_iteration = None
            restore()
        started = last = time.monotonic()
        for index, batch in enumerate(self.loader, 1):
            yield batch
            now = time.monotonic()
            if index == 1 or index == len(self) or now - last >= 30:
                self.callback(phase=self.phase, batch=index, batches=len(self),
                              phase_seconds=now - started)
                last = now


def config_path(dataset, method):
    folder = Path("configs/experiment/real") / dataset / method
    canonical = folder / f"{method}_{dataset}_nlb_5ms.yaml"
    if canonical.exists():
        return canonical
    return folder / f"{method}_{dataset}_nlb_5ms_train.yaml"


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def write_status(path, state):
    temporary = path.with_suffix(".tmp")
    _write_json(temporary, state)
    temporary.replace(path)


def existing_run(folder):
    path = folder / "status.json"
    if path.exists():
        return json.loads(path.read_text())
    artifacts = ("config.json", "latest_model.pt", "training_state.pt", "model.pt", "best_model.pt")
    return {"status": "incomplete"} if any((folder / name).exists() for name in artifacts) else None


def assert_inactive(state, method, dataset):
    pid = state.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    except PermissionError:
        raise RuntimeError(f"Cannot establish whether recorded worker PID {pid} is inactive.")
    process = subprocess.run(["ps", "-p", str(pid), "-o", "args="], capture_output=True, text=True)
    command = process.stdout.strip()
    if process.returncode != 0 or not command:
        raise RuntimeError(f"Cannot inspect live recorded worker PID {pid}; refusing to resume.")
    tokens = shlex.split(command)
    is_runner = any(Path(token).name == Path(__file__).name for token in tokens)
    if is_runner and "--worker" in tokens:
        index = tokens.index("--worker")
        if tokens[index + 1:index + 3] == [method, dataset]:
            raise RuntimeError(f"Worker PID {pid} is still active for {method}/{dataset}.")


def comparable_config(payload):
    result = deepcopy(payload)
    for key in ("device", "epochs"):
        result.get("trainer", {}).pop(key, None)
    result.get("experiment", {}).pop("output_dir", None)
    return result


def validate_resume_config(saved, current):
    if comparable_config(saved) != comparable_config(current):
        raise ValueError(
            "Resume configuration differs from the saved run; only device, total epochs, "
            "and output directory may change."
        )


def capture_rng(device):
    numpy_state = np.random.get_state()
    return dict(
        rng_state=torch.get_rng_state(),
        cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        numpy_rng_state=[numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        python_rng_state=random.getstate(), device=str(device),
    )


def restore_rng(state, device):
    torch.set_rng_state(state["rng_state"].cpu())
    cuda_states = state.get("cuda_rng_state", [])
    if cuda_states and str(device).startswith("cuda"):
        saved_device = torch.device(state.get("device", "cuda"))
        index = saved_device.index or 0
        torch.cuda.set_rng_state(cuda_states[min(index, len(cuda_states) - 1)].cpu(), device=device)
    if "numpy_rng_state" in state:
        name, values, position, has_gauss, cached = state["numpy_rng_state"]
        np.random.set_state((name, np.asarray(values, dtype=np.uint32), position, has_gauss, cached))
    if "python_rng_state" in state:
        random.setstate(state["python_rng_state"])


def load_resume_checkpoint(folder, payload):
    saved_config_path = folder / "config.json"
    state_path = folder / "training_state.pt"
    if not saved_config_path.exists() or not state_path.exists():
        raise FileNotFoundError(f"Resume requires config.json and training_state.pt in {folder}.")
    validate_resume_config(json.loads(saved_config_path.read_text()), payload)
    # Legacy sqrt-decay schedulers stored NumPy float64 learning rates.
    numpy_scalar = np.float64(0).__reduce__()[0]
    numeric_globals = [
        (numpy_scalar, "numpy.core.multiarray.scalar"),
        (numpy_scalar, "numpy._core.multiarray.scalar"),
        np.dtype, type(np.dtype("float64")),
    ]
    with torch.serialization.safe_globals(numeric_globals):
        state = torch.load(state_path, map_location="cpu", weights_only=True)
    if "config" in state:
        validate_resume_config(state["config"], payload)
    epoch = state.get("epoch")
    if not isinstance(epoch, int) or epoch < 0 or epoch > payload["trainer"]["epochs"]:
        raise ValueError("Saved checkpoint epoch must lie within the requested total training budget.")
    if epoch > 0 and not state.get("strategy"):
        raise ValueError("Cannot resume trained epochs with empty strategy state; this strategy is not resumable.")
    if "rng_state" not in state:
        raise ValueError("Resume checkpoint is missing its training RNG state.")
    if "model" not in state:
        model_path = folder / "latest_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Legacy resume requires {model_path}.")
        if model_path.stat().st_mtime_ns > state_path.stat().st_mtime_ns:
            raise ValueError("Legacy model checkpoint is newer than training_state.pt; the pair may be inconsistent.")
        state["model"] = torch.load(model_path, map_location="cpu", weights_only=True)
    return state


def load_history(path, through_epoch):
    if not path.exists():
        return []
    reports = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            epoch = int(row["epoch"])
            if epoch > through_epoch:
                continue
            train = StepResult(loss=float(row["train_loss"]), batch_size=0,
                               objective=row["objective"], metrics=json.loads(row["train_metrics"]))
            valid = None if not row["valid_loss"] else StepResult(
                loss=float(row["valid_loss"]), batch_size=0, objective=row["objective"],
                metrics=json.loads(row["valid_metrics"]),
            )
            reports.append(EpochReport(epoch=epoch - 1, seconds=float(row["seconds"]),
                                       train=train, valid=valid, metrics=json.loads(row["metrics"])))
    return reports


def run_one(args):
    method, dataset = args.worker
    folder = args.output_dir / dataset / method
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    elapsed_before = 0.0
    claimed = False
    lock = None
    state = dict(dataset=dataset, model=method, split="val", bin_size_ms=5,
                 status="running", phase="setup", device=args.device,
                 seed=args.seed, pid=os.getpid(), epoch=0)

    def status(**updates):
        state.update(updates, seconds=elapsed_before + time.monotonic() - started)
        write_status(folder / "status.json", state)
        print(json.dumps(state), flush=True)

    try:
        lock = (folder / ".worker.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = existing_run(folder)
        if previous is not None:
            if not getattr(args, "resume", False):
                raise FileExistsError(f"Refusing to overwrite a previous run: {folder}; use --resume.")
            if previous.get("status") == "complete":
                print(f"Already complete: {method}/{dataset}", flush=True)
                return 0
            assert_inactive(previous, method, dataset)
        else:
            claimed = True
            status()
        source_config = config_path(dataset, method)
        config = load_experiment_config(source_config)
        config.dataset.data_path = str((args.data_root / f"{dataset}_val_5ms.h5").resolve())
        config.dataset.split = "val"
        config.dataset.max_trials = None
        config.dataset.seed = args.seed
        if method == "mint":
            config.model.lfads_seed = args.seed
        elif method == "ilqr_vae":
            config.model.init_seed = args.seed
        config.trainer.device = args.device
        config.trainer.live_eval_interval = 0
        if args.epochs is not None:
            config.trainer.epochs = args.epochs
        if args.batch_size is not None:
            config.batch_size = args.batch_size
        config.output_dir = str(args.output_dir)
        config.run_name = f"{method}_{dataset}_val_5ms_seed{args.seed}"
        payload = experiment_config_to_dict(config)
        payload["trainer"]["batch_size"] = payload.pop("batch_size")
        resume_state = load_resume_checkpoint(folder, payload) if previous is not None else None
        start_epoch = resume_state["epoch"] if resume_state is not None else 0
        if previous is not None:
            elapsed_before = float(previous.get("seconds", 0.0))
            state.update(previous)
            state.pop("error", None)
            state.update(status="running", device=args.device, pid=os.getpid(),
                         epoch=start_epoch, resumed_from_epoch=start_epoch)
            claimed = True
            status(phase="resuming")
        _write_json(folder / "config.json", payload)
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        experiment = Experiment(config)
        experiment.data.setup()
        model = experiment.build_model()
        if resume_state is not None:
            model.load_state_dict(resume_state["model"])
            experiment.trainer.history = load_history(folder / "history.csv", start_epoch)
        strategy = build_strategy(config.model.optimization)
        data = experiment.data
        train = ProgressLoader(data.train_loader(
            shuffle=not strategy.requires_ordered_training_data), status, "training",
            before_first_iteration=(lambda: restore_rng(resume_state, args.device)) if resume_state is not None else None)
        valid = ProgressLoader(data.valid_loader(), status, "validation")
        # bGPFA's full training posterior already matches this exact ordered batch.
        readout_train = DataLoader(data.train_dataset, batch_size=len(data.train_dataset), shuffle=False) if method == "bgpfa" else data.train_loader(shuffle=False)
        readout_train = ProgressLoader(readout_train, status, "readout_fit")
        interval = args.eval_every or (50 if method == "bgpfa" else 5)
        status(train_trials=len(data.train_dataset), val_trials=len(data.valid_dataset),
               time_bins=data.n_time, input_neurons=data.n_neurons,
               epochs=config.trainer.epochs, eval_every=interval,
               source_config=str(source_config), phase="fitting")
        evaluations_path = folder / "evaluations.json"
        evaluations = json.loads(evaluations_path.read_text()) if resume_state is not None and evaluations_path.exists() else []
        evaluations = [row for row in evaluations if row["epoch"] <= start_epoch]
        best_score = -math.inf
        best_path = folder / "best_metrics.json"
        best_artifacts = [folder / name for name in ("best_model.pt", "best_metrics.json", "predictions.npz")]
        if resume_state is not None and all(path.exists() for path in best_artifacts):
            best = json.loads(best_path.read_text())
            if best["epoch"] > start_epoch:
                raise ValueError("Best checkpoint is newer than the resumable training checkpoint.")
            best_score = float(best["co_bps"])
            state.update(best_co_bps=best_score, best_epoch=best["epoch"])

        def checkpoint(epoch):
            # Commit model, optimizer, and RNG together; latest_model remains a convenience copy.
            saved = dict(epoch=epoch, model=model.state_dict(), strategy=strategy.state_dict(),
                         config=payload, **capture_rng(args.device))
            save_checkpoint(folder / "training_state.pt", saved)
            save_checkpoint(folder / "latest_model.pt", saved["model"])

        def evaluate(epoch):
            nonlocal best_score, evaluations
            status(phase="evaluation", epoch=epoch)
            eval_start = time.monotonic()
            # Isolate inference randomness from the subsequent training stream.
            devices = [torch.cuda.current_device()] if args.device.startswith("cuda") else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(args.seed + 10000)
                result = evaluate_model(model, valid, device=args.device, train_loader=readout_train)
            score = nlb_bits_per_spike(result.predictions["rates"], result.targets["spikes"])
            if not math.isfinite(score):
                raise RuntimeError(f"Nonfinite full-validation co-BPS: {result.metrics}")
            row = dict(epoch=epoch, co_bps=score, seconds=elapsed_before + time.monotonic() - started,
                       evaluation_seconds=time.monotonic() - eval_start, metrics=result.metrics)
            if score > best_score:
                best_score = score
                save_checkpoint(folder / "best_model.pt", model.state_dict())
                _write_predictions(folder / "predictions.npz", result)
                write_status(folder / "best_metrics.json", row)
                state.update(best_co_bps=score, best_epoch=epoch)
            evaluations = [item for item in evaluations if item["epoch"] != epoch] + [row]
            write_status(evaluations_path, evaluations)
            status(phase="evaluated", co_bps=score, evaluation_seconds=row["evaluation_seconds"])

        def on_epoch(report):
            epoch = report.epoch + 1
            if not math.isfinite(report.train.loss):
                raise RuntimeError(f"Nonfinite training loss at epoch {epoch}: {report.train.loss}")
            _write_history(folder / "history.csv", experiment.trainer.history)
            status(epoch=epoch, train_loss=report.train.loss,
                   epoch_seconds=report.seconds, phase="trained_epoch")
            if epoch == 1 or epoch % interval == 0 or epoch == config.trainer.epochs:
                checkpoint(epoch)
                evaluate(epoch)

        if resume_state is not None:
            saved_evaluation = next((row for row in evaluations if row["epoch"] == start_epoch), None)
            if saved_evaluation is None or best_score < saved_evaluation["co_bps"] or previous.get("phase") in {"evaluation", "validation", "readout_fit"}:
                evaluate(start_epoch)
        experiment.trainer.fit(model, strategy, train, valid_loader=None, epoch_callback=on_epoch,
                               start_epoch=start_epoch,
                               strategy_state=resume_state["strategy"] if resume_state is not None else None)
        if not evaluations:
            checkpoint(config.trainer.epochs)
            evaluate(config.trainer.epochs)
        save_checkpoint(folder / "model.pt", model.state_dict())
        if hasattr(model, "library_training"):
            _write_json(folder / "library_training.json", model.library_training)
        status(status="complete", phase="complete", epoch=config.trainer.epochs)
        return 0
    except Exception as exc:
        if claimed:
            status(status="failed", error=repr(exc))
            (folder / "error.txt").write_text(traceback.format_exc())
        traceback.print_exc()
        return 1
    finally:
        if lock is not None:
            lock.close()


def write_summary(output_dir, datasets, methods):
    rows = []
    for dataset in datasets:
        for method in methods:
            path = output_dir / dataset / method / "status.json"
            row = json.loads(path.read_text()) if path.exists() else dict(dataset=dataset, model=method, status="queued")
            rows.append(row)
    write_status(output_dir / "summary.json", rows)
    lines = ["# NLB 5 ms Validation", "", "Full training and validation splits; one seed. Best validation checkpoints.", "",
             "| Dataset | Model | Status | Epoch | Best co-BPS | Minutes |",
             "| --- | --- | --- | ---: | ---: | ---: |"]
    for row in rows:
        score = row.get("best_co_bps")
        value = "" if score is None else f"{score:.5f}"
        lines.append(f"| {row['dataset']} | {row['model']} | {row['status']} | {row.get('epoch', '')} | {value} | {row.get('seconds', 0) / 60:.1f} |")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, help="Optional explicit training budget override.")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--resume", action="store_true", help="Resume unfinished runs and skip completed runs.")
    parser.add_argument("--worker", nargs=2, metavar=("MODEL", "DATASET"), help=argparse.SUPPRESS)
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    if args.workers < 1 or args.threads < 1:
        parser.error("workers and threads must be positive")
    if args.epochs is not None and args.epochs < 0:
        parser.error("epochs must be nonnegative")
    if any(value is not None and value < 1 for value in (args.batch_size, args.eval_every)):
        parser.error("batch size and evaluation interval must be positive")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        return run_one(args)
    queues = {method: [] for method in args.models}
    for method in args.models:
        for dataset in args.datasets:
            folder = args.output_dir / dataset / method
            previous = existing_run(folder)
            if previous is not None:
                if not args.resume:
                    raise FileExistsError(f"Refusing to overwrite a previous run: {folder}; use --resume.")
                if previous.get("status") == "complete":
                    continue
                assert_inactive(previous, method, dataset)
            queues[method].append(dataset)
    active = {}
    while any(queues.values()) or active:
        for method in args.models:
            if method in active or not queues[method] or len(active) >= args.workers:
                continue
            dataset = queues[method].pop(0)
            folder = args.output_dir / dataset / method
            folder.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, __file__, *arguments, "--worker", method, dataset]
            log = (folder / "run.log").open("a" if args.resume else "w")
            active[method] = (subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log)
            print(f"Started {method}/{dataset}", flush=True)
        for method, (process, log) in list(active.items()):
            if process.poll() is not None:
                log.close()
                print(f"Finished {method}: exit={process.returncode}", flush=True)
                del active[method]
        write_summary(args.output_dir, args.datasets, args.models)
        if active or any(queues.values()):
            time.sleep(5)
    return int(any(row["status"] != "complete" for row in write_summary(args.output_dir, args.datasets, args.models)))


if __name__ == "__main__":
    raise SystemExit(main())
