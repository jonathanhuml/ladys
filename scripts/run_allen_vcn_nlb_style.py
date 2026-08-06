#!/usr/bin/env python3
"""Run NLB-style held-out-neuron benchmarks on Allen VCN prepared H5 groups.

The script writes generated LaDyS YAML configs under the output directory, runs
each config through ``python -m ladys.cli run``, and maintains a CSV summary.
It is intentionally file-based so long HAL jobs can be resumed safely.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import h5py
import yaml


DEFAULT_GROUPS = (
    "flash_fc_top",
    "flash_bo_top",
    "flash_fc_second",
    "bo_drifting_one_tf",
)
DEFAULT_METHODS = (
    "psth",
    "smoothing",
    "gpfa",
    "cassm",
    "kalman",
    "lfads",
    "ndt",
    "stndt",
    "langevin_flow",
    "mint",
)
OPTIONAL_METHODS = ("bgpfa", "ilqr_vae")
ALL_METHODS = DEFAULT_METHODS + OPTIONAL_METHODS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", default=list(DEFAULT_GROUPS))
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument("--data-path", default="data/real/allen_vcn/allen_vcn_low_trial_20ms.h5")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--python", default="/home/jon/torch-gpu/bin/python")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--write-configs-only", action="store_true")
    parser.add_argument(
        "--validate-build-only",
        action="store_true",
        help="write configs, load each through LaDyS, and instantiate the model without training",
    )
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--timeout-hours", type=float, default=0.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir or _default_output_dir())
    config_dir = output_dir / "configs"
    log_dir = output_dir / "_logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    shape_info = read_shape_info(Path(args.data_path), args.groups)
    rows = _read_existing_summary(output_dir / "summary.csv")
    completed = {(row["group"], row["method"]): row for row in rows if row.get("status") == "ok"}

    jobs = []
    for group in args.groups:
        for method in _ordered_methods(args.methods):
            dataset = f"allen_vcn_{group}"
            run_name = f"{method}_{dataset}_20ms_nlb_style"
            config = build_config(
                group=group,
                method=method,
                data_path=args.data_path,
                output_dir=str(output_dir),
                run_name=run_name,
                device=args.device,
                shape=shape_info[group],
            )
            config_path = config_dir / f"{run_name}.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            jobs.append((group, method, dataset, run_name, config_path))

    _write_manifest(
        output_dir / "run_manifest.json",
        groups=args.groups,
        methods=_ordered_methods(args.methods),
        data_path=args.data_path,
        shape_info=shape_info,
        jobs=jobs,
    )

    if args.write_configs_only:
        print(f"Wrote {len(jobs)} configs to {config_dir}", flush=True)
        return 0
    if args.validate_build_only:
        validate_generated_configs(jobs)
        return 0

    timeout = None if args.timeout_hours <= 0 else args.timeout_hours * 3600.0
    for index, (group, method, dataset, run_name, config_path) in enumerate(jobs, start=1):
        if args.resume and (group, method) in completed:
            print(f"[{index}/{len(jobs)}] skip complete {dataset} {method}", flush=True)
            continue

        run_dir = output_dir / run_name
        metrics_path = run_dir / "metrics.json"
        if args.resume and metrics_path.exists():
            row = row_from_run(
                group=group,
                method=method,
                dataset=dataset,
                run_name=run_name,
                config_path=config_path,
                log_path=log_dir / f"{run_name}.log",
                run_dir=run_dir,
                seconds=0.0,
                status="ok",
                error="",
                rows=rows,
            )
            rows = _upsert_row(rows, row)
            _write_summary(output_dir / "summary.csv", rows)
            print(f"[{index}/{len(jobs)}] found metrics {dataset} {method}", flush=True)
            continue

        log_path = log_dir / f"{run_name}.log"
        print(f"[{index}/{len(jobs)}] run {dataset} {method}", flush=True)
        started = time.perf_counter()
        cmd = [args.python, "-m", "ladys.cli", "run", "-c", str(config_path)]
        try:
            proc = subprocess.run(
                cmd,
                cwd=Path.cwd(),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
            elapsed = time.perf_counter() - started
            log_path.write_text(proc.stdout)
            status = "ok" if proc.returncode == 0 else f"exit_{proc.returncode}"
            error = "" if proc.returncode == 0 else proc.stdout[-1600:].replace("\n", " | ")
            run_dir = _run_dir_from_stdout(proc.stdout, default=run_dir)
        except subprocess.TimeoutExpired as exc:
            elapsed = time.perf_counter() - started
            out = exc.stdout or ""
            if isinstance(out, bytes):
                out = out.decode(errors="replace")
            log_path.write_text(out)
            status = "timeout"
            error = f"timed out after {timeout:.0f}s"

        row = row_from_run(
            group=group,
            method=method,
            dataset=dataset,
            run_name=run_name,
            config_path=config_path,
            log_path=log_path,
            run_dir=run_dir,
            seconds=elapsed,
            status=status,
            error=error,
            rows=rows,
        )
        rows = _upsert_row(rows, row)
        _write_summary(output_dir / "summary.csv", rows)
        co_bps = row.get("co_bps", "")
        margin = row.get("baseline_margin", "")
        print(
            f"[{index}/{len(jobs)}] {status} {dataset} {method} "
            f"co_bps={co_bps} margin={margin} seconds={elapsed:.1f}",
            flush=True,
        )

    return 0


def _default_output_dir() -> str:
    date = datetime.now().strftime("%Y%m%d")
    return f"runs/allen_vcn_nlb_style_{date}"


def _write_manifest(
    path: Path,
    *,
    groups: list[str],
    methods: list[str],
    data_path: str,
    shape_info: dict[str, dict[str, int]],
    jobs: list[tuple[str, str, str, str, Path]],
) -> None:
    payload = {
        "data_path": data_path,
        "groups": list(groups),
        "methods": list(methods),
        "shape_info": shape_info,
        "jobs": [
            {
                "group": group,
                "method": method,
                "dataset": dataset,
                "run_name": run_name,
                "config": str(config_path),
            }
            for group, method, dataset, run_name, config_path in jobs
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _ordered_methods(methods: list[str]) -> list[str]:
    requested = set(methods)
    return [method for method in ALL_METHODS if method in requested]


def read_shape_info(path: Path, groups: list[str]) -> dict[str, dict[str, int]]:
    info: dict[str, dict[str, int]] = {}
    with h5py.File(path, "r") as handle:
        for group_name in groups:
            group = handle[group_name]
            train_heldin = group["train_spikes_heldin"].shape
            train_heldout = group["train_spikes_heldout"].shape
            valid_heldin = group["eval_spikes_heldin"].shape
            info[group_name] = {
                "train_trials": int(train_heldin[0]),
                "valid_trials": int(valid_heldin[0]),
                "time": int(train_heldin[1]),
                "heldin": int(train_heldin[2]),
                "heldout": int(train_heldout[2]),
                "full": int(train_heldin[2] + train_heldout[2]),
            }
    return info


def validate_generated_configs(jobs: list[tuple[str, str, str, str, Path]]) -> None:
    from ladys.config import load_experiment_config
    from ladys.experiment import Experiment

    for index, (_group, method, dataset, _run_name, config_path) in enumerate(jobs, start=1):
        if method == "mint":
            print(f"[{index}/{len(jobs)}] ok {dataset} {method} build=skipped_mint", flush=True)
            continue
        config = load_experiment_config(str(config_path))
        model = Experiment(config).build_model()
        input_neurons = getattr(model, "n_neurons", getattr(model, "input_neurons", ""))
        output_neurons = getattr(
            model,
            "output_neurons",
            getattr(model, "readout_neurons", input_neurons),
        )
        readout_neurons = getattr(model, "readout_neurons", output_neurons)
        print(
            f"[{index}/{len(jobs)}] ok {dataset} {method} "
            f"input_neurons={input_neurons} output_neurons={output_neurons} "
            f"readout_neurons={readout_neurons}",
            flush=True,
        )


def build_config(
    *,
    group: str,
    method: str,
    data_path: str,
    output_dir: str,
    run_name: str,
    device: str,
    shape: dict[str, int],
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "dataset": {
            "name": f"allen_vcn_{group}",
            "data_path": data_path,
            "group": group,
        },
        "model": {},
        "preprocessing": {"observations": None},
        "trainer": {"epochs": 0, "batch_size": 64, "device": device, "live_eval_interval": 0},
        "experiment": {"output_dir": output_dir, "run_name": run_name, "save_predictions": True},
    }

    if method == "psth":
        config["model"] = {
            "name": "psth",
            "objective": "psth_poisson_nll",
            "kern_sd_ms": 70.0,
            "bin_size_ms": 20.0,
            "prediction_floor": 1.0e-9,
            "optimization": {"name": "inference_only"},
        }
        config["trainer"].update({"device": "cpu", "batch_size": 4096})
    elif method == "smoothing":
        config["model"] = {
            "name": "smoothing",
            "objective": "smoothed_poisson_nll",
            "kern_sd_ms": 50.0,
            "bin_size_ms": 20.0,
            "log_offset": 1.0e-4,
            "nlb_decoder_alpha": 0.01,
            "nlb_poisson_max_iter": 20,
            "prediction_floor": 1.0e-9,
            "optimization": {"name": "inference_only"},
        }
        config["trainer"].update({"device": "cpu", "batch_size": 4096})
    elif method == "gpfa":
        config["model"] = {
            "name": "gpfa",
            "objective": "negative_log_marginal_likelihood",
            "latent_dim": 32 if shape["time"] <= 25 else 48,
            "bin_width": 20.0,
            "start_tau": 100.0,
            "start_eps": 1.0e-3,
            "min_var_frac": 0.01,
            "init_method": "fa",
            "init_seed": 0,
            "learn_kernel_params": False,
            "fa_max_iters": 500,
            "fa_tol": 1.0e-8,
            "kernel_param_max_iters": 8,
            "kernel_param_lr": 1.0,
            "jitter": 1.0e-5,
            "optimization": {"name": "inference_only"},
        }
        config["trainer"].update({"device": "cpu", "batch_size": 4096})
    elif method == "cassm":
        config["model"] = {
            "name": "cassm",
            "objective": "cassm_elbo",
            "projection_dim": 96,
            "dt": 0.02,
            "dataset_name": f"allen_vcn_{group}",
            "save_model": False,
            "use_dense_projection": True,
            "health_checks": True,
            "nlb_feature_source": "predict_rates",
            "nlb_decoder": "ridge",
            "nlb_ridge_alpha": 500.0,
            "optimization": {
                "name": "gradient",
                "optimizer": "Adam",
                "lr": 5.0e-2,
                "weight_decay": 0.0,
                "gradient_clip": 300.0,
            },
        }
        config["preprocessing"] = {
            "observations": {
                "name": "smooth_firing_rate",
                "sampling_precision": 20.0,
                "kern_sd_ms": 50.0,
            }
        }
        config["trainer"].update({"epochs": 100, "batch_size": 64, "live_eval_interval": 0})
    elif method == "kalman":
        config["model"] = {
            "name": "kalman",
            "objective": "negative_log_marginal_likelihood",
            "dt": 0.02,
            "dataset_name": f"allen_vcn_{group}",
            "save_model": False,
            "nlb_ridge_alpha": 500.0,
            "optimization": {"name": "inference_only"},
        }
        config["preprocessing"] = {
            "observations": {
                "name": "smooth_firing_rate",
                "sampling_precision": 20.0,
                "kern_sd_ms": 50.0,
            }
        }
        config["trainer"].update({"epochs": 0, "batch_size": 64, "device": "cpu"})
    elif method == "lfads":
        config["model"] = {
            "name": "lfads",
            "objective": "lfads_elbo",
            "generator_dim": 200,
            "initial_condition_dim": 64,
            "inferred_input_dim": 4,
            "factor_dim": 100,
            "g0_encoder_dim": 64,
            "controller_encoder_dim": 64,
            "controller_dim": 64,
            "g0_prior_kappa": 0.1,
            "inferred_input_prior_kappa": 0.1,
            "inferred_input_prior": "autoregressive",
            "inferred_input_prior_tau": 10.0,
            "keep_prob": 0.7,
            "clip_val": 5.0,
            "dt": 0.02,
            "log_rate_min": -8.0,
            "log_rate_max": 8.0,
            "posterior_logvar_min": -9.210340371976182,
            "posterior_logvar_max": 5.0,
            "controller_lag": 1,
            "reconstruction_time_steps": shape["time"],
            "readout_neurons": shape["full"],
            "output_neuron_start": shape["heldin"],
            "output_neurons": shape["heldout"],
            "use_log1p_encoder_inputs": False,
            "initialize_log_rate_bias": True,
            "prediction_samples": 1,
            "loss_scale": 10000.0,
            "reconstruction_reduce_mean": True,
            "coordinated_dropout_rate": 0.3,
            "coordinated_dropout_pass_rate": 0.0,
            "coordinated_dropout_ic_enc_seq_len": 0,
            "kl_weight_schedule_start": 0,
            "kl_weight_schedule_dur": 2880,
            "kl_g0_scale": 1.0e-7,
            "kl_u_scale": 1.0e-7,
            "l2_weight_schedule_start": 0,
            "l2_weight_schedule_dur": 2880,
            "l2_gen_scale": 0.0,
            "l2_con_scale": 0.0,
            "optimization": {
                "name": "gradient",
                "optimizer": "AdamW",
                "lr": 4.0e-3,
                "weight_decay": 0.0,
                "gradient_clip": 200.0,
            },
        }
        config["trainer"].update({"epochs": 100, "batch_size": 64, "live_eval_interval": 10})
    elif method == "ndt":
        config["model"] = {
            "name": "ndt",
            "objective": "masked_poisson_nll",
            "output_neurons": shape["full"] + (shape["full"] % 2),
            "output_mode": "auto",
            "fwd_steps": 0,
            "context_forward": min(64, max(shape["time"] - 1, 1)),
            "context_backward": min(64, max(shape["time"] - 1, 1)),
            "context_wrap_initial": False,
            "full_context": False,
            "hidden_size": 128,
            "dropout": 0.2,
            "dropout_rates": 0.2,
            "dropout_embedding": 0.2,
            "num_heads": 2,
            "num_layers": 4,
            "activation": "relu",
            "linear_embedder": True,
            "embed_dim": 1,
            "learnable_position": True,
            "max_spike_count": 20,
            "lograte": True,
            "log_rate_min": -8.0,
            "log_rate_max": 8.0,
            "spike_log_init": False,
            "fixup_init": True,
            "pre_norm": True,
            "scale_norm": False,
            "decoder_layers": 1,
            "position_offset": True,
            "mask_ratio": 0.25,
            "mask_mode": "full",
            "mask_token_ratio": 0.75,
            "mask_random_ratio": 0.9,
            "mask_max_span": 7,
            "mask_span_expand_prob": 0.0,
            "mask_span_ramp_start": 240,
            "mask_span_ramp_end": 360,
            "use_zero_mask": True,
            "topk_loss_fraction": 1.0,
            "nlb_decoder": "direct",
            "optimization": {
                "name": "gradient",
                "optimizer": "AdamW",
                "lr": 1.0e-3,
                "weight_decay": 5.0e-5,
                "gradient_clip": 200.0,
                "lr_scheduler": "warmup_cosine",
                "warmup_steps": 150,
                "total_steps": 1500,
                "scheduler_step": "epoch",
            },
        }
        config["trainer"].update({"epochs": 1500, "batch_size": 64, "live_eval_interval": 250})
    elif method == "stndt":
        config["model"] = {
            "name": "stndt",
            "objective": "stndt_masked_poisson_nll",
            "ensemble": False,
            "ensemble_size": 2,
            "output_mode": "auto",
            "fwd_steps": 0,
            "context_forward": min(46, max(shape["time"] - 1, 1)),
            "context_backward": min(7, max(shape["time"] - 1, 1)),
            "context_wrap_initial": False,
            "full_context": False,
            "hidden_size": 128,
            "dropout": 0.3258805092088328,
            "dropout_rates": 0.3331559445029162,
            "dropout_embedding": 0.5567310695526302,
            "num_heads": 2 if shape["full"] % 2 == 0 and shape["time"] % 2 == 0 else 1,
            "num_layers": 4,
            "activation": "relu",
            "linear_embedder": True,
            "embed_dim": 1,
            "learnable_position": False,
            "max_spike_count": 20,
            "lograte": True,
            "log_rate_min": -8.0,
            "log_rate_max": 8.0,
            "spike_log_init": False,
            "fixup_init": True,
            "pre_norm": True,
            "scale_norm": False,
            "decoder_layers": 1,
            "position_offset": False,
            "mask_ratio": 0.25473718401208967,
            "mask_mode": "full",
            "mask_token_ratio": 0.8355074373250815,
            "mask_random_ratio": 0.8578455430626231,
            "mask_max_span": 5,
            "mask_span_expand_prob": 0.0,
            "mask_span_ramp_start": 8000,
            "mask_span_ramp_end": 12000,
            "use_zero_mask": True,
            "topk_loss_fraction": 1.0,
            "do_contrast": False,
            "contrast_mask_ratio": 0.16105417937781183,
            "contrast_mask_mode": "full",
            "contrast_mask_token_ratio": 0.9925137916228178,
            "contrast_mask_random_ratio": 0.8592738189960607,
            "contrast_mask_max_span": 1,
            "contrast_mask_span_expand_prob": 0.0,
            "temperature": 0.07,
            "contrast_lambda": 0.5706354364772466,
            "use_contrast_projector": False,
            "linear_projector": True,
            "contrast_layer": "embedder",
            "nlb_decoder": "direct",
            "optimization": {
                "name": "gradient",
                "optimizer": "AdamW",
                "lr": 8.249215636338611e-3,
                "weight_decay": 3.8970415504287285e-4,
                "gradient_clip": 200.0,
                "lr_scheduler": "warmup_cosine",
                "warmup_steps": 958,
                "total_steps": 3000,
                "scheduler_step": "epoch",
            },
        }
        config["trainer"].update({"epochs": 3000, "batch_size": 64, "live_eval_interval": 500})
    elif method == "langevin_flow":
        config["model"] = {
            "name": "langevin_flow",
            "objective": "langevin_flow_elbo",
            "hidden_size": 280,
            "initialization": "ladys",
            "output_mode": "auto",
            "fwd_steps": 0,
            "dropout": 0.05,
            "gamma": 0.55,
            "langevin_step": 0.01,
            "potential_groups": 4,
            "potential_kernel_size": 3,
            "transformer_heads": 2,
            "transformer_feedforward": 512,
            "coordinated_dropout_rate": 0.5,
            "kl_weight": 0.1,
            "kl_warmup_epochs": 500,
            "weight_decay_warmup_epochs": 500,
            "velocity_prior_var": 0.1,
            "log_rate_min": -8.0,
            "log_rate_max": 8.0,
            "posterior_logvar_min": -9.210340371976182,
            "posterior_logvar_max": 5.0,
            "sample_train": True,
            "sample_eval": False,
            "prediction_samples": 50,
            "optimization": {
                "name": "gradient",
                "optimizer": "Adam",
                "lr": 3.0e-3,
                "weight_decay": 2.0e-5,
                "gradient_clip": 200.0,
                "lr_scheduler": "ReduceLROnPlateau",
                "scheduler_factor": 0.95,
                "scheduler_patience": 10,
                "scheduler_threshold": 0.0,
                "scheduler_min_lr": 1.0e-5,
            },
        }
        config["trainer"].update({"epochs": 500, "batch_size": 64, "live_eval_interval": 250})
    elif method == "mint":
        config["model"] = {
            "name": "mint",
            "objective": "mint_likelihood_recursion",
            "dataset": "allen_vcn",
            "lorenz_library_source": "smoothed_spikes",
            "n_candidates": 8,
            "window_length": 4,
            "delta": 1,
            "sigma": 4,
            "min_rate": 1.0e-4,
            "causal": False,
            "optimization": {"name": "inference_only"},
        }
        config["trainer"].update({"epochs": 0, "batch_size": 8, "device": "cpu"})
    elif method == "bgpfa":
        config["model"] = {
            "name": "bgpfa",
            "objective": "negative_elbo",
            "latent_dim": 12,
            "binsize": 20.0,
            "ell0": None,
            "rho": 2.0,
            "n_mc_train": 1,
            "n_mc_eval": 3,
            "kl_burnin_epochs": 25,
            "latent_scale_init": 1.0,
            "likelihood": "gaussian",
            "learn_scale": False,
            "ard": True,
            "dtype": "float32",
            "optimization": {
                "name": "mgplvm_full_batch_gradient",
                "optimizer": "Adam",
                "lr": 2.0e-2,
                "steps_per_epoch": 1,
                "burnin": 50,
                "n_mc": 1,
                "weight_decay": 0.0,
                "gradient_clip": 200.0,
            },
        }
        config["trainer"].update({"epochs": 150, "batch_size": 4096, "live_eval_interval": 50})
    elif method == "ilqr_vae":
        heldin = int(shape["heldin"])
        heldout = int(shape["heldout"])
        config["model"] = {
            "name": "ilqr_vae",
            "objective": "ilqr_vae_elbo",
            "params_path": None,
            "initialization": "checkpoint_transfer",
            "template_params_path": "data/real/ilqr_vae/final_params.bin",
            "random_init_profile": "tutorial_mc_maze",
            "readout_bias_initialization": "empirical_rates",
            "empirical_rate_floor_hz": 1.0e-3,
            "latent_dim": 90,
            "input_dim": 15,
            "init_seed": 0,
            "solver": "ilqr",
            "max_iter": 2,
            "lr": None,
            "differentiate_controls": False,
            "trainable_parameters": True,
            "n_posterior_samples": 1,
            "include_elbo_constants": True,
            "dynamics_regularizer": 1.0e-5,
            "held_in_neurons": heldin,
            "output_neuron_start": heldin,
            "output_neurons": heldout,
            "rate_mode": "likelihood",
            "dt": 0.02,
            "optimization": {
                "name": "gradient",
                "optimizer": "Adam",
                "lr": 4.0e-3,
                "lr_scheduler": "sqrt_decay",
                "sqrt_decay_scale": 1.0,
                "weight_decay": 0.0,
                "gradient_clip": 200.0,
            },
        }
        config["trainer"].update({"epochs": 15, "batch_size": 2, "device": "cpu"})
    else:
        raise ValueError(f"Unknown method: {method}")

    return config


def row_from_run(
    *,
    group: str,
    method: str,
    dataset: str,
    run_name: str,
    config_path: Path,
    log_path: Path,
    run_dir: Path,
    seconds: float,
    status: str,
    error: str,
    rows: list[dict[str, str]],
) -> dict[str, str]:
    metrics = _read_json(run_dir / "metrics.json")
    co_bps = _metric(metrics, "co_bps")
    poisson_nll = _metric(metrics, "poisson_nll")
    target = _baseline_target(rows, group)
    margin = ""
    beats = ""
    if co_bps != "" and target != "":
        margin_value = float(co_bps) - float(target)
        margin = f"{margin_value:.10g}"
        beats = str(margin_value > 0.0).lower()
    return {
        "group": group,
        "dataset": dataset,
        "method": method,
        "run_name": run_name,
        "status": status,
        "seconds": f"{seconds:.3f}",
        "co_bps": co_bps,
        "poisson_nll": poisson_nll,
        "baseline_target": target,
        "baseline_margin": margin,
        "beats_baseline": beats,
        "run_dir": str(run_dir),
        "config": str(config_path),
        "log": str(log_path),
        "error": error,
    }


def _metric(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    if value is None:
        return ""
    try:
        return f"{float(value):.10g}"
    except (TypeError, ValueError):
        return ""


def _baseline_target(rows: list[dict[str, str]], group: str) -> str:
    values = []
    for row in rows:
        if row.get("group") != group or row.get("method") not in {"psth", "smoothing"}:
            continue
        value = row.get("co_bps", "")
        if value:
            values.append(float(value))
    if not values:
        return ""
    return f"{max(values):.10g}"


def _run_dir_from_stdout(stdout: str, *, default: Path) -> Path:
    for line in stdout.splitlines():
        if line.startswith("Wrote LaDyS run:"):
            value = line.split(":", 1)[1].strip()
            if value:
                return Path(value)
    return default


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _read_existing_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _upsert_row(rows: list[dict[str, str]], row: dict[str, str]) -> list[dict[str, str]]:
    key = (row["group"], row["method"])
    out = [existing for existing in rows if (existing.get("group"), existing.get("method")) != key]
    out.append(row)
    return sorted(out, key=lambda item: (item["group"], _method_order(item["method"])))


def _method_order(method: str) -> int:
    try:
        return ALL_METHODS.index(method)
    except ValueError:
        return len(ALL_METHODS)


def _write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "group",
        "dataset",
        "method",
        "run_name",
        "status",
        "seconds",
        "co_bps",
        "poisson_nll",
        "baseline_target",
        "baseline_margin",
        "beats_baseline",
        "run_dir",
        "config",
        "log",
        "error",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


if __name__ == "__main__":
    raise SystemExit(main())
