"""Generate temporary iLQR-VAE NLB sweep configs."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


BASE_CONFIGS = {
    "area2_bump": Path(
        "configs/experiment/real/area2_bump/ilqr_vae/ilqr_vae_area2_bump_nlb_5ms_train.yaml"
    ),
    "dmfc_rsg": Path(
        "configs/experiment/real/dmfc_rsg/ilqr_vae/ilqr_vae_dmfc_rsg_nlb_5ms_train.yaml"
    ),
    "mc_rtt": Path(
        "configs/experiment/real/mc_rtt/ilqr_vae/ilqr_vae_mc_rtt_nlb_5ms_train.yaml"
    ),
}

DEFAULT_BATCH_SIZES = {
    "area2_bump": 8,
    "dmfc_rsg": 4,
    "mc_rtt": 8,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="runs/ilqr_vae_other_dataset_sweeps")
    parser.add_argument("--datasets", nargs="+", default=list(BASE_CONFIGS))
    parser.add_argument("--max-trials", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.0e-2)
    parser.add_argument(
        "--objective",
        choices=["posterior_control", "ilqr_vae_elbo"],
        default="ilqr_vae_elbo",
    )
    parser.add_argument(
        "--differentiate-controls",
        action="store_true",
        help="Backpropagate through the unrolled iLQR updates during ELBO training.",
    )
    parser.add_argument(
        "--initialization",
        choices=["checkpoint_transfer", "random"],
        default="checkpoint_transfer",
    )
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--input-dim", type=int)
    parser.add_argument(
        "--control-hessian-mode",
        choices=["true", "fisher", "clamped"],
    )
    parser.add_argument("--fallback-max-iter", type=int)
    parser.add_argument("--fallback-lr", type=float)
    parser.add_argument("--live-eval-interval", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = Path(args.output_dir)
    config_dir = root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        if dataset not in BASE_CONFIGS:
            known = ", ".join(sorted(BASE_CONFIGS))
            raise KeyError(f"Unknown iLQR-VAE NLB dataset {dataset!r}. Known: {known}")
        cfg = yaml.safe_load(BASE_CONFIGS[dataset].read_text())
        cfg["dataset"]["max_trials"] = int(args.max_trials)
        cfg["model"]["objective"] = args.objective
        cfg["model"]["differentiate_controls"] = (
            args.objective == "ilqr_vae_elbo" and args.differentiate_controls
        )
        cfg["model"]["initialization"] = args.initialization
        cfg["model"]["params_path"] = None
        if args.initialization == "random":
            cfg["model"]["template_params_path"] = None
        if args.latent_dim is not None:
            cfg["model"]["latent_dim"] = int(args.latent_dim)
        if args.input_dim is not None:
            cfg["model"]["input_dim"] = int(args.input_dim)
        if args.control_hessian_mode is not None:
            cfg["model"]["control_hessian_mode"] = args.control_hessian_mode
        cfg["model"]["max_iter"] = int(args.max_iter)
        if args.fallback_max_iter is not None:
            cfg["model"]["ilqr_failure_fallback"] = "adam"
            cfg["model"]["ilqr_fallback_max_iter"] = int(args.fallback_max_iter)
        if args.fallback_lr is not None:
            cfg["model"]["ilqr_failure_fallback"] = "adam"
            cfg["model"]["ilqr_fallback_lr"] = float(args.fallback_lr)
        cfg["model"]["trainable_parameters"] = True
        cfg["model"]["optimization"] = {
            "name": "gradient",
            "optimizer": "Adam",
            "lr": float(args.lr),
            "weight_decay": 0.0,
            "gradient_clip": 200.0,
        }
        cfg["trainer"]["epochs"] = int(args.epochs)
        cfg["trainer"]["batch_size"] = DEFAULT_BATCH_SIZES[dataset]
        cfg["trainer"]["device"] = args.device
        cfg["trainer"]["live_eval_interval"] = int(args.live_eval_interval)
        cfg["experiment"]["output_dir"] = str(root)
        init_tag = "rand" if args.initialization == "random" else "xfer"
        dim_tag = ""
        if args.latent_dim is not None or args.input_dim is not None:
            dim_tag = f"_z{cfg['model']['latent_dim']}u{cfg['model']['input_dim']}"
        cfg["experiment"]["run_name"] = (
            f"ilqr_vae_{dataset}_nlb_5ms_{args.objective}_{init_tag}"
            f"{dim_tag}_ilqr{args.max_iter}_mt{args.max_trials}_lr{args.lr:g}".replace(".", "p")
        )
        if args.fallback_max_iter is not None:
            cfg["experiment"]["run_name"] += f"_fb{args.fallback_max_iter}"
        if args.control_hessian_mode is not None:
            cfg["experiment"]["run_name"] += f"_{args.control_hessian_mode}"
        if args.differentiate_controls:
            cfg["experiment"]["run_name"] += "_diffctl"
        out = config_dir / f"{cfg['experiment']['run_name']}.yaml"
        out.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
