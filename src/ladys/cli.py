"""Command-line interface for LaDyS experiments."""

from __future__ import annotations

import argparse
from dataclasses import replace
import math
from pathlib import Path
import sys

from ladys.config import ExperimentConfig, load_experiment_config
from ladys.data import available_datasets, build_dataset_config
from ladys.experiment import Experiment
from ladys.models.base import BaseModelConfig, load_model_config
from ladys.preprocessing import PreprocessingConfig
from ladys.training import TrainerConfig
from ladys.utils.yaml import load_yaml


def main(argv: list[str] | None = None) -> int:
    """Run the LaDyS CLI."""

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0].startswith("-") and not _is_help_request(argv):
        argv = ["run", *argv]

    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 0
    return int(args.handler(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ladys",
        description="Run and inspect LaDyS latent dynamics experiments.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run",
        help="run an experiment",
        description="Run a LaDyS experiment and write a self-contained run folder.",
    )
    run_parser.add_argument("-d", "--dataset", default="lorenz", help="dataset name")
    run_parser.add_argument("-m", "--model", default="cassm", help="model name")
    run_parser.add_argument("-c", "--config", help="full experiment YAML config")
    run_parser.add_argument("--dataset-config", help="dataset YAML config")
    run_parser.add_argument("--model-config", help="model YAML config")
    run_parser.add_argument("--epochs", type=int, help="number of training epochs")
    run_parser.add_argument("--batch-size", type=int, help="training batch size")
    run_parser.add_argument("--device", help="PyTorch device")
    run_parser.add_argument("--output-dir", help="directory where run folders are written")
    run_parser.add_argument("--run-name", help="run folder name")
    run_parser.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="skip predictions.npz artifact",
    )
    run_parser.set_defaults(handler=run_command)

    list_parser = subparsers.add_parser("list", help="list registered components")
    list_parser.add_argument("kind", choices=["datasets", "models"])
    list_parser.set_defaults(handler=list_command)

    return parser


def run_command(args: argparse.Namespace) -> int:
    config = build_experiment_config(args)
    result = Experiment(config).run()
    print(f"Wrote LaDyS run: {result.run_dir}")
    if result.metrics:
        print("Metrics:")
        for key, value in sorted(result.metrics.items()):
            display = "nan" if not math.isfinite(value) else f"{value:.6g}"
            print(f"  {key}: {display}")
    else:
        print("Metrics: none available")
    return 0


def list_command(args: argparse.Namespace) -> int:
    if args.kind == "datasets":
        for name in available_datasets():
            print(name)
        return 0

    from ladys import models as _models  # noqa: F401

    for name in sorted(BaseModelConfig.registry):
        print(name)
    return 0


def build_experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    if args.config:
        config = load_experiment_config(args.config)
    else:
        config = ExperimentConfig(
            dataset=_load_dataset_config(args.dataset, args.dataset_config),
            model=_load_model_config(args.model, args.model_config),
            trainer=TrainerConfig(),
            preprocessing=_load_default_preprocessing(args.dataset, args.model),
        )

    trainer = config.trainer
    if args.epochs is not None:
        trainer = replace(trainer, epochs=args.epochs)
    if args.device is not None:
        trainer = replace(trainer, device=args.device)

    return replace(
        config,
        trainer=trainer,
        batch_size=args.batch_size if args.batch_size is not None else config.batch_size,
        output_dir=args.output_dir if args.output_dir is not None else config.output_dir,
        run_name=args.run_name if args.run_name is not None else config.run_name,
        save_predictions=False if args.no_save_predictions else config.save_predictions,
    )


def _load_dataset_config(name: str, path: str | None) -> object:
    if path is None:
        return build_dataset_config(name)
    data = load_yaml(path)
    dataset_name = data.get("name", name)
    return build_dataset_config(dataset_name, data)


def _load_model_config(name: str, path: str | None) -> BaseModelConfig:
    from ladys import models as _models  # noqa: F401

    if path is not None:
        return load_model_config(path)
    if name not in BaseModelConfig.registry:
        known = ", ".join(sorted(BaseModelConfig.registry)) or "<none>"
        raise KeyError(f"Unknown model '{name}'. Registered models: {known}.")
    return BaseModelConfig.registry[name]()


def _load_default_preprocessing(dataset: str, model: str) -> PreprocessingConfig:
    path = Path("configs") / "experiment" / f"{model}_{dataset}.yaml"
    if not path.exists():
        return PreprocessingConfig()
    data = load_yaml(path)
    return PreprocessingConfig.model_validate(data.get("preprocessing", {}))


def _is_help_request(argv: list[str]) -> bool:
    return len(argv) == 1 and argv[0] in {"-h", "--help"}


if __name__ == "__main__":
    raise SystemExit(main())
