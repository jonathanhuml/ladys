from pathlib import Path

import torch

from ladys.config import load_experiment_config
from ladys.models.stndt import STNDTConfig
from scripts.run_stndt_nlb_reproduction import _load_configs


VALID_STNDT_NLB_CONFIGS = [
    Path("configs/experiment/real/area2_bump/stndt/stndt_area2_bump_nlb_5ms.yaml"),
    Path("configs/experiment/real/dmfc_rsg/stndt/stndt_dmfc_rsg_nlb_5ms.yaml"),
    Path("configs/experiment/real/mc_rtt/stndt/stndt_mc_rtt_nlb_5ms.yaml"),
    Path("configs/experiment/real/mc_maze/stndt/stndt_mc_maze_nlb_5ms.yaml"),
]

OBSOLETE_STNDT_CONFIGS = [
    Path("configs/experiment/real/mc_maze/stndt/stndt_mc_maze_stage1_mask23a4b6e8_nlb_5ms.yaml"),
    Path("configs/experiment/real/mc_maze/stndt/stndt_mc_maze_stage2_contrast23a4b6e8_nlb_5ms.yaml"),
    Path("configs/experiment/synthetic/lorenz/stndt/stndt_lorenz.yaml"),
]


def test_stndt_nlb_experiment_configs_load():
    configs = [load_experiment_config(path) for path in VALID_STNDT_NLB_CONFIGS]

    assert [config.dataset.name for config in configs] == [
        "area2_bump",
        "dmfc_rsg",
        "mc_rtt",
        "mc_maze",
    ]
    assert all(isinstance(config.model, STNDTConfig) for config in configs)
    assert [config.model.num_heads for config in configs] == [1, 2, 2, 2]
    assert [config.model.num_layers for config in configs] == [4, 6, 4, 4]
    assert all(config.model.output_mode == "auto" for config in configs)
    assert all(config.model.nlb_decoder == "direct" for config in configs)
    assert all(config.model.mask_mode == "full" for config in configs)
    assert all(config.model.contrast_mask_mode == "full" for config in configs)
    assert [config.trainer.epochs for config in configs] == [40001, 50501, 50501, 3000]
    assert [config.model.context_forward for config in configs] == [4, 4, 4, 46]
    assert [config.model.context_backward for config in configs] == [8, 8, 8, 7]
    assert [config.model.dropout for config in configs] == [
        0.1,
        0.1,
        0.1,
        0.3258805092088328,
    ]
    assert [config.model.mask_token_ratio for config in configs] == [
        1.0,
        1.0,
        1.0,
        0.8355074373250815,
    ]
    assert [config.model.mask_random_ratio for config in configs] == [
        0.5,
        0.5,
        0.5,
        0.8578455430626231,
    ]
    assert [config.model.mask_max_span for config in configs] == [1, 1, 1, 5]
    assert [config.model.mask_span_ramp_start for config in configs] == [600, 8000, 8000, 8000]
    assert [config.model.mask_span_ramp_end for config in configs] == [1200, 12000, 12000, 12000]
    assert all(config.model.contrast_mask_span_ramp_start == 8000 for config in configs)
    assert all(config.model.contrast_mask_span_ramp_end == 12000 for config in configs)
    assert configs[-1].model.do_contrast is False


def test_stndt_mc_maze_single_yaml_expands_to_two_runner_stages():
    stages = _load_configs(
        path=Path("configs/experiment/real/mc_maze/stndt/stndt_mc_maze_nlb_5ms.yaml"),
        epochs=None,
        batch_size=None,
        device="cpu",
        output_dir="runs/stndt_nlb_reproduction",
        run_name_suffix="",
    )

    assert [name for name, _ in stages] == ["mask_only", "contrast"]
    assert [config.dataset.name for _, config in stages] == ["mc_maze", "mc_maze"]
    assert [config.trainer.epochs for _, config in stages] == [3000, 3000]
    assert [config.model.do_contrast for _, config in stages] == [False, True]
    assert [config.model.optimization.lr for _, config in stages] == [
        8.249215636338611e-3,
        1.0e-3,
    ]
    assert [config.run_name for _, config in stages] == [
        "stndt_mc_maze_nlb_5ms_mask_only",
        "stndt_mc_maze_nlb_5ms_contrast",
    ]


def test_stndt_mask_span_ramp_uses_training_epoch():
    config = STNDTConfig(
        mask_max_span=3,
        mask_span_expand_prob=0.25,
        mask_span_ramp_start=10,
        mask_span_ramp_end=20,
        contrast_mask_max_span=3,
    )
    model = config.build(n_neurons=4, n_time=8)

    model.set_training_epoch(5)
    assert model._current_mask_span_expand_prob(contrast=False) == 0.0
    assert model._current_mask_span_expand_prob(contrast=True) == 0.0

    model.set_training_epoch(15)
    assert model._current_mask_span_expand_prob(contrast=False) == 0.5
    assert model._current_mask_span_expand_prob(contrast=True) == 0.5

    model.set_training_epoch(25)
    assert model._current_mask_span_expand_prob(contrast=False) == 1.0
    assert model._current_mask_span_expand_prob(contrast=True) == 1.0


def test_stndt_upstream_initialization_parity():
    torch.manual_seed(123)
    model = STNDTConfig(
        linear_embedder=True,
        num_layers=3,
        num_heads=4,
        hidden_size=16,
        dropout=0.0,
        dropout_rates=0.0,
        dropout_embedding=0.0,
        do_contrast=False,
    ).build(n_neurons=16, n_time=8)

    assert torch.max(torch.abs(model.embedder.weight)).item() <= 0.1
    first = model.encoder.layers[0]
    second = model.encoder.layers[1]
    for (first_name, first_param), (second_name, second_param) in zip(
        first.named_parameters(),
        second.named_parameters(),
    ):
        assert first_name == second_name
        assert first_param.data_ptr() != second_param.data_ptr()
        assert torch.equal(first_param, second_param)


def test_obsolete_stndt_experiment_configs_are_absent():
    for path in OBSOLETE_STNDT_CONFIGS:
        assert not path.exists(), f"obsolete STNDT config should not exist: {path}"

    stndt_paths = sorted(Path("configs/experiment").rglob("*stndt*.yaml"))
    assert stndt_paths == sorted(VALID_STNDT_NLB_CONFIGS)
    for path in stndt_paths:
        config = load_experiment_config(path)
        assert config.trainer.epochs != 5, f"obsolete 5-epoch STNDT config found: {path}"
