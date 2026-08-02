from pathlib import Path

from ladys.config import load_experiment_config
from ladys.models.lfads import LFADSConfig


LFADS_NLB_CONFIGS = [
    Path("configs/experiment/real/area2_bump/lfads/lfads_area2_bump_nlb_5ms_ladys.yaml"),
    Path("configs/experiment/real/dmfc_rsg/lfads/lfads_dmfc_rsg_nlb_5ms_ladys.yaml"),
    Path("configs/experiment/real/mc_maze/lfads/lfads_mc_maze_nlb_5ms_ladys.yaml"),
    Path("configs/experiment/real/mc_rtt/lfads/lfads_mc_rtt_nlb_5ms_ladys.yaml"),
]


def test_lfads_nlb_experiment_configs_load():
    configs = [load_experiment_config(path) for path in LFADS_NLB_CONFIGS]

    assert all(isinstance(config.model, LFADSConfig) for config in configs)
    assert all(config.model.objective == "lfads_elbo" for config in configs)
    assert all(config.dataset.input_mode == "heldin_full_reconstruction" for config in configs)
    assert all(config.batch_size == 64 for config in configs)


def test_real_lfads_folder_contains_only_verified_ladys_configs():
    lfads_paths = sorted(
        Path("configs/experiment/real").glob("*/lfads/*.yaml")
    )

    assert lfads_paths == sorted(LFADS_NLB_CONFIGS)
    assert all(path.name.endswith("_ladys.yaml") for path in lfads_paths)


def test_model_folder_has_single_generic_lfads_preset():
    lfads_model_paths = sorted(Path("configs/model").glob("*lfads*.yaml"))

    assert lfads_model_paths == [Path("configs/model/lfads.yaml")]
