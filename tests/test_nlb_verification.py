import h5py
import numpy as np
import pytest

from scripts.verify_nlb_models import subset


def test_diagnostic_window_uses_only_scoring_mask_and_preserves_alignment(tmp_path):
    source, target = tmp_path / "source.h5", tmp_path / "subset.h5"
    values = np.arange(4 * 20 * 2, dtype=np.float32).reshape(4, 20, 2)
    with h5py.File(source, "w") as handle:
        for prefix in ("train", "eval"):
            for partition in ("heldin", "heldout"):
                handle[f"{prefix}_spikes_{partition}"] = values
        handle["eval_spikes_heldout"][:, :7] = np.nan
    start = subset(source, target, "dmfc_rsg", steps=4)
    assert start == 7
    with h5py.File(target) as handle:
        np.testing.assert_array_equal(handle["train_spikes_heldin"], values[:, 7:11])
        np.testing.assert_array_equal(handle["eval_spikes_heldout"], values[:2, 7:11])
    with h5py.File(source, "a") as handle:
        handle["eval_spikes_heldout"][:, 7:] = 0
    assert subset(source, target, "dmfc_rsg", steps=4) == start


def test_diagnostic_rejects_an_entirely_unscored_window(tmp_path):
    source = tmp_path / "unscored.h5"
    with h5py.File(source, "w") as handle:
        handle["eval_spikes_heldout"] = np.full((2, 10, 1), np.nan)
    with pytest.raises(ValueError, match="fully scored interval"):
        subset(source, tmp_path / "subset.h5", "dmfc_rsg", steps=4)
