from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.merge_nlb_full_rate_shards import _merge_split


def _shard(path: str, start: int, stop: int, total: int) -> dict:
    rows = stop - start
    return {
        "path": Path(path),
        "split": "eval",
        "start": start,
        "stop": stop,
        "total": total,
        "parts": {
            "rates_heldin": np.full((rows, 2, 3), start, dtype=np.float32),
            "rates_heldout": np.full((rows, 2, 1), stop, dtype=np.float32),
        },
    }


def test_merge_split_concatenates_contiguous_shards():
    merged = _merge_split(
        [_shard("a.npz", 0, 2, 5), _shard("b.npz", 2, 5, 5)],
        split="eval",
        expected_total=5,
    )

    assert merged["rates_heldin"].shape == (5, 2, 3)
    assert merged["rates_heldout"].shape == (5, 2, 1)
    np.testing.assert_array_equal(merged["rates_heldin"][:2], 0.0)
    np.testing.assert_array_equal(merged["rates_heldin"][2:], 2.0)


def test_merge_split_rejects_gaps():
    with pytest.raises(ValueError, match="not contiguous"):
        _merge_split(
            [_shard("a.npz", 0, 2, 5), _shard("b.npz", 3, 5, 5)],
            split="eval",
            expected_total=5,
        )
