"""Combining tile outputs keeps every tile's data or raises.

_aggregate_tiles returned only the first tile for an unknown strategy and for auto-detected
non-array outputs; spatial concatenation dropped every non-array key (so SNODAS's
failed_files vanished in tiled runs) and could misplace tiles when a key was missing from one.
"""

import numpy as np
import pytest

from terrain_maker.terrain.gridded_data import GriddedDataLoader, TileSpecGridded, TiledDataConfig


def _loader(strategy="auto"):
    loader = object.__new__(GriddedDataLoader)
    loader.tile_config = TiledDataConfig(aggregation_strategy=strategy)
    return loader


def _specs():
    return [
        TileSpecGridded(src_slice=(slice(0, 2), slice(0, 4)), out_slice=(slice(0, 2), slice(0, 4)),
                        extent=(0, 0, 1, 1), target_shape=(2, 4)),
        TileSpecGridded(src_slice=(slice(2, 4), slice(0, 4)), out_slice=(slice(2, 4), slice(0, 4)),
                        extent=(0, 1, 1, 2), target_shape=(2, 4)),
    ]


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="median"):
        _loader("median")._aggregate_tiles([np.zeros((2, 4))] * 2, _specs(), (4, 4))


def test_non_array_outputs_from_several_tiles_raise():
    with pytest.raises(ValueError, match="2 tiles"):
        _loader()._aggregate_tiles([{"path": "a"}, {"path": "b"}], _specs(), (4, 4))


def test_single_tile_non_array_output_passes_through():
    assert _loader()._aggregate_tiles([{"path": "a"}], _specs()[:1], (2, 4)) == {"path": "a"}


def test_concatenation_keeps_lists_and_per_tile_values():
    outputs = [
        {"depth": np.full((2, 4), 1.0), "failed_files": [("f1", "bad")], "metadata": {"tile": 0}},
        {"depth": np.full((2, 4), 2.0), "failed_files": [("f2", "worse")], "metadata": {"tile": 1}},
    ]
    merged = _loader()._aggregate_tiles(outputs, _specs(), (4, 4))
    np.testing.assert_array_equal(merged["depth"][:2], 1.0)
    np.testing.assert_array_equal(merged["depth"][2:], 2.0)
    assert merged["failed_files"] == [("f1", "bad"), ("f2", "worse")]
    assert merged["metadata"] == [{"tile": 0}, {"tile": 1}]


def test_array_key_missing_from_a_tile_raises():
    outputs = [{"depth": np.ones((2, 4))}, {"other": np.ones((2, 4))}]
    with pytest.raises(ValueError, match="depth"):
        _loader("concatenate")._aggregate_tiles(outputs, _specs(), (4, 4))
