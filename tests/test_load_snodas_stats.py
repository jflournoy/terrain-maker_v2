"""Tests for load_snodas_stats library function.

Verifies the SNODAS pipeline orchestration function that wraps
GriddedDataLoader + batch_process_snodas_data + calculate_snow_statistics.
"""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.snow.snodas import load_snodas_stats
from src.terrain.gridded_data import TiledDataConfig


EXPECTED_KEYS = {"median_max_depth", "mean_snow_day_ratio", "interseason_cv", "mean_intraseason_cv"}


class TestLoadSnodasStatsMockData:
    """Test mock data mode."""

    def test_mock_returns_dict_with_expected_keys(self):
        result = load_snodas_stats(mock_data=True, mock_shape=(50, 50))
        assert isinstance(result, dict)
        assert set(result.keys()) >= EXPECTED_KEYS

    def test_mock_shape_matches_requested(self):
        result = load_snodas_stats(mock_data=True, mock_shape=(100, 200))
        assert result["median_max_depth"].shape == (100, 200)
        assert result["mean_snow_day_ratio"].shape == (100, 200)

    def test_mock_default_shape(self):
        result = load_snodas_stats(mock_data=True)
        assert result["median_max_depth"].shape == (500, 500)

    def test_result_excludes_metadata_keys(self):
        result = load_snodas_stats(mock_data=True)
        assert "metadata" not in result
        assert "failed_files" not in result

    def test_mock_values_are_finite(self):
        result = load_snodas_stats(mock_data=True, mock_shape=(30, 30))
        for key in EXPECTED_KEYS:
            assert np.all(np.isfinite(result[key])), f"{key} contains non-finite values"


class TestLoadSnodasStatsFallback:
    """Test fallback to mock data when real loading can't proceed."""

    def test_falls_back_when_snodas_dir_missing(self):
        result = load_snodas_stats(
            terrain=MagicMock(),
            snodas_dir=Path("/definitely/not/a/real/path"),
            mock_shape=(30, 30),
        )
        assert isinstance(result, dict)
        assert result["median_max_depth"].shape == (30, 30)

    def test_falls_back_when_terrain_is_none(self):
        result = load_snodas_stats(
            terrain=None,
            snodas_dir=Path("/some/path"),
            mock_shape=(30, 30),
        )
        assert isinstance(result, dict)
        assert result["median_max_depth"].shape == (30, 30)

    def test_falls_back_when_snodas_dir_is_none(self):
        result = load_snodas_stats(
            terrain=MagicMock(),
            snodas_dir=None,
            mock_shape=(25, 25),
        )
        assert isinstance(result, dict)
        assert result["median_max_depth"].shape == (25, 25)


class TestLoadSnodasStatsConfiguration:
    """Test configuration passthrough."""

    def test_accepts_custom_tile_config(self):
        config = TiledDataConfig(
            max_output_pixels=4 * 1024 * 1024,
            target_tile_outputs=1000,
            max_memory_percent=75.0,
        )
        result = load_snodas_stats(
            mock_data=True,
            tile_config=config,
            mock_shape=(40, 40),
        )
        assert isinstance(result, dict)

    def test_accepts_custom_cache_dir_and_name(self):
        result = load_snodas_stats(
            mock_data=True,
            cache_dir=Path("/tmp/my_cache"),
            cache_name="my_snodas",
            mock_shape=(40, 40),
        )
        assert isinstance(result, dict)
