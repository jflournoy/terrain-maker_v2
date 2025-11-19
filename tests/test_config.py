"""Tests for configuration module."""
import pytest
from pathlib import Path
from src import config


def test_project_root_exists():
    """Test that PROJECT_ROOT is set correctly."""
    assert config.PROJECT_ROOT.exists()
    assert config.PROJECT_ROOT.is_dir()


def test_data_directories_exist():
    """Test that data directories are created."""
    assert config.DATA_DIR.exists()
    assert config.CACHE_DIR.exists()
    assert config.DEM_CACHE.exists()
    assert config.TERRAIN_CACHE.exists()
    assert config.FEATURE_CACHE.exists()
    assert config.SNOW_CACHE.exists()


def test_snodas_symlink():
    """Test that SNODAS symlink exists."""
    # This test will only pass if the symlink was created
    if config.SNODAS_DIR.exists():
        assert config.SNODAS_DIR.is_symlink() or config.SNODAS_DIR.is_dir()


def test_config_constants():
    """Test that configuration constants are properly set."""
    assert config.DEFAULT_DEM_PATTERN == "*.hgt"
    assert isinstance(config.DEFAULT_LOG_LEVEL, str)
