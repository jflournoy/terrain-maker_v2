"""Pytest configuration and fixtures for terrain-maker tests."""
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import numpy as np


@pytest.fixture
def sample_dem():
    """Create a small synthetic DEM for testing."""
    # Create a simple 100x100 elevation grid
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    # Create a simple terrain with a peak in the center
    Z = 1000 + 100 * np.exp(-(X**2 + Y**2) / 50)
    return Z.astype(np.float32)


@pytest.fixture
def sample_dem_file(tmp_path):
    """Path to a temporary sample DEM file."""
    return tmp_path / "sample.tif"


@pytest.fixture
def cache_dir(tmp_path):
    """Temporary cache directory for tests."""
    cache = tmp_path / "cache"
    cache.mkdir()
    return cache


@pytest.fixture
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent
