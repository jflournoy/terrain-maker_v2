"""Pytest configuration and fixtures for terrain-maker tests."""

import functools
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


PRISM_URL = "https://ftp.prism.oregonstate.edu"


@functools.lru_cache(maxsize=1)
def _network_available() -> bool:
    """Check once per session whether the PRISM data server is reachable."""
    import requests

    try:
        requests.head(PRISM_URL, timeout=5)
        return True
    except requests.RequestException:
        return False


def pytest_collection_modifyitems(config, items):
    """Skip tests marked `network` when external data servers are unreachable."""
    network_items = [item for item in items if "network" in item.keywords]
    if network_items and not _network_available():
        skip = pytest.mark.skip(reason=f"network unavailable ({PRISM_URL} unreachable)")
        for item in network_items:
            item.add_marker(skip)
