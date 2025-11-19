"""Basic tests to verify project setup."""
import sys
from pathlib import Path
import pytest


def test_python_version():
    """Test that Python version is 3.11 or higher."""
    assert sys.version_info >= (3, 11), f"Python 3.11+ required, got {sys.version}"


def test_src_modules_importable():
    """Test that src modules can be imported."""
    # Test basic imports
    try:
        from src import config
        assert config.PROJECT_ROOT is not None
    except ImportError as e:
        pytest.fail(f"Failed to import src.config: {e}")


def test_project_structure():
    """Test that expected project directories exist."""
    project_root = Path(__file__).parent.parent

    expected_dirs = [
        "src",
        "src/terrain",
        "src/snow",
        "src/utils",
        "tests",
        "data",
        "docs",
    ]

    for dir_path in expected_dirs:
        full_path = project_root / dir_path
        assert full_path.exists(), f"Expected directory {dir_path} not found"
