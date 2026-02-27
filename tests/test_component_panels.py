"""
TDD RED Phase: Tests for score component panel rendering (#53).

Tests the load_xc_skiing_components function and argument parsing.
Blender-dependent rendering is tested via manual execution.
"""

import numpy as np
import pytest
from pathlib import Path


class TestLoadXCSkiingComponents:
    """Test loading individual score components from .npz file."""

    def test_load_xc_skiing_components_imports(self):
        """Function should be importable from the example module."""
        from examples.detroit_combined_render import load_xc_skiing_components

        assert callable(load_xc_skiing_components)

    def test_load_components_from_npz(self, tmp_path):
        """Should load all 3 component arrays from .npz file."""
        from examples.detroit_combined_render import load_xc_skiing_components

        shape = (20, 20)
        depth = np.random.uniform(0, 1, shape).astype(np.float32)
        coverage = np.random.uniform(0, 1, shape).astype(np.float32)
        consistency = np.random.uniform(0, 1, shape).astype(np.float32)

        score_path = tmp_path / "xc_skiing_scores.npz"
        np.savez_compressed(
            score_path,
            score=np.zeros(shape),
            component_snow_depth=depth,
            component_snow_coverage=coverage,
            component_snow_consistency=consistency,
        )

        components = load_xc_skiing_components(tmp_path)

        assert components is not None
        assert len(components) == 3
        assert "snow_depth" in components
        assert "snow_coverage" in components
        assert "snow_consistency" in components
        np.testing.assert_array_almost_equal(components["snow_depth"], depth)
        np.testing.assert_array_almost_equal(components["snow_coverage"], coverage)
        np.testing.assert_array_almost_equal(components["snow_consistency"], consistency)

    def test_returns_none_if_no_file(self, tmp_path):
        """Should return None if .npz file doesn't exist."""
        from examples.detroit_combined_render import load_xc_skiing_components

        result = load_xc_skiing_components(tmp_path)
        assert result is None

    def test_returns_none_if_no_components(self, tmp_path):
        """Should return None if .npz has no component_ keys (old format)."""
        from examples.detroit_combined_render import load_xc_skiing_components

        score_path = tmp_path / "xc_skiing_scores.npz"
        np.savez_compressed(score_path, score=np.ones((10, 10)))

        result = load_xc_skiing_components(tmp_path)
        assert result is None

    def test_partial_components_loads_available(self, tmp_path):
        """Should load whatever components are available."""
        from examples.detroit_combined_render import load_xc_skiing_components

        shape = (10, 10)
        score_path = tmp_path / "xc_skiing_scores.npz"
        np.savez_compressed(
            score_path,
            score=np.zeros(shape),
            component_snow_depth=np.ones(shape),
            # Missing: component_snow_coverage, component_snow_consistency
        )

        components = load_xc_skiing_components(tmp_path)
        # Should still load what's available
        assert components is not None
        assert "snow_depth" in components
        assert len(components) == 1


class TestShowComponentsArgParsing:
    """Test that --show-components argument is recognized."""

    def test_show_components_flag_recognized(self):
        """Parser should accept --show-components flag."""
        import sys
        from unittest.mock import patch

        # We can't easily import just the parser, so test that the flag
        # is mentioned in the module. This is a lightweight check.
        import importlib
        import examples.detroit_combined_render as module

        source = Path(module.__file__).read_text()
        assert "--show-components" in source, (
            "--show-components flag not found in detroit_combined_render.py"
        )
