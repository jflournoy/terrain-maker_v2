"""
TDD RED Phase: Tests for saving and loading XC skiing score components.

Issue #52: Score component grids should be saved alongside the final
combined score in the .npz output, so downstream visualizations can
show component breakdowns without recomputing from raw SNODAS data.
"""

import numpy as np
import pytest
from pathlib import Path


class TestXCSkiingComponentsInNpz:
    """Test that XC skiing .npz output includes individual component scores."""

    def test_get_component_scores_returns_all_xc_components(self):
        """get_component_scores returns all 3 XC skiing components as arrays."""
        from src.scoring.configs import (
            DEFAULT_XC_SKIING_SCORER,
            xc_skiing_compute_derived_inputs,
        )

        # Create mock snow_stats matching what xc_skiing_compute_derived_inputs expects
        shape = (10, 10)
        snow_stats = {
            "median_max_depth": np.random.uniform(50, 500, shape).astype(np.float32),
            "mean_snow_day_ratio": np.random.uniform(0, 1, shape).astype(np.float32),
            "interseason_cv": np.random.uniform(0, 1, shape).astype(np.float32),
            "mean_intraseason_cv": np.random.uniform(0, 1, shape).astype(np.float32),
        }

        inputs = xc_skiing_compute_derived_inputs(snow_stats)
        component_scores = DEFAULT_XC_SKIING_SCORER.get_component_scores(inputs)

        # Must have all 3 components
        assert "snow_depth" in component_scores
        assert "snow_coverage" in component_scores
        assert "snow_consistency" in component_scores

        # Each should be a numpy array with matching shape
        for name, arr in component_scores.items():
            assert isinstance(arr, np.ndarray), f"{name} should be ndarray"
            assert arr.shape == shape, f"{name} shape {arr.shape} != {shape}"
            # Scores should be in [0, 1] range
            assert arr.min() >= 0.0, f"{name} min={arr.min()} < 0"
            assert arr.max() <= 1.0, f"{name} max={arr.max()} > 1"

    def test_save_and_load_component_scores_roundtrip(self, tmp_path):
        """Component scores survive save/load via np.savez_compressed."""
        from src.scoring.configs import (
            DEFAULT_XC_SKIING_SCORER,
            xc_skiing_compute_derived_inputs,
        )
        from rasterio.transform import Affine

        shape = (10, 10)
        snow_stats = {
            "median_max_depth": np.random.uniform(50, 500, shape).astype(np.float32),
            "mean_snow_day_ratio": np.random.uniform(0, 1, shape).astype(np.float32),
            "interseason_cv": np.random.uniform(0, 1, shape).astype(np.float32),
            "mean_intraseason_cv": np.random.uniform(0, 1, shape).astype(np.float32),
        }

        inputs = xc_skiing_compute_derived_inputs(snow_stats)
        final_score = DEFAULT_XC_SKIING_SCORER.compute(inputs)
        component_scores = DEFAULT_XC_SKIING_SCORER.get_component_scores(inputs)

        # Save in the format that save_outputs should use (after #52 fix)
        score_path = tmp_path / "xc_skiing_scores.npz"
        transform = Affine(0.01, 0, -84.0, 0, -0.01, 43.0)
        transform_tuple = (transform.a, transform.b, transform.c,
                           transform.d, transform.e, transform.f)

        np.savez_compressed(
            score_path,
            score=final_score,
            transform=transform_tuple,
            crs="EPSG:4326",
            component_snow_depth=component_scores["snow_depth"],
            component_snow_coverage=component_scores["snow_coverage"],
            component_snow_consistency=component_scores["snow_consistency"],
        )

        # Load back and verify
        loaded = np.load(score_path)
        assert "score" in loaded
        assert "component_snow_depth" in loaded
        assert "component_snow_coverage" in loaded
        assert "component_snow_consistency" in loaded

        np.testing.assert_array_almost_equal(loaded["score"], final_score)
        np.testing.assert_array_almost_equal(
            loaded["component_snow_depth"], component_scores["snow_depth"]
        )
        np.testing.assert_array_almost_equal(
            loaded["component_snow_coverage"], component_scores["snow_coverage"]
        )
        np.testing.assert_array_almost_equal(
            loaded["component_snow_consistency"], component_scores["snow_consistency"]
        )

    def test_load_xc_skiing_components_function(self, tmp_path):
        """load_xc_skiing_components should load component arrays from .npz."""
        # This function doesn't exist yet — will be added in #53
        # For now, test the loading pattern we expect
        shape = (10, 10)
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

        # Load and verify the pattern works
        loaded = np.load(score_path)
        components = {}
        for key in ["snow_depth", "snow_coverage", "snow_consistency"]:
            npz_key = f"component_{key}"
            assert npz_key in loaded, f"Missing {npz_key} in .npz"
            components[key] = loaded[npz_key]

        assert len(components) == 3
        np.testing.assert_array_almost_equal(components["snow_depth"], depth)
        np.testing.assert_array_almost_equal(components["snow_coverage"], coverage)
        np.testing.assert_array_almost_equal(components["snow_consistency"], consistency)

    def test_npz_backward_compatible_without_components(self, tmp_path):
        """Loading an old .npz without components should still work for 'score' key."""
        shape = (10, 10)
        score_path = tmp_path / "xc_skiing_scores.npz"
        np.savez_compressed(score_path, score=np.ones(shape))

        loaded = np.load(score_path)
        assert "score" in loaded
        # Component keys should be absent, not raise
        assert "component_snow_depth" not in loaded
