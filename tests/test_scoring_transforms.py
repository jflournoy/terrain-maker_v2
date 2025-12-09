"""
TDD RED Phase: Tests for scoring transformation functions.

These transformations convert raw values into 0-1 scores for sledding suitability.

Transformation types:
1. Trapezoidal - sweet spot with ramp-up/down (e.g., slope_mean, snow_depth)
2. Dealbreaker - step function with optional soft falloff (e.g., cliff detection)
3. Linear - simple normalization (e.g., snow coverage ratio)

All transformations return values in [0, 1].
"""

import numpy as np
import pytest


# =============================================================================
# TRAPEZOIDAL TRANSFORMATION TESTS
# =============================================================================
# Shape:     ___________
#           /           \
#          /             \
# ________/               \________
# 0      ramp_start  sweet_start  sweet_end  ramp_end    max


class TestTrapezoidalTransform:
    """Test trapezoidal transformation with sweet spot."""

    def test_in_sweet_spot_returns_one(self):
        """Values in the sweet spot should return 1.0."""
        from src.scoring.transforms import trapezoidal

        # Slope sweet spot: 5-15 degrees
        result = trapezoidal(10.0, sweet_range=(5, 15), ramp_range=(3, 25))
        assert result == 1.0

        # Test edges of sweet spot
        assert trapezoidal(5.0, sweet_range=(5, 15), ramp_range=(3, 25)) == 1.0
        assert trapezoidal(15.0, sweet_range=(5, 15), ramp_range=(3, 25)) == 1.0

    def test_below_sweet_spot_ramps_up(self):
        """Values below sweet spot should ramp up from 0 to 1."""
        from src.scoring.transforms import trapezoidal

        # At ramp_start (3), should be 0
        assert trapezoidal(3.0, sweet_range=(5, 15), ramp_range=(3, 25)) == 0.0

        # Midway between ramp_start and sweet_start
        result = trapezoidal(4.0, sweet_range=(5, 15), ramp_range=(3, 25))
        assert 0.4 < result < 0.6  # Should be ~0.5

    def test_above_sweet_spot_ramps_down(self):
        """Values above sweet spot should ramp down from 1 to 0."""
        from src.scoring.transforms import trapezoidal

        # At ramp_end (25), should be 0
        assert trapezoidal(25.0, sweet_range=(5, 15), ramp_range=(3, 25)) == 0.0

        # Midway between sweet_end and ramp_end
        result = trapezoidal(20.0, sweet_range=(5, 15), ramp_range=(3, 25))
        assert 0.4 < result < 0.6  # Should be ~0.5

    def test_at_extremes_returns_zero(self):
        """Values outside ramp range should return 0."""
        from src.scoring.transforms import trapezoidal

        assert trapezoidal(0.0, sweet_range=(5, 15), ramp_range=(3, 25)) == 0.0
        assert trapezoidal(2.0, sweet_range=(5, 15), ramp_range=(3, 25)) == 0.0
        assert trapezoidal(30.0, sweet_range=(5, 15), ramp_range=(3, 25)) == 0.0
        assert trapezoidal(100.0, sweet_range=(5, 15), ramp_range=(3, 25)) == 0.0

    def test_works_with_numpy_arrays(self):
        """Should work with numpy arrays element-wise."""
        from src.scoring.transforms import trapezoidal

        values = np.array([0.0, 4.0, 10.0, 20.0, 30.0])
        result = trapezoidal(values, sweet_range=(5, 15), ramp_range=(3, 25))

        assert isinstance(result, np.ndarray)
        assert result.shape == values.shape
        assert result[0] == 0.0  # Below ramp
        assert result[2] == 1.0  # In sweet spot
        assert result[4] == 0.0  # Above ramp

    def test_asymmetric_ramps(self):
        """Should support asymmetric ramp-up and ramp-down."""
        from src.scoring.transforms import trapezoidal

        # Quick ramp up (3->5), slow ramp down (15->30)
        result_up = trapezoidal(4.0, sweet_range=(5, 15), ramp_range=(3, 30))
        result_down = trapezoidal(22.5, sweet_range=(5, 15), ramp_range=(3, 30))

        # Both at midpoint of their respective ramps
        assert 0.4 < result_up < 0.6
        assert 0.4 < result_down < 0.6


# =============================================================================
# DEALBREAKER TRANSFORMATION TESTS
# =============================================================================
# Shape (hard):  _______
#                       |
#                       |________
#               threshold
#
# Shape (soft):  _______
#                       \
#                        \______
#               threshold  falloff_end


class TestDealbreakerTransform:
    """Test dealbreaker (step function) transformation."""

    def test_below_threshold_returns_one(self):
        """Values below threshold should return 1.0 (no penalty)."""
        from src.scoring.transforms import dealbreaker

        # Cliff threshold: 25 degrees
        assert dealbreaker(10.0, threshold=25) == 1.0
        assert dealbreaker(24.9, threshold=25) == 1.0
        assert dealbreaker(0.0, threshold=25) == 1.0

    def test_at_threshold_hard_cutoff(self):
        """At threshold with hard cutoff, should return 0."""
        from src.scoring.transforms import dealbreaker

        # Hard cutoff (no falloff)
        assert dealbreaker(25.0, threshold=25, falloff=0) == 0.0

    def test_above_threshold_returns_zero(self):
        """Values above threshold should return 0.0 (full penalty)."""
        from src.scoring.transforms import dealbreaker

        assert dealbreaker(30.0, threshold=25, falloff=0) == 0.0
        assert dealbreaker(100.0, threshold=25, falloff=0) == 0.0

    def test_soft_falloff(self):
        """With falloff, should ramp down gradually after threshold."""
        from src.scoring.transforms import dealbreaker

        # Threshold at 25, falloff over 10 degrees
        assert dealbreaker(24.0, threshold=25, falloff=10) == 1.0  # Before threshold
        assert dealbreaker(25.0, threshold=25, falloff=10) == 1.0  # At threshold start
        assert dealbreaker(35.0, threshold=25, falloff=10) == 0.0  # At falloff end

        # Midway through falloff
        result = dealbreaker(30.0, threshold=25, falloff=10)
        assert 0.4 < result < 0.6  # Should be ~0.5

    def test_works_with_numpy_arrays(self):
        """Should work with numpy arrays element-wise."""
        from src.scoring.transforms import dealbreaker

        values = np.array([10.0, 25.0, 30.0, 40.0])
        result = dealbreaker(values, threshold=25, falloff=10)

        assert isinstance(result, np.ndarray)
        assert result[0] == 1.0  # Below threshold
        assert result[1] == 1.0  # At threshold
        assert result[3] == 0.0  # Past falloff

    def test_inverted_dealbreaker(self):
        """Should support 'below threshold is bad' mode."""
        from src.scoring.transforms import dealbreaker

        # For slope_min: we WANT low values (runout zones)
        # Below 5 degrees = good, above = bad
        result = dealbreaker(3.0, threshold=5, falloff=0, below_is_good=False)
        assert result == 0.0  # Below threshold, but below_is_good=False means penalty

        result = dealbreaker(10.0, threshold=5, falloff=0, below_is_good=False)
        assert result == 1.0  # Above threshold = no penalty


# =============================================================================
# LINEAR TRANSFORMATION TESTS
# =============================================================================
# Shape:        /
#              /
#             /
# ___________/
# min       max


class TestLinearTransform:
    """Test linear normalization transformation."""

    def test_at_min_returns_zero(self):
        """Value at min should return 0."""
        from src.scoring.transforms import linear

        assert linear(0.0, value_range=(0, 100)) == 0.0

    def test_at_max_returns_one(self):
        """Value at max should return 1."""
        from src.scoring.transforms import linear

        assert linear(100.0, value_range=(0, 100)) == 1.0

    def test_midpoint_returns_half(self):
        """Value at midpoint should return 0.5."""
        from src.scoring.transforms import linear

        assert linear(50.0, value_range=(0, 100)) == 0.5

    def test_clamps_below_min(self):
        """Values below min should clamp to 0."""
        from src.scoring.transforms import linear

        assert linear(-10.0, value_range=(0, 100)) == 0.0

    def test_clamps_above_max(self):
        """Values above max should clamp to 1."""
        from src.scoring.transforms import linear

        assert linear(150.0, value_range=(0, 100)) == 1.0

    def test_inverted_linear(self):
        """Should support inverted mode (high value = low score)."""
        from src.scoring.transforms import linear

        # For CV: high variability = bad
        assert linear(0.0, value_range=(0, 1), invert=True) == 1.0  # Low CV = good
        assert linear(1.0, value_range=(0, 1), invert=True) == 0.0  # High CV = bad

    def test_works_with_numpy_arrays(self):
        """Should work with numpy arrays element-wise."""
        from src.scoring.transforms import linear

        values = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
        result = linear(values, value_range=(0, 100))

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(
            result, np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        )

    def test_sqrt_scaling(self):
        """Should support sqrt scaling for diminishing returns."""
        from src.scoring.transforms import linear

        # Snow coverage: going from 60% to 80% is less important than 20% to 40%
        result_linear = linear(0.5, value_range=(0, 1), power=1.0)
        result_sqrt = linear(0.5, value_range=(0, 1), power=0.5)

        assert result_linear == 0.5
        assert result_sqrt == pytest.approx(np.sqrt(0.5), rel=1e-5)


# =============================================================================
# TERRAIN CONSISTENCY (COMBINED METRIC) TESTS
# =============================================================================


class TestTerrainConsistency:
    """Test combined roughness + slope_std metric."""

    def test_both_zero_returns_one(self):
        """Zero roughness and zero slope_std = perfect consistency."""
        from src.scoring.transforms import terrain_consistency

        result = terrain_consistency(roughness=0.0, slope_std=0.0)
        assert result == 1.0

    def test_both_at_threshold_returns_zero(self):
        """Both at threshold = fully inconsistent."""
        from src.scoring.transforms import terrain_consistency

        # Default thresholds: roughness=30m, slope_std=10deg
        result = terrain_consistency(roughness=30.0, slope_std=10.0)
        assert result == 0.0

    def test_one_bad_one_good(self):
        """One metric bad, one good = partial penalty."""
        from src.scoring.transforms import terrain_consistency

        # Only roughness is bad
        result = terrain_consistency(roughness=30.0, slope_std=0.0)
        assert 0.2 < result < 0.4  # RMS of (1, 0) = 0.707, so 1-0.707 ≈ 0.29

    def test_custom_thresholds(self):
        """Should accept custom thresholds."""
        from src.scoring.transforms import terrain_consistency

        result = terrain_consistency(
            roughness=50.0,
            slope_std=15.0,
            roughness_threshold=100.0,
            slope_std_threshold=30.0,
        )
        # 50/100 = 0.5, 15/30 = 0.5, RMS = 0.5, consistency = 0.5
        assert result == pytest.approx(0.5, rel=0.01)

    def test_works_with_numpy_arrays(self):
        """Should work with numpy arrays element-wise."""
        from src.scoring.transforms import terrain_consistency

        roughness = np.array([0.0, 15.0, 30.0])
        slope_std = np.array([0.0, 5.0, 10.0])

        result = terrain_consistency(roughness, slope_std)

        assert isinstance(result, np.ndarray)
        assert result[0] == 1.0  # Both zero
        assert result[2] == 0.0  # Both at threshold
