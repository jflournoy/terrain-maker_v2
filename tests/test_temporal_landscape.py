"""Tests for temporal landscape data functions (build_combined_matrix, build_ridge_dem)."""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers to build mock data matching build_timeseries() output structure
# ---------------------------------------------------------------------------

def _make_mock_data(n_seasons: int = 3, n_days: int = 50) -> dict:
    """
    Create a mock data dict matching the structure returned by build_timeseries().

    Each season gets a simple bell-shaped score profile peaking mid-season.
    """
    seasons = [f"{2010 + i}-{2011 + i}" for i in range(n_seasons)]
    doy = {}
    median_depth_score = {}
    median_coverage_score = {}

    for s in seasons:
        d = np.arange(0, n_days, dtype=np.int32)
        # Bell-shaped scores: peak at center of season
        t = np.linspace(0, np.pi, n_days)
        depth = np.sin(t).astype(np.float32) * 0.8
        coverage = np.sin(t).astype(np.float32) * 0.9
        doy[s] = d
        median_depth_score[s] = depth
        median_coverage_score[s] = coverage

    return {
        "seasons": seasons,
        "doy": doy,
        "median_depth_score": median_depth_score,
        "median_coverage_score": median_coverage_score,
    }


def _make_mock_data_with_nan(n_seasons: int = 3, n_days: int = 50) -> dict:
    """Mock data where one season has NaN values at the tails."""
    data = _make_mock_data(n_seasons, n_days)
    # Add NaN to first season's edges
    first_season = data["seasons"][0]
    data["median_depth_score"][first_season][:5] = np.nan
    data["median_depth_score"][first_season][-5:] = np.nan
    return data


# ---------------------------------------------------------------------------
# Tests for build_combined_matrix
# ---------------------------------------------------------------------------

class TestBuildCombinedMatrix:
    def test_shape(self):
        from examples.xc_skiing_temporal import build_combined_matrix

        data = _make_mock_data(n_seasons=3, n_days=50)
        matrix, seasons, median = build_combined_matrix(data)

        assert matrix.shape == (3, 182)
        assert len(seasons) == 3
        assert median.shape == (182,)

    def test_scores_range(self):
        from examples.xc_skiing_temporal import build_combined_matrix

        data = _make_mock_data(n_seasons=5)
        matrix, _, median = build_combined_matrix(data)

        assert np.all(matrix >= 0.0)
        assert np.all(matrix <= 1.0)
        assert np.all(median >= 0.0)
        assert np.all(median <= 1.0)

    def test_no_nan_in_output(self):
        from examples.xc_skiing_temporal import build_combined_matrix

        data = _make_mock_data_with_nan(n_seasons=3)
        matrix, _, median = build_combined_matrix(data)

        assert not np.any(np.isnan(matrix))
        assert not np.any(np.isnan(median))

    def test_seasons_sorted_chronologically(self):
        from examples.xc_skiing_temporal import build_combined_matrix

        data = _make_mock_data(n_seasons=4)
        # Shuffle the seasons list
        data["seasons"] = list(reversed(data["seasons"]))
        _, seasons, _ = build_combined_matrix(data)

        assert seasons == sorted(seasons)

    def test_custom_n_days(self):
        from examples.xc_skiing_temporal import build_combined_matrix

        data = _make_mock_data(n_seasons=2, n_days=100)
        matrix, _, median = build_combined_matrix(data, n_days=100)

        assert matrix.shape == (2, 100)
        assert median.shape == (100,)


# ---------------------------------------------------------------------------
# Tests for build_ridge_dem
# ---------------------------------------------------------------------------

class TestBuildRidgeDem:
    def _make_inputs(self, n_seasons=3, n_days=182):
        """Create simple season_matrix and median_scores for testing."""
        matrix = np.full((n_seasons, n_days), 0.5, dtype=np.float32)
        median = np.full(n_days, 0.5, dtype=np.float32)
        return matrix, median

    def test_shape(self):
        from examples.xc_skiing_temporal import build_ridge_dem

        matrix, median = self._make_inputs(n_seasons=3)
        dem, color = build_ridge_dem(
            matrix, median, ridge_rows=7, gap_rows=3, median_width_mult=2,
        )

        # Total rows: 3 seasons * (7+3) + 2*7 + 3 = 30 + 14 + 3 = 47
        assert dem.shape == (47, 182)
        assert color.shape == (47, 182)

    def test_valley_zeros(self):
        from examples.xc_skiing_temporal import build_ridge_dem

        matrix, median = self._make_inputs(n_seasons=3)
        dem, color = build_ridge_dem(
            matrix, median, ridge_rows=7, gap_rows=3, median_width_mult=2,
        )

        # Find gap rows (they should be zero in both dem and color)
        # First ridge: rows 0-6, gap: rows 7-9
        # The gap rows in the DEM should be exactly zero
        for row in range(7, 10):  # first gap after first ridge
            assert np.all(dem[row] == 0.0), f"Gap row {row} in dem not zero"
            assert np.all(color[row] == 0.0), f"Gap row {row} in color not zero"

    def test_ridge_peak_at_center(self):
        from examples.xc_skiing_temporal import build_ridge_dem

        # Uniform score=1.0 so bell shape is the only variation
        n_days = 182
        matrix = np.ones((3, n_days), dtype=np.float32)
        median = np.ones(n_days, dtype=np.float32)
        dem, _ = build_ridge_dem(
            matrix, median, ridge_rows=7, gap_rows=3, median_width_mult=2,
        )

        # First ridge spans rows 0-6; peak should be at row 3 (center of 7)
        ridge_slice = dem[0:7, 0]  # first column of first ridge
        peak_row = np.argmax(ridge_slice)
        assert peak_row == 3, f"Peak at row {peak_row}, expected 3"

    def test_median_at_center_position(self):
        from examples.xc_skiing_temporal import build_ridge_dem

        n_seasons = 4
        matrix = np.ones((n_seasons, 182), dtype=np.float32)
        median = np.ones(182, dtype=np.float32)
        dem, _ = build_ridge_dem(
            matrix, median,
            ridge_rows=7, gap_rows=3,
            median_width_mult=2, median_height_scale=1.5,
        )

        # With 4 seasons (first_half=2): layout is
        # Season 0: rows 0-6, gap 7-9
        # Season 1: rows 10-16, gap 17-19
        # Median: rows 20-33 (14 rows = 7*2), gap 34-36
        # Season 2: rows 37-43, gap 44-46
        # Season 3: rows 47-53, gap 54-56
        # Total: 4*10 + 14 + 3 = 57

        # The median ridge should be taller (1.5x) than season ridges
        season_0_peak = dem[3, 90]  # center of first ridge, middle day
        median_peak_row = 20 + 7  # center of 14-row median ridge (index 7 of 14)
        median_peak = dem[median_peak_row, 90]

        assert median_peak > season_0_peak, (
            f"Median peak {median_peak} should be > season peak {season_0_peak}"
        )

    def test_no_nan(self):
        from examples.xc_skiing_temporal import build_ridge_dem

        matrix, median = self._make_inputs(n_seasons=5)
        dem, color = build_ridge_dem(matrix, median)

        assert not np.any(np.isnan(dem))
        assert not np.any(np.isnan(color))

    def test_color_flat_across_ridge(self):
        from examples.xc_skiing_temporal import build_ridge_dem

        # Use varying scores so we can check color is flat but not zero
        n_days = 182
        matrix = np.random.RandomState(42).rand(3, n_days).astype(np.float32)
        median = np.nanmedian(matrix, axis=0)
        dem, color = build_ridge_dem(
            matrix, median, ridge_rows=7, gap_rows=3, median_width_mult=2,
        )

        # All rows within the first ridge (0-6) should have the same color values
        for row in range(1, 7):
            np.testing.assert_array_equal(
                color[0], color[row],
                err_msg=f"Color row {row} differs from row 0 within first ridge",
            )

    def test_dem_varies_across_ridge(self):
        """DEM values should vary across a ridge (bell shape), unlike color."""
        from examples.xc_skiing_temporal import build_ridge_dem

        matrix = np.ones((3, 182), dtype=np.float32) * 0.8
        median = np.ones(182, dtype=np.float32) * 0.8
        dem, color = build_ridge_dem(
            matrix, median, ridge_rows=7, gap_rows=3, median_width_mult=2,
        )

        # DEM edge rows (0, 6) should be less than center row (3) for first ridge
        assert dem[0, 90] < dem[3, 90], "DEM edge should be less than center"
        assert dem[6, 90] < dem[3, 90], "DEM edge should be less than center"

        # Color should be constant across the ridge
        assert color[0, 90] == color[3, 90] == color[6, 90]
