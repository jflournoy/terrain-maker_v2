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
        matrix, seasons, median, q1, q3 = build_combined_matrix(data)

        assert matrix.shape == (3, 243)
        assert len(seasons) == 3
        assert median.shape == (243,)

    def test_scores_range(self):
        from examples.xc_skiing_temporal import build_combined_matrix

        data = _make_mock_data(n_seasons=5)
        matrix, _, median, _, _ = build_combined_matrix(data)

        assert np.all(matrix >= 0.0)
        assert np.all(matrix <= 1.0)
        assert np.all(median >= 0.0)
        assert np.all(median <= 1.0)

    def test_no_nan_in_output(self):
        from examples.xc_skiing_temporal import build_combined_matrix

        data = _make_mock_data_with_nan(n_seasons=3)
        matrix, _, median, _, _ = build_combined_matrix(data)

        assert not np.any(np.isnan(matrix))
        assert not np.any(np.isnan(median))

    def test_seasons_sorted_chronologically(self):
        from examples.xc_skiing_temporal import build_combined_matrix

        data = _make_mock_data(n_seasons=4)
        # Shuffle the seasons list
        data["seasons"] = list(reversed(data["seasons"]))
        _, seasons, _, _, _ = build_combined_matrix(data)

        assert seasons == sorted(seasons)

    def test_custom_n_days(self):
        from examples.xc_skiing_temporal import build_combined_matrix

        data = _make_mock_data(n_seasons=2, n_days=100)
        matrix, _, median, _, _ = build_combined_matrix(data, n_days=100)

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

    def test_color_ramps_with_height(self):
        """Color should ramp from 0 at base to full score at ridge peak."""
        from examples.xc_skiing_temporal import build_ridge_dem

        n_days = 182
        matrix = np.ones((3, n_days), dtype=np.float32) * 0.8
        median = np.ones(n_days, dtype=np.float32) * 0.8
        dem, color = build_ridge_dem(
            matrix, median, ridge_rows=7, gap_rows=3, median_width_mult=2,
        )

        # First ridge: rows 0-6, peak at center row 3
        # Color should be highest at the peak and lower at the edges
        col = 90
        peak_color = color[3, col]
        edge_color = color[0, col]
        assert peak_color > edge_color, (
            f"Peak color ({peak_color}) should be > edge color ({edge_color})"
        )
        # Peak color should equal the score (frac=1 at peak)
        assert peak_color > 0, "Peak color should be > 0"

    def test_dem_varies_across_ridge(self):
        """DEM values should vary across a ridge (bell shape)."""
        from examples.xc_skiing_temporal import build_ridge_dem

        matrix = np.ones((3, 182), dtype=np.float32) * 0.8
        median = np.ones(182, dtype=np.float32) * 0.8
        dem, color = build_ridge_dem(
            matrix, median, ridge_rows=7, gap_rows=3, median_width_mult=2,
        )

        # DEM edge rows (0, 6) should be less than center row (3) for first ridge
        assert dem[0, 90] < dem[3, 90], "DEM edge should be less than center"
        assert dem[6, 90] < dem[3, 90], "DEM edge should be less than center"

        # Color should also ramp (not flat) — lower at edges, higher at center
        assert color[0, 90] < color[3, 90], "Color at edge should be less than at center"


class TestIqrRidge:
    """Tests for the IQR-envelope summary ridge (cuesta path with q1/q3)."""

    def test_single_summary_ridge(self):
        """Summary should be one IQR ridge, not 3 separate ridges."""
        from examples.xc_skiing_temporal import build_ridge_dem

        n_days = 182
        matrix = np.ones((4, n_days), dtype=np.float32) * 0.5
        median = np.ones(n_days, dtype=np.float32) * 0.5
        q1 = np.ones(n_days, dtype=np.float32) * 0.3
        q3 = np.ones(n_days, dtype=np.float32) * 0.7

        dem, color = build_ridge_dem(
            matrix, median, q1_scores=q1, q3_scores=q3,
            ridge_rows=2, gap_rows=0, tail_rows=6, shear=20.0,
        )

        # The summary block should be ridge_rows + gap_rows = 2 (one ridge),
        # not 3 * (ridge_rows + gap_rows) = 6 (three ridges).
        # Total = shear_pad + 2 + summary_gap(6) + shear_pad + 4*2
        expected_rows = 20 + 2 + 6 + 20 + 8
        assert dem.shape == (expected_rows, n_days)

    def test_iqr_color_peaks_at_median(self):
        """Peak color in the IQR summary ridge should be the median score."""
        from examples.xc_skiing_temporal import build_ridge_dem

        n_days = 50
        matrix = np.ones((3, n_days), dtype=np.float32) * 0.4
        median = np.ones(n_days, dtype=np.float32) * 0.3
        q1 = np.ones(n_days, dtype=np.float32) * 0.2
        q3 = np.ones(n_days, dtype=np.float32) * 0.4

        dem, color = build_ridge_dem(
            matrix, median, q1_scores=q1, q3_scores=q3,
            ridge_rows=2, gap_rows=0, tail_rows=6, shear=10.0,
        )

        # Color ramps with height: 0 at base, median at peak.
        # The max color in the summary area should approach median score.
        shear_pad = 10
        summary_colors = color[:shear_pad + 2, :]
        peak_color = summary_colors.max()
        assert peak_color > 0, "IQR ridge should have non-zero color"
        assert peak_color <= 0.3 + 1e-5, (
            f"Peak color ({peak_color}) should not exceed median (0.3)"
        )
        # Color should ramp (not all the same value)
        active = summary_colors > 0
        if np.any(active):
            assert summary_colors[active].min() < peak_color, (
                "Color should ramp from low to high, not be flat"
            )

    def test_wider_iqr_makes_wider_ridge(self):
        """Wider IQR should produce a wider (more active rows) summary ridge."""
        from examples.xc_skiing_temporal import build_ridge_dem

        n_days = 50
        matrix = np.ones((3, n_days), dtype=np.float32) * 0.5
        median = np.ones(n_days, dtype=np.float32) * 0.5

        # Narrow IQR
        q1_narrow = np.ones(n_days, dtype=np.float32) * 0.45
        q3_narrow = np.ones(n_days, dtype=np.float32) * 0.55

        # Wide IQR
        q1_wide = np.ones(n_days, dtype=np.float32) * 0.2
        q3_wide = np.ones(n_days, dtype=np.float32) * 0.8

        dem_narrow, _ = build_ridge_dem(
            matrix, median, q1_scores=q1_narrow, q3_scores=q3_narrow,
            ridge_rows=2, gap_rows=0, tail_rows=6, shear=20.0,
        )
        dem_wide, _ = build_ridge_dem(
            matrix, median, q1_scores=q1_wide, q3_scores=q3_wide,
            ridge_rows=2, gap_rows=0, tail_rows=6, shear=20.0,
        )

        # Count active summary rows (before season start)
        shear_pad = 20
        narrow_active = np.any(dem_narrow[:shear_pad + 2, :] > 0, axis=1).sum()
        wide_active = np.any(dem_wide[:shear_pad + 2, :] > 0, axis=1).sum()

        assert wide_active > narrow_active, (
            f"Wide IQR ({wide_active} rows) should have more rows than "
            f"narrow IQR ({narrow_active} rows)"
        )

    def test_summary_color_scale_boosts_iqr_color(self):
        """summary_color_scale > 1 should increase IQR ridge color values."""
        from examples.xc_skiing_temporal import build_ridge_dem

        n_days = 50
        matrix = np.ones((3, n_days), dtype=np.float32) * 0.5
        median = np.ones(n_days, dtype=np.float32) * 0.3
        q1 = np.ones(n_days, dtype=np.float32) * 0.2
        q3 = np.ones(n_days, dtype=np.float32) * 0.4

        _, color_1x = build_ridge_dem(
            matrix, median, q1_scores=q1, q3_scores=q3,
            ridge_rows=2, gap_rows=0, tail_rows=6, shear=10.0,
            summary_color_scale=1.0,
        )
        _, color_2x = build_ridge_dem(
            matrix, median, q1_scores=q1, q3_scores=q3,
            ridge_rows=2, gap_rows=0, tail_rows=6, shear=10.0,
            summary_color_scale=2.0,
        )

        # Summary area: rows before the season group
        shear_pad = 10
        summary_1x = color_1x[:shear_pad + 2, :]
        summary_2x = color_2x[:shear_pad + 2, :]

        peak_1x = summary_1x.max()
        peak_2x = summary_2x.max()

        assert peak_2x > peak_1x, (
            f"2x scale peak ({peak_2x}) should exceed 1x peak ({peak_1x})"
        )
        # Season ridges should be unaffected
        season_1x = color_1x[shear_pad + 2 + 6 + shear_pad:, :]
        season_2x = color_2x[shear_pad + 2 + 6 + shear_pad:, :]
        np.testing.assert_array_equal(
            season_1x, season_2x,
            err_msg="Season ridge colors should be unaffected by summary_color_scale",
        )


class TestColScale:
    """Tests for col_scale score upsampling (line-graph fidelity)."""

    def test_col_scale_no_peak_sag(self):
        """Score upsampling (col_scale in build_ridge_dem) should produce
        higher intermediate peaks than post-hoc 2D zoom, because each column
        gets its own ramp profile at the interpolated score."""
        from examples.xc_skiing_temporal import build_ridge_dem
        from scipy.ndimage import zoom

        # Two adjacent days with very different scores + shear
        # to maximize the sag from 2D profile interpolation
        n_days = 10
        matrix = np.zeros((1, n_days), dtype=np.float32)
        matrix[0, 3] = 0.2   # low score
        matrix[0, 4] = 0.8   # high score (big shear shift)
        median = matrix[0].copy()

        col_scale = 4

        # New approach: upsample scores first, then build ridges
        dem_new, _ = build_ridge_dem(
            matrix, median,
            ridge_rows=2, gap_rows=1, tail_rows=6, shear=10.0,
            row_scale=4, col_scale=col_scale,
        )

        # Old approach: build at original resolution, then 2D zoom
        dem_old, _ = build_ridge_dem(
            matrix, median,
            ridge_rows=2, gap_rows=1, tail_rows=6, shear=10.0,
            row_scale=4, col_scale=1,
        )
        dem_old_zoomed = np.clip(zoom(dem_old, (1, col_scale), order=1), 0, None)

        # Compare intermediate columns between day 3 and day 4.
        # The new approach should have equal or higher peaks because each
        # intermediate column gets its own ramp at the interpolated score,
        # rather than blending two offset 2D profiles.
        col3_new = 3 * col_scale
        col4_new = col3_new + col_scale
        sag_count = 0
        for col in range(col3_new + 1, col4_new):
            peak_new = dem_new[:, col].max()
            # Map to old_zoomed column (different output sizes)
            col_old = int(round(col * dem_old_zoomed.shape[1] / dem_new.shape[1]))
            col_old = min(col_old, dem_old_zoomed.shape[1] - 1)
            peak_old = dem_old_zoomed[:, col_old].max()
            if peak_new > peak_old + 0.001:
                sag_count += 1

        assert sag_count > 0, (
            "Score upsampling should produce higher intermediate peaks "
            "than 2D zoom for columns between days with different shear"
        )

    def test_col_scale_output_shape(self):
        """col_scale should multiply the number of columns."""
        from examples.xc_skiing_temporal import build_ridge_dem

        n_days = 20
        matrix = np.ones((3, n_days), dtype=np.float32) * 0.5
        median = np.ones(n_days, dtype=np.float32) * 0.5

        dem_1x, _ = build_ridge_dem(
            matrix, median, ridge_rows=7, gap_rows=3, col_scale=1,
        )
        dem_4x, _ = build_ridge_dem(
            matrix, median, ridge_rows=7, gap_rows=3, col_scale=4,
        )

        assert dem_1x.shape[1] == n_days
        # np.interp upsampling: (n_days - 1) * col_scale + 1
        assert dem_4x.shape[1] == (n_days - 1) * 4 + 1
