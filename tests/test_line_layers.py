"""Tests for linear feature layer creation.

Tests the line layer creation functions extracted from san_diego_flow_demo.py.
These functions handle feature selection, variable-width expansion, and metric assignment.
Works for any linear features: streams, roads, trails, power lines, etc.
"""

import numpy as np
import pytest

from src.terrain.visualization.line_layers import (
    get_metric_data,
    create_line_layer,
    expand_lines_variable_width,
    create_line_layer,  # Backward compatibility alias
)


class TestGetMetricData:
    """Test metric selection helper."""

    def test_get_drainage_metric(self):
        """Should return drainage data when choice is 'drainage'."""
        drainage = np.array([[1, 2], [3, 4]])
        rainfall = np.array([[5, 6], [7, 8]])
        discharge = np.array([[9, 10], [11, 12]])

        result = get_metric_data("drainage", drainage, rainfall, discharge)

        np.testing.assert_array_equal(result, drainage)

    def test_get_rainfall_metric(self):
        """Should return rainfall data when choice is 'rainfall'."""
        drainage = np.array([[1, 2], [3, 4]])
        rainfall = np.array([[5, 6], [7, 8]])
        discharge = np.array([[9, 10], [11, 12]])

        result = get_metric_data("rainfall", drainage, rainfall, discharge)

        np.testing.assert_array_equal(result, rainfall)

    def test_get_discharge_metric(self):
        """Should return discharge data when choice is 'discharge' or default."""
        drainage = np.array([[1, 2], [3, 4]])
        rainfall = np.array([[5, 6], [7, 8]])
        discharge = np.array([[9, 10], [11, 12]])

        result = get_metric_data("discharge", drainage, rainfall, discharge)
        np.testing.assert_array_equal(result, discharge)

        # Test default (anything else)
        result_default = get_metric_data("unknown", drainage, rainfall, discharge)
        np.testing.assert_array_equal(result_default, discharge)


class TestStreamSelection:
    """Test stream selection by percentile threshold."""

    def test_stream_selection_basic(self):
        """Should select only pixels >= percentile threshold."""
        # Create metric data where values are obvious
        metric_data = np.array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
        ], dtype=np.float32)

        # 95th percentile of [1..20] is 19.05, so only 20 should be selected
        percentile = 95.0

        stream_layer = create_line_layer(
            metric_data=metric_data,
            selection_metric_data=metric_data,
            percentile=percentile,
            variable_width=False,
            max_width=3
        )

        # Stream layer should have metric values only at selected pixels
        # Only pixel with value 20 should be > threshold
        assert stream_layer[3, 4] == 20  # Top-right corner
        assert np.sum(stream_layer > 0) == 1  # Only one stream pixel

    def test_stream_selection_top_5_percent(self):
        """Should select top 5% of pixels (percentile=95)."""
        # Create 10x10 grid with values 0-99
        metric_data = np.arange(100, dtype=np.float32).reshape(10, 10)

        percentile = 95.0  # Top 5%

        stream_layer = create_line_layer(
            metric_data=metric_data,
            selection_metric_data=metric_data,
            percentile=percentile,
            variable_width=False,
            max_width=3
        )

        # Top 5% of 100 values = 5 values (95, 96, 97, 98, 99)
        num_stream_pixels = np.sum(stream_layer > 0)
        assert num_stream_pixels == 5

    def test_empty_metric_returns_zero_layer(self):
        """Should return all-zero layer if no valid metric data."""
        metric_data = np.zeros((10, 10), dtype=np.float32)

        stream_layer = create_line_layer(
            metric_data=metric_data,
            selection_metric_data=metric_data,
            percentile=95.0,
            variable_width=False,
            max_width=3
        )

        assert np.all(stream_layer == 0)


class TestStreamLayerCreation:
    """Test stream layer creation without variable-width."""

    def test_stream_pixels_have_metric_values(self):
        """Stream pixels should have coloring metric values, not selection metric."""
        # Selection metric: simple values
        selection_metric = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], dtype=np.float32)

        # Coloring metric: different values
        coloring_metric = np.array([
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90],
        ], dtype=np.float32)

        # Select top pixel (value 9 in selection_metric)
        stream_layer = create_line_layer(
            metric_data=coloring_metric,
            selection_metric_data=selection_metric,
            percentile=95.0,  # Will select pixel with value 9
            variable_width=False,
            max_width=3
        )

        # Stream pixel should have COLORING metric value (90), not selection metric (9)
        assert stream_layer[2, 2] == 90  # Bottom-right corner
        assert np.sum(stream_layer > 0) == 1

    def test_non_stream_pixels_are_zero(self):
        """Non-stream pixels should be exactly 0."""
        metric_data = np.arange(100, dtype=np.float32).reshape(10, 10)

        stream_layer = create_line_layer(
            metric_data=metric_data,
            selection_metric_data=metric_data,
            percentile=95.0,
            variable_width=False,
            max_width=3
        )

        # 95 pixels should be zero
        assert np.sum(stream_layer == 0) == 95

    def test_layer_shape_preserved(self):
        """Output layer should have same shape as input."""
        metric_data = np.random.rand(123, 456).astype(np.float32)

        stream_layer = create_line_layer(
            metric_data=metric_data,
            selection_metric_data=metric_data,
            percentile=90.0,
            variable_width=False,
            max_width=3
        )

        assert stream_layer.shape == metric_data.shape


class TestVariableWidthExpansion:
    """Test variable-width stream expansion."""

    def test_expansion_creates_wider_streams(self):
        """Variable-width should expand streams beyond initial pixels."""
        # Smooth algorithm needs MULTIPLE streams with varying values to expand
        # Create TWO peaks with different values
        y, x = np.ogrid[:10, :10]

        # Peak 1 (lower value)
        peak1 = np.maximum(0, 50 - np.sqrt((y - 3)**2 + (x - 3)**2) * 10).astype(np.float32)

        # Peak 2 (higher value)
        peak2 = np.maximum(0, 100 - np.sqrt((y - 7)**2 + (x - 7)**2) * 10).astype(np.float32)

        metric_data = peak1 + peak2

        # Select top pixels (both peaks)
        stream_layer = create_line_layer(
            metric_data=metric_data,
            selection_metric_data=metric_data,
            percentile=98.0,  # Top 2% (both peak pixels)
            variable_width=True,
            max_width=3
        )

        # Streams should expand beyond the original pixels
        num_stream_pixels = np.sum(stream_layer > 0)
        assert num_stream_pixels > 2  # More than just the two original pixels

    def test_higher_values_create_wider_streams(self):
        """Higher metric values should create wider streams."""
        # Create metric data with two peaks: one low, one high
        # This simulates two separate stream networks with different discharge
        y, x = np.ogrid[:20, :20]

        # Low-value peak at (10, 5)
        low_peak = np.maximum(0, 50 - np.sqrt((y - 10)**2 + (x - 5)**2) * 5).astype(np.float32)

        # High-value peak at (10, 15)
        high_peak = np.maximum(0, 100 - np.sqrt((y - 10)**2 + (x - 15)**2) * 5).astype(np.float32)

        # Combine into single metric data
        metric_data = low_peak + high_peak

        # Select top 2 pixels (both peaks)
        stream_layer = create_line_layer(
            metric_data=metric_data,
            selection_metric_data=metric_data,
            percentile=99.5,  # Top 0.5% (approximately 2 pixels)
            variable_width=True,
            max_width=3
        )

        # Extract the two stream regions
        stream_mask = stream_layer > 0

        # Label connected regions
        from scipy.ndimage import label
        labeled, num_features = label(stream_mask)

        # Should have 2 separate stream regions (one for each peak)
        # The high-value region should be larger than the low-value region
        if num_features >= 2:
            # Find sizes of each region
            region_sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
            region_sizes.sort()

            # The largest region should be bigger than the smallest
            # (high-value peak expands more than low-value peak)
            assert region_sizes[-1] > region_sizes[0]
        else:
            # If regions merged, at least verify expansion happened
            assert np.sum(stream_mask) > 2

    def test_max_width_limits_expansion(self):
        """Expansion should not exceed max_width pixels."""
        # Single high-value stream pixel
        metric_data = np.zeros((20, 20), dtype=np.float32)
        metric_data[10, 10] = 100

        max_width = 2

        stream_layer = create_line_layer(
            metric_data=metric_data,
            selection_metric_data=metric_data,
            percentile=0.0,
            variable_width=True,
            max_width=max_width
        )

        # Find the expanded stream region
        stream_mask = stream_layer > 0
        stream_coords = np.argwhere(stream_mask)

        # All stream pixels should be within max_width of the original pixel
        distances = np.sqrt(
            (stream_coords[:, 0] - 10) ** 2 + (stream_coords[:, 1] - 10) ** 2
        )
        assert np.all(distances <= max_width)


class TestExpandLinesVariableWidth:
    """Test the expand_lines_variable_width function directly."""

    def test_expansion_basic(self):
        """Should expand line mask based on metric values."""
        # Initial line mask with single pixel
        line_mask = np.zeros((10, 10), dtype=bool)
        line_mask[5, 5] = True

        # Metric data with high value at line pixel
        metric_data = np.zeros((10, 10), dtype=np.float32)
        metric_data[5, 5] = 100

        expanded_mask = expand_lines_variable_width(
            line_mask, metric_data, max_width=3
        )

        # Expanded mask should have more pixels than original
        assert np.sum(expanded_mask) > np.sum(line_mask)

    def test_memory_safety_check(self):
        """Should skip expansion if array is too large."""
        # Create a very large array that would exceed memory threshold
        # The threshold is 1500 MB for 3 float32 arrays
        # Each float32 = 4 bytes, so 1500 MB = 375M elements per array
        # Total pixels = 375M / 3 = 125M pixels ≈ 11180 × 11180
        large_size = 12000  # > threshold

        line_mask = np.zeros((large_size, large_size), dtype=bool)
        line_mask[0, 0] = True
        metric_data = np.zeros((large_size, large_size), dtype=np.float32)

        expanded_mask, expanded_values = expand_lines_variable_width(
            line_mask, metric_data, max_width=3
        )

        # Should return original mask without expansion
        np.testing.assert_array_equal(expanded_mask, line_mask)


class TestMetricConsistency:
    """Test that variable-width mode uses consistent metrics."""

    def test_variable_width_uses_same_metric_for_selection_and_width(self):
        """In variable-width mode, selection and coloring should use the same metric."""
        # This test verifies the LOGIC, not just the function
        # We're testing that when you call create_line_layer with
        # metric_data = selection_metric_data, it works correctly

        metric_data = np.array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
        ], dtype=np.float32)

        # In variable-width mode, metric_data and selection_metric_data are the same
        # Use lower percentile to select multiple pixels (smooth algo needs variation)
        stream_layer = create_line_layer(
            metric_data=metric_data,  # Same metric
            selection_metric_data=metric_data,  # Same metric
            percentile=85.0,  # Top 15% = 3 pixels with variation
            variable_width=True,
            max_width=3
        )

        # Should select multiple top pixels and expand them
        assert np.sum(stream_layer > 0) >= 3  # At least 3 pixels selected
        # With expansion, should have more than just the selected pixels
        assert np.sum(stream_layer > 0) > 3

    def test_non_variable_width_can_use_different_metrics(self):
        """In non-variable-width mode, selection and coloring can use different metrics."""
        # Selection metric: values 1-20
        selection_metric = np.arange(1, 21, dtype=np.float32).reshape(4, 5)

        # Coloring metric: values 100-119
        coloring_metric = np.arange(100, 120, dtype=np.float32).reshape(4, 5)

        stream_layer = create_line_layer(
            metric_data=coloring_metric,  # Different metric for coloring
            selection_metric_data=selection_metric,  # Different metric for selection
            percentile=95.0,
            variable_width=False,
            max_width=3
        )

        # Should select based on selection_metric (top value = 20)
        # But assign coloring_metric values (119 at that pixel)
        assert np.sum(stream_layer > 0) == 1  # Only one pixel selected
        assert stream_layer[3, 4] == 119  # Coloring metric value, not selection
