"""
Test for variable-width line layer value propagation.

Tests that expanded pixels get metric values from their nearest line pixel,
not from their own location (which is often 0).
"""

import numpy as np
import pytest
from src.terrain.visualization.line_layers import create_line_layer


def test_variable_width_propagates_metric_values():
    """
    Expanded pixels should get values from nearest stream pixel.

    With smooth expansion, need multiple streams with varying values
    to trigger expansion. Single streams don't expand (no variation).
    """
    # Create 20×20 grid
    shape = (20, 20)

    # Selection metric: two streams with different values
    selection_data = np.zeros(shape, dtype=np.float32)
    selection_data[10, 10] = 1000.0  # Stream 1
    selection_data[15, 15] = 1000.0  # Stream 2

    # Color metric: different values for each stream
    color_data = np.zeros(shape, dtype=np.float32)
    color_data[10, 10] = 100.0  # Stream 1 (min → gets min_width = 1px)
    color_data[15, 15] = 200.0  # Stream 2 (max → gets max_width = 3px)

    # Create stream layer with variable width
    stream_layer = create_line_layer(
        metric_data=color_data,
        selection_metric_data=selection_data,
        percentile=0.0,  # Select all non-zero pixels
        variable_width=True,
        max_width=3,  # Expand by 3 pixels max
    )

    # The original stream pixels should have their values
    assert stream_layer[10, 10] == 100.0, "Stream 1 should have value 100.0"
    assert stream_layer[15, 15] == 200.0, "Stream 2 should have value 200.0"

    # Stream 1 (value=100, min) gets min_width=1px → 1 pixel expansion
    assert stream_layer[10, 11] == 100.0, "Pixel 1 away from stream 1 should get 100.0"

    # Stream 2 (value=200, max) gets max_width=3px → 3 pixel expansion
    assert stream_layer[15, 16] == 200.0, "Pixel 1 away from stream 2 should get 200.0"
    assert stream_layer[15, 17] == 200.0, "Pixel 2 away from stream 2 should get 200.0"
    assert stream_layer[15, 18] == 200.0, "Pixel 3 away from stream 2 should get 200.0"

    # Pixels far from streams should be 0
    assert stream_layer[5, 5] == 0.0, "Pixel far from streams should be 0"


def test_variable_width_with_multiple_streams():
    """
    Test that each expanded pixel gets the value from its nearest stream.

    Two streams with different values - expanded pixels should get the
    value from whichever stream is nearest.

    Note: Variable-width means lower values → narrower streams!
    """
    shape = (30, 30)

    # Three streams with different values (need 3 to get mid-range widths)
    selection_data = np.zeros(shape, dtype=np.float32)
    selection_data[5, 5] = 1000.0  # Stream 0 (min value → min width)
    selection_data[10, 10] = 1000.0  # Stream 1 (mid value → mid width)
    selection_data[20, 20] = 1000.0  # Stream 2 (max value → max width)

    # Color values: 100 (min), 150 (mid), 200 (max)
    color_data = np.zeros(shape, dtype=np.float32)
    color_data[5, 5] = 100.0  # Stream 0 value (min → 0px width)
    color_data[10, 10] = 150.0  # Stream 1 value (mid-range → 1.5px width)
    color_data[20, 20] = 200.0  # Stream 2 value (max → 3px width)

    stream_layer = create_line_layer(
        metric_data=color_data,
        selection_metric_data=selection_data,
        percentile=0.0,
        variable_width=True,
        max_width=3,
    )

    # Original stream pixels
    assert stream_layer[5, 5] == 100.0, "Stream 0 should have value 100.0"
    assert stream_layer[10, 10] == 150.0, "Stream 1 should have value 150.0"
    assert stream_layer[20, 20] == 200.0, "Stream 2 should have value 200.0"

    # Stream 0 (value=100, min) gets min_width=1px → 1 pixel expansion
    assert stream_layer[5, 6] == 100.0, "Stream 0 (min value) gets min_width=1px expansion"

    # Stream 1 (value=150, mid) gets width ≈ 2px
    assert stream_layer[10, 11] == 150.0, "Pixel 1 away from stream 1 should get 150.0"
    assert stream_layer[10, 12] == 150.0, "Pixel 2 away from stream 1 should get 150.0"

    # Stream 2 (value=200, max) gets max_width=3px
    assert stream_layer[20, 21] == 200.0, "Pixel 1 away from stream 2 should get 200.0"
    assert stream_layer[20, 22] == 200.0, "Pixel 2 away from stream 2 should get 200.0"
    assert stream_layer[20, 23] == 200.0, "Pixel 3 away from stream 2 should get 200.0"

    # Midpoint between streams should be 0 (outside expansion radius)
    assert stream_layer[15, 15] == 0.0, "Midpoint should be 0 (no streams nearby)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
