"""
Targeted tests for slope_statistics edge cases.

Tests the reshape bug where edge tiles have mismatched sizes.
"""

import numpy as np
import pytest
from src.snow.slope_statistics import aggregate_by_geographic_mapping


class TestAggregateByGeographicMapping:
    """Test edge cases in aggregate_by_geographic_mapping."""

    def test_exact_fit(self):
        """Test when input exactly matches expected size."""
        # 300x600 input with stride 30x30 = 10x20 output
        slope = np.random.rand(300, 600).astype(np.float32)
        aspect = np.random.rand(300, 600).astype(np.float32)
        elevation = np.random.rand(300, 600).astype(np.float32)

        result = aggregate_by_geographic_mapping(
            slope, aspect, elevation,
            row_stride=30, col_stride=30,
            output_shape=(10, 20),
        )

        assert result["mean"].shape == (10, 20)
        assert result["max"].shape == (10, 20)

    def test_input_smaller_than_expected(self):
        """Test when input is smaller than expected (edge tile)."""
        # Expected: 300x600 (10x20 output)
        # Actual: 295x595 (slightly smaller - edge tile)
        slope = np.random.rand(295, 595).astype(np.float32)
        aspect = np.random.rand(295, 595).astype(np.float32)
        elevation = np.random.rand(295, 595).astype(np.float32)

        result = aggregate_by_geographic_mapping(
            slope, aspect, elevation,
            row_stride=30, col_stride=30,
            output_shape=(10, 20),  # Expected, but input is smaller
        )

        # Should truncate to fit: 295//30=9 rows, 595//30=19 cols
        assert result["mean"].shape == (9, 19)

    def test_input_larger_than_expected(self):
        """Test when input is larger than expected."""
        # Expected: 300x600 (10x20 output)
        # Actual: 305x605 (slightly larger)
        slope = np.random.rand(305, 605).astype(np.float32)
        aspect = np.random.rand(305, 605).astype(np.float32)
        elevation = np.random.rand(305, 605).astype(np.float32)

        result = aggregate_by_geographic_mapping(
            slope, aspect, elevation,
            row_stride=30, col_stride=30,
            output_shape=(10, 20),
        )

        # Should truncate to expected output shape
        assert result["mean"].shape == (10, 20)

    def test_input_smaller_than_one_stride(self):
        """Test when input is smaller than a single stride (problematic case)."""
        # This is the edge case that causes out_h=0 or out_w=0
        slope = np.random.rand(25, 25).astype(np.float32)  # < 30 stride
        aspect = np.random.rand(25, 25).astype(np.float32)
        elevation = np.random.rand(25, 25).astype(np.float32)

        # This should not crash - it should handle gracefully
        result = aggregate_by_geographic_mapping(
            slope, aspect, elevation,
            row_stride=30, col_stride=30,
            output_shape=(1, 1),
        )

        # When input is too small, we can't produce any output
        # Function should handle this gracefully
        assert result["mean"].shape[0] == 0 or result["mean"].size == 0

    def test_detroit_realistic_edge_case(self):
        """
        Reproduce the actual Detroit error case.

        DEM shape: (36001, 39601)
        Target: 100x100 output
        Strides: ~360x396

        Edge tiles may have partial coverage.
        """
        # Simulate an edge tile that doesn't fill the expected output
        # This mimics what happens with 3005 rows when expecting 3000
        row_stride = 30
        col_stride = 30

        # Case: tile expects 100 output rows but only has 3005 source rows
        # 3005 // 30 = 100.166... -> should fit 100 rows (3000 pixels)
        # But 3005 - 3000 = 5 extra pixels that can't form a row
        slope = np.random.rand(3005, 600).astype(np.float32)
        aspect = np.random.rand(3005, 600).astype(np.float32)
        elevation = np.random.rand(3005, 600).astype(np.float32)

        result = aggregate_by_geographic_mapping(
            slope, aspect, elevation,
            row_stride=30, col_stride=30,
            output_shape=(100, 20),  # Expected based on tile planning
        )

        # Should produce 100x20 (fitting exactly) or truncate if needed
        assert result["mean"].shape == (100, 20)

    def test_zero_dimension_input(self):
        """Test edge case with zero-sized input."""
        slope = np.array([]).reshape(0, 100).astype(np.float32)
        aspect = np.array([]).reshape(0, 100).astype(np.float32)
        elevation = np.array([]).reshape(0, 100).astype(np.float32)

        result = aggregate_by_geographic_mapping(
            slope, aspect, elevation,
            row_stride=30, col_stride=30,
            output_shape=(10, 10),
        )

        # Should handle gracefully - empty output
        assert result["mean"].shape[0] == 0


class TestTileOutputMismatch:
    """Test output shape mismatches between tile stats and output slice."""

    def test_partial_tile_output(self):
        """
        Simulate edge tile where computed stats are smaller than expected.

        This is the Detroit bug: output_shape=(100, 100) but input only
        produces (99, 99) due to truncation at edges.
        """
        # Expected: 100x100 output based on tile planning
        # But input only has 2990 rows (99 * 30 = 2970, can fit 99 rows)
        slope = np.random.rand(2990, 2990).astype(np.float32)
        aspect = np.random.rand(2990, 2990).astype(np.float32)
        elevation = np.random.rand(2990, 2990).astype(np.float32)

        result = aggregate_by_geographic_mapping(
            slope, aspect, elevation,
            row_stride=30, col_stride=30,
            output_shape=(100, 100),  # Expected but can't fit
        )

        # Should truncate to what fits: 99x99
        assert result["mean"].shape == (99, 99)

    def test_tile_output_assignment_simulation(self):
        """
        Simulate the full tiled processing with edge effects.

        Edge tiles may produce smaller outputs than expected.
        The output array assignment must handle this.
        """
        # Setup: 100x100 output target
        output = {
            "slope_mean": np.zeros((100, 100), dtype=np.float32),
        }

        # Simulate an edge tile that expects to fill output[90:100, 90:100]
        # but only produces (9, 9) due to input truncation
        slope = np.random.rand(285, 285).astype(np.float32)  # 9*30=270, fits 9
        aspect = np.random.rand(285, 285).astype(np.float32)
        elevation = np.random.rand(285, 285).astype(np.float32)

        tile_stats = aggregate_by_geographic_mapping(
            slope, aspect, elevation,
            row_stride=30, col_stride=30,
            output_shape=(10, 10),  # Expected from out_slice
        )

        # tile_stats should be (9, 9), not (10, 10)
        assert tile_stats["mean"].shape == (9, 9)

        # Simulate the safe assignment pattern
        out_slice = (slice(90, 100), slice(90, 100))
        actual_h, actual_w = tile_stats["mean"].shape

        # Safe assignment using actual dimensions
        safe_slice = (
            slice(out_slice[0].start, out_slice[0].start + actual_h),
            slice(out_slice[1].start, out_slice[1].start + actual_w),
        )
        output["slope_mean"][safe_slice] = tile_stats["mean"]

        # Verify assignment worked
        assert output["slope_mean"][90:99, 90:99].sum() > 0
        # Last row/col should still be zero (not filled)
        assert output["slope_mean"][99, :].sum() == 0


class TestP95Calculation:
    """Test the 95th percentile calculation is correct."""

    def test_p95_respects_block_boundaries(self):
        """
        Verify p95 is computed correctly per output pixel block.

        The bug: reshape(out_h, out_w, -1) mixes data from different blocks.
        Each output pixel should only use values from its own block.
        """
        # Create data where each block has distinct, non-overlapping values
        # Block (i,j) has values in range [i*100 + j*10, i*100 + j*10 + 5]
        row_stride, col_stride = 10, 10
        out_h, out_w = 5, 6

        slope = np.zeros((out_h * row_stride, out_w * col_stride), dtype=np.float32)
        for i in range(out_h):
            for j in range(out_w):
                base_val = i * 100 + j * 10
                # Fill block with values in narrow range
                block_vals = base_val + np.random.rand(row_stride, col_stride) * 5
                slope[
                    i * row_stride:(i + 1) * row_stride,
                    j * col_stride:(j + 1) * col_stride
                ] = block_vals

        aspect = np.zeros_like(slope)
        elevation = np.zeros_like(slope)

        result = aggregate_by_geographic_mapping(
            slope, aspect, elevation,
            row_stride=row_stride, col_stride=col_stride,
            output_shape=(out_h, out_w),
        )

        # p95 for block (i,j) should be approximately i*100 + j*10 + 4.75 (95% of 5)
        # It should NOT contain values from other blocks
        for i in range(out_h):
            for j in range(out_w):
                expected_base = i * 100 + j * 10
                expected_p95 = expected_base + 5 * 0.95  # ~4.75
                actual_p95 = result["p95"][i, j]

                # p95 should be within the block's value range
                assert actual_p95 >= expected_base, \
                    f"p95[{i},{j}]={actual_p95} below block minimum {expected_base}"
                assert actual_p95 <= expected_base + 5, \
                    f"p95[{i},{j}]={actual_p95} above block maximum {expected_base + 5}"

                # p95 should be close to expected value
                assert abs(actual_p95 - expected_p95) < 1.0, \
                    f"p95[{i},{j}]={actual_p95} not close to expected {expected_p95}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
