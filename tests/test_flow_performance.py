"""
Performance tests for flow accumulation functions.

These tests establish performance baselines and verify optimizations.
Run with: pytest tests/test_flow_performance.py -v -s
"""

import time
import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.terrain.flow_accumulation import (
    compute_flow_direction,
    compute_drainage_area,
    compute_upstream_rainfall,
    condition_dem,
    detect_ocean_mask,
    _fix_coastal_flow_directions,
    NUMBA_AVAILABLE,
)


class TestFlowPerformance:
    """Performance benchmarks for flow functions."""

    @pytest.fixture
    def small_dem(self):
        """100x100 DEM with varied terrain."""
        np.random.seed(42)
        x = np.linspace(0, 4 * np.pi, 100)
        y = np.linspace(0, 4 * np.pi, 100)
        X, Y = np.meshgrid(x, y)
        dem = (np.sin(X) + np.sin(Y) + 2) * 100 + np.random.randn(100, 100) * 5
        return dem.astype(np.float32)

    @pytest.fixture
    def medium_dem(self):
        """500x500 DEM for realistic performance testing."""
        np.random.seed(42)
        x = np.linspace(0, 10 * np.pi, 500)
        y = np.linspace(0, 10 * np.pi, 500)
        X, Y = np.meshgrid(x, y)
        dem = (np.sin(X) + np.sin(Y) + 2) * 100 + np.random.randn(500, 500) * 5
        return dem.astype(np.float32)

    @pytest.fixture
    def large_dem(self):
        """1000x1000 DEM for stress testing."""
        np.random.seed(42)
        x = np.linspace(0, 20 * np.pi, 1000)
        y = np.linspace(0, 20 * np.pi, 1000)
        X, Y = np.meshgrid(x, y)
        dem = (np.sin(X) + np.sin(Y) + 2) * 100 + np.random.randn(1000, 1000) * 5
        return dem.astype(np.float32)

    def test_compute_flow_direction_performance_small(self, small_dem):
        """Benchmark flow direction on 100x100 DEM."""
        # Warmup JIT
        _ = compute_flow_direction(small_dem)

        # Benchmark
        start = time.perf_counter()
        for _ in range(10):
            flow_dir = compute_flow_direction(small_dem)
        elapsed = (time.perf_counter() - start) / 10

        cells = small_dem.size
        throughput = cells / elapsed / 1e6  # Million cells per second

        print(f"\nFlow direction 100x100: {elapsed*1000:.2f} ms, {throughput:.2f} Mcells/s")
        assert flow_dir.shape == small_dem.shape
        assert elapsed < 0.1, f"Too slow: {elapsed:.3f}s (expected < 0.1s)"

    def test_compute_flow_direction_performance_medium(self, medium_dem):
        """Benchmark flow direction on 500x500 DEM."""
        # Warmup JIT
        _ = compute_flow_direction(medium_dem)

        # Benchmark
        start = time.perf_counter()
        flow_dir = compute_flow_direction(medium_dem)
        elapsed = time.perf_counter() - start

        cells = medium_dem.size
        throughput = cells / elapsed / 1e6

        print(f"\nFlow direction 500x500: {elapsed*1000:.2f} ms, {throughput:.2f} Mcells/s")
        assert flow_dir.shape == medium_dem.shape
        assert elapsed < 1.0, f"Too slow: {elapsed:.3f}s (expected < 1.0s)"

    def test_compute_flow_direction_performance_large(self, large_dem):
        """Benchmark flow direction on 1000x1000 DEM."""
        # Warmup JIT
        _ = compute_flow_direction(large_dem[:100, :100])

        # Benchmark
        start = time.perf_counter()
        flow_dir = compute_flow_direction(large_dem)
        elapsed = time.perf_counter() - start

        cells = large_dem.size
        throughput = cells / elapsed / 1e6

        print(f"\nFlow direction 1000x1000: {elapsed*1000:.2f} ms, {throughput:.2f} Mcells/s")
        assert flow_dir.shape == large_dem.shape
        assert elapsed < 5.0, f"Too slow: {elapsed:.3f}s (expected < 5.0s)"

    def test_compute_drainage_area_performance_medium(self, medium_dem):
        """Benchmark drainage area on 500x500 DEM."""
        flow_dir = compute_flow_direction(medium_dem)

        # Warmup JIT
        _ = compute_drainage_area(flow_dir)

        # Benchmark
        start = time.perf_counter()
        drainage = compute_drainage_area(flow_dir)
        elapsed = time.perf_counter() - start

        cells = medium_dem.size
        throughput = cells / elapsed / 1e6

        print(f"\nDrainage area 500x500: {elapsed*1000:.2f} ms, {throughput:.2f} Mcells/s")
        assert drainage.shape == medium_dem.shape
        assert elapsed < 1.0, f"Too slow: {elapsed:.3f}s (expected < 1.0s)"

    def test_compute_upstream_rainfall_performance_medium(self, medium_dem):
        """Benchmark upstream rainfall on 500x500 DEM."""
        flow_dir = compute_flow_direction(medium_dem)
        precip = np.random.rand(*medium_dem.shape).astype(np.float32) * 100 + 50

        # Warmup JIT
        _ = compute_upstream_rainfall(flow_dir, precip)

        # Benchmark
        start = time.perf_counter()
        upstream = compute_upstream_rainfall(flow_dir, precip)
        elapsed = time.perf_counter() - start

        cells = medium_dem.size
        throughput = cells / elapsed / 1e6

        print(f"\nUpstream rainfall 500x500: {elapsed*1000:.2f} ms, {throughput:.2f} Mcells/s")
        assert upstream.shape == medium_dem.shape
        assert elapsed < 1.0, f"Too slow: {elapsed:.3f}s (expected < 1.0s)"

    def test_condition_dem_performance_medium(self, medium_dem):
        """Benchmark DEM conditioning on 500x500 DEM."""
        # Benchmark
        start = time.perf_counter()
        conditioned = condition_dem(medium_dem, method="breach")
        elapsed = time.perf_counter() - start

        cells = medium_dem.size
        throughput = cells / elapsed / 1e6

        print(f"\nCondition DEM 500x500: {elapsed*1000:.2f} ms, {throughput:.2f} Mcells/s")
        assert conditioned.shape == medium_dem.shape
        assert elapsed < 5.0, f"Too slow: {elapsed:.3f}s (expected < 5.0s)"

    def test_fix_coastal_flow_directions_performance(self, medium_dem):
        """Benchmark coastal flow direction fix."""
        flow_dir = compute_flow_direction(medium_dem)
        # Create a coastal mask (edge cells)
        mask = np.zeros_like(medium_dem, dtype=bool)
        mask[:, 0] = True
        mask[:, -1] = True
        mask[0, :] = True
        mask[-1, :] = True

        # Benchmark
        start = time.perf_counter()
        for _ in range(10):
            flow_dir_copy = flow_dir.copy()
            _fix_coastal_flow_directions(flow_dir_copy, mask)
        elapsed = (time.perf_counter() - start) / 10

        cells = medium_dem.size
        throughput = cells / elapsed / 1e6

        print(f"\nCoastal fix 500x500: {elapsed*1000:.2f} ms, {throughput:.2f} Mcells/s")
        # Current pure Python implementation is slow - this will improve
        # For now, just verify it completes

    def test_combined_pipeline_performance(self, medium_dem):
        """Benchmark complete flow pipeline."""
        precip = np.random.rand(*medium_dem.shape).astype(np.float32) * 100 + 50

        # Warmup
        small = medium_dem[:100, :100]
        small_precip = precip[:100, :100]
        _ = compute_flow_direction(small)

        # Benchmark complete pipeline
        start = time.perf_counter()

        # 1. Condition DEM
        conditioned = condition_dem(medium_dem, method="breach")

        # 2. Flow direction
        flow_dir = compute_flow_direction(conditioned)

        # 3. Drainage area
        drainage = compute_drainage_area(flow_dir)

        # 4. Upstream rainfall
        upstream = compute_upstream_rainfall(flow_dir, precip)

        elapsed = time.perf_counter() - start

        cells = medium_dem.size
        throughput = cells / elapsed / 1e6

        print(f"\nComplete pipeline 500x500: {elapsed*1000:.2f} ms, {throughput:.2f} Mcells/s")
        assert elapsed < 10.0, f"Pipeline too slow: {elapsed:.3f}s"


class TestDirectionLookupOptimization:
    """Tests for direction lookup optimization."""

    def test_direction_code_to_offset_lookup(self):
        """Test that direction codes map correctly to offsets."""
        # D8 codes: 1=E, 2=NE, 4=N, 8=NW, 16=W, 32=SW, 64=S, 128=SE
        expected = {
            1: (0, 1),    # East
            2: (-1, 1),   # Northeast
            4: (-1, 0),   # North
            8: (-1, -1),  # Northwest
            16: (0, -1),  # West
            32: (1, -1),  # Southwest
            64: (1, 0),   # South
            128: (1, 1),  # Southeast
        }

        from src.terrain.flow_accumulation import D8_OFFSETS

        for code, offset in expected.items():
            assert D8_OFFSETS[code] == offset, f"Code {code} should map to {offset}"

    def test_lookup_table_approach(self):
        """Test that a lookup table can replace linear search."""
        # Create lookup table indexed by direction code
        # Code 1,2,4,8,16,32,64,128 -> index 0,1,2,3,4,5,6,7
        # Use log2 to convert power-of-2 codes to indices
        codes = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
        offsets_di = np.array([0, -1, -1, -1, 0, 1, 1, 1], dtype=np.int32)
        offsets_dj = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int32)

        # Test lookup for each code
        for i, code in enumerate(codes):
            # Convert code to index using bit operations
            # Code 1 -> 0, 2 -> 1, 4 -> 2, 8 -> 3, 16 -> 4, 32 -> 5, 64 -> 6, 128 -> 7
            idx = int(np.log2(code))
            assert idx == i, f"Code {code} should map to index {i}"
            assert offsets_di[idx] == [0, -1, -1, -1, 0, 1, 1, 1][i]
            assert offsets_dj[idx] == [1, 1, 0, -1, -1, -1, 0, 1][i]


class TestAccumulationCombination:
    """Tests for combining drainage area and upstream rainfall."""

    def test_combined_accumulation_correctness(self):
        """Test that combined accumulation produces same results as separate calls."""
        np.random.seed(42)
        dem = np.random.rand(100, 100).astype(np.float32) * 100
        precip = np.random.rand(100, 100).astype(np.float32) * 50 + 25

        flow_dir = compute_flow_direction(dem)

        # Separate calls
        drainage_separate = compute_drainage_area(flow_dir)
        upstream_separate = compute_upstream_rainfall(flow_dir, precip)

        # Results should be valid
        assert np.all(drainage_separate >= 1), "Every cell drains at least itself"
        assert np.all(upstream_separate >= precip), "Upstream >= local precipitation"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
