"""
Tests for iterative refinement parallel breaching method.

Following TDD RED-GREEN-REFACTOR cycle.
This is the RED phase - tests should fail initially.

The iterative refinement method addresses the limitation of the current parallel
breaching approach which can only terminate at outlets, missing chaining opportunities.

Key insight: Sinks A -> B -> C where C is near outlet should all breach via chaining:
- Iteration 1: C breaches to outlet
- Iteration 2: B chains to C's resolved path
- Iteration 3: A chains to B's resolved path

Without iterative refinement, only C succeeds and B, A are filled instead of breached.
"""

import pytest
import numpy as np

from src.terrain.flow_accumulation import (
    breach_depressions_constrained,
    _identify_sinks,
    identify_outlets,
)


class TestIterativeRefinementBreaching:
    """Test suite for iterative refinement parallel breaching method."""

    def test_chained_sinks_all_breach_with_iterative_method(self):
        """
        Chained sinks should all be breached when iterative refinement is enabled.

        DEM layout (20x5):
        - Outlet at [0, 0] (elevation 0)
        - Sink C at [4, 2] - close to outlet, should breach directly
        - Sink B at [9, 2] - far from outlet, near C's path
        - Sink A at [14, 2] - far from outlet, near B's path
        - Ridge of 100m everywhere else

        With max_breach_length=6:
        - C is ~5 cells from outlet -> can breach directly
        - B is ~10 cells from outlet -> cannot breach directly, needs to chain to C
        - A is ~15 cells from outlet -> cannot breach directly, needs to chain to B

        Expected behavior:
        - Without iterative: Only C breaches, B and A fail
        - With iterative: All three breach via chaining
        """
        # Create DEM with three chained sinks
        dem = np.ones((20, 5), dtype=np.float32) * 100.0

        # Outlet at top-left
        dem[0, 0] = 0.0
        dem[0, 1] = 5.0  # Slight slope toward outlet
        dem[1, 0] = 5.0
        dem[1, 1] = 10.0

        # Sink C - close to outlet (row 4)
        dem[4, 2] = 20.0  # Sink
        dem[3, 2] = 95.0  # Rim
        dem[5, 2] = 95.0  # Rim
        dem[4, 1] = 95.0  # Rim
        dem[4, 3] = 95.0  # Rim

        # Sink B - middle (row 9)
        dem[9, 2] = 25.0  # Sink
        dem[8, 2] = 95.0
        dem[10, 2] = 95.0
        dem[9, 1] = 95.0
        dem[9, 3] = 95.0

        # Sink A - far (row 14)
        dem[14, 2] = 30.0  # Sink
        dem[13, 2] = 95.0
        dem[15, 2] = 95.0
        dem[14, 1] = 95.0
        dem[14, 3] = 95.0

        # Create corridor between sinks (lower elevation path)
        for r in range(2, 16):
            if r not in [4, 9, 14]:  # Not the sink rows
                dem[r, 2] = 50.0  # Corridor

        outlets = np.zeros((20, 5), dtype=bool)
        outlets[0, 0] = True

        # Verify we have 3 sinks before breaching
        initial_sinks = _identify_sinks(dem, outlets, nodata_mask=None)
        sink_positions = [(s[0], s[1]) for s in initial_sinks]
        assert (4, 2) in sink_positions, "Sink C should be identified"
        assert (9, 2) in sink_positions, "Sink B should be identified"
        assert (14, 2) in sink_positions, "Sink A should be identified"

        # Breach with iterative refinement enabled
        # max_breach_length=6 means only C can reach outlet directly
        breached = breach_depressions_constrained(
            dem,
            outlets,
            max_breach_depth=100.0,
            max_breach_length=6,
            epsilon=1e-4,
            parallel_method="iterative",  # NEW PARAMETER
        )

        # Check that all three sinks have been resolved (have downslope neighbor)
        def has_downslope(arr, r, c):
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1),
                          (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                ni, nj = r + di, c + dj
                if 0 <= ni < arr.shape[0] and 0 <= nj < arr.shape[1]:
                    if arr[ni, nj] < arr[r, c]:
                        return True
            return False

        assert has_downslope(breached, 4, 2), "Sink C should be breached"
        assert has_downslope(breached, 9, 2), "Sink B should be breached via chaining to C"
        assert has_downslope(breached, 14, 2), "Sink A should be breached via chaining to B"

    def test_parallel_method_parameter_accepted(self):
        """breach_depressions_constrained should accept parallel_method parameter."""
        dem = np.array([
            [10, 10, 10],
            [10,  5, 10],
            [10, 10,  0]
        ], dtype=np.float32)

        outlets = np.zeros((3, 3), dtype=bool)
        outlets[2, 2] = True

        # Should not raise TypeError for unknown parameter
        breached = breach_depressions_constrained(
            dem,
            outlets,
            max_breach_depth=10.0,
            max_breach_length=10,
            epsilon=1e-4,
            parallel_method="iterative",
        )

        assert breached is not None
        assert breached.shape == dem.shape

    def test_parallel_method_default_is_checkerboard(self):
        """Default parallel_method should be 'checkerboard' (current behavior)."""
        dem = np.array([
            [10, 10, 10],
            [10,  5, 10],
            [10, 10,  0]
        ], dtype=np.float32)

        outlets = np.zeros((3, 3), dtype=bool)
        outlets[2, 2] = True

        # Without specifying parallel_method, should use checkerboard (existing behavior)
        breached = breach_depressions_constrained(
            dem,
            outlets,
            max_breach_depth=10.0,
            max_breach_length=10,
            epsilon=1e-4,
            # No parallel_method specified
        )

        assert breached is not None

    def test_iterative_method_finds_more_breaches_than_checkerboard(self):
        """
        Iterative method should find >= as many breaches as checkerboard method.

        This test creates a scenario where checkerboard fails on interior sinks
        but iterative succeeds via chaining.
        """
        # Create DEM with interior sink that requires chaining
        dem = np.ones((15, 15), dtype=np.float32) * 100.0

        # Outlet at corner
        dem[0, 0] = 0.0
        dem[0, 1] = 5.0
        dem[1, 0] = 5.0
        dem[1, 1] = 10.0

        # Near-outlet sink (should succeed with both methods)
        dem[3, 3] = 15.0

        # Far-interior sink (should only succeed with iterative)
        dem[10, 10] = 20.0

        # Create path corridor
        for i in range(2, 12):
            dem[i, i] = 30.0 + i  # Diagonal corridor with slope

        outlets = np.zeros((15, 15), dtype=bool)
        outlets[0, 0] = True

        # Breach with checkerboard (current default)
        breached_checkerboard = breach_depressions_constrained(
            dem.copy(),
            outlets,
            max_breach_depth=100.0,
            max_breach_length=5,  # Short length forces chaining
            epsilon=1e-4,
            parallel_method="checkerboard",
        )

        # Breach with iterative
        breached_iterative = breach_depressions_constrained(
            dem.copy(),
            outlets,
            max_breach_depth=100.0,
            max_breach_length=5,
            epsilon=1e-4,
            parallel_method="iterative",
        )

        # Count resolved sinks
        def count_resolved_sinks(original, breached, outlets):
            sinks = _identify_sinks(original, outlets, nodata_mask=None)
            resolved = 0
            for r, c, _ in sinks:
                # Check if sink now has downslope neighbor
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1),
                              (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    ni, nj = r + di, c + dj
                    if 0 <= ni < breached.shape[0] and 0 <= nj < breached.shape[1]:
                        if breached[ni, nj] < breached[r, c]:
                            resolved += 1
                            break
            return resolved

        checkerboard_resolved = count_resolved_sinks(dem, breached_checkerboard, outlets)
        iterative_resolved = count_resolved_sinks(dem, breached_iterative, outlets)

        assert iterative_resolved >= checkerboard_resolved, \
            f"Iterative ({iterative_resolved}) should resolve >= checkerboard ({checkerboard_resolved})"

    def test_iterative_converges(self):
        """Iterative method should converge (not loop forever)."""
        # Create a complex DEM that requires multiple iterations
        dem = np.ones((30, 30), dtype=np.float32) * 100.0

        # Outlet
        dem[0, 0] = 0.0

        # Chain of 5 sinks
        for i, row in enumerate([5, 10, 15, 20, 25]):
            dem[row, row] = 10.0 + i * 2

        # Corridor
        for i in range(1, 28):
            dem[i, i] = 50.0

        outlets = np.zeros((30, 30), dtype=bool)
        outlets[0, 0] = True

        # Should complete without timeout
        import time
        start = time.time()
        breached = breach_depressions_constrained(
            dem,
            outlets,
            max_breach_depth=100.0,
            max_breach_length=6,
            epsilon=1e-4,
            parallel_method="iterative",
        )
        elapsed = time.time() - start

        assert elapsed < 30, f"Iterative method took too long: {elapsed:.1f}s"
        assert breached is not None


class TestIterativeBreachingEdgeCases:
    """Edge case tests for iterative refinement breaching."""

    def test_no_sinks_returns_unchanged_dem(self):
        """DEM with no sinks should return unchanged."""
        # Simple slope - no sinks
        dem = np.arange(25, dtype=np.float32).reshape(5, 5)

        outlets = np.zeros((5, 5), dtype=bool)
        outlets[4, 4] = True  # Lowest point is outlet

        breached = breach_depressions_constrained(
            dem,
            outlets,
            max_breach_depth=100.0,
            max_breach_length=10,
            epsilon=1e-4,
            parallel_method="iterative",
        )

        # Should be essentially unchanged (maybe epsilon differences)
        assert np.allclose(breached, dem, atol=1e-3)

    def test_single_sink_same_as_checkerboard(self):
        """Single sink should produce same result regardless of method."""
        dem = np.array([
            [10, 10, 10, 10, 10],
            [10, 8,  8,  8, 10],
            [10, 8,  3,  8, 10],
            [10, 8,  8,  8, 10],
            [10, 10, 10, 10,  0]
        ], dtype=np.float32)

        outlets = np.zeros((5, 5), dtype=bool)
        outlets[4, 4] = True

        breached_checkerboard = breach_depressions_constrained(
            dem.copy(),
            outlets,
            max_breach_depth=10.0,
            max_breach_length=10,
            epsilon=1e-4,
            parallel_method="checkerboard",
        )

        breached_iterative = breach_depressions_constrained(
            dem.copy(),
            outlets,
            max_breach_depth=10.0,
            max_breach_length=10,
            epsilon=1e-4,
            parallel_method="iterative",
        )

        # Results should be identical for single sink
        assert np.allclose(breached_checkerboard, breached_iterative, atol=1e-6), \
            "Single sink should produce identical results"

    def test_unreachable_sink_goes_to_fill(self):
        """Sink that cannot be breached should still be handled (by fill stage)."""
        # Sink surrounded by very high rim, outlet too far
        dem = np.ones((20, 20), dtype=np.float32) * 1000.0
        dem[10, 10] = 5.0  # Deep sink
        dem[0, 0] = 0.0    # Far outlet

        outlets = np.zeros((20, 20), dtype=bool)
        outlets[0, 0] = True

        # With very restrictive constraints, sink cannot be breached
        breached = breach_depressions_constrained(
            dem,
            outlets,
            max_breach_depth=10.0,  # Not enough to breach 1000m rim
            max_breach_length=5,    # Not enough to reach outlet
            epsilon=1e-4,
            parallel_method="iterative",
        )

        # Sink should remain (will be handled by fill stage later)
        # This test just verifies no crash and reasonable output
        assert breached is not None
        assert breached.shape == dem.shape
