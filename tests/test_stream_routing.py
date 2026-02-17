"""
Tests for stream routing through lake networks.

🔴 TDD RED: These tests cover gaps in the stream routing test suite:
- Cascading lakes (lake → stream → lake chains)
- Direct cycle detection via _trace_flows_to_lake
- Lake inlet identification (identify_lake_inlets)
- End-to-end drainage through multi-lake networks
- Adjacent/touching lake edge cases

Written FIRST before any implementation changes (TDD workflow).
"""

import numpy as np
import pytest


# D8 direction encoding (ESRI convention)
D8_OFFSETS = {
    1: (0, 1), 2: (-1, 1), 4: (-1, 0), 8: (-1, -1),
    16: (0, -1), 32: (1, -1), 64: (1, 0), 128: (1, 1),
}


def _trace_to_terminal(flow_dir, start_r, start_c, max_steps=200):
    """Follow flow directions from a cell until terminal or max steps.

    Returns list of (row, col) cells visited.
    """
    path = [(start_r, start_c)]
    r, c = start_r, start_c
    for _ in range(max_steps):
        d = flow_dir[r, c]
        if d == 0 or d not in D8_OFFSETS:
            break
        dr, dc = D8_OFFSETS[d]
        r, c = r + dr, c + dc
        if not (0 <= r < flow_dir.shape[0] and 0 <= c < flow_dir.shape[1]):
            break
        path.append((r, c))
    return path


# ── _trace_flows_to_lake (cycle detection helper) ─────────────────────


class TestTraceFlowsToLake:
    """Direct tests for the cycle-detection tracing function.

    _trace_flows_to_lake follows a cell's flow path and returns True
    if it re-enters the specified lake — indicating a cycle risk.
    """

    def test_path_reaching_terminal_returns_false(self):
        """Flow path that ends at terminal cell (flow_dir=0) is safe."""
        from src.terrain.water_bodies import _trace_flows_to_lake

        flow_dir = np.zeros((10, 10), dtype=np.uint8)
        # Path: (5,5) → South(64) → (6,5) → South(64) → (7,5) → terminal
        flow_dir[5, 5] = 64
        flow_dir[6, 5] = 64
        flow_dir[7, 5] = 0  # terminal

        lake_mask = np.zeros((10, 10), dtype=np.uint8)
        lake_mask[2:5, 4:7] = 1  # Lake above the start cell

        result = _trace_flows_to_lake(flow_dir, lake_mask, 5, 5, lake_id=1)
        assert result is False, "Path to terminal should not be flagged as cycle"

    def test_path_reentering_lake_returns_true(self):
        """Flow path that re-enters the source lake is a cycle."""
        from src.terrain.water_bodies import _trace_flows_to_lake

        flow_dir = np.zeros((10, 10), dtype=np.uint8)
        # Path: (5,5) → West(16) → (5,4) → North(4) → (4,4) = lake cell!
        flow_dir[5, 5] = 16
        flow_dir[5, 4] = 4
        flow_dir[4, 4] = 0  # won't reach here, lake_mask check comes first

        lake_mask = np.zeros((10, 10), dtype=np.uint8)
        lake_mask[3:5, 3:6] = 1  # Lake covers (4,4)

        result = _trace_flows_to_lake(flow_dir, lake_mask, 5, 5, lake_id=1)
        assert result is True, "Path re-entering lake should be flagged as cycle"

    def test_path_leaving_grid_returns_false(self):
        """Flow path that exits the grid boundary is safe."""
        from src.terrain.water_bodies import _trace_flows_to_lake

        flow_dir = np.zeros((10, 10), dtype=np.uint8)
        # Start at edge, flow off-grid
        flow_dir[0, 5] = 4  # North → row -1 = off grid

        lake_mask = np.zeros((10, 10), dtype=np.uint8)
        lake_mask[3:6, 4:7] = 1

        result = _trace_flows_to_lake(flow_dir, lake_mask, 0, 5, lake_id=1)
        assert result is False, "Path leaving grid should not be flagged as cycle"

    def test_path_entering_different_lake_returns_false(self):
        """Flow path entering a DIFFERENT lake (not source) is not a cycle."""
        from src.terrain.water_bodies import _trace_flows_to_lake

        flow_dir = np.zeros((10, 10), dtype=np.uint8)
        # Path: (5,5) → South → (6,5) → South → (7,5) which is lake 2
        flow_dir[5, 5] = 64
        flow_dir[6, 5] = 64

        lake_mask = np.zeros((10, 10), dtype=np.uint8)
        lake_mask[2:4, 4:7] = 1  # Lake 1 (source)
        lake_mask[7:9, 4:7] = 2  # Lake 2 (different)

        # Tracing from lake 1's perspective: entering lake 2 is NOT a cycle
        result = _trace_flows_to_lake(flow_dir, lake_mask, 5, 5, lake_id=1)
        assert result is False, "Entering different lake should not be a cycle"

    def test_max_steps_limit_returns_false(self):
        """If max_steps reached without re-entering lake, assume safe."""
        from src.terrain.water_bodies import _trace_flows_to_lake

        # Create a long looping path that never re-enters the lake
        # but also never terminates (cycles among non-lake cells)
        flow_dir = np.zeros((10, 10), dtype=np.uint8)
        # Two cells looping: (5,5) → East → (5,6) → West → (5,5) → ...
        flow_dir[5, 5] = 1   # East
        flow_dir[5, 6] = 16  # West

        lake_mask = np.zeros((10, 10), dtype=np.uint8)
        lake_mask[0:2, 0:2] = 1  # Lake far away

        result = _trace_flows_to_lake(
            flow_dir, lake_mask, 5, 5, lake_id=1, max_steps=10
        )
        assert result is False, "Should return False after max_steps"

    def test_immediate_lake_reentry_in_one_step(self):
        """Cell that flows directly back into the lake in one step."""
        from src.terrain.water_bodies import _trace_flows_to_lake

        flow_dir = np.zeros((10, 10), dtype=np.uint8)
        flow_dir[5, 6] = 16  # West → (5,5) which is lake

        lake_mask = np.zeros((10, 10), dtype=np.uint8)
        lake_mask[4:6, 4:6] = 1  # Lake covers (5,5)

        result = _trace_flows_to_lake(flow_dir, lake_mask, 5, 6, lake_id=1)
        assert result is True, "One-step re-entry should be detected"


# ── identify_lake_inlets ──────────────────────────────────────────────


class TestIdentifyLakeInlets:
    """Tests for identify_lake_inlets().

    Lake inlets are boundary cells where surrounding terrain is lower
    or equal to the lake surface, meaning water naturally flows in.
    """

    def test_finds_inlets_on_uphill_boundary(self):
        """Boundary cells adjacent to lower terrain are inlets."""
        from src.terrain.water_bodies import identify_lake_inlets

        # Terrain slopes down toward lake from the north
        dem = np.zeros((10, 10))
        for r in range(10):
            dem[r, :] = 100.0 - r * 10.0  # row 0=100, row 9=10

        # Lake in lower portion
        lake_mask = np.zeros((10, 10), dtype=np.uint8)
        lake_mask[6:9, 3:7] = 1
        dem[lake_mask > 0] = 30.0  # Lake surface

        inlets = identify_lake_inlets(lake_mask, dem)

        assert 1 in inlets, "Lake 1 should have inlets"
        assert len(inlets[1]) > 0, "Should find at least one inlet"

        # Inlets should be on the north boundary (row 6) where
        # adjacent terrain (row 5) is higher and flows into lake
        inlet_rows = [r for r, c in inlets[1]]
        assert 6 in inlet_rows, "North boundary cells should be inlets"

    def test_excludes_outlet_cells(self):
        """Cells marked as outlets should not be identified as inlets."""
        from src.terrain.water_bodies import identify_lake_inlets

        dem = np.zeros((10, 10))
        for r in range(10):
            dem[r, :] = 100.0 - r * 10.0

        lake_mask = np.zeros((10, 10), dtype=np.uint8)
        lake_mask[4:7, 4:7] = 1
        dem[lake_mask > 0] = 50.0

        outlet_mask = np.zeros((10, 10), dtype=bool)
        outlet_mask[6, 5] = True  # Outlet at bottom of lake

        inlets = identify_lake_inlets(lake_mask, dem, outlet_mask=outlet_mask)

        if 1 in inlets:
            for r, c in inlets[1]:
                assert not outlet_mask[r, c], (
                    f"Outlet cell ({r},{c}) should not be identified as inlet"
                )

    def test_no_inlets_for_lake_in_bowl(self):
        """Lake at the bottom of a bowl has no inlets (all terrain higher).

        Wait — actually, inlets are cells where neighbor_elev <= lake_elev + 0.1.
        In a bowl, all neighbors are HIGHER, so no inlets should be found...
        unless the tolerance catches some.
        """
        from src.terrain.water_bodies import identify_lake_inlets

        # Bowl: elevation increases away from center
        dem = np.zeros((15, 15))
        for r in range(15):
            for c in range(15):
                dem[r, c] = ((r - 7) ** 2 + (c - 7) ** 2) * 2.0

        lake_mask = np.zeros((15, 15), dtype=np.uint8)
        lake_mask[6:9, 6:9] = 1
        dem[lake_mask > 0] = 0.0  # Lake at bottom

        # All non-lake neighbors have elevation > 0 (much higher than lake)
        inlets = identify_lake_inlets(lake_mask, dem)

        # In a bowl, neighbor elevations are all higher than lake.
        # But the function checks neighbor_elev <= lake_elev + 0.1.
        # Lake is at 0.0, tolerance is 0.1.
        # Nearest non-lake cell has elev >= 2.0. So no inlets.
        if 1 in inlets:
            assert len(inlets[1]) == 0 or inlets[1] == [], (
                "Lake in deep bowl should have no inlets"
            )

    def test_multiple_lakes_get_separate_inlets(self):
        """Each lake should have its own inlet list."""
        from src.terrain.water_bodies import identify_lake_inlets

        dem = np.zeros((20, 20))
        for r in range(20):
            dem[r, :] = 100.0 - r * 5.0

        lake_mask = np.zeros((20, 20), dtype=np.uint8)
        lake_mask[4:7, 4:7] = 1   # Lake 1
        lake_mask[12:15, 12:15] = 2  # Lake 2
        dem[lake_mask == 1] = 75.0
        dem[lake_mask == 2] = 35.0

        inlets = identify_lake_inlets(lake_mask, dem)

        # Both lakes have terrain flowing into them from above
        assert 1 in inlets, "Lake 1 should have inlets"
        assert 2 in inlets, "Lake 2 should have inlets"


# ── Cascading Lakes (Lake → Stream → Lake) ───────────────────────────


class TestCascadingLakes:
    """Tests for cascading lake chains: lake₁ → stream → lake₂ → stream.

    Real-world rivers frequently pass through multiple lakes. Drainage
    area and rainfall must propagate through the entire chain without
    resetting at each lake boundary.
    """

    def _make_cascading_lakes(self):
        """Create terrain with two lakes in a river chain.

        Layout (30×10 grid, slopes top→bottom):
        - Rows 0-7: upstream terrain
        - Rows 8-11: Lake 1 (surface=60)
        - Rows 12-15: stream between lakes
        - Rows 16-19: Lake 2 (surface=40)
        - Rows 20-29: downstream terrain

        Returns (dem, lake_mask, outlet_mask, basin_mask)
        """
        dem = np.zeros((30, 10))
        for r in range(30):
            dem[r, :] = 100.0 - r * 3.0  # Gentle slope top→bottom

        lake_mask = np.zeros((30, 10), dtype=np.uint8)

        # Lake 1 (upper)
        lake_mask[8:12, 3:7] = 1
        dem[lake_mask == 1] = 60.0

        # Lake 2 (lower)
        lake_mask[16:20, 3:7] = 2
        dem[lake_mask == 2] = 40.0

        # Outlets at bottom edge of each lake
        outlet_mask = np.zeros((30, 10), dtype=bool)
        outlet_mask[11, 5] = True   # Lake 1 outlet
        outlet_mask[19, 5] = True   # Lake 2 outlet

        basin_mask = np.zeros((30, 10), dtype=bool)

        return dem, lake_mask, outlet_mask, basin_mask

    def test_both_outlets_get_downstream_directions(self):
        """Both lake outlets should connect to downstream terrain."""
        from src.terrain.water_bodies import (
            create_lake_flow_routing,
            compute_outlet_downstream_directions,
        )
        from src.terrain.flow_accumulation import compute_flow_direction

        dem, lake_mask, outlet_mask, basin_mask = self._make_cascading_lakes()

        flow_dir = compute_flow_direction(dem)
        lake_flow = create_lake_flow_routing(lake_mask, outlet_mask, dem)
        flow_dir[lake_mask > 0] = lake_flow[lake_mask > 0]

        result = compute_outlet_downstream_directions(
            flow_dir, lake_mask, outlet_mask, dem, basin_mask=basin_mask
        )

        # Both outlets should have non-zero downstream direction
        assert result[11, 5] != 0, "Lake 1 outlet should connect downstream"
        assert result[19, 5] != 0, "Lake 2 outlet should connect downstream"

    def test_no_cycles_in_cascading_network(self):
        """The full flow network with cascading lakes must be acyclic."""
        from src.terrain.water_bodies import (
            create_lake_flow_routing,
            compute_outlet_downstream_directions,
        )
        from src.terrain.flow_accumulation import (
            compute_flow_direction,
            compute_drainage_area,
        )

        dem, lake_mask, outlet_mask, basin_mask = self._make_cascading_lakes()

        flow_dir = compute_flow_direction(dem)
        lake_flow = create_lake_flow_routing(lake_mask, outlet_mask, dem)
        flow_dir[lake_mask > 0] = lake_flow[lake_mask > 0]

        flow_dir = compute_outlet_downstream_directions(
            flow_dir, lake_mask, outlet_mask, dem, basin_mask=basin_mask
        )

        # compute_drainage_area raises RuntimeError on cycles
        drainage = compute_drainage_area(flow_dir)
        assert drainage is not None, "Drainage computation should succeed"

    def test_lake2_drainage_exceeds_lake1_drainage(self):
        """Downstream lake should accumulate more drainage than upstream lake.

        Lake 2 receives drainage from Lake 1 + inter-lake terrain + its own area.
        """
        from src.terrain.water_bodies import (
            create_lake_flow_routing,
            compute_outlet_downstream_directions,
        )
        from src.terrain.flow_accumulation import (
            compute_flow_direction,
            compute_drainage_area,
        )

        dem, lake_mask, outlet_mask, basin_mask = self._make_cascading_lakes()

        flow_dir = compute_flow_direction(dem)
        lake_flow = create_lake_flow_routing(lake_mask, outlet_mask, dem)
        flow_dir[lake_mask > 0] = lake_flow[lake_mask > 0]

        flow_dir = compute_outlet_downstream_directions(
            flow_dir, lake_mask, outlet_mask, dem, basin_mask=basin_mask
        )

        drainage = compute_drainage_area(flow_dir)

        lake1_outlet_drainage = drainage[11, 5]
        lake2_outlet_drainage = drainage[19, 5]

        assert lake2_outlet_drainage > lake1_outlet_drainage, (
            f"Lake 2 outlet drainage ({lake2_outlet_drainage}) should exceed "
            f"Lake 1 outlet drainage ({lake1_outlet_drainage})"
        )

    def test_rainfall_accumulates_through_lake_chain(self):
        """Upstream rainfall should propagate through both lakes."""
        from src.terrain.water_bodies import (
            create_lake_flow_routing,
            compute_outlet_downstream_directions,
        )
        from src.terrain.flow_accumulation import (
            compute_flow_direction,
            compute_upstream_rainfall,
        )

        dem, lake_mask, outlet_mask, basin_mask = self._make_cascading_lakes()

        flow_dir = compute_flow_direction(dem)
        lake_flow = create_lake_flow_routing(lake_mask, outlet_mask, dem)
        flow_dir[lake_mask > 0] = lake_flow[lake_mask > 0]

        flow_dir = compute_outlet_downstream_directions(
            flow_dir, lake_mask, outlet_mask, dem, basin_mask=basin_mask
        )

        precip = np.ones((30, 10)) * 10.0  # Uniform precipitation

        rainfall = compute_upstream_rainfall(flow_dir, precip)

        lake1_outlet_rainfall = rainfall[11, 5]
        lake2_outlet_rainfall = rainfall[19, 5]

        assert lake1_outlet_rainfall > 0, "Lake 1 should accumulate rainfall"
        assert lake2_outlet_rainfall > lake1_outlet_rainfall, (
            f"Lake 2 rainfall ({lake2_outlet_rainfall}) should exceed "
            f"Lake 1 rainfall ({lake1_outlet_rainfall})"
        )

    def test_flow_path_connects_lake1_to_lake2(self):
        """Tracing flow from Lake 1 outlet should eventually reach Lake 2."""
        from src.terrain.water_bodies import (
            create_lake_flow_routing,
            compute_outlet_downstream_directions,
        )
        from src.terrain.flow_accumulation import compute_flow_direction

        dem, lake_mask, outlet_mask, basin_mask = self._make_cascading_lakes()

        flow_dir = compute_flow_direction(dem)
        lake_flow = create_lake_flow_routing(lake_mask, outlet_mask, dem)
        flow_dir[lake_mask > 0] = lake_flow[lake_mask > 0]

        flow_dir = compute_outlet_downstream_directions(
            flow_dir, lake_mask, outlet_mask, dem, basin_mask=basin_mask
        )

        # Trace from Lake 1 outlet downstream
        path = _trace_to_terminal(flow_dir, 11, 5)

        # Path should pass through Lake 2 cells
        entered_lake2 = any(
            lake_mask[r, c] == 2
            for r, c in path
            if 0 <= r < 30 and 0 <= c < 10
        )
        assert entered_lake2, (
            "Flow from Lake 1 outlet should reach Lake 2. "
            f"Path visited rows: {[r for r, c in path]}"
        )


# ── Adjacent/Touching Lakes ──────────────────────────────────────────


class TestAdjacentLakes:
    """Tests for lakes that are directly adjacent or share boundary cells.

    These are edge cases where the BFS routing and outlet detection
    must correctly distinguish which cells belong to which lake.
    """

    def test_adjacent_lakes_have_independent_routing(self):
        """Two lakes sharing a boundary should route independently."""
        from src.terrain.water_bodies import create_lake_flow_routing
        from src.terrain.flow_accumulation import compute_flow_direction

        dem = np.zeros((15, 15))
        for r in range(15):
            dem[r, :] = 100.0 - r * 5.0

        # Two lakes sharing column boundary
        lake_mask = np.zeros((15, 15), dtype=np.uint8)
        lake_mask[5:10, 3:7] = 1   # Lake 1 (left)
        lake_mask[5:10, 7:11] = 2  # Lake 2 (right, shares col 7 boundary)

        dem[lake_mask == 1] = 60.0
        dem[lake_mask == 2] = 60.0

        # Separate outlets
        outlet_mask = np.zeros((15, 15), dtype=bool)
        outlet_mask[9, 5] = True   # Lake 1 outlet
        outlet_mask[9, 9] = True   # Lake 2 outlet

        lake_flow = create_lake_flow_routing(lake_mask, outlet_mask, dem)

        # Lake 1 cells should converge to Lake 1 outlet
        for r in range(5, 10):
            for c in range(3, 7):
                if not outlet_mask[r, c]:
                    assert lake_flow[r, c] != 0, (
                        f"Lake 1 cell ({r},{c}) should have flow direction"
                    )

        # Lake 2 cells should converge to Lake 2 outlet
        for r in range(5, 10):
            for c in range(7, 11):
                if not outlet_mask[r, c]:
                    assert lake_flow[r, c] != 0, (
                        f"Lake 2 cell ({r},{c}) should have flow direction"
                    )

    def test_adjacent_lakes_no_cross_contamination(self):
        """Flow from Lake 1 cells should only reach Lake 1's outlet, not Lake 2's."""
        from src.terrain.water_bodies import create_lake_flow_routing

        dem = np.zeros((15, 15))
        for r in range(15):
            dem[r, :] = 100.0 - r * 5.0

        lake_mask = np.zeros((15, 15), dtype=np.uint8)
        lake_mask[5:10, 3:7] = 1
        lake_mask[5:10, 7:11] = 2

        dem[lake_mask > 0] = 60.0

        outlet_mask = np.zeros((15, 15), dtype=bool)
        outlet_mask[9, 5] = True
        outlet_mask[9, 9] = True

        lake_flow = create_lake_flow_routing(lake_mask, outlet_mask, dem)

        # Trace a Lake 1 cell to verify it reaches Lake 1's outlet
        path = _trace_to_terminal(lake_flow, 5, 5)
        reached_correct_outlet = any(
            (r, c) == (9, 5) for r, c in path
        )
        assert reached_correct_outlet, (
            "Lake 1 cell should route to Lake 1 outlet (9,5), "
            f"path: {path}"
        )


# ── Spillway + Outlet Routing Integration ─────────────────────────────


class TestSpillwayOutletIntegration:
    """Tests combining spillway detection with outlet downstream routing.

    In the full pipeline, find_lake_spillways() provides fallback
    directions that compute_outlet_downstream_directions() uses when
    no strictly lower neighbor exists.
    """

    def test_spillway_detected_then_used_as_outlet(self):
        """End-to-end: spillway detection → outlet routing → drainage."""
        from src.terrain.water_bodies import (
            find_lake_spillways,
            create_lake_flow_routing,
            compute_outlet_downstream_directions,
        )
        from src.terrain.flow_accumulation import (
            compute_flow_direction,
            compute_drainage_area,
        )

        # Lake in a slight depression — no strictly lower neighbor
        dem = np.ones((15, 15)) * 60.0

        lake_mask = np.zeros((15, 15), dtype=np.uint8)
        lake_mask[5:10, 5:10] = 1
        dem[lake_mask > 0] = 55.0  # Lake surface below surrounding terrain

        # Create a low point in the rim (spillway location)
        dem[10, 7] = 56.0  # Just above lake — lowest rim point
        dem[11, 7] = 50.0  # Below lake — downstream escape route

        # The outlet IS at the spillway cell
        outlet_mask = np.zeros((15, 15), dtype=bool)
        outlet_mask[9, 7] = True  # Bottom boundary of lake, near low rim

        basin_mask = np.zeros((15, 15), dtype=bool)

        # Step 1: detect spillway
        spillways = find_lake_spillways(lake_mask, dem)
        assert 1 in spillways, "Should detect spillway"

        # Step 2: build lake flow
        flow_dir = compute_flow_direction(dem)
        lake_flow = create_lake_flow_routing(lake_mask, outlet_mask, dem)
        flow_dir[lake_mask > 0] = lake_flow[lake_mask > 0]

        # Step 3: connect outlets (with spillway fallback)
        result = compute_outlet_downstream_directions(
            flow_dir, lake_mask, outlet_mask, dem,
            basin_mask=basin_mask, spillways=spillways,
        )

        # Step 4: verify no cycles
        drainage = compute_drainage_area(result)

        # Lake outlet should have accumulated lake cells
        lake_cell_count = np.sum(lake_mask > 0)
        outlet_drainage = drainage[9, 7]
        assert outlet_drainage >= lake_cell_count, (
            f"Outlet drainage ({outlet_drainage}) should be >= "
            f"lake area ({lake_cell_count})"
        )


# ── Flow Network Integrity ────────────────────────────────────────────


class TestFlowNetworkIntegrity:
    """Structural integrity tests for the complete flow network.

    These tests verify global properties that should hold for ANY
    valid flow network, regardless of the specific terrain.
    """

    def test_all_cells_reach_terminal_or_edge(self):
        """Every cell in the flow network should eventually reach a terminal
        cell (flow_dir=0) or the grid boundary. No infinite loops.
        """
        from src.terrain.water_bodies import (
            create_lake_flow_routing,
            compute_outlet_downstream_directions,
        )
        from src.terrain.flow_accumulation import compute_flow_direction

        # Random-ish terrain with a lake
        np.random.seed(42)
        dem = np.zeros((20, 20))
        for r in range(20):
            dem[r, :] = 100.0 - r * 5.0
        dem += np.random.uniform(-1.0, 1.0, (20, 20))  # Add noise

        lake_mask = np.zeros((20, 20), dtype=np.uint8)
        lake_mask[8:12, 8:12] = 1
        dem[lake_mask > 0] = 55.0

        outlet_mask = np.zeros((20, 20), dtype=bool)
        outlet_mask[11, 10] = True

        basin_mask = np.zeros((20, 20), dtype=bool)

        flow_dir = compute_flow_direction(dem)
        lake_flow = create_lake_flow_routing(lake_mask, outlet_mask, dem)
        flow_dir[lake_mask > 0] = lake_flow[lake_mask > 0]

        flow_dir = compute_outlet_downstream_directions(
            flow_dir, lake_mask, outlet_mask, dem, basin_mask=basin_mask
        )

        # Every cell should eventually terminate
        for r in range(20):
            for c in range(20):
                path = _trace_to_terminal(flow_dir, r, c, max_steps=500)
                last_r, last_c = path[-1]
                # Should end at terminal (flow_dir=0) or grid edge
                terminated = (
                    flow_dir[last_r, last_c] == 0
                    or last_r == 0 or last_r == 19
                    or last_c == 0 or last_c == 19
                )
                assert terminated, (
                    f"Cell ({r},{c}) didn't reach terminal after {len(path)} steps. "
                    f"Ended at ({last_r},{last_c}) with dir={flow_dir[last_r, last_c]}"
                )

    def test_outlet_downstream_cell_has_higher_drainage(self):
        """Cell immediately downstream of any outlet should have higher
        drainage than the outlet itself (it receives the outlet's flow).
        """
        from src.terrain.water_bodies import (
            create_lake_flow_routing,
            compute_outlet_downstream_directions,
        )
        from src.terrain.flow_accumulation import (
            compute_flow_direction,
            compute_drainage_area,
        )

        dem = np.zeros((20, 20))
        for r in range(20):
            dem[r, :] = 100.0 - r * 5.0

        lake_mask = np.zeros((20, 20), dtype=np.uint8)
        lake_mask[8:12, 8:12] = 1
        dem[lake_mask > 0] = 55.0

        outlet_mask = np.zeros((20, 20), dtype=bool)
        outlet_mask[11, 10] = True

        basin_mask = np.zeros((20, 20), dtype=bool)

        flow_dir = compute_flow_direction(dem)
        lake_flow = create_lake_flow_routing(lake_mask, outlet_mask, dem)
        flow_dir[lake_mask > 0] = lake_flow[lake_mask > 0]

        flow_dir = compute_outlet_downstream_directions(
            flow_dir, lake_mask, outlet_mask, dem, basin_mask=basin_mask
        )

        drainage = compute_drainage_area(flow_dir)

        # For each connected outlet, check downstream cell
        outlet_cells = np.argwhere(outlet_mask)
        for r, c in outlet_cells:
            d = flow_dir[r, c]
            if d == 0:
                continue  # Terminal outlet, skip

            dr, dc = D8_OFFSETS[d]
            nr, nc = r + dr, c + dc
            if 0 <= nr < 20 and 0 <= nc < 20:
                assert drainage[nr, nc] > drainage[r, c], (
                    f"Downstream cell ({nr},{nc}) drainage "
                    f"({drainage[nr, nc]}) should exceed outlet ({r},{c}) "
                    f"drainage ({drainage[r, c]})"
                )
