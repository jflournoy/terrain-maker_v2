"""
Tests for water body handling in flow accumulation.

Tests downloading, rasterization, outlet detection, and lake flow routing.
Following TDD: these tests are written FIRST before implementation.
"""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from rasterio import Affine


class TestLakeRasterization:
    """Tests for rasterizing lake polygons to mask."""

    def test_rasterize_creates_mask_matching_bbox(self):
        """Lake mask should match the specified bbox and resolution."""
        from src.terrain.water_bodies import rasterize_lakes_to_mask

        # Simple lake polygon (square)
        lakes_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"lake_id": 1, "name": "Test Lake"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [-117.0, 33.0],
                            [-117.0, 33.1],
                            [-116.9, 33.1],
                            [-116.9, 33.0],
                            [-117.0, 33.0],
                        ]],
                    },
                }
            ],
        }

        bbox = (32.9, -117.1, 33.2, -116.8)  # (south, west, north, east)
        resolution = 0.01  # ~1km in degrees

        mask, transform = rasterize_lakes_to_mask(lakes_geojson, bbox, resolution)

        # Check mask dimensions match bbox/resolution (ceil for partial cells)
        expected_rows = int(np.ceil((33.2 - 32.9) / resolution))
        expected_cols = int(np.ceil((-116.8 - (-117.1)) / resolution))

        assert mask.shape[0] == expected_rows
        assert mask.shape[1] == expected_cols

        # Check some lake cells are marked
        assert np.any(mask > 0), "Lake should have non-zero cells in mask"

    def test_rasterize_labels_separate_lakes(self):
        """Each lake should have a unique label in the mask."""
        from src.terrain.water_bodies import rasterize_lakes_to_mask

        lakes_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"lake_id": 1},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [-117.0, 33.0], [-117.0, 33.05],
                            [-116.95, 33.05], [-116.95, 33.0], [-117.0, 33.0]
                        ]],
                    },
                },
                {
                    "type": "Feature",
                    "properties": {"lake_id": 2},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [-116.8, 33.1], [-116.8, 33.15],
                            [-116.75, 33.15], [-116.75, 33.1], [-116.8, 33.1]
                        ]],
                    },
                },
            ],
        }

        bbox = (32.9, -117.1, 33.2, -116.7)
        mask, _ = rasterize_lakes_to_mask(lakes_geojson, bbox, resolution=0.01)

        # Should have at least 2 unique non-zero labels
        unique_labels = np.unique(mask[mask > 0])
        assert len(unique_labels) >= 2, "Should have separate labels for each lake"

    def test_rasterize_returns_proper_affine_transform(self):
        """Transform should correctly map pixel coordinates to geographic."""
        from src.terrain.water_bodies import rasterize_lakes_to_mask

        lakes_geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {"lake_id": 1},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-117.0, 33.0], [-117.0, 33.1],
                        [-116.9, 33.1], [-116.9, 33.0], [-117.0, 33.0]
                    ]],
                },
            }],
        }

        bbox = (32.9, -117.1, 33.2, -116.8)
        resolution = 0.01

        mask, transform = rasterize_lakes_to_mask(lakes_geojson, bbox, resolution)

        # Transform should be an Affine object
        assert isinstance(transform, Affine)

        # Top-left corner should map to (west, north)
        x, y = transform * (0, 0)
        assert abs(x - (-117.1)) < 0.001, f"Expected west=-117.1, got {x}"
        assert abs(y - 33.2) < 0.001, f"Expected north=33.2, got {y}"


class TestLakeOutletDetection:
    """Tests for identifying lake outlets."""

    def test_identify_outlet_from_pour_point(self):
        """Should identify outlet cell from HydroLAKES pour point."""
        from src.terrain.water_bodies import identify_outlet_cells

        # Create a simple lake mask (10x10 lake at center of 20x20 grid)
        lake_mask = np.zeros((20, 20), dtype=np.uint8)
        lake_mask[5:15, 5:15] = 1  # Lake in center (rows 5-14, cols 5-14)

        # Transform: origin at top-left = (-117.0, 33.2), resolution = 0.01
        # Pixel (row, col) maps to (west + col*res, north - row*res)
        # We want outlet at row=10, col=10 (inside lake)
        # lon = -117.0 + 10*0.01 = -116.90
        # lat = 33.2 - 10*0.01 = 33.10
        outlets = {1: (-116.90, 33.10)}  # (lon, lat)

        transform = Affine(0.01, 0, -117.0, 0, -0.01, 33.2)

        outlet_mask = identify_outlet_cells(lake_mask, outlets, transform)

        # Should have exactly one outlet cell
        assert np.sum(outlet_mask) == 1, "Should have exactly one outlet cell"

        # Outlet should be within the lake boundary
        outlet_row, outlet_col = np.where(outlet_mask)
        assert lake_mask[outlet_row[0], outlet_col[0]] > 0, "Outlet must be within lake"

    def test_endorheic_lake_has_no_outlet(self):
        """Lake with no outlet should be marked as endorheic."""
        from src.terrain.water_bodies import identify_outlet_cells

        lake_mask = np.zeros((20, 20), dtype=np.uint8)
        lake_mask[5:15, 5:15] = 1

        # Empty outlets dict
        outlets = {}
        transform = Affine(0.01, 0, -117.0, 0, -0.01, 33.2)

        outlet_mask = identify_outlet_cells(lake_mask, outlets, transform)

        # Should have no outlet cells
        assert np.sum(outlet_mask) == 0, "Endorheic lake should have no outlets"


class TestLakeFlowRouting:
    """Tests for flow direction routing within lakes."""

    def test_create_lake_flow_routing_routes_to_outlet(self):
        """All lake cells should have flow path to outlet."""
        from src.terrain.water_bodies import create_lake_flow_routing
        from src.terrain.flow_accumulation import D8_OFFSETS

        # Create lake mask (5x5 lake)
        lake_mask = np.zeros((10, 10), dtype=np.uint8)
        lake_mask[2:7, 2:7] = 1

        # Outlet at bottom center of lake
        outlet_mask = np.zeros((10, 10), dtype=bool)
        outlet_mask[6, 4] = True

        # Flat DEM
        dem = np.ones((10, 10)) * 100.0

        flow_dir = create_lake_flow_routing(lake_mask, outlet_mask, dem)

        # Verify all lake cells (except outlet) have valid flow direction
        for r in range(2, 7):
            for c in range(2, 7):
                if not outlet_mask[r, c]:
                    # Cell should have a non-zero flow direction
                    assert flow_dir[r, c] != 0, f"Lake cell ({r},{c}) should have flow direction"

        # Outlet should have flow_dir = 0 (it's the terminal)
        assert flow_dir[6, 4] == 0, "Outlet should have flow_dir=0"

    def test_lake_flow_routing_converges_to_outlet(self):
        """Following flow directions from any lake cell should reach outlet."""
        from src.terrain.water_bodies import create_lake_flow_routing

        lake_mask = np.zeros((10, 10), dtype=np.uint8)
        lake_mask[2:7, 2:7] = 1

        outlet_mask = np.zeros((10, 10), dtype=bool)
        outlet_mask[6, 4] = True

        dem = np.ones((10, 10)) * 100.0
        flow_dir = create_lake_flow_routing(lake_mask, outlet_mask, dem)

        # D8 direction to offset mapping
        D8_OFFSETS = {
            1: (0, 1), 2: (-1, 1), 4: (-1, 0), 8: (-1, -1),
            16: (0, -1), 32: (1, -1), 64: (1, 0), 128: (1, 1)
        }

        # Trace flow from each lake cell to verify it reaches outlet
        for r in range(2, 7):
            for c in range(2, 7):
                if outlet_mask[r, c]:
                    continue

                # Follow flow path
                curr_r, curr_c = r, c
                steps = 0
                max_steps = 100

                while steps < max_steps:
                    if outlet_mask[curr_r, curr_c]:
                        break  # Reached outlet

                    direction = flow_dir[curr_r, curr_c]
                    if direction == 0:
                        break

                    if direction not in D8_OFFSETS:
                        pytest.fail(f"Invalid direction {direction} at ({curr_r},{curr_c})")

                    dr, dc = D8_OFFSETS[direction]
                    curr_r += dr
                    curr_c += dc
                    steps += 1

                assert outlet_mask[curr_r, curr_c], \
                    f"Flow from ({r},{c}) should reach outlet, ended at ({curr_r},{curr_c})"

    def test_endorheic_lake_is_terminal_sink(self):
        """Lake with no outlet should have all cells with flow_dir=0."""
        from src.terrain.water_bodies import create_lake_flow_routing

        lake_mask = np.zeros((10, 10), dtype=np.uint8)
        lake_mask[2:7, 2:7] = 1

        # No outlet
        outlet_mask = np.zeros((10, 10), dtype=bool)

        dem = np.ones((10, 10)) * 100.0
        flow_dir = create_lake_flow_routing(lake_mask, outlet_mask, dem)

        # All lake cells should be outlets (terminal sinks)
        for r in range(2, 7):
            for c in range(2, 7):
                assert flow_dir[r, c] == 0, \
                    f"Endorheic lake cell ({r},{c}) should have flow_dir=0"


class TestFlowAccumulationWithLakes:
    """Integration tests for flow accumulation with lake routing."""

    def test_outlet_receives_upstream_and_lake_accumulation(self):
        """Lake outlet should receive all upstream + lake interior drainage."""
        from src.terrain.water_bodies import create_lake_flow_routing
        from src.terrain.flow_accumulation import compute_drainage_area

        # Create simple terrain: slope from top to bottom with lake in middle
        dem = np.zeros((20, 20))
        for r in range(20):
            dem[r, :] = 100 - r  # Descending from top to bottom

        # Lake in center (flat)
        lake_mask = np.zeros((20, 20), dtype=np.uint8)
        lake_mask[8:12, 8:12] = 1
        dem[lake_mask > 0] = dem[8, 10]  # Flatten lake

        # Outlet at bottom of lake
        outlet_mask = np.zeros((20, 20), dtype=bool)
        outlet_mask[11, 10] = True

        # Get lake routing
        lake_flow = create_lake_flow_routing(lake_mask, outlet_mask, dem)

        # Compute terrain flow direction (would need to merge with lake flow)
        from src.terrain.flow_accumulation import compute_flow_direction
        terrain_flow = compute_flow_direction(dem)

        # Merge: lake cells use lake_flow, others use terrain_flow
        merged_flow = terrain_flow.copy()
        merged_flow[lake_mask > 0] = lake_flow[lake_mask > 0]

        # Compute drainage area
        drainage = compute_drainage_area(merged_flow)

        # Outlet should have drainage >= lake area (16 cells)
        outlet_drainage = drainage[11, 10]
        lake_area = np.sum(lake_mask > 0)

        assert outlet_drainage >= lake_area, \
            f"Outlet drainage ({outlet_drainage}) should be >= lake area ({lake_area})"


class TestWaterBodyDownload:
    """Tests for downloading water body data."""

    @pytest.mark.skipif(True, reason="Requires network access")
    def test_download_nhd_returns_geojson(self):
        """NHD download should return valid GeoJSON with waterbodies."""
        from src.terrain.water_bodies import download_nhd_water_bodies

        bbox = (32.5, -117.0, 32.6, -116.9)
        output_dir = Path("/tmp/test_nhd")
        output_dir.mkdir(exist_ok=True)

        waterbodies, flowlines = download_nhd_water_bodies(bbox, str(output_dir))

        assert "type" in waterbodies
        assert waterbodies["type"] == "FeatureCollection"

    def test_download_respects_bbox(self):
        """Downloaded lakes should be within the requested bbox."""
        # This would test that filtering works correctly
        # Mocked for now
        pass

    def test_download_caches_results(self):
        """Subsequent downloads should use cached data."""
        # This would test caching behavior
        pass
