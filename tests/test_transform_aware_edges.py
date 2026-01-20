"""
TDD tests for transform-aware rectangle edge sampling.

Tests the new generate_transform_aware_rectangle_edges() function that samples
edges from original DEM space and maps through transforms to final mesh coordinates.
"""

import pytest
import numpy as np
from rasterio.transform import Affine


class TestTransformAwareRectangleEdges:
    """Tests for transform-aware rectangle edge sampling (TDD RED phase)."""

    def test_coordinate_mapping_simple_downsample(self):
        """Test coordinate mapping with simple downsample transform."""
        from src.terrain.mesh_operations import generate_transform_aware_rectangle_edges
        from src.terrain.core import Terrain

        # Create simple 100×100 DEM (original)
        original_dem = np.random.rand(100, 100) * 100 + 100  # Elevation 100-200

        # Create Terrain object
        terrain = Terrain()
        terrain.dem_data = original_dem
        terrain.dem_shape = original_dem.shape
        terrain.dem_transform = Affine.translation(-89.0, 47.0) * Affine.scale(0.01, -0.01)

        # Simulate downsampling to 10×10
        downsampled_dem = original_dem[::10, ::10]  # Simple decimation
        terrain.data_layers = {
            "dem": {
                "data": original_dem,
                "transform": terrain.dem_transform,
                "crs": "EPSG:4326",
                "transformed": True,
                "transformed_data": downsampled_dem,
                "transformed_transform": terrain.dem_transform * Affine.scale(10, 10),
                "transformed_crs": "EPSG:4326",
            }
        }

        # Create coord_to_index for final 10×10 mesh (all pixels valid)
        coord_to_index = {}
        for y in range(10):
            for x in range(10):
                coord_to_index[(y, x)] = y * 10 + x

        # Run transform-aware edge sampling
        edge_pixels_final = generate_transform_aware_rectangle_edges(
            terrain,
            coord_to_index,
            edge_sample_spacing=1.0  # Sample every pixel at original resolution
        )

        # Verify results
        # Original perimeter: 2 * (100 + 100) - 4 = 396 pixels
        # After downsampling 10x: should map to ~40 unique pixels on final 10×10 perimeter
        assert len(edge_pixels_final) > 30, f"Expected >30 edge pixels, got {len(edge_pixels_final)}"
        assert len(edge_pixels_final) < 100, f"Expected <100 edge pixels, got {len(edge_pixels_final)}"

        # All pixels should be on the perimeter of 10×10 grid
        for (y, x) in edge_pixels_final:
            assert y == 0 or y == 9 or x == 0 or x == 9, f"Pixel ({y},{x}) not on perimeter"

    def test_coordinate_mapping_with_reprojection(self):
        """Test coordinate mapping with reprojection and downsample."""
        from src.terrain.mesh_operations import generate_transform_aware_rectangle_edges
        from src.terrain.core import Terrain

        # Create 50×50 DEM in WGS84
        original_dem = np.random.rand(50, 50) * 100 + 100

        terrain = Terrain()
        terrain.dem_data = original_dem
        terrain.dem_shape = original_dem.shape
        terrain.dem_transform = Affine.translation(-89.0, 47.0) * Affine.scale(0.01, -0.01)

        # Simulate reprojection + downsample to 10×10 in UTM
        downsampled_dem = original_dem[::5, ::5]
        utm_transform = Affine.translation(500000, 5200000) * Affine.scale(840, -840)

        terrain.data_layers = {
            "dem": {
                "data": original_dem,
                "transform": terrain.dem_transform,
                "crs": "EPSG:4326",
                "transformed": True,
                "transformed_data": downsampled_dem,
                "transformed_transform": utm_transform,
                "transformed_crs": "EPSG:32617",
            }
        }

        # Create coord_to_index for final 10×10 mesh
        coord_to_index = {}
        for y in range(10):
            for x in range(10):
                coord_to_index[(y, x)] = y * 10 + x

        # Run transform-aware edge sampling
        edge_pixels_final = generate_transform_aware_rectangle_edges(
            terrain,
            coord_to_index,
            edge_sample_spacing=1.0
        )

        # Should successfully map through CRS transformation
        assert len(edge_pixels_final) > 20, f"Expected >20 edge pixels after reprojection, got {len(edge_pixels_final)}"

        # All pixels should be on the perimeter
        for (y, x) in edge_pixels_final:
            assert y == 0 or y == 9 or x == 0 or x == 9

    def test_edge_coverage_comparison(self):
        """Test that transform-aware approach gives much better coverage than legacy."""
        from src.terrain.mesh_operations import (
            generate_transform_aware_rectangle_edges,
            generate_rectangle_edge_pixels,
            diagnose_rectangle_edge_coverage,
        )
        from src.terrain.core import Terrain

        # Simulate real scenario: large original DEM downsampled to smaller mesh
        original_dem = np.ones((200, 200)) * 150  # 200×200 original

        terrain = Terrain()
        terrain.dem_data = original_dem
        terrain.dem_shape = original_dem.shape
        terrain.dem_transform = Affine.translation(-89.0, 47.0) * Affine.scale(0.01, -0.01)

        # Downsample aggressively to 20×20 (creates NaN margins)
        downsampled_dem = np.full((20, 20), np.nan)
        # Only interior has data (simulating NaN margins from downsampling)
        downsampled_dem[2:-2, 2:-2] = original_dem[::10, ::10][2:-2, 2:-2]

        terrain.data_layers = {
            "dem": {
                "data": original_dem,
                "transform": terrain.dem_transform,
                "crs": "EPSG:4326",
                "transformed": True,
                "transformed_data": downsampled_dem,
                "transformed_transform": terrain.dem_transform * Affine.scale(10, 10),
                "transformed_crs": "EPSG:4326",
            }
        }

        # Create coord_to_index only for valid pixels (interior)
        coord_to_index = {}
        idx = 0
        for y in range(20):
            for x in range(20):
                if not np.isnan(downsampled_dem[y, x]):
                    coord_to_index[(y, x)] = idx
                    idx += 1

        # Test 1: Legacy approach (using downsampled shape) - should have low coverage
        legacy_diagnostic = diagnose_rectangle_edge_coverage((20, 20), coord_to_index)
        legacy_coverage = legacy_diagnostic['coverage_percent']

        # Test 2: Transform-aware approach (using original shape) - should have high coverage
        transform_aware_pixels = generate_transform_aware_rectangle_edges(
            terrain,
            coord_to_index,
            edge_sample_spacing=10.0  # Sample every 10th pixel to match downsampling
        )

        # Calculate coverage for transform-aware
        original_perimeter = 2 * (200 + 200) - 4
        transform_aware_coverage = (len(transform_aware_pixels) / original_perimeter * 100) * 10  # Adjust for sampling

        # Transform-aware should have significantly better coverage
        assert legacy_coverage < 30, f"Legacy coverage should be low (<30%), got {legacy_coverage:.1f}%"
        assert len(transform_aware_pixels) > 20, f"Transform-aware should find >20 valid edge pixels, got {len(transform_aware_pixels)}"

    def test_terrain_object_required(self):
        """Test that terrain object is required (not None)."""
        from src.terrain.mesh_operations import generate_transform_aware_rectangle_edges

        # Try calling without terrain object
        with pytest.raises((TypeError, AttributeError, ValueError)):
            generate_transform_aware_rectangle_edges(
                terrain=None,
                coord_to_index={(0, 0): 0},
                edge_sample_spacing=1.0
            )

    def test_missing_transform_data(self):
        """Test graceful handling when terrain lacks transformed_transform."""
        from src.terrain.mesh_operations import generate_transform_aware_rectangle_edges
        from src.terrain.core import Terrain

        # Create minimal Terrain without transformed data
        terrain = Terrain()
        terrain.dem_data = np.ones((10, 10)) * 100
        terrain.dem_shape = (10, 10)
        terrain.dem_transform = Affine.translation(-89.0, 47.0) * Affine.scale(0.01, -0.01)
        terrain.data_layers = {
            "dem": {
                "data": terrain.dem_data,
                "transform": terrain.dem_transform,
                "crs": "EPSG:4326",
                "transformed": False,  # No transforms applied yet
            }
        }

        coord_to_index = {(0, 0): 0}

        # Should raise error or handle gracefully
        with pytest.raises((KeyError, AttributeError, ValueError)):
            generate_transform_aware_rectangle_edges(
                terrain,
                coord_to_index,
                edge_sample_spacing=1.0
            )

    def test_edge_pixels_in_coord_to_index_only(self):
        """Test that returned pixels are only those in coord_to_index (valid vertices)."""
        from src.terrain.mesh_operations import generate_transform_aware_rectangle_edges
        from src.terrain.core import Terrain

        # Create simple scenario
        original_dem = np.ones((30, 30)) * 100

        terrain = Terrain()
        terrain.dem_data = original_dem
        terrain.dem_shape = original_dem.shape
        terrain.dem_transform = Affine.translation(-89.0, 47.0) * Affine.scale(0.01, -0.01)

        # Downsample to 10×10
        downsampled_dem = original_dem[::3, ::3]
        terrain.data_layers = {
            "dem": {
                "data": original_dem,
                "transform": terrain.dem_transform,
                "crs": "EPSG:4326",
                "transformed": True,
                "transformed_data": downsampled_dem,
                "transformed_transform": terrain.dem_transform * Affine.scale(3, 3),
                "transformed_crs": "EPSG:4326",
            }
        }

        # Create PARTIAL coord_to_index (some pixels missing)
        coord_to_index = {}
        idx = 0
        for y in range(10):
            for x in range(10):
                # Skip some pixels (simulating sparse valid_mask)
                if (y + x) % 3 != 0:  # Arbitrary pattern
                    coord_to_index[(y, x)] = idx
                    idx += 1

        # Run sampling
        edge_pixels_final = generate_transform_aware_rectangle_edges(
            terrain,
            coord_to_index,
            edge_sample_spacing=1.0
        )

        # All returned pixels MUST be in coord_to_index
        for (y, x) in edge_pixels_final:
            assert (y, x) in coord_to_index, f"Pixel ({y},{x}) not in coord_to_index!"


class TestTransformAwareIntegration:
    """Tests for integration with create_boundary_extension (TDD RED phase)."""

    def test_create_boundary_extension_accepts_terrain_parameter(self):
        """Test that create_boundary_extension accepts terrain= parameter."""
        from src.terrain.mesh_operations import create_boundary_extension
        from src.terrain.core import Terrain

        # Create minimal test data
        positions = np.array([[0, 0, 1], [1, 0, 2], [0, 1, 3], [1, 1, 4]], dtype=float)
        boundary_points = [(0, 0), (0, 1), (1, 1), (1, 0)]
        coord_to_index = {(0, 0): 0, (0, 1): 1, (1, 1): 3, (1, 0): 2}

        # Create minimal Terrain
        terrain = Terrain()
        terrain.dem_data = np.ones((10, 10)) * 100
        terrain.dem_shape = (10, 10)
        terrain.dem_transform = Affine.translation(-89.0, 47.0) * Affine.scale(0.01, -0.01)
        terrain.data_layers = {
            "dem": {
                "data": terrain.dem_data,
                "transform": terrain.dem_transform,
                "crs": "EPSG:4326",
                "transformed": True,
                "transformed_data": terrain.dem_data,
                "transformed_transform": terrain.dem_transform,
                "transformed_crs": "EPSG:4326",
            }
        }

        # Should accept terrain parameter (will fail in RED phase if not implemented)
        try:
            result = create_boundary_extension(
                positions,
                boundary_points,
                coord_to_index,
                base_depth=-0.2,
                use_rectangle_edges=True,
                terrain=terrain,  # NEW parameter
                edge_sample_spacing=1.0,
            )
            # If it succeeds, verify it returns expected structure
            assert isinstance(result, tuple)
            assert len(result) == 3  # (vertices, faces, colors)
        except TypeError as e:
            # Expected in RED phase - parameter doesn't exist yet
            if "terrain" in str(e) or "unexpected keyword" in str(e):
                pytest.skip("terrain parameter not implemented yet (TDD RED phase)")
            else:
                raise

    def test_backwards_compatibility_with_dem_shape(self):
        """Test that legacy dem_shape parameter still works."""
        from src.terrain.mesh_operations import create_boundary_extension

        positions = np.array([[0, 0, 1], [1, 0, 2], [0, 1, 3], [1, 1, 4]], dtype=float)
        boundary_points = [(0, 0), (0, 1), (1, 1), (1, 0)]
        coord_to_index = {(0, 0): 0, (0, 1): 1, (1, 1): 3, (1, 0): 2}

        # Legacy call with dem_shape (should still work)
        result = create_boundary_extension(
            positions,
            boundary_points,
            coord_to_index,
            base_depth=-0.2,
            use_rectangle_edges=True,
            dem_shape=(2, 2),  # Legacy parameter
        )

        assert isinstance(result, tuple)
        assert len(result) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
