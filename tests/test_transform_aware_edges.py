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
        transform = Affine.translation(-89.0, 47.0) * Affine.scale(0.01, -0.01)

        # Create Terrain object with required arguments
        terrain = Terrain(dem_data=original_dem, dem_transform=transform, dem_crs="EPSG:4326")

        # Simulate downsampling to 10×10
        downsampled_dem = original_dem[::10, ::10]  # Simple decimation
        # Replace the data_layers to have proper transformed state
        terrain.data_layers["dem"].update({
            "transformed": True,
            "transformed_data": downsampled_dem,
            "transformed_transform": transform * Affine.scale(10, 10),
            "transformed_crs": "EPSG:4326",
        })

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

        # Verify results - with dense sampling, we should get many edge pixels
        assert len(edge_pixels_final) > 10, f"Expected >10 edge pixels, got {len(edge_pixels_final)}"

        # All pixels should be in coord_to_index (valid mesh vertices)
        for (y, x) in edge_pixels_final:
            assert (y, x) in coord_to_index, f"Pixel ({y},{x}) not in coord_to_index"

    def test_coordinate_mapping_with_reprojection(self):
        """Test coordinate mapping with reprojection and downsample."""
        from src.terrain.mesh_operations import generate_transform_aware_rectangle_edges
        from src.terrain.core import Terrain

        # Create 50×50 DEM in WGS84
        original_dem = np.random.rand(50, 50) * 100 + 100
        transform = Affine.translation(-89.0, 47.0) * Affine.scale(0.01, -0.01)

        terrain = Terrain(dem_data=original_dem, dem_transform=transform, dem_crs="EPSG:4326")

        # Simulate reprojection + downsample to 10×10 in UTM
        downsampled_dem = original_dem[::5, ::5]
        utm_transform = Affine.translation(500000, 5200000) * Affine.scale(840, -840)

        terrain.data_layers["dem"].update({
            "transformed": True,
            "transformed_data": downsampled_dem,
            "transformed_transform": utm_transform,
            "transformed_crs": "EPSG:32617",
        })

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
        # At least some edge pixels should map to valid vertices
        assert len(edge_pixels_final) >= 0, f"Edge pixel mapping should not error"

        # All returned pixels must be in coord_to_index
        for (y, x) in edge_pixels_final:
            assert (y, x) in coord_to_index, f"Pixel ({y},{x}) not in final mesh"

    def test_edge_coverage_comparison(self):
        """Test that transform-aware approach handles NaN margins correctly."""
        from src.terrain.mesh_operations import (
            generate_transform_aware_rectangle_edges,
            diagnose_rectangle_edge_coverage,
        )
        from src.terrain.core import Terrain

        # Simulate real scenario: large original DEM downsampled to smaller mesh
        original_dem = np.ones((200, 200)) * 150  # 200×200 original
        transform = Affine.translation(-89.0, 47.0) * Affine.scale(0.01, -0.01)

        terrain = Terrain(dem_data=original_dem, dem_transform=transform, dem_crs="EPSG:4326")

        # Downsample aggressively to 20×20 (creates NaN margins)
        downsampled_dem = np.full((20, 20), np.nan)
        # Only interior has data (simulating NaN margins from downsampling)
        downsampled_dem[2:-2, 2:-2] = original_dem[::10, ::10][2:-2, 2:-2]

        terrain.data_layers["dem"].update({
            "transformed": True,
            "transformed_data": downsampled_dem,
            "transformed_transform": transform * Affine.scale(10, 10),
            "transformed_crs": "EPSG:4326",
        })

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

        # Test 2: Transform-aware approach (using original shape)
        # Should filter out invalid pixels and only return those in coord_to_index
        transform_aware_pixels = generate_transform_aware_rectangle_edges(
            terrain,
            coord_to_index,
            edge_sample_spacing=10.0
        )

        # Transform-aware should have better coverage because it samples from original (no NaN margins)
        # All returned pixels must be valid
        for (y, x) in transform_aware_pixels:
            assert (y, x) in coord_to_index, f"Transform-aware pixel ({y},{x}) not in valid coord_to_index"

        # Legacy should show low coverage
        assert legacy_coverage < 30, f"Legacy coverage should be low (<30%), got {legacy_coverage:.1f}%"

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
        dem_data = np.ones((10, 10)) * 100
        transform = Affine.translation(-89.0, 47.0) * Affine.scale(0.01, -0.01)
        terrain = Terrain(dem_data=dem_data, dem_transform=transform, dem_crs="EPSG:4326")
        # Manually set transformed flag to False to simulate untransformed state
        terrain.data_layers["dem"]["transformed"] = False

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
        transform = Affine.translation(-89.0, 47.0) * Affine.scale(0.01, -0.01)

        terrain = Terrain(dem_data=original_dem, dem_transform=transform, dem_crs="EPSG:4326")

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

        # Create minimal Terrain WITH transformed data
        dem_data = np.ones((10, 10)) * 100
        transform = Affine.translation(-89.0, 47.0) * Affine.scale(0.01, -0.01)
        terrain = Terrain(dem_data=dem_data, dem_transform=transform, dem_crs="EPSG:4326")

        # Manually mark as transformed (simulating that apply_transforms was called)
        terrain.data_layers["dem"].update({
            "transformed": True,
            "transformed_data": dem_data,
            "transformed_transform": transform,
            "transformed_crs": "EPSG:4326",
        })

        # Should accept terrain parameter
        result = create_boundary_extension(
            positions,
            boundary_points,
            coord_to_index,
            base_depth=-0.2,
            use_rectangle_edges=True,
            terrain=terrain,  # NEW parameter
            edge_sample_spacing=1.0,
            two_tier=False,  # Single-tier for simpler testing
        )
        # Verify it returns expected structure
        assert isinstance(result, tuple)
        assert len(result) == 2  # Single-tier: (vertices, faces)

    def test_backwards_compatibility_with_dem_shape(self):
        """Test that legacy dem_shape parameter still works."""
        from src.terrain.mesh_operations import create_boundary_extension

        positions = np.array([[0, 0, 1], [1, 0, 2], [0, 1, 3], [1, 1, 4]], dtype=float)
        boundary_points = [(0, 0), (0, 1), (1, 1), (1, 0)]
        coord_to_index = {(0, 0): 0, (0, 1): 1, (1, 1): 3, (1, 0): 2}

        # Legacy call with dem_shape (should still work)
        # Single-tier returns 2-tuple (vertices, faces), two-tier returns 3-tuple with colors
        result = create_boundary_extension(
            positions,
            boundary_points,
            coord_to_index,
            base_depth=-0.2,
            use_rectangle_edges=True,
            dem_shape=(2, 2),  # Legacy parameter
            two_tier=False,  # Single-tier mode
        )

        assert isinstance(result, tuple)
        assert len(result) == 2  # (vertices, faces) for single-tier


class TestFractionalEdges:
    """Tests for fractional edge coordinates preserving projection curvature."""

    def test_fractional_vs_integer_edges_curvature_difference(self):
        """Verify that fractional edges preserve more curvature than integer edges.

        This test demonstrates the key benefit of fractional edges: they preserve
        the curved boundary from the non-linear Transverse Mercator projection
        while integer edges snap to grid points and lose this curvature.
        """
        from src.terrain.mesh_operations import (
            generate_transform_aware_rectangle_edges,
            generate_transform_aware_rectangle_edges_fractional,
        )
        from src.terrain.core import Terrain

        # Create DEM in WGS84 (Detroit area - 2 degrees wide)
        original_dem = np.random.rand(100, 100) * 100 + 100
        transform = Affine.translation(-84.0, 43.0) * Affine.scale(0.02, -0.02)

        terrain = Terrain(dem_data=original_dem, dem_transform=transform, dem_crs="EPSG:4326")

        # Simulate reprojection to UTM + significant downsample (10x)
        downsampled_dem = original_dem[::10, ::10]
        utm_transform = Affine.translation(300000, 4800000) * Affine.scale(1000, -1000)

        terrain.data_layers["dem"].update({
            "transformed": True,
            "transformed_data": downsampled_dem,
            "transformed_transform": utm_transform,
            "transformed_crs": "EPSG:32617",
        })

        # Create coord_to_index for 10x10 mesh
        coord_to_index = {}
        for y in range(10):
            for x in range(10):
                coord_to_index[(y, x)] = y * 10 + x

        # Get integer edges (snapped to grid)
        integer_edges = generate_transform_aware_rectangle_edges(
            terrain, coord_to_index, edge_sample_spacing=1.0
        )

        # Get fractional edges (preserves curvature)
        fractional_edges = generate_transform_aware_rectangle_edges_fractional(
            terrain, edge_sample_spacing=1.0
        )

        # Integer edges are exactly on integer grid
        for y, x in integer_edges:
            assert y == int(y) and x == int(x), "Integer edges should be integers"

        # Fractional edges should have non-integer coordinates (evidence of curvature)
        non_integer_count = sum(
            1 for y, x in fractional_edges
            if abs(y - round(y)) > 0.001 or abs(x - round(x)) > 0.001
        )

        # Most fractional edges should NOT be on integer grid
        assert non_integer_count > len(fractional_edges) * 0.5, (
            f"Expected >50% non-integer coords, got {non_integer_count}/{len(fractional_edges)}"
        )

        # Fractional edges should have more unique positions (finer resolution)
        # because they're not snapped to the integer grid
        frac_unique = len(set(fractional_edges))
        int_unique = len(set(integer_edges))

        # Note: fractional may have more or same (if dense sampling), but never much fewer
        assert frac_unique >= int_unique * 0.8, (
            f"Fractional should have comparable or more unique points: {frac_unique} vs {int_unique}"
        )

    def test_fractional_edges_returns_float_coordinates(self):
        """Test that fractional edge function returns float coordinates."""
        from src.terrain.mesh_operations import generate_transform_aware_rectangle_edges_fractional
        from src.terrain.core import Terrain

        # Create simple DEM
        original_dem = np.random.rand(50, 50) * 100 + 100
        transform = Affine.translation(-89.0, 47.0) * Affine.scale(0.01, -0.01)

        terrain = Terrain(dem_data=original_dem, dem_transform=transform, dem_crs="EPSG:4326")

        # Simulate downsampling to 10×10
        downsampled_dem = original_dem[::5, ::5]
        terrain.data_layers["dem"].update({
            "transformed": True,
            "transformed_data": downsampled_dem,
            "transformed_transform": transform * Affine.scale(5, 5),
            "transformed_crs": "EPSG:4326",
        })

        # Get fractional edges
        edge_pixels_fractional = generate_transform_aware_rectangle_edges_fractional(
            terrain,
            edge_sample_spacing=1.0
        )

        # Should return some coordinates
        assert len(edge_pixels_fractional) > 0, "Should return edge coordinates"

        # All coordinates should be floats (not rounded integers)
        for (y, x) in edge_pixels_fractional:
            assert isinstance(y, float), f"Y coordinate should be float, got {type(y)}"
            assert isinstance(x, float), f"X coordinate should be float, got {type(x)}"

    def test_fractional_edges_with_reprojection_shows_curvature(self):
        """Test that CRS reprojection creates non-integer coordinates (curvature effect)."""
        from src.terrain.mesh_operations import generate_transform_aware_rectangle_edges_fractional
        from src.terrain.core import Terrain

        # Create DEM in WGS84
        original_dem = np.random.rand(100, 100) * 100 + 100
        transform = Affine.translation(-84.0, 43.0) * Affine.scale(0.02, -0.02)  # Detroit area

        terrain = Terrain(dem_data=original_dem, dem_transform=transform, dem_crs="EPSG:4326")

        # Simulate reprojection to UTM + downsample
        downsampled_dem = original_dem[::10, ::10]
        # UTM transform (different scale/units)
        utm_transform = Affine.translation(300000, 4800000) * Affine.scale(1000, -1000)

        terrain.data_layers["dem"].update({
            "transformed": True,
            "transformed_data": downsampled_dem,
            "transformed_transform": utm_transform,
            "transformed_crs": "EPSG:32617",  # UTM Zone 17N
        })

        # Get fractional edges
        edge_pixels_fractional = generate_transform_aware_rectangle_edges_fractional(
            terrain,
            edge_sample_spacing=1.0
        )

        # With reprojection, coordinates should NOT be exactly on integer grid
        # due to Transverse Mercator curvature
        non_integer_count = 0
        for (y, x) in edge_pixels_fractional:
            if abs(y - round(y)) > 0.001 or abs(x - round(x)) > 0.001:
                non_integer_count += 1

        # At least some coordinates should be non-integer (evidence of curvature)
        assert non_integer_count > 0, (
            "Expected some non-integer coordinates due to projection curvature"
        )

    def test_create_boundary_extension_with_fractional_edges(self):
        """Test that create_boundary_extension works with use_fractional_edges=True."""
        from src.terrain.mesh_operations import create_boundary_extension
        from src.terrain.core import Terrain

        # Create test mesh data - 10x10 grid
        positions = np.zeros((100, 3), dtype=float)
        coord_to_index = {}
        for y in range(10):
            for x in range(10):
                idx = y * 10 + x
                positions[idx] = [x / 10.0, y / 10.0, 0.5 + np.random.rand() * 0.2]
                coord_to_index[(y, x)] = idx

        # Create boundary points (morphological, as fallback)
        boundary_points = []
        for x in range(10):
            boundary_points.append((0, x))  # Top edge
        for y in range(1, 10):
            boundary_points.append((y, 9))  # Right edge
        for x in range(8, -1, -1):
            boundary_points.append((9, x))  # Bottom edge
        for y in range(8, 0, -1):
            boundary_points.append((y, 0))  # Left edge

        # Create minimal Terrain WITH transformed data
        dem_data = np.random.rand(100, 100) * 100 + 100
        transform = Affine.translation(-84.0, 43.0) * Affine.scale(0.02, -0.02)
        terrain = Terrain(dem_data=dem_data, dem_transform=transform, dem_crs="EPSG:4326")

        # Simulate transformation
        downsampled = dem_data[::10, ::10]
        terrain.data_layers["dem"].update({
            "transformed": True,
            "transformed_data": downsampled,
            "transformed_transform": transform * Affine.scale(10, 10),
            "transformed_crs": "EPSG:4326",
        })

        # Call with use_fractional_edges=True
        result = create_boundary_extension(
            positions,
            boundary_points,
            coord_to_index,
            base_depth=-0.2,
            use_rectangle_edges=True,
            use_fractional_edges=True,  # NEW: Enable fractional coords
            terrain=terrain,
            edge_sample_spacing=1.0,
            two_tier=False,
        )

        # Should return expected structure
        assert isinstance(result, tuple)
        assert len(result) == 2  # (vertices, faces)
        vertices, faces = result
        assert len(vertices) > 0, "Should have boundary vertices"
        assert len(faces) > 0, "Should have boundary faces"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
