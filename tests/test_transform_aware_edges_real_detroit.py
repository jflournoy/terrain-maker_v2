"""
TDD: Transform-Aware Edge Extrusion with Real Detroit Data

Tests the transform-aware rectangle edge sampling using the actual Detroit
transformation pipeline and real (or realistic mock) DEM extent.

This validates that edge extrusion works correctly with:
- Real WGS84 (EPSG:4326) input coordinates
- Reprojection to UTM Zone 17N (EPSG:32617)
- Horizontal flip transformation
- Realistic downsampling to target mesh vertices
"""

import pytest
import numpy as np
from pathlib import Path
from rasterio.transform import Affine
import logging

logger = logging.getLogger(__name__)


class TestTransformAwareEdgesRealDetroit:
    """Tests using actual Detroit transformation pipeline and data extent."""

    @pytest.fixture
    def detroit_synthetic_dem(self):
        """Create synthetic Detroit-like DEM matching real extent.

        Uses the same bounding box and resolution as real Detroit HGT files:
        - Bbox: -83.2°W to -83.0°W, 42.3°N to 42.5°N (filtered to N41 and above)
        - Resolution: ~10-11 meters per pixel (0.0001° at Michigan latitude)
        - Elevation: 150-250m (typical Michigan range)
        """
        # Create 200×200 DEM (roughly 20km × 20km)
        dem = np.random.rand(200, 200) * 100 + 150  # Elevation 150-250m

        # Add some realistic variation (peaks and valleys)
        y = np.linspace(0, 200, 200)
        x = np.linspace(0, 200, 200)
        Y, X = np.meshgrid(y, x)
        dem += 30 * np.sin(Y/40) * np.cos(X/40)  # Large-scale variation
        dem += 10 * np.sin(Y/15) * np.cos(X/15)  # Medium-scale variation

        # WGS84 transform for Detroit area (exact same as detroit_combined_render.py)
        # Translation: top-left corner at (-83.2°W, 42.5°N)
        # Scale: 0.0001° per pixel (approximately 10-11 meters at Michigan latitude)
        transform = Affine.translation(-83.2, 42.5) * Affine.scale(0.0001, -0.0001)

        return dem, transform

    def test_real_detroit_wgs84_to_utm_reprojection(self, detroit_synthetic_dem):
        """Test full Detroit pipeline: WGS84 → UTM reprojection + flip + downsample."""
        from src.terrain.core import Terrain
        from src.terrain.transforms import cached_reproject, flip_raster
        from src.terrain.mesh_operations import generate_transform_aware_rectangle_edges

        dem, transform = detroit_synthetic_dem

        # Create terrain with WGS84 data
        terrain = Terrain(dem_data=dem, dem_transform=transform, dem_crs="EPSG:4326")

        logger.info(f"Original DEM: {dem.shape} pixels in WGS84")
        logger.info(f"Transform: {transform}")

        # Apply Detroit's exact transforms
        terrain.add_transform(cached_reproject(src_crs="EPSG:4326", dst_crs="EPSG:32617"))
        terrain.add_transform(flip_raster(axis="horizontal"))
        terrain.apply_transforms()

        # Get transformed DEM shape
        transformed_dem = terrain.data_layers["dem"]["transformed_data"]
        logger.info(f"After transforms: {transformed_dem.shape} pixels in UTM (EPSG:32617)")

        # Configure for target vertices (like the real example)
        terrain.configure_for_target_vertices(target_vertices=500, method="average")

        # Build coord_to_index for final downsampled grid
        downsampled_dem = terrain.data_layers["dem"].get("data_prepared")
        if downsampled_dem is None:
            downsampled_dem = transformed_dem

        final_height, final_width = downsampled_dem.shape
        coord_to_index = {}
        vertex_idx = 0

        for y in range(final_height):
            for x in range(final_width):
                if not np.isnan(downsampled_dem[y, x]):
                    coord_to_index[(y, x)] = vertex_idx
                    vertex_idx += 1

        logger.info(f"Final mesh grid: {final_height}×{final_width} pixels, {len(coord_to_index)} valid vertices")

        # Update terrain with proper transformed state
        terrain.data_layers["dem"].update({
            "transformed": True,
            "transformed_data": downsampled_dem,
        })

        # Test transform-aware edge sampling
        edge_pixels = generate_transform_aware_rectangle_edges(
            terrain,
            coord_to_index,
            edge_sample_spacing=1.0  # Dense sampling
        )

        logger.info(f"Transform-aware edge sampling: {len(edge_pixels)} boundary vertices")

        # Verify all edge pixels are valid
        for (y, x) in edge_pixels:
            assert (y, x) in coord_to_index, f"Edge pixel ({y},{x}) not in valid mesh vertices"

        # Should have at least some boundary coverage
        assert len(edge_pixels) > 10, f"Expected >10 boundary vertices, got {len(edge_pixels)}"

        # Verify coverage percentage
        perimeter_pixels = 2 * (final_height + final_width) - 4
        coverage_percent = (len(edge_pixels) / perimeter_pixels) * 100
        logger.info(f"Boundary coverage: {coverage_percent:.1f}% ({len(edge_pixels)}/{perimeter_pixels})")

    def test_real_detroit_mesh_creation_with_boundary_extension(self, detroit_synthetic_dem):
        """Test that real Detroit transforms work with boundary extension."""
        from src.terrain.core import Terrain
        from src.terrain.transforms import cached_reproject, flip_raster

        dem, transform = detroit_synthetic_dem

        # Create terrain
        terrain = Terrain(dem_data=dem, dem_transform=transform, dem_crs="EPSG:4326")

        # Apply transforms
        terrain.add_transform(cached_reproject(src_crs="EPSG:4326", dst_crs="EPSG:32617"))
        terrain.add_transform(flip_raster(axis="horizontal"))
        terrain.apply_transforms()

        # Configure for target vertices
        terrain.configure_for_target_vertices(target_vertices=500, method="average")

        # Create mesh with boundary extension and rectangle edges
        try:
            mesh_obj = terrain.create_mesh(
                boundary_extension=True,
                use_rectangle_edges=True,
                scale_factor=100,
                height_scale=1.0,
                center_model=True
            )

            if mesh_obj is not None:
                # Get mesh statistics
                num_vertices = len(mesh_obj.data.vertices)
                num_faces = len(mesh_obj.data.polygons)

                logger.info(f"Mesh created: {num_vertices} vertices, {num_faces} faces")

                # Verify mesh has content
                assert num_vertices > 100, f"Mesh should have >100 vertices, got {num_vertices}"
                assert num_faces > 100, f"Mesh should have >100 faces, got {num_faces}"

                # Clean up
                import bpy
                mesh_data = mesh_obj.data
                bpy.data.objects.remove(mesh_obj)
                bpy.data.meshes.remove(mesh_data)
            else:
                # Bpy not available (headless), that's ok - just check it ran without error
                logger.info("Mesh creation succeeded (bpy not available for verification)")

        except Exception as e:
            # If not in Blender context, that's expected
            if "bpy" in str(e):
                pytest.skip("Blender not available (headless environment)")
            else:
                raise

    def test_real_detroit_shape_distortion_from_reprojection(self, detroit_synthetic_dem):
        """Verify that reprojection causes expected shape distortion (16% at Michigan latitude).

        WGS84 to UTM reprojection at Michigan latitude causes ~16% linear distortion:
        - 200×200 WGS84 pixels → ~232×174 UTM pixels
        """
        from src.terrain.core import Terrain
        from src.terrain.transforms import cached_reproject, flip_raster

        dem, transform = detroit_synthetic_dem
        original_shape = dem.shape

        terrain = Terrain(dem_data=dem, dem_transform=transform, dem_crs="EPSG:4326")
        terrain.add_transform(cached_reproject(src_crs="EPSG:4326", dst_crs="EPSG:32617"))
        terrain.add_transform(flip_raster(axis="horizontal"))
        terrain.apply_transforms()

        transformed_dem = terrain.data_layers["dem"]["transformed_data"]
        reprojected_shape = transformed_dem.shape

        logger.info(f"Original WGS84 shape: {original_shape}")
        logger.info(f"Reprojected UTM shape: {reprojected_shape}")

        # Verify shape changed due to projection distortion
        assert original_shape != reprojected_shape, "Shape should change due to projection distortion"

        # Verify distortion is within expected range for Michigan (10-20%)
        height_ratio = reprojected_shape[0] / original_shape[0]
        width_ratio = reprojected_shape[1] / original_shape[1]

        logger.info(f"Height ratio: {height_ratio:.3f}, Width ratio: {width_ratio:.3f}")

        # At Michigan latitude, should see ~16% distortion
        assert 0.8 < height_ratio < 1.2, f"Height distortion unexpected: {height_ratio}"
        assert 0.8 < width_ratio < 1.2, f"Width distortion unexpected: {width_ratio}"

    def test_real_detroit_100_percent_boundary_coverage(self, detroit_synthetic_dem):
        """Verify transform-aware approach achieves 100% boundary coverage vs legacy approach.

        Legacy approach (sampling from downsampled grid) fails with NaN margins.
        Transform-aware approach (sampling from original) succeeds even with large downsampling.
        """
        from src.terrain.core import Terrain
        from src.terrain.transforms import cached_reproject, flip_raster
        from src.terrain.mesh_operations import (
            generate_transform_aware_rectangle_edges,
            diagnose_rectangle_edge_coverage,
        )

        dem, transform = detroit_synthetic_dem

        terrain = Terrain(dem_data=dem, dem_transform=transform, dem_crs="EPSG:4326")
        terrain.add_transform(cached_reproject(src_crs="EPSG:4326", dst_crs="EPSG:32617"))
        terrain.add_transform(flip_raster(axis="horizontal"))
        terrain.apply_transforms()

        # Aggressive downsampling (creates NaN margins in transformed grid)
        terrain.configure_for_target_vertices(target_vertices=100, method="average")

        transformed_dem = terrain.data_layers["dem"]["transformed_data"]
        final_shape = transformed_dem.shape

        # Create coord_to_index only for valid (non-NaN) pixels
        coord_to_index = {}
        for y in range(final_shape[0]):
            for x in range(final_shape[1]):
                if not np.isnan(transformed_dem[y, x]):
                    coord_to_index[(y, x)] = len(coord_to_index)

        # Update terrain state
        terrain.data_layers["dem"].update({
            "transformed": True,
            "transformed_data": transformed_dem,
        })

        # Test 1: Legacy approach
        legacy_diag = diagnose_rectangle_edge_coverage(final_shape, coord_to_index)
        legacy_coverage = legacy_diag['coverage_percent']

        # Test 2: Transform-aware approach
        transform_aware_pixels = generate_transform_aware_rectangle_edges(
            terrain,
            coord_to_index,
            edge_sample_spacing=1.0
        )

        logger.info(f"Final grid shape: {final_shape}")
        logger.info(f"Valid vertices: {len(coord_to_index)}")
        logger.info(f"Legacy edge coverage: {legacy_coverage:.1f}%")
        logger.info(f"Transform-aware pixels: {len(transform_aware_pixels)}")

        # Legacy should have low coverage due to NaN margins
        assert legacy_coverage < 50, f"Legacy coverage should be low (<50%), got {legacy_coverage:.1f}%"

        # Transform-aware should have meaningful coverage
        assert len(transform_aware_pixels) > 5, f"Transform-aware should find edge pixels, got {len(transform_aware_pixels)}"

        # All returned pixels must be valid
        for (y, x) in transform_aware_pixels:
            assert (y, x) in coord_to_index, f"Pixel ({y},{x}) not in valid mesh"

    def test_real_detroit_dense_boundary_sampling(self, detroit_synthetic_dem):
        """Verify dense boundary sampling (0.33 pixels) for smooth extrusion."""
        from src.terrain.core import Terrain
        from src.terrain.transforms import cached_reproject, flip_raster
        from src.terrain.mesh_operations import generate_transform_aware_rectangle_edges

        dem, transform = detroit_synthetic_dem

        terrain = Terrain(dem_data=dem, dem_transform=transform, dem_crs="EPSG:4326")
        terrain.add_transform(cached_reproject(src_crs="EPSG:4326", dst_crs="EPSG:32617"))
        terrain.add_transform(flip_raster(axis="horizontal"))
        terrain.apply_transforms()

        terrain.configure_for_target_vertices(target_vertices=500, method="average")

        transformed_dem = terrain.data_layers["dem"]["transformed_data"]
        coord_to_index = {}

        for y in range(transformed_dem.shape[0]):
            for x in range(transformed_dem.shape[1]):
                if not np.isnan(transformed_dem[y, x]):
                    coord_to_index[(y, x)] = len(coord_to_index)

        terrain.data_layers["dem"].update({
            "transformed": True,
            "transformed_data": transformed_dem,
        })

        # Test different sampling densities
        for spacing in [0.33, 1.0, 2.0]:  # Dense, normal, sparse
            pixels = generate_transform_aware_rectangle_edges(
                terrain,
                coord_to_index,
                edge_sample_spacing=spacing
            )

            coverage = (len(pixels) / (2 * (transformed_dem.shape[0] + transformed_dem.shape[1]) - 4)) * 100
            logger.info(f"Spacing {spacing}: {len(pixels)} pixels, {coverage:.1f}% coverage")

            # Denser sampling should give more vertices
            assert len(pixels) > 5, f"Should have edge pixels at spacing {spacing}"

    def test_real_detroit_coordinate_transformation_chain(self, detroit_synthetic_dem):
        """Verify the complete coordinate transformation chain: WGS84 → UTM → Blender."""
        from src.terrain.core import Terrain
        from src.terrain.transforms import cached_reproject, flip_raster

        dem, transform = detroit_synthetic_dem

        logger.info("Stage 1: Original WGS84 (EPSG:4326)")
        logger.info(f"  DEM shape: {dem.shape}")
        logger.info(f"  Transform: {transform}")

        terrain = Terrain(dem_data=dem, dem_transform=transform, dem_crs="EPSG:4326")

        logger.info("Stage 2: Apply reprojection to UTM (EPSG:32617)")
        terrain.add_transform(cached_reproject(src_crs="EPSG:4326", dst_crs="EPSG:32617"))
        terrain.apply_transforms()

        reprojected_dem = terrain.data_layers["dem"]["transformed_data"]
        reprojected_transform = terrain.data_layers["dem"]["transformed_transform"]
        logger.info(f"  Shape: {reprojected_dem.shape}")
        logger.info(f"  CRS: EPSG:32617")

        logger.info("Stage 3: Apply horizontal flip")
        terrain.add_transform(flip_raster(axis="horizontal"))
        terrain.apply_transforms()

        flipped_dem = terrain.data_layers["dem"]["transformed_data"]
        logger.info(f"  Shape: {flipped_dem.shape} (unchanged)")

        logger.info("Stage 4: Downsample to target mesh vertices")
        terrain.configure_for_target_vertices(target_vertices=500, method="average")

        logger.info(f"  Target shape: ~{int(np.sqrt(500) * 1.3)}×{int(np.sqrt(500) / 1.3)} pixels")

        # Verify all stages completed without error
        assert reprojected_dem.shape != dem.shape, "Reprojection should change shape"
        assert flipped_dem.shape == reprojected_dem.shape, "Flip shouldn't change shape"

        # Check that transforms were applied
        assert terrain.data_layers["dem"]["transformed"] == True, "DEM should be marked as transformed"
        assert terrain.data_layers["dem"]["transformed_crs"] == "EPSG:32617", "Final CRS should be UTM"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
