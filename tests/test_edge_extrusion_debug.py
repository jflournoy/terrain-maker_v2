"""
TDD: Edge Extrusion Transformation Debug Visualizations

Tests for creating diagnostic plots showing:
1. Original rectangle sampling in DEM pixel space
2. Transformed rectangle in final mesh space
3. Boundary point distribution before/after transformations
4. Edge distribution analysis (north/south/east/west)
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile


class TestEdgeExtrusionVisualizations:
    """Test suite for edge extrusion debugging visualizations."""

    def test_visualize_rectangle_sampling_stage(self):
        """Create plot of original rectangle edge sampling in DEM space."""
        from src.terrain.mesh_operations import generate_rectangle_edge_pixels
        from src.terrain.visualization.edge_debug import plot_rectangle_edge_sampling

        # Create a simple 100×100 DEM
        dem_shape = (100, 100)

        # Sample edges at different spacings
        edges_dense = generate_rectangle_edge_pixels(dem_shape, edge_sample_spacing=1.0)
        edges_sparse = generate_rectangle_edge_pixels(dem_shape, edge_sample_spacing=5.0)

        # Verify edges were sampled
        assert len(edges_dense) > 0, "Dense edges should have points"
        assert len(edges_sparse) > 0, "Sparse edges should have points"
        assert len(edges_dense) > len(edges_sparse), "Dense should have more points than sparse"

        # Create temporary directory for plots
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "rectangle_sampling.png"

            # Generate visualization
            plot_rectangle_edge_sampling(
                dem_shape=dem_shape,
                edge_pixels_dense=edges_dense,
                edge_pixels_sparse=edges_sparse,
                output_path=output_path,
            )

            # Verify plot was created
            assert output_path.exists(), f"Plot should be created at {output_path}"
            assert output_path.stat().st_size > 0, "Plot file should not be empty"

    def test_visualize_coordinate_transformation_stages(self):
        """Create plots showing each stage of coordinate transformation."""
        pytest.importorskip("bpy")
        import bpy
        from src.terrain.core import Terrain
        from src.terrain.visualization.edge_debug import plot_transformation_pipeline
        from rasterio.transform import Affine

        # Create a simple test DEM
        dem = np.ones((50, 50), dtype=np.float32) * 100.0
        # Add a central peak to make terrain interesting
        dem[20:30, 20:30] = 150.0

        # Define a transform
        transform = Affine.translation(-89.0, 47.0) * Affine.scale(0.01, -0.01)

        # Create terrain (will use default transforms)
        terrain = Terrain(dem_data=dem, dem_crs="EPSG:4326", dem_transform=transform)

        # Apply transforms
        terrain.add_transform(lambda data, transform: (data, transform))  # Identity transform
        terrain.apply_transforms()

        # Configure for target vertices to create mesh vertex grid
        terrain.configure_for_target_vertices(target_vertices=100, method="average")

        # Create mesh to compute vertex positions (but don't render)
        # This creates terrain.x_valid and terrain.y_valid attributes
        mesh_obj = terrain.create_mesh(scale_factor=100, height_scale=1.0, center_model=True)
        if mesh_obj is not None:
            # Clean up the Blender object (don't render it)
            mesh_data = mesh_obj.data
            bpy.data.objects.remove(mesh_obj)
            bpy.data.meshes.remove(mesh_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Generate transformation visualization
            plot_transformation_pipeline(
                terrain=terrain,
                output_dir=output_dir,
                edge_sample_spacing=2.0,
            )

            # Verify plots were created
            expected_plots = [
                "stage_1_original_rectangle.png",
                "stage_2_geographic_coords.png",
                "stage_3_transformed_rectangle.png",
                "stage_4_mesh_vertices.png",
            ]

            for plot_name in expected_plots:
                plot_path = output_dir / plot_name
                assert plot_path.exists(), f"Expected {plot_name} to be created"

    def test_visualize_edge_distribution(self):
        """Create plot showing point distribution on edges (N/S/E/W)."""
        from src.terrain.mesh_operations import generate_rectangle_edge_pixels
        from src.terrain.visualization.edge_debug import plot_edge_distribution

        dem_shape = (100, 100)
        edge_pixels = generate_rectangle_edge_pixels(dem_shape, edge_sample_spacing=1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "edge_distribution.png"

            # Generate distribution visualization
            plot_edge_distribution(
                boundary_points=edge_pixels,
                output_path=output_path,
                title="Edge Distribution: Original Rectangle",
            )

            # Verify plot was created
            assert output_path.exists(), f"Plot should be created at {output_path}"
            assert output_path.stat().st_size > 0, "Plot file should not be empty"

    def test_visualize_deduplication_impact(self):
        """Show before/after of deduplication on boundary points."""
        from src.terrain.mesh_operations import (
            generate_rectangle_edge_pixels,
            deduplicate_boundary_points,
        )
        from src.terrain.visualization.edge_debug import plot_deduplication_comparison

        dem_shape = (100, 100)
        edge_pixels = generate_rectangle_edge_pixels(dem_shape, edge_sample_spacing=1.0)
        edge_pixels_dedup = deduplicate_boundary_points(edge_pixels)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "deduplication_comparison.png"

            # Generate comparison plot
            plot_deduplication_comparison(
                before=edge_pixels,
                after=edge_pixels_dedup,
                output_path=output_path,
            )

            # Verify plot was created
            assert output_path.exists(), f"Plot should be created at {output_path}"

    def test_visualize_boundary_sorting(self):
        """Show the effect of angular sorting on boundary points."""
        from src.terrain.mesh_operations import (
            generate_rectangle_edge_pixels,
            sort_boundary_points_angular,
        )
        from src.terrain.visualization.edge_debug import plot_sorting_effect

        dem_shape = (100, 100)
        edge_pixels = generate_rectangle_edge_pixels(dem_shape, edge_sample_spacing=1.0)

        # Convert to coordinate space for sorting
        edge_pixels_array = np.array(edge_pixels, dtype=float)

        # For testing, create some "transformed" points with mixed order
        # (simulating what happens after coordinate transformation)
        np.random.seed(42)
        shuffled_indices = np.random.permutation(len(edge_pixels_array))
        shuffled_points = edge_pixels_array[shuffled_indices]

        sorted_points = sort_boundary_points_angular(shuffled_points)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sorting_effect.png"

            # Generate sorting visualization
            plot_sorting_effect(
                before=shuffled_points,
                after=np.array(sorted_points),
                output_path=output_path,
            )

            # Verify plot was created
            assert output_path.exists(), f"Plot should be created at {output_path}"

    def test_full_pipeline_debug_visualization(self, tmp_path):
        """Create comprehensive debug visualization of entire rectangle edge pipeline."""
        from src.terrain.visualization.edge_debug import create_full_pipeline_debug_plot
        from src.terrain.core import Terrain
        from rasterio.transform import Affine

        # Create simple test data
        dem = np.ones((50, 50), dtype=np.float32) * 100.0
        transform = Affine.translation(-89.0, 47.0) * Affine.scale(0.01, -0.01)

        terrain = Terrain(dem_data=dem, dem_crs="EPSG:4326", dem_transform=transform)

        # Generate comprehensive debug plot
        output_path = tmp_path / "full_pipeline_debug.png"

        create_full_pipeline_debug_plot(
            terrain=terrain,
            output_path=output_path,
            edge_sample_spacing=2.0,
        )

        # Verify comprehensive plot was created
        assert output_path.exists(), f"Full pipeline debug plot should exist at {output_path}"
        assert output_path.stat().st_size > 0, "Debug plot should not be empty"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
