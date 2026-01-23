"""
Tests for color computation and application in mesh creation.

Tests the full pipeline: set_color_mapping -> compute_colors -> create_mesh -> apply colors.
Focuses on the grid-space vs vertex-space color handling bug.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile


class TestColorComputationShape:
    """Tests for color computation output shapes."""

    def test_compute_multi_overlay_colors_returns_vertex_space(self):
        """_compute_multi_overlay_colors should return vertex-space colors (N, 4)."""
        pytest.importorskip("bpy")
        import bpy
        from src.terrain.core import Terrain
        from rasterio.transform import Affine

        # Create a simple DEM
        dem = np.ones((10, 10), dtype=np.float32) * 100.0
        transform = Affine.identity()

        # Create terrain
        terrain = Terrain(dem_data=dem, dem_crs="EPSG:4326", dem_transform=transform)

        # Add a simple score layer
        scores = np.random.rand(10, 10).astype(np.float32)
        terrain.add_data_layer("scores", scores, transform, "EPSG:4326", target_layer="dem")

        # Apply minimal transforms via configure_for_target_vertices
        terrain.configure_for_target_vertices(target_vertices=50, method="average")

        # Apply transforms (required for create_mesh)
        terrain.add_transform(lambda data, transform: (data, transform))  # Identity transform
        terrain.apply_transforms()

        # Set up color mapping
        from src.terrain.color_mapping import elevation_colormap

        terrain.set_multi_color_mapping(
            base_colormap=lambda score: elevation_colormap(
                score / np.nanmax(score), cmap_name="viridis", min_elev=0.0, max_elev=1.0
            ),
            base_source_layers=["scores"],
            overlays=[],
        )

        # Create mesh to set up y_valid and x_valid
        mesh_obj = terrain.create_mesh(scale_factor=100, height_scale=1.0)

        # Compute colors (now that mesh vertices exist)
        colors = terrain.compute_colors()

        # Get valid vertices count
        n_vertices = len(terrain.y_valid)

        # Colors should be vertex-space (N, 3/4), not grid-space (H, W, 3/4)
        assert colors.ndim == 2, f"Colors should be 2D vertex-space, got shape {colors.shape}"
        assert colors.shape[1] in (3, 4), f"Colors should have 3 or 4 channels, got {colors.shape[1]}"
        assert (
            colors.shape[0] == n_vertices
        ), f"Color count {colors.shape[0]} should match vertex count {n_vertices}"

        # Clean up
        if mesh_obj is not None:
            mesh_data = mesh_obj.data
            bpy.data.objects.remove(mesh_obj)
            bpy.data.meshes.remove(mesh_data)

    def test_create_blender_mesh_with_vertex_space_colors(self):
        """create_blender_mesh should handle vertex-space colors (N, 4) correctly."""
        pytest.importorskip("bpy")
        import bpy
        from src.terrain.blender_integration import create_blender_mesh

        # Simple quad mesh
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32
        )
        faces = [(0, 1, 2, 3)]

        # Vertex-space colors (N, 4) - one color per vertex
        vertex_space_colors = np.array(
            [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 255, 0, 255]],
            dtype=np.uint8,
        )

        y_valid = np.array([0, 0, 1, 1])
        x_valid = np.array([0, 1, 1, 0])

        # This should not raise an error
        obj = create_blender_mesh(
            vertices,
            faces,
            colors=vertex_space_colors,
            y_valid=y_valid,
            x_valid=x_valid,
            name="VertexSpaceColorMesh",
        )

        assert obj is not None
        assert obj.name == "VertexSpaceColorMesh"
        assert len(obj.data.vertices) == 4

        # Cleanup
        mesh_data = obj.data
        bpy.data.objects.remove(obj)
        bpy.data.meshes.remove(mesh_data)

    def test_create_blender_mesh_with_grid_space_colors(self):
        """create_blender_mesh should handle grid-space colors (H, W, 4) correctly."""
        pytest.importorskip("bpy")
        import bpy
        from src.terrain.blender_integration import create_blender_mesh

        # Simple quad mesh
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32
        )
        faces = [(0, 1, 2, 3)]

        # Grid-space colors (H, W, 4)
        grid_space_colors = np.ones((2, 2, 4), dtype=np.uint8) * 128

        y_valid = np.array([0, 0, 1, 1])
        x_valid = np.array([0, 1, 1, 0])

        # This should not raise an error
        obj = create_blender_mesh(
            vertices,
            faces,
            colors=grid_space_colors,
            y_valid=y_valid,
            x_valid=x_valid,
            name="GridSpaceColorMesh",
        )

        assert obj is not None
        assert obj.name == "GridSpaceColorMesh"
        assert len(obj.data.vertices) == 4

        # Cleanup
        mesh_data = obj.data
        bpy.data.objects.remove(obj)
        bpy.data.meshes.remove(mesh_data)

    def test_vertex_colors_are_non_uniform(self, sample_dem):
        """Vertex colors should vary across mesh when data is non-uniform."""
        from src.terrain.core import Terrain
        from rasterio.transform import Affine
        from src.terrain.color_mapping import elevation_colormap

        # Create DEM with variation (sample_dem has peak in center)
        dem = sample_dem  # Has Gaussian peak
        transform = Affine.identity()

        terrain = Terrain(dem_data=dem, dem_crs="EPSG:4326", dem_transform=transform)

        # Set color mapping based on elevation
        terrain.set_color_mapping(
            lambda dem: elevation_colormap(
                (dem - np.nanmin(dem)) / (np.nanmax(dem) - np.nanmin(dem)),
                cmap_name="viridis",
                min_elev=0.0,
                max_elev=1.0,
            ),
            source_layers=["dem"],
        )

        # Need to configure for target vertices
        terrain.configure_for_target_vertices(target_vertices=50, method="average")

        # Compute colors
        colors = terrain.compute_colors()

        # Compute color variation (standard deviation)
        color_std = np.std(colors[:, :3], axis=0)  # RGB channels only
        max_std = np.max(color_std)

        # Colors should be non-uniform (high variation)
        # If colors are uniform white/gray, max_std would be near 0
        assert (
            max_std > 10.0
        ), f"Color variation too low: {max_std}. Colors may be uniform instead of varying."


class TestEndToEndColorPipeline:
    """Tests for the complete color computation and application pipeline."""

    def test_multi_overlay_color_mapping_end_to_end(self):
        """Full pipeline: set_multi_color_mapping -> create_mesh -> verify colors applied."""
        pytest.importorskip("bpy")
        import bpy
        from src.terrain.core import Terrain
        from src.terrain.color_mapping import elevation_colormap
        from rasterio.transform import Affine

        # Create test data
        dem = np.ones((20, 20), dtype=np.float32) * 100.0
        dem[5:15, 5:15] = 150.0  # Add a region with different elevation

        transform = Affine.identity()
        terrain = Terrain(dem_data=dem, dem_crs="EPSG:4326", dem_transform=transform)

        # Add overlay data (e.g., score layer)
        scores = np.zeros((20, 20), dtype=np.float32)
        scores[5:15, 5:15] = 1.0  # High scores in center region

        terrain.add_data_layer("scores", scores, transform, "EPSG:4326", target_layer="dem")

        # Configure for target vertices
        terrain.configure_for_target_vertices(target_vertices=50, method="average")

        # Apply transforms (required for create_mesh)
        terrain.add_transform(lambda data, transform: (data, transform))  # Identity transform
        terrain.apply_transforms()

        # Set up multi-overlay mapping
        terrain.set_multi_color_mapping(
            base_colormap=lambda dem: elevation_colormap(
                (dem - np.nanmin(dem)) / (np.nanmax(dem) - np.nanmin(dem) + 1e-8),
                cmap_name="viridis",
                min_elev=0.0,
                max_elev=1.0,
            ),
            base_source_layers=["dem"],
            overlays=[
                {
                    "colormap": lambda score: elevation_colormap(
                        score, cmap_name="plasma", min_elev=0.0, max_elev=1.0
                    ),
                    "source_layers": ["scores"],
                    "priority": 10,
                }
            ],
        )

        # Create mesh
        mesh_obj = terrain.create_mesh(
            scale_factor=100, height_scale=1.0, center_model=True
        )

        # Verify mesh was created
        assert mesh_obj is not None
        assert isinstance(mesh_obj, bpy.types.Object)
        assert len(mesh_obj.data.vertices) > 0

        # Verify mesh has vertex colors
        has_colors = len(mesh_obj.data.vertex_colors) > 0
        assert has_colors, "Mesh should have vertex colors applied"

        # Cleanup
        mesh_data = mesh_obj.data
        bpy.data.objects.remove(mesh_obj)
        bpy.data.meshes.remove(mesh_data)

    def test_colors_actually_applied_to_vertices(self):
        """Verify that computed colors are actually applied to mesh vertices."""
        pytest.importorskip("bpy")
        import bpy
        from src.terrain.core import Terrain
        from src.terrain.color_mapping import elevation_colormap
        from rasterio.transform import Affine

        # Create gradient DEM
        dem = np.zeros((10, 10), dtype=np.float32)
        for i in range(10):
            dem[i, :] = 100.0 + i * 10.0  # Gradient from 100 to 190

        transform = Affine.identity()
        terrain = Terrain(dem_data=dem, dem_crs="EPSG:4326", dem_transform=transform)

        # Configure for target vertices
        terrain.configure_for_target_vertices(target_vertices=50, method="average")

        # Apply transforms (required for create_mesh)
        terrain.add_transform(lambda data, transform: (data, transform))  # Identity transform
        terrain.apply_transforms()

        # Set simple color mapping
        terrain.set_color_mapping(
            lambda dem: elevation_colormap(
                (dem - np.nanmin(dem)) / (np.nanmax(dem) - np.nanmin(dem) + 1e-8),
                cmap_name="viridis",
            ),
            source_layers=["dem"],
        )

        # Create mesh
        mesh_obj = terrain.create_mesh(scale_factor=100, height_scale=1.0)

        assert mesh_obj is not None

        # Get actual vertex colors from mesh
        if len(mesh_obj.data.vertex_colors) > 0:
            color_layer = mesh_obj.data.vertex_colors[0]
            colors = np.array(
                [color_layer.data[i].color for i in range(len(color_layer.data))]
            )

            # Colors should not all be the same (gradient should produce variation)
            color_variation = np.std(colors[:, :3], axis=0)
            max_variation = np.max(color_variation)

            assert (
                max_variation > 0.01
            ), f"Vertex colors should vary, got max std={max_variation}"

        # Cleanup
        mesh_data = mesh_obj.data
        bpy.data.objects.remove(mesh_obj)
        bpy.data.meshes.remove(mesh_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
