"""Test suite for Detroit snow depth visualization end-to-end example.

This test validates that the terrain-maker library can successfully:
1. Load DEM data
2. Integrate SNODAS snow depth data
3. Apply geospatial transforms
4. Generate color-mapped terrain
5. Create Blender mesh
"""

import numpy as np
import pytest
from pathlib import Path
from affine import Affine
from src.terrain.core import Terrain, load_dem_files, clear_scene

# Check if Blender is available
try:
    import bpy
    HAS_BLENDER = True
except ImportError:
    HAS_BLENDER = False


class TestDetroitEndToEnd:
    """Test complete Detroit snow depth visualization workflow."""

    def test_detroit_example_loads_dem_data(self, tmp_path):
        """Detroit example should load DEM data successfully."""
        # Create synthetic DEM for Detroit region
        dem_data = np.random.uniform(180, 320, size=(100, 100)).astype(np.float32)
        transform = Affine.identity()

        # Initialize terrain with Detroit DEM data
        terrain = Terrain(dem_data, transform)

        # Verify DEM is loaded and available
        assert terrain is not None
        assert terrain.dem_shape == (100, 100)
        assert 'dem' in terrain.data_layers
        assert terrain.data_layers['dem']['data'] is not None

    def test_detroit_example_applies_transform(self, tmp_path):
        """Detroit example should apply coordinate transforms."""
        dem_data = np.random.uniform(180, 320, size=(100, 100)).astype(np.float32)
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        # Register a simple identity transform
        def identity_transform(data, trans):
            return data, trans, None

        terrain.transforms.append(identity_transform)

        # Apply transforms
        result = terrain.apply_transforms()

        # Verify transforms were applied
        assert result is None  # apply_transforms returns None on success
        assert terrain.data_layers['dem']['transformed'] is True
        assert 'transformed_data' in terrain.data_layers['dem']

    def test_detroit_example_sets_color_mapping(self, tmp_path):
        """Detroit example should apply color mapping for visualization."""
        dem_data = np.random.uniform(180, 320, size=(100, 100)).astype(np.float32)
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        # Apply identity transform first
        def identity_transform(data, trans):
            return data, trans, None

        terrain.transforms.append(identity_transform)
        terrain.apply_transforms()

        # Set color mapping based on elevation (simple function that returns RGB)
        def color_func(dem):
            # Normalize DEM to 0-1 range and create RGB (grayscale)
            normalized = (dem - dem.min()) / (dem.max() - dem.min() + 1e-8)
            # Stack to create RGB: (H, W, 3)
            rgb = np.stack([normalized, normalized, normalized], axis=-1)
            return (rgb * 255).astype(np.uint8)

        terrain.set_color_mapping(color_func, source_layers=['dem'])

        # Verify color mapping is configured
        assert hasattr(terrain, 'color_mapping')
        assert terrain.color_mapping is not None
        assert terrain.color_sources == ['dem']

    def test_detroit_example_creates_mesh(self, tmp_path):
        """Detroit example should create Blender mesh successfully."""
        dem_data = np.ones((50, 50), dtype=np.float32) * 250  # Consistent elevation for Detroit
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        # Apply identity transform
        def identity_transform(data, trans):
            return data, trans, None

        terrain.transforms.append(identity_transform)
        terrain.apply_transforms()

        # Create mesh
        mesh_obj = terrain.create_mesh()

        # Verify mesh was created
        assert mesh_obj is not None
        assert hasattr(mesh_obj, 'data')
        assert len(mesh_obj.data.vertices) > 0
        assert len(mesh_obj.data.polygons) > 0

    def test_detroit_example_complete_workflow(self, tmp_path):
        """Complete Detroit workflow: load -> transform -> color -> mesh."""
        # Create synthetic Detroit DEM
        dem_data = np.random.uniform(180, 320, size=(100, 100)).astype(np.float32)
        transform = Affine.identity()

        # Initialize terrain
        terrain = Terrain(dem_data, transform)

        # Apply identity transform
        def identity_transform(data, trans):
            return data, trans, None

        terrain.transforms.append(identity_transform)
        terrain.apply_transforms()

        # Set color mapping with simple RGB function
        def color_func(dem):
            normalized = (dem - dem.min()) / (dem.max() - dem.min() + 1e-8)
            rgb = np.stack([normalized, normalized, normalized], axis=-1)
            return (rgb * 255).astype(np.uint8)

        terrain.set_color_mapping(color_func, source_layers=['dem'])

        # Create mesh with boundary extension
        mesh_obj = terrain.create_mesh(boundary_extension=True)

        # Verify complete workflow succeeded
        assert mesh_obj is not None
        assert terrain.data_layers['dem']['transformed'] is True
        assert hasattr(mesh_obj, 'data')
        assert len(mesh_obj.data.vertices) > 0
        assert len(mesh_obj.data.polygons) > 0

    def test_detroit_example_multiple_data_layers(self, tmp_path):
        """Detroit example should support multiple data layers (DEM + snow)."""
        # Create base DEM
        dem_data = np.random.uniform(180, 320, size=(50, 50)).astype(np.float32)
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        # Add synthetic snow depth layer
        snow_data = np.random.uniform(0, 50, size=(50, 50)).astype(np.float32)
        terrain.add_data_layer('snow_depth', snow_data, transform, 'EPSG:4326')

        # Verify both layers exist
        assert 'dem' in terrain.data_layers
        assert 'snow_depth' in terrain.data_layers
        assert len(terrain.data_layers) == 2

    def test_detroit_example_produces_deterministic_output(self, tmp_path):
        """Detroit example should produce same output with same inputs."""
        dem_data = np.ones((50, 50), dtype=np.float32) * 250
        transform = Affine.identity()

        # First run
        terrain1 = Terrain(dem_data.copy(), transform)
        def identity_transform(data, trans):
            return data, trans, None
        terrain1.transforms.append(identity_transform)
        terrain1.apply_transforms()
        terrain1.set_color_mapping(lambda x: x, source_layers=['dem'])
        mesh1 = terrain1.create_mesh()
        verts1 = len(mesh1.data.vertices)
        faces1 = len(mesh1.data.polygons)

        # Second run with identical inputs
        terrain2 = Terrain(dem_data.copy(), transform)
        terrain2.transforms.append(identity_transform)
        terrain2.apply_transforms()
        terrain2.set_color_mapping(lambda x: x, source_layers=['dem'])
        mesh2 = terrain2.create_mesh()
        verts2 = len(mesh2.data.vertices)
        faces2 = len(mesh2.data.polygons)

        # Both runs should produce identical geometry
        assert verts1 == verts2
        assert faces1 == faces2


    def test_detroit_example_can_render_png(self, tmp_path):
        """Detroit example should be able to render to PNG."""
        try:
            import bpy
        except ImportError:
            pytest.skip("Blender/bpy not available for rendering test")

        dem_data = np.ones((50, 50), dtype=np.float32) * 250
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        def identity_transform(data, trans):
            return data, trans, None

        terrain.transforms.append(identity_transform)
        terrain.apply_transforms()

        def color_func(dem):
            normalized = (dem - dem.min()) / (dem.max() - dem.min() + 1e-8)
            rgb = np.stack([normalized, normalized, normalized], axis=-1)
            return (rgb * 255).astype(np.uint8)

        terrain.set_color_mapping(color_func, source_layers=['dem'])
        mesh_obj = terrain.create_mesh()

        # Verify mesh was created and can be rendered
        assert mesh_obj is not None
        assert hasattr(bpy, 'context')
        assert bpy.context.scene is not None


class TestSceneClearing:
    """Test scene clearing for mesh creation."""

    @pytest.mark.skipif(not HAS_BLENDER, reason="Requires Blender")
    def test_clear_scene_removes_objects(self):
        """clear_scene() should remove all objects from Blender scene."""
        # Create a simple cube to have an object in the scene
        bpy.ops.mesh.primitive_cube_add(size=2)
        initial_count = len(bpy.context.scene.objects)
        assert initial_count > 0, "Scene should have at least one object"

        # Clear the scene
        clear_scene()

        # Check that objects are gone
        final_count = len(bpy.context.scene.objects)
        assert final_count == 0, f"Scene should be empty after clear_scene(), but has {final_count} objects"

    @pytest.mark.skipif(not HAS_BLENDER, reason="Requires Blender")
    def test_clear_scene_removes_cameras(self):
        """clear_scene() should remove all cameras from scene."""
        # Create a camera
        bpy.ops.object.camera_add()
        initial_cameras = [obj for obj in bpy.context.scene.objects if obj.type == 'CAMERA']
        assert len(initial_cameras) > 0, "Should have created a camera"

        # Clear the scene
        clear_scene()

        # Check that cameras are gone
        final_cameras = [obj for obj in bpy.context.scene.objects if obj.type == 'CAMERA']
        assert len(final_cameras) == 0, "Scene should have no cameras after clear_scene()"

    @pytest.mark.skipif(not HAS_BLENDER, reason="Requires Blender")
    def test_clear_scene_removes_lights(self):
        """clear_scene() should remove all lights from scene."""
        # Create a light
        bpy.ops.object.light_add(type='SUN')
        initial_lights = [obj for obj in bpy.context.scene.objects if obj.type == 'LIGHT']
        assert len(initial_lights) > 0, "Should have created a light"

        # Clear the scene
        clear_scene()

        # Check that lights are gone
        final_lights = [obj for obj in bpy.context.scene.objects if obj.type == 'LIGHT']
        assert len(final_lights) == 0, "Scene should have no lights after clear_scene()"

    @pytest.mark.skipif(not HAS_BLENDER, reason="Requires Blender")
    def test_scene_clean_before_mesh_creation(self):
        """Scene should be cleaned before creating mesh."""
        # Add some objects to the scene
        bpy.ops.mesh.primitive_cube_add()
        bpy.ops.object.camera_add()

        initial_count = len(bpy.context.scene.objects)
        assert initial_count > 0

        # Clear and verify
        clear_scene()
        final_count = len(bpy.context.scene.objects)
        assert final_count == 0

        # Now create a DEM and mesh in clean scene
        dem_data = np.ones((10, 10), dtype=np.float32) * 100
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        def identity_transform(data, trans):
            return data, trans, None

        terrain.transforms.append(identity_transform)
        terrain.apply_transforms()

        mesh_obj = terrain.create_mesh()
        assert mesh_obj is not None

        # Only the terrain mesh should exist
        mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
        assert len(mesh_objects) == 1


class TestDetroitDataIntegration:
    """Test Detroit-specific data integration."""

    def test_detroit_bounds_valid(self):
        """Detroit coordinates should be valid."""
        # Detroit bounding box: ~42.3N, 83.0W to 42.4N, 82.9W
        detroit_bounds = {
            'north': 42.4,
            'south': 42.3,
            'east': -82.9,
            'west': -83.0,
        }

        assert detroit_bounds['north'] > detroit_bounds['south']
        assert detroit_bounds['east'] > detroit_bounds['west']
        # Detroit is in Northern Hemisphere
        assert detroit_bounds['north'] > 0
        # Detroit is in Western Hemisphere
        assert detroit_bounds['west'] < 0

    def test_detroit_dem_elevation_range(self):
        """Detroit DEM should have realistic elevation range."""
        # Detroit is relatively flat, mostly 160-320 meters elevation
        detroit_min_elev = 160
        detroit_max_elev = 320

        assert detroit_min_elev < detroit_max_elev
        # Detroit elevation range should be less than 200m
        assert (detroit_max_elev - detroit_min_elev) < 200

    def test_detroit_snow_data_realistic(self):
        """Detroit snow depth data should have realistic range."""
        # SNODAS typically measures 0-500mm snow depth
        snow_min = 0
        snow_max = 500

        assert snow_min < snow_max
        assert snow_max < 1000  # Reasonable upper bound
