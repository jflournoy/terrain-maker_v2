"""
Tests for scene setup operations.

Tests Blender scene, camera, and lighting setup functions.
"""

import pytest

# These tests require Blender environment
pytest.importorskip("bpy")

import numpy as np


class TestClearScene:
    """Tests for clear_scene function."""

    def test_clear_scene_imports(self):
        """Test that clear_scene can be imported."""
        from src.terrain.scene_setup import clear_scene

        assert callable(clear_scene)

    def test_clear_scene_removes_objects(self):
        """Test that clear_scene removes all objects."""
        import bpy
        from src.terrain.scene_setup import clear_scene

        # Add some test objects
        bpy.ops.mesh.primitive_cube_add()
        bpy.ops.mesh.primitive_uv_sphere_add()

        initial_count = len(bpy.data.objects)
        assert initial_count >= 2

        clear_scene()

        # Should have no objects
        assert len(bpy.data.objects) == 0


class TestSetupCamera:
    """Tests for setup_camera function."""

    def test_setup_camera_imports(self):
        """Test that setup_camera can be imported."""
        from src.terrain.scene_setup import setup_camera

        assert callable(setup_camera)

    def test_setup_camera_perspective(self):
        """Test creating a perspective camera."""
        import bpy
        from src.terrain.scene_setup import setup_camera

        # Clear scene first
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj)

        camera = setup_camera(
            camera_angle=(0, 0, 0), camera_location=(0, 0, 5), scale=10, camera_type="PERSP"
        )

        assert camera is not None
        assert camera.data.type == "PERSP"
        assert camera.location[2] == 5

    def test_setup_camera_orthographic(self):
        """Test creating an orthographic camera."""
        import bpy
        from src.terrain.scene_setup import setup_camera

        # Clear scene
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj)

        camera = setup_camera(
            camera_angle=(0, 0, 0), camera_location=(0, 0, 10), scale=15, camera_type="ORTHO"
        )

        assert camera.data.type == "ORTHO"
        assert camera.data.ortho_scale == 15

    def test_setup_camera_invalid_type_raises(self):
        """Test that invalid camera type raises ValueError."""
        from src.terrain.scene_setup import setup_camera

        with pytest.raises(ValueError):
            setup_camera(
                camera_angle=(0, 0, 0), camera_location=(0, 0, 5), scale=10, camera_type="INVALID"
            )


class TestSetupLight:
    """Tests for setup_light function."""

    def test_setup_light_imports(self):
        """Test that setup_light can be imported."""
        from src.terrain.scene_setup import setup_light

        assert callable(setup_light)

    def test_setup_light_creates_sun(self):
        """Test that setup_light creates a sun light."""
        import bpy
        from src.terrain.scene_setup import setup_light

        # Clear lights
        for obj in list(bpy.data.objects):
            if obj.type == "LIGHT":
                bpy.data.objects.remove(obj)

        light = setup_light(location=(1, 2, 3), angle=5, energy=10)

        assert light is not None
        assert light.data.type == "SUN"
        # Blender stores angle in radians, not degrees, so just check it's set
        assert light.data.angle > 0
        assert light.data.energy == 10


class TestSetupCameraAndLight:
    """Tests for setup_camera_and_light function."""

    def test_setup_camera_and_light_imports(self):
        """Test that setup_camera_and_light can be imported."""
        from src.terrain.scene_setup import setup_camera_and_light

        assert callable(setup_camera_and_light)

    def test_setup_camera_and_light_creates_both(self):
        """Test that both camera and light are created."""
        import bpy
        from src.terrain.scene_setup import setup_camera_and_light

        # Clear scene
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj)

        camera, light = setup_camera_and_light(
            camera_angle=(0, 0, 0), camera_location=(0, 0, 5), scale=10
        )

        assert camera is not None
        assert light is not None
        assert camera.data.type in ["PERSP", "ORTHO"]
        assert light.data.type == "SUN"


class TestPositionCameraRelative:
    """Tests for position_camera_relative function."""

    def test_position_camera_relative_imports(self):
        """Test that position_camera_relative can be imported."""
        from src.terrain.scene_setup import position_camera_relative

        assert callable(position_camera_relative)

    def test_position_camera_relative_south(self):
        """Test positioning camera south of mesh."""
        import bpy
        from src.terrain.scene_setup import position_camera_relative

        # Clear scene and create a simple mesh
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj)

        # Create test cube
        bpy.ops.mesh.primitive_cube_add()
        mesh_obj = bpy.context.active_object

        camera = position_camera_relative(mesh_obj, direction="south", distance=2.0)

        assert camera is not None
        # South means negative Y
        assert camera.location[1] < 0

    def test_position_camera_relative_above(self):
        """Test positioning camera above mesh."""
        import bpy
        from src.terrain.scene_setup import position_camera_relative

        # Clear and create mesh
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj)

        bpy.ops.mesh.primitive_cube_add()
        mesh_obj = bpy.context.active_object

        camera = position_camera_relative(mesh_obj, direction="above")

        # Above means high Z
        assert camera.location[2] > mesh_obj.location[2]

    def test_position_camera_relative_invalid_direction_raises(self):
        """Test that invalid direction raises ValueError."""
        import bpy
        from src.terrain.scene_setup import position_camera_relative

        # Create test mesh
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj)

        bpy.ops.mesh.primitive_cube_add()
        mesh_obj = bpy.context.active_object

        with pytest.raises(ValueError):
            position_camera_relative(mesh_obj, direction="invalid_direction")


class TestSetupWorldAtmosphere:
    """Tests for setup_world_atmosphere function."""

    def test_setup_world_atmosphere_imports(self):
        """Test that setup_world_atmosphere can be imported."""
        from src.terrain.scene_setup import setup_world_atmosphere

        assert callable(setup_world_atmosphere)

    def test_setup_world_atmosphere_creates_world(self):
        """Test that world atmosphere is created."""
        import bpy
        from src.terrain.scene_setup import setup_world_atmosphere

        world = setup_world_atmosphere(density=0.05, anisotropy=0.1)

        assert world is not None
        assert world.use_nodes is True
        # Should have volume node
        assert any("Volume" in node.bl_idname for node in world.node_tree.nodes)

    def test_setup_world_atmosphere_default_params(self):
        """Test world atmosphere with default parameters."""
        from src.terrain.scene_setup import setup_world_atmosphere

        world = setup_world_atmosphere()

        assert world is not None
        assert world.use_nodes is True
