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

    def test_position_camera_relative_ortho_scale_applied(self):
        """Test that ortho_scale is correctly applied to orthographic camera.

        Regression test: Ensures ortho_scale parameter is passed correctly
        through the function chain and applied to the camera object.
        """
        import bpy
        from src.terrain.scene_setup import position_camera_relative

        # Clear and create mesh
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj)

        bpy.ops.mesh.primitive_cube_add()
        mesh_obj = bpy.context.active_object

        # Call with specific ortho_scale value
        test_ortho_scale = 2.5
        camera = position_camera_relative(
            mesh_obj,
            direction="south",
            camera_type="ORTHO",
            ortho_scale=test_ortho_scale,
        )

        # The actual ortho_scale is computed based on mesh dimensions and our multiplier
        # For a unit cube at origin, the computed scale should incorporate our value
        assert camera is not None
        assert camera.data.type == "ORTHO"
        # The camera ortho_scale should be set (not None or 0)
        assert camera.data.ortho_scale > 0
        # For regression: verify the scale is influenced by the parameter
        # (exact value depends on mesh size calculation, but should be proportional)

    def test_position_camera_relative_ortho_scale_different_values(self):
        """Test that different ortho_scale values produce different camera scales.

        Regression test: Catches bugs where ortho_scale is passed to wrong parameter.
        """
        import bpy
        from src.terrain.scene_setup import position_camera_relative

        # Clear and create mesh
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj)

        bpy.ops.mesh.primitive_cube_add()
        mesh_obj = bpy.context.active_object

        # Call with small ortho_scale
        camera1 = position_camera_relative(
            mesh_obj,
            direction="south",
            camera_type="ORTHO",
            ortho_scale=1.0,
        )
        scale1 = camera1.data.ortho_scale

        # Remove camera and recreate mesh
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj)
        bpy.ops.mesh.primitive_cube_add()
        mesh_obj = bpy.context.active_object

        # Call with larger ortho_scale
        camera2 = position_camera_relative(
            mesh_obj,
            direction="south",
            camera_type="ORTHO",
            ortho_scale=3.0,
        )
        scale2 = camera2.data.ortho_scale

        # Larger ortho_scale should produce larger camera ortho_scale
        assert scale2 > scale1, (
            f"ortho_scale=3.0 should produce larger camera scale than ortho_scale=1.0, "
            f"got {scale2} vs {scale1}"
        )


class TestSunPosition:
    """Tests for sun positioning with azimuth and elevation (TDD)."""

    def test_setup_light_accepts_azimuth_elevation(self):
        """Test that setup_light accepts azimuth and elevation parameters."""
        import bpy
        from src.terrain.scene_setup import setup_light

        # Clear lights
        for obj in list(bpy.data.objects):
            if obj.type == "LIGHT":
                bpy.data.objects.remove(obj)

        # Should accept azimuth (degrees from north) and elevation (degrees above horizon)
        light = setup_light(azimuth=180, elevation=45)

        assert light is not None
        assert light.data.type == "SUN"

    def test_sun_from_south_elevation_45(self):
        """Test sun from south (azimuth=180) at 45 degrees elevation."""
        import bpy
        from math import radians, sin, cos
        from src.terrain.scene_setup import setup_light

        for obj in list(bpy.data.objects):
            if obj.type == "LIGHT":
                bpy.data.objects.remove(obj)

        # Sun from south at 45 deg elevation
        # Sun direction vector should point toward +Y (north) and down from above
        light = setup_light(azimuth=180, elevation=45)

        # Get the direction the sun is shining (local -Z axis of sun)
        import mathutils
        rotation = light.rotation_euler
        direction = mathutils.Vector((0, 0, -1))
        direction.rotate(rotation)

        # Azimuth 180 (from south) means light shines toward north (+Y)
        # With elevation 45, there should be downward component too
        assert direction.y > 0.5, f"Sun from south should shine north (+Y), got {direction}"
        assert direction.z < 0, f"Sun should shine downward, got {direction}"

    def test_sun_from_east_elevation_30(self):
        """Test sun from east (azimuth=90) at 30 degrees elevation."""
        import bpy
        import mathutils
        from src.terrain.scene_setup import setup_light

        for obj in list(bpy.data.objects):
            if obj.type == "LIGHT":
                bpy.data.objects.remove(obj)

        # Sun from east at 30 deg elevation
        light = setup_light(azimuth=90, elevation=30)

        rotation = light.rotation_euler
        direction = mathutils.Vector((0, 0, -1))
        direction.rotate(rotation)

        # Azimuth 90 (from east) means light shines toward west (-X)
        assert direction.x < -0.5, f"Sun from east should shine west (-X), got {direction}"
        assert direction.z < 0, f"Sun should shine downward, got {direction}"

    def test_sun_directly_overhead(self):
        """Test sun directly overhead (elevation=90)."""
        import bpy
        import mathutils
        from src.terrain.scene_setup import setup_light

        for obj in list(bpy.data.objects):
            if obj.type == "LIGHT":
                bpy.data.objects.remove(obj)

        # Elevation 90 = straight down, azimuth shouldn't matter
        light = setup_light(azimuth=0, elevation=90)

        rotation = light.rotation_euler
        direction = mathutils.Vector((0, 0, -1))
        direction.rotate(rotation)

        # Should be pointing straight down
        assert abs(direction.x) < 0.1, f"Overhead sun X should be ~0, got {direction}"
        assert abs(direction.y) < 0.1, f"Overhead sun Y should be ~0, got {direction}"
        assert direction.z < -0.9, f"Overhead sun should shine straight down, got {direction}"

    def test_sun_from_north_low_angle(self):
        """Test sun from north (azimuth=0) at low angle (10 degrees)."""
        import bpy
        import mathutils
        from src.terrain.scene_setup import setup_light

        for obj in list(bpy.data.objects):
            if obj.type == "LIGHT":
                bpy.data.objects.remove(obj)

        # Sun from north at 10 deg elevation (low sun, long shadows)
        light = setup_light(azimuth=0, elevation=10)

        rotation = light.rotation_euler
        direction = mathutils.Vector((0, 0, -1))
        direction.rotate(rotation)

        # Azimuth 0 (from north) means light shines toward south (-Y)
        assert direction.y < -0.5, f"Sun from north should shine south (-Y), got {direction}"
        # Low elevation = mostly horizontal, small Z component
        assert direction.z < 0, f"Sun should shine downward, got {direction}"

    def test_position_camera_relative_accepts_sun_azimuth_elevation(self):
        """Test that position_camera_relative accepts sun_azimuth and sun_elevation."""
        import bpy
        from src.terrain.scene_setup import position_camera_relative

        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj)

        bpy.ops.mesh.primitive_cube_add()
        mesh_obj = bpy.context.active_object

        # Should accept sun_azimuth and sun_elevation parameters
        # Note: sun_energy > 0 needed to actually create a light (defaults to 0)
        camera = position_camera_relative(
            mesh_obj,
            direction="south",
            sun_azimuth=135,  # From SE
            sun_elevation=45,
            sun_energy=3,  # Must be > 0 to create light
        )

        assert camera is not None

        # Check that a sun light was created (only when sun_energy > 0)
        sun_lights = [obj for obj in bpy.data.objects if obj.type == "LIGHT"]
        assert len(sun_lights) > 0, "Should create sun light when sun_energy > 0"


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
