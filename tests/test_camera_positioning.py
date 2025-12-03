"""Tests for cardinal direction camera positioning.

TDD RED phase: Define expected behavior for position_camera_relative() function.
Tests verify camera position, direction, and return values.
"""

import pytest
import math

try:
    import bpy
    from mathutils import Vector

    BLENDER_AVAILABLE = True
except ImportError:
    bpy = None
    Vector = None
    BLENDER_AVAILABLE = False


@pytest.fixture
def mock_mesh():
    """Create a simple test mesh with known bounds for camera positioning tests."""
    if bpy is None:
        pytest.skip("Blender not available")

    # Create a 10x10 base mesh with 2 units height, centered at origin
    bpy.ops.mesh.primitive_cube_add(size=10, location=(0, 0, 1))
    mesh = bpy.context.active_object
    mesh.scale = (1, 1, 0.1)  # Make it flat like terrain
    bpy.context.view_layer.update()

    yield mesh

    # Cleanup
    bpy.data.objects.remove(mesh, do_unlink=True)


def get_mesh_center(mesh_obj):
    """Calculate mesh center from bounding box."""
    bbox = mesh_obj.bound_box
    xs = [v[0] for v in bbox]
    ys = [v[1] for v in bbox]
    zs = [v[2] for v in bbox]
    return ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2)


class TestPositionCameraRelative:
    """Tests for position_camera_relative() function."""

    # === Function Existence ===
    def test_function_exists(self):
        """Function should be importable from core module."""
        from src.terrain.core import position_camera_relative

        assert callable(position_camera_relative)

    # === Position Tests ===
    def test_direction_above_positions_on_z_axis(self, mock_mesh):
        """Camera 'above' should be directly over mesh center on Z axis."""
        from src.terrain.core import position_camera_relative

        camera = position_camera_relative(mock_mesh, direction="above")
        center = get_mesh_center(mock_mesh)
        assert camera.location.x == pytest.approx(center[0], abs=0.1)
        assert camera.location.y == pytest.approx(center[1], abs=0.1)
        assert camera.location.z > center[2]  # Above the mesh

    def test_direction_south_positions_negative_y(self, mock_mesh):
        """Camera 'south' should be at -Y from mesh center."""
        from src.terrain.core import position_camera_relative

        camera = position_camera_relative(mock_mesh, direction="south")
        center = get_mesh_center(mock_mesh)
        assert camera.location.y < center[1]  # South = negative Y

    def test_direction_north_positions_positive_y(self, mock_mesh):
        """Camera 'north' should be at +Y from mesh center."""
        from src.terrain.core import position_camera_relative

        camera = position_camera_relative(mock_mesh, direction="north")
        center = get_mesh_center(mock_mesh)
        assert camera.location.y > center[1]  # North = positive Y

    def test_distance_parameter_affects_position(self, mock_mesh):
        """Larger distance should place camera further from mesh."""
        from src.terrain.core import position_camera_relative

        cam1 = position_camera_relative(mock_mesh, direction="south", distance=1.0)
        cam2 = position_camera_relative(mock_mesh, direction="south", distance=2.0)
        dist1 = abs(cam1.location.y - get_mesh_center(mock_mesh)[1])
        dist2 = abs(cam2.location.y - get_mesh_center(mock_mesh)[1])
        assert dist2 > dist1

    def test_elevation_parameter_affects_z_position(self, mock_mesh):
        """Elevation parameter should control camera Z position."""
        from src.terrain.core import position_camera_relative

        cam1 = position_camera_relative(mock_mesh, direction="south", elevation=0.3)
        cam2 = position_camera_relative(mock_mesh, direction="south", elevation=0.7)
        assert cam2.location.z > cam1.location.z

    # === Direction/Rotation Tests (using forward vector) ===
    def test_camera_points_at_center(self, mock_mesh):
        """Camera forward vector should point toward mesh center."""
        from src.terrain.core import position_camera_relative

        camera = position_camera_relative(mock_mesh, direction="south", look_at="center")

        # Get actual forward direction from rotation matrix
        # Camera's local -Z axis is its forward direction
        rot_matrix = camera.rotation_euler.to_matrix()
        actual_forward = rot_matrix @ Vector((0, 0, -1))

        # Expected: unit vector from camera to mesh center
        center = Vector(get_mesh_center(mock_mesh))
        expected_forward = (center - camera.location).normalized()

        # Dot product ~1.0 means vectors are aligned
        alignment = actual_forward.dot(expected_forward)
        assert alignment > 0.99, f"Camera not pointing at center, alignment={alignment}"

    def test_above_camera_looks_straight_down(self, mock_mesh):
        """Camera 'above' should look straight down (-Z direction in world space)."""
        from src.terrain.core import position_camera_relative

        camera = position_camera_relative(mock_mesh, direction="above", look_at="center")

        rot_matrix = camera.rotation_euler.to_matrix()
        forward = rot_matrix @ Vector((0, 0, -1))

        # Should point straight down: (0, 0, -1)
        # Allow small tolerances for X,Y but Z should be strongly negative
        assert abs(forward.x) < 0.05, f"Camera X tilt too much: {forward.x}"
        assert abs(forward.y) < 0.05, f"Camera Y tilt too much: {forward.y}"
        assert forward.z < -0.99, f"Camera not looking down, forward.z={forward.z}"

    def test_south_camera_has_positive_pitch(self, mock_mesh):
        """Camera from 'south' should have positive X rotation (pitch up to look at terrain)."""
        from src.terrain.core import position_camera_relative

        camera = position_camera_relative(mock_mesh, direction="south", look_at="center")

        # X rotation should be positive (pitching up to see terrain above)
        assert (
            camera.rotation_euler.x > 0.1
        ), f"Expected positive pitch, got {camera.rotation_euler.x}"

    # === Return Value Tests ===
    def test_returns_camera_object(self, mock_mesh):
        """Should return camera object."""
        from src.terrain.core import position_camera_relative

        camera = position_camera_relative(mock_mesh, direction="south")
        assert camera is not None
        assert camera.type == "CAMERA"

    # === Cardinal Directions ===
    def test_all_cardinal_directions_supported(self, mock_mesh):
        """Function should support all 8 cardinal/vertical directions."""
        from src.terrain.core import position_camera_relative

        directions = [
            "north",
            "south",
            "east",
            "west",
            "above",
            "northeast",
            "northwest",
            "southeast",
            "southwest",
        ]

        for direction in directions:
            camera = position_camera_relative(mock_mesh, direction=direction)
            assert camera is not None
            assert camera.type == "CAMERA"
