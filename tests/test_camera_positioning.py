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


class TestPositionCameraRelativeMultiMesh:
    """Tests for position_camera_relative() with multiple mesh objects.

    TDD RED phase: These tests define expected behavior for multi-mesh support.
    The function should accept a list of meshes and compute a combined bounding box.
    """

    @pytest.fixture
    def two_meshes_side_by_side(self):
        """Create two meshes positioned side-by-side (like dual terrain render)."""
        if bpy is None:
            pytest.skip("Blender not available")

        # Left mesh at X = -10
        bpy.ops.mesh.primitive_cube_add(size=10, location=(-10, 0, 1))
        mesh_left = bpy.context.active_object
        mesh_left.name = "TerrainLeft"
        mesh_left.scale = (1, 1, 0.1)

        # Right mesh at X = +10
        bpy.ops.mesh.primitive_cube_add(size=10, location=(10, 0, 1))
        mesh_right = bpy.context.active_object
        mesh_right.name = "TerrainRight"
        mesh_right.scale = (1, 1, 0.1)

        bpy.context.view_layer.update()

        yield [mesh_left, mesh_right]

        # Cleanup
        bpy.data.objects.remove(mesh_left, do_unlink=True)
        bpy.data.objects.remove(mesh_right, do_unlink=True)

    @pytest.fixture
    def three_meshes_triangle(self):
        """Create three meshes in a triangular arrangement."""
        if bpy is None:
            pytest.skip("Blender not available")

        meshes = []

        # Mesh at origin
        bpy.ops.mesh.primitive_cube_add(size=5, location=(0, 0, 1))
        mesh1 = bpy.context.active_object
        mesh1.name = "Terrain1"
        meshes.append(mesh1)

        # Mesh at upper-left
        bpy.ops.mesh.primitive_cube_add(size=5, location=(-10, 10, 1))
        mesh2 = bpy.context.active_object
        mesh2.name = "Terrain2"
        meshes.append(mesh2)

        # Mesh at upper-right
        bpy.ops.mesh.primitive_cube_add(size=5, location=(10, 10, 1))
        mesh3 = bpy.context.active_object
        mesh3.name = "Terrain3"
        meshes.append(mesh3)

        bpy.context.view_layer.update()

        yield meshes

        # Cleanup
        for mesh in meshes:
            bpy.data.objects.remove(mesh, do_unlink=True)

    # === Accepts List of Meshes ===
    def test_accepts_list_of_meshes(self, two_meshes_side_by_side):
        """Function should accept a list of mesh objects."""
        from src.terrain.core import position_camera_relative

        meshes = two_meshes_side_by_side
        camera = position_camera_relative(meshes, direction="south")

        assert camera is not None
        assert camera.type == "CAMERA"

    def test_accepts_single_mesh_in_list(self, mock_mesh):
        """Function should work with a single-element list for consistency."""
        from src.terrain.core import position_camera_relative

        camera = position_camera_relative([mock_mesh], direction="south")

        assert camera is not None
        assert camera.type == "CAMERA"

    # === Combined Bounding Box Tests ===
    def test_combined_bbox_wider_than_single_mesh(self, two_meshes_side_by_side):
        """Combined bounding box should span both meshes."""
        from src.terrain.core import position_camera_relative

        meshes = two_meshes_side_by_side
        mesh_left, mesh_right = meshes

        # Camera for single mesh
        cam_single = position_camera_relative(mesh_left, direction="south", distance=1.0)
        single_pos_y = cam_single.location[1]

        # Camera for both meshes - should be further back due to larger combined bbox
        cam_both = position_camera_relative(meshes, direction="south", distance=1.0)
        both_pos_y = cam_both.location[1]

        # Camera should be further south (more negative Y) for combined bbox
        assert both_pos_y < single_pos_y, (
            f"Combined camera Y={both_pos_y:.2f} should be < single camera Y={single_pos_y:.2f}"
        )

    def test_combined_bbox_center_between_meshes(self, two_meshes_side_by_side):
        """Camera should target the center of the combined bounding box."""
        from src.terrain.core import position_camera_relative

        meshes = two_meshes_side_by_side

        # For above view, camera X should be at center of combined bbox (X=0)
        camera = position_camera_relative(meshes, direction="above")

        # Combined center should be at X=0 (between -10 and +10)
        assert abs(camera.location[0]) < 1.0, (
            f"Camera X={camera.location[0]:.2f} should be near 0 (center of combined bbox)"
        )

    def test_ortho_scale_covers_all_meshes(self, two_meshes_side_by_side):
        """Orthographic scale should be large enough to see all meshes."""
        from src.terrain.core import position_camera_relative

        meshes = two_meshes_side_by_side
        mesh_left = meshes[0]

        # Single mesh ortho scale
        cam_single = position_camera_relative(mesh_left, direction="above", camera_type="ORTHO")
        single_scale = cam_single.data.ortho_scale

        # Combined ortho scale should be larger
        cam_both = position_camera_relative(meshes, direction="above", camera_type="ORTHO")
        both_scale = cam_both.data.ortho_scale

        assert both_scale > single_scale, (
            f"Combined ortho_scale={both_scale:.2f} should be > single scale={single_scale:.2f}"
        )

    # === Direction Tests with Multiple Meshes ===
    def test_all_directions_work_with_multiple_meshes(self, two_meshes_side_by_side):
        """All cardinal directions should work with a list of meshes."""
        from src.terrain.core import position_camera_relative

        meshes = two_meshes_side_by_side
        directions = [
            "north", "south", "east", "west", "above",
            "northeast", "northwest", "southeast", "southwest",
        ]

        for direction in directions:
            camera = position_camera_relative(meshes, direction=direction)
            assert camera is not None, f"Failed for direction: {direction}"
            assert camera.type == "CAMERA"

    # === Three Meshes Tests ===
    def test_three_meshes_combined_bbox(self, three_meshes_triangle):
        """Should handle three or more meshes correctly."""
        from src.terrain.core import position_camera_relative

        meshes = three_meshes_triangle

        camera = position_camera_relative(meshes, direction="above")

        # Center should be between all three meshes
        # Meshes at (0,0), (-10,10), (10,10) -> center roughly at (0, 6.67)
        assert camera is not None
        assert camera.type == "CAMERA"

    # === Backward Compatibility ===
    def test_single_mesh_still_works(self, mock_mesh):
        """Single mesh (not in list) should still work for backward compatibility."""
        from src.terrain.core import position_camera_relative

        # This is the existing API - should continue to work
        camera = position_camera_relative(mock_mesh, direction="south")

        assert camera is not None
        assert camera.type == "CAMERA"

    def test_single_mesh_same_result_as_single_element_list(self, mock_mesh):
        """Single mesh and [single_mesh] should produce equivalent camera positions."""
        from src.terrain.core import position_camera_relative

        cam_single = position_camera_relative(mock_mesh, direction="south", distance=1.5)
        cam_list = position_camera_relative([mock_mesh], direction="south", distance=1.5)

        # Positions should be essentially the same
        assert abs(cam_single.location[0] - cam_list.location[0]) < 0.01
        assert abs(cam_single.location[1] - cam_list.location[1]) < 0.01
        assert abs(cam_single.location[2] - cam_list.location[2]) < 0.01
