"""
Tests for camera projection and frame centering.

TDD RED phase: Define expected behavior for projection-based centering.
"""

import pytest
import math
import bpy
from mathutils import Vector, Euler


@pytest.fixture
def clear_scene():
    """Clear Blender scene before each test."""
    # Clear all objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Clear all meshes, cameras, lights
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    for camera in bpy.data.cameras:
        bpy.data.cameras.remove(camera)
    for light in bpy.data.lights:
        bpy.data.lights.remove(light)

    yield

    # Cleanup after test
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def create_test_cube(size=1.0, location=(0, 0, 0)):
    """Create a simple cube mesh for testing.

    Args:
        size: Cube size (default: 1.0)
        location: Cube center location (default: origin)

    Returns:
        Blender mesh object
    """
    # Create cube mesh data
    mesh_data = bpy.data.meshes.new("TestCube")
    mesh_obj = bpy.data.objects.new("TestCube", mesh_data)
    bpy.context.scene.collection.objects.link(mesh_obj)

    # Define cube vertices (8 corners)
    half = size / 2
    verts = [
        (-half, -half, -half),  # 0: back-bottom-left
        (half, -half, -half),   # 1: back-bottom-right
        (half, half, -half),    # 2: back-top-right
        (-half, half, -half),   # 3: back-top-left
        (-half, -half, half),   # 4: front-bottom-left
        (half, -half, half),    # 5: front-bottom-right
        (half, half, half),     # 6: front-top-right
        (-half, half, half),    # 7: front-top-left
    ]

    # Define cube faces (6 faces, each a quad)
    faces = [
        (0, 1, 2, 3),  # Back
        (4, 5, 6, 7),  # Front
        (0, 1, 5, 4),  # Bottom
        (2, 3, 7, 6),  # Top
        (0, 3, 7, 4),  # Left
        (1, 2, 6, 5),  # Right
    ]

    # Build mesh
    mesh_data.from_pydata(verts, [], faces)
    mesh_data.update()

    # Set location
    mesh_obj.location = location

    # Update scene
    bpy.context.view_layer.update()

    return mesh_obj


def create_test_camera(location=(0, 0, -10), look_at=(0, 0, 0), focal_length=50):
    """Create a perspective camera for testing.

    Args:
        location: Camera location
        look_at: Point the camera looks at
        focal_length: Camera focal length in mm (default: 50mm)

    Returns:
        Blender camera object
    """
    # Create camera
    cam_data = bpy.data.cameras.new("TestCamera")
    cam_obj = bpy.data.objects.new("TestCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)

    # Set camera type and focal length
    cam_data.type = 'PERSP'
    cam_data.lens = focal_length

    # Position camera
    cam_obj.location = location

    # Point camera at target
    direction = Vector(look_at) - Vector(location)
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()

    # Set as active camera
    bpy.context.scene.camera = cam_obj

    # Update scene
    bpy.context.view_layer.update()

    return cam_obj


class TestProjectionBasics:
    """Test basic screen-space projection math."""

    def test_cube_at_origin_projects_to_center(self, clear_scene):
        """Cube at origin with camera on -Z axis should project to screen center."""
        from src.terrain.scene_setup import compute_mesh_screen_bbox

        # Create 1x1 cube at origin
        cube = create_test_cube(size=1.0, location=(0, 0, 0))

        # Camera 10 units away on -Z axis, looking at origin
        camera = create_test_camera(location=(0, 0, -10), look_at=(0, 0, 0))

        # Compute screen bbox
        bbox = compute_mesh_screen_bbox(cube, camera, bpy.context.scene)

        # Cube should project to screen center
        assert bbox['center_x'] == pytest.approx(0.5, abs=0.05)
        assert bbox['center_y'] == pytest.approx(0.5, abs=0.05)

    def test_offset_cube_projects_off_center(self, clear_scene):
        """Cube offset in +X world appears LEFT on screen (Blender coordinate system)."""
        from src.terrain.scene_setup import compute_mesh_screen_bbox

        # Cube offset 3 units in +X direction (world coords)
        cube = create_test_cube(size=1.0, location=(3, 0, 0))

        # Camera on -Z axis looking toward +Z (at origin)
        # In Blender: camera looks down local -Z, so world +Z is "into" the camera
        camera = create_test_camera(location=(0, 0, -10), look_at=(0, 0, 0))

        bbox = compute_mesh_screen_bbox(cube, camera, bpy.context.scene)

        # In Blender's screen space with camera at -Z looking at origin:
        # - World +X appears on LEFT of screen (x < 0.5)
        # - World -X appears on RIGHT of screen (x > 0.5)
        # This is because camera's view is mirrored
        assert bbox['center_x'] < 0.4, \
            f"Cube at +X should appear left of center, got {bbox['center_x']}"
        # Should be vertically centered
        assert bbox['center_y'] == pytest.approx(0.5, abs=0.05)

    def test_cube_above_projects_above_center(self, clear_scene):
        """Cube offset upward should project above screen center."""
        from src.terrain.scene_setup import compute_mesh_screen_bbox

        # Cube offset 3 units up (+Y in Blender)
        cube = create_test_cube(size=1.0, location=(0, 0, 3))

        # Camera on -Y axis looking at origin
        camera = create_test_camera(location=(0, -10, 0), look_at=(0, 0, 0))

        bbox = compute_mesh_screen_bbox(cube, camera, bpy.context.scene)

        # Should be horizontally centered
        assert bbox['center_x'] == pytest.approx(0.5, abs=0.05)
        # Should be above center (y > 0.5)
        assert bbox['center_y'] > 0.6

    def test_bbox_size_scales_with_distance(self, clear_scene):
        """Farther objects should appear smaller in screen space."""
        from src.terrain.scene_setup import compute_mesh_screen_bbox

        cube = create_test_cube(size=2.0, location=(0, 0, 0))

        # Close camera
        camera_close = create_test_camera(location=(0, 0, -5), look_at=(0, 0, 0))
        bbox_close = compute_mesh_screen_bbox(cube, camera_close, bpy.context.scene)

        # Re-create scene for far camera
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        cube = create_test_cube(size=2.0, location=(0, 0, 0))

        # Far camera
        camera_far = create_test_camera(location=(0, 0, -20), look_at=(0, 0, 0))
        bbox_far = compute_mesh_screen_bbox(cube, camera_far, bpy.context.scene)

        # Farther cube should have smaller screen bbox
        assert bbox_far['width'] < bbox_close['width']
        assert bbox_far['height'] < bbox_close['height']


class TestCameraOrientation:
    """Test camera orientation vector calculations."""

    def test_camera_right_vector_points_right(self, clear_scene):
        """Camera's local +X (right) vector in world space."""
        # Camera at -Z looking toward +Z (at origin)
        camera = create_test_camera(location=(0, 0, -10), look_at=(0, 0, 0))

        # Get camera's right vector (local +X axis transformed to world space)
        cam_matrix = camera.matrix_world
        cam_right = cam_matrix.to_3x3() @ Vector((1, 0, 0))

        # When camera looks from -Z toward +Z:
        # Camera's local +X points toward world -X (screen-right is world-left)
        # This is because the camera's view is mirrored
        assert cam_right.x == pytest.approx(-1.0, abs=0.01)
        assert cam_right.y == pytest.approx(0.0, abs=0.01)
        assert cam_right.z == pytest.approx(0.0, abs=0.01)

    def test_camera_up_vector_points_up(self, clear_scene):
        """Camera's local +Y (up) vector in world space."""
        # Camera at -Z looking toward +Z
        camera = create_test_camera(location=(0, 0, -10), look_at=(0, 0, 0))

        # Get camera's up vector (local +Y axis transformed to world space)
        cam_matrix = camera.matrix_world
        cam_up = cam_matrix.to_3x3() @ Vector((0, 1, 0))

        # When camera looks from -Z, camera's local +Y points toward world +Y
        # NOT world +Z (Blender's up axis)
        # The camera is oriented so its local Y axis aligns with world Y
        assert cam_up.x == pytest.approx(0.0, abs=0.01)
        assert cam_up.y == pytest.approx(1.0, abs=0.01)
        assert cam_up.z == pytest.approx(0.0, abs=0.01)


class TestFrameCentering:
    """Test iterative frame centering convergence."""

    def test_centering_moves_camera_target(self, clear_scene):
        """Frame centering should adjust camera look-at target."""
        # This test will verify the centering algorithm works
        # We'll implement this after compute_mesh_screen_bbox is fixed
        pytest.skip("Requires working compute_mesh_screen_bbox implementation")

    def test_centering_converges_within_iterations(self, clear_scene):
        """Centering should converge to center within 3 iterations."""
        pytest.skip("Requires working compute_mesh_screen_bbox implementation")

    def test_already_centered_needs_no_adjustment(self, clear_scene):
        """Mesh already centered should require zero or minimal adjustment."""
        pytest.skip("Requires working compute_mesh_screen_bbox implementation")
