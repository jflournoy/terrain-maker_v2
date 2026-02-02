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


class TestWrapperParameterPassthrough:
    """Regression tests for core.py wrapper passing parameters to scene_setup.py.

    These tests catch bugs where wrapper functions pass arguments positionally
    and become misaligned when the underlying function signature changes.
    """

    def test_ortho_scale_passed_through_wrapper(self, mock_mesh):
        """Verify ortho_scale is correctly passed through core.py wrapper.

        Regression test: Catches bugs where ortho_scale is passed to wrong param
        due to positional argument mismatch between wrapper and underlying function.
        """
        from src.terrain.core import position_camera_relative

        # Call with small ortho_scale
        cam1 = position_camera_relative(
            mock_mesh,
            direction="south",
            camera_type="ORTHO",
            ortho_scale=1.0,
        )
        scale1 = cam1.data.ortho_scale

        # Call with larger ortho_scale (need fresh mesh to avoid camera reuse)
        for obj in list(bpy.data.objects):
            if obj.type == "CAMERA":
                bpy.data.objects.remove(obj)

        cam2 = position_camera_relative(
            mock_mesh,
            direction="south",
            camera_type="ORTHO",
            ortho_scale=3.0,
        )
        scale2 = cam2.data.ortho_scale

        # Different ortho_scale values must produce different camera scales
        assert scale2 > scale1, (
            f"Wrapper bug: ortho_scale=3.0 should produce larger scale than 1.0, "
            f"got {scale2:.2f} vs {scale1:.2f}. "
            f"ortho_scale may be passed to wrong parameter."
        )

    def test_focal_length_passed_through_wrapper(self, mock_mesh):
        """Verify focal_length is correctly passed through core.py wrapper."""
        from src.terrain.core import position_camera_relative

        cam = position_camera_relative(
            mock_mesh,
            direction="south",
            camera_type="PERSP",
            focal_length=85,  # Portrait lens
        )

        # Camera should have the specified focal length
        assert cam.data.lens == pytest.approx(85, abs=0.1), (
            f"focal_length not passed correctly: expected 85mm, got {cam.data.lens}mm"
        )

    def test_sun_azimuth_elevation_passed_through_wrapper(self, mock_mesh):
        """Verify sun_azimuth and sun_elevation are correctly passed through."""
        from src.terrain.core import position_camera_relative

        # Need sun_energy > 0 to create light
        cam = position_camera_relative(
            mock_mesh,
            direction="south",
            sun_azimuth=180,  # From south
            sun_elevation=45,
            sun_energy=3,
        )

        # Should have created a sun light
        sun_lights = [obj for obj in bpy.data.objects if obj.type == "LIGHT"]
        assert len(sun_lights) > 0, (
            "sun_azimuth/sun_elevation not passed correctly: no sun light created"
        )


class TestFittingDistanceMode:
    """Tests for fitting-distance camera positioning mode.

    TDD RED phase: Define photographer-friendly distance behavior.
    When distance_mode="fit", distance parameter controls framing tightness,
    not absolute position. Camera distance adjusts with elevation to maintain
    consistent framing.

    distance=1.0: Mesh exactly fits in frame
    distance=1.5: 1.5x farther for wider context
    distance=0.7: Closer for tighter crop
    """

    def test_fitting_distance_mode_accepted(self, mock_mesh):
        """Function should accept distance_mode='fit' parameter."""
        from src.terrain.core import position_camera_relative

        camera = position_camera_relative(
            mock_mesh,
            direction="south",
            distance_mode="fit",
            distance=1.0
        )

        assert camera is not None
        assert camera.type == "CAMERA"

    def test_fit_distance_1_fills_frame(self, mock_mesh):
        """distance=1.0 with mode='fit' should position camera to fill frame with mesh."""
        from src.terrain.core import position_camera_relative
        import math

        camera = position_camera_relative(
            mock_mesh,
            direction="south",
            distance_mode="fit",
            distance=1.0,
            elevation=0.5,
            focal_length=50  # Standard lens
        )

        # Calculate expected fitting distance based on camera FOV and mesh size
        # For a 50mm lens: FOV ≈ 39.6 degrees
        sensor_width = 36.0  # mm
        fov_radians = 2 * math.atan(sensor_width / (2 * 50))

        # Get mesh width (X dimension)
        bbox = mock_mesh.bound_box
        xs = [v[0] for v in bbox]
        mesh_width = max(xs) - min(xs)

        # Distance to fit mesh in frame: width / (2 * tan(FOV/2))
        expected_distance = mesh_width / (2 * math.tan(fov_radians / 2))

        # Actual distance from camera to mesh center
        center = get_mesh_center(mock_mesh)
        cam_to_center = math.sqrt(
            (camera.location[0] - center[0])**2 +
            (camera.location[1] - center[1])**2 +
            (camera.location[2] - center[2])**2
        )

        # Should match expected fitting distance
        assert cam_to_center == pytest.approx(expected_distance, rel=0.1)

    def test_fit_distance_scales_with_multiplier(self, mock_mesh):
        """distance=1.5 should be 1.5x farther than distance=1.0."""
        from src.terrain.core import position_camera_relative

        cam1 = position_camera_relative(
            mock_mesh,
            direction="south",
            distance_mode="fit",
            distance=1.0,
            elevation=0.5
        )

        cam2 = position_camera_relative(
            mock_mesh,
            direction="south",
            distance_mode="fit",
            distance=1.5,
            elevation=0.5
        )

        center = get_mesh_center(mock_mesh)
        dist1 = math.sqrt(sum((cam1.location[i] - center[i])**2 for i in range(3)))
        dist2 = math.sqrt(sum((cam2.location[i] - center[i])**2 for i in range(3)))

        # dist2 should be 1.5x dist1
        assert dist2 == pytest.approx(1.5 * dist1, rel=0.05)

    def test_fit_maintains_framing_across_elevations(self, mock_mesh):
        """Changing elevation maintains framing (distance from mesh adjusts)."""
        from src.terrain.core import position_camera_relative

        # Low elevation view
        cam_low = position_camera_relative(
            mock_mesh,
            direction="south",
            distance_mode="fit",
            distance=1.0,
            elevation=0.3
        )

        # High elevation view
        cam_high = position_camera_relative(
            mock_mesh,
            direction="south",
            distance_mode="fit",
            distance=1.0,
            elevation=0.7
        )

        # Both cameras should be same distance from mesh center
        # (even though absolute positions differ)
        center = get_mesh_center(mock_mesh)
        dist_low = math.sqrt(sum((cam_low.location[i] - center[i])**2 for i in range(3)))
        dist_high = math.sqrt(sum((cam_high.location[i] - center[i])**2 for i in range(3)))

        assert dist_low == pytest.approx(dist_high, rel=0.05), (
            f"Fitting distance should stay constant across elevations: "
            f"low={dist_low:.2f}, high={dist_high:.2f}"
        )

    def test_diagonal_mode_is_default(self, mock_mesh):
        """Without distance_mode parameter, should use diagonal mode (backward compat)."""
        from src.terrain.core import position_camera_relative

        # Default behavior (no distance_mode)
        cam_default = position_camera_relative(
            mock_mesh,
            direction="south",
            distance=1.0,
            elevation=0.5
        )

        # Explicit diagonal mode
        cam_diagonal = position_camera_relative(
            mock_mesh,
            direction="south",
            distance_mode="diagonal",
            distance=1.0,
            elevation=0.5
        )

        # Should produce same result
        assert abs(cam_default.location[0] - cam_diagonal.location[0]) < 0.01
        assert abs(cam_default.location[1] - cam_diagonal.location[1]) < 0.01
        assert abs(cam_default.location[2] - cam_diagonal.location[2]) < 0.01


class TestAboveTiltedDirection:
    """Tests for 'above-tilted' camera direction.

    This direction is like 'above' but with a slight offset toward southwest,
    providing just enough tilt to show the 3D skirt/edge extrusion of terrain meshes.

    Camera angle: ~5 degrees from vertical, toward southwest
    Use case: Showcasing 3D printed terrain with visible edge thickness
    """

    def test_above_tilted_direction_exists(self, mock_mesh):
        """The 'above-tilted' direction should be recognized without error."""
        from src.terrain.core import position_camera_relative

        # Should not raise ValueError for unrecognized direction
        camera = position_camera_relative(mock_mesh, direction="above-tilted")

        assert camera is not None
        assert camera.type == "CAMERA"

    def test_above_tilted_looks_mostly_down(self, mock_mesh):
        """Camera should look mostly downward (Z component strongly negative)."""
        from src.terrain.core import position_camera_relative

        camera = position_camera_relative(mock_mesh, direction="above-tilted")

        rot_matrix = camera.rotation_euler.to_matrix()
        forward = rot_matrix @ Vector((0, 0, -1))

        # Should point mostly down - Z should be strongly negative
        # Allow more tolerance than pure "above" which requires z < -0.99
        assert forward.z < -0.95, (
            f"Camera should look mostly down, but forward.z={forward.z:.3f}"
        )

    def test_above_tilted_has_slight_offset(self, mock_mesh):
        """Camera should NOT be directly above - should have SW offset."""
        from src.terrain.core import position_camera_relative

        camera = position_camera_relative(mock_mesh, direction="above-tilted")

        rot_matrix = camera.rotation_euler.to_matrix()
        forward = rot_matrix @ Vector((0, 0, -1))

        # Unlike pure "above" which has near-zero X/Y, this should have measurable tilt
        # Looking from southwest means forward vector has positive X and Y components
        # (camera is SW of center, looking toward center = +X, +Y)
        assert forward.x > 0.05, (
            f"Camera should have X offset to show skirt, but forward.x={forward.x:.3f}"
        )
        assert forward.y > 0.05, (
            f"Camera should have Y offset to show skirt, but forward.y={forward.y:.3f}"
        )

    def test_above_tilted_position_directly_above(self, mock_mesh):
        """Camera position should be directly above mesh center (like 'above')."""
        from src.terrain.core import position_camera_relative

        camera = position_camera_relative(mock_mesh, direction="above-tilted")
        center = get_mesh_center(mock_mesh)

        # Camera should be directly above center (X,Y match center)
        assert abs(camera.location.x - center[0]) < 0.1, (
            f"Camera X={camera.location.x:.2f} should be at mesh center X={center[0]:.2f}"
        )
        assert abs(camera.location.y - center[1]) < 0.1, (
            f"Camera Y={camera.location.y:.2f} should be at mesh center Y={center[1]:.2f}"
        )

    def test_above_tilted_still_high_elevation(self, mock_mesh):
        """Camera should still be high above the mesh (like 'above')."""
        from src.terrain.core import position_camera_relative

        camera = position_camera_relative(mock_mesh, direction="above-tilted")
        center = get_mesh_center(mock_mesh)

        # Should be well above the mesh center
        assert camera.location.z > center[2] + 5, (
            f"Camera Z={camera.location.z:.2f} should be well above mesh center "
            f"Z={center[2]:.2f}"
        )

    def test_above_tilted_different_from_pure_above(self, mock_mesh):
        """'above-tilted' should have different rotation than 'above'."""
        from src.terrain.core import position_camera_relative

        cam_above = position_camera_relative(mock_mesh, direction="above")
        cam_tilted = position_camera_relative(mock_mesh, direction="above-tilted")

        # Positions should be the same (both directly above)
        pos_diff = (
            abs(cam_above.location.x - cam_tilted.location.x) +
            abs(cam_above.location.y - cam_tilted.location.y)
        )
        assert pos_diff < 0.2, "Both cameras should be positioned directly above"

        # But rotations should differ (tilted has non-zero rotation)
        rot_above = cam_above.rotation_euler
        rot_tilted = cam_tilted.rotation_euler
        rot_diff = abs(rot_above.x - rot_tilted.x) + abs(rot_above.y - rot_tilted.y)
        assert rot_diff > 0.01, (
            f"'above-tilted' rotation should differ from 'above'. "
            f"above={rot_above[:]} vs tilted={rot_tilted[:]}"
        )

    def test_above_tilted_in_all_directions_list(self, mock_mesh):
        """'above-tilted' should be included in the full list of supported directions."""
        from src.terrain.core import position_camera_relative

        # Test alongside other directions to ensure it's a first-class citizen
        directions = [
            "north", "south", "east", "west", "above", "above-tilted",
            "northeast", "northwest", "southeast", "southwest",
        ]

        for direction in directions:
            camera = position_camera_relative(mock_mesh, direction=direction)
            assert camera is not None, f"Failed for direction: {direction}"
