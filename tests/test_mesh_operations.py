"""
Tests for mesh_operations module.

Tests the extracted mesh generation functions that were refactored from core.py.
"""

import pytest
import numpy as np


class TestVertexPositionGeneration:
    """Tests for generate_vertex_positions function."""

    def test_generate_vertex_positions_imports(self):
        """Test that generate_vertex_positions can be imported."""
        from src.terrain.mesh_operations import generate_vertex_positions

        assert callable(generate_vertex_positions)

    def test_generate_vertex_positions_basic(self):
        """Test basic vertex position generation from DEM."""
        from src.terrain.mesh_operations import generate_vertex_positions

        # Simple 3x3 DEM
        dem_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        valid_mask = ~np.isnan(dem_data)

        positions, y_valid, x_valid = generate_vertex_positions(
            dem_data, valid_mask, scale_factor=1.0, height_scale=1.0
        )

        # Should have 9 vertices (3x3 grid)
        assert positions.shape == (9, 3)

        # Check first vertex (0,0) - should be (0, 0, 1)
        assert np.allclose(positions[0], [0, 0, 1])

        # Check last vertex (2,2) - should be (2, 2, 9)
        assert np.allclose(positions[-1], [2, 2, 9])

    def test_generate_vertex_positions_with_scaling(self):
        """Test vertex generation with scale factor."""
        from src.terrain.mesh_operations import generate_vertex_positions

        dem_data = np.array([[10.0, 20.0], [30.0, 40.0]])
        valid_mask = ~np.isnan(dem_data)

        positions, y_valid, x_valid = generate_vertex_positions(
            dem_data, valid_mask, scale_factor=10.0, height_scale=2.0
        )

        # With scale_factor=10, x,y should be divided by 10
        # With height_scale=2, z should be multiplied by 2
        # First vertex (0,0): x=0/10=0, y=0/10=0, z=10*2=20
        assert np.allclose(positions[0], [0, 0, 20])

        # Last vertex (1,1): x=1/10=0.1, y=1/10=0.1, z=40*2=80
        assert np.allclose(positions[-1], [0.1, 0.1, 80])

    def test_generate_vertex_positions_handles_nan(self):
        """Test that NaN values are excluded from positions."""
        from src.terrain.mesh_operations import generate_vertex_positions

        # DEM with NaN in center
        dem_data = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])

        valid_mask = ~np.isnan(dem_data)

        positions, y_valid, x_valid = generate_vertex_positions(
            dem_data, valid_mask, scale_factor=1.0, height_scale=1.0
        )

        # Should have 8 vertices (9 - 1 NaN)
        assert positions.shape == (8, 3)

        # No NaN values in output
        assert not np.any(np.isnan(positions))

    def test_generate_vertex_positions_returns_xy_indices(self):
        """Test that function also returns x_valid, y_valid indices."""
        from src.terrain.mesh_operations import generate_vertex_positions

        dem_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        valid_mask = ~np.isnan(dem_data)

        positions, y_valid, x_valid = generate_vertex_positions(
            dem_data, valid_mask, scale_factor=1.0, height_scale=1.0
        )

        # Should return indices
        assert len(y_valid) == 4
        assert len(x_valid) == 4

        # Indices should match grid positions
        expected_y = [0, 0, 1, 1]
        expected_x = [0, 1, 0, 1]
        assert np.array_equal(y_valid, expected_y)
        assert np.array_equal(x_valid, expected_x)


class TestFindBoundaryPoints:
    """Tests for find_boundary_points function."""

    def test_find_boundary_points_imports(self):
        """Test that find_boundary_points can be imported."""
        from src.terrain.mesh_operations import find_boundary_points

        assert callable(find_boundary_points)

    def test_find_boundary_points_simple_rectangle(self):
        """Test finding boundary of simple filled rectangle."""
        from src.terrain.mesh_operations import find_boundary_points

        # 4x4 filled grid - boundary should be perimeter only
        valid_mask = np.ones((4, 4), dtype=bool)

        boundary_coords = find_boundary_points(valid_mask)

        # 4x4 grid has 12 boundary points (perimeter - 4 corners counted once each)
        # Top: 4, Right: 4, Bottom: 4, Left: 4 = 16 but corners overlap
        # Actually: 4*4 - 2*2 = 16 - 4 = 12 perimeter points
        assert len(boundary_coords) == 12

    def test_find_boundary_points_with_hole(self):
        """Test finding boundary with internal hole (should only find outer boundary)."""
        from src.terrain.mesh_operations import find_boundary_points

        # 5x5 grid with 3x3 hole in center
        valid_mask = np.ones((5, 5), dtype=bool)
        valid_mask[1:4, 1:4] = False  # Create hole

        boundary_coords = find_boundary_points(valid_mask)

        # Should find boundary points (the frame around the hole)
        # All True points that have at least one False neighbor
        assert len(boundary_coords) > 0

        # All boundary points should be True in the mask
        for y, x in boundary_coords:
            assert valid_mask[y, x]

    def test_find_boundary_points_all_false(self):
        """Test that all-False mask returns empty boundary."""
        from src.terrain.mesh_operations import find_boundary_points

        valid_mask = np.zeros((5, 5), dtype=bool)

        boundary_coords = find_boundary_points(valid_mask)

        assert len(boundary_coords) == 0

    def test_find_boundary_points_returns_tuples(self):
        """Test that boundary points are (y, x) tuples."""
        from src.terrain.mesh_operations import find_boundary_points

        valid_mask = np.ones((3, 3), dtype=bool)

        boundary_coords = find_boundary_points(valid_mask)

        # Should be list of tuples
        assert isinstance(boundary_coords, list)
        assert all(isinstance(coord, tuple) for coord in boundary_coords)
        assert all(len(coord) == 2 for coord in boundary_coords)


class TestFaceGeneration:
    """Tests for generate_faces function."""

    def test_generate_faces_imports(self):
        """Test that generate_faces can be imported."""
        from src.terrain.mesh_operations import generate_faces

        assert callable(generate_faces)

    def test_generate_faces_simple_grid(self):
        """Test face generation for simple 2x2 grid (1 quad)."""
        from src.terrain.mesh_operations import generate_faces

        # 2x2 grid has all valid points
        height, width = 2, 2
        coord_to_index = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}

        faces = generate_faces(height, width, coord_to_index)

        # Should generate 1 face for the single quad
        assert len(faces) == 1

        # Face should have 4 vertices (quad corners)
        assert len(faces[0]) == 4

        # Face should reference all 4 vertices
        assert set(faces[0]) == {0, 1, 2, 3}

    def test_generate_faces_3x3_grid(self):
        """Test face generation for 3x3 grid (4 quads)."""
        from src.terrain.mesh_operations import generate_faces

        # 3x3 grid, all valid
        height, width = 3, 3
        coord_to_index = {
            (0, 0): 0,
            (0, 1): 1,
            (0, 2): 2,
            (1, 0): 3,
            (1, 1): 4,
            (1, 2): 5,
            (2, 0): 6,
            (2, 1): 7,
            (2, 2): 8,
        }

        faces = generate_faces(height, width, coord_to_index)

        # 3x3 grid has 4 quads: (0,0), (0,1), (1,0), (1,1)
        assert len(faces) == 4

        # All faces should be quads (4 vertices)
        assert all(len(face) == 4 for face in faces)

    def test_generate_faces_with_missing_point(self):
        """Test that missing points create triangular faces."""
        from src.terrain.mesh_operations import generate_faces

        # 2x2 grid with one missing corner
        height, width = 2, 2
        coord_to_index = {(0, 0): 0, (0, 1): 1, (1, 0): 2}
        # Missing: (1, 1)

        faces = generate_faces(height, width, coord_to_index)

        # Should still create 1 face (triangle with 3 vertices)
        assert len(faces) == 1

        # Face should have 3 vertices (triangle)
        assert len(faces[0]) == 3

    def test_generate_faces_skips_insufficient_points(self):
        """Test that quads with <3 valid points are skipped."""
        from src.terrain.mesh_operations import generate_faces

        # 2x2 grid with only 2 points
        height, width = 2, 2
        coord_to_index = {(0, 0): 0, (1, 1): 1}
        # Missing: (0, 1) and (1, 0)

        faces = generate_faces(height, width, coord_to_index)

        # Should not create any faces (only 2 points, need at least 3)
        assert len(faces) == 0

    def test_generate_faces_returns_tuples(self):
        """Test that faces are returned as tuples of vertex indices."""
        from src.terrain.mesh_operations import generate_faces

        height, width = 2, 2
        coord_to_index = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}

        faces = generate_faces(height, width, coord_to_index)

        # Faces should be list of tuples
        assert isinstance(faces, list)
        assert all(isinstance(face, tuple) for face in faces)


class TestBoundaryExtension:
    """Tests for create_boundary_extension function."""

    def test_create_boundary_extension_imports(self):
        """Test that create_boundary_extension can be imported."""
        from src.terrain.mesh_operations import create_boundary_extension

        assert callable(create_boundary_extension)

    def test_create_boundary_extension_simple_square(self):
        """Test boundary extension for simple square perimeter."""
        from src.terrain.mesh_operations import create_boundary_extension

        # Simple 2x2 grid positions
        positions = np.array([[0, 0, 1], [1, 0, 2], [0, 1, 3], [1, 1, 4]], dtype=float)

        # Boundary points in order
        boundary_points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        coord_to_index = {(0, 0): 0, (1, 0): 1, (1, 1): 2, (0, 1): 3}

        boundary_vertices, boundary_faces = create_boundary_extension(
            positions, boundary_points, coord_to_index, base_depth=-0.5
        )

        # Should create 4 bottom vertices (one per boundary point)
        assert boundary_vertices.shape == (4, 3)

        # Bottom vertices should have base_depth as z-coordinate
        assert all(v[2] == -0.5 for v in boundary_vertices)

        # Should create 4 side faces (one per boundary segment)
        assert len(boundary_faces) == 4

        # Each face should be a quad (4 vertices)
        assert all(len(face) == 4 for face in boundary_faces)

    def test_create_boundary_extension_preserves_xy(self):
        """Test that boundary vertices preserve x,y coordinates."""
        from src.terrain.mesh_operations import create_boundary_extension

        positions = np.array([[2.5, 3.7, 10.0], [5.1, 6.2, 15.0]], dtype=float)

        boundary_points = [(0, 0), (0, 1)]
        coord_to_index = {(0, 0): 0, (0, 1): 1}

        boundary_vertices, boundary_faces = create_boundary_extension(
            positions, boundary_points, coord_to_index, base_depth=-1.0
        )

        # X,Y should match original positions, only Z changes
        assert np.allclose(boundary_vertices[0, :2], positions[0, :2])
        assert np.allclose(boundary_vertices[1, :2], positions[1, :2])

        # Z should be base_depth
        assert boundary_vertices[0, 2] == -1.0
        assert boundary_vertices[1, 2] == -1.0

    def test_create_boundary_extension_face_indices(self):
        """Test that side faces correctly reference top and bottom vertices."""
        from src.terrain.mesh_operations import create_boundary_extension

        positions = np.array([[0, 0, 5], [1, 0, 6]], dtype=float)
        boundary_points = [(0, 0), (1, 0)]
        coord_to_index = {(0, 0): 0, (1, 0): 1}

        boundary_vertices, boundary_faces = create_boundary_extension(
            positions, boundary_points, coord_to_index, base_depth=-2.0
        )

        # Should create 1 face connecting the 2 boundary points (wraps around)
        assert len(boundary_faces) == 2  # Each point to next (including wrap)

        # Faces should reference indices 0,1 (top) and 2,3 (bottom)
        # Bottom indices start after original positions
        n_positions = len(positions)
        for face in boundary_faces:
            # Face should have mix of top indices (<n_positions) and bottom (>=n_positions)
            assert any(idx < n_positions for idx in face)
            assert any(idx >= n_positions for idx in face)


class TestBoundaryPointSorting:
    """Tests for sort_boundary_points function."""

    def test_sort_boundary_points_imports(self):
        """Test that sort_boundary_points can be imported from mesh_operations."""
        from src.terrain.mesh_operations import sort_boundary_points

        assert callable(sort_boundary_points)

    def test_sort_boundary_points_simple_rectangle(self):
        """Test sorting boundary points of a simple rectangle."""
        from src.terrain.mesh_operations import sort_boundary_points

        # Simple 4x4 rectangle boundary points (perimeter only)
        boundary_coords = [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),  # Top edge
            (1, 3),
            (2, 3),
            (3, 3),  # Right edge
            (3, 2),
            (3, 1),
            (3, 0),  # Bottom edge
            (2, 0),
            (1, 0),  # Left edge
        ]

        # Shuffle to test sorting
        import random

        shuffled = boundary_coords.copy()
        random.shuffle(shuffled)

        # Sort the boundary points
        sorted_points = sort_boundary_points(shuffled)

        # Should form a continuous path around the perimeter
        # Check that consecutive points are adjacent (differ by 1 in x or y)
        for i in range(len(sorted_points)):
            p1 = sorted_points[i]
            p2 = sorted_points[(i + 1) % len(sorted_points)]

            # Calculate Manhattan distance
            distance = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

            # Adjacent points should be 1 unit apart
            assert (
                distance == 1
            ), f"Points {p1} and {p2} are not adjacent (distance={distance})"

    def test_sort_boundary_points_empty_list(self):
        """Test that empty boundary list returns empty list."""
        from src.terrain.mesh_operations import sort_boundary_points

        result = sort_boundary_points([])
        assert result == []

    def test_sort_boundary_points_single_point(self):
        """Test that single point returns list with that point."""
        from src.terrain.mesh_operations import sort_boundary_points

        result = sort_boundary_points([(5, 7)])
        assert result == [(5, 7)]


class TestTwoTierBoundaryExtension:
    """Tests for two-tier boundary extension functionality."""

    def test_two_tier_vertex_generation(self):
        """Test that two-tier mode creates mid and base vertices."""
        from src.terrain.mesh_operations import create_boundary_extension

        positions = np.array([[0, 0, 1], [1, 0, 2], [0, 1, 3], [1, 1, 4]], dtype=float)
        boundary_points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        # Maps (y, x) coordinates to position indices:
        # (0,0)->0 (z=1), (1,0)->1 (z=2), (1,1)->3 (z=4), (0,1)->2 (z=3)
        coord_to_index = {(0, 0): 0, (1, 0): 1, (1, 1): 3, (0, 1): 2}

        boundary_vertices, boundary_faces, boundary_colors = create_boundary_extension(
            positions,
            boundary_points,
            coord_to_index,
            base_depth=-0.4,
            two_tier=True,
            mid_depth=-0.1,
            base_material="clay",
        )

        # Should create 2*N vertices (N mid + N base)
        assert boundary_vertices.shape == (8, 3), "Should have 8 vertices (4 mid + 4 base)"

        # Mid tier vertices (first 4): each at its position's Z + mid_depth
        # positions Z values are [1, 2, 4, 3], mid_depth = -0.1
        # Expected: [0.9, 1.9, 3.9, 2.9]
        expected_mid_z = [0.9, 1.9, 3.9, 2.9]
        for i, v in enumerate(boundary_vertices[:4]):
            assert np.isclose(v[2], expected_mid_z[i]), f"Mid vertex {i} Z mismatch"

        # Base tier vertices (last 4): each at its position's Z + base_depth
        # positions Z values are [1, 2, 4, 3], base_depth = -0.4
        # Expected: [0.6, 1.6, 3.6, 2.6]
        expected_base_z = [0.6, 1.6, 3.6, 2.6]
        for i, v in enumerate(boundary_vertices[4:]):
            assert np.isclose(v[2], expected_base_z[i]), f"Base vertex {i} Z mismatch"

    def test_two_tier_mid_depth_auto_calculation(self):
        """Test that mid_depth auto-calculates as 25% of base_depth."""
        from src.terrain.mesh_operations import create_boundary_extension

        positions = np.array([[0, 0, 5]], dtype=float)
        boundary_points = [(0, 0)]
        coord_to_index = {(0, 0): 0}

        boundary_vertices, _, _ = create_boundary_extension(
            positions,
            boundary_points,
            coord_to_index,
            base_depth=-0.8,
            two_tier=True,
            mid_depth=None,  # Should auto-calculate
            base_material="clay",
        )

        # Mid depth auto-calculated as base_depth * 0.25 = -0.8 * 0.25 = -0.2
        # Mid vertex: surface + mid_depth = 5 + (-0.2) = 4.8
        # Base vertex: surface + base_depth = 5 + (-0.8) = 4.2
        assert np.isclose(boundary_vertices[0, 2], 4.8), "Mid vertex should be at surface + mid_depth"
        assert np.isclose(boundary_vertices[1, 2], 4.2), "Base vertex should be at surface + base_depth"

    def test_two_tier_face_generation(self):
        """Test that two-tier mode creates upper and lower tier faces."""
        from src.terrain.mesh_operations import create_boundary_extension

        positions = np.array([[0, 0, 1], [1, 0, 2]], dtype=float)
        boundary_points = [(0, 0), (1, 0)]
        coord_to_index = {(0, 0): 0, (1, 0): 1}

        _, boundary_faces, _ = create_boundary_extension(
            positions,
            boundary_points,
            coord_to_index,
            base_depth=-0.4,
            two_tier=True,
            mid_depth=-0.1,
            base_material="clay",
        )

        # Should create 2*N faces (N upper + N lower)
        # With 2 boundary points, expect 4 faces (2 per tier, includes wrap-around)
        assert len(boundary_faces) == 4, "Should have 4 faces (2 upper + 2 lower)"

        # All faces should be quads
        assert all(len(face) == 4 for face in boundary_faces), "All faces should be quads"

    def test_two_tier_color_assignment_blend_enabled(self):
        """Test color assignment with blend_edge_colors=True."""
        from src.terrain.mesh_operations import create_boundary_extension

        positions = np.array([[0, 0, 1], [1, 0, 2]], dtype=float)
        boundary_points = [(0, 0), (1, 0)]
        coord_to_index = {(0, 0): 0, (1, 0): 1}

        # Create terrain with colors
        terrain_colors = np.array([[255, 128, 64], [200, 150, 100]], dtype=np.uint8)

        _, _, boundary_colors = create_boundary_extension(
            positions,
            boundary_points,
            coord_to_index,
            base_depth=-0.4,
            two_tier=True,
            mid_depth=-0.1,
            base_material="clay",
            blend_edge_colors=True,
            surface_colors=terrain_colors,
        )

        # Should return colors array (2*N rows for mid + base)
        assert boundary_colors.shape == (4, 3), "Should have 4 color rows (2 mid + 2 base)"

        # Mid vertices (first 2) should have blended/copied surface colors
        # (exact values depend on implementation)
        assert boundary_colors[:2].dtype == np.uint8, "Colors should be uint8"

        # Base vertices (last 2) should have clay color
        clay_rgb = (0.5, 0.48, 0.45)
        expected_clay = (np.array(clay_rgb) * 255).astype(np.uint8)
        assert np.allclose(boundary_colors[2], expected_clay, atol=1), "Base colors should be clay"
        assert np.allclose(boundary_colors[3], expected_clay, atol=1), "Base colors should be clay"

    def test_two_tier_color_assignment_blend_disabled(self):
        """Test color assignment with blend_edge_colors=False."""
        from src.terrain.mesh_operations import create_boundary_extension

        positions = np.array([[0, 0, 1]], dtype=float)
        boundary_points = [(0, 0)]
        coord_to_index = {(0, 0): 0}

        _, _, boundary_colors = create_boundary_extension(
            positions,
            boundary_points,
            coord_to_index,
            base_depth=-0.4,
            two_tier=True,
            mid_depth=-0.1,
            base_material="clay",
            blend_edge_colors=False,
        )

        # Both mid and base should use clay color when blending disabled
        clay_rgb = (0.5, 0.48, 0.45)
        expected_clay = (np.array(clay_rgb) * 255).astype(np.uint8)
        assert np.allclose(boundary_colors[0], expected_clay, atol=1), "Mid should be clay when blending disabled"
        assert np.allclose(boundary_colors[1], expected_clay, atol=1), "Base should be clay"

    def test_two_tier_material_presets(self):
        """Test that material preset names resolve correctly."""
        from src.terrain.mesh_operations import create_boundary_extension

        positions = np.array([[0, 0, 1]], dtype=float)
        boundary_points = [(0, 0)]
        coord_to_index = {(0, 0): 0}

        # Test gold material
        _, _, boundary_colors = create_boundary_extension(
            positions,
            boundary_points,
            coord_to_index,
            base_depth=-0.4,
            two_tier=True,
            base_material="gold",
            blend_edge_colors=False,
        )

        # Base should have gold color
        gold_rgb = (1.0, 0.766, 0.336)
        expected_gold = (np.array(gold_rgb) * 255).astype(np.uint8)
        assert np.allclose(boundary_colors[1], expected_gold, atol=1), "Should use gold color"

    def test_two_tier_material_rgb_tuple(self):
        """Test that custom RGB tuples work."""
        from src.terrain.mesh_operations import create_boundary_extension

        positions = np.array([[0, 0, 1]], dtype=float)
        boundary_points = [(0, 0)]
        coord_to_index = {(0, 0): 0}

        custom_rgb = (0.6, 0.55, 0.5)
        _, _, boundary_colors = create_boundary_extension(
            positions,
            boundary_points,
            coord_to_index,
            base_depth=-0.4,
            two_tier=True,
            base_material=custom_rgb,
            blend_edge_colors=False,
        )

        # Base should have custom color
        expected_custom = (np.array(custom_rgb) * 255).astype(np.uint8)
        assert np.allclose(boundary_colors[1], expected_custom, atol=1), "Should use custom RGB"

    def test_two_tier_backwards_compatibility(self):
        """Test that two_tier=False produces same results as before."""
        from src.terrain.mesh_operations import create_boundary_extension

        positions = np.array([[0, 0, 1], [1, 0, 2]], dtype=float)
        boundary_points = [(0, 0), (1, 0)]
        coord_to_index = {(0, 0): 0, (1, 0): 1}

        # Old API (no two_tier parameter)
        verts_old, faces_old = create_boundary_extension(
            positions, boundary_points, coord_to_index, base_depth=-0.5
        )

        # Should return 2-tuple when two_tier=False (backwards compatible)
        assert isinstance(verts_old, np.ndarray), "Should return vertices array"
        assert isinstance(faces_old, list), "Should return faces list"
        assert verts_old.shape == (2, 3), "Should have 2 vertices (single tier)"
        assert len(faces_old) == 2, "Should have 2 faces (single tier)"

    def test_two_tier_invalid_material_name_raises(self):
        """Test that invalid material name raises helpful error."""
        from src.terrain.mesh_operations import create_boundary_extension

        positions = np.array([[0, 0, 1]], dtype=float)
        boundary_points = [(0, 0)]
        coord_to_index = {(0, 0): 0}

        with pytest.raises(ValueError) as exc_info:
            create_boundary_extension(
                positions,
                boundary_points,
                coord_to_index,
                two_tier=True,
                base_material="invalid_material",
            )

        # Check error message
        error_msg = str(exc_info.value)
        assert "Unknown base material" in error_msg, "Should mention unknown material"
        assert "invalid_material" in error_msg, "Should mention the invalid name"


class TestBoundarySmoothing:
    """Tests for boundary point smoothing to eliminate stair-step edges."""

    def test_smooth_boundary_points_imports(self):
        """Test that smooth_boundary_points can be imported."""
        from src.terrain.mesh_operations import smooth_boundary_points

        assert callable(smooth_boundary_points)

    def test_smooth_boundary_points_basic(self):
        """Test basic smoothing of boundary points."""
        from src.terrain.mesh_operations import smooth_boundary_points

        # Create a simple stair-step boundary (zigzag pattern)
        # This simulates pixel-grid aligned boundary
        boundary = [
            (0, 0),
            (0, 1),
            (1, 1),
            (1, 2),
            (2, 2),
            (2, 3),
        ]

        # Smooth with window size 3 as open path (not closed loop)
        smoothed = smooth_boundary_points(boundary, window_size=3, closed_loop=False)

        # Should return same number of points
        assert len(smoothed) == len(boundary)

        # Points should be floating point (not integer grid)
        assert isinstance(smoothed[0][0], (float, np.floating))

        # First and last points should be relatively unchanged (endpoints for open path)
        assert np.allclose(smoothed[0], boundary[0], atol=0.5)
        assert np.allclose(smoothed[-1], boundary[-1], atol=0.5)

        # Middle points should be smoothed (not exactly on grid)
        middle_idx = len(boundary) // 2
        # Should not be exactly the same as input (unless by chance)
        assert not np.allclose(smoothed[middle_idx], boundary[middle_idx], atol=0.01)

    def test_smooth_boundary_points_closed_loop(self):
        """Test smoothing preserves closed loop property."""
        from src.terrain.mesh_operations import smooth_boundary_points

        # Create a closed loop (rectangle-ish)
        boundary = [
            (0, 0),
            (0, 5),
            (5, 5),
            (5, 0),
        ]

        # Smooth as closed loop
        smoothed = smooth_boundary_points(boundary, window_size=3, closed_loop=True)

        # Should return same number of points
        assert len(smoothed) == len(boundary)

        # First and last should be close (closed loop)
        # When smoothing closed loop, endpoints also get smoothed
        assert len(smoothed) == len(boundary)

    def test_smooth_boundary_points_no_smoothing(self):
        """Test that window_size=1 returns original points."""
        from src.terrain.mesh_operations import smooth_boundary_points

        boundary = [(0, 0), (1, 2), (3, 4), (5, 6)]

        smoothed = smooth_boundary_points(boundary, window_size=1)

        # Should return essentially the same points (converted to float)
        assert len(smoothed) == len(boundary)
        for i in range(len(boundary)):
            assert np.allclose(smoothed[i], boundary[i], atol=1e-10)

    def test_smooth_boundary_points_increasing_strength(self):
        """Test that larger window size produces more smoothing."""
        from src.terrain.mesh_operations import smooth_boundary_points

        # Create a zigzag pattern
        boundary = [(i, i % 2) for i in range(10)]

        smoothed_3 = smooth_boundary_points(boundary, window_size=3)
        smoothed_5 = smooth_boundary_points(boundary, window_size=5)

        # More smoothing should reduce deviation from straight line
        # Calculate variance in y-coordinates for middle points
        middle_range = slice(2, 8)
        var_3 = np.var([p[1] for p in smoothed_3[middle_range]])
        var_5 = np.var([p[1] for p in smoothed_5[middle_range]])

        # Larger window should produce less variance (more smoothing)
        assert var_5 < var_3, "Larger window should smooth more"

    def test_smooth_boundary_points_empty_list(self):
        """Test handling of empty boundary list."""
        from src.terrain.mesh_operations import smooth_boundary_points

        smoothed = smooth_boundary_points([], window_size=3)

        assert smoothed == []

    def test_smooth_boundary_points_single_point(self):
        """Test handling of single point."""
        from src.terrain.mesh_operations import smooth_boundary_points

        boundary = [(5, 10)]
        smoothed = smooth_boundary_points(boundary, window_size=3)

        assert len(smoothed) == 1
        assert np.allclose(smoothed[0], boundary[0])

    def test_smooth_boundary_points_two_points(self):
        """Test handling of two points."""
        from src.terrain.mesh_operations import smooth_boundary_points

        boundary = [(0, 0), (10, 10)]
        smoothed = smooth_boundary_points(boundary, window_size=3)

        assert len(smoothed) == 2
        # With only 2 points, smoothing should have minimal effect
        assert np.allclose(smoothed[0], boundary[0], atol=0.5)
        assert np.allclose(smoothed[1], boundary[1], atol=0.5)


class TestCatmullRomCurve:
    """Tests for Catmull-Rom curve fitting and interpolation."""

    def test_catmull_rom_curve_imports(self):
        """Test that catmull_rom_curve can be imported."""
        from src.terrain.mesh_operations import catmull_rom_curve

        assert callable(catmull_rom_curve)

    def test_catmull_rom_curve_basic_interpolation(self):
        """Test basic Catmull-Rom curve interpolation."""
        from src.terrain.mesh_operations import catmull_rom_curve

        # Four control points
        p0 = np.array([0.0, 0.0])
        p1 = np.array([1.0, 0.0])
        p2 = np.array([2.0, 1.0])
        p3 = np.array([3.0, 1.0])

        # Sample at t=0.5 (midpoint between p1 and p2)
        result = catmull_rom_curve(p0, p1, p2, p3, 0.5)

        # Should return a point between p1 and p2
        assert isinstance(result, np.ndarray)
        assert len(result) == 2
        # Midpoint should be within bounds
        assert p1[0] <= result[0] <= p2[0]

    def test_catmull_rom_curve_t_boundaries(self):
        """Test Catmull-Rom curve at t=0 and t=1."""
        from src.terrain.mesh_operations import catmull_rom_curve

        p0 = np.array([0.0, 0.0])
        p1 = np.array([1.0, 1.0])
        p2 = np.array([2.0, 2.0])
        p3 = np.array([3.0, 3.0])

        # At t=0, should be close to p1
        result_0 = catmull_rom_curve(p0, p1, p2, p3, 0.0)
        assert np.allclose(result_0, p1, atol=0.01)

        # At t=1, should be close to p2
        result_1 = catmull_rom_curve(p0, p1, p2, p3, 1.0)
        assert np.allclose(result_1, p2, atol=0.01)

    def test_fit_catmull_rom_boundary_curve_imports(self):
        """Test that fit_catmull_rom_boundary_curve can be imported."""
        from src.terrain.mesh_operations import fit_catmull_rom_boundary_curve

        assert callable(fit_catmull_rom_boundary_curve)

    def test_fit_catmull_rom_boundary_curve_basic(self):
        """Test fitting Catmull-Rom curve through boundary points."""
        from src.terrain.mesh_operations import fit_catmull_rom_boundary_curve

        # Simple staircase boundary
        boundary = [(0, 0), (1, 0), (1, 1), (0, 1)]

        # Fit curve with subdivision
        smooth_curve = fit_catmull_rom_boundary_curve(boundary, subdivisions=5, closed_loop=True)

        # Should return more points than input due to subdivision
        assert len(smooth_curve) > len(boundary)

        # All points should be close to the original boundary area
        smooth_array = np.array(smooth_curve)
        assert np.all(smooth_array >= -1.0)  # Should not go far outside bounds
        assert np.all(smooth_array <= 2.0)

    def test_fit_catmull_rom_boundary_curve_smooth_output(self):
        """Test that fitted curve produces smooth output."""
        from src.terrain.mesh_operations import fit_catmull_rom_boundary_curve

        # Staircase pattern (jagged)
        boundary = [
            (0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)
        ]

        # Fit smooth curve
        smooth_curve = fit_catmull_rom_boundary_curve(boundary, subdivisions=10, closed_loop=False)

        # Check that curve doesn't have harsh directional changes like the original staircase
        smooth_array = np.array(smooth_curve)

        # Calculate directional vectors between consecutive points
        if len(smooth_curve) > 2:
            diffs = np.diff(smooth_array, axis=0)
            # Should have smooth transitions (not just axis-aligned moves)
            # At least some segments should have both x and y components
            diagonal_moves = np.sum((np.abs(diffs[:, 0]) > 0) & (np.abs(diffs[:, 1]) > 0))
            assert diagonal_moves > 0, "Smooth curve should have diagonal transitions"

    def test_fit_catmull_rom_boundary_curve_closed_loop(self):
        """Test Catmull-Rom curve with closed loop."""
        from src.terrain.mesh_operations import fit_catmull_rom_boundary_curve

        # Rectangle
        boundary = [(0, 0), (2, 0), (2, 2), (0, 2)]

        smooth_curve = fit_catmull_rom_boundary_curve(boundary, subdivisions=5, closed_loop=True)

        # First and last points should be close (closed loop)
        if len(smooth_curve) > 1:
            assert np.allclose(smooth_curve[0], smooth_curve[-1], atol=0.5)

    def test_fit_catmull_rom_boundary_curve_open_path(self):
        """Test Catmull-Rom curve with open path."""
        from src.terrain.mesh_operations import fit_catmull_rom_boundary_curve

        # Line of points
        boundary = [(0, 0), (1, 1), (2, 0), (3, 1)]

        smooth_curve = fit_catmull_rom_boundary_curve(boundary, subdivisions=5, closed_loop=False)

        # First and last should match endpoints
        assert np.allclose(smooth_curve[0], boundary[0], atol=0.5)
        assert np.allclose(smooth_curve[-1], boundary[-1], atol=0.5)


class TestCreateBoundaryExtensionCatmullRom:
    """Tests for create_boundary_extension with Catmull-Rom curve integration."""

    def test_create_boundary_extension_catmull_rom_parameter(self):
        """Test that use_catmull_rom parameter is accepted."""
        from src.terrain.mesh_operations import create_boundary_extension

        # Simple 2x2 DEM
        dem = np.array([[1.0, 2.0], [3.0, 4.0]])
        positions = np.array([[0, 0, 1], [1, 0, 2], [0, 1, 3], [1, 1, 4]], dtype=float)
        boundary_points = [(0, 0), (0, 1), (1, 1), (1, 0)]
        coord_to_index = {(0, 0): 0, (0, 1): 1, (1, 1): 3, (1, 0): 2}

        # Call with use_catmull_rom=True (should not raise error)
        result = create_boundary_extension(
            positions,
            boundary_points,
            coord_to_index,
            use_catmull_rom=True,
            catmull_rom_subdivisions=5,
        )

        # Should return tuple
        assert isinstance(result, tuple)
        assert len(result) == 2  # Single-tier mode returns (vertices, faces)

    def test_create_boundary_extension_catmull_rom_increases_vertices(self):
        """Test that Catmull-Rom increases boundary vertex count via interpolation."""
        from src.terrain.mesh_operations import create_boundary_extension

        positions = np.array(
            [[0, 0, 1], [1, 0, 2], [1, 1, 3], [0, 1, 4]], dtype=float
        )
        boundary_points = [(0, 0), (0, 1), (1, 1), (1, 0)]
        coord_to_index = {(0, 0): 0, (0, 1): 3, (1, 1): 2, (1, 0): 1}

        # Without Catmull-Rom
        result_without = create_boundary_extension(
            positions,
            boundary_points,
            coord_to_index,
            use_catmull_rom=False,
        )
        vertices_without, _ = result_without

        # With Catmull-Rom (subdivisions=10)
        result_with = create_boundary_extension(
            positions,
            boundary_points,
            coord_to_index,
            use_catmull_rom=True,
            catmull_rom_subdivisions=10,
        )
        vertices_with, _ = result_with

        # Catmull-Rom should produce more vertices due to interpolation
        # Original: 4 boundary points
        # With subdivisions=10: ~4*10 = 40 interpolated points + original
        assert len(vertices_with) > len(vertices_without)

    def test_create_boundary_extension_catmull_rom_subdivisions_effect(self):
        """Test that higher subdivisions produce more vertices."""
        from src.terrain.mesh_operations import create_boundary_extension

        positions = np.array(
            [[0, 0, 1], [1, 0, 2], [1, 1, 3], [0, 1, 4]], dtype=float
        )
        boundary_points = [(0, 0), (0, 1), (1, 1), (1, 0)]
        coord_to_index = {(0, 0): 0, (0, 1): 3, (1, 1): 2, (1, 0): 1}

        # With lower subdivisions
        result_low = create_boundary_extension(
            positions,
            boundary_points,
            coord_to_index,
            use_catmull_rom=True,
            catmull_rom_subdivisions=5,
        )
        vertices_low, _ = result_low

        # With higher subdivisions
        result_high = create_boundary_extension(
            positions,
            boundary_points,
            coord_to_index,
            use_catmull_rom=True,
            catmull_rom_subdivisions=20,
        )
        vertices_high, _ = result_high

        # Higher subdivisions should produce more vertices
        assert len(vertices_high) > len(vertices_low)

    def test_create_boundary_extension_catmull_rom_backwards_compatible(self):
        """Test that default (use_catmull_rom=False) preserves original behavior."""
        from src.terrain.mesh_operations import create_boundary_extension

        positions = np.array(
            [[0, 0, 1], [1, 0, 2], [1, 1, 3], [0, 1, 4]], dtype=float
        )
        boundary_points = [(0, 0), (0, 1), (1, 1), (1, 0)]
        coord_to_index = {(0, 0): 0, (0, 1): 3, (1, 1): 2, (1, 0): 1}

        # Default call (use_catmull_rom=False)
        result1 = create_boundary_extension(
            positions,
            boundary_points,
            coord_to_index,
        )

        # Explicit False
        result2 = create_boundary_extension(
            positions,
            boundary_points,
            coord_to_index,
            use_catmull_rom=False,
        )

        vertices1, faces1 = result1
        vertices2, faces2 = result2

        # Should be identical
        assert np.allclose(vertices1, vertices2)
        assert faces1 == faces2

    def test_create_boundary_extension_catmull_rom_two_tier(self):
        """Test Catmull-Rom with two-tier edge mode."""
        from src.terrain.mesh_operations import create_boundary_extension

        positions = np.array(
            [[0, 0, 1], [1, 0, 2], [1, 1, 3], [0, 1, 4]], dtype=float
        )
        boundary_points = [(0, 0), (0, 1), (1, 1), (1, 0)]
        coord_to_index = {(0, 0): 0, (0, 1): 3, (1, 1): 2, (1, 0): 1}

        # Two-tier with Catmull-Rom
        result = create_boundary_extension(
            positions,
            boundary_points,
            coord_to_index,
            two_tier=True,
            use_catmull_rom=True,
            catmull_rom_subdivisions=5,
        )

        # Should return 3-tuple for two-tier mode
        assert isinstance(result, tuple)
        assert len(result) == 3
        vertices, faces, colors = result

        # Vertices should be positive (has content)
        assert vertices.shape[0] > 0
        assert vertices.shape[1] == 3  # x, y, z

        # Colors should match vertex count
        assert colors.shape[0] == vertices.shape[0]
        assert colors.shape[1] == 3  # RGB

    def test_create_boundary_extension_catmull_rom_smooth_vertices(self):
        """Test that Catmull-Rom produces smooth vertex transitions."""
        from src.terrain.mesh_operations import create_boundary_extension

        # Create staircase boundary pattern
        positions = np.array(
            [
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [2, 1, 1],
                [2, 2, 1],
            ],
            dtype=float,
        )
        boundary_points = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)]
        coord_to_index = {
            (0, 0): 0,
            (0, 1): 1,
            (1, 1): 2,
            (1, 2): 3,
            (2, 2): 4,
        }

        # With Catmull-Rom
        result = create_boundary_extension(
            positions,
            boundary_points,
            coord_to_index,
            use_catmull_rom=True,
            catmull_rom_subdivisions=10,
        )
        vertices_cr, _ = result

        # Check that we have smooth transitions (not just axis-aligned)
        if len(vertices_cr) > 2:
            diffs = np.diff(vertices_cr[:, :2], axis=0)  # Just x,y
            # Should have some diagonal moves (both x and y components)
            diagonal_moves = np.sum(
                (np.abs(diffs[:, 0]) > 0.01) & (np.abs(diffs[:, 1]) > 0.01)
            )
            assert diagonal_moves > 0, "Catmull-Rom should produce diagonal transitions"
