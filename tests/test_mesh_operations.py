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
        coord_to_index = {(0, 0): 0, (1, 0): 1, (1, 1): 2, (0, 1): 3}

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

        # First 4 vertices (mid tier) should be at mid_depth
        assert all(v[2] == -0.1 for v in boundary_vertices[:4]), "Mid vertices at wrong depth"

        # Last 4 vertices (base tier) should be at base_depth
        assert all(v[2] == -0.4 for v in boundary_vertices[4:]), "Base vertices at wrong depth"

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

        # mid_depth should be base_depth * 0.25 = -0.8 * 0.25 = -0.2
        assert np.isclose(boundary_vertices[0, 2], -0.2), "Mid depth not auto-calculated correctly"

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
