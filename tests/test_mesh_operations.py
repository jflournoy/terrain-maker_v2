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
