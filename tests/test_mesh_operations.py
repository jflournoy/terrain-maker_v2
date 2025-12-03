"""
Tests for mesh_operations module.

Tests the extracted mesh generation functions that were refactored from core.py.
"""

import pytest
import numpy as np


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
