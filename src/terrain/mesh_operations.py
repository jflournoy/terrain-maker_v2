"""
Mesh generation operations for terrain visualization.

This module contains functions for creating and manipulating terrain meshes,
extracted from the core Terrain class for better modularity and testability.
"""

import numpy as np
from scipy.spatial import cKDTree


def sort_boundary_points(boundary_coords):
    """
    Sort boundary points efficiently using spatial relationships.

    Uses a KD-tree for efficient nearest neighbor queries to create a continuous
    path along the boundary points. This is useful for creating side faces that
    close a terrain mesh into a solid object.

    Args:
        boundary_coords: List of (y, x) coordinate tuples representing boundary points

    Returns:
        list: Sorted boundary points forming a continuous path around the perimeter
    """
    # Quick return for small boundaries
    if len(boundary_coords) <= 2:
        return boundary_coords

    # Start with leftmost-topmost point for consistency
    start_point = min(boundary_coords, key=lambda p: (p[1], p[0]))

    # Use a KD-tree for nearest neighbor queries - much faster than manual distance calculation
    # Convert to numpy array for KD-tree
    points_array = np.array(boundary_coords)
    kdtree = cKDTree(points_array)

    # Initialize result with start point
    ordered = [start_point]

    # Track points we've already used - faster lookups
    used_indices = set([boundary_coords.index(start_point)])

    current = start_point

    # Find next closest point until all points are used
    while len(ordered) < len(boundary_coords):
        # Query KD-tree for 10 nearest neighbors (more than enough)
        distances, indices = kdtree.query(current, k=10)

        # Find the closest unused point
        next_point = None
        for i in range(len(indices)):
            idx = indices[i]
            if idx < len(points_array) and idx not in used_indices:
                next_point = tuple(points_array[idx])
                used_indices.add(idx)
                break

        # If no more valid neighbors, break
        if next_point is None:
            break

        ordered.append(next_point)
        current = next_point

    return ordered
