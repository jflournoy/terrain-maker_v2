"""
Mesh generation operations for terrain visualization.

This module contains functions for creating and manipulating terrain meshes,
extracted from the core Terrain class for better modularity and testability.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage


def find_boundary_points(valid_mask):
    """
    Find boundary points using morphological operations.

    Identifies points on the edge of valid regions using binary erosion.
    A point is considered a boundary point if it is valid but has at least
    one invalid neighbor in a 4-connected neighborhood.

    Args:
        valid_mask (np.ndarray): Boolean mask indicating valid points (True for valid)

    Returns:
        list: List of (y, x) coordinate tuples representing boundary points
    """
    # Interior points have 4 neighbors in a 4-connected neighborhood
    struct = ndimage.generate_binary_structure(2, 1)  # 4-connected structure
    eroded = ndimage.binary_erosion(valid_mask, struct)
    boundary_mask = valid_mask & ~eroded

    # Get boundary coords as (y,x) tuples
    boundary_indices = np.where(boundary_mask)
    boundary_coords = list(zip(boundary_indices[0], boundary_indices[1]))

    return boundary_coords


def generate_vertex_positions(dem_data, valid_mask, scale_factor=100.0, height_scale=1.0):
    """
    Generate 3D vertex positions from DEM data.

    Converts 2D elevation grid into 3D positions for mesh vertices, applying
    scaling factors for visualization. Only generates vertices for valid (non-NaN)
    DEM values.

    Args:
        dem_data (np.ndarray): 2D array of elevation values (height x width)
        valid_mask (np.ndarray): Boolean mask indicating valid points (True for non-NaN)
        scale_factor (float): Horizontal scale divisor for x/y coordinates (default: 100.0).
            Higher values produce smaller meshes. E.g., 100 means 100 DEM units = 1 unit.
        height_scale (float): Multiplier for elevation values (default: 1.0).
            Values > 1 exaggerate terrain, < 1 flatten it.

    Returns:
        tuple: (positions, y_valid, x_valid) where:
            - positions: np.ndarray of shape (n_valid, 3) with (x, y, z) coordinates
            - y_valid: np.ndarray of y indices for valid points
            - x_valid: np.ndarray of x indices for valid points
    """
    height, width = dem_data.shape

    # Use NumPy for coordinate generation
    y_indices, x_indices = np.mgrid[0:height, 0:width]
    y_valid = y_indices[valid_mask]
    x_valid = x_indices[valid_mask]

    # Generate vertex positions with scaling
    positions = np.column_stack(
        [
            x_valid / scale_factor,  # x position
            y_valid / scale_factor,  # y position
            dem_data[valid_mask] * height_scale,  # z position with height scaling
        ]
    )

    return positions, y_valid, x_valid


def generate_faces(height, width, coord_to_index, batch_size=10000):
    """
    Generate mesh faces from a grid of valid points.

    Creates quad faces for the mesh by checking each potential quad position
    and verifying that its corners exist in the coordinate-to-index mapping.
    If a quad has all 4 corners, creates a quad face. If it has 3 corners,
    creates a triangle face. Skips quads with fewer than 3 corners.

    Args:
        height (int): Height of the DEM grid
        width (int): Width of the DEM grid
        coord_to_index (dict): Mapping from (y, x) coordinates to vertex indices
        batch_size (int): Number of quads to process in each batch (default: 10000)

    Returns:
        list: List of face tuples, where each tuple contains vertex indices
    """
    # Generate all potential quad faces
    y_quads, x_quads = np.mgrid[0 : height - 1, 0 : width - 1]
    y_quads = y_quads.flatten()
    x_quads = x_quads.flatten()

    # Collect faces efficiently
    faces = []

    # Use batch processing to reduce Python loop overhead
    n_quads = len(y_quads)

    for batch_start in range(0, n_quads, batch_size):
        batch_end = min(batch_start + batch_size, n_quads)
        batch_y = y_quads[batch_start:batch_end]
        batch_x = x_quads[batch_start:batch_end]

        # For each quad, check if corners exist in valid points
        for i in range(batch_end - batch_start):
            y, x = batch_y[i], batch_x[i]
            quad_points = [(y, x), (y, x + 1), (y + 1, x + 1), (y + 1, x)]

            # Get indices for each corner that exists
            valid_indices = []
            for point in quad_points:
                if point in coord_to_index:
                    valid_indices.append(coord_to_index[point])

            # Only create faces with at least 3 points
            if len(valid_indices) >= 3:
                faces.append(tuple(valid_indices))

    return faces


def create_boundary_extension(positions, boundary_points, coord_to_index, base_depth=-0.2):
    """
    Create boundary extension vertices and faces to close the mesh.

    Creates a "skirt" around the terrain by adding bottom vertices at base_depth
    and connecting them to the top boundary with quad faces. This closes the mesh
    into a solid object suitable for 3D printing or solid rendering.

    Args:
        positions (np.ndarray): Array of (n, 3) vertex positions
        boundary_points (list): List of (y, x) tuples representing ordered boundary points
        coord_to_index (dict): Mapping from (y, x) coordinates to vertex indices
        base_depth (float): Z-coordinate for the bottom of the extension (default: -0.2)

    Returns:
        tuple: (boundary_vertices, boundary_faces) where:
            - boundary_vertices: np.ndarray of (n_boundary, 3) bottom vertex positions
            - boundary_faces: list of tuples defining side face quad connectivity
    """
    n_boundary = len(boundary_points)
    boundary_vertices = np.zeros((n_boundary, 3), dtype=float)

    # Create bottom vertices for each boundary point
    for i, (y, x) in enumerate(boundary_points):
        # Get the original position
        original_idx = coord_to_index.get((y, x))
        if original_idx is None:
            continue

        # Copy position but set z to base_depth
        pos = positions[original_idx].copy()
        pos[2] = base_depth
        boundary_vertices[i] = pos

    # Create side faces efficiently
    boundary_indices = [coord_to_index.get((y, x)) for y, x in boundary_points]
    base_indices = list(range(len(positions), len(positions) + len(boundary_points)))

    boundary_faces = []
    for i in range(n_boundary):
        # Skip invalid indices
        if boundary_indices[i] is None:
            continue

        next_i = (i + 1) % n_boundary
        # Skip if next boundary point is invalid
        if boundary_indices[next_i] is None:
            continue

        # Create quad connecting top boundary to bottom
        boundary_faces.append(
            (
                boundary_indices[i],
                boundary_indices[next_i],
                base_indices[next_i],
                base_indices[i],
            )
        )

    return boundary_vertices, boundary_faces


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

    # Find start index using numpy (O(n) comparison, but vectorized)
    start_idx = np.where((points_array == start_point).all(axis=1))[0][0]
    # Track points we've already used - faster lookups
    used_indices = set([start_idx])

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
