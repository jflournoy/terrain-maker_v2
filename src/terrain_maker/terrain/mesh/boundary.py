"""Boundary point ordering and smoothing, including Catmull-Rom curve fitting."""

import logging

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


def catmull_rom_curve(p0, p1, p2, p3, t):
    """
    Catmull-Rom spline interpolation between two points.

    Evaluates a Catmull-Rom spline at parameter t, using four control points.
    The curve passes through p1 and p2, and is influenced by p0 and p3.

    Args:
        p0, p1, p2, p3: Control points (numpy arrays or tuples)
        t: Parameter in [0, 1] where 0=p1 and 1=p2

    Returns:
        Point on the curve at parameter t
    """
    t = np.clip(t, 0.0, 1.0)
    t2 = t * t
    t3 = t2 * t

    # Catmull-Rom basis functions
    q = 0.5 * np.array([-t3 + 2 * t2 - t, 3 * t3 - 5 * t2 + 2, -3 * t3 + 4 * t2 + t, t3 - t2])

    # Weighted sum of control points
    return q[0] * p0 + q[1] * p1 + q[2] * p2 + q[3] * p3


def fit_catmull_rom_boundary_curve(boundary_points, subdivisions=10, closed_loop=True):
    """
    Fit a Catmull-Rom spline curve through boundary points.

    Creates a smooth curve that passes through all boundary points by fitting
    Catmull-Rom spline segments between consecutive points.

    Args:
        boundary_points: List of (y, x) or (x, y) tuples representing the boundary
        subdivisions: Number of interpolated points per segment (higher = smoother)
        closed_loop: If True, treat the boundary as a closed loop

    Returns:
        List of interpolated points along the smooth curve
    """
    if len(boundary_points) < 2:
        return list(boundary_points)

    boundary_points = np.array(boundary_points, dtype=float)
    n = len(boundary_points)

    # Handle different boundary types
    if closed_loop and n >= 3:
        # For closed loop, extend boundary to wrap around
        extended = np.vstack(
            [
                boundary_points[-1:],  # Previous point
                boundary_points,
                boundary_points[:2],  # Next points
            ]
        )
    else:
        # For open path, duplicate endpoints for boundary handling
        extended = np.vstack(
            [
                boundary_points[0:1],  # Duplicate first point
                boundary_points,
                boundary_points[-1:],  # Duplicate last point
            ]
        )

    # Interpolate along the curve
    smooth_curve = []

    # Number of segments to process
    n_segments = n - 1 if not closed_loop else n

    for i in range(n_segments):
        # Get four consecutive points for this segment
        p0 = extended[i]
        p1 = extended[i + 1]
        p2 = extended[i + 2]
        p3 = extended[i + 3]

        # Generate subdivisions for this segment
        for j in range(subdivisions):
            t = j / float(subdivisions)
            point = catmull_rom_curve(p0, p1, p2, p3, t)
            smooth_curve.append(point)

    # Add final point for open path
    if not closed_loop:
        smooth_curve.append(boundary_points[-1])

    # For closed loop: do NOT append smooth_curve[0] at the end.
    # The face creation loop uses modulo arithmetic (next_i = (i+1) % n_boundary)
    # which naturally wraps around, so the duplicate point would create a degenerate
    # zero-area face with incorrect normals (appears dark/black when rendered).

    # Clamp curve points to stay within original boundary bounds
    # Catmull-Rom splines can overshoot between control points, creating coordinates
    # outside the valid DEM area. Clamp to min/max of original boundary points.
    if len(boundary_points) > 0:
        min_y = boundary_points[:, 0].min()
        max_y = boundary_points[:, 0].max()
        min_x = boundary_points[:, 1].min()
        max_x = boundary_points[:, 1].max()

        smooth_curve = [
            np.array([np.clip(pt[0], min_y, max_y), np.clip(pt[1], min_x, max_x)])
            for pt in smooth_curve
        ]

    # Remove duplicate/very-close points (can occur from curve wrapping or coincidental
    # interpolation): they create degenerate zero-area faces that render dark
    return _drop_near_duplicates(smooth_curve)


def _drop_near_duplicates(points, atol=1e-6, rtol=1e-5):
    """Keep the first occurrence of each point, dropping later points that are
    np.allclose(point, kept, atol=1e-6) to an earlier kept point.

    That is |point - kept| <= atol + rtol * |kept| on both axes. Candidates are found
    through a grid hash whose cells are at least the largest tolerance, so only the
    3x3 neighboring cells need checking: linear time instead of comparing every pair.
    """
    if len(points) == 0:
        return []
    coords = np.asarray(points, dtype=float)
    if not np.all(np.isfinite(coords)):
        raise ValueError("boundary curve contains non-finite points")
    cell = atol + rtol * float(np.abs(coords).max())
    buckets = {}
    kept = []
    for point, (y, x) in zip(points, coords):
        cy, cx = int(np.floor(y / cell)), int(np.floor(x / cell))
        duplicate = any(
            abs(y - py) <= atol + rtol * abs(py) and abs(x - px) <= atol + rtol * abs(px)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            for py, px in buckets.get((cy + dy, cx + dx), ())
        )
        if not duplicate:
            kept.append(point)
            buckets.setdefault((cy, cx), []).append((y, x))
    return kept


def smooth_boundary_points(boundary_coords, window_size=3, closed_loop=True):
    """
    Smooth boundary points using moving average to eliminate stair-step edges.

    Applies a moving average filter to boundary coordinates to create smoother
    curves instead of following pixel grid exactly. This reduces the jagged
    appearance on curved edges while preserving overall shape.

    Args:
        boundary_coords: List of (y, x) coordinate tuples representing boundary points
        window_size: Size of smoothing window (must be odd, default: 3).
                    Larger values produce more smoothing.
        closed_loop: If True, treat boundary as closed loop (wrap edges).
                    If False, treat as open path (endpoints less smoothed).

    Returns:
        list: Smoothed boundary points as list of (y, x) float tuples

    Examples:
        >>> boundary = [(0, 0), (0, 1), (1, 1), (1, 2)]
        >>> smoothed = smooth_boundary_points(boundary, window_size=3)
        >>> # Returns smoothed coordinates with reduced stair-stepping
    """
    # Handle edge cases
    if len(boundary_coords) == 0:
        return []

    if len(boundary_coords) == 1:
        return [tuple(float(c) for c in boundary_coords[0])]

    if len(boundary_coords) == 2:
        return [tuple(float(c) for c in pt) for pt in boundary_coords]

    # Ensure window size is odd and at least 1
    window_size = max(1, window_size)
    if window_size % 2 == 0:
        window_size += 1

    # No smoothing for window_size=1
    if window_size == 1:
        return [tuple(float(c) for c in pt) for pt in boundary_coords]

    # Convert to numpy array for efficient computation
    coords_array = np.array(boundary_coords, dtype=float)
    n_points = len(coords_array)

    # Create smoothed array
    smoothed = np.zeros_like(coords_array)

    # Half window size for indexing
    half_window = window_size // 2

    # Apply moving average
    for i in range(n_points):
        if closed_loop:
            # Wrap around for closed loop
            indices = [(i + offset - half_window) % n_points for offset in range(window_size)]
        else:
            # Clamp to edges for open path
            indices = [
                max(0, min(n_points - 1, i + offset - half_window)) for offset in range(window_size)
            ]

        # Average the coordinates
        smoothed[i] = coords_array[indices].mean(axis=0)

    # Convert back to list of tuples
    return [tuple(pt) for pt in smoothed]


def deduplicate_boundary_points(boundary_coords):
    """
    Remove duplicate points while preserving the original order.

    After coordinate transformations, many boundary points map to the same
    pixel coordinates, creating duplicates. This function removes duplicates
    while preserving the original perimeter traversal order.

    Args:
        boundary_coords: List of (y, x) coordinate tuples

    Returns:
        list: Deduplicated boundary points in original order
    """
    if len(boundary_coords) <= 1:
        return boundary_coords

    seen = set()
    unique_points = []

    for point in boundary_coords:
        point_tuple = tuple(point)
        if point_tuple not in seen:
            seen.add(point_tuple)
            unique_points.append(point)

    duplicates_removed = len(boundary_coords) - len(unique_points)
    if duplicates_removed > 0:
        logger.info(f"    Removed {duplicates_removed} duplicate points")

    return unique_points


def sort_boundary_points_angular(boundary_coords):
    """
    Sort boundary points by angle from centroid to form a closed loop.

    This is much faster than nearest-neighbor sorting and works well for dense
    boundaries (>10K points). Computes the centroid of all boundary points,
    then sorts by angle, creating a natural closed loop around the perimeter.

    After angular sorting, rotates the list so the largest gap between consecutive
    points becomes the start/end, preventing diagonal faces across the mesh.

    Args:
        boundary_coords: List of (y, x) coordinate tuples representing boundary points

    Returns:
        list: Sorted boundary points forming a continuous closed loop
    """
    # Quick return for small boundaries
    if len(boundary_coords) <= 2:
        return boundary_coords

    # Convert to numpy for vectorized operations
    points_array = np.array(boundary_coords, dtype=float)

    # Compute centroid
    centroid = points_array.mean(axis=0)

    # Compute angle from centroid for each point
    # Using atan2(y - cy, x - cx) gives angle in range [-pi, pi]
    dy = points_array[:, 0] - centroid[0]
    dx = points_array[:, 1] - centroid[1]
    angles = np.arctan2(dy, dx)

    # Sort by angle (counter-clockwise from -pi to pi)
    sorted_indices = np.argsort(angles)
    sorted_array = points_array[sorted_indices]

    # Find the largest gap between consecutive points
    # This is where we should split the loop to avoid a diagonal face
    distances = np.zeros(len(sorted_array))
    for i in range(len(sorted_array)):
        next_i = (i + 1) % len(sorted_array)
        dy = sorted_array[next_i, 0] - sorted_array[i, 0]
        dx = sorted_array[next_i, 1] - sorted_array[i, 1]
        distances[i] = np.sqrt(dy**2 + dx**2)

    # Find the index with the largest gap
    max_gap_idx = np.argmax(distances)
    max_gap_distance = distances[max_gap_idx]

    # Rotate the list so the largest gap is at the end (becomes wrap-around)
    # This puts the start/end at adjacent points on the perimeter
    rotated_array = np.roll(sorted_array, -max_gap_idx - 1, axis=0)

    # Report the wrap-around gap (will be the max gap we just found)
    logger.info(
        f"  Angular sorting: max gap = {max_gap_distance:.2f} pixels (placed at wrap-around)"
    )

    # Convert back to list of tuples
    sorted_points = [tuple(pt) for pt in rotated_array]

    return sorted_points


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
    # Dynamically adjust k based on boundary density
    # For dense boundaries (>10K points), query more neighbors to avoid getting stuck
    n_points = len(boundary_coords)
    k_neighbors = min(100, n_points)  # Query up to 100 neighbors for dense boundaries

    while len(ordered) < n_points:
        # Query KD-tree for k nearest neighbors
        # For dense boundaries, we need to search farther to find the next sequential point
        distances, indices = kdtree.query(current, k=k_neighbors)

        # Find the closest unused point
        next_point = None
        for i in range(len(indices)):
            idx = indices[i]
            if idx < len(points_array) and idx not in used_indices:
                next_point = tuple(points_array[idx])
                used_indices.add(idx)
                break

        # If no more valid neighbors, break
        # This can happen if the boundary has disconnected components
        if next_point is None:
            # DEBUG: Report incomplete sorting
            missing = n_points - len(ordered)
            if missing > n_points * 0.01:  # More than 1% points missing
                logger.warning(
                    f"  ⚠️  Warning: Boundary sorting incomplete - {missing}/{n_points} points not connected"
                )
            break

        ordered.append(next_point)
        current = next_point

    return ordered
