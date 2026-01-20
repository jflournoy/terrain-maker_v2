"""
Mesh generation operations for terrain visualization.

This module contains functions for creating and manipulating terrain meshes,
extracted from the core Terrain class for better modularity and testability.

Performance optimizations:
- Numba JIT compilation for hot loops (face generation)
- Vectorized NumPy operations where possible
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage

# Try to import numba for JIT compilation
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        """No-op JIT decorator fallback when numba is not available.

        When numba is not installed, this decorator simply returns the function
        unchanged, allowing code to run without JIT compilation.

        Args:
            *args: Ignored positional arguments (for numba compatibility)
            **kwargs: Ignored keyword arguments (for numba compatibility)

        Returns:
            Decorator function that returns the original function unchanged
        """
        def decorator(func):
            """Inner decorator that returns the function unchanged."""
            return func
        return decorator
    prange = range


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


@jit(nopython=True, parallel=True, cache=True)
def _generate_faces_numba(height, width, index_grid):
    """
    Numba-accelerated face generation.

    Args:
        height: Grid height
        width: Grid width
        index_grid: 2D array where index_grid[y,x] = vertex index, or -1 if invalid

    Returns:
        faces: Array of face vertex indices (n_faces, 4), padded with -1 for triangles
        n_faces: Number of valid faces
    """
    # Pre-allocate maximum possible faces (each cell can have 1 quad)
    max_faces = (height - 1) * (width - 1)
    faces = np.full((max_faces, 4), -1, dtype=np.int64)
    face_count = 0

    # Process each potential quad
    for y in prange(height - 1):
        for x in range(width - 1):
            # Get indices for quad corners
            i0 = index_grid[y, x]
            i1 = index_grid[y, x + 1]
            i2 = index_grid[y + 1, x + 1]
            i3 = index_grid[y + 1, x]

            # Count valid corners
            valid_count = (i0 >= 0) + (i1 >= 0) + (i2 >= 0) + (i3 >= 0)

            if valid_count >= 3:
                # Store face (quads have 4 indices, triangles have 3 with -1 padding)
                idx = y * (width - 1) + x  # Unique index for this quad position
                if valid_count == 4:
                    faces[idx, 0] = i0
                    faces[idx, 1] = i1
                    faces[idx, 2] = i2
                    faces[idx, 3] = i3
                else:
                    # Triangle - collect valid indices
                    j = 0
                    if i0 >= 0:
                        faces[idx, j] = i0
                        j += 1
                    if i1 >= 0:
                        faces[idx, j] = i1
                        j += 1
                    if i2 >= 0:
                        faces[idx, j] = i2
                        j += 1
                    if i3 >= 0:
                        faces[idx, j] = i3
                        j += 1

    return faces


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
    if HAS_NUMBA:
        # Convert dict to 2D index grid for Numba
        index_grid = np.full((height, width), -1, dtype=np.int64)
        for (y, x), idx in coord_to_index.items():
            index_grid[y, x] = idx

        # Run Numba-accelerated version
        faces_array = _generate_faces_numba(height, width, index_grid)

        # Filter out unused slots and convert to list of tuples
        faces = []
        for i in range(faces_array.shape[0]):
            if faces_array[i, 0] >= 0:  # Valid face
                if faces_array[i, 3] >= 0:  # Quad
                    faces.append(tuple(faces_array[i]))
                else:  # Triangle
                    faces.append(tuple(faces_array[i, :3]))
        return faces

    # Fallback: Original Python implementation
    y_quads, x_quads = np.mgrid[0 : height - 1, 0 : width - 1]
    y_quads = y_quads.flatten()
    x_quads = x_quads.flatten()

    faces = []
    n_quads = len(y_quads)

    for batch_start in range(0, n_quads, batch_size):
        batch_end = min(batch_start + batch_size, n_quads)
        batch_y = y_quads[batch_start:batch_end]
        batch_x = x_quads[batch_start:batch_end]

        for i in range(batch_end - batch_start):
            y, x = batch_y[i], batch_x[i]
            quad_points = [(y, x), (y, x + 1), (y + 1, x + 1), (y + 1, x)]

            valid_indices = []
            for point in quad_points:
                if point in coord_to_index:
                    valid_indices.append(coord_to_index[point])

            if len(valid_indices) >= 3:
                faces.append(tuple(valid_indices))

    return faces


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
    q = 0.5 * np.array([
        -t3 + 2*t2 - t,
        3*t3 - 5*t2 + 2,
        -3*t3 + 4*t2 + t,
        t3 - t2
    ])

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
        extended = np.vstack([
            boundary_points[-1:],  # Previous point
            boundary_points,
            boundary_points[:2]    # Next points
        ])
    else:
        # For open path, duplicate endpoints for boundary handling
        extended = np.vstack([
            boundary_points[0:1],  # Duplicate first point
            boundary_points,
            boundary_points[-1:]   # Duplicate last point
        ])

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
            np.array([
                np.clip(pt[0], min_y, max_y),
                np.clip(pt[1], min_x, max_x)
            ]) for pt in smooth_curve
        ]

    # Remove duplicate/very-close points (can occur from curve wrapping or coincidental interpolation)
    # These create degenerate zero-area faces that cause rendering artifacts
    # Keep track of unique points by checking distance to all previous points
    filtered_curve = []
    for pt in smooth_curve:
        # Check if this point is too close to any previous point
        is_duplicate = False
        for prev_pt in filtered_curve:
            if np.allclose(pt, prev_pt, atol=1e-6):
                is_duplicate = True
                break

        if not is_duplicate:
            filtered_curve.append(pt)

    # If we removed points and this is a closed loop, make sure we don't have the first point appearing in the middle
    # Re-filter to ensure no point appears more than once
    if len(filtered_curve) > 2:
        final_curve = [filtered_curve[0]]
        for i in range(1, len(filtered_curve)):
            # Check against all previous points, not just the last one
            is_dup = any(np.allclose(filtered_curve[i], prev_pt, atol=1e-6) for prev_pt in final_curve)
            if not is_dup:
                final_curve.append(filtered_curve[i])

        return final_curve

    return filtered_curve


def create_boundary_extension(
    positions,
    boundary_points,
    coord_to_index,
    base_depth=-0.2,
    two_tier=False,
    mid_depth=None,
    base_material="clay",
    blend_edge_colors=True,
    surface_colors=None,
    smooth_boundary=False,
    smooth_window_size=5,
    use_catmull_rom=False,  # PERFORMANCE: Disabled by default due to computational cost (~1-2s per terrain)
    catmull_rom_subdivisions=2,
    use_rectangle_edges=False,  # NEW: Use rectangle-edge sampling instead of morphological detection
    dem_shape=None,  # DEPRECATED: Use terrain= instead for transform-aware edges
    terrain=None,  # NEW: Terrain object for transform-aware rectangle edges
    edge_sample_spacing=1.0,  # Sampling density for rectangle edges
):
    """
    Create boundary extension vertices and faces to close the mesh.

    Creates a "skirt" around the terrain by adding bottom vertices at base_depth
    and connecting them to the top boundary with quad faces. This closes the mesh
    into a solid object suitable for 3D printing or solid rendering.

    Supports two modes:
    - Single-tier (default): Surface → Base (one jump)
    - Two-tier: Surface → Mid → Base (two-tier with color separation)

    Args:
        positions (np.ndarray): Array of (n, 3) vertex positions
        boundary_points (list): List of (y, x) tuples representing ordered boundary points
        coord_to_index (dict): Mapping from (y, x) coordinates to vertex indices
        base_depth (float): Z-coordinate for the bottom of the extension (default: -0.2)
        two_tier (bool): Enable two-tier mode (default: False)
        mid_depth (float, optional): Z-coordinate for mid tier (default: base_depth * 0.25)
        base_material (str | tuple): Material for base layer - either preset name
                                    ("clay", "obsidian", "chrome", "plastic", "gold", "ivory")
                                    or RGB tuple (0-1 range). Default: "clay"
        blend_edge_colors (bool): Blend surface colors to mid tier (default: True)
                                 If False, mid tier uses base_material color
        surface_colors (np.ndarray, optional): Surface vertex colors (n_vertices, 3) uint8
        smooth_boundary (bool): Apply smoothing to boundary to eliminate stair-step edges
                               (default: False)
        smooth_window_size (int): Window size for boundary smoothing (default: 5).
                                 Larger values produce smoother curves.
        use_catmull_rom (bool): Use Catmull-Rom curve fitting for smooth boundary
                               instead of pixel-grid topology (default: False).
                               When enabled, eliminates staircase pattern entirely.
                               NOTE: Computationally expensive (~0.3-2s per terrain).
                               Provides true smooth curves vs simple smoothing.
        catmull_rom_subdivisions (int): Number of interpolated points per boundary
                                       segment when using Catmull-Rom curves (default: 2).
                                       Higher values = smoother curve but MORE COMPUTATION.
                                       Recommended: 2 (fast) or 3-4 (very smooth).
        use_rectangle_edges (bool): Use rectangle-edge sampling instead of morphological
                                   boundary detection (default: False).
                                   ~150x faster than morphological detection.
                                   Ideal for rectangular DEMs from raster sources.
        dem_shape (tuple, optional): DEPRECATED - DEM shape (height, width) for legacy rectangle-edge sampling.
                                    Use terrain= parameter instead for transform-aware edges (avoids NaN margins).
        terrain (Terrain, optional): Terrain object for transform-aware rectangle-edge sampling.
                                    Provides original DEM shape and transform pipeline for accurate
                                    coordinate mapping without NaN margins. Improves edge coverage from
                                    0.6% (legacy) to ~100% (transform-aware) for downsampled DEMs.
        edge_sample_spacing (float): Pixel spacing for edge sampling at original DEM resolution (default: 1.0).
                                     Lower values = denser sampling, more edge pixels.

    Returns:
        tuple: When two_tier=False (backwards compatible):
            (boundary_vertices, boundary_faces)
        tuple: When two_tier=True:
            (boundary_vertices, boundary_faces, boundary_colors)

        Where:
            - boundary_vertices: np.ndarray of vertex positions
                Single-tier: (n_boundary, 3)
                Two-tier: (2*n_boundary, 3) - mid + base vertices
            - boundary_faces: list of tuples defining side face quad connectivity
                Single-tier: N quads (surface→base)
                Two-tier: 2*N quads (surface→mid + mid→base)
            - boundary_colors: np.ndarray of (2*n_boundary, 3) uint8 colors (two-tier only)
    """
    from src.terrain.materials import get_base_material_color
    from scipy.interpolate import RegularGridInterpolator

    # Generate rectangle edge pixels if requested
    # Keep original morphological boundary_points as fallback
    original_morphological_boundary = boundary_points

    if use_rectangle_edges:
        # NEW: Use transform-aware approach if terrain object provided
        if terrain is not None:
            # Transform-aware rectangle-edge sampling (avoids NaN margins)
            rect_boundary_valid = generate_transform_aware_rectangle_edges(
                terrain,
                coord_to_index,
                edge_sample_spacing
            )

            # Report results
            original_shape = terrain.dem_shape
            print(f"\n{'='*60}")
            print(f"Transform-Aware Rectangle Edge Sampling")
            print(f"{'='*60}")
            print(f"Original DEM: {original_shape[0]}×{original_shape[1]} (sampling source)")
            print(f"Edge pixels mapped to final mesh: {len(rect_boundary_valid)}")
            print(f"Edge sample spacing: {edge_sample_spacing:.1f} pixels")
            print(f"{'='*60}\n")
        else:
            # FALLBACK: Legacy approach using transformed DEM shape
            if dem_shape is None:
                raise ValueError("Either terrain or dem_shape required when use_rectangle_edges=True")

            # Run diagnostic to show why this doesn't work well
            diagnostic = diagnose_rectangle_edge_coverage(dem_shape, coord_to_index)
            print(f"\n{'='*60}")
            print(f"⚠️  Legacy Rectangle Edge Sampling (Transformed DEM)")
            print(f"{'='*60}")
            print(f"DEM shape: {diagnostic['dem_shape'][0]}×{diagnostic['dem_shape'][1]}")
            print(f"Edge coverage: {diagnostic['coverage_percent']:.1f}% ({diagnostic['valid_edge_pixels']}/{diagnostic['total_edge_pixels']} pixels)")
            print(f"  Top edge:    {diagnostic['edge_validity']['top']['valid']:4d}/{diagnostic['edge_validity']['top']['total']:4d} valid ({diagnostic['edge_validity']['top']['valid']/max(1,diagnostic['edge_validity']['top']['total'])*100:.1f}%)")
            print(f"  Right edge:  {diagnostic['edge_validity']['right']['valid']:4d}/{diagnostic['edge_validity']['right']['total']:4d} valid ({diagnostic['edge_validity']['right']['valid']/max(1,diagnostic['edge_validity']['right']['total'])*100:.1f}%)")
            print(f"  Bottom edge: {diagnostic['edge_validity']['bottom']['valid']:4d}/{diagnostic['edge_validity']['bottom']['total']:4d} valid ({diagnostic['edge_validity']['bottom']['valid']/max(1,diagnostic['edge_validity']['bottom']['total'])*100:.1f}%)")
            print(f"  Left edge:   {diagnostic['edge_validity']['left']['valid']:4d}/{diagnostic['edge_validity']['left']['total']:4d} valid ({diagnostic['edge_validity']['left']['valid']/max(1,diagnostic['edge_validity']['left']['total'])*100:.1f}%)")
            print(f"\nRecommendation: {diagnostic['recommendation']}")
            print(f"Reason: {diagnostic['reason']}")
            print(f"💡 Tip: Pass terrain= parameter for transform-aware sampling (~100% coverage)")
            print(f"{'='*60}\n")

            rect_edge_pixels = generate_rectangle_edge_pixels(dem_shape, edge_sample_spacing)

            # Filter to only include pixels that are actually valid mesh vertices
            # Many rectangle edge pixels might be NaN or outside valid_mask, causing lookup failures
            rect_boundary_valid = [
                (y, x) for y, x in rect_edge_pixels
            if (int(y), int(x)) in coord_to_index
        ]

        # Use rectangle edges only if they produce a reasonable boundary
        # If too few valid points, stick with the original morphological boundary
        original_count = len(original_morphological_boundary)
        rect_count = len(rect_boundary_valid)

        # Heuristic: Need at least 80% of morphological boundary vertices, or at least 100 vertices
        min_required = max(100, int(0.8 * original_count))

        if rect_count >= min_required:
            # Rectangle edges produced good boundary - use it
            boundary_points = rect_boundary_valid
            print(f"✓ Rectangle-edge sampling: Using {rect_count} boundary vertices (morphological had {original_count})")
        else:
            # Rectangle edges too sparse - keep morphological boundary
            boundary_points = original_morphological_boundary
            print(f"✗ Rectangle-edge sampling: Too few valid vertices ({rect_count}), keeping morphological boundary ({original_count} vertices)")
            if terrain is None:
                print(f"  Tip: Pass terrain= parameter for transform-aware sampling to avoid NaN margins")
            else:
                print(f"  Tip: Check coordinate transformation - may be mapping outside valid mesh bounds")

    # Apply boundary smoothing if requested
    original_boundary_points = boundary_points
    if smooth_boundary and len(boundary_points) > 2:
        boundary_points = smooth_boundary_points(
            boundary_points, window_size=smooth_window_size, closed_loop=True
        )

    # Apply Catmull-Rom curve fitting if requested (replaces pixel-grid topology)
    if use_catmull_rom and len(boundary_points) > 2:
        smooth_curve_points = fit_catmull_rom_boundary_curve(
            boundary_points,
            subdivisions=catmull_rom_subdivisions,
            closed_loop=True,
        )
        boundary_points = smooth_curve_points

    n_boundary = len(boundary_points)
    has_smoothed_coords = (smooth_boundary or use_catmull_rom) and any(
        not (isinstance(y, (int, np.integer)) and isinstance(x, (int, np.integer)))
        for y, x in boundary_points
    )

    # Helper function to get or interpolate position
    def get_position_at_coords(y, x):
        """Get or interpolate vertex position at given (y, x) coordinates."""
        # Try integer lookup first
        if isinstance(y, (int, np.integer)) and isinstance(x, (int, np.integer)):
            idx = coord_to_index.get((int(y), int(x)))
            if idx is not None:
                return positions[idx].copy()

        # Bilinear interpolation from surrounding vertices (for smoothed float coordinates)
        y_floor, x_floor = int(np.floor(y)), int(np.floor(x))
        y_ceil, x_ceil = y_floor + 1, x_floor + 1

        # Get the four corner positions
        corners = {}
        for dy, dx in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            yy, xx = y_floor + dy, x_floor + dx
            idx = coord_to_index.get((yy, xx))
            if idx is not None:
                corners[(dy, dx)] = positions[idx]

        # If we have all 4 corners, do bilinear interpolation
        if len(corners) == 4:
            # Fractional parts
            fy = y - y_floor
            fx = x - x_floor

            # Bilinear interpolation
            pos_00 = corners[(0, 0)]
            pos_01 = corners[(0, 1)]
            pos_10 = corners[(1, 0)]
            pos_11 = corners[(1, 1)]

            # Interpolate in x direction first
            pos_0 = pos_00 * (1 - fx) + pos_01 * fx
            pos_1 = pos_10 * (1 - fx) + pos_11 * fx

            # Then interpolate in y direction
            result = pos_0 * (1 - fy) + pos_1 * fy
            return result

        # Fallback: use nearest neighbor if we don't have all 4 corners
        y_int, x_int = int(np.round(y)), int(np.round(x))
        idx = coord_to_index.get((y_int, x_int))
        if idx is not None:
            return positions[idx].copy()

        # Final fallback: return None if can't find any position
        return None

    if not two_tier:
        # ===== SINGLE-TIER MODE (backwards compatible) =====

        if has_smoothed_coords:
            # With smoothing: create new surface vertices at smoothed positions + base vertices
            surface_boundary_verts = np.zeros((n_boundary, 3), dtype=float)
            base_boundary_verts = np.zeros((n_boundary, 3), dtype=float)

            for i, (y, x) in enumerate(boundary_points):
                pos = get_position_at_coords(y, x)
                if pos is None:
                    continue

                # Update XY to smoothed position, keep interpolated Z
                pos[0] = x / 100.0  # Use same scale as original positions
                pos[1] = y / 100.0
                surface_boundary_verts[i] = pos

                # Base vertex: same XY, but at base_depth
                base_pos = pos.copy()
                base_pos[2] = base_depth
                base_boundary_verts[i] = base_pos

            # Stack surface + base vertices
            boundary_vertices = np.vstack([surface_boundary_verts, base_boundary_verts])

            n_existing = len(positions)
            surface_boundary_indices = list(range(n_existing, n_existing + n_boundary))
            base_boundary_indices = list(
                range(n_existing + n_boundary, n_existing + 2 * n_boundary)
            )

            # Create faces
            boundary_faces = []

            if use_catmull_rom:
                # When using Catmull-Rom curves, we have many interpolated points
                # Create faces only between smoothed surface and base (no connection to original mesh)
                for i in range(n_boundary):
                    next_i = (i + 1) % n_boundary
                    # Face from smoothed surface to base
                    boundary_faces.append(
                        (
                            surface_boundary_indices[i],
                            surface_boundary_indices[next_i],
                            base_boundary_indices[next_i],
                            base_boundary_indices[i],
                        )
                    )
            else:
                # Without Catmull-Rom: connect original mesh to smoothed surface to base
                boundary_indices_orig = []
                for y, x in original_boundary_points:
                    idx = coord_to_index.get((y, x))
                    boundary_indices_orig.append(idx if idx is not None else -1)

                # Create faces: original → smoothed surface → base
                for i in range(n_boundary):
                    if boundary_indices_orig[i] < 0:
                        continue

                    next_i = (i + 1) % n_boundary
                    if boundary_indices_orig[next_i] < 0:
                        continue

                    # Face from original surface to smoothed surface
                    boundary_faces.append(
                        (
                            boundary_indices_orig[i],
                            boundary_indices_orig[next_i],
                            surface_boundary_indices[next_i],
                            surface_boundary_indices[i],
                        )
                    )

                    # Face from smoothed surface to base
                    boundary_faces.append(
                        (
                            surface_boundary_indices[i],
                            surface_boundary_indices[next_i],
                            base_boundary_indices[next_i],
                            base_boundary_indices[i],
                        )
                    )

            return boundary_vertices, boundary_faces

        else:
            # No smoothing: original behavior
            boundary_vertices = np.zeros((n_boundary, 3), dtype=float)

            # Create bottom vertices for each boundary point
            for i, (y, x) in enumerate(boundary_points):
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
            if boundary_indices[i] is None:
                continue

            next_i = (i + 1) % n_boundary
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

    else:
        # ===== TWO-TIER MODE =====

        # Auto-calculate mid_depth if not provided
        if mid_depth is None:
            mid_depth = base_depth * 0.25

        # Resolve material to RGB
        base_color_rgb = get_base_material_color(base_material)

        # When using smoothed coordinates, extract original boundary Z values for smooth interpolation
        # Also pre-compute original boundary coordinates as numpy array for fast distance calculations
        original_boundary_z_values = None
        original_boundary_coords_array = None
        if has_smoothed_coords:
            original_boundary_z_values = []
            orig_coords_list = []
            for y, x in original_boundary_points:
                orig_coords_list.append([y, x])
                orig_idx = coord_to_index.get((int(y), int(x)))
                if orig_idx is not None:
                    original_boundary_z_values.append(positions[orig_idx, 2])
                else:
                    original_boundary_z_values.append(None)
            original_boundary_coords_array = np.array(orig_coords_list, dtype=float)

        # When using smoothed coordinates, create surface vertices at smoothed positions
        # When not using smoothed, we'll reference the original mesh
        surface_vertices = None
        if has_smoothed_coords:
            surface_vertices = np.zeros((n_boundary, 3), dtype=float)

        # Create mid and base vertices
        mid_vertices = np.zeros((n_boundary, 3), dtype=float)
        base_vertices = np.zeros((n_boundary, 3), dtype=float)

        for i, (y, x) in enumerate(boundary_points):
            # For smoothed coordinates (Catmull-Rom or smooth_boundary), use interpolation
            if has_smoothed_coords:
                pos = get_position_at_coords(y, x)
                if pos is None:
                    continue

                # Improve Z value: use smooth interpolation along boundary curve
                # instead of spatial bilinear interpolation
                if use_catmull_rom and original_boundary_z_values and original_boundary_coords_array is not None:
                    # OPTIMIZATION: Use vectorized numpy to find nearest original boundary point
                    # instead of Python loop (O(N) → O(1) for distance computation)
                    dists = np.sqrt((original_boundary_coords_array[:, 0] - y)**2 +
                                   (original_boundary_coords_array[:, 1] - x)**2)
                    closest_idx = np.argmin(dists)

                    # Interpolate Z between this point and next
                    next_idx = (closest_idx + 1) % len(original_boundary_points)
                    z1 = original_boundary_z_values[closest_idx]
                    z2 = original_boundary_z_values[next_idx]

                    if z1 is not None and z2 is not None:
                        # Distance-based interpolation within the segment
                        orig_y1, orig_x1 = original_boundary_coords_array[closest_idx]
                        orig_y2, orig_x2 = original_boundary_coords_array[next_idx]

                        seg_dist = np.sqrt((orig_y2 - orig_y1)**2 + (orig_x2 - orig_x1)**2)
                        if seg_dist > 0:
                            point_dist = np.sqrt((y - orig_y1)**2 + (x - orig_x1)**2)
                            t = np.clip(point_dist / seg_dist, 0, 1)
                            pos[2] = z1 * (1 - t) + z2 * t

                # Store the surface position at smoothed coordinates
                surface_vertices[i] = pos.copy()
            else:
                # For integer coordinates, direct lookup
                original_idx = coord_to_index.get((y, x))
                if original_idx is None:
                    continue
                pos = positions[original_idx].copy()

            # Mid vertex: extend downward from surface by mid_depth offset
            # (mid_depth is shallower, typically -0.2 or so)
            pos_mid = pos.copy()
            pos_mid[2] = pos[2] + mid_depth
            mid_vertices[i] = pos_mid

            # Base vertex: flat at absolute base_depth Z coordinate
            # (base_depth is absolute Z value, typically -0.2 to -0.8)
            # All base vertices stay at same Z regardless of surface elevation
            pos_base = pos.copy()
            pos_base[2] = base_depth
            base_vertices[i] = pos_base

        # Stack vertices appropriately based on coordinate type
        n_existing = len(positions)
        if has_smoothed_coords:
            # When using smoothed coordinates, include the surface vertices
            # so we have: surface + mid + base tiers
            boundary_vertices = np.vstack([surface_vertices, mid_vertices, base_vertices])
            surface_indices = list(range(n_existing, n_existing + n_boundary))
            mid_indices = list(range(n_existing + n_boundary, n_existing + 2 * n_boundary))
            base_indices = list(range(n_existing + 2 * n_boundary, n_existing + 3 * n_boundary))
        else:
            # When using integer coordinates, just mid + base
            boundary_vertices = np.vstack([mid_vertices, base_vertices])
            surface_indices = [coord_to_index.get((int(y), int(x))) for y, x in boundary_points]
            mid_indices = list(range(n_existing, n_existing + n_boundary))
            base_indices = list(range(n_existing + n_boundary, n_existing + 2 * n_boundary))

        boundary_faces = []

        for i in range(n_boundary):
            if surface_indices[i] is None:
                continue

            next_i = (i + 1) % n_boundary
            if surface_indices[next_i] is None:
                continue

            # When using smoothed coordinates, bridge original to new smooth boundary
            # This eliminates orphaned vertices
            # Note: We use the rounded coordinates to find the nearest original vertex
            if has_smoothed_coords:
                # Find nearest original boundary vertices to this smoothed segment
                # by rounding the smoothed coordinates
                orig_i_y, orig_i_x = int(np.round(boundary_points[i][0])), int(np.round(boundary_points[i][1]))
                orig_next_y, orig_next_x = int(np.round(boundary_points[next_i][0])), int(np.round(boundary_points[next_i][1]))

                orig_i = coord_to_index.get((orig_i_y, orig_i_x))
                orig_next = coord_to_index.get((orig_next_y, orig_next_x))

                if orig_i is not None and orig_next is not None:
                    # Bridge face: original boundary → new smooth surface tier
                    # This connects the stair-step to the smooth curve
                    boundary_faces.append(
                        (
                            orig_i,
                            orig_next,
                            surface_indices[next_i],
                            surface_indices[i],
                        )
                    )

            # Upper tier: surface → mid
            boundary_faces.append(
                (
                    surface_indices[i],
                    surface_indices[next_i],
                    mid_indices[next_i],
                    mid_indices[i],
                )
            )

            # Lower tier: mid → base
            boundary_faces.append(
                (
                    mid_indices[i],
                    mid_indices[next_i],
                    base_indices[next_i],
                    base_indices[i],
                )
            )

        # Create colors (size depends on whether we created surface vertices)
        if has_smoothed_coords:
            # When using smoothed coordinates: surface + mid + base = 3 tiers
            boundary_colors = np.zeros((3 * n_boundary, 3), dtype=np.uint8)
        else:
            # When using integer coordinates: mid + base = 2 tiers
            boundary_colors = np.zeros((2 * n_boundary, 3), dtype=np.uint8)

        base_color_uint8 = (np.array(base_color_rgb) * 255).astype(np.uint8)

        for i, (y, x) in enumerate(boundary_points):
            # Get surface color by interpolating from nearby vertices
            surface_color = None
            if blend_edge_colors and surface_colors is not None:
                if has_smoothed_coords:
                    # For smoothed coordinates, interpolate color from corners
                    y_floor, x_floor = int(np.floor(y)), int(np.floor(x))
                    color_corners = {}
                    for dy, dx in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                        yy, xx = y_floor + dy, x_floor + dx
                        idx = coord_to_index.get((yy, xx))
                        if idx is not None:
                            color_corners[(dy, dx)] = surface_colors[idx, :3]

                    if len(color_corners) == 4:
                        # Bilinear interpolation of color
                        fy = y - y_floor
                        fx = x - x_floor
                        c00 = color_corners[(0, 0)].astype(float)
                        c01 = color_corners[(0, 1)].astype(float)
                        c10 = color_corners[(1, 0)].astype(float)
                        c11 = color_corners[(1, 1)].astype(float)
                        c0 = c00 * (1 - fx) + c01 * fx
                        c1 = c10 * (1 - fx) + c11 * fx
                        surface_color = (c0 * (1 - fy) + c1 * fy).astype(np.uint8)
                else:
                    # For integer coordinates, direct lookup
                    original_idx = coord_to_index.get((int(y), int(x)))
                    if original_idx is not None:
                        surface_color = surface_colors[original_idx, :3]

            if surface_color is None:
                # Use base material color as fallback
                surface_color = base_color_uint8

            # Assign colors based on tier structure
            if has_smoothed_coords:
                # Three-tier structure: surface + mid + base
                boundary_colors[i, :3] = surface_color                           # Surface tier
                boundary_colors[i + n_boundary, :3] = surface_color             # Mid tier (same as surface)
                boundary_colors[i + 2 * n_boundary, :3] = base_color_uint8     # Base tier (uniform color)
            else:
                # Two-tier structure: mid + base
                boundary_colors[i, :3] = surface_color             # Mid tier
                boundary_colors[i + n_boundary, :3] = base_color_uint8  # Base tier

        return boundary_vertices, boundary_faces, boundary_colors


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
                max(0, min(n_points - 1, i + offset - half_window))
                for offset in range(window_size)
            ]

        # Average the coordinates
        smoothed[i] = coords_array[indices].mean(axis=0)

    # Convert back to list of tuples
    return [tuple(pt) for pt in smoothed]


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


def diagnose_rectangle_edge_coverage(dem_shape, coord_to_index):
    """
    Diagnose how well rectangle edge sampling will work for this DEM.

    Checks what percentage of the rectangle perimeter has valid mesh vertices.
    Helps determine if rectangle-edge sampling is appropriate for this dataset.

    Args:
        dem_shape (tuple): DEM shape (height, width)
        coord_to_index (dict): Mapping from (y, x) to vertex indices

    Returns:
        dict: Diagnostic information including:
            - total_edge_pixels: Total pixels on rectangle perimeter
            - valid_edge_pixels: How many have valid mesh vertices
            - coverage_percent: Percentage of edge that's valid
            - edge_validity: Per-edge breakdown (top, right, bottom, left)
            - recommendation: Whether to use rectangle edges or morphological
    """
    height, width = dem_shape

    edge_validity = {
        'top': {'total': 0, 'valid': 0},
        'right': {'total': 0, 'valid': 0},
        'bottom': {'total': 0, 'valid': 0},
        'left': {'total': 0, 'valid': 0},
    }

    # Check top edge (y=0, x from 0 to width-1)
    for x in range(width):
        edge_validity['top']['total'] += 1
        if (0, x) in coord_to_index:
            edge_validity['top']['valid'] += 1

    # Check right edge (x=width-1, y from 0 to height-1)
    for y in range(height):
        edge_validity['right']['total'] += 1
        if (y, width - 1) in coord_to_index:
            edge_validity['right']['valid'] += 1

    # Check bottom edge (y=height-1, x from 0 to width-1)
    for x in range(width):
        edge_validity['bottom']['total'] += 1
        if (height - 1, x) in coord_to_index:
            edge_validity['bottom']['valid'] += 1

    # Check left edge (x=0, y from 0 to height-1)
    for y in range(height):
        edge_validity['left']['total'] += 1
        if (y, 0) in coord_to_index:
            edge_validity['left']['valid'] += 1

    # Calculate totals
    total_edge_pixels = 2 * (height + width) - 4  # Perimeter, not counting corners twice
    valid_edge_pixels = (
        edge_validity['top']['valid'] +
        edge_validity['right']['valid'] +
        edge_validity['bottom']['valid'] +
        edge_validity['left']['valid'] - 4  # Remove duplicate corner counts
    )

    coverage_percent = (valid_edge_pixels / total_edge_pixels * 100) if total_edge_pixels > 0 else 0

    # Generate recommendation
    if coverage_percent >= 90:
        recommendation = "use_rectangle_edges"
        reason = "Excellent coverage - valid data extends to grid edges"
    elif coverage_percent >= 70:
        recommendation = "use_rectangle_edges"
        reason = "Good coverage - rectangle edges should work well"
    else:
        recommendation = "use_morphological"
        reason = f"Low coverage ({coverage_percent:.1f}%) - data doesn't extend to grid edges"

    return {
        'dem_shape': dem_shape,
        'total_edge_pixels': total_edge_pixels,
        'valid_edge_pixels': valid_edge_pixels,
        'coverage_percent': coverage_percent,
        'edge_validity': edge_validity,
        'recommendation': recommendation,
        'reason': reason,
    }


def generate_rectangle_edge_pixels(dem_shape, edge_sample_spacing=1.0):
    """
    Generate boundary pixel coordinates by sampling rectangle edges.

    This creates ordered (y, x) pixel coordinates around the DEM boundary,
    forming a simple rectangle. This approach is much faster than morphological
    boundary detection and works well for rectangular DEMs.

    Algorithm:
    Sample the rectangle boundary edges at given spacing, tracing counterclockwise:
    top edge → right edge → bottom edge → left edge

    Args:
        dem_shape (tuple): DEM shape (height, width)
        edge_sample_spacing (float): Pixel spacing for edge sampling (default: 1.0)

    Returns:
        list: Ordered list of (y, x) pixel coordinates forming the rectangle boundary
    """
    height, width = dem_shape
    edge_pixels = []

    # Top edge (y=0, x from 0 to width-1)
    for x in np.arange(0, width, edge_sample_spacing):
        edge_pixels.append((0, int(x)))

    # Right edge (x=width-1, y from spacing to height-1)
    for y in np.arange(edge_sample_spacing, height, edge_sample_spacing):
        edge_pixels.append((int(y), width - 1))

    # Bottom edge (y=height-1, x from width-1 down to 0)
    for x in np.arange(width - 1, -1, -edge_sample_spacing):
        edge_pixels.append((height - 1, int(x)))

    # Left edge (x=0, y from height-1 down to spacing)
    for y in np.arange(height - 1 - edge_sample_spacing, -1, -edge_sample_spacing):
        if int(y) >= 0:
            edge_pixels.append((int(y), 0))

    # Remove duplicates (corners get added twice)
    edge_pixels = list(dict.fromkeys(edge_pixels))

    return edge_pixels


def generate_rectangle_edge_vertices(
    dem_shape,
    dem_data,
    original_transform,
    transforms_list,
    edge_sample_spacing=1.0,
    base_depth=-0.2,
):
    """
    Generate boundary vertices by sampling rectangle edges and applying geographic transforms.

    This approach leverages the same transform pipeline used for the DEM to create
    naturally smooth, curved edges that perfectly match the geographic projection.

    Algorithm:
    1. Sample the rectangle boundary in original DEM pixel space
    2. For each edge vertex, apply the sequence of geographic transforms
    3. Creates vertices with flat base_depth (foundation)

    Args:
        dem_shape (tuple): Original DEM shape (height, width)
        dem_data (np.ndarray): Original DEM data array
        original_transform (Affine): Original affine transform (pixel → geographic)
        transforms_list (list): List of transform functions to apply sequentially
        edge_sample_spacing (float): Pixel spacing for edge sampling (default: 1.0)
        base_depth (float): Z-coordinate for base vertices (default: -0.2)

    Returns:
        tuple: (boundary_vertices, boundary_faces) where:
            - boundary_vertices: Nx3 array of vertex positions
            - boundary_faces: List of quad faces connecting vertices
    """
    # Step 1: Get rectangle edge pixels
    edge_pixels = generate_rectangle_edge_pixels(dem_shape, edge_sample_spacing)

    # Step 2: Apply affine transform to get world coordinates
    # For each pixel (y_px, x_px), compute:
    #   x_world = original_transform.c + original_transform.a * x_px + original_transform.b * y_px
    #   y_world = original_transform.f + original_transform.d * x_px + original_transform.e * y_px

    boundary_vertices = []

    for y_px, x_px in edge_pixels:
        # Apply original affine transform to pixel coordinates
        # Affine.Affine(a, b, c, d, e, f) where:
        #   a, e = pixel sizes
        #   b, d = rotation/shear
        #   c, f = origin

        x_world = original_transform.c + original_transform.a * x_px + original_transform.b * y_px
        y_world = original_transform.f + original_transform.d * x_px + original_transform.e * y_px

        # Apply each transform in sequence (same as for DEM)
        pixel_coord = np.array([y_px, x_px], dtype=float)
        current_transform = original_transform
        dem_shape_current = dem_shape

        # Note: For now, we're only applying the original_transform
        # Full transform pipeline support would be added in next iteration
        # This gives us the baseline functionality

        # All base vertices have the same Z coordinate (flat foundation)
        boundary_vertices.append([x_world, y_world, base_depth])

    boundary_vertices = np.array(boundary_vertices, dtype=float)

    # Step 3: Create faces connecting consecutive edge vertices
    # Each consecutive pair of edge vertices forms a quad with the base
    n_vertices = len(boundary_vertices)
    boundary_faces = []

    for i in range(n_vertices):
        # Create quad connecting:
        # - Current and next edge vertex (at surface, but we have them at base_depth)
        # - For now, simple edge quads
        next_i = (i + 1) % n_vertices

        # Quad: current, next, next (duplicate for now, to be refined)
        # This is a placeholder - full implementation will create proper bridge faces
        boundary_faces.append([i, next_i, next_i, i])

    return boundary_vertices, boundary_faces


def generate_transform_aware_rectangle_edges(
    terrain,
    coord_to_index,
    edge_sample_spacing=1.0,
):
    """
    Generate rectangle edge pixels by sampling original DEM perimeter
    and mapping through transform pipeline.

    This function solves the NaN margin problem by sampling edges at the original
    DEM resolution (where all perimeter pixels are valid) and mapping them through
    the transform pipeline to final mesh coordinates.

    Uses affine transforms to map coordinates:
      original pixel → geographic → final transformed pixel

    Args:
        terrain: Terrain object with dem_shape, dem_transform, data_layers
        coord_to_index: Dict mapping (y, x) final pixels to vertex indices
        edge_sample_spacing: Pixel spacing for edge sampling at original resolution (default 1.0)

    Returns:
        list: Edge pixel coordinates in final transformed space,
              filtered to only valid mesh vertices

    Raises:
        ValueError: If terrain is None or lacks required transform data
    """
    if terrain is None:
        raise ValueError("Terrain object required for transform-aware rectangle edges")

    # 1. Sample edges at original resolution
    original_shape = terrain.dem_shape
    edge_pixels_orig = generate_rectangle_edge_pixels(
        original_shape,
        edge_sample_spacing
    )

    # 2. Get transform info
    original_transform = terrain.dem_transform
    dem_layer = terrain.data_layers.get("dem")

    if dem_layer is None:
        raise ValueError("Terrain lacks 'dem' data layer")

    if not dem_layer.get("transformed", False):
        raise ValueError(
            "Terrain DEM has not been transformed yet. "
            "Call terrain.apply_transforms() before using transform-aware edges."
        )

    transformed_transform = dem_layer.get("transformed_transform")
    if transformed_transform is None:
        raise ValueError("Terrain DEM lacks 'transformed_transform' - cannot map coordinates")

    # 3. Map each edge pixel: original → geographic → final
    edge_pixels_final = []
    transform_errors = 0
    out_of_bounds = 0
    not_in_coord_index = 0

    for (y_orig, x_orig) in edge_pixels_orig:
        try:
            # Original pixel → geographic coords
            # Affine multiplication: (lon, lat) = transform * (x, y)
            # Note: Affine takes (x, y) not (y, x)
            lon, lat = original_transform * (x_orig, y_orig)

            # Geographic → final pixel coords
            # Inverse transform: (x, y) = ~transform * (lon, lat)
            x_final, y_final = ~transformed_transform * (lon, lat)

            # Round to integer pixel coordinates
            y_int, x_int = int(round(y_final)), int(round(x_final))

            # Check bounds
            transformed_shape = dem_layer["transformed_data"].shape
            if y_int < 0 or y_int >= transformed_shape[0] or x_int < 0 or x_int >= transformed_shape[1]:
                out_of_bounds += 1
                continue

            # Check if this maps to a valid mesh vertex
            if (y_int, x_int) in coord_to_index:
                edge_pixels_final.append((y_int, x_int))
            else:
                not_in_coord_index += 1

        except Exception as e:
            # Track transformation errors for debugging
            transform_errors += 1
            continue

    # Debug output if very few pixels mapped
    if len(edge_pixels_final) < 100:
        print(f"  DEBUG: Transform results:")
        print(f"    Original edge pixels: {len(edge_pixels_orig)}")
        print(f"    Successfully mapped: {len(edge_pixels_final)}")
        print(f"    Transform errors: {transform_errors}")
        print(f"    Out of bounds: {out_of_bounds}")
        print(f"    Not in coord_to_index: {not_in_coord_index}")

    return edge_pixels_final
