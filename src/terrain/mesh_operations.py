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
    base_depth=0.2,
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
    edge_sample_spacing=0.33,  # Sampling density for rectangle edges (0.33 = 3x denser, ~80K boundary vertices for smooth curves)
    boundary_winding="counter-clockwise",  # NEW: Boundary winding direction for correct face normals
    use_fractional_edges=False,  # NEW: Use fractional coords preserving projection curvature
    scale_factor=100.0,  # Scale factor used for mesh positions (for fractional edge X,Y computation)
    model_offset=None,  # Model centering offset [x, y, z] (for fractional edge X,Y computation)
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
        base_depth (float): Positive depth offset below minimum surface elevation (default: 0.2).
                           Creates a flat base plane at: min_surface_z - base_depth.
                           Positive values extend below surface, negative extend above.
        two_tier (bool): Enable two-tier mode (default: False)
        mid_depth (float, optional): Positive depth offset below surface for mid tier
                                    (default: base_depth * 0.25, typically 0.05).
                                    Positive values extend below surface, negative extend above.
        base_material (str | tuple): Material for base layer - either preset name
                                    ("clay", "obsidian", "chrome", "plastic", "gold", "ivory")
                                    or RGB tuple (0-1 range). Default: "clay"
        blend_edge_colors (bool): Blend surface colors to mid tier (default: True)
                                 If False, mid tier uses base_material color for sharp transition
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
        use_fractional_edges (bool): Use fractional coordinates that preserve projection curvature
                                    (default: False). When True, creates smooth curved edge by:
                                    1. Surface tier aligned with mesh boundary (bilinear interpolation, no gap)
                                    2. Mid tier at fractional X,Y positions with offset Z (smooth curve below surface)
                                    3. Base tier at fractional X,Y positions with flat Z (smooth curved base)
                                    This eliminates gaps while preserving smooth projection-aware edge curves.
                                    Requires terrain= parameter.

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
        # NEW: Use fractional coordinates to preserve projection curvature
        if use_fractional_edges and terrain is not None:
            # Fractional edge sampling - preserves curved boundary from projection
            rect_boundary_fractional = generate_transform_aware_rectangle_edges_fractional(
                terrain,
                edge_sample_spacing
            )

            # Report results
            original_shape = terrain.dem_shape
            print(f"\n{'='*60}")
            print(f"Transform-Aware Fractional Edge Sampling (Curved Boundary)")
            print(f"{'='*60}")
            print(f"Original DEM: {original_shape[0]}×{original_shape[1]} (sampling source)")
            print(f"Fractional edge vertices: {len(rect_boundary_fractional)}")
            print(f"Edge sample spacing: {edge_sample_spacing:.1f} pixels")
            print(f"NOTE: Fractional coordinates preserve projection curvature")
            print(f"{'='*60}\n")

            # Use fractional edges directly - they'll be processed by bilinear interpolation
            # No need to filter through coord_to_index since these are fractional coords
            rect_boundary_valid = rect_boundary_fractional

        # Use transform-aware INTEGER approach if terrain provided but not fractional
        elif terrain is not None:
            # Transform-aware rectangle-edge sampling (avoids NaN margins)
            rect_boundary_valid = generate_transform_aware_rectangle_edges(
                terrain,
                coord_to_index,
                edge_sample_spacing
            )

            # Report results
            original_shape = terrain.dem_shape
            print(f"\n{'='*60}")
            print(f"Transform-Aware Rectangle Edge Sampling (Integer)")
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
            # IMPORTANT: For rectangle edges, the points are ALREADY in order from generate_rectangle_edge_pixels()
            # which traces: top→right→bottom→left in a continuous loop
            # DON'T sort them - sorting with KD-tree nearest-neighbor breaks down on dense point clouds (82K+ points)
            # and can reduce the boundary from 82K points to just 10 points!
            print(f"✓ Rectangle-edge sampling: Using {rect_count} boundary vertices (morphological had {original_count})")

            # CRITICAL: After coordinate transformation, the natural rectangle order is destroyed!
            # First deduplicate, then re-sort spatially to form a closed loop
            print(f"  Deduplicating boundary points...")
            rect_boundary_unique = deduplicate_boundary_points(rect_boundary_valid)

            # For dense boundaries, use angular sorting (faster and more robust)
            # For sparse boundaries, use nearest-neighbor
            if len(rect_boundary_unique) >= 100:
                print(f"  Sorting {len(rect_boundary_unique)} points using angular method...")
                boundary_points = sort_boundary_points_angular(rect_boundary_unique)
            else:
                print(f"  Sorting {len(rect_boundary_unique)} points using nearest-neighbor...")
                boundary_points = sort_boundary_points(rect_boundary_unique)
            print(f"  ✓ Boundary sorted into continuous path")

            # DEBUG: Check spatial distribution of boundary points
            boundary_array = np.array(boundary_points)
            y_min, y_max = boundary_array[:, 0].min(), boundary_array[:, 0].max()
            x_min, x_max = boundary_array[:, 1].min(), boundary_array[:, 1].max()

            # Count points on each edge (with 5% margin)
            y_range = y_max - y_min
            x_range = x_max - x_min
            margin = 0.05

            top_count = np.sum(boundary_array[:, 0] <= y_min + margin * y_range)
            bottom_count = np.sum(boundary_array[:, 0] >= y_max - margin * y_range)
            left_count = np.sum(boundary_array[:, 1] <= x_min + margin * x_range)
            right_count = np.sum(boundary_array[:, 1] >= x_max - margin * x_range)

            print(f"  Boundary point distribution:")
            print(f"    Top edge (north):    {top_count:6d} points")
            print(f"    Bottom edge (south): {bottom_count:6d} points")
            print(f"    Left edge (west):    {left_count:6d} points")
            print(f"    Right edge (east):   {right_count:6d} points")

            # Check if distribution is severely uneven (any edge has < 5% of points)
            total_points = len(boundary_points)
            min_percent = min(top_count, bottom_count, left_count, right_count) / total_points * 100
            if min_percent < 5.0:
                print(f"  ⚠️  Warning: Uneven distribution detected (min={min_percent:.1f}%)")
                print(f"  Sparse edges may have lower visual quality")
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
    # Check if we have fractional coordinates that need bilinear interpolation
    # This can happen from: smooth_boundary, use_catmull_rom, OR use_fractional_edges
    has_smoothed_coords = (smooth_boundary or use_catmull_rom or use_fractional_edges) and any(
        not (isinstance(y, (int, np.integer)) and isinstance(x, (int, np.integer)))
        for y, x in boundary_points
    )

    # Helper function to get or interpolate position
    # Precompute mesh bounds for edge clamping (once, not per-call)
    mesh_bounds = None
    if coord_to_index:
        all_yx = list(coord_to_index.keys())
        mesh_y = [yx[0] for yx in all_yx]
        mesh_x = [yx[1] for yx in all_yx]
        mesh_bounds = (min(mesh_y), max(mesh_y), min(mesh_x), max(mesh_x))

    def get_position_at_coords(y, x):
        """Get or interpolate vertex position at given (y, x) coordinates.

        Handles edge coordinates that extend slightly beyond mesh bounds
        by clamping to valid range. This is essential for transform-aware
        rectangle edges where projection curvature causes boundary pixels
        to land outside the integer mesh grid.
        """
        # Clamp coordinates to mesh bounds to handle projection curvature
        # that causes edge pixels to extend slightly outside the mesh
        if mesh_bounds is not None:
            y_min, y_max, x_min, x_max = mesh_bounds
            # Clamp to valid interpolation range (one less than max for bilinear)
            y = np.clip(y, y_min, y_max - 0.001)
            x = np.clip(x, x_min, x_max - 0.001)

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
        missing_corners = []
        for dy, dx in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            yy, xx = y_floor + dy, x_floor + dx
            idx = coord_to_index.get((yy, xx))
            if idx is not None:
                corners[(dy, dx)] = positions[idx]
            else:
                missing_corners.append((yy, xx))

        # Fractional parts (used for all interpolation modes)
        fy = y - y_floor
        fx = x - x_floor

        # If we have all 4 corners, do bilinear interpolation
        if len(corners) == 4:
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

        # Partial corner interpolation - handle trapezoidal mesh edges
        # UTM projection can create meshes where edge pixels don't form complete rectangles
        if len(corners) >= 2:
            # Try to interpolate with available corners
            available = list(corners.values())

            # If we have top or bottom row complete, do linear interpolation
            if (0, 0) in corners and (0, 1) in corners:
                # Have top row - interpolate and extrapolate
                pos_0 = corners[(0, 0)] * (1 - fx) + corners[(0, 1)] * fx
                if (1, 0) in corners and (1, 1) in corners:
                    pos_1 = corners[(1, 0)] * (1 - fx) + corners[(1, 1)] * fx
                    return pos_0 * (1 - fy) + pos_1 * fy
                return pos_0  # Use top row only

            if (1, 0) in corners and (1, 1) in corners:
                # Have bottom row only - use it
                pos_1 = corners[(1, 0)] * (1 - fx) + corners[(1, 1)] * fx
                return pos_1

            # If we have left or right column complete, do linear interpolation
            if (0, 0) in corners and (1, 0) in corners:
                # Have left column - interpolate
                pos_left = corners[(0, 0)] * (1 - fy) + corners[(1, 0)] * fy
                if (0, 1) in corners and (1, 1) in corners:
                    pos_right = corners[(0, 1)] * (1 - fy) + corners[(1, 1)] * fy
                    return pos_left * (1 - fx) + pos_right * fx
                return pos_left  # Use left column only

            if (0, 1) in corners and (1, 1) in corners:
                # Have right column only - use it
                pos_right = corners[(0, 1)] * (1 - fy) + corners[(1, 1)] * fy
                return pos_right

            # Diagonal corners - average them
            return np.mean(available, axis=0)

        # If we have exactly 1 corner, use it
        if len(corners) == 1:
            return list(corners.values())[0].copy()

        # Fallback: use nearest neighbor if we don't have all 4 corners
        y_int, x_int = int(np.round(y)), int(np.round(x))
        # Clamp to mesh bounds
        if mesh_bounds is not None:
            y_min, y_max, x_min, x_max = mesh_bounds
            y_int = max(y_min, min(y_int, y_max))
            x_int = max(x_min, min(x_int, x_max))
        idx = coord_to_index.get((y_int, x_int))
        if idx is not None:
            return positions[idx].copy()

        # Expanding search for nearest valid vertex (handles trapezoidal meshes)
        # The mesh may have gaps at corners due to UTM projection
        if mesh_bounds is not None:
            y_min, y_max, x_min, x_max = mesh_bounds
            for radius in range(1, 50):  # Search up to 50 pixels away
                # Search in a square ring at this radius
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        if abs(dy) != radius and abs(dx) != radius:
                            continue  # Only check ring, not filled square
                        yy = y_int + dy
                        xx = x_int + dx
                        if y_min <= yy <= y_max and x_min <= xx <= x_max:
                            idx = coord_to_index.get((yy, xx))
                            if idx is not None:
                                return positions[idx].copy()

        # DEBUG: Log when we truly can't find a position
        if missing_corners:
            get_position_at_coords.missing_corner_samples.append({
                'y': y, 'x': x,
                'y_floor': y_floor, 'x_floor': x_floor,
                'missing': missing_corners,
                'n_corners': len(corners)
            })

        # Final fallback: return None if can't find any position
        return None

    # Initialize debug tracking
    get_position_at_coords.missing_corner_samples = []

    if not two_tier:
        # ===== SINGLE-TIER MODE (backwards compatible) =====

        # Calculate minimum surface elevation for base depth reference
        # Base vertices will be positioned at: min_z - base_depth
        min_surface_z = np.min(positions[:, 2])

        if has_smoothed_coords:
            # With smoothing: create new surface vertices at smoothed positions + base vertices
            surface_boundary_verts = np.zeros((n_boundary, 3), dtype=float)
            base_boundary_verts = np.zeros((n_boundary, 3), dtype=float)

            for i, (y, x) in enumerate(boundary_points):
                pos = get_position_at_coords(y, x)
                if pos is None:
                    continue

                # For fractional edges: compute X,Y directly from fractional coordinates
                # The bilinear interpolation correctly gets Z, but X,Y get clamped to mesh bounds
                # which causes stair-stepping. Use the true fractional coords for smooth edges.
                if use_fractional_edges and model_offset is not None:
                    pos[0] = x / scale_factor - model_offset[0]
                    pos[1] = y / scale_factor - model_offset[1]
                    # Z remains from bilinear interpolation (elevation data)

                surface_boundary_verts[i] = pos.copy()

                # Base vertex: same XY, flat plane below min surface
                # (base_depth is positive offset below min surface)
                base_pos = pos.copy()
                base_pos[2] = min_surface_z - base_depth
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

            if use_catmull_rom or use_fractional_edges:
                # When using Catmull-Rom curves or fractional edges, we have many interpolated points
                # that don't map to existing mesh vertices.
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

                # Copy position but set z to flat plane below min surface
                # (base_depth is positive offset below min surface)
                pos = positions[original_idx].copy()
                pos[2] = min_surface_z - base_depth
                boundary_vertices[i] = pos

            # Create side faces efficiently
            boundary_indices = [coord_to_index.get((y, x)) for y, x in boundary_points]
        base_indices = list(range(len(positions), len(positions) + len(boundary_points)))

        boundary_faces = []

        # DEBUG: Track face generation statistics
        faces_created = 0
        faces_skipped_none = 0
        faces_skipped_distance = 0

        for i in range(n_boundary):
            if boundary_indices[i] is None:
                faces_skipped_none += 1
                continue

            next_i = (i + 1) % n_boundary
            if boundary_indices[next_i] is None:
                faces_skipped_none += 1
                continue

            # Check if this is the wrap-around edge (last → first vertex)
            # For rectangle edges with angular sorting, check if gap is reasonable
            if i == n_boundary - 1 and use_rectangle_edges:
                # Get positions of last and first boundary points
                y_last, x_last = boundary_points[i]
                y_first, x_first = boundary_points[0]

                # Calculate wrap-around distance
                distance = np.sqrt((y_last - y_first)**2 + (x_last - x_first)**2)

                # Calculate median edge distance for comparison
                # Sample distances between consecutive boundary points to establish "normal" edge spacing
                sample_size = min(100, n_boundary - 1)
                sample_distances = []
                for j in range(sample_size):
                    y_curr, x_curr = boundary_points[j]
                    y_next, x_next = boundary_points[j + 1]
                    d = np.sqrt((y_next - y_curr)**2 + (x_next - x_curr)**2)
                    sample_distances.append(d)

                median_edge_distance = np.median(sample_distances)

                # Allow wrap-around only if it's within 10x the median edge distance
                # This prevents diagonal artifacts across the mesh
                threshold = max(median_edge_distance * 10.0, 50.0)

                if distance > threshold:
                    # Skip this wrap-around edge - gap is too large relative to normal edge spacing
                    faces_skipped_distance += 1
                    print(f"  ⚠️  Wrap-around face skipped: distance = {distance:.2f} > {threshold:.1f} (median edge = {median_edge_distance:.2f})")
                    continue
                elif distance > median_edge_distance * 2.0:
                    # Warn but still create the face (gap is large but acceptable)
                    print(f"  ℹ️  Wrap-around face: distance = {distance:.2f} pixels ({distance/median_edge_distance:.1f}x median, closing loop)")

            # Create quad connecting top boundary to bottom
            # Face winding must match boundary direction for correct normals
            if boundary_winding == "clockwise":
                # Clockwise boundary: reverse face winding for outward normals
                boundary_faces.append(
                    (
                        boundary_indices[i],
                        base_indices[i],
                        base_indices[next_i],
                        boundary_indices[next_i],
                    )
                )
            else:
                # Counter-clockwise boundary: standard winding
                boundary_faces.append(
                    (
                        boundary_indices[i],
                        boundary_indices[next_i],
                        base_indices[next_i],
                        base_indices[i],
                    )
                )
            faces_created += 1

        # DEBUG: Print face generation statistics
        print(f"\n{'='*60}")
        print(f"Boundary Face Generation (Single-Tier)")
        print(f"{'='*60}")
        print(f"Boundary winding: {boundary_winding}")
        print(f"Boundary vertices: {n_boundary}")
        print(f"Boundary indices (valid): {n_boundary - sum(1 for idx in boundary_indices if idx is None)}")
        print(f"Boundary indices (None): {sum(1 for idx in boundary_indices if idx is None)}")
        print(f"Faces created: {faces_created}")
        print(f"Faces skipped (None index): {faces_skipped_none}")
        print(f"Faces skipped (distance check): {faces_skipped_distance}")
        print(f"Total boundary faces: {len(boundary_faces)}")
        expected_faces = n_boundary  # 1 face per boundary segment
        coverage = faces_created / expected_faces * 100 if expected_faces > 0 else 0
        print(f"Expected faces (ideal): {expected_faces}")
        print(f"Coverage: {coverage:.1f}%")
        print(f"{'='*60}\n")

        return boundary_vertices, boundary_faces

    else:
        # ===== TWO-TIER MODE =====

        # Auto-calculate mid_depth if not provided
        # mid_depth is a positive offset below surface (e.g., 0.05)
        # base_depth is positive offset below min surface (e.g., 0.2)
        # Default: shallow tier at 25% of base depth distance
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

        # Calculate minimum surface elevation for base depth reference
        # Base vertices will be positioned at: min_z - base_depth
        min_surface_z = np.min(positions[:, 2])

        # Create mid and base vertices
        mid_vertices = np.zeros((n_boundary, 3), dtype=float)
        base_vertices = np.zeros((n_boundary, 3), dtype=float)

        # Track which boundary vertices were successfully initialized
        # (needed for smoothed coords where interpolation may fail)
        valid_boundary_vertex = [False] * n_boundary

        # DEBUG: Track interpolation failures by edge region
        interp_success = 0
        interp_fail_no_corners = 0
        interp_fail_fallback = 0
        failed_coords = []

        # Analyze boundary coordinate ranges
        if has_smoothed_coords and boundary_points:
            y_coords = [bp[0] for bp in boundary_points]
            x_coords = [bp[1] for bp in boundary_points]
            print(f"\n[DIAG] Boundary coordinate ranges:")
            print(f"  Y: min={min(y_coords):.2f}, max={max(y_coords):.2f}")
            print(f"  X: min={min(x_coords):.2f}, max={max(x_coords):.2f}")
            # Get mesh bounds from coord_to_index
            if coord_to_index:
                all_yx = list(coord_to_index.keys())
                mesh_y = [yx[0] for yx in all_yx]
                mesh_x = [yx[1] for yx in all_yx]
                print(f"  Mesh Y: min={min(mesh_y)}, max={max(mesh_y)}")
                print(f"  Mesh X: min={min(mesh_x)}, max={max(mesh_x)}")

        # Track position samples for diagnostics
        position_samples = []

        for i, (y, x) in enumerate(boundary_points):
            # For smoothed coordinates (Catmull-Rom or smooth_boundary), use interpolation
            if has_smoothed_coords:
                pos = get_position_at_coords(y, x)
                if pos is None:
                    interp_fail_no_corners += 1
                    if len(failed_coords) < 20:  # Limit debug output
                        failed_coords.append((y, x))
                    continue
                interp_success += 1
                valid_boundary_vertex[i] = True

                # Store bilinear-interpolated values for diagnostics (before fractional edge correction)
                bilinear_x = pos[0]
                bilinear_y = pos[1]

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

                # For fractional edges: DON'T adjust surface tier X,Y
                # Keep surface vertices aligned with mesh boundary (from bilinear interpolation)
                # This eliminates gaps - surface tier shares vertex positions with mesh edge
                # Mid and base tiers will use fractional X,Y for smooth curves

                # Sample positions for diagnostic output
                # Surface tier uses bilinear interpolation (aligned with mesh)
                if len(position_samples) < 80:
                    position_samples.append({
                        'i': i,
                        'y_in': y,
                        'x_in': x,
                        'bilinear_x': bilinear_x,
                        'bilinear_y': bilinear_y,
                        'x_out': pos[0],  # Surface tier (snapped to mesh)
                        'y_out': pos[1],
                        'z_out': pos[2],
                    })

                # Store the surface position
                surface_vertices[i] = pos.copy()
            else:
                # For integer coordinates, direct lookup
                original_idx = coord_to_index.get((y, x))
                if original_idx is None:
                    continue
                pos = positions[original_idx].copy()
                valid_boundary_vertex[i] = True

            # Mid vertex: extend downward from surface by mid_depth offset
            # (mid_depth is positive depth below surface, typically 0.05 to 0.2)
            pos_mid = pos.copy()
            pos_mid[2] = pos[2] - mid_depth

            # For fractional edges: mid tier uses fractional X,Y for smooth curve
            # (surface tier stays aligned with mesh, mid/base follow smooth boundary)
            if use_fractional_edges and model_offset is not None:
                pos_mid[0] = x / scale_factor - model_offset[0]
                pos_mid[1] = y / scale_factor - model_offset[1]

            mid_vertices[i] = pos_mid

            # Base vertex: flat plane below minimum surface elevation
            # (base_depth is positive offset below min surface, typically 0.2 to 1.0)
            pos_base = pos.copy()
            pos_base[2] = min_surface_z - base_depth

            # For fractional edges: base tier uses fractional X,Y for smooth curve
            if use_fractional_edges and model_offset is not None:
                pos_base[0] = x / scale_factor - model_offset[0]
                pos_base[1] = y / scale_factor - model_offset[1]

            base_vertices[i] = pos_base

        # DEBUG: Print interpolation summary
        if has_smoothed_coords:
            total_boundary = interp_success + interp_fail_no_corners
            success_rate = interp_success / total_boundary * 100 if total_boundary > 0 else 0
            print(f"\n[DIAG] Vertex interpolation summary:")
            print(f"  Success: {interp_success}/{total_boundary} ({success_rate:.1f}%)")
            print(f"  Failed (no corners): {interp_fail_no_corners}")
            if failed_coords:
                print(f"  First failed coords (up to 20):")
                for y, x in failed_coords[:10]:
                    print(f"    (y={y:.2f}, x={x:.2f})")
                if len(failed_coords) > 10:
                    print(f"    ... and {len(failed_coords) - 10} more")

            # Print detailed missing corner info
            if hasattr(get_position_at_coords, 'missing_corner_samples') and get_position_at_coords.missing_corner_samples:
                samples = get_position_at_coords.missing_corner_samples[:10]
                print(f"\n[DIAG] Missing corner details (first {len(samples)}):")
                for s in samples:
                    print(f"    coord=({s['y']:.2f}, {s['x']:.2f}) floor=({s['y_floor']}, {s['x_floor']}) "
                          f"missing={s['missing']} had={s['n_corners']}/4 corners")

            # Print position interpolation samples to verify smoothness
            if position_samples:
                frac_mode = use_fractional_edges and model_offset is not None
                print(f"\n[DIAG] Position interpolation samples (first {len(position_samples)}):")
                print(f"  Fractional edge mode: {'ENABLED' if frac_mode else 'DISABLED'}")
                if frac_mode:
                    print(f"  Surface tier: Bilinear interpolation (aligned with mesh, no gap)")
                    print(f"  Mid/Base tiers: Fractional X,Y coords (smooth curved edge)")
                if frac_mode:
                    print(f"  {'i':>4} | {'y_in':>8} {'x_in':>8} | {'surface tier (bilinear)':>23} | {'z':>8}")
                    print(f"  {'-'*4}-+-{'-'*8}-{'-'*8}-+-{'-'*23}-+-{'-'*8}")
                    for s in position_samples[:20]:
                        print(f"  {s['i']:4d} | {s['y_in']:8.3f} {s['x_in']:8.3f} | "
                              f"({s['x_out']:9.4f}, {s['y_out']:9.4f}) | {s['z_out']:8.4f}")
                else:
                    print(f"  {'i':>4} | {'y_in':>8} {'x_in':>8} | {'x_out':>10} {'y_out':>10} {'z_out':>8}")
                    print(f"  {'-'*4}-+-{'-'*8}-{'-'*8}-+-{'-'*10}-{'-'*10}-{'-'*8}")
                    for s in position_samples[:20]:
                        print(f"  {s['i']:4d} | {s['y_in']:8.3f} {s['x_in']:8.3f} | "
                              f"{s['x_out']:10.5f} {s['y_out']:10.5f} {s['z_out']:8.4f}")
                if len(position_samples) > 20:
                    print(f"  ... ({len(position_samples) - 20} more samples)")

                # Check for stair-stepping: are X,Y outputs changing smoothly?
                x_outs = [s['x_out'] for s in position_samples]
                y_outs = [s['y_out'] for s in position_samples]
                x_diffs = [abs(x_outs[i+1] - x_outs[i]) for i in range(len(x_outs)-1)]
                y_diffs = [abs(y_outs[i+1] - y_outs[i]) for i in range(len(y_outs)-1)]
                print(f"\n  Output position deltas (smoothness check):")
                print(f"    X: min={min(x_diffs) if x_diffs else 0:.6f}, max={max(x_diffs) if x_diffs else 0:.6f}, "
                      f"mean={sum(x_diffs)/len(x_diffs) if x_diffs else 0:.6f}")
                print(f"    Y: min={min(y_diffs) if y_diffs else 0:.6f}, max={max(y_diffs) if y_diffs else 0:.6f}, "
                      f"mean={sum(y_diffs)/len(y_diffs) if y_diffs else 0:.6f}")

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

        # DEBUG: Track face generation statistics
        faces_created = 0
        faces_skipped_none = 0
        faces_skipped_distance = 0
        bridge_faces_created = 0

        for i in range(n_boundary):
            # Skip if surface index is None (integer coords) or vertex wasn't initialized (smoothed coords)
            if surface_indices[i] is None or not valid_boundary_vertex[i]:
                faces_skipped_none += 1
                continue

            next_i = (i + 1) % n_boundary
            if surface_indices[next_i] is None or not valid_boundary_vertex[next_i]:
                faces_skipped_none += 1
                continue

            # Check if this is the wrap-around edge (last → first vertex)
            # For rectangle edges with angular sorting, check if gap is reasonable
            if i == n_boundary - 1 and use_rectangle_edges:
                # Get positions of last and first boundary points
                y_last, x_last = boundary_points[i]
                y_first, x_first = boundary_points[0]

                # Calculate wrap-around distance
                distance = np.sqrt((y_last - y_first)**2 + (x_last - x_first)**2)

                # Calculate median edge distance for comparison
                # Sample distances between consecutive boundary points to establish "normal" edge spacing
                sample_size = min(100, n_boundary - 1)
                sample_distances = []
                for j in range(sample_size):
                    y_curr, x_curr = boundary_points[j]
                    y_next, x_next = boundary_points[j + 1]
                    d = np.sqrt((y_next - y_curr)**2 + (x_next - x_curr)**2)
                    sample_distances.append(d)

                median_edge_distance = np.median(sample_distances)

                # Allow wrap-around only if it's within 10x the median edge distance
                # This prevents diagonal artifacts across the mesh
                threshold = max(median_edge_distance * 10.0, 50.0)

                if distance > threshold:
                    # Skip this wrap-around edge - gap is too large relative to normal edge spacing
                    faces_skipped_distance += 1
                    print(f"  ⚠️  Wrap-around face skipped: distance = {distance:.2f} > {threshold:.1f} (median edge = {median_edge_distance:.2f})")
                    continue
                elif distance > median_edge_distance * 2.0:
                    # Warn but still create the face (gap is large but acceptable)
                    print(f"  ℹ️  Wrap-around face: distance = {distance:.2f} pixels ({distance/median_edge_distance:.1f}x median, closing loop)")

            # When using smoothed coordinates (but NOT fractional/Catmull-Rom), bridge original
            # mesh edge to new smooth boundary surface tier.
            # Skip for fractional edges: surface tier already aligned with mesh (no gap)
            # Skip for Catmull-Rom: creates too many interpolated points
            if has_smoothed_coords and not (use_fractional_edges or use_catmull_rom):
                # Find nearest original boundary vertices to this smoothed segment
                # by rounding the smoothed coordinates
                orig_i_y, orig_i_x = int(np.round(boundary_points[i][0])), int(np.round(boundary_points[i][1]))
                orig_next_y, orig_next_x = int(np.round(boundary_points[next_i][0])), int(np.round(boundary_points[next_i][1]))

                orig_i = coord_to_index.get((orig_i_y, orig_i_x))
                orig_next = coord_to_index.get((orig_next_y, orig_next_x))

                if orig_i is not None and orig_next is not None:
                    # Bridge face: original boundary → new smooth surface tier
                    # This connects the stair-step to the smooth curve
                    # Face winding must match boundary direction
                    if boundary_winding == "clockwise":
                        # Clockwise boundary: reverse face winding
                        boundary_faces.append(
                            (
                                orig_i,
                                surface_indices[i],
                                surface_indices[next_i],
                                orig_next,
                            )
                        )
                    else:
                        # Counter-clockwise boundary: standard winding
                        boundary_faces.append(
                            (
                                orig_i,
                                orig_next,
                                surface_indices[next_i],
                                surface_indices[i],
                            )
                        )
                    bridge_faces_created += 1

            # Upper tier: surface → mid
            # Face winding must match boundary direction for correct normals
            if boundary_winding == "clockwise":
                # Clockwise boundary: reverse face winding for outward normals
                boundary_faces.append(
                    (
                        surface_indices[i],
                        mid_indices[i],
                        mid_indices[next_i],
                        surface_indices[next_i],
                    )
                )
            else:
                # Counter-clockwise boundary: standard winding
                boundary_faces.append(
                    (
                        surface_indices[i],
                        surface_indices[next_i],
                        mid_indices[next_i],
                        mid_indices[i],
                    )
                )
            faces_created += 1

            # Lower tier: mid → base
            if boundary_winding == "clockwise":
                # Clockwise boundary: reverse face winding for outward normals
                boundary_faces.append(
                    (
                        mid_indices[i],
                        base_indices[i],
                        base_indices[next_i],
                        mid_indices[next_i],
                    )
                )
            else:
                # Counter-clockwise boundary: standard winding
                boundary_faces.append(
                    (
                        mid_indices[i],
                        mid_indices[next_i],
                        base_indices[next_i],
                        base_indices[i],
                    )
                )
            faces_created += 1

        # DEBUG: Print face generation statistics
        print(f"\n{'='*60}")
        print(f"Boundary Face Generation (Two-Tier)")
        print(f"{'='*60}")
        print(f"Boundary winding: {boundary_winding}")
        print(f"Boundary vertices: {n_boundary}")
        print(f"Surface indices (valid): {n_boundary - sum(1 for idx in surface_indices if idx is None)}")
        print(f"Surface indices (None): {sum(1 for idx in surface_indices if idx is None)}")
        if has_smoothed_coords:
            print(f"Bridge faces created: {bridge_faces_created}")
        print(f"Tier faces created: {faces_created}")
        print(f"Faces skipped (None index): {faces_skipped_none}")
        print(f"Faces skipped (distance check): {faces_skipped_distance}")
        print(f"Total boundary faces: {len(boundary_faces)}")
        expected_faces = n_boundary * 2  # 2 faces per boundary segment (upper + lower)
        coverage = faces_created / expected_faces * 100 if expected_faces > 0 else 0
        print(f"Expected faces (ideal): {expected_faces}")
        print(f"Coverage: {coverage:.1f}%")

        # DEBUG: Sample a few face windings to verify correctness
        if len(boundary_faces) > 0:
            print(f"\nSample face indices (first 3 faces):")
            for i in range(min(3, len(boundary_faces))):
                face = boundary_faces[i]
                print(f"  Face {i}: {face}")

        print(f"{'='*60}\n")

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
                    # For integer coordinates, try direct lookup first
                    y_int, x_int = int(y), int(x)
                    original_idx = coord_to_index.get((y_int, x_int))
                    if original_idx is not None:
                        surface_color = surface_colors[original_idx, :3]
                    else:
                        # Direct lookup failed - interpolate from nearby valid pixels
                        # This happens with rectangle-edge sampling after downsampling
                        y_floor, x_floor = int(np.floor(y)), int(np.floor(x))
                        color_corners = {}
                        for dy, dx in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                            yy, xx = y_floor + dy, x_floor + dx
                            idx = coord_to_index.get((yy, xx))
                            if idx is not None:
                                color_corners[(dy, dx)] = surface_colors[idx, :3]

                        if len(color_corners) >= 1:
                            # Bilinear interpolation if we have all 4 corners
                            if len(color_corners) == 4:
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
                                # Fallback: average available corners
                                colors_array = np.array(list(color_corners.values()), dtype=float)
                                surface_color = np.mean(colors_array, axis=0).astype(np.uint8)

            if surface_color is None:
                # Use base material color as fallback
                surface_color = base_color_uint8

            # Assign colors based on tier structure
            if has_smoothed_coords:
                # Three-tier structure: surface + mid + base
                # Surface tier uses mesh colors, mid/base use edge material or blended
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
        print(f"    Removed {duplicates_removed} duplicate points")

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
    print(f"  Angular sorting: max gap = {max_gap_distance:.2f} pixels (placed at wrap-around)")

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
                print(f"  ⚠️  Warning: Boundary sorting incomplete - {missing}/{n_points} points not connected")
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

    # IMPORTANT: Keep fractional coordinates! They're meaningful for sub-pixel sampling.
    # Only round after coordinate transformation to preserve edge density.

    # Top edge (y=0, x from 0 to width-1)
    for x in np.arange(0, width, edge_sample_spacing):
        edge_pixels.append((0.0, float(x)))

    # Right edge (x=width-1, y from spacing to height-1)
    for y in np.arange(edge_sample_spacing, height, edge_sample_spacing):
        edge_pixels.append((float(y), float(width - 1)))

    # Bottom edge (y=height-1, x from width-1 down to 0)
    for x in np.arange(width - 1, -1, -edge_sample_spacing):
        edge_pixels.append((float(height - 1), float(x)))

    # Left edge (x=0, y from height-1 down to spacing)
    for y in np.arange(height - 1 - edge_sample_spacing, -1, -edge_sample_spacing):
        if y >= 0:
            edge_pixels.append((float(y), 0.0))

    # Remove duplicates (corners get added twice - should be rare with fractional coords)
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
    3. Creates BOTH surface vertices (at DEM elevation) AND base vertices (at base_depth)
    4. Generates quad faces forming vertical walls ("skirt") around the terrain edge

    Vertex layout:
    - Indices 0 to n-1: Surface vertices (at DEM elevation)
    - Indices n to 2n-1: Base vertices (at base_depth, same x,y as surface)

    Args:
        dem_shape (tuple): Original DEM shape (height, width)
        dem_data (np.ndarray): Original DEM data array
        original_transform (Affine): Original affine transform (pixel → geographic)
        transforms_list (list): List of transform functions to apply sequentially
        edge_sample_spacing (float): Pixel spacing for edge sampling (default: 1.0)
        base_depth (float): Z-coordinate for base vertices (default: -0.2)

    Returns:
        tuple: (boundary_vertices, boundary_faces) where:
            - boundary_vertices: (2*n)x3 array of vertex positions (surface + base)
            - boundary_faces: List of quad faces forming vertical walls
    """
    # Step 1: Get rectangle edge pixels
    edge_pixels = generate_rectangle_edge_pixels(dem_shape, edge_sample_spacing)

    # Step 2: Create BOTH surface and base vertices
    surface_vertices = []
    base_vertices = []

    for y_px, x_px in edge_pixels:
        # Apply original affine transform to pixel coordinates
        x_world = original_transform.c + original_transform.a * x_px + original_transform.b * y_px
        y_world = original_transform.f + original_transform.d * x_px + original_transform.e * y_px

        # Sample DEM elevation at this edge pixel (with boundary clamping)
        y_idx = int(round(y_px))
        x_idx = int(round(x_px))
        y_idx = max(0, min(y_idx, dem_shape[0] - 1))
        x_idx = max(0, min(x_idx, dem_shape[1] - 1))
        elevation = dem_data[y_idx, x_idx]

        # Surface vertex at DEM elevation
        surface_vertices.append([x_world, y_world, elevation])
        # Base vertex at base_depth (same x, y)
        base_vertices.append([x_world, y_world, base_depth])

    # Stack: surface vertices first (0 to n-1), then base vertices (n to 2n-1)
    boundary_vertices = np.array(surface_vertices + base_vertices, dtype=float)

    # Step 3: Create quad faces forming vertical walls
    # Each quad connects surface and base vertices to form a vertical wall
    n_edge = len(edge_pixels)
    boundary_faces = []

    for i in range(n_edge):
        next_i = (i + 1) % n_edge

        # Indices: surface = 0..n-1, base = n..2n-1
        surface_i = i
        surface_next = next_i
        base_i = i + n_edge
        base_next = next_i + n_edge

        # Face winding for outward normals (boundary traces clockwise in image coords)
        # Order: surface[i] → base[i] → base[i+1] → surface[i+1]
        boundary_faces.append([surface_i, base_i, base_next, surface_next])

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
    from pyproj import Transformer

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

    # Get CRS information for reprojection
    original_crs = dem_layer.get("crs", "EPSG:4326")
    transformed_crs = dem_layer.get("transformed_crs", original_crs)

    # Create coordinate transformer if CRS changed
    transformer = None
    if original_crs != transformed_crs:
        transformer = Transformer.from_crs(original_crs, transformed_crs, always_xy=True)

    # 3. Map each edge pixel: original → geographic → reprojected → final
    edge_pixels_final = []
    transform_errors = 0
    out_of_bounds = 0
    not_in_coord_index = 0

    for (y_orig, x_orig) in edge_pixels_orig:
        try:
            # Original pixel → geographic coords in original CRS
            # Affine multiplication: (x_geo, y_geo) = transform * (x_px, y_px)
            # Note: Affine takes (x, y) not (y, x)
            x_geo, y_geo = original_transform * (x_orig, y_orig)

            # Reproject if CRS changed
            if transformer is not None:
                x_geo, y_geo = transformer.transform(x_geo, y_geo)

            # Geographic coords (in transformed CRS) → final pixel coords
            # Inverse transform: (x_px, y_px) = ~transform * (x_geo, y_geo)
            x_final, y_final = ~transformed_transform * (x_geo, y_geo)

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

    return edge_pixels_final


def generate_transform_aware_rectangle_edges_fractional(
    terrain,
    edge_sample_spacing=1.0,
):
    """
    Generate rectangle edge vertices with FRACTIONAL coordinates preserving projection curvature.

    Unlike generate_transform_aware_rectangle_edges() which rounds to integers and filters
    to existing mesh vertices, this function returns the true fractional coordinates
    from the non-linear projection transformation.

    This preserves the curved boundary that results from:
    - WGS84 → UTM Transverse Mercator projection (non-linear, causes curvature)
    - Horizontal flip transform
    - Downsampling

    Args:
        terrain: Terrain object with dem_shape, dem_transform, data_layers
        edge_sample_spacing: Pixel spacing for edge sampling at original resolution (default 1.0)

    Returns:
        list of (y, x) tuples: Fractional edge coordinates in final mesh space.
            These coordinates preserve the true curved boundary and may extend
            slightly beyond the integer grid bounds.

    Raises:
        ValueError: If terrain is None or lacks required transform data
    """
    from pyproj import Transformer

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

    # Get CRS information for reprojection
    original_crs = dem_layer.get("crs", "EPSG:4326")
    transformed_crs = dem_layer.get("transformed_crs", original_crs)

    # Create coordinate transformer if CRS changed
    transformer = None
    if original_crs != transformed_crs:
        transformer = Transformer.from_crs(original_crs, transformed_crs, always_xy=True)

    # 3. Map each edge pixel: original → geographic → reprojected → final (FRACTIONAL)
    edge_pixels_fractional = []
    transform_errors = 0

    for (y_orig, x_orig) in edge_pixels_orig:
        try:
            # Original pixel → geographic coords in original CRS
            x_geo, y_geo = original_transform * (x_orig, y_orig)

            # Reproject if CRS changed (this is the NON-LINEAR step!)
            if transformer is not None:
                x_geo, y_geo = transformer.transform(x_geo, y_geo)

            # Geographic coords (in transformed CRS) → final pixel coords
            # KEEP FRACTIONAL - do NOT round to integer!
            x_final, y_final = ~transformed_transform * (x_geo, y_geo)

            edge_pixels_fractional.append((y_final, x_final))

        except Exception:
            transform_errors += 1
            continue

    return edge_pixels_fractional
