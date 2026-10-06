"""Rectangle-edge boundary sampling, including transform-aware (projection-curved) edges."""

import logging

import numpy as np

from terrain_maker.terrain.mesh.boundary import (
    deduplicate_boundary_points,
    sort_boundary_points,
    sort_boundary_points_angular,
)

logger = logging.getLogger(__name__)


def _select_rectangle_boundary(
    boundary_points, coord_to_index, terrain, dem_shape, use_fractional_edges, edge_sample_spacing
):
    """Replace the morphological boundary with rectangle-edge samples when they cover it well.

    Samples the DEM rectangle edges (fractional or integer, transform-aware when
    ``terrain`` is given, else from ``dem_shape``). Keeps the morphological
    boundary if the rectangle samples are too sparse.
    """
    original_morphological_boundary = boundary_points

    # NEW: Use fractional coordinates to preserve projection curvature
    if use_fractional_edges and terrain is not None:
        # Fractional edge sampling - preserves curved boundary from projection
        rect_boundary_fractional = generate_transform_aware_rectangle_edges_fractional(
            terrain, edge_sample_spacing
        )

        # Report results
        original_shape = terrain.dem_shape
        logger.info(f"\n{'='*60}")
        logger.info(f"Transform-Aware Fractional Edge Sampling (Curved Boundary)")
        logger.info(f"{'='*60}")
        logger.info(f"Original DEM: {original_shape[0]}×{original_shape[1]} (sampling source)")
        logger.info(f"Fractional edge vertices: {len(rect_boundary_fractional)}")
        logger.info(f"Edge sample spacing: {edge_sample_spacing:.1f} pixels")
        logger.info(f"NOTE: Fractional coordinates preserve projection curvature")
        logger.info(f"{'='*60}\n")

        # Use fractional edges directly - they'll be processed by bilinear interpolation
        # No need to filter through coord_to_index since these are fractional coords
        rect_boundary_valid = rect_boundary_fractional

    # Use transform-aware INTEGER approach if terrain provided but not fractional
    elif terrain is not None:
        # Transform-aware rectangle-edge sampling (avoids NaN margins)
        rect_boundary_valid = generate_transform_aware_rectangle_edges(
            terrain, coord_to_index, edge_sample_spacing
        )

        # Report results
        original_shape = terrain.dem_shape
        logger.info(f"\n{'='*60}")
        logger.info(f"Transform-Aware Rectangle Edge Sampling (Integer)")
        logger.info(f"{'='*60}")
        logger.info(f"Original DEM: {original_shape[0]}×{original_shape[1]} (sampling source)")
        logger.info(f"Edge pixels mapped to final mesh: {len(rect_boundary_valid)}")
        logger.info(f"Edge sample spacing: {edge_sample_spacing:.1f} pixels")
        logger.info(f"{'='*60}\n")
    else:
        # FALLBACK: Legacy approach using transformed DEM shape
        if dem_shape is None:
            raise ValueError("Either terrain or dem_shape required when use_rectangle_edges=True")

        # Run diagnostic to show why this doesn't work well
        diagnostic = diagnose_rectangle_edge_coverage(dem_shape, coord_to_index)
        logger.info(f"\n{'='*60}")
        logger.warning(f"⚠️  Legacy Rectangle Edge Sampling (Transformed DEM)")
        logger.info(f"{'='*60}")
        logger.info(f"DEM shape: {diagnostic['dem_shape'][0]}×{diagnostic['dem_shape'][1]}")
        logger.info(
            f"Edge coverage: {diagnostic['coverage_percent']:.1f}% ({diagnostic['valid_edge_pixels']}/{diagnostic['total_edge_pixels']} pixels)"
        )
        logger.info(
            f"  Top edge:    {diagnostic['edge_validity']['top']['valid']:4d}/{diagnostic['edge_validity']['top']['total']:4d} valid ({diagnostic['edge_validity']['top']['valid']/max(1,diagnostic['edge_validity']['top']['total'])*100:.1f}%)"
        )
        logger.info(
            f"  Right edge:  {diagnostic['edge_validity']['right']['valid']:4d}/{diagnostic['edge_validity']['right']['total']:4d} valid ({diagnostic['edge_validity']['right']['valid']/max(1,diagnostic['edge_validity']['right']['total'])*100:.1f}%)"
        )
        logger.info(
            f"  Bottom edge: {diagnostic['edge_validity']['bottom']['valid']:4d}/{diagnostic['edge_validity']['bottom']['total']:4d} valid ({diagnostic['edge_validity']['bottom']['valid']/max(1,diagnostic['edge_validity']['bottom']['total'])*100:.1f}%)"
        )
        logger.info(
            f"  Left edge:   {diagnostic['edge_validity']['left']['valid']:4d}/{diagnostic['edge_validity']['left']['total']:4d} valid ({diagnostic['edge_validity']['left']['valid']/max(1,diagnostic['edge_validity']['left']['total'])*100:.1f}%)"
        )
        logger.info(f"\nRecommendation: {diagnostic['recommendation']}")
        logger.info(f"Reason: {diagnostic['reason']}")
        logger.info(
            f"💡 Tip: Pass terrain= parameter for transform-aware sampling (~100% coverage)"
        )
        logger.info(f"{'='*60}\n")

        rect_edge_pixels = generate_rectangle_edge_pixels(dem_shape, edge_sample_spacing)

        # Filter to only include pixels that are actually valid mesh vertices
        # Many rectangle edge pixels might be NaN or outside valid_mask, causing lookup failures
        rect_boundary_valid = [
            (y, x) for y, x in rect_edge_pixels if (int(y), int(x)) in coord_to_index
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
        logger.info(
            f"✓ Rectangle-edge sampling: Using {rect_count} boundary vertices (morphological had {original_count})"
        )

        # CRITICAL: After coordinate transformation, the natural rectangle order is destroyed!
        # First deduplicate, then re-sort spatially to form a closed loop
        logger.info(f"  Deduplicating boundary points...")
        rect_boundary_unique = deduplicate_boundary_points(rect_boundary_valid)

        # For dense boundaries, use angular sorting (faster and more robust)
        # For sparse boundaries, use nearest-neighbor
        if len(rect_boundary_unique) >= 100:
            logger.info(f"  Sorting {len(rect_boundary_unique)} points using angular method...")
            boundary_points = sort_boundary_points_angular(rect_boundary_unique)
        else:
            logger.info(f"  Sorting {len(rect_boundary_unique)} points using nearest-neighbor...")
            boundary_points = sort_boundary_points(rect_boundary_unique)
        logger.info(f"  ✓ Boundary sorted into continuous path")

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

        logger.info(f"  Boundary point distribution:")
        logger.info(f"    Top edge (north):    {top_count:6d} points")
        logger.info(f"    Bottom edge (south): {bottom_count:6d} points")
        logger.info(f"    Left edge (west):    {left_count:6d} points")
        logger.info(f"    Right edge (east):   {right_count:6d} points")

        # Check if distribution is severely uneven (any edge has < 5% of points)
        total_points = len(boundary_points)
        min_percent = min(top_count, bottom_count, left_count, right_count) / total_points * 100
        if min_percent < 5.0:
            logger.warning(f"  ⚠️  Warning: Uneven distribution detected (min={min_percent:.1f}%)")
            logger.info(f"  Sparse edges may have lower visual quality")
    else:
        # Rectangle edges too sparse - keep morphological boundary
        boundary_points = original_morphological_boundary
        logger.warning(
            f"✗ Rectangle-edge sampling: Too few valid vertices ({rect_count}), keeping morphological boundary ({original_count} vertices)"
        )
        if terrain is None:
            logger.info(
                f"  Tip: Pass terrain= parameter for transform-aware sampling to avoid NaN margins"
            )
        else:
            logger.info(
                f"  Tip: Check coordinate transformation - may be mapping outside valid mesh bounds"
            )

    return boundary_points


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
        "top": {"total": 0, "valid": 0},
        "right": {"total": 0, "valid": 0},
        "bottom": {"total": 0, "valid": 0},
        "left": {"total": 0, "valid": 0},
    }

    # Check top edge (y=0, x from 0 to width-1)
    for x in range(width):
        edge_validity["top"]["total"] += 1
        if (0, x) in coord_to_index:
            edge_validity["top"]["valid"] += 1

    # Check right edge (x=width-1, y from 0 to height-1)
    for y in range(height):
        edge_validity["right"]["total"] += 1
        if (y, width - 1) in coord_to_index:
            edge_validity["right"]["valid"] += 1

    # Check bottom edge (y=height-1, x from 0 to width-1)
    for x in range(width):
        edge_validity["bottom"]["total"] += 1
        if (height - 1, x) in coord_to_index:
            edge_validity["bottom"]["valid"] += 1

    # Check left edge (x=0, y from 0 to height-1)
    for y in range(height):
        edge_validity["left"]["total"] += 1
        if (y, 0) in coord_to_index:
            edge_validity["left"]["valid"] += 1

    # Calculate totals
    total_edge_pixels = 2 * (height + width) - 4  # Perimeter, not counting corners twice
    valid_edge_pixels = (
        edge_validity["top"]["valid"]
        + edge_validity["right"]["valid"]
        + edge_validity["bottom"]["valid"]
        + edge_validity["left"]["valid"]
        - 4  # Remove duplicate corner counts
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
        "dem_shape": dem_shape,
        "total_edge_pixels": total_edge_pixels,
        "valid_edge_pixels": valid_edge_pixels,
        "coverage_percent": coverage_percent,
        "edge_validity": edge_validity,
        "recommendation": recommendation,
        "reason": reason,
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


def _require_finite(x, y, y_orig, x_orig):
    """pyproj reports a failed transform as inf rather than raising; refuse it."""
    if not (np.isfinite(x) and np.isfinite(y)):
        raise ValueError(f"transform produced non-finite coordinates ({x}, {y})")


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
    edge_pixels_orig = generate_rectangle_edge_pixels(original_shape, edge_sample_spacing)

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
    if not dem_layer.get("crs"):
        raise ValueError("Terrain DEM layer has no 'crs'; cannot map edge pixels")
    original_crs = dem_layer["crs"]
    transformed_crs = dem_layer.get("transformed_crs", original_crs)

    # Create coordinate transformer if CRS changed
    transformer = None
    if original_crs != transformed_crs:
        transformer = Transformer.from_crs(original_crs, transformed_crs, always_xy=True)

    # 3. Map each edge pixel: original → geographic → reprojected → final
    edge_pixels_final = []
    out_of_bounds = 0
    not_in_coord_index = 0

    for y_orig, x_orig in edge_pixels_orig:
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
            _require_finite(x_final, y_final, y_orig, x_orig)

            # Round to integer pixel coordinates
            y_int, x_int = int(round(y_final)), int(round(x_final))

            # Check bounds
            transformed_shape = dem_layer["transformed_data"].shape
            if (
                y_int < 0
                or y_int >= transformed_shape[0]
                or x_int < 0
                or x_int >= transformed_shape[1]
            ):
                out_of_bounds += 1
                continue

            # Check if this maps to a valid mesh vertex
            if (y_int, x_int) in coord_to_index:
                edge_pixels_final.append((y_int, x_int))
            else:
                not_in_coord_index += 1

        except Exception as e:
            raise ValueError(
                f"Edge pixel (row {y_orig}, col {x_orig}) could not be mapped from "
                f"{original_crs} to {transformed_crs}: {e}"
            ) from e

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
    edge_pixels_orig = generate_rectangle_edge_pixels(original_shape, edge_sample_spacing)

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
    if not dem_layer.get("crs"):
        raise ValueError("Terrain DEM layer has no 'crs'; cannot map edge pixels")
    original_crs = dem_layer["crs"]
    transformed_crs = dem_layer.get("transformed_crs", original_crs)

    # Create coordinate transformer if CRS changed
    transformer = None
    if original_crs != transformed_crs:
        transformer = Transformer.from_crs(original_crs, transformed_crs, always_xy=True)

    # 3. Map each edge pixel: original → geographic → reprojected → final (FRACTIONAL)
    edge_pixels_fractional = []

    for y_orig, x_orig in edge_pixels_orig:
        try:
            # Original pixel → geographic coords in original CRS
            x_geo, y_geo = original_transform * (x_orig, y_orig)

            # Reproject if CRS changed (this is the NON-LINEAR step!)
            if transformer is not None:
                x_geo, y_geo = transformer.transform(x_geo, y_geo)

            # Geographic coords (in transformed CRS) → final pixel coords
            # KEEP FRACTIONAL - do NOT round to integer!
            x_final, y_final = ~transformed_transform * (x_geo, y_geo)
            _require_finite(x_final, y_final, y_orig, x_orig)

            edge_pixels_fractional.append((y_final, x_final))

        except Exception as e:
            raise ValueError(
                f"Edge pixel (row {y_orig}, col {x_orig}) could not be mapped from "
                f"{original_crs} to {transformed_crs}: {e}"
            ) from e

    return edge_pixels_fractional
