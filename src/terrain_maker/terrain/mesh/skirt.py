"""Side skirt that closes the mesh into a solid: single- and two-tier edge extrusion."""

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from terrain_maker.terrain.mesh.boundary import (
    fit_catmull_rom_boundary_curve,
    smooth_boundary_points,
)
from terrain_maker.terrain.mesh.rectangle_edges import _select_rectangle_boundary

logger = logging.getLogger(__name__)


def _make_position_lookup(positions, coord_to_index):
    """Build get_position_at_coords(y, x) for integer or fractional grid coordinates.

    Returns exact vertex positions for integer coordinates and interpolates
    (bilinear, partial-corner, or nearest-vertex search) for fractional ones.
    Unresolvable lookups return None and are recorded in
    ``get_position_at_coords.missing_corner_samples``.
    """
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
            get_position_at_coords.missing_corner_samples.append(
                {
                    "y": y,
                    "x": x,
                    "y_floor": y_floor,
                    "x_floor": x_floor,
                    "missing": missing_corners,
                    "n_corners": len(corners),
                }
            )

        # Final fallback: return None if can't find any position
        return None

    # Initialize debug tracking
    get_position_at_coords.missing_corner_samples = []

    return get_position_at_coords


def _skirt_quad(top_a, top_b, bottom_a, bottom_b, clockwise):
    """Quad face between two tiers along boundary segment a -> b, wound for outward normals.

    Clockwise boundaries need the reversed vertex order to keep normals facing out.
    """
    if clockwise:
        return (top_a, bottom_a, bottom_b, top_b)
    return (top_a, top_b, bottom_b, bottom_a)


def _wraparound_gap_too_large(boundary_points):
    """True when the last -> first closing segment is far longer than normal edge spacing.

    Rectangle edges are angle-sorted, so a huge closing gap would draw a diagonal
    face across the mesh. Logs the decision either way.
    """
    n_boundary = len(boundary_points)
    y_last, x_last = boundary_points[n_boundary - 1]
    y_first, x_first = boundary_points[0]
    distance = np.sqrt((y_last - y_first) ** 2 + (x_last - x_first) ** 2)

    # "Normal" spacing: median of up to 100 consecutive segment lengths
    sample_distances = []
    for j in range(min(100, n_boundary - 1)):
        y_curr, x_curr = boundary_points[j]
        y_next, x_next = boundary_points[j + 1]
        sample_distances.append(np.sqrt((y_next - y_curr) ** 2 + (x_next - x_curr) ** 2))
    median_edge_distance = np.median(sample_distances)

    threshold = max(median_edge_distance * 10.0, 50.0)
    if distance > threshold:
        logger.warning(
            f"  ⚠️  Wrap-around face skipped: distance = {distance:.2f} > {threshold:.1f} "
            f"(median edge = {median_edge_distance:.2f})"
        )
        return True
    if distance > median_edge_distance * 2.0:
        logger.info(
            f"  ℹ️  Wrap-around face: distance = {distance:.2f} pixels "
            f"({distance/median_edge_distance:.1f}x median, closing loop)"
        )
    return False


@dataclass
class _SkirtInputs:
    """Resolved boundary and options shared by the single- and two-tier skirt builders."""

    positions: np.ndarray
    boundary_points: list
    original_boundary_points: list
    coord_to_index: dict
    get_position_at_coords: Callable
    has_smoothed_coords: bool
    base_depth: float
    boundary_winding: str
    use_catmull_rom: bool
    use_fractional_edges: bool
    use_rectangle_edges: bool
    scale_factor: float
    model_offset: Optional[np.ndarray]


def _build_single_tier_skirt(inputs: _SkirtInputs):
    """Skirt from the surface edge straight down to a flat base plane.

    Returns (boundary_vertices, boundary_faces).
    """
    positions = inputs.positions
    boundary_points = inputs.boundary_points
    original_boundary_points = inputs.original_boundary_points
    coord_to_index = inputs.coord_to_index
    get_position_at_coords = inputs.get_position_at_coords
    has_smoothed_coords = inputs.has_smoothed_coords
    base_depth = inputs.base_depth
    boundary_winding = inputs.boundary_winding
    clockwise = boundary_winding == "clockwise"
    use_catmull_rom = inputs.use_catmull_rom
    use_fractional_edges = inputs.use_fractional_edges
    use_rectangle_edges = inputs.use_rectangle_edges
    scale_factor = inputs.scale_factor
    model_offset = inputs.model_offset
    n_boundary = len(boundary_points)

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
        base_boundary_indices = list(range(n_existing + n_boundary, n_existing + 2 * n_boundary))

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

        # Closing segment of angle-sorted rectangle edges can span the mesh
        if (
            i == n_boundary - 1
            and use_rectangle_edges
            and _wraparound_gap_too_large(boundary_points)
        ):
            faces_skipped_distance += 1
            continue

        # Create quad connecting top boundary to bottom
        # Face winding must match boundary direction for correct normals
        boundary_faces.append(
            _skirt_quad(
                boundary_indices[i],
                boundary_indices[next_i],
                base_indices[i],
                base_indices[next_i],
                clockwise,
            )
        )
        faces_created += 1

    # DEBUG: Print face generation statistics
    logger.info(f"\n{'='*60}")
    logger.info(f"Boundary Face Generation (Single-Tier)")
    logger.info(f"{'='*60}")
    logger.info(f"Boundary winding: {boundary_winding}")
    logger.info(f"Boundary vertices: {n_boundary}")
    logger.info(
        f"Boundary indices (valid): {n_boundary - sum(1 for idx in boundary_indices if idx is None)}"
    )
    logger.info(f"Boundary indices (None): {sum(1 for idx in boundary_indices if idx is None)}")
    logger.info(f"Faces created: {faces_created}")
    logger.info(f"Faces skipped (None index): {faces_skipped_none}")
    logger.info(f"Faces skipped (distance check): {faces_skipped_distance}")
    logger.info(f"Total boundary faces: {len(boundary_faces)}")
    expected_faces = n_boundary  # 1 face per boundary segment
    coverage = faces_created / expected_faces * 100 if expected_faces > 0 else 0
    logger.info(f"Expected faces (ideal): {expected_faces}")
    logger.info(f"Coverage: {coverage:.1f}%")
    logger.info(f"{'='*60}\n")

    return boundary_vertices, boundary_faces


def _interpolated_surface_color(y, x, coord_to_index, surface_colors, allow_partial):
    """Bilinear color from the four mesh vertices around (y, x), or None.

    With allow_partial, fewer than four corners fall back to their mean color.
    """
    y_floor, x_floor = int(np.floor(y)), int(np.floor(x))
    corners = {}
    for dy, dx in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        idx = coord_to_index.get((y_floor + dy, x_floor + dx))
        if idx is not None:
            corners[(dy, dx)] = surface_colors[idx, :3]

    if len(corners) == 4:
        fy, fx = y - y_floor, x - x_floor
        c0 = corners[(0, 0)].astype(float) * (1 - fx) + corners[(0, 1)].astype(float) * fx
        c1 = corners[(1, 0)].astype(float) * (1 - fx) + corners[(1, 1)].astype(float) * fx
        return (c0 * (1 - fy) + c1 * fy).astype(np.uint8)
    if allow_partial and corners:
        return np.mean(np.array(list(corners.values()), dtype=float), axis=0).astype(np.uint8)
    return None


def _z_along_original_boundary(y, x, coords, z_values):
    """Z for a smoothed boundary point, interpolated along the nearest original boundary segment.

    Smoother than bilinear surface interpolation for Catmull-Rom curves. Returns None when
    the segment has a missing endpoint Z or zero length.
    """
    closest_idx = np.argmin(np.sqrt((coords[:, 0] - y) ** 2 + (coords[:, 1] - x) ** 2))
    next_idx = (closest_idx + 1) % len(coords)
    z1, z2 = z_values[closest_idx], z_values[next_idx]
    if z1 is None or z2 is None:
        return None

    orig_y1, orig_x1 = coords[closest_idx]
    orig_y2, orig_x2 = coords[next_idx]
    seg_dist = np.sqrt((orig_y2 - orig_y1) ** 2 + (orig_x2 - orig_x1) ** 2)
    if seg_dist <= 0:
        return None
    point_dist = np.sqrt((y - orig_y1) ** 2 + (x - orig_x1) ** 2)
    t = np.clip(point_dist / seg_dist, 0, 1)
    return z1 * (1 - t) + z2 * t


def _two_tier_colors(inputs: _SkirtInputs, base_color_rgb, blend_edge_colors, surface_colors):
    """Per-vertex skirt colors: surface color on the upper tiers, base material on the base.

    Tiers are surface + mid + base with smoothed coordinates, otherwise mid + base.
    The surface color is blended from the mesh when requested, else the base color.
    """
    has_smoothed_coords = inputs.has_smoothed_coords
    n_boundary = len(inputs.boundary_points)
    boundary_points = inputs.boundary_points
    coord_to_index = inputs.coord_to_index
    n_tiers = 3 if has_smoothed_coords else 2
    boundary_colors = np.zeros((n_tiers * n_boundary, 3), dtype=np.uint8)
    base_color_uint8 = (np.array(base_color_rgb) * 255).astype(np.uint8)
    blend = blend_edge_colors and surface_colors is not None

    for i, (y, x) in enumerate(boundary_points):
        surface_color = None
        if blend and has_smoothed_coords:
            surface_color = _interpolated_surface_color(
                y, x, coord_to_index, surface_colors, allow_partial=False
            )
        elif blend:
            # Integer coords: direct lookup, else interpolate (rectangle edges after downsampling)
            original_idx = coord_to_index.get((int(y), int(x)))
            if original_idx is not None:
                surface_color = surface_colors[original_idx, :3]
            else:
                surface_color = _interpolated_surface_color(
                    y, x, coord_to_index, surface_colors, allow_partial=True
                )
        if surface_color is None:
            surface_color = base_color_uint8

        # Every tier but the base carries the surface color
        for tier in range(n_tiers - 1):
            boundary_colors[i + tier * n_boundary, :3] = surface_color
        boundary_colors[i + (n_tiers - 1) * n_boundary, :3] = base_color_uint8

    return boundary_colors


def _log_two_tier_face_stats(
    boundary_winding,
    n_boundary,
    surface_indices,
    has_smoothed_coords,
    bridge_faces_created,
    faces_created,
    faces_skipped_none,
    faces_skipped_distance,
    boundary_faces,
):
    """Log face-generation statistics for the two-tier skirt."""
    # DEBUG: Print face generation statistics
    logger.info(f"\n{'='*60}")
    logger.info(f"Boundary Face Generation (Two-Tier)")
    logger.info(f"{'='*60}")
    logger.info(f"Boundary winding: {boundary_winding}")
    logger.info(f"Boundary vertices: {n_boundary}")
    logger.info(
        f"Surface indices (valid): {n_boundary - sum(1 for idx in surface_indices if idx is None)}"
    )
    logger.info(f"Surface indices (None): {sum(1 for idx in surface_indices if idx is None)}")
    if has_smoothed_coords:
        logger.info(f"Bridge faces created: {bridge_faces_created}")
    logger.info(f"Tier faces created: {faces_created}")
    logger.info(f"Faces skipped (None index): {faces_skipped_none}")
    logger.info(f"Faces skipped (distance check): {faces_skipped_distance}")
    logger.info(f"Total boundary faces: {len(boundary_faces)}")
    expected_faces = n_boundary * 2  # 2 faces per boundary segment (upper + lower)
    coverage = faces_created / expected_faces * 100 if expected_faces > 0 else 0
    logger.info(f"Expected faces (ideal): {expected_faces}")
    logger.info(f"Coverage: {coverage:.1f}%")

    # DEBUG: Sample a few face windings to verify correctness
    if len(boundary_faces) > 0:
        logger.info(f"\nSample face indices (first 3 faces):")
        for i in range(min(3, len(boundary_faces))):
            face = boundary_faces[i]
            logger.info(f"  Face {i}: {face}")

    logger.info(f"{'='*60}\n")


def _log_interpolation_diagnostics(
    has_smoothed_coords,
    failed_coords,
    position_samples,
    interp_success,
    interp_fail_no_corners,
    get_position_at_coords,
    use_fractional_edges,
    model_offset,
):
    """Log how many smoothed boundary points interpolated and sample the resulting positions."""
    # DEBUG: Print interpolation summary
    if has_smoothed_coords:
        total_boundary = interp_success + interp_fail_no_corners
        success_rate = interp_success / total_boundary * 100 if total_boundary > 0 else 0
        logger.info(f"\n[DIAG] Vertex interpolation summary:")
        logger.info(f"  Success: {interp_success}/{total_boundary} ({success_rate:.1f}%)")
        logger.info(f"  Failed (no corners): {interp_fail_no_corners}")
        if failed_coords:
            logger.info(f"  First failed coords (up to 20):")
            for y, x in failed_coords[:10]:
                logger.info(f"    (y={y:.2f}, x={x:.2f})")
            if len(failed_coords) > 10:
                logger.info(f"    ... and {len(failed_coords) - 10} more")

        # Print detailed missing corner info
        if (
            hasattr(get_position_at_coords, "missing_corner_samples")
            and get_position_at_coords.missing_corner_samples
        ):
            samples = get_position_at_coords.missing_corner_samples[:10]
            logger.info(f"\n[DIAG] Missing corner details (first {len(samples)}):")
            for s in samples:
                logger.info(
                    f"    coord=({s['y']:.2f}, {s['x']:.2f}) floor=({s['y_floor']}, {s['x_floor']}) "
                    f"missing={s['missing']} had={s['n_corners']}/4 corners"
                )

        # Print position interpolation samples to verify smoothness
        if position_samples:
            frac_mode = use_fractional_edges and model_offset is not None
            logger.info(f"\n[DIAG] Position interpolation samples (first {len(position_samples)}):")
            logger.info(f"  Fractional edge mode: {'ENABLED' if frac_mode else 'DISABLED'}")
            if frac_mode:
                logger.info(f"  Surface tier: Bilinear interpolation (aligned with mesh, no gap)")
                logger.info(f"  Mid/Base tiers: Fractional X,Y coords (smooth curved edge)")
            if frac_mode:
                logger.info(
                    f"  {'i':>4} | {'y_in':>8} {'x_in':>8} | {'surface tier (bilinear)':>23} | {'z':>8}"
                )
                logger.info(f"  {'-'*4}-+-{'-'*8}-{'-'*8}-+-{'-'*23}-+-{'-'*8}")
                for s in position_samples[:20]:
                    logger.info(
                        f"  {s['i']:4d} | {s['y_in']:8.3f} {s['x_in']:8.3f} | "
                        f"({s['x_out']:9.4f}, {s['y_out']:9.4f}) | {s['z_out']:8.4f}"
                    )
            else:
                logger.info(
                    f"  {'i':>4} | {'y_in':>8} {'x_in':>8} | {'x_out':>10} {'y_out':>10} {'z_out':>8}"
                )
                logger.info(f"  {'-'*4}-+-{'-'*8}-{'-'*8}-+-{'-'*10}-{'-'*10}-{'-'*8}")
                for s in position_samples[:20]:
                    logger.info(
                        f"  {s['i']:4d} | {s['y_in']:8.3f} {s['x_in']:8.3f} | "
                        f"{s['x_out']:10.5f} {s['y_out']:10.5f} {s['z_out']:8.4f}"
                    )
            if len(position_samples) > 20:
                logger.info(f"  ... ({len(position_samples) - 20} more samples)")

            # Check for stair-stepping: are X,Y outputs changing smoothly?
            x_outs = [s["x_out"] for s in position_samples]
            y_outs = [s["y_out"] for s in position_samples]
            x_diffs = [abs(x_outs[i + 1] - x_outs[i]) for i in range(len(x_outs) - 1)]
            y_diffs = [abs(y_outs[i + 1] - y_outs[i]) for i in range(len(y_outs) - 1)]
            logger.info(f"\n  Output position deltas (smoothness check):")
            logger.info(
                f"    X: min={min(x_diffs) if x_diffs else 0:.6f}, max={max(x_diffs) if x_diffs else 0:.6f}, "
                f"mean={sum(x_diffs)/len(x_diffs) if x_diffs else 0:.6f}"
            )
            logger.info(
                f"    Y: min={min(y_diffs) if y_diffs else 0:.6f}, max={max(y_diffs) if y_diffs else 0:.6f}, "
                f"mean={sum(y_diffs)/len(y_diffs) if y_diffs else 0:.6f}"
            )


def _two_tier_faces(
    inputs: _SkirtInputs, surface_indices, valid_boundary_vertex, mid_indices, base_indices
):
    """Quad faces for the skirt: optional bridge from the stair-step mesh edge, then surface -> mid -> base."""
    n_boundary = len(inputs.boundary_points)
    use_rectangle_edges = inputs.use_rectangle_edges
    has_smoothed_coords = inputs.has_smoothed_coords
    boundary_points = inputs.boundary_points
    clockwise = inputs.boundary_winding == "clockwise"
    use_fractional_edges = inputs.use_fractional_edges
    use_catmull_rom = inputs.use_catmull_rom
    coord_to_index = inputs.coord_to_index
    boundary_winding = inputs.boundary_winding
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

        # Closing segment of angle-sorted rectangle edges can span the mesh
        if (
            i == n_boundary - 1
            and use_rectangle_edges
            and _wraparound_gap_too_large(boundary_points)
        ):
            faces_skipped_distance += 1
            continue

        # When using smoothed coordinates (but NOT fractional/Catmull-Rom), bridge original
        # mesh edge to new smooth boundary surface tier.
        # Skip for fractional edges: surface tier already aligned with mesh (no gap)
        # Skip for Catmull-Rom: creates too many interpolated points
        if has_smoothed_coords and not (use_fractional_edges or use_catmull_rom):
            # Find nearest original boundary vertices to this smoothed segment
            # by rounding the smoothed coordinates
            orig_i_y, orig_i_x = int(np.round(boundary_points[i][0])), int(
                np.round(boundary_points[i][1])
            )
            orig_next_y, orig_next_x = int(np.round(boundary_points[next_i][0])), int(
                np.round(boundary_points[next_i][1])
            )

            orig_i = coord_to_index.get((orig_i_y, orig_i_x))
            orig_next = coord_to_index.get((orig_next_y, orig_next_x))

            if orig_i is not None and orig_next is not None:
                # Bridge face: original boundary → new smooth surface tier
                # This connects the stair-step to the smooth curve
                # Face winding must match boundary direction
                boundary_faces.append(
                    _skirt_quad(
                        orig_i, orig_next, surface_indices[i], surface_indices[next_i], clockwise
                    )
                )
                bridge_faces_created += 1

        # Upper tier: surface → mid
        # Face winding must match boundary direction for correct normals
        boundary_faces.append(
            _skirt_quad(
                surface_indices[i],
                surface_indices[next_i],
                mid_indices[i],
                mid_indices[next_i],
                clockwise,
            )
        )
        faces_created += 1

        # Lower tier: mid → base
        boundary_faces.append(
            _skirt_quad(
                mid_indices[i],
                mid_indices[next_i],
                base_indices[i],
                base_indices[next_i],
                clockwise,
            )
        )
        faces_created += 1

    _log_two_tier_face_stats(
        boundary_winding=boundary_winding,
        n_boundary=n_boundary,
        surface_indices=surface_indices,
        has_smoothed_coords=has_smoothed_coords,
        bridge_faces_created=bridge_faces_created,
        faces_created=faces_created,
        faces_skipped_none=faces_skipped_none,
        faces_skipped_distance=faces_skipped_distance,
        boundary_faces=boundary_faces,
    )
    return boundary_faces


def _two_tier_vertices(inputs: _SkirtInputs, mid_depth):
    """Surface (smoothed coords only), mid and base tier vertices, plus which boundary points resolved."""
    has_smoothed_coords = inputs.has_smoothed_coords
    original_boundary_points = inputs.original_boundary_points
    coord_to_index = inputs.coord_to_index
    positions = inputs.positions
    n_boundary = len(inputs.boundary_points)
    boundary_points = inputs.boundary_points
    use_fractional_edges = inputs.use_fractional_edges
    base_depth = inputs.base_depth
    get_position_at_coords = inputs.get_position_at_coords
    use_catmull_rom = inputs.use_catmull_rom
    model_offset = inputs.model_offset
    scale_factor = inputs.scale_factor
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
    failed_coords = []

    # Analyze boundary coordinate ranges
    if has_smoothed_coords and boundary_points:
        y_coords = [bp[0] for bp in boundary_points]
        x_coords = [bp[1] for bp in boundary_points]
        logger.info(f"\n[DIAG] Boundary coordinate ranges:")
        logger.info(f"  Y: min={min(y_coords):.2f}, max={max(y_coords):.2f}")
        logger.info(f"  X: min={min(x_coords):.2f}, max={max(x_coords):.2f}")
        # Get mesh bounds from coord_to_index
        if coord_to_index:
            all_yx = list(coord_to_index.keys())
            mesh_y = [yx[0] for yx in all_yx]
            mesh_x = [yx[1] for yx in all_yx]
            logger.info(f"  Mesh Y: min={min(mesh_y)}, max={max(mesh_y)}")
            logger.info(f"  Mesh X: min={min(mesh_x)}, max={max(mesh_x)}")

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
            if (
                use_catmull_rom
                and original_boundary_z_values
                and original_boundary_coords_array is not None
            ):
                z = _z_along_original_boundary(
                    y, x, original_boundary_coords_array, original_boundary_z_values
                )
                if z is not None:
                    pos[2] = z

            # For fractional edges: DON'T adjust surface tier X,Y
            # Keep surface vertices aligned with mesh boundary (from bilinear interpolation)
            # This eliminates gaps - surface tier shares vertex positions with mesh edge
            # Mid and base tiers will use fractional X,Y for smooth curves

            # Sample positions for diagnostic output
            # Surface tier uses bilinear interpolation (aligned with mesh)
            if len(position_samples) < 80:
                position_samples.append(
                    {
                        "i": i,
                        "y_in": y,
                        "x_in": x,
                        "bilinear_x": bilinear_x,
                        "bilinear_y": bilinear_y,
                        "x_out": pos[0],  # Surface tier (snapped to mesh)
                        "y_out": pos[1],
                        "z_out": pos[2],
                    }
                )

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

    _log_interpolation_diagnostics(
        has_smoothed_coords=has_smoothed_coords,
        failed_coords=failed_coords,
        position_samples=position_samples,
        interp_success=interp_success,
        interp_fail_no_corners=interp_fail_no_corners,
        get_position_at_coords=get_position_at_coords,
        use_fractional_edges=use_fractional_edges,
        model_offset=model_offset,
    )
    return base_vertices, mid_vertices, surface_vertices, valid_boundary_vertex


def _build_two_tier_skirt(
    inputs: _SkirtInputs, mid_depth, base_material, blend_edge_colors, surface_colors
):
    """Skirt with a mid tier that follows the surface and a flat colored base tier.

    Returns (boundary_vertices, boundary_faces, boundary_colors).
    """
    from terrain_maker.terrain.materials import get_base_material_color

    positions = inputs.positions
    boundary_points = inputs.boundary_points
    coord_to_index = inputs.coord_to_index
    has_smoothed_coords = inputs.has_smoothed_coords
    base_depth = inputs.base_depth
    n_boundary = len(boundary_points)

    # Auto-calculate mid_depth if not provided
    # mid_depth is a positive offset below surface (e.g., 0.05)
    # base_depth is positive offset below min surface (e.g., 0.2)
    # Default: shallow tier at 25% of base depth distance
    if mid_depth is None:
        mid_depth = base_depth * 0.25

    # Resolve material to RGB
    base_color_rgb = get_base_material_color(base_material)

    base_vertices, mid_vertices, surface_vertices, valid_boundary_vertex = _two_tier_vertices(
        inputs, mid_depth=mid_depth
    )

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

    boundary_faces = _two_tier_faces(
        inputs,
        surface_indices=surface_indices,
        valid_boundary_vertex=valid_boundary_vertex,
        mid_indices=mid_indices,
        base_indices=base_indices,
    )

    boundary_colors = _two_tier_colors(
        inputs,
        base_color_rgb=base_color_rgb,
        blend_edge_colors=blend_edge_colors,
        surface_colors=surface_colors,
    )
    return boundary_vertices, boundary_faces, boundary_colors


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
    # Use rectangle-edge sampling if requested (falls back to the morphological boundary)
    if use_rectangle_edges:
        boundary_points = _select_rectangle_boundary(
            boundary_points,
            coord_to_index,
            terrain,
            dem_shape,
            use_fractional_edges,
            edge_sample_spacing,
        )

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

    # Check if we have fractional coordinates that need bilinear interpolation
    # This can happen from: smooth_boundary, use_catmull_rom, OR use_fractional_edges
    has_smoothed_coords = (smooth_boundary or use_catmull_rom or use_fractional_edges) and any(
        not (isinstance(y, (int, np.integer)) and isinstance(x, (int, np.integer)))
        for y, x in boundary_points
    )

    # Position lookup for integer or fractional boundary coordinates
    get_position_at_coords = _make_position_lookup(positions, coord_to_index)

    inputs = _SkirtInputs(
        positions=positions,
        boundary_points=boundary_points,
        original_boundary_points=original_boundary_points,
        coord_to_index=coord_to_index,
        get_position_at_coords=get_position_at_coords,
        has_smoothed_coords=has_smoothed_coords,
        base_depth=base_depth,
        boundary_winding=boundary_winding,
        use_catmull_rom=use_catmull_rom,
        use_fractional_edges=use_fractional_edges,
        use_rectangle_edges=use_rectangle_edges,
        scale_factor=scale_factor,
        model_offset=model_offset,
    )
    if not two_tier:
        return _build_single_tier_skirt(inputs)
    return _build_two_tier_skirt(
        inputs, mid_depth, base_material, blend_edge_colors, surface_colors
    )
