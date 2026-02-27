"""
Temporal XC Skiing Ridge Sculpture.

Creates a 3D data sculpture from seasonal XC skiing scores. Each season
becomes a steep ridge running west-to-east (X = day of season, Z = score),
stacked north-to-south. The median ridge sits at the center, wider and
taller, as the visual anchor.

This module is self-contained and only loaded when --temporal-landscape
is passed to the combined render script.

Requirements:
    - Blender Python API available (bpy)
    - Temporal timeseries cache (from xc_skiing_temporal.py)

Usage:
    Called from detroit_combined_render.py via create_temporal_sculpture().
"""

import logging
from pathlib import Path
from typing import Optional

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from rasterio.transform import from_origin

from src.terrain.core import Terrain, scale_elevation
from src.terrain.scene_setup import position_mesh_at

logger = logging.getLogger(__name__)


def create_temporal_sculpture(
    main_mesh: "bpy.types.Object",
    snodas_dir: Path,
    cache_file: Path,
    height_scale: float = 25.0,
    ridge_height_fraction: float = 0.5,
    ridge_rows: int = 2,
    gap_rows: int = 0,
    tail_rows: int = 6,
    shear: float = 20.0,
    smooth_sigma: float | None = None,
    gap_fraction: float = 0.10,
    row_scale: int = 8,
    col_scale: int = 0,
    depth_fraction: float = 0.35,
    summary_color_scale: float = 1.0,
    diagnostic_dir: Path | None = None,
    score_colormap: str = "boreal_mako",
) -> Optional[tuple["bpy.types.Object", "bpy.types.Object"]]:
    """
    Build and position the temporal XC skiing ridge sculpture.

    Loads temporal data, builds ridge DEM, creates Terrain mesh,
    scales to match main_mesh width, normalizes Z to a fraction of the
    main terrain's Z extent, and positions below main_mesh.

    Between-ridge gaps are NaN-masked, so each ridge becomes an isolated
    island mesh floating just above the scene background.
    ``boundary_extension=False`` is required to prevent the boundary algorithm
    from hanging on hundreds of disconnected island loops.

    Parameters
    ----------
    main_mesh : bpy.types.Object
        The main terrain mesh to position below and match width to.
    snodas_dir : Path
        Directory containing SNODAS snow depth data.
    cache_file : Path
        Path to the .npz temporal timeseries cache.
    height_scale : float
        Mesh height_scale passed to create_mesh().
    ridge_height_fraction : float
        Fraction of the main terrain's Z range for temporal ridge peaks.
        0.5 means temporal peaks are half the main terrain's Z extent.
    ridge_rows : int
        Number of rows per season ridge cross-section.
    gap_rows : int
        Number of zero-height rows between ridges.
    tail_rows : int
        Extra rows of exponential decay tail per ridge.  Creates cascading
        overlap where each ridge builds on the residual of previous ones.
    shear : float
        Northward shift (in rows) for the highest-score columns.  High-score
        days lean north over the previous ridge; low-score days stay put.
    smooth_sigma : float or None
        Gaussian smoothing sigma along N-S after ridge construction.
        None (default) means no smoothing — float-precision shifts produce
        inherently smooth ramps.
    gap_fraction : float
        Gap between the main terrain bottom and the sculpture top, expressed
        as a fraction of the *main terrain width*.  0.10 (default) = 10% of
        main width.  Independent of depth_fraction and col_scale.
    row_scale : int
        Multiplier for Y (row) resolution.  The ridge DEM grid gets
        row_scale× more rows so float scores aren't quantized to coarse
        integer positions.  8 (default) gives 8× finer grid than the
        base ridge_rows/shear parameters specify.
    col_scale : int
        Column (X / day-of-season) upscaling factor.  Bicubic interpolation
        along the day axis for smoother E-W profiles.  0 (default) auto-
        computes from the main mesh to match ~1 vertex per pixel.
    depth_fraction : float
        Target N-S depth of the sculpture as a fraction of the main terrain
        width.  0.35 (default) ≈ 35% as deep as the terrain is wide.
        Converted at runtime to an internal ns_squash after the ridge DEM is
        built so that the value is stable across changes to col_scale.
    diagnostic_dir : Path or None
        If provided, save a diagnostic plot of the temporal data and ridge
        DEM to this directory.

    Returns
    -------
    (bpy.types.Object, bpy.types.Object) or None
        ``(ridge_mesh, bg_plane)`` — the positioned ridge sculpture and
        a flat background plane at ``main_z_min``.  Returns ``None`` on
        failure.
    """
    from examples.xc_skiing_temporal import (
        build_timeseries,
        build_combined_matrix,
        build_ridge_dem,
    )

    # --- Load data + build ridges ---
    logger.info("Loading temporal XC skiing data...")
    data = build_timeseries(snodas_dir, cache_file, force=False)
    season_matrix, seasons, median_scores, q1_scores, q3_scores = build_combined_matrix(data)

    # --- Auto col_scale: match main mesh's pixel density ---
    # Both meshes use scale_factor=100, so main_width_pixels = main_world_width * 100.
    # For 1 vertex per pixel: col_scale = main_width_pixels / n_days.
    n_days = season_matrix.shape[1]
    if col_scale == 0:
        bpy.context.view_layer.update()
        main_corners = [main_mesh.matrix_world @ Vector(c) for c in main_mesh.bound_box]
        main_w = max(c.x for c in main_corners) - min(c.x for c in main_corners)
        col_scale = max(1, int(round(main_w * 100 / n_days)))
        logger.info(
            "Auto col_scale: main_width=%.1f BU -> %d pixels, col_scale=%d",
            main_w, int(main_w * 100), col_scale,
        )

    # Score upsampling happens inside build_ridge_dem (col_scale param)
    # so every intermediate column gets its own ramp profile — no peak sag.
    dem_ridges, color_ridges = build_ridge_dem(
        season_matrix, median_scores,
        ridge_rows=ridge_rows, gap_rows=gap_rows,
        tail_rows=tail_rows, shear=shear, smooth_sigma=smooth_sigma,
        row_scale=row_scale,
        col_scale=col_scale,
    )
    n_seasons = len(seasons)

    # Convert depth_fraction → internal ns_squash now that DEM shape is known.
    # Final N-S depth = ns_squash × (total_rows / n_cols) × main_width.
    # Solving: ns_squash = depth_fraction × n_cols / total_rows.
    # This makes depth_fraction invariant to col_scale changes.
    _total_rows, _n_cols = dem_ridges.shape
    ns_squash = depth_fraction * _n_cols / _total_rows
    logger.info(
        "depth_fraction=%.2f → ns_squash=%.4f (DEM %d rows × %d cols)",
        depth_fraction, ns_squash, _total_rows, _n_cols,
    )
    logger.info(
        "Temporal data: %d seasons x %d days -> ridge DEM %s (col_scale=%d)",
        n_seasons, n_days, dem_ridges.shape, col_scale,
    )

    # Diagnostic plot (uses pre-upsampled season_matrix for clarity)
    if diagnostic_dir is not None:
        _save_diagnostic(
            season_matrix, seasons, median_scores, q1_scores, q3_scores,
            dem_ridges, color_ridges, diagnostic_dir,
        )

    # Flip rows (N-S) so ridge peaks point away from the main terrain
    # (south in the scene).  Columns are NOT flipped: column 0 = Oct 1
    # already maps to X=0 (west) via the dummy transform.
    dem_ridges = dem_ridges[::-1, :].copy()
    color_ridges = color_ridges[::-1, :].copy()

    # --- NaN-mask between-ridge areas ---
    # Floor/gap cells become NaN → create_mesh skips them entirely.
    # Each ridge becomes an isolated island mesh floating just above the scene
    # background.  boundary_extension=False (below) prevents the boundary
    # algorithm from tracing hundreds of disconnected island loops (which would
    # hang indefinitely).
    dem_max = float(np.max(dem_ridges)) if dem_ridges.size > 0 else 1.0
    threshold = dem_max * 0.005
    no_data = (dem_ridges < threshold) & (color_ridges < threshold)
    dem_ridges[no_data] = np.nan  # excluded from mesh — isolated ridge islands

    # --- Create Terrain with no-op transform ---
    # Dummy transform: 1 unit per pixel, origin at top-left.
    # Scale DEM values up so ridges dominate the base_depth in create_mesh.
    # The z_scale normalization after mesh creation brings the total height
    # to the correct proportion of the main terrain.
    _DEM_SCALE = 1000.0
    dummy_transform = from_origin(0.0, dem_ridges.shape[0], 1.0, 1.0)

    terrain = Terrain(
        dem_ridges * _DEM_SCALE,
        dummy_transform,
        dem_crs="EPSG:32617",
    )
    # No-op transform satisfies create_mesh()'s "transformed" guard
    terrain.add_transform(scale_elevation(scale_factor=1.0))
    terrain.apply_transforms()

    # --- Color mapping: score → boreal_mako colormap ---
    # All surviving cells are ridge cells (NaN cells are excluded by create_mesh).
    # Normalize to use the full colormap range.
    score_max = float(np.max(color_ridges[color_ridges > 0])) if np.any(color_ridges > 0) else 1.0
    logger.info("Temporal color scaling: score_max=%.3f", score_max)

    terrain.add_data_layer(
        "scores", color_ridges, dummy_transform, "EPSG:32617",
        target_layer="dem",
    )
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap(score_colormap)

    def _temporal_color_func(scores, _mx=score_max, _cm=cmap):
        return _cm(np.clip(scores / _mx, 0, 1))[..., :3].astype(np.float32)

    terrain.set_color_mapping(
        color_func=_temporal_color_func,
        source_layers=["scores"],
    )

    # --- Create mesh — isolated ridge islands, no floor geometry ---
    # NaN cells are skipped entirely by create_mesh, leaving each ridge as a
    # disconnected island mesh.  boundary_extension=False is REQUIRED here:
    # with hundreds of disconnected island loops the boundary algorithm would
    # hang indefinitely.  The ridges float just above the scene background.
    mesh = terrain.create_mesh(
        scale_factor=100,
        height_scale=height_scale,
        center_model=True,
        boundary_extension=False,
    )
    if mesh is None:
        logger.warning("Temporal sculpture mesh creation returned None")
        return None  # type: ignore[return-value]

    # --- Scale + position below main terrain ---
    bpy.context.view_layer.update()

    # Measure main mesh world-space extents
    main_corners = [main_mesh.matrix_world @ Vector(c) for c in main_mesh.bound_box]
    main_min_x = min(c.x for c in main_corners)
    main_max_x = max(c.x for c in main_corners)
    main_width = main_max_x - main_min_x
    main_center_x = (main_min_x + main_max_x) / 2
    main_min_y = min(c.y for c in main_corners)
    main_z_range = max(c.z for c in main_corners) - min(c.z for c in main_corners)

    # Measure temporal mesh (before any scaling)
    temp_corners = [mesh.matrix_world @ Vector(c) for c in mesh.bound_box]
    temp_width = max(c.x for c in temp_corners) - min(c.x for c in temp_corners)
    temp_z_range = max(c.z for c in temp_corners) - min(c.z for c in temp_corners)

    # Scale: X to match main terrain width, Y squashed by ns_squash,
    #        Z normalized so peaks = ridge_height_fraction * main terrain Z range
    width_scale = main_width / temp_width if temp_width > 0 else 1.0
    y_scale = width_scale * ns_squash
    z_scale = (main_z_range * ridge_height_fraction) / temp_z_range if temp_z_range > 0 else 1.0
    mesh.scale = (width_scale, y_scale, z_scale)
    bpy.context.view_layer.update()

    logger.info(
        "Temporal sculpture scale: width=%.2f, z=%.2f "
        "(main Z range=%.1f, temporal Z range=%.1f, fraction=%.2f)",
        width_scale, z_scale, main_z_range, temp_z_range, ridge_height_fraction,
    )

    # Re-measure after scaling, position below main mesh
    temp_corners = [mesh.matrix_world @ Vector(c) for c in mesh.bound_box]
    temp_depth = max(c.y for c in temp_corners) - min(c.y for c in temp_corners)
    temp_z_max = max(c.z for c in temp_corners)
    temp_z_min = min(c.z for c in temp_corners)
    # Gap expressed as fraction of main terrain width — independent of temp_depth.
    gap_bu = main_width * gap_fraction
    target_y = main_min_y - gap_bu - temp_depth / 2

    position_mesh_at(mesh, main_center_x, target_y)

    # Align Z: place temporal sculpture so its north edge (the base contact of
    # the northernmost ridge) sits at the main terrain's base Z.
    #
    # We must use the north-edge min Z rather than the global bound_box min Z.
    # Reason: the base-contact row of each ridge is z=0, which gets NaN-masked.
    # The first surviving north-edge row has z ≈ a few × threshold — slightly
    # above zero.  The global min comes from tiny fringe pixels elsewhere in the
    # mesh (also near threshold but potentially different).  Using the global
    # min would leave the north edge hovering above the background plane.
    #
    # Solution: find the min Z among the northernmost 5% of vertices (by Y)
    # and align *that* to main_z_min.
    bpy.context.view_layer.update()
    _bm = bmesh.new()
    _bm.from_mesh(mesh.data)
    _bm.transform(mesh.matrix_world)
    _bm.verts.ensure_lookup_table()
    _all_y = [v.co.y for v in _bm.verts]
    if _all_y:
        _max_y = max(_all_y)
        _y_range = max(_max_y - min(_all_y), 1e-6)
        _north_zs = [v.co.z for v in _bm.verts if v.co.y >= _max_y - _y_range * 0.05]
        north_edge_z_min = min(_north_zs) if _north_zs else temp_z_min
    else:
        north_edge_z_min = temp_z_min
    _bm.free()

    main_z_min = min(c.z for c in main_corners)
    z_offset = main_z_min - north_edge_z_min
    mesh.location.z += z_offset

    bpy.context.view_layer.update()
    logger.info(
        "Temporal sculpture positioned: x=%.1f, y=%.1f, z_offset=%.1f "
        "(gap=%.2f BU = %.0f%% of main width, north_edge_z_min=%.4f vs global_min=%.4f)",
        main_center_x, target_y, z_offset, gap_bu, gap_fraction * 100,
        north_edge_z_min, temp_z_min,
    )

    # Match material to main mesh so lighting/shading is consistent.
    if main_mesh.data.materials and mesh.data.materials:
        mesh.data.materials[0] = main_mesh.data.materials[0]
        logger.info("Temporal mesh material set to match main mesh")

    return mesh, None  # second element reserved; no separate bg plane needed


def _add_background_plane(
    temporal_mesh: "bpy.types.Object",
    z_level: float,
    name: str = "temporal_bg_plane",
) -> "bpy.types.Object":
    """Create a flat dark background plane under the temporal ridge sculpture.

    The plane sits at ``z_level`` (world Z) and spans the full XY extent of
    ``temporal_mesh``.  It fills the empty space between ridges so the
    sculpture looks like terrain sitting on a solid floor rather than floating
    wire-frame islands.

    The plane uses an obsidian-dark Principled BSDF material (no vertex
    colors) so it blends with the background without distracting from the
    ridge colors.

    Parameters
    ----------
    temporal_mesh : bpy.types.Object
        Already-positioned temporal ridge sculpture.
    z_level : float
        World-space Z for the plane (should be ``main_z_min``).
    name : str
        Blender object name for the new plane.

    Returns
    -------
    bpy.types.Object
        The linked background plane object.
    """
    bpy.context.view_layer.update()

    # World-space XY bounds of the temporal mesh
    corners = [temporal_mesh.matrix_world @ Vector(c) for c in temporal_mesh.bound_box]
    x_min = min(c.x for c in corners)
    x_max = max(c.x for c in corners)
    y_min = min(c.y for c in corners)
    y_max = max(c.y for c in corners)

    # Build the plane with vertices at world-space coords (object at origin)
    mesh_data = bpy.data.meshes.new(name)
    bm = bmesh.new()
    v1 = bm.verts.new((x_min, y_min, z_level))
    v2 = bm.verts.new((x_max, y_min, z_level))
    v3 = bm.verts.new((x_max, y_max, z_level))
    v4 = bm.verts.new((x_min, y_max, z_level))
    bm.faces.new([v1, v2, v3, v4])
    bm.to_mesh(mesh_data)
    bm.free()
    mesh_data.update()

    bg_obj = bpy.data.objects.new(name, mesh_data)
    bpy.context.collection.objects.link(bg_obj)

    # Dark Principled BSDF — obsidian-like near-black
    mat = bpy.data.materials.new(f"{name}_mat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    if principled:
        principled.inputs["Base Color"].default_value = (0.011, 0.008, 0.004, 1.0)
        principled.inputs["Roughness"].default_value = 1.0
        try:
            principled.inputs["Specular IOR Level"].default_value = 0.0
        except KeyError:
            try:
                principled.inputs["Specular"].default_value = 0.0
            except KeyError:
                pass
    bg_obj.data.materials.append(mat)

    logger.info(
        "Background plane created: X=[%.1f, %.1f] Y=[%.1f, %.1f] Z=%.3f",
        x_min, x_max, y_min, y_max, z_level,
    )
    return bg_obj


def extend_to_scene_width(
    mesh: "bpy.types.Object",
    main_mesh: "bpy.types.Object",
    scene_meshes: list["bpy.types.Object"],
    bg_plane: "bpy.types.Object | None" = None,
) -> None:
    """Extend temporal mesh X-scale to span the full scene width.

    Stretches the temporal mesh so it runs from the main mesh's west edge
    to the easternmost scene mesh's east edge (typically component panels).
    Y and Z scales are unchanged, only X is adjusted.  If ``bg_plane`` is
    provided it is extended to match the same X range by directly modifying
    its vertex coordinates (the plane has location (0,0,0) so world coords
    == local coords).

    Parameters
    ----------
    mesh : bpy.types.Object
        The temporal sculpture mesh (already scaled and positioned).
    main_mesh : bpy.types.Object
        The main terrain mesh (defines the west edge).
    scene_meshes : list[bpy.types.Object]
        Additional scene meshes (e.g. component panels) whose east extent
        should be matched.
    bg_plane : bpy.types.Object or None
        Optional flat background plane to extend alongside the temporal mesh.
    """
    if not scene_meshes:
        return

    bpy.context.view_layer.update()

    # West edge: main mesh min X
    main_corners = [main_mesh.matrix_world @ Vector(c) for c in main_mesh.bound_box]
    main_min_x = min(c.x for c in main_corners)

    # East edge: max X across main mesh + all scene meshes
    scene_max_x = max(c.x for c in main_corners)
    for obj in scene_meshes:
        corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        obj_max_x = max(c.x for c in corners)
        scene_max_x = max(scene_max_x, obj_max_x)

    target_width = scene_max_x - main_min_x
    target_center_x = (main_min_x + scene_max_x) / 2

    # Measure current temporal mesh width
    temp_corners = [mesh.matrix_world @ Vector(c) for c in mesh.bound_box]
    current_width = max(c.x for c in temp_corners) - min(c.x for c in temp_corners)

    if current_width <= 0 or target_width <= current_width:
        return

    # Scale X to match target width (keep Y and Z unchanged)
    x_ratio = target_width / current_width
    mesh.scale.x *= x_ratio
    bpy.context.view_layer.update()

    # Align west edge to main_min_x (don't assume local BB is centered)
    temp_corners = [mesh.matrix_world @ Vector(c) for c in mesh.bound_box]
    current_min_x = min(c.x for c in temp_corners)
    x_shift = main_min_x - current_min_x
    mesh.location.x += x_shift

    logger.info(
        "Temporal sculpture extended: width %.1f -> %.1f (x_ratio=%.2f), "
        "center_x=%.1f",
        current_width, target_width, x_ratio, target_center_x,
    )

    # Extend the background plane to match the new X range.
    # The plane was created with world-space vertex coords and object at origin,
    # so v.co.x is already in world space.
    if bg_plane is not None:
        bpy.context.view_layer.update()
        temp_corners_final = [mesh.matrix_world @ Vector(c) for c in mesh.bound_box]
        new_min_x = min(c.x for c in temp_corners_final)
        new_max_x = max(c.x for c in temp_corners_final)
        # Get current bg_plane X bounds from vertices
        xs = [v.co.x for v in bg_plane.data.vertices]
        if xs:
            old_min_x = min(xs)
            old_max_x = max(xs)
            for v in bg_plane.data.vertices:
                if abs(v.co.x - old_max_x) < 1e-3:
                    v.co.x = new_max_x
                if abs(v.co.x - old_min_x) < 1e-3:
                    v.co.x = new_min_x
            bg_plane.data.update()
        logger.info(
            "Background plane extended: X=[%.1f, %.1f]",
            new_min_x, new_max_x,
        )


def _save_diagnostic(
    season_matrix: np.ndarray,
    seasons: list[str],
    median_scores: np.ndarray,
    q1_scores: np.ndarray,
    q3_scores: np.ndarray,
    dem_ridges: np.ndarray,
    color_ridges: np.ndarray,
    output_dir: Path,
) -> None:
    """Save a diagnostic plot of the temporal data and ridge DEM."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    output_dir.mkdir(parents=True, exist_ok=True)
    n_days = season_matrix.shape[1]
    days = np.arange(n_days)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Panel 1: All seasons as lines
    ax = axes[0, 0]
    for i, s in enumerate(seasons):
        ax.plot(days, season_matrix[i], alpha=0.4, linewidth=0.8, label=s)
    ax.fill_between(days, q1_scores, q3_scores, alpha=0.25, color="steelblue", label="IQR")
    ax.plot(days, median_scores, color="navy", linewidth=2, label="Median")
    ax.set_xlabel("Day of season (0 = Oct 1)")
    ax.set_ylabel("Combined score")
    ax.set_title("XC Skiing Scores by Season")
    ax.legend(fontsize=6, ncol=3, loc="upper right")

    # Panel 2: Score heatmap (season × day)
    ax = axes[0, 1]
    im = ax.imshow(
        season_matrix, aspect="auto", cmap="mako",
        norm=Normalize(vmin=0, vmax=season_matrix.max()),
    )
    ax.set_xlabel("Day of season")
    ax.set_ylabel("Season")
    ax.set_yticks(range(len(seasons)))
    ax.set_yticklabels(seasons, fontsize=7)
    ax.set_title("Score Heatmap")
    plt.colorbar(im, ax=ax, label="Score")

    # Panel 3: Ridge DEM
    ax = axes[1, 0]
    dem_display = np.where(dem_ridges > 0, dem_ridges, np.nan)
    im2 = ax.imshow(dem_display, aspect="auto", cmap="terrain")
    ax.set_xlabel("Day of season (column)")
    ax.set_ylabel("Row")
    ax.set_title("Ridge DEM (height)")
    plt.colorbar(im2, ax=ax, label="Height")

    # Panel 4: Ridge color map
    ax = axes[1, 1]
    col_display = np.where(color_ridges > 0, color_ridges, np.nan)
    im3 = ax.imshow(col_display, aspect="auto", cmap="mako")
    ax.set_xlabel("Day of season (column)")
    ax.set_ylabel("Row")
    ax.set_title("Ridge Color (score)")
    plt.colorbar(im3, ax=ax, label="Score")

    fig.suptitle("Temporal XC Skiing Ridge Sculpture — Diagnostic", fontsize=14)
    plt.tight_layout()

    out_path = output_dir / "temporal_ridge_diagnostic.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Temporal diagnostic saved to %s", out_path)
