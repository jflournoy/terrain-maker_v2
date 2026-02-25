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

import bpy
import numpy as np
from mathutils import Vector
from rasterio.transform import from_origin

from src.terrain.core import Terrain, scale_elevation
from src.terrain.color_mapping import boreal_mako_cmap
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
    gap_below_main: float = 0.3,
    base_depth: float = 0.2,
    edge_base_material: str = "clay",
    edge_blend_colors: bool = False,
    use_fractional_edges: bool = True,
) -> Optional["bpy.types.Object"]:
    """
    Build and position the temporal XC skiing ridge sculpture.

    Loads temporal data, builds ridge DEM, creates Terrain mesh,
    scales to match main_mesh width, normalizes Z to a fraction of the
    main terrain's Z extent, and positions below main_mesh.

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
    gap_below_main : float
        Gap between main terrain bottom and sculpture top, as a fraction
        of the temporal mesh depth (like component panels use 0.3).
    base_depth : float
        Depth of the flat base/skirt below the surface (passed to create_mesh).
    edge_base_material : str
        Material preset for the skirt base layer (e.g. "clay", "obsidian").
    edge_blend_colors : bool
        Whether to blend surface colors into the edge.
    use_fractional_edges : bool
        Use fractional edge sampling for smooth curved boundaries.

    Returns
    -------
    bpy.types.Object or None
        The positioned Blender mesh object, or None on failure.
    """
    from examples.xc_skiing_temporal import (
        build_timeseries,
        build_combined_matrix,
        build_ridge_dem,
    )

    # --- Load data + build ridges ---
    logger.info("Loading temporal XC skiing data...")
    data = build_timeseries(snodas_dir, cache_file, force=False)
    season_matrix, seasons, median_scores = build_combined_matrix(data)
    dem_ridges, color_ridges = build_ridge_dem(
        season_matrix, median_scores,
        ridge_rows=ridge_rows, gap_rows=gap_rows,
        tail_rows=tail_rows, shear=shear,
    )
    n_seasons = len(seasons)
    logger.info(
        "Temporal data: %d seasons x %d days -> ridge DEM %s",
        n_seasons, dem_ridges.shape[1], dem_ridges.shape,
    )

    # Flip rows (N-S) so ridge peaks point away from the main terrain
    # (south in the scene), and flip columns so earliest calendar days
    # (Nov) are at the west (left) when viewed from above.
    dem_ridges = dem_ridges[::-1, ::-1].copy()
    color_ridges = color_ridges[::-1, ::-1].copy()

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

    # --- Color mapping: score -> boreal_mako, scaled to actual data range ---
    # Actual scores peak around 0.5, so normalize to use the full colormap.
    score_max = float(np.max(color_ridges[color_ridges > 0])) if np.any(color_ridges > 0) else 1.0
    logger.info("Temporal color scaling: score_max=%.3f", score_max)

    terrain.add_data_layer(
        "scores", color_ridges, dummy_transform, "EPSG:32617",
        target_layer="dem",
    )
    terrain.set_color_mapping(
        color_func=lambda scores, _mx=score_max: boreal_mako_cmap(
            np.clip(scores / _mx, 0, 1)
        )[..., :3],
        source_layers=["scores"],
    )

    # --- Create mesh (skirt params match the main terrain) ---
    mesh = terrain.create_mesh(
        scale_factor=100,
        height_scale=height_scale,
        center_model=True,
        boundary_extension=True,
        two_tier_edge=True,
        base_depth=base_depth,
        edge_base_material=edge_base_material,
        edge_blend_colors=edge_blend_colors,
        use_fractional_edges=use_fractional_edges,
        edge_sample_spacing=2.0,
    )
    if mesh is None:
        logger.warning("Temporal sculpture mesh creation returned None")
        return None

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

    # Scale: X/Y to match main terrain width,
    #        Z normalized so peaks = ridge_height_fraction * main terrain Z range
    width_scale = main_width / temp_width if temp_width > 0 else 1.0
    z_scale = (main_z_range * ridge_height_fraction) / temp_z_range if temp_z_range > 0 else 1.0
    mesh.scale = (width_scale, width_scale, z_scale)
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
    # Proportional gap (like component panels: panel_width * 0.3)
    gap_bu = temp_depth * gap_below_main
    target_y = main_min_y - gap_bu - temp_depth / 2

    position_mesh_at(mesh, main_center_x, target_y)

    # Align Z: place temporal sculpture's base at the main terrain's base,
    # so it sits at the same Z level (just south in Y).
    main_z_min = min(c.z for c in main_corners)
    z_offset = main_z_min - temp_z_min
    mesh.location.z += z_offset

    bpy.context.view_layer.update()
    logger.info(
        "Temporal sculpture positioned: x=%.1f, y=%.1f, z_offset=%.1f "
        "(gap=%.1f BU, %.0f%% of depth)",
        main_center_x, target_y, z_offset, gap_bu, gap_below_main * 100,
    )

    return mesh
