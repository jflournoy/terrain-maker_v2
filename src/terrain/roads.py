"""
Road network visualization for terrain rendering - Rasterized approach.

This module provides functions to render road networks by rasterizing them onto
the terrain grid and coloring them based on elevation, integrated as a
post-processing step after terrain color computation.

Much more efficient than creating individual Blender objects - rasterizes 8922
roads in ~5 seconds instead of 20+ minutes.

Usage:
    from src.terrain.roads import apply_road_elevation_overlay
    from examples.detroit_roads import get_roads

    # After terrain colors are computed:
    roads_geojson = get_roads(bbox)
    apply_road_elevation_overlay(
        terrain=terrain,
        roads_geojson=roads_geojson,
        dem_crs="EPSG:32617",
        colormap_name="viridis",
    )
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import sys

logger = logging.getLogger(__name__)

# Road widths in pixels for different OSM highway types
ROAD_WIDTHS = {
    'motorway': 5,      # ~150m at 30m resolution
    'trunk': 3,         # ~90m
    'primary': 2,       # ~60m
    'secondary': 1,     # ~30m (minimal)
}


# =============================================================================
# ROAD RASTERIZATION
# =============================================================================


def rasterize_roads_to_mask(
    roads_geojson: Dict[str, Any],
    dem_shape: Tuple[int, int],
    dem_transform,
    dem_crs: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rasterize GeoJSON roads onto a grid mask.

    Args:
        roads_geojson: GeoJSON FeatureCollection with road LineStrings
        dem_shape: (height, width) of DEM grid
        dem_transform: Affine transform for DEM grid (assumes WGS84 coordinates)
        dem_crs: CRS of DEM (used for validation, roads assumed to be in WGS84)

    Returns:
        Tuple of:
        - road_mask: bool array (height, width), True where roads exist
        - road_widths: int array (height, width), road width in pixels (0 if no road)
    """
    height, width = dem_shape
    road_mask = np.zeros(dem_shape, dtype=bool)
    road_widths = np.zeros(dem_shape, dtype=int)

    logger.info(f"Rasterizing {len(roads_geojson.get('features', []))} road segments...")
    logger.info(f"  DEM shape: {dem_shape}")
    logger.info(f"  Transform: {dem_transform}")
    if hasattr(dem_transform, 'c'):
        logger.info(f"    Origin: ({dem_transform.c}, {dem_transform.f})")
        logger.info(f"    Scale: ({dem_transform.a}, {dem_transform.e})")
    else:
        logger.info(f"    Transform has no affine attributes")

    processed = 0
    skipped = 0
    for feature in roads_geojson.get("features", []):
        try:
            geometry = feature.get("geometry", {})
            if geometry.get("type") != "LineString":
                skipped += 1
                continue

            # Get road type and width
            highway_type = feature.get("properties", {}).get("highway", "primary")
            width = ROAD_WIDTHS.get(highway_type, 1)

            # Extract coordinates (WGS84 lon/lat - no transformation needed)
            coords = geometry.get("coordinates", [])
            if len(coords) < 2:
                skipped += 1
                continue

            # Convert WGS84 coordinates directly to pixel coordinates
            # dem_transform maps: pixel (col, row) -> (lon, lat)
            # Invert: (lon, lat) -> (col, row)
            road_pixels = []
            for lon, lat in coords:
                try:
                    col = (lon - dem_transform.c) / dem_transform.a
                    row = (lat - dem_transform.f) / dem_transform.e
                    road_pixels.append((col, row))
                except Exception:
                    continue

            if len(road_pixels) < 2:
                skipped += 1
                continue

            # Rasterize line with width using Bresenham-like algorithm
            for i in range(len(road_pixels) - 1):
                x0, y0 = road_pixels[i]
                x1, y1 = road_pixels[i + 1]

                # Draw line segment with thickness
                _draw_line_with_width(road_mask, road_widths, x0, y0, x1, y1, width, dem_shape)

            processed += 1
            if processed % 1000 == 0:
                logger.debug(f"  Rasterized {processed} road segments...")

        except Exception as e:
            logger.debug(f"Failed to rasterize road feature: {e}")
            skipped += 1
            continue

    logger.info(f"  Rasterized {processed}/{len(roads_geojson.get('features', []))} road segments to grid ({skipped} skipped)")
    return road_mask, road_widths


def _draw_line_with_width(
    mask: np.ndarray,
    widths: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    line_width: int,
    shape: Tuple[int, int],
) -> None:
    """
    Draw a line segment with thickness on a grid using Bresenham algorithm.

    Args:
        mask: boolean grid to mark road pixels
        widths: grid to store road width for each pixel
        x0, y0, x1, y1: line endpoints in pixel coordinates
        line_width: thickness in pixels
        shape: (height, width) of grid
    """
    height, width = shape

    # Bresenham line algorithm
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    x, y = x0, y0
    half_width = line_width // 2

    while True:
        # Mark pixels around this point
        for px in range(max(0, x - half_width), min(width, x + half_width + 1)):
            for py in range(max(0, y - half_width), min(height, y + half_width + 1)):
                mask[py, px] = True
                widths[py, px] = max(widths[py, px], line_width)

        if x == x1 and y == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


# =============================================================================
# COLOR APPLICATION
# =============================================================================


def get_viridis_colormap():
    """
    Get viridis colormap function.

    Returns:
        Function that maps normalized values (0-1) to (R, G, B)
    """
    try:
        from matplotlib import cm
        viridis = cm.get_cmap('viridis')
        return lambda x: viridis(np.clip(x, 0, 1))[:3]
    except ImportError:
        logger.warning("matplotlib not available, using simple grayscale colormap")
        return lambda x: (x, x, x)


def apply_road_elevation_overlay(
    terrain,
    roads_geojson: Dict[str, Any],
    dem_crs: str = "EPSG:4326",
    colormap_name: str = "viridis",
) -> None:
    """
    Apply roads as elevation-colored overlay to terrain.

    Rasterizes roads to DEM grid, samples elevation at road pixels,
    applies colormap, and replaces vertex colors where roads exist.

    Note: Roads are assumed to be in WGS84 (EPSG:4326). The DEM transform
    must also be in WGS84 for correct coordinate mapping.

    Args:
        terrain: Terrain object with computed colors and vertices
        roads_geojson: GeoJSON FeatureCollection with road LineStrings (WGS84)
        dem_crs: CRS of DEM (for reference; roads/DEM must share same CRS)
        colormap_name: matplotlib colormap name ('viridis', 'plasma', 'inferno', etc.)

    Raises:
        ValueError: If terrain missing required attributes
    """
    sys.stderr.write(f"[ROADS] apply_road_elevation_overlay called with {len(roads_geojson.get('features', []))} roads\n")
    sys.stderr.flush()

    if not hasattr(terrain, "colors") or terrain.colors is None:
        sys.stderr.write(f"[ROADS] ERROR: colors not computed\n")
        sys.stderr.flush()
        raise ValueError("Terrain colors not computed. Call compute_colors() first.")

    if not hasattr(terrain, "data_layers") or "dem" not in terrain.data_layers:
        sys.stderr.write(f"[ROADS] ERROR: dem not in data_layers\n")
        sys.stderr.flush()
        raise ValueError("Terrain DEM data layer not found.")

    sys.stderr.write(f"[ROADS] Starting road rasterization...\n")
    sys.stderr.flush()
    logger.info("Applying road elevation overlay...")

    # Get DEM data for elevation sampling
    dem_data = terrain.data_layers["dem"].get("transformed_data")
    logger.info(f"  Transformed DEM data: {dem_data.shape if dem_data is not None else 'None'}")

    if dem_data is None:
        dem_data = terrain.data_layers["dem"].get("data")
        logger.info(f"  Using original DEM data: {dem_data.shape if dem_data is not None else 'None'}")

    if dem_data is None:
        logger.error("Could not get DEM data for elevation sampling")
        return

    dem_shape = dem_data.shape

    # Get the proper WGS84 transform for the downsampled DEM
    # Construct from the DEM shape and known WGS84 bounds
    # The roads are in WGS84 (EPSG:4326), so we need a WGS84 transform
    dem_transform = None

    # Try to get WGS84 transform from DEM layer's original data
    dem_layer = terrain.data_layers.get("dem", {})

    # Try to read the score file to get the proper WGS84 transform
    try:
        from pathlib import Path
        score_file = Path("docs/images/sledding/sledding_scores.npz")
        logger.info(f"  Looking for score file: {score_file}")
        if score_file.exists():
            import numpy as np
            from affine import Affine

            logger.info(f"  Loading transform from score file...")
            loaded = np.load(score_file)
            if "transform" in loaded:
                # Score file has the proper WGS84 transform
                transform_array = loaded["transform"]
                a, b, c, d, e, f = transform_array
                # Affine(a, b, c, d, e, f)
                dem_transform = Affine(a, b, c, d, e, f)
                logger.info(f"  ✓ Loaded WGS84 transform from score file: a={a:.8f}, e={e:.8f}")
            else:
                logger.warning(f"  Score file has no transform array")
        else:
            logger.warning(f"  Score file not found")
    except Exception as e:
        logger.error(f"Could not load transform from score file: {e}", exc_info=True)

    if dem_transform is None:
        logger.error("Could not get WGS84 transform for road rasterization")
        return

    logger.debug(f"DEM data shape: {dem_shape}")
    logger.debug(f"DEM transform: {dem_transform}")

    # Rasterize roads to grid mask
    road_mask, road_widths = rasterize_roads_to_mask(
        roads_geojson, dem_shape, dem_transform, dem_crs
    )

    if not np.any(road_mask):
        logger.warning(f"No roads were rasterized to the terrain grid")
        logger.warning(f"  DEM shape: {dem_shape}")
        logger.warning(f"  DEM transform: {dem_transform}")
        if hasattr(dem_transform, 'c'):
            logger.warning(f"    Bounds: lon [{dem_transform.c}, {dem_transform.c + dem_shape[1] * dem_transform.a}], lat [{dem_transform.f + dem_shape[0] * dem_transform.e}, {dem_transform.f}]")
        return

    # Get elevation values at road pixels
    road_elevations = dem_data[road_mask]
    elev_min = np.nanmin(dem_data)
    elev_max = np.nanmax(dem_data)

    # Normalize elevations to 0-1 range
    if elev_max > elev_min:
        elev_norm = (road_elevations - elev_min) / (elev_max - elev_min)
    else:
        elev_norm = np.ones_like(road_elevations) * 0.5

    # Apply colormap
    logger.debug(f"Applying {colormap_name} colormap to {len(road_elevations)} road pixels...")
    colormap_func = get_viridis_colormap()
    road_colors = np.array([colormap_func(val) for val in elev_norm])

    # Ensure RGB values are normalized to 0-1
    road_colors = np.clip(road_colors, 0, 1)

    # Map grid pixels to vertex indices and update colors
    logger.debug("Mapping road pixels to terrain vertices...")
    _apply_road_colors_to_vertices(
        terrain, road_mask, road_colors, dem_shape
    )

    logger.info(f"✓ Road overlay applied ({np.sum(road_mask)} pixels colored)")


def _apply_road_colors_to_vertices(
    terrain,
    road_mask: np.ndarray,
    road_colors: np.ndarray,
    dem_shape: Tuple[int, int],
) -> None:
    """
    Apply road colors to terrain vertex colors.

    Args:
        terrain: Terrain object with colors and vertices
        road_mask: Boolean mask of road pixels on DEM grid
        road_colors: RGB colors for road pixels
        dem_shape: DEM grid shape
    """
    if not hasattr(terrain, "y_valid") or not hasattr(terrain, "x_valid"):
        logger.warning("Terrain missing y_valid/x_valid indices, cannot map pixels to vertices")
        return

    # Get valid pixel indices
    y_valid = terrain.y_valid
    x_valid = terrain.x_valid

    if y_valid is None or x_valid is None:
        logger.warning("y_valid or x_valid is None")
        return

    # Create mapping from pixel coords to vertex index
    height, width = dem_shape
    pixel_to_vertex = np.full((height, width), -1, dtype=int)
    for vertex_idx in range(len(y_valid)):
        y, x = y_valid[vertex_idx], x_valid[vertex_idx]
        if 0 <= y < height and 0 <= x < width:
            pixel_to_vertex[y, x] = vertex_idx

    # Apply road colors to corresponding vertices
    road_y_coords, road_x_coords = np.where(road_mask)
    color_idx = 0

    for y, x in zip(road_y_coords, road_x_coords):
        vertex_idx = pixel_to_vertex[y, x]
        if vertex_idx >= 0 and vertex_idx < len(terrain.colors):
            # Replace RGB, keep alpha
            terrain.colors[vertex_idx, :3] = road_colors[color_idx]
        color_idx += 1

    logger.debug(f"Updated colors for vertices under roads")
