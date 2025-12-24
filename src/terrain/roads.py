"""
Road network visualization for terrain rendering using data layer pipeline.

This module provides functions to render road networks by rasterizing them as
a data layer that flows through the same transformation pipeline as elevation
and score data. Roads are treated as a proper geographic data layer with
coordinate system and transform, ensuring correct alignment when downsampling
or reprojecting.

Much more efficient than creating individual Blender objects - rasterizes
roads in ~5 seconds and automatically handles coordinate transformations.

Usage - Simple API:
    from src.terrain.roads import add_roads_layer
    from examples.detroit_roads import get_roads

    # After creating terrain:
    roads_geojson = get_roads(bbox)
    add_roads_layer(
        terrain=terrain,
        roads_geojson=roads_geojson,
        bbox=bbox,  # (south, west, north, east) in WGS84
        colormap_name="viridis",
    )

Usage - Manual pipeline:
    from src.terrain.roads import rasterize_roads_to_layer
    from examples.detroit_roads import get_roads

    roads_geojson = get_roads(bbox)
    road_grid, road_transform = rasterize_roads_to_layer(
        roads_geojson, bbox, resolution=30  # 30m pixels
    )

    # Add as data layer (library handles transform automatically)
    terrain.add_data_layer(
        "roads",
        road_grid,
        road_transform,
        "EPSG:4326",
        target_layer="dem",  # Align to DEM grid
    )
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


def road_colormap(road_grid, score=None):
    """
    Map roads to colors using inverse mako colormap on score values.

    Roads are colored using the inverse of the underlying score (1 - score)
    with the mako colormap. This creates visual contrast - roads appear as
    the opposite color of the surrounding terrain.

    Args:
        road_grid: 2D array of road values (0=no road, >0=road)
        score: 2D array of score values (same shape as road_grid), typically
            sledding scores. If None, falls back to coloring by road type.

    Returns:
        Array of RGB colors with shape (height, width, 3) as uint8
    """
    import matplotlib.pyplot as plt

    # Create road mask (any road value > 0)
    road_mask = road_grid > 0.5  # Use 0.5 threshold to handle resampling artifacts

    height, width = road_grid.shape
    colors = np.zeros((height, width, 3), dtype=np.uint8)

    if score is not None and np.any(road_mask):
        # Color roads by inverse score using mako (same colormap as terrain)
        # Apply same gamma correction as terrain (gamma=0.5), then invert
        mako_cmap = plt.colormaps.get_cmap('mako')

        # Apply gamma correction like terrain does
        score_clipped = np.clip(score, 0, 1)
        score_gamma = np.power(score_clipped, 0.5)
        # Normalize to terrain's range (0 to sqrt(1.5) ≈ 1.22)
        max_val = np.power(1.5, 0.5)
        score_normalized = score_gamma / max_val

        # Invert: high score terrain = low value roads (dark), low score = high (bright)
        inverse_score = 1.0 - score_normalized

        # Apply mako to all pixels, then mask
        all_colors = mako_cmap(inverse_score)[:, :, :3]  # RGB only
        colors[road_mask] = (all_colors[road_mask] * 255).astype(np.uint8)
    else:
        # Fallback: color by road type using mako
        mako_cmap = plt.colormaps.get_cmap('mako')
        road_grid_quantized = np.clip(np.round(road_grid).astype(int), 0, 4)

        # Map road types across mako range
        road_colors_rgb = {
            1: tuple(int(c * 255) for c in mako_cmap(0.3)[:3]),
            2: tuple(int(c * 255) for c in mako_cmap(0.5)[:3]),
            3: tuple(int(c * 255) for c in mako_cmap(0.7)[:3]),
            4: tuple(int(c * 255) for c in mako_cmap(0.9)[:3]),
        }

        for road_value, rgb in road_colors_rgb.items():
            mask = road_grid_quantized == road_value
            colors[mask] = rgb

    return colors


def get_viridis_colormap():
    """
    Get viridis colormap function.

    Returns:
        Function that maps normalized values (0-1) to (R, G, B)
    """
    try:
        import matplotlib.pyplot as plt
        viridis = plt.colormaps.get_cmap('viridis')
        return lambda x: viridis(np.clip(x, 0, 1))[:3]
    except ImportError:
        logger.warning("matplotlib not available, using simple grayscale colormap")
        return lambda x: (x, x, x)


# =============================================================================
# DATA LAYER API
# =============================================================================


def rasterize_roads_to_layer(
    roads_geojson: Dict[str, Any],
    bbox: Tuple[float, float, float, float],
    resolution: float = 30.0,
    road_width_pixels: int = 3,
) -> Tuple[np.ndarray, Any]:
    """
    Rasterize GeoJSON roads to a layer grid with proper geographic transform.

    Converts vector road data (GeoJSON LineStrings) to a raster grid where each
    pixel represents road presence/type. The result includes an Affine transform
    in WGS84 (EPSG:4326) coordinates for proper geographic alignment.

    This is the key function that treats roads as a data layer - the output
    can be added to terrain via add_data_layer() and will automatically align
    to the DEM through reprojection and resampling.

    Args:
        roads_geojson: GeoJSON FeatureCollection with road LineStrings
        bbox: Bounding box as (south, west, north, east) in WGS84 degrees
        resolution: Pixel size in meters (default: 30.0). At Detroit latitude,
            ~30m/pixel gives good detail without excessive memory use.
        road_width_pixels: Width of roads in raster pixels (default: 3). Draws roads
            with thickness instead of 1-pixel lines. Use 1 for thin roads, 3-5 for
            more visible roads. At 30m resolution, 3 pixels ≈ 90m visual width.

    Returns:
        Tuple of:
        - road_grid: 2D array of uint8, values 0=no road, 1-4=road type (motorway > trunk > primary > secondary)
        - road_transform: rasterio.Affine transform mapping pixels to WGS84 coordinates
    """
    from rasterio import Affine

    south, west, north, east = bbox

    # Calculate grid dimensions based on resolution
    # Convert meters to degrees (1 degree ≈ 111 km at equator, ~90 km at 45°N)
    # Detroit is at ~42°N where 1 degree ≈ 84 km
    meters_per_degree = 111000 * np.cos(np.radians((north + south) / 2))
    pixel_size_degrees = resolution / meters_per_degree

    # Calculate grid dimensions
    width = int(np.ceil((east - west) / pixel_size_degrees))
    height = int(np.ceil((north - south) / pixel_size_degrees))

    logger.info(
        f"Rasterizing roads to grid: {width}×{height} pixels "
        f"({resolution}m resolution, bbox={bbox})"
    )

    # Create output grid
    road_grid = np.zeros((height, width), dtype=np.uint8)

    # Create transform: maps pixel (col, row) to geographic (lon, lat)
    # Note: standard raster convention has Y decreasing (north-positive becomes row increasing)
    road_transform = Affine(
        pixel_size_degrees, 0, west,        # pixel_width, skew_x, origin_x
        0, -pixel_size_degrees, north       # skew_y, pixel_height (negative = north-up), origin_y
    )

    logger.debug(f"Road transform: {road_transform}")

    # Rasterize each road feature
    processed = 0
    skipped = 0

    for feature in roads_geojson.get("features", []):
        try:
            geometry = feature.get("geometry", {})
            if geometry.get("type") != "LineString":
                skipped += 1
                continue

            # Get road type and priority
            highway_type = feature.get("properties", {}).get("highway", "primary")
            # Encode road type as uint8: motorway=4, trunk=3, primary=2, secondary=1, other=1
            road_value = {
                'motorway': 4,
                'trunk': 3,
                'primary': 2,
                'secondary': 1,
            }.get(highway_type, 1)

            # Extract coordinates (WGS84 lon/lat)
            coords = geometry.get("coordinates", [])
            if len(coords) < 2:
                skipped += 1
                continue

            # Convert geographic coords to pixel coordinates
            road_pixels = []
            for lon, lat in coords:
                # Check bounds
                if not (west <= lon <= east and south <= lat <= north):
                    continue

                # Map WGS84 to pixel space
                col = (lon - west) / pixel_size_degrees
                row = (north - lat) / pixel_size_degrees  # North-up: lat decreases = row increases

                road_pixels.append((col, row))

            if len(road_pixels) < 2:
                skipped += 1
                continue

            # Draw line with width using Bresenham algorithm
            for i in range(len(road_pixels) - 1):
                x0, y0 = road_pixels[i]
                x1, y1 = road_pixels[i + 1]
                _draw_line_on_layer(road_grid, x0, y0, x1, y1, road_value, (height, width), road_width_pixels)

            processed += 1
            if processed % 1000 == 0:
                logger.debug(f"  Rasterized {processed} road segments...")

        except Exception as e:
            logger.debug(f"Failed to rasterize road feature: {e}")
            skipped += 1
            continue

    logger.info(
        f"Rasterized {processed}/{len(roads_geojson.get('features', []))} road segments "
        f"({skipped} skipped, {np.count_nonzero(road_grid)} pixels with roads)"
    )

    return road_grid, road_transform


def _draw_line_on_layer(
    grid: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    value: int,
    shape: Tuple[int, int],
    line_width: int = 1,
) -> None:
    """
    Draw a thick line on a raster grid using Bresenham algorithm.

    Args:
        grid: 2D array to draw on
        x0, y0, x1, y1: line endpoints in pixel coordinates
        value: value to write to grid
        shape: (height, width) of grid
        line_width: thickness of line in pixels (default: 1). For width N,
            draws a line with perpendicular thickness of N pixels.
    """
    height, width = shape

    # Convert to integers
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

    # Bresenham line algorithm
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    x, y = x0, y0
    half_width = line_width // 2

    while True:
        # Mark pixels in a cross/box pattern for thickness
        if line_width == 1:
            # Single pixel - fast path
            if 0 <= y < height and 0 <= x < width:
                grid[y, x] = max(grid[y, x], value)
        else:
            # Draw thick line using numpy slicing (vectorized)
            # Calculate bounds with clipping to grid
            y_start = max(0, y - half_width)
            y_end = min(height, y + half_width + 1)
            x_start = max(0, x - half_width)
            x_end = min(width, x + half_width + 1)
            if y_start < y_end and x_start < x_end:
                grid[y_start:y_end, x_start:x_end] = np.maximum(
                    grid[y_start:y_end, x_start:x_end], value
                )

        if x == x1 and y == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def add_roads_layer(
    terrain,
    roads_geojson: Dict[str, Any],
    bbox: Tuple[float, float, float, float],
    resolution: float = 30.0,
    road_width_pixels: int = 3,
) -> None:
    """
    Add roads as a data layer to terrain with automatic coordinate alignment.

    This is the high-level API for road integration. Roads are rasterized to
    a grid with proper geographic metadata and added as a data layer. The library's
    data layer pipeline ensures proper alignment even if terrain is downsampled
    or reprojected.

    To color roads, use the multi-overlay color mapping system:

        roads_geojson = get_roads(bbox)
        terrain.add_roads_layer(terrain, roads_geojson, bbox, road_width_pixels=3)
        terrain.set_multi_color_mapping(
            base_colormap=lambda dem: elevation_colormap(dem, 'michigan'),
            base_source_layers=['dem'],
            overlays=[{
                'colormap': road_colormap,
                'source_layers': ['roads'],
                'priority': 10,
            }],
        )
        terrain.compute_colors()

    Args:
        terrain: Terrain object (must have DEM data layer)
        roads_geojson: GeoJSON FeatureCollection with road LineStrings
        bbox: Bounding box as (south, west, north, east) in WGS84 degrees
        resolution: Pixel size in meters for rasterization (default: 30.0)
        road_width_pixels: Width of roads in raster pixels (default: 3). Higher values
            make roads more visible. At 30m resolution, 3 pixels ≈ 90m visual width.

    Raises:
        ValueError: If terrain missing DEM data layer
    """
    if not hasattr(terrain, "data_layers") or "dem" not in terrain.data_layers:
        raise ValueError("Terrain DEM data layer not found.")

    logger.info("Adding roads as data layer...")

    # Rasterize roads to grid with WGS84 transform
    road_grid, road_transform = rasterize_roads_to_layer(
        roads_geojson, bbox, resolution, road_width_pixels
    )

    if not np.any(road_grid):
        logger.warning("No roads were rasterized, skipping road layer")
        return

    # Add as data layer - library handles coordinate alignment automatically
    # The target_layer="dem" ensures roads are reprojected to match DEM's grid
    try:
        terrain.add_data_layer(
            "roads",
            road_grid.astype(np.float32),  # Convert to float for compatibility
            road_transform,
            "EPSG:4326",
            target_layer="dem",
        )
        logger.info(f"✓ Roads added as data layer, aligned to DEM grid")
    except Exception as e:
        logger.error(f"Failed to add roads data layer: {e}")
        raise


def _color_vertices_from_road_layer(
    terrain,
    colormap_name: str = "viridis",
) -> None:
    """
    Color vertices based on road layer values (DEPRECATED).

    This function is deprecated. Use the multi-overlay color mapping system instead:

        terrain.set_multi_color_mapping(
            base_colormap=...,
            base_source_layers=['dem'],
            overlays=[{
                'colormap': road_colormap,
                'source_layers': ['roads'],
                'priority': 10,
            }],
        )
        terrain.compute_colors()

    Uses the transformed (downsampled/reprojected) road layer to color terrain
    vertices. Road pixels are colored using a colormap based on road type.

    Args:
        terrain: Terrain object with aligned road layer and computed colors
        colormap_name: Matplotlib colormap name (deprecated)
    """
    if "roads" not in terrain.data_layers:
        logger.warning("Roads layer not found in terrain")
        return

    roads_info = terrain.data_layers["roads"]

    # Get the aligned/transformed road data (after reprojection to DEM grid)
    # The add_data_layer function stores reprojected data in the layer
    if roads_info.get("transformed", False) and "transformed_data" in roads_info:
        road_data = roads_info["transformed_data"]
        logger.info("Using transformed (aligned) road layer data")
    else:
        # Fall back to original if not yet transformed
        road_data = roads_info.get("data")
        logger.info("Using original road layer data")

    if road_data is None:
        logger.warning("No road data available for coloring")
        return

    if not hasattr(terrain, "y_valid") or not hasattr(terrain, "x_valid"):
        logger.warning("Terrain missing y_valid/x_valid indices")
        return

    # Get valid pixel indices
    y_valid = terrain.y_valid
    x_valid = terrain.x_valid

    if y_valid is None or x_valid is None:
        logger.warning("y_valid or x_valid is None")
        return

    # Map road values to colors
    # Road values: 0=no road, 1-4=road types
    road_colormap = get_viridis_colormap()

    # Normalize road values (1-4) to colormap range (0-1)
    # Assign colors: motorway=0.8 (red), trunk=0.6 (orange), primary=0.4 (yellow), secondary=0.2 (green)
    road_colors_map = {
        0: None,  # No road - don't color
        1: 0.2,   # Secondary - green
        2: 0.4,   # Primary - yellow
        3: 0.6,   # Trunk - orange
        4: 0.8,   # Motorway - red
    }

    height, width = road_data.shape
    road_height, road_width = road_data.shape

    logger.info(
        f"Coloring vertices from road layer ({road_height}×{road_width}), "
        f"{len(y_valid)} valid terrain vertices"
    )

    colored_count = 0

    # For each valid vertex, check if it's on a road and apply color
    for vertex_idx in range(len(y_valid)):
        y, x = y_valid[vertex_idx], x_valid[vertex_idx]

        # Bounds check
        if not (0 <= y < road_height and 0 <= x < road_width):
            continue

        road_value = int(road_data[y, x])
        if road_value == 0:
            # No road at this location
            continue

        # Get color for this road type
        color_norm = road_colors_map.get(road_value, None)
        if color_norm is None:
            continue

        # Apply colormap
        road_color = road_colormap(color_norm)

        # Update vertex color (keep alpha)
        if vertex_idx < len(terrain.colors):
            terrain.colors[vertex_idx, :3] = road_color
            colored_count += 1

    logger.info(f"✓ Colored {colored_count} vertices on roads")
