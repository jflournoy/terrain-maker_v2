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
    if not hasattr(terrain, "colors") or terrain.colors is None:
        raise ValueError("Terrain colors not computed. Call compute_colors() first.")

    if not hasattr(terrain, "data_layers") or "dem" not in terrain.data_layers:
        raise ValueError("Terrain DEM data layer not found.")

    logger.info("Applying road elevation overlay...")

    # Get DEM data for elevation sampling
    dem_data = terrain.data_layers["dem"].get("transformed_data")
    if dem_data is None:
        dem_data = terrain.data_layers["dem"].get("data")

    if dem_data is None:
        logger.error("Could not get DEM data for elevation sampling")
        return

    dem_shape = dem_data.shape

    # Get the proper WGS84 transform for the downsampled DEM
    # The roads are in WGS84 (EPSG:4326), so we need a WGS84 transform
    # (not the reprojected UTM transform that the DEM layer has after reprojection)
    dem_transform = None

    # Load the WGS84 transform from the score file
    try:
        score_file = Path("docs/images/sledding/sledding_scores.npz")
        if score_file.exists():
            import numpy as np
            from affine import Affine

            loaded = np.load(score_file)
            if "transform" in loaded:
                # Score file has the proper WGS84 transform matching the downsampled grid
                transform_array = loaded["transform"]
                a, b, c, d, e, f = transform_array
                dem_transform = Affine(a, b, c, d, e, f)
                logger.info(f"  Using WGS84 transform from score file")
    except Exception as e:
        logger.warning(f"Could not load transform from score file: {e}")

    if dem_transform is None:
        logger.error("Could not get WGS84 transform for road rasterization")
        return

    # Rasterize roads to grid mask
    road_mask, road_widths = rasterize_roads_to_mask(
        roads_geojson, dem_shape, dem_transform, dem_crs
    )

    if not np.any(road_mask):
        logger.warning("No roads were rasterized to the terrain grid")
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


# =============================================================================
# DATA LAYER API (NEW)
# =============================================================================


def rasterize_roads_to_layer(
    roads_geojson: Dict[str, Any],
    bbox: Tuple[float, float, float, float],
    resolution: float = 30.0,
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
                _draw_line_on_layer(road_grid, x0, y0, x1, y1, road_value, (height, width))

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
) -> None:
    """
    Draw a line on a raster grid using Bresenham algorithm.

    Args:
        grid: 2D array to draw on
        x0, y0, x1, y1: line endpoints in pixel coordinates
        value: value to write to grid
        shape: (height, width) of grid
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

    while True:
        # Mark pixel
        if 0 <= y < height and 0 <= x < width:
            # Keep highest road type value at this pixel
            grid[y, x] = max(grid[y, x], value)

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
    colormap_name: str = "viridis",
    resolution: float = 30.0,
) -> None:
    """
    Add roads as a data layer to terrain with automatic coordinate alignment.

    This is the high-level API for road integration. Roads are rasterized to
    a grid, added as a data layer, and then used to color vertices. The library's
    data layer pipeline ensures proper alignment even if terrain is downsampled
    or reprojected.

    Args:
        terrain: Terrain object (must have colors computed)
        roads_geojson: GeoJSON FeatureCollection with road LineStrings
        bbox: Bounding box as (south, west, north, east) in WGS84 degrees
        colormap_name: Matplotlib colormap for road coloring (default: 'viridis')
        resolution: Pixel size in meters for rasterization (default: 30.0)

    Raises:
        ValueError: If terrain missing required attributes
    """
    if not hasattr(terrain, "colors") or terrain.colors is None:
        raise ValueError("Terrain colors must be computed first. Call compute_colors().")

    if not hasattr(terrain, "data_layers") or "dem" not in terrain.data_layers:
        raise ValueError("Terrain DEM data layer not found.")

    logger.info("Adding roads as data layer...")

    # Rasterize roads to grid with WGS84 transform
    road_grid, road_transform = rasterize_roads_to_layer(roads_geojson, bbox, resolution)

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

    # Now apply road colors to vertices using the aligned road layer
    _color_vertices_from_road_layer(terrain, colormap_name)


def _color_vertices_from_road_layer(
    terrain,
    colormap_name: str = "viridis",
) -> None:
    """
    Color vertices based on road layer values.

    Uses the transformed (downsampled/reprojected) road layer to color terrain
    vertices. Road pixels are colored using a colormap based on road type.

    Args:
        terrain: Terrain object with aligned road layer and computed colors
        colormap_name: Matplotlib colormap name
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
