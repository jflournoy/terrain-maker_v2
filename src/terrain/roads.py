"""
Road network visualization for terrain rendering using data layer pipeline.

This module provides functions to:
1. Fetch road data from OpenStreetMap Overpass API (with caching)
2. Rasterize roads as a data layer for terrain visualization
3. Apply roads through the transformation pipeline (reproject, downsample)

Roads are treated as a proper geographic data layer with coordinate system
and transform, ensuring correct alignment when downsampling or reprojecting.

Usage - Simple API:
    from src.terrain.roads import get_roads, add_roads_layer

    # Fetch roads for a bounding box (cached for 30 days)
    bbox = (32.5, -117.6, 33.5, -116.0)  # (south, west, north, east)
    roads_geojson = get_roads(bbox, road_types=['motorway', 'trunk', 'primary'])

    # Add to terrain
    add_roads_layer(terrain, roads_geojson, bbox)

Usage - Large areas (tiled fetching):
    from src.terrain.roads import get_roads_tiled

    # Automatically tiles large areas to respect API limits
    bbox = (37.0, -95.0, 45.0, -85.0)  # Large multi-state area
    roads = get_roads_tiled(bbox, tile_size=2.0)

Usage - Manual pipeline:
    from src.terrain.roads import get_roads, rasterize_roads_to_layer

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

import hashlib
import json
import logging
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


# =============================================================================
# ROAD DATA FETCHING (OpenStreetMap Overpass API)
# =============================================================================


def build_overpass_query(
    bbox: Tuple[float, float, float, float],
    road_types: List[str],
) -> str:
    """
    Build Overpass QL query for road network data.

    Args:
        bbox: (south, west, north, east) in WGS84 (EPSG:4326)
        road_types: List of OSM highway tags ('motorway', 'trunk', 'primary')

    Returns:
        Overpass QL query string
    """
    south, west, north, east = bbox
    bbox_str = f"{south},{west},{north},{east}"

    # Build query for each road type
    way_queries = []
    for road_type in road_types:
        way_queries.append(f'  way["highway"="{road_type}"]({bbox_str});')

    way_clauses = "\n".join(way_queries)

    query = f"""
[out:json][timeout:60];
(
{way_clauses}
);
out geom;
"""
    return query


def _compute_bbox_hash(bbox: Tuple[float, float, float, float]) -> str:
    """
    Compute hash for bounding box (used for cache keys).

    Args:
        bbox: (south, west, north, east)

    Returns:
        SHA256 hash of bbox coordinates
    """
    bbox_str = f"{bbox[0]:.6f},{bbox[1]:.6f},{bbox[2]:.6f},{bbox[3]:.6f}"
    return hashlib.sha256(bbox_str.encode()).hexdigest()


def _get_cache_dir() -> Path:
    """
    Get or create cache directory for road data.

    Returns:
        Path to cache directory (data/cache/roads/)
    """
    cache_dir = Path.cwd() / "data" / "cache" / "roads"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _load_cached_roads(
    bbox: Tuple[float, float, float, float],
) -> Optional[Dict[str, Any]]:
    """
    Load road data from cache if it exists and is fresh.

    Args:
        bbox: (south, west, north, east)

    Returns:
        GeoJSON dict if cache exists and is fresh (< 30 days), None otherwise
    """
    bbox_hash = _compute_bbox_hash(bbox)
    cache_dir = _get_cache_dir()
    cache_file = cache_dir / f"roads_{bbox_hash}.geojson"
    meta_file = cache_dir / f"roads_{bbox_hash}_meta.json"

    if not cache_file.exists() or not meta_file.exists():
        logger.debug(f"  No cache found for bbox {bbox_hash[:8]}...")
        return None

    # Check cache expiration (30 days)
    try:
        with open(meta_file) as f:
            meta = json.load(f)

        created_time = datetime.fromisoformat(meta.get("created_at", ""))
        age = datetime.now() - created_time
        max_age = timedelta(days=30)

        if age > max_age:
            logger.debug(f"  Cache expired ({age.days} days old)")
            return None

        logger.debug(f"  Loading cached roads (age: {age.days} days)")
        with open(cache_file) as f:
            return json.load(f)

    except Exception as e:
        logger.warning(f"  Error loading cache: {e}")
        return None


def _cache_road_data(
    bbox: Tuple[float, float, float, float],
    geojson_data: Dict[str, Any],
) -> None:
    """
    Save road data to cache.

    Args:
        bbox: (south, west, north, east)
        geojson_data: GeoJSON FeatureCollection
    """
    bbox_hash = _compute_bbox_hash(bbox)
    cache_dir = _get_cache_dir()
    cache_file = cache_dir / f"roads_{bbox_hash}.geojson"
    meta_file = cache_dir / f"roads_{bbox_hash}_meta.json"

    try:
        # Save GeoJSON
        with open(cache_file, "w") as f:
            json.dump(geojson_data, f, indent=2)

        # Save metadata
        meta = {
            "created_at": datetime.now().isoformat(),
            "bbox": bbox,
            "num_features": len(geojson_data.get("features", [])),
        }
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"  Cached {len(geojson_data.get('features', []))} roads")

    except Exception as e:
        logger.warning(f"  Error caching road data: {e}")


def _fetch_roads_from_osm(
    bbox: Tuple[float, float, float, float],
    road_types: List[str],
    timeout: int = 60,
) -> Optional[Dict[str, Any]]:
    """
    Fetch road data from OpenStreetMap Overpass API.

    Args:
        bbox: (south, west, north, east) in WGS84
        road_types: List of OSM highway tags
        timeout: Request timeout in seconds

    Returns:
        GeoJSON FeatureCollection or None if fetch fails
    """
    south, west, north, east = bbox
    logger.info("Fetching road data from OpenStreetMap Overpass API...")
    logger.info(f"  Extent: lat [{south:.3f}, {north:.3f}], lon [{west:.3f}, {east:.3f}]")
    logger.info(f"  Road types: {', '.join(road_types)}")

    query = build_overpass_query(bbox, road_types)

    try:
        overpass_url = "https://overpass-api.de/api/interpreter"
        logger.debug(f"  Sending query to {overpass_url} (timeout: {timeout}s)...")

        response = requests.post(overpass_url, data={"data": query}, timeout=timeout)
        response.raise_for_status()
        data = response.json()

    except requests.exceptions.Timeout:
        logger.error(f"  Overpass API timeout after {timeout}s")
        return None
    except requests.exceptions.HTTPError as e:
        if response.status_code == 429:
            logger.error("  Overpass API rate limited (429)")
        else:
            logger.error(f"  Overpass API error: {response.status_code}")
        return None
    except Exception as e:
        logger.error(f"  Failed to fetch from Overpass API: {e}")
        return None

    # Parse OSM data into GeoJSON
    features = []
    for element in data.get("elements", []):
        try:
            if element.get("type") != "way":
                continue

            # Extract road geometry (coordinates)
            nodes = element.get("nodes", [])
            if not nodes or len(nodes) < 2:
                continue

            # Get coordinates from geometry if available
            if "geometry" not in element:
                continue

            coords = [[node["lon"], node["lat"]] for node in element["geometry"]]
            if len(coords) < 2:
                continue

            # Extract tags
            tags = element.get("tags", {})
            highway_type = tags.get("highway", "unknown")
            name = tags.get("name", f"Road_{element['id']}")
            ref = tags.get("ref", "")

            # Create GeoJSON feature
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords,
                },
                "properties": {
                    "osm_id": element["id"],
                    "name": name,
                    "ref": ref,
                    "highway": highway_type,
                },
            }
            features.append(feature)

        except Exception as e:
            logger.debug(f"  Skipped element {element.get('id')}: {e}")
            continue

    logger.info(f"  Found {len(features)} road segments")

    # Return as GeoJSON FeatureCollection
    return {
        "type": "FeatureCollection",
        "features": features,
    }


def get_roads(
    bbox: Tuple[float, float, float, float],
    road_types: Optional[List[str]] = None,
    force_refresh: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Get road network data for a bounding box from OpenStreetMap.

    Fetches from cache if available, otherwise queries Overpass API.
    Results are cached locally for 30 days.

    Args:
        bbox: (south, west, north, east) in WGS84 (EPSG:4326)
        road_types: OSM highway tags to include. Default: ['motorway', 'trunk', 'primary']
            Common types:
            - 'motorway': Interstates, major freeways
            - 'trunk': Major state routes
            - 'primary': Primary state/county roads
            - 'secondary': Secondary roads
            - 'tertiary': Local connecting roads
        force_refresh: Force fresh fetch, skip cache

    Returns:
        GeoJSON FeatureCollection with road LineStrings, or None if fetch fails

    Example:
        >>> from src.terrain.roads import get_roads
        >>> bbox = (32.5, -117.6, 33.5, -116.0)  # San Diego
        >>> roads = get_roads(bbox, road_types=['motorway', 'trunk'])
        >>> print(f"Found {len(roads['features'])} road segments")
    """
    if road_types is None:
        road_types = ["motorway", "trunk", "primary"]

    logger.info(f"Loading roads for bbox: {bbox}")

    # Try cache first
    if not force_refresh:
        cached = _load_cached_roads(bbox)
        if cached is not None:
            logger.info(f"  Using cached roads ({len(cached['features'])} segments)")
            return cached

    # Fetch from API
    start_time = time.time()
    geojson = _fetch_roads_from_osm(bbox, road_types)

    if geojson is None:
        logger.warning("  Failed to fetch roads from API, returning empty collection")
        return {"type": "FeatureCollection", "features": []}

    elapsed = time.time() - start_time
    logger.info(f"  Fetch completed in {elapsed:.1f}s")

    # Cache the result
    _cache_road_data(bbox, geojson)

    return geojson


def get_roads_tiled(
    bbox: Tuple[float, float, float, float],
    road_types: Optional[List[str]] = None,
    tile_size: float = 2.0,
    retry_count: int = 1,
    retry_delay: float = 2.0,
    force_refresh: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Fetch roads for large areas by tiling and merging results.

    Works globally - anywhere OpenStreetMap has road coverage.
    Automatically tiles large bounding boxes to respect Overpass API limits
    and handles retries for failed tiles.

    Args:
        bbox: (south, west, north, east) in WGS84 - works for any location worldwide
        road_types: OSM highway tags to include. Default: ['motorway', 'trunk', 'primary']
        tile_size: Size of tiles in degrees (default: 2.0° recommended for Overpass API)
        retry_count: Number of retries for failed tiles (default: 1)
        retry_delay: Delay in seconds before retrying (default: 2.0)
        force_refresh: Force fresh fetch, skip cache (default: False)

    Returns:
        Merged GeoJSON FeatureCollection with all road segments from all tiles,
        or empty FeatureCollection if all fetches fail.

    Example:
        >>> # Large area covering multiple US states
        >>> bbox = (37.0, -95.0, 45.0, -85.0)
        >>> roads = get_roads_tiled(bbox, road_types=["motorway", "trunk"])
        >>> print(f"Found {len(roads['features'])} road segments")

        >>> # Works globally - Alps region
        >>> alps_bbox = (45.5, 6.0, 47.5, 14.0)
        >>> roads = get_roads_tiled(alps_bbox, tile_size=1.5)
    """
    import math

    if road_types is None:
        road_types = ["motorway", "trunk", "primary"]

    south, west, north, east = bbox
    lat_span = north - south
    lon_span = east - west

    logger.info(f"Fetching roads for bbox: lat [{south:.2f}, {north:.2f}], lon [{west:.2f}, {east:.2f}]")
    logger.info(f"  Coverage: {lat_span:.1f}° lat × {lon_span:.1f}° lon")

    # Check if tiling is needed
    if lat_span <= tile_size and lon_span <= tile_size:
        # Small bbox - single fetch
        logger.info("  Small area - single fetch (no tiling needed)")
        result = get_roads(bbox, road_types, force_refresh=force_refresh)
        return result if result else {"type": "FeatureCollection", "features": []}

    # Large bbox - tile and merge
    lat_tiles = int(math.ceil(lat_span / tile_size))
    lon_tiles = int(math.ceil(lon_span / tile_size))
    total_tiles = lat_tiles * lon_tiles

    logger.info(f"  Large area - fetching in {tile_size}° tiles ({lat_tiles}×{lon_tiles} = {total_tiles} tiles)")

    all_features = []
    failed_tiles = []

    # Fetch all tiles
    for lat_idx in range(lat_tiles):
        for lon_idx in range(lon_tiles):
            tile_south = south + lat_idx * tile_size
            tile_north = min(tile_south + tile_size, north)
            tile_west = west + lon_idx * tile_size
            tile_east = min(tile_west + tile_size, east)

            tile_bbox = (tile_south, tile_west, tile_north, tile_east)
            tile_num = lat_idx * lon_tiles + lon_idx + 1

            logger.info(f"  Tile {tile_num}/{total_tiles}: lat [{tile_south:.2f}, {tile_north:.2f}], lon [{tile_west:.2f}, {tile_east:.2f}]")

            tile_roads = get_roads(tile_bbox, road_types, force_refresh=force_refresh)
            if tile_roads and tile_roads.get("features"):
                feature_count = len(tile_roads["features"])
                all_features.extend(tile_roads["features"])
                logger.info(f"    ✓ {feature_count} road segments")
            else:
                failed_tiles.append((tile_num, tile_bbox))
                logger.warning(f"    ✗ No roads returned (timeout or error)")

    # Retry failed tiles
    if failed_tiles and retry_count > 0:
        logger.info(f"  Retrying {len(failed_tiles)} failed tiles...")
        time.sleep(retry_delay)

        still_failed = []
        for tile_num, tile_bbox in failed_tiles:
            logger.info(f"  Retry tile {tile_num}...")
            tile_roads = get_roads(tile_bbox, road_types, force_refresh=True)
            if tile_roads and tile_roads.get("features"):
                feature_count = len(tile_roads["features"])
                all_features.extend(tile_roads["features"])
                logger.info(f"    ✓ Retry succeeded: {feature_count} segments")
            else:
                still_failed.append(tile_num)
                logger.warning(f"    ✗ Retry failed")

        if still_failed:
            logger.warning(f"  {len(still_failed)} tiles failed after retry: {still_failed}")

    logger.info(f"  Loaded {len(all_features)} total road segments from {total_tiles} tiles")
    return {"type": "FeatureCollection", "features": all_features}


# =============================================================================
# ROAD VISUALIZATION
# =============================================================================


def road_colormap(road_grid, score=None):
    """
    Map roads to a distinctive red color for special material treatment.

    The red color (180, 30, 30) is used as a marker so the Blender material
    shader can identify road pixels and apply a glassy/emissive effect.

    Args:
        road_grid: 2D array of road values (0=no road, >0=road)
        score: Unused, kept for API compatibility.

    Returns:
        Array of RGB colors with shape (height, width, 3) as uint8
    """
    # Create road mask (any road value > 0)
    road_mask = road_grid > 0.5  # Use 0.5 threshold to handle resampling artifacts

    height, width = road_grid.shape
    colors = np.zeros((height, width, 3), dtype=np.uint8)

    # Deep red for roads - will be made glassy/emissive by material shader
    road_red_color = (180, 30, 30)
    colors[road_mask] = road_red_color

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


def smooth_road_mask(
    road_mask: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Apply Gaussian blur to road mask for anti-aliased edges (GPU-accelerated).

    The Bresenham line algorithm creates stair-step (aliased) edges.
    Applying Gaussian smoothing creates soft anti-aliased boundaries that
    render more smoothly, especially after the mask goes through resampling.

    Uses PyTorch GPU acceleration when available (6x speedup on CUDA).

    Args:
        road_mask: 2D array of road values (0=no road, >0=road)
        sigma: Gaussian blur sigma in pixels (default: 1.0).
            Higher values = softer edges. Typical range: 0.5-2.0.
            - 0.5: Minimal softening
            - 1.0: Standard anti-aliasing (recommended)
            - 2.0: Very soft/blurry edges

    Returns:
        Smoothed road mask as float32 array. Values are now continuous
        (not binary) and may need thresholding if binary mask is needed.
    """
    from src.terrain.gpu_ops import gpu_gaussian_blur

    if sigma <= 0:
        return road_mask.astype(np.float32)

    logger.info(f"Anti-aliasing road mask (sigma={sigma})...")

    # Convert to float for smooth blending
    mask_float = road_mask.astype(np.float32)

    # Apply Gaussian blur (GPU-accelerated)
    smoothed = gpu_gaussian_blur(mask_float, sigma=sigma)

    # Preserve original values where roads are strong, blend at edges
    # This keeps road centers at original intensity while softening edges
    road_pixels = np.sum(road_mask > 0.5)
    edge_pixels = np.sum((smoothed > 0.1) & (smoothed < 0.9))

    logger.info(f"✓ Road mask anti-aliased: {road_pixels} road pixels, {edge_pixels} edge pixels")

    return smoothed


def smooth_dem_along_roads(
    dem: np.ndarray,
    road_mask: np.ndarray,
    smoothing_radius: int = 2,
) -> np.ndarray:
    """
    Smooth the DEM along roads to reduce elevation detail (GPU-accelerated).

    Applies Gaussian smoothing only to road pixels. Non-road pixels
    are unchanged. The smoothing kernel should be about half the road width.

    Uses PyTorch GPU acceleration when available (6x speedup on CUDA).

    Args:
        dem: 2D array of elevation values
        road_mask: 2D boolean or float array where >0.5 = road
        smoothing_radius: Radius for Gaussian smoothing (default: 2 pixels)

    Returns:
        Smoothed DEM array (same shape as input)
    """
    from src.terrain.gpu_ops import gpu_gaussian_blur

    logger.info(f"Smoothing DEM along roads (radius={smoothing_radius})...")

    # Create binary road mask
    road_binary = road_mask > 0.5

    # Apply Gaussian smoothing to entire DEM (GPU-accelerated)
    smoothed_dem = gpu_gaussian_blur(dem.astype(np.float32), sigma=float(smoothing_radius))

    # Only replace road pixels with smoothed values
    result = dem.copy()
    result[road_binary] = smoothed_dem[road_binary]

    road_pixels = np.sum(road_binary)
    logger.info(f"✓ Smoothed {road_pixels} road pixels")

    return result.astype(dem.dtype)


def smooth_road_vertices(
    vertices: np.ndarray,
    road_mask: np.ndarray,
    y_valid: np.ndarray,
    x_valid: np.ndarray,
    smoothing_radius: int = 2,
) -> np.ndarray:
    """
    Smooth Z coordinates of mesh vertices that are on roads.

    Reconstructs a 2D Z grid from vertex positions, applies 2D Gaussian
    smoothing in grid space, then writes smoothed values back to road
    vertices only. Non-road vertices are unchanged.

    This ensures smoothing follows the spatial layout of the road rather
    than the arbitrary vertex index order.

    Args:
        vertices: Mesh vertex positions (N, 3) array with [x, y, z] coords
        road_mask: 2D array (H, W) where >0.5 indicates road pixels
        y_valid: Array (N,) of y indices mapping vertices to road_mask rows
        x_valid: Array (N,) of x indices mapping vertices to road_mask columns
        smoothing_radius: Gaussian smoothing sigma (default: 2, use 0 to disable)

    Returns:
        Modified vertices array with smoothed Z values on roads.
        X and Y coordinates are never modified.
    """
    from scipy.ndimage import gaussian_filter

    # Return unchanged if no smoothing
    if smoothing_radius <= 0:
        return vertices.copy()

    result = vertices.copy()
    n_surface_vertices = len(y_valid)
    h, w = road_mask.shape

    # Find which surface vertices are on roads
    in_bounds = (
        (y_valid >= 0) & (y_valid < h) &
        (x_valid >= 0) & (x_valid < w)
    )

    road_vertex_mask = np.zeros(n_surface_vertices, dtype=bool)
    road_vertex_mask[in_bounds] = road_mask[y_valid[in_bounds], x_valid[in_bounds]] > 0.5

    road_indices = np.where(road_vertex_mask)[0]
    if len(road_indices) == 0:
        return result

    # Reconstruct 2D Z grid from vertex positions
    z_grid = np.full((h, w), np.nan, dtype=np.float64)
    z_grid[y_valid[in_bounds], x_valid[in_bounds]] = result[
        np.where(in_bounds)[0], 2
    ]

    # Create a road-only Z grid: set non-road pixels to NaN so the Gaussian
    # only averages road Z values with neighboring road Z values.
    # To handle NaN in gaussian_filter, use the normalized convolution trick:
    # smoothed = convolve(data_with_zeros) / convolve(mask)
    is_road = road_mask > 0.5
    valid_road = is_road & ~np.isnan(z_grid)

    z_for_smooth = np.where(valid_road, z_grid, 0.0)
    weight = valid_road.astype(np.float64)

    sigma = float(smoothing_radius)
    z_smoothed = gaussian_filter(z_for_smooth, sigma=sigma, mode='nearest')
    w_smoothed = gaussian_filter(weight, sigma=sigma, mode='nearest')

    # Avoid division by zero where no valid road neighbors exist
    safe_mask = w_smoothed > 1e-10
    z_result = np.where(safe_mask, z_smoothed / w_smoothed, z_grid)

    # Write smoothed Z back to road vertices only
    result[road_indices, 2] = z_result[
        y_valid[road_indices], x_valid[road_indices]
    ]

    logger.info(f"✓ Smoothed {len(road_indices)} road vertices in 2D grid space "
                f"(sigma={smoothing_radius})")

    return result


def offset_road_vertices(
    vertices: np.ndarray,
    road_mask: np.ndarray,
    y_valid: np.ndarray,
    x_valid: np.ndarray,
    offset: float = 0.0,
) -> np.ndarray:
    """
    Offset Z coordinates of mesh vertices that are on roads by a fixed amount.

    A simpler alternative to smooth_road_vertices. Raises or lowers all road
    vertices by a constant offset, making roads visually distinct from terrain.

    Args:
        vertices: Mesh vertex positions (N, 3) array with [x, y, z] coords
        road_mask: 2D array (H, W) where >0.5 indicates road pixels
        y_valid: Array (N,) of y indices mapping vertices to road_mask rows
        x_valid: Array (N,) of x indices mapping vertices to road_mask columns
        offset: Z offset to apply to road vertices. Positive = raise, negative = lower.
                Default: 0.0 (no change)

    Returns:
        Modified vertices array with offset Z values on roads.
        X and Y coordinates are never modified.
    """
    # Return unchanged if no offset
    if offset == 0.0:
        return vertices.copy()

    result = vertices.copy()
    n_surface_vertices = len(y_valid)

    # Find which surface vertices are on roads
    in_bounds = (
        (y_valid >= 0) & (y_valid < road_mask.shape[0]) &
        (x_valid >= 0) & (x_valid < road_mask.shape[1])
    )

    # Only surface vertices (with grid mappings) can be on roads
    road_vertex_mask = np.zeros(n_surface_vertices, dtype=bool)
    road_vertex_mask[in_bounds] = road_mask[y_valid[in_bounds], x_valid[in_bounds]] > 0.5

    # Apply offset to road vertices
    road_indices = np.where(road_vertex_mask)[0]
    if len(road_indices) > 0:
        result[road_indices, 2] += offset

    return result


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

    To color roads, use the multi-overlay color mapping system::

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

    This function is deprecated. Use the multi-overlay color mapping system instead::

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
