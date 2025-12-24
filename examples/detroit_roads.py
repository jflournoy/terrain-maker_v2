#!/usr/bin/env python3
"""
Detroit Road Network Data Fetcher.

This module provides functions to fetch road network data (interstates and state roads)
from OpenStreetMap Overpass API with caching support.

Pipeline:
1. Build Overpass QL query for specified road types
2. Fetch road geometries from OSM Overpass API
3. Cache results locally with hash-based keys
4. Return as GeoJSON FeatureCollection

Usage:
    from examples.detroit_roads import get_roads

    # Fetch roads for a bounding box
    bbox = (42.3, -83.5, 42.5, -82.8)  # (south, west, north, east)
    roads = get_roads(bbox, road_types=['motorway', 'trunk', 'primary'])

    # With caching (subsequent calls are instant)
    roads = get_roads(bbox, road_types=['motorway'], force_refresh=False)
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# =============================================================================
# QUERY BUILDING
# =============================================================================


def build_overpass_query(bbox: tuple, road_types: list[str]) -> str:
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


# =============================================================================
# CACHING
# =============================================================================


def compute_bbox_hash(bbox: tuple) -> str:
    """
    Compute hash for bounding box.

    Args:
        bbox: (south, west, north, east)

    Returns:
        SHA256 hash of bbox coordinates
    """
    bbox_str = f"{bbox[0]:.6f},{bbox[1]:.6f},{bbox[2]:.6f},{bbox[3]:.6f}"
    return hashlib.sha256(bbox_str.encode()).hexdigest()


def get_cache_dir() -> Path:
    """
    Get or create cache directory for road data.

    Returns:
        Path to cache directory
    """
    cache_dir = Path.cwd() / "data" / "cache" / "roads"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def load_cached_roads(bbox: tuple) -> Optional[Dict[str, Any]]:
    """
    Load road data from cache if it exists and is fresh.

    Args:
        bbox: (south, west, north, east)

    Returns:
        GeoJSON dict if cache exists and is fresh, None otherwise
    """
    bbox_hash = compute_bbox_hash(bbox)
    cache_dir = get_cache_dir()
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


def cache_road_data(bbox: tuple, geojson_data: Dict[str, Any]) -> None:
    """
    Save road data to cache.

    Args:
        bbox: (south, west, north, east)
        geojson_data: GeoJSON FeatureCollection
    """
    bbox_hash = compute_bbox_hash(bbox)
    cache_dir = get_cache_dir()
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


# =============================================================================
# FETCHING FROM OSM
# =============================================================================


def fetch_roads_from_osm(
    bbox: tuple, road_types: list[str], timeout: int = 60
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
    try:
        import requests
    except ImportError:
        logger.warning("requests library not available for road fetching")
        return None

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


# =============================================================================
# PUBLIC API
# =============================================================================


def get_roads(
    bbox: tuple,
    road_types: list[str] = None,
    force_refresh: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Get road network data for a bounding box.

    Fetches from cache if available, otherwise queries Overpass API.

    Args:
        bbox: (south, west, north, east) in WGS84 (EPSG:4326)
        road_types: OSM highway tags to include. Default: ['motorway', 'trunk', 'primary']
        force_refresh: Force fresh fetch, skip cache

    Returns:
        GeoJSON FeatureCollection with road LineStrings, or None if fetch fails
    """
    if road_types is None:
        road_types = ["motorway", "trunk", "primary"]

    logger.info(f"Loading roads for bbox: {bbox}")

    # Try cache first
    if not force_refresh:
        cached = load_cached_roads(bbox)
        if cached is not None:
            logger.info(f"  Using cached roads ({len(cached['features'])} segments)")
            return cached

    # Fetch from API
    start_time = time.time()
    geojson = fetch_roads_from_osm(bbox, road_types)

    if geojson is None:
        logger.warning("  Failed to fetch roads from API, returning empty collection")
        return {"type": "FeatureCollection", "features": []}

    elapsed = time.time() - start_time
    logger.info(f"  Fetch completed in {elapsed:.1f}s")

    # Cache the result
    cache_road_data(bbox, geojson)

    return geojson


def get_roads_tiled(
    bbox: tuple,
    road_types: list[str] = None,
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
        logger.info(f"  Small area - single fetch (no tiling needed)")
        result = get_roads(bbox, road_types, force_refresh=force_refresh)
        return result if result else {"type": "FeatureCollection", "features": []}

    # Large bbox - tile and merge
    import math

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


if __name__ == "__main__":
    # Simple test
    import argparse

    parser = argparse.ArgumentParser(description="Fetch road data for Detroit area")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=[42.25, -83.5, 42.5, -82.8],
        help="Bounding box: south west north east",
    )
    parser.add_argument(
        "--road-types",
        nargs="+",
        default=["motorway", "trunk"],
        help="Road types to fetch",
    )
    parser.add_argument(
        "--output",
        help="Output GeoJSON file",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force fresh fetch, skip cache",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Fetch roads
    bbox = tuple(args.bbox)
    roads = get_roads(bbox, args.road_types, force_refresh=args.force_refresh)

    if roads and args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(roads, f, indent=2)
        print(f"Saved {len(roads['features'])} roads to {output_path}")
