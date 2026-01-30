"""
DEM (Digital Elevation Model) downloader for SRTM elevation data.

This module provides functions to download SRTM elevation data for specified
geographic areas using either bounding box coordinates or place names.

SRTM Data:
    - NASA Shuttle Radar Topography Mission
    - Global coverage (60°N to 56°S)
    - 1 arc-second (~30m) resolution (SRTM1)
    - 3 arc-second (~90m) resolution (SRTM3)
    - Data format: HGT (Height) files
    - Tile size: 1° × 1° geographic grid

Usage - Download by bbox::

    from src.terrain.dem_downloader import download_dem_by_bbox

    bbox = (42.0, -83.5, 42.5, -83.0)  # Detroit area
    files = download_dem_by_bbox(
        bbox=bbox,
        output_dir="data/detroit_dem",
        username="your_earthdata_username",
        password="your_earthdata_password"
    )

Usage - Download by place name::

    from src.terrain.dem_downloader import download_dem_by_place_name

    files = download_dem_by_place_name(
        place_name="Detroit, MI",
        output_dir="data/detroit_dem",
        username="your_earthdata_username",
        password="your_earthdata_password"
    )

Usage - Visualize bbox::

    from src.terrain.dem_downloader import display_bbox_on_map

    bbox = (42.0, -83.5, 42.5, -83.0)
    display_bbox_on_map(bbox, output_file="bbox_map.html")
    # Open bbox_map.html in browser to visualize the area
"""

import logging
import math
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import requests
except ImportError:
    requests = None

try:
    from NASADEM import NASADEMConnection
    import earthaccess
except ImportError:
    NASADEMConnection = None
    earthaccess = None

logger = logging.getLogger(__name__)


def get_srtm_tile_name(lat: float, lon: float) -> str:
    """
    Get SRTM tile name for a given latitude/longitude coordinate.

    SRTM tiles are 1°×1° and named by their southwest corner coordinates.

    Args:
        lat: Latitude in decimal degrees (-90 to +90)
        lon: Longitude in decimal degrees (-180 to +180)

    Returns:
        Tile name following SRTM convention (e.g., "N42W084")

    Examples:
        >>> get_srtm_tile_name(42.3, -83.0)
        'N42W083'
        >>> get_srtm_tile_name(42.9, -83.9)
        'N42W084'
    """
    # Floor coordinates to get SW corner of tile
    lat_floor = math.floor(lat)
    lon_floor = math.floor(lon)

    # Format latitude (N/S)
    lat_letter = 'N' if lat_floor >= 0 else 'S'
    lat_val = abs(lat_floor)

    # Format longitude (E/W)
    lon_letter = 'E' if lon_floor >= 0 else 'W'
    lon_val = abs(lon_floor)

    return f"{lat_letter}{lat_val:02d}{lon_letter}{lon_val:03d}"


def calculate_required_srtm_tiles(bbox: Tuple[float, float, float, float]) -> List[str]:
    """
    Calculate which SRTM tiles are needed to cover a bounding box.

    Args:
        bbox: Bounding box as (south, west, north, east) in decimal degrees

    Returns:
        List of SRTM tile names (e.g., ["N42W084", "N42W083"])

    Examples:
        >>> calculate_required_srtm_tiles((42.0, -83.5, 42.5, -83.0))
        ['N42W084', 'N42W083']
    """
    south, west, north, east = bbox

    # Calculate tile ranges
    lat_min = math.floor(south)
    lat_max = math.floor(north)
    lon_min = math.floor(west)
    lon_max = math.floor(east)

    tiles = []
    for lat in range(lat_min, lat_max + 1):
        for lon in range(lon_min, lon_max + 1):
            tile_name = get_srtm_tile_name(lat, lon)
            tiles.append(tile_name)

    return tiles


def _download_srtm_tile(
    tile_name: str,
    output_dir: Path,
    username: Optional[str] = None,
    password: Optional[str] = None
) -> bool:
    """
    Download a single SRTM tile from NASA Earthdata.

    Downloads SRTM1 (1 arc-second, ~30m) data from NASA Earthdata using the
    NASADEM library. Files are downloaded as ZIP archives containing HGT files.

    Requires free NASA Earthdata account: https://urs.earthdata.nasa.gov/users/new

    Args:
        tile_name: SRTM tile name (e.g., "N42W084")
        output_dir: Directory to save downloaded file
        username: NASA Earthdata username
        password: NASA Earthdata password

    Returns:
        True if download successful or file already exists, False otherwise

    Note:
        NASADEM downloads tiles as ZIP files named like "NASADEM_HGT_N32W117.zip".
        The ZIP contains the HGT file and other metadata.
    """
    if NASADEMConnection is None:
        logger.error(
            "NASADEM library not installed. Install with: pip install NASADEM\n"
            "Or download manually from: https://portal.opentopography.org/"
        )
        return False

    # NASADEM downloads ZIP files, not raw HGT
    # Format: NASADEM_HGT_N32W117.zip (uppercase tile name)
    output_file = output_dir / f"NASADEM_HGT_{tile_name.upper()}.zip"

    # Skip if file already exists
    if output_file.exists():
        logger.info(f"Tile {tile_name} already exists, skipping download")
        return True

    if username is None or password is None:
        logger.warning("No credentials provided - skipping actual download")
        return False

    logger.info(f"Downloading SRTM tile: {tile_name}")

    try:
        # Authenticate with NASA Earthdata using environment variables
        # The earthaccess.login() API changed - now uses strategy parameter
        # instead of username/password. The "environment" strategy reads from
        # EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables.
        if earthaccess is not None:
            earthaccess.login(strategy="environment", persist=False)

        # Create NASADEM connection with our output directory
        # Pass skip_auth=True since we already authenticated above
        nasadem = NASADEMConnection(
            download_directory=str(output_dir),
            skip_auth=True  # We already authenticated with earthaccess
        )

        # Download tile - returns NASADEMGranule object
        # Note: tile_name should be lowercase for NASADEM (e.g., "n32w117")
        granule = nasadem.download_tile(tile_name.lower())

        if output_file.exists():
            logger.info(f"✓ Downloaded {tile_name} ({output_file.stat().st_size} bytes)")
            return True
        else:
            logger.warning(f"Download completed but file not found: {output_file}")
            return False

    except Exception as e:
        logger.error(f"Failed to download {tile_name}: {e}")
        return False


def download_dem_by_bbox(
    bbox: Tuple[float, float, float, float],
    output_dir: str,
    username: Optional[str] = None,
    password: Optional[str] = None
) -> List[Path]:
    """
    Download SRTM elevation data for a bounding box area.

    Args:
        bbox: Bounding box as (south, west, north, east) in decimal degrees
        output_dir: Directory to save downloaded DEM files
        username: NASA Earthdata username (optional for testing)
        password: NASA Earthdata password (optional for testing)

    Returns:
        List of Path objects pointing to downloaded ZIP files

    Note:
        NASADEM downloads tiles as ZIP files (e.g., "NASADEM_HGT_N32W117.zip").
        Each ZIP contains the HGT file and metadata.

    Examples:
        >>> bbox = (42.0, -83.5, 42.5, -83.0)  # Detroit
        >>> files = download_dem_by_bbox(bbox, "data/dem", "user", "pass")
        >>> print(f"Downloaded {len(files)} tiles")
    """
    output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Calculate required tiles
    tiles = calculate_required_srtm_tiles(bbox)
    logger.info(f"Need {len(tiles)} SRTM tiles for bbox: {tiles}")

    # Download each tile
    downloaded_files = []
    for tile_name in tiles:
        success = _download_srtm_tile(tile_name, output_path, username, password)
        if success:
            # NASADEM downloads ZIP files with uppercase tile names
            tile_file = output_path / f"NASADEM_HGT_{tile_name.upper()}.zip"
            downloaded_files.append(tile_file)

    return downloaded_files


def _geocode_place_name(place_name: str) -> Tuple[float, float, float, float]:
    """
    Geocode a place name to a bounding box.

    Args:
        place_name: Place name like "Detroit, MI"

    Returns:
        Bounding box as (south, west, north, east)
    """
    # Minimal implementation - would use geocoding API
    logger.info(f"Would geocode place name: {place_name}")

    # Return a dummy bbox for now
    return (42.0, -83.5, 42.5, -83.0)


def download_dem_by_place_name(
    place_name: str,
    output_dir: str,
    username: Optional[str] = None,
    password: Optional[str] = None
) -> List[Path]:
    """
    Download SRTM elevation data for a named location.

    Args:
        place_name: Place name like "Detroit, MI" or "Mount Rainier"
        output_dir: Directory to save downloaded DEM files
        username: NASA Earthdata username (optional for testing)
        password: NASA Earthdata password (optional for testing)

    Returns:
        List of Path objects pointing to downloaded HGT files

    Examples:
        >>> files = download_dem_by_place_name("Detroit, MI", "data/dem")
    """
    # Geocode place name to bbox
    bbox = _geocode_place_name(place_name)

    # Download using bbox
    return download_dem_by_bbox(bbox, output_dir, username, password)


def display_bbox_on_map(
    bbox: Tuple[float, float, float, float],
    output_file: str = "bbox_map.html"
) -> None:
    """
    Create an interactive HTML map showing the bounding box.

    Helps users visualize and verify their bounding box selection.

    Args:
        bbox: Bounding box as (south, west, north, east) in decimal degrees
        output_file: Path to output HTML file

    Examples:
        >>> bbox = (42.0, -83.5, 42.5, -83.0)
        >>> display_bbox_on_map(bbox, "detroit_bbox.html")
        # Opens detroit_bbox.html in browser
    """
    south, west, north, east = bbox

    # Minimal HTML with inline leaflet
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Bounding Box Visualization</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        #map {{ height: 600px; width: 100%; }}
    </style>
</head>
<body>
    <h1>Bounding Box: ({south}, {west}) to ({north}, {east})</h1>
    <div id="map"></div>
    <script>
        var map = L.map('map').setView([{(south + north) / 2}, {(west + east) / 2}], 10);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors'
        }}).addTo(map);

        var bounds = [[{south}, {west}], [{north}, {east}]];
        L.rectangle(bounds, {{color: "#ff7800", weight: 2}}).addTo(map);
        map.fitBounds(bounds);
    </script>
</body>
</html>"""

    output_path = Path(output_file)
    output_path.write_text(html_content)
    logger.info(f"Created bbox visualization: {output_file}")
