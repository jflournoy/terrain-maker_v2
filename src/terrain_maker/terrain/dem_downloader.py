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

    from terrain_maker.terrain.dem_downloader import download_dem_by_bbox

    bbox = (42.0, -83.5, 42.5, -83.0)  # Detroit area
    files = download_dem_by_bbox(
        bbox=bbox,
        output_dir="data/detroit_dem",
        username="your_earthdata_username",
        password="your_earthdata_password"
    )

Usage - Download by place name::

    from terrain_maker.terrain.dem_downloader import download_dem_by_place_name

    files = download_dem_by_place_name(
        place_name="Detroit, MI",
        output_dir="data/detroit_dem",
        username="your_earthdata_username",
        password="your_earthdata_password"
    )

Usage - Visualize bbox::

    from terrain_maker.terrain.dem_downloader import display_bbox_on_map

    bbox = (42.0, -83.5, 42.5, -83.0)
    display_bbox_on_map(bbox, output_file="bbox_map.html")
    # Open bbox_map.html in browser to visualize the area
"""

import logging
import math
import os
from pathlib import Path
from typing import Collection, List, Tuple, Optional

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


def _load_earthdata_credentials(
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve NASA Earthdata credentials from args, environment, or .env file.

    Checks in order:
    1. Explicit username/password arguments (if provided)
    2. EARTHDATA_USERNAME / EARTHDATA_PASSWORD environment variables
    3. .env file in the current working directory or project root

    Returns:
        (username, password) tuple, either or both may be None if not found
    """
    if username and password:
        return username, password

    # Check environment variables
    username = username or os.environ.get("EARTHDATA_USERNAME")
    password = password or os.environ.get("EARTHDATA_PASSWORD")
    if username and password:
        return username, password

    # Try loading from .env file (no dependency required)
    for env_path in [Path.cwd() / ".env", Path.cwd().parent / ".env"]:
        if env_path.exists():
            try:
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" not in line:
                            continue
                        key, _, value = line.partition("=")
                        key, value = key.strip(), value.strip()
                        if key == "EARTHDATA_USERNAME" and not username:
                            username = value
                        elif key == "EARTHDATA_PASSWORD" and not password:
                            password = value
                if username and password:
                    logger.debug(f"Loaded Earthdata credentials from {env_path}")
                    # Also set in environment so earthaccess.login(strategy="environment") works
                    os.environ.setdefault("EARTHDATA_USERNAME", username)
                    os.environ.setdefault("EARTHDATA_PASSWORD", password)
                    return username, password
            except Exception as e:
                logger.debug(f"Could not read {env_path}: {e}")

    return username, password


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
        True when the tile's ZIP is present (downloaded now or already on disk).

    Raises:
        ImportError: the NASADEM library is not installed.
        RuntimeError: no Earthdata credentials, or the download failed or left no file.

    Note:
        NASADEM downloads tiles as ZIP files named like "NASADEM_HGT_N32W117.zip".
        The ZIP contains the HGT file and other metadata.
    """
    if NASADEMConnection is None:
        raise ImportError(
            "NASADEM library not installed (uv sync installs it); or download tiles "
            "manually from https://portal.opentopography.org/"
        )

    # NASADEM downloads ZIP files, not raw HGT
    # Format: NASADEM_HGT_N32W117.zip (uppercase tile name)
    output_file = output_dir / f"NASADEM_HGT_{tile_name.upper()}.zip"

    # Skip if file already exists
    if output_file.exists():
        logger.info(f"Tile {tile_name} already exists, skipping download")
        return True

    # Resolve credentials from args → env vars → .env file
    username, password = _load_earthdata_credentials(username, password)
    if username is None or password is None:
        raise RuntimeError(
            "No Earthdata credentials found. Provide via username/password arguments, "
            "EARTHDATA_USERNAME/EARTHDATA_PASSWORD env vars, or a .env file in the project root"
        )

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

    except Exception as e:
        raise RuntimeError(f"Failed to download {tile_name}: {type(e).__name__}: {e}") from e

    if not output_file.exists():
        raise RuntimeError(f"Download of {tile_name} reported success but {output_file} is missing")
    logger.info(f"✓ Downloaded {tile_name} ({output_file.stat().st_size} bytes)")
    return True


def download_dem_by_bbox(
    bbox: Tuple[float, float, float, float],
    output_dir: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
    expected_missing: Collection[str] = (),
) -> List[Path]:
    """
    Download SRTM elevation data for a bounding box area.

    Every tile the bbox needs must arrive, or this raises naming each failed tile and why.
    NASADEM has no tiles over open ocean; name such tiles in expected_missing to say their
    absence is known (they are still downloaded if they exist).

    Args:
        bbox: Bounding box as (south, west, north, east) in decimal degrees
        output_dir: Directory to save downloaded DEM files
        username: NASA Earthdata username (optional for testing)
        password: NASA Earthdata password (optional for testing)
        expected_missing: Tile names (e.g. "N32W118") allowed to be unavailable

    Returns:
        List of Path objects pointing to downloaded ZIP files

    Raises:
        RuntimeError: one or more tiles not in expected_missing could not be downloaded.

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

    # Try every tile, then report all failures at once
    expected = {t.upper() for t in expected_missing}
    downloaded_files = []
    failures = {}
    for tile_name in tiles:
        try:
            _download_srtm_tile(tile_name, output_path, username, password)
        except Exception as e:
            if tile_name.upper() in expected:
                logger.info(f"Tile {tile_name} unavailable, as expected: {e}")
                continue
            failures[tile_name] = f"{type(e).__name__}: {e}"
            continue
        # NASADEM downloads ZIP files with uppercase tile names
        downloaded_files.append(output_path / f"NASADEM_HGT_{tile_name.upper()}.zip")

    if failures:
        detail = "\n".join(f"  {tile}: {why}" for tile, why in failures.items())
        raise RuntimeError(
            f"{len(failures)} of {len(tiles)} DEM tiles could not be downloaded:\n{detail}\n"
            "If a tile is open ocean (no NASADEM data), pass it in expected_missing."
        )
    return downloaded_files


def _geocode_place_name(place_name: str) -> Tuple[float, float, float, float]:
    """
    Geocode a place name to a bounding box.

    Args:
        place_name: Place name like "Detroit, MI"

    Returns:
        Bounding box as (south, west, north, east)
    """
    raise NotImplementedError(
        f"Geocoding is not implemented, so {place_name!r} cannot be turned into a bbox "
        "(it used to return Detroit's bbox for every name). Use download_dem_by_bbox."
    )


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
