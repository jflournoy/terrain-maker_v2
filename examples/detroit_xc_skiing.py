#!/usr/bin/env python3
"""
Detroit Cross-Country Skiing Analysis Example.

This example demonstrates analyzing snow conditions to identify optimal
cross-country skiing locations, and locating nearby skiing parks.

Pipeline:
1. Load DEM and SNODAS snow data
2. Compute snow statistics
3. Fetch XC skiing park locations from OpenStreetMap
4. Compute snow suitability scores (grid-based)
5. Score each park based on vicinity snow conditions
6. Generate visualizations and save outputs

Usage:
    # Run with real SNODAS data
    python examples/detroit_xc_skiing.py --snodas-dir data/snodas_data

    # Run with mock data (fast, for testing)
    python examples/detroit_xc_skiing.py --mock-data

    # Specify output directory
    python examples/detroit_xc_skiing.py --output-dir ./outputs --snodas-dir data/snodas_data
"""

import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from affine import Affine
from rasterio.transform import rowcol

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.terrain.core import Terrain
from src.terrain.data_loading import load_dem_files
from src.terrain.gridded_data import (
    downsample_for_viz,
    create_mock_snow_data,
    GriddedDataLoader,
    TiledDataConfig,
)
from src.snow import (
    batch_process_snodas_data,
    calculate_snow_statistics,
)
from src.scoring.configs import (
    DEFAULT_XC_SKIING_SCORER,
    xc_skiing_compute_derived_inputs,
)

# Configure logging
LOG_FILE = Path(__file__).parent / "detroit_xc_skiing.log"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers = []

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE, mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s"))
logger.addHandler(file_handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
    handlers=[file_handler]
)

# Default mock data resolution - small enough to never OOM
# This matches what real SNODAS data would be resampled to
MOCK_DATA_SHAPE = (500, 500)  # 250K pixels per array, ~1MB each


# =============================================================================
# DEM LOADING
# =============================================================================

def run_step_dem(output_dir: Path, mock_data: bool = False):
    """Load and visualize DEM data."""
    logger.info("\n" + "=" * 70)
    logger.info("Step 1: Loading DEM Data")
    logger.info("=" * 70)

    if mock_data:
        logger.info("Generating mock DEM data...")
        height, width = 1024, 1024
        dem = np.random.randint(150, 250, (height, width))
        # Use bounds matching real DEM: lon[-89, -78], lat[37, 47]
        # This covers the full Great Lakes region
        pixel_width = 11.0 / width   # 11 degrees longitude span
        pixel_height = -10.0 / height  # 10 degrees latitude span (negative for north-up)
        transform = Affine(pixel_width, 0, -89.0, 0, pixel_height, 47.0)
        logger.info(f"Mock DEM shape: {dem.shape}, bounds: lon[-89, -78], lat[37, 47]")
        return dem, transform

    # Load real DEM files
    dem_dir = Path("data/dem/detroit")
    if not dem_dir.exists():
        raise FileNotFoundError(f"DEM directory not found: {dem_dir}")

    logger.info(f"Loading DEM files from {dem_dir}...")
    dem, transform = load_dem_files(dem_dir)
    logger.info(f"Loaded DEM shape: {dem.shape}, bounds: {transform}")

    # Visualize
    (output_dir / "01_dem").mkdir(parents=True, exist_ok=True)
    viz_dem, stride = downsample_for_viz(dem)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(viz_dem, cmap="terrain")
    ax.set_title("Detroit Elevation Model")
    plt.colorbar(im, ax=ax, label="Elevation (m)")
    output_path = output_dir / "01_dem" / "dem_elevation.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ DEM visualization: {output_path}")

    return dem, transform


# =============================================================================
# OPENSTREETMAP PARK FETCHING
# =============================================================================

def fetch_xc_skiing_parks(extent, output_path: Optional[Path] = None) -> list[dict]:
    """
    Fetch cross-country skiing parks from OpenStreetMap Overpass API.

    Args:
        extent: (minx, miny, maxx, maxy) in EPSG:4326
        output_path: Optional path to save GeoJSON

    Returns:
        List of park dicts with name, lat, lon
    """
    try:
        import requests
    except ImportError:
        logger.warning("requests not installed - using mock parks")
        return generate_mock_parks(extent)

    minx, miny, maxx, maxy = extent
    logger.info("Querying OpenStreetMap for XC skiing parks...")
    logger.info(f"  Extent: lat [{miny:.3f}, {maxy:.3f}], lon [{minx:.3f}, {maxx:.3f}]")

    # Overpass API query optimized for cross-country skiing facilities
    # Uses longer server-side timeout for large areas
    # Key tags: piste:type=nordic, sport=cross-country_skiing, landuse=winter_sports
    bbox = f"{miny},{minx},{maxy},{maxx}"
    overpass_query = f"""
    [out:json][timeout:90];
    (
      // Nordic/XC ski trails and pistes
      way["piste:type"="nordic"]({bbox});
      relation["piste:type"="nordic"]({bbox});
      way["piste:type"="skitour"]({bbox});

      // Cross-country skiing sport facilities
      way["sport"="cross-country_skiing"]({bbox});
      node["sport"="cross-country_skiing"]({bbox});
      way["sport"~"skiing"]({bbox});

      // Winter sports areas and ski resorts with nordic trails
      way["landuse"="winter_sports"]({bbox});
      node["landuse"="winter_sports"]({bbox});
      way["leisure"="sports_centre"]["sport"~"skiing"]({bbox});

      // Named ski trails and areas
      way["route"="ski"]({bbox});
      relation["route"="ski"]({bbox});
    );
    out center;
    """

    try:
        overpass_url = "https://overpass-api.de/api/interpreter"
        logger.info("  Sending query to Overpass API (90s timeout)...")
        response = requests.post(overpass_url, data={"data": overpass_query}, timeout=120)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.warning(f"Failed to fetch from Overpass API: {e}")
        logger.info("Using mock parks instead")
        return generate_mock_parks(extent)

    # Parse results and deduplicate by name
    parks = []
    seen_names = set()

    for element in data.get("elements", []):
        try:
            # Get coordinates
            if element["type"] == "way":
                if "center" not in element:
                    continue
                lat = element["center"]["lat"]
                lon = element["center"]["lon"]
            elif element["type"] == "relation":
                # Relations may have bounds instead of center
                if "center" in element:
                    lat = element["center"]["lat"]
                    lon = element["center"]["lon"]
                else:
                    continue
            else:
                lat = element.get("lat")
                lon = element.get("lon")
                if lat is None or lon is None:
                    continue

            # Get name (prefer name, fallback to piste:name or ID)
            tags = element.get("tags", {})
            name = tags.get("name") or tags.get("piste:name") or f"Trail_{element['id']}"

            # Deduplicate by name (keep first occurrence)
            if name in seen_names:
                continue
            seen_names.add(name)

            parks.append({
                "name": name,
                "lat": lat,
                "lon": lon,
                "osm_id": element["id"],
                "osm_type": element["type"],
            })
        except KeyError:
            continue

    logger.info(f"Found {len(parks)} XC skiing locations from OpenStreetMap")

    # Save as GeoJSON if requested
    if output_path and parks:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
                    "properties": {"name": p["name"]}
                }
                for p in parks
            ]
        }
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        logger.info(f"✓ Parks saved to: {output_path}")

    return parks


def generate_mock_parks(extent) -> list[dict]:
    """
    Generate mock parks using real Great Lakes region XC skiing coordinates.

    Uses actual park coordinates (not random) to ensure markers are on land
    and represent realistic skiing destinations across MI, WI, OH, IN, IL.
    """
    minx, miny, maxx, maxy = extent

    # Real XC skiing parks across the Great Lakes region
    # Covers the full DEM extent: lon[-89, -78], lat[37, 47]
    real_parks = [
        # Michigan - Southeast (Detroit area)
        {"name": "Stony Creek Metropark", "lat": 42.7377, "lon": -83.0583},
        {"name": "Kensington Metropark", "lat": 42.5303, "lon": -83.6581},
        {"name": "Indian Springs Metropark", "lat": 42.6514, "lon": -83.5028},
        {"name": "Proud Lake Recreation Area", "lat": 42.5583, "lon": -83.5553},
        {"name": "Bald Mountain Recreation Area", "lat": 42.7958, "lon": -83.1961},
        {"name": "Pinckney Recreation Area", "lat": 42.4214, "lon": -84.0461},
        # Michigan - Central/North
        {"name": "Sleepy Hollow State Park", "lat": 42.9089, "lon": -84.4233},
        {"name": "Hartwick Pines State Park", "lat": 44.7442, "lon": -84.6647},
        {"name": "Cross Country Ski Headquarters (Roscommon)", "lat": 44.5031, "lon": -84.5917},
        {"name": "Hanson Hills Recreation Area", "lat": 44.6553, "lon": -84.7372},
        # Michigan - Upper Peninsula (partial coverage)
        {"name": "Tahquamenon Falls State Park", "lat": 46.5983, "lon": -85.2072},
        # Wisconsin
        {"name": "Lapham Peak State Park", "lat": 43.0408, "lon": -88.3978},
        {"name": "Kettle Moraine State Forest", "lat": 43.0036, "lon": -88.1467},
        {"name": "Blue Mound State Park", "lat": 43.0256, "lon": -89.8389},
        {"name": "Devil's Lake State Park", "lat": 43.4175, "lon": -89.7289},
        {"name": "Mirror Lake State Park", "lat": 43.5558, "lon": -89.8150},
        # Ohio
        {"name": "Cuyahoga Valley National Park", "lat": 41.2808, "lon": -81.5678},
        {"name": "Mohican State Park", "lat": 40.6117, "lon": -82.3111},
        {"name": "Salt Fork State Park", "lat": 40.1172, "lon": -81.5067},
        {"name": "Findley State Park", "lat": 41.1203, "lon": -82.2311},
        # Indiana
        {"name": "Pokagon State Park", "lat": 41.7078, "lon": -85.0178},
        {"name": "Chain O'Lakes State Park", "lat": 41.3269, "lon": -85.5217},
        {"name": "Potato Creek State Park", "lat": 41.5456, "lon": -86.3564},
        {"name": "Indiana Dunes National Park", "lat": 41.6533, "lon": -87.0522},
        # Illinois
        {"name": "Starved Rock State Park", "lat": 41.3197, "lon": -88.9942},
        {"name": "Matthiessen State Park", "lat": 41.2867, "lon": -89.0197},
        {"name": "Rock Cut State Park", "lat": 42.3983, "lon": -89.0342},
        # Kentucky (southern edge of DEM)
        {"name": "Big Bone Lick State Park", "lat": 38.8828, "lon": -84.7528},
    ]

    # Filter to parks within the extent
    parks = [
        p for p in real_parks
        if minx <= p["lon"] <= maxx and miny <= p["lat"] <= maxy
    ]

    logger.info(f"Found {len(parks)} real parks within extent (from {len(real_parks)} known)")
    return parks


# =============================================================================
# PARK SCORING
# =============================================================================

def compute_park_scores(
    parks: list[dict],
    score_grid: np.ndarray,
    transform: Affine,
    radius_km: float = 1.0,
) -> list[dict]:
    """
    Score each park based on vicinity snow suitability.

    Args:
        parks: List of parks with name, lat, lon
        score_grid: 2D array of XC skiing suitability scores (0-1)
        transform: Affine transform for grid
        radius_km: Radius in km for vicinity averaging

    Returns:
        Parks with added "score" and "pixel_coords" fields
    """
    logger.info(f"Scoring parks based on {radius_km}km vicinity...")
    scored_parks = []

    for park in parks:
        try:
            # Convert lat/lon to pixel coordinates
            row, col = rowcol(transform, park["lon"], park["lat"])

            # Skip if outside grid
            if not (0 <= row < score_grid.shape[0] and 0 <= col < score_grid.shape[1]):
                logger.debug(f"Park {park['name']} outside grid bounds, skipping")
                continue

            # Compute radius in pixels
            # At Detroit latitude (~42°), 1° ≈ 85km
            pixel_size_deg = transform.a
            radius_pixels = max(1, int(radius_km / (pixel_size_deg * 85)))

            # Extract vicinity window
            r_min = max(0, int(row) - radius_pixels)
            r_max = min(score_grid.shape[0], int(row) + radius_pixels + 1)
            c_min = max(0, int(col) - radius_pixels)
            c_max = min(score_grid.shape[1], int(col) + radius_pixels + 1)

            vicinity = score_grid[r_min:r_max, c_min:c_max]

            # Average score in vicinity
            if vicinity.size > 0:
                park_score = float(np.nanmean(vicinity))
            else:
                park_score = 0.0

            scored_parks.append({
                **park,
                "score": park_score,
                "pixel_coords": [int(row), int(col)],
            })
        except Exception as e:
            logger.warning(f"Error scoring park {park['name']}: {e}")

    # Sort by score descending
    scored_parks.sort(key=lambda p: p["score"], reverse=True)

    logger.info(f"Scored {len(scored_parks)} parks")
    if scored_parks:
        logger.info("Top 5 parks:")
        for p in scored_parks[:5]:
            logger.info(f"  {p['name']}: {p['score']:.3f}")

    return scored_parks


# =============================================================================
# ANALYSIS STEPS
# =============================================================================

def run_step_snow(
    output_dir: Path,
    dem: np.ndarray,
    transform: Affine,
    mock_data: bool = False,
    terrain: Optional["Terrain"] = None,
    snodas_dir: Optional[Path] = None,
) -> dict:
    """
    Load snow statistics using GriddedDataLoader with automatic tiling.

    For real data: Uses GriddedDataLoader with memory-safe tiled processing.
    For mock data: Creates arrays at a fixed safe resolution (MOCK_DATA_SHAPE).

    Returns:
        Snow statistics dict with arrays at processed resolution
    """
    logger.info("\n" + "=" * 70)
    logger.info("Step 2: Snow Statistics (Memory-Safe via GriddedDataLoader)")
    logger.info("=" * 70)

    if mock_data:
        # Create mock data at safe fixed resolution
        logger.info("Using mock snow data")
        snow_stats = create_mock_snow_data(MOCK_DATA_SHAPE)
        logger.info(f"Mock snow data shape: {MOCK_DATA_SHAPE}")
    else:
        # Try to load real SNODAS data using GriddedDataLoader
        if snodas_dir and snodas_dir.exists() and terrain:
            logger.info(f"Loading real SNODAS data from: {snodas_dir}")
            try:
                # Configure for memory-safe processing
                tile_config = TiledDataConfig(
                    max_output_pixels=4 * 1024 * 1024,  # 4M pixels max per array
                    target_tile_outputs=1000,  # 1000x1000 tiles
                    max_memory_percent=75.0,  # Conservative memory limit
                )

                # Define pipeline steps
                pipeline = [
                    ("load_snodas", batch_process_snodas_data, {}),
                    ("compute_stats", calculate_snow_statistics, {}),
                ]

                # Create loader with terrain context
                loader = GriddedDataLoader(
                    terrain=terrain,
                    cache_dir=Path("xc_skiing_cache"),
                    tile_config=tile_config,
                )

                # Run pipeline with automatic tiling
                result = loader.run_pipeline(
                    data_source=snodas_dir,
                    pipeline=pipeline,
                    cache_name="xc_snodas",
                )

                # Extract snow statistics (remove metadata)
                snow_stats = {k: v for k, v in result.items() if k not in ("metadata", "failed_files")}
                failed = result.get("failed_files", [])
                logger.info(f"✓ Loaded SNODAS data ({len(failed)} files failed)")

            except Exception as e:
                logger.warning(f"Failed to load SNODAS data: {e}")
                logger.info("Falling back to mock data")
                snow_stats = create_mock_snow_data(MOCK_DATA_SHAPE)
        else:
            if not snodas_dir:
                logger.warning("No SNODAS directory specified (use --snodas-dir)")
            elif snodas_dir and not snodas_dir.exists():
                logger.warning(f"SNODAS directory not found: {snodas_dir}")
            else:
                logger.warning("Terrain object not available for SNODAS processing")
            logger.info("Falling back to mock data")
            snow_stats = create_mock_snow_data(MOCK_DATA_SHAPE)

    # Log snow statistics info
    snow_shape = snow_stats["median_max_depth"].shape
    logger.info(f"Snow statistics shape: {snow_shape}")
    logger.info(f"  median_max_depth: {np.min(snow_stats['median_max_depth']):.1f}-{np.max(snow_stats['median_max_depth']):.1f} mm")
    logger.info(f"  mean_snow_day_ratio: {np.min(snow_stats['mean_snow_day_ratio']):.3f}-{np.max(snow_stats['mean_snow_day_ratio']):.3f}")

    # Visualize snow depth
    (output_dir / "01_snow").mkdir(parents=True, exist_ok=True)
    snow_viz, stride = downsample_for_viz(snow_stats["median_max_depth"])
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(snow_viz, cmap="cividis")
    ax.set_title("Median Max Snow Depth")
    plt.colorbar(im, ax=ax, label="Depth (mm)")
    output_path = output_dir / "01_snow" / "snow_depth.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Snow visualization: {output_path}")

    return snow_stats


def run_step_score(
    output_dir: Path,
    snow_stats: dict,
) -> np.ndarray:
    """
    Compute XC skiing suitability scores.

    Args:
        output_dir: Output directory
        snow_stats: Snow statistics dict

    Returns:
        Score grid at snow_stats resolution
    """
    logger.info("\n" + "=" * 70)
    logger.info("Step 3: XC Skiing Suitability Scoring")
    logger.info("=" * 70)

    # Compute derived inputs
    inputs = xc_skiing_compute_derived_inputs(snow_stats)
    logger.info("Input metrics (gridded):")
    for key, val in inputs.items():
        if isinstance(val, np.ndarray):
            logger.info(f"  {key}: shape={val.shape}, range=[{np.min(val):.3f}, {np.max(val):.3f}]")
        elif isinstance(val, (int, float)):
            logger.info(f"  {key}: {val:.3f}")

    # Compute scores grid (same shape as snow_stats arrays)
    score_grid = DEFAULT_XC_SKIING_SCORER.compute(inputs)

    # Get expected shape from snow stats
    expected_shape = snow_stats["median_max_depth"].shape

    # Handle scalar vs array results
    if np.isscalar(score_grid):
        score_grid = np.full(expected_shape, score_grid, dtype=np.float32)
    else:
        score_grid = np.asarray(score_grid, dtype=np.float32)

    # Ensure proper shape
    if score_grid.shape != expected_shape:
        logger.warning(f"Score shape {score_grid.shape} != expected {expected_shape}, broadcasting...")
        score_grid = np.full(expected_shape, score_grid.flat[0] if score_grid.size > 0 else 0.5, dtype=np.float32)

    logger.info(f"Score grid shape: {score_grid.shape}, range: [{score_grid.min():.3f}, {score_grid.max():.3f}]")

    # Visualize
    (output_dir / "02_xc_scores").mkdir(parents=True, exist_ok=True)
    viz_score, viz_stride = downsample_for_viz(score_grid)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(viz_score, cmap="viridis", vmin=0, vmax=1)
    ax.set_title("XC Skiing Suitability Score")
    plt.colorbar(im, ax=ax, label="Suitability (0-1)")
    output_path = output_dir / "02_xc_scores" / "xc_skiing_scores.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Score visualization: {output_path}")

    return score_grid


def run_step_parks(
    output_dir: Path,
    dem: np.ndarray,
    transform: Affine,
    score_grid: np.ndarray,
) -> tuple[list[dict], Affine]:
    """
    Fetch and score parks.

    Args:
        output_dir: Output directory
        dem: DEM array (for getting geographic bounds)
        transform: Affine transform for DEM
        score_grid: Score grid

    Returns:
        Tuple of (scored_parks list, score_transform Affine)
    """
    logger.info("\n" + "=" * 70)
    logger.info("Step 4: Park Location & Scoring")
    logger.info("=" * 70)

    # Get extent from transform (using full DEM dimensions for geographic bounds)
    rows, cols = dem.shape
    bounds = (
        transform.c,  # minx
        transform.f + transform.e * rows,  # miny
        transform.c + transform.a * cols,  # maxx
        transform.f,  # maxy
    )
    logger.info(f"DEM bounds (minx, miny, maxx, maxy): {bounds}")

    # Fetch parks from OpenStreetMap
    parks_geojson = output_dir / "xc_skiing_parks.geojson"
    parks = fetch_xc_skiing_parks(bounds, output_path=parks_geojson)

    if not parks:
        logger.warning("No parks found!")
        return []

    # Create transform for score_grid based on its shape and the DEM extent
    score_rows, score_cols = score_grid.shape
    minx, miny, maxx, maxy = bounds
    score_transform = Affine(
        (maxx - minx) / score_cols,  # pixel width
        0,
        minx,  # x origin
        0,
        -(maxy - miny) / score_rows,  # pixel height (negative for north-up)
        maxy,  # y origin
    )
    logger.info(f"Score grid transform: pixel size = {score_transform.a:.6f}°")

    # Score parks based on vicinity
    scored_parks = compute_park_scores(parks, score_grid, score_transform, radius_km=1.0)

    # Visualize parks on score map
    if scored_parks:
        (output_dir / "03_parks").mkdir(parents=True, exist_ok=True)
        viz_score, viz_stride = downsample_for_viz(score_grid)
        fig, ax = plt.subplots(figsize=(12, 9))
        im = ax.imshow(viz_score, cmap="viridis", vmin=0, vmax=1)

        # Plot parks as circles colored by their local skiing score
        park_cols = [p["pixel_coords"][1] / viz_stride for p in scored_parks]
        park_rows = [p["pixel_coords"][0] / viz_stride for p in scored_parks]
        park_scores = [p["score"] for p in scored_parks]

        scatter = ax.scatter(
            park_cols, park_rows,
            c=park_scores,
            cmap="plasma",
            vmin=0, vmax=1,
            s=120,  # marker size
            marker="o",  # circle marker
            edgecolors="white",
            linewidths=1.5,
            zorder=10,  # draw on top
        )

        ax.set_title(f"XC Skiing Parks ({len(scored_parks)} found)")
        plt.colorbar(im, ax=ax, label="Grid Suitability (0-1)")
        cbar_parks = plt.colorbar(scatter, ax=ax, label="Park Score (0-1)", shrink=0.8, pad=0.12)
        output_path = output_dir / "03_parks" / "parks_on_score_map.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"✓ Parks visualization: {output_path}")

    return scored_parks, score_transform


# =============================================================================
# OUTPUT SAVING
# =============================================================================

def save_outputs(
    score_grid: np.ndarray,
    scored_parks: list[dict],
    output_dir: Path,
    score_transform: Affine,
):
    """Save analysis outputs.

    Args:
        score_grid: Score grid array
        scored_parks: List of parks with scores
        output_dir: Directory for visualizations (docs/images/)
        score_transform: Affine transform for the score grid
    """
    logger.info("\n" + "=" * 70)
    logger.info("Step 5: Saving Outputs")
    logger.info("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save data files to examples/output/xc_skiing/ for use by other examples
    data_dir = Path("examples/output/xc_skiing")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Save score grid as .npz with transform metadata
    score_path = data_dir / "xc_skiing_scores.npz"
    # Save transform as tuple (a, b, c, d, e, f) for Affine reconstruction
    transform_tuple = (
        score_transform.a, score_transform.b, score_transform.c,
        score_transform.d, score_transform.e, score_transform.f
    )
    np.savez_compressed(score_path, score=score_grid, transform=transform_tuple, crs="EPSG:4326")
    logger.info(f"✓ Score grid saved: {score_path} ({score_path.stat().st_size / 1024:.1f} KB) (with transform)")

    # Save parks as .json
    parks_path = data_dir / "xc_skiing_parks.json"
    with open(parks_path, 'w') as f:
        json.dump(scored_parks, f, indent=2)
    logger.info(f"✓ Parks saved: {parks_path}")


# =============================================================================
# 3D RENDERING
# =============================================================================

def run_step_render_3d(
    output_dir: Path,
    dem: np.ndarray,
    transform: Affine,
    score_grid: np.ndarray,
    score_transform: Affine,
    scored_parks: list[dict],
):
    """
    Render 3D terrain with dual colormaps: elevation base + score overlay near parks.

    Args:
        output_dir: Output directory for render
        dem: DEM array
        transform: Affine transform for DEM
        score_grid: XC skiing suitability scores
        score_transform: Affine transform for score grid
        scored_parks: List of park dictionaries with lat/lon
    """
    logger.info("\n" + "=" * 70)
    logger.info("Step 6: 3D Terrain Rendering (Dual Colormap)")
    logger.info("=" * 70)

    try:
        import bpy
        from src.terrain.color_mapping import elevation_colormap
        from src.terrain.transforms import reproject_raster, flip_raster, scale_elevation
        from math import radians
        from src.terrain.core import setup_camera, setup_light, clear_scene

        # Clear scene
        clear_scene()

        # Create terrain with DEM
        logger.info("Creating terrain from DEM...")
        terrain = Terrain(dem, transform, dem_crs="EPSG:4326")

        # Add transforms
        terrain.add_transform(reproject_raster(src_crs="EPSG:4326", dst_crs="EPSG:32617"))
        terrain.add_transform(flip_raster(axis="horizontal"))
        terrain.add_transform(scale_elevation(scale_factor=0.0001))

        # Configure downsampling BEFORE applying transforms
        terrain.configure_for_target_vertices(target_vertices=1920 * 1080)

        # Apply transforms (including downsampling)
        terrain.apply_transforms()

        # Add score layer
        terrain.add_data_layer("score", score_grid, score_transform, "EPSG:4326", target_layer="dem")

        # Create mesh FIRST (needed for proximity mask)
        logger.info("Creating mesh...")
        mesh_obj = terrain.create_mesh(
            scale_factor=100,
            height_scale=15.0,  # Moderate vertical exaggeration
            center_model=True,
            boundary_extension=True,
        )

        if mesh_obj is None:
            logger.error("Failed to create mesh")
            return

        # Compute proximity mask for parks
        park_lons = np.array([p["lon"] for p in scored_parks])
        park_lats = np.array([p["lat"] for p in scored_parks])

        logger.info("Computing proximity mask for park zones...")
        park_mask = terrain.compute_proximity_mask(
            park_lons,
            park_lats,
            radius_meters=2000,  # 2km radius around parks
            cluster_threshold_meters=500,  # Merge parks within 500m
        )

        # Set dual colormaps (viridis family for perceptual uniformity)
        logger.info("Setting blended color mapping...")
        terrain.set_blended_color_mapping(
            base_colormap=lambda elev: elevation_colormap(elev, cmap_name="cividis"),
            base_source_layers=["dem"],
            overlay_colormap=lambda score: elevation_colormap(
                score, cmap_name="plasma", min_elev=0.0, max_elev=1.0
            ),
            overlay_source_layers=["score"],
            overlay_mask=park_mask,
        )

        # Compute colors (now with blending)
        terrain.compute_colors()

        # Re-apply colors to mesh
        from src.terrain.blender_integration import apply_vertex_colors

        apply_vertex_colors(mesh_obj, terrain.colors, terrain.y_valid, terrain.x_valid)

        # Setup camera
        camera_angle = (radians(60), 0, radians(0))
        camera = setup_camera(
            camera_angle=camera_angle,
            camera_location=(0, -15, 10),
            scale=1.0,
            focal_length=35,
            camera_type="PERSP",
        )

        # Setup lighting
        sun = setup_light(
            location=(10, -5, 15),
            angle=2,
            energy=3.5,
            rotation_euler=(radians(60), 0, radians(-30)),
        )

        # Render
        bpy.context.scene.render.resolution_x = 1920
        bpy.context.scene.render.resolution_y = 1080
        bpy.context.scene.render.image_settings.file_format = "PNG"

        render_path = output_dir / "04_render_3d" / "xc_skiing_terrain_3d.png"
        render_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.context.scene.render.filepath = str(render_path)

        logger.info("Rendering...")
        bpy.ops.render.render(write_still=True)
        logger.info(f"✓ 3D render saved: {render_path}")

    except ImportError:
        logger.warning("Blender Python API (bpy) not available - skipping 3D rendering")
        logger.info("  Install blender-python or run in Blender environment for 3D rendering")
    except Exception as e:
        logger.error(f"Error during 3D rendering: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Detroit Cross-Country Skiing Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with mock data (fast)
  python examples/detroit_xc_skiing.py --mock-data

  # Run with real SNODAS data
  python examples/detroit_xc_skiing.py --snodas-dir data/snodas_data

  # Specify output directory
  python examples/detroit_xc_skiing.py --output-dir ./outputs --snodas-dir data/snodas_data
        """,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/images/xc_skiing"),
        help="Output directory (default: docs/images/xc_skiing/)",
    )

    parser.add_argument(
        "--mock-data",
        action="store_true",
        help="Use mock data instead of real DEM/SNODAS",
    )

    parser.add_argument(
        "--snodas-dir",
        type=Path,
        help="Path to SNODAS data directory",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "=" * 70)
    logger.info("Detroit Cross-Country Skiing Analysis")
    logger.info("=" * 70)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Using mock data: {args.mock_data}")

    # Run analysis pipeline with memory-safe processing via GriddedDataLoader
    dem, transform = run_step_dem(args.output_dir, args.mock_data)

    # Create Terrain object for GriddedDataLoader (needed for real SNODAS processing)
    terrain = None
    if not args.mock_data:
        terrain = Terrain(dem, transform, dem_crs="EPSG:4326")

    # Load snow statistics (uses GriddedDataLoader for real data, mock for testing)
    snow_stats = run_step_snow(
        args.output_dir,
        dem,
        transform,
        mock_data=args.mock_data,
        terrain=terrain,
        snodas_dir=args.snodas_dir,
    )

    # Compute suitability scores
    score_grid = run_step_score(args.output_dir, snow_stats)

    # Find and score parks
    scored_parks, score_transform = run_step_parks(args.output_dir, dem, transform, score_grid)

    # Save outputs (with transform metadata for use by other examples)
    save_outputs(score_grid, scored_parks, args.output_dir, score_transform)

    # Render 3D visualization (if bpy available)
    if scored_parks:
        run_step_render_3d(
            args.output_dir,
            dem,
            transform,
            score_grid,
            score_transform,
            scored_parks,
        )

    logger.info("\n" + "=" * 70)
    logger.info("✓ Analysis complete!")
    logger.info(f"Outputs saved to: {args.output_dir}")
    logger.info("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\n[✗] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n[✗] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
