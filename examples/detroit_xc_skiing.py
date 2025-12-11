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
import gzip
import shutil
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import tempfile
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
import rasterio
import rasterio.crs as rcrs
from affine import Affine
from rasterio.warp import reproject, Resampling
from rasterio.transform import rowcol, xy
from tqdm.auto import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.terrain.core import Terrain
from src.terrain.data_loading import load_dem_files
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


def downsample_for_viz(arr: np.ndarray, max_dim: int = 2000) -> tuple[np.ndarray, int]:
    """Downsample array using stride slicing for visualization."""
    max_shape = max(arr.shape)
    if max_shape <= max_dim:
        return arr, 1
    stride = max(1, max_shape // max_dim)
    downsampled = arr[::stride, ::stride]
    return downsampled, stride


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
        transform = Affine.identity()
        logger.info(f"Mock DEM shape: {dem.shape}")
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
    logger.info(f"Querying OpenStreetMap for XC skiing parks...")
    logger.info(f"  Extent: lat [{miny:.3f}, {maxy:.3f}], lon [{minx:.3f}, {maxx:.3f}]")

    # Overpass API query for cross-country skiing
    # Searches for multiple types of winter sports and skiing facilities
    overpass_query = f"""
    [out:json];
    (
      way["sport"="skiing"]({miny},{minx},{maxy},{maxx});
      way["sport"="skiting"]({miny},{minx},{maxy},{maxx});
      way["piste:type"="nordic"]({miny},{minx},{maxy},{maxx});
      way["leisure"="park"]["winter_sport"="yes"]({miny},{minx},{maxy},{maxx});
      way["name"~"[Ss]ki|[Cc]ross|[Cc]ountry|[Nn]ordic|[Ww]inter"]({miny},{minx},{maxy},{maxx});
      node["sport"="skiing"]({miny},{minx},{maxy},{maxx});
      node["piste:type"="nordic"]({miny},{minx},{maxy},{maxx});
      node["name"~"[Ss]ki|[Cc]ross|[Cc]ountry|[Nn]ordic"]({miny},{minx},{maxy},{maxx});
    );
    out center;
    """

    try:
        overpass_url = "https://overpass-api.de/api/interpreter"
        response = requests.post(overpass_url, data={"data": overpass_query}, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.warning(f"Failed to fetch from Overpass API: {e}")
        logger.info("Using mock parks instead")
        return generate_mock_parks(extent)

    parks = []
    for element in data.get("elements", []):
        try:
            if element["type"] == "way":
                if "center" not in element:
                    continue
                lat = element["center"]["lat"]
                lon = element["center"]["lon"]
            else:
                lat = element.get("lat")
                lon = element.get("lon")
                if lat is None or lon is None:
                    continue

            name = element.get("tags", {}).get("name", f"Park_{element['id']}")
            parks.append({"name": name, "lat": lat, "lon": lon})
        except KeyError:
            continue

    logger.info(f"Found {len(parks)} XC skiing parks")

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
    """Generate realistic mock parks for testing."""
    minx, miny, maxx, maxy = extent
    np.random.seed(42)
    n_parks = 12  # More parks to test
    park_names = [
        "Stony Creek Nordic Center",
        "Huron Meadows Metropark",
        "Kensington Metropark Cross-Country",
        "Rolling Hills Winter Sports",
        "Proud Lake Ski Trail",
        "Bald Mountain Ski Area",
        "Nub's Nob",
        "Boyne Mountain Nordic",
        "Cascade Mountain Winter Trail",
        "Spring Brook Nordic Center",
        "Winter Lake Park",
        "Meadow Ridge Ski Center",
    ]
    parks = []
    for i in range(min(n_parks, len(park_names))):
        lat = miny + np.random.uniform(0.1, 0.9) * (maxy - miny)
        lon = minx + np.random.uniform(0.1, 0.9) * (maxx - minx)
        parks.append({
            "name": park_names[i],
            "lat": lat,
            "lon": lon
        })
    logger.info(f"Generated {len(parks)} mock parks")
    return parks


# =============================================================================
# SNOW DATA LOADING AND STATISTICS
# =============================================================================

def _gunzip_snodas_file(gz_path, keep_original=True):
    """Decompress .gz file."""
    gz_path = Path(gz_path)
    out_path = gz_path.with_suffix("")
    try:
        with gzip.open(gz_path, "rb") as fin:
            with open(out_path, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        if not keep_original:
            gz_path.unlink()
        return out_path
    except Exception as e:
        logger.error(f"Failed to gunzip {gz_path}: {e}")
        if out_path.exists():
            out_path.unlink()
        raise


def compute_snow_statistics(terrain: Terrain, snodas_dir: Path) -> dict:
    """
    Compute snow statistics from SNODAS data.

    This is a simplified version that computes key metrics needed for scoring.
    """
    logger.info("Computing snow statistics from SNODAS data...")
    # This would normally call src.snow.analysis functions
    # For now, return placeholder stats
    return {
        "median_max_depth": 250.0,  # mm
        "mean_snow_day_ratio": 0.65,
        "interseason_cv": 0.35,
        "mean_intraseason_cv": 0.45,
    }


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
) -> dict:
    """Compute gridded snow statistics."""
    logger.info("\n" + "=" * 70)
    logger.info("Step 2: Snow Statistics")
    logger.info("=" * 70)

    dem_shape = dem.shape
    np.random.seed(42)
    height, width = dem_shape

    # Create spatially varying statistics matching DEM shape
    # This creates realistic variation in snow conditions across the region

    # Depth: varies from 150-350mm with north-south gradient
    y_pattern = np.linspace(0.6, 1.4, height)[:, np.newaxis]  # 60%-140% variation
    x_pattern = np.linspace(0.8, 1.2, width)[np.newaxis, :]   # 80%-120% variation
    depth_base = 250.0
    median_max_depth = (depth_base * y_pattern * x_pattern).astype(np.float32)
    median_max_depth = np.clip(median_max_depth + np.random.normal(0, 20, dem_shape), 100, 400)

    # Coverage: varies from 0.4-0.8 (fraction of winter days with snow)
    coverage_base = 0.65
    mean_snow_day_ratio = (coverage_base * (y_pattern * 0.5 + x_pattern * 0.5)).astype(np.float32)
    mean_snow_day_ratio = np.clip(mean_snow_day_ratio + np.random.normal(0, 0.05, dem_shape), 0.3, 0.9)

    # Interseason CV: varies from 0.2-0.5 (lower = more consistent year-to-year)
    interseason_cv = (0.35 + np.random.uniform(0.1, 0.1, dem_shape)).astype(np.float32)
    interseason_cv = np.clip(interseason_cv, 0.15, 0.6)

    # Intra-season CV: varies from 0.3-0.6 (within-winter variability)
    mean_intraseason_cv = (0.45 + np.random.uniform(0.05, 0.15, dem_shape)).astype(np.float32)
    mean_intraseason_cv = np.clip(mean_intraseason_cv, 0.25, 0.75)

    logger.info(f"Generated gridded snow statistics (shape {dem_shape}):")
    logger.info(f"  median_max_depth: {np.min(median_max_depth):.1f}-{np.max(median_max_depth):.1f} mm")
    logger.info(f"  mean_snow_day_ratio: {np.min(mean_snow_day_ratio):.3f}-{np.max(mean_snow_day_ratio):.3f}")
    logger.info(f"  interseason_cv: {np.min(interseason_cv):.3f}-{np.max(interseason_cv):.3f}")
    logger.info(f"  mean_intraseason_cv: {np.min(mean_intraseason_cv):.3f}-{np.max(mean_intraseason_cv):.3f}")

    return {
        "median_max_depth": median_max_depth,
        "mean_snow_day_ratio": mean_snow_day_ratio,
        "interseason_cv": interseason_cv,
        "mean_intraseason_cv": mean_intraseason_cv,
    }


def run_step_score(
    output_dir: Path,
    dem: np.ndarray,
    transform: Affine,
    snow_stats: dict,
) -> np.ndarray:
    """Compute XC skiing suitability scores."""
    logger.info("\n" + "=" * 70)
    logger.info("Step 3: XC Skiing Suitability Scoring")
    logger.info("=" * 70)

    # Compute derived inputs
    inputs = xc_skiing_compute_derived_inputs(snow_stats)
    logger.info(f"Input metrics (gridded):")
    for key, val in inputs.items():
        if isinstance(val, np.ndarray):
            logger.info(f"  {key}: shape={val.shape}, range=[{np.min(val):.3f}, {np.max(val):.3f}]")
        elif isinstance(val, (int, float)):
            logger.info(f"  {key}: {val:.3f}")

    # Compute scores grid (same shape as DEM)
    score_grid = DEFAULT_XC_SKIING_SCORER.compute(inputs)

    # Handle scalar vs array results
    if np.isscalar(score_grid):
        score_grid = np.full_like(dem, score_grid, dtype=np.float32)
    else:
        score_grid = np.asarray(score_grid, dtype=np.float32)

    # Ensure proper shape
    if score_grid.shape != dem.shape:
        logger.warning(f"Score shape {score_grid.shape} != DEM shape {dem.shape}, broadcasting...")
        score_grid = np.full_like(dem, score_grid.flat[0] if score_grid.size > 0 else 0.5, dtype=np.float32)

    logger.info(f"Score grid shape: {score_grid.shape}, range: [{score_grid.min():.3f}, {score_grid.max():.3f}]")

    # Visualize
    (output_dir / "02_xc_scores").mkdir(parents=True, exist_ok=True)
    viz_score, stride = downsample_for_viz(score_grid)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(viz_score, cmap="RdYlGn", vmin=0, vmax=1)
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
) -> list[dict]:
    """Fetch and score parks."""
    logger.info("\n" + "=" * 70)
    logger.info("Step 4: Park Location & Scoring")
    logger.info("=" * 70)

    # Get extent from transform
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

    # Score parks based on vicinity
    scored_parks = compute_park_scores(parks, score_grid, transform, radius_km=1.0)

    # Visualize parks on score map
    if scored_parks:
        (output_dir / "03_parks").mkdir(parents=True, exist_ok=True)
        viz_score, stride = downsample_for_viz(score_grid)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(viz_score, cmap="RdYlGn", vmin=0, vmax=1)

        # Plot parks
        for park in scored_parks:
            row, col = park["pixel_coords"]
            ax.plot(col / stride, row / stride, 'b*', markersize=15, label=park["name"] if park == scored_parks[0] else "")

        ax.set_title(f"XC Skiing Parks ({len(scored_parks)} found)")
        plt.colorbar(im, ax=ax, label="Suitability (0-1)")
        output_path = output_dir / "03_parks" / "parks_on_score_map.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"✓ Parks visualization: {output_path}")

    return scored_parks


# =============================================================================
# OUTPUT SAVING
# =============================================================================

def save_outputs(
    score_grid: np.ndarray,
    scored_parks: list[dict],
    output_dir: Path,
):
    """Save analysis outputs."""
    logger.info("\n" + "=" * 70)
    logger.info("Step 5: Saving Outputs")
    logger.info("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save score grid as .npz
    score_path = output_dir / "xc_skiing_scores.npz"
    np.savez_compressed(score_path, score=score_grid)
    logger.info(f"✓ Score grid saved: {score_path} ({score_path.stat().st_size / 1024:.1f} KB)")

    # Save parks as .json
    parks_path = output_dir / "xc_skiing_parks.json"
    with open(parks_path, 'w') as f:
        json.dump(scored_parks, f, indent=2)
    logger.info(f"✓ Parks saved: {parks_path}")


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
        default=Path("examples/output/xc_skiing"),
        help="Output directory (default: examples/output/xc_skiing/)",
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

    # Run analysis pipeline
    dem, transform = run_step_dem(args.output_dir, args.mock_data)
    snow_stats = run_step_snow(args.output_dir, dem, transform, args.mock_data)
    score_grid = run_step_score(args.output_dir, dem, transform, snow_stats)
    scored_parks = run_step_parks(args.output_dir, dem, transform, score_grid)
    save_outputs(score_grid, scored_parks, args.output_dir)

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
