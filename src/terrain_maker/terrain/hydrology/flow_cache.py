"""Caching of flow_accumulation results keyed on inputs and parameters."""

import logging
from pathlib import Path
from typing import Dict, Optional
import datetime
import json
import os

import rasterio

logger = logging.getLogger(__name__)


# ============================================================================
# Flow Computation Caching
# ============================================================================


def _get_cache_key_params(
    dem_path: str,
    backend: str,
    max_cells: Optional[int],
    target_vertices: Optional[int],
    fill_method: str,
    mask_ocean: bool,
    ocean_elevation_threshold: float,
    coastal_elev_threshold: float,
    edge_mode: str,
    max_breach_depth: float,
    max_breach_length: int,
    epsilon: float,
) -> Dict:
    """
    Build dictionary of parameters that affect cache validity.

    Returns
    -------
    dict
        Parameters that form the cache key
    """
    return {
        "dem_path": str(dem_path),
        "backend": backend,
        "max_cells": max_cells,
        "target_vertices": target_vertices,
        "fill_method": fill_method,
        "mask_ocean": mask_ocean,
        "ocean_elevation_threshold": ocean_elevation_threshold,
        "coastal_elev_threshold": coastal_elev_threshold,
        "edge_mode": edge_mode,
        "max_breach_depth": max_breach_depth,
        "max_breach_length": max_breach_length,
        "epsilon": epsilon,
    }


def _get_dem_mtime(dem_path: Path) -> float:
    """Get modification time of DEM file."""
    return os.path.getmtime(dem_path)


def _validate_cache(
    cache_dir: Path,
    cache_params: Dict,
    dem_mtime: float,
) -> bool:
    """
    Check if valid cache exists.

    Parameters
    ----------
    cache_dir : Path
        Directory containing cached files
    cache_params : dict
        Current computation parameters
    dem_mtime : float
        Modification time of DEM file

    Returns
    -------
    bool
        True if cache is valid, False otherwise
    """
    metadata_file = cache_dir / "flow_cache_metadata.json"
    if not metadata_file.exists():
        return False

    # Check all required output files exist
    required_files = [
        "flow_direction.tif",
        "flow_accumulation_area.tif",
        "flow_accumulation_rainfall.tif",
        "dem_conditioned.tif",
    ]
    for filename in required_files:
        if not (cache_dir / filename).exists():
            return False

    # Load and validate metadata
    try:
        with open(metadata_file) as f:
            cached_metadata = json.load(f)
    except (json.JSONDecodeError, IOError):
        return False

    # Check DEM modification time
    cached_dem_mtime = cached_metadata.get("dem_mtime")
    if cached_dem_mtime is None or cached_dem_mtime != dem_mtime:
        return False

    # Check all cache key parameters match
    cached_params = cached_metadata.get("cache_params", {})
    for key, value in cache_params.items():
        if cached_params.get(key) != value:
            return False

    return True


def _load_from_cache(cache_dir: Path) -> Dict:
    """
    Load flow computation results from cache.

    Parameters
    ----------
    cache_dir : Path
        Directory containing cached files

    Returns
    -------
    dict
        Dictionary with flow_direction, drainage_area, upstream_rainfall,
        conditioned_dem, metadata, and files
    """
    # Load metadata
    metadata_file = cache_dir / "flow_cache_metadata.json"
    with open(metadata_file) as f:
        full_metadata = json.load(f)

    # Extract just the computation metadata (not cache params)
    metadata = full_metadata.get("computation_metadata", {})
    metadata["cache_hit"] = True

    # Load rasters
    files = {
        "flow_direction": str(cache_dir / "flow_direction.tif"),
        "drainage_area": str(cache_dir / "flow_accumulation_area.tif"),
        "upstream_rainfall": str(cache_dir / "flow_accumulation_rainfall.tif"),
        "conditioned_dem": str(cache_dir / "dem_conditioned.tif"),
    }

    with rasterio.open(files["flow_direction"]) as src:
        flow_direction = src.read(1)
    with rasterio.open(files["drainage_area"]) as src:
        drainage_area = src.read(1)
    with rasterio.open(files["upstream_rainfall"]) as src:
        upstream_rainfall = src.read(1)
    with rasterio.open(files["conditioned_dem"]) as src:
        conditioned_dem = src.read(1)

    return {
        "flow_direction": flow_direction,
        "drainage_area": drainage_area,
        "upstream_rainfall": upstream_rainfall,
        "conditioned_dem": conditioned_dem,
        "breached_dem": None,  # Not saved to cache, available only on first run
        "metadata": metadata,
        "files": files,
    }


def _save_to_cache(
    cache_dir: Path,
    cache_params: Dict,
    dem_mtime: float,
    result: Dict,
) -> None:
    """
    Save flow computation results to cache.

    Parameters
    ----------
    cache_dir : Path
        Directory for cached files
    cache_params : dict
        Parameters that form the cache key
    dem_mtime : float
        Modification time of DEM file
    result : dict
        Flow computation results
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    metadata_file = cache_dir / "flow_cache_metadata.json"
    full_metadata = {
        # Top-level fields for easy access
        "dem_path": cache_params["dem_path"],
        "backend": cache_params["backend"],
        "timestamp": datetime.datetime.now().isoformat(),
        # Full cache params for validation
        "cache_params": cache_params,
        "dem_mtime": dem_mtime,
        "computation_metadata": result["metadata"],
    }
    with open(metadata_file, "w") as f:
        json.dump(full_metadata, f, indent=2)
