"""
Flow accumulation module for hydrological analysis.

Implements D8 flow routing algorithm with precipitation weighting.
Based on flow-spec.md requirements.

Supports two backends:
- "pysheds": Uses pysheds library for core hydrology (recommended)
- "custom": Uses custom numba-accelerated implementation
"""

from pathlib import Path
from typing import Dict, Tuple, Union, Optional, Literal
import datetime
import json
import os
import shutil
import numpy as np
import rasterio
from rasterio import Affine
from scipy import ndimage
from scipy.ndimage import grey_dilation, grey_erosion

# Try to import pysheds
try:
    from pysheds.grid import Grid as PyshedsGrid
    PYSHEDS_AVAILABLE = True
except ImportError:
    PYSHEDS_AVAILABLE = False
    PyshedsGrid = None

# Try to import numba for performance optimizations
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create no-op decorator if numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    # Mock prange as regular range
    prange = range


# D8 flow direction encoding: (row_offset, col_offset) -> direction_code
# Directions follow standard D8 encoding used in ArcGIS/GRASS
D8_DIRECTIONS = {
    (0, 1): 1,  # East
    (-1, 1): 2,  # Northeast
    (-1, 0): 4,  # North
    (-1, -1): 8,  # Northwest
    (0, -1): 16,  # West
    (1, -1): 32,  # Southwest
    (1, 0): 64,  # South
    (1, 1): 128,  # Southeast
}

# Reverse mapping for flow routing
D8_OFFSETS = {v: k for k, v in D8_DIRECTIONS.items()}


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


def flow_accumulation(
    dem_path: str,
    precipitation_path: str,
    output_dir: Optional[str] = None,
    flow_algorithm: str = "d8",
    # Legacy parameters (used when backend="legacy" or "custom")
    fill_method: str = "breach",
    min_basin_size: Optional[int] = 10000,
    max_fill_depth: Optional[float] = None,
    # Spec-compliant parameters (used when backend="spec")
    coastal_elev_threshold: float = 10.0,
    edge_mode: Literal["all", "local_minima", "outward_slope", "none"] = "all",
    max_breach_depth: float = 50.0,
    max_breach_length: int = 100,
    epsilon: float = 1e-4,
    masked_basin_outlets: Optional[np.ndarray] = None,
    # Common parameters
    cell_size: Optional[float] = None,
    max_cells: Optional[int] = None,
    target_vertices: Optional[int] = None,
    mask_ocean: bool = True,
    ocean_elevation_threshold: float = 0.0,
    backend: Literal["legacy", "spec", "pysheds"] = "spec",
    lake_mask: Optional[np.ndarray] = None,
    lake_outlets: Optional[np.ndarray] = None,
    # Caching parameters
    cache: bool = False,
    cache_dir: Optional[str] = None,
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Compute flow accumulation with precipitation weighting.

    Automatically downsamples DEM if it exceeds max_cells to improve performance.

    Parameters
    ----------
    dem_path : str
        Path to DEM raster file (GeoTIFF)
    precipitation_path : str
        Path to annual precipitation raster (mm/year)
    output_dir : str, optional
        Directory for output files (default: same as DEM)
    flow_algorithm : str, default 'd8'
        Flow routing method ('d8' only for now)
    fill_method : str, default 'breach'
        Depression handling ('breach' or 'fill')
    cell_size : float, optional
        Override DEM resolution in meters
    max_cells : int, optional
        Maximum number of cells for flow computation. DEM will be downsampled
        if it exceeds this limit. Mutually exclusive with target_vertices.
    target_vertices : int, optional
        Target number of vertices for final rendering. Automatically sets
        max_cells = target_vertices * 3 for flow accuracy.
        Mutually exclusive with max_cells.
    mask_ocean : bool, default True
        If True, detect and exclude ocean/water bodies from flow computation.
        Ocean cells (elevation <= ocean_elevation_threshold, connected to border)
        will not be filled and will have flow_dir = 0.
    ocean_elevation_threshold : float, default 0.0
        Elevation threshold (meters) for ocean detection. Cells at or below
        this elevation connected to the border are considered ocean.
    min_basin_size : int, optional, default 10000
        Minimum basin size (cells) to preserve. Endorheic basins >= this size
        will not be filled (preserves large natural basins like Salton Sea).
        Set to None to disable basin preservation.
    max_fill_depth : float, optional
        Maximum fill depth (meters). Depressions requiring fill > this depth
        will be preserved. Set to None to allow unlimited fill depth.
    backend : {"legacy", "spec", "pysheds"}, default "spec"
        Which implementation to use for core hydrology algorithms:
        - "spec": Spec-compliant 4-stage pipeline (outlet ID + breaching + fill) - RECOMMENDED
        - "legacy": Morphological reconstruction + workarounds (deprecated)
        - "pysheds": PySheds library integration (experimental, may produce cycles)

    Legacy parameters (used when backend="legacy" or backend="pysheds"):
        fill_method : str, default "breach"
            Depression handling ("breach" or "fill")
        min_basin_size : int, default 10000
            Minimum basin size (cells) to preserve
        max_fill_depth : float, optional
            Maximum fill depth (meters) for preserving deep basins

    Spec-compliant parameters (used when backend="spec"):
        coastal_elev_threshold : float, default 10.0
            Maximum elevation for coastal outlets (meters above sea level)
        edge_mode : {"all", "local_minima", "outward_slope", "none"}, default "all"
            Boundary outlet strategy (see flow-spec.md for details)
        max_breach_depth : float, default 50.0
            Maximum elevation drop at any cell during breaching (meters)
        max_breach_length : int, default 100
            Maximum breach path length (cells)
        epsilon : float, default 1e-4
            Minimum gradient in filled areas (meters/cell)
        masked_basin_outlets : np.ndarray (bool), optional
            User-supplied outlet locations for known lakes/basins
    lake_mask : np.ndarray, optional
        Labeled mask of known water bodies (0 = no lake, N = lake ID).
        Lake interior cells will route flow toward their outlets.
    lake_outlets : np.ndarray (bool), optional
        Boolean mask of lake outlet cells. Required if lake_mask is provided.
        Outlets receive accumulated flow from all lake cells.
    cache : bool, default False
        If True, cache computation results and load from cache if valid.
        Cache is invalidated if DEM file is modified or parameters change.
    cache_dir : str, optional
        Directory for cached files. Defaults to DEM's directory if not specified.

    Returns
    -------
    dict
        Dictionary with keys:
        - flow_direction: np.ndarray (D8 encoded)
        - drainage_area: np.ndarray (cells draining to each pixel)
        - upstream_rainfall: np.ndarray (mm·m² total upstream)
        - conditioned_dem: np.ndarray (pit-filled DEM)
        - metadata: dict (processing info, including downsampling details)
        - files: dict (output file paths)

    Raises
    ------
    FileNotFoundError
        If DEM or precipitation file doesn't exist
    ValueError
        If spatial alignment fails or both max_cells and target_vertices specified
    """
    # Validate inputs
    print("  flow_accumulation: validating inputs...", flush=True)
    dem_path = Path(dem_path)
    precip_path = Path(precipitation_path)

    if not dem_path.exists():
        raise FileNotFoundError(f"DEM file not found: {dem_path}")
    if not precip_path.exists():
        raise FileNotFoundError(f"Precipitation file not found: {precip_path}")

    # Validate max_cells and target_vertices
    if max_cells is not None and target_vertices is not None:
        raise ValueError("Cannot specify both max_cells and target_vertices")

    # Calculate max_cells from target_vertices if specified
    original_target_vertices = target_vertices
    if target_vertices is not None:
        # Use 3x target_vertices for flow accuracy
        max_cells = target_vertices * 3

    # === CACHE CHECK ===
    if cache:
        # Determine cache directory
        if cache_dir is None:
            cache_path = dem_path.parent
        else:
            cache_path = Path(cache_dir)

        # Build cache key parameters
        cache_params = _get_cache_key_params(
            dem_path=str(dem_path),
            backend=backend,
            max_cells=max_cells,
            target_vertices=original_target_vertices,
            fill_method=fill_method,
            mask_ocean=mask_ocean,
            ocean_elevation_threshold=ocean_elevation_threshold,
            coastal_elev_threshold=coastal_elev_threshold,
            edge_mode=edge_mode,
            max_breach_depth=max_breach_depth,
            max_breach_length=max_breach_length,
            epsilon=epsilon,
        )

        # Check cache validity
        dem_mtime = _get_dem_mtime(dem_path)
        if _validate_cache(cache_path, cache_params, dem_mtime):
            print("  flow_accumulation: loading from cache...", flush=True)
            return _load_from_cache(cache_path)

        print("  flow_accumulation: cache miss, computing...", flush=True)

    # Load DEM
    print("  flow_accumulation: loading DEM...", flush=True)
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1).astype(np.float32)
        dem_transform = src.transform
        dem_crs = src.crs
        original_shape = dem_data.shape
    print(f"  flow_accumulation: DEM loaded {original_shape}", flush=True)

    # Adaptive resolution: downsample if DEM exceeds max_cells
    downsampling_applied = False
    downsample_factor = 1.0
    dem_shape = original_shape

    if max_cells is not None and (original_shape[0] * original_shape[1]) > max_cells:
        # Calculate downsample factor to achieve max_cells
        current_cells = original_shape[0] * original_shape[1]
        downsample_factor = np.sqrt(current_cells / max_cells)

        # Calculate new shape
        new_height = int(original_shape[0] / downsample_factor)
        new_width = int(original_shape[1] / downsample_factor)
        downsampled_shape = (new_height, new_width)

        print(f"  Downsampling DEM from {original_shape} ({current_cells:,} cells) to "
              f"{downsampled_shape} ({new_height * new_width:,} cells) "
              f"[{downsample_factor:.2f}x factor]...", flush=True)

        # Downsample DEM using rasterio
        from rasterio.warp import reproject, Resampling

        dem_downsampled = np.empty(downsampled_shape, dtype=np.float32)

        # Calculate new transform (larger pixels)
        downsampled_transform = dem_transform * Affine.scale(downsample_factor)

        reproject(
            source=dem_data,
            destination=dem_downsampled,
            src_transform=dem_transform,
            src_crs=dem_crs,
            dst_transform=downsampled_transform,
            dst_crs=dem_crs,
            resampling=Resampling.bilinear
        )

        dem_data = dem_downsampled
        dem_transform = downsampled_transform
        dem_shape = downsampled_shape
        downsampling_applied = True

        print(f"  ✓ Downsampled DEM to {dem_shape}", flush=True)
    elif max_cells is not None:
        print(f"DEM size ({original_shape[0] * original_shape[1]:,} cells) below max_cells ({max_cells:,}), "
              f"no downsampling needed")

    # Load precipitation
    print("  flow_accumulation: loading precipitation...", flush=True)
    with rasterio.open(precip_path) as src:
        precip_data = src.read(1).astype(np.float32)
        precip_transform = src.transform
        precip_crs = src.crs

        print(f"  flow_accumulation: precipitation loaded {precip_data.shape}", flush=True)

        # Check spatial alignment and resample if needed
        if precip_data.shape != dem_shape:
            print(f"  Resampling precipitation from {precip_data.shape} to match DEM {dem_shape}...", flush=True)

            # Resample precipitation to match DEM
            from rasterio.warp import reproject, Resampling

            precip_resampled = np.empty(dem_shape, dtype=np.float32)

            reproject(
                source=precip_data,
                destination=precip_resampled,
                src_transform=precip_transform,
                src_crs=precip_crs,
                dst_transform=dem_transform,
                dst_crs=dem_crs,
                resampling=Resampling.bilinear
            )

            precip_data = precip_resampled
            print(f"✓ Resampled precipitation to {precip_data.shape}")

    # Determine cell size (convert from degrees to meters if geographic CRS)
    if cell_size is None:
        from rasterio.crs import CRS
        import math

        # Check if CRS is geographic (lat/lon in degrees)
        if dem_crs is not None and CRS.from_user_input(dem_crs).is_geographic:
            # Cell size is in degrees - convert to meters using Haversine approximation
            # Calculate at center latitude for best accuracy
            pixel_size_deg = abs(dem_transform.a)

            # Get bounds to find center latitude
            height, width = dem_shape
            left = dem_transform.c
            top = dem_transform.f
            bottom = top + (height * dem_transform.e)
            center_lat = (top + bottom) / 2

            # Haversine approximation for degrees to meters
            # At center latitude
            lon_to_m = 111320 * math.cos(math.radians(center_lat))
            lat_to_m = 110540

            cell_width_m = pixel_size_deg * lon_to_m
            cell_height_m = abs(dem_transform.e) * lat_to_m

            # Use average of width and height for area calculations
            cell_size = math.sqrt(cell_width_m * cell_height_m)

            print(f"  Geographic CRS detected (cell size: {pixel_size_deg:.8f}° ≈ {cell_size:.1f}m at lat {center_lat:.2f}°)")
        else:
            # Projected CRS - pixel size is already in CRS units (meters)
            cell_size = abs(dem_transform.a)

    # Step 1: Detect ocean mask (if enabled)
    ocean_mask = None
    if mask_ocean:
        print(f"Detecting ocean (elevation <= {ocean_elevation_threshold}m, border-connected)...")
        ocean_mask = detect_ocean_mask(
            dem_data, threshold=ocean_elevation_threshold, border_only=True
        )
        ocean_cells = np.sum(ocean_mask)
        ocean_pct = 100 * ocean_cells / ocean_mask.size
        print(f"  Ocean detected: {ocean_cells:,} cells ({ocean_pct:.1f}%)")

    # Note: We DON'T mask basins from flow computation to allow internal drainage patterns.
    # Basins are only preserved from filling (via condition_dem), but flow is computed inside them.
    # This lets us see how water drains through closed basins like the Imperial Valley.

    # Use ocean mask for flow computation
    flow_mask = ocean_mask if ocean_mask is not None else None

    # Branch based on backend
    if backend == "spec":
        # === SPEC-COMPLIANT BACKEND ===
        # Use spec-compliant 4-stage pipeline (outlet ID + breaching + fill)
        print(f"Using spec-compliant backend (flow-spec.md)...")

        # Emit warnings if legacy parameters are specified
        if fill_method != "breach":
            import warnings
            warnings.warn(
                f"fill_method='{fill_method}' is ignored when backend='spec'. "
                "Use epsilon parameter instead (epsilon=0 for fill, epsilon>0 for breach-like).",
                DeprecationWarning
            )
        if min_basin_size != 10000:
            import warnings
            warnings.warn(
                "min_basin_size is ignored when backend='spec'. "
                "Use max_breach_depth/max_breach_length to control basin preservation.",
                DeprecationWarning
            )

        # Step 2: Condition DEM using spec-compliant pipeline
        conditioned_dem, outlets = condition_dem_spec(
            dem_data,
            nodata_mask=ocean_mask,
            coastal_elev_threshold=coastal_elev_threshold,
            edge_mode=edge_mode,
            max_breach_depth=max_breach_depth,
            max_breach_length=max_breach_length,
            epsilon=epsilon,
            masked_basin_outlets=masked_basin_outlets,
        )

        # Step 3: Compute flow directions
        print("Computing flow directions...")
        flow_direction = compute_flow_direction(conditioned_dem, mask=ocean_mask)
        # Note: No _fix_coastal_flow_directions needed - outlets handle it properly

    elif backend == "pysheds":
        if not PYSHEDS_AVAILABLE:
            raise ImportError("pysheds is not installed. Install with: pip install pysheds")

        # === PYSHEDS BACKEND ===
        # Use pysheds for core hydrology (depression filling, flow direction, accumulation)
        # WARNING: PySheds may produce flow cycles in some cases. Use custom backend for
        # production work.
        import warnings
        warnings.warn(
            "PySheds backend is experimental and may produce flow cycles. "
            "Use backend='custom' for reliable results.",
            UserWarning
        )
        print(f"Using pysheds backend for flow computation...")

        # Create pysheds grid from numpy array
        # PySheds expects nodata value - use a large negative number for ocean/masked areas
        dem_for_pysheds = dem_data.copy()
        nodata_value = -9999.0
        if ocean_mask is not None and np.any(ocean_mask):
            dem_for_pysheds[ocean_mask] = nodata_value

        # Create ViewFinder and Raster objects for pysheds
        from pysheds.sview import Raster, ViewFinder

        # Create ViewFinder with proper metadata
        viewfinder = ViewFinder(
            shape=dem_for_pysheds.shape,
            affine=dem_transform,
            crs=dem_crs,
            nodata=nodata_value,
        )

        # Wrap numpy array as Raster and create grid from viewfinder
        dem_raster = Raster(dem_for_pysheds, viewfinder=viewfinder)
        grid = PyshedsGrid(viewfinder=viewfinder)

        # Step 2: Condition DEM using pysheds
        print("  pysheds: Filling pits...")
        pit_filled = grid.fill_pits(dem_raster)

        print("  pysheds: Filling depressions...")
        flooded = grid.fill_depressions(pit_filled)

        print("  pysheds: Resolving flats...")
        inflated = grid.resolve_flats(flooded)

        conditioned_dem = np.array(inflated).astype(np.float32)

        # Restore original ocean values to conditioned DEM
        if ocean_mask is not None and np.any(ocean_mask):
            conditioned_dem[ocean_mask] = dem_data[ocean_mask]

        # Step 3: Compute flow direction using pysheds
        print("  pysheds: Computing flow direction...")
        fdir = grid.flowdir(inflated)

        # Convert pysheds flow direction to our D8 encoding
        # PySheds uses same D8 encoding by default (1,2,4,8,16,32,64,128)
        # PySheds returns negative values for outlets/boundaries (e.g., -2)
        # We need to convert these to 0 (outlet) before casting to uint8
        fdir_arr = np.array(fdir)
        fdir_arr[fdir_arr < 0] = 0  # Convert negative values to outlet (0)
        flow_direction = fdir_arr.astype(np.uint8)

        # Apply ocean mask to flow direction (ocean cells = outlet)
        if ocean_mask is not None and np.any(ocean_mask):
            flow_direction[ocean_mask] = 0
            # Fix coastal cells to flow toward ocean (pysheds doesn't do this automatically)
            _fix_coastal_flow_directions(flow_direction, ocean_mask)

    else:
        # === LEGACY BACKEND (morphological reconstruction + workarounds) ===
        # Step 2: Condition DEM (fill pits/depressions with masking)
        # Scale min_basin_size if downsampling was applied (area scales with factor²)
        scaled_min_basin_size = min_basin_size
        if min_basin_size is not None and downsample_factor > 1.0:
            scaled_min_basin_size = max(100, int(min_basin_size / (downsample_factor ** 2)))
            print(f"Conditioning DEM (method={fill_method}, min_basin_size={min_basin_size} → {scaled_min_basin_size} scaled)...")
        else:
            print(f"Conditioning DEM (method={fill_method}, min_basin_size={min_basin_size})...")
        conditioned_dem = condition_dem(
            dem_data,
            method=fill_method,
            ocean_mask=ocean_mask,
            min_basin_size=scaled_min_basin_size,
            max_fill_depth=max_fill_depth,
        )

        # Step 3: Compute flow directions (with combined mask)
        print("Computing flow directions...")
        flow_direction = compute_flow_direction(
            conditioned_dem, mask=flow_mask if np.any(flow_mask) else None
        )

    # Step 3.5: Apply lake routing if lake_mask provided
    if lake_mask is not None and lake_outlets is not None:
        from src.terrain.water_bodies import create_lake_flow_routing
        print("Applying lake flow routing...")
        lake_flow = create_lake_flow_routing(lake_mask, lake_outlets, conditioned_dem)
        # Merge: lake cells use lake_flow, others keep terrain_flow
        flow_direction = np.where(lake_mask > 0, lake_flow, flow_direction)
        lake_count = len(np.unique(lake_mask[lake_mask > 0]))
        outlet_count = np.sum(lake_outlets)
        print(f"  Routed {lake_count} lakes with {outlet_count} outlets")

    # Step 4: Compute drainage area and upstream rainfall
    if backend == "pysheds":
        # Use pysheds accumulation
        print("  pysheds: Computing drainage area...")
        acc = grid.accumulation(fdir)
        drainage_area = np.array(acc).astype(np.float32)

        print("  pysheds: Computing upstream rainfall (weighted)...")
        # Wrap precipitation as Raster for weighted accumulation
        precip_raster = Raster(precip_data, viewfinder=viewfinder)
        weighted_acc = grid.accumulation(fdir, weights=precip_raster)
        upstream_rainfall = np.array(weighted_acc).astype(np.float32)

        # Apply ocean mask to accumulation outputs
        if ocean_mask is not None and np.any(ocean_mask):
            drainage_area[ocean_mask] = 0
            upstream_rainfall[ocean_mask] = 0
    else:
        # Use custom backend
        print("Computing drainage area...")
        drainage_area = compute_drainage_area(flow_direction)

        print("Computing upstream rainfall...")
        upstream_rainfall = compute_upstream_rainfall(flow_direction, precip_data)

    # Compute metadata
    total_area_km2 = (dem_shape[0] * dem_shape[1] * cell_size**2) / 1e6
    max_drainage_cells = np.max(drainage_area)
    max_drainage_area_km2 = (max_drainage_cells * cell_size**2) / 1e6
    max_upstream_m3 = np.max(upstream_rainfall) * cell_size**2 / 1000

    metadata = {
        "cell_size_m": cell_size,
        "drainage_area_units": "cells",
        "total_area_km2": total_area_km2,
        "max_drainage_area_km2": max_drainage_area_km2,
        "max_upstream_rainfall_m3": max_upstream_m3,
        "algorithm": flow_algorithm,
        "fill_method": fill_method,
        "backend": backend,
        "downsampling_applied": downsampling_applied,
        "original_shape": original_shape,
        "downsampled_shape": dem_shape if downsampling_applied else original_shape,
        "downsample_factor": downsample_factor,
    }

    # Add target_vertices to metadata if specified
    if target_vertices is not None:
        metadata["target_vertices"] = target_vertices

    # Mark as fresh computation (not from cache)
    if cache:
        metadata["cache_hit"] = False

    # Save outputs
    if output_dir is None:
        output_dir = dem_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    files = {}
    files["flow_direction"] = str(output_dir / "flow_direction.tif")
    files["drainage_area"] = str(output_dir / "flow_accumulation_area.tif")
    files["upstream_rainfall"] = str(output_dir / "flow_accumulation_rainfall.tif")
    files["conditioned_dem"] = str(output_dir / "dem_conditioned.tif")

    # Write output GeoTIFFs
    _write_geotiff(files["flow_direction"], flow_direction.astype(np.uint8), dem_transform, dem_crs)
    _write_geotiff(files["drainage_area"], drainage_area, dem_transform, dem_crs)
    _write_geotiff(files["upstream_rainfall"], upstream_rainfall, dem_transform, dem_crs)
    _write_geotiff(files["conditioned_dem"], conditioned_dem, dem_transform, dem_crs)

    result = {
        "flow_direction": flow_direction,
        "drainage_area": drainage_area,
        "upstream_rainfall": upstream_rainfall,
        "conditioned_dem": conditioned_dem,
        "metadata": metadata,
        "files": files,
    }

    # === CACHE SAVE ===
    if cache:
        # Determine cache directory (use output_dir if cache_dir not specified)
        if cache_dir is None:
            cache_save_path = dem_path.parent
        else:
            cache_save_path = Path(cache_dir)

        # Copy output files to cache location if different from output_dir
        if cache_save_path != output_dir:
            cache_save_path.mkdir(parents=True, exist_ok=True)
            for filename in ["flow_direction.tif", "flow_accumulation_area.tif",
                            "flow_accumulation_rainfall.tif", "dem_conditioned.tif"]:
                src_file = output_dir / filename
                dst_file = cache_save_path / filename
                if src_file.exists():
                    shutil.copy2(src_file, dst_file)
            # Update files dict to point to cache location
            result["files"] = {
                "flow_direction": str(cache_save_path / "flow_direction.tif"),
                "drainage_area": str(cache_save_path / "flow_accumulation_area.tif"),
                "upstream_rainfall": str(cache_save_path / "flow_accumulation_rainfall.tif"),
                "conditioned_dem": str(cache_save_path / "dem_conditioned.tif"),
            }

        # Save cache metadata
        _save_to_cache(cache_save_path, cache_params, dem_mtime, result)
        print(f"  flow_accumulation: cached to {cache_save_path}", flush=True)

    return result


@jit(nopython=True, cache=True)
def _compute_flow_direction_jit(dem: np.ndarray, flow_dir: np.ndarray) -> None:
    """
    JIT-compiled flow direction computation (numba accelerated).

    Modifies flow_dir in-place for maximum performance.

    Parameters
    ----------
    dem : np.ndarray
        Digital elevation model
    flow_dir : np.ndarray
        Output array for flow directions (modified in-place)
    """
    rows, cols = dem.shape

    # D8 direction offsets and codes (explicit for numba)
    offsets = np.array([
        (0, 1),    # East: 1
        (-1, 1),   # Northeast: 2
        (-1, 0),   # North: 4
        (-1, -1),  # Northwest: 8
        (0, -1),   # West: 16
        (1, -1),   # Southwest: 32
        (1, 0),    # South: 64
        (1, 1),    # Southeast: 128
    ], dtype=np.int32)

    codes = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)

    # Pre-compute distance factors for diagonal vs cardinal neighbors
    distances = np.array([1.0, 1.414, 1.0, 1.414, 1.0, 1.414, 1.0, 1.414], dtype=np.float32)

    # For each cell, find steepest downslope neighbor
    for i in range(rows):
        for j in range(cols):
            max_slope = 0.0
            best_dir = 0

            current_elev = dem[i, j]

            # Check all 8 neighbors
            # Priority order for tie-breaking: S, E, W, N, SE, SW, NE, NW
            # This ensures consistent flow direction when slopes are equal
            priority_order = np.array([6, 0, 4, 2, 7, 5, 1, 3], dtype=np.int32)

            for p in range(8):
                k = priority_order[p]
                di, dj = offsets[k]
                ni = i + di
                nj = j + dj

                # Check bounds
                if 0 <= ni < rows and 0 <= nj < cols:
                    neighbor_elev = dem[ni, nj]
                    slope = (current_elev - neighbor_elev) / distances[k]

                    # Use >= for first priority direction, > for others
                    # This ensures consistent tie-breaking
                    if slope > max_slope:
                        max_slope = slope
                        best_dir = codes[k]

            # Handle pits and boundary outlets
            # A pit is an interior cell where all neighbors are higher
            # Pits become outlets (flow_dir = 0) to avoid cycles
            if best_dir == 0:
                # No downslope neighbor found - this is a pit or boundary
                # Mark as outlet (flow_dir = 0)
                best_dir = 0

            flow_dir[i, j] = best_dir


def compute_flow_direction(dem: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """
    Compute D8 flow direction from DEM.

    Parameters
    ----------
    dem : np.ndarray
        Digital elevation model (2D array)
    mask : np.ndarray (bool), optional
        Boolean mask indicating cells to exclude from flow computation.
        Masked cells will have flow_dir = 0 (no flow).

    Returns
    -------
    np.ndarray (uint8)
        Flow direction encoded as D8 (1,2,4,8,16,32,64,128)
        0 indicates outlet/nodata/masked
    """
    rows, cols = dem.shape
    flow_dir = np.zeros((rows, cols), dtype=np.uint8)

    # Use JIT-compiled version if available (10-20x faster)
    if NUMBA_AVAILABLE:
        _compute_flow_direction_jit(dem, flow_dir)
        # Apply mask after computation
        if mask is not None:
            flow_dir[mask] = 0

        # CRITICAL FIX: Force land cells adjacent to masked cells (ocean/sinks)
        # to flow toward the masked cell, even if there's no downslope.
        # This ensures watersheds properly drain to ocean/water bodies.
        if mask is not None:
            _fix_coastal_flow_directions(flow_dir, mask)

        return flow_dir

    # Fallback: Pure Python implementation
    for i in range(rows):
        for j in range(cols):
            max_slope = 0.0
            best_dir = 0

            current_elev = dem[i, j]

            # Check all 8 neighbors
            for (di, dj), direction_code in D8_DIRECTIONS.items():
                ni, nj = i + di, j + dj

                # Check bounds
                if 0 <= ni < rows and 0 <= nj < cols:
                    neighbor_elev = dem[ni, nj]
                    slope = (current_elev - neighbor_elev) / np.sqrt(di**2 + dj**2)

                    if slope > max_slope:
                        max_slope = slope
                        best_dir = direction_code

            # Handle pits and boundary outlets
            # A pit is an interior cell where all neighbors are higher
            # Pits become outlets (flow_dir = 0) to avoid cycles
            if best_dir == 0:
                # No downslope neighbor found - this is a pit or boundary
                # Mark as outlet (flow_dir = 0)
                best_dir = 0

            flow_dir[i, j] = best_dir

    # Apply mask
    if mask is not None:
        flow_dir[mask] = 0

    return flow_dir


@jit(nopython=True, cache=True)
def _fix_coastal_flow_directions_jit(flow_dir: np.ndarray, mask: np.ndarray) -> int:
    """
    JIT-compiled coastal flow direction fix.

    Only fixes pit cells (flow_dir == 0) adjacent to masked cells.
    Does not override valid flow directions.

    Parameters
    ----------
    flow_dir : np.ndarray (uint8)
        Flow direction grid (modified in-place)
    mask : np.ndarray (bool)
        Mask of ocean/sink cells

    Returns
    -------
    int
        Number of cells fixed
    """
    rows, cols = flow_dir.shape

    # D8 offsets and codes (explicit arrays for numba)
    offsets = np.array([
        (0, 1), (-1, 1), (-1, 0), (-1, -1),
        (0, -1), (1, -1), (1, 0), (1, 1)
    ], dtype=np.int32)
    codes = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)

    fixed_count = 0

    for i in range(rows):
        for j in range(cols):
            # Skip if already masked (ocean/sink)
            if mask[i, j]:
                continue

            # Only fix cells that are pits (flow_dir == 0)
            # Don't override valid flow directions
            if flow_dir[i, j] != 0:
                continue

            # Check if any neighbor is masked (ocean/sink)
            for k in range(8):
                di = offsets[k, 0]
                dj = offsets[k, 1]
                ni = i + di
                nj = j + dj

                if 0 <= ni < rows and 0 <= nj < cols and mask[ni, nj]:
                    # Found adjacent ocean/sink - point flow toward it
                    flow_dir[i, j] = codes[k]
                    fixed_count += 1
                    break

    return fixed_count


def _fix_coastal_flow_directions(flow_dir: np.ndarray, mask: np.ndarray) -> None:
    """
    Fix pit cells adjacent to masked cells (ocean/sinks) to flow toward them.

    Only modifies cells with flow_dir == 0 (pits). This ensures coastal pits
    drain to ocean while preserving valid inland flow directions.

    This prevents coastal pits from fragmenting the drainage network.

    Parameters
    ----------
    flow_dir : np.ndarray
        Flow direction grid (modified in-place)
    mask : np.ndarray (bool)
        Mask of ocean/sink cells
    """
    if NUMBA_AVAILABLE:
        fixed_count = _fix_coastal_flow_directions_jit(flow_dir, mask)
    else:
        # Pure Python fallback
        rows, cols = flow_dir.shape
        D8_OFFSETS = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
        D8_CODES = [1, 2, 4, 8, 16, 32, 64, 128]

        fixed_count = 0
        for i in range(rows):
            for j in range(cols):
                if mask[i, j]:
                    continue
                # Only fix cells that are pits (flow_dir == 0)
                if flow_dir[i, j] != 0:
                    continue
                for (di, dj), code in zip(D8_OFFSETS, D8_CODES):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols and mask[ni, nj]:
                        flow_dir[i, j] = code
                        fixed_count += 1
                        break

    print(f"  Fixed {fixed_count:,} coastal cells to flow toward ocean/sinks")


@jit(nopython=True, cache=True)
def _compute_drainage_area_jit(flow_dir: np.ndarray, drainage_area: np.ndarray) -> None:
    """
    JIT-compiled drainage area computation (numba accelerated).

    Uses array-based topological sorting for massive speedup over dict/set approach.

    Parameters
    ----------
    flow_dir : np.ndarray
        D8 flow direction grid
    drainage_area : np.ndarray
        Output array (modified in-place), initialized to 1.0
    """
    rows, cols = flow_dir.shape

    # D8 direction offsets (same as in _compute_flow_direction_jit)
    offsets = np.array([
        (0, 1),    # 1: East
        (-1, 1),   # 2: Northeast
        (-1, 0),   # 4: North
        (-1, -1),  # 8: Northwest
        (0, -1),   # 16: West
        (1, -1),   # 32: Southwest
        (1, 0),    # 64: South
        (1, 1),    # 128: Southeast
    ], dtype=np.int32)

    codes = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)

    # Count how many cells flow INTO each cell
    contributor_count = np.zeros((rows, cols), dtype=np.int32)

    for i in range(rows):
        for j in range(cols):
            direction = flow_dir[i, j]
            if direction > 0:
                # Find which offset corresponds to this direction
                for k in range(8):
                    if codes[k] == direction:
                        di, dj = offsets[k]
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            contributor_count[ni, nj] += 1
                        break

    # Create queue of cells to process (cells with 0 contributors = ridgelines)
    # Use flat indexing for efficiency
    queue_size = 0
    queue = np.zeros(rows * cols, dtype=np.int32)

    for i in range(rows):
        for j in range(cols):
            if contributor_count[i, j] == 0:
                queue[queue_size] = i * cols + j
                queue_size += 1

    # Process cells in topological order
    queue_pos = 0
    while queue_pos < queue_size:
        # Dequeue
        flat_idx = queue[queue_pos]
        queue_pos += 1

        i = flat_idx // cols
        j = flat_idx % cols

        # Get receiver
        direction = flow_dir[i, j]
        if direction > 0:
            # Find offset for this direction
            for k in range(8):
                if codes[k] == direction:
                    di, dj = offsets[k]
                    ni, nj = i + di, j + dj

                    if 0 <= ni < rows and 0 <= nj < cols:
                        # Accumulate drainage area
                        drainage_area[ni, nj] += drainage_area[i, j]

                        # Decrement contributor count
                        contributor_count[ni, nj] -= 1

                        # If all contributors processed, add to queue
                        if contributor_count[ni, nj] == 0:
                            queue[queue_size] = ni * cols + nj
                            queue_size += 1
                    break
        # Note: Cells with direction==0 (outlets/ocean) are processed but don't
        # send flow anywhere. They act as sinks that accumulate drainage from
        # upstream but have no outflow.


def compute_drainage_area(flow_dir: np.ndarray) -> np.ndarray:
    """
    Compute drainage area (unweighted flow accumulation).

    Parameters
    ----------
    flow_dir : np.ndarray
        D8 flow direction grid

    Returns
    -------
    np.ndarray
        Number of cells draining through each pixel (including itself)
    """
    rows, cols = flow_dir.shape
    drainage_area = np.ones((rows, cols), dtype=np.float32)

    # Use JIT-compiled version if available (50-100x faster)
    if NUMBA_AVAILABLE:
        _compute_drainage_area_jit(flow_dir, drainage_area)
        return drainage_area

    # Fallback: Pure Python implementation with dict/set
    # Build flow network: for each cell, track which cells flow INTO it
    contributors = {}  # contributors[cell] = list of cells that flow to it
    receivers = {}  # receivers[cell] = cell it flows to

    for i in range(rows):
        for j in range(cols):
            direction = flow_dir[i, j]
            if direction > 0 and direction in D8_OFFSETS:
                di, dj = D8_OFFSETS[direction]
                receiver = (i + di, j + dj)
                if 0 <= receiver[0] < rows and 0 <= receiver[1] < cols:
                    receivers[(i, j)] = receiver
                    # Track that (i,j) contributes to receiver
                    if receiver not in contributors:
                        contributors[receiver] = []
                    contributors[receiver].append((i, j))

    # Process cells from high to low elevation using topological sort
    # Cells with no contributors are processed first
    processed = set()
    to_process = []

    # Find all cells with no contributors (ridgelines/peaks)
    for i in range(rows):
        for j in range(cols):
            if (i, j) not in contributors:
                to_process.append((i, j))

    # Process in order
    while to_process:
        cell = to_process.pop(0)
        if cell in processed:
            continue

        i, j = cell
        processed.add(cell)

        # Add this cell's drainage area to its receiver
        receiver = receivers.get(cell)
        if receiver is not None:
            ri, rj = receiver
            drainage_area[ri, rj] += drainage_area[i, j]

            # Check if all contributors to receiver are now processed
            receiver_contributors = contributors.get(receiver, [])
            if all(c in processed for c in receiver_contributors):
                to_process.append(receiver)

    return drainage_area


@jit(nopython=True, cache=True)
def _compute_upstream_rainfall_jit(
    flow_dir: np.ndarray, precipitation: np.ndarray, upstream_rainfall: np.ndarray
) -> None:
    """
    JIT-compiled upstream rainfall computation (numba accelerated).

    Uses array-based topological sorting for massive speedup over dict/set approach.

    Parameters
    ----------
    flow_dir : np.ndarray
        D8 flow direction grid
    precipitation : np.ndarray
        Annual precipitation (mm/year)
    upstream_rainfall : np.ndarray
        Output array (modified in-place), initialized to precipitation values
    """
    rows, cols = flow_dir.shape

    # D8 direction offsets
    offsets = np.array([
        (0, 1),    # 1: East
        (-1, 1),   # 2: Northeast
        (-1, 0),   # 4: North
        (-1, -1),  # 8: Northwest
        (0, -1),   # 16: West
        (1, -1),   # 32: Southwest
        (1, 0),    # 64: South
        (1, 1),    # 128: Southeast
    ], dtype=np.int32)

    codes = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)

    # Count contributors
    contributor_count = np.zeros((rows, cols), dtype=np.int32)

    for i in range(rows):
        for j in range(cols):
            direction = flow_dir[i, j]
            if direction > 0:
                for k in range(8):
                    if codes[k] == direction:
                        di, dj = offsets[k]
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            contributor_count[ni, nj] += 1
                        break

    # Initialize queue with ridgelines (cells with 0 contributors)
    queue_size = 0
    queue = np.zeros(rows * cols, dtype=np.int32)

    for i in range(rows):
        for j in range(cols):
            if contributor_count[i, j] == 0:
                queue[queue_size] = i * cols + j
                queue_size += 1

    # Process in topological order
    queue_pos = 0
    while queue_pos < queue_size:
        flat_idx = queue[queue_pos]
        queue_pos += 1

        i = flat_idx // cols
        j = flat_idx % cols

        # Get receiver
        direction = flow_dir[i, j]
        if direction > 0:
            for k in range(8):
                if codes[k] == direction:
                    di, dj = offsets[k]
                    ni, nj = i + di, j + dj

                    if 0 <= ni < rows and 0 <= nj < cols:
                        # Accumulate upstream rainfall
                        upstream_rainfall[ni, nj] += upstream_rainfall[i, j]

                        # Decrement contributor count
                        contributor_count[ni, nj] -= 1

                        # If ready, add to queue
                        if contributor_count[ni, nj] == 0:
                            queue[queue_size] = ni * cols + nj
                            queue_size += 1
                    break


def compute_upstream_rainfall(flow_dir: np.ndarray, precipitation: np.ndarray) -> np.ndarray:
    """
    Compute precipitation-weighted flow accumulation.

    Parameters
    ----------
    flow_dir : np.ndarray
        D8 flow direction grid
    precipitation : np.ndarray
        Annual precipitation (mm/year)

    Returns
    -------
    np.ndarray
        Total upstream precipitation (mm·m²) at each pixel
        Represents cumulative rainfall from entire upstream area
    """
    rows, cols = flow_dir.shape
    upstream_rainfall = precipitation.copy().astype(np.float32)

    # Use JIT-compiled version if available (50-100x faster)
    if NUMBA_AVAILABLE:
        _compute_upstream_rainfall_jit(flow_dir, precipitation, upstream_rainfall)
        return upstream_rainfall

    # Fallback: Pure Python implementation with dict/set
    # Build flow network
    contributors = {}
    receivers = {}

    for i in range(rows):
        for j in range(cols):
            direction = flow_dir[i, j]
            if direction > 0 and direction in D8_OFFSETS:
                di, dj = D8_OFFSETS[direction]
                receiver = (i + di, j + dj)
                if 0 <= receiver[0] < rows and 0 <= receiver[1] < cols:
                    receivers[(i, j)] = receiver
                    if receiver not in contributors:
                        contributors[receiver] = []
                    contributors[receiver].append((i, j))

    # Topological sort: process from ridgelines to outlets
    processed = set()
    to_process = []

    # Find cells with no contributors (ridgelines)
    for i in range(rows):
        for j in range(cols):
            if (i, j) not in contributors:
                to_process.append((i, j))

    # Process in topological order
    while to_process:
        cell = to_process.pop(0)
        if cell in processed:
            continue

        i, j = cell
        processed.add(cell)

        # Add this cell's upstream rainfall to its receiver
        receiver = receivers.get(cell)
        if receiver is not None:
            ri, rj = receiver
            upstream_rainfall[ri, rj] += upstream_rainfall[i, j]

            # Check if receiver is ready to process
            receiver_contributors = contributors.get(receiver, [])
            if all(c in processed for c in receiver_contributors):
                to_process.append(receiver)

    return upstream_rainfall


def identify_outlets(
    dem: np.ndarray,
    nodata_mask: np.ndarray,
    coastal_elev_threshold: float = 10.0,
    edge_mode: Literal["all", "local_minima", "outward_slope", "none"] = "all",
    masked_basin_outlets: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Identify drainage outlet cells (Stage 1 of flow-spec.md).

    Classifies cells that act as drainage termini where water leaves the system.
    Implements three outlet types: coastal, edge, and masked basin outlets.

    Parameters
    ----------
    dem : np.ndarray
        Digital elevation model
    nodata_mask : np.ndarray (bool)
        Mask of ocean/off-grid cells (True = NoData)
    coastal_elev_threshold : float, default 10.0
        Maximum elevation for coastal outlets (meters above sea level).
        Prevents high cliffs adjacent to ocean from being spurious outlets.
    edge_mode : {"all", "local_minima", "outward_slope", "none"}, default "all"
        Boundary outlet strategy:
        - "all": All boundary cells are outlets (safest, prevents artificial basins)
        - "local_minima": Only boundary cells that are local minima
        - "outward_slope": Boundary cells with interior neighbors sloping toward them
        - "none": No edge outlets (for islands fully surrounded by coastline)
    masked_basin_outlets : np.ndarray (bool), optional
        User-supplied outlet locations for known lakes/basins

    Returns
    -------
    np.ndarray (bool)
        Boolean mask where True = outlet cell

    Notes
    -----
    This implements Stage 1 of flow-spec.md (lines 42-108).

    Edge mode "all" is the safest default - it ensures no artificial endorheic
    basins form at boundaries. The cost is some fragmentation of edge drainage
    networks, but this is usually preferable to missed outlets.

    Examples
    --------
    >>> dem = np.array([[5, 5, 5],
    ...                 [5, 3, 5],
    ...                 [5, 5, 5]], dtype=np.float32)
    >>> nodata = np.array([[False, False, False],
    ...                    [True,  False, False],
    ...                    [False, False, False]])
    >>> outlets = identify_outlets(dem, nodata, coastal_elev_threshold=10.0)
    >>> outlets[1, 1]  # Low coastal cell adjacent to ocean
    True
    """
    rows, cols = dem.shape
    outlets = np.zeros((rows, cols), dtype=bool)

    # --- Coastal outlets ---
    # Land cells adjacent to NoData AND elevation <= threshold
    for i in range(rows):
        for j in range(cols):
            # Skip if already NoData
            if nodata_mask[i, j]:
                continue

            # Check if adjacent to NoData (8-connected)
            adjacent_to_nodata = False
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1),
                          (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    if nodata_mask[ni, nj]:
                        adjacent_to_nodata = True
                        break

            # Mark as coastal outlet if low enough
            if adjacent_to_nodata and dem[i, j] <= coastal_elev_threshold:
                outlets[i, j] = True

    # --- Edge outlets ---
    if edge_mode == "all":
        # All boundary cells are outlets
        outlets[0, :] = True  # Top edge
        outlets[-1, :] = True  # Bottom edge
        outlets[:, 0] = True  # Left edge
        outlets[:, -1] = True  # Right edge
        # Don't override NoData cells
        outlets[nodata_mask] = False

    elif edge_mode == "local_minima":
        # Only boundary cells that are local minima among edge neighbors
        # Top edge
        for j in range(cols):
            if nodata_mask[0, j]:
                continue
            is_min = True
            # Check edge neighbors (left, right, below)
            for dj in [-1, 0, 1]:
                nj = j + dj
                if 0 <= nj < cols and nj != j:
                    if not nodata_mask[0, nj] and dem[0, nj] < dem[0, j]:
                        is_min = False
                        break
            # Check interior neighbor (below)
            if rows > 1 and not nodata_mask[1, j]:
                if dem[1, j] < dem[0, j]:
                    is_min = False
            if is_min:
                outlets[0, j] = True

        # Bottom edge
        for j in range(cols):
            if nodata_mask[-1, j]:
                continue
            is_min = True
            for dj in [-1, 0, 1]:
                nj = j + dj
                if 0 <= nj < cols and nj != j:
                    if not nodata_mask[-1, nj] and dem[-1, nj] < dem[-1, j]:
                        is_min = False
                        break
            # Check interior neighbor (above)
            if rows > 1 and not nodata_mask[-2, j]:
                if dem[-2, j] < dem[-1, j]:
                    is_min = False
            if is_min:
                outlets[-1, j] = True

        # Left edge
        for i in range(rows):
            if nodata_mask[i, 0]:
                continue
            is_min = True
            for di in [-1, 0, 1]:
                ni = i + di
                if 0 <= ni < rows and ni != i:
                    if not nodata_mask[ni, 0] and dem[ni, 0] < dem[i, 0]:
                        is_min = False
                        break
            # Check interior neighbor (right)
            if cols > 1 and not nodata_mask[i, 1]:
                if dem[i, 1] < dem[i, 0]:
                    is_min = False
            if is_min:
                outlets[i, 0] = True

        # Right edge
        for i in range(rows):
            if nodata_mask[i, -1]:
                continue
            is_min = True
            for di in [-1, 0, 1]:
                ni = i + di
                if 0 <= ni < rows and ni != i:
                    if not nodata_mask[ni, -1] and dem[ni, -1] < dem[i, -1]:
                        is_min = False
                        break
            # Check interior neighbor (left)
            if cols > 1 and not nodata_mask[i, -2]:
                if dem[i, -2] < dem[i, -1]:
                    is_min = False
            if is_min:
                outlets[i, -1] = True

    elif edge_mode == "outward_slope":
        # Boundary cells with interior neighbors sloping more steeply toward edge
        # than toward any other neighbor
        # Top edge
        for j in range(cols):
            if nodata_mask[0, j]:
                continue
            if rows > 1 and not nodata_mask[1, j]:
                # Interior neighbor below
                slope_to_edge = (dem[1, j] - dem[0, j]) / 1.0
                # Check if this is steepest slope from interior cell
                max_slope_elsewhere = -np.inf
                for di, dj in [(-1, 0), (0, -1), (0, 1),
                              (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    ni, nj = 1 + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        if not nodata_mask[ni, nj]:
                            dist = np.sqrt(di**2 + dj**2)
                            slope = (dem[1, j] - dem[ni, nj]) / dist
                            max_slope_elsewhere = max(max_slope_elsewhere, slope)
                if slope_to_edge > max_slope_elsewhere:
                    outlets[0, j] = True

        # Similar logic for other edges (bottom, left, right)
        # Bottom edge
        for j in range(cols):
            if nodata_mask[-1, j]:
                continue
            if rows > 1 and not nodata_mask[-2, j]:
                slope_to_edge = (dem[-2, j] - dem[-1, j]) / 1.0
                max_slope_elsewhere = -np.inf
                for di, dj in [(1, 0), (0, -1), (0, 1),
                              (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    ni, nj = -2 + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        if not nodata_mask[ni, nj]:
                            dist = np.sqrt(di**2 + dj**2)
                            slope = (dem[-2, j] - dem[ni, nj]) / dist
                            max_slope_elsewhere = max(max_slope_elsewhere, slope)
                if slope_to_edge > max_slope_elsewhere:
                    outlets[-1, j] = True

        # Left edge
        for i in range(rows):
            if nodata_mask[i, 0]:
                continue
            if cols > 1 and not nodata_mask[i, 1]:
                slope_to_edge = (dem[i, 1] - dem[i, 0]) / 1.0
                max_slope_elsewhere = -np.inf
                for di, dj in [(0, -1), (-1, 0), (1, 0),
                              (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    ni, nj = i + di, 1 + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        if not nodata_mask[ni, nj]:
                            dist = np.sqrt(di**2 + dj**2)
                            slope = (dem[i, 1] - dem[ni, nj]) / dist
                            max_slope_elsewhere = max(max_slope_elsewhere, slope)
                if slope_to_edge > max_slope_elsewhere:
                    outlets[i, 0] = True

        # Right edge
        for i in range(rows):
            if nodata_mask[i, -1]:
                continue
            if cols > 1 and not nodata_mask[i, -2]:
                slope_to_edge = (dem[i, -2] - dem[i, -1]) / 1.0
                max_slope_elsewhere = -np.inf
                for di, dj in [(0, 1), (-1, 0), (1, 0),
                              (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    ni, nj = i + di, -2 + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        if not nodata_mask[ni, nj]:
                            dist = np.sqrt(di**2 + dj**2)
                            slope = (dem[i, -2] - dem[ni, nj]) / dist
                            max_slope_elsewhere = max(max_slope_elsewhere, slope)
                if slope_to_edge > max_slope_elsewhere:
                    outlets[i, -1] = True

    elif edge_mode == "none":
        # No edge outlets (for islands)
        pass

    # --- Masked basin outlets ---
    if masked_basin_outlets is not None:
        outlets |= masked_basin_outlets

    return outlets


@jit(nopython=True, parallel=True, cache=True)
def _identify_sinks_jit(
    dem: np.ndarray,
    outlets: np.ndarray,
    nodata_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    JIT-compiled parallel sink identification (10-20x faster than pure Python).

    Uses numba parallel execution to check cells concurrently across multiple CPU cores.

    Returns
    -------
    sink_rows, sink_cols, sink_elevs : np.ndarray
        Arrays of sink coordinates and elevations, sorted by elevation
    """
    rows, cols = dem.shape

    # First pass: count sinks per row (parallel)
    row_sink_counts = np.zeros(rows, dtype=np.int32)

    for i in prange(rows):
        count = 0
        for j in range(cols):
            # Skip outlets and nodata
            if outlets[i, j] or nodata_mask[i, j]:
                continue

            # Check if any neighbor is lower
            has_downslope = False
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        if not nodata_mask[ni, nj] and dem[ni, nj] < dem[i, j]:
                            has_downslope = True
                            break
                if has_downslope:
                    break

            if not has_downslope:
                count += 1

        row_sink_counts[i] = count

    # Calculate total sinks and row offsets
    total_sinks = np.sum(row_sink_counts)
    row_offsets = np.zeros(rows + 1, dtype=np.int32)
    for i in range(rows):
        row_offsets[i + 1] = row_offsets[i] + row_sink_counts[i]

    # Allocate output arrays
    sink_rows = np.empty(total_sinks, dtype=np.int32)
    sink_cols = np.empty(total_sinks, dtype=np.int32)
    sink_elevs = np.empty(total_sinks, dtype=np.float64)

    # Second pass: fill sink arrays (parallel)
    for i in prange(rows):
        offset = row_offsets[i]
        local_count = 0

        for j in range(cols):
            # Skip outlets and nodata
            if outlets[i, j] or nodata_mask[i, j]:
                continue

            # Check if any neighbor is lower
            has_downslope = False
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        if not nodata_mask[ni, nj] and dem[ni, nj] < dem[i, j]:
                            has_downslope = True
                            break
                if has_downslope:
                    break

            if not has_downslope:
                sink_rows[offset + local_count] = i
                sink_cols[offset + local_count] = j
                sink_elevs[offset + local_count] = dem[i, j]
                local_count += 1

    # Sort by elevation (argsort)
    sort_indices = np.argsort(sink_elevs)
    return sink_rows[sort_indices], sink_cols[sort_indices], sink_elevs[sort_indices]


def _identify_sinks(
    dem: np.ndarray,
    outlets: np.ndarray,
    nodata_mask: Optional[np.ndarray] = None,
) -> list:
    """
    Identify all sink cells (cells with no downslope neighbor).

    Returns sinks sorted by elevation (lowest first) to reduce cascading breaches.

    Parameters
    ----------
    dem : np.ndarray
        Digital elevation model
    outlets : np.ndarray (bool)
        Outlet mask from identify_outlets()
    nodata_mask : np.ndarray (bool), optional
        Cells to exclude

    Returns
    -------
    list of tuples
        List of (row, col, elevation) for each sink, sorted by elevation ascending
    """
    if nodata_mask is None:
        nodata_mask = np.zeros_like(dem, dtype=bool)

    # Use JIT-compiled version if available
    if NUMBA_AVAILABLE:
        sink_rows, sink_cols, sink_elevs = _identify_sinks_jit(dem, outlets, nodata_mask)
        return [(int(r), int(c), float(e)) for r, c, e in zip(sink_rows, sink_cols, sink_elevs)]

    # Fallback to pure Python
    rows, cols = dem.shape
    sinks = []

    for i in range(rows):
        for j in range(cols):
            # Skip outlets and nodata
            if outlets[i, j] or nodata_mask[i, j]:
                continue

            # Check if any neighbor is lower
            has_downslope = False
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1),
                          (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    if not nodata_mask[ni, nj] and dem[ni, nj] < dem[i, j]:
                        has_downslope = True
                        break

            if not has_downslope:
                sinks.append((i, j, dem[i, j]))

    # Sort by elevation (process lowest first to avoid cascading)
    sinks.sort(key=lambda x: x[2])
    return sinks


def _reconstruct_path(parent_map: dict, start_r: int, start_c: int,
                     end_r: int, end_c: int) -> list:
    """
    Reconstruct breach path from Dijkstra parent map.

    Parameters
    ----------
    parent_map : dict
        Mapping from (row, col) to (parent_row, parent_col)
    start_r, start_c : int
        Starting cell (sink)
    end_r, end_c : int
        Ending cell (drain point)

    Returns
    -------
    list of tuples
        Path from sink to drain as [(row, col), ...] in order
    """
    path = []
    r, c = end_r, end_c

    while (r, c) != (None, None):
        path.append((r, c))
        if r == start_r and c == start_c:
            break
        r, c = parent_map.get((r, c), (None, None))

    path.reverse()
    return path


@jit(nopython=True, cache=True)
def _find_breach_path_dijkstra_jit(
    dem: np.ndarray,
    start_row: int,
    start_col: int,
    outlets: np.ndarray,
    resolved: np.ndarray,
    max_depth: float,
    max_length: int,
) -> Tuple[bool, np.ndarray, np.ndarray]:
    """
    JIT-compiled Dijkstra breach path finder (5-10x faster).

    Returns
    -------
    found : bool
        Whether a valid path was found
    path_rows, path_cols : np.ndarray
        Path coordinates (empty if not found)
    """
    start_elev = dem[start_row, start_col]
    rows, cols = dem.shape

    # Use arrays instead of dicts for JIT compatibility
    visited = np.zeros((rows, cols), dtype=np.bool_)
    parent_r = np.full((rows, cols), -1, dtype=np.int32)
    parent_c = np.full((rows, cols), -1, dtype=np.int32)

    # Simple priority queue (using lists, heapq not fully supported in nopython)
    # Store: (cost, length, row, col, parent_row, parent_col)
    # We'll use a simplified approach: expand lowest cost first
    max_queue_size = min(10000, rows * cols)  # Reasonable limit
    queue_costs = np.full(max_queue_size, np.inf, dtype=np.float64)
    queue_lengths = np.zeros(max_queue_size, dtype=np.int32)
    queue_rows = np.zeros(max_queue_size, dtype=np.int32)
    queue_cols = np.zeros(max_queue_size, dtype=np.int32)
    queue_parent_r = np.zeros(max_queue_size, dtype=np.int32)
    queue_parent_c = np.zeros(max_queue_size, dtype=np.int32)
    queue_size = 1
    queue_costs[0] = 0.0
    queue_lengths[0] = 0
    queue_rows[0] = start_row
    queue_cols[0] = start_col
    queue_parent_r[0] = -1
    queue_parent_c[0] = -1

    end_r, end_c = -1, -1
    found = False

    while queue_size > 0:
        # Find minimum cost item (simple linear search for now)
        min_idx = 0
        min_cost = queue_costs[0]
        for i in range(1, queue_size):
            if queue_costs[i] < min_cost:
                min_cost = queue_costs[i]
                min_idx = i

        # Pop item
        cost = queue_costs[min_idx]
        length = queue_lengths[min_idx]
        r = queue_rows[min_idx]
        c = queue_cols[min_idx]
        pr = queue_parent_r[min_idx]
        pc = queue_parent_c[min_idx]

        # Remove from queue (swap with last)
        queue_size -= 1
        if min_idx < queue_size:
            queue_costs[min_idx] = queue_costs[queue_size]
            queue_lengths[min_idx] = queue_lengths[queue_size]
            queue_rows[min_idx] = queue_rows[queue_size]
            queue_cols[min_idx] = queue_cols[queue_size]
            queue_parent_r[min_idx] = queue_parent_r[queue_size]
            queue_parent_c[min_idx] = queue_parent_c[queue_size]

        # Skip if already visited
        if visited[r, c]:
            continue

        visited[r, c] = True
        parent_r[r, c] = pr
        parent_c[r, c] = pc

        # Check termination: reached outlet
        if outlets[r, c]:
            end_r, end_c = r, c
            found = True
            break

        # Check termination: reached resolved cell at or below start elevation
        if resolved[r, c] and dem[r, c] <= start_elev:
            end_r, end_c = r, c
            found = True
            break

        # Check length constraint
        if length >= max_length:
            continue

        # Explore neighbors
        for di in range(-1, 2):
            for dj in range(-1, 2):
                if di == 0 and dj == 0:
                    continue

                ni, nj = r + di, c + dj

                # Bounds check
                if not (0 <= ni < rows and 0 <= nj < cols):
                    continue
                if visited[ni, nj]:
                    continue

                # Cost to breach through this neighbor
                breach_depth_here = max(0.0, dem[ni, nj] - start_elev)

                # Check depth constraint
                if breach_depth_here > max_depth:
                    continue

                new_cost = cost + breach_depth_here
                new_length = length + 1

                # Add to queue if space available
                if queue_size < max_queue_size:
                    queue_costs[queue_size] = new_cost
                    queue_lengths[queue_size] = new_length
                    queue_rows[queue_size] = ni
                    queue_cols[queue_size] = nj
                    queue_parent_r[queue_size] = r
                    queue_parent_c[queue_size] = c
                    queue_size += 1

    if not found:
        return False, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    # Reconstruct path
    path_list_r = []
    path_list_c = []
    curr_r, curr_c = end_r, end_c

    while curr_r != -1:
        path_list_r.append(curr_r)
        path_list_c.append(curr_c)
        if curr_r == start_row and curr_c == start_col:
            break
        next_r = parent_r[curr_r, curr_c]
        next_c = parent_c[curr_r, curr_c]
        curr_r, curr_c = next_r, next_c

    # Convert to arrays and reverse
    path_r = np.array(path_list_r[::-1], dtype=np.int32)
    path_c = np.array(path_list_c[::-1], dtype=np.int32)

    return True, path_r, path_c


@jit(nopython=True, cache=True)
def _dijkstra_single_sink(
    dem: np.ndarray,
    start_row: int,
    start_col: int,
    outlets: np.ndarray,
    max_depth: float,
    max_length: int,
    path_r_out: np.ndarray,
    path_c_out: np.ndarray,
) -> int:
    """
    Single sink Dijkstra worker for parallel processing.

    Similar to _find_breach_path_dijkstra_jit but:
    1. Ignores `resolved` array (for Phase 1 parallelism)
    2. Writes path to pre-allocated output arrays
    3. Returns path length (0 if not found)

    This allows calling from within prange loops.
    """
    start_elev = dem[start_row, start_col]
    rows, cols = dem.shape

    # Use arrays for JIT compatibility
    visited = np.zeros((rows, cols), dtype=np.bool_)
    parent_r = np.full((rows, cols), -1, dtype=np.int32)
    parent_c = np.full((rows, cols), -1, dtype=np.int32)

    # Simple priority queue
    max_queue_size = min(10000, rows * cols)
    queue_costs = np.full(max_queue_size, np.inf, dtype=np.float64)
    queue_lengths = np.zeros(max_queue_size, dtype=np.int32)
    queue_rows = np.zeros(max_queue_size, dtype=np.int32)
    queue_cols = np.zeros(max_queue_size, dtype=np.int32)
    queue_parent_r = np.zeros(max_queue_size, dtype=np.int32)
    queue_parent_c = np.zeros(max_queue_size, dtype=np.int32)
    queue_size = 1
    queue_costs[0] = 0.0
    queue_lengths[0] = 0
    queue_rows[0] = start_row
    queue_cols[0] = start_col
    queue_parent_r[0] = -1
    queue_parent_c[0] = -1

    end_r, end_c = -1, -1
    found = False

    while queue_size > 0:
        # Find minimum cost item
        min_idx = 0
        min_cost = queue_costs[0]
        for i in range(1, queue_size):
            if queue_costs[i] < min_cost:
                min_cost = queue_costs[i]
                min_idx = i

        # Pop item
        cost = queue_costs[min_idx]
        length = queue_lengths[min_idx]
        r = queue_rows[min_idx]
        c = queue_cols[min_idx]
        pr = queue_parent_r[min_idx]
        pc = queue_parent_c[min_idx]

        # Remove from queue (swap with last)
        queue_size -= 1
        if min_idx < queue_size:
            queue_costs[min_idx] = queue_costs[queue_size]
            queue_lengths[min_idx] = queue_lengths[queue_size]
            queue_rows[min_idx] = queue_rows[queue_size]
            queue_cols[min_idx] = queue_cols[queue_size]
            queue_parent_r[min_idx] = queue_parent_r[queue_size]
            queue_parent_c[min_idx] = queue_parent_c[queue_size]

        if visited[r, c]:
            continue

        visited[r, c] = True
        parent_r[r, c] = pr
        parent_c[r, c] = pc

        # Termination: reached outlet
        if outlets[r, c]:
            end_r, end_c = r, c
            found = True
            break

        # Length constraint
        if length >= max_length:
            continue

        # Explore neighbors
        for di in range(-1, 2):
            for dj in range(-1, 2):
                if di == 0 and dj == 0:
                    continue

                ni, nj = r + di, c + dj

                if not (0 <= ni < rows and 0 <= nj < cols):
                    continue
                if visited[ni, nj]:
                    continue

                breach_depth_here = max(0.0, dem[ni, nj] - start_elev)

                if breach_depth_here > max_depth:
                    continue

                new_cost = cost + breach_depth_here
                new_length = length + 1

                if queue_size < max_queue_size:
                    queue_costs[queue_size] = new_cost
                    queue_lengths[queue_size] = new_length
                    queue_rows[queue_size] = ni
                    queue_cols[queue_size] = nj
                    queue_parent_r[queue_size] = r
                    queue_parent_c[queue_size] = c
                    queue_size += 1

    if not found:
        return 0  # No path found

    # Reconstruct path into output arrays
    path_idx = 0
    curr_r, curr_c = end_r, end_c
    max_path_len = len(path_r_out)

    while curr_r != -1 and path_idx < max_path_len:
        path_r_out[path_idx] = curr_r
        path_c_out[path_idx] = curr_c
        path_idx += 1
        if curr_r == start_row and curr_c == start_col:
            break
        next_r = parent_r[curr_r, curr_c]
        next_c = parent_c[curr_r, curr_c]
        curr_r, curr_c = next_r, next_c

    # Reverse path in-place (sink -> outlet becomes sink -> outlet order)
    for i in range(path_idx // 2):
        j = path_idx - 1 - i
        path_r_out[i], path_r_out[j] = path_r_out[j], path_r_out[i]
        path_c_out[i], path_c_out[j] = path_c_out[j], path_c_out[i]

    return path_idx


@jit(nopython=True, parallel=True, cache=True)
def _breach_sinks_parallel_batch(
    dem: np.ndarray,
    sinks_r: np.ndarray,
    sinks_c: np.ndarray,
    outlets: np.ndarray,
    max_depth: float,
    max_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find breach paths for a batch of sinks in parallel using prange.

    This is Phase 1 of two-phase parallel processing:
    - Uses outlets only (no resolved array) for termination
    - Each sink's Dijkstra is independent
    - Sinks should be spatially distant (checkerboard partitioning)

    Parameters
    ----------
    dem : np.ndarray
        Digital elevation model
    sinks_r, sinks_c : np.ndarray
        Row and column coordinates of sinks to process
    outlets : np.ndarray (bool)
        Outlet mask
    max_depth : float
        Maximum breach depth per cell
    max_length : int
        Maximum path length

    Returns
    -------
    found : np.ndarray (bool)
        Whether path was found for each sink
    paths_r, paths_c : np.ndarray (n_sinks, max_length+1)
        Path coordinates for each sink (-1 for unused slots)
    path_lengths : np.ndarray (int)
        Actual path length for each sink
    """
    n_sinks = len(sinks_r)
    max_path_len = max_length + 1

    # Pre-allocate outputs
    found = np.zeros(n_sinks, dtype=np.bool_)
    paths_r = np.full((n_sinks, max_path_len), -1, dtype=np.int32)
    paths_c = np.full((n_sinks, max_path_len), -1, dtype=np.int32)
    path_lengths = np.zeros(n_sinks, dtype=np.int32)

    # Process each sink in parallel
    for i in prange(n_sinks):
        sink_r = sinks_r[i]
        sink_c = sinks_c[i]

        # Run Dijkstra for this sink
        path_len = _dijkstra_single_sink(
            dem, sink_r, sink_c, outlets,
            max_depth, max_length,
            paths_r[i], paths_c[i]
        )

        found[i] = path_len > 0
        path_lengths[i] = path_len

    return found, paths_r, paths_c, path_lengths


def _cluster_sinks_checkerboard(
    sinks: list,
    grid_size: int,
    dem_shape: Tuple[int, int],
) -> Tuple[list, list]:
    """
    Cluster sinks into two batches using checkerboard pattern.

    Divides the DEM into grid cells of size `grid_size`. Sinks in
    "black" cells (even row+col) go in batch 1, "white" cells (odd)
    in batch 2. This ensures sinks in the same batch are at least
    `grid_size` cells apart diagonally.

    Parameters
    ----------
    sinks : list
        List of (row, col, elev, depth) tuples
    grid_size : int
        Size of grid cells (should be >= max_breach_length)
    dem_shape : tuple
        Shape of DEM (rows, cols)

    Returns
    -------
    batch_black, batch_white : list
        Two lists of sinks for parallel processing
    """
    batch_black = []  # Even grid cells
    batch_white = []  # Odd grid cells

    for sink in sinks:
        r, c, elev, depth = sink
        grid_r = r // grid_size
        grid_c = c // grid_size

        if (grid_r + grid_c) % 2 == 0:
            batch_black.append(sink)
        else:
            batch_white.append(sink)

    return batch_black, batch_white


def _find_breach_path_dijkstra(
    dem: np.ndarray,
    start_row: int,
    start_col: int,
    outlets: np.ndarray,
    resolved: np.ndarray,
    max_depth: float,
    max_length: int,
) -> Optional[list]:
    """
    Find least-cost breach path from sink to draining cell using Dijkstra.

    Cost metric: Total elevation that must be removed along the path.

    Termination conditions:
    1. Reached an outlet cell
    2. Reached a resolved cell with elevation <= start_elev
    3. No more cells to explore within constraints (breach failed)

    Parameters
    ----------
    dem : np.ndarray
        Digital elevation model
    start_row, start_col : int
        Sink cell coordinates
    outlets : np.ndarray (bool)
        Outlet mask
    resolved : np.ndarray (bool)
        Tracks which cells already have drainage paths
    max_depth : float
        Maximum elevation drop allowed at any single cell (meters)
    max_length : int
        Maximum path length (cells)

    Returns
    -------
    list of tuples or None
        Breach path [(row, col), ...] from sink to drain, or None if failed
    """
    # Use JIT-compiled version if available
    if NUMBA_AVAILABLE:
        found, path_r, path_c = _find_breach_path_dijkstra_jit(
            dem, start_row, start_col, outlets, resolved, max_depth, max_length
        )
        if found:
            return [(int(r), int(c)) for r, c in zip(path_r, path_c)]
        else:
            return None

    # Fallback to pure Python with heapq
    import heapq

    start_elev = dem[start_row, start_col]
    rows, cols = dem.shape

    # Priority queue: (cost, length, row, col, parent_row, parent_col)
    pq = [(0, 0, start_row, start_col, None, None)]
    visited = {}  # {(row, col): cost}
    parent_map = {}  # {(row, col): (parent_row, parent_col)}

    while pq:
        cost, length, r, c, pr, pc = heapq.heappop(pq)

        # Skip if already visited with lower cost
        if (r, c) in visited:
            continue
        visited[(r, c)] = cost
        parent_map[(r, c)] = (pr, pc)

        # Check termination: reached outlet
        if outlets[r, c]:
            return _reconstruct_path(parent_map, start_row, start_col, r, c)

        # Check termination: reached resolved cell at or below start elevation
        if resolved[r, c] and dem[r, c] <= start_elev:
            return _reconstruct_path(parent_map, start_row, start_col, r, c)

        # Check length constraint
        if length >= max_length:
            continue  # Don't expand further from this cell

        # Explore neighbors
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ni, nj = r + di, c + dj

            # Bounds check
            if not (0 <= ni < rows and 0 <= nj < cols):
                continue
            if (ni, nj) in visited:
                continue

            # Cost to breach through this neighbor
            # Must carve down to start_elev (or below)
            breach_depth_here = max(0, dem[ni, nj] - start_elev)

            # Check depth constraint
            if breach_depth_here > max_depth:
                continue  # Would exceed max breach depth at this cell

            new_cost = cost + breach_depth_here
            new_length = length + 1

            heapq.heappush(pq, (new_cost, new_length, ni, nj, r, c))

    # No viable path found within constraints
    return None


def _apply_breach(dem: np.ndarray, path: list, epsilon: float) -> None:
    """
    Carve breach path into DEM with monotonic gradient.

    Works backward from drain point to sink, ensuring each cell is epsilon
    lower than the previous cell in the path.

    Only lowers cells, never raises them.

    Parameters
    ----------
    dem : np.ndarray
        Digital elevation model (modified in-place)
    path : list of tuples
        Breach path [(row, col), ...] from sink to drain
    epsilon : float
        Small gradient for breach paths (meters per cell)
    """
    n = len(path)
    if n < 2:
        return

    # Start from drain point (end of path)
    drain_r, drain_c = path[-1]
    base_elev = dem[drain_r, drain_c]

    # Work backward toward sink
    for i in range(n - 2, -1, -1):
        r, c = path[i]
        # Each cell should be epsilon higher than the next cell in path
        required_elev = base_elev + epsilon * (n - 1 - i)
        # Only lower cells, never raise them
        if dem[r, c] > required_elev:
            dem[r, c] = required_elev


def breach_depressions_constrained(
    dem: np.ndarray,
    outlets: np.ndarray,
    max_breach_depth: float = 50.0,
    max_breach_length: int = 100,
    epsilon: float = 1e-4,
    nodata_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Remove depressions via constrained breaching (Stage 2a of flow-spec.md).

    Uses Lindsay (2016) constrained least-cost breaching algorithm.
    Implements two-pass approach:
    1. Identify all sinks (cells with no downslope neighbor)
    2. For each sink, attempt Dijkstra breach within constraints

    Parameters
    ----------
    dem : np.ndarray
        Input digital elevation model
    outlets : np.ndarray (bool)
        Outlet mask from identify_outlets()
    max_breach_depth : float, default 50.0
        Maximum elevation drop allowed at any single cell (meters)
    max_breach_length : int, default 100
        Maximum breach path length (cells)
    epsilon : float, default 1e-4
        Small gradient for breach paths (meters per cell)
    nodata_mask : np.ndarray (bool), optional
        Cells to exclude from breaching

    Returns
    -------
    np.ndarray
        DEM with depressions breached where possible

    Notes
    -----
    This implements Stage 2a of flow-spec.md (lines 115-289).

    Sinks that cannot be breached within constraints are left for
    Stage 2b (priority-flood fill) to handle.

    References
    ----------
    Lindsay, J.B. (2016). Efficient hybrid breaching-filling sink removal
    methods for flow path enforcement in digital elevation models.
    Hydrological Processes, 30, 846–857.
    """
    breached = dem.copy().astype(np.float64)
    rows, cols = dem.shape

    if nodata_mask is None:
        nodata_mask = np.zeros_like(dem, dtype=bool)

    # Track which cells have drainage paths
    resolved = outlets.copy()

    print("  Stage 2a: Identifying sinks...")

    # Show threading info
    if NUMBA_AVAILABLE:
        try:
            from numba import get_num_threads
            num_threads = get_num_threads()
            print(f"    Numba: {num_threads} threads available for parallel operations")
        except:
            pass

    sinks = _identify_sinks(breached, outlets, nodata_mask)
    print(f"    Found {len(sinks):,} sink cells")

    if len(sinks) == 0:
        print("    No sinks to breach")
        return breached.astype(np.float32)

    # Precompute fill depths to identify shallow sinks (optimization)
    print("    Computing sink depths...")
    from skimage.morphology import reconstruction
    filled_preview = breached.copy().astype(np.float64)
    seed = filled_preview.copy()
    seed[1:-1, 1:-1] = filled_preview.max() + 1000
    if nodata_mask is not None:
        seed[nodata_mask] = filled_preview[nodata_mask]
    filled_preview = reconstruction(seed, filled_preview, method='erosion')

    # Calculate depth for each sink
    sink_depths = []
    for sink_r, sink_c, sink_elev in sinks:
        depth = filled_preview[sink_r, sink_c] - sink_elev
        sink_depths.append(depth)

    # Filter to significant sinks (> 1.0m depth)
    # Shallow sinks will be handled by Stage 2b (priority-flood fill)
    depth_threshold = 1.0  # meters
    significant_sinks = [
        (sink_r, sink_c, sink_elev, depth)
        for (sink_r, sink_c, sink_elev), depth in zip(sinks, sink_depths)
        if depth > depth_threshold
    ]

    shallow_skipped = len(sinks) - len(significant_sinks)
    print(f"    Skipping {shallow_skipped:,} shallow sinks (<{depth_threshold}m deep)")
    print(f"    Processing {len(significant_sinks):,} significant sinks (>{depth_threshold}m deep)")

    print(f"  Stage 2a: Attempting constrained breaching (max_depth={max_breach_depth}m, max_length={max_breach_length} cells)...")

    total_sinks = len(significant_sinks)
    breached_count = 0
    failed_count = 0
    already_resolved_count = 0

    # Estimate if parallel is useful: only helps when sinks are near outlets
    # With large max_breach_length and large DEM, most sinks are interior
    # and parallel (outlets-only) will find ~0 breaches - wasted effort
    avg_dim = (rows + cols) / 2
    parallel_useful = max_breach_length < avg_dim / 4  # Sinks likely near outlets

    if not parallel_useful and NUMBA_AVAILABLE:
        print(f"    Note: max_breach_length ({max_breach_length}) is large relative to DEM ({rows}x{cols})")
        print(f"    Using serial JIT (better for interior sinks that chain together)")

    # Use two-phase parallel processing with numba prange
    if NUMBA_AVAILABLE and total_sinks > 100 and parallel_useful:
        # Two-phase parallel processing: uses numba prange for true multi-core parallelism
        try:
            from numba import get_num_threads
            num_threads = get_num_threads()
            print(f"    Using two-phase parallel breaching ({num_threads} CPU cores)")
        except:
            print(f"    Using two-phase parallel breaching")

        # Cluster sinks using checkerboard pattern (grid_size = 2 * max_breach_length)
        # Sinks in same batch are guaranteed to be far enough apart that paths won't overlap
        grid_size = 2 * max_breach_length
        batch_black, batch_white = _cluster_sinks_checkerboard(
            significant_sinks, grid_size, (rows, cols)
        )
        print(f"    Checkerboard clustering: {len(batch_black):,} black cells, {len(batch_white):,} white cells")

        # Sub-batch size for progress reporting (process in chunks)
        # Use 1% intervals like serial version (~100 updates per phase)
        sub_batch_size = max(10, total_sinks // 100)

        def process_batch_with_progress(batch, phase_name, start_breached, start_failed, start_resolved):
            """Process a batch in sub-batches with progress reporting."""
            nonlocal breached, resolved
            batch_breached = start_breached
            batch_failed = start_failed
            batch_resolved = start_resolved
            n = len(batch)

            for chunk_start in range(0, n, sub_batch_size):
                chunk_end = min(chunk_start + sub_batch_size, n)
                chunk = batch[chunk_start:chunk_end]

                # Prepare arrays for this chunk
                chunk_r = np.array([s[0] for s in chunk], dtype=np.int32)
                chunk_c = np.array([s[1] for s in chunk], dtype=np.int32)

                # Run parallel Dijkstra on chunk
                found, paths_r, paths_c, path_lengths = _breach_sinks_parallel_batch(
                    breached, chunk_r, chunk_c, outlets,
                    max_breach_depth, max_breach_length
                )

                # Apply successful breaches
                for i in range(len(chunk)):
                    sink_r, sink_c = chunk_r[i], chunk_c[i]
                    if resolved[sink_r, sink_c]:
                        batch_resolved += 1
                        continue

                    if found[i]:
                        path_len = path_lengths[i]
                        path = [(int(paths_r[i, j]), int(paths_c[i, j])) for j in range(path_len)]
                        _apply_breach(breached, path, epsilon)
                        for r, c in path:
                            resolved[r, c] = True
                        batch_breached += 1
                    else:
                        batch_failed += 1

                # Progress report
                processed = chunk_end
                percent = 100.0 * processed / n
                print(f"      {phase_name}: {percent:5.1f}% ({processed:,}/{n:,}) - {batch_breached:,} breached, {batch_failed:,} failed")

            return batch_breached, batch_failed, batch_resolved

        # Phase 1: Process "black" cells in parallel sub-batches
        if len(batch_black) > 0:
            print(f"    Phase 1: Processing {len(batch_black):,} sinks...")
            breached_count, failed_count, already_resolved_count = process_batch_with_progress(
                batch_black, "Phase 1", breached_count, failed_count, already_resolved_count
            )

        # Phase 2: Process "white" cells in parallel sub-batches
        if len(batch_white) > 0:
            print(f"    Phase 2: Processing {len(batch_white):,} sinks...")
            breached_count, failed_count, already_resolved_count = process_batch_with_progress(
                batch_white, "Phase 2", breached_count, failed_count, already_resolved_count
            )

        # Note: Failed sinks are left for Stage 2b (priority-flood) to handle
        # Priority-flood is O(n log n) and much faster than re-running Dijkstra

    else:
        # Serial processing for small sink counts or when numba unavailable
        if NUMBA_AVAILABLE:
            print(f"    Using serial JIT-compiled Dijkstra (small sink count)")
        else:
            print(f"    Using serial pure-Python Dijkstra (numba unavailable)")

        # Progress reporting
        if total_sinks > 0:
            progress_interval = max(1, total_sinks // 100)  # Report every 1%
            print(f"    Progress (showing every 1%):")

        for idx, (sink_r, sink_c, sink_elev, depth) in enumerate(significant_sinks):
            # Progress reporting
            if total_sinks > 0 and idx % progress_interval == 0:
                percent = 100.0 * idx / total_sinks
                print(f"      {percent:5.1f}% ({idx:,} / {total_sinks:,} sinks, {breached_count:,} breached, {failed_count:,} failed)")

            # Skip if already resolved by previous breach
            if resolved[sink_r, sink_c]:
                already_resolved_count += 1
                continue

            # Attempt to find breach path
            path = _find_breach_path_dijkstra(
                breached, sink_r, sink_c, outlets, resolved,
                max_breach_depth, max_breach_length
            )

            if path is not None:
                # Apply breach
                _apply_breach(breached, path, epsilon)

                # Mark all cells in path as resolved
                for r, c in path:
                    resolved[r, c] = True

                breached_count += 1
            else:
                # Could not breach within constraints
                failed_count += 1

        if total_sinks > 0:
            print(f"      100.0% ({total_sinks:,} / {total_sinks:,} sinks, {breached_count:,} breached, {failed_count:,} failed)")

    print(f"    Results: {breached_count:,} breached, {already_resolved_count:,} already resolved, {failed_count:,} failed")
    print(f"    Total sinks handled: {shallow_skipped:,} shallow (skipped) + {breached_count:,} breached + {already_resolved_count:,} resolved = {shallow_skipped + breached_count + already_resolved_count:,}")
    print(f"    Remaining for Stage 2b: {failed_count + shallow_skipped:,} sinks")

    return breached.astype(np.float32)


def priority_flood_fill_epsilon(
    dem: np.ndarray,
    outlets: np.ndarray,
    epsilon: float = 1e-4,
    nodata_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Fill residual depressions with epsilon gradient (Stage 2b of flow-spec.md).

    Uses Barnes et al. (2014) priority-flood algorithm with epsilon gradient
    applied DURING fill (not after). This creates flow-directing gradients
    as flats form, ensuring proper drainage without post-processing.

    Parameters
    ----------
    dem : np.ndarray
        Input DEM (already breached by Stage 2a)
    outlets : np.ndarray (bool)
        Outlet mask from identify_outlets()
    epsilon : float, default 1e-4
        Minimum elevation increment per cell (meters per cell).
        Creates gradients in filled areas to ensure drainage.
    nodata_mask : np.ndarray (bool), optional
        Cells to exclude from filling

    Returns
    -------
    np.ndarray
        DEM with residual depressions filled

    Notes
    -----
    This implements Stage 2b of flow-spec.md (lines 291-328).

    Key difference from morphological reconstruction:
    - Epsilon is applied DURING fill as cells are processed
    - Creates natural gradients toward outlets as fill progresses
    - Eliminates need for separate flat resolution step

    References
    ----------
    Barnes, R., Lehman, C., & Mulla, D. (2014). Priority-flood: An optimal
    depression-filling and watershed-labeling algorithm for digital elevation
    models. Computers & Geosciences, 62, 117–127.
    """
    import heapq

    rows, cols = dem.shape
    filled = dem.copy().astype(np.float64)

    if nodata_mask is None:
        nodata_mask = np.zeros_like(dem, dtype=bool)

    # Priority queue: (elevation, row, col)
    pq = []
    in_queue = np.zeros((rows, cols), dtype=bool)

    # Seed priority queue with outlets
    for i in range(rows):
        for j in range(cols):
            if outlets[i, j] and not nodata_mask[i, j]:
                heapq.heappush(pq, (filled[i, j], i, j))
                in_queue[i, j] = True

    # Also seed with cells adjacent to NoData (border of domain)
    for i in range(rows):
        for j in range(cols):
            if nodata_mask[i, j] or in_queue[i, j]:
                continue

            # Check if adjacent to NoData
            adjacent_to_nodata = False
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1),
                          (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    if nodata_mask[ni, nj]:
                        adjacent_to_nodata = True
                        break

            if adjacent_to_nodata:
                heapq.heappush(pq, (filled[i, j], i, j))
                in_queue[i, j] = True

    # Process cells in elevation order (priority-flood)
    filled_count = 0
    while pq:
        elev, r, c = heapq.heappop(pq)

        # Check all 8 neighbors
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ni, nj = r + di, c + dj

            # Bounds and status checks
            if not (0 <= ni < rows and 0 <= nj < cols):
                continue
            if in_queue[ni, nj]:
                continue
            if nodata_mask[ni, nj]:
                continue

            # If neighbor is in depression (below current + epsilon), raise it
            # This is the KEY DIFFERENCE: epsilon applied DURING fill
            if filled[ni, nj] < elev + epsilon:
                filled[ni, nj] = elev + epsilon
                filled_count += 1

            # Add neighbor to queue
            heapq.heappush(pq, (filled[ni, nj], ni, nj))
            in_queue[ni, nj] = True

    if filled_count > 0:
        print(f"    Priority-flood raised {filled_count:,} cells to resolve depressions")

    return filled.astype(np.float32)


def condition_dem_spec(
    dem: np.ndarray,
    nodata_mask: Optional[np.ndarray] = None,
    coastal_elev_threshold: float = 10.0,
    edge_mode: Literal["all", "local_minima", "outward_slope", "none"] = "all",
    max_breach_depth: float = 50.0,
    max_breach_length: int = 100,
    epsilon: float = 1e-4,
    masked_basin_outlets: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Condition DEM using spec-compliant 4-stage pipeline.

    Orchestrates the complete flow-spec.md pipeline:
    1. Stage 1: Identify outlets (coastal, edge, masked basins)
    2. Stage 2a: Constrained breaching (Dijkstra least-cost)
    3. Stage 2b: Priority-flood fill residuals (with epsilon)
    4. (External) D8 flow direction → see compute_flow_direction()
    5. (External) Flow accumulation → see compute_drainage_area()

    Parameters
    ----------
    dem : np.ndarray
        Input digital elevation model
    nodata_mask : np.ndarray (bool), optional
        Mask of ocean/off-grid cells (True = NoData)
    coastal_elev_threshold : float, default 10.0
        Maximum elevation for coastal outlets (meters above sea level)
    edge_mode : {"all", "local_minima", "outward_slope", "none"}, default "all"
        Boundary outlet strategy
    max_breach_depth : float, default 50.0
        Maximum elevation drop at any single cell during breaching (meters)
    max_breach_length : int, default 100
        Maximum breach path length (cells)
    epsilon : float, default 1e-4
        Minimum elevation increment per cell in filled areas (meters)
    masked_basin_outlets : np.ndarray (bool), optional
        User-supplied outlet locations for known lakes/basins

    Returns
    -------
    conditioned_dem : np.ndarray
        Breached and filled DEM
    outlets : np.ndarray (bool)
        Outlet mask (useful for diagnostics and visualization)

    Notes
    -----
    This implements the complete spec-compliant pipeline from flow-spec.md.

    Key advantages over legacy approach:
    - Explicit outlet classification prevents boundary artifacts
    - Selective breaching preserves legitimate basins
    - Epsilon gradient applied during fill (not after)
    - No workarounds needed (_fix_coastal_flow, fill_small_sinks)

    Examples
    --------
    >>> dem = np.array([[5, 5, 5],
    ...                 [5, 3, 5],
    ...                 [5, 5, 5]], dtype=np.float32)
    >>> nodata = np.zeros((3, 3), dtype=bool)
    >>> conditioned, outlets = condition_dem_spec(dem, nodata)
    >>> # Now use with compute_flow_direction() and compute_drainage_area()
    """
    if nodata_mask is None:
        nodata_mask = np.zeros_like(dem, dtype=bool)

    print("  Stage 1: Identifying outlets...")
    outlets = identify_outlets(
        dem,
        nodata_mask,
        coastal_elev_threshold,
        edge_mode,
        masked_basin_outlets
    )
    num_outlets = np.sum(outlets)
    print(f"    Found {num_outlets:,} outlet cells")

    print("  Stage 2a: Constrained breaching...")
    breached = breach_depressions_constrained(
        dem, outlets, max_breach_depth, max_breach_length, epsilon, nodata_mask
    )

    print("  Stage 2b: Priority-flood fill residuals...")
    filled = priority_flood_fill_epsilon(
        breached, outlets, epsilon, nodata_mask
    )

    # Ensure masked cells maintain original elevation
    filled[nodata_mask] = dem[nodata_mask]

    # Ensure we never lowered elevations (only raised for filling)
    filled = np.maximum(filled, dem)

    print("  DEM conditioning complete (spec-compliant pipeline)")
    return filled, outlets


def detect_ocean_mask(
    dem: np.ndarray, threshold: float = 0.0, border_only: bool = True
) -> np.ndarray:
    """
    Detect ocean or water bodies in DEM.

    Identifies cells at or below elevation threshold that are connected to
    the border (assumed to be ocean/large water bodies).

    Uses connected component labeling for efficient O(n) detection.

    Parameters
    ----------
    dem : np.ndarray
        Digital elevation model
    threshold : float, default 0.0
        Elevation threshold (meters). Cells <= threshold are candidates.
    border_only : bool, default True
        If True, only return border-connected low-elevation areas (ocean).
        If False, return all areas below threshold (includes inland lakes).

    Returns
    -------
    np.ndarray (bool)
        Boolean mask where True = ocean/water

    Examples
    --------
    >>> dem = np.array([[0, 0, 5], [0, 1, 6], [5, 6, 7]])
    >>> ocean = detect_ocean_mask(dem, threshold=0.0, border_only=True)
    >>> ocean
    array([[ True,  True, False],
           [ True, False, False],
           [False, False, False]])
    """
    from scipy.ndimage import label

    # Find all cells at or below threshold
    low_elevation = dem <= threshold

    if not border_only:
        return low_elevation

    # No low-elevation cells? No ocean.
    if not np.any(low_elevation):
        return np.zeros_like(dem, dtype=bool)

    # Label connected regions (O(n) operation)
    structure = np.ones((3, 3), dtype=bool)  # 8-connectivity
    labeled, num_features = label(low_elevation, structure=structure)

    # Find labels that touch any border
    border_labels = set()
    border_labels.update(labeled[0, :])   # Top border
    border_labels.update(labeled[-1, :])  # Bottom border
    border_labels.update(labeled[:, 0])   # Left border
    border_labels.update(labeled[:, -1])  # Right border
    border_labels.discard(0)  # Remove background label

    # Create mask for all border-connected regions
    ocean_mask = np.isin(labeled, list(border_labels))

    return ocean_mask


def detect_endorheic_basins(
    dem: np.ndarray,
    min_size: int = 10,
    exclude_mask: np.ndarray | None = None,
    min_depth: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """
    Detect endorheic (closed) basins in DEM.

    Identifies closed depressions (basins with no outlet) that exceed
    a minimum size threshold. Used to preserve large natural basins
    like the Salton Sea, Death Valley, etc.

    Parameters
    ----------
    dem : np.ndarray
        Digital elevation model
    min_size : int, default 10
        Minimum basin size in cells to be considered significant
    exclude_mask : np.ndarray (bool), optional
        Mask of areas to exclude from basin detection (e.g., ocean).
        This improves performance by only filling land areas.
    min_depth : float, default 0.5
        Minimum depression depth in meters to be considered a basin.
        Higher values = only preserve truly deep basins, fill shallower ones.

    Returns
    -------
    basin_mask : np.ndarray (bool)
        Boolean mask where True = part of endorheic basin
    basin_sizes : dict
        Dictionary mapping basin_id to size in cells

    Examples
    --------
    >>> # Create closed basin surrounded by mountains
    >>> dem = np.array([[50, 50, 50],
    ...                 [50, 10, 50],
    ...                 [50, 50, 50]])
    >>> mask, sizes = detect_endorheic_basins(dem, min_size=1)
    >>> mask[1, 1]  # Center is basin
    True
    """
    from scipy.ndimage import label

    # Fill depressions to find what WOULD be filled
    # Pass exclude_mask to avoid filling ocean (performance optimization)
    filled = _fill_depressions(dem, epsilon=0.0, mask=exclude_mask)
    fill_depth = filled - dem

    # Cells that would be filled are part of depressions
    # Higher min_depth = only preserve truly deep basins, fill shallower ones
    depressions = fill_depth > min_depth

    if not np.any(depressions):
        # No depressions found
        return np.zeros_like(dem, dtype=bool), {}

    # Label connected depression regions (O(n) operation)
    structure = np.ones((3, 3), dtype=bool)  # 8-connectivity
    labeled, num_features = label(depressions, structure=structure)

    # Calculate size of each basin (vectorized for performance)
    unique_labels, label_counts = np.unique(labeled[labeled > 0], return_counts=True)
    basin_sizes = dict(zip(unique_labels.tolist(), label_counts.tolist()))

    # Create mask for large basins (vectorized operation)
    basin_mask = np.zeros_like(dem, dtype=bool)
    large_basin_ids = [bid for bid, size in basin_sizes.items() if size >= min_size]
    if large_basin_ids:
        basin_mask = np.isin(labeled, large_basin_ids)

    return basin_mask, basin_sizes


def condition_dem(
    dem: np.ndarray,
    method: str = "fill",
    ocean_mask: np.ndarray | None = None,
    min_basin_size: int | None = None,
    max_fill_depth: float | None = None,
    min_basin_depth: float = 0.5,
    fill_small_sinks: int | None = None,
) -> np.ndarray:
    """
    Condition DEM by filling pits and resolving depressions.

    Uses morphological reconstruction (priority flood algorithm) to properly
    fill depressions and pits. This is the standard algorithm used by most
    hydrological analysis tools (e.g., GRASS, WhiteboxTools, ArcGIS).

    Supports masking ocean areas and preserving large endorheic basins.

    Parameters
    ----------
    dem : np.ndarray
        Input DEM
    method : str, default 'fill'
        Depression handling method ('fill' or 'breach')
        - 'fill': Complete depression filling (raises elevations)
        - 'breach': Minimal filling to preserve terrain (uses epsilon)
    ocean_mask : np.ndarray (bool), optional
        Boolean mask indicating ocean/water cells to exclude from conditioning.
        Masked cells maintain original elevation.
    min_basin_size : int, optional
        Minimum basin size (cells) to preserve. Basins >= this size are
        not filled (preserves large endorheic basins like Salton Sea).
    max_fill_depth : float, optional
        Maximum fill depth (meters). Depressions requiring fill > this
        depth are preserved (protects deep natural basins).
    min_basin_depth : float, default 0.5
        Minimum depression depth (meters) to be considered a preservable basin.
        Higher values = only preserve truly deep basins, fill shallower ones.
        Increase this for noisy high-resolution DEMs.
    fill_small_sinks : int, optional
        Maximum sink size (cells) to fill. After main filling, any remaining
        local minima (sinks) with contributing area < this size are filled.
        This removes small artifacts that create fragmented drainage.
        Typical values: 10-100 cells.

    Returns
    -------
    np.ndarray
        Conditioned DEM with depressions resolved

    Examples
    --------
    >>> # Basic usage
    >>> conditioned = condition_dem(dem)

    >>> # Mask ocean
    >>> ocean = detect_ocean_mask(dem, threshold=0.0)
    >>> conditioned = condition_dem(dem, ocean_mask=ocean)

    >>> # Preserve large basins
    >>> conditioned = condition_dem(dem, min_basin_size=10000)

    >>> # Fill small sinks (< 50 cells) to reduce fragmentation
    >>> conditioned = condition_dem(dem, fill_small_sinks=50)
    """
    # ========== BASIN PRESERVATION LOGIC ==========
    # Large endorheic basins (e.g., Salton Sea, Death Valley) are preserved
    # using TWO independent mechanisms that can work together:
    #
    # 1. Size-based preservation (min_basin_size):
    #    - Detects closed depressions larger than min_basin_size cells
    #    - Preserves these naturally-occurring basins by excluding from filling
    #    - Example: min_basin_size=10000 preserves basins >= 10,000 cells
    #
    # 2. Depth-based preservation (max_fill_depth):
    #    - After filling, restores any cells that required > max_fill_depth meters
    #    - Example: max_fill_depth=50 preserves basins requiring >50m fill
    #
    # These mechanisms preserve real geographic features while still filling
    # noise and local pits.

    # Create combined exclusion mask
    exclude_mask = np.zeros_like(dem, dtype=bool)

    if ocean_mask is not None:
        exclude_mask |= ocean_mask

    # === Size-based preservation: Preserve large closed basins ===
    if min_basin_size is not None:
        basin_mask, basin_sizes = detect_endorheic_basins(
            dem, min_size=min_basin_size, exclude_mask=ocean_mask, min_depth=min_basin_depth
        )
        total_depressions = len(basin_sizes)
        large_basins = sum(1 for size in basin_sizes.values() if size >= min_basin_size)
        num_cells_masked = np.sum(basin_mask)
        pct_masked = 100 * num_cells_masked / basin_mask.size
        print(f"  Basin preservation: {total_depressions} depressions >{min_basin_depth}m deep, {large_basins} >= {min_basin_size} cells")
        print(f"  Masked {num_cells_masked:,} cells ({pct_masked:.1f}% of DEM)")
        exclude_mask |= basin_mask

    # === Main depression filling ===
    # Priority-flood algorithm properly fills depressions while respecting
    # exclusion masks (ocean, large basins, etc.)
    if method == "fill":
        filled = _fill_depressions(dem, epsilon=0.0, mask=exclude_mask)
    elif method == "breach":
        # Use 1e-4 (0.1mm) to avoid fragmentation from rounding (10mm precision)
        filled = _fill_depressions(dem, epsilon=1e-4, mask=exclude_mask)
    else:
        raise ValueError(f"Unknown fill method: {method}")

    # === Depth-based preservation: Preserve deep basins ===
    # If a depression would require > max_fill_depth meters of fill,
    # preserve it at original elevation (it's likely a real natural feature)
    if max_fill_depth is not None:
        fill_depth = filled - dem
        deep_basins = fill_depth > max_fill_depth
        filled[deep_basins] = dem[deep_basins]

    # Ensure masked cells maintain original elevation
    filled[exclude_mask] = dem[exclude_mask]

    # Ensure we never lower original elevations
    filled = np.maximum(filled, dem)

    # Fill small sinks if requested
    if fill_small_sinks is not None and fill_small_sinks > 0:
        filled = _fill_small_sinks(filled, max_sink_size=fill_small_sinks, mask=exclude_mask)

    return filled


def _fill_small_sinks(
    dem: np.ndarray,
    max_sink_size: int = 50,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Fill small sinks (local minima) that would create fragmented drainage.

    Iteratively finds local minima and fills small ones to their spill point.
    This removes DEM noise artifacts that create many tiny outlets.

    Optimized implementation: O(m + n*k) where m=grid size, n=num_sinks, k=avg_sink_size.
    Uses single-pass index collection instead of per-sink grid scans.

    Parameters
    ----------
    dem : np.ndarray
        Input DEM (already filled/breached)
    max_sink_size : int, default 50
        Maximum size (cells) of sinks to fill. Sinks larger than this
        are preserved (they may be real features).
    mask : np.ndarray (bool), optional
        Cells to exclude from filling (e.g., ocean, large basins).

    Returns
    -------
    np.ndarray
        DEM with small sinks filled
    """
    from scipy.ndimage import label, minimum_filter, maximum_filter, find_objects

    filled = dem.copy().astype(np.float64)
    rows, cols = dem.shape

    # Find local minima: cells that are strictly lower than all 8 neighbors
    footprint = np.ones((3, 3), dtype=bool)
    local_min_filter = minimum_filter(filled, footprint=footprint, mode='constant', cval=np.inf)
    local_max_filter = maximum_filter(filled, footprint=footprint, mode='constant', cval=-np.inf)
    local_minima = (filled == local_min_filter) & (filled < local_max_filter)

    # Exclude masked cells and boundary
    if mask is not None:
        local_minima &= ~mask
    local_minima[0, :] = False
    local_minima[-1, :] = False
    local_minima[:, 0] = False
    local_minima[:, -1] = False

    num_minima = np.sum(local_minima)
    print(f"  Small sink detection: found {num_minima:,} local minima cells")
    if num_minima == 0:
        return filled.astype(np.float32)

    # Label connected sink regions
    structure = np.ones((3, 3), dtype=bool)  # 8-connectivity
    labeled, num_features = label(local_minima, structure=structure)
    print(f"  Small sink detection: {num_features:,} connected sink regions")

    # Get bounding boxes for all regions in one pass
    slices = find_objects(labeled)

    # Precompute sink indices using vectorized numpy operations
    # Get all labeled cell coordinates at once
    labeled_rows, labeled_cols = np.where(labeled > 0)
    labeled_ids = labeled[labeled_rows, labeled_cols]

    # Sort by label ID to group cells belonging to same sink
    sort_order = np.argsort(labeled_ids)
    sorted_rows = labeled_rows[sort_order]
    sorted_cols = labeled_cols[sort_order]
    sorted_ids = labeled_ids[sort_order]

    # Find split points between different labels
    # np.diff finds where labels change, np.where finds those positions
    split_points = np.where(np.diff(sorted_ids) > 0)[0] + 1
    split_points = np.concatenate([[0], split_points, [len(sorted_ids)]])

    # Build dict mapping sink_id -> (row_indices, col_indices)
    sink_indices = {}
    for i in range(len(split_points) - 1):
        start, end = split_points[i], split_points[i + 1]
        if start < end:
            sink_id = sorted_ids[start]
            sink_indices[sink_id] = (sorted_rows[start:end], sorted_cols[start:end])

    # Process each sink using precomputed indices
    sinks_filled = 0
    cells_raised = 0

    for sink_id in range(1, num_features + 1):
        if sink_id not in sink_indices:
            continue

        sink_rows, sink_cols = sink_indices[sink_id]
        sink_size = len(sink_rows)

        if sink_size > max_sink_size:
            continue  # Skip large sinks

        obj_slice = slices[sink_id - 1]
        if obj_slice is None:
            continue

        # Expand slice by 1 for boundary detection
        r_start = max(0, obj_slice[0].start - 1)
        r_stop = min(rows, obj_slice[0].stop + 1)
        c_start = max(0, obj_slice[1].start - 1)
        c_stop = min(cols, obj_slice[1].stop + 1)

        # Work on cropped region
        local_labeled = labeled[r_start:r_stop, c_start:c_stop]
        local_filled = filled[r_start:r_stop, c_start:c_stop]
        local_mask = mask[r_start:r_stop, c_start:c_stop] if mask is not None else None

        # Create sink mask for this region only (small array)
        sink_mask_local = local_labeled == sink_id

        # Find boundary using dilation on small region
        from scipy.ndimage import binary_dilation
        expanded = binary_dilation(sink_mask_local, structure=structure)
        boundary = expanded & ~sink_mask_local

        if local_mask is not None:
            boundary &= ~local_mask

        if not np.any(boundary):
            continue

        sink_elevation = np.min(local_filled[sink_mask_local])
        boundary_elevs = local_filled[boundary]
        higher_neighbors = boundary_elevs > sink_elevation

        # Determine new elevation
        if np.any(higher_neighbors):
            new_elev = np.min(boundary_elevs[higher_neighbors]) + 1e-6
        else:
            lowest_boundary_elev = np.min(boundary_elevs)
            if lowest_boundary_elev <= sink_elevation:
                new_elev = lowest_boundary_elev + 1e-6
            else:
                continue

        # Update using precomputed indices (vectorized, avoids full-grid scan)
        filled[sink_rows, sink_cols] = new_elev
        sinks_filled += 1
        cells_raised += sink_size

    if sinks_filled > 0:
        print(f"  Filled {sinks_filled} small sinks ({cells_raised:,} cells) with max_size={max_sink_size}")

    return filled.astype(np.float32)


def _fill_depressions(
    dem: np.ndarray, epsilon: float = 0.0, mask: np.ndarray | None = None
) -> np.ndarray:
    """
    Fill depressions in DEM using morphological reconstruction.

    This implements depression filling via morphological reconstruction.
    Algorithm: Reconstruct from seed (borders at DEM elevation, interior at +inf)
    downward, constrained by original DEM.

    Parameters
    ----------
    dem : np.ndarray
        Input digital elevation model
    epsilon : float, default 0.0
        Small gradient to add in flat areas (for 'breach' method)
        If > 0, creates minimal gradients instead of true flats
    mask : np.ndarray (bool), optional
        Boolean mask indicating cells to exclude from filling.
        Masked cells maintain original elevation.

    Returns
    -------
    np.ndarray
        Depression-filled DEM
    """
    from skimage.morphology import reconstruction

    dem = dem.astype(np.float64)  # Use float64 for precision

    # Create seed: borders at DEM elevation, interior slightly higher
    # This allows reconstruction to fill depressions
    seed = dem.copy()
    seed[1:-1, 1:-1] = dem.max() + 1000  # Interior much higher

    # If mask provided, set masked cells as borders (won't be filled)
    if mask is not None:
        seed[mask] = dem[mask]

    # Morphological reconstruction by erosion
    # Erode seed downward, constrained by mask (original DEM)
    # This fills depressions to their spill point elevation
    filled = reconstruction(seed, dem, method='erosion')

    # Restore masked cells to original elevation
    if mask is not None:
        filled[mask] = dem[mask]

    # Add epsilon gradients if requested (for breach method)
    if epsilon > 0:
        filled = _resolve_flats(filled, epsilon)

    return filled.astype(np.float32)


def _resolve_flats(dem: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Resolve flat areas using Garbrecht-Martz (1997) dual-gradient algorithm.

    Uses TWO gradients combined:
    1. Gradient TOWARD lower terrain (pour points) - water flows to outlets
    2. Gradient AWAY from higher terrain (high points) - water flows from ridges

    The combined gradient ensures natural flow convergence in flat regions.

    References:
    - Garbrecht & Martz (1997): "The assignment of drainage direction over
      flat surfaces in raster digital elevation models" J. Hydrol. 193: 204-213
    - Barnes et al. (2014): "An Efficient Assignment of Drainage Direction
      Over Flat Surfaces" arXiv:1511.04433

    Parameters
    ----------
    dem : np.ndarray
        Input DEM (potentially with flat areas)
    epsilon : float
        Small value for gradient increment (default: 1e-5 m)

    Returns
    -------
    np.ndarray
        DEM with flats resolved using dual gradients
    """
    resolved = dem.copy().astype(np.float64)
    rows, cols = dem.shape

    # Round DEM to eliminate floating-point precision issues
    # Use 10mm precision (0.01m) to avoid fragmentation from tiny breach gradients
    dem_rounded = np.round(dem, 2)

    # Find flat cells: cells with at least one neighbor at the SAME elevation
    flat_cells = np.zeros((rows, cols), dtype=bool)

    # Create padded DEM for safe neighbor access
    padded = np.full((rows + 2, cols + 2), np.nan, dtype=dem_rounded.dtype)
    padded[1:-1, 1:-1] = dem_rounded

    # Check all 8 neighbors for equal elevation (vectorized)
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        neighbor = padded[1 + di:rows + 1 + di, 1 + dj:cols + 1 + dj]
        flat_cells |= (dem_rounded == neighbor)

    flat_count = np.sum(flat_cells)
    if flat_count == 0:
        return resolved.astype(np.float32)

    print(f"  Flat resolution: {flat_count:,} flat cells found")

    # Find pour points: flat cells adjacent to STRICTLY LOWER terrain
    pour_points = np.zeros((rows, cols), dtype=bool)

    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        padded_inf = np.full((rows + 2, cols + 2), np.inf, dtype=dem_rounded.dtype)
        padded_inf[1:-1, 1:-1] = dem_rounded
        neighbor = padded_inf[1 + di:rows + 1 + di, 1 + dj:cols + 1 + dj]
        pour_points |= flat_cells & (neighbor < dem_rounded)

    # Boundary flat cells are also pour points
    boundary_mask = np.zeros((rows, cols), dtype=bool)
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True
    pour_points |= flat_cells & boundary_mask

    # NEW: Find high points: flat cells adjacent to STRICTLY HIGHER terrain
    high_points = np.zeros((rows, cols), dtype=bool)

    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        padded_neg = np.full((rows + 2, cols + 2), -np.inf, dtype=dem_rounded.dtype)
        padded_neg[1:-1, 1:-1] = dem_rounded
        neighbor = padded_neg[1 + di:rows + 1 + di, 1 + dj:cols + 1 + dj]
        high_points |= flat_cells & (neighbor > dem_rounded)

    pour_count = np.sum(pour_points)
    high_count = np.sum(high_points)

    if pour_count == 0 and high_count == 0:
        return resolved.astype(np.float32)

    print(f"  Flat resolution: {pour_count:,} pour points, {high_count:,} high points")

    # Compute DUAL gradients (Garbrecht-Martz algorithm)
    # Gradient 1: Distance from pour points (toward lower terrain)
    dist_to_low = _compute_flat_gradient_bfs(flat_cells, pour_points, dem_rounded)

    # Gradient 2: Distance from high points (away from higher terrain)
    dist_from_high = _compute_flat_gradient_bfs(flat_cells, high_points, dem_rounded)

    # Combine gradients: cells should be HIGHER if:
    # - farther from pour points (dist_to_low is larger)
    # - closer to high points (dist_from_high is smaller)
    # Formula: combined = dist_to_low + (max_dist - dist_from_high)
    # Simplified: combined = dist_to_low - dist_from_high + const
    # The constant doesn't matter since we're adding relative gradients

    # Find max distances for normalization
    max_dist_low = np.max(dist_to_low[flat_cells]) if pour_count > 0 else 0
    max_dist_high = np.max(dist_from_high[flat_cells]) if high_count > 0 else 0

    # Combine: cells farther from outlets AND closer to ridges get higher elevation
    # This creates natural flow convergence toward outlets and away from ridges
    combined_gradient = np.zeros_like(dist_to_low)

    if pour_count > 0 and high_count > 0:
        # Full dual-gradient: both components
        # dist_to_low: higher = farther from outlet = should be higher
        # dist_from_high: higher = farther from ridge = should be lower
        # Combined: dist_to_low adds elevation, dist_from_high subtracts
        combined_gradient[flat_cells] = (
            dist_to_low[flat_cells] +
            (max_dist_high - dist_from_high[flat_cells])
        )
    elif pour_count > 0:
        # Only pour points: just use distance toward outlets
        combined_gradient[flat_cells] = dist_to_low[flat_cells]
    elif high_count > 0:
        # Only high points: just use distance from ridges (inverted)
        combined_gradient[flat_cells] = max_dist_high - dist_from_high[flat_cells]

    # Apply combined gradient
    resolved[flat_cells] += combined_gradient[flat_cells] * epsilon

    return resolved.astype(np.float32)


@jit(nopython=True, cache=True)
def _compute_flat_gradient_bfs_jit(
    flat_cells: np.ndarray,
    pour_points: np.ndarray,
    dem_rounded: np.ndarray,
    gradient: np.ndarray
) -> None:
    """
    JIT-compiled multi-source BFS to compute gradient from pour points.

    Computes geodesic distance from pour points within each flat region,
    respecting elevation boundaries (cells at different elevations are barriers).

    Parameters
    ----------
    flat_cells : np.ndarray (bool)
        Mask of flat cells
    pour_points : np.ndarray (bool)
        Mask of pour points (sources for BFS)
    dem_rounded : np.ndarray
        Rounded DEM for elevation comparison
    gradient : np.ndarray (float64)
        Output gradient array (modified in-place)
    """
    rows, cols = flat_cells.shape

    # D8 offsets for neighbor checking
    offsets = np.array([
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ], dtype=np.int32)

    # Initialize queue with all pour points
    queue = np.zeros(rows * cols, dtype=np.int32)
    queue_start = 0
    queue_end = 0

    # Visited array (distance from nearest pour point at same elevation)
    visited = np.zeros((rows, cols), dtype=np.int32)
    visited[:, :] = -1  # -1 = not visited

    # Add all pour points to queue with distance 0
    for i in range(rows):
        for j in range(cols):
            if pour_points[i, j]:
                queue[queue_end] = i * cols + j
                queue_end += 1
                visited[i, j] = 0

    # BFS from all pour points simultaneously
    while queue_start < queue_end:
        flat_idx = queue[queue_start]
        queue_start += 1

        i = flat_idx // cols
        j = flat_idx % cols
        current_dist = visited[i, j]
        current_elev = dem_rounded[i, j]

        # Check all 8 neighbors
        for k in range(8):
            di = offsets[k, 0]
            dj = offsets[k, 1]
            ni = i + di
            nj = j + dj

            # Bounds check
            if 0 <= ni < rows and 0 <= nj < cols:
                # Only expand to flat cells at SAME elevation that haven't been visited
                if (flat_cells[ni, nj] and
                    visited[ni, nj] == -1 and
                    dem_rounded[ni, nj] == current_elev):
                    visited[ni, nj] = current_dist + 1
                    queue[queue_end] = ni * cols + nj
                    queue_end += 1

    # Copy distances to gradient (pour points stay at 0)
    for i in range(rows):
        for j in range(cols):
            if visited[i, j] > 0:
                gradient[i, j] = visited[i, j]


def _compute_flat_gradient_bfs(
    flat_cells: np.ndarray,
    pour_points: np.ndarray,
    dem_rounded: np.ndarray
) -> np.ndarray:
    """
    Compute gradient from pour points using multi-source BFS.

    Parameters
    ----------
    flat_cells : np.ndarray (bool)
        Mask of flat cells
    pour_points : np.ndarray (bool)
        Mask of pour points (sources for BFS)
    dem_rounded : np.ndarray
        Rounded DEM for elevation comparison

    Returns
    -------
    np.ndarray
        Gradient values (distance from nearest pour point at same elevation)
    """
    gradient = np.zeros(flat_cells.shape, dtype=np.float64)

    if NUMBA_AVAILABLE:
        _compute_flat_gradient_bfs_jit(flat_cells, pour_points, dem_rounded, gradient)
    else:
        # Pure Python fallback (slower but works)
        rows, cols = flat_cells.shape
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (-1, 1), (1, -1), (1, 1)]

        from collections import deque
        queue = deque()
        visited = np.full((rows, cols), -1, dtype=np.int32)

        # Initialize with pour points
        pour_coords = np.argwhere(pour_points)
        for i, j in pour_coords:
            queue.append((i, j, 0))
            visited[i, j] = 0

        # BFS
        while queue:
            i, j, dist = queue.popleft()
            current_elev = dem_rounded[i, j]

            for di, dj in offsets:
                ni, nj = i + di, j + dj
                if (0 <= ni < rows and 0 <= nj < cols and
                    flat_cells[ni, nj] and visited[ni, nj] == -1 and
                    dem_rounded[ni, nj] == current_elev):
                    visited[ni, nj] = dist + 1
                    gradient[ni, nj] = dist + 1
                    queue.append((ni, nj, dist + 1))

    return gradient


def _write_geotiff(
    path: str, data: np.ndarray, transform: Affine, crs: rasterio.crs.CRS
) -> None:
    """
    Write numpy array to GeoTIFF file.

    Parameters
    ----------
    path : str
        Output file path
    data : np.ndarray
        Data array to write
    transform : Affine
        Affine transform
    crs : rasterio.crs.CRS
        Coordinate reference system
    """
    height, width = data.shape

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(data, 1)
