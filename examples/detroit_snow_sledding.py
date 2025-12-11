#!/usr/bin/env python3
"""
Detroit Snow & Sledding Analysis Example.

This example demonstrates integrating terrain elevation data with snow analysis
to identify optimal sledding locations. It produces visual outputs at multiple
stages of the pipeline:

1. DEM Visualization - Shows the Detroit elevation model
2. Snow Depth - Displays snow depth distribution
3. Sledding Score - Highlights areas with best sledding potential
4. 3D Render - Creates 3D terrain mesh with snow overlay (requires Blender)

Usage:
    # Run specific step with mock data (fast)
    python examples/detroit_snow_sledding.py --step dem --mock-data

    # Run all steps with real data
    python examples/detroit_snow_sledding.py --all-steps

    # Specify output directory
    python examples/detroit_snow_sledding.py --output-dir ./outputs --step score

    # Get help
    python examples/detroit_snow_sledding.py --help
"""

import sys
import argparse
import logging
import hashlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import gzip
import tempfile
import tarfile
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
import rasterio
import rasterio.crs as rcrs
from affine import Affine
from rasterio.warp import reproject, Resampling
from tqdm.auto import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.terrain.core import Terrain
from src.terrain.data_loading import load_dem_files

# Configure logging with both console and file output
LOG_FILE = Path(__file__).parent / "detroit_snow_sledding.log"

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Clear any existing handlers
logger.handlers = []

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logger.addHandler(console_handler)

# File handler (persists past crashes)
file_handler = logging.FileHandler(LOG_FILE, mode='w')  # 'w' = overwrite each run
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s"))
logger.addHandler(file_handler)

# Also capture root logger for library logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
    handlers=[file_handler]
)


def downsample_for_viz(arr: np.ndarray, max_dim: int = 2000) -> tuple[np.ndarray, int]:
    """
    Downsample array using stride slicing for cheap visualization.

    Args:
        arr: Input array to downsample
        max_dim: Maximum dimension size for output

    Returns:
        Tuple of (downsampled_array, stride_used)
    """
    max_shape = max(arr.shape)
    if max_shape <= max_dim:
        return arr, 1

    stride = max(1, max_shape // max_dim)
    downsampled = arr[::stride, ::stride]
    return downsampled, stride

try:
    import bpy

    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False


# =============================================================================
# SNODAS Helper Functions (from src.snow.analysis)
# =============================================================================


def _gunzip_snodas_file(gz_path, keep_original=True):
    """Decompress .gz to the same folder, remove if keep_original=False."""
    gz_path = Path(gz_path)
    out_path = gz_path.with_suffix("")
    logger.debug(f"Gunzip {gz_path} -> {out_path}")
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


def _read_snodas_header(header_path):
    """Parse header to get transform, etc."""
    logger.debug(f"Reading SNODAS header: {header_path}")
    field_types = {
        "data_units": str,
        "data_intercept": float,
        "data_slope": float,
        "no_data_value": float,
        "number_of_columns": int,
        "number_of_rows": int,
        "data_bytes_per_pixel": int,
        "minimum_data_value": float,
        "maximum_data_value": float,
        "horizontal_datum": str,
        "horizontal_precision": float,
        "projected": str,
        "geographically_corrected": str,
        "benchmark_x-axis_coordinate": float,
        "benchmark_y-axis_coordinate": float,
        "x-axis_resolution": float,
        "y-axis_resolution": float,
        "x-axis_offset": float,
        "y-axis_offset": float,
        "minimum_x-axis_coordinate": float,
        "maximum_x-axis_coordinate": float,
        "minimum_y-axis_coordinate": float,
        "maximum_y-axis_coordinate": float,
        "benchmark_column": int,
        "benchmark_row": int,
        "start_year": int,
        "start_month": int,
        "start_day": int,
        "start_hour": int,
        "start_minute": int,
        "start_second": int,
    }
    meta = {}
    try:
        with open(header_path, "r") as f:
            for line in f:
                parts = line.split(":", 1)
                if len(parts) != 2:
                    continue
                key, val = parts
                key = key.strip().lower().replace(" ", "_")
                val = val.strip()
                if val == "Not applicable":
                    continue
                if key in field_types:
                    try:
                        meta[key] = field_types[key](val)
                    except ValueError:
                        meta[key] = val
                else:
                    meta[key] = val

        meta["width"] = meta["number_of_columns"]
        meta["height"] = meta["number_of_rows"]
        # Build transform only if coordinate fields are present
        required_coords = [
            "minimum_x-axis_coordinate",
            "minimum_y-axis_coordinate",
            "maximum_x-axis_coordinate",
            "maximum_y-axis_coordinate",
        ]
        if all(coord in meta for coord in required_coords):
            meta["transform"] = rasterio.transform.from_bounds(
                meta["minimum_x-axis_coordinate"],
                meta["minimum_y-axis_coordinate"],
                meta["maximum_x-axis_coordinate"],
                meta["maximum_y-axis_coordinate"],
                meta["width"],
                meta["height"],
            )
        meta["crs"] = "EPSG:4326"
        return meta
    except Exception as e:
        logger.error(f"Error parsing header {header_path}: {e}")
        raise


def _read_snodas_binary(binary_path, meta):
    """Read big-endian 16-bit .dat, applying slope/intercept, masking no_data."""
    logger.debug(f"Reading SNODAS binary: {binary_path}")
    try:
        arr = np.fromfile(binary_path, dtype=">i2")
        arr = arr.reshape((meta["height"], meta["width"]))
        if "data_slope" in meta and "data_intercept" in meta:
            arr = arr * meta["data_slope"] + meta["data_intercept"]
        arr = np.ma.masked_equal(arr, meta["no_data_value"])
        return arr
    except Exception as e:
        logger.error(f"Error reading binary {binary_path}: {e}")
        raise


def _load_snodas_data(binary_path, header_path):
    """
    Load raw SNODAS .dat + .txt into a masked array + metadata.
    """
    logger.debug(f"Loading SNODAS from {binary_path}")
    binary_path = Path(binary_path)
    header_path = Path(header_path)

    # Handle binary file decompression
    loaded_uncompressed = False
    if binary_path.suffix == ".gz":
        uncompressed_binary_path = binary_path.with_suffix("")
        if not uncompressed_binary_path.exists():
            uncompressed_binary_path = _gunzip_snodas_file(binary_path)
        else:
            loaded_uncompressed = True
    else:
        # Already uncompressed
        uncompressed_binary_path = binary_path

    # Handle header file decompression
    if header_path.suffix == ".gz":
        uncompressed_header_path = header_path.with_suffix("")
        if not uncompressed_header_path.exists():
            uncompressed_header_path = _gunzip_snodas_file(header_path)
        else:
            loaded_uncompressed = True
    else:
        # Already uncompressed
        uncompressed_header_path = header_path

    if loaded_uncompressed:
        logger.debug("Decompressed file exists, did not decompress.")

    meta = _read_snodas_header(uncompressed_header_path)
    data = _read_snodas_binary(uncompressed_binary_path, meta)
    return data, meta


def _process_single_file(args):
    """Helper for parallel .dat.gz -> processed .npz conversion."""
    (
        dat_file,  # Path to .dat.gz
        extent,  # (minx, miny, maxx, maxy)
        processed_dir,  # Path to output directory
        target_dims,  # (tgt_h, tgt_w, tgt_transform)
        resample_to_extent,
    ) = args

    try:
        date_str = dat_file.stem.split("_")[2].split(".")[0]
        header_file = dat_file.parent / f"snow_depth_metadata_{date_str}.txt.gz"
        if not header_file.exists():
            logger.warning(f"Missing header for {dat_file}")
            return None
        if resample_to_extent:
            proc_tag = "processed"
        else:
            proc_tag = "raw"
        date = datetime.strptime(date_str, "%Y%m%d")
        out_file = processed_dir / f"snodas_{proc_tag}_{date_str}.npz"
        target_height, target_width, target_transform = target_dims

        # If out_file exists with correct shape, skip
        if out_file.exists() and resample_to_extent:
            logger.debug(f"Checking existing file {out_file}")
            try:
                with np.load(out_file) as npz:
                    d = npz["data"]
                    if d.shape == (target_height, target_width):
                        return (date, out_file)
            except Exception as e:
                logger.error(f"Failed to read {out_file} for shape check: {e}", exc_info=True)
                pass

        if out_file.exists() and not resample_to_extent:
            logger.debug(f"Existing raw file exists {out_file}")
            return (date, out_file)

        # Load raw SNODAS
        data, meta = _load_snodas_data(dat_file, header_file)

        logger.debug(f"Loaded {dat_file} with shape {data.shape} and metadata: {meta}")

        save_kwargs = {}
        if resample_to_extent:
            # 1) Allocate a buffer the size of your DEM grid
            cropped = np.zeros((target_height, target_width), dtype=np.float32)

            # 2) Reproject from SNODAS (geographic WGS84) → target (also WGS84 here)
            reproject(
                data,
                cropped,
                src_transform=meta["transform"],
                dst_transform=target_transform,
                src_crs="EPSG:4326",
                dst_crs="EPSG:4326",
                resampling=Resampling.bilinear,
            )

            # 3) Build a NumPy array of the six Affine coefficients of the *target* grid
            transform_arr = np.array(list(target_transform), dtype=np.float64)

            # 4) SNODAS is always geographic WGS84 → so we hard‐code the CRS string
            crs_str = "EPSG:4326"

            # 5) Fill in everything to save
            save_kwargs = {
                "data": cropped,
                "transform": transform_arr,
                "crs": np.string_(crs_str),
                "height": np.int32(target_height),
                "width": np.int32(target_width),
                "crop_extent": np.array(extent, dtype=np.float64),
                "no_data_value": np.array(meta["no_data_value"], dtype=np.float32),
            }

        else:
            # RAW branch: keep the original SNODAS data & transform
            data  # 2D array at native SNODAS resolution

            # 1) Convert the native Affine → 1×6 array
            native_transform = meta["transform"]  # rasterio.Affine
            transform_arr = np.array(list(native_transform), dtype=np.float64)

            # 2) Hard‐code CRS = WGS84
            crs_str = "EPSG:4326"

            # 3) Build kwargs to save the raw grid
            save_kwargs = {
                "data": data,
                "transform": transform_arr,
                "crs": np.string_(crs_str),
                "height": np.int32(data.shape[0]),
                "width": np.int32(data.shape[1]),
                "no_data_value": np.array(meta["no_data_value"], dtype=np.float32),
            }

        # Finally, write out the .npz with only numpy‐friendly types
        np.savez_compressed(out_file, **save_kwargs)
        return (date, out_file)

    except Exception as e:
        logger.error(f"Error processing {dat_file}: {e}")
        return None


def _load_processed_snodas(processed_file):
    """Load processed .npz SNODAS and reconstruct spatial metadata."""
    with np.load(processed_file) as npz:
        # 1) Pull out the data (masked if necessary)
        data = npz["data"]
        if "no_data_value" in npz.files:
            data = np.ma.masked_equal(data, npz["no_data_value"])
        else:
            data = np.ma.masked_invalid(data)

        meta = {}

        # 2) Transform → Affine
        if "transform" in npz.files:
            t_arr = npz["transform"]  # shape (6,)
            meta["transform"] = Affine(*t_arr.tolist())

        # 3) CRS → rasterio.crs.CRS
        if "crs" in npz.files:
            crs_val = npz["crs"].item()  # byte‐string or unicode
            if isinstance(crs_val, bytes):
                crs_str = crs_val.decode()
            else:
                crs_str = str(crs_val)
            meta["crs"] = rcrs.CRS.from_user_input(crs_str)

        # 4) Height/Width → int
        if "height" in npz.files:
            meta["height"] = int(npz["height"])
        if "width" in npz.files:
            meta["width"] = int(npz["width"])

        # 5) Crop extent → tuple of floats
        if "crop_extent" in npz.files:
            ext = npz["crop_extent"]  # array([minx, miny, maxx, maxy])
            meta["crop_extent"] = tuple(ext.tolist())

        # 6) Copy any remaining fields verbatim (e.g. no_data_value)
        for key in npz.files:
            if key in ("data", "transform", "crs", "height", "width", "crop_extent"):
                continue
            meta[key] = npz[key]

    return data, meta


# =============================================================================
# Pipeline Step Functions for GriddedDataLoader
# =============================================================================


def batch_process_snodas_data(snodas_dir, extent, target_shape, processed_dir="processed_snodas", max_workers=14):
    """
    Step 1: Load and process SNODAS files.

    Process all SNODAS .dat.gz files in snodas_dir,
    cropping to 'extent' & reprojecting to ~1/120° (EPSG:4326).

    Returns:
        Dict with processed_files mapping dates to file paths
    """
    snodas_dir = Path(snodas_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not snodas_dir.exists():
        logger.error(f"No SNODAS directory at {snodas_dir}")
        return {"processed_files": {}, "num_files": 0}

    dat_files = list(snodas_dir.glob("**/snow_depth_*.dat.gz"))
    if not dat_files:
        logger.error(f"No SNODAS .dat.gz files found in {snodas_dir}")
        return {"processed_files": {}, "num_files": 0}

    logger.info(f"Found {len(dat_files)} SNODAS data files to process")

    # Snodas standard pixel size in geographic degrees
    pixel_size = 1.0 / 120.0
    target_height, target_width = target_shape
    minx, miny, maxx, maxy = extent
    target_transform = rasterio.transform.from_bounds(
        minx, miny, maxx, maxy, target_width, target_height
    )

    args_list = [
        (
            f,
            extent,
            processed_dir,
            (target_height, target_width, target_transform),
            True,  # resample_to_extent
        )
        for f in dat_files
    ]
    processed_files = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_process_single_file, arg): arg[0] for arg in args_list}
        with tqdm(total=len(dat_files), desc="Processing SNODAS files") as pbar:
            for future in as_completed(future_map):
                dat_file = future_map[future]
                try:
                    result = future.result()
                    if result is not None:
                        date, out_file = result
                        processed_files[date] = out_file
                except Exception as e:
                    logger.error(f"Error processing {dat_file}: {e}")
                finally:
                    pbar.update(1)

    logger.info(f"Successfully processed {len(processed_files)} SNODAS files")
    return {"processed_files": processed_files, "num_files": len(processed_files)}


def calculate_snow_statistics(
    input_data, snow_season_start_month=11, snow_season_end_month=4
):
    """
    Step 2: Compute snow statistics from loaded files.

    Group processed SNODAS by winter season and compute aggregated stats.

    Returns:
        Dict with final statistics and metadata
    """
    processed_files = input_data["processed_files"]
    failed_files = []

    # Group by winter season
    seasons = {}
    for date, filepath in processed_files.items():
        if (
            snow_season_start_month <= date.month <= 12
            or 1 <= date.month <= snow_season_end_month
        ):
            season_year = date.year if date.month >= snow_season_start_month else date.year - 1
            key = f"{season_year}-{season_year+1}"
            seasons.setdefault(key, []).append(filepath)

    logger.info(
        f"Computing stats for {len(seasons)} winter seasons, months {snow_season_start_month} to {snow_season_end_month}"
    )

    seasonal_stats = []
    expected_shape = None
    metadata = None

    for season, filelist in tqdm(seasons.items(), desc="Processing winter seasons"):
        # Initialize running accumulators to None
        running_max = None
        running_sum = None
        running_sum_sq = None
        snow_days = None
        N_successful = 0

        for fpath in tqdm(filelist, desc=f"Loading {season} data", leave=False):
            try:
                data, meta = _load_processed_snodas(fpath)
                if expected_shape is None:
                    expected_shape = data.shape
                    logger.info(f"Setting expected shape: {expected_shape}")
                elif data.shape != expected_shape:
                    logger.error(f"Shape mismatch {fpath}")
                    failed_files.append((fpath, "Shape mismatch"))
                    continue
                if running_max is None:
                    # first day → bootstrap all accumulators
                    running_max = data.copy()
                    filled = data.filled(0.0).astype(np.float64)
                    running_sum = filled.copy()
                    running_sum_sq = filled**2
                    snow_days = (data > 0).astype(np.int32)
                else:
                    # update maximum
                    running_max = np.maximum(running_max, data)
                    # update sum & sum of squares
                    filled = data.filled(0.0).astype(np.float64)
                    running_sum += filled
                    running_sum_sq += filled**2
                    # update snow-days count
                    snow_days += (data > 0).astype(np.int32)
                N_successful += 1
                if metadata is None:
                    metadata = meta
            except Exception as e:
                logger.error(f"Error loading {fpath}: {e}")
                failed_files.append((fpath, str(e)))

        # After the loop, compute final summaries:
        if N_successful > 0:
            mean_depth = running_sum / N_successful
            std_depth = np.sqrt(running_sum_sq / N_successful - mean_depth**2)
        else:
            mean_depth = np.NaN
            std_depth = np.NaN

        stats_dict = {
            "max_depth": running_max,
            "mean_depth": mean_depth,
            "std_depth": std_depth,
            "snow_days": snow_days,
            "total_days": N_successful,
        }

        logger.debug(
            f"Computed stats for season {season}: shapes -> "
            f"max {running_max.shape}, mean {mean_depth.shape}, std {std_depth.shape}"
        )

        logger.debug(f"Computing cv_depth for season {season}")
        with np.errstate(divide="ignore", invalid="ignore"):
            cv_depth = stats_dict["std_depth"] / stats_dict["mean_depth"]
            # Replace NaN/Inf with 0 for areas with no snow (mean_depth ≈ 0)
            # These areas have no meaningful variability to measure
            cv_depth = np.where(np.isfinite(cv_depth), cv_depth, 0.0)
            stats_dict["cv_depth"] = cv_depth

        seasonal_stats.append(stats_dict)
        logger.info(
            f"Season {season}: "
            f"MaxDepth={float(np.ma.max(stats_dict['max_depth'])):.2f}, "
            f"MeanSnowDays={float(np.ma.mean(stats_dict['snow_days'])):.1f} / {stats_dict['total_days']}"
        )

    if not seasonal_stats:
        raise ValueError("No valid seasons found after processing SNODAS data.")

    # Aggregation across multiple seasons
    final_stats = {}
    final_stats["median_max_depth"] = np.ma.median(
        [s["max_depth"] for s in seasonal_stats], axis=0
    )
    # Snow day ratio
    ratios = [s["snow_days"] / s["total_days"] for s in seasonal_stats]
    final_stats["mean_snow_day_ratio"] = np.ma.mean(ratios, axis=0)
    # Inter-season variability (year-to-year variation in snow)
    seasonal_means = np.ma.stack([s["mean_depth"] for s in seasonal_stats])
    with np.errstate(divide="ignore", invalid="ignore"):
        interseason_cv = np.ma.std(seasonal_means, axis=0) / np.ma.mean(
            seasonal_means, axis=0
        )
        # Replace NaN/Inf with 0 for areas with no snow across all seasons
        interseason_cv = np.where(np.isfinite(interseason_cv), interseason_cv, 0.0)
        final_stats["interseason_cv"] = interseason_cv

    # Intra-season average CV (within-winter variability, averaged across years)
    mean_intraseason_cv = np.ma.mean(
        [s["cv_depth"] for s in seasonal_stats], axis=0
    )
    # Ensure no NaN values remain (should already be handled above, but be safe)
    mean_intraseason_cv = np.where(
        np.isfinite(mean_intraseason_cv), mean_intraseason_cv, 0.0
    )
    final_stats["mean_intraseason_cv"] = mean_intraseason_cv

    logger.info(
        f"Final stats computed: "
        f"median_max_depth range={float(np.nanmin(final_stats['median_max_depth'])):.1f}-"
        f"{float(np.nanmax(final_stats['median_max_depth'])):.1f}, "
        f"mean_intraseason_cv range={float(np.nanmin(final_stats['mean_intraseason_cv'])):.2f}-"
        f"{float(np.nanmax(final_stats['mean_intraseason_cv'])):.2f}"
    )

    # Add metadata to result
    final_stats["metadata"] = metadata
    final_stats["failed_files"] = failed_files

    return final_stats


# =============================================================================
# Slope Statistics Caching
# =============================================================================


def compute_cached_slope_statistics(dem, dem_transform, dem_crs, target_shape, target_transform, target_crs, cache_dir=None):
    """
    Compute slope statistics with caching based on DEM content hash.

    Args:
        dem: DEM array
        dem_transform: DEM affine transform
        dem_crs: DEM coordinate reference system
        target_shape: Target output shape
        target_transform: Target affine transform
        target_crs: Target coordinate reference system
        cache_dir: Cache directory (default: .slope_cache)

    Returns:
        Slope statistics object
    """
    if cache_dir is None:
        cache_dir = Path(".slope_cache")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Compute cache key from DEM content
    dem_hash = hashlib.sha256(dem.tobytes()).hexdigest()[:16]
    target_hash = hashlib.sha256(str((target_shape, target_crs)).encode()).hexdigest()[:16]
    cache_key = f"{dem_hash}_{target_hash}"
    cache_file = cache_dir / f"slope_stats_{cache_key}.npz"

    # Try to load from cache
    if cache_file.exists():
        logger.info(f"Loading cached slope statistics: {cache_file.name}")
        try:
            with np.load(cache_file, allow_pickle=True) as npz:
                # Reconstruct the slope stats object
                from dataclasses import dataclass

                @dataclass
                class SlopeStats:
                    slope_mean: np.ndarray
                    slope_max: np.ndarray
                    slope_min: np.ndarray
                    slope_std: np.ndarray
                    slope_p95: np.ndarray
                    roughness: np.ndarray
                    aspect_sin: np.ndarray
                    aspect_cos: np.ndarray

                    @property
                    def dominant_aspect(self) -> np.ndarray:
                        """Compute dominant aspect from vector-averaged sin/cos components."""
                        return np.degrees(np.arctan2(self.aspect_sin, self.aspect_cos)) % 360

                stats = SlopeStats(
                    slope_mean=npz['slope_mean'],
                    slope_max=npz['slope_max'],
                    slope_min=npz['slope_min'],
                    slope_std=npz['slope_std'],
                    slope_p95=npz['slope_p95'],
                    roughness=npz['roughness'],
                    aspect_sin=npz['aspect_sin'],
                    aspect_cos=npz['aspect_cos'],
                )
                logger.info("✓ Cache hit for slope statistics")
                return stats
        except Exception as e:
            logger.warning(f"Failed to load slope stats cache: {e}")

    # Compute slope statistics
    logger.info("Computing slope statistics from DEM (tiled processing)...")
    from src.snow.slope_statistics import compute_tiled_slope_statistics

    slope_stats = compute_tiled_slope_statistics(
        dem=dem,
        dem_transform=dem_transform,
        dem_crs=dem_crs,
        target_shape=target_shape,
        target_transform=target_transform,
        target_crs=target_crs,
    )

    # Save to cache
    try:
        np.savez_compressed(
            cache_file,
            slope_mean=slope_stats.slope_mean,
            slope_max=slope_stats.slope_max,
            slope_min=slope_stats.slope_min,
            slope_std=slope_stats.slope_std,
            slope_p95=slope_stats.slope_p95,
            roughness=slope_stats.roughness,
            aspect_sin=slope_stats.aspect_sin,
            aspect_cos=slope_stats.aspect_cos,
        )
        logger.debug(f"Cached slope statistics to {cache_file.name}")
    except Exception as e:
        logger.warning(f"Failed to save slope stats cache: {e}")

    return slope_stats


# =============================================================================
# Mock Data and Visualization Helpers
# =============================================================================


def create_mock_snow_data(shape: tuple) -> dict:
    """
    Create mock snow data for testing.

    Args:
        shape: Shape of the snow data arrays

    Returns:
        Dictionary with mock snow statistics
    """
    logger.info("Creating mock snow data for testing...")

    # Create realistic-looking mock data
    np.random.seed(42)

    # Snow depth (0-300cm, concentrated in certain areas)
    median_max_depth = np.random.gamma(2, 30, shape).astype(np.float32)
    median_max_depth = np.clip(median_max_depth, 0, 300)

    # Snow coverage ratio (0-1, mostly high in winter)
    mean_snow_day_ratio = np.random.beta(8, 2, shape).astype(np.float32)

    # Variability (0-1, lower is more consistent)
    interseason_cv = np.random.beta(2, 8, shape).astype(np.float32) * 0.5
    mean_intraseason_cv = np.random.beta(2, 8, shape).astype(np.float32) * 0.3

    return {
        "median_max_depth": median_max_depth,
        "mean_snow_day_ratio": mean_snow_day_ratio,
        "interseason_cv": interseason_cv,
        "mean_intraseason_cv": mean_intraseason_cv,
    }


def visualize_dem(dem: np.ndarray, output_path: Path):
    """
    Create DEM visualization.

    Args:
        dem: Digital elevation model array
        output_path: Where to save the visualization
    """
    logger.info(f"Creating DEM visualization: {output_path}")

    fig, ax = plt.subplots(figsize=(12, 9))

    # Use viridis colormap for elevation
    im = ax.imshow(dem, cmap="viridis", aspect="auto")
    ax.set_title("Detroit Digital Elevation Model", fontsize=16, fontweight="bold")
    ax.set_xlabel("Longitude (pixels)")
    ax.set_ylabel("Latitude (pixels)")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Elevation (meters)", rotation=270, labelpad=20)

    # Add statistics text
    stats_text = f"Min: {np.nanmin(dem):.1f}m\nMax: {np.nanmax(dem):.1f}m\nMean: {np.nanmean(dem):.1f}m"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ DEM visualization saved: {output_path}")


def visualize_snow_depth(snow_depth: np.ndarray, output_path: Path):
    """
    Create snow depth visualization.

    Args:
        snow_depth: Snow depth array (cm)
        output_path: Where to save the visualization
    """
    logger.info(f"Creating snow depth visualization: {output_path}")

    fig, ax = plt.subplots(figsize=(12, 9))

    # Use viridis colormap for snow depth
    # Mask zero values for better visualization
    snow_depth_masked = np.ma.masked_where(snow_depth == 0, snow_depth)

    im = ax.imshow(snow_depth_masked, cmap="viridis", aspect="auto", vmin=0, vmax=np.nanmax(snow_depth))
    ax.set_title("Snow Depth Distribution", fontsize=16, fontweight="bold")
    ax.set_xlabel("Longitude (pixels)")
    ax.set_ylabel("Latitude (pixels)")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Snow Depth (cm)", rotation=270, labelpad=20)

    # Add statistics text
    valid_depth = snow_depth[snow_depth > 0]
    if len(valid_depth) > 0:
        stats_text = (
            f"Min: {np.nanmin(valid_depth):.1f}cm\n"
            f"Max: {np.nanmax(valid_depth):.1f}cm\n"
            f"Mean: {np.nanmean(valid_depth):.1f}cm\n"
            f"Coverage: {(len(valid_depth) / snow_depth.size) * 100:.1f}%"
        )
    else:
        stats_text = "No snow detected"

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Snow depth visualization saved: {output_path}")


def visualize_sledding_score(score: np.ndarray, output_path: Path):
    """
    Create sledding score visualization.

    Args:
        score: Sledding suitability score (0-1)
        output_path: Where to save the visualization
    """
    logger.info(f"Creating sledding score visualization: {output_path}")

    fig, ax = plt.subplots(figsize=(12, 9))

    # Use viridis colormap (yellow = good sledding, purple = poor)
    im = ax.imshow(score, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_title("Sledding Suitability Score", fontsize=16, fontweight="bold")
    ax.set_xlabel("Longitude (pixels)")
    ax.set_ylabel("Latitude (pixels)")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Sledding Score (0=poor, 1=excellent)", rotation=270, labelpad=20)

    # Add statistics and interpretation text
    stats_text = (
        f"Min: {np.nanmin(score):.2f}\n"
        f"Max: {np.nanmax(score):.2f}\n"
        f"Mean: {np.nanmean(score):.2f}\n"
        f"\n"
        f"Yellow: Great sledding\n"
        f"Green: Moderate\n"
        f"Purple: Poor"
    )

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Sledding score visualization saved: {output_path}")


def save_sledding_score_histogram(score: np.ndarray, output_path: Path):
    """
    Create histogram of sledding score distribution.

    Args:
        score: Sledding suitability score (0-1)
        output_path: Where to save the histogram
    """
    logger.info(f"Creating sledding score histogram: {output_path}")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Flatten and remove NaNs
    scores_flat = score.flatten()
    scores_clean = scores_flat[~np.isnan(scores_flat)]

    # Create histogram with percentile lines
    ax.hist(scores_clean, bins=50, color="steelblue", alpha=0.7, edgecolor="black")

    # Calculate percentiles
    p25 = np.percentile(scores_clean, 25)
    p50 = np.percentile(scores_clean, 50)
    p75 = np.percentile(scores_clean, 75)

    # Add percentile lines
    ax.axvline(p25, color="orange", linestyle="--", linewidth=2, label=f"25th percentile: {p25:.3f}")
    ax.axvline(p50, color="red", linestyle="--", linewidth=2, label=f"50th percentile: {p50:.3f}")
    ax.axvline(p75, color="green", linestyle="--", linewidth=2, label=f"75th percentile: {p75:.3f}")

    ax.set_xlabel("Sledding Score", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency (pixels)", fontsize=12, fontweight="bold")
    ax.set_title("Sledding Score Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Sledding score histogram saved: {output_path}")


def save_sledding_score_percentiles(score: np.ndarray, output_path: Path):
    """
    Create percentile map showing where each location falls in the score distribution.

    Args:
        score: Sledding suitability score (0-1)
        output_path: Where to save the percentile map
    """
    logger.info(f"Creating sledding score percentile map: {output_path}")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Flatten to get percentiles
    scores_flat = score.flatten()
    scores_clean = scores_flat[~np.isnan(scores_flat)]

    # Create percentile ranks (0-100)
    percentile_map = np.zeros_like(score)
    valid_mask = ~np.isnan(score)
    percentile_map[valid_mask] = (
        np.searchsorted(np.sort(scores_clean), score[valid_mask]) / len(scores_clean) * 100
    )
    percentile_map[~valid_mask] = np.nan

    # Use turbo colormap for percentiles (0=red, 100=blue)
    im = ax.imshow(percentile_map, cmap="turbo", aspect="equal", interpolation="nearest", vmin=0, vmax=100)
    ax.set_title("Sledding Score Percentiles", fontweight="bold", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add colorbar with percentile labels
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, ticks=[0, 25, 50, 75, 100])
    cbar.ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    cbar.set_label("Percentile Rank", rotation=270, labelpad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Sledding score percentile map saved: {output_path}")


def save_sledding_score_filtered(score: np.ndarray, output_path: Path, threshold: float = 0.7):
    """
    Create filtered map showing only excellent sledding locations (score > threshold).

    Args:
        score: Sledding suitability score (0-1)
        output_path: Where to save the filtered map
        threshold: Score threshold for excellent locations (default: 0.7)
    """
    logger.info(f"Creating filtered sledding score map (threshold > {threshold}): {output_path}")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create filtered score: excellent locations retain value, others become 0
    excellent_spots = score > threshold
    filtered_score = np.where(excellent_spots, score, 0)

    # Count excellent pixels
    excellent_count = np.sum(excellent_spots)
    total_pixels = np.sum(~np.isnan(score))
    excellent_pct = (excellent_count / total_pixels * 100) if total_pixels > 0 else 0

    # Use RdYlGn colormap with vmin/vmax to highlight only the excellent range (0.7-1.0)
    im = ax.imshow(filtered_score, cmap="RdYlGn", aspect="equal", interpolation="nearest", vmin=0.7, vmax=1.0)
    ax.set_title(f"Excellent Sledding Locations (Score > {threshold})", fontweight="bold", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Sledding Score", rotation=270, labelpad=20)

    # Add annotation showing how many excellent pixels found
    ax.text(
        0.5, -0.12,
        f"{excellent_count:,} excellent pixels ({excellent_pct:.1f}% of area)",
        transform=ax.transAxes,
        ha='center',
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Sledding score filtered map saved: {output_path}")


def save_slope_stat_panels(slope_stats, output_dir: Path):
    """
    Save individual slope statistics panels to separate PNG files.

    Args:
        slope_stats: SlopeStatistics object from compute_tiled_slope_statistics()
        output_dir: Directory to save individual panel PNG files
    """
    logger.info(f"Saving individual slope statistics panels to {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Panel 1: Slope Mean
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(slope_stats.slope_mean, cmap="viridis", aspect="equal", interpolation="nearest")
    ax.set_title("Slope Mean (°)", fontweight="bold", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"{np.nanmin(slope_stats.slope_mean):.1f}-{np.nanmax(slope_stats.slope_mean):.1f}°", rotation=270, labelpad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "mean.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Saved: {output_dir / 'mean.png'}")

    # Panel 2: Slope Max
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(slope_stats.slope_max, cmap="inferno", aspect="equal", interpolation="nearest")
    ax.set_title("Slope Max (°) - Cliff Detection", fontweight="bold", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"{np.nanmin(slope_stats.slope_max):.1f}-{np.nanmax(slope_stats.slope_max):.1f}°", rotation=270, labelpad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "max.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Saved: {output_dir / 'max.png'}")

    # Panel 3: Slope Min
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(slope_stats.slope_min, cmap="cividis", aspect="equal", interpolation="nearest")
    ax.set_title("Slope Min (°) - Flat Spots", fontweight="bold", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"{np.nanmin(slope_stats.slope_min):.1f}-{np.nanmax(slope_stats.slope_min):.1f}°", rotation=270, labelpad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "min.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Saved: {output_dir / 'min.png'}")

    # Panel 4: Slope Std Dev
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(slope_stats.slope_std, cmap="magma", aspect="equal", interpolation="nearest")
    ax.set_title("Slope Std Dev (°) - Terrain Consistency", fontweight="bold", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"{np.nanmin(slope_stats.slope_std):.1f}-{np.nanmax(slope_stats.slope_std):.1f}°", rotation=270, labelpad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "std.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Saved: {output_dir / 'std.png'}")

    # Panel 5: Slope P95
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(slope_stats.slope_p95, cmap="plasma", aspect="equal", interpolation="nearest")
    ax.set_title("Slope P95 (°) - 95th Percentile", fontweight="bold", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"{np.nanmin(slope_stats.slope_p95):.1f}-{np.nanmax(slope_stats.slope_p95):.1f}°", rotation=270, labelpad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "p95.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Saved: {output_dir / 'p95.png'}")

    # Panel 6: Roughness
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(slope_stats.roughness, cmap="inferno", aspect="equal", interpolation="nearest")
    ax.set_title("Roughness (m elev std)", fontweight="bold", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"{np.nanmin(slope_stats.roughness):.1f}-{np.nanmax(slope_stats.roughness):.1f}m", rotation=270, labelpad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "roughness.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Saved: {output_dir / 'roughness.png'}")

    # Panel 7: Aspect (dominant slope direction)
    from matplotlib.colors import Normalize
    fig, ax = plt.subplots(figsize=(10, 8))
    aspect_deg = slope_stats.dominant_aspect

    # Use twilight colormap with 180° offset: North (0°) = light, South (180°) = dark
    # Add 180° to flip the coordinate system so North appears light and South appears dark
    norm = Normalize(vmin=0, vmax=360)
    aspect_colors = plt.cm.twilight(norm((aspect_deg + 180) % 360))

    # Fade to gray for low slopes
    slope_fade_threshold = 3.0
    slope_fade = np.clip(slope_stats.slope_mean / slope_fade_threshold, 0, 1)
    gray = 0.5
    for i in range(3):
        aspect_colors[:, :, i] = gray + slope_fade * (aspect_colors[:, :, i] - gray)

    im = ax.imshow(aspect_colors, aspect="equal", interpolation="nearest")
    ax.set_title("Dominant Aspect (°) - N=light, S=dark", fontweight="bold", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.twilight, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, ticks=[0, 90, 180, 270, 360])
    cbar.ax.set_yticklabels(['N', 'E', 'S', 'W', 'N'])
    plt.tight_layout()
    plt.savefig(output_dir / "aspect.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Saved: {output_dir / 'aspect.png'}")

    # Panel 8: Aspect Strength
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(slope_stats.aspect_strength, cmap="viridis", aspect="equal", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("Aspect Strength (0=varied, 1=uniform)", fontweight="bold", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_dir / "aspect_strength.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Saved: {output_dir / 'aspect_strength.png'}")


def save_slope_penalty_panels(slope_stats, output_dir: Path):
    """
    Save individual slope penalty panels to separate PNG files.

    Args:
        slope_stats: SlopeStatistics object
        output_dir: Directory to save individual panel PNG files
    """
    from src.scoring import trapezoidal, dealbreaker, terrain_consistency

    logger.info(f"Saving individual slope penalty panels to {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute base trapezoidal score
    slope_score = trapezoidal(
        slope_stats.slope_mean,
        sweet_range=(5, 15),
        ramp_range=(3, 25),
    )

    # Compute cliff penalty
    cliff_penalty = dealbreaker(
        slope_stats.slope_p95,
        threshold=25,
        falloff=10,
    )
    cliff_mask = slope_stats.slope_p95 > 25

    # Compute terrain consistency
    consistency_penalty = terrain_consistency(
        slope_stats.roughness,
        slope_stats.slope_std,
        roughness_threshold=30,
        slope_std_threshold=10,
    )
    rough_mask = (slope_stats.roughness > 15) | (slope_stats.slope_std > 5)

    # Final slope score
    final_score = slope_score * cliff_penalty * consistency_penalty
    score_reduction = slope_score - final_score

    # Panel 1: Base score
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(slope_score, cmap="viridis", aspect="equal", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("Base Slope Score", fontweight="bold", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_dir / "base_score.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Saved: {output_dir / 'base_score.png'}")

    # Panel 2: Cliff penalty
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cliff_penalty, cmap="plasma", aspect="equal", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("Cliff Penalty (p95 > 25°)", fontweight="bold", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8)
    cliff_pixels = np.sum(cliff_mask)
    ax.text(0.5, -0.15, f"{cliff_pixels} pixels penalized",
            transform=ax.transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7))
    plt.tight_layout()
    plt.savefig(output_dir / "cliff_penalty.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Saved: {output_dir / 'cliff_penalty.png'}")

    # Panel 3: Terrain consistency
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(consistency_penalty, cmap="plasma", aspect="equal", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("Terrain Consistency", fontweight="bold", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8)
    rough_pixels = np.sum(rough_mask)
    ax.text(0.5, -0.15, f"{rough_pixels} pixels with variation",
            transform=ax.transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7))
    plt.tight_layout()
    plt.savefig(output_dir / "terrain_consistency.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Saved: {output_dir / 'terrain_consistency.png'}")

    # Panel 4: Combined penalty
    combined_penalty = cliff_penalty * consistency_penalty
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(combined_penalty, cmap="inferno", aspect="equal", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("Combined Penalties", fontweight="bold", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_dir / "combined_penalty.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Saved: {output_dir / 'combined_penalty.png'}")

    # Panel 5: Final score
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(final_score, cmap="viridis", aspect="equal", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("Final Slope Score", fontweight="bold", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_dir / "final_score.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Saved: {output_dir / 'final_score.png'}")

    # Panel 6: Score reduction
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(score_reduction, cmap="magma", aspect="equal", interpolation="nearest", vmin=0)
    ax.set_title("Score Reduction", fontweight="bold", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_dir / "score_reduction.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Saved: {output_dir / 'score_reduction.png'}")


def save_score_component_panels(
    slope_stats,
    snow_stats: dict,
    final_score: np.ndarray,
    output_dir: Path,
    scorer=None,
):
    """
    Save individual score component panels to separate PNG files.

    Args:
        slope_stats: SlopeStatistics object
        snow_stats: Dictionary with snow statistics
        final_score: Final sledding score array
        output_dir: Directory to save individual panel PNG files
        scorer: Optional ScoreCombiner to generate formula from config
    """
    from src.scoring import trapezoidal, dealbreaker, linear, snow_consistency, terrain_consistency
    from src.scoring.configs import DEFAULT_SLEDDING_SCORER

    logger.info(f"Saving individual score component panels to {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Use provided scorer or default
    if scorer is None:
        scorer = DEFAULT_SLEDDING_SCORER

    # Compute all transformations
    slope_score = trapezoidal(slope_stats.slope_mean, sweet_range=(5, 15), ramp_range=(3, 25))
    cliff_penalty = dealbreaker(slope_stats.slope_p95, threshold=25, falloff=10)
    terrain_cons = terrain_consistency(slope_stats.roughness, slope_stats.slope_std)
    depth_score = trapezoidal(snow_stats["median_max_depth"], sweet_range=(150, 500), ramp_range=(50, 1000))
    coverage_score = linear(snow_stats["mean_snow_day_ratio"], value_range=(0, 1), power=0.5)
    consistency_score = snow_consistency(
        snow_stats["interseason_cv"],
        snow_stats["mean_intraseason_cv"],
    )

    # Aspect bonus: northness × strength
    northness = np.cos(np.radians(slope_stats.dominant_aspect))
    aspect_bonus_raw = northness * slope_stats.aspect_strength * 0.05
    aspect_score = linear(aspect_bonus_raw, value_range=(-0.05, 0.05))

    # Runout bonus
    runout_bonus = np.where(slope_stats.slope_min < 5, 1.0, 0.0)

    # Get weights from scorer config
    weights = {comp.name: comp.weight for comp in scorer.components if comp.role == "additive"}
    w_slope = weights.get("slope_mean", 0.30)
    w_depth = weights.get("snow_depth", 0.15)
    w_coverage = weights.get("snow_coverage", 0.25)
    w_consistency = weights.get("snow_consistency", 0.20)
    w_aspect = weights.get("aspect_bonus", 0.05)
    w_runout = weights.get("runout_bonus", 0.05)

    # Combination using config weights
    additive_sum = (
        w_slope * slope_score +
        w_depth * depth_score +
        w_coverage * coverage_score +
        w_consistency * consistency_score +
        w_aspect * aspect_score +
        w_runout * runout_bonus
    )
    multiplicative = cliff_penalty * terrain_cons
    combined_final = additive_sum * multiplicative

    # Save equation to markdown file
    additive_parts = []
    multiplicative_parts = []
    for comp in scorer.components:
        name = comp.name.replace("_", " ").title()
        if comp.role == "additive":
            additive_parts.append(f"{comp.weight:.2f} × {name}")
        else:
            multiplicative_parts.append(name)

    additive_formula = "  +  ".join(additive_parts)
    multiplicative_formula = "  ×  ".join(multiplicative_parts)

    additive_compact = " + ".join(
        f"{comp.weight:.2f}×{comp.name.replace('_', '').title()[:6]}"
        for comp in scorer.components if comp.role == "additive"
    )
    mult_compact = " × ".join(
        comp.name.replace("_", "").title()[:8]
        for comp in scorer.components if comp.role == "multiplicative"
    )

    equation_text = (
        f"# Sledding Score Formula\n\n"
        f"**Scorer**: {scorer.name}\n\n"
        f"## Additive Components (weighted sum = base score)\n\n"
        f"```\n{additive_formula}\n```\n\n"
        f"## Multiplicative Penalties\n\n"
        f"```\n× {multiplicative_formula}\n```\n\n"
        f"## Final Equation\n\n"
        f"```\nFINAL = ({additive_compact}) × {mult_compact}\n```\n"
    )

    equation_path = output_dir / "equation.md"
    with open(equation_path, 'w') as f:
        f.write(equation_text)
    logger.info(f"✓ Saved: {equation_path}")

    # Helper function to save individual panels
    def save_single_panel(data, filename, title, cmap, norm=None, vmin=0, vmax=1):
        fig, ax = plt.subplots(figsize=(10, 8))
        if norm:
            im = ax.imshow(data, cmap=cmap, aspect="equal", interpolation="nearest", norm=norm)
        else:
            im = ax.imshow(data, cmap=cmap, aspect="equal", interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontweight="bold", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"✓ Saved: {output_dir / filename}")

    # Row 1: Additive components
    save_single_panel(slope_score, "slope_score.png", "Slope Score", "viridis")
    save_single_panel(depth_score, "depth_score.png", "Depth Score", "viridis")
    save_single_panel(coverage_score, "coverage_score.png", "Coverage Score", "viridis")
    save_single_panel(consistency_score, "consistency_score.png", "Consistency Score", "viridis")

    # Row 2: Additional components + penalties
    save_single_panel(aspect_score, "aspect_score.png", "Aspect Score", "viridis")
    save_single_panel(runout_bonus, "runout_bonus.png", "Runout Bonus", "viridis")
    save_single_panel(cliff_penalty, "cliff_penalty.png", "Cliff Penalty", "plasma")
    save_single_panel(terrain_cons, "terrain_consistency.png", "Terrain Consistency", "plasma")

    # Row 3: Raw inputs
    save_single_panel(slope_stats.slope_mean, "raw_slope_mean.png", "Raw: Slope Mean", "cividis",
                     vmin=np.nanmin(slope_stats.slope_mean), vmax=np.nanmax(slope_stats.slope_mean))
    save_single_panel(snow_stats["median_max_depth"], "raw_snow_depth.png", "Raw: Snow Depth", "viridis",
                     vmin=np.nanmin(snow_stats["median_max_depth"]), vmax=np.nanmax(snow_stats["median_max_depth"]))
    save_single_panel(snow_stats["interseason_cv"], "raw_interseason_cv.png", "Raw: Interseason CV", "magma",
                     vmin=np.nanmin(snow_stats["interseason_cv"]), vmax=np.nanmax(snow_stats["interseason_cv"]))
    save_single_panel(snow_stats["mean_intraseason_cv"], "raw_intraseason_cv.png", "Raw: Intraseason CV", "magma",
                     vmin=np.nanmin(snow_stats["mean_intraseason_cv"]), vmax=np.nanmax(snow_stats["mean_intraseason_cv"]))

    # Row 4: Combination steps
    save_single_panel(additive_sum, "additive_sum.png", "Additive Sum", "viridis")
    save_single_panel(multiplicative, "multiplicative.png", "Multiplicative Product", "inferno")

    # Final score with linear color gradient
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(combined_final, cmap="viridis", aspect="equal", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("Final Score", fontweight="bold", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_dir / "final_score.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Saved: {output_dir / 'final_score.png'}")

    # Score reduction
    score_reduction = additive_sum - combined_final
    save_single_panel(score_reduction, "score_reduction.png", "Score Reduction", "magma",
                     vmin=0, vmax=max(0.01, np.nanmax(score_reduction)))


def visualize_sledding_factors(
    snow_stats: dict,
    sledding_score: np.ndarray,
    output_path: Path,
    slope_score: np.ndarray = None,
    slope_deg: np.ndarray = None,
    depth_score: np.ndarray = None,
    coverage_score: np.ndarray = None,
    variability_score: np.ndarray = None,
):
    """
    Create multi-panel visualization showing all factors in sledding score calculation.

    Args:
        snow_stats: Dictionary with snow statistics
        sledding_score: Final sledding score array
        output_path: Where to save the visualization
        slope_score: Slope suitability score (0-1)
        slope_deg: Raw slope in degrees
        depth_score: Pre-computed depth score from SnowAnalysis
        coverage_score: Pre-computed coverage score from SnowAnalysis
        variability_score: Pre-computed variability/consistency score from SnowAnalysis
    """
    logger.info(f"Creating sledding factors visualization: {output_path}")

    # Use pre-computed scores if provided, otherwise raise error (DRY - don't duplicate logic)
    if depth_score is None or coverage_score is None or variability_score is None:
        raise ValueError(
            "depth_score, coverage_score, and variability_score must be provided. "
            "Get these from SnowAnalysis after calling calculate_sledding_score()."
        )

    # Create 3x3 grid for 9 panels (or 2x5 for 10 if we have slope)
    has_slope = slope_score is not None and slope_deg is not None
    if has_slope:
        fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    else:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    panel_idx = 0

    # Panel: Slope (degrees) - raw
    # Use adaptive vmax based on data range (95th percentile) for better visibility
    if has_slope:
        ax = axes[panel_idx]
        actual_max = np.nanmax(slope_deg)
        slope_vmax = min(15, np.nanpercentile(slope_deg, 99))  # Cap display at 15° for visibility
        im = ax.imshow(slope_deg, cmap="cividis", aspect="equal", interpolation="nearest", vmin=0, vmax=slope_vmax)
        ax.set_title(f"Slope (actual max={actual_max:.1f}°)", fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8)
        panel_idx += 1

    # Panel: Median Max Snow Depth (raw)
    ax = axes[panel_idx]
    im = ax.imshow(snow_stats["median_max_depth"], cmap="viridis", aspect="equal", interpolation="nearest")
    ax.set_title("Median Max Snow Depth (mm)", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    panel_idx += 1

    # Panel: Snow Coverage Ratio (raw)
    ax = axes[panel_idx]
    im = ax.imshow(snow_stats["mean_snow_day_ratio"], cmap="plasma", aspect="equal", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("Snow Day Ratio (persistence)", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    panel_idx += 1

    # Panel: Interseason CV (raw)
    ax = axes[panel_idx]
    im = ax.imshow(snow_stats["interseason_cv"], cmap="magma", aspect="equal", interpolation="nearest")
    ax.set_title("Interseason CV (year-to-year)", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    panel_idx += 1

    # Panel: Intraseason CV (raw)
    ax = axes[panel_idx]
    im = ax.imshow(snow_stats["mean_intraseason_cv"], cmap="magma", aspect="equal", interpolation="nearest")
    ax.set_title("Mean Intraseason CV (within-winter)", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    panel_idx += 1

    # Panel: Slope Score (normalized, 25% weight)
    if has_slope:
        ax = axes[panel_idx]
        im = ax.imshow(slope_score, cmap="plasma", aspect="equal", interpolation="nearest", vmin=0, vmax=1)
        ax.set_title("Slope Score (25% weight)", fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8)
        panel_idx += 1

    # Panel: Depth Score (normalized, 30% weight)
    ax = axes[panel_idx]
    im = ax.imshow(depth_score, cmap="plasma", aspect="equal", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("Depth Score (30% weight)", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    panel_idx += 1

    # Panel: Coverage Score (normalized, 30% weight)
    ax = axes[panel_idx]
    im = ax.imshow(coverage_score, cmap="plasma", aspect="equal", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("Coverage Score (30% weight)", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    panel_idx += 1

    # Panel: Variability Score (normalized, 15% weight)
    ax = axes[panel_idx]
    im = ax.imshow(variability_score, cmap="plasma", aspect="equal", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("Consistency Score (15% weight)", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    panel_idx += 1

    # Panel: Final Sledding Score
    ax = axes[panel_idx]
    im = ax.imshow(sledding_score, cmap="viridis", aspect="equal", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("Final Sledding Score", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Remove axis ticks for cleaner look
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    # Add overall title with formula
    if has_slope:
        formula = "Score = 0.25×Slope + 0.30×Depth + 0.30×Coverage + 0.15×Consistency"
    else:
        formula = "Score = 0.30×Depth + 0.30×Coverage + 0.15×Consistency (no slope)"
    fig.suptitle(
        f"Sledding Score Components\n{formula}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Sledding factors visualization saved: {output_path}")


def render_3d_with_snow(terrain: Terrain, output_path: Path):
    """
    Create 3D render with snow overlay using Blender.

    Args:
        terrain: Terrain object with snow data layers
        output_path: Where to save the render
    """
    if not BLENDER_AVAILABLE:
        logger.warning("Blender not available - skipping 3D render")
        return

    logger.info(f"Creating 3D render with snow overlay: {output_path}")

    # pylint: disable=import-outside-toplevel
    from src.terrain.core import (
        clear_scene,
        position_camera_relative,
        setup_light,
        setup_render_settings,
        render_scene_to_file,
    )

    # Clear Blender scene
    clear_scene()

    # Apply transforms if not already applied
    if not terrain.data_layers["dem"].get("transformed", False):
        logger.info("Applying identity transforms to terrain for mesh creation")
        # For mock data without real transforms, just mark the original data as transformed
        terrain.data_layers["dem"]["transformed_data"] = terrain.data_layers["dem"]["data"]
        # Use crs from metadata or default to EPSG:4326
        terrain.data_layers["dem"]["transformed_crs"] = terrain.data_layers["dem"].get("crs", "EPSG:4326")
        # Mark as transformed
        terrain.data_layers["dem"]["transformed"] = True

    # Compute colors if not already computed
    if terrain.vertex_colors is None:
        logger.info("Computing vertex colors")
        terrain.compute_colors()

    # Create mesh with sledding score coloring
    mesh_obj = terrain.create_mesh(
        scale_factor=100,
        height_scale=2.0,
        center_model=True,
        boundary_extension=0.1,
    )

    # Setup camera
    position_camera_relative(
        mesh_obj,
        direction="south",
        distance=0.264,
        elevation=0.396,
    )

    # Setup lighting and rendering
    setup_light(angle=2, energy=3)
    setup_render_settings(use_gpu=True, samples=512, use_denoising=False)

    # Render
    render_scene_to_file(
        output_path=output_path,
        width=960,
        height=720,
        file_format="PNG",
        color_mode="RGBA",
        compression=90,
        save_blend_file=False,
    )

    logger.info(f"✓ 3D render saved: {output_path}")


def run_step_dem(output_dir: Path, use_mock: bool):
    """Run DEM visualization step."""
    logger.info("\n" + "=" * 70)
    logger.info("Step 1: DEM Visualization")
    logger.info("=" * 70)

    if use_mock:
        # Create mock DEM with realistic Detroit geographic extent
        logger.info("Using mock DEM data")
        from rasterio.transform import Affine

        # Detroit bounding box (approximate)
        detroit_bounds = (-83.3, 42.2, -82.9, 42.5)
        dem_height, dem_width = 500, 500
        dem = np.random.rand(dem_height, dem_width) * 100 + 150  # 150-250m elevation

        # Create proper geographic transform
        transform = Affine.translation(detroit_bounds[0], detroit_bounds[3]) * Affine.scale(
            (detroit_bounds[2] - detroit_bounds[0]) / dem_width,
            -(detroit_bounds[3] - detroit_bounds[1]) / dem_height
        )
    else:
        # Load real DEM
        dem_dir = project_root / "data" / "dem" / "detroit"
        if not dem_dir.exists():
            logger.warning(f"DEM directory not found: {dem_dir}")
            logger.info("Falling back to mock data")
            from rasterio.transform import Affine

            detroit_bounds = (-83.3, 42.2, -82.9, 42.5)
            dem_height, dem_width = 500, 500
            dem = np.random.rand(dem_height, dem_width) * 100 + 150
            transform = Affine.translation(detroit_bounds[0], detroit_bounds[3]) * Affine.scale(
                (detroit_bounds[2] - detroit_bounds[0]) / dem_width,
                -(detroit_bounds[3] - detroit_bounds[1]) / dem_height
            )
        else:
            try:
                # Load all HGT files in directory (110 files covering full extent)
                # This defines the full geographic bounds for SNODAS processing
                dem, transform = load_dem_files(dem_dir, pattern="*.hgt")
            except ValueError as e:
                logger.warning(f"Failed to load DEM files: {e}")
                logger.info("Falling back to mock data")
                from rasterio.transform import Affine

                detroit_bounds = (-83.3, 42.2, -82.9, 42.5)
                dem_height, dem_width = 500, 500
                dem = np.random.rand(dem_height, dem_width) * 100 + 150
                transform = Affine.translation(detroit_bounds[0], detroit_bounds[3]) * Affine.scale(
                    (detroit_bounds[2] - detroit_bounds[0]) / dem_width,
                    -(detroit_bounds[3] - detroit_bounds[1]) / dem_height
                )

    # Visualize (downsample if needed for memory efficiency)
    dem_viz, stride = downsample_for_viz(dem)
    if stride > 1:
        logger.info(f"Downsampling DEM for visualization: {dem.shape} -> {dem_viz.shape} (stride={stride})")
    (output_dir / "01_raw").mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "01_raw" / "dem.png"
    visualize_dem(dem_viz, output_path)

    return dem, transform


def run_step_snow(output_dir: Path, dem: np.ndarray, use_mock: bool, terrain=None, snodas_dir=None):
    """Run snow depth visualization step."""
    logger.info("\n" + "=" * 70)
    logger.info("Step 2: Snow Depth Analysis")
    logger.info("=" * 70)

    if use_mock:
        # Create mock snow data
        logger.info("Using mock snow data")
        snow_stats = create_mock_snow_data(dem.shape)
    else:
        # Try to load real SNODAS data using GriddedDataLoader
        if snodas_dir and snodas_dir.exists() and terrain:
            logger.info(f"Loading real SNODAS data from: {snodas_dir}")
            try:
                from src.terrain.gridded_data import GriddedDataLoader

                # Define pipeline steps
                pipeline = [
                    ("load_snodas", batch_process_snodas_data, {}),
                    ("compute_stats", calculate_snow_statistics, {}),
                ]

                # Create loader with terrain context
                loader = GriddedDataLoader(terrain=terrain, cache_dir=Path("snow_analysis_cache"))

                # Run pipeline
                result = loader.run_pipeline(
                    data_source=snodas_dir,
                    pipeline=pipeline,
                    cache_name="snodas"
                )

                # Extract snow statistics (remove metadata and failed_files)
                snow_stats = {k: v for k, v in result.items() if k not in ("metadata", "failed_files")}
                failed = result.get("failed_files", [])
                logger.info(f"✓ Loaded SNODAS data ({len(failed)} files failed)")
            except Exception as e:
                logger.warning(f"Failed to load SNODAS data: {e}")
                logger.info("Falling back to mock data")
                snow_stats = create_mock_snow_data(dem.shape)
        else:
            if not snodas_dir:
                logger.warning("No SNODAS directory specified (use --snodas-dir)")
            elif not snodas_dir.exists():
                logger.warning(f"SNODAS directory not found: {snodas_dir}")
            else:
                logger.warning("Terrain object not available for SNODAS processing")
            logger.info("Falling back to mock data")
            snow_stats = create_mock_snow_data(dem.shape)

    # Visualize snow depth (downsample if needed for memory efficiency)
    snow_data = snow_stats["median_max_depth"]
    snow_viz, stride = downsample_for_viz(snow_data)
    if stride > 1:
        logger.info(f"Downsampling snow data for visualization: {snow_data.shape} -> {snow_viz.shape} (stride={stride})")
    (output_dir / "01_raw").mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "01_raw" / "snow_depth.png"
    visualize_snow_depth(snow_viz, output_path)

    return snow_stats


def run_step_score(output_dir: Path, dem: np.ndarray, snow_stats: dict, transform=None):
    """Run sledding score calculation and visualization step."""
    logger.info("\n" + "=" * 70)
    logger.info("Step 3: Sledding Score Calculation")
    logger.info("=" * 70)

    # Create Terrain object
    if transform is None:
        from rasterio.transform import Affine

        transform = Affine.identity()

    terrain = Terrain(dem, transform, dem_crs="EPSG:4326")

    # Import scoring system
    from src.scoring.configs.sledding import (
        DEFAULT_SLEDDING_SCORER,
        compute_derived_inputs,
    )

    # Calculate slope statistics from DEM (with caching)
    target_shape = snow_stats["median_max_depth"].shape
    slope_stats = compute_cached_slope_statistics(
        dem=dem,
        dem_transform=transform,
        dem_crs="EPSG:4326",
        target_shape=target_shape,
        target_transform=transform,
        target_crs="EPSG:4326",
        cache_dir=Path(".slope_cache"),
    )

    logger.info(f"Slope stats computed: mean={np.mean(slope_stats.slope_mean):.1f}°, "
               f"max={np.max(slope_stats.slope_max):.1f}°, "
               f"p95={np.mean(slope_stats.slope_p95):.1f}°")

    # Prepare inputs for the scorer
    logger.info("Preparing scorer inputs...")
    inputs = compute_derived_inputs(slope_stats, snow_stats)

    # Compute the score using the configurable scorer
    logger.info(f"Computing score with {len(DEFAULT_SLEDDING_SCORER.components)} components...")
    sledding_score = DEFAULT_SLEDDING_SCORER.compute(inputs)

    # Get individual component scores for visualization/debugging
    component_scores = DEFAULT_SLEDDING_SCORER.get_component_scores(inputs)

    # Mask invalid values
    sledding_score = np.ma.masked_invalid(sledding_score)

    logger.info("Sledding score stats:")
    logger.info(f"  Range: {np.ma.min(sledding_score):.3f} to {np.ma.max(sledding_score):.3f}")
    logger.info(f"  Mean: {np.ma.mean(sledding_score):.3f}")
    high_score_pixels = np.ma.sum(sledding_score > 0.7)
    logger.info(
        f"  Pixels > 0.7: {high_score_pixels} "
        f"({high_score_pixels / sledding_score.count() * 100:.1f}%)"
    )

    # Add sledding score to terrain
    terrain.add_data_layer(
        "sledding_score",
        sledding_score,
        transform,
        "EPSG:4326",
        target_layer="dem",
    )

    logger.info(f"Terrain layers: {list(terrain.data_layers.keys())}")

    # Visualize sledding score (downsample if needed for memory efficiency)
    score_viz, stride = downsample_for_viz(sledding_score)
    if stride > 1:
        logger.info(f"Downsampling score for visualization: {sledding_score.shape} -> {score_viz.shape} (stride={stride})")
    (output_dir / "05_final").mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "05_final" / "sledding_score.png"
    visualize_sledding_score(score_viz, output_path)

    # Save histogram of sledding scores
    histogram_path = output_dir / "05_final" / "sledding_score_histogram.png"
    save_sledding_score_histogram(score_viz, histogram_path)

    # Save percentile map of sledding scores
    percentile_path = output_dir / "05_final" / "sledding_score_percentiles.png"
    save_sledding_score_percentiles(score_viz, percentile_path)

    # Save filtered map showing only excellent locations (score > 0.7)
    filtered_path = output_dir / "05_final" / "sledding_score_excellent.png"
    save_sledding_score_filtered(score_viz, filtered_path, threshold=0.7)

    # Downsample snow_stats arrays to match
    if stride > 1:
        snow_stats_viz = {
            k: v[::stride, ::stride] if hasattr(v, '__getitem__') and len(v.shape) == 2 else v
            for k, v in snow_stats.items()
        }
    else:
        snow_stats_viz = snow_stats

    # Visualize slope statistics in detail
    if slope_stats is not None:
        logger.info("Generating slope statistics visualizations...")

        # Downsample slope stats for visualization
        slope_stats_viz = slope_stats
        if stride > 1:
            # Create downsampled version of slope stats
            from dataclasses import replace
            slope_stats_viz = replace(
                slope_stats_viz,
                slope_mean=slope_stats_viz.slope_mean[::stride, ::stride],
                slope_max=slope_stats_viz.slope_max[::stride, ::stride],
                slope_min=slope_stats_viz.slope_min[::stride, ::stride],
                slope_std=slope_stats_viz.slope_std[::stride, ::stride],
                slope_p95=slope_stats_viz.slope_p95[::stride, ::stride],
                roughness=slope_stats_viz.roughness[::stride, ::stride],
                aspect_sin=slope_stats_viz.aspect_sin[::stride, ::stride],
                aspect_cos=slope_stats_viz.aspect_cos[::stride, ::stride],
            )

        # Save individual slope statistics panels
        slope_stats_dir = output_dir / "02_slope_stats"
        save_slope_stat_panels(slope_stats_viz, slope_stats_dir)

        # Save individual slope penalty panels
        penalty_dir = output_dir / "03_slope_penalties"
        save_slope_penalty_panels(slope_stats_viz, penalty_dir)

        # Save individual score component panels
        component_dir = output_dir / "04_score_components"
        save_score_component_panels(
            slope_stats=slope_stats_viz,
            snow_stats=snow_stats_viz,
            final_score=score_viz,
            output_dir=component_dir,
        )
    else:
        logger.warning("No slope_stats available - skipping slope statistics visualizations")

    return terrain, sledding_score


def run_step_render(output_dir: Path, terrain: Terrain):
    """Run 3D rendering step."""
    logger.info("\n" + "=" * 70)
    logger.info("Step 4: 3D Rendering")
    logger.info("=" * 70)

    if not BLENDER_AVAILABLE:
        logger.warning("Blender not available - skipping 3D render")
        logger.info("Install Blender Python module (bpy) to enable 3D rendering")
        return

    # Set color mapping to blend sledding score
    from src.terrain.core import elevation_colormap

    def blend_elevation_and_sledding(dem_data, sledding_score_data):
        """Blend elevation colors with sledding score."""
        dem_colors = elevation_colormap(dem_data, cmap_name="viridis")

        # Use sledding score to modulate saturation
        # High score = more vibrant, low score = more muted
        dem_colors[:, :, :3] = dem_colors[:, :, :3] * (0.5 + 0.5 * sledding_score_data[:, :, np.newaxis])

        return dem_colors

    terrain.set_color_mapping(blend_elevation_and_sledding, source_layers=["dem", "sledding_score"])

    # Render
    (output_dir / "06_render").mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "06_render" / "render_3d.png"
    render_3d_with_snow(terrain, output_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Detroit Snow & Sledding Analysis Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run DEM visualization with mock data
  python examples/detroit_snow_sledding.py --step dem --mock-data

  # Run all steps with real data
  python examples/detroit_snow_sledding.py --all-steps

  # Run specific step with custom output directory
  python examples/detroit_snow_sledding.py --output-dir ./my_outputs --step score
        """,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/images"),
        help="Output directory for visualizations (default: docs/images/)",
    )

    parser.add_argument(
        "--step",
        choices=["dem", "snow", "score", "render"],
        help="Run a specific step (dem, snow, score, render)",
    )

    parser.add_argument(
        "--all-steps", action="store_true", help="Run all visualization steps"
    )

    parser.add_argument(
        "--mock-data",
        action="store_true",
        help="Use mock data instead of loading real DEM/SNODAS data",
    )

    parser.add_argument(
        "--snodas-dir",
        type=Path,
        help="Path to SNODAS data directory (for real snow data)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.step and not args.all_steps:
        parser.error("Must specify either --step or --all-steps")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "=" * 70)
    logger.info("Detroit Snow & Sledding Analysis")
    logger.info("=" * 70)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Using mock data: {args.mock_data}")

    # State variables for multi-step execution
    dem = None
    transform = None
    snow_stats = None
    terrain = None

    # Run requested steps
    steps_to_run = []
    if args.all_steps:
        # All steps = visualization steps only (not 3D render)
        # Use --step render separately for 3D rendering
        steps_to_run = ["dem", "snow", "score"]
    else:
        steps_to_run = [args.step]

    for step in steps_to_run:
        if step == "dem":
            dem, transform = run_step_dem(args.output_dir, args.mock_data)

        elif step == "snow":
            if dem is None:
                dem, transform = run_step_dem(args.output_dir, args.mock_data)
            # Create terrain early for SNODAS processing
            if terrain is None and not args.mock_data:
                from rasterio.transform import Affine
                if transform is None:
                    transform = Affine.identity()
                terrain = Terrain(dem, transform, dem_crs="EPSG:4326")
            snow_stats = run_step_snow(args.output_dir, dem, args.mock_data, terrain, args.snodas_dir)

        elif step == "score":
            if dem is None:
                dem, transform = run_step_dem(args.output_dir, args.mock_data)
            if snow_stats is None:
                # Create terrain early for SNODAS processing
                if terrain is None and not args.mock_data:
                    from rasterio.transform import Affine
                    if transform is None:
                        transform = Affine.identity()
                    terrain = Terrain(dem, transform, dem_crs="EPSG:4326")
                snow_stats = run_step_snow(args.output_dir, dem, args.mock_data, terrain, args.snodas_dir)
            terrain, _ = run_step_score(args.output_dir, dem, snow_stats, transform)

        elif step == "render":
            if dem is None:
                dem, transform = run_step_dem(args.output_dir, args.mock_data)
            if snow_stats is None:
                # Create terrain early for SNODAS processing
                if terrain is None and not args.mock_data:
                    from rasterio.transform import Affine
                    if transform is None:
                        transform = Affine.identity()
                    terrain = Terrain(dem, transform, dem_crs="EPSG:4326")
                snow_stats = run_step_snow(args.output_dir, dem, args.mock_data, terrain, args.snodas_dir)
            # Ensure sledding_score layer exists (required for 3D rendering)
            if terrain is None or "sledding_score" not in terrain.data_layers:
                terrain, _ = run_step_score(args.output_dir, dem, snow_stats, transform)
            run_step_render(args.output_dir, terrain)

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
