"""
SNODAS data processing utilities.

Functions for loading, processing, and computing statistics from
NOAA SNODAS (Snow Data Assimilation System) snow depth data.

Pipeline steps:
- batch_process_snodas_data: Load and reproject SNODAS files
- calculate_snow_statistics: Compute aggregated snow statistics
"""

import gzip
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import rasterio
import rasterio.crs as rcrs
from affine import Affine
from rasterio.warp import reproject, Resampling
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


# =============================================================================
# SNODAS File I/O Helpers
# =============================================================================


def _gunzip_snodas_file(gz_path, keep_original: bool = True) -> Path:
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


def _read_snodas_header(header_path) -> Dict[str, Any]:
    """Parse SNODAS header file to get transform, dimensions, etc."""
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


def _read_snodas_binary(binary_path, meta: Dict[str, Any]) -> np.ma.MaskedArray:
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


def _load_snodas_data(binary_path, header_path) -> Tuple[np.ma.MaskedArray, Dict]:
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


def _process_single_file(args) -> Optional[Tuple[datetime, Path]]:
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

            # 2) Reproject from SNODAS (geographic WGS84) -> target (also WGS84 here)
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

            # 4) SNODAS is always geographic WGS84 -> so we hard-code the CRS string
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

            # 1) Convert the native Affine -> 1x6 array
            native_transform = meta["transform"]  # rasterio.Affine
            transform_arr = np.array(list(native_transform), dtype=np.float64)

            # 2) Hard-code CRS = WGS84
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

        # Finally, write out the .npz with only numpy-friendly types
        np.savez_compressed(out_file, **save_kwargs)
        return (date, out_file)

    except Exception as e:
        logger.error(f"Error processing {dat_file}: {e}")
        return None


def _load_processed_snodas(processed_file) -> Tuple[np.ma.MaskedArray, Dict]:
    """Load processed .npz SNODAS and reconstruct spatial metadata."""
    with np.load(processed_file) as npz:
        # 1) Pull out the data (masked if necessary)
        data = npz["data"]
        if "no_data_value" in npz.files:
            data = np.ma.masked_equal(data, npz["no_data_value"])
        else:
            data = np.ma.masked_invalid(data)

        meta = {}

        # 2) Transform -> Affine
        if "transform" in npz.files:
            t_arr = npz["transform"]  # shape (6,)
            meta["transform"] = Affine(*t_arr.tolist())

        # 3) CRS -> rasterio.crs.CRS
        if "crs" in npz.files:
            crs_val = npz["crs"].item()  # byte-string or unicode
            if isinstance(crs_val, bytes):
                crs_str = crs_val.decode()
            else:
                crs_str = str(crs_val)
            meta["crs"] = rcrs.CRS.from_user_input(crs_str)

        # 4) Height/Width -> int
        if "height" in npz.files:
            meta["height"] = int(npz["height"])
        if "width" in npz.files:
            meta["width"] = int(npz["width"])

        # 5) Crop extent -> tuple of floats
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


def batch_process_snodas_data(
    snodas_dir,
    extent: Tuple[float, float, float, float],
    target_shape: Tuple[int, int],
    processed_dir: str = "processed_snodas",
    max_workers: int = 14,
) -> Dict[str, Any]:
    """
    Step 1: Load and process SNODAS files.

    Process all SNODAS .dat.gz files in snodas_dir,
    cropping to 'extent' & reprojecting to ~1/120 deg (EPSG:4326).

    Args:
        snodas_dir: Directory containing SNODAS .dat.gz files
        extent: (minx, miny, maxx, maxy) bounding box
        target_shape: (height, width) for output arrays
        processed_dir: Directory for processed .npz files
        max_workers: Number of parallel workers

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
    input_data: Dict[str, Any],
    snow_season_start_month: int = 11,
    snow_season_end_month: int = 4,
) -> Dict[str, np.ndarray]:
    """
    Step 2: Compute snow statistics from loaded files.

    Group processed SNODAS by winter season and compute aggregated stats.

    Args:
        input_data: Dict with "processed_files" from batch_process_snodas_data
        snow_season_start_month: First month of snow season (default: November)
        snow_season_end_month: Last month of snow season (default: April)

    Returns:
        Dict with final statistics:
        - median_max_depth: Median of seasonal max depths
        - mean_snow_day_ratio: Average fraction of days with snow
        - interseason_cv: Year-to-year variability
        - mean_intraseason_cv: Within-winter variability
        - metadata: Processing metadata
        - failed_files: List of files that failed to process
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
                    # first day -> bootstrap all accumulators
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
            # Replace NaN/Inf with 0 for areas with no snow (mean_depth ~ 0)
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
