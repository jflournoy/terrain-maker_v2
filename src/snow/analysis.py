# snow_analysis.py
import os
import numpy as np
import logging
import hashlib
import json
import gzip
import tempfile
import tarfile
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import requests
import urllib3
import rasterio
import rasterio.crs as rcrs
from affine import Affine
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds
from pathlib import Path
from datetime import datetime, timedelta
from tqdm.auto import tqdm

# For optional terrain feature calculations
try:
    import richdem as rd

    HAS_RICHDEM = True
except ImportError:
    HAS_RICHDEM = False
    rd = None
from scipy import ndimage
from matplotlib.colors import SymLogNorm

logger = logging.getLogger(__name__)


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


def _load_snodas_data(binary_path, header_path):
    """
    Load raw SNODAS .dat + .txt into a masked array + metadata.
    """
    logger.debug(f"Loading SNODAS from {binary_path}")
    binary_path = Path(binary_path)
    header_path = Path(header_path)
    uncompressed_binary_path = binary_path.with_suffix("")
    uncompressed_header_path = header_path.with_suffix("")

    loaded_uncompressed = False
    if binary_path.suffix == ".gz":
        if not uncompressed_binary_path.exists():
            uncompressed_binary_path = _gunzip_snodas_file(binary_path)
        else:
            loaded_uncompressed = True
    if header_path.suffix == ".gz":
        if not uncompressed_header_path.exists():
            uncompressed_header_path = _gunzip_snodas_file(header_path)
        else:
            loaded_uncompressed = True
    if loaded_uncompressed:
        logger.debug("Decompressed file exists, did not decompress.")

    meta = _read_snodas_header(uncompressed_header_path)
    data = _read_snodas_binary(uncompressed_binary_path, meta)
    return data, meta


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


class SnowAnalysis:
    """Handles processing and analysis of SNODAS snow data for terrain visualization,
    including optional terrain feature computations and SNODAS downloading."""

    def __init__(
        self,
        terrain=None,
        snodas_root_dir=None,
        cache_dir="snow_analysis_cache",
        resample_to_extent: bool = True,
    ):
        """
        Initialize snow analysis with optional Terrain and SNODAS data.

        Args:
            terrain: Optional Terrain object (with .dem_bounds, .data_layers, etc.)
            snodas_root_dir: Directory containing SNODAS data files
            cache_dir: Directory to store cached computations
        """
        self.terrain = terrain
        self.snodas_root_dir = Path(snodas_root_dir) if snodas_root_dir else None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.resample_to_extent = resample_to_extent

        # Storage for processed data
        self.processed_files = {}
        self.stats = None
        self.metadata = None
        self.sledding_score = None

    # -------------------------------------------------------------------------
    # Public API: SNODAS Processing & Statistics
    # -------------------------------------------------------------------------
    def set_terrain(self, terrain):
        """Set or update the terrain to analyze."""
        self.terrain = terrain

    def process_snow_data(self, force_reprocess=False):
        """
        Process SNODAS files and calculate snow statistics.

        Args:
            force_reprocess: Whether to force reprocessing cached data

        Returns:
            tuple: (stats, metadata, failed_files)
                   Snow statistics, metadata, and any files that failed.
        """
        if self.terrain is None:
            raise ValueError("No terrain provided. Call set_terrain() first.")

        if self.snodas_root_dir is None or not self.snodas_root_dir.exists():
            raise ValueError(f"SNODAS directory not found: {self.snodas_root_dir}")

        # Use bounding box from the Terrain
        extent = self.terrain.dem_bounds  # (minx, miny, maxx, maxy)
        logger.info(f"Using DEM extent from Terrain: {extent}")

        # Create cache key from extent & SNODAS path
        cache_key = hashlib.md5(f"{extent}:{self.snodas_root_dir}".encode()).hexdigest()

        cache_file = self.cache_dir / f"snow_analysis_{cache_key}.npz"
        failed_cache = self.cache_dir / f"snow_analysis_failed_{cache_key}.json"

        # Attempt to load from cache
        if not force_reprocess and cache_file.exists():
            logger.info(f"Loading snow analysis from cache: {cache_file}")
            with np.load(cache_file, allow_pickle=True) as npz:
                stats = {k: npz[k] for k in npz.files if k != "metadata"}
                metadata = npz["metadata"].item()

                # Load any failed files
                failed_files = []
                if failed_cache.exists():
                    with open(failed_cache) as f:
                        failed_files = json.load(f)

            self.stats = stats
            self.metadata = metadata
            # Optionally add stats arrays to Terrain
            self._store_snow_stats_in_terrain(stats, metadata)
            return stats, metadata, failed_files

        # Process SNODAS
        logger.info("Batch-processing SNODAS data...")
        processed_files = self.batch_process_snodas_data(extent)
        self.processed_files = processed_files

        # Calculate stats
        logger.info("Calculating snow statistics...")
        stats, metadata, failed_files = self.calculate_snow_statistics(processed_files)

        # Save to cache
        logger.info(f"Saving analysis results to cache: {cache_file}")
        np.savez_compressed(
            cache_file,
            metadata=metadata,
            **{k: v for k, v in stats.items() if isinstance(v, np.ndarray)},
        )
        with open(failed_cache, "w") as f:
            json.dump(failed_files, f)

        self.stats = stats
        self.metadata = metadata
        # Optionally store stats in Terrain
        self._store_snow_stats_in_terrain(stats, metadata)
        return stats, metadata, failed_files

    def calculate_sledding_score(self, stats=None, min_depth_mm=100, min_coverage=0.3):
        """
        Calculate a sledding suitability score combining depth and consistency metrics.

        Args:
            stats: Dictionary of snow statistics (defaults to self.stats).
            min_depth_mm: Minimum snow depth for sledding in mm (default: 100).
            min_coverage: Min proportion of snow days (default: 0.3 or 30%).

        Returns:
            numpy.ma.MaskedArray: Sledding score [0-1].
        """
        if stats is None:
            if self.stats is None:
                raise ValueError("No snow stats. Call process_snow_data() first.")
            stats = self.stats

        logger.info("Computing sledding potential score...")

        with np.errstate(divide="ignore", invalid="ignore"):
            depth_score = np.clip(stats["median_max_depth"] / (min_depth_mm * 2), 0, 1)
            coverage_score = np.clip(stats["mean_snow_day_ratio"] / min_coverage, 0, 1)
            variability_score = 1 - np.clip(
                (stats["interseason_cv"] + stats["mean_intraseason_cv"]) / 4, 0, 1
            )

        # Weighted combination
        sledding_score = 0.4 * depth_score + 0.4 * coverage_score + 0.2 * variability_score
        sledding_score = np.ma.masked_invalid(sledding_score)

        logger.info("Sledding score stats:")
        logger.info(f"  Range: {np.ma.min(sledding_score):.3f} to {np.ma.max(sledding_score):.3f}")
        logger.info(f"  Mean: {np.ma.mean(sledding_score):.3f}")
        high_score_pixels = np.ma.sum(sledding_score > 0.7)
        logger.info(
            f"  Pixels > 0.7: {high_score_pixels} "
            f"({high_score_pixels / sledding_score.count() * 100:.1f}%)"
        )

        # Store in terrain
        if self.terrain is not None:
            # SNODAS metadata is typically EPSG:4326, but let's confirm
            transform = self.metadata.get("transform", self.terrain.dem_transform)
            crs = self.metadata.get("crs", "EPSG:4326")
            # Add to Terrain, reprojecting to match the 'dem' layer
            self.terrain.add_data_layer(
                "sledding_score",
                sledding_score,
                transform,
                crs,
                target_layer="dem",  # ensures alignment with your DEM
            )

        self.sledding_score = sledding_score
        return sledding_score

    # -------------------------------------------------------------------------
    # Public API: DEM Feature Calculations (Optional)
    # -------------------------------------------------------------------------
    def calculate_tpi(self, dem_data, window_size=5):
        """
        Calculate Topographic Position Index (TPI): difference between
        a cell's elevation and the mean elevation of neighbors.
        """
        logger.info(f"Calculating TPI (window={window_size})...")
        kernel = np.ones((window_size, window_size), dtype=np.float32)
        kernel[window_size // 2, window_size // 2] = 0
        kernel /= kernel.sum()
        neighborhood_mean = ndimage.convolve(dem_data, kernel, mode="reflect")
        return dem_data - neighborhood_mean

    def calculate_roughness(self, dem_data, window_size=3):
        """
        Compute terrain roughness via local std dev in a window.
        """
        logger.info(f"Calculating roughness (window={window_size})...")
        mean_sq = ndimage.uniform_filter(dem_data**2, size=window_size)
        mean_elev = ndimage.uniform_filter(dem_data, size=window_size)
        variance = mean_sq - mean_elev**2
        variance = np.maximum(variance, 0)
        return np.sqrt(variance)

    def create_dem_features(
        self, dem_files, snodas_data, snodas_metadata, cache_dir="feature_cache"
    ):
        """
        (Optional) Merge DEM files, compute TPI/slope/curvature,
        and resample them to SNODAS grid. Then you could add them back
        to your Terrain if you wish.
        """
        logger.info("Creating DEM-based features & resampling to SNODAS resolution.")
        # (This logic is mostly identical to prior code blocks—omitted for brevity.)
        # The main difference is that after computing, you could do:
        #
        # self.terrain.add_data_layer('tpi', tpi_array, transform, crs, target_layer='dem')
        # etc.
        #
        # Return the feature dictionary if needed.
        pass

    # -------------------------------------------------------------------------
    # Public API: SNODAS Downloading (Optional)
    # -------------------------------------------------------------------------
    def download_snodas_data(
        self,
        start_date,
        end_date,
        output_dir=None,
        masked=False,
        max_retries=3,
        max_workers=4,
        rate_limit=1.0,
    ):
        """
        Download SNODAS data in parallel for the given date range
        (similar to snodas.py).
        """
        if output_dir is None:
            if self.snodas_root_dir is None:
                raise ValueError("Must provide output_dir if snodas_root_dir not set.")
            output_dir = self.snodas_root_dir

        # Silence InsecureRequestWarning
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        date_list = [
            start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)
        ]
        logger.info(
            f"Downloading SNODAS from {start_date.date()} to {end_date.date()}, "
            f"masked={masked}, saving to {output_dir}"
        )

        args_list = [(d, output_dir, masked, max_retries, rate_limit) for d in date_list]
        all_downloaded = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(self._download_single_date, arg): arg[0] for arg in args_list
            }
            with tqdm(total=len(date_list), desc="Downloading SNODAS") as pbar:
                for future in as_completed(future_map):
                    date = future_map[future]
                    try:
                        result = future.result()
                        all_downloaded.extend(result)
                    except Exception as e:
                        logger.error(f"Error downloading data for {date}: {e}")
                    pbar.update(1)

        logger.info(f"Downloaded {len(all_downloaded)} files total.")
        return all_downloaded

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    def visualize_snow_data(self, data_type="sledding_score", cmap="viridis", figsize=(12, 8)):
        """
        Simple visualization of a SNODAS-derived array (score or stats).
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import SymLogNorm

        if data_type == "sledding_score" and self.sledding_score is not None:
            data = self.sledding_score
            title = "Sledding Potential Score"
            colorbar_label = "Score (0-1)"
            norm = None
        elif data_type in (self.stats or {}):
            data = self.stats[data_type]
            title = f"Snow {data_type.replace('_', ' ').title()}"
            colorbar_label = self.metadata.get("data_units", "Value")
            norm = SymLogNorm(linthresh=1) if "depth" in data_type else None
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

        plt.figure(figsize=figsize)
        im = plt.imshow(data, cmap=cmap, norm=norm)
        plt.colorbar(im, label=colorbar_label)
        plt.title(title)
        plt.axis("off")
        plt.show()

    # -------------------------------------------------------------------------
    # Internal/Private
    # -------------------------------------------------------------------------
    def _store_snow_stats_in_terrain(self, stats, metadata):
        """
        Optionally store the SNODAS stats arrays in self.terrain,
        reprojecting to match the DEM so they're readily available.
        """
        if not self.terrain:
            return  # no terrain to store into
        logger.info("Adding snow statistic arrays to Terrain data_layers...")

        transform = metadata.get("transform", self.terrain.dem_transform)
        crs = metadata.get("crs", "EPSG:4326")

        # Only add array stats
        for key, arr in stats.items():
            if not isinstance(arr, np.ndarray):
                continue
            layer_name = f"snodas_{key}"  # e.g., "snodas_median_max_depth"
            logger.info(f"Storing {layer_name} in Terrain.")
            self.terrain.add_data_layer(
                layer_name, arr, transform, crs, target_layer="dem"  # ensures alignment
            )

    def batch_process_snodas_data(self, extent, processed_dir="processed_snodas", max_workers=14):
        """
        Process all SNODAS .dat.gz files in snodas_root_dir,
        cropping to 'extent' & reprojecting to ~1/120° (EPSG:4326).
        """
        processed_dir = Path(processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)

        if not self.snodas_root_dir:
            logger.error("No SNODAS root directory set.")
            return {}

        dat_files = list(self.snodas_root_dir.glob("**/snow_depth_*.dat.gz"))
        if not dat_files:
            logger.error(f"No SNODAS .dat.gz files found in {self.snodas_root_dir}")
            return {}

        logger.info(f"Found {len(dat_files)} SNODAS data files to process")

        # Snodas standard pixel size in geographic degrees
        pixel_size = 1.0 / 120.0
        minx, miny, maxx, maxy = extent
        target_width = int(round((maxx - minx) / pixel_size))
        target_height = int(round((maxy - miny) / pixel_size))
        target_transform = rasterio.transform.from_bounds(
            minx, miny, maxx, maxy, target_width, target_height
        )

        args_list = [
            (
                f,
                extent,
                processed_dir,
                (target_height, target_width, target_transform),
                self.resample_to_extent,
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
        return processed_files

    def calculate_snow_statistics(
        self, processed_files, snow_season_start_month=11, snow_season_end_month=4
    ):
        """
        Group processed SNODAS by winter season and compute aggregated stats.
        """
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
                    data, meta = self._load_processed_snodas(fpath)
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
                stats_dict["cv_depth"] = stats_dict["std_depth"] / stats_dict["mean_depth"]

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
        # Inter-season variability
        seasonal_means = np.ma.stack([s["mean_depth"] for s in seasonal_stats])
        with np.errstate(divide="ignore", invalid="ignore"):
            final_stats["interseason_cv"] = np.ma.std(seasonal_means, axis=0) / np.ma.mean(
                seasonal_means, axis=0
            )
        # Intra-season average CV
        final_stats["mean_intraseason_cv"] = np.ma.mean(
            [s["cv_depth"] for s in seasonal_stats], axis=0
        )

        return final_stats, metadata, failed_files

    def _load_processed_snodas(self, processed_file):
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

    def _download_single_date(self, args):
        """Helper for concurrent SNODAS downloads from NSIDC."""
        current_date, output_dir, masked, max_retries, rate_limit = args
        downloaded = []

        year_str = current_date.strftime("%Y")
        month_str = current_date.strftime("%m_%b")
        base_url = "https://noaadata.apps.nsidc.org/NOAA/G02158"
        data_type = "masked" if masked else "unmasked"

        date_dir = Path(output_dir) / year_str / month_str
        date_dir.mkdir(parents=True, exist_ok=True)

        date_str = current_date.strftime("%Y%m%d")
        snow_depth_path = date_dir / f"snow_depth_{date_str}.dat.gz"
        metadata_path = date_dir / f"snow_depth_metadata_{date_str}.txt.gz"

        if snow_depth_path.exists() and metadata_path.exists():
            return [snow_depth_path, metadata_path]

        tar_name = f"SNODAS_{'' if masked else 'unmasked_'}{date_str}.tar"
        url = f"{base_url}/{data_type}/{year_str}/{month_str}/{tar_name}"
        tar_path = date_dir / tar_name

        time.sleep(rate_limit)
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, verify=False, stream=True)
                resp.raise_for_status()

                with open(tar_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)

                with tarfile.open(tar_path) as tar:
                    snow_files = [
                        f
                        for f in tar.getnames()
                        if "1034" in f and (f.endswith(".dat.gz") or f.endswith(".txt.gz"))
                    ]
                    for sf in snow_files:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            tar.extract(sf, tmpdir)
                            extracted_path = Path(tmpdir) / sf
                            if sf.endswith(".dat.gz"):
                                extracted_path.rename(snow_depth_path)
                                downloaded.append(snow_depth_path)
                            elif sf.endswith(".txt.gz"):
                                extracted_path.rename(metadata_path)
                                downloaded.append(metadata_path)

                tar_path.unlink(missing_ok=True)
                break

            except Exception as e:
                logger.error(f"Error downloading {tar_name} for {current_date}: {e}")
                if tar_path.exists():
                    tar_path.unlink()
                if attempt == max_retries - 1:
                    logger.error(f"Failed after {max_retries} attempts")

        return downloaded
