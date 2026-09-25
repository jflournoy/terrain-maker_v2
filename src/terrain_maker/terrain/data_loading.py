"""
Data loading operations for terrain processing.

This module contains functions for loading and merging DEM (Digital Elevation Model)
files from various sources.
"""

import logging
import zipfile
from pathlib import Path
import numpy as np
import rasterio
from rasterio.merge import merge
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _extract_dem_from_zips(directory: Path, pattern: str = "*.hgt") -> int:
    """
    Extract DEM files from ZIP archives if no matching files are found.

    NASADEM downloads come as ZIP files (e.g., NASADEM_HGT_N32W117.zip).
    This function automatically extracts matching files from ZIPs when needed.

    Args:
        directory: Directory containing ZIP files
        pattern: File pattern to extract (e.g., "*.hgt")

    Returns:
        Number of files extracted
    """
    # Check if matching files already exist
    glob_pattern = pattern
    existing_files = list(directory.glob(glob_pattern))
    if existing_files:
        return 0  # Files already exist, no extraction needed

    # Look for ZIP files
    zip_files = list(directory.glob("*.zip"))
    if not zip_files:
        return 0  # No ZIP files to extract

    logger.info(f"No {pattern} files found, extracting from {len(zip_files)} ZIP archives...")

    extracted_count = 0
    for zip_path in zip_files:
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Extract files matching the pattern
                # Convert glob pattern to simple extension check
                ext = pattern.lstrip("*")  # e.g., "*.hgt" -> ".hgt"

                matching_members = [
                    m for m in zf.namelist()
                    if m.lower().endswith(ext.lower())
                ]

                for member in matching_members:
                    zf.extract(member, directory)
                    extracted_count += 1
                    logger.debug(f"  Extracted {member} from {zip_path.name}")

        except zipfile.BadZipFile:
            logger.warning(f"Skipping invalid ZIP file: {zip_path.name}")
        except Exception as e:
            logger.warning(f"Failed to extract from {zip_path.name}: {e}")

    if extracted_count > 0:
        logger.info(f"Extracted {extracted_count} files from ZIP archives")

    return extracted_count


def load_dem_files(
    directory_path: str, pattern: str = "*.hgt", recursive: bool = False
) -> tuple[np.ndarray, rasterio.Affine]:
    """
    Load and merge DEM files from a directory into a single elevation dataset.
    Supports any raster format readable by rasterio (HGT, GeoTIFF, etc.).

    Automatically extracts files from ZIP archives if no matching files are found.
    This is useful for NASADEM downloads which come as ZIP files.

    Args:
        directory_path: Path to directory containing DEM files (or ZIP archives)
        pattern: File pattern to match (default: ``*.hgt``)
        recursive: Whether to search subdirectories recursively (default: False)

    Returns:
        tuple: (merged_dem, transform) where:
            - merged_dem: numpy array containing the merged elevation data
            - transform: affine transform mapping pixel to geographic coordinates

    Raises:
        ValueError: If no valid DEM files are found or directory doesn't exist
        OSError: If directory access fails or file reading fails
        rasterio.errors.RasterioIOError: If there are issues reading the DEM files
    """
    logger.info(f"Searching for DEM files matching '{pattern}' in: {directory_path}")

    try:
        directory = Path(directory_path)

        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory}")

        if not directory.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        # Extract from ZIP archives if no matching files exist
        _extract_dem_from_zips(directory, pattern)

        # Find all matching files
        glob_func = directory.rglob if recursive else directory.glob
        dem_files = sorted(glob_func(pattern))

        if not dem_files:
            raise ValueError(f"No files matching '{pattern}' found in {directory}")

        # Validate and open files
        dem_datasets = []
        with tqdm(dem_files, desc="Opening DEM files") as pbar:
            for file in pbar:
                try:
                    ds = rasterio.open(file)

                    # Basic validation
                    if ds.count == 0:
                        logger.warning(f"No raster bands found in {file}")
                        ds.close()
                        continue

                    if ds.dtypes[0] not in ("int16", "int32", "float32", "float64"):
                        logger.warning(f"Unexpected data type in {file}: {ds.dtypes[0]}")
                        ds.close()
                        continue

                    dem_datasets.append(ds)
                    pbar.set_postfix({"opened": len(dem_datasets)})

                except rasterio.errors.RasterioIOError as e:
                    logger.warning(f"Failed to open {file}: {str(e)}")
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error with {file}: {str(e)}")
                    continue

        if not dem_datasets:
            raise ValueError("No valid DEM files could be opened")

        logger.info(f"Successfully opened {len(dem_datasets)} DEM files")

        # Merge datasets
        try:
            with rasterio.Env():
                merged_dem, transform = merge(dem_datasets)

                # Extract first band - merge() returns 3D array (bands, height, width)
                merged_dem = merged_dem[0]

                logger.info(f"Successfully merged DEMs:")
                logger.info(f"  Output shape: {merged_dem.shape}")
                logger.info(
                    f"  Value range: {np.nanmin(merged_dem):.2f} to {np.nanmax(merged_dem):.2f}"
                )
                logger.info(f"  Transform: {transform}")

                return merged_dem, transform

        finally:
            # Clean up
            for ds in dem_datasets:
                ds.close()

    except Exception as e:
        logger.error(f"Error processing DEM files: {str(e)}")
        raise


def load_score_grid(
    file_path: Path,
    data_keys: list[str] = None,
) -> tuple[np.ndarray, rasterio.Affine | None]:
    """
    Load georeferenced raster data from an NPZ file.

    Works with any NPZ file containing a 2D array and optional Affine transform.
    Common use cases: score grids, classification maps, derived terrain products.

    The function searches for data arrays using the provided keys, falling back
    to common key names and finally to the first available array.

    Args:
        file_path: Path to .npz file
        data_keys: Keys to try for data array. If None, tries ["data", "score", "values"]
            then falls back to first array in file.

    Returns:
        Tuple of (data_array, transform) where:
            - data_array: 2D numpy array with the raster data
            - transform: Affine transform or None if not present in file

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file contains no arrays

    Example:
        >>> scores, transform = load_score_grid("path/to/scores.npz")
        >>> if transform:
        ...     terrain.add_data_layer("scores", scores, transform, "EPSG:4326")
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Score file not found: {file_path}")

    logger.info(f"Loading score grid from {file_path}")

    # Load NPZ file
    data = np.load(file_path)

    # Determine keys to search for data
    if data_keys is None:
        data_keys = ["data", "score", "values"]

    # Find the data array
    data_array = None
    for key in data_keys:
        if key in data:
            data_array = data[key]
            logger.debug(f"  Found data under key '{key}'")
            break

    # Fallback to first available array
    if data_array is None:
        available_keys = list(data.files)
        # Exclude 'transform' and other metadata keys
        array_keys = [k for k in available_keys if k not in ("transform", "crs")]
        if array_keys:
            first_key = array_keys[0]
            data_array = data[first_key]
            logger.debug(f"  Using first available array under key '{first_key}'")
        else:
            raise ValueError(f"No data arrays found in {file_path}")

    # Extract transform if present
    transform = None
    if "transform" in data:
        t = data["transform"]
        transform = rasterio.Affine(t[0], t[1], t[2], t[3], t[4], t[5])
        logger.debug(f"  Loaded transform: origin=({t[2]:.4f}, {t[5]:.4f})")

    logger.info(f"  Shape: {data_array.shape}, dtype: {data_array.dtype}")
    if transform:
        logger.info(f"  Transform: pixel size=({transform.a:.6f}, {transform.e:.6f})")
    else:
        logger.info("  No transform metadata (will need same_extent_as for alignment)")

    return data_array, transform


def save_score_grid(
    file_path: Path,
    data: np.ndarray,
    transform: "rasterio.Affine | None" = None,
    data_key: str = "data",
    **metadata,
) -> Path:
    """
    Save georeferenced raster data to an NPZ file.

    Creates an NPZ file compatible with load_score_grid(). The transform
    is stored as a 6-element array that can be reconstructed as an Affine.

    Args:
        file_path: Output path for .npz file
        data: 2D numpy array with raster data
        transform: Optional Affine transform for georeferencing
        data_key: Key name for the data array (default: "data")
        **metadata: Additional key=value pairs to store in the file

    Returns:
        Path to the saved file

    Example:
        >>> from rasterio import Affine
        >>> scores = compute_sledding_scores(dem)
        >>> transform = Affine.translation(-83.5, 42.5) * Affine.scale(0.01, -0.01)
        >>> save_score_grid("scores.npz", scores, transform, crs="EPSG:4326")

        >>> # Load it back
        >>> loaded_scores, loaded_transform = load_score_grid("scores.npz")
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Build save dict
    save_dict = {data_key: data}

    # Add transform as array
    if transform is not None:
        save_dict["transform"] = [
            transform.a, transform.b, transform.c,
            transform.d, transform.e, transform.f,
        ]

    # Add any additional metadata
    save_dict.update(metadata)

    # Save
    np.savez(file_path, **save_dict)

    logger.info(f"Saved score grid to {file_path}")
    logger.info(f"  Shape: {data.shape}, dtype: {data.dtype}")
    if transform:
        logger.info(f"  Transform: origin=({transform.c:.4f}, {transform.f:.4f})")

    return file_path


def find_score_file(
    name: str,
    search_dirs: list[Path] = None,
    subdirs: list[str] = None,
) -> Path | None:
    """
    Search for a score file in common locations.

    Useful for finding pre-computed score files that may be in various
    locations depending on how the pipeline was run.

    Args:
        name: Base filename to search for (e.g., "sledding_scores.npz")
        search_dirs: List of directories to search. Defaults to common locations.
        subdirs: Subdirectories to check within each search_dir
            (e.g., ["sledding", "xc_skiing"])

    Returns:
        Path to found file, or None if not found

    Example:
        >>> path = find_score_file("sledding_scores.npz",
        ...                         search_dirs=[Path("docs/images"), Path("output")],
        ...                         subdirs=["sledding", ""])
        >>> if path:
        ...     scores, transform = load_score_grid(path)
    """
    if search_dirs is None:
        search_dirs = [
            Path("docs/images"),
            Path("examples/output"),
            Path("output"),
            Path("."),
        ]

    if subdirs is None:
        subdirs = [""]  # Just search in the directory itself

    for search_dir in search_dirs:
        for subdir in subdirs:
            if subdir:
                check_path = search_dir / subdir / name
            else:
                check_path = search_dir / name

            if check_path.exists():
                logger.debug(f"Found score file: {check_path}")
                return check_path

    logger.debug(f"Score file not found: {name}")
    return None


# =============================================================================
# HGT FILE UTILITIES
# =============================================================================


def parse_hgt_filename(filename: str | Path) -> tuple[int | None, int | None]:
    """
    Parse SRTM HGT filename to extract latitude and longitude.

    Works globally with standard SRTM naming convention:
    - N42W083.hgt (Northern/Western hemisphere) -> lat=42, lon=-83
    - S15E028.hgt (Southern/Eastern hemisphere) -> lat=-15, lon=28

    Args:
        filename: HGT filename or Path (e.g., "N42W083.hgt" or Path("/path/to/N42W083.hgt"))

    Returns:
        Tuple of (latitude, longitude) as signed integers, or (None, None) if invalid
    """
    import re

    # Extract just the filename if it's a path
    name = Path(filename).stem.upper()

    # Match pattern: N/S followed by 2 digits, then E/W followed by 3 digits
    pattern = r'^([NS])(\d{2})([EW])(\d{3})$'
    match = re.match(pattern, name)

    if not match:
        return None, None

    ns, lat_str, ew, lon_str = match.groups()

    lat = int(lat_str)
    lon = int(lon_str)

    # Apply sign based on hemisphere
    if ns == 'S':
        lat = -lat
    if ew == 'W':
        lon = -lon

    return lat, lon


def load_filtered_hgt_files(
    dem_dir: Path | str,
    min_latitude: int = None,
    max_latitude: int = None,
    min_longitude: int = None,
    max_longitude: int = None,
    bbox: tuple = None,
    pattern: str = "*.hgt",
) -> tuple[np.ndarray, rasterio.Affine]:
    """
    Load SRTM HGT files filtered by latitude/longitude range.

    Works globally with standard SRTM naming convention. Filters files
    before loading to reduce memory usage for large DEM directories.

    Args:
        dem_dir: Directory containing HGT files
        min_latitude: Southern bound (e.g., -45 for S45, 42 for N42)
        max_latitude: Northern bound (e.g., 60 for N60)
        min_longitude: Western bound (e.g., -120 for W120)
        max_longitude: Eastern bound (e.g., 30 for E30)
        bbox: Bounding box as (west, south, east, north) tuple. If provided,
            overrides individual min/max parameters. Uses standard GIS convention:
            (min_lon, min_lat, max_lon, max_lat).
        pattern: File pattern to match (default: ``*.hgt``)

    Returns:
        Tuple of (merged_dem, transform)

    Raises:
        ValueError: If no matching files found after filtering

    Example:
        >>> # Load only tiles in Michigan area using individual params
        >>> dem, transform = load_filtered_hgt_files(
        ...     "/path/to/srtm",
        ...     min_latitude=41, max_latitude=47,
        ...     min_longitude=-90, max_longitude=-82
        ... )

        >>> # Same area using bbox (west, south, east, north)
        >>> dem, transform = load_filtered_hgt_files(
        ...     "/path/to/srtm",
        ...     bbox=(-90, 41, -82, 47)
        ... )

        >>> # Alps region (Switzerland/Austria)
        >>> dem, transform = load_filtered_hgt_files(
        ...     "/path/to/srtm",
        ...     bbox=(5, 45, 15, 48)
        ... )
    """
    # If bbox provided, unpack into individual params
    if bbox is not None:
        min_longitude, min_latitude, max_longitude, max_latitude = bbox
    dem_dir = Path(dem_dir)

    logger.info(f"Loading HGT files from {dem_dir}")
    if min_latitude is not None or max_latitude is not None:
        logger.info(f"  Latitude filter: [{min_latitude}, {max_latitude}]")
    if min_longitude is not None or max_longitude is not None:
        logger.info(f"  Longitude filter: [{min_longitude}, {max_longitude}]")

    # Find all matching files
    all_files = list(dem_dir.glob(pattern))

    if not all_files:
        raise ValueError(f"No files matching '{pattern}' found in {dem_dir}")

    # Filter by lat/lon if specified
    filtered_files = []
    for f in all_files:
        lat, lon = parse_hgt_filename(f)

        # Skip files that couldn't be parsed
        if lat is None or lon is None:
            continue

        # Apply filters
        if min_latitude is not None and lat < min_latitude:
            continue
        if max_latitude is not None and lat > max_latitude:
            continue
        if min_longitude is not None and lon < min_longitude:
            continue
        if max_longitude is not None and lon > max_longitude:
            continue

        filtered_files.append(f)

    if not filtered_files:
        raise ValueError(
            f"No HGT files found matching lat/lon filters in {dem_dir}"
        )

    logger.info(f"  Found {len(filtered_files)} files after filtering (from {len(all_files)} total)")

    # Load the filtered files directly using rasterio
    dem_datasets = []
    for file in filtered_files:
        try:
            ds = rasterio.open(file)
            if ds.count > 0:
                dem_datasets.append(ds)
        except rasterio.errors.RasterioIOError as e:
            logger.warning(f"Failed to open {file}: {str(e)}")
            continue

    if not dem_datasets:
        raise ValueError("No valid HGT files could be opened after filtering")

    try:
        with rasterio.Env():
            merged_dem, transform = merge(dem_datasets)
            merged_dem = merged_dem[0]  # Extract first band

            logger.info(f"  Merged {len(dem_datasets)} HGT files:")
            logger.info(f"    Shape: {merged_dem.shape}")
            logger.info(f"    Value range: {np.nanmin(merged_dem):.2f} to {np.nanmax(merged_dem):.2f}")

            return merged_dem, transform

    finally:
        for ds in dem_datasets:
            ds.close()


def load_geotiff_cropped_to_dem(
    geotiff_path: Path,
    dem_shape: tuple,
    dem_transform,
    dem_crs: str,
    use_windowed_read: bool = True,
) -> tuple:
    """
    Load a GeoTIFF file cropped to DEM's geographic bounds.

    This is a common pattern for loading auxiliary data (precipitation, land cover, etc.)
    that needs to be aligned with a DEM. The function:
    1. Crops to DEM's geographic bounds (via windowed reading if same CRS)
    2. Returns data with its transform for further processing

    Args:
        geotiff_path: Path to GeoTIFF file (e.g., precipitation, land cover)
        dem_shape: DEM shape (height, width) as tuple
        dem_transform: DEM's affine transform
        dem_crs: DEM's coordinate reference system (e.g., "EPSG:4326")
        use_windowed_read: If True and CRS match, use windowed reading for efficiency

    Returns:
        tuple: (data, transform, crs) where:
            - data: np.ndarray cropped to DEM bounds
            - transform: Affine transform for the cropped data
            - crs: Coordinate reference system

    Example:
        >>> precip_data, precip_transform, precip_crs = load_geotiff_cropped_to_dem(
        ...     precip_path, dem.shape, dem_transform, "EPSG:4326"
        ... )
        >>> # precip_data is now cropped to DEM's geographic bounds

    Notes:
        - If CRS match and windowed reading is enabled, only loads overlapping region (memory efficient)
        - If CRS differ, loads full file (caller should use rasterio.reproject)
        - Falls back to full read if windowed read fails
    """
    logger.info(f"Loading {geotiff_path.name} cropped to DEM bounds...")

    with rasterio.open(geotiff_path) as src:
        src_crs = src.crs

        # If CRS differ, can't use windowed reading - return full file
        if src_crs != dem_crs:
            logger.info(f"  CRS differ ({src_crs} vs {dem_crs}), loading full file for reprojection")
            data = src.read(1).astype(np.float32)
            return data, src.transform, src_crs

        # Same CRS - use windowed reading if enabled
        if use_windowed_read:
            try:
                from rasterio.windows import from_bounds
                from rasterio.transform import array_bounds

                # Get DEM bounds
                dem_bounds = array_bounds(dem_shape[0], dem_shape[1], dem_transform)

                # Calculate window that overlaps with DEM
                window = from_bounds(*dem_bounds, transform=src.transform)

                # Read windowed data
                data = src.read(1, window=window).astype(np.float32)
                transform = src.window_transform(window)

                logger.info(f"  ✓ Cropped {src.shape} → {data.shape} using windowed read")
                return data, transform, src_crs

            except Exception as e:
                logger.warning(f"  Windowed read failed ({e}), falling back to full read")

        # Fallback: load full file
        logger.info(f"  Loading full file ({src.shape})")
        data = src.read(1).astype(np.float32)
        return data, src.transform, src_crs
