"""
Data loading operations for terrain processing.

This module contains functions for loading and merging DEM (Digital Elevation Model)
files from various sources.
"""

import logging
from pathlib import Path
import numpy as np
import rasterio
from rasterio.merge import merge
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_dem_files(
    directory_path: str, pattern: str = "*.hgt", recursive: bool = False
) -> tuple[np.ndarray, rasterio.Affine]:
    """
    Load and merge DEM files from a directory into a single elevation dataset.
    Supports any raster format readable by rasterio (HGT, GeoTIFF, etc.).

    Args:
        directory_path: Path to directory containing DEM files
        pattern: File pattern to match (default: "*.hgt")
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
