"""
Water body detection and identification from elevation data.

Provides functions to identify water bodies from DEM data using slope analysis.
Water is characterized by flat surfaces (low slope), while terrain typically has
higher slope values.
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import generic_filter
import logging

logger = logging.getLogger(__name__)


def identify_water_by_slope(dem_data, slope_threshold=0.1, fill_holes=True):
    """
    Identify water bodies by detecting flat areas (low slope).

    Water bodies typically have very flat surfaces with near-zero slope.
    This function calculates local slope using Horn's method and identifies pixels
    below the threshold as potential water. Optionally applies morphological
    operations to fill small gaps and smooth the water mask.

    Args:
        dem_data (np.ndarray): Digital elevation model as 2D array (height values)
        slope_threshold (float): Maximum slope magnitude to classify as water.
                               Default: 0.1 (very flat surfaces)
                               Typical range: 0.05 to 0.5 depending on DEM resolution
                               Values are gradient magnitude from Horn's method
        fill_holes (bool): Apply morphological operations to fill small gaps
                          in water mask and smooth boundaries. Default: True

    Returns:
        np.ndarray: Boolean mask (dtype=bool) where True = water, False = land.
                   Same shape as dem_data.

    Raises:
        ValueError: If dem_data is not 2D or slope_threshold is negative

    Examples:
        >>> dem = np.array([[100, 100, 110], [100, 100, 110], [110, 110, 120]])
        >>> water_mask = identify_water_by_slope(dem, slope_threshold=0.1)
        >>> water_mask.dtype
        dtype('bool')
        >>> water_mask.shape
        (3, 3)
    """
    if dem_data.ndim != 2:
        raise ValueError(f"DEM data must be 2D, got shape {dem_data.shape}")

    if slope_threshold < 0:
        raise ValueError(f"slope_threshold must be non-negative, got {slope_threshold}")

    logger.info(f"Identifying water by slope (threshold: {slope_threshold})")

    # Create a working copy, handling NaN values
    dem_work = np.array(dem_data, dtype=np.float32)
    has_nan = np.isnan(dem_work)
    nan_count = np.sum(has_nan)

    if nan_count > 0:
        logger.debug(f"Found {nan_count} NaN values in DEM, will handle separately")

    # Calculate slope magnitude using Sobel operators
    # Slope is the gradient magnitude (rise over run in degrees)
    slope_mask = _calculate_slope(dem_work)

    # Identify flat areas as water
    water_mask = slope_mask < slope_threshold

    # Handle NaN values - mark them as not-water to preserve them
    if nan_count > 0:
        water_mask[has_nan] = False

    # Optionally smooth the mask
    if fill_holes:
        water_mask = _smooth_water_mask(water_mask)
        logger.debug("Applied morphological operations to smooth water mask")

    logger.info(f"Water detection complete: {np.sum(water_mask)} water pixels identified")

    return water_mask.astype(np.bool_)


def _calculate_slope(dem_data):
    """
    Calculate slope magnitude using Horn's method.

    Uses Horn's method with proper convolution kernels for more accurate slope
    computation on downsampled DEM data. This is more robust than Sobel for
    terrain slope analysis.

    Args:
        dem_data (np.ndarray): 2D elevation data

    Returns:
        np.ndarray: Slope magnitude (gradient magnitude), same shape as input
    """
    # Save the original NaN mask
    nan_mask = np.isnan(dem_data)

    # Fill NaN values with interpolation
    dem_filled = dem_data.copy()
    mask = np.isnan(dem_filled)
    if np.any(mask) and np.any(~mask):
        # Use linear interpolation for NaN values
        dem_filled[mask] = np.interp(
            np.flatnonzero(mask),
            np.flatnonzero(~mask),
            dem_filled[~mask]
        )
    elif np.all(mask):
        # All NaN - return all zeros
        return np.zeros_like(dem_data)

    # Calculate gradients using Horn's method
    # These kernels weight central differences more heavily
    dx = ndimage.convolve(
        dem_filled,
        np.array([[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]]) / 8.0
    )
    dy = ndimage.convolve(
        dem_filled,
        np.array([[-1, -2, -1],
                  [0, 0, 0],
                  [1, 2, 1]]) / 8.0
    )

    # Calculate slope magnitude (gradient magnitude)
    slope = np.hypot(dx, dy)

    # Restore NaN values to their original locations
    slope[nan_mask] = np.nan

    return slope


def _smooth_water_mask(water_mask, structure_size=3):
    """
    Smooth water mask using morphological operations.

    Applies closing (dilation then erosion) to fill small holes within water
    bodies and smooth boundaries. Preserves water regions while removing noise.

    Args:
        water_mask (np.ndarray): Boolean water mask
        structure_size (int): Size of morphological structuring element

    Returns:
        np.ndarray: Smoothed boolean water mask
    """
    # Create structuring element for morphological operations
    structure = ndimage.generate_binary_structure(2, 2)

    # Apply closing: dilation followed by erosion
    # This fills small holes within water bodies and smooths boundaries
    # while preserving the overall water extent
    smoothed = ndimage.binary_closing(water_mask, structure=structure, iterations=1)

    return smoothed
