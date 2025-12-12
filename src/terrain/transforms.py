"""
Raster transformation operations for terrain processing.

This module contains functions for transforming raster data including
downsampling, smoothing, flipping, and elevation scaling.
"""

import logging
import numpy as np
from scipy.ndimage import zoom
from scipy import ndimage
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio import Affine
import rasterio


def downsample_raster(zoom_factor=0.1, order=4, nodata_value=np.nan):
    """
    Create a raster downsampling transform function with specified parameters.

    Args:
        zoom_factor: Scaling factor for downsampling (default: 0.1)
        order: Interpolation order (default: 4)
        nodata_value: Value to treat as no data (default: np.nan)

    Returns:
        function: A transform function that downsamples raster data
    """
    logger = logging.getLogger(__name__)

    def transform(raster_data, transform=None):
        """
        Downsample raster data using scipy.ndimage.zoom

        Args:
            raster_data: Input raster numpy array
            transform: Optional affine transform

        Returns:
            tuple: (downsampled_data, new_transform, None)
        """
        logger.info(f"Downsampling raster by factor {zoom_factor}")

        # Mask out nodata values before downsampling
        mask = raster_data == nodata_value

        # Prepare data for downsampling
        processed_data = raster_data.copy()
        processed_data[mask] = np.nan

        # Downsample
        downsampled = zoom(
            processed_data, zoom=zoom_factor, order=order, prefilter=False, mode="reflect"
        )

        # Restore nodata values
        if np.any(mask):
            mask_downsampled = zoom(mask.astype(float), zoom=zoom_factor, order=0, prefilter=False)
            downsampled[mask_downsampled > 0.5] = nodata_value

        # Update transform if provided
        if transform is not None:
            # Scale the transform to match new resolution
            new_transform = transform * transform.scale(1 / zoom_factor)
        else:
            new_transform = None

        logger.info(f"Original shape: {raster_data.shape}")
        logger.info(f"Downsampled shape: {downsampled.shape}")
        logger.info(f"Value range: {np.nanmin(downsampled):.2f} to {np.nanmax(downsampled):.2f}")

        # Return 3-tuple for consistency with transform pipeline
        return downsampled, new_transform, None

    return transform


def smooth_raster(window_size=None, nodata_value=np.nan):
    """
    Create a raster smoothing transform function with specified parameters.

    Args:
        window_size: Size of median filter window
                    (defaults to 5% of smallest dimension if None)
        nodata_value: Value to treat as no data (default: np.nan)

    Returns:
        function: A transform function that smooths raster data
    """
    logger = logging.getLogger(__name__)

    def transform(raster_data, transform=None):
        """
        Apply median smoothing to raster data

        Args:
            raster_data: Input raster numpy array
            transform: Optional affine transform (unchanged by smoothing)

        Returns:
            tuple: (smoothed_data, transform, None)
        """
        logger.info("Applying median smoothing to raster")

        # Create a mask for nodata values
        mask = raster_data == nodata_value

        # Prepare data for smoothing
        processed_data = raster_data.copy()
        processed_data[mask] = np.nan

        # Calculate window size if not provided
        actual_window_size = window_size
        if actual_window_size is None:
            actual_window_size = int(np.floor(np.min(processed_data.shape) * 0.05))
            logger.info(f"Using calculated window size: {actual_window_size}")

        # Apply median filter
        smoothed = ndimage.median_filter(processed_data, size=actual_window_size)

        # Restore nodata values
        smoothed[mask] = nodata_value

        logger.info(f"Smoothing window size: {actual_window_size}")
        logger.info(
            f"Value range before smoothing: {np.nanmin(processed_data):.2f} to {np.nanmax(processed_data):.2f}"
        )
        logger.info(
            f"Value range after smoothing: {np.nanmin(smoothed):.2f} to {np.nanmax(smoothed):.2f}"
        )

        # Transform is unchanged by smoothing, return 3-tuple for consistency
        return smoothed, transform, None

    return transform


def flip_raster(axis="horizontal"):
    """
    Create a transform function that mirrors (flips) the DEM data.
    If axis='horizontal', it flips top ↔ bottom.
    (In terms of rows, row=0 becomes row=(height-1).)

    If axis='vertical', you could do left ↔ right (np.fliplr).
    """

    def transform_func(data, transform=None):
        """Flip array along specified axis and update transform if provided."""
        # 1) Flip the array in pixel space
        if axis == "horizontal":
            # Top <-> bottom
            new_data = np.flipud(data)
            flip_code = "horizontal"
        elif axis == "vertical":
            # Left <-> right
            new_data = np.fliplr(data)
            flip_code = "vertical"
        else:
            raise ValueError("axis must be 'horizontal' or 'vertical'.")

        if transform is None:
            # No transform to update, just return
            return (new_data, None, None)

        # 2) Original array shape
        old_height, old_width = data.shape

        # 3) If we keep the same shape after flip
        new_height, new_width = new_data.shape
        assert (
            new_height == old_height and new_width == old_width
        ), "Flip changed array size unexpectedly!"

        # 4) Update the affine transform

        # Typical georeferenced transform looks like:
        #   Affine(a, b, xoff,
        #          d, e, yoff)
        # For a north-up raster with no rotation:
        #   a = pixel_width, e = -pixel_height, b=d=0
        #   xoff, yoff = top-left corner in world coords
        #
        # Flipping top ↔ bottom effectively inverts rows:
        #   new_row = (height-1) - old_row
        #
        # That means:
        #   new_yoff = old_yoff + (height-1)*e
        #   new_e = -old_e    (so that row moves in the opposite direction)

        # GDAL order is: (xoff, a, b, yoff, d, e)
        # which maps to Affine(a, b, c, d, e, f) as: c=xoff, f=yoff
        xoff, a, b, yoff, d, e = transform.to_gdal()

        if flip_code == "horizontal":
            # top <-> bottom flip => invert "row" direction
            new_e = -e
            new_yoff = yoff + (old_height - 1) * e
            # Everything else remains the same
            new_transform = Affine(a, b, xoff, d, new_e, new_yoff)

        elif flip_code == "vertical":
            # left <-> right flip => invert "col" direction
            new_a = -a
            new_xoff = xoff + (old_width - 1) * a
            # Others remain the same
            new_transform = Affine(new_a, b, new_xoff, d, e, yoff)

        return (new_data, new_transform, None)

    return transform_func


def scale_elevation(scale_factor=1.0, nodata_value=np.nan):
    """
    Create a raster elevation scaling transform function.

    Multiplies all elevation values by the scale factor. Useful for reducing
    or amplifying terrain height without changing horizontal scale.

    Args:
        scale_factor (float): Multiplication factor for elevation values (default: 1.0)
        nodata_value: Value to treat as no data (default: np.nan)

    Returns:
        function: A transform function that scales elevation data
    """
    logger = logging.getLogger(__name__)

    def transform(raster_data, transform=None):
        """
        Scale elevation values in raster data.

        Args:
            raster_data: Input raster numpy array
            transform: Optional affine transform (unchanged by scaling)

        Returns:
            tuple: (scaled_data, transform, None)
        """
        logger.info(f"Scaling elevation by factor {scale_factor}")

        # Create output array
        scaled_data = raster_data.copy()

        # Mask out nodata values
        mask = raster_data == nodata_value

        # Scale the valid data
        scaled_data[~mask] = raster_data[~mask] * scale_factor

        # Preserve nodata values
        scaled_data[mask] = nodata_value

        logger.info(f"Original range: {np.nanmin(raster_data):.2f} to {np.nanmax(raster_data):.2f}")
        logger.info(f"Scaled range: {np.nanmin(scaled_data):.2f} to {np.nanmax(scaled_data):.2f}")

        # Transform is unchanged by scaling (affects Z only)
        return scaled_data, transform, None

    return transform


def reproject_raster(src_crs="EPSG:4326", dst_crs="EPSG:32617", nodata_value=np.nan, num_threads=4):
    """
    Generalized raster reprojection function

    Args:
        src_crs: Source coordinate reference system
        dst_crs: Destination coordinate reference system
        nodata_value: Value to use for areas outside original data
        num_threads: Number of threads for parallel processing

    Returns:
        Function that transforms data and returns (data, transform, new_crs)
    """

    def _reproject_raster(src_data, src_transform):
        logger = logging.getLogger(__name__)
        logger.info(f"Reprojecting raster from {src_crs} to {dst_crs}")

        with rasterio.Env(GDAL_NUM_THREADS=str(num_threads)):
            # Calculate transform and dimensions for destination CRS
            dst_transform, width, height = calculate_default_transform(
                src_crs,
                dst_crs,
                src_data.shape[1],
                src_data.shape[0],
                *rasterio.transform.array_bounds(
                    src_data.shape[0], src_data.shape[1], src_transform
                ),
            )

            # Create destination array
            dst_data = np.full((height, width), nodata_value, dtype=src_data.dtype)

            # Reproject with bilinear interpolation
            reproject(
                source=src_data,
                destination=dst_data,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                dst_nodata=nodata_value,
                num_threads=num_threads,
                warp_mem_limit=512,
            )

            logger.info(f"Reprojection complete. New shape: {dst_data.shape}")
            logger.info(f"Value range: {np.nanmin(dst_data):.2f} to {np.nanmax(dst_data):.2f}")

            # Return all three values
            return dst_data, dst_transform, dst_crs

    return _reproject_raster
