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

    # Set descriptive name for logging
    transform.__name__ = f"downsample({zoom_factor:.3f})"
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

    # Set descriptive name for logging
    transform.__name__ = f"median_smooth(w={window_size or 'auto'})"
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

    # Set descriptive name for logging
    transform_func.__name__ = f"flip_{axis}"
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

    # Set descriptive name for logging
    transform.__name__ = f"scale_elev(×{scale_factor})"
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

    # Set descriptive name for logging
    _reproject_raster.__name__ = f"reproject({src_crs}→{dst_crs})"
    return _reproject_raster


def feature_preserving_smooth(sigma_spatial=3.0, sigma_intensity=None, nodata_value=np.nan):
    """
    Create a feature-preserving smoothing transform using bilateral filtering.

    Removes high-frequency noise while preserving ridges, valleys, and drainage patterns.
    Uses bilateral filtering: spatial Gaussian weighted by intensity similarity.

    Args:
        sigma_spatial: Spatial smoothing extent in pixels (default: 3.0).
            Larger = more smoothing. Typical range: 1-10 pixels.
        sigma_intensity: Intensity similarity threshold in elevation units.
            Larger = more smoothing across elevation differences.
            If None, auto-calculated as 5% of elevation range.
        nodata_value: Value to treat as no data (default: np.nan)

    Returns:
        function: Transform function compatible with terrain.add_transform()
    """
    logger = logging.getLogger(__name__)

    def transform(raster_data, transform=None):
        """
        Apply bilateral filtering to raster data.

        Args:
            raster_data: Input raster numpy array
            transform: Optional affine transform (unchanged by smoothing)

        Returns:
            tuple: (smoothed_data, transform, None)
        """
        logger.info(f"Applying feature-preserving smoothing (sigma_spatial={sigma_spatial})")

        # Create a mask for nodata values
        if np.isnan(nodata_value):
            mask = np.isnan(raster_data)
        else:
            mask = raster_data == nodata_value

        # Prepare data for smoothing (work with float64 for precision)
        processed_data = raster_data.astype(np.float64).copy()
        processed_data[mask] = np.nan

        # Calculate intensity sigma if not provided (5% of elevation range)
        actual_sigma_intensity = sigma_intensity
        if actual_sigma_intensity is None:
            valid_data = processed_data[~mask]
            if len(valid_data) > 0:
                elev_range = np.nanmax(valid_data) - np.nanmin(valid_data)
                actual_sigma_intensity = max(elev_range * 0.05, 1.0)
            else:
                actual_sigma_intensity = 10.0
            logger.info(f"Auto-calculated sigma_intensity: {actual_sigma_intensity:.2f}")

        # Apply bilateral filter
        smoothed = _bilateral_filter_2d(
            processed_data, sigma_spatial, actual_sigma_intensity, mask
        )

        # Restore nodata values
        smoothed[mask] = nodata_value

        logger.info(
            f"Value range before: {np.nanmin(processed_data):.2f} to {np.nanmax(processed_data):.2f}"
        )
        logger.info(
            f"Value range after: {np.nanmin(smoothed):.2f} to {np.nanmax(smoothed):.2f}"
        )

        # Transform is unchanged by smoothing, return 3-tuple for consistency
        return smoothed.astype(raster_data.dtype), transform, None

    # Set descriptive name for logging
    transform.__name__ = f"bilateral_smooth(σ={sigma_spatial})"
    return transform


def _bilateral_filter_2d(data, sigma_spatial, sigma_intensity, mask):
    """
    Apply bilateral filtering to 2D elevation data.

    Implementation priority (fastest first):
    1. OpenCV (~250M pixels/sec) - highly optimized SIMD implementation
    2. skimage (~0.8M pixels/sec) - good C implementation
    3. Pure numpy fallback (~1K pixels/sec) - last resort

    Args:
        data: 2D numpy array of elevation values
        sigma_spatial: Spatial Gaussian sigma in pixels
        sigma_intensity: Intensity Gaussian sigma in elevation units
        mask: Boolean mask where True = nodata

    Returns:
        Filtered 2D array
    """
    import time

    logger = logging.getLogger(__name__)
    height, width = data.shape
    n_pixels = height * width

    # Sanity check: cap sigma_spatial to avoid memory issues
    # OpenCV kernel diameter ≈ 6*sigma, so sigma=100 → ~600px kernel → 360K comparisons/pixel
    MAX_SIGMA_SPATIAL = 15.0  # Practical limit for bilateral filter
    if sigma_spatial > MAX_SIGMA_SPATIAL:
        logger.warning(
            f"sigma_spatial={sigma_spatial} is very large (max recommended: {MAX_SIGMA_SPATIAL}). "
            f"Capping to {MAX_SIGMA_SPATIAL} to avoid memory issues. "
            f"For stronger smoothing, apply multiple passes or downsample first."
        )
        sigma_spatial = MAX_SIGMA_SPATIAL

    logger.info(f"Bilateral filter: {height}×{width} = {n_pixels/1e6:.1f}M pixels")

    start_time = time.time()

    # Fill NaN values temporarily (neither OpenCV nor skimage handle NaN)
    data_filled = data.copy().astype(np.float32)
    if np.any(mask):
        valid_median = np.nanmedian(data[~mask])
        data_filled[mask] = valid_median

    # Try OpenCV first (500x faster than skimage)
    try:
        import cv2

        logger.info("  Using OpenCV (highly optimized)...")

        # OpenCV bilateral: d=-1 means auto-calculate diameter from sigmaSpace
        # Use explicit diameter for better control: d = ceil(6*sigma) | 1 (ensure odd)
        diameter = int(np.ceil(6 * sigma_spatial)) | 1
        logger.info(f"  Kernel diameter: {diameter}px")

        result = cv2.bilateralFilter(
            data_filled,
            d=diameter,
            sigmaColor=float(sigma_intensity),
            sigmaSpace=float(sigma_spatial),
        )

        # Restore NaN values
        result = result.astype(np.float64)
        result[mask] = np.nan

        elapsed = time.time() - start_time
        throughput = n_pixels / elapsed / 1e6
        logger.info(f"  ✓ Bilateral filter complete in {elapsed:.2f}s ({throughput:.1f}M pixels/sec)")

        return result

    except ImportError:
        pass  # Fall through to skimage
    except cv2.error as e:
        logger.warning(f"  OpenCV bilateral filter failed: {e}")
        logger.warning("  Falling back to skimage...")
    except MemoryError:
        logger.warning("  OpenCV ran out of memory, falling back to skimage...")

    # Try skimage as fallback
    try:
        from skimage.restoration import denoise_bilateral

        logger.info("  Using skimage (install opencv-python-headless for 500x speedup)...")

        result = denoise_bilateral(
            data_filled.astype(np.float64),
            sigma_color=sigma_intensity,
            sigma_spatial=sigma_spatial,
            mode="reflect",
            channel_axis=None,
        )

        result[mask] = np.nan

        elapsed = time.time() - start_time
        throughput = n_pixels / elapsed / 1e6
        logger.info(f"  ✓ Bilateral filter complete in {elapsed:.1f}s ({throughput:.1f}M pixels/sec)")

        return result

    except ImportError:
        logger.warning(
            "  Neither OpenCV nor skimage available, using slow fallback. "
            "Install opencv-python-headless for 500x speedup."
        )
        return _bilateral_filter_2d_fallback(
            data, sigma_spatial, sigma_intensity, mask, start_time
        )


def _bilateral_filter_2d_fallback(data, sigma_spatial, sigma_intensity, mask, start_time=None):
    """
    Pure-numpy bilateral filter fallback (slow but works without dependencies).

    This is O(H * W * K^2) and will be slow for large images.
    """
    import time

    logger = logging.getLogger(__name__)
    height, width = data.shape
    total_pixels = height * width

    if start_time is None:
        start_time = time.time()

    # Estimate: ~1000 pixels/sec for fallback
    estimated_time = total_pixels / 1000
    logger.warning(
        f"  ⚠ Slow fallback: {height}×{width} = {total_pixels:,} pixels. "
        f"Estimated ~{estimated_time/60:.0f} minutes!"
    )

    # Determine kernel size (3 * sigma covers 99.7% of Gaussian)
    kernel_radius = int(np.ceil(3 * sigma_spatial))
    kernel_size = 2 * kernel_radius + 1

    # Create spatial weight kernel (constant, compute once)
    y_offsets, x_offsets = np.ogrid[
        -kernel_radius : kernel_radius + 1, -kernel_radius : kernel_radius + 1
    ]
    spatial_weights = np.exp(-(x_offsets**2 + y_offsets**2) / (2 * sigma_spatial**2))

    # Pad data for boundary handling (reflect mode preserves edge values)
    padded = np.pad(data, kernel_radius, mode="reflect")
    padded_mask = np.pad(mask, kernel_radius, mode="constant", constant_values=True)

    # Output array
    result = np.zeros_like(data)

    # Process with progress logging and ETA
    log_interval = max(1, height // 10)
    for i in range(height):
        if i > 0 and i % log_interval == 0:
            elapsed = time.time() - start_time
            rows_per_sec = i / elapsed
            remaining_rows = height - i
            eta = remaining_rows / rows_per_sec if rows_per_sec > 0 else 0
            logger.info(
                f"  Progress: {i}/{height} rows ({100*i/height:.0f}%) - "
                f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s"
            )

        for j in range(width):
            if mask[i, j]:
                result[i, j] = np.nan
                continue

            # Extract neighborhood
            neighborhood = padded[i : i + kernel_size, j : j + kernel_size]
            neighborhood_mask = padded_mask[i : i + kernel_size, j : j + kernel_size]

            # Center value
            center_value = data[i, j]

            # Intensity weights based on difference from center
            intensity_diff = neighborhood - center_value
            intensity_weights = np.exp(
                -(intensity_diff**2) / (2 * sigma_intensity**2)
            )

            # Combined weights (spatial * intensity)
            combined_weights = spatial_weights * intensity_weights
            combined_weights[neighborhood_mask] = 0
            combined_weights[np.isnan(neighborhood)] = 0

            # Weighted average
            weight_sum = np.sum(combined_weights)
            if weight_sum > 0:
                valid_neighborhood = np.where(
                    np.isnan(neighborhood), 0, neighborhood
                )
                result[i, j] = np.sum(valid_neighborhood * combined_weights) / weight_sum
            else:
                result[i, j] = center_value

    elapsed = time.time() - start_time
    throughput = total_pixels / elapsed
    logger.info(
        f"  ✓ Bilateral filter complete in {elapsed:.1f}s "
        f"({throughput:.0f} pixels/sec)"
    )
    return result


def smooth_score_data(
    scores: np.ndarray,
    sigma_spatial: float = 3.0,
    sigma_intensity: float = None,
) -> np.ndarray:
    """
    Smooth score data using bilateral filtering.

    Applies feature-preserving smoothing to reduce blocky pixelation from
    low-resolution source data (e.g., SNODAS ~925m) when displayed on
    high-resolution terrain (~30m DEM).

    Uses bilateral filtering: smooths within similar-intensity regions while
    preserving edges between different score zones.

    Args:
        scores: 2D numpy array of score values (typically 0-1 range)
        sigma_spatial: Spatial smoothing extent in pixels (default: 3.0).
            Larger = more smoothing. Typical range: 1-10 pixels.
        sigma_intensity: Intensity similarity threshold in score units.
            Larger = more smoothing across score differences.
            If None, auto-calculated as 15% of score range (good for 0-1 data).

    Returns:
        Smoothed score array with same shape as input.
        NaN values are preserved. Output is clipped to [0, 1] range.

    Example:
        >>> # Smooth blocky SNODAS-derived scores
        >>> sledding_scores = load_score_grid("sledding_scores.npz")
        >>> smoothed = smooth_score_data(sledding_scores, sigma_spatial=5.0)
    """
    logger = logging.getLogger(__name__)

    # Create mask for NaN values
    mask = np.isnan(scores)

    # Calculate sigma_intensity if not provided
    # For score data (0-1 range), use ~15% of range for good edge preservation
    if sigma_intensity is None:
        valid_data = scores[~mask]
        if len(valid_data) > 0:
            score_range = np.nanmax(valid_data) - np.nanmin(valid_data)
            sigma_intensity = max(score_range * 0.15, 0.05)  # At least 0.05
        else:
            sigma_intensity = 0.15  # Default for 0-1 range
        logger.debug(f"Auto-calculated sigma_intensity: {sigma_intensity:.3f}")

    logger.info(
        f"Smoothing score data (sigma_spatial={sigma_spatial}, sigma_intensity={sigma_intensity:.3f})"
    )

    # Apply bilateral filter
    smoothed = _bilateral_filter_2d(
        scores.astype(np.float64),
        sigma_spatial=sigma_spatial,
        sigma_intensity=sigma_intensity,
        mask=mask,
    )

    # Clip to valid score range [0, 1] (bilateral can slightly exceed bounds)
    smoothed = np.clip(smoothed, 0.0, 1.0)

    # Restore NaN values
    smoothed[mask] = np.nan

    return smoothed


def despeckle_scores(
    scores: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """
    Remove isolated speckles from score data using median filtering.

    Unlike bilateral filtering which preserves edges, median filtering
    replaces each pixel with the median of its neighborhood. This effectively
    removes isolated outlier pixels (speckles) while preserving larger regions.

    Use case: SNODAS snow data upsampled to high-res DEM often has isolated
    low-score pixels (speckles) in otherwise high-score regions due to
    resolution mismatch. These appear as visual noise in the rendered terrain.

    Args:
        scores: 2D array of score values (typically 0-1 range)
        kernel_size: Size of median filter kernel (default: 3 for 3x3).
            Larger kernels remove larger speckle clusters but may affect
            legitimate small features. Common values: 3, 5, 7.

    Returns:
        Despeckled score array with same shape as input.
        NaN values are preserved.

    Example:
        >>> # Remove single-pixel speckles
        >>> despeckled = despeckle_scores(scores, kernel_size=3)
        >>> # Remove up to 2x2 speckle clusters
        >>> despeckled = despeckle_scores(scores, kernel_size=5)
    """
    from scipy.ndimage import median_filter

    logger = logging.getLogger(__name__)

    if scores.ndim != 2:
        raise ValueError(f"scores must be 2D, got {scores.ndim}D")

    # Handle NaN values - replace with 0 temporarily for filtering
    mask = np.isnan(scores)
    scores_filled = scores.copy()
    scores_filled[mask] = 0.0

    logger.info(f"Despeckle score data (kernel_size={kernel_size})")

    # Apply median filter
    despeckled = median_filter(scores_filled.astype(np.float64), size=kernel_size)

    # Clip to valid range (median should preserve range but ensure it)
    despeckled = np.clip(despeckled, 0.0, 1.0)

    # Restore NaN values
    despeckled[mask] = np.nan

    return despeckled.astype(np.float32)
