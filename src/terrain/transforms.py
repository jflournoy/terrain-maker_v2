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


def downsample_raster(zoom_factor=0.1, method="average", nodata_value=np.nan):
    """
    Create a raster downsampling transform function with specified parameters.

    Args:
        zoom_factor: Scaling factor for downsampling (default: 0.1)
        method: Downsampling method (default: "average")
            - "average": Area averaging - best for DEMs, no overshoot
            - "lanczos": Lanczos resampling - sharp, minimal aliasing
            - "cubic": Cubic spline interpolation
            - "bilinear": Bilinear interpolation - safe fallback
        nodata_value: Value to treat as no data (default: np.nan)

    Returns:
        function: A transform function that downsamples raster data
    """
    logger = logging.getLogger(__name__)

    def transform(raster_data, transform=None):
        """
        Downsample raster data using the specified method.

        Args:
            raster_data: Input raster numpy array
            transform: Optional affine transform

        Returns:
            tuple: (downsampled_data, new_transform, None)
        """
        logger.info(f"Downsampling raster by factor {zoom_factor} using {method}")

        # Detect nodata mask (don't copy data yet)
        if np.isnan(nodata_value):
            mask = np.isnan(raster_data)
        else:
            mask = raster_data == nodata_value

        has_nodata = np.any(mask)

        # Calculate output shape
        out_shape = (
            max(1, int(raster_data.shape[0] * zoom_factor)),
            max(1, int(raster_data.shape[1] * zoom_factor)),
        )

        if method == "average":
            # Area averaging - best for DEMs, no upfront copy needed
            # PyTorch handles float conversion efficiently
            downsampled = _downsample_average(raster_data, out_shape, mask if has_nodata else None)
        elif method == "lanczos":
            # Lanczos resampling - need float processing
            # Prepare data only for this method
            processed_data = raster_data.astype(np.float64)
            if has_nodata:
                processed_data[mask] = np.nan
            downsampled = _downsample_lanczos(processed_data, out_shape)
        elif method == "cubic":
            # Cubic spline interpolation - need float processing
            processed_data = raster_data.astype(np.float64)
            if has_nodata:
                processed_data[mask] = np.nan
            downsampled = zoom(
                processed_data, zoom=zoom_factor, order=3, prefilter=True, mode="reflect"
            )
        elif method == "bilinear":
            # Bilinear interpolation - need float processing
            processed_data = raster_data.astype(np.float64)
            if has_nodata:
                processed_data[mask] = np.nan
            downsampled = zoom(
                processed_data, zoom=zoom_factor, order=1, prefilter=False, mode="reflect"
            )
        else:
            raise ValueError(f"Unknown downsampling method: {method}")

        # Restore nodata values (on much smaller downsampled data)
        if has_nodata and method != "average":
            # For non-average methods, resize mask on downsampled result (efficient)
            from skimage.transform import resize
            mask_downsampled = resize(
                mask.astype(float), downsampled.shape, order=0, preserve_range=True
            )
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
    transform.__name__ = f"downsample({zoom_factor:.3f}, {method})"
    return transform


def downsample_raster_optimized(zoom_factor=0.1, method="average", nodata_value=np.nan):
    """
    Create an optimized raster downsampling transform using two-pass for large compression.

    For large compression ratios (>100:1), uses two-pass downsampling to reduce memory
    bandwidth and improve cache efficiency. For smaller ratios, uses single-pass.

    Expected speedup: ~35x for billion-pixel DEMs (79s → ~2s).

    Args:
        zoom_factor: Scaling factor for downsampling (default: 0.1)
        method: Downsampling method (default: "average")
            - "average": Area averaging - best for DEMs, no overshoot
            - "lanczos": Lanczos resampling - sharp, minimal aliasing
            - "cubic": Cubic spline interpolation
            - "bilinear": Bilinear interpolation - safe fallback
        nodata_value: Value to treat as no data (default: np.nan)

    Returns:
        function: A transform function that downsamples raster data
    """
    logger = logging.getLogger(__name__)

    def transform(raster_data, transform=None):
        """
        Downsample raster data, using two-pass for large compression ratios.

        Args:
            raster_data: Input raster numpy array
            transform: Optional affine transform

        Returns:
            tuple: (downsampled_data, new_transform, None)
        """
        # Calculate compression ratio
        input_pixels = np.prod(raster_data.shape)
        output_pixels = input_pixels * (zoom_factor ** 2)
        compression_ratio = input_pixels / output_pixels

        logger.info(
            f"Downsampling {input_pixels:,} pixels to {output_pixels:,.0f} pixels "
            f"(compression {compression_ratio:.1f}:1)"
        )

        # Use two-pass for large compression ratios (>100:1)
        # Intermediate resolution: ~316:1 compression in first pass
        if compression_ratio > 100:
            logger.info("Using two-pass downsampling for large compression ratio")

            # First pass: compress by ~316x (zoom_factor ≈ 0.316 or 1/sqrt(10))
            zoom_first = 1 / np.sqrt(compression_ratio / 10)  # ~0.316 for 100:1 ratio
            zoom_first = min(zoom_first, 0.5)  # Cap at 50% to avoid too-large intermediate
            zoom_second = zoom_factor / zoom_first

            logger.info(f"Two-pass: {zoom_first:.3f} → {zoom_second:.3f}")

            # Get individual downsample functions
            downsample_first = downsample_raster(zoom_first, method, nodata_value)
            downsample_second = downsample_raster(zoom_second, method, nodata_value)

            # Apply first pass
            data_intermediate, transform_intermediate, _ = downsample_first(
                raster_data, transform
            )

            # Apply second pass
            result, final_transform, _ = downsample_second(
                data_intermediate, transform_intermediate
            )

            return result, final_transform, None
        else:
            # Single-pass for small compression ratios
            logger.info("Using single-pass downsampling")
            downsample = downsample_raster(zoom_factor, method, nodata_value)
            return downsample(raster_data, transform)

    # Set descriptive name for logging
    transform.__name__ = f"downsample_opt({zoom_factor:.3f}, {method})"
    return transform


def _downsample_average(data: np.ndarray, out_shape: tuple, mask: np.ndarray = None) -> np.ndarray:
    """
    Downsample using area averaging (mean of source pixels in each target cell).

    This is the most physically accurate method for DEMs - it preserves the
    mean elevation and doesn't create values outside the original range.

    Uses PyTorch's F.interpolate with mode='area' for true area averaging.
    PyTorch GPU is ~1.3x faster than CPU even for 1B pixel datasets due to
    efficient memory layout and optimized kernels.

    Args:
        data: Input raster array (any dtype, will be converted to float)
        out_shape: Target output shape (height, width)
        mask: Optional boolean mask where True = nodata (avoided in averaging)
    """
    import torch
    import torch.nn.functional as F

    # PyTorch requires C-contiguous arrays (no negative strides from flipud/fliplr)
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)

    # Convert to torch tensor: (H, W) -> (1, 1, H, W)
    tensor = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)

    # Handle nodata masking: temporarily replace with NaN for area averaging
    # (PyTorch's area mode ignores NaN values in averaging)
    if mask is not None:
        tensor[0, 0, mask] = float('nan')

    # Use GPU if available (faster even for very large datasets)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = tensor.to(device)

    # Area interpolation - true area averaging, O(n) regardless of scale
    result = F.interpolate(tensor, size=out_shape, mode="area")

    # Convert back to numpy
    downsampled = result.squeeze().cpu().numpy()

    # Restore nodata on downsampled result if mask was provided
    # Much faster: resize small downsampled mask instead of large original mask
    if mask is not None:
        from skimage.transform import resize
        mask_downsampled = resize(
            mask.astype(float), downsampled.shape, order=0, preserve_range=True
        )
        downsampled[mask_downsampled > 0.5] = np.nan

    return downsampled


def _downsample_lanczos(data: np.ndarray, out_shape: tuple) -> np.ndarray:
    """
    Downsample using Lanczos resampling (sharp with good anti-aliasing).

    Lanczos provides sharper results than area averaging while still
    handling aliasing well. Good for preserving terrain detail.
    """
    try:
        # Try PIL/Pillow first (best Lanczos implementation)
        from PIL import Image

        # Handle NaN by temporarily replacing with mean
        nan_mask = np.isnan(data)
        if np.any(nan_mask):
            data_filled = data.copy()
            data_filled[nan_mask] = np.nanmean(data)
        else:
            data_filled = data

        # Convert to PIL Image
        img = Image.fromarray(data_filled.astype(np.float32))
        # Resize with Lanczos
        resized = img.resize((out_shape[1], out_shape[0]), Image.Resampling.LANCZOS)
        result = np.array(resized, dtype=np.float64)

        # Restore NaN
        if np.any(nan_mask):
            mask_resized = np.array(
                Image.fromarray(nan_mask.astype(np.float32)).resize(
                    (out_shape[1], out_shape[0]), Image.Resampling.NEAREST
                )
            )
            result[mask_resized > 0.5] = np.nan

        return result

    except ImportError:
        # Fall back to skimage
        from skimage.transform import resize

        return resize(
            data,
            out_shape,
            order=3,  # Cubic as approximation
            anti_aliasing=True,
            preserve_range=True,
        )


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

    # Store metadata for algorithms that need to know the scale factor
    # This allows slope_adaptive_smooth to compensate for prior scaling
    transform._elevation_scale_factor = scale_factor

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
                num_threads=num_threads if num_threads > 0 else None,
                warp_mem_limit=2048,  # Increased from 512MB to 2GB for better performance
            )

            logger.info(f"Reprojection complete. New shape: {dst_data.shape}")
            logger.info(f"Value range: {np.nanmin(dst_data):.2f} to {np.nanmax(dst_data):.2f}")

            # Return all three values
            return dst_data, dst_transform, dst_crs

    # Set descriptive name for logging
    _reproject_raster.__name__ = f"reproject({src_crs}→{dst_crs})"
    return _reproject_raster


def cached_reproject(
    src_crs="EPSG:4326",
    dst_crs="EPSG:32617",
    cache_dir="data/cache/reprojected",
    nodata_value=np.nan,
    num_threads=0,  # 0 = auto-detect CPU cores (much faster on multi-core systems)
):
    """
    Cached raster reprojection - saves reprojected DEM to disk for instant reuse.

    First call computes and caches the reprojection. Subsequent calls load from
    cache (~0.5s). Cache is keyed by CRS pair and source data hash.

    Optimizations:
    - num_threads=0 (auto-detect): Uses all available CPU cores (much faster)
    - warp_mem_limit=2048MB: Increased from 512MB for better tiling and fewer passes

    Args:
        src_crs: Source coordinate reference system
        dst_crs: Destination coordinate reference system
        cache_dir: Directory to store cached reprojections
        nodata_value: Value to use for areas outside original data
        num_threads: Number of GDAL threads (0=auto-detect, 4 is default GDAL)

    Returns:
        Function that transforms data and returns (data, transform, new_crs)

    Example:
        >>> # First run: computes and caches
        >>> terrain.add_transform(cached_reproject(src_crs="EPSG:4326", dst_crs="EPSG:32617"))
        >>> # Second run: ~0.5s (loads from cache)
    """
    from pathlib import Path
    import hashlib

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    def _cached_reproject(src_data, src_transform):
        logger = logging.getLogger(__name__)

        # Create cache key from data shape, transform, and CRS
        data_hash = hashlib.md5(
            f"{src_data.shape}_{src_transform}_{src_crs}_{dst_crs}".encode()
        ).hexdigest()[:12]
        cache_file = cache_path / f"reproject_{src_crs.replace(':', '_')}_to_{dst_crs.replace(':', '_')}_{data_hash}.npz"
        meta_file = cache_path / f"reproject_{src_crs.replace(':', '_')}_to_{dst_crs.replace(':', '_')}_{data_hash}_meta.npz"

        # Try to load from cache
        if cache_file.exists() and meta_file.exists():
            logger.info(f"Loading cached reprojection from {cache_file.name}")
            cached = np.load(cache_file)
            meta = np.load(meta_file, allow_pickle=True)

            dst_data = cached["data"]
            dst_transform = Affine(*meta["transform"])

            logger.info(f"  Loaded {dst_data.shape} in cache hit (instant)")
            return dst_data, dst_transform, dst_crs

        # Cache miss - compute reprojection
        logger.info(f"Cache miss - reprojecting {src_crs} → {dst_crs} (will cache result)")

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
                num_threads=num_threads if num_threads > 0 else None,
                warp_mem_limit=2048,  # Increased from 512MB to 2GB for better performance
            )

        # Save to cache
        logger.info(f"Saving reprojection to cache: {cache_file.name}")
        np.savez_compressed(cache_file, data=dst_data)
        np.savez(meta_file, transform=list(dst_transform)[:6])

        logger.info(f"Reprojection complete. New shape: {dst_data.shape}")
        return dst_data, dst_transform, dst_crs

    _cached_reproject.__name__ = f"cached_reproject({src_crs}→{dst_crs})"
    return _cached_reproject


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
    Remove isolated speckles from score data using median filtering (GPU-accelerated).

    Unlike bilateral filtering which preserves edges, median filtering
    replaces each pixel with the median of its neighborhood. This effectively
    removes isolated outlier pixels (speckles) while preserving larger regions.

    Uses PyTorch GPU acceleration when available (5-10x speedup on CUDA).

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
    from src.terrain.gpu_ops import gpu_median_filter

    logger = logging.getLogger(__name__)

    if scores.ndim != 2:
        raise ValueError(f"scores must be 2D, got {scores.ndim}D")

    # Handle NaN values - replace with 0 temporarily for filtering
    mask = np.isnan(scores)
    scores_filled = scores.copy()
    scores_filled[mask] = 0.0

    logger.info(f"Despeckle score data (kernel_size={kernel_size})")

    # Apply median filter (GPU-accelerated)
    despeckled = gpu_median_filter(scores_filled.astype(np.float32), kernel_size=kernel_size)

    # Clip to valid range (median should preserve range but ensure it)
    despeckled = np.clip(despeckled, 0.0, 1.0)

    # Restore NaN values
    despeckled[mask] = np.nan

    return despeckled.astype(np.float32)


def wavelet_denoise_dem(
    nodata_value=np.nan,
    wavelet: str = "db4",
    levels: int = 3,
    threshold_sigma: float = 2.0,
    preserve_structure: bool = True,
):
    """
    Create a transform that removes noise while preserving terrain structure.

    Uses wavelet decomposition to separate terrain features (ridges, valleys,
    drainage patterns) from high-frequency noise. This is smarter than median
    filtering because it understands that terrain has structure at certain
    spatial frequencies.

    How it works:
    1. Decompose DEM into frequency bands using wavelets
    2. Estimate noise level from finest (highest-frequency) band
    3. Apply soft thresholding to remove coefficients below noise threshold
    4. Reconstruct DEM from cleaned coefficients

    The result preserves terrain structure while removing sensor noise, SRTM
    artifacts, and other high-frequency disturbances.

    Args:
        nodata_value: Value to treat as no data (default: np.nan)
        wavelet: Wavelet type (default: "db4" - Daubechies 4).
            Options: "db4" (smooth), "haar" (sharp edges), "sym4" (symmetric).
            - "db4": Best for natural terrain (smooth transitions)
            - "haar": Best for urban/artificial structures
            - "sym4": Good balance, symmetric filtering
        levels: Decomposition levels (default: 3). More levels = coarser
            structure preserved. Each level halves the resolution.
            - 2: Preserves finer detail, removes less noise
            - 3: Good balance (recommended)
            - 4: Aggressive smoothing, may blur small features
        threshold_sigma: Noise threshold multiplier (default: 2.0).
            Higher = more aggressive denoising.
            - 1.5: Light denoising, preserves more detail
            - 2.0: Standard denoising (recommended)
            - 3.0: Aggressive, may remove subtle features
        preserve_structure: If True, only denoise highest-frequency band
            to maximize structure preservation. If False, denoise all bands.

    Returns:
        function: A transform function for use with terrain.add_transform()

    Example:
        >>> # Standard terrain denoising
        >>> terrain.add_transform(wavelet_denoise_dem())

        >>> # Aggressive denoising for very noisy DEM
        >>> terrain.add_transform(wavelet_denoise_dem(threshold_sigma=3.0, levels=4))

        >>> # Light denoising, preserve maximum detail
        >>> terrain.add_transform(wavelet_denoise_dem(threshold_sigma=1.5, levels=2))
    """
    logger = logging.getLogger(__name__)

    def transform(raster_data, transform=None):
        """Apply wavelet denoising to DEM."""
        try:
            import pywt
        except ImportError:
            logger.warning(
                "PyWavelets (pywt) not installed. Install with: pip install PyWavelets"
            )
            logger.warning("Falling back to median filter despeckle")
            # Fallback to simple median filter
            from scipy.ndimage import median_filter
            return median_filter(raster_data, size=3), transform, None

        logger.info(
            f"Wavelet denoising DEM (wavelet={wavelet}, levels={levels}, "
            f"sigma={threshold_sigma}, preserve_structure={preserve_structure})"
        )

        # Handle nodata
        if np.isnan(nodata_value):
            mask = np.isnan(raster_data)
        else:
            mask = raster_data == nodata_value

        # Check if we have enough valid data to denoise
        valid_count = np.sum(~mask)
        total_count = mask.size
        valid_ratio = valid_count / total_count

        if valid_ratio < 0.1:
            logger.warning(
                f"Only {100*valid_ratio:.1f}% valid data ({valid_count}/{total_count} pixels). "
                "Skipping wavelet denoising."
            )
            return raster_data.copy(), transform, None

        logger.info(f"  Valid data: {100*valid_ratio:.1f}% ({valid_count} pixels)")

        # Fill nodata with local median for processing
        data = raster_data.copy().astype(np.float64)
        if np.any(mask):
            from scipy.ndimage import median_filter
            # Use valid data mean as fallback for areas where median is also NaN
            valid_mean = np.nanmean(raster_data)
            filled = median_filter(np.nan_to_num(data, nan=valid_mean), size=5)
            data[mask] = filled[mask]

        # Wavelet decomposition
        coeffs = pywt.wavedec2(data, wavelet, level=levels)

        # Estimate noise from finest detail coefficients (HH band)
        # MAD (median absolute deviation) is robust noise estimator
        detail_coeffs = coeffs[-1]  # Finest level: (cH, cV, cD)
        hh = detail_coeffs[2]  # Diagonal detail (usually noisiest)
        sigma = np.median(np.abs(hh)) / 0.6745  # MAD to sigma conversion

        logger.info(f"  Estimated noise sigma: {sigma:.4f}")

        # Apply soft thresholding
        threshold = threshold_sigma * sigma

        def soft_threshold(c, thresh):
            """Soft thresholding: shrink coefficients toward zero."""
            return np.sign(c) * np.maximum(np.abs(c) - thresh, 0)

        # Threshold detail coefficients
        denoised_coeffs = [coeffs[0]]  # Keep approximation (coarsest level)

        for i, (cH, cV, cD) in enumerate(coeffs[1:]):
            if preserve_structure and i < len(coeffs) - 2:
                # Only denoise finest level(s), preserve coarser structure
                denoised_coeffs.append((cH, cV, cD))
            else:
                # Apply thresholding
                denoised_coeffs.append((
                    soft_threshold(cH, threshold),
                    soft_threshold(cV, threshold),
                    soft_threshold(cD, threshold),
                ))

        # Reconstruct
        denoised = pywt.waverec2(denoised_coeffs, wavelet)

        # Trim to original size (wavelet reconstruction may add padding)
        denoised = denoised[:raster_data.shape[0], :raster_data.shape[1]]

        # Restore nodata
        denoised[mask] = nodata_value

        logger.info(
            f"  Value range: {np.nanmin(raster_data):.2f}→{np.nanmin(denoised):.2f} to "
            f"{np.nanmax(raster_data):.2f}→{np.nanmax(denoised):.2f}"
        )

        return denoised.astype(raster_data.dtype), transform, None

    transform.__name__ = f"wavelet_denoise(w={wavelet},L={levels},σ={threshold_sigma})"
    return transform


def despeckle_dem(nodata_value=np.nan, kernel_size: int = 3):
    """
    Create a transform that removes isolated elevation noise using median filtering (GPU-accelerated).

    Unlike bilateral smoothing (--smooth) which preserves edges but can look patchy,
    median filtering uniformly removes local outliers/speckles across the entire DEM.
    This is better for removing sensor noise or small DEM artifacts.

    Uses PyTorch GPU acceleration when available (5-10x speedup on CUDA).

    For smarter frequency-aware denoising that preserves terrain structure,
    use wavelet_denoise_dem() instead.

    Args:
        nodata_value: Value to treat as no data (default: np.nan)
        kernel_size: Size of median filter kernel (default: 3 for 3x3).
            Must be odd integer ≥3. Larger = more smoothing.
            - 3: Removes single-pixel noise (recommended)
            - 5: Removes 2x2 artifacts
            - 7: Stronger smoothing, may affect small terrain features

    Returns:
        function: A transform function for use with terrain.add_transform()

    Example:
        >>> terrain.add_transform(despeckle_dem(kernel_size=3))
    """
    logger = logging.getLogger(__name__)

    def transform(raster_data, transform=None):
        """
        Apply median filter to DEM to remove isolated speckles/noise.

        Args:
            raster_data: Input raster numpy array (elevation data)
            transform: Optional affine transform (unchanged by filtering)

        Returns:
            tuple: (despeckled_data, transform, None)
        """
        from src.terrain.gpu_ops import gpu_median_filter

        logger.info(f"Despeckle DEM (kernel_size={kernel_size})")

        # Create mask for nodata values
        if np.isnan(nodata_value):
            mask = np.isnan(raster_data)
        else:
            mask = raster_data == nodata_value

        # Fill nodata temporarily with median for filtering
        data_filled = raster_data.copy().astype(np.float32)
        if np.any(mask):
            valid_median = np.nanmedian(raster_data[~mask])
            data_filled[mask] = valid_median

        # Apply median filter (GPU-accelerated)
        despeckled = gpu_median_filter(data_filled, kernel_size=kernel_size)

        # Restore nodata values
        despeckled[mask] = nodata_value

        logger.info(
            f"Value range before: {np.nanmin(raster_data):.2f} to {np.nanmax(raster_data):.2f}"
        )
        logger.info(
            f"Value range after: {np.nanmin(despeckled):.2f} to {np.nanmax(despeckled):.2f}"
        )

        return despeckled.astype(raster_data.dtype), transform, None

    # Set descriptive name for logging
    transform.__name__ = f"despeckle_dem(k={kernel_size})"
    return transform


def remove_bumps(kernel_size: int = 3, structure: str = "disk", strength: float = 1.0):
    """
    Remove local maxima (bumps) from DEM using morphological opening.

    Morphological opening = erosion followed by dilation. This operation:
    - Removes small bright features (buildings, trees, noise)
    - Never creates new local maxima (mathematically guaranteed)
    - Preserves larger terrain features and overall shape
    - Leaves valleys and depressions untouched

    This is the standard approach for "removing buildings from DEMs" in
    geospatial processing.

    Args:
        kernel_size: Size of the structuring element (default: 3).
            Controls the maximum size of bumps to remove:
            - 1: Removes features up to ~2 pixels across (very subtle)
            - 3: Removes features up to ~6 pixels across
            - 5: Removes features up to ~10 pixels across
            For 30m DEMs, size=3 removes ~180m features
        structure: Shape of structuring element (default: "disk").
            - "disk": Circular, isotropic (recommended)
            - "square": Faster but may create artifacts on diagonals
        strength: Blend factor between original and opened result (default: 1.0).
            - 0.0: No effect (returns original)
            - 0.5: Half the bump removal effect (subtle)
            - 1.0: Full bump removal (original behavior)
            Values between 0 and 1 provide fine-grained control.

    Returns:
        function: A transform function for use with terrain.add_transform()

    Example:
        >>> # Remove small bumps (buildings on 30m DEM)
        >>> terrain.add_transform(remove_bumps(kernel_size=3))

        >>> # Subtle bump reduction (50% strength)
        >>> terrain.add_transform(remove_bumps(kernel_size=1, strength=0.5))

        >>> # More aggressive bump removal
        >>> terrain.add_transform(remove_bumps(kernel_size=5))
    """
    logger = logging.getLogger(__name__)

    def transform(raster_data, affine_transform=None):
        from scipy.ndimage import grey_opening
        from skimage.morphology import disk, square

        strength_str = f", strength={strength}" if strength < 1.0 else ""
        logger.info(f"Removing bumps via morphological opening (kernel={kernel_size}, structure={structure}{strength_str})")

        # Handle nodata
        if np.issubdtype(raster_data.dtype, np.floating):
            mask = np.isnan(raster_data)
            nodata_value = np.nan
        else:
            # For integer types, use min value as nodata indicator
            mask = np.zeros(raster_data.dtype, dtype=bool)
            nodata_value = None

        data = raster_data.astype(np.float64).copy()

        # Fill nodata temporarily with local median
        if np.any(mask):
            valid_median = np.nanmedian(data[~mask])
            data[mask] = valid_median

        # Create structuring element
        if structure == "disk":
            selem = disk(kernel_size)
        else:
            selem = square(kernel_size)

        # Morphological opening: erosion then dilation
        # - Erosion shrinks bright regions, removing small peaks
        # - Dilation restores shape, but removed peaks stay removed
        opened = grey_opening(data, footprint=selem)

        # Apply strength blending: result = original * (1-strength) + opened * strength
        # This gives continuous control from no effect (0) to full effect (1)
        if strength < 1.0:
            result = data * (1.0 - strength) + opened * strength
        else:
            result = opened

        # Compute statistics (based on actual effect after blending)
        diff = data - result
        bumps_removed = np.sum(diff > 0.1)  # Pixels lowered by >0.1m
        max_reduction = np.max(diff)
        mean_reduction = np.mean(diff[diff > 0]) if np.any(diff > 0) else 0

        logger.info(f"  Bumps affected: {bumps_removed:,} pixels")
        logger.info(f"  Max height reduction: {max_reduction:.2f}m")
        logger.info(f"  Mean reduction (where applied): {mean_reduction:.2f}m")

        # Restore nodata
        if nodata_value is not None and np.any(mask):
            result[mask] = nodata_value

        return result.astype(raster_data.dtype), affine_transform, None

    transform.__name__ = f"remove_bumps(k={kernel_size},{structure},s={strength})"
    return transform


def slope_adaptive_smooth(
    slope_threshold: float = 2.0,
    smooth_sigma: float = 5.0,
    transition_width: float = 1.0,
    nodata_value=np.nan,
    elevation_scale: float = 1.0,
    edge_threshold: float | None = None,
    edge_window: int = 5,
    strength: float = 1.0,
):
    """
    Create a transform that smooths flat areas more aggressively than hilly areas.

    This addresses the problem of buildings/structures appearing as bumps in
    flat regions. Flat areas get strong Gaussian smoothing to remove these
    artifacts, while slopes and hills are preserved with minimal smoothing.

    How it works:

    1. Compute local slope at each pixel using gradient magnitude
    2. Create a smooth weight mask: 1.0 where flat, 0.0 where steep
    3. Apply Gaussian blur to entire DEM
    4. Blend: output = original * (1-weight) + smoothed * weight

    The transition from "flat" to "steep" is smooth (using sigmoid) to avoid
    visible boundaries in the output.

    Args:
        slope_threshold: Slope angle in degrees below which terrain is
            considered "flat" (default: 2.0 degrees).
            - 1.0°: Very aggressive, only smooths nearly horizontal areas
            - 2.0°: Good default, smooths typical flat areas with buildings
            - 5.0°: Smooths gentle slopes too
        smooth_sigma: Gaussian blur sigma in pixels (default: 5.0).
            Controls the strength of smoothing in flat areas.
            - 3.0: Light smoothing
            - 5.0: Moderate smoothing (recommended)
            - 10.0: Very strong smoothing, may blur valid terrain features
        transition_width: Width of transition zone in degrees (default: 1.0).
            Controls how quickly smoothing fades off above threshold.
            - 0.5: Sharp transition
            - 1.0: Smooth transition (recommended)
            - 2.0: Very gradual transition
        nodata_value: Value to treat as no data (default: np.nan)
        elevation_scale: Scale factor that was applied to elevation data (default: 1.0).
            If elevation was scaled (e.g., by scale_elevation(0.0001)), pass that
            factor here so slope computation uses real-world elevation differences.
            The gradient is divided by this factor to recover true slopes.
        edge_threshold: Elevation difference threshold for edge preservation (default: None).
            If set, sharp elevation discontinuities (like lake boundaries) are preserved.
            Areas where local elevation range exceeds this threshold are not smoothed.
            - None: Disabled (original behavior)
            - 5.0: Preserve edges with >5m elevation change
            - 10.0: Only preserve very sharp edges (>10m change)
            Recommended: 3-10m depending on terrain features to preserve.
        edge_window: Window size for edge detection (default: 5).
            Larger windows detect edges over broader areas but may over-protect.
        strength: Overall smoothing strength multiplier (default: 1.0).
            Scales the maximum smoothing effect in flat areas.
            - 1.0: Full smoothing (original behavior)
            - 0.5: Half the smoothing effect
            - 0.25: Gentle smoothing
            - 0.0: No smoothing (transform has no effect)

    Returns:
        function: A transform function for use with terrain.add_transform()

    Example:
        >>> # Standard: smooth flat areas with >2° slopes preserved
        >>> terrain.add_transform(slope_adaptive_smooth())

        >>> # Aggressive: smooth anything below 5° slope
        >>> terrain.add_transform(slope_adaptive_smooth(slope_threshold=5.0, smooth_sigma=8.0))

        >>> # Conservative: only smooth very flat areas, light blur
        >>> terrain.add_transform(slope_adaptive_smooth(slope_threshold=1.0, smooth_sigma=3.0))

        >>> # Compensate for prior scale_elevation(0.0001)
        >>> terrain.add_transform(slope_adaptive_smooth(elevation_scale=0.0001))

        >>> # Preserve lake boundaries and other sharp edges
        >>> terrain.add_transform(slope_adaptive_smooth(edge_threshold=5.0))

        >>> # Gentle smoothing (25% of full effect)
        >>> terrain.add_transform(slope_adaptive_smooth(strength=0.25))
    """
    logger = logging.getLogger(__name__)

    def transform(raster_data, affine_transform=None):
        """
        Apply slope-adaptive smoothing to DEM.

        Args:
            raster_data: Input raster numpy array (elevation data)
            affine_transform: Optional affine transform (unchanged by smoothing)

        Returns:
            tuple: (smoothed_data, affine_transform, None)
        """
        from scipy.ndimage import gaussian_filter

        strength_str = f", strength={strength}" if strength != 1.0 else ""
        logger.info(
            f"Slope-adaptive smoothing (threshold={slope_threshold}°, "
            f"sigma={smooth_sigma}, transition={transition_width}°{strength_str})"
        )

        # Handle nodata mask
        if np.isnan(nodata_value):
            mask = np.isnan(raster_data)
        else:
            mask = raster_data == nodata_value
        total_valid = np.sum(~mask)

        # Work with float64 for precision
        data = raster_data.astype(np.float64).copy()

        # Fill nodata temporarily for gradient calculation
        if np.any(mask):
            valid_median = np.nanmedian(raster_data[~mask])
            data[mask] = valid_median

        # Compute slope magnitude using Sobel gradients
        # Sobel gives dz/dx and dz/dy in elevation units per pixel
        dy = ndimage.sobel(data, axis=0, mode="reflect") / 8.0  # Sobel normalizes by 8
        dx = ndimage.sobel(data, axis=1, mode="reflect") / 8.0

        # Account for pixel size to get actual slope
        # Affine transform stores pixel dimensions: a=x_size, e=y_size (usually negative)
        if affine_transform is not None:
            pixel_size_x = abs(affine_transform.a)
            pixel_size_y = abs(affine_transform.e)
            # Scale gradients from per-pixel to per-meter
            dx = dx / pixel_size_x
            dy = dy / pixel_size_y
            logger.info(f"  Pixel size: {pixel_size_x:.1f}m x {pixel_size_y:.1f}m")
        else:
            logger.warning("  No affine transform - assuming 1m pixels (slopes may be wrong)")

        # Compensate for elevation scaling (if elevation was scaled by 0.0001, gradients are too)
        # Divide by elevation_scale to recover true gradient magnitude
        if elevation_scale != 1.0:
            dx = dx / elevation_scale
            dy = dy / elevation_scale
            logger.info(f"  Compensating for elevation_scale={elevation_scale}")

        # Gradient magnitude (rise/run in elevation per meter)
        gradient_magnitude = np.sqrt(dx**2 + dy**2)

        # Convert to slope angle in degrees
        slope_degrees = np.degrees(np.arctan(gradient_magnitude))

        logger.info(
            f"  Slope range: {np.nanmin(slope_degrees):.2f}° to "
            f"{np.nanmax(slope_degrees):.2f}°"
        )
        logger.info(
            f"  Median slope: {np.nanmedian(slope_degrees):.2f}°"
        )

        # Create smooth weight mask using sigmoid
        # weight = 1.0 where flat (slope < threshold), 0.0 where steep
        # Sigmoid: 1 / (1 + exp(k * (x - threshold)))
        # k controls steepness of transition
        k = 4.0 / transition_width  # Scale factor for desired transition width
        smoothing_weight = 1.0 / (1.0 + np.exp(k * (slope_degrees - slope_threshold)))

        # Edge preservation: detect sharp elevation discontinuities and protect them
        if edge_threshold is not None:
            from scipy.ndimage import maximum_filter, minimum_filter

            # Compute local elevation range (max - min in window)
            # This detects sharp edges like lake boundaries regardless of slope
            local_max = maximum_filter(data, size=edge_window, mode="reflect")
            local_min = minimum_filter(data, size=edge_window, mode="reflect")
            local_range = local_max - local_min

            # Account for elevation scaling if applied
            if elevation_scale != 1.0:
                # local_range is in scaled units, threshold is in real units
                effective_threshold = edge_threshold * elevation_scale
            else:
                effective_threshold = edge_threshold

            # Create edge mask: 1.0 near edges, 0.0 elsewhere
            # Use sigmoid for smooth transition
            edge_k = 4.0 / (effective_threshold * 0.5)  # Transition over half the threshold
            edge_weight = 1.0 / (1.0 + np.exp(-edge_k * (local_range - effective_threshold)))

            # Reduce smoothing weight near edges
            # edge_weight=1 means edge detected → smoothing_weight should be 0
            smoothing_weight = smoothing_weight * (1.0 - edge_weight)

            edge_pixels = np.sum(edge_weight > 0.5)
            logger.info(
                f"  Edge preservation: {100*edge_pixels/total_valid:.1f}% pixels near edges "
                f"(threshold={edge_threshold}m, window={edge_window})"
            )

        # Log weight distribution
        flat_pixels = np.sum(smoothing_weight > 0.9)
        steep_pixels = np.sum(smoothing_weight < 0.1)
        logger.info(
            f"  Pixels: {100*flat_pixels/total_valid:.1f}% flat (weight>0.9), "
            f"{100*steep_pixels/total_valid:.1f}% steep (weight<0.1)"
        )

        # Apply Gaussian smoothing to entire DEM
        smoothed = gaussian_filter(data, sigma=smooth_sigma, mode="reflect")

        # Apply strength multiplier to smoothing weight
        # strength=1.0: full effect, strength=0.5: half effect, strength=0.0: no effect
        if strength != 1.0:
            smoothing_weight = smoothing_weight * strength
            logger.info(f"  Strength={strength}: max weight scaled to {np.max(smoothing_weight):.2f}")

        # Blend: output = original * (1-weight) + smoothed * weight
        result = data * (1.0 - smoothing_weight) + smoothed * smoothing_weight

        # Restore nodata values
        result[mask] = nodata_value

        logger.info(
            f"  Value range before: {np.nanmin(raster_data):.2f} to "
            f"{np.nanmax(raster_data):.2f}"
        )
        logger.info(
            f"  Value range after: {np.nanmin(result):.2f} to "
            f"{np.nanmax(result):.2f}"
        )

        return result.astype(raster_data.dtype), affine_transform, None

    # Set descriptive name for logging
    edge_str = f",edge={edge_threshold}m" if edge_threshold is not None else ""
    strength_str = f",s={strength}" if strength != 1.0 else ""
    transform.__name__ = (
        f"slope_adaptive_smooth(θ={slope_threshold}°,σ={smooth_sigma}{edge_str}{strength_str})"
    )
    return transform


def upscale_scores(
    scores: np.ndarray,
    scale: int = 4,
    method: str = "auto",
    nodata_value: float = np.nan,
) -> np.ndarray:
    """
    Upscale score grid to reduce blockiness when applied to terrain.

    Uses AI super-resolution (Real-ESRGAN) when available, falling back to
    bilateral upscaling for edge-preserving smoothness.

    Args:
        scores: Input score grid (2D numpy array)
        scale: Upscaling factor (default: 4, meaning 4x resolution)
        method: Upscaling method:
            - "auto": Try Real-ESRGAN, fall back to bilateral
            - "esrgan": Use Real-ESRGAN (requires optional realesrgan package)
            - "bilateral": Use bilateral filter upscaling (no extra dependencies)
            - "bicubic": Simple bicubic interpolation
        nodata_value: Value treated as no data (default: np.nan)

    Returns:
        Upscaled score grid with smoother gradients

    Note:
        The "esrgan" method requires the optional ``realesrgan`` package::

            pip install realesrgan

        Or install terrain-maker with the upscale extra::

            pip install terrain-maker[upscale]

        Without it, "auto" will fall back to "bilateral" which produces
        good results without ML dependencies.

    Example:
        >>> scores_hires = upscale_scores(sledding_scores, scale=4)
        >>> # Now scores_hires is 4x the resolution with smoother transitions

        >>> # Force bilateral method (no ML dependencies)
        >>> scores_hires = upscale_scores(scores, scale=4, method="bilateral")
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Upscaling scores {scores.shape} by {scale}x using method='{method}'")

    # Handle nodata mask
    if np.isnan(nodata_value):
        mask = np.isnan(scores)
    else:
        mask = scores == nodata_value

    # Fill nodata with nearest valid values for upscaling
    data = scores.copy()
    if np.any(mask):
        from scipy.ndimage import distance_transform_edt
        # Fill nodata with nearest neighbor
        indices = distance_transform_edt(mask, return_distances=False, return_indices=True)
        data = data[tuple(indices)]

    # Normalize to 0-1 for processing
    data_min, data_max = np.nanmin(data), np.nanmax(data)
    if data_max > data_min:
        normalized = (data - data_min) / (data_max - data_min)
    else:
        normalized = np.zeros_like(data)

    # Try methods in order of preference: quality → speed
    if method == "auto":
        # ESRGAN (best quality) → bilinear (fast, good quality) → nearest (fastest fallback)
        methods_to_try = ["esrgan", "bilinear", "nearest"]
    else:
        methods_to_try = [method]

    result = None
    used_method = None

    for m in methods_to_try:
        try:
            if m == "esrgan":
                result = _upscale_esrgan(normalized, scale)
                used_method = "esrgan"
                break
            elif m == "bilateral":
                result = _upscale_bilateral(normalized, scale)
                used_method = "bilateral"
                break
            elif m == "bicubic":
                result = zoom(normalized, scale, order=3, mode="reflect")
                used_method = "bicubic"
                break
            elif m == "bilinear":
                # Fast upscaling with bilinear interpolation
                result = zoom(normalized, scale, order=1, mode="reflect")
                used_method = "bilinear"
                break
            elif m == "nearest":
                # Fastest upscaling with nearest neighbor
                result = zoom(normalized, scale, order=0, mode="constant")
                used_method = "nearest"
                break
        except ImportError as e:
            # Log at INFO level so user can see why ESRGAN isn't working
            logger.info(f"Method '{m}' not available: {e}")
            logger.info(f"  Falling back to next method...")
            continue
        except Exception as e:
            logger.warning(f"Method '{m}' failed: {e}")
            logger.warning(f"  Falling back to next method...")
            continue

    if result is None:
        # Last resort: fastest possible (nearest neighbor)
        logger.warning("All upscaling methods failed, using fastest fallback (nearest neighbor)")
        result = zoom(normalized, scale, order=0, mode="constant")
        used_method = "nearest"

    # Denormalize back to original range
    result = result * (data_max - data_min) + data_min

    # Clamp to original range (interpolation can overshoot)
    result = np.clip(result, data_min, data_max)

    # Restore nodata mask (upscaled)
    if np.any(mask):
        mask_upscaled = zoom(mask.astype(np.float32), scale, order=0, mode="constant") > 0.5
        if np.isnan(nodata_value):
            result[mask_upscaled] = np.nan
        else:
            result[mask_upscaled] = nodata_value

    logger.info(f"Upscaled scores: {scores.shape} -> {result.shape} using {used_method}")
    return result.astype(scores.dtype)


def _upscale_esrgan(normalized: np.ndarray, scale: int) -> np.ndarray:
    """Upscale using Real-ESRGAN neural network.

    For large scales (>4x), uses multi-step upscaling by chaining multiple
    ESRGAN passes (e.g., 32x = 4x × 4x × 2x).
    """
    logger = logging.getLogger(__name__)

    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        import torch
    except ImportError as e:
        raise ImportError(
            f"Real-ESRGAN import failed: {e}\n"
            f"Install with: uv sync --extra upscale"
        )

    logger.info("Using Real-ESRGAN for AI upscaling...")

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Running on: {device}")

    # For large scales, chain multiple ESRGAN passes
    # RealESRGAN-x4plus pre-trained model only supports 4x
    # Break down: 32x = 4x × 4x × 2x, 16x = 4x × 4x, 8x = 4x × 2x
    if scale > 4:
        # Decompose scale into chain of 4x and 2x passes
        import math
        num_4x_passes = 0
        remaining_scale = scale

        # Use as many 4x passes as possible
        while remaining_scale >= 4 and remaining_scale % 4 == 0:
            num_4x_passes += 1
            remaining_scale //= 4

        # Handle remaining scale (should be 1, 2, or odd number)
        if remaining_scale > 1 and remaining_scale != 2:
            # For odd scales or scales not power-of-2, log warning and round
            logger.warning(
                f"Scale {scale} not evenly divisible by 4. "
                f"Using {num_4x_passes} passes of 4x + 1 pass of {remaining_scale}x"
            )

        passes = [(4, num_4x_passes)]
        if remaining_scale == 2:
            passes.append((2, 1))
        elif remaining_scale > 2:
            passes.append((remaining_scale, 1))

        logger.info(f"  Multi-step upscaling for {scale}x: " +
                   " × ".join([f"{s}x" for s, n in passes for _ in range(n)]))

        # Apply each pass
        current = normalized
        try:
            for step_scale, num_passes in passes:
                for pass_idx in range(num_passes):
                    current = _upscale_esrgan_single(current, step_scale, device)
                    logger.info(f"    ✓ Pass {pass_idx + 1}: {step_scale}x → shape {current.shape}")
        except Exception as e:
            logger.error(f"  Multi-step ESRGAN failed at pass {pass_idx + 1} (scale={step_scale}x): {e}")
            raise

        return current
    else:
        # Single pass for 2x or 4x
        return _upscale_esrgan_single(normalized, scale, device)


def _upscale_esrgan_single(normalized: np.ndarray, scale: int, device) -> np.ndarray:
    """Single-pass ESRGAN upscaling (supports 2x or 4x only)."""
    logger = logging.getLogger(__name__)

    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet

    # Convert to uint8 image (ESRGAN expects 0-255)
    img_uint8 = (normalized * 255).astype(np.uint8)

    # ESRGAN expects HWC format with 3 channels
    img_rgb = np.stack([img_uint8, img_uint8, img_uint8], axis=-1)

    # Initialize model with matching scale
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=scale  # Must match pre-trained weights (2 or 4)
    )

    # Use default Real-ESRGAN pre-trained models from GitHub releases
    # Download to weights directory if not present
    import os
    from pathlib import Path

    weights_dir = Path.home() / '.cache' / 'realesrgan' / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)

    if scale == 4:
        model_filename = 'RealESRGAN_x4plus.pth'
        model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    elif scale == 2:
        model_filename = 'RealESRGAN_x2plus.pth'
        model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
    else:
        raise ValueError(f"Scale {scale} not supported by Real-ESRGAN (only 2x and 4x)")

    model_path = weights_dir / model_filename

    # Download if not present
    if not model_path.exists():
        logger.info(f"  Downloading {model_filename} (~67MB, one-time)...")
        import urllib.request
        urllib.request.urlretrieve(model_url, model_path)
        logger.info(f"  ✓ Downloaded to {model_path}")

    upsampler = RealESRGANer(
        scale=scale,
        model_path=str(model_path),
        model=model,
        tile=400,  # Process in tiles to save memory
        tile_pad=10,
        pre_pad=0,
        half=device.type == "cuda",  # Use half precision on GPU
        device=device,
    )

    # Run upscaling
    output, _ = upsampler.enhance(img_rgb, outscale=scale)

    # Convert back to single channel float
    result = output[:, :, 0].astype(np.float32) / 255.0

    return result


def _upscale_bilateral(normalized: np.ndarray, scale: int) -> np.ndarray:
    """Upscale using bicubic + bilateral filter for edge-preserving smoothness."""
    logger = logging.getLogger(__name__)
    logger.info("Using bilateral upscaling (edge-preserving)...")

    # First, bicubic upscale
    upscaled = zoom(normalized, scale, order=3, mode="reflect")

    # Then apply bilateral filter to smooth while preserving edges
    try:
        from skimage.restoration import denoise_bilateral

        # sigma_spatial controls smoothing extent
        # sigma_color controls how much color/value difference matters
        result = denoise_bilateral(
            upscaled,
            sigma_color=0.1,
            sigma_spatial=scale * 1.5,  # Scale with upscaling factor
            channel_axis=None,  # Grayscale
        )
        logger.info(f"  Applied bilateral filter (sigma_spatial={scale * 1.5})")
    except ImportError:
        logger.warning("scikit-image not available, using Gaussian smoothing")
        from scipy.ndimage import gaussian_filter
        result = gaussian_filter(upscaled, sigma=scale * 0.5)

    return result
