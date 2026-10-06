"""Entry points: flow_accumulation (file based) and compute_flow_with_basins (arrays)."""

import logging
from pathlib import Path
from typing import Dict, Union, Optional, Literal
import shutil

import numpy as np
import rasterio
from rasterio import Affine

try:
    from pysheds.grid import Grid as PyshedsGrid

    PYSHEDS_AVAILABLE = True
except ImportError:
    PYSHEDS_AVAILABLE = False
    PyshedsGrid = None

from terrain_maker.terrain.hydrology.accumulation import (
    compute_drainage_area,
    compute_upstream_rainfall,
)
from terrain_maker.terrain.hydrology.conditioning import (
    condition_dem,
    condition_dem_spec,
    detect_endorheic_basins,
    detect_ocean_mask,
)
from terrain_maker.terrain.hydrology.flow_cache import (
    _get_cache_key_params,
    _get_dem_mtime,
    _load_from_cache,
    _save_to_cache,
    _validate_cache,
)
from terrain_maker.terrain.hydrology.routing import (
    _fix_coastal_flow_directions,
    compute_flow_direction,
)

logger = logging.getLogger(__name__)


def compute_flow_with_basins(
    dem: np.ndarray,
    dem_transform: Affine,
    precipitation: Optional[np.ndarray] = None,
    precip_transform: Optional[Affine] = None,
    lake_mask: Optional[np.ndarray] = None,
    lake_outlets: Optional[np.ndarray] = None,
    detect_basins: bool = True,
    min_basin_size: int = 5000,
    min_basin_depth: float = 1.0,
    backend: str = "spec",
    coastal_elev_threshold: float = 0.0,
    edge_mode: str = "all",
    max_breach_depth: float = 25.0,
    max_breach_length: int = 150,
    epsilon: float = 1e-4,
    ocean_threshold: float = 0.0,
    ocean_border_only: bool = True,
    upscale_precip: bool = False,
    upscale_factor: int = 4,
    upscale_method: str = "bilinear",
    verbose: bool = True,
) -> Dict[str, any]:
    """
    Compute flow using validated basin preservation pattern.

    Steps:
    1. Detect ocean mask
    2. Detect endorheic basins (optional)
    3. Create conditioning mask (ocean + basins + selective lakes)
    4. Condition DEM with combined mask
    5. Compute flow direction with lake routing (if lakes provided)
    6. Compute drainage area
    7. Compute upstream rainfall (if precipitation provided)

    **Basin-aware lake handling:**
    - Lakes INSIDE preserved basins → pre-masked (act as drainage sinks)
    - Lakes OUTSIDE basins → NOT masked (act as river connectors)

    Parameters
    ----------
    dem : np.ndarray
        Digital elevation model (2D array)
    dem_transform : Affine
        Geographic transform for DEM
    precipitation : np.ndarray, optional
        Precipitation data (same shape as DEM). If None, upstream rainfall not computed.
    precip_transform : Affine, optional
        Geographic transform for precipitation (currently unused, assumed same as DEM)
    lake_mask : np.ndarray, optional
        Labeled mask of water bodies (0 = no lake, >0 = lake ID)
    lake_outlets : np.ndarray, optional
        Boolean mask of lake outlet cells
    detect_basins : bool, default=True
        Whether to detect and preserve endorheic basins
    min_basin_size : int, default=5000
        Minimum basin size in cells to preserve. When set to the default (5000),
        uses adaptive scaling (4e-5 × total_cells) to handle different DEM sizes.
        Set to a specific value to override adaptive scaling.
    min_basin_depth : float, default=1.0
        Minimum basin depth in meters to be considered endorheic
    backend : str, default="spec"
        Flow algorithm backend ('spec' or 'legacy')
    coastal_elev_threshold : float, default=0.0
        Max elevation for coastal outlets in meters (spec backend only)
    edge_mode : str, default="all"
        Boundary outlet strategy: 'all', 'local_minima', 'outward_slope', 'none'
    max_breach_depth : float, default=25.0
        Max vertical breach per cell in meters (spec backend only)
    max_breach_length : int, default=150
        Max breach path length in cells (spec backend only)
    epsilon : float, default=1e-4
        Min gradient in filled areas (spec backend only)
    ocean_threshold : float, default=0.0
        Elevation threshold for ocean detection
    ocean_border_only : bool, default=True
        Only detect ocean from border pixels
    upscale_precip : bool, default=False
        Whether to upscale precipitation data (with upscale_method) before accumulation
        at the integer DEM/precipitation ratio; non-integer ratios use cubic interpolation
    upscale_factor : int, default=4
        Upscaling factor for precipitation (2, 4, or 8)
    upscale_method : str, default="bilinear"
        Upscaling method: "bilinear", "esrgan", "bilateral", "bicubic" or "nearest" (see upscale_scores)
    verbose : bool, default=True
        Print progress messages

    Returns
    -------
    dict
        Dictionary with keys:
        - 'flow_direction': np.ndarray, D8 flow direction codes
        - 'drainage_area': np.ndarray, drainage area in cells
        - 'dem_conditioned': np.ndarray, depression-filled DEM
        - 'ocean_mask': np.ndarray, boolean ocean mask
        - 'basin_mask': np.ndarray or None, boolean endorheic basin mask
        - 'lake_inlets': np.ndarray or None, boolean mask of lake inlet cells
        - 'upstream_rainfall': np.ndarray or None, upstream rainfall (if precip provided)
        - 'conditioning_mask': np.ndarray, combined mask used for DEM conditioning

    Examples
    --------
    Basic usage without lakes:

    >>> results = compute_flow_with_basins(dem, dem_transform)
    >>> flow_dir = results['flow_direction']
    >>> drainage = results['drainage_area']

    With lakes and precipitation:

    >>> results = compute_flow_with_basins(
    ...     dem, dem_transform,
    ...     precipitation=precip,
    ...     lake_mask=lakes,
    ...     lake_outlets=outlets,
    ...     detect_basins=True
    ... )
    >>> upstream_rain = results['upstream_rainfall']

    Notes
    -----
    This function uses the 'spec' backend by default, which provides:
    - Breaching-based depression handling
    - Configurable coastal outlet detection
    - Endorheic basin preservation
    """
    from terrain_maker.terrain.transforms import upscale_scores

    if verbose:
        logger.info("=" * 60)
        logger.info("FLOW PIPELINE WITH BASIN PRESERVATION")
        logger.info("=" * 60)

    # Step 1: Detect ocean
    if verbose:
        logger.info("\n1. Detecting ocean...")
    ocean_mask = detect_ocean_mask(dem, threshold=ocean_threshold, border_only=ocean_border_only)
    if verbose:
        ocean_pct = 100 * np.sum(ocean_mask) / dem.size
        logger.info(f"   Ocean cells: {np.sum(ocean_mask):,} ({ocean_pct:.1f}%)")

    # Step 2: Detect endorheic basins (optional)
    basin_mask = None

    if detect_basins:
        if verbose:
            logger.info("\n2. Detecting endorheic basins...")
        total_cells = dem.size
        adaptive_min_size = int(1e-3 * total_cells)
        effective_min_size = adaptive_min_size if min_basin_size == 5000 else min_basin_size

        if verbose and effective_min_size != min_basin_size:
            logger.info(
                f"   Adaptive basin size: {effective_min_size:,} cells "
                f"({100*effective_min_size/total_cells:.4f}% of domain)"
            )

        basin_mask, endorheic_basins = detect_endorheic_basins(
            dem,
            min_size=effective_min_size,
            exclude_mask=ocean_mask,
            min_depth=min_basin_depth,
        )

        if basin_mask is not None and np.any(basin_mask):
            num_basins = len(endorheic_basins)
            if verbose:
                logger.info(f"   Found {num_basins} endorheic basin(s)")
        else:
            if verbose:
                logger.info("   No significant endorheic basins detected")
            basin_mask = None

    # Step 3: Create conditioning mask (basin-aware lake pre-masking)
    if verbose:
        logger.info("\n3. Creating DEM conditioning mask...")
    conditioning_mask = ocean_mask.copy()

    if lake_mask is not None and basin_mask is not None and np.any(basin_mask):
        lakes_in_basins = (lake_mask > 0) & basin_mask
        if np.any(lakes_in_basins):
            if verbose:
                logger.info(
                    f"   Pre-masking {np.sum(lakes_in_basins):,} lake cells "
                    "inside basins (drainage sinks)"
                )
            conditioning_mask = conditioning_mask | lakes_in_basins
        lakes_outside = (lake_mask > 0) & ~basin_mask
        if np.any(lakes_outside) and verbose:
            logger.info(
                f"   NOT masking {np.sum(lakes_outside):,} lake cells "
                "outside basins (river connectors)"
            )
    elif lake_mask is not None and np.any(lake_mask > 0):
        if verbose:
            logger.info(
                f"   NOT masking {np.sum(lake_mask > 0):,} lake cells "
                "(no basins detected, all are connectors)"
            )

    if basin_mask is not None and np.any(basin_mask):
        if verbose:
            logger.info(
                f"   Pre-masking {np.sum(basin_mask):,} basin cells " "to preserve topography"
            )
        conditioning_mask = conditioning_mask | basin_mask

    # Step 4: Condition DEM
    if verbose:
        logger.info(f"\n4. Conditioning DEM (backend={backend})...")
    if backend == "spec":
        dem_conditioned, outlets, breached_dem = condition_dem_spec(
            dem,
            nodata_mask=conditioning_mask,
            coastal_elev_threshold=coastal_elev_threshold,
            edge_mode=edge_mode,
            max_breach_depth=max_breach_depth,
            max_breach_length=max_breach_length,
            epsilon=epsilon,
        )
    else:
        dem_conditioned = condition_dem(
            dem,
            method="breach",
            ocean_mask=conditioning_mask,
            min_basin_size=min_basin_size,
            min_basin_depth=min_basin_depth,
        )
        breached_dem = None

    # Step 5: Identify lake inlets
    # Step 5-6: Compute flow direction, then route lakes outside basins via DEM spillways
    if verbose:
        logger.info("\n6. Computing flow direction...")
    flow_dir_base = compute_flow_direction(dem_conditioned, mask=ocean_mask)
    flow_dir, lake_inlets = _route_lakes_and_find_inlets(
        lake_mask, lake_outlets, basin_mask, dem_conditioned, flow_dir_base
    )

    # Step 7: Compute drainage area
    if verbose:
        logger.info("\n7. Computing drainage area...")
    drainage_area = compute_drainage_area(flow_dir)

    # Step 8: Compute upstream rainfall (optional)
    upstream_rainfall = None
    if precipitation is not None:
        if verbose:
            logger.info("\n8. Computing upstream rainfall...")
        precip_for_accumulation = precipitation.copy()

        if upscale_precip and precipitation.shape != dem.shape:
            scale_y = dem.shape[0] / precipitation.shape[0]
            scale_x = dem.shape[1] / precipitation.shape[1]
            if abs(scale_y - scale_x) < 0.01 and abs(scale_y - round(scale_y)) < 0.01:
                scale_int = int(round(scale_y))
                precip_for_accumulation = upscale_scores(
                    precipitation, scale=scale_int, method=upscale_method, nodata_value=0.0
                )
            else:
                from scipy.ndimage import zoom

                precip_for_accumulation = zoom(
                    precipitation, (scale_y, scale_x), order=3, mode="reflect"
                )
            if verbose:
                logger.info(
                    f"   Upscaled precipitation: {precipitation.shape} → "
                    f"{precip_for_accumulation.shape}"
                )

        precip_masked = precip_for_accumulation.copy()
        precip_masked[ocean_mask] = 0
        upstream_rainfall = compute_upstream_rainfall(flow_dir, precip_masked)

    if verbose:
        logger.info("\nFlow pipeline complete.")
        logger.info("=" * 60)

    return {
        "flow_direction": flow_dir,
        "drainage_area": drainage_area,
        "dem_conditioned": dem_conditioned,
        "breached_dem": breached_dem,
        "ocean_mask": ocean_mask,
        "basin_mask": basin_mask,
        "lake_inlets": lake_inlets,
        "upstream_rainfall": upstream_rainfall,
        "conditioning_mask": conditioning_mask,
    }


def _load_aligned_precipitation(
    precip_path, dem_shape, dem_transform, dem_crs, upscale_precip, upscale_method
):
    """Load precipitation cropped to the DEM, fill nodata, and resample it onto the DEM grid."""
    # Load precipitation (cropped to DEM bounds using library function)
    logger.info("  flow_accumulation: loading precipitation...")
    from terrain_maker.terrain.data_loading import load_geotiff_cropped_to_dem

    precip_data, precip_transform, precip_crs = load_geotiff_cropped_to_dem(
        precip_path,
        dem_shape=dem_shape,
        dem_transform=dem_transform,
        dem_crs=dem_crs,
        use_windowed_read=True,
    )

    logger.info(f"  flow_accumulation: precipitation loaded {precip_data.shape}")

    # Fill missing values (nodata) using nearest neighbor interpolation
    # Common nodata values: -9999, -32768, 0, NaN, or any negative values (precipitation can't be negative)
    nodata_mask = (
        np.isnan(precip_data)
        | (precip_data < -1000)  # Catch extreme negative nodata values like -9999, -32768
        | (precip_data < 0)  # Any negative value is invalid for precipitation
    )

    if np.any(nodata_mask):
        num_missing = np.sum(nodata_mask)
        total_pixels = precip_data.size
        pct_missing = 100.0 * num_missing / total_pixels
        logger.info(
            f"  Imputing {num_missing:,} missing values ({pct_missing:.1f}%) using nearest neighbor..."
        )

        from scipy.ndimage import distance_transform_edt

        from terrain_maker.terrain._memory import EDT_INDICES, check_memory

        check_memory(nodata_mask.size, EDT_INDICES, "Precipitation nodata imputation")
        # Find indices of nearest valid values
        # Returns shape (ndim, *input_shape) - for 2D: (2, H, W)
        indices = distance_transform_edt(nodata_mask, return_distances=False, return_indices=True)

        # Fill missing values with nearest valid neighbors
        # indices[0][nodata_mask] = row indices, indices[1][nodata_mask] = col indices
        precip_data[nodata_mask] = precip_data[indices[0][nodata_mask], indices[1][nodata_mask]]

        logger.info(f"  ✓ Imputation complete")

    # Check spatial alignment and resample if needed
    if precip_data.shape != dem_shape:
        # Calculate required scale factor
        scale_y = dem_shape[0] / precip_data.shape[0]
        scale_x = dem_shape[1] / precip_data.shape[1]
        is_upscaling = scale_y > 1.0 and scale_x > 1.0

        # Upscale with the requested method if asked AND actually upscaling
        if upscale_precip and is_upscaling:
            logger.info(
                f"  Upscaling precipitation from {precip_data.shape} to {dem_shape} using {upscale_method}..."
            )

            # Use upscale_scores if scale is uniform and an integer
            if abs(scale_y - scale_x) < 0.01 and abs(scale_y - round(scale_y)) < 0.01:
                from terrain_maker.terrain.transforms import upscale_scores

                scale_int = int(round(scale_y))
                logger.info(f"    Running {upscale_method} {scale_int}x upscaling...")
                precip_upscaled = upscale_scores(
                    precip_data, scale=scale_int, method=upscale_method, nodata_value=0.0
                )
                precip_data = precip_upscaled
                logger.info(
                    f"  ✓ Upscaled precipitation using {upscale_method}: {precip_data.shape}"
                )
            else:
                # Non-uniform scaling - Detroit-style approach for GPU acceleration
                # Step 1: Over-upscale to next power-of-2 with the requested method
                # Step 2: Downsample to exact target with rasterio reproject
                import math

                avg_scale = (scale_y + scale_x) / 2
                # Round UP to next power of 2 (e.g., 29.458 → 32)
                power_of_2_scale = 2 ** math.ceil(math.log2(avg_scale))

                if power_of_2_scale >= 2:
                    # Over-upscale with the requested method, then downsample to the exact grid
                    logger.info(
                        f"  Two-step upscaling: {upscale_method} {power_of_2_scale}x + downsample to exact shape..."
                    )

                    from terrain_maker.terrain.transforms import upscale_scores

                    # Step 1: over-upscale to the power-of-2 scale
                    logger.info(
                        f"    Running {upscale_method} {power_of_2_scale}x upscaling..."
                    )
                    precip_esrgan = upscale_scores(
                        precip_data, scale=power_of_2_scale, method=upscale_method, nodata_value=0.0
                    )
                    logger.info(
                        f"    ✓ {upscale_method} complete: {precip_data.shape} → {precip_esrgan.shape}"
                    )

                    # Step 2: Downsample to exact target shape with rasterio reproject
                    from rasterio.warp import reproject, Resampling

                    logger.info(f"    Downsampling to exact target shape...")
                    precip_final = np.empty(dem_shape, dtype=np.float32)

                    # Create transforms for intermediate and target shapes
                    if precip_transform is not None and dem_transform is not None:
                        # Calculate intermediate transform (after over-upscaling)
                        esrgan_transform = precip_transform * Affine.scale(1.0 / power_of_2_scale)

                        reproject(
                            source=precip_esrgan,
                            destination=precip_final,
                            src_transform=esrgan_transform,
                            src_crs=precip_crs,
                            dst_transform=dem_transform,
                            dst_crs=dem_crs,
                            resampling=Resampling.bilinear,
                        )
                        logger.info(
                            f"    ✓ Downsampling complete: {precip_esrgan.shape} → {precip_final.shape}"
                        )
                    else:
                        # No transform available - use scipy zoom for final adjustment
                        from scipy.ndimage import zoom

                        scale_y_final = dem_shape[0] / precip_esrgan.shape[0]
                        scale_x_final = dem_shape[1] / precip_esrgan.shape[1]
                        precip_final = zoom(
                            precip_esrgan, (scale_y_final, scale_x_final), order=1, mode="reflect"
                        )
                        logger.info(
                            f"    ✓ Downsampling complete: {precip_esrgan.shape} → {precip_final.shape}"
                        )

                    precip_data = precip_final
                    logger.info(f"  ✓ Two-step upscaling complete: {precip_final.shape}")
                else:
                    # Scale below 2x: resample directly with rasterio
                    from rasterio.warp import reproject, Resampling

                    logger.info(
                        f"  Non-uniform scaling ({scale_y:.2f}x, {scale_x:.2f}x), using rasterio reproject..."
                    )

                    precip_resampled = np.empty(dem_shape, dtype=np.float32)
                    reproject(
                        source=precip_data,
                        destination=precip_resampled,
                        src_transform=precip_transform,
                        src_crs=precip_crs,
                        dst_transform=dem_transform,
                        dst_crs=dem_crs,
                        resampling=Resampling.bilinear,
                    )
                    precip_data = precip_resampled
                    logger.info(f"  ✓ Resampled precipitation to {precip_data.shape}")
        elif upscale_precip and not is_upscaling:
            # User requested upscaling but data is being downscaled - inform and use standard resampling
            logger.info(
                f"  Precipitation is being downscaled ({scale_y:.2f}x, {scale_x:.2f}x), using bilinear resampling..."
            )
            from rasterio.warp import reproject, Resampling

            precip_resampled = np.empty(dem_shape, dtype=np.float32)
            reproject(
                source=precip_data,
                destination=precip_resampled,
                src_transform=precip_transform,
                src_crs=precip_crs,
                dst_transform=dem_transform,
                dst_crs=dem_crs,
                resampling=Resampling.bilinear,
            )
            precip_data = precip_resampled
            logger.info(f"  ✓ Downsampled precipitation to {precip_data.shape}")
        else:
            # Standard resampling (no upscaling requested)
            logger.info(
                f"  Resampling precipitation from {precip_data.shape} to match DEM {dem_shape}..."
            )

            from rasterio.warp import reproject, Resampling

            precip_resampled = np.empty(dem_shape, dtype=np.float32)

            reproject(
                source=precip_data,
                destination=precip_resampled,
                src_transform=precip_transform,
                src_crs=precip_crs,
                dst_transform=dem_transform,
                dst_crs=dem_crs,
                resampling=Resampling.bilinear,
            )

            precip_data = precip_resampled
            logger.info(f"  ✓ Resampled precipitation to {precip_data.shape}")

    return precip_data


def _downsample_to_max_cells(
    original_shape, max_cells, dem_transform, lake_mask, lake_outlets, dem_data, dem_crs
):
    """Downsample the DEM (and lake rasters) to at most max_cells; returns the possibly-updated arrays and flags."""
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

        logger.info(
            f"  Downsampling DEM from {original_shape} ({current_cells:,} cells) to "
            f"{downsampled_shape} ({new_height * new_width:,} cells) "
            f"[{downsample_factor:.2f}x factor]..."
        )

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
            resampling=Resampling.bilinear,
        )

        dem_data = dem_downsampled
        dem_transform = downsampled_transform
        dem_shape = downsampled_shape
        downsampling_applied = True

        # Also downsample lake_mask and lake_outlets if provided
        if lake_mask is not None:
            from scipy.ndimage import zoom

            scale_y = downsampled_shape[0] / original_shape[0]
            scale_x = downsampled_shape[1] / original_shape[1]
            lake_mask = zoom(lake_mask, (scale_y, scale_x), order=0)
            logger.info(f"  ✓ Downsampled lake_mask to {lake_mask.shape}")

        if lake_outlets is not None:
            from scipy.ndimage import zoom

            scale_y = downsampled_shape[0] / original_shape[0]
            scale_x = downsampled_shape[1] / original_shape[1]
            lake_outlets = zoom(lake_outlets.astype(np.uint8), (scale_y, scale_x), order=0).astype(
                bool
            )
            logger.info(f"  ✓ Downsampled lake_outlets to {lake_outlets.shape}")

        logger.info(f"  ✓ Downsampled DEM to {dem_shape}")
    elif max_cells is not None:
        logger.info(
            f"DEM size ({original_shape[0] * original_shape[1]:,} cells) below max_cells ({max_cells:,}), "
            f"no downsampling needed"
        )
    return (
        dem_data,
        dem_shape,
        dem_transform,
        downsample_factor,
        downsampling_applied,
        lake_mask,
        lake_outlets,
    )


def _write_flow_outputs(
    output_dir,
    dem_path,
    dem_transform,
    dem_crs,
    flow_direction,
    drainage_area,
    upstream_rainfall,
    conditioned_dem,
):
    """Write flow direction, drainage, rainfall and conditioned DEM GeoTIFFs; return (paths, output_dir)."""
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
    return files, output_dir


def _route_lakes_and_find_inlets(
    lake_mask, lake_outlets, basin_mask, conditioned_dem, flow_direction
):
    """Route flow through lakes outside endorheic basins (DEM spillways) and mark lake inlet cells."""
    # Step 3.5: Apply lake routing if lake_mask provided
    # Uses DEM-based spillway detection (boundary cells) instead of HydroLAKES
    # pour points (which land in lake interiors and create terminal outlets).
    # Basin-aware: only routes lakes outside preserved endorheic basins.
    if lake_mask is not None and lake_outlets is not None and np.any(lake_mask > 0):
        from terrain_maker.terrain.water_bodies import (
            create_lake_flow_routing,
            find_lake_spillways,
            compute_outlet_downstream_directions,
        )

        logger.info("Applying lake flow routing (DEM-based spillways)...")

        # Basin-aware: only route lakes OUTSIDE preserved basins
        labeled_lakes = lake_mask.copy()
        if basin_mask is not None and np.any(basin_mask):
            labeled_lakes[basin_mask] = 0
            lakes_outside = (lake_mask > 0) & ~basin_mask
            n_in = len(np.unique(lake_mask[(lake_mask > 0) & basin_mask]))
            n_out = len(np.unique(lake_mask[lakes_outside]))
            logger.info(
                f"  {n_in} lakes inside basins (natural flow), "
                f"{n_out} lakes outside basins (explicit routing)"
            )
        else:
            lakes_outside = lake_mask > 0

        if np.any(lakes_outside):
            # DEM-based spillways: find lowest boundary cell for each lake
            spillways = find_lake_spillways(labeled_lakes, conditioned_dem)
            spillway_outlets = np.zeros(lake_mask.shape, dtype=bool)
            for lake_id, (sr, sc, _sdir) in spillways.items():
                spillway_outlets[sr, sc] = True

            logger.info(
                f"  DEM spillway detection: {len(spillways)} spillways "
                f"(replacing {int(np.sum(lake_outlets)):,} HydroLAKES pour points)"
            )

            # BFS routing: all lake cells route toward DEM spillway
            lake_flow = create_lake_flow_routing(labeled_lakes, spillway_outlets, conditioned_dem)
            flow_direction = np.where(lakes_outside, lake_flow, flow_direction)

            # Connect spillway outlets to downstream terrain (cycle-safe)
            if np.any(spillway_outlets):
                flow_direction = compute_outlet_downstream_directions(
                    flow_direction,
                    labeled_lakes,
                    spillway_outlets,
                    conditioned_dem,
                    basin_mask=basin_mask,
                    spillways=spillways,
                )

            logger.info(
                f"  Applied routing to {np.sum(lakes_outside):,} cells "
                f"with {len(spillways)} spillway outlets"
            )

    # Step 3.6: Identify lake inlets (after DEM conditioning + lake routing)
    lake_inlets = None
    if lake_mask is not None and np.any(lake_mask > 0):
        from terrain_maker.terrain.water_bodies import identify_lake_inlets

        outlet_mask_for_inlets = lake_outlets if lake_outlets is not None else None
        inlets_dict = identify_lake_inlets(
            lake_mask, conditioned_dem, outlet_mask=outlet_mask_for_inlets
        )
        if inlets_dict:
            lake_inlets = np.zeros_like(lake_mask, dtype=bool)
            for lake_id, inlet_cells in inlets_dict.items():
                for row, col in inlet_cells:
                    if 0 <= row < lake_inlets.shape[0] and 0 <= col < lake_inlets.shape[1]:
                        lake_inlets[row, col] = True
            logger.info(f"  Lake inlets: {np.sum(lake_inlets):,} cells")
    return flow_direction, lake_inlets


def _condition_and_route(
    backend,
    fill_method,
    min_basin_size,
    detect_basins,
    conditioning_mask,
    ocean_mask,
    dem_data,
    coastal_elev_threshold,
    edge_mode,
    max_breach_depth,
    max_breach_length,
    epsilon,
    masked_basin_outlets,
    parallel_method,
    dem_transform,
    dem_crs,
    downsample_factor,
    max_fill_depth,
    flow_mask,
):
    """Condition the DEM and compute D8 flow direction with the selected backend (spec, pysheds or legacy)."""
    pysheds_state = None
    # Initialize breached_dem (only set by spec backend, None for others)
    breached_dem = None

    # Branch based on backend
    if backend == "spec":
        # === SPEC-COMPLIANT BACKEND ===
        # Use spec-compliant 4-stage pipeline (outlet ID + breaching + fill)
        logger.info(f"Using spec-compliant backend (flow-spec.md)...")

        # Emit warnings if legacy parameters are specified
        if fill_method != "breach":
            import warnings

            warnings.warn(
                f"fill_method='{fill_method}' is ignored when backend='spec'. "
                "Use epsilon parameter instead (epsilon=0 for fill, epsilon>0 for breach-like).",
                DeprecationWarning,
            )
        if min_basin_size != 10000:
            import warnings

            warnings.warn(
                "min_basin_size is ignored when backend='spec'. "
                "Use max_breach_depth/max_breach_length to control basin preservation.",
                DeprecationWarning,
            )

        # Step 2: Condition DEM using spec-compliant pipeline
        # Use combined conditioning mask (ocean + basins) to preserve topography
        logger.info("Conditioning DEM (outlets + breach + fill)...")
        nodata_mask_for_spec = conditioning_mask if detect_basins else ocean_mask
        conditioned_dem, outlets, breached_dem = condition_dem_spec(
            dem_data,
            nodata_mask=nodata_mask_for_spec,
            coastal_elev_threshold=coastal_elev_threshold,
            edge_mode=edge_mode,
            max_breach_depth=max_breach_depth,
            max_breach_length=max_breach_length,
            epsilon=epsilon,
            masked_basin_outlets=masked_basin_outlets,
            parallel_method=parallel_method,
        )

        # Step 3: Compute flow directions
        logger.info("Computing flow directions...")
        # CRITICAL: Combine ocean/nodata with all identified outlets (edge, coastal, masked basin)
        # so that ALL outlet types are properly used as flow direction terminals
        outlet_mask = ocean_mask | outlets
        flow_direction = compute_flow_direction(conditioned_dem, mask=outlet_mask)
        logger.info("  ✓ Flow directions computed")

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
            UserWarning,
        )
        logger.info(f"Using pysheds backend for flow computation...")

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
        logger.info("  pysheds: Filling pits...")
        pit_filled = grid.fill_pits(dem_raster)

        logger.info("  pysheds: Filling depressions...")
        flooded = grid.fill_depressions(pit_filled)

        logger.info("  pysheds: Resolving flats...")
        inflated = grid.resolve_flats(flooded)

        conditioned_dem = np.array(inflated).astype(np.float32)

        # Restore original ocean values to conditioned DEM
        if ocean_mask is not None and np.any(ocean_mask):
            conditioned_dem[ocean_mask] = dem_data[ocean_mask]

        # Step 3: Compute flow direction using pysheds
        logger.info("  pysheds: Computing flow direction...")
        fdir = grid.flowdir(inflated)
        pysheds_state = (grid, fdir, viewfinder)  # reused for accumulation

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
            scaled_min_basin_size = max(100, int(min_basin_size / (downsample_factor**2)))
            logger.info(
                f"Conditioning DEM (method={fill_method}, min_basin_size={min_basin_size} → {scaled_min_basin_size} scaled)..."
            )
        else:
            logger.info(
                f"Conditioning DEM (method={fill_method}, min_basin_size={min_basin_size})..."
            )
        conditioned_dem = condition_dem(
            dem_data,
            method=fill_method,
            ocean_mask=ocean_mask,
            min_basin_size=scaled_min_basin_size,
            max_fill_depth=max_fill_depth,
        )

        # Step 3: Compute flow directions (with combined mask)
        logger.info("Computing flow directions...")
        flow_direction = compute_flow_direction(
            conditioned_dem, mask=flow_mask if np.any(flow_mask) else None
        )
    return breached_dem, conditioned_dem, flow_direction, pysheds_state


def _build_conditioning_masks(
    mask_ocean,
    dem_data,
    ocean_elevation_threshold,
    detect_basins,
    backend,
    min_basin_size,
    min_basin_depth,
    lake_mask,
):
    """Detect ocean and endorheic basins and combine them (plus lakes inside basins) into the conditioning mask."""
    # Step 1: Detect ocean mask (if enabled)
    ocean_mask = None
    if mask_ocean:
        logger.info(
            f"Detecting ocean (elevation <= {ocean_elevation_threshold}m, border-connected)..."
        )
        ocean_mask = detect_ocean_mask(
            dem_data, threshold=ocean_elevation_threshold, border_only=True
        )
        ocean_cells = np.sum(ocean_mask)
        ocean_pct = 100 * ocean_cells / ocean_mask.size
        logger.info(f"  Ocean detected: {ocean_cells:,} cells ({ocean_pct:.1f}%)")

    # Step 1b: Detect endorheic basins (if enabled and using spec backend)
    basin_mask = None
    if detect_basins and backend == "spec":
        logger.info(
            f"Detecting endorheic basins (min_size={min_basin_size}, min_depth={min_basin_depth:.1f}m)..."
        )
        basin_mask, endorheic_basins = detect_endorheic_basins(
            dem_data,
            min_size=min_basin_size,
            min_depth=min_basin_depth,
            exclude_mask=ocean_mask,
        )

        if basin_mask is not None and np.any(basin_mask):
            num_basins = len(endorheic_basins)
            basin_coverage = 100 * np.sum(basin_mask) / dem_data.size
            logger.info(
                f"  Found {num_basins} endorheic basin(s) ({basin_coverage:.2f}% of domain)"
            )
            logger.info(f"  Basins will be masked during conditioning to preserve topography")
        else:
            logger.info("  No significant endorheic basins detected")
            basin_mask = None
    elif detect_basins and backend != "spec":
        import warnings

        warnings.warn(
            f"detect_basins=True is only supported with backend='spec'. "
            f"Use min_basin_size parameter with legacy backend instead.",
            UserWarning,
        )

    # Create combined conditioning mask for spec backend
    # Strategy: ocean + endorheic basins + selective lakes
    # (Lakes handled later in the pipeline based on basin location)
    conditioning_mask = (
        ocean_mask.copy() if ocean_mask is not None else np.zeros(dem_data.shape, dtype=bool)
    )

    # Basin-aware lake pre-masking:
    # Lakes INSIDE basins → masked (drainage sinks like Salton Sea)
    # Lakes OUTSIDE basins → NOT masked (river connectors)
    if lake_mask is not None and basin_mask is not None and np.any(basin_mask):
        lakes_in_basins = (lake_mask > 0) & basin_mask
        if np.any(lakes_in_basins):
            conditioning_mask = conditioning_mask | lakes_in_basins
            logger.info(
                f"  Pre-masking {np.sum(lakes_in_basins):,} lake cells inside basins (drainage sinks)"
            )

        lakes_outside = (lake_mask > 0) & ~basin_mask
        if np.any(lakes_outside):
            logger.info(
                f"  NOT masking {np.sum(lakes_outside):,} lake cells outside basins (river connectors)"
            )
    elif lake_mask is not None and np.any(lake_mask > 0):
        logger.info(
            f"  NOT masking {np.sum(lake_mask > 0):,} lake cells (no basins detected, all are connectors)"
        )

    if basin_mask is not None and np.any(basin_mask):
        conditioning_mask = conditioning_mask | basin_mask
        logger.info(
            f"  Combined conditioning mask: {np.sum(conditioning_mask):,} cells "
            f"({100*np.sum(conditioning_mask)/conditioning_mask.size:.1f}%)"
        )

    # Use ocean mask for flow computation (flow direction terminals)
    # Note: Basins are NOT masked from flow - flow is computed inside them
    flow_mask = ocean_mask if ocean_mask is not None else None
    return basin_mask, conditioning_mask, flow_mask, ocean_mask


def _cell_size_m(cell_size, dem_shape, dem_crs, dem_transform):
    """Cell size in meters (converted from degrees for geographic CRS) unless given."""
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

            logger.info(
                f"  Geographic CRS detected (cell size: {pixel_size_deg:.8f}° ≈ {cell_size:.1f}m at lat {center_lat:.2f}°)"
            )
        else:
            # Projected CRS - pixel size is already in CRS units (meters)
            cell_size = abs(dem_transform.a)
    return cell_size


def _accumulate_flow(backend, pysheds_state, flow_direction, precip_data, ocean_mask):
    """Drainage area (cells) and precipitation-weighted upstream rainfall, with ocean excluded."""
    # Step 4: Compute drainage area and upstream rainfall
    if backend == "pysheds":
        # Use pysheds accumulation
        from pysheds.sview import Raster

        grid, fdir, viewfinder = pysheds_state
        logger.info("  pysheds: Computing drainage area...")
        acc = grid.accumulation(fdir)
        drainage_area = np.array(acc).astype(np.float32)

        logger.info("  pysheds: Computing upstream rainfall (weighted)...")
        # CRITICAL: Mask ocean in precipitation BEFORE accumulation
        # Otherwise ocean precip accumulates into coastal cells (coastline artifacts)
        precip_for_pysheds = precip_data.copy()
        if ocean_mask is not None and np.any(ocean_mask):
            precip_for_pysheds[ocean_mask] = 0
        # Wrap precipitation as Raster for weighted accumulation
        precip_raster = Raster(precip_for_pysheds, viewfinder=viewfinder)
        weighted_acc = grid.accumulation(fdir, weights=precip_raster)
        upstream_rainfall = np.array(weighted_acc).astype(np.float32)

        # Apply ocean mask to drainage area output (already masked for upstream_rainfall)
        if ocean_mask is not None and np.any(ocean_mask):
            drainage_area[ocean_mask] = 0
    else:
        # Use custom backend
        logger.info("Computing drainage area...")
        drainage_area = compute_drainage_area(flow_direction)

        logger.info("Computing upstream rainfall...")
        # CRITICAL: Mask ocean in precipitation BEFORE accumulation
        # Otherwise ocean precip accumulates into coastal cells (coastline artifacts)
        precip_masked = precip_data.copy()
        if ocean_mask is not None and np.any(ocean_mask):
            precip_masked[ocean_mask] = 0
        upstream_rainfall = compute_upstream_rainfall(flow_direction, precip_masked)
    return drainage_area, upstream_rainfall


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
    parallel_method: str = "checkerboard",
    epsilon: Optional[float] = None,
    masked_basin_outlets: Optional[np.ndarray] = None,
    # Basin detection parameters
    detect_basins: bool = False,
    min_basin_depth: float = 1.0,
    # Common parameters
    cell_size: Optional[float] = None,
    max_cells: Optional[int] = None,
    target_vertices: Optional[int] = None,
    mask_ocean: bool = True,
    ocean_elevation_threshold: float = 0.0,
    backend: Literal["legacy", "spec", "pysheds"] = "spec",
    lake_mask: Optional[np.ndarray] = None,
    lake_outlets: Optional[np.ndarray] = None,
    # Precipitation upscaling parameters
    upscale_precip: bool = False,
    upscale_factor: int = 4,
    upscale_method: str = "bilinear",
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
        epsilon : float, optional
            Minimum gradient in filled areas (meters/cell). If None (default),
            automatically calculated as 1e-5 * cell_resolution per flow-spec.md
            guidelines (e.g., 1e-4 for 10m DEM). Pass explicit value to override.
        masked_basin_outlets : np.ndarray (bool), optional
            User-supplied outlet locations for known lakes/basins
    lake_mask : np.ndarray, optional
        Labeled mask of known water bodies (0 = no lake, N = lake ID).
        Lake interior cells will route flow toward their outlets.
    lake_outlets : np.ndarray (bool), optional
        Boolean mask of lake outlet cells. Required if lake_mask is provided.
        Outlets receive accumulated flow from all lake cells.
    detect_basins : bool, default False
        If True, automatically detect and preserve endorheic basins (closed drainage
        basins). Basins are masked during DEM conditioning to preserve their original
        topography. Works with spec backend only.
    min_basin_depth : float, default 1.0
        Minimum basin depth (meters) to be considered endorheic. Only used when
        detect_basins=True. Basins shallower than this threshold are not preserved.
    upscale_precip : bool, default False
        If True, upscale precipitation data to match DEM resolution using upscale_method
        upscaling before computing upstream rainfall. This preserves fine-scale precipitation
        patterns and reduces coastal artifacts. Upscaling happens BEFORE ocean masking.
    upscale_factor : int, default 4
        Target upscaling factor for precipitation (2, 4, or 8). Only used if upscale_precip=True.
    upscale_method : str, default "bilinear"
        Upscaling method passed to upscale_scores: "bilinear", "esrgan" (needs the upscale
        extra), "bilateral", "bicubic" or "nearest". A method that fails raises.
        Only used if upscale_precip=True.
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
    logger.info("  flow_accumulation: validating inputs...")
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
            logger.info("  flow_accumulation: loading from cache...")
            return _load_from_cache(cache_path)

        logger.info("  flow_accumulation: cache miss, computing...")

    # Load DEM
    logger.info("  flow_accumulation: loading DEM...")
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1).astype(np.float32)
        dem_transform = src.transform
        dem_crs = src.crs
        original_shape = dem_data.shape
    logger.info(f"  flow_accumulation: DEM loaded {original_shape}")

    (
        dem_data,
        dem_shape,
        dem_transform,
        downsample_factor,
        downsampling_applied,
        lake_mask,
        lake_outlets,
    ) = _downsample_to_max_cells(
        original_shape=original_shape,
        max_cells=max_cells,
        dem_transform=dem_transform,
        lake_mask=lake_mask,
        lake_outlets=lake_outlets,
        dem_data=dem_data,
        dem_crs=dem_crs,
    )

    precip_data = _load_aligned_precipitation(
        precip_path=precip_path,
        dem_shape=dem_shape,
        dem_transform=dem_transform,
        dem_crs=dem_crs,
        upscale_precip=upscale_precip,
        upscale_method=upscale_method,
    )
    cell_size = _cell_size_m(
        cell_size=cell_size, dem_shape=dem_shape, dem_crs=dem_crs, dem_transform=dem_transform
    )

    # Auto-calculate epsilon if not provided (flow-spec.md guideline)
    # epsilon = 1e-5 * cell_resolution (e.g., 1e-4 for 10m DEM)
    if epsilon is None:
        epsilon = 1e-5 * cell_size
        logger.info(
            f"  Auto-calculated epsilon: {epsilon:.2e} m/cell (= 1e-5 × {cell_size:.1f}m cell size)"
        )

    basin_mask, conditioning_mask, flow_mask, ocean_mask = _build_conditioning_masks(
        mask_ocean=mask_ocean,
        dem_data=dem_data,
        ocean_elevation_threshold=ocean_elevation_threshold,
        detect_basins=detect_basins,
        backend=backend,
        min_basin_size=min_basin_size,
        min_basin_depth=min_basin_depth,
        lake_mask=lake_mask,
    )

    breached_dem, conditioned_dem, flow_direction, pysheds_state = _condition_and_route(
        backend=backend,
        fill_method=fill_method,
        min_basin_size=min_basin_size,
        detect_basins=detect_basins,
        conditioning_mask=conditioning_mask,
        ocean_mask=ocean_mask,
        dem_data=dem_data,
        coastal_elev_threshold=coastal_elev_threshold,
        edge_mode=edge_mode,
        max_breach_depth=max_breach_depth,
        max_breach_length=max_breach_length,
        epsilon=epsilon,
        masked_basin_outlets=masked_basin_outlets,
        parallel_method=parallel_method,
        dem_transform=dem_transform,
        dem_crs=dem_crs,
        downsample_factor=downsample_factor,
        max_fill_depth=max_fill_depth,
        flow_mask=flow_mask,
    )

    flow_direction, lake_inlets = _route_lakes_and_find_inlets(
        lake_mask=lake_mask,
        lake_outlets=lake_outlets,
        basin_mask=basin_mask,
        conditioned_dem=conditioned_dem,
        flow_direction=flow_direction,
    )

    drainage_area, upstream_rainfall = _accumulate_flow(
        backend=backend,
        pysheds_state=pysheds_state,
        flow_direction=flow_direction,
        precip_data=precip_data,
        ocean_mask=ocean_mask,
    )

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
        # Store serializable versions of transform and crs
        "transform": tuple(dem_transform),  # Affine as 6-tuple (a, b, c, d, e, f)
        "crs": str(dem_crs),  # CRS as string (e.g., "EPSG:4326")
    }

    # Add target_vertices to metadata if specified
    if target_vertices is not None:
        metadata["target_vertices"] = target_vertices

    # Mark as fresh computation (not from cache)
    if cache:
        metadata["cache_hit"] = False

    files, output_dir = _write_flow_outputs(
        output_dir=output_dir,
        dem_path=dem_path,
        dem_transform=dem_transform,
        dem_crs=dem_crs,
        flow_direction=flow_direction,
        drainage_area=drainage_area,
        upstream_rainfall=upstream_rainfall,
        conditioned_dem=conditioned_dem,
    )

    result = {
        "flow_direction": flow_direction,
        "drainage_area": drainage_area,
        "upstream_rainfall": upstream_rainfall,
        "conditioned_dem": conditioned_dem,
        "breached_dem": breached_dem,
        "lake_mask": lake_mask,  # Downsampled lake mask (or None if not provided)
        "lake_inlets": lake_inlets,
        "basin_mask": basin_mask,
        "ocean_mask": ocean_mask,
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
            for filename in [
                "flow_direction.tif",
                "flow_accumulation_area.tif",
                "flow_accumulation_rainfall.tif",
                "dem_conditioned.tif",
            ]:
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
        logger.info(f"  flow_accumulation: cached to {cache_save_path}")

    return result


def _write_geotiff(path: str, data: np.ndarray, transform: Affine, crs: rasterio.crs.CRS) -> None:
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
