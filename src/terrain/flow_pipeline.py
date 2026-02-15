"""
Flow computation pipeline with basin preservation.

This module provides a validated flow computation pattern that properly handles:
1. Ocean detection
2. Endorheic basin detection and preservation
3. Water body (lake) integration with basin awareness
4. DEM conditioning with combined masks
5. Flow direction and drainage area computation

The pattern is extracted from validate_flow_with_water_bodies.py and provides
a reusable orchestrator for flow analysis.
"""

import numpy as np
from rasterio import Affine
from typing import Optional, Dict, Any, Tuple

from src.terrain.flow_accumulation import (
    compute_flow_direction,
    compute_drainage_area,
    compute_upstream_rainfall,
    condition_dem_spec,
    detect_ocean_mask,
    detect_endorheic_basins,
)
from src.terrain.water_bodies import (
    identify_lake_inlets,
    create_lake_flow_routing,
    compute_outlet_downstream_directions,
)
from src.terrain.transforms import upscale_scores


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
    upscale_method: str = "auto",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compute flow using validated basin preservation pattern.

    This function implements the approach validated in validate_flow_with_water_bodies.py:
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
        Whether to upscale precipitation data using ESRGAN before accumulation
    upscale_factor : int, default=4
        Upscaling factor for precipitation (2, 4, or 8)
    upscale_method : str, default="auto"
        Upscaling method: "auto" (try ESRGAN, fall back to bilateral), "esrgan", "bilateral", or "bicubic"
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
    if verbose:
        print("=" * 60)
        print("FLOW PIPELINE WITH BASIN PRESERVATION")
        print("=" * 60)

    # Step 1: Detect ocean
    if verbose:
        print("\n1. Detecting ocean...")
    ocean_mask = detect_ocean_mask(
        dem, threshold=ocean_threshold, border_only=ocean_border_only
    )
    if verbose:
        ocean_pct = 100 * np.sum(ocean_mask) / dem.size
        print(f"   Ocean cells: {np.sum(ocean_mask):,} ({ocean_pct:.1f}%)")

    # Step 2: Detect endorheic basins (optional)
    basin_mask = None
    endorheic_basins = {}

    if detect_basins:
        if verbose:
            print("\n2. Detecting endorheic basins...")

        # Adaptive basin size scaling: min_size scales with total DEM area
        # Formula derived from empirical testing: 1000 cells works well for 1000×1000 DEM
        # Scaling factor: 1000 / (1000 × 1000) = 1e-3
        total_cells = dem.size
        adaptive_min_size = int(1e-3 * total_cells)

        # Use adaptive size if default is still set, otherwise respect user override
        effective_min_size = adaptive_min_size if min_basin_size == 5000 else min_basin_size

        if verbose and effective_min_size != min_basin_size:
            print(f"   Adaptive basin size: {effective_min_size:,} cells "
                  f"({100*effective_min_size/total_cells:.4f}% of domain)")

        basin_mask, endorheic_basins = detect_endorheic_basins(
            dem,
            min_size=effective_min_size,
            exclude_mask=ocean_mask,
            min_depth=min_basin_depth,
        )

        if basin_mask is not None and np.any(basin_mask):
            num_basins = len(endorheic_basins)
            basin_coverage = 100 * np.sum(basin_mask) / dem.size
            if verbose:
                print(f"   Found {num_basins} endorheic basin(s)")
                print(f"   Basin coverage: {basin_coverage:.2f}% of domain")
        else:
            if verbose:
                print("   No significant endorheic basins detected")
            basin_mask = None

    # Step 3: Create conditioning mask
    # Strategy:
    # - Ocean: always masked
    # - Endorheic basins: always masked (preserve topography)
    # - Lakes INSIDE basins: masked (drainage sinks like Salton Sea)
    # - Lakes OUTSIDE basins: NOT masked (river connectors)
    if verbose:
        print("\n3. Creating DEM conditioning mask...")
    conditioning_mask = ocean_mask.copy()

    if lake_mask is not None and basin_mask is not None and np.any(basin_mask):
        # Only mask lakes that are inside preserved basins
        lakes_in_basins = (lake_mask > 0) & basin_mask
        if np.any(lakes_in_basins):
            if verbose:
                print(
                    f"   Pre-masking {np.sum(lakes_in_basins):,} lake cells "
                    "inside basins (drainage sinks)"
                )
            conditioning_mask = conditioning_mask | lakes_in_basins

        # Lakes outside basins are NOT masked (act as connecting segments)
        lakes_outside = (lake_mask > 0) & ~basin_mask
        if np.any(lakes_outside) and verbose:
            print(
                f"   NOT masking {np.sum(lakes_outside):,} lake cells "
                "outside basins (river connectors)"
            )
    elif lake_mask is not None and np.any(lake_mask > 0):
        # No basins detected - don't mask any lakes (all are river connectors)
        if verbose:
            print(
                f"   NOT masking {np.sum(lake_mask > 0):,} lake cells "
                "(no basins detected, all are connectors)"
            )

    if basin_mask is not None and np.any(basin_mask):
        if verbose:
            print(
                f"   Pre-masking {np.sum(basin_mask):,} basin cells "
                "to preserve topography"
            )
        conditioning_mask = conditioning_mask | basin_mask

    # Step 4: Condition DEM
    if verbose:
        print(f"\n4. Conditioning DEM (backend={backend})...")

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
        # Legacy backend
        from src.terrain.flow_accumulation import condition_dem

        dem_conditioned = condition_dem(
            dem,
            method="breach",  # Always use breach for consistency
            ocean_mask=conditioning_mask,
            min_basin_size=min_basin_size,
            min_basin_depth=min_basin_depth,
        )
        # Legacy backend doesn't expose breached DEM separately
        breached_dem = None

    # Step 5: Identify lake inlets (after DEM conditioning)
    lake_inlets = None
    if lake_mask is not None and np.any(lake_mask > 0):
        if verbose:
            print("\n5. Identifying lake inlets...")
        outlet_mask_for_inlets = lake_outlets if lake_outlets is not None else None
        inlets_dict = identify_lake_inlets(
            lake_mask, dem_conditioned, outlet_mask=outlet_mask_for_inlets
        )

        if inlets_dict:
            # Convert inlet dict to boolean mask
            lake_inlets = np.zeros_like(lake_mask, dtype=bool)
            for lake_id, inlet_cells in inlets_dict.items():
                for row, col in inlet_cells:
                    if 0 <= row < lake_inlets.shape[0] and 0 <= col < lake_inlets.shape[1]:
                        lake_inlets[row, col] = True

            inlet_count = np.sum(lake_inlets)
            if verbose:
                print(f"   Inlet cells: {inlet_count}")

    # Step 6: Compute flow direction (with lake routing if applicable)
    if verbose:
        print("\n6. Computing flow direction...")
    flow_dir_base = compute_flow_direction(dem_conditioned, mask=ocean_mask)

    # Apply lake routing (with basin awareness)
    flow_dir = flow_dir_base.copy()
    if lake_mask is not None and lake_outlets is not None and np.any(lake_mask > 0):
        if verbose:
            print("   Applying lake flow routing...")

        # Only route lakes OUTSIDE preserved basins
        lakes_outside_basin = lake_mask.copy()
        if basin_mask is not None and np.any(basin_mask):
            lakes_outside_basin = (lake_mask > 0) & ~basin_mask
            num_lakes_in = len(np.unique(lake_mask[lake_mask & basin_mask]))
            num_lakes_out = len(np.unique(lake_mask[lakes_outside_basin]))

            if verbose and num_lakes_in > 0:
                cells_in = np.sum((lake_mask > 0) & basin_mask)
                print(
                    f"   {num_lakes_in} water bodies INSIDE basins "
                    f"({cells_in:,} cells) use natural basin flow"
                )

            if verbose and num_lakes_out > 0:
                cells_out = np.sum(lakes_outside_basin)
                print(
                    f"   {num_lakes_out} water bodies OUTSIDE basins "
                    f"({cells_out:,} cells) use explicit outlet routing"
                )

        # Apply explicit routing only to lakes outside basins
        if np.any(lakes_outside_basin):
            # Create mask of outlets only for lakes outside basins
            lake_outlets_outside = (
                lake_outlets & ~basin_mask if basin_mask is not None else lake_outlets
            )

            lake_flow = create_lake_flow_routing(
                lakes_outside_basin, lake_outlets_outside, dem_conditioned
            )
            flow_dir = np.where(lakes_outside_basin, lake_flow, flow_dir_base)

            # Connect lake outlets to downstream terrain so drainage
            # propagates through lakes instead of resetting at outlets
            if np.any(lake_outlets_outside):
                flow_dir = compute_outlet_downstream_directions(
                    flow_dir,
                    lakes_outside_basin,
                    lake_outlets_outside,
                    dem_conditioned,
                    basin_mask=basin_mask,
                )

            if verbose:
                num_outlets_outside = np.sum(lake_outlets_outside)
                print(
                    f"   Applied routing to {np.sum(lakes_outside_basin):,} cells "
                    f"with {num_outlets_outside} outlets (connected downstream)"
                )

    # Step 7: Compute drainage area
    if verbose:
        print("\n7. Computing drainage area...")
    drainage_area = compute_drainage_area(flow_dir)

    # Step 8: Compute upstream rainfall (optional)
    upstream_rainfall = None
    if precipitation is not None:
        if verbose:
            print("\n8. Computing upstream rainfall...")

        # Upscale/resample precipitation to match DEM resolution if requested
        # This is done BEFORE ocean masking to preserve detail
        precip_for_accumulation = precipitation.copy()

        if upscale_precip:
            # Determine target shape (match DEM/flow_dir)
            target_shape = dem.shape

            if precipitation.shape != target_shape:
                if verbose:
                    print(f"   Upscaling precipitation {precipitation.shape} → {target_shape} using {upscale_method}...")

                # Calculate required scale factor
                scale_y = target_shape[0] / precipitation.shape[0]
                scale_x = target_shape[1] / precipitation.shape[1]

                # Use upscale_scores if scale is uniform and an integer
                if abs(scale_y - scale_x) < 0.01 and abs(scale_y - round(scale_y)) < 0.01:
                    # Uniform integer scale - use ESRGAN/bilateral directly
                    scale_int = int(round(scale_y))
                    precip_upscaled = upscale_scores(
                        precipitation,
                        scale=scale_int,
                        method=upscale_method,
                        nodata_value=0.0
                    )
                    precip_for_accumulation = precip_upscaled
                else:
                    # Non-uniform scaling - two-step approach for GPU acceleration
                    # Step 1: Upscale to nearest integer scale with ESRGAN (GPU)
                    # Step 2: Fine-tune to exact target with rasterio reproject
                    avg_scale = (scale_y + scale_x) / 2
                    nearest_scale = int(round(avg_scale))

                    if nearest_scale >= 2 and upscale_method in ("auto", "esrgan"):
                        # Use ESRGAN for bulk upscaling, then reproject for final adjustment
                        if verbose:
                            print(f"   Two-step upscaling: ESRGAN {nearest_scale}x + reproject to exact shape...")

                        # Step 1: ESRGAN upscaling to nearest integer scale (GPU-accelerated)
                        precip_esrgan = upscale_scores(
                            precipitation,
                            scale=nearest_scale,
                            method=upscale_method,
                            nodata_value=0.0
                        )

                        # Step 2: Fine-tune to exact target shape with rasterio reproject
                        from rasterio.warp import reproject, Resampling
                        precip_final = np.empty(target_shape, dtype=np.float32)

                        # Create transforms for intermediate and target shapes
                        if precip_transform is not None and dem_transform is not None:
                            # Calculate intermediate transform (after ESRGAN upscaling)
                            esrgan_transform = precip_transform * Affine.scale(1.0 / nearest_scale)

                            reproject(
                                source=precip_esrgan,
                                destination=precip_final,
                                src_transform=esrgan_transform,
                                src_crs="EPSG:4326",  # Assume WGS84 for precipitation
                                dst_transform=dem_transform,
                                dst_crs="EPSG:4326",
                                resampling=Resampling.bilinear,
                            )
                        else:
                            # No transform available - use scipy zoom for final adjustment
                            from scipy.ndimage import zoom
                            scale_y_final = target_shape[0] / precip_esrgan.shape[0]
                            scale_x_final = target_shape[1] / precip_esrgan.shape[1]
                            precip_final = zoom(precip_esrgan, (scale_y_final, scale_x_final), order=1, mode='reflect')

                        precip_for_accumulation = precip_final
                        if verbose:
                            print(f"   ✓ ESRGAN upscaled {precipitation.shape} → {precip_esrgan.shape} → {precip_final.shape}")
                    else:
                        # Fall back to basic bicubic for small scales or when bilateral is requested
                        from scipy.ndimage import zoom
                        if verbose:
                            print(f"   Using bicubic zoom for non-uniform scaling ({scale_y:.2f}x, {scale_x:.2f}x)...")
                        precip_for_accumulation = zoom(precipitation, (scale_y, scale_x), order=3, mode='reflect')

                if verbose:
                    print(f"   ✓ Upscaled precipitation: {precipitation.shape} → {precip_for_accumulation.shape}")
                    print(f"   Value range: {precip_for_accumulation.min():.2f} - {precip_for_accumulation.max():.2f} mm")
            else:
                if verbose:
                    print(f"   Precipitation already at DEM resolution {target_shape}, no upscaling needed")

        # Mask ocean cells (AFTER upscaling)
        precip_masked = precip_for_accumulation.copy()
        precip_masked[ocean_mask] = 0

        upstream_rainfall = compute_upstream_rainfall(flow_dir, precip_masked)

    if verbose:
        print("\nFlow pipeline complete.")
        print("=" * 60)

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
