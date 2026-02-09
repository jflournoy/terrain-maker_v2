#!/usr/bin/env python3
"""
Flow validation with REAL water bodies from NHD or HydroLAKES.

This script demonstrates using actual lake data instead of synthetic lakes:
- NHD (National Hydrography Dataset) - USA only, very detailed
- HydroLAKES - Global coverage, includes pour points

Usage:
    # Using NHD (USA only)
    python examples/validate_flow_with_nhd_data.py --data-source nhd

    # Using HydroLAKES (global)
    python examples/validate_flow_with_nhd_data.py --data-source hydrolakes

    # Specify output directory
    python examples/validate_flow_with_nhd_data.py --data-source nhd --output my_results/
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from the synthetic validation script
from validate_flow_with_water_bodies import (
    main as synthetic_main,
    create_validation_images,
    load_real_precipitation,
    validate_no_cycles,
    compute_mass_balance,
    validate_drainage_connectivity,
    trace_flow_paths,
    find_coastal_subset,
    BIGNESS_SIZES,
)

# Import water body functions
from src.terrain.water_bodies import (
    download_water_bodies,
    rasterize_lakes_to_mask,
    identify_outlet_cells,
    create_lake_flow_routing,
)

# Import flow functions
from src.terrain.flow_accumulation import (
    compute_flow_direction,
    compute_drainage_area,
    compute_upstream_rainfall,
    condition_dem,
    detect_ocean_mask,
)

import rasterio
from rasterio import Affine
from scipy.ndimage import zoom


def load_real_water_bodies(
    bbox: tuple,
    data_source: str,
    output_dir: Path,
    min_area_km2: float = 0.01,
) -> tuple:
    """
    Download and load real water body data from NHD or HydroLAKES.

    Parameters
    ----------
    bbox : tuple
        Bounding box (south, west, north, east) in degrees
    data_source : str
        "nhd" for National Hydrography Dataset (USA only)
        "hydrolakes" for global HydroLAKES data
    output_dir : Path
        Directory for cached data
    min_area_km2 : float
        Minimum lake area in km²

    Returns
    -------
    tuple
        (lakes_geojson, outlets_dict)
    """
    print(f"\n  Downloading {data_source.upper()} water body data...")
    print(f"    Bounding box: {bbox}")
    print(f"    Min area: {min_area_km2} km²")

    # Download water bodies
    geojson_path = download_water_bodies(
        bbox=bbox,
        output_dir=str(output_dir / "water_bodies"),
        data_source=data_source,
        min_area_km2=min_area_km2,
        force_download=False,
    )

    # Load GeoJSON
    import json
    with open(geojson_path, 'r') as f:
        lakes_geojson = json.load(f)

    num_lakes = len(lakes_geojson.get("features", []))
    print(f"    Loaded {num_lakes} water bodies from {geojson_path.name}")

    # Extract outlets from GeoJSON properties
    outlets_dict = {}
    for idx, feature in enumerate(lakes_geojson["features"], start=1):
        props = feature.get("properties", {})

        # Try different outlet field names
        outlet = None
        if "outlet" in props:
            outlet = props["outlet"]
        elif "pour_point" in props:
            outlet = props["pour_point"]
        elif "Pour_long" in props and "Pour_lat" in props:
            outlet = [props["Pour_long"], props["Pour_lat"]]

        if outlet and isinstance(outlet, (list, tuple)) and len(outlet) == 2:
            outlets_dict[idx] = tuple(outlet)
        else:
            # If no outlet specified, use centroid (not ideal but workable)
            geom = feature.get("geometry", {})
            if geom.get("type") == "Polygon" and geom.get("coordinates"):
                coords = geom["coordinates"][0]  # Exterior ring
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                centroid = (sum(lons)/len(lons), sum(lats)/len(lats))
                outlets_dict[idx] = centroid

    print(f"    Identified {len(outlets_dict)} lake outlets")

    return lakes_geojson, outlets_dict, bbox


def main():
    parser = argparse.ArgumentParser(description="Flow validation with real water body data")
    parser.add_argument('--bigness', type=str, default='small',
                        choices=['small', 'medium', 'large', 'full'],
                        help='Size of DEM to process (default: small)')
    parser.add_argument('--data-source', type=str, default='nhd',
                        choices=['nhd', 'hydrolakes'],
                        help='Water body data source (default: nhd)')
    parser.add_argument('--min-area-km2', type=float, default=0.01,
                        help='Minimum lake area in km² (default: 0.01)')
    parser.add_argument('--target-size', type=int, default=None,
                        help='Target resolution for full mode (default: 1000)')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output directory')
    parser.add_argument('--fill-method', type=str, default='breach',
                        choices=['fill', 'breach'],
                        help='Depression filling method (default: breach)')
    parser.add_argument('--min-basin-depth', type=float, default=5.0,
                        help='Minimum basin depth (m) to preserve (default: 5.0)')
    parser.add_argument('--min-basin-size', type=int, default=50000,
                        help='Minimum basin size (cells) to preserve (default: 50000)')
    args = parser.parse_args()

    # Override target size for full mode if specified
    if args.target_size and args.bigness == 'full':
        BIGNESS_SIZES['full'] = args.target_size

    # Set output directory
    if args.output is None:
        args.output = Path(f'examples/output/flow_{args.data_source}_{args.bigness}')

    args.output.mkdir(parents=True, exist_ok=True)

    # Find DEM file
    dem_path = Path("examples/output/merged_dem.tif")
    if not dem_path.exists():
        dem_path = Path("examples/output/flow_data/merged_dem.tif")
    if not dem_path.exists():
        print("ERROR: No DEM found. Run san_diego_flow_demo.py first.")
        return 1

    print(f"Loading DEM from {dem_path}...")
    with rasterio.open(dem_path) as src:
        full_dem = src.read(1)
        dem_transform_full = src.transform
        dem_crs = src.crs
        print(f"Full DEM: {full_dem.shape}, range {full_dem.min():.0f}-{full_dem.max():.0f}m")

    # Select subset based on bigness
    target_size = BIGNESS_SIZES[args.bigness]

    if args.bigness == 'full':
        original_shape = full_dem.shape
        scale_factor = target_size / max(original_shape)

        if scale_factor < 1.0:
            print(f"\nDownsampling FULL DEM from {original_shape} to ~{target_size}x{target_size}...")
            dem = zoom(full_dem, scale_factor, order=1)
            dem_transform = dem_transform_full * Affine.scale(1/scale_factor)
            print(f"  Downsampled to: {dem.shape[0]}x{dem.shape[1]}")
        else:
            dem = full_dem.copy()
            dem_transform = dem_transform_full
            print(f"\nUsing FULL DEM ({dem.shape[0]}x{dem.shape[1]})...")
    else:
        dem, row, col = find_coastal_subset(full_dem, target_size)
        dem_transform = dem_transform_full * Affine.translation(col, row)
        print(f"\nUsing {args.bigness} subset ({target_size}x{target_size}):")
        print(f"  Location: rows {row}-{row+target_size}, cols {col}-{col+target_size}")

    ocean_preview = 100 * np.sum(dem <= 0) / dem.size
    print(f"  Ocean coverage: ~{ocean_preview:.0f}%")
    print(f"  Elevation range: {dem.min():.0f} - {dem.max():.0f}m")

    # Calculate bounding box from DEM
    rows, cols = dem.shape
    west, north = dem_transform * (0, 0)
    east, south = dem_transform * (cols, rows)
    bbox = (south, west, north, east)

    print(f"\n{'='*60}")
    print(f"RUNNING FLOW PIPELINE WITH {args.data_source.upper()} DATA")
    print(f"{'='*60}")

    # Step 1: Detect ocean
    print("\n1. Detecting ocean...")
    ocean_mask = detect_ocean_mask(dem, threshold=0.0, border_only=True)
    print(f"   Ocean cells: {np.sum(ocean_mask):,} ({100*np.sum(ocean_mask)/dem.size:.1f}%)")

    # Step 2: Load real water bodies
    print(f"\n2. Loading {args.data_source.upper()} water bodies...")
    try:
        lakes_geojson, outlets_dict, bbox = load_real_water_bodies(
            bbox=bbox,
            data_source=args.data_source,
            output_dir=args.output,
            min_area_km2=args.min_area_km2,
        )

        if len(lakes_geojson["features"]) == 0:
            print("   WARNING: No water bodies found in bounding box.")
            lake_mask = None
            lake_outlets = None
        else:
            # Rasterize lakes to mask
            print("\n3. Rasterizing water bodies...")
            resolution = abs(dem_transform.a)  # pixel width in degrees
            lake_mask_raw, lake_transform = rasterize_lakes_to_mask(
                lakes_geojson, bbox, resolution=resolution
            )
            print(f"   Lake mask (raw): {lake_mask_raw.shape}")

            # Resample to match DEM shape
            if lake_mask_raw.shape != dem.shape:
                print(f"   Resampling from {lake_mask_raw.shape} to {dem.shape}")
                scale_y = dem.shape[0] / lake_mask_raw.shape[0]
                scale_x = dem.shape[1] / lake_mask_raw.shape[1]
                lake_mask = zoom(lake_mask_raw, (scale_y, scale_x), order=0)
            else:
                lake_mask = lake_mask_raw

            print(f"   Lake mask (final): {lake_mask.shape}")
            print(f"   Unique lakes: {len(np.unique(lake_mask[lake_mask > 0]))}")

            # Identify outlet cells
            print("\n4. Identifying lake outlets...")
            lake_outlets_raw = identify_outlet_cells(lake_mask_raw, outlets_dict, lake_transform)

            # Resample outlets
            if lake_outlets_raw.shape != dem.shape:
                scale_y = dem.shape[0] / lake_outlets_raw.shape[0]
                scale_x = dem.shape[1] / lake_outlets_raw.shape[1]
                lake_outlets = zoom(lake_outlets_raw.astype(np.uint8), (scale_y, scale_x), order=0).astype(bool)
            else:
                lake_outlets = lake_outlets_raw

            print(f"   Outlet cells: {np.sum(lake_outlets)}")

    except Exception as e:
        print(f"   ERROR loading water bodies: {e}")
        print(f"   Continuing without lake routing...")
        lake_mask = None
        lake_outlets = None

    # Continue with rest of pipeline (same as synthetic version)
    # Step 5: Condition DEM
    print(f"\n5. Conditioning DEM (method={args.fill_method})...")
    dem_conditioned = condition_dem(
        dem, method=args.fill_method, ocean_mask=ocean_mask,
        min_basin_size=args.min_basin_size, min_basin_depth=args.min_basin_depth
    )

    # Step 6: Compute flow direction WITHOUT lakes
    print("\n6. Computing flow direction (without lakes)...")
    flow_dir_no_lakes = compute_flow_direction(dem_conditioned, mask=ocean_mask)

    # Compute drainage area without lakes
    print("\n7. Computing drainage area (without lakes)...")
    drainage_area_no_lakes = compute_drainage_area(flow_dir_no_lakes)

    # Step 7: Apply lake routing
    if lake_mask is not None and lake_outlets is not None:
        print("\n8. Applying lake flow routing...")
        lake_flow = create_lake_flow_routing(lake_mask, lake_outlets, dem_conditioned)
        flow_dir = np.where(lake_mask > 0, lake_flow, flow_dir_no_lakes)

        num_lakes = len(np.unique(lake_mask[lake_mask > 0]))
        num_outlets = np.sum(lake_outlets)
        lake_cells = np.sum(lake_mask > 0)
        print(f"   Routed {num_lakes} lakes ({lake_cells:,} cells) with {num_outlets} outlets")
    else:
        print("\n8. Skipping lake routing (no lakes detected)")
        flow_dir = flow_dir_no_lakes

    # Step 8: Compute drainage area WITH lakes
    print("\n9. Computing drainage area (with lakes)...")
    drainage_area = compute_drainage_area(flow_dir)

    # Step 9: Compute upstream rainfall
    print("\n10. Computing upstream rainfall...")
    precip, is_real_precip = load_real_precipitation(dem.shape, args.output)

    if not is_real_precip:
        print("  No real precipitation data, using synthetic")
        precip = 200 + (dem - dem.min()) / (dem.max() - dem.min() + 1) * 300
        precip = precip.astype(np.float32)

    precip[ocean_mask] = 0
    upstream_rainfall = compute_upstream_rainfall(flow_dir, precip)

    # Step 10: Validate
    print("\n11. Validating...")
    cycles, sample_size = validate_no_cycles(flow_dir, sample_size=1000)
    mass_balance = compute_mass_balance(flow_dir, drainage_area)
    drainage_violations, violation_mask = validate_drainage_connectivity(flow_dir, drainage_area)

    print(f"   Cycles: {cycles}/{sample_size}")
    print(f"   Mass Balance: {mass_balance:.1f}%")
    print(f"   Drainage violations: {drainage_violations:,} cells")

    trace_flow_paths(flow_dir, drainage_area, num_traces=5)

    status = "PASS" if cycles == 0 and mass_balance > 95 else "FAIL"
    if drainage_violations > 0:
        status = "WARN" if status == "PASS" else status
        print(f"   WARNING: {drainage_violations} cells flow to neighbors with LESS drainage")
    print(f"\n   Status: {status}")

    # Step 11: Generate images
    print("\n12. Generating validation images...")
    output_dir = create_validation_images(
        dem=dem,
        dem_conditioned=dem_conditioned,
        ocean_mask=ocean_mask,
        flow_dir=flow_dir,
        drainage_area=drainage_area,
        upstream_rainfall=upstream_rainfall,
        precip=precip,
        output_dir=args.output,
        cycles=cycles,
        sample_size=sample_size,
        mass_balance=mass_balance,
        bigness=args.bigness,
        is_real_precip=is_real_precip,
        drainage_violations=drainage_violations,
        violation_mask=violation_mask,
        lake_mask=lake_mask,
        lake_outlets=lake_outlets,
        flow_dir_no_lakes=flow_dir_no_lakes,
        drainage_area_no_lakes=drainage_area_no_lakes,
    )

    print(f"\n{'='*60}")
    print(f"VALIDATION COMPLETE: {status}")
    print(f"{'='*60}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Data source: {args.data_source.upper()}")

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
