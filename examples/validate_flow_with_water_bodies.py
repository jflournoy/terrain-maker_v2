#!/usr/bin/env python3
"""
Flow validation with water bodies: Test lake flow routing integration.

This script extends validate_flow_complete.py to test water body handling:
1. Loads a DEM and detects ocean
2. Downloads/loads water body polygons (lakes, reservoirs)
3. Rasterizes lake polygons and identifies outlets
4. Computes flow direction with lake routing
5. Validates flow correctness and generates visualizations

Water bodies are detected using synthetic lakes created from low-lying flat areas
for testing purposes. In production, use real NHD or HydroLAKES data.

Usage:
    python examples/validate_flow_with_water_bodies.py                    # Small subset (200x200)
    python examples/validate_flow_with_water_bodies.py --bigness medium   # Medium subset (500x500)
    python examples/validate_flow_with_water_bodies.py --bigness large    # Large subset (1000x1000)
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import rasterio
from rasterio import Affine
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import flow functions
from src.terrain.flow_accumulation import (
    compute_flow_direction,
    compute_drainage_area,
    detect_ocean_mask,
    compute_flow_with_basins,
    D8_OFFSETS,
)

# Import water body functions
from src.terrain.water_bodies import (
    rasterize_lakes_to_mask,
    identify_outlet_cells,
)

# Import visualization functions from library
from src.terrain.visualization.flow_diagnostics import (
    save_flow_plot,
    FLOW_COLORMAPS,
    D8_VECTORS,
)

# Import validation functions from validate_flow_complete
from validate_flow_complete import (
    validate_no_cycles,
    compute_mass_balance,
    validate_drainage_connectivity,
    trace_flow_paths,
    find_coastal_subset,
    load_real_precipitation,
    BIGNESS_SIZES,
)


def create_synthetic_lakes(dem: np.ndarray, dem_transform: Affine) -> tuple:
    """
    Create synthetic lake masks directly from low-lying flat areas.

    Instead of creating polygons and rasterizing, this directly returns
    a labeled mask with the actual irregular lake shapes detected in the DEM.

    In a real scenario, you would use download_water_bodies() to get NHD or
    HydroLAKES data. This function creates test lakes for demonstration.

    Parameters
    ----------
    dem : np.ndarray
        Digital elevation model
    dem_transform : Affine
        Geographic transform for DEM

    Returns
    -------
    tuple
        (lake_mask, outlets_dict)
    """
    from scipy.ndimage import label, binary_dilation

    rows, cols = dem.shape

    # Find low-lying flat areas (candidate lakes)
    # Use areas between -5m and 50m elevation that are relatively flat
    low_areas = (dem > -5) & (dem < 50)

    # Find flat regions (small elevation variance)
    from scipy.ndimage import uniform_filter
    window_size = 5
    local_mean = uniform_filter(dem.astype(float), size=window_size)
    local_var = uniform_filter((dem.astype(float) - local_mean)**2, size=window_size)
    flat_areas = local_var < 2.0  # Variance < 2m²

    # Combine: low AND flat
    lake_candidates = low_areas & flat_areas

    # Label connected regions
    structure = np.ones((3, 3), dtype=bool)
    labeled, num_features = label(lake_candidates, structure=structure)

    # Filter by size (keep lakes with 20-500 cells)
    min_size = 20
    max_size = 500

    # Create lake mask with relabeled IDs
    lake_mask = np.zeros_like(dem, dtype=np.uint16)
    outlets_dict = {}

    lake_id = 1
    for region_id in range(1, num_features + 1):
        mask = labeled == region_id
        size = np.sum(mask)

        if size < min_size or size > max_size:
            continue

        # Find outlet: lowest elevation point on perimeter
        # Expand region by 1 pixel to find boundary
        expanded = binary_dilation(mask, structure=structure)
        boundary = expanded & ~mask

        if not np.any(boundary):
            continue

        # Find lowest point on boundary
        boundary_rows, boundary_cols = np.where(boundary)
        boundary_elevs = dem[boundary_rows, boundary_cols]
        outlet_idx = np.argmin(boundary_elevs)
        outlet_row, outlet_col = boundary_rows[outlet_idx], boundary_cols[outlet_idx]

        # Convert outlet pixel to geographic coordinates
        outlet_lon, outlet_lat = dem_transform * (outlet_col, outlet_row)

        # Add lake to mask with new ID
        lake_mask[mask] = lake_id
        outlets_dict[lake_id] = (outlet_lon, outlet_lat)
        lake_id += 1

    num_lakes = lake_id - 1
    print(f"  Created {num_lakes} synthetic lakes for testing")

    return lake_mask, outlets_dict


def create_validation_images(
    dem: np.ndarray,
    dem_conditioned: np.ndarray,
    ocean_mask: np.ndarray,
    flow_dir: np.ndarray,
    drainage_area: np.ndarray,
    upstream_rainfall: np.ndarray,
    precip: np.ndarray,
    output_dir: Path,
    cycles: int,
    sample_size: int,
    mass_balance: float,
    bigness: str,
    is_real_precip: bool = False,
    drainage_violations: int = 0,
    violation_mask: np.ndarray = None,
    # Water body specific parameters
    lake_mask: np.ndarray = None,
    lake_outlets: np.ndarray = None,
    lake_inlets: np.ndarray = None,
    flow_dir_no_lakes: np.ndarray = None,
    drainage_area_no_lakes: np.ndarray = None,
    # Basin preservation parameters
    basin_mask: np.ndarray = None,
    num_basins: int = 0,
):
    """Generate visualization images including water body integration."""

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving images to: {output_dir}")

    # 1. Original DEM
    save_flow_plot(
        dem, f'1. Original DEM ({dem.shape[0]}x{dem.shape[1]})',
        output_dir / '01_dem_original.png',
        cmap=FLOW_COLORMAPS["elevation"], label='Elevation (m)'
    )

    # 2. Ocean Mask
    ocean_pct = 100 * np.sum(ocean_mask) / ocean_mask.size
    save_flow_plot(
        ocean_mask.astype(float), f'2. Ocean Mask ({ocean_pct:.1f}% ocean)',
        output_dir / '02_ocean_mask.png',
        cmap=FLOW_COLORMAPS["binary"], label='Ocean (1) / Land (0)'
    )

    # 3. Water Bodies (NEW)
    if lake_mask is not None:
        num_lakes = len(np.unique(lake_mask[lake_mask > 0]))
        lake_pct = 100 * np.sum(lake_mask > 0) / lake_mask.size

        # Create visualization with lakes and outlets
        fig, ax = plt.subplots(figsize=(12, 10))

        # Show elevation as background
        im_dem = ax.imshow(dem, cmap=FLOW_COLORMAPS["elevation"], alpha=0.6)

        # Overlay lakes in blue
        lake_display = np.ma.masked_where(lake_mask == 0, lake_mask.astype(float))
        im_lakes = ax.imshow(lake_display, cmap='Blues', alpha=0.7, vmin=0, vmax=num_lakes)

        # Mark outlets with arrows pointing outward (red)
        if lake_outlets is not None and np.any(lake_outlets):
            outlet_rows, outlet_cols = np.where(lake_outlets)
            outlet_dirs = flow_dir[outlet_rows, outlet_cols]

            # Extract flow vectors for outlets
            outlet_u = np.zeros(len(outlet_rows), dtype=float)
            outlet_v = np.zeros(len(outlet_rows), dtype=float)

            for i, d8_code in enumerate(outlet_dirs):
                if d8_code in D8_VECTORS:
                    outlet_u[i], outlet_v[i] = D8_VECTORS[int(d8_code)]

            # Draw outlet arrows (red, pointing outward)
            ax.quiver(outlet_cols, outlet_rows, outlet_u, outlet_v,
                     color='red', scale=80, scale_units='xy', width=0.001,
                     label=f'Outlets ({len(outlet_rows)})')

        # Mark inlets with arrows pointing inward (green)
        if lake_inlets is not None and np.any(lake_inlets):
            inlet_rows, inlet_cols = np.where(lake_inlets)

            # For inlets, reverse the arrow direction (pointing into the lake)
            inlet_dirs = flow_dir[inlet_rows, inlet_cols]

            inlet_u = np.zeros(len(inlet_rows), dtype=float)
            inlet_v = np.zeros(len(inlet_rows), dtype=float)

            for i, d8_code in enumerate(inlet_dirs):
                if d8_code in D8_VECTORS:
                    # Reverse direction for inlet arrows
                    u, v = D8_VECTORS[int(d8_code)]
                    inlet_u[i], inlet_v[i] = -u, -v

            # Draw inlet arrows (green, pointing inward)
            ax.quiver(inlet_cols, inlet_rows, inlet_u, inlet_v,
                     color='green', scale=80, scale_units='xy', width=0.001,
                     label=f'Inlets ({len(inlet_rows)})')

        plt.colorbar(im_dem, ax=ax, label='Elevation (m)', shrink=0.7)
        ax.set_title(f'3. Water Bodies ({num_lakes} lakes, {lake_pct:.2f}% of area)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        if (lake_outlets is not None and np.any(lake_outlets)) or (lake_inlets is not None and np.any(lake_inlets)):
            ax.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(output_dir / '03_water_bodies.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: 03_water_bodies.png")

    # 3b. Endorheic Basins (NEW)
    if basin_mask is not None and np.any(basin_mask):
        basin_coverage = 100 * np.sum(basin_mask) / basin_mask.size

        fig, ax = plt.subplots(figsize=(12, 10))

        # Show elevation as background
        im_dem = ax.imshow(dem, cmap=FLOW_COLORMAPS["elevation"], alpha=0.6)

        # Overlay basins in purple/magenta
        basin_display = np.ma.masked_where(~basin_mask, basin_mask.astype(float))
        ax.imshow(basin_display, cmap='Purples', alpha=0.7, vmin=0, vmax=1)

        plt.colorbar(im_dem, ax=ax, label='Elevation (m)', shrink=0.8)
        ax.set_title(f'3b. Detected Endorheic Basins ({basin_coverage:.2f}% of area)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')

        plt.tight_layout()
        plt.savefig(output_dir / '03b_endorheic_basins.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: 03b_endorheic_basins.png")

    # 4. Conditioned DEM
    save_flow_plot(
        dem_conditioned, '4. Conditioned DEM (depression-filled)',
        output_dir / '04_dem_conditioned.png',
        cmap=FLOW_COLORMAPS["elevation"], label='Elevation (m)'
    )

    # 5. Fill Depth
    fill_depth = dem_conditioned - dem
    fill_depth[ocean_mask] = 0
    fill_max = np.max(fill_depth)
    save_flow_plot(
        fill_depth, f'5. Depression Fill Depth (max={fill_max:.1f}m)',
        output_dir / '05_fill_depth.png',
        cmap=FLOW_COLORMAPS["fill"], label='m', log_scale=True
    )

    # 6. Flow Direction
    flow_display = flow_dir.astype(float)
    flow_display[flow_dir == 0] = np.nan
    save_flow_plot(
        flow_display, '6. Flow Direction (D8 codes)',
        output_dir / '06_flow_direction.png',
        cmap=FLOW_COLORMAPS["direction"], label='D8 code (1-128)'
    )

    # 7. Drainage Area
    lake_overlay_mask = (lake_mask > 0) if lake_mask is not None else None
    save_flow_plot(
        drainage_area, '7. Drainage Area (log scale)',
        output_dir / '07_drainage_area.png',
        cmap=FLOW_COLORMAPS["drainage"], label='cells', log_scale=True,
        overlay_data=lake_overlay_mask, overlay_cmap='Blues'
    )

    # 8. Stream Network with lakes highlighted
    # Use top 5% drainage area threshold for stream visualization
    stream_threshold = np.percentile(drainage_area[drainage_area > 0], 95)
    streams = drainage_area >= stream_threshold

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(dem, cmap=FLOW_COLORMAPS["elevation"], alpha=0.6)

    # Overlay streams
    stream_overlay = np.ma.masked_where(~streams, streams.astype(float))
    ax.imshow(stream_overlay, cmap=FLOW_COLORMAPS["streams"], alpha=0.9)

    # Overlay lakes in blue
    if lake_mask is not None:
        lake_overlay = np.ma.masked_where(lake_mask == 0, np.ones_like(lake_mask, dtype=float))
        ax.imshow(lake_overlay, cmap='Blues', alpha=0.5)

    stream_count = np.sum(streams)
    plt.colorbar(im, ax=ax, label='Elevation (m)', shrink=0.8)
    ax.set_title(f'8. Stream Network with Lakes (top 5%, {stream_count:,} cells)',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.tight_layout()
    plt.savefig(output_dir / '08_stream_network_with_lakes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 08_stream_network_with_lakes.png")

    # 9. Precipitation
    if is_real_precip:
        precip_title = '9. WorldClim Annual Precipitation'
        precip_label = 'mm/year'
    else:
        precip_title = '9. Synthetic Precipitation (elevation-based)'
        precip_label = 'mm/year (synthetic)'
    save_flow_plot(
        precip, precip_title,
        output_dir / '09_precipitation.png',
        cmap=FLOW_COLORMAPS["precip"], label=precip_label
    )

    # 10. Upstream Rainfall
    save_flow_plot(
        upstream_rainfall, '10. Upstream Rainfall',
        output_dir / '10_upstream_rainfall.png',
        cmap=FLOW_COLORMAPS["rainfall"], label='mm·m²', log_scale=True,
        overlay_data=lake_overlay_mask, overlay_cmap='Blues'
    )

    # 10b. Discharge Potential (rainfall-weighted drainage, log scale)
    # Combines drainage area with upstream rainfall to show where biggest flows occur
    mean_rainfall = np.mean(upstream_rainfall[upstream_rainfall > 0]) if np.any(upstream_rainfall > 0) else 1.0
    discharge_potential = drainage_area.astype(np.float32) * (upstream_rainfall / mean_rainfall)
    save_flow_plot(
        discharge_potential, '10b. Discharge Potential - Log Scale (drainage × rainfall)',
        output_dir / '10b_discharge_potential_log.png',
        cmap=FLOW_COLORMAPS["rainfall"], label='Discharge Index', log_scale=True,
        overlay_data=lake_overlay_mask, overlay_cmap='Blues'
    )

    # 10c. Discharge Potential (linear scale)
    # Same metric as 10b but linear scale to show absolute discharge magnitudes
    save_flow_plot(
        discharge_potential, '10c. Discharge Potential - Linear Scale (drainage × rainfall)',
        output_dir / '10c_discharge_potential_linear.png',
        cmap=FLOW_COLORMAPS["rainfall"], label='Discharge Index', log_scale=False,
        overlay_data=lake_overlay_mask, overlay_cmap='Blues'
    )

    # 11. Validation Summary
    status = "PASS" if cycles == 0 and mass_balance > 95 and drainage_violations == 0 else "FAIL"
    if cycles == 0 and mass_balance > 95 and drainage_violations > 0:
        status = "WARN"

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')

    violation_text = f"{drainage_violations:>8,}"
    if drainage_violations > 0:
        violation_text = f"{drainage_violations:>8,} ⚠"

    num_lakes = len(np.unique(lake_mask[lake_mask > 0])) if lake_mask is not None else 0
    num_outlets = np.sum(lake_outlets) if lake_outlets is not None else 0
    basin_cells = np.sum(basin_mask) if basin_mask is not None and np.any(basin_mask) else 0

    summary_text = f"""
    ╔══════════════════════════════════════════════════════════╗
    ║         FLOW VALIDATION WITH WATER BODIES                ║
    ╠══════════════════════════════════════════════════════════╣
    ║                                                          ║
    ║  Status:        {status:>8}                                  ║
    ║                                                          ║
    ║  Cycles:        {cycles:>8} / {sample_size:<8}                     ║
    ║  Mass Balance:  {mass_balance:>8.1f}%                              ║
    ║  Drain. Viol.:  {violation_text}                            ║
    ║                                                          ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Water Bodies & Basins                                   ║
    ╠══════════════════════════════════════════════════════════╣
    ║                                                          ║
    ║  Lakes:         {num_lakes:>8}                                  ║
    ║  Outlets:       {num_outlets:>8}                                  ║
    ║  Endor. Basins: {num_basins:>8}  ({basin_cells:,} cells)     ║
    ║                                                          ║
    ╠══════════════════════════════════════════════════════════╣
    ║  DEM Statistics                                          ║
    ╠══════════════════════════════════════════════════════════╣
    ║                                                          ║
    ║  Bigness:       {bigness:>8}                                  ║
    ║  Size:          {dem.shape[0]:>4} x {dem.shape[1]:<4} pixels                    ║
    ║  Total Cells:   {dem.size:>12,}                          ║
    ║  Ocean Cells:   {np.sum(ocean_mask):>12,} ({100*np.sum(ocean_mask)/ocean_mask.size:.1f}%)             ║
    ║                                                          ║
    ║  Elevation:     {dem.min():>8.0f} - {dem.max():<8.0f} m                  ║
    ║  Max Drainage:  {np.max(drainage_area):>12,.0f} cells                 ║
    ║  Max Upstream:  {np.max(upstream_rainfall):>12,.0f} mm·m²                ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """

    color = 'darkgreen' if status == "PASS" else ('darkorange' if status == "WARN" else 'darkred')
    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor=color, linewidth=3))

    plt.savefig(output_dir / '11_validation_summary.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"  Saved: 11_validation_summary.png")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Flow validation with water bodies")
    parser.add_argument('--bigness', type=str, default='small',
                        choices=['small', 'medium', 'large', 'full'],
                        help='Size of DEM to process (default: small)')
    parser.add_argument('--data-source', type=str, default='synthetic',
                        choices=['synthetic', 'nhd', 'hydrolakes'],
                        help='Water body data source: synthetic (default), nhd (USA), hydrolakes (global)')
    parser.add_argument('--min-lake-area', type=float, default=0.01,
                        help='Minimum lake area in km² for real data (default: 0.01)')
    parser.add_argument('--target-size', type=int, default=None,
                        help='Target resolution for full mode (default: 1000)')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output directory (default: examples/output/flow_water_bodies_<bigness>)')
    parser.add_argument('--backend', type=str, default='spec',
                        choices=['legacy', 'spec', 'pysheds'],
                        help='Flow algorithm backend (default: spec)')
    parser.add_argument('--fill-method', type=str, default='breach',
                        choices=['fill', 'breach'],
                        help='Depression filling method (legacy backend only, default: breach)')
    parser.add_argument('--min-basin-depth', type=float, default=5.0,
                        help='Minimum basin depth (m) to preserve (legacy backend only, default: 5.0)')
    parser.add_argument('--min-basin-size', type=int, default=50000,
                        help='Minimum basin size (cells) to preserve (legacy backend only, default: 50000)')
    # Spec backend parameters
    parser.add_argument('--coastal-elev-threshold', type=float, default=0.0,
                        help='Max elevation for coastal outlets in meters (spec backend, default: 0.0)')
    parser.add_argument('--edge-mode', type=str, default='all',
                        choices=['all', 'local_minima', 'outward_slope', 'none'],
                        help='Boundary outlet strategy (spec backend, default: all)')
    parser.add_argument('--max-breach-depth', type=float, default=25.0,
                        help='Max vertical breach per cell in meters (spec backend, default: 25.0)')
    parser.add_argument('--max-breach-length', type=int, default=80,
                        help='Max breach path length in cells (spec backend, default: 80)')
    parser.add_argument('--epsilon', type=float, default=1e-4,
                        help='Min gradient in filled areas (spec backend, default: 1e-4)')
    # Basin detection parameters
    parser.add_argument('--detect-basins', action='store_true', default=True,
                        help='Automatically detect and preserve endorheic basins (default: True)')
    parser.add_argument('--basin-min-size', type=int, default=5000,
                        help='Minimum endorheic basin size in cells to preserve (default: 5000)')
    parser.add_argument('--basin-min-depth', type=float, default=1.0,
                        help='Minimum basin depth in meters to be considered endorheic (default: 1.0)')
    args = parser.parse_args()

    # Override target size for full mode if specified
    if args.target_size and args.bigness == 'full':
        BIGNESS_SIZES['full'] = args.target_size

    # Set output directory
    if args.output is None:
        suffix = f'_{args.data_source}' if args.data_source != 'synthetic' else ''
        args.output = Path(f'examples/output/flow_water_bodies_{args.bigness}{suffix}')

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
            # Adjust transform for downsampling
            dem_transform = dem_transform_full * Affine.scale(1/scale_factor)
            print(f"  Downsampled to: {dem.shape[0]}x{dem.shape[1]}")
        else:
            dem = full_dem.copy()
            dem_transform = dem_transform_full
            print(f"\nUsing FULL DEM ({dem.shape[0]}x{dem.shape[1]})...")
    else:
        dem, row, col = find_coastal_subset(full_dem, target_size)
        # Calculate transform for subset
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
    print(f"RUNNING FLOW PIPELINE WITH {args.data_source.upper()} WATER BODIES")
    print(f"{'='*60}")

    # Step 1: Detect ocean
    print("\n1. Detecting ocean...")
    ocean_mask = detect_ocean_mask(dem, threshold=0.0, border_only=True)
    print(f"   Ocean cells: {np.sum(ocean_mask):,} ({100*np.sum(ocean_mask)/dem.size:.1f}%)")

    # Step 2: Load water bodies (synthetic or real)
    if args.data_source == 'synthetic':
        # Create synthetic water bodies for testing (direct mask, no GeoJSON)
        print("\n2. Creating synthetic water bodies...")
        lake_mask, outlets_dict = create_synthetic_lakes(dem, dem_transform)

        if not np.any(lake_mask > 0):
            print("   WARNING: No water bodies found. Continuing without lake routing.")
            lake_mask = None
            lake_outlets = None
        else:
            print(f"   Lake mask shape: {lake_mask.shape}")
            print(f"   Unique lakes: {len(np.unique(lake_mask[lake_mask > 0]))}")

            # Create outlet mask from outlet coordinates
            print("\n3. Identifying lake outlets...")
            lake_outlets = np.zeros_like(lake_mask, dtype=bool)

            for lake_id, (outlet_lon, outlet_lat) in outlets_dict.items():
                # Convert geographic coordinates to pixel coordinates
                inv_transform = ~dem_transform
                col, row = inv_transform * (outlet_lon, outlet_lat)
                row, col = int(row), int(col)

                # Check bounds
                if 0 <= row < lake_mask.shape[0] and 0 <= col < lake_mask.shape[1]:
                    lake_outlets[row, col] = True

            print(f"   Outlet cells: {np.sum(lake_outlets)}")

            # Identify inlets (water flowing into lakes) - will be done after DEM conditioning
            lake_inlets = None
    else:
        # Load real water bodies from NHD or HydroLAKES
        print(f"\n2. Loading {args.data_source.upper()} water bodies...")
        print(f"   Bounding box: {bbox}")
        print(f"   Min lake area: {args.min_lake_area} km²")

        try:
            from src.terrain.water_bodies import download_water_bodies

            # Download/load water bodies
            geojson_path = download_water_bodies(
                bbox=bbox,
                output_dir=str(args.output / "water_bodies"),
                data_source=args.data_source,
                min_area_km2=args.min_lake_area,
                force_download=False,
            )

            # Load GeoJSON
            import json
            with open(geojson_path, 'r') as f:
                lakes_geojson = json.load(f)

            num_features = len(lakes_geojson.get("features", []))
            print(f"   Loaded {num_features} water bodies from {geojson_path.name}")

            if num_features == 0:
                print("   WARNING: No water bodies found in bounding box.")
                lake_mask = None
                lake_outlets = None
            else:
                # Extract outlets from GeoJSON
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

                print(f"   Found {len(outlets_dict)} lake outlets")

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
            lake_inlets = None

    # Step 3-8: Compute flow using validated pipeline
    step_num = 5 if args.data_source != 'synthetic' else 4
    print(f"\n{step_num}. Computing flow with basin preservation pipeline...")

    # Load precipitation for upstream rainfall computation
    print("  Loading precipitation data...")
    precip, is_real_precip = load_real_precipitation(dem.shape, args.output)
    if not is_real_precip:
        print("  No real precipitation data found, using synthetic (elevation-based)")
        precip = 200 + (dem - dem.min()) / (dem.max() - dem.min() + 1) * 300
        precip = precip.astype(np.float32)

    # Call validated flow pipeline
    flow_results = compute_flow_with_basins(
        dem=dem,
        dem_transform=dem_transform,
        precipitation=precip,
        lake_mask=lake_mask,
        lake_outlets=lake_outlets,
        detect_basins=args.detect_basins,
        min_basin_size=args.basin_min_size,
        min_basin_depth=args.basin_min_depth,
        backend=args.backend,
        coastal_elev_threshold=args.coastal_elev_threshold,
        edge_mode=args.edge_mode,
        max_breach_depth=args.max_breach_depth,
        max_breach_length=args.max_breach_length,
        epsilon=args.epsilon,
        ocean_threshold=0.0,
        ocean_border_only=True,
        verbose=False,  # Suppress verbose output (we have our own progress messages)
    )

    # Extract results
    flow_dir = flow_results["flow_direction"]
    drainage_area = flow_results["drainage_area"]
    dem_conditioned = flow_results["dem_conditioned"]
    basin_mask = flow_results["basin_mask"]
    lake_inlets = flow_results["lake_inlets"]
    upstream_rainfall = flow_results["upstream_rainfall"]

    # Count basins for reporting
    endorheic_basins = {}
    if basin_mask is not None and np.any(basin_mask):
        from scipy.ndimage import label
        labeled_basins, num_basins = label(basin_mask)
        for basin_id in range(1, num_basins + 1):
            endorheic_basins[basin_id] = np.sum(labeled_basins == basin_id)

    # Compute flow WITHOUT lakes for comparison visualizations
    print("  Computing flow without lakes (for comparison)...")
    flow_dir_no_lakes = compute_flow_direction(dem_conditioned, mask=ocean_mask)
    drainage_area_no_lakes = compute_drainage_area(flow_dir_no_lakes)

    # Step 9: Validate
    print("\n10. Validating...")
    cycles, sample_size = validate_no_cycles(flow_dir, sample_size=1000)
    mass_balance = compute_mass_balance(flow_dir, drainage_area)
    drainage_violations, violation_mask = validate_drainage_connectivity(flow_dir, drainage_area)

    print(f"   Cycles: {cycles}/{sample_size}")
    print(f"   Mass Balance: {mass_balance:.1f}%")
    print(f"   Drainage violations: {drainage_violations:,} cells")

    # Trace flow paths
    trace_flow_paths(flow_dir, drainage_area, num_traces=5)

    status = "PASS" if cycles == 0 and mass_balance > 95 else "FAIL"
    if drainage_violations > 0:
        status = "WARN" if status == "PASS" else status
        print(f"   WARNING: {drainage_violations} cells flow to neighbors with LESS drainage")
    print(f"\n   Status: {status}")

    # Step 11: Generate images
    print("\n11. Generating validation images...")
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
        # Water body specific
        lake_mask=lake_mask,
        lake_outlets=lake_outlets,
        lake_inlets=lake_inlets,
        flow_dir_no_lakes=flow_dir_no_lakes,
        drainage_area_no_lakes=drainage_area_no_lakes,
        # Basin preservation
        basin_mask=basin_mask,
        num_basins=len(endorheic_basins),
    )

    print(f"\n{'='*60}")
    print(f"VALIDATION COMPLETE: {status}")
    print(f"{'='*60}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Images generated:")
    for img in sorted(output_dir.glob('*.png')):
        print(f"  - {img.name}")

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
