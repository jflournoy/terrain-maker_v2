"""
Validate flow accumulation pipeline with individual plots per stage.

Outputs separate PNG files for each stage:
1. dem_original.png - Raw elevation
2. dem_conditioned.png - After depression filling
3. fill_depth.png - Where depressions were filled
4. flow_direction.png - D8 flow directions
5. drainage_area.png - Unweighted accumulation
6. drainage_histogram.png - Distribution
7. upstream_rainfall.png - Precipitation-weighted
8. precipitation_input.png - Local rainfall
9. validation.png - Correctness check
"""

import sys
from pathlib import Path
import numpy as np

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))


def save_individual_plots(output_dir: Path):
    """
    Create individual validation plots for each stage.

    Parameters
    ----------
    output_dir : Path
        Output directory for individual plots
    """
    import rasterio
    from rasterio.warp import reproject, Resampling

    # Input paths
    flow_dir_path = "examples/output/flow_outputs/flow_direction.tif"
    drainage_path = "examples/output/flow_outputs/flow_accumulation_area.tif"
    rainfall_path = "examples/output/flow_outputs/flow_accumulation_rainfall.tif"
    dem_cond_path = "examples/output/flow_outputs/dem_conditioned.tif"
    dem_orig_path = "examples/output/merged_dem.tif"

    # Auto-detect precipitation file (WorldClim or PRISM)
    from pathlib import Path
    output_dir = Path('examples/output')
    precip_files = list(output_dir.glob('worldclim_annual_precip*.tif'))
    if not precip_files:
        precip_files = list(output_dir.glob('prism_annual_precip*.tif'))
    if precip_files:
        precip_path = str(precip_files[0])
    else:
        raise FileNotFoundError("No precipitation file found. Run san_diego_flow_demo.py first.")

    print("Loading pre-computed flow outputs...")
    print(f"Using precipitation file: {precip_path}")

    # Load flow outputs (downsampled resolution)
    with rasterio.open(flow_dir_path) as src:
        flow_dir = src.read(1)
        flow_transform = src.transform
        flow_crs = src.crs

    with rasterio.open(drainage_path) as src:
        drainage_area = src.read(1)

    with rasterio.open(rainfall_path) as src:
        upstream_rainfall = src.read(1)

    with rasterio.open(dem_cond_path) as src:
        dem_conditioned = src.read(1)

    # Load original data (full resolution)
    with rasterio.open(dem_orig_path) as src:
        dem_full = src.read(1).astype(np.float32)
        dem_transform = src.transform

    with rasterio.open(precip_path) as src:
        precip_full = src.read(1).astype(np.float32)
        precip_transform = src.transform

    print(f"Flow output shape: {dem_conditioned.shape}")

    # Resample to match flow resolution
    flow_shape = dem_conditioned.shape
    dem_original = np.empty(flow_shape, dtype=np.float32)
    precip_data = np.empty(flow_shape, dtype=np.float32)

    reproject(
        dem_full, dem_original,
        src_transform=dem_transform,
        dst_transform=flow_transform,
        src_crs=flow_crs,
        dst_crs=flow_crs,
        resampling=Resampling.bilinear
    )

    reproject(
        precip_full, precip_data,
        src_transform=precip_transform,
        dst_transform=flow_transform,
        src_crs=flow_crs,
        dst_crs=flow_crs,
        resampling=Resampling.bilinear
    )

    print(f"Data aligned: {dem_original.shape}")

    # Calculate statistics
    dem_diff = dem_conditioned - dem_original
    filled_cells = np.sum(dem_diff > 0)
    validation = upstream_rainfall >= precip_data
    validation_pct = 100 * np.sum(validation) / validation.size

    print(f"Filled {filled_cells:,} cells ({100 * filled_cells / dem_original.size:.2f}%)")
    print(f"Validation: {validation_pct:.1f}% pass")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Original DEM
    print("1/9: Original DEM...")
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(dem_original, cmap='terrain', interpolation='nearest')
    ax.set_title('Original DEM', fontsize=14, fontweight='bold')
    ax.set_xlabel('Column (pixels)')
    ax.set_ylabel('Row (pixels)')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Elevation (m)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / '1_dem_original.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Conditioned DEM
    print("2/9: Conditioned DEM...")
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(dem_conditioned, cmap='terrain', interpolation='nearest')
    ax.set_title('Conditioned DEM (Depression-Filled)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Column (pixels)')
    ax.set_ylabel('Row (pixels)')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Elevation (m)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / '2_dem_conditioned.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Fill depth
    print("3/9: Fill depth...")
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(dem_diff, cmap='hot', interpolation='nearest', vmin=0)
    ax.set_title(f'Fill Depth ({filled_cells:,} cells filled, {100 * filled_cells / dem_original.size:.1f}%)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Column (pixels)')
    ax.set_ylabel('Row (pixels)')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Fill depth (m)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / '3_fill_depth.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 4: Flow direction
    print("4/9: Flow direction...")
    fig, ax = plt.subplots(figsize=(10, 8))
    flow_dir_vis = flow_dir.astype(np.float32)
    flow_dir_vis[flow_dir == 0] = np.nan
    im = ax.imshow(flow_dir_vis, cmap='hsv', interpolation='nearest')
    ax.set_title('Flow Direction (D8 Encoding)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Column (pixels)')
    ax.set_ylabel('Row (pixels)')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Direction code', fontsize=12)
    cbar.set_ticks([1, 2, 4, 8, 16, 32, 64, 128])
    plt.tight_layout()
    plt.savefig(output_dir / '4_flow_direction.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 5: Drainage area (log scale)
    print("5/9: Drainage area...")
    fig, ax = plt.subplots(figsize=(10, 8))
    drainage_log = np.log10(drainage_area + 1)
    im = ax.imshow(drainage_log, cmap='Blues', interpolation='nearest')
    ax.set_title(f'Drainage Area (log scale, max={drainage_area.max():,.0f} cells)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Column (pixels)')
    ax.set_ylabel('Row (pixels)')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('log₁₀(cells)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / '5_drainage_area.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 6: Drainage histogram
    print("6/9: Drainage histogram...")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist(drainage_area.flatten(), bins=100, log=True, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Drainage Area (cells)', fontsize=12)
    ax.set_ylabel('Frequency (log scale)', fontsize=12)
    ax.set_title(f'Drainage Area Distribution (ridgelines={np.sum(drainage_area == 1):,})',
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_dir / '6_drainage_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 7: Upstream rainfall (log scale)
    print("7/9: Upstream rainfall...")
    fig, ax = plt.subplots(figsize=(10, 8))
    # Filter out negative/invalid values before log
    upstream_valid = upstream_rainfall.copy()
    upstream_valid[upstream_valid <= 0] = np.nan
    rainfall_log = np.log10(upstream_valid)
    im = ax.imshow(rainfall_log, cmap='YlGnBu', interpolation='nearest')
    ax.set_title('Upstream Rainfall (log scale)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Column (pixels)')
    ax.set_ylabel('Row (pixels)')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('log₁₀(mm·m²/year)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / '7_upstream_rainfall.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 8: Local precipitation
    print("8/9: Local precipitation...")
    fig, ax = plt.subplots(figsize=(10, 8))
    # Filter out nodata values
    precip_valid = precip_data.copy()
    precip_valid[precip_valid < 0] = np.nan
    im = ax.imshow(precip_valid, cmap='YlGnBu', interpolation='nearest')
    ax.set_title(f'Local Precipitation Input (range={np.nanmin(precip_valid):.0f}-{np.nanmax(precip_valid):.0f} mm/year)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Column (pixels)')
    ax.set_ylabel('Row (pixels)')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('mm/year', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / '8_precipitation_input.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 9: Validation
    print("9/9: Validation...")
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(validation, cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=1)
    ax.set_title(f'Validation: Upstream ≥ Local ({validation_pct:.1f}% pass)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Column (pixels)')
    ax.set_ylabel('Row (pixels)')

    # Add statistics text box
    stats_text = (
        f"Statistics:\n"
        f"✓ Validation: {validation_pct:.1f}% pass\n"
        f"✓ Max drainage: {drainage_area.max():,.0f} cells\n"
        f"✓ Ridgelines (area=1): {np.sum(drainage_area == 1):,}\n"
        f"✓ Filled cells: {filled_cells:,}\n"
        f"✓ Max fill depth: {dem_diff.max():.2f} m\n"
        f"✓ All flow codes valid: {np.all((flow_dir >= 0) & (flow_dir <= 128))}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1])
    cbar.set_label('Pass/Fail', fontsize=12)
    cbar.ax.set_yticklabels(['Fail', 'Pass'])
    plt.tight_layout()
    plt.savefig(output_dir / '9_validation.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Saved 9 plots to {output_dir}/")

    # Print summary
    print("\n" + "="*60)
    print("FLOW VALIDATION SUMMARY")
    print("="*60)
    print(f"✓ Validation: {validation_pct:.1f}% pass (upstream ≥ local)")
    print(f"✓ Max drainage area: {drainage_area.max():,.0f} cells")
    print(f"✓ Ridgeline cells (drainage=1): {np.sum(drainage_area == 1):,}")
    print(f"✓ Depressions filled: {filled_cells:,} ({100 * filled_cells / dem_original.size:.2f}%)")
    print(f"✓ Max fill depth: {dem_diff.max():.2f} m")
    print(f"✓ Flow direction codes valid: {np.all((flow_dir >= 0) & (flow_dir <= 128))}")
    print("="*60)
    print("\nPlots saved:")
    for i, name in enumerate([
        "1_dem_original.png",
        "2_dem_conditioned.png",
        "3_fill_depth.png",
        "4_flow_direction.png",
        "5_drainage_area.png",
        "6_drainage_histogram.png",
        "7_upstream_rainfall.png",
        "8_precipitation_input.png",
        "9_validation.png"
    ], 1):
        print(f"  {i}. {output_dir / name}")


def main():
    """Run validation with individual plots."""
    output_dir = Path("examples/output/flow-validation")
    save_individual_plots(output_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
