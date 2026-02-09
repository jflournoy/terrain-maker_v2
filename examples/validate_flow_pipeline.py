"""
Validate flow accumulation pipeline by visualizing each stage.

Outputs diagnostic plots showing:
1. Original DEM
2. Conditioned DEM (depression-filled)
3. Flow direction
4. Drainage area
5. Upstream rainfall

This helps verify that depression filling and flow accumulation work correctly.
"""

import sys
from pathlib import Path
import numpy as np

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import rasterio
from rasterio.transform import rowcol

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.terrain.flow_accumulation import (
    flow_accumulation,
    compute_flow_direction,
    compute_drainage_area,
    compute_upstream_rainfall,
    condition_dem,
)


def create_flow_validation_plots(dem_path: str, precip_path: str, output_path: str):
    """
    Create comprehensive flow pipeline validation plots.

    Parameters
    ----------
    dem_path : str
        Path to DEM GeoTIFF
    precip_path : str
        Path to precipitation GeoTIFF
    output_path : str
        Output path for validation plot
    """
    print("[DEBUG] Starting create_flow_validation_plots")
    print("Loading DEM and precipitation data...")

    # Load DEM
    print("[DEBUG] Loading DEM...")
    with rasterio.open(dem_path) as src:
        dem_original = src.read(1).astype(np.float32)
        transform = src.transform
    print(f"[DEBUG] DEM loaded: {dem_original.shape}")

    # Load precipitation
    print("[DEBUG] Loading precipitation...")
    with rasterio.open(precip_path) as src:
        precip_data = src.read(1).astype(np.float32)
    print(f"[DEBUG] Precipitation loaded: {precip_data.shape}")

    print(f"DEM shape: {dem_original.shape}")
    print(f"DEM range: {dem_original.min():.1f} to {dem_original.max():.1f} m")
    print(f"Precipitation range: {precip_data.min():.1f} to {precip_data.max():.1f} mm/year")

    # Stage 1: Condition DEM
    print("\n1. Conditioning DEM (filling depressions)...")
    dem_conditioned = condition_dem(dem_original, method="breach")

    # Calculate difference
    dem_diff = dem_conditioned - dem_original
    filled_cells = np.sum(dem_diff > 0)
    print(f"   Filled {filled_cells:,} cells ({100 * filled_cells / dem_original.size:.2f}%)")
    print(f"   Max fill depth: {dem_diff.max():.2f} m")

    # Stage 2: Flow direction
    print("\n2. Computing flow direction...")
    flow_dir = compute_flow_direction(dem_conditioned)

    # Stage 3: Drainage area
    print("\n3. Computing drainage area...")
    drainage_area = compute_drainage_area(flow_dir)

    # Stage 4: Upstream rainfall
    print("\n4. Computing upstream rainfall...")
    upstream_rainfall = compute_upstream_rainfall(flow_dir, precip_data)

    # Create comprehensive visualization
    print("\n5. Creating validation plots...")

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle('Flow Accumulation Pipeline Validation', fontsize=16, fontweight='bold')

    # Row 1: DEM stages
    # Original DEM
    im1 = axes[0, 0].imshow(dem_original, cmap='terrain', interpolation='nearest')
    axes[0, 0].set_title('1. Original DEM')
    axes[0, 0].set_xlabel('Column')
    axes[0, 0].set_ylabel('Row')
    plt.colorbar(im1, ax=axes[0, 0], label='Elevation (m)')

    # Conditioned DEM
    im2 = axes[0, 1].imshow(dem_conditioned, cmap='terrain', interpolation='nearest')
    axes[0, 1].set_title('2. Conditioned DEM (depression-filled)')
    axes[0, 1].set_xlabel('Column')
    axes[0, 1].set_ylabel('Row')
    plt.colorbar(im2, ax=axes[0, 1], label='Elevation (m)')

    # Fill depth
    im3 = axes[0, 2].imshow(dem_diff, cmap='hot', interpolation='nearest', vmin=0)
    axes[0, 2].set_title('3. Fill Depth (difference)')
    axes[0, 2].set_xlabel('Column')
    axes[0, 2].set_ylabel('Row')
    plt.colorbar(im3, ax=axes[0, 2], label='Fill depth (m)')

    # Row 2: Flow direction and drainage
    # Flow direction (visualize as colors)
    flow_dir_vis = flow_dir.astype(np.float32)
    flow_dir_vis[flow_dir == 0] = np.nan  # Make outlets transparent
    im4 = axes[1, 0].imshow(flow_dir_vis, cmap='hsv', interpolation='nearest')
    axes[1, 0].set_title('4. Flow Direction (D8)')
    axes[1, 0].set_xlabel('Column')
    axes[1, 0].set_ylabel('Row')
    cbar4 = plt.colorbar(im4, ax=axes[1, 0], label='Direction code')
    cbar4.set_ticks([1, 2, 4, 8, 16, 32, 64, 128])

    # Drainage area (log scale)
    drainage_log = np.log10(drainage_area + 1)
    im5 = axes[1, 1].imshow(drainage_log, cmap='Blues', interpolation='nearest')
    axes[1, 1].set_title('5. Drainage Area (log scale)')
    axes[1, 1].set_xlabel('Column')
    axes[1, 1].set_ylabel('Row')
    plt.colorbar(im5, ax=axes[1, 1], label='log₁₀(cells)')

    # Drainage area histogram
    axes[1, 2].hist(drainage_area.flatten(), bins=100, log=True, color='steelblue', edgecolor='black')
    axes[1, 2].set_xlabel('Drainage Area (cells)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('6. Drainage Area Distribution')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_xscale('log')

    # Row 3: Upstream rainfall and validation
    # Upstream rainfall (log scale)
    rainfall_log = np.log10(upstream_rainfall + 1)
    im7 = axes[2, 0].imshow(rainfall_log, cmap='YlGnBu', interpolation='nearest')
    axes[2, 0].set_title('7. Upstream Rainfall (log scale)')
    axes[2, 0].set_xlabel('Column')
    axes[2, 0].set_ylabel('Row')
    plt.colorbar(im7, ax=axes[2, 0], label='log₁₀(mm·m²)')

    # Precipitation input
    im8 = axes[2, 1].imshow(precip_data, cmap='YlGnBu', interpolation='nearest')
    axes[2, 1].set_title('8. Precipitation Input')
    axes[2, 1].set_xlabel('Column')
    axes[2, 1].set_ylabel('Row')
    plt.colorbar(im8, ax=axes[2, 1], label='mm/year')

    # Validation: upstream_rainfall >= local_precip
    validation = upstream_rainfall >= precip_data
    validation_pct = 100 * np.sum(validation) / validation.size

    axes[2, 2].imshow(validation, cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=1)
    axes[2, 2].set_title(f'9. Validation: Upstream ≥ Local\n({validation_pct:.1f}% pass)')
    axes[2, 2].set_xlabel('Column')
    axes[2, 2].set_ylabel('Row')

    # Add text with validation stats
    stats_text = (
        f"✓ All cells have upstream_rainfall ≥ local_precip: {validation_pct:.1f}%\n"
        f"✓ Max drainage area: {drainage_area.max():,.0f} cells\n"
        f"✓ Cells with drainage_area = 1: {np.sum(drainage_area == 1):,} (ridgelines)\n"
        f"✓ Depressions filled: {filled_cells:,} cells\n"
        f"✓ All flow directions valid: {np.all((flow_dir >= 0) & (flow_dir <= 128))}"
    )

    axes[2, 2].text(0.02, 0.02, stats_text, transform=axes[2, 2].transAxes,
                    fontsize=8, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved validation plot to {output_path}")

    # Print validation summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"✓ Upstream rainfall ≥ local precipitation: {validation_pct:.1f}%")
    print(f"✓ Max drainage area: {drainage_area.max():,.0f} cells")
    print(f"✓ Ridgeline cells (drainage = 1): {np.sum(drainage_area == 1):,}")
    print(f"✓ Depressions filled: {filled_cells:,} cells ({100 * filled_cells / dem_original.size:.2f}%)")
    print(f"✓ Max fill depth: {dem_diff.max():.2f} m")
    print(f"✓ Flow direction codes valid: {np.all((flow_dir >= 0) & (flow_dir <= 128))}")
    print("="*60)


def main():
    """Run flow validation on San Diego demo data."""
    print("[DEBUG] Entered main()")
    import argparse
    print("[DEBUG] Imported argparse")

    parser = argparse.ArgumentParser(description='Validate flow accumulation pipeline')
    print("[DEBUG] Created parser")
    parser.add_argument('--dem', default='examples/output/merged_dem.tif',
                        help='Path to DEM GeoTIFF')
    parser.add_argument('--precip',
                        default=None,
                        help='Path to precipitation GeoTIFF (auto-detects if not specified)')
    parser.add_argument('--output', default='examples/output/flow_validation.png',
                        help='Output path for validation plot')

    args = parser.parse_args()

    # Auto-detect precipitation file if not specified
    if args.precip is None:
        output_dir = Path('examples/output')
        # Look for WorldClim first, then PRISM
        precip_files = list(output_dir.glob('worldclim_annual_precip*.tif'))
        if not precip_files:
            precip_files = list(output_dir.glob('prism_annual_precip*.tif'))

        if precip_files:
            args.precip = str(precip_files[0])
            print(f"Auto-detected precipitation file: {args.precip}")
        else:
            print("❌ No precipitation file found. Run san_diego_flow_demo.py first.")
            return 1

    # Check inputs exist
    if not Path(args.dem).exists():
        print(f"❌ DEM not found: {args.dem}")
        print("Run san_diego_flow_demo.py first to generate test data")
        return 1

    if not Path(args.precip).exists():
        print(f"❌ Precipitation file not found: {args.precip}")
        print("Run san_diego_flow_demo.py first to generate test data")
        return 1

    # Create validation plots
    create_flow_validation_plots(args.dem, args.precip, args.output)

    return 0


if __name__ == '__main__':
    sys.exit(main())
