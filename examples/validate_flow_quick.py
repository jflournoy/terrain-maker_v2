"""
Quick flow accumulation validation using downsampled DEM.

Visualizes each stage of the flow pipeline on a manageable subset.
"""

import sys
from pathlib import Path
import numpy as np

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.terrain.flow_accumulation import (
    compute_flow_direction,
    compute_drainage_area,
    compute_upstream_rainfall,
    condition_dem,
)


def main():
    """Create quick validation plots using flow_accumulation outputs."""
    # Use the already-computed flow outputs from san_diego_flow_demo
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

    import rasterio

    # Load data
    with rasterio.open(flow_dir_path) as src:
        flow_dir = src.read(1)

    with rasterio.open(drainage_path) as src:
        drainage_area = src.read(1)

    with rasterio.open(rainfall_path) as src:
        upstream_rainfall = src.read(1)

    with rasterio.open(dem_cond_path) as src:
        dem_conditioned = src.read(1)

    with rasterio.open(dem_orig_path) as src:
        dem_full = src.read(1).astype(np.float32)
        dem_transform = src.transform
        dem_crs = src.crs

    with rasterio.open(precip_path) as src:
        precip_full = src.read(1).astype(np.float32)
        precip_transform = src.transform

    print(f"Full DEM shape: {dem_full.shape}")
    print(f"Flow output shape: {dem_conditioned.shape}")

    # Resample full-res data to match flow output resolution
    if dem_full.shape != dem_conditioned.shape:
        print("Resampling DEM and precipitation to match flow resolution...")
        from rasterio.warp import reproject, Resampling

        # Get transform from conditioned DEM
        with rasterio.open(dem_cond_path) as src:
            flow_transform = src.transform
            flow_shape = dem_conditioned.shape

        # Resample DEM
        dem_original = np.empty(flow_shape, dtype=np.float32)
        reproject(
            dem_full, dem_original,
            src_transform=dem_transform,
            dst_transform=flow_transform,
            src_crs=dem_crs,
            dst_crs=dem_crs,
            resampling=Resampling.bilinear
        )

        # Resample precipitation
        precip_data = np.empty(flow_shape, dtype=np.float32)
        reproject(
            precip_full, precip_data,
            src_transform=precip_transform,
            dst_transform=flow_transform,
            src_crs=dem_crs,
            dst_crs=dem_crs,
            resampling=Resampling.bilinear
        )
    else:
        dem_original = dem_full
        precip_data = precip_full

    print(f"Data loaded and aligned: {dem_original.shape}")

    # Calculate statistics
    dem_diff = dem_conditioned - dem_original
    filled_cells = np.sum(dem_diff > 0)
    validation = upstream_rainfall >= precip_data
    validation_pct = 100 * np.sum(validation) / validation.size

    print(f"Filled {filled_cells:,} cells ({100 * filled_cells / dem_original.size:.2f}%)")
    print(f"Validation: {validation_pct:.1f}% cells pass (upstream ≥ local)")

    # Create visualization
    print("Creating plots...")

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle('Flow Accumulation Pipeline Validation', fontsize=16, fontweight='bold')

    # Row 1: DEM stages
    im1 = axes[0, 0].imshow(dem_original, cmap='terrain', interpolation='nearest')
    axes[0, 0].set_title('1. Original DEM')
    plt.colorbar(im1, ax=axes[0, 0], label='Elevation (m)', fraction=0.046)

    im2 = axes[0, 1].imshow(dem_conditioned, cmap='terrain', interpolation='nearest')
    axes[0, 1].set_title('2. Conditioned DEM')
    plt.colorbar(im2, ax=axes[0, 1], label='Elevation (m)', fraction=0.046)

    im3 = axes[0, 2].imshow(dem_diff, cmap='hot', interpolation='nearest', vmin=0)
    axes[0, 2].set_title('3. Fill Depth')
    plt.colorbar(im3, ax=axes[0, 2], label='Fill (m)', fraction=0.046)

    # Row 2: Flow direction and drainage
    flow_dir_vis = flow_dir.astype(np.float32)
    flow_dir_vis[flow_dir == 0] = np.nan
    im4 = axes[1, 0].imshow(flow_dir_vis, cmap='hsv', interpolation='nearest')
    axes[1, 0].set_title('4. Flow Direction (D8)')
    plt.colorbar(im4, ax=axes[1, 0], label='Code', fraction=0.046)

    drainage_log = np.log10(drainage_area + 1)
    im5 = axes[1, 1].imshow(drainage_log, cmap='Blues', interpolation='nearest')
    axes[1, 1].set_title('5. Drainage Area (log)')
    plt.colorbar(im5, ax=axes[1, 1], label='log₁₀(cells)', fraction=0.046)

    axes[1, 2].hist(drainage_area.flatten(), bins=100, log=True, color='steelblue')
    axes[1, 2].set_xlabel('Drainage Area (cells)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('6. Drainage Distribution')
    axes[1, 2].set_xscale('log')
    axes[1, 2].grid(True, alpha=0.3)

    # Row 3: Rainfall
    rainfall_log = np.log10(upstream_rainfall + 1)
    im7 = axes[2, 0].imshow(rainfall_log, cmap='YlGnBu', interpolation='nearest')
    axes[2, 0].set_title('7. Upstream Rainfall (log)')
    plt.colorbar(im7, ax=axes[2, 0], label='log₁₀(mm·m²)', fraction=0.046)

    im8 = axes[2, 1].imshow(precip_data, cmap='YlGnBu', interpolation='nearest')
    axes[2, 1].set_title('8. Local Precipitation')
    plt.colorbar(im8, ax=axes[2, 1], label='mm/year', fraction=0.046)

    axes[2, 2].imshow(validation, cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=1)
    axes[2, 2].set_title(f'9. Validation ({validation_pct:.1f}% pass)')

    stats_text = (
        f"✓ Upstream ≥ Local: {validation_pct:.1f}%\n"
        f"✓ Max drainage: {drainage_area.max():,.0f} cells\n"
        f"✓ Ridgelines: {np.sum(drainage_area == 1):,}\n"
        f"✓ Filled: {filled_cells:,} cells\n"
        f"✓ Max fill: {dem_diff.max():.2f} m"
    )
    axes[2, 2].text(0.02, 0.02, stats_text, transform=axes[2, 2].transAxes,
                    fontsize=8, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    output_path = "examples/output/flow_validation.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved to {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"✓ Validation: {validation_pct:.1f}% pass")
    print(f"✓ Max drainage: {drainage_area.max():,.0f} cells")
    print(f"✓ Ridgelines: {np.sum(drainage_area == 1):,}")
    print(f"✓ Filled cells: {filled_cells:,} ({100 * filled_cells / dem_original.size:.2f}%)")
    print(f"✓ Max fill depth: {dem_diff.max():.2f} m")
    print("="*60)


if __name__ == '__main__':
    sys.exit(main())
