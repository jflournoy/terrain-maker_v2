#!/usr/bin/env python3
"""
Visualize flow accumulation with continuous color gradients.

Shows two key metrics:
1. Drainage area (number of cells flowing into each cell)
2. Cumulative rainfall (annual rainfall accumulated from upstream)

Both use continuous color gradients to show how values increase from
headwaters to outlets in a smooth, natural way.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rasterio

def main():
    """Create continuous flow accumulation visualizations."""
    print("Loading flow accumulation data...")

    with rasterio.open('examples/output/flow_outputs/flow_accumulation_area.tif') as src:
        drainage = src.read(1)

    with rasterio.open('examples/output/flow_outputs/flow_accumulation_rainfall.tif') as src:
        rainfall_accum = src.read(1)

    with rasterio.open('examples/output/merged_dem.tif') as src:
        dem = src.read(1).astype(np.float32)

    # Resample DEM to match flow resolution if needed
    if dem.shape != drainage.shape:
        print(f"Resampling DEM from {dem.shape} to {drainage.shape}...")
        from rasterio.warp import reproject, Resampling

        with rasterio.open('examples/output/merged_dem.tif') as src:
            dem_transform = src.transform
            dem_crs = src.crs

        with rasterio.open('examples/output/flow_outputs/flow_accumulation_area.tif') as src:
            flow_transform = src.transform

        dem_resampled = np.empty(drainage.shape, dtype=np.float32)
        reproject(
            dem, dem_resampled,
            src_transform=dem_transform,
            dst_transform=flow_transform,
            src_crs=dem_crs,
            dst_crs=dem_crs,
            resampling=Resampling.bilinear
        )
        dem = dem_resampled

    print(f"Data shape: {drainage.shape}")
    print(f"Max drainage: {drainage.max():,.0f} cells = {drainage.max() * 0.002025:.0f} km²")
    print(f"Max rainfall accumulation: {rainfall_accum.max():.2e} mm/year")

    # Convert to km² for better readability
    drainage_km2 = drainage * 0.002025  # 45m cells = 0.002025 km²

    # Convert rainfall to km³/year (more intuitive units)
    # rainfall_accum is in mm/year * cells
    # Each cell is 0.002025 km² = 2,025 m²
    # mm/year * m² = m³/year
    # So: mm/year * 2,025 m² = mm * 2,025 m² / year = 2.025 m³/year
    rainfall_km3 = rainfall_accum * 2.025 / 1e9  # Convert m³ to km³

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Flow Accumulation: Drainage Area and Cumulative Rainfall',
                 fontsize=16, fontweight='bold')

    # Row 1: Drainage area (cell count → km²)

    # Plot 1: Terrain with drainage overlay (linear scale, threshold)
    ax = axes[0, 0]
    ax.imshow(dem, cmap='terrain', alpha=0.7)
    # Overlay significant drainage (>2 km²)
    drainage_mask = drainage_km2 > 2.0
    drainage_overlay = np.ma.masked_where(~drainage_mask, drainage_km2)
    im = ax.imshow(drainage_overlay, cmap='Blues', alpha=0.6, vmin=2, vmax=drainage_km2.max())
    ax.set_title('Drainage Area on Terrain (>2 km²)')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    plt.colorbar(im, ax=ax, label='Drainage area (km²)')

    # Plot 2: Drainage area (log scale, full range)
    ax = axes[0, 1]
    drainage_log = np.log10(drainage_km2 + 0.001)  # +0.001 to avoid log(0)
    im = ax.imshow(drainage_log, cmap='viridis', interpolation='nearest')
    ax.set_title('Drainage Area (log₁₀ scale, continuous)')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    cbar = plt.colorbar(im, ax=ax, label='log₁₀(km²)')
    # Add readable tick labels
    tick_locs = cbar.get_ticks()
    tick_labels = [f'{10**v:.1f}' if v >= 0 else f'{10**v:.3f}' for v in tick_locs]
    cbar.ax.set_yticklabels(tick_labels)

    # Plot 3: Drainage distribution
    ax = axes[0, 2]
    drainage_nonzero = drainage_km2[drainage_km2 > 0.001]
    ax.hist(drainage_nonzero, bins=100, log=True, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(2, color='green', linestyle='--', linewidth=2, label='Small streams (2 km²)')
    ax.axvline(20, color='orange', linestyle='--', linewidth=2, label='Rivers (20 km²)')
    ax.axvline(200, color='red', linestyle='--', linewidth=2, label='Major rivers (200 km²)')
    ax.set_xlabel('Drainage Area (km²)')
    ax.set_ylabel('Number of Cells (log scale)')
    ax.set_title('Drainage Area Distribution')
    ax.set_xscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Row 2: Cumulative rainfall (precipitation-weighted accumulation)

    # Plot 4: Terrain with rainfall accumulation overlay
    ax = axes[1, 0]
    ax.imshow(dem, cmap='terrain', alpha=0.7)
    # Overlay significant rainfall accumulation
    rainfall_threshold = np.percentile(rainfall_km3[rainfall_km3 > 0], 90)  # Top 10%
    rainfall_mask = rainfall_km3 > rainfall_threshold
    rainfall_overlay = np.ma.masked_where(~rainfall_mask, rainfall_km3)
    im = ax.imshow(rainfall_overlay, cmap='Blues', alpha=0.6,
                   vmin=rainfall_threshold, vmax=rainfall_km3.max())
    ax.set_title(f'Cumulative Rainfall on Terrain (top 10%)')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    plt.colorbar(im, ax=ax, label='Cumulative rainfall (km³/year)')

    # Plot 5: Rainfall accumulation (log scale, full range)
    ax = axes[1, 1]
    rainfall_log = np.log10(rainfall_km3 + 1e-6)  # Avoid log(0)
    im = ax.imshow(rainfall_log, cmap='plasma', interpolation='nearest')
    ax.set_title('Cumulative Rainfall (log₁₀ scale, continuous)')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    cbar = plt.colorbar(im, ax=ax, label='log₁₀(km³/year)')
    # Add readable tick labels
    tick_locs = cbar.get_ticks()
    tick_labels = [f'{10**v:.2e}' for v in tick_locs]
    cbar.ax.set_yticklabels(tick_labels)

    # Plot 6: Rainfall distribution
    ax = axes[1, 2]
    rainfall_nonzero = rainfall_km3[rainfall_km3 > 1e-6]
    ax.hist(rainfall_nonzero, bins=100, log=True, color='purple', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Cumulative Rainfall (km³/year)')
    ax.set_ylabel('Number of Cells (log scale)')
    ax.set_title('Cumulative Rainfall Distribution')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f"""Flow Accumulation Statistics:

Drainage Area:
  Max: {drainage.max():,.0f} cells = {drainage_km2.max():.1f} km²
  Mean: {drainage.mean():.1f} cells = {drainage_km2.mean():.3f} km²
  Cells >2 km²: {np.sum(drainage_km2 > 2):,}
  Cells >20 km²: {np.sum(drainage_km2 > 20):,}
  Cells >200 km²: {np.sum(drainage_km2 > 200):,}

Cumulative Rainfall:
  Max: {rainfall_accum.max():.2e} mm/year = {rainfall_km3.max():.3e} km³/year
  Mean: {rainfall_accum.mean():.2e} mm/year = {rainfall_km3.mean():.3e} km³/year

Main outlet at ({np.unravel_index(np.argmax(drainage), drainage.shape)[0]},
               {np.unravel_index(np.argmax(drainage), drainage.shape)[1]})
"""
    fig.text(0.98, 0.02, stats_text, fontsize=8, family='monospace',
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    output_path = 'examples/output/flow_accumulation_continuous.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved continuous flow accumulation visualization: {output_path}")

    # Also create a simplified version showing just the two key maps side-by-side
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))
    fig2.suptitle('Hydrogeography: Cell Count vs Rainfall Accumulation',
                  fontsize=14, fontweight='bold')

    # Left: Drainage area (cell count)
    ax = axes2[0]
    im = ax.imshow(drainage_log, cmap='viridis', interpolation='nearest')
    ax.set_title('Number of Upstream Cells\n(drainage area, continuous gradient)', fontsize=12)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    cbar = plt.colorbar(im, ax=ax, label='log₁₀(km²)', fraction=0.046, pad=0.04)
    tick_locs = cbar.get_ticks()
    tick_labels = [f'{10**v:.1f}' if v >= 0 else f'{10**v:.3f}' for v in tick_locs]
    cbar.ax.set_yticklabels(tick_labels)

    # Right: Cumulative rainfall
    ax = axes2[1]
    im = ax.imshow(rainfall_log, cmap='plasma', interpolation='nearest')
    ax.set_title('Cumulative Upstream Rainfall\n(rainfall "diffused down", continuous gradient)', fontsize=12)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    cbar = plt.colorbar(im, ax=ax, label='log₁₀(km³/year)', fraction=0.046, pad=0.04)
    tick_locs = cbar.get_ticks()
    tick_labels = [f'{10**v:.2e}' for v in tick_locs]
    cbar.ax.set_yticklabels(tick_labels)

    output_path2 = 'examples/output/flow_accumulation_comparison.png'
    plt.tight_layout()
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison visualization: {output_path2}")

if __name__ == '__main__':
    sys.exit(main())
