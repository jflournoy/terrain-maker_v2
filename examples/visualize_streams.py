#!/usr/bin/env python3
"""
Visualize stream networks from flow accumulation results.

Creates clear visualization of stream networks by highlighting cells
with significant drainage area.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rasterio

def main():
    """Create stream network visualization."""
    # Load data
    print("Loading flow accumulation data...")

    with rasterio.open('examples/output/flow_outputs/flow_accumulation_area.tif') as src:
        drainage = src.read(1)

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
    print(f"Max drainage: {drainage.max():,.0f} cells")

    # Define stream thresholds
    major_rivers = drainage > 10000   # >200 km² at 45m resolution
    rivers = (drainage > 1000) & (drainage <= 10000)  # 20-200 km²
    streams = (drainage > 100) & (drainage <= 1000)   # 2-20 km²

    print(f"\nStream network statistics:")
    print(f"  Major rivers (>10k cells): {np.sum(major_rivers):,} cells")
    print(f"  Rivers (1k-10k): {np.sum(rivers):,} cells")
    print(f"  Streams (100-1k): {np.sum(streams):,} cells")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('San Diego Stream Networks', fontsize=16, fontweight='bold')

    # Plot 1: DEM with all streams overlaid
    ax = axes[0, 0]
    ax.imshow(dem, cmap='terrain', alpha=0.7)
    ax.imshow(streams, cmap='Blues', alpha=0.3, vmin=0, vmax=1)
    ax.imshow(rivers, cmap='Blues', alpha=0.5, vmin=0, vmax=1)
    ax.imshow(major_rivers, cmap='Blues', alpha=0.8, vmin=0, vmax=1)
    ax.set_title('Stream Networks on Terrain')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')

    # Plot 2: Streams only (binary)
    ax = axes[0, 1]
    stream_network = np.zeros_like(drainage, dtype=np.float32)
    stream_network[streams] = 1.0
    stream_network[rivers] = 2.0
    stream_network[major_rivers] = 3.0
    im = ax.imshow(stream_network, cmap='Blues', vmin=0, vmax=3)
    ax.set_title('Stream Network (isolated)')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['No flow', 'Streams', 'Rivers', 'Major rivers'])

    # Plot 3: Drainage area (log scale)
    ax = axes[1, 0]
    drainage_log = np.log10(drainage + 1)
    im = ax.imshow(drainage_log, cmap='YlGnBu', interpolation='nearest')
    ax.set_title('Drainage Area (log₁₀ scale)')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    plt.colorbar(im, ax=ax, label='log₁₀(cells)')

    # Plot 4: Histogram of drainage areas
    ax = axes[1, 1]
    drainage_flat = drainage[drainage > 1].flatten()
    ax.hist(drainage_flat, bins=100, log=True, color='steelblue', alpha=0.7)
    ax.axvline(100, color='green', linestyle='--', label='Streams (100 cells)')
    ax.axvline(1000, color='orange', linestyle='--', label='Rivers (1k cells)')
    ax.axvline(10000, color='red', linestyle='--', label='Major rivers (10k cells)')
    ax.set_xlabel('Drainage Area (cells)')
    ax.set_ylabel('Frequency (log scale)')
    ax.set_title('Drainage Area Distribution')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f"""Stream Network Stats:
Major Rivers: {np.sum(major_rivers):,} cells
Rivers: {np.sum(rivers):,} cells
Streams: {np.sum(streams):,} cells
Max drainage: {drainage.max():,.0f} cells

Main outlet:
  Row: {np.unravel_index(np.argmax(drainage), drainage.shape)[0]}
  Col: {np.unravel_index(np.argmax(drainage), drainage.shape)[1]}
"""
    fig.text(0.98, 0.02, stats_text, fontsize=9, family='monospace',
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    output_path = 'examples/output/stream_networks.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved stream network visualization: {output_path}")

if __name__ == '__main__':
    sys.exit(main())
