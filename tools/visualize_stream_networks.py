#!/usr/bin/env python3
"""
Visualize stream networks by building upstream drainage networks.

Creates visualization showing continuous stream networks by finding
all cells that drain to major outlets.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rasterio

def find_upstream_cells(flow_dir, start_row, start_col):
    """
    Find all cells that eventually drain to the starting cell.

    Uses breadth-first search to find all upstream contributing cells.
    Returns set of (row, col) tuples.
    """
    # D8 direction encoding (matches flow_accumulation.py)
    direction_offsets = {
        1: (0, 1),    # East
        2: (-1, 1),   # Northeast
        4: (-1, 0),   # North
        8: (-1, -1),  # Northwest
        16: (0, -1),  # West
        32: (1, -1),  # Southwest
        64: (1, 0),   # South
        128: (1, 1)   # Southeast
    }

    # To find cells that flow TO (start_row, start_col), we need to:
    # 1. For each possible direction code
    # 2. Calculate which neighboring cell would flow here with that code
    # 3. Check if that neighbor actually has that flow direction
    reverse_lookup = {}
    for code, (dr, dc) in direction_offsets.items():
        # If a cell flows with direction (dr, dc), then a cell at (-dr, -dc) would flow here
        reverse_lookup[code] = (-dr, -dc)

    upstream = set()
    to_check = [(start_row, start_col)]
    checked = set()

    rows, cols = flow_dir.shape

    while to_check:
        row, col = to_check.pop(0)

        if (row, col) in checked:
            continue
        checked.add((row, col))
        upstream.add((row, col))

        # Check all 8 neighbors to see if any flow to this cell
        for code, (dr, dc) in reverse_lookup.items():
            neighbor_row = row + dr
            neighbor_col = col + dc

            # Check bounds
            if not (0 <= neighbor_row < rows and 0 <= neighbor_col < cols):
                continue

            # Check if this neighbor flows here
            neighbor_flow = flow_dir[neighbor_row, neighbor_col]
            if neighbor_flow == code:
                to_check.append((neighbor_row, neighbor_col))

    return upstream

def main():
    """Create stream network visualization by building upstream networks."""
    print("Loading flow data...")

    with rasterio.open('examples/output/flow_outputs/flow_direction.tif') as src:
        flow_dir = src.read(1)

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

    # Find outlets (local maxima in drainage with high accumulation)
    # These are cells with very high drainage and most neighbors have lower drainage
    major_threshold = 100000  # >2000 km² at 45m resolution
    major_outlets = []

    print(f"\nFinding major outlets (drainage >{major_threshold:,})...")
    candidates = np.where(drainage > major_threshold)

    for i in range(len(candidates[0])):
        row = candidates[0][i]
        col = candidates[1][i]
        drain = drainage[row, col]

        # Check if this is a local maximum (most neighbors have lower drainage)
        is_max = True
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < drainage.shape[0] and 0 <= nc < drainage.shape[1]:
                    if drainage[nr, nc] > drain:
                        is_max = False
                        break
            if not is_max:
                break

        if is_max:
            major_outlets.append((row, col, drain))

    major_outlets.sort(key=lambda x: x[2], reverse=True)
    print(f"Found {len(major_outlets)} major outlets")

    for i, (row, col, drain) in enumerate(major_outlets[:5]):
        print(f"  {i+1}. ({row}, {col}): {drain:,.0f} cells")

    # Build upstream network for the largest outlet
    if not major_outlets:
        print("No major outlets found! Using global maximum.")
        outlet_idx = np.unravel_index(np.argmax(drainage), drainage.shape)
        outlet_row, outlet_col = outlet_idx
    else:
        outlet_row, outlet_col, _ = major_outlets[0]

    print(f"\nBuilding upstream network from outlet ({outlet_row}, {outlet_col})...")
    upstream_cells = find_upstream_cells(flow_dir, outlet_row, outlet_col)
    print(f"Found {len(upstream_cells):,} upstream cells")

    # Create network mask
    network_mask = np.zeros(drainage.shape, dtype=bool)
    for row, col in upstream_cells:
        network_mask[row, col] = True

    # Define stream hierarchy within the network
    major_rivers = network_mask & (drainage > 10000)
    rivers = network_mask & (drainage > 1000) & (drainage <= 10000)
    streams = network_mask & (drainage > 100) & (drainage <= 1000)

    print(f"\nStream hierarchy in main network:")
    print(f"  Major rivers (>10k cells): {np.sum(major_rivers):,} cells")
    print(f"  Rivers (1k-10k): {np.sum(rivers):,} cells")
    print(f"  Streams (100-1k): {np.sum(streams):,} cells")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('San Diego River Network (Upstream Trace)', fontsize=16, fontweight='bold')

    # Plot 1: DEM with stream network
    ax = axes[0, 0]
    ax.imshow(dem, cmap='terrain', alpha=0.7)
    ax.imshow(network_mask, cmap='Blues', alpha=0.6, vmin=0, vmax=1)
    ax.plot(outlet_col, outlet_row, 'r*', markersize=15, label='Main outlet')
    ax.set_title('Complete Stream Network on Terrain')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.legend()

    # Plot 2: Stream hierarchy
    ax = axes[0, 1]
    stream_levels = np.zeros_like(drainage, dtype=np.float32)
    stream_levels[streams] = 1.0
    stream_levels[rivers] = 2.0
    stream_levels[major_rivers] = 3.0
    im = ax.imshow(stream_levels, cmap='Blues', vmin=0, vmax=3)
    ax.plot(outlet_col, outlet_row, 'r*', markersize=15)
    ax.set_title('Stream Hierarchy')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['No flow', 'Streams', 'Rivers', 'Major rivers'])

    # Plot 3: Drainage area (log scale) for network only
    ax = axes[1, 0]
    drainage_masked = np.where(network_mask, drainage, 0)
    drainage_log = np.log10(drainage_masked + 1)
    im = ax.imshow(drainage_log, cmap='YlGnBu', interpolation='nearest')
    ax.set_title('Drainage Area (network only, log scale)')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    plt.colorbar(im, ax=ax, label='log₁₀(cells)')

    # Plot 4: Drainage distribution within network
    ax = axes[1, 1]
    network_drainage = drainage[network_mask]
    ax.hist(network_drainage, bins=100, log=True, color='steelblue', alpha=0.7)
    ax.axvline(100, color='green', linestyle='--', label='Streams (100 cells)')
    ax.axvline(1000, color='orange', linestyle='--', label='Rivers (1k cells)')
    ax.axvline(10000, color='red', linestyle='--', label='Major rivers (10k cells)')
    ax.set_xlabel('Drainage Area (cells)')
    ax.set_ylabel('Frequency (log scale)')
    ax.set_title('Drainage Distribution in Network')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add statistics
    stats_text = f"""Network Statistics:
Total network cells: {len(upstream_cells):,}
Major rivers: {np.sum(major_rivers):,} cells
Rivers: {np.sum(rivers):,} cells
Streams: {np.sum(streams):,} cells

Main outlet:
  Location: ({outlet_row}, {outlet_col})
  Drainage: {drainage[outlet_row, outlet_col]:,.0f} cells
  Area: ~{drainage[outlet_row, outlet_col] * 0.002025:.0f} km²
"""
    fig.text(0.98, 0.02, stats_text, fontsize=9, family='monospace',
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    output_path = 'examples/output/stream_network_upstream.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved stream network visualization: {output_path}")

if __name__ == '__main__':
    sys.exit(main())
