#!/usr/bin/env python3
"""Test arrow visualization with real DEM data from San Diego."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from terrain.data_loading import load_dem_files
from terrain.flow_accumulation import compute_flow_direction

# San Diego area DEM directory
dem_dir = Path("data/san_diego_dem")

if not dem_dir.exists():
    print(f"Error: DEM directory not found at {dem_dir}")
    print("Please ensure the San Diego DEM data is available.")
    sys.exit(1)

print("Loading full combined DEM...")
dem_full, transform = load_dem_files(dem_dir)
print(f"  Full DEM shape: {dem_full.shape}")
print(f"  Full elevation range: [{np.min(dem_full):.1f}, {np.max(dem_full):.1f}] meters")

# Downsample so longest side is 1000px
print("\nDownsampling to 1000px on longest side...")
from scipy.ndimage import zoom
rows, cols = dem_full.shape
longest_side = max(rows, cols)
zoom_factor = 1000.0 / longest_side

dem = zoom(dem_full, zoom_factor, order=1)  # Bilinear interpolation

print(f"  Downsampled DEM shape: {dem.shape}")
print(f"  Downsampled elevation range: [{np.min(dem):.1f}, {np.max(dem):.1f}] meters")
print(f"  This will generate {dem.shape[0] * dem.shape[1]:,} arrows (one per pixel)")

# Compute flow direction using D8 algorithm
print("\nComputing flow direction...")
flow_dir = compute_flow_direction(dem)

print(f"  Flow direction codes: {np.unique(flow_dir)}")
outlets = np.sum(flow_dir == 0)
print(f"  Found {outlets} outlet/pit cells")

# D8 encoding and corresponding triangle marker angles
d8_markers = {
    1: (0, '>'),      # East
    2: (45, '^'),     # Northeast
    4: (90, '^'),     # North
    8: (135, '^'),    # Northwest
    16: (180, '<'),   # West
    32: (225, 'v'),   # Southwest
    64: (270, 'v'),   # South
    128: (315, 'v'),  # Southeast
}

# Create visualizations at different pixel densities
output_dir = Path("examples/output")
output_dir.mkdir(exist_ok=True, parents=True)

configs = [
    # (pixels_per_arrow, name)
    (3, "real_1000x1000_3px"),   # 1000×1000 arrows, 3px each = 3000×3000px image
]

print(f"\nGenerating {len(configs)} arrow visualizations from real DEM data...")

for pixels_per_arrow, name in configs:
    rows, cols = flow_dir.shape

    # Calculate image dimensions
    image_height_px = rows * pixels_per_arrow
    image_width_px = cols * pixels_per_arrow

    print(f"\n  Creating {name}: {rows}×{cols} arrows at {pixels_per_arrow}px/arrow = {image_width_px}×{image_height_px}px...")

    # Create figure
    dpi = 100
    fig_width = image_width_px / dpi
    fig_height = image_height_px / dpi
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Create vibrant color map for flow directions
    # Map each D8 direction to a distinct saturated color
    flow_colors = np.zeros((rows, cols, 3))
    d8_color_map = {
        0: (0.5, 0.5, 0.5),      # Outlet/pit - medium gray
        1: (1.0, 0.2, 0.2),      # East - vibrant red
        2: (1.0, 0.5, 0.0),      # Northeast - vibrant orange
        4: (0.9, 0.9, 0.0),      # North - vibrant yellow
        8: (0.5, 0.9, 0.0),      # Northwest - vibrant yellow-green
        16: (0.0, 0.8, 0.0),     # West - vibrant green
        32: (0.0, 0.8, 0.8),     # Southwest - vibrant cyan
        64: (0.0, 0.4, 1.0),     # South - vibrant blue
        128: (0.6, 0.0, 0.8),    # Southeast - vibrant purple
    }

    for code, color in d8_color_map.items():
        mask = (flow_dir == code)
        flow_colors[mask] = color

    # Show flow direction colors as background
    ax.imshow(flow_colors, origin='upper', extent=[0, cols, rows, 0], alpha=0.9)

    # Calculate marker size - 3 pixels per cell
    points_per_pixel = 72 / dpi
    marker_size = (pixels_per_arrow * points_per_pixel) ** 2

    # Plot arrow heads for each D8 direction
    for code, (angle, _) in d8_markers.items():
        mask = (flow_dir == code)
        y_coords, x_coords = np.where(mask)

        if len(y_coords) > 0:
            ax.scatter(x_coords, y_coords,
                      marker=(3, 0, angle),
                      s=marker_size,
                      c='darkblue',
                      alpha=0.7,
                      edgecolors='none',
                      rasterized=True)

    # Mark outlets
    outlets_y, outlets_x = np.where(flow_dir == 0)
    outlet_size = 0.25  # Very small outlet markers
    if len(outlets_y) > 0:
        ax.scatter(outlets_x, outlets_y, c='red', s=outlet_size, marker='o',
                  alpha=0.9, edgecolors='darkred', linewidths=2, zorder=10,
                  label=f'Outlets ({len(outlets_y)})')

    title_fontsize = max(10, min(20, image_width_px // 100))
    ax.set_title(f"Real DEM Flow Direction: {rows}×{cols} cells at {pixels_per_arrow}px/arrow",
                fontsize=title_fontsize, fontweight='bold')
    label_fontsize = max(8, min(14, image_width_px // 150))
    ax.set_xlabel("Column", fontsize=label_fontsize)
    ax.set_ylabel("Row", fontsize=label_fontsize)
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_aspect('equal')
    if len(outlets_y) > 0:
        ax.legend(loc='upper right', fontsize=label_fontsize)

    plt.tight_layout()
    output_path = output_dir / f"test_arrowheads_{name}.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"    ✓ Saved: {output_path.name} ({image_width_px}×{image_height_px}px, {file_size_mb:.1f}MB)")

print(f"\n✓ Real DEM arrow visualization test complete!")
print(f"  DEM size: {dem.shape[0]}×{dem.shape[1]} cells")
print(f"  Arrows shown: {dem.shape[0] * dem.shape[1]:,} (ONE per pixel)")
print(f"  Elevation range: {np.min(dem):.1f}m to {np.max(dem):.1f}m")
print(f"  Output: examples/output/test_arrowheads_real_1000x1000_3px.png")
