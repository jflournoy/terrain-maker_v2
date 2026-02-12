#!/usr/bin/env python3
"""Test the new arrow head visualization for flow direction."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Create a test flow direction grid (1000x1000)
# Make a simple pattern that flows toward center outlet
rows, cols = 1000, 1000
flow_dir = np.zeros((rows, cols), dtype=np.uint8)

# D8 encoding: 1=E, 2=NE, 4=N, 8=NW, 16=W, 32=SW, 64=S, 128=SE
# Create a radial flow pattern toward center
center_y, center_x = rows // 2, cols // 2
for y in range(rows):
    for x in range(cols):
        if y == center_y and x == center_x:
            flow_dir[y, x] = 0  # Center is outlet
        else:
            # Determine direction toward center
            dy = center_y - y
            dx = center_x - x

            # Map to nearest D8 direction
            if abs(dx) > abs(dy) * 2:
                flow_dir[y, x] = 1 if dx > 0 else 16  # E or W
            elif abs(dy) > abs(dx) * 2:
                flow_dir[y, x] = 64 if dy > 0 else 4  # S or N
            elif dx > 0 and dy > 0:
                flow_dir[y, x] = 128  # SE
            elif dx > 0 and dy < 0:
                flow_dir[y, x] = 2  # NE
            elif dx < 0 and dy > 0:
                flow_dir[y, x] = 32  # SW
            else:
                flow_dir[y, x] = 8  # NW

# D8 encoding and corresponding triangle marker angles (degrees)
d8_markers = {
    1: (0, '>'),      # East - right triangle
    2: (45, '^'),     # Northeast - up-right
    4: (90, '^'),     # North - up triangle
    8: (135, '^'),    # Northwest - up-left
    16: (180, '<'),   # West - left triangle
    32: (225, 'v'),   # Southwest - down-left
    64: (270, 'v'),   # South - down triangle
    128: (315, 'v'),  # Southeast - down-right
}

# Test different sampling rates and marker sizes
output_dir = Path("examples/output")
output_dir.mkdir(exist_ok=True, parents=True)

configs = [
    # (sample_rate, pixels_per_arrow, name)
    # pixels_per_arrow determines image resolution: image_size = grid_size * pixels_per_arrow
    (1, 3, "full_3px"),          # Every cell, minimal 3-pixel triangles → 3,000×3,000 image
    (1, 10, "full_10px"),        # Every cell, 10×10 pixels per arrow → 10,000×10,000 image
    (5, 10, "fifth_10px"),       # Every 5th cell, 10×10 pixels per arrow → 2,000×2,000 image
    (10, 15, "tenth_15px"),      # Every 10th cell, 15×15 pixels per arrow → 1,500×1,500 image
]

print(f"Generating {len(configs)} high-resolution test plots...")
print(f"Test grid: {rows}x{cols} with radial flow toward center outlet\n")

for sample_rate, pixels_per_arrow, name in configs:
    # Calculate image dimensions
    # Each arrow needs pixels_per_arrow × pixels_per_arrow pixels to be visible
    sampled_rows = len(np.arange(0, rows, sample_rate))
    sampled_cols = len(np.arange(0, cols, sample_rate))

    image_height_px = sampled_rows * pixels_per_arrow
    image_width_px = sampled_cols * pixels_per_arrow

    # Create figure with exact pixel dimensions
    dpi = 100
    fig_width = image_width_px / dpi
    fig_height = image_height_px / dpi

    print(f"  Creating {name}: {sampled_rows}×{sampled_cols} arrows at {pixels_per_arrow}px/arrow = {image_width_px}×{image_height_px}px image...")

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Background
    background = np.where(flow_dir == 0, np.nan, 1.0)
    ax.imshow(background, cmap='gray_r', alpha=0.3, origin='upper', extent=[0, cols, rows, 0])

    # Sample grid
    y_sample = np.arange(0, rows, sample_rate)
    x_sample = np.arange(0, cols, sample_rate)
    actual_arrows = len(y_sample) * len(x_sample)

    # Calculate marker size in points² to fill the pixel area
    # Marker size in matplotlib is in points² (1 point = dpi/72 pixels)
    # We want markers to be approximately pixels_per_arrow pixels wide
    points_per_pixel = 72 / dpi
    marker_size = (pixels_per_arrow * points_per_pixel) ** 2

    # Plot arrow heads as triangular markers for each D8 direction
    for code, (angle, marker_base) in d8_markers.items():
        # Find all cells with this flow direction in the sampled grid
        mask = np.zeros((rows, cols), dtype=bool)
        for y in y_sample:
            for x in x_sample:
                if flow_dir[y, x] == code:
                    mask[y, x] = True

        y_coords, x_coords = np.where(mask)

        if len(y_coords) > 0:
            # Plot triangles for this direction
            ax.scatter(x_coords, y_coords,
                      marker=(3, 0, angle),  # 3-sided polygon (triangle) rotated by angle
                      s=marker_size,
                      c='darkblue',
                      alpha=0.7,
                      edgecolors='none',
                      rasterized=True)

    # Outlet (larger marker for visibility)
    outlet_size = marker_size * 4
    ax.scatter(center_x, center_y, c='red', s=outlet_size, marker='o', zorder=10,
              edgecolors='darkred', linewidths=3)

    title_fontsize = max(10, min(24, image_width_px // 100))
    ax.set_title(f"Sample every {sample_rate} cell(s), {pixels_per_arrow}px/arrow ({actual_arrows:,} arrows)",
                fontsize=title_fontsize, fontweight='bold')
    label_fontsize = max(8, min(16, image_width_px // 150))
    ax.set_xlabel("Column", fontsize=label_fontsize)
    ax.set_ylabel("Row", fontsize=label_fontsize)
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal')

    plt.tight_layout()
    output_path = output_dir / f"test_arrowheads_{name}.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved: {output_path.name} ({image_width_px}×{image_height_px}px, {file_size_mb:.1f}MB)\n")

print("\nD8 direction encoding:")
for code, (angle, _) in d8_markers.items():
    direction_name = {1: "East", 2: "NE", 4: "North", 8: "NW",
                     16: "West", 32: "SW", 64: "South", 128: "SE"}[code]
    print(f"  {code:3d} = {direction_name:5s} → triangle rotated {angle:3.0f}°")
