#!/usr/bin/env python3
"""
Complete flow validation: Run production functions and generate visualization images.

This script:
1. Loads a DEM (size based on --bigness option)
2. Runs the PRODUCTION flow functions (compute_flow_direction, etc.)
3. Validates correctness (0 cycles, 100% mass balance)
4. Generates individual visualization images for each pipeline stage

Bigness options:
- small/medium/large: Extract a coastal subset of that size (200/500/1000 pixels)
- full: Use ENTIRE DEM area but downsample to target resolution (default 1000x1000)

Usage:
    python examples/validate_flow_complete.py                    # Small subset (200x200)
    python examples/validate_flow_complete.py --bigness medium   # Medium subset (500x500)
    python examples/validate_flow_complete.py --bigness large    # Large subset (1000x1000)
    python examples/validate_flow_complete.py --bigness full     # Full area, downsampled to 1000x1000
    python examples/validate_flow_complete.py --bigness full --target-size 2000  # Full area at 2000x2000
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
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import PRODUCTION flow functions
from src.terrain.flow_accumulation import (
    compute_flow_direction,
    compute_drainage_area,
    compute_upstream_rainfall,
    condition_dem,
    detect_ocean_mask,
    D8_OFFSETS,
)


def load_real_precipitation(dem_shape: tuple, output_dir: Path) -> tuple[np.ndarray, bool]:
    """
    Load real WorldClim precipitation data if available.

    Returns
    -------
    tuple[np.ndarray, bool]
        (precipitation array resampled to dem_shape, True if real data)
        or (None, False) if no real data found
    """
    # Look for WorldClim precipitation files
    precip_patterns = [
        output_dir / "worldclim_*.tif",
        output_dir.parent / "worldclim_*.tif",
        Path("examples/output") / "worldclim_*.tif",
    ]

    precip_path = None
    for pattern in precip_patterns:
        matches = list(pattern.parent.glob(pattern.name))
        if matches:
            precip_path = matches[0]
            break

    if precip_path is None or not precip_path.exists():
        return None, False

    print(f"  Loading real precipitation from {precip_path.name}")
    with rasterio.open(precip_path) as src:
        precip_raw = src.read(1).astype(np.float32)
        nodata = src.nodata

    # Handle nodata values (WorldClim uses large negative values)
    # If nodata not set in metadata, detect it from unrealistic values
    if nodata is None:
        # Precipitation can't be negative or > 15000 mm/year
        invalid_mask = (precip_raw < 0) | (precip_raw > 15000)
    else:
        invalid_mask = precip_raw == nodata

    # Replace invalid values with neighborhood mean for better interpolation
    if np.any(invalid_mask):
        valid_mean = np.nanmean(precip_raw[~invalid_mask])
        precip_raw[invalid_mask] = valid_mean
        print(f"  Replaced {np.sum(invalid_mask)} nodata cells with mean ({valid_mean:.0f} mm)")

    print(f"  Precipitation range: {precip_raw.min():.0f} - {precip_raw.max():.0f} mm/year")

    # Resample to match DEM shape
    if precip_raw.shape != dem_shape:
        scale_y = dem_shape[0] / precip_raw.shape[0]
        scale_x = dem_shape[1] / precip_raw.shape[1]
        precip = zoom(precip_raw, (scale_y, scale_x), order=1)
        print(f"  Resampled from {precip_raw.shape} to {precip.shape}")
    else:
        precip = precip_raw

    return precip.astype(np.float32), True

# Viridis family colormaps
CMAP_ELEVATION = 'viridis'      # Good for elevation data
CMAP_BINARY = 'cividis'         # Good for binary/mask data
CMAP_FILL = 'magma'             # Good for fill depth (low values important)
CMAP_DIRECTION = 'plasma'       # Good for categorical direction data
CMAP_DRAINAGE = 'viridis'       # Good for log-scale drainage
CMAP_STREAMS = 'inferno'        # Good for stream highlighting
CMAP_PRECIP = 'cividis'         # Good for precipitation
CMAP_RAINFALL = 'plasma'        # Good for accumulated rainfall

# Bigness presets
# For small/medium/large: extract a subset of that size
# For full: use entire DEM but downsample to target resolution
BIGNESS_SIZES = {
    'small': 200,
    'medium': 500,
    'large': 1000,
    'full': 1000,  # Target resolution for downsampled full DEM
}


def validate_no_cycles(flow_dir: np.ndarray, sample_size: int = 500) -> tuple[int, int]:
    """Check for flow cycles by tracing random paths."""
    rows, cols = flow_dir.shape
    non_outlet = np.where(flow_dir > 0)

    if len(non_outlet[0]) == 0:
        return 0, 0

    actual_sample = min(sample_size, len(non_outlet[0]))
    np.random.seed(42)
    sample_idx = np.random.choice(len(non_outlet[0]), actual_sample, replace=False)

    cycles = 0
    for idx in sample_idx:
        i, j = non_outlet[0][idx], non_outlet[1][idx]
        visited = set()
        while True:
            if (i, j) in visited:
                cycles += 1
                break
            visited.add((i, j))
            d = flow_dir[i, j]
            if d == 0 or d not in D8_OFFSETS:
                break
            di, dj = D8_OFFSETS[d]
            i, j = i + di, j + dj
            if i < 0 or i >= rows or j < 0 or j >= cols:
                break

    return cycles, actual_sample


def compute_mass_balance(flow_dir: np.ndarray, drainage_area: np.ndarray) -> float:
    """Compute mass balance: sum of outlet drainage / total cells."""
    outlets = flow_dir == 0
    total_drainage = np.sum(drainage_area[outlets])
    return 100 * total_drainage / flow_dir.size


def validate_drainage_connectivity(flow_dir: np.ndarray, drainage_area: np.ndarray) -> tuple[int, np.ndarray]:
    """
    Validate that drainage area is properly cumulative.

    For every non-outlet cell, its downstream neighbor must have drainage >= this cell's drainage.

    Returns
    -------
    tuple[int, np.ndarray]
        (count of violations, boolean mask of violating cells)
    """
    rows, cols = flow_dir.shape
    violations = np.zeros((rows, cols), dtype=bool)

    # D8 direction codes to (di, dj) offsets
    dir_to_offset = {
        1: (0, 1),    # E
        2: (-1, 1),   # NE
        4: (-1, 0),   # N
        8: (-1, -1),  # NW
        16: (0, -1),  # W
        32: (1, -1),  # SW
        64: (1, 0),   # S
        128: (1, 1),  # SE
    }

    for i in range(rows):
        for j in range(cols):
            fd = flow_dir[i, j]
            if fd == 0:  # Outlet - skip
                continue

            if fd not in dir_to_offset:
                continue

            di, dj = dir_to_offset[fd]
            ni, nj = i + di, j + dj

            # Check bounds
            if 0 <= ni < rows and 0 <= nj < cols:
                my_drainage = drainage_area[i, j]
                neighbor_drainage = drainage_area[ni, nj]

                # Downstream cell should have drainage >= my drainage
                # (it receives my water plus potentially others)
                if neighbor_drainage < my_drainage:
                    violations[i, j] = True

    return np.sum(violations), violations


def trace_flow_paths(flow_dir: np.ndarray, drainage_area: np.ndarray, num_traces: int = 5) -> None:
    """
    Trace a few flow paths from high-drainage cells and print drainage values.

    This helps verify that drainage increases monotonically downstream.
    """
    rows, cols = flow_dir.shape

    # D8 direction codes to (di, dj) offsets
    dir_to_offset = {
        1: (0, 1),    # E
        2: (-1, 1),   # NE
        4: (-1, 0),   # N
        8: (-1, -1),  # NW
        16: (0, -1),  # W
        32: (1, -1),  # SW
        64: (1, 0),   # S
        128: (1, 1),  # SE
    }

    # Find cells with high drainage (but not outlets - those are maxima)
    # Look for cells in the 90-95th percentile
    threshold_low = np.percentile(drainage_area[drainage_area > 1], 90)
    threshold_high = np.percentile(drainage_area[drainage_area > 1], 95)

    candidates = np.argwhere(
        (drainage_area > threshold_low) &
        (drainage_area < threshold_high) &
        (flow_dir > 0)
    )

    if len(candidates) == 0:
        print("   No suitable cells found for flow tracing")
        return

    # Sample a few random starting points
    np.random.seed(42)
    sample_idx = np.random.choice(len(candidates), min(num_traces, len(candidates)), replace=False)

    print(f"\n   Tracing {len(sample_idx)} flow paths (showing first 10 steps):")
    print("   " + "-" * 60)

    for idx in sample_idx:
        start_i, start_j = candidates[idx]

        # Trace downstream
        path = []
        i, j = start_i, start_j
        max_steps = 10

        for step in range(max_steps):
            fd = flow_dir[i, j]
            drain = drainage_area[i, j]
            path.append((i, j, drain, fd))

            if fd == 0:  # Reached outlet
                break
            if fd not in dir_to_offset:
                break

            di, dj = dir_to_offset[fd]
            ni, nj = i + di, j + dj

            if not (0 <= ni < rows and 0 <= nj < cols):
                break

            i, j = ni, nj

        # Print the path
        print(f"   Path from ({start_i}, {start_j}):")
        drain_values = [p[2] for p in path]
        is_monotonic = all(drain_values[i] <= drain_values[i+1] for i in range(len(drain_values)-1))

        for step, (pi, pj, drain, fd) in enumerate(path):
            dir_name = {1: 'E', 2: 'NE', 4: 'N', 8: 'NW', 16: 'W', 32: 'SW', 64: 'S', 128: 'SE', 0: 'OUT'}.get(fd, '?')
            arrow = "→" if step < len(path) - 1 else ""
            print(f"      [{pi:4d},{pj:4d}] drain={drain:>10,.0f}  flow={dir_name:>3} {arrow}")

        status = "✓ monotonic" if is_monotonic else "✗ NOT monotonic!"
        print(f"      {status}")
        print()


def save_plot(data: np.ndarray, title: str, output_path: Path, cmap: str,
              label: str = '', log_scale: bool = False, mask: np.ndarray = None,
              overlay_data: np.ndarray = None, overlay_cmap: str = None,
              vmin: float = None, vmax: float = None):
    """Save a single plot to file."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Handle masking
    plot_data = data.copy().astype(float)
    if mask is not None:
        plot_data = np.ma.masked_where(mask, plot_data)

    # Plot main data
    if log_scale and np.any(plot_data > 0):
        plot_data = np.log10(plot_data + 1)
        label = f'log10({label})' if label else 'log10'

    im = ax.imshow(plot_data, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label=label, shrink=0.8)

    # Add overlay if provided
    if overlay_data is not None and overlay_cmap is not None:
        overlay_masked = np.ma.masked_where(~overlay_data, overlay_data.astype(float))
        ax.imshow(overlay_masked, cmap=overlay_cmap, alpha=0.7)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


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
):
    """Generate individual visualization images for each pipeline stage."""

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving images to: {output_dir}")

    # 1. Original DEM
    save_plot(
        dem, f'1. Original DEM ({dem.shape[0]}x{dem.shape[1]})',
        output_dir / '01_dem_original.png',
        cmap=CMAP_ELEVATION, label='Elevation (m)'
    )

    # 2. Ocean Mask
    ocean_pct = 100 * np.sum(ocean_mask) / ocean_mask.size
    save_plot(
        ocean_mask.astype(float), f'2. Ocean Mask ({ocean_pct:.1f}% ocean)',
        output_dir / '02_ocean_mask.png',
        cmap=CMAP_BINARY, label='Ocean (1) / Land (0)'
    )

    # 3. Conditioned DEM
    save_plot(
        dem_conditioned, '3. Conditioned DEM (depression-filled)',
        output_dir / '03_dem_conditioned.png',
        cmap=CMAP_ELEVATION, label='Elevation (m)'
    )

    # 4. Fill Depth (log scale for better visualization of small values)
    fill_depth = dem_conditioned - dem
    fill_depth[ocean_mask] = 0
    fill_max = np.max(fill_depth)
    save_plot(
        fill_depth, f'4. Depression Fill Depth (max={fill_max:.1f}m)',
        output_dir / '04_fill_depth.png',
        cmap=CMAP_FILL, label='m', log_scale=True
    )

    # 5. Flow Direction
    flow_display = flow_dir.astype(float)
    flow_display[flow_dir == 0] = np.nan
    save_plot(
        flow_display, '5. Flow Direction (D8 codes)',
        output_dir / '05_flow_direction.png',
        cmap=CMAP_DIRECTION, label='D8 code (1-128)'
    )

    # 6a. Drainage Area (log scale)
    save_plot(
        drainage_area, '6a. Drainage Area (log scale)',
        output_dir / '06a_drainage_area_log.png',
        cmap=CMAP_DRAINAGE, label='cells', log_scale=True
    )

    # 6b. Drainage Area (linear scale)
    save_plot(
        drainage_area, '6b. Drainage Area (linear)',
        output_dir / '06b_drainage_area_linear.png',
        cmap=CMAP_DRAINAGE, label='cells', log_scale=False
    )

    # 7a. Stream Network (top 5% drainage)
    threshold_5 = np.percentile(drainage_area[drainage_area > 1], 95)
    streams_5 = drainage_area > threshold_5
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(dem, cmap=CMAP_ELEVATION, alpha=0.6)
    stream_overlay = np.ma.masked_where(~streams_5, streams_5.astype(float))
    ax.imshow(stream_overlay, cmap=CMAP_STREAMS, alpha=0.9)
    plt.colorbar(im, ax=ax, label='Elevation (m)', shrink=0.8)
    ax.set_title(f'7a. Stream Network (top 5%, threshold={threshold_5:.0f} cells)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.tight_layout()
    plt.savefig(output_dir / '07a_stream_network_5pct.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 07a_stream_network_5pct.png")

    # 7b. Stream Network (top 10% drainage)
    threshold_10 = np.percentile(drainage_area[drainage_area > 1], 90)
    streams_10 = drainage_area > threshold_10
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(dem, cmap=CMAP_ELEVATION, alpha=0.6)
    stream_overlay = np.ma.masked_where(~streams_10, streams_10.astype(float))
    ax.imshow(stream_overlay, cmap=CMAP_STREAMS, alpha=0.9)
    plt.colorbar(im, ax=ax, label='Elevation (m)', shrink=0.8)
    ax.set_title(f'7b. Stream Network (top 10%, threshold={threshold_10:.0f} cells)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.tight_layout()
    plt.savefig(output_dir / '07b_stream_network_10pct.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 07b_stream_network_10pct.png")

    # 7c. Stream Network (top 15% drainage)
    threshold_15 = np.percentile(drainage_area[drainage_area > 1], 85)
    streams_15 = drainage_area > threshold_15
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(dem, cmap=CMAP_ELEVATION, alpha=0.6)
    stream_overlay = np.ma.masked_where(~streams_15, streams_15.astype(float))
    ax.imshow(stream_overlay, cmap=CMAP_STREAMS, alpha=0.9)
    plt.colorbar(im, ax=ax, label='Elevation (m)', shrink=0.8)
    ax.set_title(f'7c. Stream Network (top 15%, threshold={threshold_15:.0f} cells)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.tight_layout()
    plt.savefig(output_dir / '07c_stream_network_15pct.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 07c_stream_network_15pct.png")

    # 7d. Drainage Connectivity Violations (if any)
    if violation_mask is not None and drainage_violations > 0:
        fig, ax = plt.subplots(figsize=(12, 10))
        # Show drainage area log scale as background
        drainage_log = np.log10(drainage_area + 1)
        im = ax.imshow(drainage_log, cmap=CMAP_DRAINAGE, alpha=0.7)
        # Overlay violations in red
        violation_overlay = np.ma.masked_where(~violation_mask, np.ones_like(violation_mask, dtype=float))
        ax.imshow(violation_overlay, cmap='Reds', alpha=0.9, vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='log10(drainage area)', shrink=0.8)
        ax.set_title(f'7d. Drainage Connectivity Violations ({drainage_violations:,} cells)',
                     fontsize=14, fontweight='bold', color='darkred')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        plt.tight_layout()
        plt.savefig(output_dir / '07d_drainage_violations.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: 07d_drainage_violations.png")

    # 8. Precipitation Input
    if is_real_precip:
        precip_title = '8. WorldClim Annual Precipitation'
        precip_label = 'mm/year'
    else:
        precip_title = '8. Synthetic Precipitation (elevation-based)'
        precip_label = 'mm/year (synthetic)'
    save_plot(
        precip, precip_title,
        output_dir / '08_precipitation.png',
        cmap=CMAP_PRECIP, label=precip_label
    )

    # 9. Upstream Rainfall (log scale)
    save_plot(
        upstream_rainfall, '9. Upstream Rainfall',
        output_dir / '09_upstream_rainfall.png',
        cmap=CMAP_RAINFALL, label='mm·m²', log_scale=True
    )

    # 10. Validation Summary
    status = "PASS" if cycles == 0 and mass_balance > 95 and drainage_violations == 0 else "FAIL"
    if cycles == 0 and mass_balance > 95 and drainage_violations > 0:
        status = "WARN"  # Passing main checks but has connectivity issues
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')

    # Format violation text with color hint
    violation_text = f"{drainage_violations:>8,}"
    if drainage_violations > 0:
        violation_text = f"{drainage_violations:>8,} ⚠"

    summary_text = f"""
    ╔══════════════════════════════════════════════════════════╗
    ║              FLOW VALIDATION RESULTS                     ║
    ╠══════════════════════════════════════════════════════════╣
    ║                                                          ║
    ║  Status:        {status:>8}                                  ║
    ║                                                          ║
    ║  Cycles:        {cycles:>8} / {sample_size:<8}                     ║
    ║  Mass Balance:  {mass_balance:>8.1f}%                              ║
    ║  Drain. Viol.:  {violation_text}                            ║
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

    plt.savefig(output_dir / '10_validation_summary.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"  Saved: 10_validation_summary.png")

    return output_dir


def find_coastal_subset(full_dem: np.ndarray, size: int) -> tuple[np.ndarray, int, int]:
    """Find a good coastal subset with mixed terrain."""
    rows, cols = full_dem.shape
    best_subset = None
    best_score = -1

    for row in range(0, rows - size, size // 4):
        for col in range(0, cols - size, size // 4):
            subset = full_dem[row:row+size, col:col+size]
            ocean_frac = np.sum(subset <= 0) / subset.size
            elev_range = subset.max() - subset.min()

            # Score: want ~20-30% ocean and good elevation range
            if 0.1 < ocean_frac < 0.5 and elev_range > 50:
                # Score based on closeness to 25% ocean and elevation variety
                ocean_score = 1 - abs(ocean_frac - 0.25) * 4
                elev_score = min(elev_range / 500, 1)
                score = ocean_score * 0.6 + elev_score * 0.4

                if score > best_score:
                    best_score = score
                    best_subset = (row, col, ocean_frac, elev_range)

    if best_subset:
        row, col, ocean_frac, elev_range = best_subset
        dem = full_dem[row:row+size, col:col+size].copy()
        return dem, row, col
    else:
        # Fallback to center
        row = (rows - size) // 2
        col = (cols - size) // 2
        return full_dem[row:row+size, col:col+size].copy(), row, col


def main():
    parser = argparse.ArgumentParser(description="Complete flow validation with images")
    parser.add_argument('--bigness', type=str, default='small',
                        choices=['small', 'medium', 'large', 'full'],
                        help='Size of DEM to process (default: small)')
    parser.add_argument('--target-size', type=int, default=None,
                        help='Target resolution for full mode (default: 1000)')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output directory (default: examples/output/flow_validation_<bigness>)')
    parser.add_argument('--no-fill', action='store_true',
                        help='Skip depression filling (use raw DEM for flow direction)')
    parser.add_argument('--fill-small-sinks', type=int, default=None,
                        help='Fill sinks smaller than N cells (reduces fragmentation, e.g., 50)')
    parser.add_argument('--fill-method', type=str, default='breach',
                        choices=['fill', 'breach'],
                        help='Depression filling method: fill (complete) or breach (minimal, default)')
    parser.add_argument('--min-basin-depth', type=float, default=100.0,
                        help='Minimum basin depth (m) to preserve (default: 100.0)')
    parser.add_argument('--min-basin-size', type=int, default=50000,
                        help='Minimum basin size (cells) to preserve (default: 50000)')
    parser.add_argument('--high-res-params', action='store_true',
                        help='Use recommended parameters for high-resolution noisy DEMs (fill method, depth=2.0, fill_small_sinks=50)')
    args = parser.parse_args()

    # Apply high-resolution preset if requested
    if args.high_res_params:
        print("Applying high-resolution DEM parameters preset:")
        print("  - fill_method: fill")
        print("  - min_basin_depth: 2.0m")
        print("  - min_basin_size: 10000 cells")
        print("  - fill_small_sinks: 50 cells")
        args.fill_method = 'fill'
        args.min_basin_depth = 2.0
        args.min_basin_size = 10000
        if args.fill_small_sinks is None:
            args.fill_small_sinks = 50

    # Override target size for full mode if specified
    if args.target_size and args.bigness == 'full':
        BIGNESS_SIZES['full'] = args.target_size

    # Set output directory based on bigness
    if args.output is None:
        suffix = '_high_res' if args.high_res_params else ''
        args.output = Path(f'examples/output/flow_validation_{args.bigness}{suffix}')

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
        print(f"Full DEM: {full_dem.shape}, range {full_dem.min():.0f}-{full_dem.max():.0f}m")

    # Select subset based on bigness
    target_size = BIGNESS_SIZES[args.bigness]

    if args.bigness == 'full':
        # Use entire DEM but downsample to target resolution
        original_shape = full_dem.shape
        scale_factor = target_size / max(original_shape)

        if scale_factor < 1.0:
            print(f"\nDownsampling FULL DEM from {original_shape} to ~{target_size}x{target_size}...")
            dem = zoom(full_dem, scale_factor, order=1)  # Bilinear interpolation
            print(f"  Downsampled to: {dem.shape[0]}x{dem.shape[1]}")
            print(f"  Scale factor: {scale_factor:.3f}")
        else:
            dem = full_dem.copy()
            print(f"\nUsing FULL DEM ({dem.shape[0]}x{dem.shape[1]})...")
    else:
        dem, row, col = find_coastal_subset(full_dem, target_size)
        print(f"\nUsing {args.bigness} subset ({target_size}x{target_size}):")
        print(f"  Location: rows {row}-{row+target_size}, cols {col}-{col+target_size}")

    ocean_preview = 100 * np.sum(dem <= 0) / dem.size
    print(f"  Ocean coverage: ~{ocean_preview:.0f}%")
    print(f"  Elevation range: {dem.min():.0f} - {dem.max():.0f}m")

    print(f"\n{'='*60}")
    print(f"RUNNING FLOW PIPELINE (bigness={args.bigness})")
    print(f"{'='*60}")

    # Step 1: Detect ocean
    print("\n1. Detecting ocean...")
    ocean_mask = detect_ocean_mask(dem, threshold=0.0, border_only=True)
    print(f"   Ocean cells: {np.sum(ocean_mask):,} ({100*np.sum(ocean_mask)/dem.size:.1f}%)")

    # Step 2: Condition DEM - Fill shallow noise, preserve deep real basins
    if args.no_fill:
        print("\n2. Skipping DEM conditioning (--no-fill mode)...")
        dem_conditioned = dem.astype(np.float32).copy()
    else:
        # DEM noise creates high-frequency false sinks. Real basins are DEEP.
        # - Salton Sink: ~80m below sea level (should be preserved)
        # - Noise: typically < 5m depth (should be filled)
        #
        # Default: Use very high min_depth (100m) to only preserve the deepest basins
        # At 50m we get 751 depressions creating 317K scattered outlets
        # At 100m we get only 2 depressions and 6K outlets (mostly boundary)
        #
        # For high-res noisy DEMs: Use lower thresholds (2m depth) to preserve
        # real features while filling noise
        min_depth = args.min_basin_depth
        min_basin = args.min_basin_size

        fill_sinks_msg = f", fill_small_sinks={args.fill_small_sinks}" if args.fill_small_sinks else ""
        print(f"\n2. Conditioning DEM (method={args.fill_method}, min_depth={min_depth}m, min_basin={min_basin} cells{fill_sinks_msg})...")
        dem_conditioned = condition_dem(dem, method=args.fill_method, ocean_mask=ocean_mask,
                                        min_basin_size=min_basin, min_basin_depth=min_depth,
                                        fill_small_sinks=args.fill_small_sinks)

    # Step 3: Compute flow direction
    print("\n3. Computing flow direction...")
    flow_dir = compute_flow_direction(dem_conditioned, mask=ocean_mask)

    # Step 4: Compute drainage area
    print("\n4. Computing drainage area...")
    drainage_area = compute_drainage_area(flow_dir)

    # Step 5: Compute upstream rainfall
    print("\n5. Computing upstream rainfall...")

    # Try to load real WorldClim precipitation data
    precip, is_real_precip = load_real_precipitation(dem.shape, args.output)

    if not is_real_precip:
        # Fall back to synthetic elevation-based precipitation
        print("  No real precipitation data found, using synthetic (elevation-based)")
        precip = 200 + (dem - dem.min()) / (dem.max() - dem.min() + 1) * 300
        precip = precip.astype(np.float32)

    precip[ocean_mask] = 0
    upstream_rainfall = compute_upstream_rainfall(flow_dir, precip)

    # Validate
    print("\n6. Validating...")
    cycles, sample_size = validate_no_cycles(flow_dir, sample_size=1000)
    mass_balance = compute_mass_balance(flow_dir, drainage_area)
    drainage_violations, violation_mask = validate_drainage_connectivity(flow_dir, drainage_area)

    print(f"   Cycles: {cycles}/{sample_size}")
    print(f"   Mass Balance: {mass_balance:.1f}%")
    print(f"   Drainage violations: {drainage_violations:,} cells")

    # Trace actual flow paths to verify drainage increases downstream
    trace_flow_paths(flow_dir, drainage_area, num_traces=5)

    status = "PASS" if cycles == 0 and mass_balance > 95 else "FAIL"
    if drainage_violations > 0:
        status = "WARN" if status == "PASS" else status
        print(f"   WARNING: {drainage_violations} cells flow to neighbors with LESS drainage (indicates bug)")
    print(f"\n   Status: {status}")

    # Generate images
    print("\n7. Generating validation images...")
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
