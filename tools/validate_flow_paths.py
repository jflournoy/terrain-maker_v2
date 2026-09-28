"""
Visualize actual flow paths and stream networks.

Shows:
1. Stream network extracted from drainage area
2. Flow paths traced from random points
3. Flow vectors overlaid on terrain
"""

import sys
from pathlib import Path
import numpy as np

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))


def trace_flow_path(flow_dir, start_row, start_col, max_steps=10000):
    """
    Trace a flow path from a starting point to outlet.

    Parameters
    ----------
    flow_dir : np.ndarray
        D8 flow direction grid
    start_row : int
        Starting row
    start_col : int
        Starting column
    max_steps : int
        Maximum steps to prevent infinite loops

    Returns
    -------
    list of tuples
        List of (row, col) coordinates along path
    """
    # D8 direction offsets
    D8_OFFSETS = {
        1: (0, 1),     # East
        2: (-1, 1),    # Northeast
        4: (-1, 0),    # North
        8: (-1, -1),   # Northwest
        16: (0, -1),   # West
        32: (1, -1),   # Southwest
        64: (1, 0),    # South
        128: (1, 1),   # Southeast
    }

    path = [(start_row, start_col)]
    row, col = start_row, start_col
    rows, cols = flow_dir.shape

    for _ in range(max_steps):
        direction = flow_dir[row, col]

        # Stop at outlet (direction = 0)
        if direction == 0 or direction not in D8_OFFSETS:
            break

        # Move to next cell
        dr, dc = D8_OFFSETS[direction]
        row, col = row + dr, col + dc

        # Check bounds
        if not (0 <= row < rows and 0 <= col < cols):
            break

        # Check for cycles
        if (row, col) in path:
            break

        path.append((row, col))

    return path


def extract_stream_network(drainage_area, threshold_cells):
    """
    Extract stream network from drainage area.

    Parameters
    ----------
    drainage_area : np.ndarray
        Drainage area in cells
    threshold_cells : int
        Minimum drainage area to be considered a stream

    Returns
    -------
    np.ndarray
        Binary mask where 1 = stream
    """
    return drainage_area >= threshold_cells


def create_flow_path_plots(output_dir: Path):
    """Create flow path visualization plots."""
    import rasterio
    from rasterio.warp import reproject, Resampling

    print("Loading flow data...")

    # Input paths
    flow_dir_path = "examples/output/flow_outputs/flow_direction.tif"
    drainage_path = "examples/output/flow_outputs/flow_accumulation_area.tif"
    dem_cond_path = "examples/output/flow_outputs/dem_conditioned.tif"

    # Load data
    with rasterio.open(flow_dir_path) as src:
        flow_dir = src.read(1)

    with rasterio.open(drainage_path) as src:
        drainage_area = src.read(1)

    with rasterio.open(dem_cond_path) as src:
        dem = src.read(1)

    print(f"Data shape: {dem.shape}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Stream network
    print("1/4: Stream network...")
    fig, ax = plt.subplots(figsize=(12, 10))

    # Show elevation as background
    ax.imshow(dem, cmap='terrain', alpha=0.6, interpolation='nearest')

    # Extract streams at different thresholds
    stream_threshold = 1000  # cells
    streams = extract_stream_network(drainage_area, stream_threshold)

    # Overlay streams
    stream_mask = np.ma.masked_where(~streams, streams)
    ax.imshow(stream_mask, cmap='Blues', alpha=0.8, interpolation='nearest')

    ax.set_title(f'Stream Network (drainage ≥ {stream_threshold:,} cells)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Column (pixels)')
    ax.set_ylabel('Row (pixels)')

    # Add stats
    stream_pixels = np.sum(streams)
    stream_pct = 100 * stream_pixels / streams.size
    ax.text(0.02, 0.98, f'Stream pixels: {stream_pixels:,} ({stream_pct:.1f}%)',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_dir / '1_stream_network.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Stream network (multiple thresholds)
    print("2/4: Multi-threshold streams...")
    fig, ax = plt.subplots(figsize=(12, 10))

    # Background
    ax.imshow(dem, cmap='gray', alpha=0.4, interpolation='nearest')

    # Multiple threshold levels
    thresholds = [500, 2000, 10000]
    colors = ['lightblue', 'blue', 'darkblue']
    labels = []

    for thresh, color in zip(thresholds, colors):
        streams = extract_stream_network(drainage_area, thresh)
        stream_mask = np.ma.masked_where(~streams, streams)
        ax.imshow(stream_mask, cmap=matplotlib.colors.ListedColormap([color]),
                  alpha=0.6, interpolation='nearest')
        labels.append(f'≥{thresh:,} cells')

    ax.set_title('Stream Network (Multiple Thresholds)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Column (pixels)')
    ax.set_ylabel('Row (pixels)')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, alpha=0.6, label=l)
                       for c, l in zip(colors, labels)]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_dir / '2_stream_thresholds.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Flow paths from random points
    print("3/4: Traced flow paths...")
    fig, ax = plt.subplots(figsize=(12, 10))

    # Background: elevation
    ax.imshow(dem, cmap='terrain', alpha=0.5, interpolation='nearest')

    # Trace paths from random high-elevation points
    np.random.seed(42)
    n_paths = 50

    # Find high elevation points to start from
    high_elev_mask = dem > np.percentile(dem, 75)
    high_elev_coords = np.argwhere(high_elev_mask)

    # Sample random starting points
    if len(high_elev_coords) > n_paths:
        indices = np.random.choice(len(high_elev_coords), n_paths, replace=False)
        start_points = high_elev_coords[indices]
    else:
        start_points = high_elev_coords

    # Trace paths
    paths_traced = 0
    for start_row, start_col in start_points:
        path = trace_flow_path(flow_dir, start_row, start_col)

        if len(path) > 10:  # Only plot paths with at least 10 steps
            path_array = np.array(path)
            ax.plot(path_array[:, 1], path_array[:, 0],
                    color='blue', alpha=0.4, linewidth=0.8)
            paths_traced += 1

    ax.set_title(f'Flow Paths Traced from {paths_traced} High-Elevation Points',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Column (pixels)')
    ax.set_ylabel('Row (pixels)')

    ax.text(0.02, 0.98, f'{paths_traced} paths traced\nStarting from elevation > 75th percentile',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.savefig(output_dir / '3_flow_paths.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 4: Flow vectors (sampled)
    print("4/4: Flow direction vectors...")
    fig, ax = plt.subplots(figsize=(12, 10))

    # Background
    ax.imshow(dem, cmap='terrain', alpha=0.6, interpolation='nearest')

    # Sample flow direction vectors (every N pixels)
    sample_rate = max(1, dem.shape[0] // 50)  # Aim for ~50 vectors per side

    # D8 direction offsets
    D8_OFFSETS = {
        1: (0, 1),     # East
        2: (-1, 1),    # Northeast
        4: (-1, 0),    # North
        8: (-1, -1),   # Northwest
        16: (0, -1),   # West
        32: (1, -1),   # Southwest
        64: (1, 0),    # South
        128: (1, 1),   # Southeast
    }

    # Create vector field
    for i in range(0, dem.shape[0], sample_rate):
        for j in range(0, dem.shape[1], sample_rate):
            direction = flow_dir[i, j]
            if direction in D8_OFFSETS:
                dr, dc = D8_OFFSETS[direction]
                # Scale arrows
                ax.arrow(j, i, dc * sample_rate * 0.6, dr * sample_rate * 0.6,
                        head_width=sample_rate * 0.3,
                        head_length=sample_rate * 0.4,
                        fc='darkblue', ec='darkblue', alpha=0.7)

    ax.set_title(f'Flow Direction Vectors (sampled every {sample_rate} pixels)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Column (pixels)')
    ax.set_ylabel('Row (pixels)')

    plt.tight_layout()
    plt.savefig(output_dir / '4_flow_vectors.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Saved 4 flow path plots to {output_dir}/")

    # Print summary
    print("\n" + "="*60)
    print("FLOW PATH VISUALIZATION SUMMARY")
    print("="*60)
    print(f"✓ Stream network extracted (threshold: {stream_threshold:,} cells)")
    print(f"✓ Flow paths traced: {paths_traced}")
    print(f"✓ Vector field sampled: every {sample_rate} pixels")
    print("="*60)


def main():
    """Run flow path visualization."""
    output_dir = Path("examples/output/flow-validation")
    create_flow_path_plots(output_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
