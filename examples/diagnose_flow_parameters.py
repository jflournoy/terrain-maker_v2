"""
Diagnostic script to test flow accumulation parameters.

Tests different combinations of:
- fill_method: 'fill' vs 'breach'
- min_basin_depth: How deep a depression must be to preserve
- max_fill_depth: Maximum depth to fill
- fill_small_sinks: Whether to fill small remaining sinks

Run with:
    python examples/diagnose_flow_parameters.py --dem data/your_dem.tif --precip data/your_precip.tif
"""

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from src.terrain.flow_accumulation import flow_accumulation


def plot_drainage_comparison(results, output_dir):
    """Plot drainage area for different parameter combinations."""
    n_configs = len(results)
    fig, axes = plt.subplots(2, (n_configs + 1) // 2, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (config_name, result) in enumerate(results.items()):
        drainage = result['drainage_area']

        # Log scale for better visualization
        drainage_log = np.log10(drainage + 1)

        im = axes[idx].imshow(drainage_log, cmap='viridis')
        axes[idx].set_title(f"{config_name}\nMax: {np.max(drainage):.0f} cells")
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx], label='log10(drainage area)')

    # Hide unused subplots
    for idx in range(n_configs, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'drainage_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to {output_dir / 'drainage_comparison.png'}")


def analyze_flow_stats(result, config_name):
    """Print statistics about flow accumulation results."""
    drainage = result['drainage_area']
    flow_dir = result['flow_direction']

    # Count outlets (cells with flow_dir = 0 and drainage > 1)
    outlets = (flow_dir == 0) & (drainage > 1)
    n_outlets = np.sum(outlets)

    # Find cells with no flow direction
    no_flow = (flow_dir == 0) & (drainage == 1)
    n_no_flow = np.sum(no_flow)

    # Drainage statistics
    max_drainage = np.max(drainage)
    mean_drainage = np.mean(drainage)

    # Check for fragmentation (many small outlets)
    outlet_drainages = drainage[outlets]
    if len(outlet_drainages) > 0:
        median_outlet_drainage = np.median(outlet_drainages)
    else:
        median_outlet_drainage = 0

    print(f"\n{config_name}:")
    print(f"  Outlets: {n_outlets:,}")
    print(f"  Isolated cells (no flow): {n_no_flow:,}")
    print(f"  Max drainage area: {max_drainage:,.0f} cells")
    print(f"  Mean drainage area: {mean_drainage:.1f} cells")
    print(f"  Median outlet drainage: {median_outlet_drainage:.0f} cells")

    # Fragmentation indicator: if we have many outlets with small drainage,
    # the DEM is fragmented
    if n_outlets > 100 and median_outlet_drainage < 1000:
        print(f"  ⚠️  WARNING: High fragmentation detected (many small outlets)")

    return {
        'n_outlets': n_outlets,
        'n_no_flow': n_no_flow,
        'max_drainage': max_drainage,
        'median_outlet_drainage': median_outlet_drainage,
    }


def main():
    parser = argparse.ArgumentParser(description='Diagnose flow accumulation parameters')
    parser.add_argument('--dem', type=str, required=True, help='Path to DEM file')
    parser.add_argument('--precip', type=str, required=True, help='Path to precipitation file')
    parser.add_argument('--output-dir', type=str, default='examples/output/flow_diagnostics',
                        help='Output directory for results')
    parser.add_argument('--max-cells', type=int, default=10000,
                        help='Max cells for flow computation (downsample larger DEMs)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Testing flow accumulation with different parameters...")
    print(f"DEM: {args.dem}")
    print(f"Precipitation: {args.precip}")
    print(f"Max cells: {args.max_cells:,}")

    # Test configurations
    configs = {
        'default_breach': {
            'fill_method': 'breach',
            'min_basin_size': 10000,
            'min_basin_depth': 0.5,
        },
        'aggressive_fill': {
            'fill_method': 'fill',
            'min_basin_size': 10000,
            'min_basin_depth': 2.0,  # Only preserve deeper basins
        },
        'no_basin_preserve': {
            'fill_method': 'fill',
            'min_basin_size': None,  # Fill everything
            'min_basin_depth': 0.5,
        },
        'breach_higher_threshold': {
            'fill_method': 'breach',
            'min_basin_size': 10000,
            'min_basin_depth': 2.0,  # Higher threshold
        },
    }

    results = {}
    stats = {}

    for config_name, config_params in configs.items():
        print(f"\n{'='*60}")
        print(f"Testing: {config_name}")
        print(f"Parameters: {config_params}")
        print(f"{'='*60}")

        result = flow_accumulation(
            dem_path=args.dem,
            precipitation_path=args.precip,
            output_dir=str(output_dir / config_name),
            max_cells=args.max_cells,
            **config_params
        )

        results[config_name] = result
        stats[config_name] = analyze_flow_stats(result, config_name)

    # Plot comparison
    plot_drainage_comparison(results, output_dir)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    best_config = min(stats.items(), key=lambda x: x[1]['n_outlets'])
    print(f"\nBest configuration (fewest outlets): {best_config[0]}")
    print(f"  Outlets: {best_config[1]['n_outlets']:,}")
    print(f"  Max drainage: {best_config[1]['max_drainage']:,.0f} cells")

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
