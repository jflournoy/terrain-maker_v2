"""
Quick test to verify flow accumulation fixes.

This creates a synthetic DEM with known drainage patterns and checks
that the flow accumulation produces reasonable results.

Run with:
    python examples/test_flow_fix.py
"""

import numpy as np
import tempfile
from pathlib import Path
import rasterio
from rasterio import Affine
from src.terrain.flow_accumulation import flow_accumulation
import matplotlib.pyplot as plt


def create_test_dem(path, dem_type='slope'):
    """Create synthetic DEM for testing."""
    if dem_type == 'slope':
        # Simple east-west slope with ocean on west side
        rows, cols = 200, 300
        dem = np.zeros((rows, cols), dtype=np.float32)

        for j in range(cols):
            dem[:, j] = j * 5.0  # Elevation increases eastward

        # Ocean on left side
        dem[:, :10] = -10.0

    elif dem_type == 'valley':
        # V-shaped valley
        rows, cols = 200, 200
        dem = np.zeros((rows, cols), dtype=np.float32)

        for i in range(rows):
            for j in range(cols):
                # Distance from center
                center = cols // 2
                dist = abs(j - center)
                # V-shaped: elevation increases away from center
                dem[i, j] = 100.0 + dist * 2.0 - i * 1.0

    else:
        raise ValueError(f"Unknown dem_type: {dem_type}")

    # Write to GeoTIFF
    transform = Affine.translation(0, 0) * Affine.scale(30, -30)  # 30m pixels
    with rasterio.open(
        path, 'w',
        driver='GTiff',
        height=rows, width=cols,
        count=1, dtype=dem.dtype,
        crs='EPSG:4326',
        transform=transform,
    ) as dst:
        dst.write(dem, 1)

    return dem


def create_test_precip(path, shape):
    """Create uniform precipitation raster."""
    precip = np.full(shape, 500.0, dtype=np.float32)

    transform = Affine.translation(0, 0) * Affine.scale(30, -30)
    with rasterio.open(
        path, 'w',
        driver='GTiff',
        height=shape[0], width=shape[1],
        count=1, dtype=precip.dtype,
        crs='EPSG:4326',
        transform=transform,
    ) as dst:
        dst.write(precip, 1)

    return precip


def plot_results(dem, result, output_path):
    """Plot DEM and drainage area side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # DEM
    im0 = axes[0].imshow(dem, cmap='terrain')
    axes[0].set_title('Input DEM')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], label='Elevation (m)')

    # Flow direction
    flow_dir = result['flow_direction']
    im1 = axes[1].imshow(flow_dir, cmap='tab20')
    axes[1].set_title('Flow Direction (D8)')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], label='Direction code')

    # Drainage area (log scale)
    drainage = result['drainage_area']
    drainage_log = np.log10(drainage + 1)
    im2 = axes[2].imshow(drainage_log, cmap='viridis')
    axes[2].set_title(f'Drainage Area\nMax: {np.max(drainage):,.0f} cells')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], label='log10(drainage area)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")


def test_flow_accumulation(dem_type='slope'):
    """Test flow accumulation with synthetic DEM."""
    print(f"\nTesting flow accumulation with {dem_type} DEM...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test data
        dem_path = tmpdir / 'test_dem.tif'
        precip_path = tmpdir / 'test_precip.tif'

        dem = create_test_dem(dem_path, dem_type=dem_type)
        precip = create_test_precip(precip_path, dem.shape)

        # Run flow accumulation
        result = flow_accumulation(
            dem_path=str(dem_path),
            precipitation_path=str(precip_path),
            output_dir=str(tmpdir / 'output'),
            mask_ocean=True,
            fill_method='fill',
            min_basin_size=None,  # Fill all basins for clean test
        )

        # Analyze results
        drainage = result['drainage_area']
        flow_dir = result['flow_direction']

        # Count outlets
        outlets = (flow_dir == 0) & (drainage > 1)
        n_outlets = np.sum(outlets)

        # Check for fragmentation
        print(f"  Shape: {drainage.shape}")
        print(f"  Max drainage: {np.max(drainage):,.0f} cells")
        print(f"  Mean drainage: {np.mean(drainage):.1f} cells")
        print(f"  Outlets: {n_outlets:,}")

        # Check for reasonable drainage
        max_drainage = np.max(drainage)
        total_cells = drainage.size

        if dem_type == 'slope':
            # For a simple slope, most water should go to single outlet
            # Max drainage should be significant fraction of total
            if max_drainage < total_cells * 0.5:
                print(f"  ⚠️  WARNING: Max drainage is only {max_drainage/total_cells:.1%} of total cells")
                print(f"     This suggests fragmentation!")
            else:
                print(f"  ✓ Good drainage convergence: {max_drainage/total_cells:.1%} of cells")

            if n_outlets > 100:
                print(f"  ⚠️  WARNING: Too many outlets ({n_outlets}), suggests fragmentation!")
            else:
                print(f"  ✓ Reasonable number of outlets")

        # Save plot
        output_dir = Path('examples/output/flow_test')
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_results(dem, result, output_dir / f'{dem_type}_test.png')

        return result


if __name__ == '__main__':
    print("="*60)
    print("Testing flow accumulation algorithm")
    print("="*60)

    # Test with simple slope (most sensitive to bugs)
    result_slope = test_flow_accumulation('slope')

    # Test with valley
    result_valley = test_flow_accumulation('valley')

    print("\n" + "="*60)
    print("Tests complete!")
    print("="*60)
    print("\nIf you see warnings about fragmentation, the algorithm still has issues.")
    print("If results look good, the fix is working!")
