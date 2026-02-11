"""
Verify that drainage area is computed from the conditioned DEM, not the raw DEM.

This script loads a small DEM, computes drainage area from:
1. Raw DEM (unmodified)
2. Conditioned DEM (after breaching and filling)

Then creates side-by-side plots to verify differences exist.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.terrain.data_loading import load_dem_files
from src.terrain.flow_accumulation import (
    compute_flow_direction,
    compute_drainage_area,
    detect_ocean_mask,
    condition_dem_spec,
)


def load_san_diego_dem():
    """Load San Diego DEM for testing."""
    dem_dir = Path(__file__).parent.parent / "data" / "san_diego_dem"
    if not dem_dir.exists():
        print(f"Error: DEM directory not found at {dem_dir}")
        print("Please download San Diego DEM first")
        return None, None

    dem, transform = load_dem_files(str(dem_dir))
    print(f"Loaded DEM: {dem.shape}")
    print(f"  Value range: {dem.min():.2f} to {dem.max():.2f} m")
    return dem, transform


def main():
    print("=" * 70)
    print("DRAINAGE AREA VERIFICATION")
    print("=" * 70)

    # Load DEM
    dem, transform = load_san_diego_dem()
    if dem is None:
        return 1

    # Detect ocean
    print("\nDetecting ocean mask...")
    ocean_mask = detect_ocean_mask(dem, threshold=0.0, border_only=True)
    print(f"  Ocean cells: {np.sum(ocean_mask):,}")

    # Compute from raw DEM
    print("\n" + "=" * 70)
    print("COMPUTING FROM RAW DEM")
    print("=" * 70)
    print("Computing flow direction from raw DEM...")
    flow_dir_raw = compute_flow_direction(dem, mask=ocean_mask)
    print(f"  Flow direction shape: {flow_dir_raw.shape}")

    print("Computing drainage area from raw DEM...")
    drainage_raw = compute_drainage_area(flow_dir_raw)
    print(f"  Drainage area range: {drainage_raw.min():.1f} to {drainage_raw.max():.1f}")
    print(f"  Non-zero cells: {np.sum(drainage_raw > 0):,}")

    # Condition DEM
    print("\n" + "=" * 70)
    print("CONDITIONING DEM")
    print("=" * 70)
    print("Conditioning DEM (breaching + filling)...")
    dem_conditioned, outlets, breached_dem = condition_dem_spec(
        dem,
        nodata_mask=ocean_mask,
        coastal_elev_threshold=0.0,
        edge_mode="all",
        max_breach_depth=25.0,
        max_breach_length=150,
        epsilon=1e-4,
    )
    print(f"  Conditioned DEM range: {dem_conditioned.min():.2f} to {dem_conditioned.max():.2f} m")
    print(f"  Breached DEM range: {breached_dem.min():.2f} to {breached_dem.max():.2f} m")

    # Compute differences
    dem_diff = dem - dem_conditioned
    breach_depth = dem - breached_dem
    print(f"  DEM change range: {dem_diff.min():.2f} to {dem_diff.max():.2f} m")
    print(f"  Breach depth range: {breach_depth.min():.2f} to {breach_depth.max():.2f} m")
    print(f"  Cells modified by conditioning: {np.sum(np.abs(dem_diff) > 1e-6):,}")
    print(f"  Cells modified by breaching: {np.sum(np.abs(breach_depth) > 1e-6):,}")

    # Compute from conditioned DEM
    print("\n" + "=" * 70)
    print("COMPUTING FROM CONDITIONED DEM")
    print("=" * 70)
    print("Computing flow direction from conditioned DEM...")
    flow_dir_cond = compute_flow_direction(dem_conditioned, mask=ocean_mask)
    print(f"  Flow direction shape: {flow_dir_cond.shape}")

    print("Computing drainage area from conditioned DEM...")
    drainage_cond = compute_drainage_area(flow_dir_cond)
    print(f"  Drainage area range: {drainage_cond.min():.1f} to {drainage_cond.max():.1f}")
    print(f"  Non-zero cells: {np.sum(drainage_cond > 0):,}")

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    drainage_diff = np.abs(drainage_raw - drainage_cond)
    print(f"Drainage area differences:")
    print(f"  Max difference: {drainage_diff.max():.1f} cells")
    print(f"  Mean difference: {drainage_diff.mean():.1f} cells")
    print(f"  Cells with differences: {np.sum(drainage_diff > 0):,}")

    if np.sum(drainage_diff > 0) == 0:
        print("\n⚠️  WARNING: Drainage areas are IDENTICAL!")
        print("   This suggests conditioning had no effect on flow routing.")
    else:
        print(f"\n✓ Drainage areas DIFFER - conditioning affects flow routing")
        print(f"  {100 * np.sum(drainage_diff > 0) / dem.size:.2f}% of cells have different drainage area")

    # Create visualization
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    output_dir = Path(__file__).parent.parent / "examples" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Drainage Area Verification: Raw vs Conditioned DEM", fontsize=16, fontweight="bold")

    # Row 1: Raw DEM analysis
    axes[0, 0].imshow(dem, cmap="gray")
    axes[0, 0].set_title("Raw DEM")
    axes[0, 0].set_ylabel("Row")

    axes[0, 1].imshow(breach_depth, cmap="RdYlBu_r", vmin=0, vmax=50)
    axes[0, 1].set_title("Breach Depth (dem - breached)")
    cbar = plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1])
    cbar.set_label("meters")

    im = axes[0, 2].imshow(np.log10(drainage_raw + 1), cmap="viridis")
    axes[0, 2].set_title("Drainage Area from Raw DEM (log10)")
    cbar = plt.colorbar(im, ax=axes[0, 2])
    cbar.set_label("log10(cells)")

    # Row 2: Conditioned DEM analysis
    axes[1, 0].imshow(dem_conditioned, cmap="gray")
    axes[1, 0].set_title("Conditioned DEM")
    axes[1, 0].set_ylabel("Row")
    axes[1, 0].set_xlabel("Column")

    im = axes[1, 1].imshow(drainage_diff, cmap="hot")
    axes[1, 1].set_title("Drainage Area Difference (|raw - conditioned|)")
    axes[1, 1].set_xlabel("Column")
    cbar = plt.colorbar(im, ax=axes[1, 1])
    cbar.set_label("cells")

    im = axes[1, 2].imshow(np.log10(drainage_cond + 1), cmap="viridis")
    axes[1, 2].set_title("Drainage Area from Conditioned DEM (log10)")
    axes[1, 2].set_xlabel("Column")
    cbar = plt.colorbar(im, ax=axes[1, 2])
    cbar.set_label("log10(cells)")

    plt.tight_layout()
    output_file = output_dir / "drainage_area_verification.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved comparison plot: {output_file}")

    # Save numerical summary
    summary_file = output_dir / "drainage_area_verification.txt"
    with open(summary_file, "w") as f:
        f.write("DRAINAGE AREA VERIFICATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"DEM Shape: {dem.shape}\n")
        f.write(f"DEM Range: {dem.min():.2f} to {dem.max():.2f} m\n\n")

        f.write("BREACHING STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Cells modified by conditioning: {np.sum(np.abs(dem_diff) > 1e-6):,}\n")
        f.write(f"Cells modified by breaching: {np.sum(np.abs(breach_depth) > 1e-6):,}\n")
        f.write(f"Max breach depth: {breach_depth.max():.2f} m\n")
        f.write(f"Mean breach depth: {breach_depth[breach_depth > 0].mean():.2f} m\n\n")

        f.write("DRAINAGE AREA COMPARISON\n")
        f.write("-" * 70 + "\n")
        f.write(f"Raw DEM drainage range: {drainage_raw.min():.1f} to {drainage_raw.max():.1f} cells\n")
        f.write(f"Conditioned DEM drainage range: {drainage_cond.min():.1f} to {drainage_cond.max():.1f} cells\n\n")

        f.write(f"Drainage area differences:\n")
        f.write(f"  Max difference: {drainage_diff.max():.1f} cells\n")
        f.write(f"  Mean difference: {drainage_diff.mean():.1f} cells\n")
        f.write(f"  Cells with differences: {np.sum(drainage_diff > 0):,}\n")
        f.write(f"  Percentage of domain: {100 * np.sum(drainage_diff > 0) / dem.size:.2f}%\n\n")

        if np.sum(drainage_diff > 0) == 0:
            f.write("⚠️  WARNING: Drainage areas are IDENTICAL!\n")
            f.write("This suggests conditioning had no effect on flow routing.\n")
        else:
            f.write("✓ Drainage areas DIFFER - conditioning affects flow routing\n")

    print(f"✓ Saved summary report: {summary_file}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
