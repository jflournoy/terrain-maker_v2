"""Validate the fixed flow accumulation algorithm on real DEM data.

Tests on a 1000x1000 subset of the San Diego DEM to verify:
1. No cycles in flow network
2. Good mass balance (all cells drain to outlets)
3. Proper drainage area calculation
"""

import sys
from pathlib import Path
import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.terrain.flow_accumulation import (
    compute_flow_direction,
    compute_drainage_area,
    condition_dem,
    detect_ocean_mask,
    D8_OFFSETS,
)


def validate_flow_on_subset():
    """Test flow algorithm on a 1000x1000 subset of San Diego DEM."""
    dem_path = Path("examples/output/merged_dem.tif")

    if not dem_path.exists():
        print(f"DEM not found at {dem_path}")
        print("Please run san_diego_flow_demo.py first to generate the merged DEM.")
        return False

    print("Loading DEM subset (1000x1000 from center)...")
    with rasterio.open(dem_path) as src:
        full_shape = src.shape
        print(f"Full DEM shape: {full_shape}")

        # Take a 1000x1000 subset from the center
        row_start = (full_shape[0] - 1000) // 2
        col_start = (full_shape[1] - 1000) // 2

        window = rasterio.windows.Window(col_start, row_start, 1000, 1000)
        dem = src.read(1, window=window)

    print(f"Subset shape: {dem.shape}")
    print(f"Elevation range: {dem.min():.1f} to {dem.max():.1f}m")

    # Step 1: Detect ocean and condition the DEM
    print("\nDetecting ocean mask...")
    ocean_mask = detect_ocean_mask(dem)
    print(f"Ocean cells: {np.sum(ocean_mask):,}")

    print("Conditioning DEM (fill depressions, resolve flats)...")
    dem_cond = condition_dem(dem, method="breach", ocean_mask=ocean_mask)

    # Step 2: Compute flow direction
    print("Computing flow directions...")
    flow_dir = compute_flow_direction(dem_cond, ocean_mask)

    # Step 3: Compute drainage area
    print("Computing drainage area...")
    drainage_area = compute_drainage_area(flow_dir)

    # Validate: Check for cycles
    print("\n--- Validation ---")
    rows, cols = flow_dir.shape

    # Sample 500 random non-outlet cells
    non_outlet_coords = np.where(flow_dir > 0)
    sample_size = min(500, len(non_outlet_coords[0]))
    np.random.seed(42)
    sample_idx = np.random.choice(len(non_outlet_coords[0]), sample_size, replace=False)

    cycles_found = 0
    path_lengths = []

    for idx in sample_idx:
        start_i = non_outlet_coords[0][idx]
        start_j = non_outlet_coords[1][idx]

        i, j = start_i, start_j
        visited = set()
        path_len = 0

        while True:
            if (i, j) in visited:
                cycles_found += 1
                break
            visited.add((i, j))
            path_len += 1

            d = flow_dir[i, j]
            if d == 0:  # Outlet
                break

            if d not in D8_OFFSETS:
                break

            di, dj = D8_OFFSETS[d]
            ni, nj = i + di, j + dj

            if ni < 0 or ni >= rows or nj < 0 or nj >= cols:
                break  # Off grid = outlet

            i, j = ni, nj

            if path_len > 10000:  # Safety limit
                cycles_found += 1
                break

        path_lengths.append(path_len)

    print(f"Cycle detection (sampled {sample_size} cells):")
    print(f"  Cycles found: {cycles_found}")
    print(f"  Max path length: {max(path_lengths)}")
    print(f"  Avg path length: {np.mean(path_lengths):.1f}")

    # Mass balance check
    all_outlets = flow_dir == 0
    total_drainage_at_outlets = np.sum(drainage_area[all_outlets])
    total_cells = flow_dir.size
    mass_balance = 100 * total_drainage_at_outlets / total_cells

    print(f"\nMass balance:")
    print(f"  Total cells: {total_cells:,}")
    print(f"  Total drainage at outlets: {total_drainage_at_outlets:,.0f}")
    print(f"  Mass balance ratio: {mass_balance:.2f}%")

    # Outlets summary
    outlet_count = np.sum(all_outlets)
    interior_outlets = all_outlets.copy()
    interior_outlets[0, :] = False
    interior_outlets[-1, :] = False
    interior_outlets[:, 0] = False
    interior_outlets[:, -1] = False

    print(f"\nOutlets:")
    print(f"  Total outlets: {outlet_count:,}")
    print(f"  Interior outlets (ocean/sinks): {np.sum(interior_outlets):,}")
    print(f"  Boundary outlets: {outlet_count - np.sum(interior_outlets):,}")

    # Success criteria
    success = cycles_found == 0 and mass_balance > 95

    print(f"\n{'='*50}")
    if success:
        print("✓ VALIDATION PASSED")
        print("  - No cycles detected")
        print("  - Mass balance > 95%")
    else:
        print("✗ VALIDATION FAILED")
        if cycles_found > 0:
            print(f"  - {cycles_found} cycles detected!")
        if mass_balance <= 95:
            print(f"  - Mass balance only {mass_balance:.2f}%")

    return success


if __name__ == "__main__":
    success = validate_flow_on_subset()
    sys.exit(0 if success else 1)
