"""Quick validation of flow algorithm on small real DEM subset."""

import sys
from pathlib import Path
import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).parent.parent))

print("Loading modules...", flush=True)
from src.terrain.flow_accumulation import (
    compute_flow_direction,
    compute_drainage_area,
    condition_dem,
    detect_ocean_mask,
    D8_OFFSETS,
)

def main():
    dem_path = Path("examples/output/merged_dem.tif")
    if not dem_path.exists():
        print(f"DEM not found at {dem_path}")
        return False

    print("Loading DEM subset (200x200)...", flush=True)
    with rasterio.open(dem_path) as src:
        full_shape = src.shape
        print(f"Full DEM: {full_shape}", flush=True)

        # Take a small 200x200 subset
        row_start = (full_shape[0] - 200) // 2
        col_start = (full_shape[1] - 200) // 2
        window = rasterio.windows.Window(col_start, row_start, 200, 200)
        dem = src.read(1, window=window)

    print(f"Subset: {dem.shape}, range {dem.min():.0f}-{dem.max():.0f}m", flush=True)

    print("Detecting ocean...", flush=True)
    ocean_mask = detect_ocean_mask(dem)
    print(f"Ocean cells: {np.sum(ocean_mask)}", flush=True)

    print("Conditioning DEM...", flush=True)
    dem_cond = condition_dem(dem, method="breach", ocean_mask=ocean_mask)

    print("Computing flow direction...", flush=True)
    flow_dir = compute_flow_direction(dem_cond, ocean_mask)

    print("Computing drainage area...", flush=True)
    drainage_area = compute_drainage_area(flow_dir)

    # Validate for cycles
    print("\nValidating...", flush=True)
    rows, cols = flow_dir.shape
    non_outlet = np.where(flow_dir > 0)
    sample_size = min(200, len(non_outlet[0]))
    np.random.seed(42)
    sample_idx = np.random.choice(len(non_outlet[0]), sample_size, replace=False)

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

    outlets = flow_dir == 0
    total_drainage = np.sum(drainage_area[outlets])
    mass_balance = 100 * total_drainage / flow_dir.size

    print(f"Cycles: {cycles}/{sample_size}", flush=True)
    print(f"Mass balance: {mass_balance:.1f}%", flush=True)

    success = cycles == 0 and mass_balance > 95
    print(f"\n{'PASS' if success else 'FAIL'}", flush=True)
    return success

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
