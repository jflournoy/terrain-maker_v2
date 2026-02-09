"""
Diagnose flow outlet handling.

This script analyzes the flow direction and accumulation to understand:
1. Where all outlets are (not just ocean)
2. How much flow reaches each outlet
3. Whether there are problematic interior "dead ends"
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import rasterio

# D8 direction encoding
D8_OFFSETS = {
    1: (0, 1),    # E
    2: (-1, 1),   # NE
    4: (-1, 0),   # N
    8: (-1, -1),  # NW
    16: (0, -1),  # W
    32: (1, -1),  # SW
    64: (1, 0),   # S
    128: (1, 1),  # SE
}

D8_NAMES = {
    1: "East",
    2: "Northeast",
    4: "North",
    8: "Northwest",
    16: "West",
    32: "Southwest",
    64: "South",
    128: "Southeast",
    0: "OUTLET",
}


def diagnose_outlets():
    """Analyze flow outlets in detail."""
    flow_dir_path = "examples/output/flow_outputs/flow_direction.tif"
    drainage_path = "examples/output/flow_outputs/flow_accumulation_area.tif"
    dem_cond_path = "examples/output/flow_outputs/dem_conditioned.tif"

    print("Loading flow data...")
    with rasterio.open(flow_dir_path) as src:
        flow_dir = src.read(1)

    with rasterio.open(drainage_path) as src:
        drainage_area = src.read(1)

    with rasterio.open(dem_cond_path) as src:
        dem = src.read(1)

    rows, cols = flow_dir.shape
    print(f"Grid size: {rows} x {cols} = {rows * cols:,} cells")

    # Find all outlets (direction = 0)
    all_outlets = flow_dir == 0
    outlet_count = np.sum(all_outlets)
    print(f"\nTotal outlets (direction=0): {outlet_count:,}")

    # Classify outlets by location
    boundary_top = all_outlets[0, :]
    boundary_bottom = all_outlets[-1, :]
    boundary_left = all_outlets[:, 0]
    boundary_right = all_outlets[:, -1]

    # Interior outlets (not on any boundary)
    interior_outlets = all_outlets.copy()
    interior_outlets[0, :] = False
    interior_outlets[-1, :] = False
    interior_outlets[:, 0] = False
    interior_outlets[:, -1] = False

    print(f"\nOutlet locations:")
    print(f"  Top edge (row 0):    {np.sum(boundary_top):,}")
    print(f"  Bottom edge (row {rows-1}): {np.sum(boundary_bottom):,}")
    print(f"  Left edge (col 0):   {np.sum(boundary_left):,}")
    print(f"  Right edge (col {cols-1}): {np.sum(boundary_right):,}")
    print(f"  Interior outlets:    {np.sum(interior_outlets):,}")

    # Ocean outlets (elevation <= 0)
    ocean_mask = dem <= 0
    ocean_outlets = all_outlets & ocean_mask
    land_outlets = all_outlets & ~ocean_mask

    print(f"\nOutlet types:")
    print(f"  Ocean outlets (elev <= 0m):  {np.sum(ocean_outlets):,}")
    print(f"  Land outlets (elev > 0m):    {np.sum(land_outlets):,}")

    # Drainage captured by each outlet type
    print(f"\nDrainage area by outlet type:")
    print(f"  Ocean outlets capture: {np.sum(drainage_area[ocean_outlets]):,.0f} cells")
    print(f"  Land outlets capture:  {np.sum(drainage_area[land_outlets]):,.0f} cells")

    # Find the largest outlets
    outlet_drainage = []
    outlet_coords = np.where(all_outlets)
    for i, j in zip(outlet_coords[0], outlet_coords[1]):
        da = drainage_area[i, j]
        elev = dem[i, j]
        edge = "interior"
        if i == 0:
            edge = "top"
        elif i == rows - 1:
            edge = "bottom"
        elif j == 0:
            edge = "left"
        elif j == cols - 1:
            edge = "right"
        outlet_drainage.append((da, i, j, elev, edge))

    outlet_drainage.sort(reverse=True)

    print(f"\nTop 20 outlets by drainage area:")
    print(f"  {'Rank':<5} {'Drainage':>12} {'Row':>6} {'Col':>6} {'Elev':>8} {'Edge':<10}")
    print(f"  {'-'*5} {'-'*12} {'-'*6} {'-'*6} {'-'*8} {'-'*10}")
    for rank, (da, i, j, elev, edge) in enumerate(outlet_drainage[:20], 1):
        print(f"  {rank:<5} {da:>12,.0f} {i:>6} {j:>6} {elev:>8.1f} {edge:<10}")

    # Check for cycles: cells that have a receiver but receiver doesn't ultimately reach an outlet
    print(f"\n--- Checking for flow network issues ---")

    # Count cells by direction
    print(f"\nFlow direction distribution:")
    for d in [0, 1, 2, 4, 8, 16, 32, 64, 128]:
        count = np.sum(flow_dir == d)
        pct = 100 * count / flow_dir.size
        print(f"  {d:>3} ({D8_NAMES.get(d, '?'):<12}): {count:>10,} ({pct:>5.2f}%)")

    # Check for invalid directions
    valid_dirs = {0, 1, 2, 4, 8, 16, 32, 64, 128}
    invalid_mask = ~np.isin(flow_dir, list(valid_dirs))
    if np.any(invalid_mask):
        print(f"\n  WARNING: {np.sum(invalid_mask):,} cells with invalid direction!")
    else:
        print(f"\n  All flow directions are valid (0, 1, 2, 4, 8, 16, 32, 64, 128)")

    # Check for cycles by tracing from random cells
    print(f"\n--- Cycle detection (sampling) ---")
    np.random.seed(42)

    # Sample 1000 random non-outlet cells
    non_outlet_coords = np.where(flow_dir > 0)
    if len(non_outlet_coords[0]) > 0:
        sample_size = min(1000, len(non_outlet_coords[0]))
        sample_idx = np.random.choice(len(non_outlet_coords[0]), sample_size, replace=False)

        cycles_found = 0
        max_path_length = 0
        path_lengths = []

        for idx in sample_idx:
            start_i = non_outlet_coords[0][idx]
            start_j = non_outlet_coords[1][idx]

            # Trace path to outlet
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

                if path_len > 100000:  # Safety limit
                    cycles_found += 1
                    break

            path_lengths.append(path_len)
            max_path_length = max(max_path_length, path_len)

        print(f"  Sampled {sample_size} random cells")
        print(f"  Cycles detected: {cycles_found}")
        print(f"  Max path length: {max_path_length}")
        print(f"  Avg path length: {np.mean(path_lengths):.1f}")

    # Find one cycle and trace it
    if cycles_found > 0:
        print(f"\n--- Tracing a cycle ---")
        for idx in sample_idx:
            start_i = non_outlet_coords[0][idx]
            start_j = non_outlet_coords[1][idx]

            i, j = start_i, start_j
            visited = {}
            step = 0

            while True:
                if (i, j) in visited:
                    cycle_start = visited[(i, j)]
                    print(f"  Cycle found! Cell ({i}, {j}) revisited at step {step}, first seen at step {cycle_start}")
                    print(f"  Cycle length: {step - cycle_start}")

                    # Trace the cycle
                    print(f"  Cycle path:")
                    ci, cj = i, j
                    for _ in range(min(step - cycle_start + 1, 10)):
                        d = flow_dir[ci, cj]
                        elev = dem[ci, cj]
                        print(f"    ({ci}, {cj}) elev={elev:.3f}m dir={d} ({D8_NAMES.get(d, '?')})")
                        if d in D8_OFFSETS:
                            di, dj = D8_OFFSETS[d]
                            ci, cj = ci + di, cj + dj
                        else:
                            break
                    break

                visited[(i, j)] = step
                step += 1

                d = flow_dir[i, j]
                if d == 0:
                    break
                if d not in D8_OFFSETS:
                    break

                di, dj = D8_OFFSETS[d]
                ni, nj = i + di, j + dj

                if ni < 0 or ni >= rows or nj < 0 or nj >= cols:
                    break

                i, j = ni, nj

                if step > 100000:
                    break

            if (i, j) in visited:
                break  # Found a cycle, stop searching

    # Mass balance check
    print(f"\n--- Mass balance analysis ---")
    total_cells = flow_dir.size
    total_drainage_at_outlets = np.sum(drainage_area[all_outlets])

    print(f"  Total cells in grid: {total_cells:,}")
    print(f"  Total drainage at all outlets: {total_drainage_at_outlets:,.0f}")
    print(f"  Ratio: {100 * total_drainage_at_outlets / total_cells:.2f}%")

    # What about cells that don't drain anywhere?
    # These would be cells with direction != 0 but no path to an outlet

    print(f"\nDone.")


if __name__ == "__main__":
    diagnose_outlets()
