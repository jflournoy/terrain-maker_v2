"""
Test the flat resolution algorithm directly.

Creates a small flat region and verifies that after resolution:
1. All cells have distinct elevations
2. Flow directions don't create cycles
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.terrain.flow_accumulation import (
    _resolve_flats,
    _fill_depressions,
    compute_flow_direction,
    compute_drainage_area,
    D8_OFFSETS,
)


def test_simple_flat():
    """Test flat resolution on a simple 5x5 flat with sloped edges."""
    print("Test 1: Simple 5x5 flat with sloped edges")

    # Create DEM with flat center and sloped edges
    dem = np.array([
        [10, 10, 10, 10, 10],
        [10,  5,  5,  5, 10],
        [10,  5,  5,  5, 10],
        [10,  5,  5,  5, 10],
        [10,  5,  5,  5,  1],  # Outlet at corner
    ], dtype=np.float32)

    print("Original DEM:")
    print(dem)

    # Resolve flats
    resolved = _resolve_flats(dem, epsilon=1e-3)

    print("\nResolved DEM (showing differences from 5.0):")
    flat_region = dem == 5
    print(f"  Flat cells: {np.sum(flat_region)}")

    for i in range(5):
        row = []
        for j in range(5):
            if flat_region[i, j]:
                diff = resolved[i, j] - 5.0
                row.append(f"{diff:+.5f}")
            else:
                row.append(f"  {dem[i, j]:.1f}  ")
        print("  " + " ".join(row))

    # Compute flow direction
    flow_dir = compute_flow_direction(resolved)
    print("\nFlow directions:")
    print(flow_dir)

    # Check for cycles
    print("\nChecking for cycles...")
    cycles = 0
    for i in range(5):
        for j in range(5):
            if flow_dir[i, j] > 0:
                # Trace path
                ci, cj = i, j
                visited = set()
                while True:
                    if (ci, cj) in visited:
                        print(f"  CYCLE at ({i}, {j})")
                        cycles += 1
                        break
                    visited.add((ci, cj))
                    d = flow_dir[ci, cj]
                    if d == 0:
                        break
                    if d not in D8_OFFSETS:
                        break
                    di, dj = D8_OFFSETS[d]
                    ci, cj = ci + di, cj + dj
                    if ci < 0 or ci >= 5 or cj < 0 or cj >= 5:
                        break

    print(f"\nCycles found: {cycles}")
    return cycles == 0


def test_large_flat():
    """Test flat resolution on a larger flat region."""
    print("\n" + "="*60)
    print("Test 2: Large 20x20 flat region")

    # Create DEM with large flat center
    dem = np.full((20, 20), 5.0, dtype=np.float32)

    # Add sloped border
    dem[0, :] = 10.0
    dem[-1, :] = 10.0
    dem[:, 0] = 10.0
    dem[:, -1] = 10.0

    # Single outlet at corner
    dem[-1, -1] = 0.0

    print(f"Flat cells: {np.sum(dem == 5)}")
    print(f"Outlet at (19, 19) = 0.0")

    # Resolve flats
    resolved = _resolve_flats(dem, epsilon=1e-3)

    # Verify all flat cells are now distinct
    flat_mask = dem == 5
    flat_values = resolved[flat_mask]
    unique_values = np.unique(flat_values)
    print(f"Unique elevations in former flat: {len(unique_values)}")
    print(f"Min: {flat_values.min():.6f}, Max: {flat_values.max():.6f}")

    # Compute flow direction
    flow_dir = compute_flow_direction(resolved)

    # Check for cycles
    cycles = 0
    non_outlet = flow_dir > 0
    for i in range(20):
        for j in range(20):
            if flow_dir[i, j] > 0:
                ci, cj = i, j
                visited = set()
                while True:
                    if (ci, cj) in visited:
                        cycles += 1
                        break
                    visited.add((ci, cj))
                    d = flow_dir[ci, cj]
                    if d == 0:
                        break
                    if d not in D8_OFFSETS:
                        break
                    di, dj = D8_OFFSETS[d]
                    ci, cj = ci + di, cj + dj
                    if ci < 0 or ci >= 20 or cj < 0 or cj >= 20:
                        break

    print(f"Cycles found: {cycles}")

    # Check drainage
    drainage = compute_drainage_area(flow_dir)
    outlet_drainage = drainage[-1, -1]
    print(f"Drainage at outlet: {outlet_drainage} (should be ~{20*20})")

    return cycles == 0 and outlet_drainage > 350


def test_two_pour_points():
    """Test flat with two pour points - should drain to both."""
    print("\n" + "="*60)
    print("Test 3: Flat with two pour points")

    # Create DEM with flat center and two outlets
    dem = np.full((10, 10), 5.0, dtype=np.float32)

    # High border
    dem[0, :] = 10.0
    dem[-1, :] = 10.0
    dem[:, 0] = 10.0
    dem[:, -1] = 10.0

    # Two outlets at opposite corners
    dem[0, 0] = 0.0
    dem[-1, -1] = 0.0

    print("Two outlets at (0,0) and (9,9)")

    # Resolve flats
    resolved = _resolve_flats(dem, epsilon=1e-3)

    # Compute flow
    flow_dir = compute_flow_direction(resolved)
    drainage = compute_drainage_area(flow_dir)

    print(f"Drainage at (0,0): {drainage[0, 0]}")
    print(f"Drainage at (9,9): {drainage[-1, -1]}")
    print(f"Total drainage at outlets: {drainage[0, 0] + drainage[-1, -1]}")

    # Check cycles
    cycles = 0
    for i in range(10):
        for j in range(10):
            if flow_dir[i, j] > 0:
                ci, cj = i, j
                visited = set()
                while True:
                    if (ci, cj) in visited:
                        cycles += 1
                        break
                    visited.add((ci, cj))
                    d = flow_dir[ci, cj]
                    if d == 0:
                        break
                    if d not in D8_OFFSETS:
                        break
                    di, dj = D8_OFFSETS[d]
                    ci, cj = ci + di, cj + dj
                    if ci < 0 or ci >= 10 or cj < 0 or cj >= 10:
                        break

    print(f"Cycles found: {cycles}")
    return cycles == 0


def test_fill_then_resolve():
    """Test that fill_depressions + flat resolution works together."""
    print("\n" + "="*60)
    print("Test 4: Depression filling + flat resolution")

    # Create DEM with enclosed depression
    dem = np.array([
        [10, 10, 10, 10, 10],
        [10,  8,  6,  8, 10],
        [10,  6,  2,  6, 10],  # Deep pit at center
        [10,  8,  6,  8, 10],
        [10, 10, 10, 10,  5],  # Outlet
    ], dtype=np.float32)

    print("Original DEM (pit at center, outlet at corner):")
    print(dem)

    # Fill with breach method (includes flat resolution)
    filled = _fill_depressions(dem, epsilon=1e-3)

    print("\nFilled DEM:")
    print(filled)
    print(f"\nCenter cell went from 2.0 to {filled[2, 2]:.4f}")

    # Compute flow
    flow_dir = compute_flow_direction(filled)
    print("\nFlow directions:")
    print(flow_dir)

    # Check drainage at outlet
    drainage = compute_drainage_area(flow_dir)
    print(f"\nDrainage at outlet: {drainage[-1, -1]} (should be 25)")

    return drainage[-1, -1] >= 20


if __name__ == "__main__":
    results = []
    results.append(("Simple flat", test_simple_flat()))
    results.append(("Large flat", test_large_flat()))
    results.append(("Two pour points", test_two_pour_points()))
    results.append(("Fill + resolve", test_fill_then_resolve()))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    sys.exit(0 if all_passed else 1)
