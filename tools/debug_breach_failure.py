"""
Debug why all breaches are failing.

Check:
1. Are outlets being created correctly?
2. What's the distance from sinks to nearest outlet?
3. Is max_breach_length too restrictive?
"""

import numpy as np
from scipy.ndimage import distance_transform_edt

# Create a simple test case that mimics the problem
def test_breach_failure():
    """Test why breaches might fail."""

    # Simple DEM: 10x10 with a depression in the middle
    dem = np.array([
        [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        [10,  9,  9,  9,  9,  9,  9,  9,  9, 10],
        [10,  9,  8,  8,  8,  8,  8,  8,  9, 10],
        [10,  9,  8,  7,  7,  7,  7,  8,  9, 10],
        [10,  9,  8,  7,  5,  5,  7,  8,  9, 10],  # Depression at (4,4) and (4,5)
        [10,  9,  8,  7,  5,  5,  7,  8,  9, 10],
        [10,  9,  8,  7,  7,  7,  7,  8,  9, 10],
        [10,  9,  8,  8,  8,  8,  8,  8,  9, 10],
        [10,  9,  9,  9,  9,  9,  9,  9,  9, 10],
        [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    ], dtype=np.float32)

    # Test different outlet configurations

    print("Test 1: Edge outlets (edge_mode='all')")
    print("=" * 60)
    outlets1 = np.zeros_like(dem, dtype=bool)
    outlets1[0, :] = True  # Top edge
    outlets1[-1, :] = True  # Bottom edge
    outlets1[:, 0] = True  # Left edge
    outlets1[:, -1] = True  # Right edge
    print(f"Outlets: {np.sum(outlets1)} cells")

    # Find distance from depression to nearest outlet
    dist = distance_transform_edt(~outlets1)
    print(f"Distance from (4,4) to nearest outlet: {dist[4, 4]:.1f} cells")
    print(f"Distance from (4,5) to nearest outlet: {dist[4, 5]:.1f} cells")
    print()

    print("Test 2: No edge outlets (edge_mode='none')")
    print("=" * 60)
    outlets2 = np.zeros_like(dem, dtype=bool)
    print(f"Outlets: {np.sum(outlets2)} cells")
    if np.sum(outlets2) == 0:
        print("WARNING: No outlets! All breaches will fail!")
    print()

    print("Test 3: Coastal outlets only (coastal_elev_threshold=10.0)")
    print("=" * 60)
    # Simulate coastal outlets (cells at elevation <= 10.0 adjacent to nodata)
    nodata_mask = np.zeros_like(dem, dtype=bool)
    nodata_mask[0, :] = True  # Simulate ocean at top

    outlets3 = np.zeros_like(dem, dtype=bool)
    # Check cells adjacent to nodata
    for i in range(1, dem.shape[0]):
        for j in range(dem.shape[1]):
            if not nodata_mask[i, j]:  # Not nodata
                # Check if adjacent to nodata
                for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < dem.shape[0] and 0 <= nj < dem.shape[1]:
                        if nodata_mask[ni, nj]:
                            # Adjacent to nodata, check elevation
                            if dem[i, j] <= 10.0:
                                outlets3[i, j] = True
                                break

    print(f"Outlets: {np.sum(outlets3)} cells")
    if np.sum(outlets3) > 0:
        dist3 = distance_transform_edt(~outlets3)
        print(f"Distance from (4,4) to nearest outlet: {dist3[4, 4]:.1f} cells")
    else:
        print("No coastal outlets found!")
    print()

    print("Test 4: What if max_breach_length is too short?")
    print("=" * 60)
    max_lengths = [10, 50, 100, 200]
    for max_len in max_lengths:
        # Count how many depression cells are reachable
        reachable = np.sum(dist <= max_len)
        depression_reachable = dist[4, 4] <= max_len
        print(f"max_breach_length={max_len:3d}: {reachable:3d} cells reachable, depression reachable: {depression_reachable}")


if __name__ == "__main__":
    test_breach_failure()

    print()
    print("=" * 60)
    print("DIAGNOSIS QUESTIONS:")
    print("=" * 60)
    print("1. Are outlets being created at all?")
    print("   → Check if outlet count > 0")
    print()
    print("2. Are sinks too far from outlets?")
    print("   → If distance > max_breach_length, breach will fail")
    print()
    print("3. What edge_mode is being used?")
    print("   → 'all' creates outlets at all boundaries")
    print("   → 'none' creates NO outlets (only coastal)")
    print("   → 'local_minima' only marks local minima")
    print()
    print("4. Is max_breach_length too restrictive?")
    print("   → Try increasing from 100 to 500")
