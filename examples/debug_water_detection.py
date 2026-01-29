#!/usr/bin/env python3
"""
Debug script to analyze water detection slope values.
Shows what slope magnitudes are being produced by Horn's method.
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.terrain.core import (
    Terrain,
    load_dem_files,
    scale_elevation,
    flip_raster,
    reproject_raster,
)
from src.terrain.water import _calculate_slope

# SRTM tiles directory
SRTM_TILES_DIR = Path(__file__).parent.parent / "data" / "dem" / "detroit"

print("=" * 70)
print("Water Detection Debug - Analyzing Slope Values")
print("=" * 70)

# Load and transform DEM
print("\n[1/3] Loading and transforming DEM...")
dem_data, transform = load_dem_files(SRTM_TILES_DIR, pattern="*.hgt")
terrain = Terrain(dem_data, transform)

# Configure downsampling
target_vertices = 960 * 720 * 2
terrain.configure_for_target_vertices(target_vertices, method="average")

# Apply transforms
terrain.transforms.append(reproject_raster("EPSG:4326", "EPSG:32617", num_threads=4))
terrain.transforms.append(flip_raster(axis="horizontal"))
terrain.transforms.append(scale_elevation(scale_factor=0.0001))
terrain.apply_transforms()

# Get transformed DEM
transformed_dem = terrain.data_layers["dem"]["transformed_data"]
print(f"Transformed DEM shape: {transformed_dem.shape}")
print(f"DEM value range: {np.nanmin(transformed_dem):.4f} to {np.nanmax(transformed_dem):.4f}")

# Calculate slopes
print("\n[2/3] Computing slope using Horn's method...")
slopes = _calculate_slope(transformed_dem)

print(f"Slope value range: {np.nanmin(slopes):.6f} to {np.nanmax(slopes):.6f}")
print(f"Slope percentiles:")
print(f"  5th:  {np.nanpercentile(slopes, 5):.6f}")
print(f"  25th: {np.nanpercentile(slopes, 25):.6f}")
print(f"  50th: {np.nanpercentile(slopes, 50):.6f}")
print(f"  75th: {np.nanpercentile(slopes, 75):.6f}")
print(f"  95th: {np.nanpercentile(slopes, 95):.6f}")

# Test different thresholds
print("\n[3/3] Testing different water detection thresholds:")
test_thresholds = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]

for threshold in test_thresholds:
    water_count = np.sum(slopes < threshold)
    water_percent = 100 * water_count / slopes.size
    print(f"  Threshold {threshold:5.2f}: {water_count:8d} pixels ({water_percent:5.1f}%)")

print("\n" + "=" * 70)
print("RECOMMENDATION:")
print("Water should be ~5-15% of terrain. Choose threshold accordingly.")
print("=" * 70)
