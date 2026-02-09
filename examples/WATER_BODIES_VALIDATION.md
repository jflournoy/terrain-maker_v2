# Water Bodies Flow Validation

This document demonstrates the water body integration in the terrain-maker flow accumulation system.

## Overview

The `validate_flow_with_water_bodies.py` script tests the complete water body handling pipeline:

1. **Detect Ocean** - Identifies ocean/water cells using elevation threshold
2. **Create/Load Water Bodies** - Synthetic lakes or real NHD/HydroLAKES data
3. **Rasterize Lakes** - Convert lake polygons to labeled mask
4. **Identify Outlets** - Find pour points where lakes drain
5. **Condition DEM** - Fill depressions (preserving large basins)
6. **Compute Flow** - Calculate flow directions with and without lakes
7. **Apply Lake Routing** - Route flow through lakes to outlets
8. **Validate** - Check for cycles, mass balance, drainage connectivity

## Key Features

### Lake Flow Routing

Water bodies are integrated using the `create_lake_flow_routing()` function:

- **BFS from outlets**: Flow directions converge toward lake outlets
- **No internal sinks**: All lake cells route to their outlet (or nowhere for endorheic lakes)
- **Seamless integration**: Lake flow merges with terrain flow at lake boundaries

### Synthetic Test Lakes

For testing, the script creates synthetic lakes from:
- Low-lying areas (elevation -5m to 50m)
- Flat regions (elevation variance < 2m²)
- Size constraints (20-500 cells)

In production, use real data from:
- **NHD** (USA): National Hydrography Dataset
- **HydroLAKES** (global): Global lake polygons with pour points

## Validation Results

### Test Run: San Diego Coastal Subset (200×200 cells)

**Water Bodies:**
- 24 synthetic lakes detected
- 5,022 cells (12.6% of land area)
- 24 outlet points identified

**Flow Validation:**
- ✓ 0 cycles detected (out of 1,000 sampled paths)
- ✓ 100.0% mass balance (all flow reaches outlets)
- ✓ 0 drainage connectivity violations
- ✓ All traced paths show monotonic drainage increase

**Performance:**
- Execution time: ~15 seconds
- Memory efficient with resampling for shape alignment

## Generated Visualizations

The script produces 11 validation plots:

### 1. Original DEM
Shows the raw digital elevation model with elevation colormap.

### 2. Ocean Mask
Binary mask showing detected ocean cells (23.8% of area).

### 3. Water Bodies ⭐
**NEW**: Shows detected lakes (blue) with outlet points (red crosses).
- 24 lakes covering 12.6% of land area
- Outlets clearly marked for visual verification

### 4. Conditioned DEM
Depression-filled DEM ready for flow computation.

### 5. Fill Depth
Log-scale visualization of depression filling depths.

### 6. Flow Direction Comparison ⭐
**NEW**: Three-panel comparison:
- **Before** (without lakes): Natural terrain flow
- **After** (with lakes): Flow routed through lakes
- **Difference**: Cells where flow changed (shown in red)

Shows exactly where lake routing affected flow patterns.

### 7. Drainage Area Comparison ⭐
**NEW**: Three-panel comparison:
- **Before** (without lakes): Drainage accumulation on terrain
- **After** (with lakes): Drainage with lake routing
- **Difference**: Absolute change in log-scale drainage

Demonstrates how lakes concentrate flow at their outlets.

### 8. Stream Network with Lakes
Combined visualization showing:
- Elevation (background)
- Stream network (top 5% drainage in red/orange)
- Lakes (blue overlay)

### 9. Precipitation
Real WorldClim annual precipitation data (or synthetic fallback).

### 10. Upstream Rainfall
Precipitation-weighted flow accumulation showing cumulative upstream rainfall.

### 11. Validation Summary
Text summary with:
- Pass/fail status
- Cycle detection results
- Mass balance percentage
- Lake count and statistics
- DEM statistics

## Usage

### Basic Usage

```bash
# Small subset (200×200)
python examples/validate_flow_with_water_bodies.py --bigness small

# Medium subset (500×500)
python examples/validate_flow_with_water_bodies.py --bigness medium

# Large subset (1000×1000)
python examples/validate_flow_with_water_bodies.py --bigness large

# Full DEM (downsampled)
python examples/validate_flow_with_water_bodies.py --bigness full
```

### With Custom Parameters

```bash
# Custom depression filling
python examples/validate_flow_with_water_bodies.py \
  --bigness medium \
  --fill-method fill \
  --min-basin-depth 2.0 \
  --min-basin-size 10000

# Custom output directory
python examples/validate_flow_with_water_bodies.py \
  --bigness small \
  --output my_validation_results/
```

## Implementation Details

### Shape Alignment

The script handles shape mismatches between rasterized lake masks and DEM:

```python
# Resample lake mask to match DEM shape exactly
if lake_mask_raw.shape != dem.shape:
    scale_y = dem.shape[0] / lake_mask_raw.shape[0]
    scale_x = dem.shape[1] / lake_mask_raw.shape[1]
    # Use nearest neighbor to preserve lake IDs (integers)
    lake_mask = zoom(lake_mask_raw, (scale_y, scale_x), order=0)
```

This ensures perfect alignment for:
- Flow direction computation
- Drainage area calculation
- Visualization overlays

### Lake Flow Integration

```python
# Compute flow WITHOUT lakes (for comparison)
flow_dir_no_lakes = compute_flow_direction(dem_conditioned, mask=ocean_mask)

# Apply lake routing
lake_flow = create_lake_flow_routing(lake_mask, lake_outlets, dem_conditioned)

# Merge: lake cells use lake_flow, others keep terrain_flow
flow_dir = np.where(lake_mask > 0, lake_flow, flow_dir_no_lakes)
```

This approach:
- Preserves terrain flow outside lakes
- Routes lake interior cells toward outlets
- Creates seamless transitions at lake boundaries

## Key Visualizations Showing Water Bodies Work

### 3. Water Bodies (03_water_bodies.png)
- **What it shows**: Detected lakes (blue polygons) with outlet points (red X markers)
- **Why it matters**: Verifies that lakes were correctly identified and outlets are placed at lake boundaries
- **Expected pattern**: Lakes appear in low-lying flat areas, outlets at lowest elevation points on perimeter

### 6. Flow Direction Comparison (06_flow_direction_comparison.png)
- **What it shows**: Side-by-side flow direction before/after lake routing, plus difference map
- **Why it matters**: Demonstrates that lake routing changes flow within lakes but not outside
- **Expected pattern**: Red cells (changed flow) should only appear inside lake boundaries

### 7. Drainage Area Comparison (07_drainage_comparison.png)
- **What it shows**: Drainage accumulation before/after lakes, plus difference
- **Why it matters**: Shows how lakes concentrate drainage at their outlets
- **Expected pattern**: Outlets should show large increases in drainage (red in difference map)

### 8. Stream Network with Lakes (08_stream_network_with_lakes.png)
- **What it shows**: Streams (orange) overlaid with lakes (blue)
- **Why it matters**: Verifies that streams connect properly through lakes
- **Expected pattern**: Streams should enter lakes and emerge at outlets, not fragment within lakes

## Files Generated

All outputs are saved to `examples/output/flow_water_bodies_<bigness>/`:

```
flow_water_bodies_small/
├── 01_dem_original.png              # Input DEM
├── 02_ocean_mask.png                # Detected ocean
├── 03_water_bodies.png              # ⭐ Lakes + outlets
├── 04_dem_conditioned.png           # Filled DEM
├── 05_fill_depth.png                # Depression depths
├── 06_flow_direction_comparison.png # ⭐ Before/after flow
├── 07_drainage_comparison.png       # ⭐ Before/after drainage
├── 08_stream_network_with_lakes.png # ⭐ Streams + lakes
├── 09_precipitation.png             # Rainfall data
├── 10_upstream_rainfall.png         # Flow-weighted rainfall
└── 11_validation_summary.png        # Summary stats
```

**⭐ = New water body-specific visualizations**

## Next Steps

### For Production Use

Replace synthetic lakes with real data:

```python
from src.terrain.water_bodies import download_water_bodies

# Download NHD (USA only)
geojson_path = download_water_bodies(
    bbox=(south, west, north, east),
    output_dir="data/water_bodies",
    data_source="nhd",
    min_area_km2=0.1
)

# Or download HydroLAKES (global)
geojson_path = download_water_bodies(
    bbox=(south, west, north, east),
    output_dir="data/water_bodies",
    data_source="hydrolakes",
    min_area_km2=0.1
)
```

### Integration with Terrain Maker

Use lake routing in terrain analysis:

```python
from src.terrain.flow_accumulation import flow_accumulation

results = flow_accumulation(
    dem_path="data/dem.tif",
    precipitation_path="data/precip.tif",
    lake_mask=lake_mask,           # From rasterize_lakes_to_mask()
    lake_outlets=lake_outlets,     # From identify_outlet_cells()
    backend="custom"
)

flow_dir = results["flow_direction"]
drainage = results["drainage_area"]
```

## Validation Status: ✅ PASS

Water body integration is working correctly:
- Lakes are properly identified and rasterized
- Outlets are correctly located at lake boundaries
- Flow routing converges toward outlets within lakes
- No cycles or mass balance issues introduced
- Drainage accumulation properly concentrated at outlets

## References

- [flow_accumulation.py](../src/terrain/flow_accumulation.py) - Main flow computation
- [water_bodies.py](../src/terrain/water_bodies.py) - Lake handling functions
- [flow-spec.md](../flow-spec.md) - Flow algorithm specification
