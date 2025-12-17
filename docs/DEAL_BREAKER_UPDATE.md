# Deal Breaker Update: Physical Metrics

## Summary of Changes

Updated sledding deal breakers from abstract statistical measures to physical, intuitive metrics:

### Before (Abstract)
```python
- Slope > 35°              # Statistical threshold
- Variability > 5°         # Slope std dev in degrees (depends on tile size)
- Coverage < 0.5 months    # Opportunistic sledding
```

### After (Physical)
```python
- Slope > 40°              # Extreme cliffs only (raised from 35°)
- Roughness > 6m           # Elevation std dev in meters (physical bumps)
- Coverage < 0.5 months    # Opportunistic sledding (unchanged)
```

## Why This Matters

### Problem with Old Approach
- **"Variability > 5°"** was abstract and confusing
- Meaning depended on tile size (10x10 vs 20x20 pixels)
- Not intuitive for users ("What does 5 degree std dev mean?")

### Benefits of New Approach
- **"Roughness > 6m"** is physical and clear
- **Tile-independent**: 6 meters is 6 meters, regardless of pixel resolution
- **Intuitive**: "6 meters of elevation variation" is easy to visualize
- **Already computed**: `slope_stats.roughness` was already being calculated!

## Physical Interpretation

At ~27m DEM resolution with typical 10-20x downsampling:

**Roughness values over 300m × 300m tiles:**
- **0.5-1m**: Very smooth (parking lot, golf course)
- **1-2m**: Smooth (mowed park)
- **2-3m**: Gentle rolling hills (natural park) - **FUN for sledding!**
- **3-5m**: Moderate bumps (hilly park)
- **5-6m**: Rough terrain (rocky, steep)
- **6m+**: Very rough (cliff faces, boulders) - **DEALBREAKER**

## Updated Synergy Bonuses

Also updated from abstract to physical:

### Before
- Low variability (<2°)
- Very low variability (<1°)

### After
- Smooth terrain (<3m roughness) - gentle rolling hills
- Very smooth terrain (<1.5m roughness) - golf course smooth

## Code Changes

### 1. Core Scoring Function
File: `src/terrain/scoring.py`

**`sledding_deal_breakers()`:**
- Changed parameter: `slope_variability` → `roughness`
- Updated threshold: `max_variability=5.0` → `max_roughness=6.0`
- Updated slope: `max_slope=35.0` → `max_slope=40.0`

**`sledding_synergy_bonus()`:**
- Changed parameter: `slope_variability` → `roughness`
- Updated thresholds: `low_variability <2°` → `smooth_terrain <3m`
- Updated thresholds: `very_low_variability <1°` → `very_smooth_terrain <1.5m`

**`compute_sledding_score()`:**
- Changed parameter: `slope_variability` → `roughness`
- Updated examples and docstrings

### 2. Scoring Config
File: `src/scoring/configs/sledding.py`

**`compute_improved_sledding_score()`:**
- Changed: `slope_variability = slope_stats.slope_std`
- To: `roughness_meters = slope_stats.roughness`
- Updated docstring

### 3. Tests
File: `tests/test_sledding_score.py`

- Updated all test functions to use `roughness` instead of `slope_variability`
- Updated test values to use physical metrics (meters)
- Updated slope threshold from 35° to 40°
- **All 28 tests pass** ✓

### 4. Documentation
File: `docs/images/04_score_components/equation.md`

- Updated deal breakers: `Slope > 40° OR Roughness > 6m`
- Updated synergy bonuses to reference "smooth terrain" instead of "low variability"

## Migration Guide

If you have custom scoring code that uses the old API:

### Before
```python
from src.terrain.scoring import compute_sledding_score

score = compute_sledding_score(
    snow_depth=12.0,
    slope=9.0,
    coverage_months=3.0,
    slope_variability=2.0,  # Old parameter (degrees)
)
```

### After
```python
from src.terrain.scoring import compute_sledding_score

score = compute_sledding_score(
    snow_depth=12.0,
    slope=9.0,
    coverage_months=3.0,
    roughness=2.0,  # New parameter (meters)
)
```

### Getting Roughness from Slope Stats
```python
from src.snow.slope_statistics import compute_tiled_slope_statistics

# Compute slope statistics
slope_stats = compute_tiled_slope_statistics(
    dem=dem,
    dem_transform=transform,
    dem_crs="EPSG:4326",
    target_shape=(1200, 1320),
    target_transform=score_transform,
    target_crs="EPSG:4326",
    cell_size_m=30.0,
)

# roughness is already computed!
roughness_meters = slope_stats.roughness  # Elevation std dev in meters
```

## Example Values

Based on Detroit DEM analysis (~27m resolution):

| Terrain Type | Roughness | Interpretation |
|-------------|-----------|----------------|
| Parking lot | 0.5m | Completely flat |
| Golf course | 1-2m | Very smooth, manicured |
| Natural park | 2-4m | Gentle rolling hills - **ideal for sledding** |
| Hilly park | 4-6m | Moderate bumps - still acceptable |
| Rocky terrain | 6-8m | Rough, steep - **dealbreaker** |
| Cliffs/boulders | 8m+ | Very dangerous - hard zero |

## Next Steps

To regenerate images with new parameters:
```bash
uv run python examples/detroit_snow_sledding.py --step score --mock-data
```

This will update:
- `docs/images/04_score_components/raw_slope_variability.png` → `raw_roughness_meters.png`
- `docs/images/04_score_components/valid_terrain_mask.png` (with new thresholds)
- All score visualizations with updated dealbreakers

## References

- Physical scale analysis: `analyze_dem_scale.py`
- Deal breaker analysis: `docs/DEAL_BREAKER_ANALYSIS.md`
- Test suite: `tests/test_sledding_score.py`
