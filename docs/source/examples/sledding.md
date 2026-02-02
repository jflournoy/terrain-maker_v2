# Snow Integration: Sledding Location Analysis

Combine elevation data with SNODAS snow statistics to identify optimal sledding locations.

![Sledding Scores](../../images/05_final/sledding_score.png)

## Overview

This example demonstrates:
- Loading SNODAS snow depth/coverage data
- Multi-layer terrain analysis
- Score-based visualization
- Pipeline with visual outputs at each stage

## Quick Start

```bash
# With mock data (fast testing)
python examples/detroit_snow_sledding.py --mock-data

# With real SNODAS data
python examples/detroit_snow_sledding.py
```

## Processing Pipeline

The sledding score pipeline produces visual outputs at each stage, making it easy to debug and understand the analysis.

### Step 1: Raw Input Data

The pipeline starts with elevation (DEM) and snow depth data.

| DEM Elevation | Snow Depth |
|---------------|------------|
| ![DEM](../../images/01_raw/dem.png) | ![Snow Depth](../../images/01_raw/snow_depth.png) |

### Step 2: Slope Statistics

High-resolution slope analysis computes various terrain metrics.

| Mean Slope | Max Slope | Roughness |
|------------|-----------|-----------|
| ![Mean](../../images/02_slope_stats/mean.png) | ![Max](../../images/02_slope_stats/max.png) | ![Roughness](../../images/02_slope_stats/roughness.png) |

### Step 3: Slope Penalties

Penalty factors are computed for hazardous terrain.

| Cliff Penalty | Terrain Consistency | Combined Penalty |
|---------------|---------------------|------------------|
| ![Cliff](../../images/03_slope_penalties/cliff_penalty.png) | ![Consistency](../../images/03_slope_penalties/terrain_consistency.png) | ![Combined](../../images/03_slope_penalties/combined_penalty.png) |

### Step 4: Score Components

Individual score components before combination.

| Snow Score | Slope Score | Coverage Score |
|------------|-------------|----------------|
| ![Snow](../../images/04_score_components/snow_trapezoid_score.png) | ![Slope](../../images/04_score_components/slope_trapezoid_score.png) | ![Coverage](../../images/04_score_components/coverage_score.png) |

### Step 5: Final Score

The final sledding score combines all components.

![Final Sledding Score](../../images/05_final/sledding_score.png)

## The Code

```python
from src.terrain.core import Terrain
from src.terrain.scoring import compute_sledding_score
from src.terrain.data_loading import load_snodas_data
from src.terrain.color_mapping import elevation_colormap

# 1. Load terrain and snow data
terrain = Terrain(dem_data, transform)

# 2. Load and process SNODAS snow data
snow_depth, snow_coverage = load_snodas_data(snodas_dir)

# 3. Add snow as data layer (auto-reprojects to match DEM)
terrain.add_data_layer(
    "snow_depth",
    snow_depth,
    snow_transform,
    "EPSG:4326",
    target_layer="dem"
)

# 4. Calculate sledding score
score = compute_sledding_score(
    terrain,
    depth_weight=0.4,
    coverage_weight=0.3,
    slope_weight=0.3
)

# 5. Visualize with score-based coloring
terrain.set_color_mapping(
    lambda s: elevation_colormap(s, cmap_name='plasma'),
    source_layers=['sledding_score']
)
```

## Key Functions Used

| Function | Purpose |
|----------|---------|
| {func}`~terrain.core.Terrain.add_data_layer` | Add georeferenced data with auto-reprojection |
| {func}`~src.terrain.scoring.compute_sledding_score` | Compute sledding suitability |
| {func}`~terrain.data_loading.load_snodas_data` | Load SNODAS snow grids |
| {func}`~terrain.transforms.smooth_score_data` | Reduce blockiness in low-res data |

## Score Components

The sledding score combines:

1. **Snow Depth** (40%) - Deeper snow = better sledding
2. **Coverage** (30%) - Consistent coverage preferred
3. **Slope** (30%) - Sweet spot: 5-15° gradient

```python
# Custom weights
score = compute_sledding_score(
    terrain,
    depth_weight=0.5,    # Prioritize deep snow
    coverage_weight=0.2,
    slope_weight=0.3
)
```

## Output Files

| File | Description |
|------|-------------|
| `01_raw/dem.png` | Input elevation data |
| `01_raw/snow_depth.png` | SNODAS snow depth |
| `02_slope_stats/` | Slope analysis panels |
| `03_slope_penalties/` | Penalty visualizations |
| `05_final/sledding_score.png` | Final score map |

## See Also

- {doc}`elevation` - Basic terrain rendering
- {doc}`combined_render` - Dual-colormap visualization
- {func}`~src.terrain.scoring.compute_sledding_score` - Score calculation details
