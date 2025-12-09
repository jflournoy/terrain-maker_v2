# Scoring Configuration Guide

This guide explains how to use and customize the terrain suitability scoring system.

## Overview

The scoring system combines multiple terrain and snow statistics into a single suitability score (0-1). It uses:

- **Transformation functions** to convert raw values into 0-1 scores
- **A combiner** to weight and combine scores using additive and multiplicative rules

## Architecture

```
Raw Statistics → Transformations → Component Scores → Combiner → Final Score
                                        ↓
                    [additive: weighted sum] × [multiplicative: penalties]
```

## Transformation Types

All transformations output values in the range [0, 1].

### 1. Trapezoidal

For values with a "sweet spot" - ideal range with ramp-up and ramp-down zones.

```
Score
  1 |        ___________
    |       /           \
    |      /             \
  0 |_____/               \_____
         |   |       |   |
       ramp sweet  sweet ramp
       start start   end  end
```

**Use for:** slope angle, snow depth, any metric with an ideal middle range

**Parameters:**
- `sweet_range`: (start, end) - values in this range score 1.0
- `ramp_range`: (start, end) - full range including ramps

**Example:**
```json
{
  "name": "slope_mean",
  "transform": "trapezoidal",
  "transform_params": {
    "sweet_range": [5, 15],
    "ramp_range": [3, 25]
  },
  "role": "additive",
  "weight": 0.25
}
```

### 2. Dealbreaker

Step function for hard cutoffs with optional soft falloff. Used for safety-critical thresholds.

```
Score (hard)          Score (soft)
  1 |_______            1 |_______
    |       |             |       \
  0 |       |_____      0 |        \_____
         threshold              threshold
```

**Use for:** cliff detection, dangerous conditions, absolute limits

**Parameters:**
- `threshold`: cutoff point
- `falloff`: width of soft transition (0 = hard cutoff)
- `below_is_good`: if True, values below threshold are safe (default)

**Example:**
```json
{
  "name": "slope_p95",
  "transform": "dealbreaker",
  "transform_params": {
    "threshold": 25,
    "falloff": 10
  },
  "role": "multiplicative"
}
```

### 3. Linear

Simple normalization with optional power scaling and inversion.

```
Score
  1 |          /
    |        /
    |      /
  0 |____/
        min    max
```

**Use for:** ratios, coverage percentages, variability metrics

**Parameters:**
- `value_range`: (min, max) to normalize
- `invert`: if True, high values → low scores (for CV, variability)
- `power`: scaling power (0.5 = sqrt for diminishing returns)

**Example:**
```json
{
  "name": "snow_coverage",
  "transform": "linear",
  "transform_params": {
    "value_range": [0, 1],
    "power": 0.5
  },
  "role": "additive",
  "weight": 0.20
}
```

### 4. Terrain Consistency

Combined metric for roughness + slope variability using RMS.

**Parameters:**
- `roughness`: elevation std dev in meters
- `slope_std`: slope std dev in degrees
- `roughness_threshold`: meters (default: 30)
- `slope_std_threshold`: degrees (default: 10)

## Component Roles

### Additive (Weighted Sum)

Components that contribute to the base score. Weights must sum to 1.0.

```python
base_score = Σ (weight_i × score_i)
```

**Properties:**
- Must have a `weight` (0.0-1.0)
- All weights must sum to 1.0
- Each contributes independently to the score

### Multiplicative (Penalties)

Components that can reduce the score. No weight needed.

```python
final_score = base_score × penalty_1 × penalty_2 × ...
```

**Properties:**
- No weight (applies as multiplier)
- Score of 1.0 = no penalty
- Score of 0.0 = complete disqualification
- Multiple penalties compound

## Complete Formula

```python
final_score = (
    w1 × additive_1 +
    w2 × additive_2 +
    ...
) × multiplicative_1 × multiplicative_2 × ...
```

## Default Sledding Configuration

Located at: `src/scoring/configs/sledding_default.json`

### Additive Components (sum to 1.0)

| Component | Transform | Weight | Description |
|-----------|-----------|--------|-------------|
| `slope_mean` | trapezoidal | 25% | Ideal: 5-15°, usable: 3-25° |
| `snow_depth` | trapezoidal | 25% | Ideal: 150-500mm (~6-20") |
| `snow_coverage` | linear (√) | 20% | Fraction of days with snow |
| `snow_consistency` | linear (inv) | 15% | Year-to-year reliability |
| `aspect_bonus` | linear | 10% | North-facing snow retention |
| `runout_bonus` | linear | 5% | Safe stopping area |

### Multiplicative Components (penalties)

| Component | Transform | Description |
|-----------|-----------|-------------|
| `slope_p95` | dealbreaker | Cliff detection (25° threshold) |
| `terrain_consistency` | linear | Combined roughness + slope_std |

## Customizing the Config

### Option 1: Edit JSON

```bash
# Copy default config
cp src/scoring/configs/sledding_default.json my_config.json

# Edit thresholds, weights, etc.
```

```python
import json
from src.scoring import ScoreCombiner

with open("my_config.json") as f:
    scorer = ScoreCombiner.from_dict(json.load(f))

score = snow.calculate_sledding_score(scorer=scorer)
```

### Option 2: Programmatic Modification

```python
from src.scoring import ScoreComponent, ScoreCombiner

# Create custom scorer
scorer = ScoreCombiner(
    name="my_sledding_score",
    components=[
        # Steeper slopes for advanced sledding
        ScoreComponent(
            name="slope_mean",
            transform="trapezoidal",
            transform_params={
                "sweet_range": (10, 25),  # Steeper!
                "ramp_range": (5, 35),
            },
            role="additive",
            weight=0.30,
        ),
        # ... more components
    ],
)

score = snow.calculate_sledding_score(scorer=scorer)
```

### Option 3: Modify Default

```python
from src.scoring.configs.sledding import create_default_sledding_scorer

scorer = create_default_sledding_scorer()

# Modify a component's parameters
for component in scorer.components:
    if component.name == "slope_mean":
        component.transform_params["sweet_range"] = (8, 20)

score = snow.calculate_sledding_score(scorer=scorer)
```

## Required Inputs

The scorer expects these inputs (computed automatically by `compute_derived_inputs`):

| Input | Source | Description |
|-------|--------|-------------|
| `slope_mean` | SlopeStatistics | Mean slope in degrees |
| `slope_p95` | SlopeStatistics | 95th percentile slope |
| `snow_depth` | SNODAS stats | `median_max_depth` in mm |
| `snow_coverage` | SNODAS stats | `mean_snow_day_ratio` (0-1) |
| `snow_consistency` | SNODAS stats | `interseason_cv` |
| `aspect_bonus` | Derived | cos(aspect) × strength × 0.05 |
| `runout_bonus` | Derived | 1.0 if slope_min < 5° |
| `terrain_consistency` | Derived | RMS of roughness + slope_std |

## Visualization

The example script (`detroit_snow_sledding.py`) generates visualizations showing:

1. **Raw statistics** - slope, snow depth, coverage, CVs
2. **Component scores** - each transformed to 0-1
3. **Penalties** - cliff detection, terrain consistency
4. **Final score** - combined result

## Tips for Customization

1. **Weights must sum to 1.0** - The system will raise an error if not

2. **Multiplicative for safety** - Use multiplicative role for anything that should disqualify an area entirely (cliffs, water, hazards)

3. **Additive for preferences** - Use additive role for quality factors that trade off against each other

4. **Test incrementally** - Use `get_component_scores()` to debug individual components

5. **Consider local conditions** - Adjust thresholds based on your region:
   - More snow? Increase depth thresholds
   - Steeper terrain? Adjust slope sweet spot
   - Different safety tolerance? Modify cliff threshold
