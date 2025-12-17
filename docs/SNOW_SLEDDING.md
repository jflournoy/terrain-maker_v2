# Snow Integration: Sledding Location Analysis

A complete example showing how to combine elevation data with SNODAS snow statistics to identify optimal sledding locations. This demonstrates the power of multi-layer terrain analysis with visual outputs at each pipeline stage.

## The Final Result

This pipeline produces a comprehensive sledding suitability score map:

![Final Sledding Suitability Score](images/05_final/sledding_score.png)

**Score Interpretation:**
- **0.8-1.3 (Bright Yellow)**: Excellent sledding locations (with synergy bonuses)
- **0.5-0.8 (Yellow-Green)**: Good sledding conditions
- **0.3-0.5 (Green-Cyan)**: Moderate - marginal conditions
- **0.0-0.3 (Dark Blue/Purple)**: Poor sledding potential

---

### What This Example Shows

✓ **Multi-Layer Analysis**: Combine terrain elevation with snow depth/coverage data
✓ **SNODAS Integration**: Process real National Snow and Ice Data Center snow grids
✓ **Pipeline Visualization**: Generate visual outputs at each analysis step
✓ **Sledding Score Calculation**: Identify optimal locations based on snow depth, coverage, and consistency
✓ **Mock Data Support**: Fast testing with realistic synthetic snow data
✓ **Step-by-Step Execution**: Run individual pipeline steps or the entire workflow

### The Pipeline

The snow integration example demonstrates a multi-stage analysis pipeline:

```
DEM → Snow Depth → Sledding Score → 3D Render (optional)
```

Each stage produces a visual artifact:
- **DEM Visualization**: Terrain elevation heatmap
- **Snow Depth Visualization**: Median maximum snow depth across the season
- **Sledding Score Visualization**: Combined suitability score (0 to ~1.3 with synergies)
- **3D Render**: Optional Blender visualization with snow overlay

### The Code

```python
from pathlib import Path
from src.terrain.io import load_dem_files
from src.terrain.core import Terrain
from src.snow.analysis import SnowAnalysis

# 1. Load elevation data
dem_dir = Path("data/dem/detroit")
dem, transform = load_dem_files(dem_dir, pattern="*.tif")

# 2. Create terrain
terrain = Terrain(dem, transform, dem_crs="EPSG:4326")

# 3. Process SNODAS snow data
snodas_dir = Path("data/snodas_data")
snow = SnowAnalysis(terrain=terrain, snodas_root_dir=snodas_dir)
stats, metadata, failed = snow.process_snow_data()

# 4. Calculate sledding suitability score
# This uses improved scoring with:
# - Trapezoid functions for sweet spots (snow 1-20", slope 1-20°)
# - Optimal ranges: snow 2-12", slope 6-12°
# - Deal breakers for unsuitable terrain (slope >35°, variability >5°)
# - Coverage with diminishing returns
# - Synergy bonuses for exceptional combinations
sledding_score = snow.calculate_sledding_score()

# The score is automatically added to terrain.data_layers
# Access it via: terrain.data_layers['sledding_score']['data']
```

### Running This Example

#### Quick Start with Mock Data

Test the pipeline quickly with synthetic data:

```bash
# Generate all visualizations with mock data
npm run py:example:snow-demo

# Or run individual steps
npm run py:example:snow-dem        # DEM visualization only
npm run py:example:snow-analysis   # Sledding score calculation
```

#### Using Real SNODAS Data

Process real snow data from the National Snow and Ice Data Center:

```bash
# Full pipeline with real SNODAS data
npm run py:example:snow-real

# Just the sledding score analysis
npm run py:example:snow-real-score
```

Or run the Python script directly:

```bash
# With mock data for testing
uv run examples/detroit_snow_sledding.py --mock-data --all-steps

# With real SNODAS data
uv run examples/detroit_snow_sledding.py --snodas-dir data/snodas_data --all-steps

# Run individual steps
uv run examples/detroit_snow_sledding.py --step dem
uv run examples/detroit_snow_sledding.py --step snow
uv run examples/detroit_snow_sledding.py --step score
```

### Pipeline Steps Explained

#### Step 1: DEM Visualization

Generates a terrain elevation heatmap using the 'terrain' colormap:

```python
def visualize_dem(dem: np.ndarray, output_path: Path):
    """Create DEM visualization with terrain colormap."""
    plt.figure(figsize=(12, 8))
    plt.imshow(dem, cmap="terrain", aspect="auto")
    plt.colorbar(label="Elevation (meters)")
    plt.title("Detroit Area - Digital Elevation Model")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
```

**Output**: `detroit_dem.png` - Elevation heatmap showing terrain features

#### Step 2: Snow Depth Visualization

Displays median maximum snow depth across the season using a blue-white gradient:

```python
def visualize_snow_depth(snow_depth: np.ndarray, output_path: Path):
    """Create snow depth visualization with blue-white colormap."""
    # Custom blue-white gradient for snow
    colors = ["#08519c", "#3182bd", "#6baed6", "#bdd7e7", "#eff3ff", "white"]
    cmap = LinearSegmentedColormap.from_list("snow", colors)

    plt.figure(figsize=(12, 8))
    plt.imshow(snow_depth, cmap=cmap, aspect="auto")
    plt.colorbar(label="Snow Depth (cm)")
    plt.title("Detroit Area - Median Maximum Snow Depth")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
```

**Output**: `detroit_snow_depth.png` - Snow depth heatmap from SNODAS data

**Data Source**: SNODAS (Snow Data Assimilation System) provides daily snow depth grids at 1km resolution. The visualization shows the median of the maximum daily snow depth for each pixel across the analysis period.

#### Step 3: Sledding Score Calculation

Combines multiple snow metrics to identify optimal sledding locations:

```python
def visualize_sledding_score(score: np.ndarray, output_path: Path):
    """Create sledding score visualization with RdYlGn colormap."""
    plt.figure(figsize=(12, 8))
    plt.imshow(score, cmap=plt.cm.RdYlGn, vmin=0, vmax=1, aspect="auto")
    plt.colorbar(label="Sledding Suitability Score")
    plt.title("Detroit Area - Sledding Location Score")

    # Add interpretation guide
    plt.text(0.02, 0.98, "Green = Excellent | Yellow = Moderate | Red = Poor",
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
```

**Output**: `detroit_sledding_score.png` - Suitability score heatmap (0 to ~1.3 with synergies)

**Improved Scoring Algorithm**:

The scoring uses trapezoid functions to identify sweet spots, deal breakers for unsuitable terrain, and synergy bonuses for exceptional combinations:

1. **Trapezoid Sweet Spots**: Snow 1-20" (optimal 2-12"), slope 1-20° (optimal 6-12°)
2. **Deal Breakers**: Hard zeros for slope >35°, variability >5°, coverage <0.5mo
3. **Multiplicative Base**: Snow × Slope × Coverage (poor factors can't be compensated)
4. **Synergy Bonuses**: +10% to +30% for exceptional combinations

The final score ranges from 0 (poor sledding) to ~1.3 (excellent with bonuses).

See [equation.md](images/04_score_components/equation.md) for the complete formula.

### Mock Data for Testing

The example includes realistic mock data generation for fast iteration:

```python
def create_mock_snow_data(shape: tuple) -> dict:
    """Create mock snow data with realistic distributions."""
    return {
        "median_max_depth": np.random.gamma(2, 30, shape),      # 0-300cm
        "mean_snow_day_ratio": np.random.beta(8, 2, shape),     # 0-1
        "interseason_cv": np.random.beta(2, 8, shape) * 0.5,    # 0-0.5
        "mean_intraseason_cv": np.random.beta(2, 8, shape) * 0.3 # 0-0.3
    }
```

Mock data uses statistical distributions that mimic real SNODAS patterns:
- **Gamma distribution** for snow depth (right-skewed, realistic accumulation)
- **Beta distribution** for ratios and coefficients (bounded 0-1)

This enables:
- ✓ Fast testing without downloading SNODAS data
- ✓ Reproducible test cases
- ✓ TDD-friendly development workflow

### Example Output

When you run the full pipeline with mock data:

```
======================================================================
Detroit Snow Sledding Analysis
======================================================================

[1/3] Generating DEM visualization...
      ✓ Saved: examples/detroit_dem.png

[2/3] Generating snow depth visualization...
      ✓ Mock snow data created (500x500)
      ✓ Saved: examples/detroit_snow_depth.png

[3/3] Calculating sledding scores...
      ✓ Sledding score calculated
      ✓ Score range: 0.00 - 1.00
      ✓ Mean score: 0.54
      ✓ Saved: examples/detroit_sledding_score.png

======================================================================
Pipeline Complete!
======================================================================

Generated outputs:
  ✓ examples/detroit_dem.png
  ✓ examples/detroit_snow_depth.png
  ✓ examples/detroit_sledding_score.png
```

### SNODAS Data Structure

The example expects SNODAS data organized by date:

```
data/snodas_data/
├── 2024-01-01/
│   └── us_ssmv11034tS__T0001TTNATS2024010105HP001.dat
├── 2024-01-02/
│   └── us_ssmv11034tS__T0001TTNATS2024010205HP001.dat
└── ...
```

**SNODAS Grid Details**:
- Resolution: 1 km × 1 km
- Coverage: Continental United States
- Temporal: Daily snapshots
- Variables: Snow depth, SWE, melt, and more

The `SnowAnalysis` class automatically:
- Discovers all daily grids in the directory
- Reprojects to match the terrain DEM coordinate system
- Calculates seasonal statistics (median, max, CV, coverage ratio)
- Caches results for fast subsequent runs

### Integration with Terrain Layers

The snow data integrates seamlessly with the terrain system via `add_data_layer()`:

```python
# Snow statistics are automatically reprojected to match terrain CRS
snow = SnowAnalysis(terrain=terrain, snodas_root_dir=snodas_dir)
stats, metadata, failed = snow.process_snow_data()

# Calculate score - this adds a new layer to terrain
sledding_score = snow.calculate_sledding_score()

# Access the score via data_layers
score_data = terrain.data_layers['sledding_score']['data']
score_crs = terrain.data_layers['sledding_score']['crs']
score_transform = terrain.data_layers['sledding_score']['transform']
```

This architecture enables:
- **Automatic reprojection** between different coordinate systems
- **Metadata preservation** (CRS, transform, units)
- **Multi-layer visualization** (combine elevation, snow, and score)
- **Modular analysis** (keep terrain and snow analysis separate)

### Customization

#### Understanding the Improved Scoring

The improved sledding score uses a fundamentally different approach than weighted sums:

```python
# The score uses trapezoid functions, deal breakers, and synergies
sledding_score = snow.calculate_sledding_score()

# Key scoring parameters (defined in src/terrain/scoring.py):
# - Snow range: 1-20 inches (optimal 2-12")
# - Slope range: 1-20 degrees (optimal 6-12°)
# - Deal breaker thresholds: slope >35°, variability >5°, coverage <0.5mo
# - Synergy bonuses: +10% to +30% for exceptional combinations
```

This multiplicative approach ensures that poor performance in any factor significantly impacts the final score, preventing compensation between unrelated factors.

#### Filter by Score Threshold

The pipeline automatically generates a filtered visualization showing only excellent locations (score > 0.7). This is saved as `sledding_score_excellent.png` in the `05_final/` output directory.

If you want to filter by a different threshold, you can customize it in code:

```python
# Get high-scoring areas (score > 0.8 for very high standards)
excellent_spots = sledding_score > 0.8

# Visualize only excellent locations with custom threshold
score_filtered = np.where(excellent_spots, sledding_score, 0)
# Then plot with same visualization logic...
```

Or adjust the pipeline directly:

```python
from examples.detroit_snow_sledding import save_sledding_score_filtered

# Generate with custom threshold (e.g., 0.75)
save_sledding_score_filtered(sledding_score, output_path="excellent_0.75.png", threshold=0.75)
```

#### Use Different Time Periods

Process SNODAS data from specific date ranges:

```python
# Filter SNODAS files by date
snow = SnowAnalysis(terrain=terrain, snodas_root_dir=snodas_dir)

# Process only December-February (peak sledding season)
stats, metadata, failed = snow.process_snow_data(
    start_date="2024-12-01",
    end_date="2025-02-28"
)
```

### Snow Analysis Pipeline: Detailed Breakdown

The sledding location analysis pipeline breaks down into distinct stages, each generating individual visualizations for detailed analysis and documentation:

#### Stage 1: Terrain Input

**Raw elevation data visualization:**

![DEM - Digital Elevation Model](images/01_raw/dem.png)

The pipeline begins with a Digital Elevation Model showing the terrain's elevation features. This is the foundation for all subsequent analysis - the terrain's slope and elevation characteristics influence sledding potential.

#### Stage 2: Snow Data Input

**Median maximum snow depth:**

![Snow Depth Distribution](images/01_raw/snow_depth.png)

SNODAS snow data is processed to show median maximum snow depth. Deeper snow generally provides better sledding conditions, but too-consistent depth can indicate areas that flood seasonally.

#### Stage 3: Slope Statistics Analysis

The terrain's slope characteristics are computed at full resolution using tiled analysis (to handle large DEMs efficiently). This breaks down into 8 detailed panels:

**Slope Mean** - Average slope angle in degrees:
```
Shows the gentle-to-moderate slopes ideal for sledding
- 5-15°: Excellent sledding terrain (not too steep, not too flat)
- 0-5°: Too flat - snow won't move
- >25°: Too steep - avalanche risk, hard to control
```

![Slope Mean - Average slope](images/02_slope_stats/mean.png)

**Slope Max** - Maximum slope within each region (cliff detection):
```
Identifies steep cliffs and drop-offs that reduce sledding suitability
- High values: Terrain has hidden cliffs (dangerous)
- Low values: Smooth, consistent terrain
```

![Slope Max - Cliff Detection](images/02_slope_stats/max.png)

**Slope Min** - Minimum slope (flat spot identification):
```
Finds the flattest areas which are good for run-out zones
- 0-5°: Ideal flat run-out areas to slow sleds
- Shows where sledders naturally decelerate
```

![Slope Min - Flat Spots](images/02_slope_stats/min.png)

**Slope Std Dev** - Slope variability (terrain roughness):
```
Measures how consistent the slope is
- Low values: Smooth, predictable descent
- High values: Choppy, variable terrain
```

![Slope Std Dev - Terrain Consistency](images/02_slope_stats/std.png)

**Slope P95** - 95th percentile slope (steep outliers):
```
Detects scattered steep areas without averaging them out
- Used to penalize regions with hidden cliffs
- More robust than slope_max to noise
```

![Slope P95 - 95th Percentile Slope](images/02_slope_stats/p95.png)

**Roughness** - Elevation variation (standard deviation of elevation within grid):
```
Identifies rough, boulder-strewn terrain vs smooth slopes
- High roughness: Rocky, dangerous terrain
- Low roughness: Smooth, sled-friendly slopes
```

![Roughness - Elevation Variability](images/02_slope_stats/roughness.png)

**Dominant Aspect** - Direction slopes face (compass direction):
```
Shows which way slopes face:
- Light colors: North-facing (shadier, retains more snow)
- Dark colors: South-facing (sunnier, more melt)
- North-facing aspects often have better sledding
```

![Dominant Aspect - Slope Direction](images/02_slope_stats/aspect.png)

**Aspect Strength** - Consistency of aspect direction:
```
Shows how uniform the slope direction is:
- High values: All slopes face the same direction
- Low values: Slopes face mixed directions
- Affects snow retention patterns
```

![Aspect Strength - Direction Consistency](images/02_slope_stats/aspect_strength.png)

#### Stage 4: Slope Penalties Analysis

Raw slope statistics are transformed into a sledding score with penalties for hazards:

**Base Slope Score** - Trapezoidal scoring function on slope mean:
```python
# Sweet range: 5-15° (ideal sledding slope)
# Ramp range: 3-25° (acceptable range)
# Outside ramp range: 0 points
slope_score = trapezoidal(slope_mean, sweet_range=(5, 15), ramp_range=(3, 25))
```

![Base Slope Score - Trapezoidal Function](images/03_slope_penalties/base_score.png)

**Cliff Penalty** - Dealbreaker penalty for steep areas:
```python
# If p95 slope > 25°, penalize heavily
# Applies smooth rolloff to avoid hard cutoffs
cliff_penalty = dealbreaker(slope_p95, threshold=25, falloff=10)
```

![Cliff Penalty - Steep Area Reduction](images/03_slope_penalties/cliff_penalty.png)

**Terrain Consistency Penalty** - Penalizes rough terrain:
```python
# Combined penalty from roughness + slope variability
# Extreme roughness or high std deviation reduces score
consistency_penalty = terrain_consistency(roughness, slope_std)
```

![Terrain Consistency - Roughness Penalty](images/03_slope_penalties/terrain_consistency.png)

**Combined Penalties** - Multiplication of all penalties:
```python
# Final slope score = base × cliff_penalty × consistency_penalty
combined_penalty = cliff_penalty × consistency_penalty
```

![Combined Penalties - Total Slope Reduction](images/03_slope_penalties/combined_penalty.png)

**Final Slope Score** - Base score after all penalties:
```python
final_slope_score = slope_score × combined_penalty
```

![Final Slope Score - Penalized Result](images/03_slope_penalties/final_score.png)

**Score Reduction Map** - Visual of penalty impact:
```python
# Shows how much score was lost to penalties
# Yellow = high penalty, Black = low penalty
score_reduction = base_score - final_slope_score
```

![Score Reduction - Penalty Impact](images/03_slope_penalties/score_reduction.png)

#### Stage 5: Improved Score Component Analysis

The improved sledding score uses trapezoid functions, deal breakers, and synergy bonuses. The detailed analysis breaks these into 11 individual panels showing each transformation:

**Raw Input Data** (for reference):

![Raw Snow Depth](images/04_score_components/raw_snow_depth_inches.png)
*Raw: Snow Depth in inches - Unscored SNODAS measurements converted to inches*

![Raw Slope Degrees](images/04_score_components/raw_slope_degrees.png)
*Raw: Slope in degrees - Unscored terrain slope from DEM analysis*

![Raw Coverage Months](images/04_score_components/raw_coverage_months.png)
*Raw: Snow coverage in months - Duration of snow coverage during season*

![Raw Slope Variability](images/04_score_components/raw_slope_variability.png)
*Raw: Slope variability (std dev) - Terrain roughness measure*

**Trapezoid Sweet Spot Scoring**:

![Snow Trapezoid Score](images/04_score_components/snow_trapezoid_score.png)
*Snow Trapezoid Score - Sweet spot 2-12" (ramps 1-20") scores highest*

![Slope Trapezoid Score](images/04_score_components/slope_trapezoid_score.png)
*Slope Trapezoid Score - Sweet spot 6-12° (ramps 1-20°) scores highest*

![Coverage Score](images/04_score_components/coverage_score.png)
*Coverage Score - Diminishing returns: 1 - exp(-months/2)*

**Deal Breakers & Validation**:

![Valid Terrain Mask](images/04_score_components/valid_terrain_mask.png)
*Valid Terrain Mask - Areas passing all deal breaker checks (slope ≤35°, variability ≤5°, coverage ≥0.5mo)*

**Synergy Bonuses**:

![Synergy Bonus](images/04_score_components/synergy_bonus.png)
*Synergy Bonus Multiplier - Hierarchical bonuses (+10% to +30%) for exceptional combinations*

**Score Combination**:

![Base Score](images/04_score_components/base_score.png)
*Base Score - Multiplicative: Snow × Slope × Coverage (before synergies)*

![Final Sledding Score](images/04_score_components/final_score.png)
*Final Sledding Score - Base × Synergy Bonus (range: 0 to ~1.3)*

**Improved Scoring Formula:**

The complete formula is generated and saved as markdown for documentation:

```
Snow Trapezoid:  1" ━━━ [2-12"] ━━━ 20"
Slope Trapezoid: 1° ━━━ [6-12°] ━━━ 20°
Coverage:        1 - exp(-months/2)

Deal Breakers:   Slope > 35° OR Variability > 5° OR Coverage < 0.5mo → Score = 0

Base Score:      Snow_Trapezoid × Slope_Trapezoid × Coverage_Score

Synergies:       Perfect Combo (+30%), Consistent (+15%),
                 Moderate Slope + Flat Runout (+20%), Low Var + Perfect Slope (+10%)

Final Score:     BASE × SYNERGY_BONUS  (range: 0 to ~1.5)
```

See [equation.md](images/04_score_components/equation.md) for the exact formula.

#### Stage 6: Final Result

The final sledding score combines all factors into a single suitability map (0 to ~1.3 scale with synergies). This is the primary output of the pipeline - a comprehensive assessment of sledding suitability across the entire region:

![Final Sledding Suitability Score](images/05_final/sledding_score.png)

This map shows where sledding conditions are best. The colors represent suitability scores from 0 (poor) to ~1.3 (excellent with synergy bonuses).

**Score Interpretation:**
- **0.8-1.3 (Bright Yellow)**: Excellent sledding locations (with synergy bonuses)
- **0.5-0.8 (Yellow-Green)**: Good sledding conditions
- **0.3-0.5 (Green-Cyan)**: Moderate - marginal conditions
- **0.0-0.3 (Dark Blue/Purple)**: Poor sledding potential

**Additional Analysis Visualizations:**

Understanding the score distribution helps identify optimal locations and assess regional variation:

![Sledding Score Distribution](images/05_final/sledding_score_histogram.png)
*Histogram showing the distribution of sledding scores across all pixels, with quartile markers (25th, 50th, 75th percentiles)*

![Sledding Score Percentiles](images/05_final/sledding_score_percentiles.png)
*Spatial distribution of score percentiles: shows which pixels fall into bottom 25% (red), 25-50% (orange), 50-75% (yellow), or top 25% (turquoise/blue) of scores*

![Excellent Sledding Locations](images/05_final/sledding_score_excellent.png)
*Filtered map highlighting only the excellent sledding locations (score > 0.7) - the most promising areas for sledding*

### Generating These Visualizations

All individual panels are generated automatically when you run the example:

```bash
# Generate all visualizations
uv run examples/detroit_snow_sledding.py --mock-data --all-steps

# Output structure:
# docs/images/
# ├── 01_raw/                    # Input data
# │   ├── dem.png
# │   └── snow_depth.png
# ├── 02_slope_stats/             # 8 slope analysis panels
# ├── 03_slope_penalties/         # 6 penalty analysis panels
# ├── 04_score_components/        # 11 improved scoring components + equation
# └── 05_final/                   # Final sledding score
```

Each panel can be embedded in documentation, presentations, or analysis reports. The individual format makes it easy to explain each step of the pipeline in detail.

---

### Why This Example Matters

**Traditional snow analysis workflow**:
1. Manually download SNODAS grids
2. Write custom reprojection code
3. Manually align snow data with terrain
4. Implement scoring algorithms
5. Handle coordinate system mismatches
6. Generate visualizations separately

**With Terrain Maker + SnowAnalysis**:
1. Point to SNODAS directory
2. Call `process_snow_data()`
3. Call `calculate_sledding_score()`
4. Visual outputs generated automatically

The example demonstrates:
- ✓ **Modular design**: Snow analysis is separate from core terrain functionality
- ✓ **Data layer system**: Automatic reprojection and metadata preservation
- ✓ **Multi-stage pipeline**: Visual feedback at each step
- ✓ **Testing-friendly**: Mock data enables fast TDD iteration
- ✓ **Production-ready**: Real SNODAS integration with caching

---

**Want to try it yourself?** See [Quick Reference](QUICK_REFERENCE.md) for API documentation and [API Reference](API_REFERENCE.md) for detailed function signatures.
