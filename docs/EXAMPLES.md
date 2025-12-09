# Examples Gallery

Real-world examples demonstrating the power and simplicity of Terrain Maker.

## Detroit Elevation Visualization

A complete example showing how to create stunning 3D terrain visualizations from real SRTM elevation data with just a few lines of Python.

### The Result

![Detroit Elevation Visualization](images/detroit_elevation_south.png)

*Professional-quality 3D terrain visualization of Detroit metro area - rendered with Blender from real SRTM elevation data with water body detection enabled*

### Multiple Views with Intelligent Camera Positioning

The example demonstrates the power of `position_camera_relative()` by generating professional renders from different camera angles - each with optimal framing for its view direction:

| North View | East View |
|:---:|:---:|
| *(looking south)* | *(looking west)* |
| ![Detroit North View](images/detroit_elevation_north.png) | ![Detroit East View](images/detroit_elevation_east.png) |

| West View | Overhead View |
|:---:|:---:|
| *(looking east)* | *(90° down, zero rotation)* |
| ![Detroit West View](images/detroit_elevation_west.png) | ![Detroit Above View](images/detroit_elevation_above.png) |

**Generate these views yourself:**
```bash
npm run py:example:detroit-north     # or any of: south, east, west, above
```

Each view is automatically framed with intelligent target offset adjustments. No manual coordinate calculations needed!

### What This Example Shows

✓ **Loading Real Geographic Data**: Automatically loads and merges SRTM HGT tiles
✓ **Intelligent Mesh Optimization**: Configure mesh density by target vertex count, not magic numbers
✓ **Coordinate Transformation**: Automatic reprojection from WGS84 to UTM coordinates
✓ **Water Body Detection**: Automatic identification of water bodies using slope-based analysis
✓ **Intuitive Camera Control**: Position cameras using cardinal directions (north, south, above, etc)
✓ **Beautiful Visualization**: Professional Blender rendering with color mapping and water shader

### The Code

```python
# 1. Load elevation data
dem_data, transform = load_dem_files(SRTM_TILES_DIR, pattern='*.hgt')
terrain = Terrain(dem_data, transform)

# 2. Define output image dimensions and compute target mesh vertices
# Target vertices = WIDTH × HEIGHT × 2
# This ensures ~2 vertices per output pixel for optimal mesh density
WIDTH = 960
HEIGHT = 720
target_vertices = WIDTH * HEIGHT * 2  # ~1.4 million vertices
terrain.configure_for_target_vertices(target_vertices, order=4)

# 3. Apply geographic transforms
terrain.transforms.append(reproject_raster(
    src_crs='EPSG:4326',      # WGS84 (from SRTM data)
    dst_crs='EPSG:32617',     # UTM Zone 17N (Detroit area)
    num_threads=4
))
terrain.transforms.append(flip_raster(axis='horizontal'))
terrain.transforms.append(scale_elevation(scale_factor=0.0001))
terrain.apply_transforms()

# 4. Set up beautiful Mako color mapping
terrain.set_color_mapping(
    lambda dem: elevation_colormap(dem, cmap_name='mako'),
    source_layers=['dem']
)

# 5. Detect water bodies on unscaled DEM (critical!)
from src.terrain.water import identify_water_by_slope
transformed_dem = terrain.data_layers['dem']['transformed_data']
unscaled_dem = transformed_dem / 0.0001  # Undo elevation scaling
water_mask = identify_water_by_slope(unscaled_dem, slope_threshold=0.01, fill_holes=True)

# 6. Create mesh with water detection and render
mesh = terrain.create_mesh(
    scale_factor=100.0,
    height_scale=4.0,
    center_model=True,           # Center mesh at origin
    boundary_extension=True,     # Extend boundaries for clean edges
    water_mask=water_mask        # Use pre-computed water mask
)
camera = position_camera_relative(mesh, direction='south', distance=1.5)
render_scene_to_file(output_path="detroit.png", width=WIDTH, height=HEIGHT)
```

### Key Features in Action

#### Intelligent Downsampling
Instead of guessing downsampling factors, compute target vertices based on your output image dimensions:

```python
# Match mesh density to output resolution: ~2 vertices per pixel
WIDTH = 960
HEIGHT = 720
target_vertices = WIDTH * HEIGHT * 2  # 1,382,400 vertices

# Terrain Maker automatically calculates the optimal downsampling factor
terrain.configure_for_target_vertices(target_vertices)

# This ensures your mesh has perfect detail for the output size
# (no wasted vertices, no under-sampling)
```

**The Math:**
- Output image: 960 × 720 = 691,200 pixels
- Target vertices: 691,200 × 2 = 1,382,400
- This gives approximately 2 vertices per output pixel, providing optimal detail without over-sampling
- Adjust the multiplier (×2, ×3, etc.) based on your quality needs and performance budget

#### Cardinal Direction Camera Positioning
No more confusing coordinate calculations. Position your camera intuitively with intelligent view-specific targeting:

```python
camera = position_camera_relative(
    mesh_obj,
    direction='south',      # Can be: north, south, east, west, northeast, etc.
    distance=1.5,           # Multiplier of mesh diagonal
    elevation=0.5,          # Height above center
    look_at=(0, -1.5, 0),   # View-specific target offset (auto-calculated)
)
```

**What makes this powerful:**
- Each cardinal direction has intelligent target offset adjustments
- North: targets (0, 2, 0) - offsets north for perfect framing
- South: targets (0, -1.5, 0) - offsets south for perfect framing
- East/West: adjust X axis similarly for optimal perspective
- The function automatically calculates rotation, distance, and elevation
- Overhead view uses zero rotation to eliminate gimbal lock artifacts

This eliminates trial-and-error camera positioning entirely!

#### Geographic Coordinate Handling
Automatic reprojection and proper coordinate system handling:

```python
terrain.transforms.append(reproject_raster(
    src_crs='EPSG:4326',      # WGS84 (from SRTM data)
    dst_crs='EPSG:32617',     # UTM Zone 17N (Detroit area)
    num_threads=4
))
```

#### Water Body Detection & Blue Coloring
Automatic water body identification using slope-based analysis with direct blue coloring:

```python
# IMPORTANT: Detect water on the UNSCALED DEM (before elevation scaling)
# This ensures slope calculations are meaningful (scaled elevation produces tiny slopes)

# 1. Apply all transforms including elevation scaling
terrain.transforms.append(reproject_raster('EPSG:4326', 'EPSG:32617'))
terrain.transforms.append(flip_raster(axis='horizontal'))
terrain.transforms.append(scale_elevation(scale_factor=0.0001))
terrain.apply_transforms()

# 2. Get transformed DEM and unscale it
transformed_dem = terrain.data_layers['dem']['transformed_data']
unscaled_dem = transformed_dem / 0.0001  # Undo elevation scaling

# 3. Detect water on unscaled DEM with very low threshold
from src.terrain.water import identify_water_by_slope
water_mask = identify_water_by_slope(
    unscaled_dem,
    slope_threshold=0.01,  # Extremely low threshold for nearly-flat water
    fill_holes=True
)

# 4. Create mesh with pre-computed water mask
mesh = terrain.create_mesh(
    scale_factor=100.0,
    height_scale=4.0,
    water_mask=water_mask  # Use pre-computed water mask
)
```

**How it works:**

**Detection Phase:**
- Uses Horn's method to compute terrain slope magnitude from elevation data
- Identifies pixels with slope below threshold as potential water bodies
- Applies morphological operations (closing: dilation then erosion) to smooth water boundaries and fill gaps
- Water is nearly flat (slope ~0), while terrain has measurable slopes

**Coloring Phase:**
- Directly colors detected water pixels blue (RGB: 26, 102, 204)
- Land pixels retain their elevation-based Mako colormap colors
- Results in clear visual distinction between water and terrain
- Water is colored during mesh creation, no shader configuration needed

**Critical Implementation Detail - Why Use Unscaled DEM?**

When elevation data is scaled (e.g., by 0.0001) for visualization:
- Original DEM: elevations 50-1400 meters → slopes ~0-100 magnitude
- Scaled DEM: elevations 0.005-0.14 meters → slopes ~0-0.027 magnitude

Using a threshold of 0.1 on scaled data would catch 90% of terrain as "water" because most slopes are below 0.027. **Always detect water on unscaled DEM before applying elevation scaling.**

**Threshold Selection:**
- For mostly flat water (lakes, reservoirs): `0.01 - 0.05` (very sensitive)
- For mixed terrain with some water: `0.1 - 0.2` (moderate)
- For all flat-ish areas: `0.5+` (lenient)
- **Default for Detroit**: `0.01` produces realistic 13-15% water coverage

**For More Details:**
See [Water Body Detection in API Reference](API_REFERENCE.md#identify_water_by_slope) for complete documentation including examples, threshold guidelines, and best practices.

### Running This Example

#### Basic Render (South View - Default)
```bash
python examples/detroit_elevation_real.py
```

This will:
1. Load SRTM elevation tiles from `data/dem/detroit/` (110 tiles)
2. Compute target vertices from output image dimensions: WIDTH × HEIGHT × 2 = 960 × 720 × 2 = 1,382,400
3. Process the data with intelligent downsampling to match target vertex count
4. Generate a ~1.4 million vertex Blender mesh (exact count varies by interpolation)
5. Detect water bodies on the unscaled DEM and color them blue
6. Render a publication-quality PNG (960×720) with elevation colors + water rendering
7. Save the Blender file for further editing

#### Generate Multiple Views
The example supports command-line arguments to easily create renders from different camera angles:

```bash
# Quick commands for each cardinal direction
npm run py:example:detroit-north    # North view
npm run py:example:detroit-south    # South view
npm run py:example:detroit-east     # East view
npm run py:example:detroit-west     # West view
npm run py:example:detroit-above    # Overhead bird's-eye view
```

Each renders the same terrain from a different perspective with intelligent view-specific framing. Perfect for creating comparison sets or presentations!

### Example Output

When you run the example, you'll see output like this:

```
======================================================================
Detroit Real Elevation Visualization
======================================================================
✓ Blender scene cleared
[1/6] Loading SRTM tiles...
Opening DEM files: 100%|██████████| 110/110 [00:00<00:00, 1530.60it/s, opened=110]

[2/6] Initializing Terrain object...
      Terrain initialized
      DEM shape: (36001, 39601)

[3/6] Applying transforms...
      Original DEM shape: (36001, 39601)
      Configured for 1,382,400 target vertices
      Calculated zoom_factor: 0.031139
      Downsampled DEM shape: (1326, 1137)
      Actual vertices: 1,507,662
      Transforms applied successfully

[4/6] Setting up color mapping...
      Color mapping configured (Mako colormap)

[5/6] Creating Blender mesh...
      ✓ Mesh created successfully!
      Vertices: 1370951
      Polygons: 1368731

[6/6] Setting up camera and rendering to PNG...
      Camera: South-facing cardinal view
      Direction: south, distance: 1.5x, elevation: 0.5x
      Type: Orthographic
      Samples: 32
      Rendering...
      ✓ Rendered successfully!
      File: detroit_elevation_real.png
      Size: 2.0 MB

======================================================================
Detroit Real Elevation Visualization Complete!
======================================================================

Summary:
  ✓ Loaded and merged all SRTM tiles (full coverage)
  ✓ Configured downsampling to target vertex count intelligently
  ✓ Applied geographic coordinate reprojection (WGS84 → UTM)
  ✓ Created Terrain object with real elevation data
  ✓ Applied transforms (reproject + flip + scale)
  ✓ Configured beautiful Mako elevation-based color mapping
  ✓ Detected and applied water bodies (slope-based identification)
  ✓ Generated Blender mesh with 1370951 vertices
  ✓ Rendered to PNG: /path/to/detroit_elevation_real.png

That's it! Professional terrain visualization with water detection in just a few lines of Python!
```

The entire process takes about 30-40 seconds on modern hardware. You'll see progress bars for DEM loading and processing, with detailed logging of each transformation step.

### Output Files

- `examples/detroit_elevation_south.png` - Main example render (south view, 960×720, 2.1 MB)
- `examples/detroit_elevation_{north,east,west,above}.png` - Alternative views
- `examples/detroit_elevation_{view}.blend` - Blender files for further editing

### Why Terrain Maker Makes This Easy

Traditional terrain visualization typically requires:
- Manually guessing optimal mesh density (under-sample → blurry, over-sample → slow)
- Complex coordinate reprojection setup
- Blender scripting knowledge
- Camera positioning through trial-and-error
- Manual water body identification and coloring

With Terrain Maker, you get:
- **Smart mesh density calculation**: `target_vertices = WIDTH × HEIGHT × 2` ensures perfect detail for your output resolution
- **Automatic mesh optimization** - calculates the exact downsampling needed
- **Built-in geographic transforms** with sensible defaults
- **Water body detection** using slope-based analysis on unscaled DEM
- **Automatic water coloring** - flat areas colored blue during mesh creation
- **Intuitive cardinal direction camera positioning** (north, south, above, etc.)
- **Professional Blender integration** out of the box
- **Color mapping from elevation data** in one line

### Customization

#### Command-Line Camera Control
The example script accepts arguments to customize rendering without code changes:

```bash
# Change view direction
python examples/detroit_elevation_real.py --view east

# Adjust camera distance (closer/farther)
python examples/detroit_elevation_real.py --view north --distance 0.15

# Change elevation (higher/lower viewpoint)
python examples/detroit_elevation_real.py --view above --elevation 0.8

# Custom output filename
python examples/detroit_elevation_real.py --view west --output my_render.png

# Switch to orthographic projection
python examples/detroit_elevation_real.py --view north --camera-type ORTHO
```

#### In-Code Customization

```python
# Change the colormap
elevation_colormap(dem, cmap_name='viridis')    # Other options: turbo, plasma, etc.

# Adjust camera angle
position_camera_relative(mesh, direction='east', elevation=1.0)

# Configure rendering
setup_render_settings(samples=4096, use_denoising=True)

# Create a presentation set of all views
for view in ['north', 'south', 'east', 'west', 'above']:
    position_camera_relative(mesh, direction=view)
    render_scene_to_file(output_path=f"detroit_{view}.png")
```

#### Generating Multiple Comparison Views

Create a full set of renders for comparison or publication:

```bash
# Generate all cardinal views at once
for view in north south east west above; do
    npm run py:example:detroit-$view
done
```

This creates 5 renders showing the terrain from different perspectives, with consistent framing and optimal camera positioning for each angle.

### Regenerating Example Images

When you update the example code or want to refresh the documentation images with the latest rendering, use these npm commands:

```bash
# Generate the main example image (south view - default)
npm run py:example:detroit-elevation

# Generate specific views
npm run py:example:detroit-south      # South view (recommended for main image)
npm run py:example:detroit-north      # North view
npm run py:example:detroit-east       # East view
npm run py:example:detroit-west       # West view
npm run py:example:detroit-above      # Overhead view

# Generate all views at once
for view in south north east west above; do
  npm run py:example:detroit-$view
done
```

**Output locations:**
- Main image: `examples/detroit_elevation_south.png` (recommended for documentation)
- View-specific: `examples/detroit_elevation_{north,south,east,west,above}.png`
- Blender files: `examples/detroit_elevation_{view}.blend`

**Each render takes 30-40 seconds on modern hardware.** The images are then automatically included in the documentation gallery when you run:

```bash
npm run docs:build
```

### Accelerating Renders with Caching

The example supports intelligent caching to dramatically speed up iterative development. Instead of reprocessing elevation data and regenerating meshes for every render, the system caches:

- **DEM Cache**: Loaded and merged elevation tiles (.npz format with hash validation)
- **Mesh Cache**: Generated Blender mesh files (.blend with parameter-based hashing)

#### How It Works

```
First Run (Full Pipeline):
├─ Load SRTM tiles → .dem_cache/dem_[hash].npz
├─ Merge & transform
├─ Create mesh → .mesh_cache/mesh_[hash].blend
└─ Render to PNG

Second Run (Same DEM):
├─ Load from .dem_cache/ (skip 10-15 seconds of file merging)
├─ Create mesh → .mesh_cache/mesh_[hash].blend
└─ Render to PNG

Subsequent Renders (Same view):
├─ Load from .dem_cache/
├─ Load from .mesh_cache/ (skip 20-30 seconds of mesh generation!)
└─ Render only to PNG
```

#### Using Caching

Enable caching by adding the `--cache` flag:

```bash
# First run: caches DEM and mesh
npm run detroit-build        # South view with caching
npm run detroit-build-north  # North view with caching
npm run detroit-build-east   # East view with caching
npm run detroit-build-west   # West view with caching
npm run detroit-build-above  # Overhead view with caching

# Generate all views with caching (faster on subsequent runs)
npm run detroit-build:all
```

Or run the Python script directly with caching:

```bash
# Enable caching for full pipeline
uv run examples/detroit_elevation_real.py --cache

# With specific view
uv run examples/detroit_elevation_real.py --cache --view north

# Generate multiple views with caching
for view in south north east west above; do
  uv run examples/detroit_elevation_real.py --cache --view $view
done
```

#### Cache Management

```bash
# Clear all cached DEM and mesh files
npm run detroit-cache-clear
uv run examples/detroit_elevation_real.py --cache --clear-cache

# View cache statistics
npm run detroit-cache-stats
```

#### Cache File Locations

```
.dem_cache/           # DEM files (.npz format)
  dem_[hash].npz      # Merged elevation data
  dem_[hash]_meta.json  # Metadata and statistics

.mesh_cache/          # Mesh files (.blend format)
  mesh_[hash].blend   # Blender scene with terrain mesh
  [hash]_meta.json    # Mesh parameters and hash
```

#### Cache Invalidation

The cache automatically invalidates when:

- **DEM files change** (file modification time or count changes)
- **Mesh parameters change** (scale_factor, height_scale, center_model, boundary_extension, water_mask)
- **Manual clear** via `--clear-cache` flag

This ensures you always have current data without manual cache management.

#### Performance Impact

**Time savings by caching DEM:**
- First render: Full merge (10-15 seconds) + mesh generation (20-30 seconds) + render (5-10 seconds) = ~35-55 seconds
- Cached DEM: Skip merge + mesh generation (20-30 seconds) + render (5-10 seconds) = ~25-40 seconds
- **Savings: 10-15 seconds per render**

**Time savings by caching mesh:**
- After first full pipeline, subsequent renders skip mesh generation entirely
- Render-only pass: Load cached mesh + render (5-10 seconds) = ~5-10 seconds
- **Savings: 20-30 seconds per render**

---

## Snow Integration: Sledding Location Analysis

A complete example showing how to combine elevation data with SNODAS snow statistics to identify optimal sledding locations. This demonstrates the power of multi-layer terrain analysis with visual outputs at each pipeline stage.

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
- **Sledding Score Visualization**: Combined suitability score (0-1 scale)
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
# This combines:
# - Snow depth (deeper is better)
# - Coverage (consistent coverage is better)
# - Consistency (low variability is better)
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

**Output**: `detroit_sledding_score.png` - Suitability score heatmap (0-1 scale)

**Scoring Algorithm**:
- **Snow Depth Component** (40% weight): Normalized median max depth (deeper is better)
- **Coverage Component** (30% weight): Snow day ratio (more consistent coverage is better)
- **Consistency Component** (30% weight): 1 - CV (lower variability is better)

The final score ranges from 0 (poor sledding) to 1 (excellent sledding).

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

#### Adjust Sledding Score Weights

Modify the weighting of different snow characteristics:

```python
# Default weights
sledding_score = snow.calculate_sledding_score()

# Custom weights (must sum to 1.0)
sledding_score = snow.calculate_sledding_score(
    depth_weight=0.5,        # Prioritize depth
    coverage_weight=0.3,     # Medium importance for coverage
    consistency_weight=0.2   # Less emphasis on consistency
)
```

#### Filter by Score Threshold

Identify only the best locations:

```python
# Get high-scoring areas (score > 0.7)
excellent_spots = sledding_score > 0.7

# Visualize only excellent locations
visualize_sledding_score(
    np.where(excellent_spots, sledding_score, 0),
    output_path="excellent_sledding.png"
)
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
