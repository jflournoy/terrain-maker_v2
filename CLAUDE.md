# CLAUDE.md - Project AI Guidelines

## State Tracking

**Use two complementary systems for tracking work:**

1. **TodoWrite tool** (primary) - Use for task management visible to the user
   - Break work into discrete, actionable items
   - Mark tasks in_progress/completed as you work
   - Good for: task lists, progress tracking, user visibility

2. **`.claude-current-status` file** (supplementary) - Higher-resolution notes
   - Timestamps, context, decisions, file references
   - Details that don't fit in todo items
   - Session continuity across conversations
   - Good for: debugging context, decision rationale, file locations

**Workflow:** Start tasks with TodoWrite, always add detailed notes to `.claude-current-status`. When you add notes, re-assess and clean up old notes.


## Development Method: TDD

**RECOMMENDED: Use Test-Driven Development for new features**

TDD helps Claude produce more focused, correct code by clarifying requirements upfront and reducing wildly wrong approaches.

### Benefits of TDD with Claude

- **Without TDD**: Claude may over-engineer or miss requirements
- **With TDD**: Claude writes targeted code that meets specific criteria

### TDD Workflow

1. 🔴 **RED**: Write a failing test to define requirements
2. 🟢 **GREEN**: Write minimal code to pass the test
3. 🔄 **REFACTOR**: Improve code with test safety net
4. ✓ **COMMIT**: Ship working, tested code

### The TDD Command

```bash
/tdd start "your feature"  # Guides through the TDD cycle
```

Consider TDD especially for complex features or when requirements are unclear.

## Critical Instructions

**ALWAYS use `date` command for dates** - Never assume or guess dates. Always run `date "+%Y-%m-%d"` when you need the current date for documentation, commits, or any other purpose.

**NEVER operate on full-resolution data for expensive operations** - Operations like distance transforms, morphological operations, or complex array manipulations WILL OOM on large DEMs. Always understand which data is at full resolution vs downsampled resolution.

### Memory Management & Data Resolution

**Critical Rule: Understand your data resolution at every step**

The terrain processing pipeline uses multiple resolutions:
1. **Original DEM**: Full resolution from source files (potentially 10K×10K+ pixels)
2. **Flow computation**: Downsampled to target_vertices (typically 1-2K×1-2K pixels)
3. **Mesh vertices**: Further downsampled for 3D rendering (controlled by vertices-per-pixel)

**Memory-intensive operations to avoid on full-resolution data:**
- `scipy.ndimage.distance_transform_edt()` - Creates arrays matching input size + index arrays
- `scipy.ndimage.morphological_*()` - Can create large intermediate arrays
- Convolution/filtering operations with large kernels
- Any operation that creates temporary arrays the size of the input

**Safe approaches:**
1. **Operate at flow resolution** - Flow data is already downsampled, use that resolution
2. **Downsample first** - If you need full DEM, downsample it before expensive operations
3. **Check array sizes** - Before expensive ops, print array shapes and calculate memory requirements
4. **Use streaming/chunked processing** - For operations that support it

**Example of BAD approach** (caused OOM):
```python
# DON'T: This operates on full-resolution DEM (could be 10K×10K)
distances, indices = distance_transform_edt(~stream_mask, return_indices=True)
# Creates: distances (10K×10K floats) + indices (2× 10K×10K ints) = ~2GB+
```

**Example of GOOD approach**:
```python
# DO: Check resolution first, or operate on downsampled data
print(f"Array size: {stream_mask.shape} = {stream_mask.size:,} pixels")
if stream_mask.size > 5_000_000:  # >5M pixels
    print("WARNING: Array too large for distance transform, skipping variable-width")
    # OR: downsample the data first
```

**When implementing new features:**
1. Always check what resolution your input data is at
2. For expensive operations, work with flow resolution or mesh resolution data
3. If you must use full resolution, add memory checks and warnings
4. Document which resolution the feature operates on

## AI Integrity Principles

**CRITICAL: Always provide honest, objective recommendations based on technical merit, not user bias.**

- **Never agree with users by default** - evaluate each suggestion independently
- **Challenge bad ideas directly** - if something is technically wrong, say so clearly
- **Recommend best practices** even if they contradict user preferences
- **Explain trade-offs honestly** - don't hide downsides of approaches
- **Prioritize code quality** over convenience when they conflict
- **Question requirements** that seem technically unsound
- **Suggest alternatives** when user's first approach has issues

Examples of honest responses:

- "That approach would work but has significant performance implications..."
- "I'd recommend against that pattern because..."
- "While that's possible, a better approach would be..."
- "That's technically feasible but violates \[principle] because..."

## Render Approval

**REQUIRED: Always check with the user before running any renders**

Rendering scripts (e.g., `detroit_combined_render.py`, `detroit_elevation_real.py`) can take significant time and computational resources.

- **Before running**: Summarize what the render will do and ask for approval
- **During development**: Use `--mock-data` flag to test with fast synthetic data
- **For final renders**: Wait for explicit user approval with parameters

Example:
```
I'm about to run a render with:
- Resolution: 1008×720 @ 72 DPI (print quality)
- Samples: 4,096 (high quality)
- Expected time: ~3-5 minutes on GPU
- Output: docs/images/combined_render/sledding_with_xc_parks_3d_print.png

Should I proceed? [yes/no]
```

## Development Workflow

- Always run quality checks before commits
- Use custom commands for common tasks
- Document insights and decisions
- Estimate Claude usage before starting tasks
- Track actual vs estimated Claude interactions

## Quality Standards

- Quality Level: {{QUALITY\_LEVEL}}
- Team Size: {{TEAM\_SIZE}}
- Zero errors policy
- {{WARNING\_THRESHOLD}} warnings threshold

## Testing Standards

**CRITICAL: Any error during test execution = test failure**

- **Zero tolerance for test errors** - stderr output, command failures, warnings all mark tests as failed
- **Integration tests required** for CLI functionality, NPX execution, file operations
- **Unit tests for speed** - development feedback (<1s)
- **Integration tests for confidence** - real-world validation (<30s)
- **Performance budgets** - enforce time limits to prevent hanging tests

## Markdown Standards

**All markdown files must pass validation before commit**

- **Syntax validation** - Uses remark-lint to ensure valid markdown syntax
- **Consistent formatting** - Enforces consistent list markers, emphasis, and code blocks
- **Link validation** - Checks that internal links point to existing files
- **Auto-fix available** - Run `npm run markdown:fix` to auto-correct formatting issues

### Markdown Quality Checks

- `npm run markdown:lint` - Validate all markdown files
- `npm run markdown:fix` - Auto-fix formatting issues
- Included in `hygiene:quick` and `commit:check` scripts
- CI validates markdown on every push/PR

### Markdown Style Guidelines

- Use `-` for unordered lists
- Use `*` for emphasis, `**` for strong emphasis
- Use fenced code blocks with language tags
- Use `.` for ordered list markers
- Ensure all internal links are valid

## Commands

- `/hygiene` - Project health check
- `/todo` - Task management
- `/commit` - Quality-checked commits
- `/design` - Feature planning
- `/estimate` - Claude usage cost estimation
- `/next` - AI-recommended priorities
- `/learn` - Capture insights
- `/docs` - Update documentation

## Documentation

**This project uses Sphinx for API documentation with `uv` for Python package management.**

### Building Sphinx Docs

```bash
# Install documentation dependencies (first time only)
uv sync --extra docs

# Build documentation
npm run docs:build

# Clean build (removes previous build)
npm run docs:build:clean

# Serve documentation locally
npm run docs:serve
# Then open http://localhost:8080

# Check for broken links
npm run docs:check
```

### Documentation Structure

- `docs/source/` - Sphinx RST source files
- `docs/source/api/` - API reference (19 modules, 100% coverage)
- `docs/source/examples/` - Usage examples
- `docs/source/guides/` - User guides
- `docs/build/html/` - Generated HTML (not in git)

### Adding New API Documentation

When adding new modules, create RST files in `docs/source/api/` and add to `docs/source/index.rst`.

### Documentation Images

Example documentation includes visualization images generated from example scripts. These images are auto-generated using **real data** for production-quality documentation.

```bash
# Generate all documentation images (uses REAL data, takes 2-5 minutes)
npm run docs:images

# Full documentation build (images + Sphinx build)
npm run docs:build:full

# Manual image generation (if needed)
bash scripts/generate-docs-images.sh
```

**Note:** Image generation uses real DEM and SNODAS data, which takes several minutes. Ensure data files are present in `data/` directory before running.

**Image locations:**
- `docs/images/01_raw/` - Raw input data (DEM, snow depth)
- `docs/images/02_slope_stats/` - Slope analysis visualizations
- `docs/images/03_slope_penalties/` - Terrain penalty visualizations
- `docs/images/04_score_components/` - Score component breakdowns
- `docs/images/05_final/` - Final score visualizations

**Adding new documentation images:**
1. Update example script to output to `docs/images/` subdirectory
2. Add script invocation to `scripts/generate-docs-images.sh`
3. Reference images in markdown with relative paths: `../../images/subdir/image.png`

## Scripts Using Blender's Python API

**Import bpy directly** - When a script uses terrain maker's Blender rendering capabilities, import bpy at the module level, just like any other dependency.

### Installing bpy (Blender Python Module)

To run rendering scripts with regular Python (instead of inside Blender), install the `bpy` package:

```bash
# Install with uv (recommended)
uv pip install bpy

# Or install optional blender dependencies
uv sync --extra blender

# Verify installation
uv run python -c "import bpy; print(f'Blender {bpy.app.version_string}')"
```

**Requirements:**
- Python 3.11+ (matches Blender's Python version)
- ~400MB disk space (bpy includes Blender's core)
- Linux, macOS, or Windows

### Guidelines for Scripts with Blender Rendering

When creating scripts that use Blender's Python API via terrain maker:

1. **Import bpy directly at the top** - No try/except wrappers
   ```python
   import bpy  # Available when running in environment with Blender Python installed
   ```

2. **Document the requirement clearly** - State that bpy/Blender is needed
   ```python
   """
   Script Name.

   Renders terrain using Blender's Python API.

   Requirements:
   - Blender Python API available (bpy)
   - Pre-computed data files

   Usage:
       python examples/script.py

   With arguments:
       python examples/script.py --option value
   """
   ```

3. **Run as regular Python** - These are standard Python scripts, not special "Blender scripts"
   ```bash
   # Regular Python execution
   python examples/detroit_dual_render.py

   # With arguments
   python examples/detroit_dual_render.py --mock-data --output-dir ./renders
   ```

4. **Use terrain maker's rendering library** - All Blender operations are handled by terrain maker, not direct bpy calls
   - Use `Terrain` class and its rendering methods
   - Leverage the library's Blender integration
   - Focus on data pipeline, not low-level Blender API

### Example Pattern

See `examples/detroit_dual_render.py` for the proper pattern:
- Direct `import bpy` at module level (no try/except)
- Clear docstring about Blender requirement
- Run as regular Python script (`python examples/...`)
- Uses terrain maker's Terrain class for rendering
- Uses standard argparse for command-line arguments

## Visualization Standards

**Prefer perceptually uniform colormaps** - For all visualizations, prioritize colormaps that are perceptually uniform and colorblind-friendly.

### Recommended Colormaps

**Custom terrain-maker colormaps:**
- **Michigan elevation**: `michigan` - Perceptually uniform Michigan landscape (Great Lakes blue → forest green → upland meadow → sand dunes)
- **Boreal-Mako**: `boreal_mako` - Perceptually uniform score colormap (boreal green → mako blue → pale mint) with edge effect for score visualizations

**Terrain/Elevation (perceptually uniform):**
- **michigan**: Custom Michigan natural landscape colormap (preferred for Detroit/Michigan examples)
- **turbo**: Google's modern rainbow replacement with better perceptual properties
- **cividis**: Optimized for colorblind accessibility
- **gist_earth**, **terrain**: Traditional earth-tones (less perceptually uniform but intuitive)

**Score maps (viridis family - perceptually uniform):**
- **viridis**, **plasma**, **inferno**, **magma**, **cividis**

**Snow/Ice data:**
- **boreal_mako**: Custom sledding/snow score colormap (boreal green → blue → mint with edge effect)
- **cool**, **winter**, **ice** (snow-like colors)

### Colormap Selection Criteria

- **Perceptual uniformity**: Equal steps in data = equal perceptual steps in color
- **Colorblind-friendly**: Accessible to users with color vision deficiencies
- **Print-friendly**: Converts well to grayscale
- **Sequential data**: Use single-hue progressions (viridis, plasma) or custom michigan
- **Diverging data**: Use center-neutral progressions (coolwarm, RdBu)

### Trade-offs

- **For scientific publication**: Use viridis family (plasma, viridis, cividis)
- **For presentations/demos**: Use terrain-specific (michigan, gist_earth, turbo) for immediate visual recognition
- **For dual colormaps**: Combine terrain-like base (michigan, gist_earth) with viridis family overlay (plasma)

Avoid rainbow colormaps (jet, hsv) due to perceptual non-uniformity.

## Example Development Guidelines

**ALWAYS use library functions where possible** - When building examples, prioritize using terrain maker's public APIs and library functions rather than implementing custom solutions.

### Benefits of Library-First Examples

- **Demonstrates intended usage** - Shows users how to properly use the library
- **Reduces maintenance burden** - Examples break less often when refactoring library internals
- **Promotes composability** - Examples become templates for users building similar features
- **Self-documenting code** - Library function names serve as inline documentation

### Example Patterns

Good example patterns:
- Use `Terrain` class methods for rendering and data operations
- Use `ScoreCombiner` and score config classes for scoring logic
- Use `GriddedDataLoader` for memory-efficient data processing
- Leverage color mapping functions from `color_mapping.py`
- Use public functions from `terrain/` package, not internal implementations

Patterns to avoid:
- Direct Blender API calls (use `Terrain` class methods instead)
- Reimplementing scoring logic (use `ScoreCombiner`)
- Manual mesh generation (use `create_mesh()`)
- Custom DEM loading (use `Terrain` class data handling)

## Terrain-Maker Library

**The terrain-maker library is a powerful, well-designed tool. Use it fully.** The library provides a clear "grammar" for data operations. Understanding and extending this grammar is how the library grows.

### Library Capabilities

**Core Data Handling:**
- Load DEMs from HGT/GeoTIFF files
- Manage multiple geographic data layers (DEM, scores, roads, etc.)
- Automatic coordinate system handling (WGS84, UTM, custom CRS)
- Efficient reprojection and resampling
- Memory-efficient tiling for large datasets
- Caching layer results to avoid recomputation

**Transforms (apply to DEM or any raster):**
- `reproject_raster()` - Reproject between coordinate systems
- `downsample_raster()` - Reduce resolution with interpolation
- `flip_raster()` - Mirror along horizontal/vertical axis
- `scale_elevation()` - Multiply elevation values
- `smooth_raster()` - Apply median filter
- Custom transforms via `add_transform()`

**Terrain Analysis:**
- `detect_water_highres()` - Find water bodies from slope
- Compute slopes, aspects, hillshade
- Custom data layers from any source

**Coloring & Visualization:**
- `elevation_colormap()` - Map values to colors (any matplotlib colormap)
- `slope_colormap()` - Specialized slope visualization
- `set_color_mapping()` - Single colormap for terrain
- `set_blended_color_mapping()` - Dual colormaps with proximity masking
- `compute_proximity_mask()` - Create zones around points (parks, cities)
- Water detection and coloring (University of Michigan blue)

**Mesh Generation:**
- `create_mesh()` - Generate 3D mesh with vertex colors
- Automatic downsampling to target vertex count
- Boundary extension for seamless edges
- Center model option for balanced positioning

**Blender Integration:**
- `position_camera_relative()` - Smart camera positioning (cardinal directions)
- `setup_light()` - Configure sun/fill lighting
- `apply_vertex_colors()` - Apply colors to mesh vertices
- `render_scene_to_file()` - Render to PNG/JPEG with custom resolution
- `create_background_plane()` - Add background for drop shadows
- Material and shader management

**Examples Using Library:**
- `detroit_elevation_real.py` - Basic elevation visualization
- `detroit_snow_sledding.py` - Score-based coloring with water detection
- `detroit_xc_skiing.py` - Multi-layer rendering with proximity zones
- `detroit_combined_render.py` - Dual colormaps, roads, multiple data sources

### Core Library Concepts

**Terrain Class** - The primary interface for all operations
- Stores data layers (DEM, scores, derived data)
- Manages coordinate systems and transforms
- Handles reprojection and resampling automatically
- Creates meshes for Blender visualization

**Data Layer Pipeline** - The grammar for adding geographic data

```python
# Universal pattern for any geographic data:
terrain.add_data_layer(
    name="my_layer",              # Unique identifier
    data=my_grid,                 # 2D numpy array
    transform=my_transform,       # Affine in source CRS
    crs="EPSG:4326",             # Source CRS (usually WGS84)
    target_layer="dem",           # Align to DEM's transformed state
)
```

This pattern works for ANY geographic raster data:
- Elevation (DEM)
- Score grids (sledding, skiing, etc.)
- Roads (rasterized)
- Trails, power lines, utilities
- Land cover, zoning, vegetation
- Custom computed layers (slopes, aspects, etc.)

**Key principle**: `target_layer="dem"` tells the library to automatically:
1. Reproject data from source CRS → DEM's CRS
2. Resample data to match DEM's shape (handles downsampling!)
3. Align coordinates perfectly

### When to Suggest New Library Functions

**Suggest new library functions when you identify a pattern** that should be reusable across the codebase.

**Suggest if:**
- You implement the same data processing twice
- Examples show similar patterns (loading, transforming, coloring)
- A new feature requires coordinate transformations (use existing transforms)
- A task involves geographic data layers (use `add_data_layer()` pattern)
- Vertex coloring logic is repeated (extract to helper function)
- Mesh operations recur across examples (add to library)

**Do NOT suggest if:**
- It's a one-off operation specific to one example
- It's simple enough to be inline (< 10 lines of straightforward code)
- It's specific to one visualization approach (keep in example, not library)

### Architecture Patterns to Follow

**1. Rasterization**: Convert vector data → grid with geographic transform
```python
def my_data_to_layer(vector_data, bbox, resolution=30.0):
    """Convert vector data to raster with proper geographic transform."""
    # Calculate grid based on bbox and resolution
    # Create Affine transform in WGS84 (EPSG:4326)
    # Rasterize vector features to grid
    return raster_grid, affine_transform
```

**2. Data Layer Integration**: Add rasterized data as a layer
```python
# Rasterize with proper geographic metadata
grid, transform = my_data_to_layer(vector_data, bbox)

# Add as data layer (library handles alignment)
terrain.add_data_layer("my_layer", grid, transform, "EPSG:4326",
                       target_layer="dem")
```

**3. Vertex Coloring**: Use aligned layer to color vertices
```python
# Get aligned layer data (automatically reprojected/resampled by library)
layer_data = terrain.data_layers["my_layer"]["data"]

# For each vertex, check corresponding pixel and apply color
for vertex_idx in range(len(terrain.y_valid)):
    y, x = terrain.y_valid[vertex_idx], terrain.x_valid[vertex_idx]
    value = layer_data[y, x]
    if value > 0:  # Has data
        color = colormap(value)
        terrain.colors[vertex_idx, :3] = color
```

**4. Mesh Creation**: Call library once with all data prepared
```python
# Compute colors from all layers (DEM, scores, water, etc.)
terrain.compute_colors(water_mask=water_mask)

# Create mesh once (all layers already integrated)
mesh = terrain.create_mesh(scale_factor=100, height_scale=30)
```

### Real Example: Roads Data Layer

The roads implementation demonstrates the proper pattern:

**Before (broken):**
- Rasterized roads **after** all transforms using hardcoded mapping
- Roads appeared tiny, in wrong location
- Didn't account for DEM downsampling

**After (correct):**
- `rasterize_roads_to_layer()` - Creates proper WGS84 raster with Affine
- `add_roads_layer()` - Adds via `add_data_layer()` with `target_layer="dem"`
- Library automatically reprojects + resamples
- Roads align perfectly with downsampled DEM

## Architecture Principles

- Keep functions under 15 complexity
- Code files under 400 lines
- Comprehensive error handling
- Prefer functional programming patterns
- Avoid mutation where possible
- **Use existing library functions before writing custom code**
- **Treat data layers as first-class citizens (follow the add_data_layer pattern)**

## Claude Usage Guidelines

- Use `/estimate` before starting any non-trivial task
- Track actual Claude interactions vs estimates
- Optimize for message efficiency in complex tasks
- Budget Claude usage for different project phases

**Typical Usage Patterns**:

- **Bug Fix**: 10-30 messages
- **Small Feature**: 30-80 messages
- **Major Feature**: 100-300 messages
- **Architecture Change**: 200-500 messages

## Collaboration Guidelines

- Always add Claude as co-author on commits
- Run `/hygiene` before asking for help
- Use `/todo` for quick task capture
- Document learnings with `/learn`
- Regular `/reflect` sessions for insights

## Project Standards

- Test coverage: 60% minimum
- Documentation: All features documented
- Error handling: Graceful failures with clear messages
- Performance: Monitor code complexity and file sizes
- ALWAYS use atomic commits
- use emojis, judiciously
- NEVER Update() a file before you Read() the file.

### TDD Examples

- [🔴 test: add failing test for updateCommandCatalog isolation (TDD RED)](../../commit/00e7a22)
- [🔴 test: add failing tests for tdd.js framework detection (TDD RED)](../../commit/2ce43d1)
- [🔴 test: add failing tests for learn.js functions (TDD RED)](../../commit/8b90d58)
- [🔴 test: add failing tests for formatBytes and estimateTokens (TDD RED)](../../commit/1fdac58)
- [🔴 test: add failing tests for findBrokenLinks (TDD RED phase)](../../commit/8ec6319)
