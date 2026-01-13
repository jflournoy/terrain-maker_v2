---
title: Quick Start
nav_order: 2
---

# Quick Start: Your First Terrain in 5 Minutes

Get terrain-maker running and create your first 3D terrain visualization.

## Prerequisites

- Python 3.9+
- Basic understanding of Python
- Blender 3.0+ (for rendering)

## 1. Install

```bash
# Install with uv (recommended)
uv pip install terrain-maker

# Or with pip
pip install terrain-maker
```

## 2. Download DEM Data

You'll need elevation data. Two options:

### Option A: Use provided example data
```bash
# The Detroit example includes 110 SRTM tiles (provided)
python examples/detroit_elevation_real.py
```

### Option B: Get your own SRTM data
Download from [NASA Earth Explorer](https://earthexplorer.usgs.gov/):
1. Search your area of interest
2. Filter: SRTM Elevation Products
3. Download .hgt files (SRTM 90m tiles)
4. Place in `data/dem/your-region/`

## 3. Simple Example: Load and Visualize

```python
from pathlib import Path
from src.terrain.core import (
    Terrain,
    load_dem_files,
    elevation_colormap,
    position_camera_relative,
    setup_light,
    setup_render_settings,
    render_scene_to_file,
    clear_scene
)

# Setup
dem_dir = Path('data/dem/detroit')
clear_scene()

# Step 1: Load DEM tiles
print("Loading DEM...")
dem_data, transform = load_dem_files(dem_dir, pattern='*.hgt')

# Step 2: Create terrain
print("Creating terrain...")
terrain = Terrain(dem_data, transform)

# Step 3: Downsample for faster processing
terrain.configure_for_target_vertices(500_000)
terrain.apply_transforms()

# Step 4: Set color by elevation
terrain.set_color_mapping(
    lambda dem: elevation_colormap(dem, cmap_name='viridis'),
    source_layers=['dem']
)

# Step 5: Create 3D mesh
print("Creating mesh...")
mesh = terrain.create_mesh(scale_factor=100, height_scale=1)

# Step 6: Position camera (south view, looking north)
camera = position_camera_relative(
    mesh,
    direction='south',
    distance=1.5,
    elevation=0.5
)

# Step 7: Setup lighting and rendering
setup_light(angle=2, energy=3)
setup_render_settings(samples=256, use_denoising=True)

# Step 8: Render to PNG
print("Rendering...")
output = render_scene_to_file(
    output_path='my_terrain.png',
    width=1280,
    height=720
)
print(f"Saved: {output}")
```

## 4. What Each Step Does

| Step | Purpose |
|------|---------|
| `load_dem_files()` | Load and merge SRTM elevation tiles |
| `Terrain()` | Create terrain object from DEM |
| `configure_for_target_vertices()` | Intelligent downsampling |
| `set_color_mapping()` | Color terrain by elevation (or custom function) |
| `create_mesh()` | Generate Blender 3D mesh |
| `position_camera_relative()` | Position camera intuitively (north, south, above, etc) |
| `setup_light()` | Add sun light for illumination |
| `setup_render_settings()` | Configure Blender cycles renderer |
| `render_scene_to_file()` | Render to PNG/JPEG/etc |

## 5. Customize It

### Different colormap
```python
# Instead of viridis
elevation_colormap(dem, cmap_name='mako')      # Beautiful perceptually uniform
elevation_colormap(dem, cmap_name='turbo')     # High contrast
elevation_colormap(dem, cmap_name='plasma')    # Scientific
```

### Different camera angle
```python
# Change direction
position_camera_relative(mesh, direction='north')     # Look south
position_camera_relative(mesh, direction='above')     # Looking down
position_camera_relative(mesh, direction='northeast') # Diagonal

# Adjust distance and elevation
position_camera_relative(mesh, direction='south', distance=2.0, elevation=1.5)
```

### Custom color function
```python
import numpy as np

def color_by_slope(dem):
    """Color terrain by steepness instead of elevation"""
    # Calculate slope
    gy, gx = np.gradient(dem)
    slope = np.sqrt(gx**2 + gy**2)

    # Normalize to 0-1
    slope_norm = (slope - slope.min()) / (slope.max() - slope.min() + 1e-8)

    # Red for steep, blue for flat
    colors = np.zeros((*dem.shape, 4), dtype=np.uint8)
    colors[..., 0] = (slope_norm * 255).astype(np.uint8)  # Red channel
    colors[..., 2] = ((1 - slope_norm) * 255).astype(np.uint8)  # Blue channel
    colors[..., 3] = 255  # Alpha

    return colors

terrain.set_color_mapping(color_by_slope, source_layers=['dem'])
```

### Detect and color water bodies
```python
from src.terrain.water import identify_water_by_slope

# Get unscaled DEM for meaningful slope calculations
# (important: detect BEFORE scaling elevation)
transformed_dem = terrain.data_layers['dem']['transformed_data']
scale_factor = 0.0001  # Must match your scale_elevation transform
unscaled_dem = transformed_dem / scale_factor

# Detect water using slope-based analysis
water_mask = identify_water_by_slope(
    unscaled_dem,
    slope_threshold=0.01,  # Adjust for your terrain (lower = more sensitive)
    fill_holes=True        # Smooth water mask
)

# Create mesh with water detection
# Water pixels will be colored blue, land by elevation
mesh = terrain.create_mesh(
    scale_factor=100,
    height_scale=1,
    water_mask=water_mask
)
```

See the [Water Body Detection section](API_REFERENCE.md#terrainwater) in the API Reference for details and threshold guidelines.

## 6. Next Steps

- **See it in action**: Check [Examples](EXAMPLES.md) for a complete Detroit example
- **Learn the API**: Browse [API Reference](API_REFERENCE.md)
- **Explore features**: Read [Best Practices](BEST_PRACTICES.md)

## Troubleshooting

**"ModuleNotFoundError: No module named 'src'"**
```bash
# Make sure you're in the project root directory
cd terrain-maker_v2
python your_script.py
```

**"Blender not found"**
```bash
# Ensure Blender 3.0+ is installed
# Set BLENDER_EXECUTABLE environment variable if not in PATH
export BLENDER_EXECUTABLE=/path/to/blender
```

**Out of memory with large DEMs**
```python
# Reduce target vertices
terrain.configure_for_target_vertices(100_000)  # Smaller mesh

# Render at lower resolution
render_scene_to_file(..., width=640, height=480)
```

---

**Ready?** Run the Detroit example: `python examples/detroit_elevation_real.py`
