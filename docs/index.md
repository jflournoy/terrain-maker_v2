# terrain-maker Documentation

Welcome to the **terrain-maker** documentation. This library makes it easy to create 3D terrain visualizations with Blender from Digital Elevation Model (DEM) data.

## Quick Start

1. **[Install](PYTHON_SETUP.md)** - Set up Python environment
2. **[Quick Start Tutorial](QUICK_START.md)** - Get your first terrain in 5 minutes
3. **[Examples](EXAMPLES.md)** - See it in action with real data
4. **[Combined Render Guide](COMBINED_RENDER.md)** - Advanced rendering with all features
5. **[API Reference](API_REFERENCE.md)** - Detailed method documentation

## What is terrain-maker?

A Python library for:

- Loading and processing Digital Elevation Models (SRTM, GeoTIFF, etc.)
- Applying transforms (downsampling, smoothing, reprojection)
- Creating custom color mappings
- Generating 3D terrain meshes for Blender
- High-quality terrain visualization and rendering

## Key Features

✅ **Multi-source DEM loading** - SRTM, GeoTIFF, and other formats
✅ **Flexible transform pipeline** - Downsample, smooth, reproject, flip
✅ **Advanced color mapping** - Elevation, slope, custom functions
✅ **Blender integration** - Direct mesh generation with materials
✅ **Smart caching** - Hash-based caching for fast reprocessing

## Example: Real Detroit Terrain

The library includes a complete example that demonstrates all key features:

```python
from src.terrain.core import (
    Terrain, load_dem_files,
    downsample_raster
)

# Load 110 SRTM tiles covering Detroit
dem, transform = load_dem_files('/path/to/tiles', pattern='*.hgt')

# Create terrain object
terrain = Terrain(dem, transform)

# Apply transforms
terrain.transforms.append(downsample_raster(zoom_factor=0.1))
terrain.apply_transforms()

# Color by elevation
def color_by_elevation(dem):
    normalized = (dem - dem.min()) / (dem.max() - dem.min() + 1e-8)
    purple = np.array([0.5, 0, 1, 1])
    yellow = np.array([1, 1, 0, 1])
    colors = np.outer(normalized.flat, purple) + np.outer(1-normalized.flat, yellow)
    return (colors.reshape(dem.shape + (4,)) * 255).astype(np.uint8)

terrain.set_color_mapping(color_by_elevation)

# Generate mesh and render
mesh = terrain.create_mesh(scale_factor=400, height_scale=0.0035)
```

## Getting Started

1. **[Install dependencies](PYTHON_SETUP.md)** - Set up Python environment
2. **Read [API Reference](API_REFERENCE.md)** - Understand available methods
3. **Explore examples/** - Run provided examples
4. **Build your own** - Adapt examples for your data

## Common Patterns

### Load, Process, Visualize

```python
# Load multiple DEMs and merge
dem, transform = load_dem_files('/srtm/tiles')

# Create terrain object
terrain = Terrain(dem, transform)

# Apply transforms in sequence
terrain.transforms.append(downsample_raster(zoom_factor=0.1))
terrain.transforms.append(smooth_raster(window_size=5))
terrain.apply_transforms()

# Set color scheme
terrain.set_color_mapping(my_color_function)

# Generate mesh
mesh = terrain.create_mesh()
```

### Blender Rendering

```python
from src.terrain.core import (
    clear_scene, setup_camera_and_light,
    setup_render_settings, create_background_plane
)
from math import radians

# Clean scene
clear_scene()

# Configure rendering
setup_render_settings(samples=128, use_denoising=True)

# Position camera
camera, light = setup_camera_and_light(
    camera_angle=(radians(63.6), 0, radians(46.7)),
    camera_location=(7.36, -6.93, 4.96),
    scale=20
)

# Add background
background = create_background_plane(mesh)

# Render to file
import bpy
bpy.context.scene.render.filepath = "/output/terrain.png"
bpy.ops.render.render(write_still=True)
```

## Architecture

**Data Pipeline:**

```
Raw DEM Files
    ↓
load_dem_files() → Merged DEM
    ↓
Terrain(dem, transform)
    ↓
transforms (downsample, smooth, reproject, etc.)
    ↓
apply_transforms() → Transformed DEM
    ↓
set_color_mapping() → Color function
    ↓
create_mesh() → Blender mesh
    ↓
Blender rendering → PNG/GLTF/etc.
```

## Documentation Structure

- **[COMBINED\_RENDER.md](COMBINED_RENDER.md)**
  - Complete CLI reference for combined render
  - All smoothing and enhancement options
  - Camera, lighting, and material settings
  - Pipeline caching and memory management

- **[API\_REFERENCE.md](API_REFERENCE.md)**
  - Complete reference for all methods
  - Parameter documentation
  - Code examples
  - Performance benchmarks

- **[PYTHON\_SETUP.md](PYTHON_SETUP.md)**
  - Environment setup
  - Dependency installation
  - Troubleshooting

## Support

For issues or questions:

1. Check the [API Reference](API_REFERENCE.md) for method details
2. Review examples in `examples/` directory
3. Check existing issues/discussions

---

**Version:** 1.0
**Last Updated:** 2026-01-12
