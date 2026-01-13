# Detroit Elevation Visualization

A complete example showing how to create 3D terrain visualizations from real SRTM elevation data.

![Detroit Elevation](../_static/detroit_elevation_south.png)

## Overview

This example demonstrates:
- Loading SRTM HGT elevation files
- Reprojecting from WGS84 to UTM coordinates
- Detecting water bodies automatically
- Rendering with Blender

## Quick Start

```bash
python examples/detroit_elevation_real.py --direction south
```

## The Code

```python
from src.terrain.core import Terrain, load_dem_files
from src.terrain.transforms import reproject_raster, flip_raster, scale_elevation
from src.terrain.color_mapping import elevation_colormap
from src.terrain.water import identify_water_by_slope
from src.terrain.scene_setup import position_camera_relative
from src.terrain.rendering import render_scene_to_file

# 1. Load elevation data
dem_data, transform = load_dem_files(SRTM_TILES_DIR, pattern='*.hgt')
terrain = Terrain(dem_data, transform)

# 2. Configure mesh density based on output resolution
WIDTH, HEIGHT = 960, 720
target_vertices = WIDTH * HEIGHT * 2
terrain.configure_for_target_vertices(target_vertices)

# 3. Apply geographic transforms
terrain.add_transform(reproject_raster(src_crs='EPSG:4326', dst_crs='EPSG:32617'))
terrain.add_transform(flip_raster(axis='horizontal'))
terrain.add_transform(scale_elevation(scale_factor=0.0001))
terrain.apply_transforms()

# 4. Set up color mapping
terrain.set_color_mapping(
    lambda dem: elevation_colormap(dem, cmap_name='mako'),
    source_layers=['dem']
)

# 5. Detect water bodies
water_mask = identify_water_by_slope(dem_data, slope_threshold=0.01)

# 6. Create mesh and render
mesh = terrain.create_mesh(
    scale_factor=100.0,
    height_scale=4.0,
    water_mask=water_mask
)
camera = position_camera_relative(mesh, direction='south')
render_scene_to_file("detroit.png", width=WIDTH, height=HEIGHT)
```

## Key Functions Used

| Function | Purpose |
|----------|---------|
| {func}`~terrain.core.Terrain` | Main terrain processing class |
| {func}`~terrain.core.load_dem_files` | Load and merge HGT/GeoTIFF files |
| {func}`~terrain.transforms.reproject_raster` | Convert between coordinate systems |
| {func}`~terrain.water.identify_water_by_slope` | Detect water bodies by slope analysis |
| {func}`~terrain.scene_setup.position_camera_relative` | Smart camera positioning |
| {func}`~terrain.rendering.render_scene_to_file` | Render to PNG/JPEG |

## Camera Views

Generate multiple views using cardinal directions:

```bash
python examples/detroit_elevation_real.py --direction north
python examples/detroit_elevation_real.py --direction east
python examples/detroit_elevation_real.py --direction above
```

| Direction | View |
|-----------|------|
| `south` | Looking north (default) |
| `north` | Looking south |
| `east` | Looking west |
| `west` | Looking east |
| `above` | Overhead view |

## See Also

- {doc}`sledding` - Adding snow analysis
- {doc}`combined_render` - Full-featured rendering
- {func}`~terrain.core.Terrain.configure_for_target_vertices` - Mesh optimization
