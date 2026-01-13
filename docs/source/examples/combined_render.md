# Combined Render: Full-Featured Example

The most comprehensive example demonstrating all terrain-maker features in one pipeline.

![Combined Render](../_static/combined_render.png)

## Overview

This example combines:
- Dual colormaps (sledding + XC skiing)
- Road rendering with glossy materials
- Print-quality output
- Advanced DEM smoothing
- Pipeline caching

## Quick Start

```bash
# Fast preview
python examples/detroit_combined_render.py

# Print quality (18x12 inches @ 300 DPI)
python examples/detroit_combined_render.py --print-quality

# With roads
python examples/detroit_combined_render.py --roads --smooth
```

## Feature Categories

### Dual Colormap System

Blend two score layers with proximity masking:

```python
from src.terrain.core import Terrain
from src.terrain.color_mapping import elevation_colormap

# Base layer: sledding scores (everywhere)
# Overlay: XC skiing scores (near parks only)
terrain.set_blended_color_mapping(
    base_colormap=lambda s: elevation_colormap(s, cmap_name='mako'),
    base_source_layers=['sledding'],
    overlay_colormap=lambda s: elevation_colormap(s, cmap_name='rocket'),
    overlay_source_layers=['xc_skiing'],
    overlay_mask=park_proximity_mask
)
```

**Key function:** {func}`~terrain.core.Terrain.set_blended_color_mapping`

### Road Rendering

```bash
python examples/detroit_combined_render.py \
    --roads \
    --road-types motorway trunk primary \
    --road-color azurite \
    --road-width 3
```

**Key functions:**
- {func}`~terrain.roads.add_roads_layer` - Add roads as data layer
- {func}`~terrain.roads.smooth_road_vertices` - Smooth road elevations
- {func}`~terrain.materials.apply_terrain_with_obsidian_roads` - Glossy road material

### DEM Smoothing

Multiple smoothing options can be combined:

```bash
# Wavelet denoising (best for sensor noise)
--wavelet-denoise --wavelet-type db4 --wavelet-levels 3

# Slope-adaptive (smooths flat areas, preserves hills)
--adaptive-smooth --adaptive-slope-threshold 2.0

# Bump removal (removes buildings/trees)
--remove-bumps 2 --remove-bumps-strength 0.7
```

**Key functions:**
- {func}`~terrain.transforms.wavelet_denoise_dem`
- {func}`~terrain.transforms.slope_adaptive_smooth`
- {func}`~terrain.transforms.remove_bumps`

### Camera & Lighting

```bash
--camera-direction south \
--sun-azimuth 225 \
--sun-elevation 30 \
--hdri-lighting
```

**Key functions:**
- {func}`~terrain.scene_setup.position_camera_relative`
- {func}`~terrain.scene_setup.setup_hdri_lighting`
- {func}`~terrain.scene_setup.setup_two_point_lighting`

### Print Quality

```bash
python examples/detroit_combined_render.py \
    --print-quality \
    --print-dpi 300 \
    --print-width 18 \
    --print-height 12 \
    --format jpeg \
    --embed-profile
```

**Key function:** {func}`~terrain.rendering.render_scene_to_file`

## CLI Reference

### Basic Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir` | `docs/images/combined_render` | Output directory |
| `--mock-data` | False | Use mock data |
| `--no-render` | False | Skip rendering |

### Quality Options

| Flag | Default | Description |
|------|---------|-------------|
| `--print-quality` | False | High-res mode |
| `--print-dpi` | 300 | Print resolution |
| `--vertex-multiplier` | auto | Mesh detail |

### Smoothing Options

| Flag | Description |
|------|-------------|
| `--smooth` | Bilateral smoothing |
| `--wavelet-denoise` | Wavelet denoising |
| `--adaptive-smooth` | Slope-adaptive |
| `--remove-bumps SIZE` | Morphological |
| `--despeckle-dem` | Median filter |

### Road Options

| Flag | Default | Description |
|------|---------|-------------|
| `--roads` | False | Enable roads |
| `--road-types` | motorway trunk primary | OSM types |
| `--road-color` | azurite | Color preset |
| `--road-width` | 3 | Width in pixels |

## Pipeline Architecture

```
Load DEM → Transform → Detect Water → Smooth → Add Layers → Color → Mesh → Render
```

1. **Load**: DEM + score grids + parks + roads
2. **Transform**: Reproject, flip, downsample
3. **Water**: Slope-based detection (before smoothing)
4. **Smooth**: Wavelet/adaptive/bilateral
5. **Layers**: Align scores and roads to DEM
6. **Color**: Dual colormap with proximity mask
7. **Mesh**: Create vertices with colors
8. **Render**: Camera, lights, background, output

## See Also

- {doc}`elevation` - Basic terrain
- {doc}`sledding` - Snow analysis
- {func}`~terrain.cache.PipelineCache` - Caching system
