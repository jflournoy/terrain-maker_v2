# Dual Terrain Rendering: Side-by-Side Activity Comparison

A complete example showing how to render two terrain suitability maps side-by-side with park location markers. This demonstrates multi-mesh Blender rendering, geographic coordinate transformation for markers, and sunset-style lighting.

## The Final Result

This pipeline produces a side-by-side comparison of sledding and XC skiing suitability:

![Dual Terrain Render](images/dual_render/sledding_and_xc_skiing_3d.png)

**Left**: Sledding suitability scores | **Right**: Cross-country skiing suitability with park markers

**Score Interpretation** (mako colormap):

- **Yellow/Green**: High suitability (0.7-1.0)
- **Cyan/Teal**: Moderate suitability (0.4-0.7)
- **Dark Blue/Purple**: Low suitability (0.0-0.4)
- **Blue areas**: Water bodies (detected by slope analysis)

**Park Markers**: Terra cotta spheres indicate park locations on the XC skiing terrain.

---

## What This Example Shows

- **Multi-Mesh Rendering**: Two terrain meshes positioned side-by-side with dynamic spacing
- **Geographic Marker Placement**: Convert lat/lon coordinates to Blender mesh space using `geo_to_mesh_coords()`
- **Coordinate Transformation**: WGS84 → UTM reprojection with automatic CRS handling
- **Water Detection**: Slope-based water body identification colored blue
- **Sunset Lighting**: Warm golden sun from southwest, cool blue fill from northeast
- **Memory Management**: Intelligent vertex limiting to prevent OOM with large DEMs

## The Pipeline

```
Sledding Scores ─┐
                 ├─→ Dual Terrain Mesh → Camera & Lighting → Render
XC Skiing Scores ┘
      └─→ Park Markers (geo_to_mesh_coords)
```

### Prerequisites

This example requires pre-computed scores from other examples:

```bash
# 1. Generate sledding scores
python examples/detroit_snow_sledding.py --steps score

# 2. Generate XC skiing scores and park data
python examples/detroit_xc_skiing.py
```

## The Code

### Loading Pre-Computed Scores

```python
from pathlib import Path
import numpy as np
from affine import Affine

def load_scores(output_dir: Path):
    """Load pre-computed scores with their transforms."""
    # Load sledding scores
    sledding_data = np.load(output_dir / "sledding_scores.npz")
    sledding_scores = sledding_data["score"]
    t = sledding_data["transform"]
    sledding_transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])

    # Load XC skiing scores
    xc_data = np.load(output_dir / "xc_skiing" / "xc_skiing_scores.npz")
    xc_scores = xc_data["score"]
    t = xc_data["transform"]
    xc_transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])

    return sledding_scores, sledding_transform, xc_scores, xc_transform
```

### Creating Terrain with Scores

```python
from src.terrain.core import (
    Terrain,
    reproject_raster,
    flip_raster,
    scale_elevation,
)
from src.terrain.color_mapping import elevation_colormap

def create_terrain_with_score(name, score_grid, dem, transform, dem_crs="EPSG:4326"):
    """Create a terrain mesh colored by suitability score."""
    terrain = Terrain(dem, transform, dem_crs=dem_crs)

    # Transform pipeline: reproject → flip → scale elevation
    terrain.add_transform(reproject_raster(src_crs=dem_crs, dst_crs="EPSG:32617"))
    terrain.add_transform(flip_raster(axis="horizontal"))
    terrain.add_transform(scale_elevation(scale_factor=0.0001))

    # Configure downsampling for memory efficiency
    terrain.configure_for_target_vertices(target_vertices=1920*1080)

    # Apply transforms
    terrain.apply_transforms()

    # Add score layer (automatically resampled to match DEM)
    terrain.add_data_layer("score", score_grid, score_transform, dem_crs, target_layer="dem")

    # Color by score using mako colormap
    terrain.set_color_mapping(
        lambda score: elevation_colormap(score, cmap_name="mako", min_elev=0.0, max_elev=1.0),
        source_layers=["score"],
    )
    terrain.compute_colors()

    # Create mesh with water detection
    mesh_obj = terrain.create_mesh(
        scale_factor=100,
        height_scale=10.0,
        center_model=True,
        boundary_extension=True,
    )

    return mesh_obj, terrain
```

### Placing Park Markers with geo_to_mesh_coords

The key feature is using `Terrain.geo_to_mesh_coords()` to convert geographic coordinates to Blender mesh space:

```python
def create_park_markers(parks, terrain, mesh_obj):
    """Create markers at park locations using proper coordinate transformation."""
    # Extract coordinates
    lons = np.array([park["lon"] for park in parks])
    lats = np.array([park["lat"] for park in parks])

    # Convert geographic coords to mesh space
    # Automatically handles CRS reprojection (WGS84 → UTM)
    xs, ys, zs = terrain.geo_to_mesh_coords(
        lons, lats,
        elevation_offset=0.1,  # Slightly above terrain
        input_crs="EPSG:4326"   # Parks are in WGS84
    )

    # Add mesh offset for multi-mesh scenes
    xs = xs + mesh_obj.location[0]
    ys = ys + mesh_obj.location[1]

    # Create marker spheres
    markers = []
    for i, park in enumerate(parks):
        bpy.ops.mesh.primitive_ico_sphere_add(
            radius=0.05,
            location=(xs[i], ys[i], zs[i])
        )
        marker = bpy.context.active_object
        marker.name = f"Park_{park['name']}"

        # Apply material (terra cotta for visibility)
        mat = bpy.data.materials.new(name=f"ParkMarker_{i}")
        bsdf = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Base Color"].default_value = (0.8, 0.4, 0.3, 1.0)
        markers.append(marker)

    return markers
```

### Sunset Lighting Setup

```python
from math import radians
from src.terrain.core import setup_light

def setup_sunset_lighting():
    """Create warm sunset lighting with cool fill."""
    # Primary sun - low angle from southwest
    sun = setup_light(
        angle=1,  # Sharp shadows
        energy=4.0,
        rotation_euler=(radians(75), 0, radians(-45)),
    )
    sun.data.color = (1.0, 0.85, 0.6)  # Warm golden

    # Fill light - cool blue from northeast
    fill = setup_light(
        angle=3,  # Soft
        energy=1.5,
        rotation_euler=(radians(60), 0, radians(135)),
    )
    fill.data.color = (0.7, 0.8, 1.0)  # Cool blue

    return [sun, fill]
```

## Running This Example

### Quick Start

```bash
# Run with pre-computed scores (requires sledding + XC skiing examples first)
python examples/detroit_dual_render.py

# Output: docs/images/dual_render/sledding_and_xc_skiing_3d.png
```

### With Mock Data

```bash
# Test rendering pipeline without real data
python examples/detroit_dual_render.py --mock-data
```

### Custom Output Directory

```bash
python examples/detroit_dual_render.py --output-dir ./my_renders
```

## Key Concepts

### geo_to_mesh_coords()

This method is essential for placing objects at geographic locations on terrain:

```python
x, y, z = terrain.geo_to_mesh_coords(lon, lat, elevation_offset=0.1)
```

It handles:

1. **CRS Reprojection**: Automatically converts from input CRS (e.g., WGS84) to the terrain's transformed CRS (e.g., UTM)
2. **Pixel Lookup**: Finds the corresponding pixel in the transformed DEM
3. **Elevation Extraction**: Gets the terrain height at that location
4. **Mesh Scaling**: Applies the same scale factors used in `create_mesh()`
5. **Centering Offset**: Accounts for model centering if enabled

### Multi-Mesh Positioning

When rendering multiple terrains side-by-side:

```python
# Calculate spacing based on mesh widths
gap = max(width_left, width_right) * 0.2
total_width = width_left + gap + width_right

# Position meshes
pos_left = (-total_width/2 + width_left/2, 0, 0)
pos_right = (total_width/2 - width_right/2, 0, 0)

mesh_left.location = pos_left
mesh_right.location = pos_right
```

### Memory Management

For large DEMs, limit vertex count:

```python
# Target ~2M vertices per mesh for dual rendering
terrain.configure_for_target_vertices(1920 * 1080)
```

## Output Files

| File | Description |
|------|-------------|
| `sledding_and_xc_skiing_3d.png` | Final rendered image (1920×1080) |
| `sledding_and_xc_skiing_3d.blend` | Blender project file for further editing |

## See Also

- [Snow Sledding Analysis](SNOW_SLEDDING.md) - Generate sledding scores
- [XC Skiing Analysis](XC_SKIING.md) - Generate XC skiing scores and park data
- [API Reference](API_REFERENCE.md) - `geo_to_mesh_coords()` documentation
