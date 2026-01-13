# Quick Start

Get your first terrain visualization in 5 minutes.

## Prerequisites

- Python 3.11+
- Blender Python API (`bpy`)

## Installation

```bash
# Clone the repository
git clone https://github.com/jflournoy/terrain-maker_v2.git
cd terrain-maker_v2

# Install with uv (recommended)
uv sync --extra blender

# Or with pip
pip install -e ".[blender]"
```

## Your First Terrain

```python
from src.terrain.core import Terrain
from src.terrain.transforms import reproject_raster, flip_raster
from src.terrain.scene_setup import position_camera_relative
from src.terrain.rendering import render_scene_to_file
import numpy as np

# Create mock elevation data
dem = np.random.randint(100, 300, (512, 512)).astype(np.float32)
from rasterio.transform import Affine
transform = Affine.translation(-83.0, 42.5) * Affine.scale(0.001, -0.001)

# Create terrain
terrain = Terrain(dem, transform)
terrain.add_transform(reproject_raster(src_crs='EPSG:4326', dst_crs='EPSG:32617'))
terrain.add_transform(flip_raster(axis='horizontal'))
terrain.apply_transforms()

# Create mesh and render
mesh = terrain.create_mesh(scale_factor=100, height_scale=5)
camera = position_camera_relative(mesh, direction='south')
render_scene_to_file('my_terrain.png', width=800, height=600)
```

## Run an Example

```bash
# Detroit elevation with real SRTM data
python examples/detroit_elevation_real.py

# With mock data (no real files needed)
python examples/detroit_elevation_real.py --mock-data
```

## Next Steps

- {doc}`examples/elevation` - Detailed elevation example
- {doc}`examples/sledding` - Snow data integration
- {doc}`examples/combined_render` - Full-featured rendering
- {doc}`api/core` - API reference
