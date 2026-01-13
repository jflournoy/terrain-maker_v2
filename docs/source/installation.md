# Installation

## Requirements

- Python 3.11+
- Blender Python API (bpy)

## Install with uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/jflournoy/terrain-maker_v2.git
cd terrain-maker_v2

# Install with Blender support
uv sync --extra blender

# Verify installation
uv run python -c "import bpy; print(f'Blender {bpy.app.version_string}')"
```

## Install with pip

```bash
pip install -e ".[blender]"
```

## Dependencies

Core dependencies are installed automatically:

- `numpy` - Array operations
- `rasterio` - GeoTIFF/HGT file reading
- `scipy` - Spatial operations
- `pyproj` - Coordinate transformations
- `shapely` - Geometry operations

Optional dependencies:

- `bpy` - Blender Python API (required for rendering)
- `numba` - JIT compilation for faster mesh generation
- `pywavelets` - Wavelet denoising
- `realesrgan` - AI super-resolution for score upscaling

## Verify Installation

```python
# Check terrain-maker
from src.terrain.core import Terrain
print("terrain-maker OK")

# Check Blender
import bpy
print(f"Blender {bpy.app.version_string}")
```

## Troubleshooting

### bpy not found

The Blender Python module requires Python 3.11. Install with:

```bash
uv pip install bpy
```

### numba warnings

Numba is optional. If not installed, mesh generation falls back to pure NumPy (slower but works).

### Memory issues with large DEMs

Use `configure_for_target_vertices()` to downsample:

```python
terrain.configure_for_target_vertices(500_000)  # 500K vertices max
```
