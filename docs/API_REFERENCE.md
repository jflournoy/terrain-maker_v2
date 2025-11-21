# terrain-maker API Reference

Comprehensive documentation of all classes and functions in the terrain-maker library.

## Table of Contents

- [Terrain Class](#terrain-class)
- [Data Loading](#data-loading)
- [Transform Functions](#transform-functions)
- [Analysis Functions](#analysis-functions)
- [Blender Rendering](#blender-rendering)
- [Utility Functions](#utility-functions)

---

## Terrain Class

Core class for managing DEM data and terrain operations.

### `Terrain.__init__(dem_data, transform, crs='EPSG:4326')`

Initialize a Terrain object with elevation data.

**Parameters:**

- `dem_data` (np.ndarray): 2D array of elevation values (height × width)
- `transform` (Affine): Affine transform mapping pixel coordinates to geographic coordinates
- `crs` (str): Coordinate reference system in EPSG format (default: 'EPSG:4326' - WGS84)

**Returns:** Terrain object

**Example:**

```python
import numpy as np
from affine import Affine
from src.terrain.core import Terrain

dem = np.random.rand(100, 100) * 1000  # 100×100 grid, elevation 0-1000m
transform = Affine.identity()
terrain = Terrain(dem, transform, crs='EPSG:4326')
```

**Properties:**

- `dem_shape`: Tuple of (height, width)
- `data_layers`: Dictionary of all data layers
- `transforms`: List of transforms to apply
- `terrain_obj`: Blender object (after mesh creation)

---

### `Terrain.apply_transforms()`

Apply all registered transforms to data layers in sequence.

**Returns:** None (modifies in-place)

**Behavior:**

- Processes each transform in the order added
- Caches results based on transform hash
- Marks data layers as "transformed"
- Stores CRS changes

**Example:**

```python
terrain.transforms.append(downsample_raster(zoom_factor=0.1))
terrain.transforms.append(smooth_raster(window_size=3))
terrain.apply_transforms()
```

---

### `Terrain.configure_for_target_vertices(target_vertices, order=4)`

Configure downsampling to achieve approximately target_vertices.

This method calculates the appropriate zoom_factor to achieve a desired vertex count for mesh generation. It provides a more intuitive API than manually calculating zoom_factor from the original DEM shape.

**Parameters:**

- `target_vertices` (int): Desired vertex count for final mesh (e.g., 500_000)
- `order` (int): Interpolation order for downsampling (0=nearest, 1=linear, 4=bicubic, default: 4)

**Returns:** float - Calculated zoom_factor that was added to transforms

**Raises:**

- `ValueError`: If target_vertices is not a positive integer

**Example:**

```python
terrain = Terrain(dem, transform)

# Configure to achieve 500,000 vertices
zoom = terrain.configure_for_target_vertices(500_000)
print(f"Calculated zoom_factor: {zoom:.6f}")

terrain.apply_transforms()
mesh = terrain.create_mesh(scale_factor=400.0)
```

**Benefits:**

- **Intuitive API**: Specify desired vertices instead of zoom_factor
- **Automatic calculation**: No need to do manual math (sqrt of ratio)
- **Safe**: Handles edge cases (target > source, invalid inputs)
- **Informative**: Logs the calculated values for verification

**Use Cases:**

- Target specific mesh complexity for rendering performance
- Achieve consistent vertex counts across different DEM sizes
- Simplify experiment configuration in scripts

---

### `Terrain.set_color_mapping(color_func, source_layers=None, mask_func=None, mask_sources=None, mask_threshold=None, **kwargs)`

Configure color mapping for terrain visualization.

**Parameters:**

- `color_func` (callable): Function that maps DEM values to RGB(A) colors
  - Signature: `func(dem_array) -> np.ndarray` with shape `(height, width, 3)` or `(height, width, 4)`
- `source_layers` (list): Data layers to use for color input (default: \['dem'])
- `mask_func` (callable): Optional function to create transparency mask
- `mask_sources` (list): Data layers for mask input
- `mask_threshold` (float): Threshold for mask application
- `**kwargs`: Additional keyword arguments passed to color\_func

**Returns:** None

**Example:**

```python
def elevation_to_color(dem):
    # Normalize elevation
    norm = (dem - dem.min()) / (dem.max() - dem.min() + 1e-8)
    # Create RGB
    rgb = np.stack([norm, norm*0.5, 1-norm], axis=-1)
    return (rgb * 255).astype(np.uint8)

terrain.set_color_mapping(elevation_to_color, source_layers=['dem'])
```

---

### `Terrain.create_mesh(base_depth=-0.2, boundary_extension=True, scale_factor=100.0, height_scale=1.0, center_model=True, verbose=True)`

Generate a Blender 3D mesh from transformed DEM data.

**Parameters:**

- `base_depth` (float): Z-coordinate for terrain base (default: -0.2)
- `boundary_extension` (bool): Whether to create side faces (default: True)
- `scale_factor` (float): Horizontal scale divisor for coordinates (default: 100.0)
- `height_scale` (float): Multiplier for elevation Z values (default: 1.0)
- `center_model` (bool): Whether to center at origin (default: True)
- `verbose` (bool): Log progress information (default: True)

**Returns:** bpy.types.Object (Blender mesh object)

**Technical Details:**

- Creates mesh using vectorized numpy operations
- Optimizes boundary detection with morphological operations
- Applies vertex colors from color mapping if available
- Automatically creates and applies material

**Example:**

```python
mesh = terrain.create_mesh(
    scale_factor=200.0,
    height_scale=0.01,
    center_model=True,
    boundary_extension=True
)
```

---

## Data Loading

### `load_dem_files(directory_path, pattern='*.hgt', recursive=False)`

Load and merge multiple DEM files (HGT, GeoTIFF, etc.) from a directory.

**Parameters:**

- `directory_path` (str or Path): Directory containing DEM files
- `pattern` (str): File glob pattern (default: '\*.hgt' for SRTM tiles)
- `recursive` (bool): Search subdirectories recursively (default: False)

**Returns:** Tuple of (dem\_array, transform)

- `dem_array` (np.ndarray): Merged elevation data
- `transform` (Affine): Georeferencing transform

**Raises:**

- `ValueError`: If no valid DEM files found
- `OSError`: If directory access fails

**Example:**

```python
from src.terrain.core import load_dem_files

dem, transform = load_dem_files('/path/to/srtm/tiles', pattern='*.hgt')
print(f"Loaded DEM: {dem.shape}, range: {dem.min():.1f}-{dem.max():.1f}m")
```

**Supported Formats:**

- HGT (SRTM, Shuttle Radar Topography Mission)
- GeoTIFF (any georeferenced TIFF)
- Any format supported by rasterio

---

## Transform Functions

Transform functions are factories that create transform operations for the transform pipeline. They modify DEM data and return (data, transform, crs) tuples.

### `downsample_raster(zoom_factor=0.1, order=4, nodata_value=np.nan)`

Create a downsampling transform using scipy interpolation.

**Parameters:**

- `zoom_factor` (float): Scaling factor (0.1 = 10:1 reduction, 0.5 = 2:1)
- `order` (int): Interpolation order (0=nearest, 1=linear, 4=bicubic) (default: 4)
- `nodata_value`: Value marking missing data (default: np.nan)

**Returns:** Transform function

**Use Cases:**

- Reduce mesh complexity for large DEMs
- Faster rendering at lower resolution
- Memory-efficient processing

**Example:**

```python
from src.terrain.core import downsample_raster

# Reduce DEM by 10x using bicubic interpolation
downsample = downsample_raster(zoom_factor=0.1, order=4)
terrain.transforms.append(downsample)
```

**Performance:**

- 0.1 zoom on 36,000×39,600 grid: ~4 seconds
- Preserves elevation statistics (min/max values)

---

### `smooth_raster(window_size=None, nodata_value=np.nan)`

Create a smoothing transform using median filtering.

**Parameters:**

- `window_size` (int or None): Filter window size in pixels
  - None = auto-calculate from raster size
  - Typical: 3-7 pixels for terrain smoothing
- `nodata_value`: Value marking missing data (default: np.nan)

**Returns:** Transform function

**Use Cases:**

- Remove noise from noisy DEM data
- Smooth terrain before analysis
- Prepare data for slope/aspect calculations

**Example:**

```python
from src.terrain.core import smooth_raster

# Smooth with 5×5 median filter
smooth = smooth_raster(window_size=5)
terrain.transforms.append(smooth)
```

**Effect:**

- Preserves edges while smoothing
- Reduces noise ~20-30%
- Applied before other transforms

---

### `flip_raster(axis='horizontal')`

Create a transform that mirrors (flips) the DEM data.

**Parameters:**

- `axis` (str): 'horizontal' (flip top↔bottom) or 'vertical' (flip left↔right)

**Returns:** Transform function

**Use Cases:**

- Correct for inverted coordinate systems
- Mirror terrain for artistic effects
- Align data from different sources

**Example:**

```python
from src.terrain.core import flip_raster

# Flip terrain vertically (invert rows)
flip = flip_raster(axis='horizontal')
terrain.transforms.append(flip)
terrain.apply_transforms()
```

**Technical Details:**

- Updates affine transform accordingly
- Handles both axis-aligned and rotated rasters
- Preserves all data statistics

---

### `reproject_raster(src_crs='EPSG:4326', dst_crs='EPSG:32617', nodata_value=np.nan, num_threads=4)`

Create a coordinate system reprojection transform.

**Parameters:**

- `src_crs` (str): Source CRS in EPSG format (default: 'EPSG:4326' - WGS84)
- `dst_crs` (str): Destination CRS (default: 'EPSG:32617' - UTM Zone 17N)
- `nodata_value`: Value for areas outside source extent (default: np.nan)
- `num_threads` (int): GDAL threads for parallel processing (default: 4)

**Returns:** Transform function

**Use Cases:**

- Convert between geographic and projected coordinates
- Combine DEMs from different CRS
- Prepare for analysis requiring projected coordinates

**Example:**

```python
from src.terrain.core import reproject_raster

# Reproject from WGS84 (lat/lon) to UTM Zone 17N (meters)
reproject = reproject_raster(
    src_crs='EPSG:4326',
    dst_crs='EPSG:32617'
)
terrain.transforms.append(reproject)
```

**Performance:**

- GDAL multi-threaded processing
- Bilinear interpolation by default
- Preserves elevation values during reprojection

---

## Analysis Functions

### `slope_colormap(slopes, cmap_name='terrain', min_slope=0, max_slope=45)`

Generate colors based on terrain slope values.

**Parameters:**

- `slopes` (np.ndarray): 2D array of slope values in degrees
- `cmap_name` (str): Matplotlib colormap name (default: 'terrain')
- `min_slope` (float): Minimum slope for color range normalization (default: 0)
- `max_slope` (float): Maximum slope for color range normalization (default: 45)

**Returns:** np.ndarray of RGBA values, shape (height, width, 4)

**Supported Colormaps:**

- 'terrain': Brown→tan (terrain)
- 'viridis': Purple→yellow (perceptually uniform)
- 'plasma': Purple→yellow (high contrast)
- 'twilight': Purple→orange (cyclic)
- Any matplotlib colormap

**Example:**

```python
from src.terrain.core import slope_colormap
from scipy.ndimage import sobel

# Calculate slope from DEM
grad_x = sobel(dem_data, axis=0)
grad_y = sobel(dem_data, axis=1)
slopes = np.arctan(np.sqrt(grad_x**2 + grad_y**2)) * 180 / np.pi

# Get colors
colors = slope_colormap(slopes, cmap_name='viridis', min_slope=0, max_slope=60)
```

**Use Cases:**

- Visualize terrain steepness
- Identify erosion-prone areas
- Highlight avalanche-prone slopes
- Distinguish terrain types

---

## Blender Rendering

All Blender rendering functions require `import bpy`.

### `clear_scene()`

Delete all objects from the current Blender scene.

**Parameters:** None

**Returns:** None

**Effect:**

- Removes all objects (meshes, lights, cameras)
- Calls `bpy.ops.wm.read_factory_settings(use_empty=True)`
- Cleans up for fresh scene setup

**Example:**

```python
from src.terrain.core import clear_scene

clear_scene()  # Start fresh
# ... create new objects ...
```

---

### `setup_camera_and_light(camera_angle, camera_location, scale, sun_angle=2, sun_energy=3, focal_length=50, camera_type='PERSP')`

Configure camera and sun light for terrain visualization.

**Parameters:**

- `camera_angle` (tuple): Rotation in radians (x, y, z)
- `camera_location` (tuple): Position in world coordinates (x, y, z)
- `scale` (float): Camera scale value (ortho_scale for orthographic cameras)
- `sun_angle` (float): Sun light angular size in degrees (default: 2)
- `sun_energy` (float): Light intensity/energy (default: 3)
- `focal_length` (float): Camera focal length in mm (default: 50, perspective only)
- `camera_type` (str): Camera type 'PERSP' (perspective) or 'ORTHO' (orthographic) (default: 'PERSP')

**Returns:** Tuple of (camera\_object, sun\_object)

**Raises:** ValueError if camera_type is not 'PERSP' or 'ORTHO'

**Examples:**

Perspective camera (default):

```python
from src.terrain.core import setup_camera_and_light
from math import radians

camera, light = setup_camera_and_light(
    camera_angle=(radians(63.6), radians(0), radians(46.7)),
    camera_location=(7.36, -6.93, 4.96),
    scale=20.0,
    camera_type='PERSP',
    focal_length=50
)
```

Orthographic camera:

```python
camera, light = setup_camera_and_light(
    camera_angle=(radians(45), radians(0), radians(45)),
    camera_location=(5.0, -5.0, 3.0),
    scale=15.0,
    camera_type='ORTHO'
)
```

**Camera Type Comparison:**

| Aspect | Perspective | Orthographic |
|--------|-------------|--------------|
| Real-world look | ✓ Natural depth | Flat, technical |
| Scale parameter | Background distance feel | Image scale |
| focal_length | Controls FOV | Ignored |
| Best for | Natural renders | Technical visualization |

**Typical Camera Angles:**

- Isometric: (63.6°, 0°, 46.7°)
- Top-down: (0°, 0°, 0°)
- Front: (90°, 0°, 0°)
- Side: (90°, 0°, 90°)

---

### `setup_world_atmosphere(density=0.02, scatter_color=(1, 1, 1, 1), anisotropy=0.0)`

Add volumetric atmospheric effects to world.

**Parameters:**

- `density` (float): Volume scatter density (default: 0.02)
- `scatter_color` (tuple): RGBA color of scattered light (default: white)
- `anisotropy` (float): Scatter direction (-1 to 1, 0 = uniform) (default: 0.0)

**Returns:** bpy.types.World object

**Use Cases:**

- Create hazy atmospheric effects
- Add volumetric fog
- Improve visual depth

**Example:**

```python
from src.terrain.core import setup_world_atmosphere

world = setup_world_atmosphere(
    density=0.02,
    scatter_color=(1, 1, 1, 1),
    anisotropy=0.0
)
```

---

### `setup_render_settings(use_gpu=True, samples=128, preview_samples=32, use_denoising=True, denoiser='OPTIX', compute_device='OPTIX')`

Configure Blender Cycles renderer settings.

**Parameters:**

- `use_gpu` (bool): Enable GPU acceleration (default: True)
- `samples` (int): Render samples/bounces (default: 128)
- `preview_samples` (int): Viewport preview samples (default: 32)
- `use_denoising` (bool): Enable denoiser (default: True)
- `denoiser` (str): 'OPTIX', 'OPENIMAGEDENOISE', or 'NLM' (default: 'OPTIX')
- `compute_device` (str): 'OPTIX', 'CUDA', 'HIP', 'METAL' (default: 'OPTIX')

**Returns:** None (configures scene in-place)

**Quality Presets:**

- Fast: samples=32, denoising=True
- Medium: samples=64, denoising=True
- High: samples=128, denoising=True
- Ultra: samples=256, denoising=False

**Example:**

```python
from src.terrain.core import setup_render_settings

# High-quality rendering
setup_render_settings(
    use_gpu=True,
    samples=128,
    use_denoising=True,
    denoiser='OPTIX'
)
```

---

### `apply_colormap_material(material)`

Configure material with vertex color support and emission.

**Parameters:**

- `material` (bpy.types.Material): Material to configure

**Returns:** None (modifies in-place)

**Shader Setup:**

- Vertex Color node → reads TerrainColors layer
- Emission shader → 70% color strength
- Principled BSDF → 30% reflective
- Mix shader → combines emission + reflection

**Example:**

```python
import bpy
from src.terrain.core import apply_colormap_material

material = bpy.data.materials.new(name="TerrainMaterial")
apply_colormap_material(material)
mesh.materials.append(material)
```

**Result:**

- Vertex colors visible regardless of lighting
- Mix of self-illuminated and reflected appearance
- Professional terrain visualization

---

### `create_background_plane(terrain_obj, depth=-2.0, scale_factor=2.0, material_params=None)`

Add a background plane beneath terrain for visual context.

**Parameters:**

- `terrain_obj` (bpy.types.Object): Terrain mesh for size reference
- `depth` (float): Z-coordinate for plane (default: -2.0)
- `scale_factor` (float): Scale relative to terrain (default: 2.0)
- `material_params` (dict): Material customization
  - 'base\_color': (R, G, B, A)
  - 'emission\_color': (R, G, B, A)
  - 'emission\_strength': float
  - 'roughness': float
  - 'metallic': float

**Returns:** bpy.types.Object (background plane)

**Example:**

```python
from src.terrain.core import create_background_plane

bg_plane = create_background_plane(
    terrain_obj=mesh,
    depth=-2.0,
    scale_factor=2.5,
    material_params={
        'emission_color': (0.8, 0.8, 0.9, 1.0),
        'emission_strength': 0.3
    }
)
```

---

## Utility Functions

### `transform_wrapper(transform_func)`

Decorator for creating standardized transform functions.

**Parameters:**

- `transform_func` (callable): Function with signature `(data, transform) -> (data, transform, crs)`

**Returns:** Wrapped transform function

**Use Cases:**

- Create custom transforms
- Ensure consistent interface
- Integrate external algorithms

**Example:**

```python
from src.terrain.core import transform_wrapper

@transform_wrapper
def my_custom_transform(data, transform):
    # Your processing here
    processed_data = data * 2  # Example: double elevations
    return processed_data, transform, None

terrain.transforms.append(my_custom_transform)
```

---

## Common Patterns

### Pattern 1: Load, Downsample, Smooth, Visualize

```python
from src.terrain.core import Terrain, load_dem_files, downsample_raster, smooth_raster

# Load DEM
dem, transform = load_dem_files('/path/to/tiles')

# Create Terrain object
terrain = Terrain(dem, transform)

# Add transforms
terrain.transforms.append(downsample_raster(zoom_factor=0.1))
terrain.transforms.append(smooth_raster(window_size=5))
terrain.apply_transforms()

# Color by elevation
def elev_color(dem):
    norm = (dem - dem.min()) / (dem.max() - dem.min() + 1e-8)
    rgb = np.stack([norm, 1-norm, 0.5], axis=-1)
    return (rgb * 255).astype(np.uint8)

terrain.set_color_mapping(elev_color)

# Create mesh and render
mesh = terrain.create_mesh(scale_factor=200, height_scale=0.01)
```

### Pattern 2: Full Rendering Pipeline

```python
from src.terrain.core import (
    Terrain, clear_scene, setup_camera_and_light,
    setup_render_settings, create_background_plane
)

# Load terrain (... steps omitted ...)

# Setup Blender scene
clear_scene()
setup_render_settings(samples=64, use_denoising=True)
cam, light = setup_camera_and_light(
    camera_angle=(1.11, 0, 0.82),
    camera_location=(10, -10, 5),
    scale=20
)
bg = create_background_plane(mesh)

# Render
import bpy
bpy.context.scene.render.filepath = "/output/render.png"
bpy.ops.render.render(write_still=True)
```

### Pattern 3: Multi-layer Analysis

```python
# Load elevation and additional data
dem, dem_trans = load_dem_files('/dem/')
slope_data, _ = load_dem_files('/slope/')

# Create terrain with multiple layers
terrain = Terrain(dem, dem_trans)
terrain.add_data_layer('slope', slope_data)

# Apply transforms to elevation only
terrain.transforms.append(downsample_raster(zoom_factor=0.1))
terrain.apply_transforms()

# Color by slope
slope_colors = slope_colormap(terrain.data_layers['slope']['transformed_data'])
terrain.set_color_mapping(lambda dem: slope_colors)
```

---

## Error Handling

### Common Errors and Solutions

**AttributeError: 'bpy' module not available**

- Solution: Ensure running within Blender or with bpy installed
- Blender must be in Python path

**ValueError: Not enough values to unpack (expected 3, got 2)**

- Solution: Transform functions must return 3-tuple: (data, transform, crs)
- See `transform_wrapper` for correct pattern

**Memory Error on large DEMs**

- Solution: Use downsampling before mesh creation
- Consider zoom\_factor=0.1 or lower

**Blender crash on very large meshes (>20M vertices)**

- Solution: Apply additional downsampling (0.05 factor)
- Consider splitting into tiles

---

## Performance Considerations

| Operation | Time | Memory | Notes |
|-----------|------|--------|-------|
| Load 100 SRTM tiles | ~10s | 2 GB | Depends on disk I/O |
| Downsample 10:1 | ~4s | Low | Bicubic interpolation |
| Smooth (5×5 median) | ~5s | Medium | Generic filter |
| Mesh creation (14M verts) | ~75s | 4 GB | Includes color application |
| Render (32 samples) | ~2m | Variable | GPU recommended |

---

## Version History

- **1.0**: Initial release with core transforms and Blender integration
- See CHANGELOG.md for detailed history
