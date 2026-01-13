---
title: API Reference
nav_order: 5
---

# terrain-maker API Reference

Comprehensive documentation of all classes and functions in the terrain-maker library.

**Auto-generated from source code docstrings.**

## Table of Contents

- [terrain.advanced_viz](#terrain-advanced_viz)
- [terrain.blender_integration](#terrain-blender_integration)
- [terrain.cache](#terrain-cache)
- [terrain.color_mapping](#terrain-color_mapping)
- [terrain.core](#terrain-core)
- [terrain.data_loading](#terrain-data_loading)
- [terrain.diagnostics](#terrain-diagnostics)
- [terrain.gridded_data](#terrain-gridded_data)
- [terrain.materials](#terrain-materials)
- [terrain.mesh_cache](#terrain-mesh_cache)
- [terrain.mesh_operations](#terrain-mesh_operations)
- [terrain.pipeline](#terrain-pipeline)
- [terrain.rendering](#terrain-rendering)
- [terrain.roads](#terrain-roads)
- [terrain.scene_setup](#terrain-scene_setup)
- [terrain.scoring](#terrain-scoring)
- [terrain.transforms](#terrain-transforms)
- [terrain.water](#terrain-water)

---

## terrain.advanced_viz

Advanced terrain visualization features.

This module provides specialized visualization capabilities for terrain data:
- Drive-time isochrone curves (3D transportation analysis)
- Slope calculation using Horn's method
- 3D legend generation for Blender scenes

Migrated from legacy helpers.py with improvements.

### Functions

#### `create_drive_time_curves(drive_time, terrain_obj, processed_dem, height_offset, bevel_depth)`

Create 3D glowing curves in Blender representing drive-time isochrones.

Generates 3D curves that follow the terrain surface with emission shaders
that glow on specific viewing angles. Great for transportation analysis
visualization.

Args:
    drive_time (geopandas.GeoDataFrame): Processed drive-time polygons
    terrain_obj: Blender terrain mesh object (for reference)
    processed_dem (np.ndarray): DEM data for coordinate centering
    height_offset (float): Height above terrain in Blender units (default: 1.0)
    bevel_depth (float): Thickness of the curve (default: 0.02)

Returns:
    list: List of created Blender curve objects

Notes:
    - Uses 'inferno' colormap for gradient coloring
    - Applies edge-emission shader for directional glow effect
    - Curves are positioned relative to DEM center

Examples:
    >>> curves = create_drive_time_curves(
    ...     drive_time, terrain_mesh, dem_data, height_offset=0.5
    ... )
    >>> print(f"Created {len(curves)} curves")

#### `create_values_legend(terrain_obj, values, mpp, colormap_name, n_samples, label, units, scale, position_offset)`

Create a 3D legend bar in the Blender scene.

Generates a vertical bar with color gradient and text labels showing
the value scale for the terrain visualization.

Args:
    terrain_obj: Blender terrain mesh object (for positioning reference)
    values (np.ndarray): Value array to create legend for
    mpp (float): Meters per pixel (default: 30)
    colormap_name (str): Matplotlib colormap name (default: 'mako_r')
    n_samples (int): Number of labels on legend (default: 10)
    label (str): Legend title (default: 'Value')
    units (str): Unit string (e.g., 'meters', 'mm') (default: '')
    scale (float): Legend bar scale factor (default: 0.2)
    position_offset (tuple): (x, y, z) offset from terrain (default: (5, 0, 0))

Returns:
    tuple: (legend_object, text_objects_list)

Examples:
    >>> legend_obj, labels = create_values_legend(
    ...     terrain_mesh, elevation_data,
    ...     label='Elevation', units='meters', n_samples=5
    ... )
    >>> print(f"Created legend with {len(labels)} labels")

#### `horn_slope(dem, window_size)`

Calculate slope using Horn's method with NaN handling.

Horn's method is a standard GIS technique for calculating terrain slope
using a 3x3 Sobel-like kernel. This implementation properly handles NaN
values through interpolation.

Args:
    dem (np.ndarray): Input DEM array (2D)
    window_size (int): Reserved for future use (currently fixed at 3)

Returns:
    np.ndarray: Slope magnitude array (same shape as input)

Examples:
    >>> dem = np.random.rand(100, 100) * 1000  # Random elevation
    >>> slopes = horn_slope(dem)
    >>> print(f"Slope range: {slopes.min():.2f} to {slopes.max():.2f}")

#### `load_drive_time_data(dem_data, utm_transform, meters_per_pixel, buffer_size, simplify_tolerance)`

Load and process drive-time polygon data for terrain visualization.

Loads GeoJSON drive-time isochrones, projects them to match DEM coordinates,
and smooths geometries for clean visualization.

Args:
    dem_data (np.ndarray): Processed DEM data array
    utm_transform (Affine): Affine transform from UTM to pixel coordinates
    meters_per_pixel (float): Spatial resolution in meters
    buffer_size (float): Buffer size for geometry smoothing (percentage)
    simplify_tolerance (float): Simplification tolerance (percentage)

Returns:
    geopandas.GeoDataFrame: Processed drive-time polygons in pixel coordinates

Notes:
    - Expects a file named '1_to_5_hr_drive_times.geojson' in current directory
    - Projects to UTM Zone 17N (EPSG:32617) for Detroit area
    - Applies smoothing via buffer operations

Examples:
    >>> drive_time = load_drive_time_data(
    ...     dem_data, utm_transform, mpp=30, buffer_size=10, simplify_tolerance=5
    ... )
    >>> print(f"Loaded {len(drive_time)} drive-time zones")

#### `transform_to_pixels(geom)`

Transform geometry from UTM coordinates to pixel coordinates.

Converts UTM coordinates to pixel coordinates using the affine transform.
Handles both Polygon and MultiPolygon geometries recursively.

Args:
    geom: Shapely geometry in UTM coordinates (Polygon or MultiPolygon)

Returns:
    Shapely geometry with coordinates transformed to pixel space

---

## terrain.blender_integration

Blender integration for terrain visualization.

This module contains Blender-specific code for creating and configuring
terrain meshes, materials, and rendering.

### Functions

#### `apply_road_mask(mesh_obj, road_mask, y_valid, x_valid, logger)`

Apply a road mask as a separate vertex color layer for material detection.

Creates a "RoadMask" vertex color layer where road vertices have R=1.0
and non-road vertices have R=0.0. This allows the material shader to
detect roads without changing the terrain colors.

Uses Blender's foreach_set for ~100x faster bulk operations.

Args:
    mesh_obj (bpy.types.Object): The Blender mesh object
    road_mask (np.ndarray): 2D boolean or float array (height, width) where >0.5 = road
    y_valid (np.ndarray): Y indices mapping vertices to grid positions
    x_valid (np.ndarray): X indices mapping vertices to grid positions
    logger (logging.Logger, optional): Logger for progress messages

#### `apply_vertex_colors(mesh_obj, vertex_colors, y_valid, x_valid, logger)`

Apply colors to an existing Blender mesh.

Accepts colors in either vertex-space (n_vertices, 3/4) or grid-space (height, width, 3/4).
When grid-space colors are provided with y_valid/x_valid indices, colors are extracted
for each vertex using those coordinates.

Uses Blender's foreach_set for ~100x faster bulk operations.

Args:
    mesh_obj (bpy.types.Object): The Blender mesh object to apply colors to
    vertex_colors (np.ndarray): Colors in one of two formats:
        - Vertex-space: shape (n_vertices, 3) or (n_vertices, 4)
        - Grid-space: shape (height, width, 3) or (height, width, 4)
    y_valid (np.ndarray, optional): Y indices for grid-space colors
    x_valid (np.ndarray, optional): X indices for grid-space colors
    logger (logging.Logger, optional): Logger for progress messages

#### `apply_vertex_positions(mesh_obj, new_positions, logger)`

Apply new 3D positions to mesh vertices.

Useful for applying smoothed vertex coordinates to an existing mesh,
e.g., after road smoothing or terrain filtering.

Args:
    mesh_obj: Blender mesh object to modify
    new_positions: Array of shape (n_vertices, 3) with new [x, y, z] positions
    logger: Optional logger for progress messages

Raises:
    ValueError: If new_positions shape doesn't match mesh vertex count

Example:
    >>> # Smooth road vertices and apply to mesh
    >>> from src.terrain.roads import smooth_road_vertices
    >>>
    >>> vertices = np.array([v.co[:] for v in mesh.data.vertices])
    >>> smoothed = smooth_road_vertices(vertices, road_mask, y_valid, x_valid)
    >>> apply_vertex_positions(mesh, smoothed)

#### `create_blender_mesh(vertices, faces, colors, y_valid, x_valid, name, logger)`

Create a Blender mesh object from vertices and faces.

Creates a new Blender mesh datablock, populates it with geometry data,
optionally applies vertex colors, and creates a material with colormap shader.

Args:
    vertices (np.ndarray): Array of (n, 3) vertex positions
    faces (list): List of tuples defining face connectivity
    colors (np.ndarray, optional): Array of RGB/RGBA colors (height, width, channels)
    y_valid (np.ndarray, optional): Array of y indices for vertex colors
    x_valid (np.ndarray, optional): Array of x indices for vertex colors
    name (str): Name for the mesh and object (default: "TerrainMesh")
    logger (logging.Logger, optional): Logger for progress messages

Returns:
    bpy.types.Object: The created terrain mesh object

Raises:
    RuntimeError: If Blender is not available or mesh creation fails

---

## terrain.cache

DEM caching module for efficient terrain visualization pipeline.

Implements .npz-based caching with hash validation to avoid reloading
and reprocessing expensive DEM merging operations.

### Classes

#### `DEMCache`

Manages caching of loaded and merged DEM data with hash validation.

The cache stores:
- DEM array as .npz file
- Metadata including file hash, timestamp, and file list

Attributes:
    cache_dir: Directory where cache files are stored
    enabled: Whether caching is enabled

**Methods:**

- `clear_cache(self, cache_name)` - Clear all cached files for a given cache name.

- `compute_source_hash(self, directory_path, pattern, recursive)` - Compute hash of source DEM files based on paths and modification times.

- `get_cache_path(self, source_hash, cache_name)` - Get the path for a cache file.

- `get_cache_stats(self)` - Get statistics about cached files.

- `get_metadata_path(self, source_hash, cache_name)` - Get the path for cache metadata file.

- `load_cache(self, source_hash, cache_name)` - Load cached DEM data.

- `save_cache(self, dem_array, transform, source_hash, cache_name)` - Save DEM array and transform to cache.


#### `PipelineCache`

Target-style caching for terrain processing pipelines.

Like a build system (Make, Bazel), this cache:
- Tracks targets with defined parameters and dependencies
- Computes cache keys that incorporate the FULL dependency chain
- Ensures downstream targets are invalidated when upstream changes
- Supports file inputs with mtime tracking

Example:
    cache = PipelineCache()
    cache.define_target("dem_loaded", params={"path": "/data"})
    cache.define_target("reprojected", params={"crs": "EPSG:32617"},
                       dependencies=["dem_loaded"])

    # First run: cache miss
    if cache.get_cached("reprojected") is None:
        data = expensive_operation()
        cache.save_target("reprojected", data)

    # Second run (same params): cache hit
    # If dem_loaded params change: cache miss (invalidated)

Attributes:
    cache_dir: Directory where cache files are stored
    enabled: Whether caching is enabled
    targets: Dict of target definitions {name: {params, dependencies, file_inputs}}

**Methods:**

- `clear_all(self)` - Clear all cache files.

- `clear_target(self, target_name)` - Clear cache files for a specific target.

- `compute_target_key(self, target_name)` - Compute cache key for a target, incorporating all upstream dependencies.

- `define_target(self, name, params, dependencies, file_inputs)` - Define a pipeline target with its parameters and dependencies.

- `get_cached(self, target_name, return_metadata)` - Get cached target output if available.

- `save_target(self, target_name, data, metadata)` - Save target output to cache.


#### `TransformCache`

Cache for transform pipeline results with dependency tracking.

Tracks chains of transforms (reproject -> smooth -> water_detect) and
computes cache keys that incorporate the full dependency chain, ensuring
downstream caches are invalidated when upstream params change.

Attributes:
    cache_dir: Directory where cache files are stored
    enabled: Whether caching is enabled
    dependencies: Graph of transform dependencies
    transforms: Registered transforms with their parameters

**Methods:**

- `compute_transform_hash(self, upstream_hash, transform_name, params)` - Compute cache key from upstream hash and transform parameters.

- `get_cache_path(self, cache_key, transform_name)` - Get path for cache file.

- `get_dependency_chain(self, transform_name)` - Get full dependency chain for a transform.

- `get_full_cache_key(self, transform_name, source_hash)` - Compute full cache key incorporating dependency chain.

- `get_metadata_path(self, cache_key, transform_name)` - Get path for metadata file.

- `invalidate_downstream(self, transform_name)` - Invalidate all caches downstream of a transform.

- `load_transform(self, cache_key, transform_name)` - Load transform result from cache.

- `register_dependency(self, child, upstream)` - Register a dependency between transforms.

- `register_transform(self, name, upstream, params)` - Register a transform with its parameters.

- `save_transform(self, cache_key, data, transform_name, metadata)` - Save transform result to cache.


### Functions

#### `clear_all(self)`

Clear all cache files.

Returns:
    Number of files deleted

#### `clear_cache(self, cache_name)`

Clear all cached files for a given cache name.

Args:
    cache_name: Name of cache item to clear

Returns:
    Number of files deleted

#### `clear_target(self, target_name)`

Clear cache files for a specific target.

Args:
    target_name: Name of target to clear

Returns:
    Number of files deleted

#### `compute_source_hash(self, directory_path, pattern, recursive)`

Compute hash of source DEM files based on paths and modification times.

This ensures the cache is invalidated if:
- Files are added/removed
- Files are modified
- Directory path changes

Args:
    directory_path: Path to DEM directory
    pattern: File pattern (e.g., "*.hgt")
    recursive: Whether search is recursive

Returns:
    SHA256 hash of source file metadata

#### `compute_target_key(self, target_name)`

Compute cache key for a target, incorporating all upstream dependencies.

The key is a SHA256 hash that changes if:
- Target's own params change
- Any upstream target's params change
- Any file inputs are modified

Args:
    target_name: Name of the target

Returns:
    64-character hex SHA256 hash, or empty string if target undefined

#### `compute_transform_hash(self, upstream_hash, transform_name, params)`

Compute cache key from upstream hash and transform parameters.

The key incorporates:
- Upstream cache key (propagating the full dependency chain)
- Transform name
- All transform parameters (sorted for determinism)

Args:
    upstream_hash: Hash of upstream data/transform
    transform_name: Name of this transform (e.g., "reproject", "smooth")
    params: Transform parameters dict

Returns:
    SHA256 hash string (64 chars)

#### `define_target(self, name, params, dependencies, file_inputs)`

Define a pipeline target with its parameters and dependencies.

Args:
    name: Unique name for this target
    params: Parameters that affect the target's output
    dependencies: List of upstream target names this depends on
    file_inputs: List of file paths whose mtimes should be tracked

Raises:
    ValueError: If adding this target would create a circular dependency

#### `get_cache_path(self, cache_key, transform_name)`

Get path for cache file.

Args:
    cache_key: Cache key hash
    transform_name: Name of transform

Returns:
    Path to cache .npz file

#### `get_cache_stats(self)`

Get statistics about cached files.

Returns:
    Dictionary with cache statistics

#### `get_cached(self, target_name, return_metadata)`

Get cached target output if available.

Args:
    target_name: Name of the target
    return_metadata: If True, return (data, metadata) tuple

Returns:
    Cached data (array or dict of arrays), or None if cache miss.
    If return_metadata=True, returns (data, metadata) or (None, None)

#### `get_dependency_chain(self, transform_name)`

Get full dependency chain for a transform.

Args:
    transform_name: Name of transform

Returns:
    List of transform names from root to target

#### `get_full_cache_key(self, transform_name, source_hash)`

Compute full cache key incorporating dependency chain.

Args:
    transform_name: Target transform name
    source_hash: Hash of original source data

Returns:
    Cache key hash

#### `get_metadata_path(self, cache_key, transform_name)`

Get path for metadata file.

Args:
    cache_key: Cache key hash
    transform_name: Name of transform

Returns:
    Path to metadata .json file

#### `invalidate_downstream(self, transform_name)`

Invalidate all caches downstream of a transform.

Args:
    transform_name: Name of transform whose downstream should be invalidated

Returns:
    Number of cache files deleted

#### `load_cache(self, source_hash, cache_name)`

Load cached DEM data.

Args:
    source_hash: Hash of source files
    cache_name: Name of cache item (default: "dem")

Returns:
    Tuple of (dem_array, transform) or None if cache doesn't exist

#### `load_transform(self, cache_key, transform_name)`

Load transform result from cache.

Args:
    cache_key: Cache key hash
    transform_name: Name of transform

Returns:
    Cached array or None if cache miss/disabled

#### `register_dependency(self, child, upstream)`

Register a dependency between transforms.

Args:
    child: Name of dependent transform
    upstream: Name of upstream transform it depends on

#### `register_transform(self, name, upstream, params)`

Register a transform with its parameters.

Args:
    name: Transform name
    upstream: Name of upstream dependency
    params: Transform parameters

#### `save_cache(self, dem_array, transform, source_hash, cache_name)`

Save DEM array and transform to cache.

Args:
    dem_array: Merged DEM array
    transform: Affine transform
    source_hash: Hash of source files
    cache_name: Name of cache item (default: "dem")

Returns:
    Tuple of (cache_file_path, metadata_file_path)

#### `save_target(self, target_name, data, metadata)`

Save target output to cache.

Args:
    target_name: Name of the target
    data: numpy array, or dict of arrays to cache
    metadata: Optional additional metadata (can include Affine transforms)

Returns:
    Path to cache file, or None if disabled

#### `save_transform(self, cache_key, data, transform_name, metadata)`

Save transform result to cache.

Args:
    cache_key: Cache key hash
    data: Transform result array
    transform_name: Name of transform
    metadata: Optional additional metadata

Returns:
    Tuple of (cache_path, metadata_path) or (None, None) if disabled

#### `serialize_value(v)`

Convert a parameter value to a deterministic string for hashing.

Handles numpy arrays specially by including shape, dtype, and content hash.
Other values are converted to their string representation.

Args:
    v: Parameter value (can be ndarray, scalar, string, etc.)

Returns:
    Deterministic string representation of the value

---

## terrain.color_mapping

Color mapping functions for terrain visualization.

This module contains functions for mapping elevation and slope data to colors
using matplotlib colormaps.

### Functions

#### `elevation_colormap(dem_data, cmap_name, min_elev, max_elev, gamma)`

Create a colormap based on elevation values.

Maps elevation data to colors using a matplotlib colormap.
Low elevations map to the start of the colormap, high elevations to the end.

Args:
    dem_data: 2D numpy array of elevation values
    cmap_name: Matplotlib colormap name (default: 'viridis')
    min_elev: Minimum elevation for normalization (default: use data min)
    max_elev: Maximum elevation for normalization (default: use data max)
    gamma: Gamma correction exponent (default: 1.0 = no correction).
           Values < 1.0 brighten midtones, values > 1.0 darken midtones.
           Common values: 0.5 = brighten, 2.2 = darken (sRGB gamma)

Returns:
    Array of RGB colors with shape (height, width, 3) as uint8

#### `slope_colormap(slopes, cmap_name, min_slope, max_slope)`

Create a simple colormap based solely on terrain slopes.

Args:
    slopes: Array of slope values in degrees
    cmap_name: Matplotlib colormap name (default: 'terrain')
    min_slope: Minimum slope value for normalization (default: 0)
    max_slope: Maximum slope value for normalization (default: 45)

Returns:
    Array of RGBA colors with shape (*slopes.shape, 4)

---

## terrain.core

### Classes

#### `Terrain`

Core class for managing Digital Elevation Model (DEM) data and terrain operations.

Handles loading, transforming, and visualizing terrain data from raster sources.
Supports coordinate reprojection, downsampling, color mapping, and 3D mesh generation
for Blender visualization. Uses efficient caching to avoid recomputation of transforms.

Attributes:
    dem_shape (tuple): Shape of DEM array as (height, width).
    dem_transform (rasterio.Affine): Affine transform for geographic coordinates.
    data_layers (dict): Dictionary of data layers (DEM, overlays, derived data).
    transforms (list): List of transform functions to apply.
    vertices (np.ndarray): Vertex positions for generated mesh.
    vertex_colors (np.ndarray): RGBA colors for mesh vertices.

Examples:
    >>> dem_data = np.random.rand(100, 100) * 1000
    >>> transform = rasterio.Affine.identity()
    >>> terrain = Terrain(dem_data, transform, dem_crs='EPSG:4326')
    >>> terrain.apply_transforms()
    >>> mesh = terrain.create_mesh(scale_factor=100.0)

**Methods:**

- `add_data_layer(self, name, data, transform, crs, target_crs, target_layer, same_extent_as, resampling)` - Add a data layer, optionally reprojecting to match another layer.

- `add_transform(self, transform_func)` - Add a transform function to the processing pipeline.

- `apply_transforms(self, cache)` - Apply all transforms to all data layers with optional caching.

- `compute_colors(self, water_mask)` - Compute colors using color_func and optionally mask_func.

- `compute_data_layer(self, name, source_layer, compute_func, transformed, cache_key)` - Compute a new data layer from an existing one using a transformation function.

- `compute_proximity_mask(self, lons, lats, radius_meters, input_crs, cluster_threshold_meters)` - Create boolean mask for vertices within radius of geographic points.

- `configure_for_target_vertices(self, target_vertices, method)` - Configure downsampling to achieve approximately target_vertices.

- `create_mesh(self, base_depth, boundary_extension, scale_factor, height_scale, center_model, verbose, detect_water, water_slope_threshold, water_mask)` - Create a Blender mesh from transformed DEM data with both performance and control.

- `detect_water(self, slope_threshold, fill_holes)` - Detect water bodies on the transformed (downsampled) DEM.

- `detect_water_highres(self, slope_threshold, fill_holes, scale_factor)` - Detect water bodies on high-resolution DEM before downsampling.

- `geo_to_mesh_coords(self, lon, lat, elevation_offset, input_crs)` - Convert geographic coordinates to Blender mesh coordinates.

- `get_bbox_wgs84(self, layer)` - Get bounding box in WGS84 coordinates (EPSG:4326).

- `set_blended_color_mapping(self, base_colormap, base_source_layers, overlay_colormap, overlay_source_layers, overlay_mask, base_color_kwargs, overlay_color_kwargs)` - Apply two different colormaps based on a spatial mask (hard transition).

- `set_color_mapping(self, color_func, source_layers, color_kwargs, mask_func, mask_layers, mask_kwargs, mask_threshold)` - Set up how to map data layers to colors (RGB) and optionally a mask/alpha channel.

- `set_multi_color_mapping(self, base_colormap, base_source_layers, overlays, base_color_kwargs)` - Apply multiple data layers as overlays with different colormaps and priority.

- `visualize_dem(self, layer, use_transformed, title, cmap, percentile_clip, clip_percentiles, max_pixels, show_histogram)` - Create diagnostic visualization of any terrain data layer.


#### `TerrainCache`

Cache manager for terrain data processing results.

Handles persistent storage and retrieval of transformed terrain data layers
as GeoTIFF files with geographic metadata. Supports loading and saving with
coordinate reference system (CRS) and custom metadata.

Attributes:
    cache_dir (Path): Root directory for cached GeoTIFF files.
    logger (logging.Logger): Logger instance for cache operations.

Examples:
    >>> cache = TerrainCache('my_cache_dir')
    >>> cache.save('dem_transformed', dem_array, transform, 'EPSG:32617')
    >>> data, transform, crs = cache.load('dem_transformed')

**Methods:**

- `exists(self, target_name)` - Check if target exists

- `get_target_path(self, target_name)` - Get path for a specific target

- `load(self, target_name)` - Load GeoTIFF and metadata if it exists

- `save(self, target_name, data, transform, crs, metadata)` - Save data as GeoTIFF with CRS and metadata


### Functions

#### `add_data_layer(self, name, data, transform, crs, target_crs, target_layer, same_extent_as, resampling)`

Add a data layer, optionally reprojecting to match another layer.

Stores data with geographic metadata (CRS and transform). Can automatically
reproject and resample to match an existing layer's grid for multi-layer analysis.

Args:
    name (str): Unique name for this data layer (e.g., 'dem', 'elevation', 'slope').
    data (np.ndarray): 2D array of data values, shape (height, width).
    transform (rasterio.Affine, optional): Affine transform mapping pixel to geographic
        coords. Required unless same_extent_as is specified.
    crs (str, optional): Coordinate reference system in EPSG format (e.g., 'EPSG:4326').
        Required unless same_extent_as is specified (inherits from reference layer).
    target_crs (str, optional): Target CRS to reproject to. If None and target_layer
        specified, uses target layer's CRS. If None and no target, uses input crs.
    target_layer (str, optional): Name of existing layer to match grid and CRS.
        If specified, data is automatically reprojected and resampled to align.
    same_extent_as (str, optional): Name of existing layer whose geographic extent
        this data covers. When specified, transform and CRS are automatically
        calculated from the reference layer's bounds and the data's shape.
        This is useful when score grids or overlays cover the same area as the
        DEM but at different resolutions. Implies target_layer if not specified.
    resampling (rasterio.enums.Resampling): Resampling method for reprojection
        (default: Resampling.bilinear). See rasterio docs for options.

Returns:
    None: Modifies internal data_layers dictionary.

Raises:
    KeyError: If target_layer or same_extent_as layer doesn't exist.
    ValueError: If neither transform nor same_extent_as is provided.

Examples:
    >>> # Add elevation data with native CRS
    >>> terrain.add_data_layer('dem', dem_array, transform, 'EPSG:4326')

    >>> # Add overlay data, reproject to match DEM
    >>> terrain.add_data_layer('landcover', lc_array, lc_transform, 'EPSG:3857',
    ...                        target_layer='dem')

    >>> # Add score data that covers the same extent as DEM (automatic transform)
    >>> terrain.add_data_layer('score', score_array, same_extent_as='dem')

    >>> # Use nearest-neighbor for categorical data
    >>> terrain.add_data_layer('zones', zone_array, zone_transform, 'EPSG:4326',
    ...                        target_layer='dem', resampling=Resampling.nearest)

#### `add_transform(self, transform_func)`

Add a transform function to the processing pipeline.

Args:
    transform_func (callable): Function that transforms DEM data. Should accept
        (dem_array: np.ndarray) and return transformed np.ndarray.

Returns:
    None: Modifies internal transforms list in place.

Examples:
    >>> terrain.add_transform(lambda dem: gaussian_filter(dem, sigma=2))
    >>> terrain.apply_transforms()

#### `apply_colormap_material(material)`

Create a simple material setup for terrain visualization using vertex colors.
Uses emission to guarantee colors are visible regardless of lighting.

Args:
    material: Blender material to configure

#### `apply_transforms(self, cache)`

Apply all transforms to all data layers with optional caching.

Processes each data layer through the transform pipeline. Results are cached
to avoid recomputation. Transforms are applied in order.

Args:
    cache (bool): Whether to cache results (default: False).

Returns:
    None: Updates internal data_layers with 'transformed_data' for each layer.

Examples:
    >>> terrain.add_transform(flip_raster(axis='horizontal'))
    >>> terrain.apply_transforms(cache=True)
    >>> dem_data = terrain.data_layers['dem']['transformed_data']

#### `apply_water_shader(material, water_color)`

Apply water shader to material, coloring water areas based on vertex alpha channel.
Uses alpha channel to mix between water color and elevation colors.
Water pixels (alpha=1.0) render as water color; land pixels (alpha=0.0) show elevation colors.

Args:
    material: Blender material to configure
    water_color: RGB tuple for water (default: University of Michigan blue #00274C)

#### `calculate_target_vertices(width, height, multiplier)`

Calculate target vertex count for optimal mesh density at a given render resolution.

This helper calculates an appropriate number of vertices for terrain meshes
based on the intended output resolution. Using ~2 vertices per output pixel
ensures good detail without excessive geometry.

Args:
    width: Render width in pixels
    height: Render height in pixels
    multiplier: Vertices per pixel (default: 2.0).
               Higher values = more detail but slower renders.
               - 1.0: Minimum detail (1 vertex per pixel)
               - 2.0: Good balance for most renders (recommended)
               - 3.0+: High detail for print or zoomed views

Returns:
    int: Target vertex count for terrain mesh creation

Example:
    ```python
    # For 1920x1080 render
    target = calculate_target_vertices(1920, 1080)  # ~4.1M vertices

    # For print quality (3000x2400 @ 300 DPI)
    target = calculate_target_vertices(3000, 2400, multiplier=2.5)  # ~18M vertices

    # Use with Terrain
    terrain.configure_for_target_vertices(target_vertices=target)
    ```

#### `clear_scene()`

Clear all objects from the Blender scene.

Resets the scene to factory settings (empty scene) and removes all default
objects. Useful before importing terrain meshes to ensure a clean workspace.

Raises:
    RuntimeError: If Blender module (bpy) is not available.

#### `compute_colors(self, water_mask)`

Compute colors using color_func and optionally mask_func.

Supports three modes:
- Standard: Single colormap applied to all vertices
- Blended: Two colormaps blended based on proximity mask
- Multi-overlay: Multiple overlays with different colormaps and priority

Args:
    water_mask (np.ndarray, optional): Boolean water mask in grid space (height × width).
        For blended mode, water pixels will be colored blue in the final vertex colors.
        For standard and multi-overlay modes, water detection is handled in create_mesh().

Returns:
    np.ndarray: RGBA color array.

#### `compute_data_layer(self, name, source_layer, compute_func, transformed, cache_key)`

Compute a new data layer from an existing one using a transformation function.

Allows creating derived layers (e.g., slope, aspect, hillshade) from existing data.
Results are stored as new layer and optionally cached.

Args:
    name (str): Name for the computed layer.
    source_layer (str): Name of existing source layer to compute from.
    compute_func (Callable): Function that accepts source array (np.ndarray)
        and returns computed array (np.ndarray). Can return same or different shape.
    transformed (bool): If True, use already-transformed source data; if False,
        use original source data (default: False).
    cache_key (str, optional): Custom cache identifier. If None, auto-generated
        from layer name and function name.

Returns:
    np.ndarray: The computed layer data array.

Raises:
    KeyError: If source_layer doesn't exist.
    ValueError: If transformed=True but source hasn't been transformed.

Examples:
    >>> # Compute slope from DEM using scipy
    >>> from scipy.ndimage import sobel
    >>> slope = terrain.compute_data_layer(
    ...     'slope', 'dem',
    ...     lambda dem: np.sqrt(sobel(dem, axis=0)**2 + sobel(dem, axis=1)**2)
    ... )

    >>> # Compute hill-shade visualization
    >>> from scipy.ndimage import gaussian_filter
    >>> hillshade = terrain.compute_data_layer(
    ...     'hillshade', 'dem',
    ...     lambda dem: np.clip(gaussian_filter(dem, 2) * 0.5, 0, 1)
    ... )

    >>> # Compute from transformed (downsampled) data
    >>> downsampled_slope = terrain.compute_data_layer(
    ...     'slope_downsampled', 'dem',
    ...     lambda dem: np.gradient(dem)[0],
    ...     transformed=True
    ... )

#### `compute_proximity_mask(self, lons, lats, radius_meters, input_crs, cluster_threshold_meters)`

Create boolean mask for vertices within radius of geographic points.

Uses KDTree for efficient spatial queries to identify mesh vertices
near specified geographic locations (e.g., parks, POIs). Optionally
clusters nearby points first to create unified proximity zones.

Args:
    lons: Array of longitudes for points of interest.
    lats: Array of latitudes for points of interest (must match lons shape).
    radius_meters: Radius in meters around each point/cluster to include.
    input_crs: CRS of input coordinates (default: "EPSG:4326" for WGS84 lon/lat).
    cluster_threshold_meters: Optional distance threshold for clustering nearby
        points using DBSCAN. Points within this distance are merged into single
        zones. If None, each point gets its own zone. Useful for merging nearby
        parks into continuous zones.

Returns:
    Boolean array of shape (num_vertices,) where True indicates vertex is
    within radius of at least one point/cluster.

Raises:
    RuntimeError: If create_mesh() has not been called yet.
    ValueError: If lons and lats have different shapes.

Example:
    >>> # Create zones around parks
    >>> park_lons = np.array([-83.1, -83.2, -83.15])
    >>> park_lats = np.array([42.4, 42.5, 42.45])
    >>> mask = terrain.compute_proximity_mask(
    ...     park_lons, park_lats,
    ...     radius_meters=1000,
    ...     cluster_threshold_meters=200  # Merge parks within 200m
    ... )
    >>> # mask is True for vertices within 1km of park clusters

#### `configure_for_target_vertices(self, target_vertices, method)`

Configure downsampling to achieve approximately target_vertices.

This method calculates the appropriate zoom_factor to achieve a desired
vertex count for mesh generation. It provides a more intuitive API than
manually calculating zoom_factor from the original DEM shape.

Args:
    target_vertices: Desired vertex count for final mesh (e.g., 500_000)
    method: Downsampling method (default: "average")
        - "average": Area averaging - best for DEMs, no overshoot
        - "lanczos": Lanczos resampling - sharp, minimal aliasing
        - "cubic": Cubic spline interpolation
        - "bilinear": Bilinear interpolation - safe fallback

Returns:
    Calculated zoom_factor that was added to transforms

Raises:
    ValueError: If target_vertices is invalid

Example:
    terrain = Terrain(dem, transform)
    zoom = terrain.configure_for_target_vertices(500_000, method="average")
    print(f"Calculated zoom_factor: {zoom:.4f}")
    terrain.apply_transforms()
    mesh = terrain.create_mesh(scale_factor=400.0)

#### `create_background_plane(terrain_obj, depth, scale_factor, material_params)`

Create a large emissive plane beneath the terrain for background illumination.

Args:
    terrain_obj: The terrain Blender object used for size reference
    depth: Z-coordinate for the plane position
    scale_factor: Scale multiplier for plane size relative to terrain
    material_params: Optional dict to override default material parameters

Returns:
    bpy.types.Object: The created background plane object

Raises:
    ValueError: If terrain_obj is None or has invalid bounds
    RuntimeError: If mesh or material creation fails

#### `create_mesh(self, base_depth, boundary_extension, scale_factor, height_scale, center_model, verbose, detect_water, water_slope_threshold, water_mask)`

Create a Blender mesh from transformed DEM data with both performance and control.

Generates vertices from DEM elevation values and faces for connectivity. Optionally
creates boundary faces to close the mesh into a solid. Supports coordinate scaling
and elevation scaling for visualization. Can optionally detect and apply water bodies
to vertex alpha channel for water rendering.

Args:
    base_depth (float): Z-coordinate for the bottom of the terrain model (default: -0.2).
        Used when boundary_extension=True to create side faces.
    boundary_extension (bool): Whether to create side faces around the terrain boundary
        to close the mesh (default: True). If False, creates open terrain surface.
    scale_factor (float): Horizontal scale divisor for x/y coordinates (default: 100.0).
        Higher values produce smaller meshes. E.g., 100 means 100 DEM units = 1 Blender unit.
    height_scale (float): Multiplier for elevation values (default: 1.0). Vertically
        exaggerates or reduces terrain features. Values > 1 exaggerate, < 1 flatten.
    center_model (bool): Whether to center the model at origin (default: True).
        Centers XY coordinates but preserves absolute Z elevation values.
    verbose (bool): Whether to log detailed progress information (default: True).
    detect_water (bool): Whether to detect water bodies and apply to alpha channel
        (default: False). Uses slope-based detection on transformed DEM.
    water_slope_threshold (float): Maximum slope magnitude to classify as water
        (default: 0.5). Only used if detect_water=True and water_mask is None.
    water_mask (np.ndarray): Pre-computed boolean water mask (True=water, False=land).
        If provided, this mask is used instead of computing water detection.
        Allows water detection on unscaled DEM before elevation scaling transforms.

Returns:
    bpy.types.Object | None: The created terrain mesh object, or None if creation failed.

Raises:
    ValueError: If transformed DEM layer is not available (apply_transforms() not called).

#### `detect_water(self, slope_threshold, fill_holes)`

Detect water bodies on the transformed (downsampled) DEM.

This is a fast, simple water detection method that works on the already-
transformed DEM. It automatically handles elevation scale factor unscaling.

For higher accuracy at the cost of speed, use detect_water_highres() instead,
which detects water on the full-resolution DEM before downsampling.

Args:
    slope_threshold: Maximum slope magnitude to classify as water (default: 0.01).
                    Water bodies have nearly zero slope.
    fill_holes: Whether to apply morphological hole filling (default: True)

Returns:
    np.ndarray: Boolean water mask matching transformed_data shape

Raises:
    ValueError: If transforms haven't been applied yet

Example:
    ```python
    terrain = Terrain(dem, transform)
    terrain.apply_transforms()
    water_mask = terrain.detect_water()
    terrain.compute_colors(water_mask=water_mask)
    ```

#### `detect_water_highres(self, slope_threshold, fill_holes, scale_factor)`

Detect water bodies on high-resolution DEM before downsampling.

This method properly handles water detection by:
1. Applying all NON-downsampling transforms to DEM (reproject, flip, etc.)
2. Detecting water on the high-resolution transformed DEM
3. Downsampling the water mask to match the final terrain resolution

This prevents false water detection that occurs when detecting on downsampled DEMs.

Args:
    slope_threshold: Maximum slope magnitude to classify as water (default: 0.01)
    fill_holes: Whether to apply morphological hole filling (default: True)
    scale_factor: Elevation scale factor to unscale before detection (default: 0.0001)

Returns:
    np.ndarray: Boolean water mask matching transformed_data shape

Raises:
    ValueError: If transforms haven't been applied yet
    ImportError: If water detection module is not available

Example:
    ```python
    terrain = Terrain(dem, transform, dem_crs="EPSG:4326")
    terrain.add_transform(reproject_raster(src_crs="EPSG:4326", dst_crs="EPSG:32617"))
    terrain.add_transform(flip_raster(axis="horizontal"))
    terrain.add_transform(scale_elevation(scale_factor=0.0001))
    terrain.configure_for_target_vertices(target_vertices=1_000_000)
    terrain.apply_transforms()  # Includes downsampling

    # Detect water on high-res BEFORE it was downsampled
    water_mask = terrain.detect_water_highres(slope_threshold=0.01)
    # water_mask shape matches downsampled terrain
    ```

Note:
    This method requires that:
    - Transforms have been applied (apply_transforms() called)
    - The DEM layer exists in data_layers
    - Water detection module (src.terrain.water) is available

#### `downsample_raster(zoom_factor, method, nodata_value)`

Create a raster downsampling transform function with specified parameters.

Args:
    zoom_factor: Scaling factor for downsampling (default: 0.1)
    method: Downsampling method (default: "average")
        - "average": Area averaging - best for DEMs, no overshoot
        - "lanczos": Lanczos resampling - sharp, minimal aliasing
        - "cubic": Cubic spline interpolation
        - "bilinear": Bilinear interpolation - safe fallback
    nodata_value: Value to treat as no data (default: np.nan)

Returns:
    function: A transform function that downsamples raster data

#### `elevation_colormap(dem_data, cmap_name, min_elev, max_elev, gamma)`

Create a colormap based on elevation values.

Maps elevation data to colors using a matplotlib colormap.
Low elevations map to the start of the colormap, high elevations to the end.

Args:
    dem_data: 2D numpy array of elevation values
    cmap_name: Matplotlib colormap name (default: 'viridis')
    min_elev: Minimum elevation for normalization (default: use data min)
    max_elev: Maximum elevation for normalization (default: use data max)
    gamma: Gamma correction exponent (default: 1.0 = no correction).
           Values < 1.0 brighten midtones, values > 1.0 darken midtones.

Returns:
    Array of RGB colors with shape (height, width, 3) as uint8

#### `exists(self, target_name)`

Check if target exists

#### `feature_preserving_smooth(sigma_spatial, sigma_intensity, nodata_value)`

Create a feature-preserving smoothing transform using bilateral filtering.

Removes high-frequency noise while preserving ridges, valleys, and drainage patterns.
Uses bilateral filtering: spatial Gaussian weighted by intensity similarity.

Args:
    sigma_spatial: Spatial smoothing extent in pixels (default: 3.0).
        Larger = more smoothing. Typical range: 1-10 pixels.
    sigma_intensity: Intensity similarity threshold in elevation units.
        Larger = more smoothing across elevation differences.
        If None, auto-calculated as 5% of elevation range.
    nodata_value: Value to treat as no data (default: np.nan)

Returns:
    function: Transform function compatible with terrain.add_transform()

#### `flip_raster(axis)`

Create a transform function that mirrors (flips) the DEM data.
If axis='horizontal', it flips top ↔ bottom.
(In terms of rows, row=0 becomes row=(height-1).)

If axis='vertical', you could do left ↔ right (np.fliplr).

#### `geo_to_mesh_coords(self, lon, lat, elevation_offset, input_crs)`

Convert geographic coordinates to Blender mesh coordinates.

Transforms lon/lat points through the same pipeline used for the terrain mesh,
producing coordinates that align with the rendered terrain. Useful for placing
markers, labels, or other objects at geographic locations on the terrain.

Args:
    lon: Longitude(s) or X coordinate(s). Can be a single float or array.
    lat: Latitude(s) or Y coordinate(s). Can be a single float or array
        (must match lon shape).
    elevation_offset: Height above terrain surface in mesh units (default: 0.0).
        Positive values place points above the terrain.
    input_crs: CRS of input coordinates (default: "EPSG:4326" for WGS84 lon/lat).
        Will be reprojected to match the transformed DEM's CRS if different.

Returns:
    Tuple of (x, y, z) arrays in Blender mesh coordinates. If single values
    were passed for lon/lat, returns single float values.

Raises:
    RuntimeError: If create_mesh() has not been called yet (model_params not set).
    ValueError: If the DEM layer has not been transformed yet.

Example:
    >>> terrain = Terrain(dem, transform)
    >>> terrain.add_transform(reproject_to_utm("EPSG:4326", "EPSG:32617"))
    >>> terrain.apply_transforms()
    >>> mesh = terrain.create_mesh()
    >>> # Get mesh coords for a park at (-83.1, 42.4) in WGS84
    >>> x, y, z = terrain.geo_to_mesh_coords(-83.1, 42.4, elevation_offset=0.1)
    >>> # Create marker at (x, y, z) in Blender

#### `get_bbox_wgs84(self, layer)`

Get bounding box in WGS84 coordinates (EPSG:4326).

Returns bbox in standard format used by OSM, web mapping APIs, etc.
Handles reprojection from any source CRS back to WGS84.

Args:
    layer: Layer name to get bbox for (default: "dem")

Returns:
    Tuple of (south, west, north, east) in WGS84 degrees

Raises:
    KeyError: If specified layer doesn't exist

Example:
    >>> terrain = Terrain(dem_data, transform, dem_crs="EPSG:32617")  # UTM
    >>> south, west, north, east = terrain.get_bbox_wgs84()
    >>> print(f"Bounds: {south:.4f}°N to {north:.4f}°N, {west:.4f}°E to {east:.4f}°E")

#### `get_render_settings_report()`

Query Blender for the actual render settings used.

Returns a dictionary of all render-relevant settings, useful for
debugging, reproducibility, and verification.

Returns:
    dict: Dictionary containing all render settings from Blender

#### `get_target_path(self, target_name)`

Get path for a specific target

#### `load(self, target_name)`

Load GeoTIFF and metadata if it exists

#### `load_dem_files(directory_path, pattern, recursive)`

Load and merge DEM files from a directory into a single elevation dataset.
Supports any raster format readable by rasterio (HGT, GeoTIFF, etc.).

Args:
    directory_path: Path to directory containing DEM files
    pattern: File pattern to match (default: "*.hgt")
    recursive: Whether to search subdirectories recursively (default: False)

Returns:
    tuple: (merged_dem, transform) where:
        - merged_dem: numpy array containing the merged elevation data
        - transform: affine transform mapping pixel to geographic coordinates

Raises:
    ValueError: If no valid DEM files are found or directory doesn't exist
    OSError: If directory access fails or file reading fails
    rasterio.errors.RasterioIOError: If there are issues reading the DEM files

#### `load_filtered_hgt_files(dem_dir, min_latitude, max_latitude, min_longitude, max_longitude, bbox, pattern)`

Load SRTM HGT files filtered by latitude/longitude range or bounding box.

Filters files before loading to reduce memory usage for large DEM directories.

Args:
    dem_dir: Directory containing HGT files
    min_latitude: Southern bound (e.g., 41 for N41)
    max_latitude: Northern bound (e.g., 47 for N47)
    min_longitude: Western bound (e.g., -90 for W090)
    max_longitude: Eastern bound (e.g., -82 for W082)
    bbox: Bounding box as (west, south, east, north). Overrides individual params.
    pattern: File pattern (default: "*.hgt")

Returns:
    Tuple of (merged_dem, transform)

Example:
    >>> # Load Detroit area using bbox
    >>> dem, transform = load_filtered_hgt_files(
    ...     "data/dem/detroit",
    ...     bbox=(-84, 41, -82, 43)
    ... )

#### `position_camera_relative(mesh_obj, direction, distance, elevation, look_at, camera_type, sun_angle, sun_energy, sun_azimuth, sun_elevation, focal_length, ortho_scale)`

Position camera relative to mesh using intuitive cardinal directions.

Simplifies camera positioning by using natural directions (north, south, etc.)
instead of absolute Blender coordinates. The camera is automatically positioned
relative to the mesh bounds and rotated to point at the mesh center.

Args:
    mesh_obj: Blender mesh object to position camera relative to
    direction: Cardinal direction - one of:
        'north', 'south', 'east', 'west' (horizontal directions)
        'northeast', 'northwest', 'southeast', 'southwest' (diagonals)
        'above' (directly overhead)
        Default: 'south'
    distance: Distance multiplier relative to mesh diagonal
        (e.g., 1.5 means 1.5x mesh_diagonal away). Default: 1.5
    elevation: Height as fraction of mesh diagonal added to Z position
        (0.0 = ground level, 1.0 = mesh_diagonal above ground). Default: 0.5
    look_at: Where camera points - 'center' to point at mesh center,
        or tuple (x, y, z) for custom target. Default: 'center'
    camera_type: 'ORTHO' (orthographic) or 'PERSP' (perspective). Default: 'ORTHO'
    sun_angle: Angle of sun light in degrees. Default: 0 (no light)
    sun_energy: Intensity of sun light. Default: 0 (no light created unless > 0)
    sun_azimuth: Direction sun comes FROM in degrees (0=N, 90=E, 180=S, 270=W).
    sun_elevation: Sun angle above horizon in degrees (0=horizon, 90=overhead).
    focal_length: Camera focal length in mm (perspective cameras only). Default: 50
    ortho_scale: Multiplier for orthographic camera scale relative to mesh diagonal.
        Higher values zoom out (show more area), lower values zoom in.
        Only affects orthographic cameras. Default: 1.2

Returns:
    Camera object

Raises:
    ValueError: If direction is not recognized or camera_type is invalid

#### `print_render_settings_report(log)`

Print a formatted report of all Blender render settings.

Queries Blender for actual settings and prints them in a readable format.
Useful for debugging and ensuring settings are correctly applied.

Args:
    log: Logger to use (defaults to module logger)

#### `render_scene_to_file(output_path, width, height, file_format, color_mode, compression, save_blend_file, show_progress, max_retries, retry_delay)`

Render the current Blender scene to file.

Includes automatic retry logic for GPU memory errors. If rendering fails
due to CUDA/GPU memory exhaustion, the function will wait and retry up to
max_retries times before giving up.

Args:
    output_path (str or Path): Path where output file will be saved
    width (int): Render width in pixels (default: 1920)
    height (int): Render height in pixels (default: 1440)
    file_format (str): Output format 'PNG', 'JPEG', etc. (default: 'PNG')
    color_mode (str): 'RGBA' or 'RGB' (default: 'RGBA')
    compression (int): PNG compression level 0-100 (default: 90)
    save_blend_file (bool): Also save .blend project file (default: True)
    show_progress (bool): Show render progress updates (default: True).
        Logs elapsed time every 5 seconds during rendering.
    max_retries (int): Maximum number of retry attempts for GPU memory
        errors (default: 3). Set to 0 to disable retries.
    retry_delay (float): Seconds to wait between retry attempts (default: 5.0).
        Allows GPU memory to be freed by other processes.

Returns:
    Path: Path to rendered file if successful, None otherwise

#### `reproject_raster(src_crs, dst_crs, nodata_value, num_threads)`

Generalized raster reprojection function

Args:
    src_crs: Source coordinate reference system
    dst_crs: Destination coordinate reference system
    nodata_value: Value to use for areas outside original data
    num_threads: Number of threads for parallel processing

Returns:
    Function that transforms data and returns (data, transform, new_crs)

#### `sample_array(arr)`

Downsample array for visualization if it exceeds max_pixels limit.

#### `save(self, target_name, data, transform, crs, metadata)`

Save data as GeoTIFF with CRS and metadata

#### `scale_elevation(scale_factor, nodata_value)`

Create a raster elevation scaling transform function.

Multiplies all elevation values by the scale factor. Useful for reducing
or amplifying terrain height without changing horizontal scale.

Args:
    scale_factor (float): Multiplication factor for elevation values (default: 1.0)
    nodata_value: Value to treat as no data (default: np.nan)

Returns:
    function: A transform function that scales elevation data

#### `set_blended_color_mapping(self, base_colormap, base_source_layers, overlay_colormap, overlay_source_layers, overlay_mask, base_color_kwargs, overlay_color_kwargs)`

Apply two different colormaps based on a spatial mask (hard transition).

Uses base colormap for most of terrain and overlay colormap for masked zones.
This is useful for showing different data types in different regions, such as
elevation colors for general terrain but suitability scores near parks.

Args:
    base_colormap: Function mapping base layer(s) to RGB/RGBA. Takes N arrays
        (one per base_source_layers) and returns (H, W, 3) or (H, W, 4).
    base_source_layers: Layer names to pass to base_colormap, e.g., ['dem'].
    overlay_colormap: Function mapping overlay layer(s) to RGB/RGBA. Takes N
        arrays (one per overlay_source_layers) and returns (H, W, 3) or (H, W, 4).
    overlay_source_layers: Layer names to pass to overlay_colormap, e.g., ['score'].
    overlay_mask: Boolean array of shape (num_vertices,) indicating where to use
        overlay colormap. True = overlay, False = base. Use compute_proximity_mask()
        to create this mask.
    base_color_kwargs: Optional kwargs passed to base_colormap.
    overlay_color_kwargs: Optional kwargs passed to overlay_colormap.

Returns:
    None: Modifies internal color mapping to use blended approach.

Raises:
    ValueError: If source layers don't exist or overlay_mask has wrong shape.
    RuntimeError: If create_mesh() hasn't been called yet (needed for mask validation).

Example:
    >>> from src.terrain.color_mapping import elevation_colormap
    >>> # Compute proximity mask for park zones
    >>> park_mask = terrain.compute_proximity_mask(
    ...     park_lons, park_lats, radius_meters=1000
    ... )
    >>> # Set dual colormaps
    >>> terrain.set_blended_color_mapping(
    ...     base_colormap=lambda elev: elevation_colormap(
    ...         elev, cmap_name="gist_earth"
    ...     ),
    ...     base_source_layers=["dem"],
    ...     overlay_colormap=lambda score: elevation_colormap(
    ...         score, cmap_name="cool", min_elev=0, max_elev=1
    ...     ),
    ...     overlay_source_layers=["score"],
    ...     overlay_mask=park_mask
    ... )
    >>> terrain.compute_colors()  # Apply the blended mapping

#### `set_color_mapping(self, color_func, source_layers, color_kwargs, mask_func, mask_layers, mask_kwargs, mask_threshold)`

Set up how to map data layers to colors (RGB) and optionally a mask/alpha channel.

Allows flexible color mapping by applying a function to one or more data layers.
Optionally applies a separate mask function for transparency/alpha channel control.
Color mapping is applied during mesh creation with `compute_colors()`.

Args:
    color_func (Callable): Function that accepts N data arrays (one per source_layers)
        and returns colored array of shape (H, W, 3) for RGB or (H, W, 4) for RGBA.
        Values should be in range [0, 1] for 8-bit output.
    source_layers (list[str]): Names of data layers to pass to color_func, in order.
        E.g., ['dem'] for single layer or ['red', 'green', 'blue'] for composite.
    color_kwargs (dict, optional): Additional keyword arguments passed to color_func.
    mask_func (Callable, optional): Function producing alpha/mask values (0-1) for
        transparency. Takes layer arrays as input. If omitted, fully opaque.
    mask_layers (list[str] | str, optional): Layer names for mask_func. If None,
        uses source_layers. Single string converted to list.
    mask_kwargs (dict, optional): Additional keyword arguments for mask_func.
    mask_threshold (float, optional): If mask_func is threshold-based, convenience
        parameter for threshold value (implementation-dependent).

Returns:
    None: Modifies internal color mapping configuration.

Raises:
    ValueError: If source_layers or mask_layers refer to non-existent layers.

Examples:
    >>> # Single-layer elevation with viridis colormap
    >>> from matplotlib.cm import viridis
    >>> terrain.set_color_mapping(
    ...     lambda dem: viridis(dem / dem.max()),
    ...     ['dem']
    ... )

    >>> # RGB composite from three layers
    >>> terrain.set_color_mapping(
    ...     lambda r, g, b: np.stack([r, g, b], axis=-1),
    ...     ['red_band', 'green_band', 'blue_band']
    ... )

    >>> # Elevation with water transparency mask
    >>> terrain.set_color_mapping(
    ...     lambda dem: elevation_colormap(dem),
    ...     ['dem'],
    ...     mask_func=lambda dem: (dem > 0).astype(float),
    ...     mask_layers=['dem']
    ... )

    >>> # Hillshade with elevation colors and slope transparency
    >>> terrain.set_color_mapping(
    ...     lambda dem: dem_colors,
    ...     ['dem'],
    ...     mask_func=lambda dem: 1 - np.clip(np.gradient(dem)[0], 0, 1),
    ...     mask_layers=['dem']
    ... )

#### `set_multi_color_mapping(self, base_colormap, base_source_layers, overlays, base_color_kwargs)`

Apply multiple data layers as overlays with different colormaps and priority.

This enables flexible data visualization where multiple geographic features
(roads, trails, land use, power lines, etc.) can each be colored independently
based on their own colormaps and source layers.

Args:
    base_colormap: Function mapping base layer(s) to RGB/RGBA. Takes N arrays
        (one per base_source_layers) and returns (H, W, 3) or (H, W, 4).
    base_source_layers: Layer names for base_colormap, e.g., ['dem'].
    overlays: List of overlay specifications. Each overlay dict contains:
        - 'colormap': Function taking data arrays and returning (H, W, 3/4).
        - 'source_layers': List of layer names for this overlay, e.g., ['roads'].
        - 'colormap_kwargs': Optional dict of kwargs for the colormap function.
        - 'threshold': Optional threshold value (default: 0.5). Only applies overlay
                      where source layer value >= threshold. Important for interpolated
                      discrete data (like roads) that may have fractional values after
                      resampling. Use 0.5 for data that should be 0 or 1+ after rasterization.
        - 'priority': Integer priority (lower = higher priority). First matching
                      overlay is applied.
    base_color_kwargs: Optional kwargs passed to base_colormap.

Returns:
    None: Modifies internal color mapping to use multi-overlay approach.

Raises:
    ValueError: If source layers don't exist or overlay specs are invalid.

Example:
    >>> # Base elevation colors with roads and trails overlays
    >>> terrain.set_multi_color_mapping(
    ...     base_colormap=lambda elev: elevation_colormap(elev, "michigan"),
    ...     base_source_layers=["dem"],
    ...     overlays=[
    ...         {
    ...             "colormap": lambda roads: colormap_roads(roads),
    ...             "source_layers": ["roads"],
    ...             "priority": 10,  # High priority roads show on top
    ...         },
    ...         {
    ...             "colormap": lambda trails: colormap_trails(trails),
    ...             "source_layers": ["trails"],
    ...             "priority": 20,  # Lower priority
    ...         },
    ...     ]
    ... )
    >>> terrain.compute_colors()

#### `setup_camera(camera_angle, camera_location, scale, focal_length, camera_type)`

Configure camera for terrain visualization.

Args:
    camera_angle: Tuple of (x,y,z) rotation angles in radians
    camera_location: Tuple of (x,y,z) camera position
    scale: Camera scale value (ortho_scale for orthographic cameras)
    focal_length: Camera focal length in mm (default: 50, used only for perspective)
    camera_type: Camera type 'PERSP' (perspective) or 'ORTHO' (orthographic) (default: 'PERSP')

Returns:
    Camera object

Raises:
    ValueError: If camera_type is not 'PERSP' or 'ORTHO'

#### `setup_camera_and_light(camera_angle, camera_location, scale, sun_angle, sun_energy, focal_length, camera_type)`

Configure camera and main light for terrain visualization.

Convenience function that calls setup_camera() and setup_light().

Args:
    camera_angle: Tuple of (x,y,z) rotation angles in radians
    camera_location: Tuple of (x,y,z) camera position
    scale: Camera scale value (ortho_scale for orthographic cameras)
    sun_angle: Angle of sun light in degrees (default: 2)
    sun_energy: Energy/intensity of sun light (default: 3)
    focal_length: Camera focal length in mm (default: 50, used only for perspective)
    camera_type: Camera type 'PERSP' (perspective) or 'ORTHO' (orthographic) (default: 'PERSP')

Returns:
    tuple: (camera object, sun light object)

#### `setup_hdri_lighting(sun_elevation, sun_rotation, sun_intensity, sun_size, air_density, visible_to_camera, camera_background, sky_strength)`

Set up HDRI-style sky lighting using Blender's Nishita sky model.

Creates realistic sky lighting that contributes to ambient illumination
without being visible in the final render (by default).

Args:
    sun_elevation: Sun elevation angle in degrees (0=horizon, 90=overhead)
    sun_rotation: Sun rotation/azimuth in degrees (0=front, 180=back)
    sun_intensity: Multiplier for sun disc brightness in sky texture (default: 1.0)
    sun_size: Angular diameter of sun disc in degrees (default: 0.545 = real sun).
              Larger values create softer shadows, smaller values create sharper.
    air_density: Atmospheric density (default: 1.0, higher=hazier)
    visible_to_camera: If False, sky is invisible but still lights scene
    camera_background: RGB tuple for background when sky invisible.
                      Use (0.9, 0.9, 0.9) for atmosphere to work.
    sky_strength: Overall sky emission strength (ambient light level).
                 If None, defaults to sun_intensity for backwards compatibility.

Returns:
    bpy.types.World: The configured world object

#### `setup_light(location, angle, energy, rotation_euler, azimuth, elevation)`

Create and configure sun light for terrain visualization.

Sun position can be specified either with rotation_euler (raw Blender angles)
or with the more intuitive azimuth/elevation system:

- azimuth: Direction the sun comes FROM, in degrees clockwise from North
  (0=North, 90=East, 180=South, 270=West)
- elevation: Angle above horizon in degrees (0=horizon, 90=directly overhead)

Args:
    location: Tuple of (x,y,z) light position (default: (1, 1, 2))
    angle: Angle of sun light in degrees (default: 2)
    energy: Energy/intensity of sun light (default: 3)
    rotation_euler: Tuple of (x,y,z) rotation angles in radians (default: sun from NW)
    azimuth: Direction sun comes FROM in degrees (0=N, 90=E, 180=S, 270=W)
    elevation: Sun angle above horizon in degrees (0=horizon, 90=overhead)

Returns:
    Sun light object

#### `setup_render_settings(use_gpu, samples, preview_samples, use_denoising, denoiser, compute_device, use_persistent_data, use_auto_tile, tile_size)`

Configure Blender render settings for high-quality terrain visualization.

Args:
    use_gpu: Whether to use GPU acceleration
    samples: Number of render samples
    preview_samples: Number of viewport preview samples
    use_denoising: Whether to enable denoising
    denoiser: Type of denoiser to use ('OPTIX', 'OPENIMAGEDENOISE', 'NLM')
    compute_device: Compute device type ('OPTIX', 'CUDA', 'HIP', 'METAL')
    use_persistent_data: Keep scene data in memory between frames (default: False)
    use_auto_tile: Enable automatic tiling for large renders (default: False).
        Splits large images into smaller GPU-friendly tiles to reduce VRAM usage.
        Essential for print-quality renders (3000x2400+ pixels).
    tile_size: Tile size in pixels when auto_tile is enabled (default: 2048).
        Smaller tiles = less VRAM but slower rendering. Try 512-1024 for limited VRAM.

#### `setup_two_point_lighting(sun_azimuth, sun_elevation, sun_energy, sun_angle, sun_color, fill_azimuth, fill_elevation, fill_energy, fill_angle, fill_color)`

Set up two-point lighting with primary sun and optional fill light.

Creates professional-quality lighting for terrain visualization:
- Primary sun: Creates shadows and defines form (warm color by default)
- Fill light: Softens shadows, adds depth (cool color by default)

Args:
    sun_azimuth: Direction sun comes FROM in degrees (0=N, 90=E, 180=S, 270=W)
    sun_elevation: Sun angle above horizon in degrees (0=horizon, 90=overhead)
    sun_energy: Sun light strength (default: 7.0)
    sun_angle: Sun angular size in degrees (default: 1.0)
    sun_color: RGB tuple for sun color (default: warm golden)
    fill_azimuth: Direction fill light comes FROM in degrees
    fill_elevation: Fill light angle above horizon in degrees
    fill_energy: Fill light strength (default: 0.0 = no fill)
    fill_angle: Fill light angular size in degrees
    fill_color: RGB tuple for fill color (default: cool blue)

Returns:
    List of created light objects

#### `setup_world_atmosphere(density, scatter_color, anisotropy)`

Set up world volume for atmospheric effects.

Args:
    density: Density of the atmospheric volume (default: 0.02)
    scatter_color: RGBA color tuple for scatter (default: white)
    anisotropy: Direction of scatter from -1 to 1 (default: 0 for uniform)

Returns:
    bpy.types.World: The configured world object

#### `slope_colormap(slopes, cmap_name, min_slope, max_slope)`

Create a simple colormap based solely on terrain slopes.

Args:
    slopes: Array of slope values in degrees
    cmap_name: Matplotlib colormap name (default: 'terrain')
    min_slope: Minimum slope value for normalization (default: 0)
    max_slope: Maximum slope value for normalization (default: 45)

Returns:
    Array of RGBA colors with shape (*slopes.shape, 4)

#### `smooth_raster(window_size, nodata_value)`

Create a raster smoothing transform function with specified parameters.

Args:
    window_size: Size of median filter window
                (defaults to 5% of smallest dimension if None)
    nodata_value: Value to treat as no data (default: np.nan)

Returns:
    function: A transform function that smooths raster data

#### `smooth_score_data(scores, sigma_spatial, sigma_intensity)`

Smooth score data using bilateral filtering.

Applies feature-preserving smoothing to reduce blocky pixelation from
low-resolution source data (e.g., SNODAS ~925m) when displayed on
high-resolution terrain (~30m DEM).

Uses bilateral filtering: smooths within similar-intensity regions while
preserving edges between different score zones.

Args:
    scores: 2D numpy array of score values (typically 0-1 range)
    sigma_spatial: Spatial smoothing extent in pixels (default: 3.0).
        Larger = more smoothing. Typical range: 1-10 pixels.
    sigma_intensity: Intensity similarity threshold in score units.
        Larger = more smoothing across score differences.
        If None, auto-calculated as 15% of score range (good for 0-1 data).

Returns:
    Smoothed score array with same shape as input.
    NaN values are preserved. Output is clipped to [0, 1] range.

Example:
    >>> # Smooth blocky SNODAS-derived scores
    >>> sledding_scores = load_score_grid("sledding_scores.npz")
    >>> smoothed = smooth_score_data(sledding_scores, sigma_spatial=5.0)

#### `transform_wrapper(transform_func)`

Standardize transform function interface with consistent output

Args:
    transform_func: The original transform function to wrap

Returns:
    A wrapped function with consistent signature and return format

#### `visualize_dem(self, layer, use_transformed, title, cmap, percentile_clip, clip_percentiles, max_pixels, show_histogram)`

Create diagnostic visualization of any terrain data layer.

Args:
    layer: Name of data layer to visualize (default: 'dem')
    use_transformed: Whether to use transformed or original data (default: False)
    title: Plot title (default: auto-generated based on layer)
    cmap: Matplotlib colormap
    percentile_clip: Whether to clip extreme values
    clip_percentiles: Tuple of (min, max) percentiles to clip (default: (1, 99))
    max_pixels: Maximum number of pixels for subsampling
    show_histogram: Whether to show the histogram panel (default: True)

#### `wrapped_transform(data, transform)`

Standardized transform wrapper with consistent signature

Args:
    data: Input numpy array to transform
    transform: Optional affine transform

Returns:
    Tuple of (transformed_data, transform, [crs]) where CRS is optional

---

## terrain.data_loading

Data loading operations for terrain processing.

This module contains functions for loading and merging DEM (Digital Elevation Model)
files from various sources.

### Functions

#### `find_score_file(name, search_dirs, subdirs)`

Search for a score file in common locations.

Useful for finding pre-computed score files that may be in various
locations depending on how the pipeline was run.

Args:
    name: Base filename to search for (e.g., "sledding_scores.npz")
    search_dirs: List of directories to search. Defaults to common locations.
    subdirs: Subdirectories to check within each search_dir
        (e.g., ["sledding", "xc_skiing"])

Returns:
    Path to found file, or None if not found

Example:
    >>> path = find_score_file("sledding_scores.npz",
    ...                         search_dirs=[Path("docs/images"), Path("output")],
    ...                         subdirs=["sledding", ""])
    >>> if path:
    ...     scores, transform = load_score_grid(path)

#### `load_dem_files(directory_path, pattern, recursive)`

Load and merge DEM files from a directory into a single elevation dataset.
Supports any raster format readable by rasterio (HGT, GeoTIFF, etc.).

Args:
    directory_path: Path to directory containing DEM files
    pattern: File pattern to match (default: "*.hgt")
    recursive: Whether to search subdirectories recursively (default: False)

Returns:
    tuple: (merged_dem, transform) where:
        - merged_dem: numpy array containing the merged elevation data
        - transform: affine transform mapping pixel to geographic coordinates

Raises:
    ValueError: If no valid DEM files are found or directory doesn't exist
    OSError: If directory access fails or file reading fails
    rasterio.errors.RasterioIOError: If there are issues reading the DEM files

#### `load_filtered_hgt_files(dem_dir, min_latitude, max_latitude, min_longitude, max_longitude, bbox, pattern)`

Load SRTM HGT files filtered by latitude/longitude range.

Works globally with standard SRTM naming convention. Filters files
before loading to reduce memory usage for large DEM directories.

Args:
    dem_dir: Directory containing HGT files
    min_latitude: Southern bound (e.g., -45 for S45, 42 for N42)
    max_latitude: Northern bound (e.g., 60 for N60)
    min_longitude: Western bound (e.g., -120 for W120)
    max_longitude: Eastern bound (e.g., 30 for E30)
    bbox: Bounding box as (west, south, east, north) tuple. If provided,
        overrides individual min/max parameters. Uses standard GIS convention:
        (min_lon, min_lat, max_lon, max_lat).
    pattern: File pattern to match (default: "*.hgt")

Returns:
    Tuple of (merged_dem, transform)

Raises:
    ValueError: If no matching files found after filtering

Example:
    >>> # Load only tiles in Michigan area using individual params
    >>> dem, transform = load_filtered_hgt_files(
    ...     "/path/to/srtm",
    ...     min_latitude=41, max_latitude=47,
    ...     min_longitude=-90, max_longitude=-82
    ... )

    >>> # Same area using bbox (west, south, east, north)
    >>> dem, transform = load_filtered_hgt_files(
    ...     "/path/to/srtm",
    ...     bbox=(-90, 41, -82, 47)
    ... )

    >>> # Alps region (Switzerland/Austria)
    >>> dem, transform = load_filtered_hgt_files(
    ...     "/path/to/srtm",
    ...     bbox=(5, 45, 15, 48)
    ... )

#### `load_score_grid(file_path, data_keys)`

Load georeferenced raster data from an NPZ file.

Works with any NPZ file containing a 2D array and optional Affine transform.
Common use cases: score grids, classification maps, derived terrain products.

The function searches for data arrays using the provided keys, falling back
to common key names and finally to the first available array.

Args:
    file_path: Path to .npz file
    data_keys: Keys to try for data array. If None, tries ["data", "score", "values"]
        then falls back to first array in file.

Returns:
    Tuple of (data_array, transform) where:
        - data_array: 2D numpy array with the raster data
        - transform: Affine transform or None if not present in file

Raises:
    FileNotFoundError: If file doesn't exist
    ValueError: If file contains no arrays

Example:
    >>> scores, transform = load_score_grid("path/to/scores.npz")
    >>> if transform:
    ...     terrain.add_data_layer("scores", scores, transform, "EPSG:4326")

#### `parse_hgt_filename(filename)`

Parse SRTM HGT filename to extract latitude and longitude.

Works globally with standard SRTM naming convention:
- N42W083.hgt (Northern/Western hemisphere) -> lat=42, lon=-83
- S15E028.hgt (Southern/Eastern hemisphere) -> lat=-15, lon=28

Args:
    filename: HGT filename or Path (e.g., "N42W083.hgt" or Path("/path/to/N42W083.hgt"))

Returns:
    Tuple of (latitude, longitude) as signed integers, or (None, None) if invalid

#### `save_score_grid(file_path, data, transform, data_key, **metadata)`

Save georeferenced raster data to an NPZ file.

Creates an NPZ file compatible with load_score_grid(). The transform
is stored as a 6-element array that can be reconstructed as an Affine.

Args:
    file_path: Output path for .npz file
    data: 2D numpy array with raster data
    transform: Optional Affine transform for georeferencing
    data_key: Key name for the data array (default: "data")
    **metadata: Additional key=value pairs to store in the file

Returns:
    Path to the saved file

Example:
    >>> from rasterio import Affine
    >>> scores = compute_sledding_scores(dem)
    >>> transform = Affine.translation(-83.5, 42.5) * Affine.scale(0.01, -0.01)
    >>> save_score_grid("scores.npz", scores, transform, crs="EPSG:4326")

    >>> # Load it back
    >>> loaded_scores, loaded_transform = load_score_grid("scores.npz")

---

## terrain.diagnostics

Diagnostic plotting utilities for terrain processing.

Provides visualization functions to understand and debug terrain transforms,
particularly wavelet denoising, slope-adaptive smoothing, and other processing steps.

### Functions

#### `generate_bump_removal_diagnostics(original, after_removal, output_dir, prefix, kernel_size, nodata_value)`

Generate bump removal diagnostic plot.

Args:
    original: Original DEM before bump removal
    after_removal: DEM after morphological opening
    output_dir: Directory to save diagnostic plot
    prefix: Filename prefix
    kernel_size: Kernel size used for removal
    nodata_value: Value treated as no data

Returns:
    Path to saved diagnostic plot

#### `generate_full_adaptive_smooth_diagnostics(original, smoothed, output_dir, prefix, slope_threshold, smooth_sigma, transition_width, nodata_value, pixel_size, edge_threshold, edge_window)`

Generate all slope-adaptive smoothing diagnostic plots.

Creates both the spatial comparison and the histogram analysis.

Args:
    original: Original DEM before smoothing
    smoothed: DEM after slope-adaptive smoothing
    output_dir: Directory to save diagnostic plots
    prefix: Filename prefix for saved plots
    slope_threshold: Slope threshold in degrees
    smooth_sigma: Gaussian sigma used for smoothing
    transition_width: Width of sigmoid transition zone
    nodata_value: Value treated as no data
    pixel_size: Pixel size in meters (e.g., 30.0 for SRTM data).
        Required for accurate slope calculation.
    edge_threshold: Edge preservation threshold in meters (default: None).
        If set, shows which areas are protected due to sharp elevation changes.
    edge_window: Window size for edge detection (default: 5).

Returns:
    Tuple of (spatial_plot_path, histogram_plot_path)

#### `generate_full_wavelet_diagnostics(original, denoised, output_dir, prefix, wavelet, levels, threshold_sigma, nodata_value)`

Generate all wavelet diagnostic plots.

Creates both the before/after comparison and the coefficient analysis.

Args:
    original: Original DEM before denoising
    denoised: DEM after wavelet denoising
    output_dir: Directory to save diagnostic plots
    prefix: Filename prefix for saved plots
    wavelet: Wavelet type used for denoising
    levels: Number of decomposition levels
    threshold_sigma: Sigma multiplier used for thresholding
    nodata_value: Value treated as no data

Returns:
    Tuple of (comparison_plot_path, coefficient_plot_path)

#### `generate_luminance_histogram(image_path, output_path)`

Generate and save a luminance (B&W) histogram of a rendered image.

Shows distribution of brightness values with annotations for pure black
and pure white pixel counts. Useful for checking exposure and clipping
in rendered outputs, especially for print preparation.

Args:
    image_path: Path to the rendered image (PNG, JPEG, etc.)
    output_path: Path to save the histogram image

Returns:
    Path to saved histogram image, or None if failed

#### `generate_rgb_histogram(image_path, output_path)`

Generate and save an RGB histogram of a rendered image.

Creates a figure with histograms for each color channel (R, G, B)
overlaid on the same axes with transparency. Useful for analyzing
color balance and distribution in rendered outputs.

Args:
    image_path: Path to the rendered image (PNG, JPEG, etc.)
    output_path: Path to save the histogram image

Returns:
    Path to saved histogram image, or None if failed

#### `generate_upscale_diagnostics(original, upscaled, output_dir, prefix, scale, method, nodata_value, cmap)`

Generate upscale diagnostic plot.

Args:
    original: Original score grid before upscaling
    upscaled: Score grid after upscaling
    output_dir: Directory to save diagnostic plot
    prefix: Filename prefix
    scale: Upscaling factor used
    method: Upscaling method name
    nodata_value: Value treated as no data
    cmap: Colormap for score visualization

Returns:
    Path to saved diagnostic plot

#### `plot_adaptive_smooth_diagnostics(original, smoothed, output_path, slope_threshold, smooth_sigma, transition_width, title_prefix, nodata_value, profile_row, cmap, pixel_size, edge_threshold, edge_window)`

Generate diagnostic plots showing slope-adaptive smoothing effects.

Creates a multi-panel figure showing:
- Original DEM
- Computed slope map
- Smoothing weight mask (where smoothing is applied)
- Smoothed DEM
- Difference (noise removed)
- Cross-section profile comparison

Args:
    original: Original DEM before smoothing
    smoothed: DEM after slope-adaptive smoothing
    output_path: Path to save the diagnostic plot
    slope_threshold: Slope threshold in degrees used for smoothing
    smooth_sigma: Gaussian sigma used for smoothing
    transition_width: Width of sigmoid transition zone
    title_prefix: Prefix for plot titles
    nodata_value: Value treated as no data
    profile_row: Row index for cross-section (default: middle row)
    cmap: Colormap for elevation visualization
    pixel_size: Pixel size in meters (e.g., 30.0 for SRTM data).
        Required for accurate slope calculation.
    edge_threshold: Edge preservation threshold in meters (default: None).
        If set, shows which areas are protected due to sharp elevation changes.
    edge_window: Window size for edge detection (default: 5).

Returns:
    Path to saved diagnostic plot

#### `plot_adaptive_smooth_histogram(original, smoothed, output_path, slope_threshold, transition_width, title_prefix, nodata_value, pixel_size)`

Generate histogram analysis of slope-adaptive smoothing.

Shows distribution of slopes and how much smoothing was applied
at different slope values.

Args:
    original: Original DEM before smoothing
    smoothed: DEM after slope-adaptive smoothing
    output_path: Path to save the diagnostic plot
    slope_threshold: Slope threshold in degrees used for smoothing
    transition_width: Width of sigmoid transition zone
    title_prefix: Prefix for plot titles
    nodata_value: Value treated as no data
    pixel_size: Pixel size in meters (e.g., 30.0 for SRTM data).
        Required for accurate slope calculation.

Returns:
    Path to saved diagnostic plot

#### `plot_bump_removal_diagnostics(original, after_removal, output_path, kernel_size, title_prefix, nodata_value, cmap)`

Generate diagnostic plot for morphological bump removal.

Shows:
- Original DEM
- After bump removal
- Bumps removed (difference) - always positive since only peaks are removed
- Histogram of bump heights

Args:
    original: Original DEM before bump removal
    after_removal: DEM after morphological opening
    output_path: Path to save the diagnostic plot
    kernel_size: Kernel size used for removal
    title_prefix: Prefix for plot titles
    nodata_value: Value treated as no data
    cmap: Colormap for elevation visualization

Returns:
    Path to saved diagnostic plot

#### `plot_processing_pipeline(stages, output_path, title, cmap, nodata_value)`

Generate comparison plots showing multiple processing stages.

Useful for visualizing an entire processing pipeline with
multiple transforms (raw → despeckle → wavelet → smooth, etc.)

Args:
    stages: Dictionary mapping stage names to DEM arrays
        Example: {"Raw": raw_dem, "Despeckled": despeckled, "Final": final}
    output_path: Path to save the diagnostic plot
    title: Plot title
    cmap: Colormap for elevation
    nodata_value: Value treated as no data

Returns:
    Path to saved diagnostic plot

#### `plot_upscale_diagnostics(original, upscaled, output_path, scale, method, title_prefix, nodata_value, cmap)`

Generate diagnostic plots showing score upscaling effects.

Creates a multi-panel figure showing:
- Original score grid
- Upscaled score grid
- Zoomed comparison of a region
- Histograms of value distributions
- Edge detail comparison

Args:
    original: Original score grid before upscaling
    upscaled: Score grid after upscaling
    output_path: Path to save the diagnostic plot
    scale: Upscaling factor used
    method: Upscaling method name (for title)
    title_prefix: Prefix for plot titles
    nodata_value: Value treated as no data
    cmap: Colormap for score visualization

Returns:
    Path to saved diagnostic plot

#### `plot_wavelet_coefficients(original, output_path, wavelet, levels, threshold_sigma, nodata_value)`

Generate diagnostic plots showing wavelet coefficient distributions.

Creates a figure showing:
- Coefficient histograms before/after thresholding
- Decomposition levels visualization
- Threshold line for each level

Args:
    original: Original DEM
    output_path: Path to save the diagnostic plot
    wavelet: Wavelet type used for decomposition
    levels: Number of decomposition levels
    threshold_sigma: Sigma multiplier for thresholding
    nodata_value: Value treated as no data

Returns:
    Path to saved diagnostic plot

#### `plot_wavelet_diagnostics(original, denoised, output_path, title_prefix, nodata_value, profile_row, cmap)`

Generate diagnostic plots showing wavelet denoising effects.

Creates a multi-panel figure showing:
- Original DEM
- Denoised DEM
- Difference (noise removed)
- Cross-section profile comparison

Args:
    original: Original DEM before denoising
    denoised: DEM after wavelet denoising
    output_path: Path to save the diagnostic plot
    title_prefix: Prefix for plot titles
    nodata_value: Value treated as no data
    profile_row: Row index for cross-section (default: middle row)
    cmap: Colormap for elevation visualization

Returns:
    Path to saved diagnostic plot

---

## terrain.gridded_data

Generic gridded data loader with pipeline caching for terrain visualization.

Handles loading external gridded datasets (SNODAS, temperature, precipitation, etc.),
processing through user-defined pipelines, and caching each step independently.

Features:
- Transparent automatic tiling for large datasets
- Memory monitoring with failsafe to prevent OOM/thrashing
- Per-step and merged result caching
- Smart aggregation (concatenation for spatial data, averaging for statistics)

### Classes

#### `GriddedDataLoader`

Load and cache external gridded data with pipeline processing.

This class provides a general framework for:
- Loading gridded data from arbitrary formats
- Processing data through multi-step pipelines
- Caching each pipeline step independently
- Smart cache invalidation based on step dependencies

Pipeline format: List of (name, function, kwargs) tuples

Example:
    >>> def load_data(source, extent, target_shape):
    ...     # Load and crop data
    ...     return {"raw": data_array}
    >>>
    >>> def compute_stats(input_data):
    ...     # Compute statistics from previous step
    ...     raw = input_data["raw"]
    ...     return {"mean": raw.mean(), "std": raw.std()}
    >>>
    >>> pipeline = [
    ...     ("load", load_data, {}),
    ...     ("stats", compute_stats, {}),
    ... ]
    >>>
    >>> loader = GriddedDataLoader(terrain, cache_dir=Path(".cache"))
    >>> result = loader.run_pipeline(
    ...     data_source="/path/to/data",
    ...     pipeline=pipeline,
    ...     cache_name="my_analysis"
    ... )

**Methods:**

- `run_pipeline(self, data_source, pipeline, cache_name, force_reprocess)` - Execute a processing pipeline with caching at each step.


#### `MemoryLimitExceeded`

Raised when memory usage exceeds configured limits.


#### `MemoryMonitor`

Monitor system memory and abort processing if limits exceeded.

**Methods:**

- `check_memory(self, force)` - Check memory usage and raise MemoryLimitExceeded if over threshold.


#### `TileSpecGridded`

Tile specification with geographic extent for gridded data.


#### `TiledDataConfig`

Configuration for automatic tiling in GriddedDataLoader.


### Functions

#### `check_memory(self, force)`

Check memory usage and raise MemoryLimitExceeded if over threshold.

Args:
    force: Force check even if check_interval hasn't elapsed

Raises:
    MemoryLimitExceeded: If memory or swap usage exceeds limits

#### `create_mock_snow_data(shape)`

Create mock snow data for testing.

Generates realistic-looking mock snow statistics using statistical
distributions that mimic real SNODAS patterns.

Args:
    shape: Shape of the snow data arrays (height, width)

Returns:
    Dictionary with mock snow statistics:
    - median_max_depth: Snow depth in mm (gamma distribution)
    - mean_snow_day_ratio: Fraction of days with snow (beta distribution)
    - interseason_cv: Year-to-year variability (beta distribution)
    - mean_intraseason_cv: Within-winter variability (beta distribution)

#### `downsample_for_viz(arr, max_dim)`

Downsample array using stride slicing for visualization.

Args:
    arr: Input array to downsample
    max_dim: Maximum dimension size for output

Returns:
    Tuple of (downsampled_array, stride_used)

#### `run_pipeline(self, data_source, pipeline, cache_name, force_reprocess)`

Execute a processing pipeline with caching at each step.

Features:
- Transparent automatic tiling for large outputs
- Memory monitoring with failsafe
- Per-step and merged result caching

Args:
    data_source: Data source (directory, file list, URL, etc.)
    pipeline: List of (step_name, function, kwargs) tuples
              Each function receives previous step's output as first arg
    cache_name: Base name for cache files
    force_reprocess: Force reprocessing all steps even if cached

Returns:
    Output of final pipeline step

Raises:
    MemoryLimitExceeded: If memory limits exceeded during tiling

---

## terrain.materials

Material and shader operations for Blender terrain visualization.

This module contains functions for creating and configuring Blender materials,
shaders, and background planes for terrain rendering.

### Functions

#### `apply_colormap_material(material)`

Create a physically-based material for terrain visualization using vertex colors.

Uses pure Principled BSDF for proper lighting response - no emission.
Terrain responds realistically to sun direction and casts proper shadows.

Args:
    material: Blender material to configure

#### `apply_glassy_road_material(material)`

Deprecated: Use apply_terrain_with_obsidian_roads() instead.

#### `apply_terrain_with_obsidian_roads(material, terrain_style, road_color)`

Create a material with glossy roads and terrain colors/test material.

Reads from two vertex color layers:
- "TerrainColors": The actual terrain colors (used for non-road areas)
- "RoadMask": R channel marks road pixels (R > 0.5 = road)

Roads render with glossy metallic properties (like polished stone).
Non-road areas use either vertex colors or a test material.

Args:
    material: Blender material to configure
    terrain_style: Optional test material for terrain ("chrome", "clay", etc.)
                  If None, uses vertex colors with pure Principled BSDF (no emission).
    road_color: Road color - either a preset name from ROAD_COLORS
               ("obsidian", "azurite", "azurite-light", "malachite", "hematite")
               or an RGB tuple (0-1 range). Default: "obsidian" (near-black).

#### `apply_test_material(material, style)`

Apply a test material to the entire terrain mesh.

Test materials ignore vertex colors and apply a uniform material style
for testing lighting, shadows, and mesh geometry.

Args:
    material: Blender material to configure
    style: Material style name - one of:
        - "obsidian": Glossy black glass (metallic, mirror-smooth)
        - "chrome": Metallic chrome with reflections
        - "clay": Matte gray clay (diffuse, no reflections)
        - "plastic": Glossy white plastic
        - "gold": Metallic gold with warm tones
        - "terrain": Normal terrain with vertex colors (default)

Raises:
    ValueError: If style is not recognized

#### `apply_water_shader(material, water_color)`

Apply water shader to material, coloring water areas based on vertex alpha channel.
Uses alpha channel to mix between water color and elevation colors.
Water pixels (alpha=1.0) render as water color; land pixels (alpha=0.0) show elevation colors.

Args:
    material: Blender material to configure
    water_color: RGB tuple for water (default: University of Michigan blue #00274C)

#### `create_background_plane(terrain_obj, depth, scale_factor, material_params)`

Create a large emissive plane beneath the terrain for background illumination.

Args:
    terrain_obj: The terrain Blender object used for size reference
    depth: Z-coordinate for the plane position
    scale_factor: Scale multiplier for plane size relative to terrain
    material_params: Optional dict to override default material parameters

Returns:
    bpy.types.Object: The created background plane object

Raises:
    ValueError: If terrain_obj is None or has invalid bounds
    RuntimeError: If mesh or material creation fails

---

## terrain.mesh_cache

Mesh caching module for Blender .blend file reuse.

Caches mesh generation results so render-only passes don't need to regenerate
the mesh. Useful for iterating on camera angles and render settings without
waiting for mesh generation.

### Classes

#### `MeshCache`

Manages caching of generated Blender mesh files.

The cache stores:
- .blend files (Blender scene with generated mesh)
- Metadata including generation parameters and hash

Attributes:
    cache_dir: Directory where mesh cache is stored
    enabled: Whether caching is enabled

**Methods:**

- `clear_cache(self, cache_name)` - Clear cached mesh files for a specific cache name.

- `compute_mesh_hash(self, dem_hash, mesh_params)` - Compute hash of mesh generation parameters.

- `get_cache_path(self, mesh_hash, cache_name)` - Get the path for a mesh cache file.

- `get_cache_stats(self)` - Get statistics about cached mesh files.

- `get_metadata_path(self, mesh_hash, cache_name)` - Get the path for mesh metadata file.

- `load_cache(self, mesh_hash, cache_name)` - Load cached mesh .blend file.

- `save_cache(self, blend_file, mesh_hash, mesh_params, cache_name)` - Cache a generated mesh by copying the .blend file.


### Functions

#### `clear_cache(self, cache_name)`

Clear cached mesh files for a specific cache name.

Args:
    cache_name: Name of cache item to clear (default: "mesh")

Returns:
    Number of files deleted

#### `compute_mesh_hash(self, dem_hash, mesh_params)`

Compute hash of mesh generation parameters.

This ensures the cache is invalidated if mesh generation parameters change:
- DEM data (via dem_hash)
- scale_factor
- height_scale
- center_model
- boundary_extension
- water_mask applied

Args:
    dem_hash: Hash of DEM data
    mesh_params: Dictionary of mesh generation parameters

Returns:
    SHA256 hash of mesh parameters

#### `get_cache_path(self, mesh_hash, cache_name)`

Get the path for a mesh cache file.

Args:
    mesh_hash: Hash of mesh parameters
    cache_name: Name of cache item (default: "mesh")

Returns:
    Path to .blend file

#### `get_cache_stats(self)`

Get statistics about cached mesh files.

Returns:
    Dictionary with cache statistics

#### `get_metadata_path(self, mesh_hash, cache_name)`

Get the path for mesh metadata file.

Args:
    mesh_hash: Hash of mesh parameters
    cache_name: Name of cache item (default: "mesh")

Returns:
    Path to metadata file

#### `load_cache(self, mesh_hash, cache_name)`

Load cached mesh .blend file.

Args:
    mesh_hash: Hash of mesh parameters
    cache_name: Name of cache item (default: "mesh")

Returns:
    Path to cached .blend file or None if not found

#### `save_cache(self, blend_file, mesh_hash, mesh_params, cache_name)`

Cache a generated mesh by copying the .blend file.

Args:
    blend_file: Path to source .blend file
    mesh_hash: Hash of mesh parameters
    mesh_params: Dictionary of mesh generation parameters
    cache_name: Name of cache item (default: "mesh")

Returns:
    Tuple of (cached_blend_path, metadata_path)

---

## terrain.mesh_operations

Mesh generation operations for terrain visualization.

This module contains functions for creating and manipulating terrain meshes,
extracted from the core Terrain class for better modularity and testability.

Performance optimizations:
- Numba JIT compilation for hot loops (face generation)
- Vectorized NumPy operations where possible

### Functions

#### `create_boundary_extension(positions, boundary_points, coord_to_index, base_depth)`

Create boundary extension vertices and faces to close the mesh.

Creates a "skirt" around the terrain by adding bottom vertices at base_depth
and connecting them to the top boundary with quad faces. This closes the mesh
into a solid object suitable for 3D printing or solid rendering.

Args:
    positions (np.ndarray): Array of (n, 3) vertex positions
    boundary_points (list): List of (y, x) tuples representing ordered boundary points
    coord_to_index (dict): Mapping from (y, x) coordinates to vertex indices
    base_depth (float): Z-coordinate for the bottom of the extension (default: -0.2)

Returns:
    tuple: (boundary_vertices, boundary_faces) where:
        - boundary_vertices: np.ndarray of (n_boundary, 3) bottom vertex positions
        - boundary_faces: list of tuples defining side face quad connectivity

#### `decorator(func)`

Inner decorator that returns the function unchanged.

#### `find_boundary_points(valid_mask)`

Find boundary points using morphological operations.

Identifies points on the edge of valid regions using binary erosion.
A point is considered a boundary point if it is valid but has at least
one invalid neighbor in a 4-connected neighborhood.

Args:
    valid_mask (np.ndarray): Boolean mask indicating valid points (True for valid)

Returns:
    list: List of (y, x) coordinate tuples representing boundary points

#### `generate_faces(height, width, coord_to_index, batch_size)`

Generate mesh faces from a grid of valid points.

Creates quad faces for the mesh by checking each potential quad position
and verifying that its corners exist in the coordinate-to-index mapping.
If a quad has all 4 corners, creates a quad face. If it has 3 corners,
creates a triangle face. Skips quads with fewer than 3 corners.

Args:
    height (int): Height of the DEM grid
    width (int): Width of the DEM grid
    coord_to_index (dict): Mapping from (y, x) coordinates to vertex indices
    batch_size (int): Number of quads to process in each batch (default: 10000)

Returns:
    list: List of face tuples, where each tuple contains vertex indices

#### `generate_vertex_positions(dem_data, valid_mask, scale_factor, height_scale)`

Generate 3D vertex positions from DEM data.

Converts 2D elevation grid into 3D positions for mesh vertices, applying
scaling factors for visualization. Only generates vertices for valid (non-NaN)
DEM values.

Args:
    dem_data (np.ndarray): 2D array of elevation values (height x width)
    valid_mask (np.ndarray): Boolean mask indicating valid points (True for non-NaN)
    scale_factor (float): Horizontal scale divisor for x/y coordinates (default: 100.0).
        Higher values produce smaller meshes. E.g., 100 means 100 DEM units = 1 unit.
    height_scale (float): Multiplier for elevation values (default: 1.0).
        Values > 1 exaggerate terrain, < 1 flatten it.

Returns:
    tuple: (positions, y_valid, x_valid) where:
        - positions: np.ndarray of shape (n_valid, 3) with (x, y, z) coordinates
        - y_valid: np.ndarray of y indices for valid points
        - x_valid: np.ndarray of x indices for valid points

#### `jit(*args, **kwargs)`

No-op JIT decorator fallback when numba is not available.

When numba is not installed, this decorator simply returns the function
unchanged, allowing code to run without JIT compilation.

Args:
    *args: Ignored positional arguments (for numba compatibility)
    **kwargs: Ignored keyword arguments (for numba compatibility)

Returns:
    Decorator function that returns the original function unchanged

#### `sort_boundary_points(boundary_coords)`

Sort boundary points efficiently using spatial relationships.

Uses a KD-tree for efficient nearest neighbor queries to create a continuous
path along the boundary points. This is useful for creating side faces that
close a terrain mesh into a solid object.

Args:
    boundary_coords: List of (y, x) coordinate tuples representing boundary points

Returns:
    list: Sorted boundary points forming a continuous path around the perimeter

---

## terrain.pipeline

Lightweight dependency graph pipeline for terrain visualization.

Provides declarative task dependencies with automatic caching, staleness detection,
and execution planning. Integrates with existing DEMCache and MeshCache.

Example:
    from src.terrain.pipeline import TerrainPipeline

    pipeline = TerrainPipeline(dem_dir="data/dem/detroit", cache_enabled=True)

    # Show execution plan
    pipeline.explain("render_view")

    # Get cache statistics
    stats = pipeline.cache_stats()

    # Clear cache
    pipeline.clear_cache()

### Classes

#### `TaskState`

Represents execution state of a task.


#### `TerrainPipeline`

Dependency graph executor for terrain rendering pipeline.

Manages task dependencies, caching, and execution planning without
external build system dependencies. All logic stays in Python.

Features:
- Automatic staleness detection via hashing
- Dry-run execution plans
- Reusable cached outputs across multiple views
- Per-layer caching support

Tasks in pipeline:
1. load_dem: Load and merge SRTM tiles (cached)
2. apply_transforms: Reproject, flip, downsample (can be cached)
3. detect_water: Identify water bodies from DEM (can be cached)
4. create_mesh: Build Blender geometry (cached, reused across views)
5. render_view: Output to PNG (cached by view/render params)

**Methods:**

- `apply_transforms(self, target_vertices, reproject_crs, scale_factor)` - Task: Apply reprojection, flipping, and elevation scaling.

- `cache_stats(self)` - Get cache statistics.

- `clear_cache(self)` - Clear all caches.

- `create_mesh(self, scale_factor, height_scale, center_model, boundary_extension, transform_params, water_params)` - Task: Create Blender mesh from DEM and water mask.

- `detect_water(self, slope_threshold, fill_holes)` - Task: Detect water bodies using slope analysis on unscaled DEM.

- `explain(self, task_name)` - Explain what would execute to build a task (show dependency tree).

- `load_dem(self, pattern)` - Task: Load and cache raw DEM from SRTM tiles.

- `render_all_views(self, views)` - Render all views efficiently.

- `render_view(self, view, width, height, distance, elevation, focal_length, camera_type, samples)` - Task: Render a view to PNG.


### Functions

#### `apply_transforms(self, target_vertices, reproject_crs, scale_factor)`

Task: Apply reprojection, flipping, and elevation scaling.

Args:
    target_vertices: Mesh density target
    reproject_crs: Target CRS for reprojection
    scale_factor: Elevation scaling factor

Returns:
    (transformed_dem, transform)

#### `cache_stats(self)`

Get cache statistics.

#### `clear_cache(self)`

Clear all caches.

#### `create_mesh(self, scale_factor, height_scale, center_model, boundary_extension, transform_params, water_params)`

Task: Create Blender mesh from DEM and water mask.

KEY: Mesh is identical for all views and cached at geometry level,
not per-view. Different camera angles reuse same mesh.

Cache key includes ALL upstream parameters per dependency graph:
- DEM source hash (load_dem)
- Transform params (apply_transforms)
- Water params (detect_water)
- Mesh params (this task)

Args:
    scale_factor: XY scaling
    height_scale: Z scaling for height exaggeration
    center_model: Center mesh at origin
    boundary_extension: Extend boundary for better rendering
    transform_params: Upstream transform parameters (for cache key)
    water_params: Upstream water detection parameters (for cache key)

Returns:
    Blender mesh object

#### `detect_water(self, slope_threshold, fill_holes)`

Task: Detect water bodies using slope analysis on unscaled DEM.

Args:
    slope_threshold: Threshold for slope magnitude (Horn's method)
    fill_holes: Apply morphological smoothing

Returns:
    water_mask (boolean array)

#### `explain(self, task_name)`

Explain what would execute to build a task (show dependency tree).

Shows:
- Task dependencies
- Execution order
- Which tasks would be computed vs cached

#### `load_dem(self, pattern)`

Task: Load and cache raw DEM from SRTM tiles.

Returns:
    (dem_array, affine_transform)

#### `render_all_views(self, views)`

Render all views efficiently.

Builds mesh once, reuses for all views.

Args:
    views: List of view names

Returns:
    Dictionary mapping view names to output paths

#### `render_view(self, view, width, height, distance, elevation, focal_length, camera_type, samples)`

Task: Render a view to PNG.

Args:
    view: Camera direction (north, south, east, west, above)
    width: Output width
    height: Output height
    distance: Camera distance multiplier
    elevation: Camera elevation multiplier
    focal_length: Focal length
    camera_type: PERSP or ORTHO
    samples: Render samples

Returns:
    Path to rendered PNG

#### `visit(task)`

Recursively visit a task and its dependencies for topological sort.

Depth-first traversal that visits all dependencies before the task itself,
building execution order for the task dependency graph.

Args:
    task: Name of the task to visit

---

## terrain.rendering

Rendering operations for Blender terrain visualization.

This module contains functions for configuring Blender render settings
and executing scene rendering.

### Classes

#### `RenderProgressTracker`

Track and report render progress for tiled rendering.

Uses Blender's render handlers to provide progress updates during
long-running renders. Particularly useful with auto-tiling enabled.

**Methods:**

- `register(self)` - Register render progress handlers.

- `unregister(self)` - Unregister render progress handlers.


### Functions

#### `get_render_settings_report()`

Query Blender for the actual render settings used.

Returns a dictionary of all render-relevant settings, useful for
debugging, reproducibility, and verification.

Returns:
    dict: Dictionary containing all render settings from Blender

#### `print_render_settings_report(log)`

Print a formatted report of all Blender render settings.

Queries Blender for actual settings and prints them in a readable format.
Useful for debugging and ensuring settings are correctly applied.

Args:
    log: Logger to use (defaults to module logger)

#### `register(self)`

Register render progress handlers.

#### `render_scene_to_file(output_path, width, height, file_format, color_mode, compression, save_blend_file, show_progress, max_retries, retry_delay)`

Render the current Blender scene to file.

Includes automatic retry logic for GPU memory errors. If rendering fails
due to CUDA/GPU memory exhaustion, the function will wait and retry up to
max_retries times before giving up.

Args:
    output_path (str or Path): Path where output file will be saved
    width (int): Render width in pixels (default: 1920)
    height (int): Render height in pixels (default: 1440)
    file_format (str): Output format 'PNG', 'JPEG', etc. (default: 'PNG')
    color_mode (str): 'RGBA' or 'RGB' (default: 'RGBA')
    compression (int): PNG compression level 0-100 (default: 90)
    save_blend_file (bool): Also save .blend project file (default: True)
    show_progress (bool): Show render progress updates (default: True).
        Logs elapsed time every 5 seconds during rendering.
    max_retries (int): Maximum number of retry attempts for GPU memory
        errors (default: 3). Set to 0 to disable retries.
    retry_delay (float): Seconds to wait between retry attempts (default: 5.0).
        Allows GPU memory to be freed by other processes.

Returns:
    Path: Path to rendered file if successful, None otherwise

#### `setup_render_settings(use_gpu, samples, preview_samples, use_denoising, denoiser, compute_device, use_ambient_occlusion, ao_distance, ao_factor, use_persistent_data, use_auto_tile, tile_size)`

Configure Blender render settings for high-quality terrain visualization.

Args:
    use_gpu: Whether to use GPU acceleration
    samples: Number of render samples
    preview_samples: Number of viewport preview samples
    use_denoising: Whether to enable denoising
    denoiser: Type of denoiser to use ('OPTIX', 'OPENIMAGEDENOISE', 'NLM')
    compute_device: Compute device type ('OPTIX', 'CUDA', 'HIP', 'METAL')
    use_ambient_occlusion: Enable ambient occlusion (darkens crevices)
    ao_distance: AO sampling distance (default: 1.0 Blender units)
    ao_factor: AO strength multiplier (default: 1.0)
    use_persistent_data: Keep scene data in memory between frames (default: False)
    use_auto_tile: Enable automatic tiling for large renders (default: False).
        Splits large images into smaller GPU-friendly tiles to reduce VRAM usage.
        Essential for print-quality renders (3000x2400+ pixels).
    tile_size: Tile size in pixels when auto_tile is enabled (default: 2048).
        Smaller tiles = less VRAM but slower rendering. Try 512-1024 for limited VRAM.

#### `unregister(self)`

Unregister render progress handlers.

---

## terrain.roads

Road network visualization for terrain rendering using data layer pipeline.

This module provides functions to render road networks by rasterizing them as
a data layer that flows through the same transformation pipeline as elevation
and score data. Roads are treated as a proper geographic data layer with
coordinate system and transform, ensuring correct alignment when downsampling
or reprojecting.

Much more efficient than creating individual Blender objects - rasterizes
roads in ~5 seconds and automatically handles coordinate transformations.

Usage - Simple API:
    from src.terrain.roads import add_roads_layer
    from examples.detroit_roads import get_roads

    # After creating terrain:
    roads_geojson = get_roads(bbox)
    add_roads_layer(
        terrain=terrain,
        roads_geojson=roads_geojson,
        bbox=bbox,  # (south, west, north, east) in WGS84
        colormap_name="viridis",
    )

Usage - Manual pipeline:
    from src.terrain.roads import rasterize_roads_to_layer
    from examples.detroit_roads import get_roads

    roads_geojson = get_roads(bbox)
    road_grid, road_transform = rasterize_roads_to_layer(
        roads_geojson, bbox, resolution=30  # 30m pixels
    )

    # Add as data layer (library handles transform automatically)
    terrain.add_data_layer(
        "roads",
        road_grid,
        road_transform,
        "EPSG:4326",
        target_layer="dem",  # Align to DEM grid
    )

### Functions

#### `add_roads_layer(terrain, roads_geojson, bbox, resolution, road_width_pixels)`

Add roads as a data layer to terrain with automatic coordinate alignment.

This is the high-level API for road integration. Roads are rasterized to
a grid with proper geographic metadata and added as a data layer. The library's
data layer pipeline ensures proper alignment even if terrain is downsampled
or reprojected.

To color roads, use the multi-overlay color mapping system:

    roads_geojson = get_roads(bbox)
    terrain.add_roads_layer(terrain, roads_geojson, bbox, road_width_pixels=3)
    terrain.set_multi_color_mapping(
        base_colormap=lambda dem: elevation_colormap(dem, 'michigan'),
        base_source_layers=['dem'],
        overlays=[{
            'colormap': road_colormap,
            'source_layers': ['roads'],
            'priority': 10,
        }],
    )
    terrain.compute_colors()

Args:
    terrain: Terrain object (must have DEM data layer)
    roads_geojson: GeoJSON FeatureCollection with road LineStrings
    bbox: Bounding box as (south, west, north, east) in WGS84 degrees
    resolution: Pixel size in meters for rasterization (default: 30.0)
    road_width_pixels: Width of roads in raster pixels (default: 3). Higher values
        make roads more visible. At 30m resolution, 3 pixels ≈ 90m visual width.

Raises:
    ValueError: If terrain missing DEM data layer

#### `get_viridis_colormap()`

Get viridis colormap function.

Returns:
    Function that maps normalized values (0-1) to (R, G, B)

#### `offset_road_vertices(vertices, road_mask, y_valid, x_valid, offset)`

Offset Z coordinates of mesh vertices that are on roads by a fixed amount.

A simpler alternative to smooth_road_vertices. Raises or lowers all road
vertices by a constant offset, making roads visually distinct from terrain.

Args:
    vertices: Mesh vertex positions (N, 3) array with [x, y, z] coords
    road_mask: 2D array (H, W) where >0.5 indicates road pixels
    y_valid: Array (N,) of y indices mapping vertices to road_mask rows
    x_valid: Array (N,) of x indices mapping vertices to road_mask columns
    offset: Z offset to apply to road vertices. Positive = raise, negative = lower.
            Default: 0.0 (no change)

Returns:
    Modified vertices array with offset Z values on roads.
    X and Y coordinates are never modified.

#### `rasterize_roads_to_layer(roads_geojson, bbox, resolution, road_width_pixels)`

Rasterize GeoJSON roads to a layer grid with proper geographic transform.

Converts vector road data (GeoJSON LineStrings) to a raster grid where each
pixel represents road presence/type. The result includes an Affine transform
in WGS84 (EPSG:4326) coordinates for proper geographic alignment.

This is the key function that treats roads as a data layer - the output
can be added to terrain via add_data_layer() and will automatically align
to the DEM through reprojection and resampling.

Args:
    roads_geojson: GeoJSON FeatureCollection with road LineStrings
    bbox: Bounding box as (south, west, north, east) in WGS84 degrees
    resolution: Pixel size in meters (default: 30.0). At Detroit latitude,
        ~30m/pixel gives good detail without excessive memory use.
    road_width_pixels: Width of roads in raster pixels (default: 3). Draws roads
        with thickness instead of 1-pixel lines. Use 1 for thin roads, 3-5 for
        more visible roads. At 30m resolution, 3 pixels ≈ 90m visual width.

Returns:
    Tuple of:
    - road_grid: 2D array of uint8, values 0=no road, 1-4=road type (motorway > trunk > primary > secondary)
    - road_transform: rasterio.Affine transform mapping pixels to WGS84 coordinates

#### `road_colormap(road_grid, score)`

Map roads to a distinctive red color for special material treatment.

The red color (180, 30, 30) is used as a marker so the Blender material
shader can identify road pixels and apply a glassy/emissive effect.

Args:
    road_grid: 2D array of road values (0=no road, >0=road)
    score: Unused, kept for API compatibility.

Returns:
    Array of RGB colors with shape (height, width, 3) as uint8

#### `smooth_dem_along_roads(dem, road_mask, smoothing_radius)`

Smooth the DEM along roads to reduce elevation detail.

Applies Gaussian smoothing only to road pixels. Non-road pixels
are unchanged. The smoothing kernel should be about half the road width.

Args:
    dem: 2D array of elevation values
    road_mask: 2D boolean or float array where >0.5 = road
    smoothing_radius: Radius for Gaussian smoothing (default: 2 pixels)

Returns:
    Smoothed DEM array (same shape as input)

#### `smooth_road_mask(road_mask, sigma)`

Apply Gaussian blur to road mask for anti-aliased edges.

The Bresenham line algorithm creates stair-step (aliased) edges.
Applying Gaussian smoothing creates soft anti-aliased boundaries that
render more smoothly, especially after the mask goes through resampling.

Args:
    road_mask: 2D array of road values (0=no road, >0=road)
    sigma: Gaussian blur sigma in pixels (default: 1.0).
        Higher values = softer edges. Typical range: 0.5-2.0.
        - 0.5: Minimal softening
        - 1.0: Standard anti-aliasing (recommended)
        - 2.0: Very soft/blurry edges

Returns:
    Smoothed road mask as float32 array. Values are now continuous
    (not binary) and may need thresholding if binary mask is needed.

#### `smooth_road_vertices(vertices, road_mask, y_valid, x_valid, smoothing_radius)`

Smooth Z coordinates of mesh vertices that are on roads.

This function operates on mesh vertices directly after mesh creation,
avoiding the coordinate alignment issues of DEM-based smoothing.

Args:
    vertices: Mesh vertex positions (N, 3) array with [x, y, z] coords
    road_mask: 2D array (H, W) where >0.5 indicates road pixels
    y_valid: Array (N,) of y indices mapping vertices to road_mask rows
    x_valid: Array (N,) of x indices mapping vertices to road_mask columns
    smoothing_radius: Gaussian smoothing sigma (default: 2, use 0 to disable)

Returns:
    Modified vertices array with smoothed Z values on roads.
    X and Y coordinates are never modified.

---

## terrain.scene_setup

Scene setup operations for Blender terrain visualization.

This module contains functions for setting up Blender scenes, cameras,
lighting, and atmosphere for terrain rendering.

### Functions

#### `calculate_camera_frustum_size(camera_type, aspect_ratio, ortho_scale, fov_degrees, distance)`

Calculate the visible area of a camera at a given distance.

Computes the width and height of the camera's frustum (visible area) at a
specified distance. Works with both orthographic and perspective cameras.

For orthographic cameras, the frustum size depends on the ortho_scale parameter.
For perspective cameras, the frustum size depends on the FOV and distance from
the camera.

Args:
    camera_type: Type of camera - "ORTHO" for orthographic or "PERSP" for perspective
    aspect_ratio: Render aspect ratio (width / height, typically 16/9 or similar)
    ortho_scale: Scale value for orthographic cameras (required for ORTHO type)
    fov_degrees: Field of view in degrees for perspective cameras (required for PERSP type)
    distance: Distance from camera for frustum calculation (required for PERSP type)

Returns:
    tuple: (width, height) of the camera frustum in Blender units

Raises:
    ValueError: If camera_type is invalid or required parameters are missing
    TypeError: If parameters have incorrect types

Examples:
    >>> # Orthographic camera with 2x ortho_scale and 16:9 aspect ratio
    >>> w, h = calculate_camera_frustum_size("ORTHO", 16/9, ortho_scale=2.0)
    >>> print(f"Width: {w:.2f}, Height: {h:.2f}")
    Width: 2.00, Height: 1.12

    >>> # Perspective camera with 49.13° FOV at 10 units distance
    >>> w, h = calculate_camera_frustum_size("PERSP", 16/9, fov_degrees=49.13, distance=10.0)
    >>> # Result width and height depend on FOV and distance

#### `clear_scene()`

Clear all objects from the Blender scene.

Resets the scene to factory settings (empty scene) and removes all default
objects. Useful before importing terrain meshes to ensure a clean workspace.

Raises:
    RuntimeError: If Blender module (bpy) is not available.

#### `create_background_plane(camera, mesh_or_meshes, distance_below, color, size_multiplier, receive_shadows, flat_color)`

Create a background plane for Blender terrain renders.

Creates a plane mesh positioned below the terrain that fills the camera view.
The plane is sized to fill the camera's frustum with a safety margin and
positioned below the lowest point of the terrain mesh(es).

This is useful for adding a clean background color to terrain renders without
drop shadows (by default) or with shadows for depth effect.

Args:
    camera: Blender camera object to size plane relative to
    mesh_or_meshes: Single mesh object or list of mesh objects to position
        plane below. The plane will be positioned below the lowest Z point.
    distance_below: Distance below the lowest mesh point to place the plane
        (default: 50.0 units)
    color: Color for the background plane as hex string (e.g., "#F5F5F0") or
        RGB tuple (default: eggshell white #F5F5F0)
    size_multiplier: How much larger than camera frustum to make the plane,
        for safety margin (default: 2.0, makes plane 2x frustum size)
    receive_shadows: Whether the plane receives shadows from objects
        (default: False for clean background)
    flat_color: If True, use emission shader for exact color that ignores
        scene lighting. If False (default), use Principled BSDF that responds
        to lighting (darker colors may appear lighter due to ambient light).

Returns:
    Blender plane object with material applied and positioned

Raises:
    ValueError: If camera or mesh is invalid
    RuntimeError: If called outside of Blender environment

#### `create_matte_material(name, color, material_roughness, receive_shadows, flat_color)`

Create a matte material for backgrounds.

Creates either a physically-based matte material or a flat emission material.
The flat option is useful when you want an exact color that doesn't respond
to scene lighting (e.g., for studio-style backgrounds).

Args:
    name: Name for the material (default: "BackgroundMaterial")
    color: Color as hex string (e.g., "#F5F5F0") or RGB tuple (default: eggshell white)
    material_roughness: Roughness value 0.0-1.0, 1.0 = fully matte (default: 1.0).
        Only used when flat_color=False.
    receive_shadows: Whether the material receives shadows (default: False).
        Only used when flat_color=False.
    flat_color: If True, use pure emission shader for exact color regardless of
        lighting. The rendered color will match the input color exactly.
        If False (default), use Principled BSDF which responds to lighting.

Returns:
    Blender Material object configured as specified

Note:
    This function requires Blender and the bpy module to be available.
    Call only from within a Blender environment.

Raises:
    RuntimeError: If called outside of Blender environment

#### `hex_to_rgb(hex_color)`

Convert hex color string to normalized RGB tuple.

Converts hex color strings in various formats (#RRGGBB, #RGB, with or without #)
to normalized RGB tuples with values in range 0.0-1.0.

Args:
    hex_color: Hex color string in format:
        - "#RRGGBB" (e.g., "#F5F5F0")
        - "#RGB" (e.g., "#FFF")
        - "RRGGBB" or "RGB" (without #)

Returns:
    Tuple of (r, g, b) floats in range 0.0-1.0

Raises:
    ValueError: If hex_color format is invalid or contains invalid characters

Examples:
    >>> r, g, b = hex_to_rgb("#F5F5F0")  # Eggshell white
    >>> r, g, b = hex_to_rgb("FFF")      # Pure white
    >>> r, g, b = hex_to_rgb("#000000")  # Pure black

#### `position_camera_relative(mesh_obj, direction, distance, elevation, look_at, camera_type, sun_angle, sun_energy, sun_azimuth, sun_elevation, focal_length, ortho_scale)`

Position camera relative to mesh(es) using intuitive cardinal directions.

Simplifies camera positioning by using natural directions (north, south, etc.)
instead of absolute Blender coordinates. The camera is automatically positioned
relative to the mesh bounds and rotated to point at the mesh center.

Supports multiple meshes by computing a combined bounding box that encompasses
all provided mesh objects. This is useful for dual terrain renders or scenes
with multiple terrain meshes that need to be viewed together.

Args:
    mesh_obj: Blender mesh object or list of mesh objects to position camera
        relative to. If a list is provided, a combined bounding box is computed.
    direction: Cardinal direction - one of:
        'north', 'south', 'east', 'west' (horizontal directions)
        'northeast', 'northwest', 'southeast', 'southwest' (diagonals)
        'above' (directly overhead)
        Default: 'south'
    distance: Distance multiplier relative to mesh diagonal
        (e.g., 1.5 means 1.5x mesh_diagonal away). Default: 1.5
    elevation: Height as fraction of mesh diagonal added to Z position
        (0.0 = ground level, 1.0 = mesh_diagonal above ground). Default: 0.5
    look_at: Where camera points - 'center' to point at mesh center,
        or tuple (x, y, z) for custom target. Default: 'center'
    camera_type: 'ORTHO' (orthographic) or 'PERSP' (perspective). Default: 'ORTHO'
    sun_angle: Angular diameter of sun in degrees (affects shadow softness). Default: 0 (no light)
    sun_energy: Intensity of sun light. Default: 0 (no light created unless > 0)
    sun_azimuth: Direction sun comes FROM in degrees (0=North, 90=East, 180=South, 270=West)
    sun_elevation: Angle of sun above horizon in degrees (0=horizon, 90=overhead)
    focal_length: Camera focal length in mm (perspective cameras only). Default: 50
    ortho_scale: Multiplier for orthographic camera scale relative to mesh diagonal.
        Higher values zoom out (show more area), lower values zoom in.
        Only affects orthographic cameras. Default: 1.2

Returns:
    Camera object

Raises:
    ValueError: If direction is not recognized or camera_type is invalid

#### `setup_camera(camera_angle, camera_location, scale, focal_length, camera_type)`

Configure camera for terrain visualization.

Args:
    camera_angle: Tuple of (x,y,z) rotation angles in radians
    camera_location: Tuple of (x,y,z) camera position
    scale: Camera scale value (ortho_scale for orthographic cameras)
    focal_length: Camera focal length in mm (default: 50, used only for perspective)
    camera_type: Camera type 'PERSP' (perspective) or 'ORTHO' (orthographic) (default: 'PERSP')

Returns:
    Camera object

Raises:
    ValueError: If camera_type is not 'PERSP' or 'ORTHO'

#### `setup_camera_and_light(camera_angle, camera_location, scale, sun_angle, sun_energy, focal_length, camera_type)`

Configure camera and main light for terrain visualization.

Convenience function that calls setup_camera() and setup_light().

Args:
    camera_angle: Tuple of (x,y,z) rotation angles in radians
    camera_location: Tuple of (x,y,z) camera position
    scale: Camera scale value (ortho_scale for orthographic cameras)
    sun_angle: Angle of sun light in degrees (default: 2)
    sun_energy: Energy/intensity of sun light (default: 3)
    focal_length: Camera focal length in mm (default: 50, used only for perspective)
    camera_type: Camera type 'PERSP' (perspective) or 'ORTHO' (orthographic) (default: 'PERSP')

Returns:
    tuple: (camera object, sun light object)

#### `setup_hdri_lighting(sun_elevation, sun_rotation, sun_intensity, sun_size, air_density, visible_to_camera, camera_background, sky_strength)`

Set up HDRI-style sky lighting using Blender's Nishita sky model.

Creates realistic sky lighting that contributes to ambient illumination
without being visible in the final render (by default).

The Nishita sky model provides physically-based atmospheric scattering
for natural-looking outdoor lighting.

Args:
    sun_elevation: Sun elevation angle in degrees (0=horizon, 90=overhead)
    sun_rotation: Sun rotation/azimuth in degrees (0=front, 180=back)
    sun_intensity: Multiplier for sun disc brightness in sky texture (default: 1.0)
    sun_size: Angular diameter of sun disc in degrees (default: 0.545 = real sun).
              Larger values create softer shadows, smaller values create sharper shadows.
    air_density: Atmospheric density (default: 1.0, higher=hazier)
    visible_to_camera: If False, sky is invisible but still lights scene
    camera_background: RGB tuple for background color when sky is invisible.
                      Default None = use transparent (black behind scene).
                      Use (0.9, 0.9, 0.9) for light gray if using atmosphere.
    sky_strength: Overall sky emission strength (ambient light level).
                 If None, defaults to sun_intensity for backwards compatibility.

Returns:
    bpy.types.World: The configured world object

#### `setup_light(location, angle, energy, rotation_euler, azimuth, elevation)`

Create and configure sun light for terrain visualization.

Sun position can be specified either with rotation_euler (raw Blender angles)
or with the more intuitive azimuth/elevation system:

- azimuth: Direction the sun comes FROM, in degrees clockwise from North
  (0=North, 90=East, 180=South, 270=West)
- elevation: Angle above horizon in degrees (0=horizon, 90=directly overhead)

If azimuth and elevation are provided, they override rotation_euler.

Args:
    location: Tuple of (x,y,z) light position (default: (1, 1, 2))
    angle: Angular diameter of sun in degrees (default: 2, affects shadow softness)
    energy: Energy/intensity of sun light (default: 3)
    rotation_euler: Tuple of (x,y,z) rotation angles in radians (legacy, use azimuth/elevation)
    azimuth: Direction sun comes FROM in degrees (0=North, 90=East, 180=South, 270=West)
    elevation: Angle above horizon in degrees (0=horizon, 90=overhead)

Returns:
    Sun light object

#### `setup_two_point_lighting(sun_azimuth, sun_elevation, sun_energy, sun_angle, sun_color, fill_azimuth, fill_elevation, fill_energy, fill_angle, fill_color)`

Set up two-point lighting with primary sun and optional fill light.

Creates professional-quality lighting for terrain visualization:
- Primary sun: Creates shadows and defines form (warm color by default)
- Fill light: Softens shadows, adds depth (cool color by default)

The warm/cool color contrast creates a natural outdoor lighting look
similar to golden hour photography.

Args:
    sun_azimuth: Direction sun comes FROM in degrees (0=N, 90=E, 180=S, 270=W).
        Default: 225° (southwest, afternoon sun)
    sun_elevation: Sun angle above horizon in degrees (0=horizon, 90=overhead).
        Default: 30° (mid-afternoon)
    sun_energy: Sun light strength. Default: 7.0
    sun_angle: Sun angular size in degrees (smaller=sharper shadows).
        Default: 1.0°
    sun_color: RGB tuple for sun color. Default: (1.0, 0.85, 0.6) warm golden
    fill_azimuth: Direction fill light comes FROM in degrees.
        Default: 45° (northeast, opposite sun)
    fill_elevation: Fill light angle above horizon in degrees.
        Default: 60° (higher angle for even fill)
    fill_energy: Fill light strength. Default: 0.0 (no fill light).
        Set to ~1-3 for subtle fill, ~5+ for strong fill.
    fill_angle: Fill light angular size in degrees.
        Default: 3.0° (softer than sun)
    fill_color: RGB tuple for fill color. Default: (0.7, 0.8, 1.0) cool blue

Returns:
    List of created light objects (1-2 lights depending on fill_energy)

Examples:
    >>> # Basic sun-only lighting
    >>> lights = setup_two_point_lighting(sun_azimuth=180, sun_elevation=45)

    >>> # Sun with fill for softer shadows
    >>> lights = setup_two_point_lighting(
    ...     sun_azimuth=225, sun_elevation=30, sun_energy=7,
    ...     fill_energy=2, fill_azimuth=45, fill_elevation=60
    ... )

    >>> # Low sun for dramatic shadows
    >>> lights = setup_two_point_lighting(sun_elevation=10)

#### `setup_world_atmosphere(density, scatter_color, anisotropy)`

Set up world volume for atmospheric effects.

This function is additive - it preserves any existing Surface shader
(like HDRI lighting) and only adds a Volume shader for atmospheric fog.

Note: Density is per-Blender-unit. For terrain scenes that are 100-500 units
across, use very low values (0.0001-0.001). Higher values will make the
scene very dark or completely black.

Args:
    density: Density of the atmospheric volume (default: 0.0002, very subtle)
             For stronger fog: 0.001. For barely visible haze: 0.0001
    scatter_color: RGBA color tuple for scatter (default: white)
    anisotropy: Direction of scatter from -1 to 1 (default: 0.8 for forward scatter,
                creates sun halo effect similar to real atmosphere)

Returns:
    bpy.types.World: The configured world object

---

## terrain.scoring

Scoring functions for terrain suitability analysis.

This module contains scoring functions for evaluating terrain suitability
for various activities like sledding and cross-country skiing.

### Functions

#### `compute_sledding_score(snow_depth, slope, coverage_months, roughness)`

Compute overall sledding suitability score.

Combines multiple factors using a multiplicative model where:
1. Deal breakers → immediate zero score
2. Base score = snow_score × slope_score × coverage_score
3. Final score = base_score × synergy_bonus

The multiplicative approach ensures that poor performance in any
one factor significantly reduces the overall score, while synergies
can boost exceptional combinations.

Args:
    snow_depth: Snow depth in inches (scalar or array)
    slope: Terrain slope in degrees (scalar or array)
    coverage_months: Months of snow coverage (scalar or array)
    roughness: Elevation std dev in meters (scalar or array) - physical terrain roughness

Returns:
    Score(s) in range [0, ~1.5], same shape as inputs
    (Can exceed 1.0 due to synergy bonuses)

Example:
    >>> # Perfect conditions
    >>> compute_sledding_score(12.0, 9.0, 4.0, 2.0)
    1.12  # High score with bonuses
    >>> # Deal breaker (too steep)
    >>> compute_sledding_score(12.0, 45.0, 4.0, 2.0)
    0.0  # Too steep (>40°)
    >>> # Deal breaker (too rough)
    >>> compute_sledding_score(12.0, 10.0, 4.0, 8.0)
    0.0  # Too rough (>6m)

#### `compute_xc_skiing_score(snow_depth, snow_coverage, snow_consistency, min_depth, optimal_depth_min, optimal_depth_max, max_depth, min_coverage)`

Compute overall cross-country skiing suitability score.

XC skiing scoring focuses on snow conditions (parks handle terrain safety):
- Snow depth trapezoid (30% weight): optimal 100-400mm, usable 50-800mm
- Snow coverage linear (60% weight): proportional to days with snow
- Snow consistency inverted (10% weight): low CV = reliable

Deal breaker:
- Snow coverage < 15% (< ~18 days per season) → Score = 0

Final score combines:
- Base score: Weighted sum of depth, coverage, consistency
- Range: 0 (poor) to 1.0 (excellent)

Args:
    snow_depth: Snow depth in mm (SNODAS native units)
    snow_coverage: Fraction of days with snow (0-1)
    snow_consistency: Coefficient of variation (lower is better, 0-1.5)
    min_depth: Minimum usable snow depth (mm)
    optimal_depth_min: Lower bound of optimal depth range (mm)
    optimal_depth_max: Upper bound of optimal depth range (mm)
    max_depth: Maximum usable snow depth (mm)
    min_coverage: Minimum snow coverage threshold (15% = 0.15)

Returns:
    XC skiing suitability score (0-1)

Examples:
    >>> compute_xc_skiing_score(250.0, 0.75, 0.3)
    0.78  # Excellent conditions

    >>> compute_xc_skiing_score(150.0, 0.1, 0.5)
    0.0  # Deal breaker - coverage too low

    >>> compute_xc_skiing_score(100.0, 0.5, 0.8)
    0.53  # Good enough conditions

#### `coverage_diminishing_returns(coverage_months, tau)`

Score snow coverage with diminishing returns.

Some coverage is critical, but beyond a certain point more coverage
doesn't add much value. Uses exponential saturation:
    score = 1 - exp(-coverage_months / tau)

This gives:
- 0 months: 0.0
- 1 month: ~0.39
- 2 months: ~0.63
- 4 months: ~0.86
- 6 months: ~0.95

Args:
    coverage_months: Months of snow coverage (scalar or array)
    tau: Time constant for saturation (default: 2.0 months)

Returns:
    Score(s) in range [0, 1), same shape as coverage_months

Example:
    >>> coverage_diminishing_returns(2.0)
    0.632  # Two months gives ~63% score
    >>> coverage_diminishing_returns(6.0)
    0.950  # Six months gives ~95% score

#### `sledding_deal_breakers(slope, roughness, coverage_months, max_slope, max_roughness, min_coverage)`

Identify deal breaker conditions for sledding.

Some terrain features are absolute deal breakers for sledding:
- Slope > 40°: Extreme cliffs (double-black-diamond terrain)
- Roughness > 6m: Very rough terrain (cliff faces, boulders)
- Coverage < 0.5 months: Not enough snow season

Args:
    slope: Terrain slope in degrees (scalar or array)
    roughness: Elevation std dev in meters (scalar or array) - physical terrain roughness
    coverage_months: Months of snow coverage (scalar or array)
    max_slope: Maximum acceptable slope (default: 40°)
    max_roughness: Maximum acceptable roughness in meters (default: 6.0m)
    min_coverage: Minimum acceptable coverage months (default: 0.5)

Returns:
    Boolean array indicating deal breaker locations (True = deal breaker)

Example:
    >>> sledding_deal_breakers(45.0, 3.0, 3.0)
    True  # Too steep (>40°)
    >>> sledding_deal_breakers(10.0, 2.0, 3.0)
    False  # Acceptable
    >>> sledding_deal_breakers(15.0, 8.0, 3.0)
    True  # Too rough (>6m)

#### `sledding_synergy_bonus(slope, snow_depth, coverage_months, roughness)`

Compute synergy bonuses for exceptional sledding combinations.

Some combinations of factors create exceptional sledding that's more
than the sum of its parts. This function identifies these synergies
and applies multiplicative bonuses.

Synergies (applied in hierarchical priority order):
1. Perfect combo (slope + snow + coverage) = +30% (highest priority)
2. Consistent coverage + good slope + smooth terrain = +15%
3. Moderate slope + very smooth terrain = +20%
4. Smooth terrain + perfect slope = +10% (lowest priority)

Higher priority bonuses prevent lower priority bonuses from applying
to avoid over-rewarding.

Args:
    slope: Terrain slope in degrees (scalar or array)
    snow_depth: Snow depth in inches (scalar or array)
    coverage_months: Months of snow coverage (scalar or array)
    roughness: Elevation std dev in meters (scalar or array) - physical terrain roughness

Returns:
    Bonus multiplier(s) >= 1.0, same shape as inputs

Example:
    >>> sledding_synergy_bonus(9.0, 12.0, 3.5, 2.0)
    1.30  # Perfect combo bonus only
    >>> sledding_synergy_bonus(9.0, 12.0, 4.0, 1.5)
    1.65  # Perfect combo + smooth terrain bonuses stack
    >>> sledding_synergy_bonus(15.0, 5.0, 1.0, 5.0)
    1.0  # No synergies

#### `trapezoid_score(value, min_value, optimal_min, optimal_max, max_value)`

Compute trapezoid (sweet spot) scoring for a parameter.

The trapezoid function creates a sweet spot scoring pattern:
- Below min_value: score = 0
- Between min_value and optimal_min: linear ramp from 0 to 1
- Between optimal_min and optimal_max: score = 1 (optimal range)
- Between optimal_max and max_value: linear ramp from 1 to 0
- Above max_value: score = 0

This is useful for parameters where there's a "just right" range and
both too little and too much are bad.

Args:
    value: The value(s) to score (scalar or array)
    min_value: Minimum acceptable value (below this scores 0)
    optimal_min: Start of optimal range (scores 1.0)
    optimal_max: End of optimal range (scores 1.0)
    max_value: Maximum acceptable value (above this scores 0)

Returns:
    Score(s) in range [0, 1], same shape as value

Example:
    >>> # Snow depth scoring: 4-8-16-24 inches
    >>> trapezoid_score(12.0, 4.0, 8.0, 16.0, 24.0)
    1.0  # In optimal range
    >>> trapezoid_score(6.0, 4.0, 8.0, 16.0, 24.0)
    0.5  # Halfway up ramp

#### `xc_skiing_deal_breakers(snow_coverage, min_coverage)`

Identify deal breaker conditions for cross-country skiing.

XC skiing depends primarily on snow reliability. Parks handle terrain safety,
so only snow coverage matters as a deal breaker.

Deal breaker:
- Snow coverage < 15% of days (< ~18 days per winter season)

Args:
    snow_coverage: Fraction of days with snow (0-1)
    min_coverage: Minimum snow coverage threshold (default 0.15)

Returns:
    Boolean or array of booleans indicating deal breaker conditions

Examples:
    >>> xc_skiing_deal_breakers(0.5)
    False  # 50% coverage is good

    >>> xc_skiing_deal_breakers(0.1)
    True  # Only 10% coverage - too unreliable

    >>> xc_skiing_deal_breakers(np.array([0.1, 0.3, 0.8]))
    array([True, False, False])

---

## terrain.transforms

Raster transformation operations for terrain processing.

This module contains functions for transforming raster data including
downsampling, smoothing, flipping, and elevation scaling.

### Functions

#### `cached_reproject(src_crs, dst_crs, cache_dir, nodata_value, num_threads)`

Cached raster reprojection - saves reprojected DEM to disk for instant reuse.

First call computes and caches the reprojection (~24s). Subsequent calls
load from cache (~0.5s). Cache is keyed by CRS pair and source data hash.

Args:
    src_crs: Source coordinate reference system
    dst_crs: Destination coordinate reference system
    cache_dir: Directory to store cached reprojections
    nodata_value: Value to use for areas outside original data
    num_threads: Number of threads for parallel processing

Returns:
    Function that transforms data and returns (data, transform, new_crs)

Example:
    >>> # First run: ~24s (computes and caches)
    >>> terrain.add_transform(cached_reproject(src_crs="EPSG:4326", dst_crs="EPSG:32617"))
    >>> # Second run: ~0.5s (loads from cache)

#### `despeckle_dem(nodata_value, kernel_size)`

Create a transform that removes isolated elevation noise using median filtering.

Unlike bilateral smoothing (--smooth) which preserves edges but can look patchy,
median filtering uniformly removes local outliers/speckles across the entire DEM.
This is better for removing sensor noise or small DEM artifacts.

For smarter frequency-aware denoising that preserves terrain structure,
use wavelet_denoise_dem() instead.

Args:
    nodata_value: Value to treat as no data (default: np.nan)
    kernel_size: Size of median filter kernel (default: 3 for 3x3).
        Must be odd integer ≥3. Larger = more smoothing.
        - 3: Removes single-pixel noise (recommended)
        - 5: Removes 2x2 artifacts
        - 7: Stronger smoothing, may affect small terrain features

Returns:
    function: A transform function for use with terrain.add_transform()

Example:
    >>> terrain.add_transform(despeckle_dem(kernel_size=3))

#### `despeckle_scores(scores, kernel_size)`

Remove isolated speckles from score data using median filtering.

Unlike bilateral filtering which preserves edges, median filtering
replaces each pixel with the median of its neighborhood. This effectively
removes isolated outlier pixels (speckles) while preserving larger regions.

Use case: SNODAS snow data upsampled to high-res DEM often has isolated
low-score pixels (speckles) in otherwise high-score regions due to
resolution mismatch. These appear as visual noise in the rendered terrain.

Args:
    scores: 2D array of score values (typically 0-1 range)
    kernel_size: Size of median filter kernel (default: 3 for 3x3).
        Larger kernels remove larger speckle clusters but may affect
        legitimate small features. Common values: 3, 5, 7.

Returns:
    Despeckled score array with same shape as input.
    NaN values are preserved.

Example:
    >>> # Remove single-pixel speckles
    >>> despeckled = despeckle_scores(scores, kernel_size=3)
    >>> # Remove up to 2x2 speckle clusters
    >>> despeckled = despeckle_scores(scores, kernel_size=5)

#### `downsample_raster(zoom_factor, method, nodata_value)`

Create a raster downsampling transform function with specified parameters.

Args:
    zoom_factor: Scaling factor for downsampling (default: 0.1)
    method: Downsampling method (default: "average")
        - "average": Area averaging - best for DEMs, no overshoot
        - "lanczos": Lanczos resampling - sharp, minimal aliasing
        - "cubic": Cubic spline interpolation
        - "bilinear": Bilinear interpolation - safe fallback
    nodata_value: Value to treat as no data (default: np.nan)

Returns:
    function: A transform function that downsamples raster data

#### `feature_preserving_smooth(sigma_spatial, sigma_intensity, nodata_value)`

Create a feature-preserving smoothing transform using bilateral filtering.

Removes high-frequency noise while preserving ridges, valleys, and drainage patterns.
Uses bilateral filtering: spatial Gaussian weighted by intensity similarity.

Args:
    sigma_spatial: Spatial smoothing extent in pixels (default: 3.0).
        Larger = more smoothing. Typical range: 1-10 pixels.
    sigma_intensity: Intensity similarity threshold in elevation units.
        Larger = more smoothing across elevation differences.
        If None, auto-calculated as 5% of elevation range.
    nodata_value: Value to treat as no data (default: np.nan)

Returns:
    function: Transform function compatible with terrain.add_transform()

#### `flip_raster(axis)`

Create a transform function that mirrors (flips) the DEM data.
If axis='horizontal', it flips top ↔ bottom.
(In terms of rows, row=0 becomes row=(height-1).)

If axis='vertical', you could do left ↔ right (np.fliplr).

#### `remove_bumps(kernel_size, structure, strength)`

Remove local maxima (bumps) from DEM using morphological opening.

Morphological opening = erosion followed by dilation. This operation:
- Removes small bright features (buildings, trees, noise)
- Never creates new local maxima (mathematically guaranteed)
- Preserves larger terrain features and overall shape
- Leaves valleys and depressions untouched

This is the standard approach for "removing buildings from DEMs" in
geospatial processing.

Args:
    kernel_size: Size of the structuring element (default: 3).
        Controls the maximum size of bumps to remove:
        - 1: Removes features up to ~2 pixels across (very subtle)
        - 3: Removes features up to ~6 pixels across
        - 5: Removes features up to ~10 pixels across
        For 30m DEMs, size=3 removes ~180m features
    structure: Shape of structuring element (default: "disk").
        - "disk": Circular, isotropic (recommended)
        - "square": Faster but may create artifacts on diagonals
    strength: Blend factor between original and opened result (default: 1.0).
        - 0.0: No effect (returns original)
        - 0.5: Half the bump removal effect (subtle)
        - 1.0: Full bump removal (original behavior)
        Values between 0 and 1 provide fine-grained control.

Returns:
    function: A transform function for use with terrain.add_transform()

Example:
    >>> # Remove small bumps (buildings on 30m DEM)
    >>> terrain.add_transform(remove_bumps(kernel_size=3))

    >>> # Subtle bump reduction (50% strength)
    >>> terrain.add_transform(remove_bumps(kernel_size=1, strength=0.5))

    >>> # More aggressive bump removal
    >>> terrain.add_transform(remove_bumps(kernel_size=5))

#### `reproject_raster(src_crs, dst_crs, nodata_value, num_threads)`

Generalized raster reprojection function

Args:
    src_crs: Source coordinate reference system
    dst_crs: Destination coordinate reference system
    nodata_value: Value to use for areas outside original data
    num_threads: Number of threads for parallel processing

Returns:
    Function that transforms data and returns (data, transform, new_crs)

#### `scale_elevation(scale_factor, nodata_value)`

Create a raster elevation scaling transform function.

Multiplies all elevation values by the scale factor. Useful for reducing
or amplifying terrain height without changing horizontal scale.

Args:
    scale_factor (float): Multiplication factor for elevation values (default: 1.0)
    nodata_value: Value to treat as no data (default: np.nan)

Returns:
    function: A transform function that scales elevation data

#### `slope_adaptive_smooth(slope_threshold, smooth_sigma, transition_width, nodata_value, elevation_scale, edge_threshold, edge_window, strength)`

Create a transform that smooths flat areas more aggressively than hilly areas.

This addresses the problem of buildings/structures appearing as bumps in
flat regions. Flat areas get strong Gaussian smoothing to remove these
artifacts, while slopes and hills are preserved with minimal smoothing.

How it works:
1. Compute local slope at each pixel using gradient magnitude
2. Create a smooth weight mask: 1.0 (full smoothing) where flat,
   0.0 (no smoothing) where steep
3. Apply Gaussian blur to entire DEM
4. Blend: output = original * (1-weight) + smoothed * weight

The transition from "flat" to "steep" is smooth (using sigmoid) to avoid
visible boundaries in the output.

Args:
    slope_threshold: Slope angle in degrees below which terrain is
        considered "flat" (default: 2.0 degrees).
        - 1.0°: Very aggressive, only smooths nearly horizontal areas
        - 2.0°: Good default, smooths typical flat areas with buildings
        - 5.0°: Smooths gentle slopes too
    smooth_sigma: Gaussian blur sigma in pixels (default: 5.0).
        Controls the strength of smoothing in flat areas.
        - 3.0: Light smoothing
        - 5.0: Moderate smoothing (recommended)
        - 10.0: Very strong smoothing, may blur valid terrain features
    transition_width: Width of transition zone in degrees (default: 1.0).
        Controls how quickly smoothing fades off above threshold.
        - 0.5: Sharp transition
        - 1.0: Smooth transition (recommended)
        - 2.0: Very gradual transition
    nodata_value: Value to treat as no data (default: np.nan)
    elevation_scale: Scale factor that was applied to elevation data (default: 1.0).
        If elevation was scaled (e.g., by scale_elevation(0.0001)), pass that
        factor here so slope computation uses real-world elevation differences.
        The gradient is divided by this factor to recover true slopes.
    edge_threshold: Elevation difference threshold for edge preservation (default: None).
        If set, sharp elevation discontinuities (like lake boundaries) are preserved.
        Areas where local elevation range exceeds this threshold are not smoothed.
        - None: Disabled (original behavior)
        - 5.0: Preserve edges with >5m elevation change
        - 10.0: Only preserve very sharp edges (>10m change)
        Recommended: 3-10m depending on terrain features to preserve.
    edge_window: Window size for edge detection (default: 5).
        Larger windows detect edges over broader areas but may over-protect.
    strength: Overall smoothing strength multiplier (default: 1.0).
        Scales the maximum smoothing effect in flat areas.
        - 1.0: Full smoothing (original behavior)
        - 0.5: Half the smoothing effect
        - 0.25: Gentle smoothing
        - 0.0: No smoothing (transform has no effect)

Returns:
    function: A transform function for use with terrain.add_transform()

Example:
    >>> # Standard: smooth flat areas with >2° slopes preserved
    >>> terrain.add_transform(slope_adaptive_smooth())

    >>> # Aggressive: smooth anything below 5° slope
    >>> terrain.add_transform(slope_adaptive_smooth(slope_threshold=5.0, smooth_sigma=8.0))

    >>> # Conservative: only smooth very flat areas, light blur
    >>> terrain.add_transform(slope_adaptive_smooth(slope_threshold=1.0, smooth_sigma=3.0))

    >>> # Compensate for prior scale_elevation(0.0001)
    >>> terrain.add_transform(slope_adaptive_smooth(elevation_scale=0.0001))

    >>> # Preserve lake boundaries and other sharp edges
    >>> terrain.add_transform(slope_adaptive_smooth(edge_threshold=5.0))

    >>> # Gentle smoothing (25% of full effect)
    >>> terrain.add_transform(slope_adaptive_smooth(strength=0.25))

#### `smooth_raster(window_size, nodata_value)`

Create a raster smoothing transform function with specified parameters.

Args:
    window_size: Size of median filter window
                (defaults to 5% of smallest dimension if None)
    nodata_value: Value to treat as no data (default: np.nan)

Returns:
    function: A transform function that smooths raster data

#### `smooth_score_data(scores, sigma_spatial, sigma_intensity)`

Smooth score data using bilateral filtering.

Applies feature-preserving smoothing to reduce blocky pixelation from
low-resolution source data (e.g., SNODAS ~925m) when displayed on
high-resolution terrain (~30m DEM).

Uses bilateral filtering: smooths within similar-intensity regions while
preserving edges between different score zones.

Args:
    scores: 2D numpy array of score values (typically 0-1 range)
    sigma_spatial: Spatial smoothing extent in pixels (default: 3.0).
        Larger = more smoothing. Typical range: 1-10 pixels.
    sigma_intensity: Intensity similarity threshold in score units.
        Larger = more smoothing across score differences.
        If None, auto-calculated as 15% of score range (good for 0-1 data).

Returns:
    Smoothed score array with same shape as input.
    NaN values are preserved. Output is clipped to [0, 1] range.

Example:
    >>> # Smooth blocky SNODAS-derived scores
    >>> sledding_scores = load_score_grid("sledding_scores.npz")
    >>> smoothed = smooth_score_data(sledding_scores, sigma_spatial=5.0)

#### `soft_threshold(c, thresh)`

Soft thresholding: shrink coefficients toward zero.

#### `transform(raster_data, affine_transform)`

Apply slope-adaptive smoothing to DEM.

Args:
    raster_data: Input raster numpy array (elevation data)
    affine_transform: Optional affine transform (unchanged by smoothing)

Returns:
    tuple: (smoothed_data, affine_transform, None)

#### `transform_func(data, transform)`

Flip array along specified axis and update transform if provided.

#### `upscale_scores(scores, scale, method, nodata_value)`

Upscale score grid to reduce blockiness when applied to terrain.

Uses AI super-resolution (Real-ESRGAN) when available, falling back to
bilateral upscaling for edge-preserving smoothness.

Args:
    scores: Input score grid (2D numpy array)
    scale: Upscaling factor (default: 4, meaning 4x resolution)
    method: Upscaling method:
        - "auto": Try Real-ESRGAN, fall back to bilateral
        - "esrgan": Use Real-ESRGAN (requires optional realesrgan package)
        - "bilateral": Use bilateral filter upscaling (no extra dependencies)
        - "bicubic": Simple bicubic interpolation
    nodata_value: Value treated as no data (default: np.nan)

Returns:
    Upscaled score grid with smoother gradients

Note:
    The "esrgan" method requires the optional ``realesrgan`` package::

        pip install realesrgan

    Or install terrain-maker with the upscale extra::

        pip install terrain-maker[upscale]

    Without it, "auto" will fall back to "bilateral" which produces
    good results without ML dependencies.

Example:
    >>> scores_hires = upscale_scores(sledding_scores, scale=4)
    >>> # Now scores_hires is 4x the resolution with smoother transitions

    >>> # Force bilateral method (no ML dependencies)
    >>> scores_hires = upscale_scores(scores, scale=4, method="bilateral")

#### `wavelet_denoise_dem(nodata_value, wavelet, levels, threshold_sigma, preserve_structure)`

Create a transform that removes noise while preserving terrain structure.

Uses wavelet decomposition to separate terrain features (ridges, valleys,
drainage patterns) from high-frequency noise. This is smarter than median
filtering because it understands that terrain has structure at certain
spatial frequencies.

How it works:
1. Decompose DEM into frequency bands using wavelets
2. Estimate noise level from finest (highest-frequency) band
3. Apply soft thresholding to remove coefficients below noise threshold
4. Reconstruct DEM from cleaned coefficients

The result preserves terrain structure while removing sensor noise, SRTM
artifacts, and other high-frequency disturbances.

Args:
    nodata_value: Value to treat as no data (default: np.nan)
    wavelet: Wavelet type (default: "db4" - Daubechies 4).
        Options: "db4" (smooth), "haar" (sharp edges), "sym4" (symmetric).
        - "db4": Best for natural terrain (smooth transitions)
        - "haar": Best for urban/artificial structures
        - "sym4": Good balance, symmetric filtering
    levels: Decomposition levels (default: 3). More levels = coarser
        structure preserved. Each level halves the resolution.
        - 2: Preserves finer detail, removes less noise
        - 3: Good balance (recommended)
        - 4: Aggressive smoothing, may blur small features
    threshold_sigma: Noise threshold multiplier (default: 2.0).
        Higher = more aggressive denoising.
        - 1.5: Light denoising, preserves more detail
        - 2.0: Standard denoising (recommended)
        - 3.0: Aggressive, may remove subtle features
    preserve_structure: If True, only denoise highest-frequency band
        to maximize structure preservation. If False, denoise all bands.

Returns:
    function: A transform function for use with terrain.add_transform()

Example:
    >>> # Standard terrain denoising
    >>> terrain.add_transform(wavelet_denoise_dem())

    >>> # Aggressive denoising for very noisy DEM
    >>> terrain.add_transform(wavelet_denoise_dem(threshold_sigma=3.0, levels=4))

    >>> # Light denoising, preserve maximum detail
    >>> terrain.add_transform(wavelet_denoise_dem(threshold_sigma=1.5, levels=2))

---

## terrain.water

Water body detection and identification from elevation data.

Provides functions to identify water bodies from DEM data using slope analysis.
Water is characterized by flat surfaces (low slope), while terrain typically has
higher slope values.

### Functions

#### `identify_water_by_slope(dem_data, slope_threshold, fill_holes)`

Identify water bodies by detecting flat areas (low slope).

Water bodies typically have very flat surfaces with near-zero slope.
This function calculates local slope using Horn's method and identifies pixels
below the threshold as potential water. Optionally applies morphological
operations to fill small gaps and smooth the water mask.

Args:
    dem_data (np.ndarray): Digital elevation model as 2D array (height values)
    slope_threshold (float): Maximum slope magnitude to classify as water.
                           Default: 0.1 (very flat surfaces)
                           Typical range: 0.05 to 0.5 depending on DEM resolution
                           Values are gradient magnitude from Horn's method
    fill_holes (bool): Apply morphological operations to fill small gaps
                      in water mask and smooth boundaries. Default: True

Returns:
    np.ndarray: Boolean mask (dtype=bool) where True = water, False = land.
               Same shape as dem_data.

Raises:
    ValueError: If dem_data is not 2D or slope_threshold is negative

Examples:
    >>> dem = np.array([[100, 100, 110], [100, 100, 110], [110, 110, 120]])
    >>> water_mask = identify_water_by_slope(dem, slope_threshold=0.1)
    >>> water_mask.dtype
    dtype('bool')
    >>> water_mask.shape
    (3, 3)
