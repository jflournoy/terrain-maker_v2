# terrain-maker API Reference

Comprehensive documentation of all classes and functions in the terrain-maker library.

**Auto-generated from source code docstrings.**

## Table of Contents

- [terrain.core](#terrain-core)

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

- `add_data_layer(self, name, data, transform, crs, target_crs, target_layer, resampling)` - Add a data layer, optionally reprojecting to match another layer.

- `add_transform(self, transform_func)` - Add a transform function to the processing pipeline.

- `apply_transforms(self, cache)` - Apply all transforms to all data layers with optional caching.

- `compute_colors(self)` - Compute colors using color_func and optionally mask_func.

- `compute_data_layer(self, name, source_layer, compute_func, transformed, cache_key)` - Compute a new data layer from an existing one using a transformation function.

- `configure_for_target_vertices(self, target_vertices, order)` - Configure downsampling to achieve approximately target_vertices.

- `create_mesh(self, base_depth, boundary_extension, scale_factor, height_scale, center_model, verbose)` - Create a Blender mesh from transformed DEM data with both performance and control.

- `set_color_mapping(self, color_func, source_layers, color_kwargs, mask_func, mask_layers, mask_kwargs, mask_threshold)` - Set up how to map data layers to colors (RGB) and optionally a mask/alpha channel.

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

#### `add_data_layer(self, name, data, transform, crs, target_crs, target_layer, resampling)`

Add a data layer, optionally reprojecting to match another layer.

Stores data with geographic metadata (CRS and transform). Can automatically
reproject and resample to match an existing layer's grid for multi-layer analysis.

Args:
    name (str): Unique name for this data layer (e.g., 'dem', 'elevation', 'slope').
    data (np.ndarray): 2D array of data values, shape (height, width).
    transform (rasterio.Affine): Affine transform mapping pixel to geographic coords.
    crs (str): Coordinate reference system in EPSG format (e.g., 'EPSG:4326').
    target_crs (str, optional): Target CRS to reproject to. If None and target_layer
        specified, uses target layer's CRS. If None and no target, uses input crs.
    target_layer (str, optional): Name of existing layer to match grid and CRS.
        If specified, data is automatically reprojected and resampled to align.
    resampling (rasterio.enums.Resampling): Resampling method for reprojection
        (default: Resampling.bilinear). See rasterio docs for options.

Returns:
    None: Modifies internal data_layers dictionary.

Raises:
    KeyError: If target_layer specified but doesn't exist.
    ValueError: If target_crs specified but no reference layer available.

Examples:
    >>> # Add elevation data with native CRS
    >>> terrain.add_data_layer('dem', dem_array, transform, 'EPSG:4326')

    >>> # Add overlay data, reproject to match DEM
    >>> terrain.add_data_layer('landcover', lc_array, lc_transform, 'EPSG:3857',
    ...                        target_layer='dem')

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

#### `clear_scene()`

Clear all objects from the Blender scene.

Resets the scene to factory settings (empty scene) and removes all default
objects. Useful before importing terrain meshes to ensure a clean workspace.

Raises:
    RuntimeError: If Blender module (bpy) is not available.

#### `compute_colors(self)`

Compute colors using color_func and optionally mask_func.

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

#### `configure_for_target_vertices(self, target_vertices, order)`

Configure downsampling to achieve approximately target_vertices.

This method calculates the appropriate zoom_factor to achieve a desired
vertex count for mesh generation. It provides a more intuitive API than
manually calculating zoom_factor from the original DEM shape.

Args:
    target_vertices: Desired vertex count for final mesh (e.g., 500_000)
    order: Interpolation order for downsampling (0=nearest, 1=linear, 4=bicubic)

Returns:
    Calculated zoom_factor that was added to transforms

Raises:
    ValueError: If target_vertices is invalid

Example:
    terrain = Terrain(dem, transform)
    zoom = terrain.configure_for_target_vertices(500_000)
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

#### `create_mesh(self, base_depth, boundary_extension, scale_factor, height_scale, center_model, verbose)`

Create a Blender mesh from transformed DEM data with both performance and control.

Generates vertices from DEM elevation values and faces for connectivity. Optionally
creates boundary faces to close the mesh into a solid. Supports coordinate scaling
and elevation scaling for visualization.

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

Returns:
    bpy.types.Object | None: The created terrain mesh object, or None if creation failed.

Raises:
    ValueError: If transformed DEM layer is not available (apply_transforms() not called).

#### `downsample_raster(zoom_factor, order, nodata_value)`

Create a raster downsampling transform function with specified parameters.

Args:
    zoom_factor: Scaling factor for downsampling (default: 0.1)
    order: Interpolation order (default: 4)
    nodata_value: Value to treat as no data (default: np.nan)

Returns:
    function: A transform function that downsamples raster data

#### `elevation_colormap(dem_data, cmap_name, min_elev, max_elev)`

Create a colormap based on elevation values.

Maps elevation data to colors using a matplotlib colormap.
Low elevations map to the start of the colormap, high elevations to the end.

Args:
    dem_data: 2D numpy array of elevation values
    cmap_name: Matplotlib colormap name (default: 'viridis')
    min_elev: Minimum elevation for normalization (default: use data min)
    max_elev: Maximum elevation for normalization (default: use data max)

Returns:
    Array of RGB colors with shape (height, width, 3) as uint8

#### `exists(self, target_name)`

Check if target exists

#### `flip_raster(axis)`

Create a transform function that mirrors (flips) the DEM data.
If axis='horizontal', it flips top ↔ bottom.
(In terms of rows, row=0 becomes row=(height-1).)

If axis='vertical', you could do left ↔ right (np.fliplr).

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

#### `position_camera_relative(mesh_obj, direction, distance, elevation, look_at, camera_type, sun_angle, sun_energy, focal_length)`

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
    sun_angle: Angle of sun light in degrees. Default: 2
    sun_energy: Intensity of sun light. Default: 3
    focal_length: Camera focal length in mm (perspective cameras only). Default: 50

Returns:
    Camera object

Raises:
    ValueError: If direction is not recognized or camera_type is invalid

#### `render_scene_to_file(output_path, width, height, file_format, color_mode, compression, save_blend_file)`

Render the current Blender scene to file.

Args:
    output_path (str or Path): Path where output file will be saved
    width (int): Render width in pixels (default: 1920)
    height (int): Render height in pixels (default: 1440)
    file_format (str): Output format 'PNG', 'JPEG', etc. (default: 'PNG')
    color_mode (str): 'RGBA' or 'RGB' (default: 'RGBA')
    compression (int): PNG compression level 0-100 (default: 90)
    save_blend_file (bool): Also save .blend project file (default: True)

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

#### `setup_light(location, angle, energy, rotation_euler)`

Create and configure sun light for terrain visualization.

Args:
    location: Tuple of (x,y,z) light position (default: (1, 1, 2))
    angle: Angle of sun light in degrees (default: 2)
    energy: Energy/intensity of sun light (default: 3)
    rotation_euler: Tuple of (x,y,z) rotation angles in radians (default: sun from NW)

Returns:
    Sun light object

#### `setup_render_settings(use_gpu, samples, preview_samples, use_denoising, denoiser, compute_device)`

Configure Blender render settings for high-quality terrain visualization.

Args:
    use_gpu: Whether to use GPU acceleration
    samples: Number of render samples
    preview_samples: Number of viewport preview samples
    use_denoising: Whether to enable denoising
    denoiser: Type of denoiser to use ('OPTIX', 'OPENIMAGEDENOISE', 'NLM')
    compute_device: Compute device type ('OPTIX', 'CUDA', 'HIP', 'METAL')

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

#### `transform(raster_data, transform)`

Scale elevation values in raster data.

Args:
    raster_data: Input raster numpy array
    transform: Optional affine transform (unchanged by scaling)

Returns:
    tuple: (scaled_data, transform, None)

#### `transform_func(data, transform)`

Flip array along specified axis and update transform if provided.

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
