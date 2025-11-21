# terrain-maker API Reference

Comprehensive documentation of all classes and functions in the terrain-maker library.

**Auto-generated from source code docstrings.**

## Table of Contents

- [terrain.core](#terrain-core)

---

## terrain.core

### Classes

#### `Terrain`

No documentation

**Methods:**

- `add_data_layer(self, name, data, transform, crs, target_crs, target_layer, resampling)` - Add a data layer, optionally reprojecting to match another layer

- `add_transform(self, transform_func)` - Add a transform function to the pipeline

- `apply_transforms(self, cache)` - Apply transforms with efficient caching

- `compute_colors(self)` - Compute colors using color_func and optionally mask_func.

- `compute_data_layer(self, name, source_layer, compute_func, transformed, cache_key)` - Compute a new data layer from an existing one

- `configure_for_target_vertices(self, target_vertices, order)` - Configure downsampling to achieve approximately target_vertices.

- `create_mesh(self, base_depth, boundary_extension, scale_factor, height_scale, center_model, verbose)` - Create a Blender mesh from transformed DEM data with both performance and control.

- `set_color_mapping(self, color_func, source_layers, color_kwargs, mask_func, mask_layers, mask_kwargs, mask_threshold)` - Set up how to map data to colors (RGB) and optionally a mask/alpha channel.

- `visualize_dem(self, layer, use_transformed, title, cmap, percentile_clip, clip_percentiles, max_pixels, show_histogram)` - Create diagnostic visualization of any terrain data layer.


#### `TerrainCache`

No documentation

**Methods:**

- `exists(self, target_name)` - Check if target exists

- `get_target_path(self, target_name)` - Get path for a specific target

- `load(self, target_name)` - Load GeoTIFF and metadata if it exists

- `save(self, target_name, data, transform, crs, metadata)` - Save data as GeoTIFF with CRS and metadata


### Functions

#### `add_data_layer(self, name, data, transform, crs, target_crs, target_layer, resampling)`

Add a data layer, optionally reprojecting to match another layer

Args:
    name: Name of the layer
    data: Input array
    transform: Affine transform for the data
    crs: CRS of input data
    target_crs: Target CRS to reproject to (if None, use target_layer's CRS)
    target_layer: Name of existing layer to match (for CRS and grid)
    resampling: Resampling method for alignment

#### `add_transform(self, transform_func)`

Add a transform function to the pipeline

#### `apply_colormap_material(material)`

Create a simple material setup for terrain visualization using vertex colors.
Uses emission to guarantee colors are visible regardless of lighting.

Args:
    material: Blender material to configure

#### `apply_transforms(self, cache)`

Apply transforms with efficient caching

#### `clear_scene()`

No documentation

#### `compute_colors(self)`

Compute colors using color_func and optionally mask_func.

Returns:
    np.ndarray: RGBA color array.

#### `compute_data_layer(self, name, source_layer, compute_func, transformed, cache_key)`

Compute a new data layer from an existing one

Args:
    name: Name for the new layer
    source_layer: Name of source layer to compute from
    compute_func: Function that takes source array and returns computed array
    transformed: Whether to use the transformed source data (default: False)
    cache_key: Optional custom cache key for the computation
    
Returns:
    np.ndarray: The computed layer data

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

Args:
    base_depth: Z-coordinate for the bottom of the terrain model (default: -0.2)
    boundary_extension: Whether to create side faces around the terrain boundary (default: True)
    scale_factor: Horizontal scale divisor for x/y coordinates (default: 100.0)
    height_scale: Multiplier for elevation values (default: 1.0)
    center_model: Whether to center the model at origin (default: True)
    verbose: Whether to log detailed progress information (default: True)

Returns:
    bpy.types.Object: The created terrain mesh object

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

No documentation

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

Set up how to map data to colors (RGB) and optionally a mask/alpha channel.

Args:
    color_func: Function that takes N data arrays (from source_layers)
                and returns an (H, W, 3 or 4) float array (RGB or RGBA).
    source_layers: List of layer names (strings) to feed to color_func.
    color_kwargs: Dict of additional kwargs specifically for color_func.
    mask_func: Optional function producing a mask (e.g., for water).
               Can take one or more arrays as input.
    mask_layers: Layer name(s) to supply to mask_func if different from source_layers.
    mask_kwargs: Dict of additional kwargs specifically for mask_func.
    mask_threshold: Optional convenience parameter if mask_func is threshold-based.

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

No documentation

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
