from __future__ import annotations

import os
import json
import time
from pathlib import Path
import glob

try:
    import bpy
except ImportError:
    bpy = None
from dataclasses import dataclass
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio import Affine
from scipy.ndimage import zoom, generic_filter, sobel
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from math import radians
from tqdm import tqdm
from matplotlib.collections import LineCollection
import numpy as np
import shapely
from shapely.geometry import Polygon, Point
from shapely.affinity import scale
import logging
from datetime import datetime
import sys
import seaborn as sns
import geopandas as gpd
from shapely.validation import make_valid
import colorsys
from matplotlib.colors import to_rgb
from typing import Optional, Dict, Any, Tuple, Callable
import functools
import inspect
import zarr
import hashlib

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create a formatter and add it to the handler
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(console_handler)


def calculate_target_vertices(
    width: int,
    height: int,
    multiplier: float = 2.0,
) -> int:
    """
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
    """
    return int(width * height * multiplier)


def clear_scene():
    """
    Clear all objects from the Blender scene.

    Resets the scene to factory settings (empty scene) and removes all default
    objects. Useful before importing terrain meshes to ensure a clean workspace.

    Raises:
        RuntimeError: If Blender module (bpy) is not available.
    """
    from src.terrain.scene_setup import clear_scene as _clear_scene

    return _clear_scene()


def setup_camera(camera_angle, camera_location, scale, focal_length=50, camera_type="PERSP"):
    """Configure camera for terrain visualization.

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
    """
    from src.terrain.scene_setup import setup_camera as _setup_camera

    return _setup_camera(camera_angle, camera_location, scale, focal_length, camera_type)


def setup_light(
    location=(1, 1, 2),
    angle=2,
    energy=3,
    rotation_euler=None,
    azimuth=None,
    elevation=None,
):
    """Create and configure sun light for terrain visualization.

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
    """
    from src.terrain.scene_setup import setup_light as _setup_light

    return _setup_light(location, angle, energy, rotation_euler, azimuth, elevation)


def setup_camera_and_light(
    camera_angle,
    camera_location,
    scale,
    sun_angle=2,
    sun_energy=3,
    focal_length=50,
    camera_type="PERSP",
):
    """Configure camera and main light for terrain visualization.

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
    """
    from src.terrain.scene_setup import setup_camera_and_light as _setup_camera_and_light

    return _setup_camera_and_light(
        camera_angle, camera_location, scale, sun_angle, sun_energy, focal_length, camera_type
    )


def setup_two_point_lighting(
    sun_azimuth: float = 225.0,
    sun_elevation: float = 30.0,
    sun_energy: float = 7.0,
    sun_angle: float = 1.0,
    sun_color: tuple = (1.0, 0.85, 0.6),
    fill_azimuth: float = 45.0,
    fill_elevation: float = 60.0,
    fill_energy: float = 0.0,
    fill_angle: float = 3.0,
    fill_color: tuple = (0.7, 0.8, 1.0),
) -> list:
    """Set up two-point lighting with primary sun and optional fill light.

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
    """
    from src.terrain.scene_setup import setup_two_point_lighting as _setup_two_point_lighting

    return _setup_two_point_lighting(
        sun_azimuth=sun_azimuth,
        sun_elevation=sun_elevation,
        sun_energy=sun_energy,
        sun_angle=sun_angle,
        sun_color=sun_color,
        fill_azimuth=fill_azimuth,
        fill_elevation=fill_elevation,
        fill_energy=fill_energy,
        fill_angle=fill_angle,
        fill_color=fill_color,
    )


def position_camera_relative(
    mesh_obj,
    direction="south",
    distance=1.5,
    elevation=0.5,
    look_at="center",
    camera_type="ORTHO",
    sun_angle=0,
    sun_energy=0,
    sun_azimuth=None,
    sun_elevation=None,
    focal_length=50,
    ortho_scale=1.2,
):
    """Position camera relative to mesh using intuitive cardinal directions.

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
    """
    from src.terrain.scene_setup import position_camera_relative as _position_camera_relative

    return _position_camera_relative(
        mesh_obj=mesh_obj,
        direction=direction,
        distance=distance,
        elevation=elevation,
        look_at=look_at,
        camera_type=camera_type,
        sun_angle=sun_angle,
        sun_energy=sun_energy,
        sun_azimuth=sun_azimuth,
        sun_elevation=sun_elevation,
        focal_length=focal_length,
        ortho_scale=ortho_scale,
    )


def setup_world_atmosphere(density=0.02, scatter_color=(1, 1, 1, 1), anisotropy=0.0):
    """Set up world volume for atmospheric effects.

    Args:
        density: Density of the atmospheric volume (default: 0.02)
        scatter_color: RGBA color tuple for scatter (default: white)
        anisotropy: Direction of scatter from -1 to 1 (default: 0 for uniform)

    Returns:
        bpy.types.World: The configured world object
    """
    from src.terrain.scene_setup import setup_world_atmosphere as _setup_world_atmosphere

    return _setup_world_atmosphere(density, scatter_color, anisotropy)


def setup_hdri_lighting(
    sun_elevation: float = 30.0,
    sun_rotation: float = 225.0,
    sun_intensity: float = 1.0,
    sun_size: float = 0.545,
    air_density: float = 1.0,
    visible_to_camera: bool = False,
    camera_background: tuple = None,
    sky_strength: float = None,
):
    """Set up HDRI-style sky lighting using Blender's Nishita sky model.

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
    """
    from src.terrain.scene_setup import setup_hdri_lighting as _setup_hdri_lighting

    return _setup_hdri_lighting(
        sun_elevation=sun_elevation,
        sun_rotation=sun_rotation,
        sun_intensity=sun_intensity,
        sun_size=sun_size,
        air_density=air_density,
        visible_to_camera=visible_to_camera,
        camera_background=camera_background,
        sky_strength=sky_strength,
    )


def setup_render_settings(
    use_gpu: bool = True,
    samples: int = 128,
    preview_samples: int = 32,
    use_denoising: bool = True,
    denoiser: str = "OPTIX",
    compute_device: str = "OPTIX",
    use_persistent_data: bool = False,
    use_auto_tile: bool = False,
    tile_size: int = 2048,
) -> None:
    """
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
    """
    from src.terrain.rendering import setup_render_settings as _setup_render_settings

    return _setup_render_settings(
        use_gpu=use_gpu,
        samples=samples,
        preview_samples=preview_samples,
        use_denoising=use_denoising,
        denoiser=denoiser,
        compute_device=compute_device,
        use_persistent_data=use_persistent_data,
        use_auto_tile=use_auto_tile,
        tile_size=tile_size,
    )


def render_scene_to_file(
    output_path,
    width=1920,
    height=1440,
    file_format="PNG",
    color_mode="RGBA",
    compression=90,
    save_blend_file=True,
    show_progress=True,
    max_retries=3,
    retry_delay=5.0,
):
    """
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
    """
    from src.terrain.rendering import render_scene_to_file as _render_scene_to_file

    return _render_scene_to_file(
        output_path=output_path,
        width=width,
        height=height,
        file_format=file_format,
        color_mode=color_mode,
        compression=compression,
        save_blend_file=save_blend_file,
        show_progress=show_progress,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )


def get_render_settings_report() -> dict:
    """
    Query Blender for the actual render settings used.

    Returns a dictionary of all render-relevant settings, useful for
    debugging, reproducibility, and verification.

    Returns:
        dict: Dictionary containing all render settings from Blender
    """
    from src.terrain.rendering import get_render_settings_report as _get_render_settings_report

    return _get_render_settings_report()


def print_render_settings_report(log=None) -> None:
    """
    Print a formatted report of all Blender render settings.

    Queries Blender for actual settings and prints them in a readable format.
    Useful for debugging and ensuring settings are correctly applied.

    Args:
        log: Logger to use (defaults to module logger)
    """
    from src.terrain.rendering import print_render_settings_report as _print_render_settings_report

    return _print_render_settings_report(log)


def apply_colormap_material(material: bpy.types.Material) -> None:
    """
    Create a simple material setup for terrain visualization using vertex colors.
    Uses emission to guarantee colors are visible regardless of lighting.

    Args:
        material: Blender material to configure
    """
    from src.terrain.materials import apply_colormap_material as _apply_colormap_material

    return _apply_colormap_material(material)


def apply_water_shader(
    material: bpy.types.Material, water_color: Tuple[float, float, float] = (0.0, 0.153, 0.298)
) -> None:
    """
    Apply water shader to material, coloring water areas based on vertex alpha channel.
    Uses alpha channel to mix between water color and elevation colors.
    Water pixels (alpha=1.0) render as water color; land pixels (alpha=0.0) show elevation colors.

    Args:
        material: Blender material to configure
        water_color: RGB tuple for water (default: University of Michigan blue #00274C)
    """
    from src.terrain.materials import apply_water_shader as _apply_water_shader

    return _apply_water_shader(material, water_color)


def create_background_plane(
    terrain_obj: bpy.types.Object,
    depth: float = -2.0,
    scale_factor: float = 2.0,
    material_params: Optional[Dict] = None,
) -> bpy.types.Object:
    """
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
    """
    from src.terrain.materials import create_background_plane as _create_background_plane

    return _create_background_plane(terrain_obj, depth, scale_factor, material_params)


def load_dem_files(
    directory_path: str, pattern: str = "*.hgt", recursive: bool = False
) -> tuple[np.ndarray, rasterio.Affine]:
    """
    Load and merge DEM files from a directory into a single elevation dataset.
    Supports any raster format readable by rasterio (HGT, GeoTIFF, etc.).

    Args:
        directory_path: Path to directory containing DEM files
        pattern: File pattern to match (default: ``*.hgt``)
        recursive: Whether to search subdirectories recursively (default: False)

    Returns:
        tuple: (merged_dem, transform) where:
            - merged_dem: numpy array containing the merged elevation data
            - transform: affine transform mapping pixel to geographic coordinates

    Raises:
        ValueError: If no valid DEM files are found or directory doesn't exist
        OSError: If directory access fails or file reading fails
        rasterio.errors.RasterioIOError: If there are issues reading the DEM files
    """
    from src.terrain.data_loading import load_dem_files as _load_dem_files

    return _load_dem_files(directory_path, pattern, recursive)


def load_filtered_hgt_files(
    dem_dir,
    min_latitude: int = None,
    max_latitude: int = None,
    min_longitude: int = None,
    max_longitude: int = None,
    bbox: tuple = None,
    pattern: str = "*.hgt",
):
    """
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
    """
    from src.terrain.data_loading import load_filtered_hgt_files as _load_filtered_hgt_files

    return _load_filtered_hgt_files(
        dem_dir,
        min_latitude=min_latitude,
        max_latitude=max_latitude,
        min_longitude=min_longitude,
        max_longitude=max_longitude,
        bbox=bbox,
        pattern=pattern,
    )


def reproject_raster(src_crs="EPSG:4326", dst_crs="EPSG:32617", nodata_value=np.nan, num_threads=4):
    """
    Generalized raster reprojection function

    Args:
        src_crs: Source coordinate reference system
        dst_crs: Destination coordinate reference system
        nodata_value: Value to use for areas outside original data
        num_threads: Number of threads for parallel processing

    Returns:
        Function that transforms data and returns (data, transform, new_crs)
    """
    from src.terrain.transforms import reproject_raster as _reproject_raster

    return _reproject_raster(src_crs, dst_crs, nodata_value, num_threads)


def downsample_raster(zoom_factor=0.1, method="average", nodata_value=np.nan):
    """
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
    """
    from src.terrain.transforms import downsample_raster as _downsample_raster

    return _downsample_raster(zoom_factor, method, nodata_value)


def smooth_raster(window_size=None, nodata_value=np.nan):
    """
    Create a raster smoothing transform function with specified parameters.

    Args:
        window_size: Size of median filter window
                    (defaults to 5% of smallest dimension if None)
        nodata_value: Value to treat as no data (default: np.nan)

    Returns:
        function: A transform function that smooths raster data
    """
    from src.terrain.transforms import smooth_raster as _smooth_raster

    return _smooth_raster(window_size, nodata_value)


def flip_raster(axis="horizontal"):
    """
    Create a transform function that mirrors (flips) the DEM data.
    If axis='horizontal', it flips top ↔ bottom.
    (In terms of rows, row=0 becomes row=(height-1).)

    If axis='vertical', you could do left ↔ right (np.fliplr).
    """
    from src.terrain.transforms import flip_raster as _flip_raster

    return _flip_raster(axis)


def scale_elevation(scale_factor=1.0, nodata_value=np.nan):
    """
    Create a raster elevation scaling transform function.

    Multiplies all elevation values by the scale factor. Useful for reducing
    or amplifying terrain height without changing horizontal scale.

    Args:
        scale_factor (float): Multiplication factor for elevation values (default: 1.0)
        nodata_value: Value to treat as no data (default: np.nan)

    Returns:
        function: A transform function that scales elevation data
    """
    from src.terrain.transforms import scale_elevation as _scale_elevation

    return _scale_elevation(scale_factor, nodata_value)


def feature_preserving_smooth(sigma_spatial=3.0, sigma_intensity=None, nodata_value=np.nan):
    """
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
    """
    from src.terrain.transforms import feature_preserving_smooth as _feature_preserving_smooth

    return _feature_preserving_smooth(sigma_spatial, sigma_intensity, nodata_value)


def smooth_score_data(scores, sigma_spatial=3.0, sigma_intensity=None):
    """
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
    """
    from src.terrain.transforms import smooth_score_data as _smooth_score_data

    return _smooth_score_data(scores, sigma_spatial, sigma_intensity)


def slope_colormap(slopes, cmap_name="terrain", min_slope=0, max_slope=45):
    """
    Create a simple colormap based solely on terrain slopes.

    Args:
        slopes: Array of slope values in degrees
        cmap_name: Matplotlib colormap name (default: 'terrain')
        min_slope: Minimum slope value for normalization (default: 0)
        max_slope: Maximum slope value for normalization (default: 45)

    Returns:
        Array of RGBA colors with shape (*slopes.shape, 4)
    """
    from src.terrain.color_mapping import slope_colormap as _slope_colormap

    return _slope_colormap(slopes, cmap_name, min_slope, max_slope)


def elevation_colormap(dem_data, cmap_name="viridis", min_elev=None, max_elev=None, gamma=1.0):
    """
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
    """
    from src.terrain.color_mapping import elevation_colormap as _elevation_colormap

    return _elevation_colormap(dem_data, cmap_name, min_elev, max_elev, gamma)


def transform_wrapper(transform_func):
    """
    Standardize transform function interface with consistent output

    Args:
        transform_func: The original transform function to wrap

    Returns:
        A wrapped function with consistent signature and return format
    """

    @functools.wraps(transform_func)
    def wrapped_transform(data: np.ndarray, transform: rasterio.Affine = None) -> tuple:
        """
        Standardized transform wrapper with consistent signature

        Args:
            data: Input numpy array to transform
            transform: Optional affine transform

        Returns:
            Tuple of (transformed_data, transform, [crs]) where CRS is optional
        """
        # Inspect function signature to determine how to call
        sig = inspect.signature(transform_func)
        params = list(sig.parameters.keys())

        try:
            # Initialize result variables
            transformed_data = None
            final_transform = transform
            crs = None

            # Case 1: Transform takes only data
            if len(params) == 1 and params[0] == "data":
                transformed_data = transform_func(data)

            # Case 2: Transform takes (data, transform)
            elif len(params) == 2 and params[0] == "data" and params[1] == "transform":
                result = transform_func(data, transform)

                # Handle different return types
                if isinstance(result, tuple):
                    if len(result) == 3:
                        # When transform returns (data, transform, crs)
                        transformed_data, final_transform, crs = result
                    elif len(result) == 2:
                        # When transform returns (data, transform)
                        transformed_data, final_transform = result
                    else:
                        transformed_data = result[0]
                else:
                    transformed_data = result

            # Case 3: More complex signature or other parameters
            else:
                result = transform_func(data, transform)

                # Handle different return types
                if isinstance(result, tuple):
                    if len(result) == 3:
                        # When transform returns (data, transform, crs)
                        transformed_data, final_transform, crs = result
                    elif len(result) == 2:
                        # When transform returns (data, transform)
                        transformed_data, final_transform = result
                    else:
                        transformed_data = result[0]
                else:
                    transformed_data = result

            # Return standardized format: always a 3-tuple with optional None for crs
            return transformed_data, final_transform, crs

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error in transform {transform_func.__name__}: {e}")
            raise

    return wrapped_transform


class TerrainCache:
    """
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
    """

    def __init__(self, cache_dir: str = "terrain_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def get_target_path(self, target_name: str) -> Path:
        """Get path for a specific target"""
        return self.cache_dir / f"{target_name}.tif"

    def exists(self, target_name: str) -> bool:
        """Check if target exists"""
        return self.get_target_path(target_name).exists()

    def save(self, target_name: str, data: np.ndarray, transform, crs="EPSG:4326", metadata=None):
        """Save data as GeoTIFF with CRS and metadata"""
        path = self.get_target_path(target_name)
        self.logger.info(f"Saving {target_name}")

        # Save main raster data
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs=crs,
            transform=transform,
            compress="lzw",
        ) as dst:
            dst.write(data, 1)

            # Save metadata if provided
            if metadata:
                # Convert any non-string values to strings for GDAL metadata
                string_metadata = {k: str(v) for k, v in metadata.items()}
                dst.update_tags(**string_metadata)

        # For more complex metadata that can't be stored in GDAL tags
        if metadata:
            # Save to a companion JSON file
            meta_path = path.with_suffix(".json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f)

    def load(self, target_name: str) -> Optional[Dict[str, Any]]:
        """Load GeoTIFF and metadata if it exists"""
        path = self.get_target_path(target_name)
        if not path.exists():
            return None

        self.logger.info(f"Loading {target_name}")

        # Load raster data
        with rasterio.open(path) as src:
            data = src.read(1)
            transform = src.transform
            crs = src.crs
            # Get basic metadata from tags
            basic_metadata = src.tags()

        # Try to load companion metadata file
        meta_path = path.with_suffix(".json")
        full_metadata = basic_metadata.copy()
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    full_metadata.update(json.load(f))
            except (json.JSONDecodeError, OSError) as e:
                self.logger.warning(f"Failed to load metadata file {meta_path}: {e}")

        return {"data": data, "transform": transform, "crs": crs, "metadata": full_metadata}


class Terrain:
    """
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
    """

    def __init__(
        self,
        dem_data: np.ndarray,
        dem_transform: rasterio.Affine,
        dem_crs: str = "EPSG:4326",
        cache_dir: str = "terrain_cache",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize terrain from DEM data.

        Args:
            dem_data (np.ndarray): DEM array of shape (height, width) containing elevation values.
                Integer types are converted to float32. Must be 2D.
            dem_transform (rasterio.Affine): Affine transform mapping pixel coordinates to
                geographic coordinates.
            dem_crs (str): Coordinate reference system in EPSG format (default: 'EPSG:4326').
            cache_dir (str): Directory for caching computations (default: 'terrain_cache').
            logger (logging.Logger, optional): Logger instance for diagnostic output.

        Raises:
            TypeError: If dem_data is not a numpy array or has unsupported dtype.
            ValueError: If dem_data is not 2D.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("Initializing Terrain Cache")
        self.cache = TerrainCache(cache_dir)
        self.logger.info(f"Cache directory contents: {list(self.cache.cache_dir.glob('**/*'))}")

        self.logger.info("Initializing Terrain...")

        # Validate input
        if not isinstance(dem_data, np.ndarray):
            raise TypeError("dem_data must be a numpy array")
        if dem_data.ndim != 2:
            raise ValueError(f"dem_data must be 2D, got shape {dem_data.shape}")

        # Convert to float32 if needed
        if np.issubdtype(dem_data.dtype, np.integer):
            self.logger.info(f"Converting DEM data from {dem_data.dtype} to float32")
            dem_data = dem_data.astype(np.float32)
        elif not np.issubdtype(dem_data.dtype, np.floating):
            raise TypeError(f"Unsupported DEM data type: {dem_data.dtype}")

        # Store original DEM data and transform
        self.dem_transform = dem_transform

        self.dem_bounds = rasterio.transform.array_bounds(
            dem_data.shape[0], dem_data.shape[1], dem_transform
        )

        # Calculate resolution in meters
        self.resolution = (abs(dem_transform[0]) * 111320, abs(dem_transform[4]) * 111320)
        self.dem_shape = dem_data.shape

        # Initialize list of transforms and data layers
        self.transforms = []
        self.data_layers = {}

        # Track cumulative transform metadata (e.g., elevation scale factors)
        # This allows algorithms to compensate for prior transforms
        self.transform_metadata = {
            "elevation_scale": 1.0,  # Cumulative scale factor applied to elevation
        }

        # Initialize containers for processed data
        self.processed_dem = None  # Will hold transformed DEM data
        self.vertices = None  # Will hold final vertex positions
        self.faces = None  # Will hold face indices
        self.vertex_colors = None  # Will hold vertex colors

        self.add_data_layer("dem", dem_data, dem_transform, dem_crs)

        self.logger.info(f"Terrain initialized with DEM data:")
        self.logger.info(f"  Shape: {dem_data.shape}")
        self.logger.info(f"  Resolution: {self.resolution[0]:.2f}m x {self.resolution[1]:.2f}m")
        self.logger.info(f"  Value range: {np.nanmin(dem_data):.2f} to {np.nanmax(dem_data):.2f}")

    def visualize_dem(
        self,
        layer: str = "dem",
        use_transformed: bool = False,
        title: str = None,
        cmap: str = "terrain",
        percentile_clip: bool = True,
        clip_percentiles: tuple = (1, 99),
        max_pixels: int = 500_000,
        show_histogram: bool = True,
    ) -> None:
        """
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
        """
        self.logger.info(f"Creating visualization for layer '{layer}'")

        # Validate requested layer exists
        if layer not in self.data_layers:
            available_layers = list(self.data_layers.keys())
            raise ValueError(f"Layer '{layer}' not found. Available layers: {available_layers}")

        layer_info = self.data_layers[layer]

        # Determine which data to use (transformed or original)
        if use_transformed and not layer_info.get("transformed", False):
            self.logger.warning(
                f"Transformed data requested for layer '{layer}' but not available. Using original."
            )
            use_transformed = False

        if use_transformed:
            plot_data = layer_info["transformed_data"]
            data_transform = layer_info["transformed_transform"]
            data_crs = layer_info["transformed_crs"]
            self.logger.info(f"Using transformed data for layer '{layer}'")
        else:
            plot_data = layer_info["data"]
            data_transform = layer_info["transform"]
            data_crs = layer_info["crs"]
            self.logger.info(f"Using original data for layer '{layer}'")

        # Generate title if not provided
        if title is None:
            transform_status = "Transformed" if use_transformed else "Original"
            title = f"{transform_status} {layer.capitalize()} Layer Visualization"

        # Remove NaN for calculations
        valid_data = plot_data[~np.isnan(plot_data)]
        if len(valid_data) == 0:
            self.logger.error(f"Layer '{layer}' contains no valid data (all NaN)")
            return

        # Logging basic statistics
        self.logger.info("Data Statistics:")
        self.logger.info(f"  Shape: {plot_data.shape}")
        self.logger.info(f"  Min Value: {valid_data.min():.4f}")
        self.logger.info(f"  Max Value: {valid_data.max():.4f}")
        self.logger.info(f"  Mean Value: {valid_data.mean():.4f}")
        self.logger.info(f"  Median Value: {np.median(valid_data):.4f}")

        # NaN analysis
        nan_percentage = np.isnan(plot_data).mean() * 100
        self.logger.info(f"  NaN Percentage: {nan_percentage:.2f}%")

        # Subsampling to prevent memory issues
        def sample_array(arr):
            """Downsample array for visualization if it exceeds max_pixels limit."""
            total_pixels = arr.size
            if total_pixels <= max_pixels:
                return arr

            sample_rate = int(np.sqrt(total_pixels / max_pixels))
            self.logger.info(f"Subsampling with rate 1/{sample_rate} for visualization")
            return arr[::sample_rate, ::sample_rate]

        sampled_data = sample_array(plot_data)

        # Determine color scaling
        if percentile_clip:
            min_percentile, max_percentile = clip_percentiles
            vmin = np.percentile(valid_data, min_percentile)
            vmax = np.percentile(valid_data, max_percentile)
            self.logger.info(
                f"Clipping to {min_percentile}-{max_percentile} percentiles: [{vmin:.4f}, {vmax:.4f}]"
            )
        else:
            vmin, vmax = valid_data.min(), valid_data.max()

        # Determine plot layout
        if show_histogram:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle(title, fontsize=16)

            # Main data heatmap
            im = ax1.imshow(sampled_data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax1.set_title(f"{layer.capitalize()} Visualization")
            ax1.set_xlabel("Column Index")
            ax1.set_ylabel("Row Index")
            plt.colorbar(im, ax=ax1, shrink=0.8, label=layer.capitalize())

            # Value distribution histogram
            ax2.hist(valid_data, bins=50, color="skyblue", alpha=0.7)
            ax2.set_title(f"{layer.capitalize()} Distribution")
            ax2.set_xlabel("Value")
            ax2.set_ylabel("Frequency")

            # Add grid lines
            ax1.grid(False)
            ax2.grid(True, alpha=0.3)

        else:
            # Simple single plot with just the heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            fig.suptitle(title, fontsize=16)

            im = ax.imshow(sampled_data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(f"{layer.capitalize()} Visualization")
            ax.set_xlabel("Column Index")
            ax.set_ylabel("Row Index")
            plt.colorbar(im, ax=ax, shrink=0.8, label=layer.capitalize())

        # Add data source and transform info in footer
        transform_str = f"Transform: [{data_transform[0]:.6f}, {data_transform[1]:.6f}, {data_transform[2]:.6f}, {data_transform[3]:.6f}, {data_transform[4]:.6f}, {data_transform[5]:.6f}]"
        plt.figtext(0.5, 0.01, f"CRS: {data_crs} | {transform_str}", ha="center", fontsize=8)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.05)  # Make room for footer
        plt.show()

        self.logger.info("Visualization complete.")

    def add_transform(self, transform_func):
        """
        Add a transform function to the processing pipeline.

        Args:
            transform_func (callable): Function that transforms DEM data. Should accept
                (dem_array: np.ndarray) and return transformed np.ndarray.

        Returns:
            None: Modifies internal transforms list in place.

        Examples:
            >>> terrain.add_transform(lambda dem: gaussian_filter(dem, sigma=2))
            >>> terrain.apply_transforms()
        """
        wrapped_transform = transform_wrapper(transform_func)

        self.transforms.append(wrapped_transform)
        self.logger.info(f"Added transform: {transform_func.__name__}")

    def add_data_layer(
        self,
        name: str,
        data: np.ndarray,
        transform: Optional[rasterio.Affine] = None,
        crs: Optional[str] = None,
        target_crs: Optional[str] = None,
        target_layer: Optional[str] = None,
        same_extent_as: Optional[str] = None,
        resampling: Resampling = Resampling.bilinear,
    ) -> None:
        """
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
        """
        self.logger.info(f"Adding data layer '{name}'")

        # Handle same_extent_as: calculate transform from reference layer's bounds
        if same_extent_as is not None:
            if same_extent_as not in self.data_layers:
                raise KeyError(f"Reference layer '{same_extent_as}' not found for same_extent_as")

            ref_info = self.data_layers[same_extent_as]

            # Use ORIGINAL extent and CRS (before transforms) since the source data
            # typically covers the same geographic area as the original reference layer.
            # The reprojection will handle coordinate transformation.
            ref_data = ref_info["data"]
            ref_transform = ref_info["transform"]
            ref_crs = ref_info["crs"]

            # Calculate geographic bounds of reference layer
            ref_height, ref_width = ref_data.shape
            # Top-left corner
            x_origin = ref_transform.c
            y_origin = ref_transform.f
            # Bottom-right corner
            x_end = x_origin + ref_transform.a * ref_width
            y_end = y_origin + ref_transform.e * ref_height

            # Calculate pixel size for source data to cover same extent
            src_height, src_width = data.shape
            pixel_width = (x_end - x_origin) / src_width
            pixel_height = (y_end - y_origin) / src_height

            # Create transform for source data
            transform = rasterio.Affine(pixel_width, 0, x_origin, 0, pixel_height, y_origin)
            crs = ref_crs

            self.logger.info(
                f"Calculated transform from '{same_extent_as}' extent: "
                f"origin=({x_origin:.4f}, {y_origin:.4f}), "
                f"pixel=({pixel_width:.6f}, {pixel_height:.6f})"
            )

            # If target_layer not specified, use same_extent_as as target
            if target_layer is None:
                target_layer = same_extent_as

        # Validate that transform and crs are provided (either directly or via same_extent_as)
        if transform is None:
            raise ValueError("transform is required (or use same_extent_as to calculate automatically)")
        if crs is None:
            raise ValueError("crs is required (or use same_extent_as to inherit from reference layer)")

        # Determine target CRS and transform
        if target_layer is not None:
            if target_layer not in self.data_layers:
                raise KeyError(f"Target layer '{target_layer}' not found")

            target_info = self.data_layers[target_layer]

            # Use transformed data dimensions if transforms have been applied,
            # otherwise fall back to original dimensions. This ensures data layers
            # added after downsampling are automatically resampled to match the
            # actual mesh dimensions.
            if target_info.get("transformed", False) and "transformed_data" in target_info:
                target_shape = target_info["transformed_data"].shape
                target_transform = target_info.get(
                    "transformed_transform", target_info["transform"]
                )
                target_crs = target_info.get("transformed_crs", target_info["crs"])
                self.logger.debug(
                    f"Using transformed target shape {target_shape} for layer alignment"
                )
            else:
                target_shape = target_info["data"].shape
                target_transform = target_info["transform"]
                target_crs = target_info["crs"]

        elif target_crs is not None:
            # If target_crs provided but no reference layer, we need a reference layer
            if not self.data_layers:
                raise ValueError("Cannot determine target grid without reference layer")

            # Use first layer as reference for grid
            reference_layer = next(iter(self.data_layers.values()))

            # Use transformed dimensions if available
            if reference_layer.get("transformed", False) and "transformed_data" in reference_layer:
                target_transform = reference_layer.get(
                    "transformed_transform", reference_layer["transform"]
                )
                target_shape = reference_layer["transformed_data"].shape
            else:
                target_transform = reference_layer["transform"]
                target_shape = reference_layer["data"].shape

        else:
            # If no target specified, keep original
            self.data_layers[name] = {
                "data": data,
                "transform": transform,
                "crs": crs,
                "transformed": False,
            }
            self.logger.info(f"Added layer '{name}' with original CRS {crs}")
            return

        # Create target array and reproject if needed
        # Note: Affine.__ne__ returns array, so use tuple comparison instead
        transforms_differ = (crs != target_crs) or (tuple(transform) != tuple(target_transform))
        if transforms_differ:
            self.logger.info(f"Reprojecting from {crs} to {target_crs}")
            self.logger.info(f"Transforms: {transform} to {target_transform}")
            aligned_data = np.zeros(target_shape, dtype=data.dtype)

            try:
                reproject(
                    data,
                    aligned_data,
                    src_transform=transform,
                    src_crs=crs,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    resampling=resampling,
                )

                # Store reprojected data
                self.data_layers[name] = {
                    "data": aligned_data,
                    "transform": target_transform,
                    "crs": target_crs,
                    "original_data": data,
                    "original_transform": transform,
                    "original_crs": crs,
                    "transformed": False,
                }

                self.logger.info(f"Successfully added layer '{name}' (reprojected):")
                self.logger.info(f"  Shape: {aligned_data.shape}")
                self.logger.info(
                    f"  Value range: {np.nanmin(aligned_data):.2f} to {np.nanmax(aligned_data):.2f}"
                )

            except Exception as e:
                self.logger.error(f"Failed to reproject layer '{name}': {str(e)}")
                raise
        else:
            # No reprojection needed
            self.data_layers[name] = {
                "data": data,
                "transform": transform,
                "crs": crs,
                "transformed": False,
            }
            self.logger.info(f"Added layer '{name}' (no reprojection needed)")

    def get_bbox_wgs84(self, layer: str = "dem") -> tuple[float, float, float, float]:
        """
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
        """
        from rasterio.warp import transform_bounds

        if layer not in self.data_layers:
            raise KeyError(f"Layer '{layer}' not found")

        layer_info = self.data_layers[layer]

        # Use transformed data/transform if available, otherwise original
        if layer_info.get("transformed") and "transformed_data" in layer_info:
            data = layer_info["transformed_data"]
            layer_transform = layer_info.get("transformed_transform", layer_info["transform"])
            layer_crs = layer_info.get("transformed_crs", layer_info["crs"])
        else:
            data = layer_info["data"]
            layer_transform = layer_info["transform"]
            layer_crs = layer_info["crs"]

        # Calculate bounds from transform and shape
        height, width = data.shape
        # Top-left corner
        x_origin = layer_transform.c
        y_origin = layer_transform.f
        # Bottom-right corner
        x_end = x_origin + layer_transform.a * width
        y_end = y_origin + layer_transform.e * height

        # Normalize bounds (west, south, east, north)
        west = min(x_origin, x_end)
        east = max(x_origin, x_end)
        south = min(y_origin, y_end)
        north = max(y_origin, y_end)

        # If already WGS84, just return
        if layer_crs in ("EPSG:4326", "epsg:4326", None):
            return (south, west, north, east)

        # Transform bounds to WGS84
        transformed_bounds = transform_bounds(
            layer_crs, "EPSG:4326", west, south, east, north
        )

        # transform_bounds returns (west, south, east, north)
        t_west, t_south, t_east, t_north = transformed_bounds

        return (t_south, t_west, t_north, t_east)

    def compute_data_layer(
        self,
        name: str,
        source_layer: str,
        compute_func: Callable[[np.ndarray], np.ndarray],
        transformed: bool = False,
        cache_key: Optional[str] = None,
    ) -> np.ndarray:
        """
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
        """
        self.logger.info(f"Computing layer '{name}' from '{source_layer}'")

        # Verify source layer exists
        if source_layer not in self.data_layers:
            raise KeyError(f"Source layer '{source_layer}' not found")

        source_layer_info = self.data_layers[source_layer]

        # Check if transformed data is requested but not available
        if transformed and not source_layer_info.get("transformed", False):
            raise ValueError(f"Source layer '{source_layer}' has not been transformed")

        # Get appropriate source data and metadata
        if transformed:
            source_data = source_layer_info["transformed_data"]
            source_transform = source_layer_info["transformed_transform"]
            source_crs = source_layer_info["transformed_crs"]
        else:
            source_data = source_layer_info["data"]
            source_transform = source_layer_info["transform"]
            source_crs = source_layer_info["crs"]

        # Generate cache key if not provided
        if cache_key is None:
            transform_suffix = "_transformed" if transformed else ""
            cache_key = f"{name}_{source_layer}{transform_suffix}_{compute_func.__name__}"

        # Try to load from cache
        cached = self.cache.load(cache_key)

        if cached is None:
            self.logger.info(f"Computing {name} from {source_layer}")
            try:
                # Apply the computation function
                computed_data = compute_func(source_data)

                # Cache the result with source metadata
                self.cache.save(
                    cache_key,
                    computed_data,
                    transform=source_transform,
                    crs=source_crs,
                    metadata={"source_layer": source_layer, "transformed": transformed},
                )

            except Exception as e:
                self.logger.error(f"Failed to compute layer '{name}': {str(e)}")
                raise
        else:
            self.logger.info(f"Loaded cached computation for '{name}'")
            computed_data = cached["data"]
            # Use cached metadata if available
            source_transform = cached.get("transform", source_transform)
            source_crs = cached.get("crs", source_crs)

        # Add the computed layer with correct transform and CRS
        self.add_data_layer(name, computed_data, source_transform, source_crs)

        return computed_data

    def apply_transforms(self, cache=False):
        """
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
        """
        if not self.transforms:
            self.logger.warning("No transforms to apply")
            return

        # Process all data layers with individual caches
        with tqdm(total=len(self.data_layers), desc="Processing data layers") as pbar:
            for name, layer in self.data_layers.items():
                # Skip already transformed layers
                if layer.get("transformed", False):
                    self.logger.debug(f"Layer {name} already transformed, skipping")
                    pbar.update(1)
                    continue

                # Create target name from transform sequence
                layer_target = f"{name}_{'_'.join(t.__name__ for t in self.transforms)}"
                cached_layer = self.cache.load(layer_target)

                if cached_layer is None:
                    import time as _time
                    self.logger.info(f"Cache miss for {layer_target}, computing transforms...")
                    layer_data = layer["data"].copy()
                    current_transform = layer["transform"]
                    current_crs = layer["crs"]

                    # Track transforms that were applied
                    applied_transforms = []
                    n_transforms = len(self.transforms)
                    pipeline_start = _time.time()

                    for i, transform_func in enumerate(self.transforms, 1):
                        transform_name = transform_func.__name__
                        input_shape = layer_data.shape
                        n_pixels = layer_data.size

                        self.logger.info(
                            f"  [{i}/{n_transforms}] Applying {transform_name}... "
                            f"(input: {input_shape[0]}×{input_shape[1]} = {n_pixels/1e6:.1f}M pixels)"
                        )
                        transform_start = _time.time()

                        # Apply transform (wrapper always returns 3-tuple)
                        try:
                            layer_data, current_transform, new_crs = transform_func(
                                layer_data, current_transform
                            )

                            # Update CRS if the transform provided a new one
                            if new_crs is not None:
                                current_crs = new_crs

                            applied_transforms.append(transform_name)

                            # Track elevation scale factor if this transform has one
                            if hasattr(transform_func, '_elevation_scale_factor'):
                                scale = transform_func._elevation_scale_factor
                                self.transform_metadata["elevation_scale"] *= scale
                                self.logger.debug(
                                    f"  Updated elevation_scale: {self.transform_metadata['elevation_scale']}"
                                )

                            transform_elapsed = _time.time() - transform_start
                            output_shape = layer_data.shape
                            self.logger.info(
                                f"  [{i}/{n_transforms}] ✓ {transform_name} complete in {transform_elapsed:.1f}s "
                                f"→ {output_shape[0]}×{output_shape[1]}"
                            )
                        except Exception as e:
                            self.logger.error(
                                f"Failed applying transform {transform_name}: {str(e)}"
                            )
                            raise

                    pipeline_elapsed = _time.time() - pipeline_start
                    self.logger.info(
                        f"  All {n_transforms} transforms complete in {pipeline_elapsed:.1f}s total"
                    )

                    # Save result with comprehensive metadata
                    metadata = {
                        "transforms": applied_transforms,
                        "original_shape": layer["data"].shape,
                        "transformed_shape": layer_data.shape,
                        "original_crs": layer["crs"],
                        "final_crs": current_crs,
                    }

                    if cache:
                        self.cache.save(
                            layer_target,
                            layer_data,
                            transform=current_transform,
                            crs=current_crs,
                            metadata=metadata,
                        )

                    # Update layer info
                    self.data_layers[name].update(
                        {
                            "transformed_data": layer_data,
                            "transformed_transform": current_transform,
                            "transformed_crs": current_crs,
                            "transformed": True,
                            "transform_metadata": metadata,
                        }
                    )
                else:
                    self.logger.info(f"Cache hit for {layer_target}")
                    self.data_layers[name].update(
                        {
                            "transformed_data": cached_layer["data"],
                            "transformed_transform": cached_layer["transform"],
                            "transformed_crs": cached_layer.get("crs", layer["crs"]),
                            "transformed": True,
                            "transform_metadata": cached_layer.get("metadata", {}),
                        }
                    )

                pbar.update(1)

        self.logger.info("Transforms applied successfully")

    def configure_for_target_vertices(
        self, target_vertices: int, method: str = "average"
    ) -> float:
        """
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
        """
        if not isinstance(target_vertices, int) or target_vertices <= 0:
            raise ValueError(f"target_vertices must be a positive integer, got {target_vertices}")

        original_h, original_w = self.dem_shape
        original_vertices = original_h * original_w

        if target_vertices > original_vertices:
            self.logger.warning(
                f"Target vertices ({target_vertices:,}) exceeds source vertices "
                f"({original_vertices:,}). Using original resolution (zoom_factor=1.0)."
            )
            zoom_factor = 1.0
        else:
            # Calculate zoom_factor: vertices = (H * zoom) * (W * zoom)
            # So: zoom_factor = sqrt(target_vertices / (H * W))
            zoom_factor = np.sqrt(target_vertices / original_vertices)

        self.logger.info(
            f"Configuring for {target_vertices:,} target vertices\n"
            f"  Original DEM: {original_h} × {original_w} ({original_vertices:,} vertices)\n"
            f"  Calculated zoom_factor: {zoom_factor:.6f}\n"
            f"  Downsampling method: {method}\n"
            f"  Resulting grid: {int(original_h * zoom_factor)} × {int(original_w * zoom_factor)}"
        )

        # Add downsampling transform to the pipeline
        self.transforms.append(downsample_raster(zoom_factor=zoom_factor, method=method))

        return zoom_factor

    def detect_water_highres(
        self,
        slope_threshold: float = 0.01,
        fill_holes: bool = True,
        scale_factor: float = 0.0001,
    ) -> np.ndarray:
        """
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

        Example::

            terrain = Terrain(dem, transform, dem_crs="EPSG:4326")
            terrain.add_transform(reproject_raster(src_crs="EPSG:4326", dst_crs="EPSG:32617"))
            terrain.add_transform(flip_raster(axis="horizontal"))
            terrain.add_transform(scale_elevation(scale_factor=0.0001))
            terrain.configure_for_target_vertices(target_vertices=1_000_000)
            terrain.apply_transforms()  # Includes downsampling

            # Detect water on high-res BEFORE it was downsampled
            water_mask = terrain.detect_water_highres(slope_threshold=0.01)
            # water_mask shape matches downsampled terrain

        Note:
            This method requires that:
            - Transforms have been applied (apply_transforms() called)
            - The DEM layer exists in data_layers
            - Water detection module (src.terrain.water) is available
        """
        if "dem" not in self.data_layers:
            raise ValueError("DEM layer not found. Cannot detect water without DEM.")

        if not self.data_layers["dem"].get("transformed", False):
            raise ValueError(
                "Transforms have not been applied yet. Call apply_transforms() first."
            )

        from src.terrain.water import identify_water_by_slope
        from scipy.ndimage import zoom

        self.logger.info("Detecting water bodies on high-resolution DEM...")

        # Get the original DEM
        original_dem = self.data_layers["dem"]["data"]
        original_transform = self.data_layers["dem"]["transform"]
        original_crs = self.data_layers["dem"]["crs"]

        # Apply all non-downsampling transforms
        highres_dem = original_dem.copy()
        current_transform = original_transform
        current_crs = original_crs

        for transform_func in self.transforms:
            # Skip downsampling transforms - we want high-res
            if "downsample" in transform_func.__name__.lower():
                self.logger.debug(f"Skipping {transform_func.__name__} for high-res water detection")
                continue

            try:
                highres_dem, current_transform, new_crs = transform_func(
                    highres_dem, current_transform
                )
                if new_crs is not None:
                    current_crs = new_crs
            except Exception as e:
                self.logger.error(
                    f"Failed applying transform {transform_func.__name__} for water detection: {e}"
                )
                raise

        self.logger.info(f"  High-res DEM shape: {highres_dem.shape}")

        # Unscale elevation if scale_factor was applied
        if scale_factor is not None and scale_factor != 1.0:
            unscaled_dem = highres_dem / scale_factor
        else:
            unscaled_dem = highres_dem

        # Detect water on high-res DEM
        water_mask_highres = identify_water_by_slope(
            unscaled_dem,
            slope_threshold=slope_threshold,
            fill_holes=fill_holes,
        )

        water_pixels_highres = np.sum(water_mask_highres)
        water_percent_highres = 100 * water_pixels_highres / water_mask_highres.size
        self.logger.info(
            f"  High-res water detection: {water_pixels_highres:,} pixels "
            f"({water_percent_highres:.1f}%)"
        )

        # Now downsample the water mask to match the transformed (downsampled) DEM
        transformed_dem = self.data_layers["dem"]["transformed_data"]
        target_shape = transformed_dem.shape

        if water_mask_highres.shape == target_shape:
            self.logger.info(f"  Water mask already matches target shape {target_shape}")
            return water_mask_highres

        # Calculate zoom factor for downsampling the mask
        zoom_y = target_shape[0] / water_mask_highres.shape[0]
        zoom_x = target_shape[1] / water_mask_highres.shape[1]

        self.logger.info(
            f"  Downsampling water mask: {water_mask_highres.shape} → {target_shape}"
        )

        # Downsample water mask using nearest-neighbor (order=0) to preserve boolean nature
        water_mask_downsampled = zoom(
            water_mask_highres.astype(np.float32),
            zoom=(zoom_y, zoom_x),
            order=0,  # Nearest neighbor for boolean mask
            prefilter=False,
        ).astype(np.bool_)

        water_pixels_final = np.sum(water_mask_downsampled)
        water_percent_final = 100 * water_pixels_final / water_mask_downsampled.size
        self.logger.info(
            f"  Final water mask: {water_pixels_final:,} pixels "
            f"({water_percent_final:.1f}%)"
        )

        return water_mask_downsampled

    def detect_water(
        self,
        slope_threshold: float = 0.01,
        fill_holes: bool = True,
    ) -> np.ndarray:
        """
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
        """
        if "dem" not in self.data_layers:
            raise ValueError("DEM layer not found. Cannot detect water without DEM.")

        if not self.data_layers["dem"].get("transformed", False):
            raise ValueError(
                "Transforms have not been applied yet. Call apply_transforms() first."
            )

        from src.terrain.water import identify_water_by_slope

        self.logger.info("Detecting water bodies on transformed DEM...")

        # Get the transformed DEM
        dem_data = self.data_layers["dem"]["transformed_data"]

        # Auto-detect elevation scale factor from model_params or estimate from data range
        scale_factor = self.model_params.get("elevation_scale", None)

        if scale_factor is None:
            # Estimate: if max elevation is << 1, it was probably scaled
            max_elev = np.nanmax(dem_data)
            if max_elev < 1.0:
                # Likely scaled by 0.0001 (max ~0.03 for 300m terrain)
                scale_factor = 0.0001
                self.logger.debug(f"Auto-detected scale factor: {scale_factor}")
            else:
                scale_factor = 1.0  # Already in meters

        # Unscale to meters for proper slope calculation
        if scale_factor != 1.0:
            unscaled_dem = dem_data / scale_factor
            self.logger.debug(f"Unscaled DEM from {np.nanmax(dem_data):.4f} to {np.nanmax(unscaled_dem):.1f}m")
        else:
            unscaled_dem = dem_data

        # Detect water
        water_mask = identify_water_by_slope(
            unscaled_dem,
            slope_threshold=slope_threshold,
            fill_holes=fill_holes,
        )

        water_pixels = np.sum(water_mask)
        water_percent = 100 * water_pixels / water_mask.size
        self.logger.info(f"  Water detected: {water_pixels:,} pixels ({water_percent:.1f}%)")

        return water_mask

    def set_color_mapping(
        self,
        color_func: Callable[[np.ndarray, ...], np.ndarray],
        source_layers: list[str],
        *,
        color_kwargs: Optional[Dict[str, Any]] = None,
        mask_func: Optional[Callable[[np.ndarray, ...], np.ndarray]] = None,
        mask_layers: Optional[list[str] | str] = None,
        mask_kwargs: Optional[Dict[str, Any]] = None,
        mask_threshold: Optional[float] = None,
    ) -> None:
        """
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
        """
        # Validate source_layers exist
        missing_layers = [name for name in source_layers if name not in self.data_layers]
        if missing_layers:
            raise ValueError(f"Source layers not found: {missing_layers}")

        # Default kwargs dicts
        if color_kwargs is None:
            color_kwargs = {}
        if mask_kwargs is None:
            mask_kwargs = {}

        # Handle mask layer defaults
        if mask_func:
            if mask_layers is None:
                mask_layers = source_layers
            elif isinstance(mask_layers, str):
                mask_layers = [mask_layers]

            # Validate mask_layers exist
            missing_mask_layers = [name for name in mask_layers if name not in self.data_layers]
            if missing_mask_layers:
                raise ValueError(f"Mask layers not found: {missing_mask_layers}")
        else:
            mask_layers = []

        # Store mapping setup
        self.color_mapping = color_func
        self.color_sources = list(source_layers)
        self.color_kwargs = color_kwargs

        self.mask_func = mask_func
        self.mask_sources = mask_layers
        self.mask_kwargs = mask_kwargs
        self.mask_threshold = mask_threshold

        # Logging
        self.logger.info(f"Color function: {color_func.__name__}")
        self.logger.info(f"Color source layers: {source_layers}")
        if color_kwargs:
            self.logger.info(f"Color kwargs: {color_kwargs}")

        if mask_func:
            self.logger.info(f"Mask function: {mask_func.__name__}")
            self.logger.info(f"Mask source layers: {mask_layers}")
            if mask_kwargs:
                self.logger.info(f"Mask kwargs: {mask_kwargs}")
            if mask_threshold is not None:
                self.logger.info(f"Mask threshold: {mask_threshold}")

    def set_blended_color_mapping(
        self,
        base_colormap: Callable[[np.ndarray], np.ndarray],
        base_source_layers: list[str],
        overlay_colormap: Callable[[np.ndarray], np.ndarray],
        overlay_source_layers: list[str],
        overlay_mask: np.ndarray,
        *,
        base_color_kwargs: Optional[Dict[str, Any]] = None,
        overlay_color_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
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
        """
        # Validate source_layers exist
        all_layers = set(base_source_layers) | set(overlay_source_layers)
        missing_layers = [name for name in all_layers if name not in self.data_layers]
        if missing_layers:
            raise ValueError(f"Source layers not found: {missing_layers}")

        # Validate overlay_mask
        if not hasattr(self, "vertices") or self.vertices is None:
            raise RuntimeError(
                "create_mesh() must be called before set_blended_color_mapping(). "
                "Vertex count is needed to validate overlay_mask shape."
            )

        overlay_mask = np.asarray(overlay_mask)
        expected_shape = (len(self.vertices),)
        if overlay_mask.shape != expected_shape:
            raise ValueError(
                f"overlay_mask must have shape {expected_shape} to match vertex count. "
                f"Got {overlay_mask.shape}. Use compute_proximity_mask() to create mask."
            )

        # Default kwargs
        if base_color_kwargs is None:
            base_color_kwargs = {}
        if overlay_color_kwargs is None:
            overlay_color_kwargs = {}

        # Store blended color mapping configuration
        self.color_mapping_mode = "blended"
        self.base_colormap = base_colormap
        self.base_color_sources = list(base_source_layers)
        self.base_color_kwargs = base_color_kwargs
        self.overlay_colormap = overlay_colormap
        self.overlay_color_sources = list(overlay_source_layers)
        self.overlay_color_kwargs = overlay_color_kwargs
        self.overlay_mask = overlay_mask

        self.logger.info("Blended color mapping configured:")
        self.logger.info(f"  Base colormap: {base_colormap.__name__} on {base_source_layers}")
        self.logger.info(f"  Overlay colormap: {overlay_colormap.__name__} on {overlay_source_layers}")
        self.logger.info(
            f"  Overlay mask: {np.sum(overlay_mask)}/{len(overlay_mask)} vertices "
            f"({100.0 * np.sum(overlay_mask) / len(overlay_mask):.1f}%)"
        )

    def set_multi_color_mapping(
        self,
        base_colormap: Callable[[np.ndarray, ...], np.ndarray],
        base_source_layers: list[str],
        overlays: list[dict],
        *,
        base_color_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Apply multiple data layers as overlays with different colormaps and priority.

        This enables flexible data visualization where multiple geographic features
        (roads, trails, land use, power lines, etc.) can each be colored independently
        based on their own colormaps and source layers.

        Args:
            base_colormap: Function mapping base layer(s) to RGB/RGBA. Takes N arrays
                (one per base_source_layers) and returns (H, W, 3) or (H, W, 4).
            base_source_layers: Layer names for base_colormap, e.g., ['dem'].
            overlays: List of overlay specifications. Each overlay dict contains
                ``colormap`` (function returning H,W,3/4), ``source_layers`` (list of
                layer names), ``colormap_kwargs`` (optional dict), ``threshold`` (value
                above which overlay applies, default 0.5), and ``priority`` (lower =
                higher priority).
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
        """
        # Validate all source_layers exist
        all_layers = set(base_source_layers)
        for overlay in overlays:
            all_layers.update(overlay.get("source_layers", []))

        missing_layers = [name for name in all_layers if name not in self.data_layers]
        if missing_layers:
            raise ValueError(f"Source layers not found: {missing_layers}")

        # Validate overlay specs
        for i, overlay in enumerate(overlays):
            if "colormap" not in overlay:
                raise ValueError(f"Overlay {i} missing required 'colormap' key")
            if "source_layers" not in overlay:
                raise ValueError(f"Overlay {i} missing required 'source_layers' key")
            if "priority" not in overlay:
                raise ValueError(f"Overlay {i} missing required 'priority' key")

        # Sort overlays by priority (lower number = higher priority = applied first)
        sorted_overlays = sorted(overlays, key=lambda x: x["priority"])

        # Default kwargs
        if base_color_kwargs is None:
            base_color_kwargs = {}

        # Store multi-overlay configuration
        self.color_mapping_mode = "multi_overlay"
        self.base_colormap = base_colormap
        self.base_color_sources = list(base_source_layers)
        self.base_color_kwargs = base_color_kwargs
        self.overlays = sorted_overlays

        self.logger.info("Multi-overlay color mapping configured:")
        self.logger.info(f"  Base colormap: {base_colormap.__name__} on {base_source_layers}")
        self.logger.info(f"  Number of overlays: {len(overlays)}")
        for i, overlay in enumerate(sorted_overlays):
            has_mask = "mask" in overlay
            mask_info = f", has_mask={has_mask}" if has_mask else ""
            threshold = overlay.get("threshold", "default")
            self.logger.info(
                f"    Overlay {i} (priority {overlay['priority']}): "
                f"{overlay['colormap'].__name__} on {overlay['source_layers']}"
                f" [threshold={threshold}{mask_info}]"
            )

    def compute_colors(self, water_mask=None):
        """
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
        """
        # Check if multi-overlay mode
        if hasattr(self, "color_mapping_mode") and self.color_mapping_mode == "multi_overlay":
            return self._compute_multi_overlay_colors(water_mask=water_mask)

        # Check if blended mode
        if hasattr(self, "color_mapping_mode") and self.color_mapping_mode == "blended":
            return self._compute_blended_colors(water_mask=water_mask)

        # Standard single colormap mode
        if not hasattr(self, "color_mapping") or not hasattr(self, "color_sources"):
            raise ValueError("Color mapping not set. Call set_color_mapping() first.")

        self.logger.info("Computing colors...")

        # Prepare color data arrays
        color_arrays = [
            (
                self.data_layers[layer]["transformed_data"]
                if self.data_layers[layer].get("transformed")
                else self.data_layers[layer]["data"]
            )
            for layer in self.color_sources
        ]

        # Compute base colors
        try:
            colors = self.color_mapping(*color_arrays, **self.color_kwargs)
        except Exception as e:
            self.logger.error(f"Error computing colors: {str(e)}")
            raise

        # Ensure RGBA
        if colors.shape[-1] == 3:
            # Create alpha channel with appropriate max value for the data type
            if colors.dtype == np.uint8:
                alpha_channel = np.full(colors.shape[:2] + (1,), 255, dtype=colors.dtype)
            else:
                alpha_channel = np.ones(colors.shape[:2] + (1,), dtype=colors.dtype)
            colors = np.concatenate([colors, alpha_channel], axis=-1)

        # Apply mask if provided
        if self.mask_func:
            mask_arrays = [
                (
                    self.data_layers[layer]["transformed_data"]
                    if self.data_layers[layer].get("transformed")
                    else self.data_layers[layer]["data"]
                )
                for layer in self.mask_sources
            ]

            try:
                mask = self.mask_func(*mask_arrays, **self.mask_kwargs)
            except Exception as e:
                self.logger.error(f"Error computing mask: {str(e)}")
                raise

            # Apply threshold if provided
            if self.mask_threshold is not None:
                mask = mask >= self.mask_threshold

            # Update alpha channel based on mask
            colors[..., 3] = np.where(mask, 0.0, 1.0)

        self.colors = colors

        self.logger.info(f"Colors computed successfully with shape {colors.shape}")

        return colors

    def _compute_blended_colors(self, water_mask=None):
        """
        Compute colors using blended colormap mode (internal method).

        Computes base colors for entire DEM, overlay colors for overlay zones,
        and blends them according to overlay_mask at the vertex level.

        Args:
            water_mask (np.ndarray, optional): Boolean water mask in grid space (height × width).
                If provided, water pixels will be colored blue in the final vertex colors.

        Returns:
            np.ndarray: RGBA color array with blended colors.
        """
        self.logger.info("Computing blended colors...")

        # Get transformed DEM data to determine grid shape
        dem_data = self.data_layers["dem"]["transformed_data"]
        height, width = dem_data.shape

        # Compute base colors for all pixels
        base_arrays = [
            (
                self.data_layers[layer]["transformed_data"]
                if self.data_layers[layer].get("transformed")
                else self.data_layers[layer]["data"]
            )
            for layer in self.base_color_sources
        ]

        try:
            base_colors_grid = self.base_colormap(*base_arrays, **self.base_color_kwargs)
        except Exception as e:
            self.logger.error(f"Error computing base colors: {str(e)}")
            raise

        # Compute overlay colors for all pixels
        overlay_arrays = [
            (
                self.data_layers[layer]["transformed_data"]
                if self.data_layers[layer].get("transformed")
                else self.data_layers[layer]["data"]
            )
            for layer in self.overlay_color_sources
        ]

        try:
            overlay_colors_grid = self.overlay_colormap(*overlay_arrays, **self.overlay_color_kwargs)
        except Exception as e:
            self.logger.error(f"Error computing overlay colors: {str(e)}")
            raise

        # Ensure both are RGBA
        for colors_grid, name in [(base_colors_grid, "base"), (overlay_colors_grid, "overlay")]:
            if colors_grid.shape[-1] == 3:
                if colors_grid.dtype == np.uint8:
                    alpha = np.full(colors_grid.shape[:2] + (1,), 255, dtype=colors_grid.dtype)
                else:
                    alpha = np.ones(colors_grid.shape[:2] + (1,), dtype=colors_grid.dtype)
                if name == "base":
                    base_colors_grid = np.concatenate([colors_grid, alpha], axis=-1)
                else:
                    overlay_colors_grid = np.concatenate([colors_grid, alpha], axis=-1)

        # Map grid colors to vertex colors using y_valid, x_valid
        # These map vertex index to (row, col) in the grid
        base_vertex_colors = base_colors_grid[self.y_valid, self.x_valid]
        overlay_vertex_colors = overlay_colors_grid[self.y_valid, self.x_valid]

        # Handle boundary vertices if they exist
        # Boundary vertices (from boundary_extension) don't map to grid pixels
        # Use nearest valid pixel color (or default color)
        num_surface_vertices = len(self.y_valid)
        num_total_vertices = len(self.vertices)

        if num_total_vertices > num_surface_vertices:
            # Has boundary vertices - pad with default color
            self.logger.info(
                f"  Padding colors for {num_total_vertices - num_surface_vertices} boundary vertices"
            )
            # Use mean color for boundary (or could use edge colors)
            default_color = np.mean(base_vertex_colors, axis=0).astype(base_vertex_colors.dtype)

            # Extend vertex colors arrays
            base_vertex_colors = np.vstack(
                [base_vertex_colors, np.tile(default_color, (num_total_vertices - num_surface_vertices, 1))]
            )
            overlay_vertex_colors = np.vstack(
                [
                    overlay_vertex_colors,
                    np.tile(default_color, (num_total_vertices - num_surface_vertices, 1)),
                ]
            )

        # Blend colors using overlay_mask: True = overlay, False = base
        colors = np.where(
            self.overlay_mask[:, None],  # Broadcast to (N, 1) for RGBA channels
            overlay_vertex_colors,
            base_vertex_colors,
        )

        num_overlay = np.sum(self.overlay_mask)
        num_base = len(self.overlay_mask) - num_overlay
        self.logger.info(
            f"Blended colors computed: {num_overlay} overlay vertices, {num_base} base vertices"
        )

        # Apply water coloring if water mask provided
        if water_mask is not None:
            self.logger.info(f"Applying water coloring to blended vertex colors...")
            self.logger.debug(f"  Water mask shape: {water_mask.shape}")
            self.logger.debug(f"  DEM shape: {dem_data.shape}")
            self.logger.debug(f"  Num surface vertices: {num_surface_vertices}")
            self.logger.debug(f"  y_valid range: {self.y_valid.min()}-{self.y_valid.max()}")
            self.logger.debug(f"  x_valid range: {self.x_valid.min()}-{self.x_valid.max()}")

            # Create depth gradient for water bodies (lighter at edges, darker at center)
            # Compute distance transform to measure distance from water edges (shores)
            from scipy.ndimage import distance_transform_edt

            water_distances = distance_transform_edt(water_mask)

            # Normalize distances (0 at edge/shore, 1 at center/deepest point)
            max_dist = water_distances.max()
            if max_dist > 0:
                normalized_distances = water_distances / max_dist
            else:
                # Single-pixel water or uniform - use uniform color
                normalized_distances = water_distances

            # Define gradient colors: blue to very dark blue (Great Lakes depth)
            # Edge (shallow water near shore): Bright recognizable blue
            edge_color = np.array([25, 85, 125], dtype=np.float32)  # Medium blue
            # Center (deep water): Very dark blue (suggests depth of Lake Superior: 1,332 ft)
            center_color = np.array([5, 20, 35], dtype=np.float32)  # Deep dark blue

            # Map water mask from grid space to vertex space (vectorized)
            # Only for surface vertices (not boundary vertices)
            water_at_vertices = water_mask[self.y_valid, self.x_valid]
            water_vertex_indices = np.where(water_at_vertices)[0]

            # Get normalized distances for all water vertices
            water_y = self.y_valid[water_vertex_indices]
            water_x = self.x_valid[water_vertex_indices]
            water_depths = normalized_distances[water_y, water_x]

            # Exponential falloff: darkens rapidly from edges to center
            # Use t^2.5 for strong exponential darkening
            t = np.power(water_depths, 2.5)[:, np.newaxis]  # Shape (N, 1) for broadcasting
            water_colors = edge_color * (1 - t) + center_color * t

            # Apply gradient colors to water vertices
            surface_colors = colors[:num_surface_vertices]
            surface_colors[water_vertex_indices, :3] = water_colors.astype(np.uint8)
            water_vertex_count = len(water_vertex_indices)

            self.logger.info(f"Water colored blue ({water_vertex_count} vertices)")

        self.colors = colors

        return colors

    def _compute_multi_overlay_colors(self, water_mask=None):
        """
        Compute colors using multi-overlay mode (internal method).

        Combines base colormap with multiple overlays. For each grid pixel, applies the
        first overlay (by priority) whose source data is non-zero and non-NaN. Falls back
        to base colormap if no overlays match.

        Args:
            water_mask (np.ndarray, optional): Boolean water mask in grid space (height × width).
                If provided, water pixels will be colored blue in the final vertex colors.

        Returns:
            np.ndarray: RGBA color array with multi-overlay colors.
        """
        self.logger.info("Computing multi-overlay colors...")

        # Get transformed DEM data to determine grid shape
        dem_data = self.data_layers["dem"]["transformed_data"]
        height, width = dem_data.shape

        # Compute base colors for all pixels
        base_arrays = [
            (
                self.data_layers[layer]["transformed_data"]
                if self.data_layers[layer].get("transformed")
                else self.data_layers[layer]["data"]
            )
            for layer in self.base_color_sources
        ]

        try:
            base_colors_grid = self.base_colormap(*base_arrays, **self.base_color_kwargs)
        except Exception as e:
            self.logger.error(f"Error computing base colors: {str(e)}")
            raise

        # Ensure base colors are RGBA
        if base_colors_grid.shape[-1] == 3:
            if base_colors_grid.dtype == np.uint8:
                alpha = np.full(base_colors_grid.shape[:2] + (1,), 255, dtype=base_colors_grid.dtype)
            else:
                alpha = np.ones(base_colors_grid.shape[:2] + (1,), dtype=base_colors_grid.dtype)
            base_colors_grid = np.concatenate([base_colors_grid, alpha], axis=-1)

        # Initialize result grid with base colors
        result_colors_grid = np.copy(base_colors_grid)

        # Apply overlays in priority order
        for overlay_idx, overlay in enumerate(self.overlays):
            overlay_colormap = overlay["colormap"]
            overlay_sources = overlay["source_layers"]
            overlay_kwargs = overlay.get("colormap_kwargs", {})
            priority = overlay["priority"]

            # Get overlay source data
            overlay_arrays = [
                (
                    self.data_layers[layer]["transformed_data"]
                    if self.data_layers[layer].get("transformed")
                    else self.data_layers[layer]["data"]
                )
                for layer in overlay_sources
            ]

            try:
                overlay_colors_grid = overlay_colormap(*overlay_arrays, **overlay_kwargs)
            except Exception as e:
                self.logger.error(f"Error computing overlay {overlay_idx} colors: {str(e)}")
                raise

            # Ensure overlay colors are RGBA
            if overlay_colors_grid.shape[-1] == 3:
                if overlay_colors_grid.dtype == np.uint8:
                    alpha = np.full(overlay_colors_grid.shape[:2] + (1,), 255, dtype=overlay_colors_grid.dtype)
                else:
                    alpha = np.ones(overlay_colors_grid.shape[:2] + (1,), dtype=overlay_colors_grid.dtype)
                overlay_colors_grid = np.concatenate([overlay_colors_grid, alpha], axis=-1)

            # Create mask for where this overlay applies
            # Option 1: Explicit mask provided (e.g., park_mask for proximity-based overlays)
            # Option 2: Threshold-based mask from first source layer value
            explicit_mask = overlay.get("mask", None)
            use_explicit_mask = False

            if explicit_mask is not None:
                # Explicit mask provided - must be vertex-space boolean array
                explicit_mask = np.asarray(explicit_mask)
                self.logger.info(
                    f"Overlay {overlay_idx}: explicit mask shape={explicit_mask.shape}, "
                    f"y_valid len={len(self.y_valid)}, grid shape={overlay_arrays[0].shape}, "
                    f"mask sum={np.sum(explicit_mask)}"
                )
                if explicit_mask.shape == (len(self.y_valid),):
                    # Convert vertex mask to grid mask (exact match)
                    overlay_mask = np.zeros(overlay_arrays[0].shape, dtype=bool)
                    overlay_mask[self.y_valid, self.x_valid] = explicit_mask
                    use_explicit_mask = True
                    self.logger.info(
                        f"Overlay {overlay_idx}: converted vertex mask to grid mask, "
                        f"{np.sum(overlay_mask)} grid pixels"
                    )
                elif len(explicit_mask.shape) == 1 and len(explicit_mask) >= len(self.y_valid):
                    # Vertex-space mask but might include boundary vertices
                    # Use only the first len(y_valid) entries (surface vertices)
                    self.logger.info(
                        f"Overlay {overlay_idx}: mask has {len(explicit_mask)} entries, "
                        f"using first {len(self.y_valid)} for surface vertices"
                    )
                    overlay_mask = np.zeros(overlay_arrays[0].shape, dtype=bool)
                    overlay_mask[self.y_valid, self.x_valid] = explicit_mask[:len(self.y_valid)]
                    use_explicit_mask = True
                    self.logger.info(
                        f"Overlay {overlay_idx}: converted vertex mask to grid mask, "
                        f"{np.sum(overlay_mask)} grid pixels"
                    )
                elif explicit_mask.shape == overlay_arrays[0].shape:
                    # Already grid-space
                    overlay_mask = explicit_mask.astype(bool)
                    use_explicit_mask = True
                    self.logger.info(f"Overlay {overlay_idx}: using grid-space mask directly")
                else:
                    self.logger.warning(
                        f"Overlay {overlay_idx}: mask shape {explicit_mask.shape} doesn't match "
                        f"vertex count {len(self.y_valid)} or grid shape {overlay_arrays[0].shape}. "
                        f"Falling back to threshold-based mask."
                    )

            if not use_explicit_mask:
                # Use threshold-based mask from source layer values
                overlay_mask_data = overlay_arrays[0]
                threshold = overlay.get("threshold", 0.5)
                overlay_mask = (overlay_mask_data >= threshold) & ~np.isnan(overlay_mask_data)
                self.logger.info(
                    f"Overlay {overlay_idx}: using threshold={threshold}, "
                    f"{np.sum(overlay_mask)} grid pixels"
                )

            # Apply overlay colors where mask is True
            result_colors_grid[overlay_mask] = overlay_colors_grid[overlay_mask]

            self.logger.debug(
                f"Overlay {overlay_idx} (priority {priority}): "
                f"applied to {np.sum(overlay_mask)} grid pixels"
            )

        # Map grid colors to vertex colors using y_valid, x_valid
        vertex_colors = result_colors_grid[self.y_valid, self.x_valid]

        # Apply water mask if provided
        if water_mask is not None:
            num_surface_vertices = len(self.y_valid)

            self.logger.debug(
                f"Applying water mask to {num_surface_vertices} surface vertices"
            )
            self.logger.debug(f"  DEM shape: {dem_data.shape}")
            self.logger.debug(f"  Num surface vertices: {num_surface_vertices}")
            self.logger.debug(f"  y_valid range: {self.y_valid.min()}-{self.y_valid.max()}")
            self.logger.debug(f"  x_valid range: {self.x_valid.min()}-{self.x_valid.max()}")

            # Create depth gradient for water bodies (lighter at edges, darker at center)
            # Compute distance transform to measure distance from water edges (shores)
            from scipy.ndimage import distance_transform_edt

            water_distances = distance_transform_edt(water_mask)

            # Normalize distances (0 at edge/shore, 1 at center/deepest point)
            max_dist = water_distances.max()
            if max_dist > 0:
                normalized_distances = water_distances / max_dist
            else:
                # Single-pixel water or uniform - use uniform color
                normalized_distances = water_distances

            # Define gradient colors: blue to very dark blue (Great Lakes depth)
            # Edge (shallow water near shore): Bright recognizable blue
            edge_color = np.array([25, 85, 125], dtype=np.float32)  # Medium blue
            # Center (deep water): Very dark blue (suggests depth of Lake Superior: 1,332 ft)
            center_color = np.array([5, 20, 35], dtype=np.float32)  # Deep dark blue

            # Map water mask from grid space to vertex space (vectorized)
            water_at_vertices = water_mask[self.y_valid, self.x_valid]
            water_vertex_indices = np.where(water_at_vertices)[0]

            # Get normalized distances for all water vertices
            water_y = self.y_valid[water_vertex_indices]
            water_x = self.x_valid[water_vertex_indices]
            water_depths = normalized_distances[water_y, water_x]

            # Exponential falloff: darkens rapidly from edges to center
            # Use t^2.5 for strong exponential darkening
            t = np.power(water_depths, 2.5)[:, np.newaxis]  # Shape (N, 1) for broadcasting
            water_colors = edge_color * (1 - t) + center_color * t

            # Apply gradient colors to water vertices
            vertex_colors[water_vertex_indices, :3] = water_colors.astype(np.uint8)
            water_vertex_count = len(water_vertex_indices)

            self.logger.info(f"Water colored blue ({water_vertex_count} vertices)")

        self.colors = vertex_colors

        self.logger.info(f"Multi-overlay colors computed: {len(self.overlays)} overlays applied")
        return vertex_colors

    def create_mesh(
        self,
        base_depth=-0.2,
        boundary_extension=True,
        scale_factor=100.0,
        height_scale=1.0,
        center_model=True,
        verbose=True,
        detect_water=False,
        water_slope_threshold=0.5,
        water_mask=None,
        two_tier_edge=False,
        edge_mid_depth=None,
        edge_base_material="clay",
        edge_blend_colors=True,
        smooth_boundary=False,
        smooth_boundary_window=5,
        use_catmull_rom=False,
        catmull_rom_subdivisions=10,
    ):
        """
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
            two_tier_edge (bool): Enable two-tier edge extrusion (default: False).
                Creates a small colored edge near the surface with a larger uniform base below.
            edge_mid_depth (float): Depth of middle tier for two-tier edge (default: auto-calculated).
                If None, automatically set to base_depth * 0.25 (25% down from surface).
            edge_base_material (str | tuple): Material for base layer (default: "clay").
                Either a preset name ("clay", "obsidian", "chrome", "plastic", "gold", "ivory")
                or an RGB tuple (0-1 range).
            edge_blend_colors (bool): Blend surface colors to mid tier in two-tier mode (default: True).
                If False, mid tier uses the base_material color for sharper transition.
            smooth_boundary (bool): Apply smoothing to boundary points to eliminate stair-step edges
                (default: False). Useful for smoother mesh transitions when using two-tier edge.
            smooth_boundary_window (int): Window size for boundary smoothing (default: 5).
                Larger values produce more smoothing. Only used when smooth_boundary=True.
            use_catmull_rom (bool): Use Catmull-Rom curve fitting for smooth boundary geometry
                (default: False). When enabled, eliminates pixel-grid staircase pattern entirely
                by fitting smooth parametric curve through boundary points.
            catmull_rom_subdivisions (int): Number of interpolated points per boundary segment
                when using Catmull-Rom curves (default: 10). Higher values = smoother curve
                but more vertices. Only used when use_catmull_rom=True.

        Returns:
            bpy.types.Object | None: The created terrain mesh object, or None if creation failed.

        Raises:
            ValueError: If transformed DEM layer is not available (apply_transforms() not called).

        Examples:
            # Default single-tier edge (backwards compatible)
            mesh = terrain.create_mesh(boundary_extension=True)

            # Two-tier edge with default clay base
            mesh = terrain.create_mesh(two_tier_edge=True)

            # Two-tier with gold base material
            mesh = terrain.create_mesh(
                two_tier_edge=True,
                edge_base_material="gold",
            )

            # Two-tier with custom RGB color
            mesh = terrain.create_mesh(
                two_tier_edge=True,
                edge_base_material=(0.6, 0.55, 0.5),
            )
        """
        start_time = time.time()
        self.logger.info("Creating terrain mesh...")

        # Get transformed DEM data
        if "dem" not in self.data_layers or not self.data_layers["dem"].get("transformed", False):
            raise ValueError("Transformed DEM layer required for mesh creation")

        # Compute colors if color mapping is set and colors haven't been computed yet
        if hasattr(self, "color_mapping") and not hasattr(self, "colors"):
            self.compute_colors()

        # Apply water detection if requested
        if detect_water or water_mask is not None:
            if water_mask is None:
                # Get transformed DEM data (needed for both water detection and coloring)
                dem_data = self.data_layers["dem"]["transformed_data"]

                # Compute water mask from transformed DEM (if no pre-computed mask provided)
                self.logger.info(
                    f"Detecting water bodies (slope threshold: {water_slope_threshold})..."
                )
                from src.terrain.water import identify_water_by_slope

                water_mask = identify_water_by_slope(
                    dem_data, slope_threshold=water_slope_threshold, fill_holes=True
                )
            else:
                # Get transformed DEM data for coloring
                dem_data = self.data_layers["dem"]["transformed_data"]

                self.logger.info(
                    f"Using pre-computed water mask ({np.sum(water_mask)} water pixels)"
                )

            # Apply water mask to colors: create depth gradient (lighter at edges, darker at center)
            # Compute distance transform to measure distance from water edges (shores)
            from scipy.ndimage import distance_transform_edt

            water_distances = distance_transform_edt(water_mask)

            # Normalize distances (0 at edge/shore, 1 at center/deepest point)
            max_dist = water_distances.max()
            if max_dist > 0:
                normalized_distances = water_distances / max_dist
            else:
                # Single-pixel water or uniform - use uniform color
                normalized_distances = water_distances

            # Define gradient colors: blue to very dark blue (Great Lakes depth)
            # Edge (shallow water near shore): Bright recognizable blue
            edge_color = np.array([25, 85, 125], dtype=np.float32)  # Medium blue
            # Center (deep water): Very dark blue (suggests depth of Lake Superior: 1,332 ft)
            center_color = np.array([5, 20, 35], dtype=np.float32)  # Deep dark blue

            # If colors exist, overwrite water pixels with gradient
            if hasattr(self, "colors") and self.colors is not None:
                # Extract RGB channels (keep alpha if present)
                has_alpha = self.colors.shape[-1] >= 4
                rgb_data = self.colors[..., :3]

                # Apply depth gradient to water pixels
                water_indices = np.where(water_mask)
                water_depths = normalized_distances[water_indices]

                # Exponential falloff: darkens rapidly from edges to center
                # Use t^2.5 for strong exponential darkening
                t = np.power(water_depths, 10)[:, np.newaxis]  # Shape (N, 1) for broadcasting
                water_colors = edge_color * (1 - t) + center_color * t

                # Apply gradient colors to water pixels
                rgb_data[water_indices] = water_colors.astype(np.uint8)

                # Reconstruct with alpha channel if needed
                if has_alpha:
                    self.colors[..., :3] = rgb_data
                else:
                    self.colors = rgb_data

                self.logger.info(f"Water colored with depth gradient ({np.sum(water_mask)} water pixels)")
            else:
                # No colors yet, create them with water coloring
                height, width = dem_data.shape
                if dem_data.dtype == np.float32 or dem_data.dtype == np.float64:
                    # Normalize elevation for color
                    valid = ~np.isnan(dem_data)
                    if np.any(valid):
                        min_elev = np.nanmin(dem_data)
                        max_elev = np.nanmax(dem_data)
                        normalized = np.zeros_like(dem_data)
                        normalized[valid] = (dem_data[valid] - min_elev) / (
                            max_elev - min_elev + 1e-8
                        )
                    else:
                        normalized = np.zeros_like(dem_data)

                    # Create grayscale RGB from elevation
                    gray = np.clip(normalized * 255, 0, 255).astype(np.uint8)
                    rgb = np.stack([gray, gray, gray], axis=-1)

                    # Apply water depth gradient coloring (not flat color)
                    water_indices = np.where(water_mask)
                    water_depths = normalized_distances[water_indices]
                    t = np.power(water_depths, 10)[:, np.newaxis]
                    water_colors = edge_color * (1 - t) + center_color * t
                    rgb[water_indices] = water_colors.astype(np.uint8)

                    self.colors = rgb
                else:
                    # If DEM is already scaled, use it directly
                    gray = np.clip(dem_data, 0, 255).astype(np.uint8)
                    rgb = np.stack([gray, gray, gray], axis=-1)

                    # Apply water depth gradient coloring (not flat color)
                    water_indices = np.where(water_mask)
                    water_depths = normalized_distances[water_indices]
                    t = np.power(water_depths, 10)[:, np.newaxis]
                    water_colors = edge_color * (1 - t) + center_color * t
                    rgb[water_indices] = water_colors.astype(np.uint8)

                    self.colors = rgb

                self.logger.info(
                    f"Colors created with water depth gradient coloring ({np.sum(water_mask)} water pixels)"
                )

        dem_data = self.data_layers["dem"]["transformed_data"]
        height, width = dem_data.shape

        # Create valid points mask (non-NaN values)
        valid_mask = ~np.isnan(dem_data)

        # Generate vertex positions with scaling
        self.logger.info("Generating vertex positions...")
        from src.terrain.mesh_operations import generate_vertex_positions

        positions, y_valid, x_valid = generate_vertex_positions(
            dem_data, valid_mask, scale_factor, height_scale
        )

        # Center the model if requested
        if center_model:
            self.logger.info("Centering model at origin...")
            # Calculate centroid
            centroid = np.mean(positions, axis=0)
            # Center horizontally (x, y) but preserve elevation (z)
            positions[:, 0] -= centroid[0]
            positions[:, 1] -= centroid[1]

            # Store offset for later reference (camera positioning)
            self.model_offset = centroid
        else:
            self.model_offset = np.array([0, 0, 0])

        # Store model parameters for reference
        self.model_params = {
            "scale_factor": scale_factor,
            "height_scale": height_scale,
            "centered": center_model,
            "offset": self.model_offset.tolist(),
            "base_depth": base_depth,
            "two_tier_edge": two_tier_edge,
            "edge_mid_depth": edge_mid_depth,
            "edge_base_material": edge_base_material,
            "edge_blend_colors": edge_blend_colors,
        }

        # Create mapping from (y,x) coords to vertex indices - using dictionaries for O(1) lookups
        self.logger.info("Creating coordinate to index mapping...")
        coord_to_index = {(y, x): i for i, (y, x) in enumerate(zip(y_valid, x_valid))}

        # OPTIMIZATION: Find boundary points using morphological operations
        self.logger.info("Finding boundary points with optimized algorithm...")
        from src.terrain.mesh_operations import find_boundary_points

        boundary_coords = find_boundary_points(valid_mask)

        # Only sort boundary points if needed (they're used for side faces)
        if boundary_extension:
            boundary_points = self._sort_boundary_points_optimized(boundary_coords)
        else:
            boundary_points = boundary_coords

        # OPTIMIZATION: Vectorized face generation using NumPy operations
        self.logger.info("Generating faces with vectorized operations...")
        from src.terrain.mesh_operations import generate_faces

        faces = generate_faces(height, width, coord_to_index)

        # Handle boundary extension if needed
        if boundary_extension:
            self.logger.info("Creating optimized boundary extension...")
            from src.terrain.mesh_operations import create_boundary_extension

            # Get surface colors for two-tier edge (if available and two_tier_edge enabled)
            surface_colors = None
            if two_tier_edge and hasattr(self, "colors") and self.colors is not None:
                # Flatten colors if they're 2D grid (H, W, 3) to 1D (N, 3)
                if self.colors.ndim == 3:
                    surface_colors = self.colors[y_valid, x_valid, :]
                else:
                    surface_colors = self.colors

            # Call create_boundary_extension with two-tier and smoothing parameters
            result = create_boundary_extension(
                positions,
                boundary_points,
                coord_to_index,
                base_depth,
                two_tier=two_tier_edge,
                mid_depth=edge_mid_depth,
                base_material=edge_base_material,
                blend_edge_colors=edge_blend_colors,
                surface_colors=surface_colors,
                smooth_boundary=smooth_boundary,
                smooth_window_size=smooth_boundary_window,
                use_catmull_rom=use_catmull_rom,
                catmull_rom_subdivisions=catmull_rom_subdivisions,
            )

            # Handle return value (2-tuple for single-tier, 3-tuple for two-tier)
            if two_tier_edge:
                boundary_vertices, boundary_faces, boundary_colors = result
                # Store boundary_colors separately for Blender vertex coloring
                # Don't extend self.colors - it stays as 2D grid for surface vertices
                # Boundary colors will be applied directly to mesh vertices after creation
                self.boundary_colors = boundary_colors
            else:
                boundary_vertices, boundary_faces = result
                self.boundary_colors = None

            # Extend vertices with boundary vertices
            vertices = np.vstack([positions, boundary_vertices])
            # Add boundary faces to complete the mesh
            faces.extend(boundary_faces)
        else:
            vertices = positions

        # Store vertices and faces for later use (e.g., proximity calculations)
        self.vertices = vertices
        self.faces = faces
        self.y_valid = y_valid
        self.x_valid = x_valid

        # Create the Blender mesh
        try:
            from src.terrain.blender_integration import create_blender_mesh

            # Prepare colors if available
            colors = self.colors if hasattr(self, "colors") else None
            boundary_colors = self.boundary_colors if hasattr(self, "boundary_colors") else None

            obj = create_blender_mesh(
                vertices,
                faces,
                colors=colors,
                y_valid=y_valid,
                x_valid=x_valid,
                boundary_colors=boundary_colors,
                name="TerrainMesh",
                logger=self.logger,
            )

            elapsed = time.time() - start_time
            self.logger.info(f"Terrain mesh created successfully in {elapsed:.2f} seconds")
            self.terrain_obj = obj
            return obj

        except Exception as e:
            self.logger.error(f"Error creating terrain mesh: {str(e)}")
            raise

    def _sort_boundary_points_optimized(self, boundary_coords):
        """
        Sort boundary points efficiently using spatial relationships.

        Args:
            boundary_coords: List of (y, x) coordinate tuples

        Returns:
            list: Sorted boundary points forming a continuous path
        """
        from src.terrain.mesh_operations import sort_boundary_points

        return sort_boundary_points(boundary_coords)

    def geo_to_mesh_coords(
        self,
        lon: np.ndarray | float,
        lat: np.ndarray | float,
        elevation_offset: float = 0.0,
        input_crs: str = "EPSG:4326",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
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
        """
        from pyproj import Transformer

        # Check prerequisites
        if not hasattr(self, "model_params") or self.model_params is None:
            raise RuntimeError(
                "create_mesh() must be called before geo_to_mesh_coords(). "
                "The mesh parameters are needed for coordinate conversion."
            )

        dem_info = self.data_layers.get("dem", {})
        if not dem_info.get("transformed", False):
            raise ValueError(
                "DEM layer must be transformed before geo_to_mesh_coords(). "
                "Call apply_transforms() first."
            )

        # Get transformed DEM and its transform
        dem_data = dem_info["transformed_data"]
        transform = dem_info.get("transformed_transform")
        dem_crs = dem_info.get("transformed_crs", "EPSG:4326")

        if transform is None:
            raise ValueError("No transform found for transformed DEM layer.")

        # Convert to arrays for uniform handling
        lon_arr = np.atleast_1d(np.asarray(lon, dtype=np.float64))
        lat_arr = np.atleast_1d(np.asarray(lat, dtype=np.float64))
        scalar_input = lon_arr.shape == (1,) and not hasattr(lon, "__len__")

        if lon_arr.shape != lat_arr.shape:
            raise ValueError(
                f"lon and lat must have the same shape. Got {lon_arr.shape} and {lat_arr.shape}"
            )

        # Reproject coordinates if input CRS differs from DEM CRS
        if input_crs != dem_crs:
            self.logger.debug(f"  Reprojecting {len(lon_arr)} points from {input_crs} to {dem_crs}")
            transformer = Transformer.from_crs(input_crs, dem_crs, always_xy=True)
            lon_arr, lat_arr = transformer.transform(lon_arr, lat_arr)

        # Convert geographic coords to pixel coords using inverse transform
        # Affine: (col, row) = ~transform * (x, y) where x=easting, y=northing
        inv_transform = ~transform
        cols, rows = inv_transform * (lon_arr, lat_arr)

        # Round to nearest pixel
        rows = np.round(rows).astype(int)
        cols = np.round(cols).astype(int)

        # Get DEM shape
        height, width = dem_data.shape

        # Clamp to valid range and track out-of-bounds points
        valid_mask = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        rows_clamped = np.clip(rows, 0, height - 1)
        cols_clamped = np.clip(cols, 0, width - 1)

        # Get elevation values from DEM
        elevations = dem_data[rows_clamped, cols_clamped]

        # Handle out-of-bounds or NaN elevations
        invalid_mask = ~valid_mask | np.isnan(elevations)
        if np.any(invalid_mask):
            # Use mean elevation for invalid points
            valid_elevations = dem_data[~np.isnan(dem_data)]
            mean_elev = np.mean(valid_elevations) if len(valid_elevations) > 0 else 0.0
            elevations = np.where(invalid_mask, mean_elev, elevations)
            self.logger.debug(
                f"  {np.sum(invalid_mask)} points outside DEM bounds, using mean elevation"
            )

        # Apply mesh scaling (same as generate_vertex_positions)
        scale_factor = self.model_params["scale_factor"]
        height_scale = self.model_params["height_scale"]

        x = cols.astype(np.float64) / scale_factor
        y = rows.astype(np.float64) / scale_factor
        z = elevations * height_scale + elevation_offset

        # Apply centering offset if model was centered
        if self.model_params.get("centered", False):
            offset = self.model_params["offset"]
            x = x - offset[0]
            y = y - offset[1]
            # Note: z offset not applied - elevation_offset handles vertical positioning

        # Return scalars if input was scalar
        if scalar_input:
            return float(x[0]), float(y[0]), float(z[0])

        return x, y, z

    def compute_proximity_mask(
        self,
        lons: np.ndarray,
        lats: np.ndarray,
        radius_meters: float,
        input_crs: str = "EPSG:4326",
        cluster_threshold_meters: Optional[float] = None,
    ) -> np.ndarray:
        """
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
        """
        from scipy.spatial import KDTree

        # Check prerequisites
        if not hasattr(self, "vertices") or self.vertices is None:
            raise RuntimeError(
                "create_mesh() must be called before compute_proximity_mask(). "
                "Mesh vertices are needed for proximity calculations."
            )

        # Validate inputs
        lons = np.asarray(lons)
        lats = np.asarray(lats)
        if lons.shape != lats.shape:
            raise ValueError(f"lons and lats must have same shape. Got {lons.shape} and {lats.shape}")

        # Filter out NaN coordinates (missing park locations, etc.)
        valid_mask = ~(np.isnan(lons) | np.isnan(lats))
        if not np.all(valid_mask):
            num_invalid = np.sum(~valid_mask)
            self.logger.warning(
                f"Filtering out {num_invalid} points with NaN coordinates "
                f"({len(lons) - num_invalid} valid points remaining)"
            )
            lons = lons[valid_mask]
            lats = lats[valid_mask]

        # Handle edge case: no valid points
        if len(lons) == 0:
            self.logger.warning("No valid points remaining after filtering NaN coordinates")
            return np.zeros(len(self.vertices), dtype=bool)

        # Convert geographic coords to mesh space (x, y only, ignore z)
        xs, ys, _ = self.geo_to_mesh_coords(lons, lats, input_crs=input_crs)
        point_coords = np.column_stack([xs, ys])

        # Filter out points that ended up with NaN mesh coordinates (outside DEM bounds)
        mesh_valid_mask = ~(np.isnan(point_coords[:, 0]) | np.isnan(point_coords[:, 1]))
        if not np.all(mesh_valid_mask):
            num_invalid = np.sum(~mesh_valid_mask)
            self.logger.warning(
                f"Filtering out {num_invalid} points outside DEM bounds "
                f"({np.sum(mesh_valid_mask)} valid points remaining)"
            )
            point_coords = point_coords[mesh_valid_mask]

        # Handle edge case: no valid points after mesh coordinate conversion
        if len(point_coords) == 0:
            self.logger.warning("No points within DEM bounds for proximity mask")
            return np.zeros(len(self.vertices), dtype=bool)

        self.logger.info(f"Computing proximity mask for {len(point_coords)} points...")

        # Get pixel size from transformed DEM for metric conversions
        dem_info = self.data_layers.get("dem", {})
        transformed_transform = dem_info.get("transformed_transform")
        if transformed_transform is None:
            raise ValueError("Transformed DEM transform not found")

        # Pixel size in meters (assumes metric CRS like UTM after transformation)
        pixel_size_meters = abs(transformed_transform.a)
        scale_factor = self.model_params["scale_factor"]

        # Calculate meters per mesh unit
        # 1 mesh unit = scale_factor pixels = scale_factor * pixel_size_meters
        meters_per_mesh_unit = scale_factor * pixel_size_meters

        self.logger.debug(
            f"  Pixel size: {pixel_size_meters:.2f}m, "
            f"Scale factor: {scale_factor}, "
            f"Meters per mesh unit: {meters_per_mesh_unit:.2f}m"
        )

        # Cluster nearby points using DBSCAN
        if cluster_threshold_meters is not None:
            from sklearn.cluster import DBSCAN

            # Convert cluster threshold from meters to mesh units
            cluster_threshold_mesh = cluster_threshold_meters / meters_per_mesh_unit

            self.logger.info(
                f"  Clustering points with threshold {cluster_threshold_meters}m "
                f"({cluster_threshold_mesh:.3f} mesh units)..."
            )

            clustering = DBSCAN(eps=cluster_threshold_mesh, min_samples=1)
            labels = clustering.fit_predict(point_coords)
            num_clusters = labels.max() + 1

            # Use cluster centroids instead of individual points
            point_coords = np.array(
                [point_coords[labels == i].mean(axis=0) for i in range(num_clusters)]
            )

            self.logger.info(f"  Clustered {len(lons)} points into {num_clusters} zones")

        # Build KDTree for efficient spatial queries
        tree = KDTree(point_coords)

        # Get mesh vertex positions (just x, y for 2D distance)
        # vertices includes boundary vertices if boundary_extension=True
        mesh_verts_2d = self.vertices[:, :2]

        # Convert radius from meters to mesh units
        radius_mesh = radius_meters / meters_per_mesh_unit

        self.logger.info(
            f"  Querying vertices within {radius_meters}m ({radius_mesh:.3f} mesh units) "
            f"of {len(point_coords)} point(s)..."
        )

        # Query: which vertices are within radius of ANY point?
        distances, _ = tree.query(mesh_verts_2d, k=1)
        mask = distances <= radius_mesh

        num_in_zone = np.sum(mask)
        pct_in_zone = 100.0 * num_in_zone / len(mask)
        self.logger.info(
            f"  Proximity mask: {num_in_zone}/{len(mask)} vertices ({pct_in_zone:.1f}%) in zones"
        )

        return mask
