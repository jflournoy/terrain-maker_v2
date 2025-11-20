import os
from pathlib import Path
import glob
import bpy
import pickle
import hashlib
from dataclasses import dataclass
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
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

def setup_logging(logger_name, log_file=None):
    """Configure logging to both file and console with different levels"""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # # File handler (DEBUG and above)
    # if log_file is None:
    #     log_file = f'rayshade_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # file_handler = logging.FileHandler(log_file)
    # file_handler.setLevel(logging.DEBUG)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)
    
    return logger

def get_logger(logger_name):
    existing_logger = logging.getLogger(logger_name)
    if not existing_logger.handlers:  # Check if handlers already exist
        return setup_logging(logger_name)
    return existing_logger

logger = get_logger(__name__)

def preview_combined_terrain(slopes, sledding_score, cmap_name='mako', min_saturation=0.1):
    """
    Preview the terrain with combined slope and sledding score visualization.
    
    Args:
        slopes: The slope data array
        sledding_score: The sledding score array (0-1)
        cmap_name: Base colormap name
        min_saturation: Minimum saturation for areas with zero sledding score
    """
    # Create figure with two subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    original_slopes_trans = slopes
    if process_values is not None:
        original_slopes_trans = process_values(slopes)
    
    # Plot 1: Original slopes
    cmap = plt.cm.get_cmap(cmap_name)
    slope_plot = ax1.imshow(original_slopes_trans, cmap=cmap)
    plt.colorbar(slope_plot, ax=ax1)
    ax1.set_title('Slopes')
    
    # Plot 2: Sledding score
    score_plot = ax2.imshow(sledding_score, cmap='viridis')
    plt.colorbar(score_plot, ax=ax2)
    ax2.set_title('Sledding Score')
    
    # Plot 3: Combined visualization
    # Generate combined colors
    colors = create_combined_colormap(slopes, sledding_score, cmap_name)
    
    # Display combined visualization
    combined_plot = ax3.imshow(colors)
    ax3.set_title('Combined Visualization')
    
    plt.tight_layout()
    plt.show()
    
    # Also show histograms of the data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.hist(slopes[~np.isnan(slopes)].ravel(), bins=50)
    ax1.set_title('Slope Distribution')
    ax1.set_xlabel('Slope')
    ax1.set_ylabel('Count')
    
    ax2.hist(sledding_score[~np.isnan(sledding_score)].ravel(), bins=50)
    ax2.set_title('Sledding Score Distribution')
    ax2.set_xlabel('Score')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()

def modify_color_by_score(base_color, score, min_saturation=0.1):
    """
    Modify a color's saturation based on a score while preserving its hue.
    
    Args:
        base_color: RGB color tuple from Mako colormap
        score: Value between 0 and 1
        min_saturation: Minimum saturation for score=0
    
    Returns:
        Modified RGB color
    """
    # Convert RGB to HSV
    h, s, v = colorsys.rgb_to_hsv(*base_color)
    
    # Modify saturation based on score
    # Keep some minimum saturation even with score=0
    new_s = s * (min_saturation + (1 - min_saturation) * score)
    
    # Convert back to RGB
    return colorsys.hsv_to_rgb(h, new_s, v)

def create_combined_colormap(slopes, sledding_score, cmap_name='mako'):
    """
    Create combined colors using slopes for base color and sledding score for saturation.
    
    Args:
        slopes: Slope values array
        sledding_score: Sledding score array (0 to 1)
        cmap_name: Base colormap name
    
    Returns:
        Array of RGB colors
    """
    logger = get_logger(__name__)
    logger.info("Creating combined slope-sledding colormap...")
    
    # Get the base colormap
    cmap = plt.cm.get_cmap(cmap_name)
    
    # Normalize slope values to 0-1
    norm_slopes = (slopes - np.nanmin(slopes)) / (np.nanmax(slopes) - np.nanmin(slopes))
    
    # Create output array
    colors = np.zeros((*slopes.shape, 3))
    
    # Apply the transformation with progress bar
    valid_mask = ~np.isnan(norm_slopes) & ~np.isnan(sledding_score)
    total_pixels = np.sum(valid_mask)
    
    logger.info(f"Processing {total_pixels} valid pixels...")
    
    # Process in chunks for efficiency
    chunk_size = 10000
    flat_indices = np.where(valid_mask.ravel())[0]
    
    with tqdm(total=total_pixels, desc="Generating colors") as pbar:
        for i in range(0, len(flat_indices), chunk_size):
            chunk_indices = flat_indices[i:i+chunk_size]
            
            # Get base colors from colormap
            base_colors = cmap(norm_slopes.ravel()[chunk_indices])[:, :3]
            scores = sledding_score.ravel()[chunk_indices]
            
            # Modify each color based on score
            modified_colors = np.array([
                modify_color_by_score(color, score)
                for color, score in zip(base_colors, scores)
            ])
            
            # Insert into output array
            y_indices, x_indices = np.unravel_index(chunk_indices, slopes.shape)
            colors[y_indices, x_indices] = modified_colors
            
            pbar.update(len(chunk_indices))
    
    logger.info("Color generation complete")
    return colors

@dataclass
class TerrainCache:
    dem_cache_dir: str = "dem_cache"
    blend_path: str = "terrain_cache.blend"
    
    def __post_init__(self):
        os.makedirs(self.dem_cache_dir, exist_ok=True)
    
    def _make_key(self, step_name: str, args: dict) -> str:
        args_str = str(sorted(args.items()))
        return hashlib.md5(f"{step_name}:{args_str}".encode()).hexdigest()
    
    def save_dem(self, args: dict, data) -> None:
        key = self._make_key('preprocess', args)
        path = os.path.join(self.dem_cache_dir, f"{key}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(data, f)
            
    def load_dem(self, args: dict):
        key = self._make_key('preprocess', args)
        path = os.path.join(self.dem_cache_dir, f"{key}.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None

def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

def horn_slope(dem, window_size=3):
    """Calculate slope using an extended Horn's method with NaN handling and configurable window size
    
    Args:
        dem: Input DEM array
        window_size: Size of the window for slope calculation (odd integer, default=5)
    """
    logger.info(f"Computing Horn slope for DEM shape: {dem.shape}")
    logger.info(f"Input DEM value range: {np.nanmin(dem):.2f} to {np.nanmax(dem):.2f}")
    logger.info(f"Input NaN count: {np.sum(np.isnan(dem))}")
    
    # Save the original NaN mask
    nan_mask = np.isnan(dem)
    
    # Fill NaN values with interpolation
    dem_filled = dem.copy()
    mask = np.isnan(dem_filled)
    dem_filled[mask] = np.interp(np.flatnonzero(mask), 
                                np.flatnonzero(~mask), 
                                dem_filled[~mask])
    
    logger.info(f"After filling NaNs - value range: {np.nanmin(dem_filled):.2f} to {np.nanmax(dem_filled):.2f}")
    
    # Calculate gradients using Horn's method
    dx = ndimage.convolve(dem_filled, np.array([[-1, 0, 1],
                                               [-2, 0, 2],
                                               [-1, 0, 1]]) / 8.0)
    dy = ndimage.convolve(dem_filled, np.array([[-1, -2, -1],
                                               [0, 0, 0],
                                               [1, 2, 1]]) / 8.0)
    
    logger.info(f"Gradient ranges - dx: {np.nanmin(dx):.2f} to {np.nanmax(dx):.2f}")
    logger.info(f"Gradient ranges - dy: {np.nanmin(dy):.2f} to {np.nanmax(dy):.2f}")
    
    # Calculate slope magnitude
    slope = np.hypot(dx, dy)
    
    # Restore NaN values to their original locations
    slope[nan_mask] = np.nan
    
    logger.info(f"Output slope value range: {np.nanmin(slope):.2f} to {np.nanmax(slope):.2f}")
    logger.info(f"Output NaN count: {np.sum(np.isnan(slope))}")
    
    return slope

def load_drive_time_data(dem_data, utm_transform, meters_per_pixel, buffer_size, tol):
    """
    Load and process drive time polygons to match DEM data.
    
    Args:
        dem_data: The processed DEM data array
        utm_transform: The Affine transform from UTM coordinates to pixel space
    """
    
    logger.info("Loading drive time data...")
    drive_time = gpd.read_file("1_to_5_hr_drive_times.geojson")
    
    # Fix any invalid geometries
    logger.info("Validating geometries...")
    drive_time['geometry'] = drive_time.geometry.apply(make_valid)
    
    # Project to UTM Zone 17N to match DEM
    logger.info("Projecting to UTM...")
    drive_time = drive_time.to_crs("EPSG:32617")
    
    # Transform from UTM coordinates to pixel coordinates using the affine transform
    def transform_to_pixels(geom):
        if geom.geom_type == 'MultiPolygon':
            polygons = [transform_to_pixels(poly) for poly in geom.geoms]
            return shapely.geometry.MultiPolygon(polygons)
        
        # Get coordinates
        coords = np.array(geom.exterior.coords)
        
        # Transform from UTM to pixel coordinates
        pixel_x = (coords[:, 0] - utm_transform.c) / meters_per_pixel
        pixel_y = (coords[:, 1] - utm_transform.f) / meters_per_pixel + dem_data.shape[0]
        
        # Create new polygon with pixel coordinates
        return shapely.geometry.Polygon(zip(pixel_x, pixel_y))
    
    # Apply the transformation
    drive_time['geometry'] = drive_time.geometry.apply(transform_to_pixels)

    # Smooth the geometries
    buffer_size = meters_per_pixel * buffer_size/100
    simplify_tolerance = meters_per_pixel * tol/100  # Adjust as needed
    
    logger.info("Smoothing geometries...")
    drive_time['geometry'] = (
        drive_time.geometry
        .simplify(tolerance=simplify_tolerance, preserve_topology=True)
        .buffer(buffer_size, join_style=2, cap_style=3)
        .buffer(-buffer_size, join_style=2, cap_style=3)
        .buffer(buffer_size/2, join_style=1, cap_style=3)
        .buffer(-buffer_size/2, join_style=1, cap_style=3)
    )
    
    logger.info(f"Processed {len(drive_time)} drive time polygons")
    logger.debug(f"Drive time bounds: {drive_time.total_bounds}")
    
    return drive_time
    
    
def process_terrain_with_cache(dem_files, cache: TerrainCache, 
                             zoom_factor=0.1, fill_value=-9999, 
                             downsample_order=4, scale_z=0.2,
                             colormap_name='twilight_r',
                             base_depth=-0.2,
                             process_values=None,
                             force_dem_reprocess=False,
                             force_mesh_reprocess=False):
    """Process terrain with both DEM and mesh caching"""
    
    logger.info(f"Starting terrain processing with {len(dem_files)} DEM files")
    logger.debug(f"Processing parameters: zoom_factor={zoom_factor}, fill_value={fill_value}, "
                f"downsample_order={downsample_order}, scale_z={scale_z}")
    
    # Initialize return variables
    processed_dem = None
    processing_steps = None
    slopes = None
    terrain = None
    utm_transform = None
    
    # First try to load preprocessed DEM data
    dem_args = {
        'zoom_factor': zoom_factor,
        'fill_value': fill_value,
        'downsample_order': downsample_order,
        'dem_hash': hashlib.md5(str(dem_files).encode()).hexdigest()
    }
    logger.debug(f"DEM cache key parameters: {dem_args}")
    
    # Handle DEM processing
    if not force_dem_reprocess:
        logger.info("Attempting to load DEM from cache...")
        dem_data = cache.load_dem(dem_args)
        if dem_data is not None:
            logger.info("Successfully loaded preprocessed DEM from cache")
            processed_dem, processing_steps, slopes, utm_transform = dem_data
            logger.debug(f"Loaded DEM shape: {processed_dem.shape}")
            logger.debug(f"Loaded slopes range: {np.nanmin(slopes):.2f} to {np.nanmax(slopes):.2f}")
        else:
            logger.info("No cached DEM found, will process from scratch")
            force_dem_reprocess = True
            force_mesh_reprocess = True
    
    if force_dem_reprocess:
        logger.info("Processing DEM from scratch...")
        try:
            elevation, transform = merge(dem_files)
            logger.info(f"Merged DEM shape: {elevation.shape}")
            logger.debug(f"Transform matrix: {transform}")
            
            logger.info("Reprojecting to UTM...")
            dem_utm, utm_transform = reproject_dem_to_utm(elevation[0,:,:], transform, 'EPSG:4326')
            logger.info(f"UTM DEM shape: {dem_utm.shape}")
            logger.debug(f"UTM transform matrix: {utm_transform}")
            
            logger.info("Preprocessing DEM...")
            processed_dem, processing_steps, slopes = preprocess_dem(
                dem_utm,
                zoom_factor=zoom_factor,
                fill_value=fill_value,
                downsample_order=downsample_order
            )
            
            logger.info(f"Processed DEM shape: {processed_dem.shape}")
            logger.debug(f"Value ranges after preprocessing:")
            for step, data in processing_steps.items():
                if isinstance(data, np.ndarray):
                    logger.debug(f"  {step}: {np.nanmin(data):.2f} to {np.nanmax(data):.2f}")
            
            logger.info("Caching processed DEM...")
            cache.save_dem(dem_args, (processed_dem, processing_steps, slopes, utm_transform))
            
        except Exception as e:
            logger.error(f"Error during DEM processing: {str(e)}", exc_info=True)
            raise
    
    # Handle mesh processing
    should_create_new_mesh = force_mesh_reprocess
    if not force_mesh_reprocess and os.path.exists(cache.blend_path):
        logger.info(f"Loading cached terrain from blend file: {cache.blend_path}")
        try:
            bpy.ops.wm.open_mainfile(filepath=cache.blend_path)
            terrain = bpy.data.objects.get("Terrain")
            if not terrain:
                logger.warning("No 'Terrain' object found in blend file")
                should_create_new_mesh = True
            else:
                logger.info("Successfully loaded cached terrain")
                logger.debug(f"Terrain vertices: {len(terrain.data.vertices)}")
                logger.debug(f"Terrain faces: {len(terrain.data.polygons)}")
        except Exception as e:
            logger.error(f"Error loading blend file: {str(e)}", exc_info=True)
            should_create_new_mesh = True
    else:
        should_create_new_mesh = True
    
    # Create new mesh if needed
    if should_create_new_mesh:
        logger.info("Creating new terrain mesh...")
        try:
            clear_scene()
            terrain, dem_structure, color_layer = create_terrain_mesh(
                processed_dem,
                values=slopes,
                process_values=process_values,
                scale_z=scale_z,
                colormap_name=colormap_name,
                base_depth=base_depth
            )
            
            logger.info(f"Created terrain mesh with {len(terrain.data.vertices)} vertices")
            logger.debug(f"Color layer '{color_layer.name}' created with {len(color_layer.data)} elements")
            
            logger.info(f"Saving blend file to: {cache.blend_path}")
            bpy.ops.wm.save_as_mainfile(filepath=cache.blend_path)
            
        except Exception as e:
            logger.error(f"Error creating terrain mesh: {str(e)}", exc_info=True)
            raise
    
    return processed_dem, processing_steps, slopes, terrain, utm_transform

def reproject_dem_to_utm(dem_data, src_transform, src_crs, num_threads=15):
    logger.info(f"Reprojecting DEM with shape: {dem_data.shape}")
    logger.info(f"Input value range: {np.nanmin(dem_data):.2f} to {np.nanmax(dem_data):.2f}")
    logger.info(f"Input NaN count: {np.sum(np.isnan(dem_data))}")

    dst_crs = 'EPSG:32617'
    
    transform, width, height = calculate_default_transform(
        src_crs, dst_crs, 
        dem_data.shape[1], dem_data.shape[0],
        *rasterio.transform.array_bounds(dem_data.shape[0], dem_data.shape[1], src_transform)
    )
    
    logger.info(f"Calculated UTM dimensions: {width}x{height}")
    
    dst_dem = np.full((height, width), np.nan, dtype=np.float32)

    with rasterio.Env(NUM_THREADS=num_threads):
        reproject(
            source=dem_data,
            destination=dst_dem,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            dst_nodata=np.nan,
            num_threads=num_threads,
            warp_mem_limit=512  # Increase memory limit for larger chunks
        )
    
    
    logger.info(f"Reprojected value range: {np.nanmin(dst_dem):.2f} to {np.nanmax(dst_dem):.2f}")
    logger.info(f"Reprojected NaN count: {np.sum(np.isnan(dst_dem))} (proportion of total: {np.sum(np.isnan(dst_dem)) / (width*height)}")
    
    return dst_dem, transform

def preprocess_dem(dem_data, zoom_factor=1/10, fill_value = -999999.0, downsample_order=4):
    """
    Preprocess DEM data using NaN for missing values
    """
    logger.info(f"Starting DEM preprocessing with shape: {dem_data.shape}")
    logger.info(f"Initial value range: {np.nanmin(dem_data):.2f} to {np.nanmax(dem_data):.2f}")
    logger.info(f"Number of NaN values: {np.sum(np.isnan(dem_data))}")
    
    steps = {
        'original': dem_data.copy(),
    }
    
    # Convert nodata values to NaN
    dem_clean = dem_data.copy()
    n_missing = np.sum(dem_clean == fill_value)
    logger.info(f"Number of fill values ({fill_value}): {n_missing}")
    if n_missing > 0:
        dem_clean[dem_clean == fill_value] = np.nan
        logger.info(f"After fill value replacement - NaN count: {np.sum(np.isnan(dem_clean))}")
    
    steps['nodata_replaced'] = dem_clean.copy()
    logger.info(f"After nodata replacement - value range: {np.nanmin(dem_clean):.2f} to {np.nanmax(dem_clean):.2f}")

    logger.info(f"Pre-downsample dem_clean shape: {dem_clean.shape}")
    logger.info(f"Pre-downsample value range: {np.nanmin(dem_clean):.2f} to {np.nanmax(dem_clean):.2f}")
    logger.info(f"Pre-downsample NaN count: {np.sum(np.isnan(dem_clean))}")
    logger.info(f"Pre-downsample total elements: {dem_clean.size}")
    
    # Downsample
    logger.info(f"Downsampling with factor {zoom_factor} and order {downsample_order}")
    dem_downsampled = zoom(dem_clean, zoom=zoom_factor, order=downsample_order, prefilter=False)
    logger.info(f"Downsampled shape: {dem_downsampled.shape}")
    logger.info(f"Downsampled value range: {np.nanmin(dem_downsampled):.2f} to {np.nanmax(dem_downsampled):.2f}")
    logger.info(f"Downsampled NaN count: {np.sum(np.isnan(dem_downsampled))}")
    steps['downsampled'] = dem_downsampled.copy()

    # Calculate slopes
    logger.info("Calculating slopes...")
    slopes = horn_slope(dem_downsampled, window_size=3)
    slopes = np.flipud(slopes)
    logger.info(f"Slopes value range: {np.nanmin(slopes):.2f} to {np.nanmax(slopes):.2f}")
    logger.info(f"Slopes NaN count: {np.sum(np.isnan(slopes))}")
    steps['slopes'] = slopes.copy()

    # smooth
    dem_smoothed = smooth_terrain(dem_downsampled,
                                  window_size=int(np.floor(np.min(dem_downsampled.shape)*.005)))
    steps['smoothed'] = dem_smoothed.copy()
    # dem_smoothed = dem_downsampled
    
    # Normalize to 0-1 (ignoring NaN values)
    logger.info("Normalizing values...")
    dem_normalized = dem_smoothed.copy()
    
    # Check for all-NaN condition before normalization
    if np.all(np.isnan(dem_normalized)):
        logger.error("ERROR: All values are NaN before normalization!")
        raise ValueError("Cannot normalize: all values are NaN")
    
    valid_min = np.nanmin(dem_normalized)
    valid_max = np.nanmax(dem_normalized)
    logger.info(f"Normalization range: {valid_min:.2f} to {valid_max:.2f}")
    
    if valid_min == valid_max:
        logger.error("ERROR: Cannot normalize - min and max values are equal!")
        raise ValueError(f"Cannot normalize: min and max values are equal ({valid_min})")
    
    dem_normalized = (dem_normalized - valid_min) / (valid_max - valid_min)
    dem_normalized = np.flipud(dem_normalized)
    
    logger.info(f"Final normalized value range: {np.nanmin(dem_normalized):.2f} to {np.nanmax(dem_normalized):.2f}")
    logger.info(f"Final NaN count: {np.sum(np.isnan(dem_normalized))}")
    steps['normalized'] = dem_normalized.copy()
    
    return dem_normalized, steps, slopes

def sort_boundary_points(points):
    """
    Sort boundary points to ensure we get a complete boundary loop
    """
    # Convert to set for faster lookup
    points_set = set(points)
    
    # Find starting point (top-left corner)
    start = min(points, key=lambda p: (p[0], p[1]))
    
    ordered = [start]
    current = start
    
    # Possible neighbor offsets (8-connected)
    neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    while len(ordered) < len(points):
        # Find next boundary point
        next_point = None
        min_dist = float('inf')
        
        # Check all remaining points
        for point in points_set - set(ordered):
            dy = point[0] - current[0]
            dx = point[1] - current[1]
            dist = dx*dx + dy*dy
            if dist < min_dist:
                min_dist = dist
                next_point = point
        
        if next_point is None:
            break
            
        ordered.append(next_point)
        current = next_point
    
    return ordered
    
def project_to_convex_boundary(points, scale_pars = None):
    # Create hull from x,y coordinates only
    poly = Polygon(points)
    if scale_pars is not None:
        x,y,z = scale_pars
        poly = scale(poly, xfact=x, yfact=y, zfact=z, origin='center')
    hull = poly.convex_hull
    hull_boundary = hull.exterior
    
    # For each original point:
    new_positions = []
    for x, y in points:
        point = Point(x, y)
        # Project point onto hull boundary
        projected = hull_boundary.interpolate(hull_boundary.project(point))
        new_positions.append((projected.x, projected.y))
        
    return new_positions
    
class DEMStructure:
    def __init__(self, dem_data, values=None, scale_z=1.0):
        self.height, self.width = dem_data.shape
        self.points = {}
        self.boundary_points = []
        self.next_index = 0
        self.index_to_coords = {}

        # Create valid points mask once
        valid_mask = ~np.isnan(dem_data)
        
        # Get all valid coordinates at once using numpy
        valid_coords = np.argwhere(valid_mask)
        logger.info(f"Found {len(valid_coords)} valid points")

        # Create points dictionary using vectorized operations
        y_coords, x_coords = valid_coords.T
        positions = np.column_stack([x_coords, y_coords, dem_data[valid_mask]])
        values_array = values[valid_mask] if values is not None else None

        logger.info(f"Creating all points data...")
        # Create all point data at once
        points_data = {
            (y, x): {
                'dem_coords': (y, x),
                'position': tuple(pos),
                'index': idx,
                'value': values_array[i] if values_array is not None else None
            }
            for i, ((y, x), pos, idx) in enumerate(zip(
                map(tuple, valid_coords), 
                positions, 
                range(len(valid_coords))
            ))
        }
        
        self.points = points_data
        self.next_index = len(valid_coords)

        logger.info(f"Creating index to coords mapping...")
        # Create index to coords mapping
        self.index_to_coords = {
            idx: (y, x) 
            for (y, x), data in points_data.items() 
            for idx in [data['index']]
        }

        # Calculate boundary points using numpy operations
        boundary_mask = np.zeros_like(valid_mask)
        boundary_mask[0, :] = valid_mask[0, :]  # Top edge
        boundary_mask[-1, :] = valid_mask[-1, :]  # Bottom edge
        boundary_mask[:, 0] = valid_mask[:, 0]  # Left edge
        boundary_mask[:, -1] = valid_mask[:, -1]  # Right edge

        
        # Interior boundaries
        valid_neighbors = np.zeros_like(dem_data)
        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
            rolled = np.roll(np.roll(valid_mask, dy, axis=0), dx, axis=1)
            valid_neighbors += rolled
        
        boundary_mask |= (valid_mask & (valid_neighbors < 4))
        
        logger.info(f"Finding interior boundaries...")
        # Get boundary points coordinates
        boundary_coords = np.argwhere(boundary_mask)
        self.boundary_points = [tuple(coord) for coord in boundary_coords]
        
        # Sort boundary points
        self.boundary_points = sort_boundary_points(self.boundary_points)

        logger.info(f"Processed {len(self.points)} points with {len(self.boundary_points)} boundary points")

        # Do convex hull transformation on boundary
        #self.transform_boundary_to_convex_hull()

        logger.info(f"Normalizing z...")
        # Finally normalize all positions
        self.normalize_positions(scale_z)

    def get_dem_coords_from_index(self, index):
        """Get the DEM coordinates (y,x) for a given vertex index"""
        return self.index_to_coords[index]
    def get_boundary_positions(self):
        """Get list of positions for boundary points"""
        return [self.points[p]['position'] for p in self.boundary_points]
    
    def update_boundary_positions(self, new_positions):
        """Update positions for boundary points"""
        for point, new_pos in zip(self.boundary_points, new_positions):
            self.points[point]['position'] = new_pos
    
    def transform_boundary_to_convex_hull(self):
        # Get current positions (x,y only)
        positions = [(x, y) for x,y,z in self.get_boundary_positions()]
        
        # Do convex hull projection
        projected_xy = project_to_convex_boundary(positions)
        
        # Convert back and update with original z values
        new_positions = [(x, y, self.points[bp]['position'][2]) 
                        for (x,y), bp in zip(projected_xy, self.boundary_points)]
        
        self.update_boundary_positions(new_positions)
    
    def normalize_positions(self, scale_z=1.0):
        """
        Normalize all positions to -1 to 1 range
        
        Args:
            scale_z: Either a float for uniform scaling or a function f(x,y,z) -> float 
                    that returns the scaling factor for each point
        """
        mean_x = np.mean([p['position'][0] for p in self.points.values()])
        mean_y = np.mean([p['position'][1] for p in self.points.values()])
        for point in self.points.values():
            x, y, z = point['position']
            norm_x = (x - mean_x) / 100
            norm_y = (y - mean_y) / 100
            
            # Apply z-scaling based on type
            if callable(scale_z):
                # Pass normalized coordinates to scaling function
                z = scale_z(norm_x, norm_y, z)
            else:
                z = scale_z * z
                
            point['position'] = (norm_x, norm_y, z)
            
    def get_index(self, y, x):
        """Get the vertex index for a point in the DEM structure"""
        if (y,x) not in self.points:
            raise KeyError(f"Point ({y},{x}) not found in DEM structure")
        return self.points[(y,x)]['index']
        
    def get_dem_coords(self, y, x):
        """
        Gets the original DEM x,y coordinates for a point.
        These are the unmodified coordinates from the terrain data.
        """
        return self.points[(y,x)]['dem_coords']
    
    def get_normalized_position(self, y, x):
        """
        Gets the current normalized position (x,y,z) for a point.
        This position may have been adjusted during smoothing operations.
        """
        return self.points[(y,x)]['position']
    
    def is_valid_point(self, y, x):
        """Check if a point exists in the structure"""
        return (y,x) in self.points
    
    def get_vertices_list(self):
        """Return vertices in index order for mesh creation"""
        sorted_points = sorted(self.points.items(), key=lambda x: x[1]['index'])
        return [point[1]['position'] for point in sorted_points]
    
    def get_boundary_vertices_and_faces(self, base_depth=-0.2):
        """Creates simple extruded side faces for terrain boundary."""
        next_index = max(point['index'] for point in self.points.values()) + 1
        
        # For each boundary point, create corresponding bottom vertex
        bottom_vertices = {}
        for y, x in self.boundary_points:
            pos = self.get_normalized_position(y, x)
            bottom_vertices[(y,x)] = {
                'position': (pos[0], pos[1], base_depth),  # Same x,y, just lower z
                'index': next_index
            }
            next_index += 1
        
        # Create quad faces by connecting each boundary segment to its bottom vertices
        side_faces = []
        n = len(self.boundary_points)
        for i in range(n):
            p1 = self.boundary_points[i]
            p2 = self.boundary_points[(i + 1) % n]  # Wrap around to first point
            
            # Get indices for top and bottom vertices
            v1 = self.points[p1]['index']
            v2 = self.points[p2]['index']
            b1 = bottom_vertices[p1]['index']
            b2 = bottom_vertices[p2]['index']
            
            # Create quad face
            side_faces.append((v1, v2, b2, b1))
        
        return bottom_vertices, side_faces

def create_terrain_mesh(dem_data, values=None, process_values=None, scale_z=1.0, base_depth=-0.2, colormap_name='twilight_r'):
    # Initialize DEM structure
    logger.info("Initializing DEM structure...")
    dem_structure = DEMStructure(dem_data, values, scale_z)
    height, width = dem_data.shape
    logger.info(f"Created dem structure from dem with {width}x{height}")
    
    # Get vertices and create faces for the top surface
    vertices = dem_structure.get_vertices_list()
    faces = []
    
    # Create top faces
    logger.info("Creating top faces...")
    mesh = bpy.data.meshes.new("TerrainMesh")
    
    with tqdm(total=(height-1)*(width-1), desc="Processing faces") as pbar:
        for y in range(height-1):
            for x in range(width-1):
                valid_points = []
                # Store in winding order
                quad_positions = [(y,x), (y,x+1), (y+1,x+1), (y+1,x)]
                for pos in quad_positions:
                    if dem_structure.is_valid_point(*pos):
                        valid_points.append(dem_structure.points[pos]['index'])
                
                if len(valid_points) >= 3:
                    faces.append(tuple(valid_points))
                pbar.update(1)
    
    # Get bottom vertices and side faces
    logger.info("Creating boundary faces...")
    bottom_vertices, boundary_faces = dem_structure.get_boundary_vertices_and_faces(base_depth)
    
    # Add bottom vertices to main vertex list
    vertices.extend([v['position'] for v in bottom_vertices.values()])
    
    # Add boundary faces to main faces list
    faces.extend(boundary_faces)
    
    # Create and return mesh
    logger.info(f"Creating mesh with {len(vertices)} vertices and {len(faces)} faces")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    if values is not None:
        logger.info("Creating vertex color layer...")
        color_layer = mesh.vertex_colors.new(name="Values")
        
        # Create colormap from matplotlib
        logger.info("Setting up colormap and normalizing values...")
        cmap = plt.colormaps.get_cmap(colormap_name)

        # Keep track of zero slopes before transformation
        zero_mask = np.abs(values) < 1e-6
        
        if process_values is not None:
            values = process_values(values)
                
        norm = Normalize(vmin=np.nanmin(values), vmax=np.nanmax(values))
        
        # Create mapping and calculate colors using vectorized operations
        logger.info("Pre-calculating vertex colors...")
        coords_map = np.array([(point_data['dem_coords'], point_data['index']) 
                              for coords, point_data in dem_structure.points.items()], 
                              dtype=[('coords', 'i4', 2), ('index', 'i4')])
    
        # Extract valid values all at once
        y_coords = coords_map['coords'][:, 0]
        x_coords = coords_map['coords'][:, 1]
        valid_values = values[y_coords, x_coords]
        valid_mask = ~np.isnan(valid_values)
    
        # Calculate all colors at once for valid values
        logger.info("Calculating colors using vectorized operations...")
        colors = np.zeros((len(coords_map), 4), dtype=np.float32)
        colors[valid_mask] = cmap(norm(valid_values[valid_mask]))
        zero_vertices = zero_mask[y_coords, x_coords]
        colors[zero_vertices, 3] = 0  # Set alpha=0 for zero slopes
    
        # Create vertex color dictionary using vectorized operations
        vertex_colors = dict(zip(coords_map['index'][valid_mask], colors[valid_mask]))
        logger.info(f"Created colors for {len(vertex_colors)} valid vertices")
    
        # Create a numpy array of all colors in the correct order
        logger.info("Creating color array...")
        total_loops = len(color_layer.data)
        colors = np.zeros((total_loops, 4), dtype=np.float32)
        
        with tqdm(total=total_loops, desc="Building color array") as pbar:
            for loop_idx in range(total_loops):
                vertex_idx = mesh.loops[loop_idx].vertex_index
                if vertex_idx in vertex_colors:
                    colors[loop_idx] = vertex_colors[vertex_idx]
                pbar.update(1)
    
        # Batch assign all colors at once
        logger.info("Batch assigning colors...")
        try:
            color_layer.data.foreach_set('color', colors.flatten())
            logger.info("Color assignment completed successfully")
        except Exception as e:
            logger.error(f"Error during batch color assignment: {e}")
            # Fall back to regular assignment if batch fails
            logger.info("Falling back to regular color assignment...")
            with tqdm(total=total_loops, desc="Assigning colors (fallback)") as pbar:
                for loop_idx in range(total_loops):
                    vertex_idx = mesh.loops[loop_idx].vertex_index
                    if vertex_idx in vertex_colors:
                        color_layer.data[loop_idx].color = vertex_colors[vertex_idx]
                    pbar.update(1)

    logger.info("Creating new terrain mesh...")
    obj = bpy.data.objects.new("Terrain", mesh)
    logger.info("Linking collection to mesh...")
    bpy.context.scene.collection.objects.link(obj)

    logger.info("Checking color layer...")
    for k in mesh.vertex_colors.keys():
        logger.info(f"    {k}")
    logger.info(f"Color layer stats: {len(color_layer.data)} vertices")
    logger.info(f"Sample of color values: {[[round(x,3) for x in c.color] for c in list(color_layer.data)[:5]]}")
    return obj, dem_structure, color_layer

def setup_camera_and_light(camera_angle, camera_location, scale, sun_angle = 2, sun_energy = 3, focal_length=50):
    # Add camera with adjusted position
    # Add camera with adjusted position
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = focal_length
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    
    # Position camera further back
    cam_obj.location = camera_location
    cam_obj.rotation_euler = camera_angle  # Looking straight down
    
    # For perspective mode, increase the field of view
    # Switch to orthographic mode
    cam_data.type = 'PERSP'
    
    # Adjust the orthographic scale to fit your scene
    cam_data.ortho_scale = scale  # Adjust this value to zoom in/out
    
    bpy.context.scene.camera = cam_obj
    
    # Add two lights for better relief visibility
    sun = bpy.data.lights.new(name="Sun", type='SUN')
    sun_obj = bpy.data.objects.new("Sun", sun)
    bpy.context.scene.collection.objects.link(sun_obj)
    sun_obj.location = (1, 1, 2)
    sun_obj.rotation_euler = (radians(0), radians(315), radians(0))
    sun.angle = sun_angle
    sun.energy = sun_energy  # Increase brightness
    
    
    
    return cam_obj, sun_obj

def setup_world_atmosphere(density=0.02, scatter_color=(1, 1, 1, 1), anisotropy=0.0):
    """
    Set up world volume for atmospheric effects.
    
    Args:
        density: Density of the atmospheric volume (default: 0.02)
        scatter_color: RGBA color for scatter (default: white)
        anisotropy: Direction of scatter (-1 to 1, default: 0 for uniform)
    """
    logger.info("Setting up world atmosphere...")
    
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    
    # Clear existing nodes
    nodes.clear()
    
    # Create nodes
    output = nodes.new('ShaderNodeOutputWorld')
    background = nodes.new('ShaderNodeBackground')
    volume = nodes.new('ShaderNodeVolumePrincipled')
    
    # Set up volume properties
    volume.inputs['Density'].default_value = density
    volume.inputs['Anisotropy'].default_value = anisotropy
    volume.inputs['Color'].default_value = scatter_color
    
    # Connect nodes
    links.new(background.outputs['Background'], output.inputs['Surface'])
    links.new(volume.outputs['Volume'], output.inputs['Volume'])
    
    logger.info(f"World atmosphere set up with density {density}")

def visualize_dem_steps(steps, cmap='viridis'):
    """Create visualizations for each preprocessing step"""
    # Set up a figure with subplots for all steps
    n_steps = len(steps)
    fig, axes = plt.subplots(1, n_steps, figsize=(5*n_steps, 4))
    
    for ax, (name, data) in zip(axes, steps.items()):
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(f"DEM - {name}")
        plt.colorbar(im, ax=ax, label='Elevation')
    
    plt.tight_layout()
    plt.show()

def setup_render_settings(use_gpu=True, samples=128, preview_samples=32, 
                         use_denoising=True, denoiser='OPTIX', 
                         compute_device='OPTIX'):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'

    # Explicit sRGB color management settings
    scene.view_settings.view_transform = 'Standard'  # This is sRGB in Blender 4.3
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    
    # Set display device to sRGB
    scene.display_settings.display_device = 'sRGB'
    
    # Set render output to sRGB
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_depth = '16'  # 16-bit for better color precision
    
    logger.info("Color management settings configured for sRGB:")
    logger.info(f"  View transform: {scene.view_settings.view_transform}")
    logger.info(f"  Display device: {scene.display_settings.display_device}")
    logger.info(f"  Color depth: {scene.render.image_settings.color_depth}")
    
    # Enable transparent film for proper water rendering
    scene.render.film_transparent = False
    
    # Light bounce settings
    scene.cycles.max_bounces = 32
    scene.cycles.transparent_max_bounces = 32
    scene.cycles.transmission_bounces = 32
    scene.cycles.volume_bounces = 2
    scene.cycles.volume_step_rate = 5.0
    scene.cycles.volume_max_steps = 32
    
    # Caustics settings (pick one state)
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False
    
    # Enable multiple importance sampling
    scene.cycles.sample_clamp_indirect = 0.0
    
    # GPU settings
    if use_gpu:
        scene.cycles.device = 'GPU'
        prefs = bpy.context.preferences
        cprefs = prefs.addons['cycles'].preferences
        cprefs.compute_device_type = compute_device
        
        for device in cprefs.devices:
            device.use = True
    
    # Sampling settings
    scene.cycles.samples = samples
    scene.cycles.preview_samples = preview_samples
    scene.cycles.use_denoising = use_denoising
    scene.cycles.denoiser = denoiser
    scene.cycles.use_adaptive_sampling = True

def apply_colormap_material(mat):
    mat.node_tree.nodes.clear()
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Output node
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (800, 300)
    
    # Mix shader for terrain/water blending
    terrain_water_mix = nodes.new('ShaderNodeMixShader')
    terrain_water_mix.location = (600, 300)
    
    # Terrain material
    terrain_shader = nodes.new('ShaderNodeBsdfPrincipled')
    terrain_shader.location = (200, 400)
    # Ceramic-like settings
    # terrain_shader.inputs['Metallic'].default_value = 0.0  # Non-metallic
    # terrain_shader.inputs['Specular IOR Level'].default_value = 0.5  # Low specular
    # terrain_shader.inputs['Roughness'].default_value = 0.7  # Very matte
    # terrain_shader.inputs['IOR'].default_value = 1.5  # Typical ceramic IOR
    # terrain_shader.inputs['Sheen Weight'].default_value = 0.1  # Slight sheen for micro-detail
    # terrain_shader.inputs['Coat Weight'].default_value = 0.15  # Tiny bit of clearcoat
    # terrain_shader.inputs['Coat Roughness'].default_value = 0.7  # Matte clearcoat
    terrain_shader.inputs['Metallic'].default_value = 1.0  # Fully metallic
    terrain_shader.inputs['Roughness'].default_value = 0.95  # Fully rough
    terrain_shader.inputs['Specular IOR Level'].default_value = 0.6  # Mid specular
    terrain_shader.inputs['IOR'].default_value = 3.0  # Higher IOR for more dramatic effect
    
    # Water material
    water_shader = nodes.new('ShaderNodeBsdfPrincipled')
    water_shader.location = (200, 100)
    water_shader.inputs['Base Color'].default_value = (0.1, 0.2, 0.35, 1.0)
    water_shader.inputs['Metallic'].default_value = 0.9
    water_shader.inputs['Roughness'].default_value = 0.1
    water_shader.inputs['IOR'].default_value = 1.33  # Updated for Blender 4.3
    
    # Water surface detail
    noise = nodes.new('ShaderNodeTexNoise')
    noise.location = (0, 100)
    noise.inputs['Scale'].default_value = 50
    noise.inputs['Detail'].default_value = 3.0
    noise.inputs['Roughness'].default_value = 0.5
    noise.inputs['Distortion'].default_value = 0.3
    
    mapping = nodes.new('ShaderNodeMapping')
    mapping.location = (-200, 100)
    
    texcoord = nodes.new('ShaderNodeTexCoord')
    texcoord.location = (-400, 100)
    
    bump = nodes.new('ShaderNodeBump')
    bump.location = (200, 0)
    bump.inputs['Strength'].default_value = 0.3
    bump.inputs['Distance'].default_value = 0.1
    
    # Vertex color input
    vertex_color = nodes.new('ShaderNodeVertexColor')
    vertex_color.layer_name = "Values"
    vertex_color.location = (0, 300)
    
    # Connections
    links.new(vertex_color.outputs['Color'], terrain_shader.inputs['Base Color'])
    
    # Water texture connections
    links.new(texcoord.outputs['Generated'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'], noise.inputs['Vector'])
    links.new(noise.outputs['Fac'], bump.inputs['Height'])
    links.new(bump.outputs['Normal'], water_shader.inputs['Normal'])
    
    # Final mixing
    links.new(vertex_color.outputs['Alpha'], terrain_water_mix.inputs[0])
    links.new(terrain_shader.outputs['BSDF'], terrain_water_mix.inputs[2])
    links.new(water_shader.outputs['BSDF'], terrain_water_mix.inputs[1])
    
    # Output connection
    links.new(terrain_water_mix.outputs[0], output.inputs['Surface'])

def smooth_terrain(dem_data, window_size=10):
    """
    Smooth terrain using median filtering while preserving NaN values.
    
    Args:
        dem_data: Input DEM array
        window_size: Size of the median filter window (odd integer)
    
    Returns:
        Smoothed DEM array with original NaN values preserved
    """
    logger.info(f"Applying median filter with window size {window_size}")
    
    # Save original NaN mask
    nan_mask = np.isnan(dem_data)
    logger.info(f"Original NaN count: {np.sum(nan_mask)}")
    
    # Create a copy and fill NaNs with a sentinel value
    dem_filled = dem_data.copy()
    fill_value = np.nanmin(dem_data) - 1  # Use value less than all valid data
    dem_filled[nan_mask] = fill_value
    
    logger.info(f"Value range before smoothing: {np.nanmin(dem_filled):.2f} to {np.nanmax(dem_filled):.2f}")
    
    # Apply median filter
    with tqdm(total=1, desc="Applying median filter") as pbar:
        smoothed = ndimage.median_filter(dem_filled, size=window_size)
        pbar.update(1)
    
    # Restore NaN values to their original positions
    smoothed[nan_mask] = np.nan
    
    logger.info(f"Value range after smoothing: {np.nanmin(smoothed):.2f} to {np.nanmax(smoothed):.2f}")
    logger.info(f"Final NaN count: {np.sum(np.isnan(smoothed))}")
    
    return smoothed
    
def get_hgt_dimensions(file_path):
    """
    Determine HGT dimensions based on file size
    Returns (width, height)
    """
    file_size = os.path.getsize(file_path)
    
    # Each pixel is 2 bytes (16-bit integers)
    num_pixels = file_size // 2
    
    if num_pixels == 3601 * 3601:
        return (3601, 3601)  # SRTM1
    elif num_pixels == 1201 * 1201:
        return (1201, 1201)  # SRTM3
    else:
        raise ValueError(f"Unexpected file size {file_size} bytes. Cannot determine HGT dimensions.")

def load_hgt_files_from_directory(directory_path):
    """
    Load all .hgt files from a directory
    
    Args:
        directory_path (str): Path to directory containing HGT files
        
    Returns:
        list: Paths to all HGT files found
    """
    # Get all .hgt files in directory
    hgt_pattern = os.path.join(directory_path, "*.hgt")
    hgt_files = glob.glob(hgt_pattern)
    
    if not hgt_files:
        raise ValueError(f"No .hgt files found in {directory_path}")
    
    logger.info(f"Found {len(hgt_files)} HGT files in {directory_path}")
    
    return hgt_files

def get_slope_samples(slopes, n_samples=10):
    """Get n_samples evenly spaced slope values between 5th and 95th percentile"""
    p5 = np.percentile(slopes, 5)
    p95 = np.percentile(slopes, 95)
    return np.linspace(p5, p95, n_samples)

def create_values_legend(terrain_obj, values, mpp=30, colormap_name='mako_r', n_samples=10, 
                        magnify_slopes=10, spacing_factor=0.2, process_values=None, 
                        inv_process_values=None, depth=-0.2):
    """
    Create a slope legend with triangles showing slope angles matching terrain computation.
    The slope is visualized in both y and z dimensions on the front face, while the back
    remains flat at the specified depth.
    
    Args:
        terrain_obj: Blender terrain object
        values: Array of slope values
        mpp: Meters per pixel
        colormap_name: Name of matplotlib colormap to use
        n_samples: Number of slope samples to show in legend
        magnify_slopes: Factor to multiply slopes by for visibility
        spacing_factor: Controls space between legend elements (0=no space, 1=one triangle_base space)
        process_values: Optional function to process values before color mapping
        inv_process_values: Optional function to inverse process_values
        depth: Base depth for the back faces of the legend
    """
    
    # Create a completely new mesh for the legend
    legend_mesh = bpy.data.meshes.new(name="SlopeLegend")
    legend_obj = bpy.data.objects.new("SlopeLegend", legend_mesh)
    
    # Create the vertices and faces lists before adding to mesh
    vertices = []
    faces = []
    face_colors = []
    
    # Calculate legend positioning relative to terrain bounds
    bound_box = terrain_obj.bound_box
    terrain_min = np.min(bound_box, axis=0)
    terrain_max = np.max(bound_box, axis=0)
    terrain_size = terrain_max - terrain_min
    
    triangle_base = terrain_size[0] / n_samples * .95 / (1 + spacing_factor)
    total_width = triangle_base * n_samples * (1 + spacing_factor)
    rectangle_height = triangle_base * 0.2
    
    legend_y_start = terrain_max[1] + terrain_size[1] * 0.035
    x_start_offset = (terrain_min[0] + terrain_max[0])/2 - total_width/2
    
    # Setup colormap and normalization
    cmap = plt.colormaps.get_cmap(colormap_name)
    if process_values is not None:
        values = process_values(values)
    norm = Normalize(vmin=np.nanmin(values), vmax=np.nanmax(values))
    
    values_samples = np.linspace(np.nanmin(values), np.nanmax(values), n_samples)

    vertex_count = 0
    
    logger.info(f"Creating legend with {n_samples} elements")
    logger.info(f"Value range: {np.nanmin(values):.3f} to {np.nanmax(values):.3f}")
    
    for i, value in enumerate(tqdm(values_samples, desc="Creating legend elements")):
        # Get color from colormap for this value
        color = cmap(norm(value))[:3]  # Only take RGB, drop alpha
        
        # Convert processed value back to original scale if needed
        if inv_process_values is not None:
            value = inv_process_values(value)
            
        # Calculate actual slope considering magnification and meters per pixel
        actual_slope = value * magnify_slopes / mpp
        
        # Calculate positions for this legend element
        x_start = x_start_offset + i * triangle_base * (1 + spacing_factor)
        x_end = x_start + triangle_base
        y_offset = actual_slope * triangle_base
        
        # Calculate heights for the sloped face
        z_start = 0
        z_end = actual_slope * triangle_base
        
        logger.debug(f"Element {i}: value={value:.3f}, slope={actual_slope:.3f}")
        logger.debug(f"  Position: x={x_start:.3f} to {x_end:.3f}, y_offset={y_offset:.3f}")
        
        # Define vertices for this element
        element_vertices = [
            # Front rectangle vertices
            (x_start, legend_y_start - rectangle_height, z_end),  # 0: Bottom left front (swapped z)
            (x_end, legend_y_start - rectangle_height, z_start),  # 1: Bottom right front (swapped z)
            (x_end, legend_y_start, z_start),                     # 2: Top right front (swapped z)
            (x_start, legend_y_start, z_end),                     # 3: Top left front (swapped z)
            
            # Back rectangle vertices (flat at depth)
            (x_start, legend_y_start - rectangle_height, depth),  # 4: Bottom left back
            (x_end, legend_y_start - rectangle_height, depth),    # 5: Bottom right back
            (x_end, legend_y_start, depth),                       # 6: Top right back
            (x_start, legend_y_start, depth),                     # 7: Top left back
            
            # Triangle vertices for slope
            (x_start, legend_y_start + y_offset, z_end),         # 8: Front peak (moved to left side)
            (x_start, legend_y_start + y_offset, depth)          # 9: Back peak (moved to left side)
        ]
        
        # Define faces using local indices
        element_faces = [
            # Rectangle faces
            (vertex_count + 0, vertex_count + 1, vertex_count + 2, vertex_count + 3),  # Front
            (vertex_count + 4, vertex_count + 5, vertex_count + 6, vertex_count + 7),  # Back
            (vertex_count + 0, vertex_count + 4, vertex_count + 7, vertex_count + 3),  # Left
            (vertex_count + 1, vertex_count + 5, vertex_count + 6, vertex_count + 2),  # Right
            (vertex_count + 0, vertex_count + 1, vertex_count + 5, vertex_count + 4),  # Bottom
            
            # Triangle faces
            (vertex_count + 2, vertex_count + 3, vertex_count + 8),                    # Front slope
            (vertex_count + 6, vertex_count + 7, vertex_count + 9),                    # Back slope
            (vertex_count + 2, vertex_count + 6, vertex_count + 9, vertex_count + 8),  # Right slope side
            (vertex_count + 3, vertex_count + 7, vertex_count + 9, vertex_count + 8),  # Left slope side
        ]
        
        # Update vertex count for next element
        vertex_count += len(element_vertices)
        
        # Add this element's vertices and faces to the main lists
        vertices.extend(element_vertices)
        faces.extend(element_faces)
        
        # Add colors for all faces of this element
        face_colors.extend([color] * len(element_faces))
        
        logger.debug(f"  Added {len(element_vertices)} vertices and {len(element_faces)} faces")
    
    logger.info(f"Legend creation complete:")
    logger.info(f"Legend creation stats:")
    logger.info(f"  Vertices: {len(vertices)}")
    logger.info(f"  Faces: {len(faces)}")
    logger.info(f"  Colors: {len(face_colors)}")
    logger.info(f"  Final mesh vertices: {len(legend_mesh.vertices)}")
    logger.info(f"  Final mesh faces: {len(legend_mesh.polygons)}")
    
    logger.info("Creating mesh from vertices and faces...")
    # Create the mesh from the accumulated data
    legend_mesh.from_pydata(vertices, [], faces)
    legend_mesh.update()
    
    # Add to scene after mesh is complete
    bpy.context.scene.collection.objects.link(legend_obj)
    
    # Create materials and colors
    color_layer = legend_mesh.vertex_colors.new(name="Values")
    
    # Verify face and color counts match
    assert len(legend_mesh.polygons) == len(face_colors), "Mismatch in face and color counts"
    
    # Apply colors to faces
    for poly, face_color in zip(legend_mesh.polygons, face_colors):
        for loop_idx in poly.loop_indices:
            color_layer.data[loop_idx].color = (*face_color, 1.0)
    
    # Create and apply material
    mat = bpy.data.materials.new(name="LegendMaterial")
    mat.use_nodes = True
    legend_obj.data.materials.append(mat)
    apply_colormap_material(mat)
    
    return legend_obj

def get_pixel_size(dem_files, zoom_factor):
    """Calculate the size of pixels in meters after UTM projection and downsampling"""
    logger.info("Calculating pixel sizes...")
    
    with rasterio.open(dem_files[0]) as src:
        # First get the original data and transform
        elevation = src.read(1)
        transform = src.transform
        
        # Reproject to UTM (you're using zone 17N for Detroit)
        dst_crs = 'EPSG:32617'
        utm_transform, width, height = calculate_default_transform(
            src.crs, dst_crs,
            elevation.shape[1], elevation.shape[0],
            *rasterio.transform.array_bounds(elevation.shape[0], 
                                           elevation.shape[1], 
                                           transform)
        )
        
        # UTM coordinates are already in meters
        # Get the pixel size in meters
        meters_per_pixel = abs(utm_transform[0])  # In UTM, pixels are square
        
        # Account for downsampling
        meters_per_pixel /= zoom_factor
        
        logger.info(f"After projection and {zoom_factor:.3f}x downsampling:")
        logger.info(f"Each pixel represents: {meters_per_pixel:.2f}m")
        
        return meters_per_pixel
        
def create_background_plane(terrain_obj, depth=-2.0, scale_factor=2.0):
    """
    Create a large white plane beneath the terrain.
    
    Args:
        terrain_obj: The terrain Blender object (for size reference)
        depth: Z-coordinate for the plane (default: -2.0)
        scale_factor: How much larger than the terrain the plane should be (default: 2.0)
    
    Returns:
        The created plane object
    """
    logger.info("Creating background plane...")
    
    # Get terrain bounds
    bound_box = terrain_obj.bound_box
    terrain_min = np.min(bound_box, axis=0)
    terrain_max = np.max(bound_box, axis=0)
    terrain_size = terrain_max - terrain_min
    terrain_center = (terrain_max + terrain_min) / 2
    
    # Calculate plane dimensions
    plane_size = max(terrain_size[0], terrain_size[1]) * scale_factor
    
    # Create plane mesh
    plane_mesh = bpy.data.meshes.new("BackgroundPlane")
    plane_obj = bpy.data.objects.new("BackgroundPlane", plane_mesh)
    
    # Create vertices
    half_size = plane_size / 2
    vertices = [
        (terrain_center[0] - half_size, terrain_center[1] - half_size, depth),
        (terrain_center[0] + half_size, terrain_center[1] - half_size, depth),
        (terrain_center[0] + half_size, terrain_center[1] + half_size, depth),
        (terrain_center[0] - half_size, terrain_center[1] + half_size, depth)
    ]
    
    # Create face
    faces = [(0, 1, 2, 3)]
    
    # Create mesh from vertices and faces
    plane_mesh.from_pydata(vertices, [], faces)
    plane_mesh.update()
    
    # Link object to scene
    bpy.context.scene.collection.objects.link(plane_obj)
    
    # Create white material
    # Create white material
    mat = bpy.data.materials.new(name="WhitePlaneMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    # Create Principled BSDF shader for pure white
    output = nodes.new('ShaderNodeOutputMaterial')
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    
    # Set to pure white with emission
    principled.inputs['Base Color'].default_value = (1, 1, 1, 1)
    principled.inputs['Emission Color'].default_value = (1, 1, 1, 1)
    principled.inputs['Emission Strength'].default_value = .35
    principled.inputs['Roughness'].default_value = 0.0
    principled.inputs['Metallic'].default_value = 0.0
    principled.inputs['IOR'].default_value = 1.0
    
    # Position nodes in the node editor
    output.location = (300, 0)
    principled.location = (0, 0)
    
    # Link nodes
    mat.node_tree.links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    
    # Assign material to plane
    plane_obj.data.materials.append(mat)
    
    logger.info(f"Created background plane at depth {depth} with size {plane_size:.2f}")
    
    return plane_obj
    
def create_drive_time_curves(drive_time, terrain_obj, processed_dem, height_offset=1.0, bevel_depth=0.02):
    logger.info("Creating drive time boundary curves...")

    mean_x = processed_dem.shape[1] / 2
    mean_y = processed_dem.shape[0] / 2
    logger.info(f"DEM coordinate means x,y: {mean_x}, {mean_y}")
    
    curves = []
    
    # Create viridis colormap for the number of unique polygons
    n_polygons = len(drive_time)
    cmap = plt.colormaps.get_cmap('inferno')
    
    start = .8
    end = .1
    colors = cmap(np.linspace(start, end, n_polygons))
    
    logger.info(f"Created {n_polygons} colors from turbo colormap")
    
    for idx, zone in enumerate(tqdm(drive_time.geometry, desc="Processing drive time zones")):
        # Get color for this polygon
        color = colors[idx]
        
        # Create material for this specific curve
        # Create material for this specific curve
        mat = bpy.data.materials.new(name=f"DriveTimeMaterial_{idx}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        
        # Create nodes
        principled = nodes.new('ShaderNodeBsdfPrincipled')
        geometry = nodes.new('ShaderNodeNewGeometry')
        vector_math = nodes.new('ShaderNodeVectorMath')
        vector_math.operation = 'CROSS_PRODUCT'  # Cross product with UP vector gives right-hand direction
        
        # Create vector for UP
        combine_xyz = nodes.new('ShaderNodeCombineXYZ')
        combine_xyz.inputs[2].default_value = 1.0  # Z=1 for UP vector
        
        # Set up principled shader
        principled.inputs['Base Color'].default_value = (*color[:3], 1.0)
        principled.inputs['Metallic'].default_value = 0.0
        principled.inputs['Roughness'].default_value = 1.0
        principled.inputs['Emission Color'].default_value = (*color[:3], 1.0)
        principled.inputs['Emission Strength'].default_value = 0
        principled.inputs['Transmission Weight'].default_value = 0
        principled.inputs['IOR'].default_value = 1.0
        
        # Calculate right-hand direction using cross product of tangent and UP
        links.new(geometry.outputs['Tangent'], vector_math.inputs[0])
        links.new(combine_xyz.outputs[0], vector_math.inputs[1])
        
        # Compare with normal using dot product
        dot = nodes.new('ShaderNodeVectorMath')
        dot.operation = 'DOT_PRODUCT'
        links.new(vector_math.outputs[0], dot.inputs[0])
        links.new(geometry.outputs['Normal'], dot.inputs[1])

        map_range = nodes.new('ShaderNodeMapRange')
        map_range.inputs['From Min'].default_value = -1.0
        map_range.inputs['From Max'].default_value = 1.0
        map_range.inputs['To Min'].default_value = 0.0
        map_range.inputs['To Max'].default_value = 1.0
        
        emit = (350.0, 350.0, 350.0, 1.0)
        no_emit = (0.0, 0.0, 0.0, 0.0)
        
        # Use dot product to control emission
        color_ramp = nodes.new('ShaderNodeValToRGB')
        # color_ramp.color_ramp.elements[0].color = emit
        # color_ramp.color_ramp.elements[1].color = emit
        color_ramp.color_ramp.elements.new(0.7)
        color_ramp.color_ramp.elements.new(0.8)
        color_ramp.color_ramp.elements.new(0.9)
        color_ramp.color_ramp.elements.new(1.0)
        color_ramp.color_ramp.elements[0].position = 0
        color_ramp.color_ramp.elements[0].color = emit
        color_ramp.color_ramp.elements[1].position = 0.049
        color_ramp.color_ramp.elements[1].color = emit
        color_ramp.color_ramp.elements[2].position = 0.050
        color_ramp.color_ramp.elements[2].color = no_emit
        color_ramp.color_ramp.elements[3].position = 0.349
        color_ramp.color_ramp.elements[3].color = no_emit
        color_ramp.color_ramp.elements[4].position = 0.350
        color_ramp.color_ramp.elements[4].color = no_emit
        color_ramp.color_ramp.elements[5].position = 1
        color_ramp.color_ramp.elements[5].color = no_emit


        links.new(dot.outputs['Value'], map_range.inputs['Value'])
        links.new(map_range.outputs['Result'], color_ramp.inputs[0])
        links.new(color_ramp.outputs['Color'], principled.inputs['Emission Strength'])
        
        output = nodes.new('ShaderNodeOutputMaterial')
        links.new(principled.outputs['BSDF'], output.inputs['Surface'])

        boundaries = [zone.exterior.coords] if zone.geom_type == 'Polygon' else [poly.exterior.coords for poly in zone.geoms]
        
        for boundary_idx, boundary in enumerate(boundaries):
            # Create the curve data
            curve_data = bpy.data.curves.new(f'DriveTimeCurve_{idx}_{boundary_idx}', 'CURVE')
            curve_data.dimensions = '3D'
            curve_data.resolution_u = 12
            curve_data.bevel_depth = bevel_depth
            
            # Create the curve object
            curve_obj = bpy.data.objects.new(f'DriveTime_{idx}_{boundary_idx}', curve_data)
            
            # Create a new spline in the curve
            spline = curve_data.splines.new('POLY')
            
            # Get coordinates and scale/center them
            coords = [((x-mean_x)/100, (y-mean_y)/100, height_offset) for x,y in list(boundary)]
            
            # Set the number of points
            spline.points.add(len(coords)-1)  # -1 because one point is created by default
            
            # Assign the coordinates
            for point, coord in zip(spline.points, coords):
                point.co = (*coord, 1)  # The fourth component is weight, usually 1
            
            # Assign material
            curve_obj.data.materials.append(mat)
            
            # Link to scene
            bpy.context.scene.collection.objects.link(curve_obj)
            curves.append(curve_obj)
            
            logger.debug(f"Created curve with {len(coords)} points")

    logger.info(f"Created {len(curves)} drive time boundary curves")
    return curves