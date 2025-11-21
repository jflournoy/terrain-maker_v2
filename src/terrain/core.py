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
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(console_handler)

def clear_scene():
    """
    Clear all objects from the Blender scene.

    Resets the scene to factory settings (empty scene) and removes all default
    objects. Useful before importing terrain meshes to ensure a clean workspace.

    Raises:
        RuntimeError: If Blender module (bpy) is not available.
    """
    logger.info("Clearing Blender scene...")
    
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Count objects to be removed
    n_objects = len(bpy.data.objects)
    logger.debug(f"Removing {n_objects} objects")
    
    for obj in bpy.data.objects:
        logger.debug(f"Removing object: {obj.name}")
        bpy.data.objects.remove(obj, do_unlink=True)
        
    logger.info("Scene cleared successfully")

def setup_camera(camera_angle, camera_location, scale, focal_length=50, camera_type='PERSP'):
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
    # Validate camera_type
    if camera_type not in ('PERSP', 'ORTHO'):
        raise ValueError("camera_type must be 'PERSP' or 'ORTHO'")

    logger.debug(f"Creating {camera_type} camera...")
    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)

    cam_obj.location = camera_location
    cam_obj.rotation_euler = camera_angle

    # Set camera type and type-specific parameters
    cam_data.type = camera_type

    if camera_type == 'PERSP':
        cam_data.lens = focal_length
        logger.debug(f"Perspective camera configured at {camera_location} with {focal_length}mm lens")
    else:  # ORTHO
        cam_data.ortho_scale = scale
        logger.debug(f"Orthographic camera configured at {camera_location} with scale {scale}")

    bpy.context.scene.camera = cam_obj

    return cam_obj


def setup_light(location=(1, 1, 2), angle=2, energy=3, rotation_euler=(0, radians(315), 0)):
    """Create and configure sun light for terrain visualization.

    Args:
        location: Tuple of (x,y,z) light position (default: (1, 1, 2))
        angle: Angle of sun light in degrees (default: 2)
        energy: Energy/intensity of sun light (default: 3)
        rotation_euler: Tuple of (x,y,z) rotation angles in radians (default: sun from NW)

    Returns:
        Sun light object
    """
    logger.debug("Creating sun light...")
    sun = bpy.data.lights.new(name="Sun", type='SUN')
    sun_obj = bpy.data.objects.new("Sun", sun)
    bpy.context.scene.collection.objects.link(sun_obj)

    sun_obj.location = location
    sun_obj.rotation_euler = rotation_euler
    sun.angle = angle
    sun.energy = energy

    logger.debug(f"Sun light configured with angle {angle}° and energy {energy}")

    return sun_obj


def setup_camera_and_light(camera_angle, camera_location, scale, sun_angle=2, sun_energy=3, focal_length=50, camera_type='PERSP'):
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
    logger.info("Setting up camera and lighting...")
    camera = setup_camera(camera_angle, camera_location, scale, focal_length, camera_type)
    light = setup_light(angle=sun_angle, energy=sun_energy)
    logger.info("Camera and lighting setup complete")
    return camera, light


def position_camera_relative(
    mesh_obj,
    direction='south',
    distance=1.5,
    elevation=0.5,
    look_at='center',
    camera_type='ORTHO',
    sun_angle=2,
    sun_energy=3,
    focal_length=50
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
        sun_angle: Angle of sun light in degrees. Default: 2
        sun_energy: Intensity of sun light. Default: 3
        focal_length: Camera focal length in mm (perspective cameras only). Default: 50

    Returns:
        Camera object

    Raises:
        ValueError: If direction is not recognized or camera_type is invalid
    """
    # Direction vector mappings (Blender Z-up coordinate system)
    DIRECTIONS = {
        'north': (0, 1, 0),           # +Y
        'south': (0, -1, 0),          # -Y
        'east': (1, 0, 0),            # +X
        'west': (-1, 0, 0),           # -X
        'above': (0, 0, 1),           # +Z (straight up)
        'northeast': (0.7071, 0.7071, 0),
        'northwest': (-0.7071, 0.7071, 0),
        'southeast': (0.7071, -0.7071, 0),
        'southwest': (-0.7071, -0.7071, 0),
    }

    if direction not in DIRECTIONS:
        raise ValueError(f"Invalid direction '{direction}'. Must be one of: {', '.join(DIRECTIONS.keys())}")

    logger.info(f"Positioning camera {direction} of mesh with {camera_type} camera")

    # Get mesh bounds
    bbox = mesh_obj.bound_box
    xs = [v[0] for v in bbox]
    ys = [v[1] for v in bbox]
    zs = [v[2] for v in bbox]

    center = np.array([
        (min(xs) + max(xs)) / 2,
        (min(ys) + max(ys)) / 2,
        (min(zs) + max(zs)) / 2
    ])

    # Calculate mesh diagonal for scaling
    mesh_size = np.array([max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)])
    mesh_diagonal = np.linalg.norm(mesh_size)

    # Get direction vector and calculate camera position
    dir_vector = np.array(DIRECTIONS[direction])

    # For 'above', position directly overhead; for others, position offset from center
    if direction == 'above':
        camera_pos = center.copy()
    else:
        camera_pos = center + dir_vector * distance * mesh_diagonal

    # Add elevation (height above ground)
    camera_pos[2] = center[2] + elevation * mesh_diagonal

    # Determine look-at target
    if look_at == 'center':
        target_pos = center
    else:
        target_pos = np.array(look_at)

    # Calculate rotation to point at target
    # In Blender, camera's local -Z axis is forward, +Y is up
    from mathutils import Vector

    cam_to_target = Vector(target_pos - camera_pos).normalized()

    # Use to_track_quat to calculate rotation
    # Arguments: (forward_axis, up_axis) where -Z is camera forward, Y is world up
    track_quat = cam_to_target.to_track_quat('-Z', 'Y')
    camera_angle = track_quat.to_euler()

    logger.debug(f"Camera position: {camera_pos}, angle: {camera_angle}")

    # Set ortho scale based on mesh size if orthographic
    ortho_scale = mesh_diagonal * 1.2 if camera_type == 'ORTHO' else distance * mesh_diagonal

    # Create camera directly, optionally with light
    camera = setup_camera(
        camera_angle=camera_angle,
        camera_location=tuple(camera_pos),
        scale=ortho_scale,
        focal_length=focal_length,
        camera_type=camera_type
    )

    # Create light if sun parameters are provided
    if sun_angle > 0 or sun_energy > 0:
        setup_light(angle=sun_angle, energy=sun_energy)

    return camera


def setup_world_atmosphere(density=0.02, scatter_color=(1, 1, 1, 1), anisotropy=0.0):
    """Set up world volume for atmospheric effects.
    
    Args:
        density: Density of the atmospheric volume (default: 0.02)
        scatter_color: RGBA color tuple for scatter (default: white)
        anisotropy: Direction of scatter from -1 to 1 (default: 0 for uniform)
    
    Returns:
        bpy.types.World: The configured world object
    """
    logger.info("Setting up world atmosphere...")
    
    try:
        world = bpy.context.scene.world
        if world is None:
            world = bpy.data.worlds.new("World")
            bpy.context.scene.world = world
        
        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links
        
        logger.debug("Clearing existing world nodes...")
        nodes.clear()
        
        # Create node network
        logger.debug("Creating atmosphere node network...")
        output = nodes.new('ShaderNodeOutputWorld')
        background = nodes.new('ShaderNodeBackground')
        volume = nodes.new('ShaderNodeVolumePrincipled')
        
        # Configure volume properties
        volume.inputs['Density'].default_value = density
        volume.inputs['Anisotropy'].default_value = anisotropy
        volume.inputs['Color'].default_value = scatter_color
        
        # Connect nodes
        links.new(background.outputs['Background'], output.inputs['Surface'])
        links.new(volume.outputs['Volume'], output.inputs['Volume'])
        
        logger.info(f"World atmosphere configured with density {density}")
        return world
        
    except Exception as e:
        logger.error(f"Failed to setup world atmosphere: {str(e)}")
        raise

def setup_render_settings(
    use_gpu: bool = True, 
    samples: int = 128, 
    preview_samples: int = 32,
    use_denoising: bool = True, 
    denoiser: str = 'OPTIX',
    compute_device: str = 'OPTIX'
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
    """
    logger.info("Configuring render settings...")
    
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'

    # Configure color management for sRGB
    logger.info("Setting up color management...")
    scene.view_settings.view_transform = 'Standard'  # sRGB in Blender 4.3
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    scene.display_settings.display_device = 'sRGB'
    
    # Configure render output settings
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_depth = '16'
    scene.render.film_transparent = False
    
    logger.info("Configuring Cycles settings...")
    # Configure Cycles render settings
    cycles = scene.cycles
    cycles.max_bounces = 32
    cycles.transparent_max_bounces = 32
    cycles.transmission_bounces = 32
    cycles.volume_bounces = 2
    cycles.volume_step_rate = 5.0
    cycles.volume_max_steps = 32
    cycles.caustics_reflective = False
    cycles.caustics_refractive = False
    cycles.sample_clamp_indirect = 0.0
    
    # Configure sampling settings
    cycles.samples = samples
    cycles.preview_samples = preview_samples
    cycles.use_denoising = use_denoising
    cycles.denoiser = denoiser
    cycles.use_adaptive_sampling = True
    
    # Configure GPU settings if requested
    if use_gpu:
        logger.info(f"Setting up GPU rendering with {compute_device}...")
        try:
            cycles.device = 'GPU'
            prefs = bpy.context.preferences
            cprefs = prefs.addons['cycles'].preferences
            cprefs.compute_device_type = compute_device
            
            # Enable all available devices
            device_count = 0
            for device in cprefs.devices:
                device.use = True
                device_count += 1
            logger.info(f"Enabled {device_count} compute devices")
            
        except Exception as e:
            logger.error(f"Failed to configure GPU rendering: {str(e)}")
            logger.warning("Falling back to CPU rendering")
            cycles.device = 'CPU'
    
    logger.info("Render settings configured successfully")


def render_scene_to_file(
    output_path,
    width=1920,
    height=1440,
    file_format='PNG',
    color_mode='RGBA',
    compression=90,
    save_blend_file=True
):
    """
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
    """
    try:
        import bpy
    except ImportError:
        logger.warning("Blender/bpy not available - skipping render")
        return None

    output_path = Path(output_path).resolve()
    logger.info(f"Rendering scene to {output_path}")

    try:
        # Configure render output
        bpy.context.scene.render.filepath = str(output_path)
        bpy.context.scene.render.image_settings.file_format = file_format
        bpy.context.scene.render.image_settings.color_mode = color_mode

        if file_format == 'PNG':
            bpy.context.scene.render.image_settings.compression = compression

        bpy.context.scene.render.resolution_x = width
        bpy.context.scene.render.resolution_y = height
        bpy.context.scene.render.resolution_percentage = 100

        logger.info(f"Render: {width}×{height} {file_format} ({color_mode})")

        # Execute render
        bpy.ops.render.render(write_still=True)

        # Verify output
        if output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"Rendered successfully: {file_size_mb:.1f} MB")

            # Save Blender file if requested
            if save_blend_file:
                blend_path = output_path.parent / output_path.stem
                blend_path = blend_path.with_suffix('.blend')
                try:
                    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
                    logger.info(f"Saved Blender file: {blend_path.name}")
                except Exception as e:
                    logger.warning(f"Could not save Blender file: {e}")

            return output_path
        else:
            logger.error("Render file was not created")
            return None

    except Exception as e:
        logger.error(f"Render failed: {str(e)}")
        return None


def apply_colormap_material(material: bpy.types.Material) -> None:
    """
    Create a simple material setup for terrain visualization using vertex colors.
    Uses emission to guarantee colors are visible regardless of lighting.

    Args:
        material: Blender material to configure
    """
    logger.info(f"Setting up material nodes for {material.name}")

    # Clear existing nodes
    material.node_tree.nodes.clear()
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    try:
        # Create simple shader nodes with vertex color + emission
        output = nodes.new('ShaderNodeOutputMaterial')
        principled = nodes.new('ShaderNodeBsdfPrincipled')
        emission = nodes.new('ShaderNodeEmission')
        mix_shader = nodes.new('ShaderNodeMixShader')
        vertex_color = nodes.new('ShaderNodeVertexColor')

        # Position nodes
        output.location = (600, 300)
        mix_shader.location = (400, 300)
        principled.location = (200, 400)
        emission.location = (200, 200)
        vertex_color.location = (0, 300)

        # Set vertex color layer
        vertex_color.layer_name = "TerrainColors"

        # Configure principled shader for reflectance
        principled.inputs['Base Color'].default_value = (0.5, 0.5, 0.5, 1.0)
        principled.inputs['Roughness'].default_value = 0.8

        # Configure emission to use vertex colors
        # Use vertex color directly as emission color with moderate strength
        emission.inputs['Strength'].default_value = 1.5

        # Create connections
        logger.debug("Creating node connections")
        # Vertex color drives both emission and base color
        links.new(vertex_color.outputs['Color'], emission.inputs['Color'])
        links.new(vertex_color.outputs['Color'], principled.inputs['Base Color'])

        # Mix between principled shader (reflected light) and emission (self-illuminated)
        # Use mostly emission with some reflected light for better color visibility
        links.new(principled.outputs['BSDF'], mix_shader.inputs[1])
        links.new(emission.outputs['Emission'], mix_shader.inputs[2])

        # Set mix factor to favor emission (70% emission, 30% principled)
        # This ensures vertex colors are visible and properly colored
        mix_shader.inputs[0].default_value = 0.3

        # Connect to output
        links.new(mix_shader.outputs['Shader'], output.inputs['Surface'])

        logger.info("Material setup completed successfully")

    except Exception as e:
        logger.error(f"Error setting up material: {str(e)}")
        raise

def create_background_plane(
    terrain_obj: bpy.types.Object, 
    depth: float = -2.0, 
    scale_factor: float = 2.0,
    material_params: Optional[Dict] = None
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
    logger.info("Creating background plane...")
    
    if terrain_obj is None:
        raise ValueError("Terrain object cannot be None")
    
    default_material_params = {
        'base_color': (1, 1, 1, 1),
        'emission_color': (1, 1, 1, 1),
        'emission_strength': 0.35,
        'roughness': 0.0,
        'metallic': 0.0,
        'ior': 1.0
    }
    
    try:
        # Calculate terrain bounds
        bound_box = np.array(terrain_obj.bound_box)
        terrain_min = np.min(bound_box, axis=0)
        terrain_max = np.max(bound_box, axis=0)
        terrain_size = terrain_max - terrain_min
        terrain_center = (terrain_max + terrain_min) / 2
        
        logger.debug(f"Terrain bounds - min: {terrain_min}, max: {terrain_max}")
        logger.debug(f"Terrain size: {terrain_size}")
        
        # Calculate plane dimensions
        plane_size = max(terrain_size[0], terrain_size[1]) * scale_factor
        half_size = plane_size / 2
        
        logger.debug(f"Plane size: {plane_size}")
        
        # Create plane geometry
        try:
            plane_mesh = bpy.data.meshes.new("BackgroundPlane")
            plane_obj = bpy.data.objects.new("BackgroundPlane", plane_mesh)
            
            # Define vertices
            vertices = [
                (terrain_center[0] - half_size, terrain_center[1] - half_size, depth),
                (terrain_center[0] + half_size, terrain_center[1] - half_size, depth),
                (terrain_center[0] + half_size, terrain_center[1] + half_size, depth),
                (terrain_center[0] - half_size, terrain_center[1] + half_size, depth)
            ]
            faces = [(0, 1, 2, 3)]
            
            # Create mesh
            plane_mesh.from_pydata(vertices, [], faces)
            plane_mesh.update()
            
            # Link to scene
            bpy.context.scene.collection.objects.link(plane_obj)
            
        except Exception as e:
            logger.error(f"Failed to create plane mesh: {str(e)}")
            raise RuntimeError("Mesh creation failed") from e
            
        # Create material
        try:
            # Merge default params with any provided overrides
            params = default_material_params.copy()
            if material_params:
                params.update(material_params)
            
            mat = bpy.data.materials.new(name="BackgroundPlaneMaterial")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            nodes.clear()
            
            # Create shader nodes
            output = nodes.new('ShaderNodeOutputMaterial')
            principled = nodes.new('ShaderNodeBsdfPrincipled')
            
            # Configure material properties using merged params
            principled.inputs['Base Color'].default_value = params['base_color']
            principled.inputs['Emission Color'].default_value = params['emission_color']
            principled.inputs['Emission Strength'].default_value = params['emission_strength']
            principled.inputs['Roughness'].default_value = params['roughness']
            principled.inputs['Metallic'].default_value = params['metallic']
            principled.inputs['IOR'].default_value = params['ior']
            
            # Position nodes
            output.location = (300, 0)
            principled.location = (0, 0)
            
            # Connect nodes
            mat.node_tree.links.new(principled.outputs['BSDF'], output.inputs['Surface'])
            
            # Assign material
            plane_obj.data.materials.append(mat)
            
        except Exception as e:
            logger.error(f"Failed to create plane material: {str(e)}")
            raise RuntimeError("Material creation failed") from e
            
        logger.info(f"Successfully created background plane:")
        logger.info(f"  Size: {plane_size:.2f}")
        logger.info(f"  Depth: {depth}")
        logger.info(f"  Center: ({terrain_center[0]:.2f}, {terrain_center[1]:.2f})")
        
        return plane_obj
        
    except Exception as e:
        logger.error(f"Background plane creation failed: {str(e)}")
        raise

def load_dem_files(directory_path: str, pattern: str = "*.hgt", recursive: bool = False) -> tuple[np.ndarray, rasterio.Affine]:
    """
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
    """
    logger.info(f"Searching for DEM files matching '{pattern}' in: {directory_path}")
    
    try:
        directory = Path(directory_path)
        
        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        
        if not directory.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")
            
        # Find all matching files
        glob_func = directory.rglob if recursive else directory.glob
        dem_files = sorted(glob_func(pattern))
        
        if not dem_files:
            raise ValueError(f"No files matching '{pattern}' found in {directory}")
        
        # Validate and open files
        dem_datasets = []
        with tqdm(dem_files, desc="Opening DEM files") as pbar:
            for file in pbar:
                try:
                    ds = rasterio.open(file)
                    
                    # Basic validation
                    if ds.count == 0:
                        logger.warning(f"No raster bands found in {file}")
                        ds.close()
                        continue
                        
                    if ds.dtypes[0] not in ('int16', 'int32', 'float32', 'float64'):
                        logger.warning(f"Unexpected data type in {file}: {ds.dtypes[0]}")
                        ds.close()
                        continue
                    
                    dem_datasets.append(ds)
                    pbar.set_postfix({"opened": len(dem_datasets)})
                    
                except rasterio.errors.RasterioIOError as e:
                    logger.warning(f"Failed to open {file}: {str(e)}")
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error with {file}: {str(e)}")
                    continue
        
        if not dem_datasets:
            raise ValueError("No valid DEM files could be opened")
            
        logger.info(f"Successfully opened {len(dem_datasets)} DEM files")
        
        # Merge datasets
        try:
            with rasterio.Env():
                merged_dem, transform = merge(dem_datasets)
                
                # Extract first band - merge() returns 3D array (bands, height, width)
                merged_dem = merged_dem[0]
                
                logger.info(f"Successfully merged DEMs:")
                logger.info(f"  Output shape: {merged_dem.shape}")
                logger.info(f"  Value range: {np.nanmin(merged_dem):.2f} to {np.nanmax(merged_dem):.2f}")
                logger.info(f"  Transform: {transform}")
                
                return merged_dem, transform
                
        finally:
            # Clean up
            for ds in dem_datasets:
                ds.close()
        
    except Exception as e:
        logger.error(f"Error processing DEM files: {str(e)}")
        raise

def reproject_raster(src_crs='EPSG:4326', dst_crs='EPSG:32617', nodata_value=np.nan, num_threads=4):
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
    def _reproject_raster(src_data, src_transform):
        logger = logging.getLogger(__name__)
        logger.info(f"Reprojecting raster from {src_crs} to {dst_crs}")
        
        with rasterio.Env(GDAL_NUM_THREADS=str(num_threads)):
            # Calculate transform and dimensions for destination CRS
            dst_transform, width, height = calculate_default_transform(
                src_crs, dst_crs, 
                src_data.shape[1], src_data.shape[0],
                *rasterio.transform.array_bounds(
                    src_data.shape[0], 
                    src_data.shape[1], 
                    src_transform
                )
            )
            
            # Create destination array
            dst_data = np.full((height, width), nodata_value, dtype=src_data.dtype)

            # Reproject with bilinear interpolation
            reproject(
                source=src_data,
                destination=dst_data,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                dst_nodata=nodata_value,
                num_threads=num_threads, 
                warp_mem_limit=512  
            )
            
            logger.info(f"Reprojection complete. New shape: {dst_data.shape}")
            logger.info(f"Value range: {np.nanmin(dst_data):.2f} to {np.nanmax(dst_data):.2f}")
            
            # Return all three values
            return dst_data, dst_transform, dst_crs
    
    return _reproject_raster

def downsample_raster(zoom_factor=0.1, order=4, nodata_value=np.nan):
    """
    Create a raster downsampling transform function with specified parameters.
    
    Args:
        zoom_factor: Scaling factor for downsampling (default: 0.1)
        order: Interpolation order (default: 4)
        nodata_value: Value to treat as no data (default: np.nan)
    
    Returns:
        function: A transform function that downsamples raster data
    """
    logger = logging.getLogger(__name__)
    
    def transform(raster_data, transform=None):
        """
        Downsample raster data using scipy.ndimage.zoom
        
        Args:
            raster_data: Input raster numpy array
            transform: Optional affine transform
            
        Returns:
            tuple: (downsampled_data, new_transform)
        """
        logger.info(f"Downsampling raster by factor {zoom_factor}")
        
        # Mask out nodata values before downsampling
        mask = raster_data == nodata_value
        
        # Prepare data for downsampling
        processed_data = raster_data.copy()
        processed_data[mask] = np.nan
        
        # Downsample
        downsampled = zoom(processed_data, zoom=zoom_factor, order=order, 
                         prefilter=False, mode='reflect')
        
        # Restore nodata values
        if np.any(mask):
            mask_downsampled = zoom(mask.astype(float), zoom=zoom_factor, 
                                  order=0, prefilter=False)
            downsampled[mask_downsampled > 0.5] = nodata_value
        
        # Update transform if provided
        if transform is not None:
            # Scale the transform to match new resolution
            new_transform = transform * transform.scale(1/zoom_factor)
        else:
            new_transform = None
            
        logger.info(f"Original shape: {raster_data.shape}")
        logger.info(f"Downsampled shape: {downsampled.shape}")
        logger.info(f"Value range: {np.nanmin(downsampled):.2f} to {np.nanmax(downsampled):.2f}")

        # Return 3-tuple for consistency with transform pipeline
        return downsampled, new_transform, None
    
    return transform

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
    logger = logging.getLogger(__name__)
    
    def transform(raster_data, transform=None):
        """
        Apply median smoothing to raster data
        
        Args:
            raster_data: Input raster numpy array
            transform: Optional affine transform (unchanged by smoothing)
            
        Returns:
            tuple: (smoothed_data, transform)
        """
        logger.info("Applying median smoothing to raster")
        
        # Create a mask for nodata values
        mask = raster_data == nodata_value
        
        # Prepare data for smoothing
        processed_data = raster_data.copy()
        processed_data[mask] = np.nan
        
        # Calculate window size if not provided
        actual_window_size = window_size
        if actual_window_size is None:
            actual_window_size = int(np.floor(np.min(processed_data.shape) * 0.05))
            logger.info(f"Using calculated window size: {actual_window_size}")
        
        # Apply median filter
        smoothed = ndimage.median_filter(processed_data, size=actual_window_size)
        
        # Restore nodata values
        smoothed[mask] = nodata_value
        
        logger.info(f"Smoothing window size: {actual_window_size}")
        logger.info(f"Value range before smoothing: {np.nanmin(processed_data):.2f} to {np.nanmax(processed_data):.2f}")
        logger.info(f"Value range after smoothing: {np.nanmin(smoothed):.2f} to {np.nanmax(smoothed):.2f}")

        # Transform is unchanged by smoothing, return 3-tuple for consistency
        return smoothed, transform, None
    
    return transform

def flip_raster(axis='horizontal'):
    """
    Create a transform function that mirrors (flips) the DEM data.
    If axis='horizontal', it flips top ↔ bottom.
    (In terms of rows, row=0 becomes row=(height-1).)

    If axis='vertical', you could do left ↔ right (np.fliplr).
    """

    def transform_func(data, transform=None):
        """Flip array along specified axis and update transform if provided."""
        # 1) Flip the array in pixel space
        if axis == 'horizontal':
            # Top <-> bottom
            new_data = np.flipud(data)
            flip_code = 'horizontal'
        elif axis == 'vertical':
            # Left <-> right
            new_data = np.fliplr(data)
            flip_code = 'vertical'
        else:
            raise ValueError("axis must be 'horizontal' or 'vertical'.")

        if transform is None:
            # No transform to update, just return
            return (new_data, None, None)

        # 2) Original array shape
        old_height, old_width = data.shape

        # 3) If we keep the same shape after flip
        new_height, new_width = new_data.shape
        assert new_height == old_height and new_width == old_width, \
            "Flip changed array size unexpectedly!"

        # 4) Update the affine transform

        # Typical georeferenced transform looks like:
        #   Affine(a, b, xoff, 
        #          d, e, yoff)
        # For a north-up raster with no rotation:
        #   a = pixel_width, e = -pixel_height, b=d=0
        #   xoff, yoff = top-left corner in world coords
        # 
        # Flipping top ↔ bottom effectively inverts rows:
        #   new_row = (height-1) - old_row
        #
        # That means:
        #   new_yoff = old_yoff + (height-1)*e
        #   new_e = -old_e    (so that row moves in the opposite direction)

        a, b, xoff, d, e, yoff = transform.to_gdal()

        if flip_code == 'horizontal':
            # top <-> bottom flip => invert "row" direction
            new_e = -e
            new_yoff = yoff + (old_height - 1) * e
            # Everything else remains the same
            new_transform = Affine(a, b, xoff, d, new_e, new_yoff)

        elif flip_code == 'vertical':
            # left <-> right flip => invert "col" direction
            new_a = -a
            new_xoff = xoff + (old_width - 1) * a
            # Others remain the same
            new_transform = Affine(new_a, b, new_xoff, d, e, yoff)

        return (new_data, new_transform, None)

    return transform_func


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
    logger = logging.getLogger(__name__)

    def transform(raster_data, transform=None):
        """
        Scale elevation values in raster data.

        Args:
            raster_data: Input raster numpy array
            transform: Optional affine transform (unchanged by scaling)

        Returns:
            tuple: (scaled_data, transform, None)
        """
        logger.info(f"Scaling elevation by factor {scale_factor}")

        # Create output array
        scaled_data = raster_data.copy()

        # Mask out nodata values
        mask = raster_data == nodata_value

        # Scale the valid data
        scaled_data[~mask] = raster_data[~mask] * scale_factor

        # Preserve nodata values
        scaled_data[mask] = nodata_value

        logger.info(f"Original range: {np.nanmin(raster_data):.2f} to {np.nanmax(raster_data):.2f}")
        logger.info(f"Scaled range: {np.nanmin(scaled_data):.2f} to {np.nanmax(scaled_data):.2f}")

        # Transform is unchanged by scaling (affects Z only)
        return scaled_data, transform, None

    return transform


def slope_colormap(slopes, cmap_name='terrain', min_slope=0, max_slope=45):
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
    logger = logging.getLogger(__name__)
    logger.info(f"Creating slope colormap using {cmap_name}")
    
    # Get the colormap from matplotlib
    cmap = plt.cm.get_cmap(cmap_name)
    
    # Get dimensions
    height, width = slopes.shape
    
    # Create output RGBA array (4 channels instead of 3)
    colors = np.zeros((height, width, 4))
    
    # Mask for valid (non-NaN) slope values
    valid_mask = ~np.isnan(slopes)
    
    # Set boundaries for color normalization
    actual_min = min_slope if min_slope is not None else np.nanmin(slopes)
    actual_max = max_slope if max_slope is not None else np.nanmax(slopes)
    
    logger.info(f"Normalizing slopes from {actual_min:.2f} to {actual_max:.2f}")
    
    # Normalize slope values between 0-1 for the colormap
    normalized_slopes = np.zeros_like(slopes)
    normalized_slopes[valid_mask] = np.clip(
        (slopes[valid_mask] - actual_min) / (actual_max - actual_min),
        0, 1
    )
    
    # Apply colormap to get RGBA values
    valid_indices = np.where(valid_mask)
    total_valid = np.sum(valid_mask)
    
    with tqdm(total=total_valid, desc="Generating slope colors") as pbar:
        # Process in chunks to manage memory
        chunk_size = min(10000, total_valid)
        num_chunks = (total_valid + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_valid)
            
            # Get indices for this chunk
            chunk_indices = (
                valid_indices[0][start_idx:end_idx],
                valid_indices[1][start_idx:end_idx]
            )
            
            # Get slope values for this chunk
            chunk_slopes = normalized_slopes[chunk_indices]
            
            # Apply colormap to get RGBA values (keep alpha channel)
            chunk_colors = cmap(chunk_slopes)  # This returns RGBA by default
            
            # Assign colors to output array
            for i in range(len(chunk_slopes)):
                y, x = chunk_indices[0][i], chunk_indices[1][i]
                colors[y, x] = chunk_colors[i]
            
            pbar.update(end_idx - start_idx)
    
    # Set invalid (NaN) areas to transparent black
    colors[~valid_mask] = (0, 0, 0, 0)
    
    logger.info(f"Created slope colormap with shape {colors.shape}")
    return colors


def elevation_colormap(dem_data, cmap_name='viridis', min_elev=None, max_elev=None):
    """
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
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating elevation colormap using {cmap_name}")

    # Handle NaN values
    valid_mask = ~np.isnan(dem_data)

    # Auto-calculate min/max if not provided
    if min_elev is None:
        min_elev = np.nanmin(dem_data)
    if max_elev is None:
        max_elev = np.nanmax(dem_data)

    logger.info(f"Elevation range: {min_elev:.1f} to {max_elev:.1f} meters")

    # Normalize elevation to 0-1 range
    if min_elev == max_elev:
        normalized = np.zeros_like(dem_data, dtype=np.float32)
    else:
        normalized = np.zeros_like(dem_data, dtype=np.float32)
        normalized[valid_mask] = (dem_data[valid_mask] - min_elev) / (max_elev - min_elev + 1e-8)

    # Get colormap from matplotlib
    try:
        # Try new API (matplotlib >= 3.7)
        import matplotlib
        cmap = matplotlib.colormaps.get_cmap(cmap_name)
    except (AttributeError, TypeError):
        # Fall back to old API for older matplotlib versions
        cmap = plt.cm.get_cmap(cmap_name)

    # Apply colormap (returns RGBA with shape (H, W, 4))
    rgba_array = cmap(normalized)

    # Extract RGB channels only (drop alpha)
    rgb = rgba_array[:, :, :3]

    # Set invalid (NaN) areas to dark gray
    rgb[~valid_mask] = [0.2, 0.2, 0.2]

    # Convert to uint8
    colors_uint8 = (rgb * 255).astype(np.uint8)

    logger.info(f"Created elevation colormap with shape {colors_uint8.shape}")
    return colors_uint8


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
            if len(params) == 1 and params[0] == 'data':
                transformed_data = transform_func(data)
                
            # Case 2: Transform takes (data, transform)
            elif len(params) == 2 and params[0] == 'data' and params[1] == 'transform':
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
    def __init__(self, cache_dir: str = 'terrain_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def get_target_path(self, target_name: str) -> Path:
        """Get path for a specific target"""
        return self.cache_dir / f"{target_name}.tif"

    def exists(self, target_name: str) -> bool:
        """Check if target exists"""
        return self.get_target_path(target_name).exists()

    def save(self, target_name: str, data: np.ndarray, transform, crs='EPSG:4326', metadata=None):
        """Save data as GeoTIFF with CRS and metadata"""
        path = self.get_target_path(target_name)
        self.logger.info(f"Saving {target_name}")
        
        # Save main raster data
        with rasterio.open(
            path,
            'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs=crs,
            transform=transform,
            compress='lzw'
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
            meta_path = path.with_suffix('.json')
            with open(meta_path, 'w') as f:
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
        meta_path = path.with_suffix('.json')
        full_metadata = basic_metadata.copy()
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    full_metadata.update(json.load(f))
            except (json.JSONDecodeError, OSError) as e:
                self.logger.warning(f"Failed to load metadata file {meta_path}: {e}")
        
        return {
            'data': data,
            'transform': transform,
            'crs': crs,
            'metadata': full_metadata
        }

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
    def __init__(self, dem_data: np.ndarray,
                 dem_transform: rasterio.Affine,
                 dem_crs: str = 'EPSG:4326',
                 cache_dir: str = 'terrain_cache',
                 logger: Optional[logging.Logger] = None) -> None:
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
                dem_data.shape[0], dem_data.shape[1], dem_transform)
        
        # Calculate resolution in meters
        self.resolution = (
            abs(dem_transform[0]) * 111320,
            abs(dem_transform[4]) * 111320
        )
        self.dem_shape = dem_data.shape
        
        # Initialize list of transforms and data layers
        self.transforms = []
        self.data_layers = {}

        # Initialize containers for processed data
        self.processed_dem = None  # Will hold transformed DEM data
        self.vertices = None       # Will hold final vertex positions
        self.faces = None         # Will hold face indices
        self.vertex_colors = None # Will hold vertex colors

        self.add_data_layer('dem', dem_data, dem_transform, dem_crs)
        
        self.logger.info(f"Terrain initialized with DEM data:")
        self.logger.info(f"  Shape: {dem_data.shape}")
        self.logger.info(f"  Resolution: {self.resolution[0]:.2f}m x {self.resolution[1]:.2f}m")
        self.logger.info(f"  Value range: {np.nanmin(dem_data):.2f} to {np.nanmax(dem_data):.2f}")
    
    def visualize_dem(self, 
                      layer: str = 'dem',
                      use_transformed: bool = False,
                      title: str = None,
                      cmap: str = 'terrain', 
                      percentile_clip: bool = True,
                      clip_percentiles: tuple = (1, 99),
                      max_pixels: int = 500_000,
                      show_histogram: bool = True) -> None:
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
        if use_transformed and not layer_info.get('transformed', False):
            self.logger.warning(f"Transformed data requested for layer '{layer}' but not available. Using original.")
            use_transformed = False
        
        if use_transformed:
            plot_data = layer_info['transformed_data']
            data_transform = layer_info['transformed_transform']
            data_crs = layer_info['transformed_crs']
            self.logger.info(f"Using transformed data for layer '{layer}'")
        else:
            plot_data = layer_info['data']
            data_transform = layer_info['transform']
            data_crs = layer_info['crs']
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
            self.logger.info(f"Clipping to {min_percentile}-{max_percentile} percentiles: [{vmin:.4f}, {vmax:.4f}]")
        else:
            vmin, vmax = valid_data.min(), valid_data.max()
        
        # Determine plot layout
        if show_histogram:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle(title, fontsize=16)
            
            # Main data heatmap
            im = ax1.imshow(sampled_data, 
                           cmap=cmap, 
                           vmin=vmin, 
                           vmax=vmax, 
                           interpolation='nearest')
            ax1.set_title(f"{layer.capitalize()} Visualization")
            ax1.set_xlabel("Column Index")
            ax1.set_ylabel("Row Index")
            plt.colorbar(im, ax=ax1, shrink=0.8, label=layer.capitalize())
            
            # Value distribution histogram
            ax2.hist(valid_data, bins=50, color='skyblue', alpha=0.7)
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
            
            im = ax.imshow(sampled_data, 
                          cmap=cmap, 
                          vmin=vmin, 
                          vmax=vmax, 
                          interpolation='nearest')
            ax.set_title(f"{layer.capitalize()} Visualization")
            ax.set_xlabel("Column Index")
            ax.set_ylabel("Row Index")
            plt.colorbar(im, ax=ax, shrink=0.8, label=layer.capitalize())
        
        # Add data source and transform info in footer
        transform_str = f"Transform: [{data_transform[0]:.6f}, {data_transform[1]:.6f}, {data_transform[2]:.6f}, {data_transform[3]:.6f}, {data_transform[4]:.6f}, {data_transform[5]:.6f}]"
        plt.figtext(0.5, 0.01, f"CRS: {data_crs} | {transform_str}", ha='center', fontsize=8)
        
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
       
    def add_data_layer(self, name: str, data: np.ndarray, transform: rasterio.Affine,
                       crs: str, target_crs: Optional[str] = None,
                       target_layer: Optional[str] = None,
                       resampling: Resampling = Resampling.bilinear) -> None:
        """
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
        """
        self.logger.info(f"Adding data layer '{name}'")
        
        # Determine target CRS and transform
        if target_layer is not None:
            if target_layer not in self.data_layers:
                raise KeyError(f"Target layer '{target_layer}' not found")
                
            target_info = self.data_layers[target_layer]
            target_crs = target_info['crs']
            target_transform = target_info['transform']
            target_shape = target_info['data'].shape
            
        elif target_crs is not None:
            # If target_crs provided but no reference layer, we need a reference layer
            if not self.data_layers:
                raise ValueError("Cannot determine target grid without reference layer")
                
            # Use first layer as reference for grid
            reference_layer = next(iter(self.data_layers.values()))
            target_transform = reference_layer['transform']
            target_shape = reference_layer['data'].shape
            
        else:
            # If no target specified, keep original
            self.data_layers[name] = {
                'data': data,
                'transform': transform,
                'crs': crs,
                'transformed': False
            }
            self.logger.info(f"Added layer '{name}' with original CRS {crs}")
            return
        
        # Create target array and reproject if needed
        if crs != target_crs or transform != target_transform:
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
                    resampling=resampling
                )
                
                # Store reprojected data
                self.data_layers[name] = {
                    'data': aligned_data,
                    'transform': target_transform,
                    'crs': target_crs,
                    'original_data': data,
                    'original_transform': transform,
                    'original_crs': crs,
                    'transformed': False
                }
                
                self.logger.info(f"Successfully added layer '{name}' (reprojected):")
                self.logger.info(f"  Shape: {aligned_data.shape}")
                self.logger.info(f"  Value range: {np.nanmin(aligned_data):.2f} to {np.nanmax(aligned_data):.2f}")
                
            except Exception as e:
                self.logger.error(f"Failed to reproject layer '{name}': {str(e)}")
                raise
        else:
            # No reprojection needed
            self.data_layers[name] = {
                'data': data,
                'transform': transform,
                'crs': crs,
                'transformed': False
            }
            self.logger.info(f"Added layer '{name}' (no reprojection needed)")
        
    def compute_data_layer(self, name: str, source_layer: str,
                          compute_func: Callable[[np.ndarray], np.ndarray],
                          transformed: bool = False,
                          cache_key: Optional[str] = None) -> np.ndarray:
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
        if transformed and not source_layer_info.get('transformed', False):
            raise ValueError(f"Source layer '{source_layer}' has not been transformed")
        
        # Get appropriate source data and metadata
        if transformed:
            source_data = source_layer_info['transformed_data']
            source_transform = source_layer_info['transformed_transform']
            source_crs = source_layer_info['transformed_crs']
        else:
            source_data = source_layer_info['data']
            source_transform = source_layer_info['transform']
            source_crs = source_layer_info['crs']
        
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
                    metadata={'source_layer': source_layer, 'transformed': transformed}
                )
                
            except Exception as e:
                self.logger.error(f"Failed to compute layer '{name}': {str(e)}")
                raise
        else:
            self.logger.info(f"Loaded cached computation for '{name}'")
            computed_data = cached['data']
            # Use cached metadata if available
            source_transform = cached.get('transform', source_transform)
            source_crs = cached.get('crs', source_crs)
        
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
                if layer.get('transformed', False):
                    self.logger.debug(f"Layer {name} already transformed, skipping")
                    pbar.update(1)
                    continue
                
                # Create target name from transform sequence
                layer_target = f"{name}_{'_'.join(t.__name__ for t in self.transforms)}"
                cached_layer = self.cache.load(layer_target)
                
                if cached_layer is None:
                    self.logger.info(f"Cache miss for {layer_target}, computing transforms...")
                    layer_data = layer['data'].copy()
                    current_transform = layer['transform']
                    current_crs = layer['crs']
                    
                    # Track transforms that were applied
                    applied_transforms = []
                    
                    for transform_func in self.transforms:
                        # Apply transform (wrapper always returns 3-tuple)
                        try:
                            layer_data, current_transform, new_crs = transform_func(layer_data, current_transform)
                            
                            # Update CRS if the transform provided a new one
                            if new_crs is not None:
                                current_crs = new_crs
                                
                            applied_transforms.append(transform_func.__name__)
                        except Exception as e:
                            self.logger.error(f"Failed applying transform {transform_func.__name__}: {str(e)}")
                            raise
                    
                    # Save result with comprehensive metadata
                    metadata = {
                        'transforms': applied_transforms,
                        'original_shape': layer['data'].shape,
                        'transformed_shape': layer_data.shape,
                        'original_crs': layer['crs'],
                        'final_crs': current_crs
                    }
                    
                    if cache:
                        self.cache.save(layer_target, layer_data, transform=current_transform, crs=current_crs, metadata=metadata)
                    
                    # Update layer info
                    self.data_layers[name].update({
                        'transformed_data': layer_data,
                        'transformed_transform': current_transform,
                        'transformed_crs': current_crs,
                        'transformed': True,
                        'transform_metadata': metadata
                    })
                else:
                    self.logger.info(f"Cache hit for {layer_target}")
                    self.data_layers[name].update({
                        'transformed_data': cached_layer['data'],
                        'transformed_transform': cached_layer['transform'],
                        'transformed_crs': cached_layer.get('crs', layer['crs']),
                        'transformed': True,
                        'transform_metadata': cached_layer.get('metadata', {})
                    })
                
                pbar.update(1)

        self.logger.info("Transforms applied successfully")

    def configure_for_target_vertices(self, target_vertices: int, order: int = 4) -> float:
        """
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
            f"  Resulting grid: {int(original_h * zoom_factor)} × {int(original_w * zoom_factor)}"
        )

        # Add downsampling transform to the pipeline
        self.transforms.append(downsample_raster(zoom_factor=zoom_factor, order=order))

        return zoom_factor

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

    def compute_colors(self):
        """
        Compute colors using color_func and optionally mask_func.
    
        Returns:
            np.ndarray: RGBA color array.
        """
        if not hasattr(self, 'color_mapping') or not hasattr(self, 'color_sources'):
            raise ValueError("Color mapping not set. Call set_color_mapping() first.")
    
        self.logger.info("Computing colors...")
    
        # Prepare color data arrays
        color_arrays = [
            self.data_layers[layer]['transformed_data']
            if self.data_layers[layer].get('transformed')
            else self.data_layers[layer]['data']
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
                self.data_layers[layer]['transformed_data']
                if self.data_layers[layer].get('transformed')
                else self.data_layers[layer]['data']
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

    def create_mesh(self, base_depth=-0.2, boundary_extension=True,
                    scale_factor=100.0, height_scale=1.0, center_model=True, verbose=True):
        """
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
        """
        start_time = time.time()
        self.logger.info("Creating terrain mesh...")

        # Get transformed DEM data
        if 'dem' not in self.data_layers or not self.data_layers['dem'].get('transformed', False):
            raise ValueError("Transformed DEM layer required for mesh creation")

        # Compute colors if color mapping is set and colors haven't been computed yet
        if hasattr(self, 'color_mapping') and not hasattr(self, 'colors'):
            self.compute_colors()

        dem_data = self.data_layers['dem']['transformed_data']
        height, width = dem_data.shape
        
        # Create valid points mask (non-NaN values)
        valid_mask = ~np.isnan(dem_data)
        
        # Use NumPy for coordinate generation
        y_indices, x_indices = np.mgrid[0:height, 0:width]
        y_valid = y_indices[valid_mask]
        x_valid = x_indices[valid_mask]
        
        # Generate vertex positions with scaling
        self.logger.info("Generating vertex positions...")
        positions = np.column_stack([
            x_valid / scale_factor,            # x position
            y_valid / scale_factor,            # y position
            dem_data[valid_mask] * height_scale # z position with height scaling
        ])
        
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
            'scale_factor': scale_factor,
            'height_scale': height_scale,
            'centered': center_model,
            'offset': self.model_offset.tolist(),
            'base_depth': base_depth
        }
        
        # Create mapping from (y,x) coords to vertex indices - using dictionaries for O(1) lookups
        self.logger.info("Creating coordinate to index mapping...")
        coord_to_index = {(y, x): i for i, (y, x) in enumerate(zip(y_valid, x_valid))}
        
        # OPTIMIZATION: Find boundary points using morphological operations
        self.logger.info("Finding boundary points with optimized algorithm...")
        # Interior points have 4 neighbors in a 4-connected neighborhood
        from scipy import ndimage
        struct = ndimage.generate_binary_structure(2, 1)  # 4-connected structure
        eroded = ndimage.binary_erosion(valid_mask, struct)
        boundary_mask = valid_mask & ~eroded
        
        # Get boundary coords as (y,x) tuples
        boundary_indices = np.where(boundary_mask)
        boundary_coords = list(zip(boundary_indices[0], boundary_indices[1]))
        
        # Only sort boundary points if needed (they're used for side faces)
        if boundary_extension:
            boundary_points = self._sort_boundary_points_optimized(boundary_coords)
        else:
            boundary_points = boundary_coords
        
        # OPTIMIZATION: Vectorized face generation using NumPy operations
        self.logger.info("Generating faces with vectorized operations...")
        
        # Generate all potential quad faces
        y_quads, x_quads = np.mgrid[0:height-1, 0:width-1]
        y_quads = y_quads.flatten()
        x_quads = x_quads.flatten()
        
        # Collect faces efficiently
        faces = []
        
        # Use batch processing to reduce Python loop overhead
        batch_size = 10000
        n_quads = len(y_quads)
        
        for batch_start in range(0, n_quads, batch_size):
            batch_end = min(batch_start + batch_size, n_quads)
            batch_y = y_quads[batch_start:batch_end]
            batch_x = x_quads[batch_start:batch_end]
            
            # For each quad, check if corners exist in valid points
            for i in range(batch_end - batch_start):
                y, x = batch_y[i], batch_x[i]
                quad_points = [(y, x), (y, x+1), (y+1, x+1), (y+1, x)]
                
                # Get indices for each corner that exists
                valid_indices = []
                for point in quad_points:
                    if point in coord_to_index:
                        valid_indices.append(coord_to_index[point])
                
                # Only create faces with at least 3 points
                if len(valid_indices) >= 3:
                    faces.append(tuple(valid_indices))
        
        # Handle boundary extension if needed
        if boundary_extension:
            self.logger.info("Creating optimized boundary extension...")
            
            # Preallocate arrays for better performance
            n_boundary = len(boundary_points)
            boundary_vertices = np.zeros((n_boundary, 3), dtype=float)
            boundary_faces = []
            
            # Create bottom vertices for each boundary point
            for i, (y, x) in enumerate(boundary_points):
                # Get the original position
                original_idx = coord_to_index.get((y, x))
                if original_idx is None:
                    continue
                    
                # Copy position but set z to base_depth
                pos = positions[original_idx].copy()
                pos[2] = base_depth
                boundary_vertices[i] = pos
            
            # Create side faces efficiently
            boundary_indices = [coord_to_index.get((y, x)) for y, x in boundary_points]
            base_indices = list(range(len(positions), len(positions) + len(boundary_points)))
            
            for i in range(n_boundary):
                # Skip invalid indices
                if boundary_indices[i] is None:
                    continue
                    
                next_i = (i + 1) % n_boundary
                # Skip if next boundary point is invalid
                if boundary_indices[next_i] is None:
                    continue
                    
                # Create quad connecting top boundary to bottom
                boundary_faces.append((
                    boundary_indices[i],
                    boundary_indices[next_i],
                    base_indices[next_i],
                    base_indices[i]
                ))
            
            # Extend vertices with boundary vertices
            vertices = np.vstack([positions, boundary_vertices])
            # Add boundary faces to complete the mesh
            faces.extend(boundary_faces)
        else:
            vertices = positions

        # Create the Blender mesh
        self.logger.info(f"Creating Blender mesh with {len(vertices)} vertices and {len(faces)} faces...")
        
        try:
            # Create mesh datablock
            mesh = bpy.data.meshes.new("TerrainMesh")
            mesh.from_pydata(vertices.tolist(), [], faces)
            mesh.update(calc_edges=True)
            
            # OPTIMIZATION: Apply colors efficiently if available
            if hasattr(self, 'colors') and self.colors is not None:
                self.logger.info("Applying vertex colors with optimized method...")
                color_layer = mesh.vertex_colors.new(name="TerrainColors")

                # OPTIMIZATION: Use numpy operations where possible
                if len(color_layer.data) > 0:
                    # Create color data array (Blender expects normalized 0-1 floats)
                    color_data = np.zeros((len(color_layer.data), 4), dtype=np.float32)

                    # Normalize colors to 0-1 range if they're uint8
                    colors_normalized = self.colors.astype(np.float32)
                    if colors_normalized.max() > 1.0:
                        colors_normalized = colors_normalized / 255.0

                    # For each polygon loop, get vertex and set color
                    for poly in mesh.polygons:
                        for loop_idx in poly.loop_indices:
                            vertex_idx = mesh.loops[loop_idx].vertex_index

                            # Only apply colors to top vertices
                            if vertex_idx < len(positions):
                                y, x = y_valid[vertex_idx], x_valid[vertex_idx]

                                # Check bounds
                                if 0 <= y < colors_normalized.shape[0] and 0 <= x < colors_normalized.shape[1]:
                                    color_data[loop_idx] = colors_normalized[y, x]

                    # Batch assign all colors at once
                    try:
                        color_layer.data.foreach_set('color', color_data.flatten())
                    except Exception as e:
                        self.logger.warning(f"Batch color assignment failed: {e}")
                        # Fallback to slower per-loop assignment
                        for i, color in enumerate(color_data):
                            color_layer.data[i].color = color
            
            # Create object and link to scene
            obj = bpy.data.objects.new("Terrain", mesh)
            bpy.context.scene.collection.objects.link(obj)
            
            # Create and assign material
            mat = bpy.data.materials.new(name="TerrainMaterial")
            mat.use_nodes = True
            obj.data.materials.append(mat)
            apply_colormap_material(mat)
            
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
        # Quick return for small boundaries
        if len(boundary_coords) <= 2:
            return boundary_coords
            
        # Start with leftmost-topmost point for consistency
        start_point = min(boundary_coords, key=lambda p: (p[1], p[0]))
        
        # Use a KD-tree for nearest neighbor queries - much faster than manual distance calculation
        from scipy.spatial import cKDTree
        
        # Convert to numpy array for KD-tree
        points_array = np.array(boundary_coords)
        kdtree = cKDTree(points_array)
        
        # Initialize result with start point
        ordered = [start_point]
        
        # Track points we've already used - faster lookups
        used_indices = set([boundary_coords.index(start_point)])
        
        current = start_point
        
        # Find next closest point until all points are used
        while len(ordered) < len(boundary_coords):
            # Query KD-tree for 10 nearest neighbors (more than enough)
            distances, indices = kdtree.query(current, k=10)
            
            # Find the closest unused point
            next_point = None
            for i in range(len(indices)):
                idx = indices[i]
                if idx < len(points_array) and idx not in used_indices:
                    next_point = tuple(points_array[idx])
                    used_indices.add(idx)
                    break
            
            # If no more valid neighbors, break
            if next_point is None:
                break
                
            ordered.append(next_point)
            current = next_point
        
        return ordered