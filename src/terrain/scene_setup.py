"""
Scene setup operations for Blender terrain visualization.

This module contains functions for setting up Blender scenes, cameras,
lighting, and atmosphere for terrain rendering.
"""

import logging
import numpy as np
from math import radians

import bpy


def clear_scene():
    """
    Clear all objects from the Blender scene.

    Resets the scene to factory settings (empty scene) and removes all default
    objects. Useful before importing terrain meshes to ensure a clean workspace.

    Raises:
        RuntimeError: If Blender module (bpy) is not available.
    """
    logger = logging.getLogger(__name__)
    logger.info("Clearing Blender scene...")

    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Count objects to be removed
    n_objects = len(bpy.data.objects)
    logger.debug(f"Removing {n_objects} objects")

    for obj in bpy.data.objects:
        logger.debug(f"Removing object: {obj.name}")
        bpy.data.objects.remove(obj, do_unlink=True)

    logger.info("Scene cleared successfully")


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
    logger = logging.getLogger(__name__)

    # Validate camera_type
    if camera_type not in ("PERSP", "ORTHO"):
        raise ValueError("camera_type must be 'PERSP' or 'ORTHO'")

    logger.debug(f"Creating {camera_type} camera...")
    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)

    cam_obj.location = camera_location
    cam_obj.rotation_euler = camera_angle

    # Set camera type and type-specific parameters
    cam_data.type = camera_type

    if camera_type == "PERSP":
        cam_data.lens = focal_length
        logger.debug(
            f"Perspective camera configured at {camera_location} with {focal_length}mm lens"
        )
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
    logger = logging.getLogger(__name__)
    logger.debug("Creating sun light...")
    sun = bpy.data.lights.new(name="Sun", type="SUN")
    sun_obj = bpy.data.objects.new("Sun", sun)
    bpy.context.scene.collection.objects.link(sun_obj)

    sun_obj.location = location
    sun_obj.rotation_euler = rotation_euler
    sun.angle = angle
    sun.energy = energy

    logger.debug(f"Sun light configured with angle {angle}° and energy {energy}")

    return sun_obj


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
    logger = logging.getLogger(__name__)
    logger.info("Setting up camera and lighting...")
    camera = setup_camera(camera_angle, camera_location, scale, focal_length, camera_type)
    light = setup_light(angle=sun_angle, energy=sun_energy)
    logger.info("Camera and lighting setup complete")
    return camera, light


def position_camera_relative(
    mesh_obj,
    direction="south",
    distance=1.5,
    elevation=0.5,
    look_at="center",
    camera_type="ORTHO",
    sun_angle=2,
    sun_energy=3,
    focal_length=50,
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
    logger = logging.getLogger(__name__)

    # Direction vector mappings (Blender Z-up coordinate system)
    DIRECTIONS = {
        "north": (0, 1, 0),  # +Y
        "south": (0, -1, 0),  # -Y
        "east": (1, 0, 0),  # +X
        "west": (-1, 0, 0),  # -X
        "above": (0, 0, 1),  # +Z (straight up)
        "northeast": (0.7071, 0.7071, 0),
        "northwest": (-0.7071, 0.7071, 0),
        "southeast": (0.7071, -0.7071, 0),
        "southwest": (-0.7071, -0.7071, 0),
    }

    if direction not in DIRECTIONS:
        raise ValueError(
            f"Invalid direction '{direction}'. Must be one of: {', '.join(DIRECTIONS.keys())}"
        )

    logger.info(f"Positioning camera {direction} of mesh with {camera_type} camera")

    # Get mesh bounds
    bbox = mesh_obj.bound_box
    xs = [v[0] for v in bbox]
    ys = [v[1] for v in bbox]
    zs = [v[2] for v in bbox]

    center = np.array([(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2])

    # Calculate mesh diagonal for scaling
    mesh_size = np.array([max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)])
    mesh_diagonal = np.linalg.norm(mesh_size)

    # Get direction vector and calculate camera position
    dir_vector = np.array(DIRECTIONS[direction])

    # For 'above', position directly overhead; for others, position offset from center
    if direction == "above":
        camera_pos = center.copy()
    else:
        camera_pos = center + dir_vector * distance * mesh_diagonal

    # Add elevation (height above ground)
    camera_pos[2] = center[2] + elevation * mesh_diagonal

    # Determine look-at target
    if look_at == "center":
        target_pos = center
    else:
        target_pos = np.array(look_at)

    # Calculate rotation to point at target
    # Special case: for overhead (above) view, use zero rotation
    # This avoids gimbal lock ambiguity when looking straight down
    if direction == "above":
        from mathutils import Euler

        camera_angle = Euler((0, 0, 0), "XYZ")
    else:
        # In Blender, camera's local -Z axis is forward, +Y is up
        from mathutils import Vector

        cam_to_target = Vector(target_pos - camera_pos).normalized()

        # Use to_track_quat to calculate rotation
        # Arguments: (forward_axis, up_axis) where -Z is camera forward, Y is world up
        track_quat = cam_to_target.to_track_quat("-Z", "Y")
        camera_angle = track_quat.to_euler()

    logger.debug(f"Camera position: {camera_pos}, angle: {camera_angle}")

    # Set ortho scale based on mesh size if orthographic
    ortho_scale = mesh_diagonal * 1.2 if camera_type == "ORTHO" else distance * mesh_diagonal

    # Create camera directly, optionally with light
    camera = setup_camera(
        camera_angle=camera_angle,
        camera_location=tuple(camera_pos),
        scale=ortho_scale,
        focal_length=focal_length,
        camera_type=camera_type,
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
    logger = logging.getLogger(__name__)
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
        output = nodes.new("ShaderNodeOutputWorld")
        background = nodes.new("ShaderNodeBackground")
        volume = nodes.new("ShaderNodeVolumePrincipled")

        # Configure volume properties
        volume.inputs["Density"].default_value = density
        volume.inputs["Anisotropy"].default_value = anisotropy
        volume.inputs["Color"].default_value = scatter_color

        # Connect nodes
        links.new(background.outputs["Background"], output.inputs["Surface"])
        links.new(volume.outputs["Volume"], output.inputs["Volume"])

        logger.info(f"World atmosphere configured with density {density}")
        return world

    except Exception as e:
        logger.error(f"Failed to setup world atmosphere: {str(e)}")
        raise
