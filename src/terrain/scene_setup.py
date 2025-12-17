"""
Scene setup operations for Blender terrain visualization.

This module contains functions for setting up Blender scenes, cameras,
lighting, and atmosphere for terrain rendering.
"""

import logging
import numpy as np
from math import radians

import bpy
from mathutils import Euler, Vector


def hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """Convert hex color string to normalized RGB tuple.

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
    """
    # Remove hash if present
    hex_str = hex_color.lstrip("#")

    # Handle shorthand format (#RGB -> #RRGGBB)
    if len(hex_str) == 3:
        hex_str = "".join([c * 2 for c in hex_str])

    # Validate length
    if len(hex_str) != 6:
        raise ValueError(
            f"Invalid hex color: '{hex_color}'. "
            f"Expected #RRGGBB, #RGB, RRGGBB, or RGB format."
        )

    # Validate characters are valid hex
    try:
        int(hex_str, 16)
    except ValueError:
        raise ValueError(f"Invalid hex color: '{hex_color}'. Contains non-hex characters.")

    # Convert to RGB tuple (normalize to 0.0-1.0)
    r = int(hex_str[0:2], 16) / 255.0
    g = int(hex_str[2:4], 16) / 255.0
    b = int(hex_str[4:6], 16) / 255.0

    return (r, g, b)


def create_matte_material(
    name: str = "BackgroundMaterial",
    color: str | tuple = "#F5F5F0",
    material_roughness: float = 1.0,
    receive_shadows: bool = False,
) -> "bpy.types.Material":
    """Create a matte Principled BSDF material for backgrounds.

    Creates a physically-based matte material with configurable color and
    shadow receiving behavior. Useful for background planes and other
    non-reflective surfaces.

    Args:
        name: Name for the material (default: "BackgroundMaterial")
        color: Color as hex string (e.g., "#F5F5F0") or RGB tuple (default: eggshell white)
        material_roughness: Roughness value 0.0-1.0, 1.0 = fully matte (default: 1.0)
        receive_shadows: Whether the material receives shadows (default: False)

    Returns:
        Blender Material object configured as specified

    Note:
        This function requires Blender and the bpy module to be available.
        Call only from within a Blender environment.

    Raises:
        RuntimeError: If called outside of Blender environment
    """
    logger = logging.getLogger(__name__)

    # Convert hex color to RGB if needed
    if isinstance(color, str):
        color = hex_to_rgb(color)

    # Create material
    material = bpy.data.materials.new(name)
    material.use_nodes = True

    # Clear default nodes
    material.node_tree.nodes.clear()

    # Create Principled BSDF shader
    nodes = material.node_tree.nodes
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")

    # Set Principled BSDF properties
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)  # RGBA
    bsdf.inputs["Roughness"].default_value = material_roughness
    bsdf.inputs["Metallic"].default_value = 0.0  # Not metallic

    # Create output node
    output = nodes.new(type="ShaderNodeOutputMaterial")

    if receive_shadows:
        # Simple pass-through: just connect BSDF to output
        material.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
        logger.debug(f"Created material '{name}' with shadow receiving enabled")
    else:
        # Create a shader that doesn't receive shadows by using an emission
        # fallback on shadow rays, making the material appear as if it doesn't
        # receive shadows (appears as the base color without darkening)

        # For a simple approach: use a Mix Shader based on Is Shadow Ray
        light_path = nodes.new(type="ShaderNodeLightPath")
        mix_shader = nodes.new(type="ShaderNodeMix")
        mix_shader.data_type = "SHADER"

        # Emission shader for shadow rays (uses the base color without shadowing)
        emission = nodes.new(type="ShaderNodeEmission")
        emission.inputs["Color"].default_value = (*color, 1.0)
        emission.inputs["Strength"].default_value = 1.0

        # Wire up: if is_shadow_ray, use emission; otherwise use BSDF
        material.node_tree.links.new(
            light_path.outputs["Is Shadow Ray"], mix_shader.inputs["Selection"]
        )
        material.node_tree.links.new(bsdf.outputs["BSDF"], mix_shader.inputs["A"])
        material.node_tree.links.new(emission.outputs["Emission"], mix_shader.inputs["B"])
        material.node_tree.links.new(mix_shader.outputs["Result"], output.inputs["Surface"])

        logger.debug(f"Created material '{name}' with shadow receiving disabled")

    logger.info(f"✓ Created material '{name}' (roughness={material_roughness}, receive_shadows={receive_shadows})")
    return material


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
    ortho_scale=1.2,
):
    """Position camera relative to mesh(es) using intuitive cardinal directions.

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
        sun_angle: Angle of sun light in degrees. Default: 2
        sun_energy: Intensity of sun light. Default: 3
        focal_length: Camera focal length in mm (perspective cameras only). Default: 50
        ortho_scale: Multiplier for orthographic camera scale relative to mesh diagonal.
            Higher values zoom out (show more area), lower values zoom in.
            Only affects orthographic cameras. Default: 1.2

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

    # Normalize input: handle both single mesh and list of meshes
    if isinstance(mesh_obj, list):
        meshes = mesh_obj
    else:
        meshes = [mesh_obj]

    # Compute combined bounding box across all meshes
    all_xs = []
    all_ys = []
    all_zs = []

    for mesh in meshes:
        bbox = mesh.bound_box
        # bound_box coordinates are in local space, need to transform to world space
        world_matrix = mesh.matrix_world
        for vertex in bbox:
            world_vertex = world_matrix @ Vector(vertex)
            all_xs.append(world_vertex[0])
            all_ys.append(world_vertex[1])
            all_zs.append(world_vertex[2])

    # Combined bounds
    min_x, max_x = min(all_xs), max(all_xs)
    min_y, max_y = min(all_ys), max(all_ys)
    min_z, max_z = min(all_zs), max(all_zs)

    center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2])

    # Calculate combined mesh diagonal for scaling
    mesh_size = np.array([max_x - min_x, max_y - min_y, max_z - min_z])
    mesh_diagonal = np.linalg.norm(mesh_size)

    if len(meshes) > 1:
        logger.debug(f"Combined bounding box: size={mesh_size}, diagonal={mesh_diagonal:.2f}")

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
        camera_angle = Euler((0, 0, 0), "XYZ")
    else:
        # In Blender, camera's local -Z axis is forward, +Y is up
        cam_to_target = Vector(target_pos - camera_pos).normalized()

        # Use to_track_quat to calculate rotation
        # Arguments: (forward_axis, up_axis) where -Z is camera forward, Y is world up
        track_quat = cam_to_target.to_track_quat("-Z", "Y")
        camera_angle = track_quat.to_euler()

    logger.debug(f"Camera position: {camera_pos}, angle: {camera_angle}")

    # Set ortho scale based on mesh size if orthographic
    computed_ortho_scale = mesh_diagonal * ortho_scale if camera_type == "ORTHO" else distance * mesh_diagonal

    # Create camera directly, optionally with light
    camera = setup_camera(
        camera_angle=camera_angle,
        camera_location=tuple(camera_pos),
        scale=computed_ortho_scale,
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
