"""
Scene setup operations for Blender terrain visualization.

This module contains functions for setting up Blender scenes, cameras,
lighting, and atmosphere for terrain rendering.
"""

import logging
import numpy as np
from math import radians, tan, atan, degrees

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
    flat_color: bool = False,
) -> "bpy.types.Material":
    """Create a matte material for backgrounds.

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

    nodes = material.node_tree.nodes
    output = nodes.new(type="ShaderNodeOutputMaterial")

    if flat_color:
        # Pure emission shader - exact color regardless of lighting
        emission = nodes.new(type="ShaderNodeEmission")
        emission.inputs["Color"].default_value = (*color, 1.0)
        emission.inputs["Strength"].default_value = 1.0

        material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
        logger.info(f"✓ Created flat color material '{name}' (emission, ignores lighting)")
        return material

    # Physically-based material using Principled BSDF
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")

    # Set Principled BSDF properties
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)  # RGBA
    bsdf.inputs["Roughness"].default_value = material_roughness
    bsdf.inputs["Metallic"].default_value = 0.0  # Not metallic

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


def calculate_camera_frustum_size(
    camera_type: str,
    aspect_ratio: float,
    ortho_scale: float = None,
    fov_degrees: float = None,
    distance: float = None,
) -> tuple[float, float]:
    """Calculate the visible area of a camera at a given distance.

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
    """
    logger = logging.getLogger(__name__)

    if camera_type == "ORTHO":
        if ortho_scale is None:
            raise ValueError("ortho_scale parameter is required for orthographic cameras")

        width = float(ortho_scale)
        height = width / aspect_ratio
        logger.debug(
            f"Orthographic frustum: scale={ortho_scale}, aspect={aspect_ratio:.3f} "
            f"-> width={width:.2f}, height={height:.2f}"
        )
        return (width, height)

    elif camera_type == "PERSP":
        if fov_degrees is None or distance is None:
            raise ValueError("fov_degrees and distance parameters are required for perspective cameras")

        # Convert FOV from degrees to radians
        fov_radians = radians(fov_degrees)

        # Calculate frustum width using the tangent of half the FOV angle
        # frustum_width = 2 * distance * tan(FOV / 2)
        width = 2.0 * distance * tan(fov_radians / 2)
        height = width / aspect_ratio

        logger.debug(
            f"Perspective frustum: fov={fov_degrees:.2f}°, distance={distance:.2f}, "
            f"aspect={aspect_ratio:.3f} -> width={width:.2f}, height={height:.2f}"
        )
        return (width, height)

    else:
        raise ValueError(f"Invalid camera_type '{camera_type}'. Must be 'ORTHO' or 'PERSP'.")


def create_background_plane(
    camera,
    mesh_or_meshes,
    distance_below: float = 50.0,
    color: str | tuple = "#F5F5F0",
    size_multiplier: float = 2.0,
    receive_shadows: bool = False,
    flat_color: bool = False,
) -> "bpy.types.Object":
    """Create a background plane for Blender terrain renders.

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
    """
    logger = logging.getLogger(__name__)

    logger.info("Creating background plane...")

    # Normalize input: handle both single mesh and list of meshes
    if isinstance(mesh_or_meshes, list):
        meshes = mesh_or_meshes
    else:
        meshes = [mesh_or_meshes]

    if not meshes:
        raise ValueError("At least one mesh must be provided")

    # Compute combined bounding box to find lowest Z point
    all_zs = []
    center_z = 0

    for mesh in meshes:
        bbox = mesh.bound_box
        world_matrix = mesh.matrix_world
        for vertex in bbox:
            world_vertex = world_matrix @ Vector(vertex)
            all_zs.append(world_vertex[2])
        # Also track center for positioning
        center_z += (min([world_matrix @ Vector(v) for v in bbox], key=lambda x: x[2])[2])

    min_z = min(all_zs)
    plane_z = min_z - distance_below

    logger.debug(f"Mesh lowest Z: {min_z:.2f}, plane position Z: {plane_z:.2f}")

    # Get camera properties to calculate frustum size
    cam_data = camera.data
    if cam_data.type == "ORTHO":
        camera_type = "ORTHO"
        ortho_scale = cam_data.ortho_scale
        fov_degrees = None
        distance = None
    elif cam_data.type == "PERSP":
        camera_type = "PERSP"
        ortho_scale = None
        # Convert focal length (mm) to FOV in degrees
        # FOV (degrees) = 2 * arctan(sensor_width / (2 * focal_length))
        # Blender's default sensor width is 36mm for full-frame
        sensor_width = 36.0  # mm, standard full-frame
        fov_radians = 2 * atan(sensor_width / (2 * cam_data.lens))
        fov_degrees = degrees(fov_radians)

        # Calculate distance from camera to plane
        cam_to_plane = plane_z - camera.location[2]
        distance = abs(cam_to_plane)
    else:
        raise ValueError(f"Unsupported camera type: {cam_data.type}")

    # Get render aspect ratio
    scene = bpy.context.scene
    render = scene.render
    aspect_ratio = render.resolution_x / render.resolution_y

    logger.debug(f"Camera type: {camera_type}, aspect ratio: {aspect_ratio:.3f}")

    # Calculate frustum size at plane location
    frustum_width, frustum_height = calculate_camera_frustum_size(
        camera_type=camera_type,
        aspect_ratio=aspect_ratio,
        ortho_scale=ortho_scale,
        fov_degrees=fov_degrees,
        distance=distance,
    )

    # Apply size multiplier for safety margin
    plane_width = frustum_width * size_multiplier
    plane_height = frustum_height * size_multiplier

    logger.debug(
        f"Frustum: {frustum_width:.2f}x{frustum_height:.2f}, "
        f"plane (after {size_multiplier}x multiplier): {plane_width:.2f}x{plane_height:.2f}"
    )

    # Create plane mesh
    mesh_data = bpy.data.meshes.new("BackgroundPlane")
    plane_obj = bpy.data.objects.new("BackgroundPlane", mesh_data)
    bpy.context.scene.collection.objects.link(plane_obj)

    # Create plane vertices (4 corners)
    half_w = plane_width / 2
    half_h = plane_height / 2
    verts = [
        (-half_w, -half_h, 0),
        (half_w, -half_h, 0),
        (half_w, half_h, 0),
        (-half_w, half_h, 0),
    ]

    # Create faces (single quad)
    faces = [(0, 1, 2, 3)]

    # Fill mesh
    mesh_data.from_pydata(verts, [], faces)
    mesh_data.update()

    # Position plane below mesh, centered on camera XY
    cam_xy = (camera.location[0], camera.location[1])
    plane_obj.location = (*cam_xy, plane_z)

    logger.debug(f"Plane positioned at: {plane_obj.location}")

    # Create and assign material
    material = create_matte_material(
        name="BackgroundMaterial",
        color=color,
        material_roughness=1.0,
        receive_shadows=receive_shadows,
        flat_color=flat_color,
    )

    if mesh_data.materials:
        mesh_data.materials[0] = material
    else:
        mesh_data.materials.append(material)

    logger.info(
        f"✓ Created background plane: {plane_width:.2f}x{plane_height:.2f} "
        f"at Z={plane_z:.2f} (color={color}, shadows={receive_shadows})"
    )

    return plane_obj


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

    # Set clipping planes to handle large scenes with background planes
    # Default clip_end of 100 is too small for terrain visualization
    cam_data.clip_start = 0.1
    cam_data.clip_end = 10000  # Sufficient for most terrain scenes

    bpy.context.scene.camera = cam_obj

    return cam_obj


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
    """
    from math import sin, cos
    from mathutils import Vector

    logger = logging.getLogger(__name__)
    logger.debug("Creating sun light...")

    sun = bpy.data.lights.new(name="Sun", type="SUN")
    sun_obj = bpy.data.objects.new("Sun", sun)
    bpy.context.scene.collection.objects.link(sun_obj)

    sun_obj.location = location

    # Calculate rotation from azimuth/elevation if provided
    if azimuth is not None and elevation is not None:
        # Convert to radians
        az_rad = radians(azimuth)
        el_rad = radians(elevation)

        # Calculate direction vector (where light shines TO, not FROM)
        # Sun comes FROM azimuth, so light travels in opposite direction
        # In Blender: +Y = North, +X = East, +Z = Up
        x = -sin(az_rad) * cos(el_rad)
        y = -cos(az_rad) * cos(el_rad)
        z = -sin(el_rad)

        direction = Vector((x, y, z)).normalized()

        # Convert direction to Euler rotation
        # Sun's local -Z axis should point in this direction
        rotation = direction.to_track_quat("-Z", "Y").to_euler()
        sun_obj.rotation_euler = rotation

        logger.debug(f"Sun from azimuth {azimuth}° elevation {elevation}° → direction {direction}")
    elif rotation_euler is not None:
        sun_obj.rotation_euler = rotation_euler
    else:
        # Default: sun from northwest (azimuth ~315°)
        sun_obj.rotation_euler = (0, radians(315), 0)

    sun.angle = angle
    sun.energy = energy

    logger.debug(f"Sun light configured with angle {angle}° and energy {energy}")

    return sun_obj


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
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Setting up two-point lighting (sun: azimuth={sun_azimuth}°, elevation={sun_elevation}°)...")

    lights = []

    # Primary sun - only create if energy > 0
    if sun_energy > 0:
        sun_light = setup_light(
            angle=sun_angle,
            energy=sun_energy,
            azimuth=sun_azimuth,
            elevation=sun_elevation,
        )
        sun_light.data.color = sun_color
        lights.append(sun_light)
        logger.debug(f"  Sun: azimuth={sun_azimuth}°, elevation={sun_elevation}°, energy={sun_energy}")

    # Fill light - only create if energy > 0
    if fill_energy > 0:
        fill_light = setup_light(
            angle=fill_angle,
            energy=fill_energy,
            azimuth=fill_azimuth,
            elevation=fill_elevation,
        )
        fill_light.data.color = fill_color
        lights.append(fill_light)
        logger.info(f"  Fill: azimuth={fill_azimuth}°, elevation={fill_elevation}°, energy={fill_energy}")

    logger.info("✓ Two-point lighting setup complete")
    return lights


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
    sun_angle=0,
    sun_energy=0,
    sun_azimuth=None,
    sun_elevation=None,
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
        setup_light(
            angle=sun_angle,
            energy=sun_energy,
            azimuth=sun_azimuth,
            elevation=sun_elevation,
        )

    return camera


def setup_world_atmosphere(density=0.0002, scatter_color=(1, 1, 1, 1), anisotropy=0.8):
    """Set up world volume for atmospheric effects.

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

        # Find existing World Output node, or create one
        output = None
        for node in nodes:
            if node.type == "OUTPUT_WORLD":
                output = node
                break

        if output is None:
            # No existing world setup - create full setup
            logger.debug("Creating new world node network...")
            nodes.clear()
            output = nodes.new("ShaderNodeOutputWorld")
            background = nodes.new("ShaderNodeBackground")
            links.new(background.outputs["Background"], output.inputs["Surface"])
        else:
            logger.debug("Adding atmosphere to existing world setup...")

        # Create or find volume node
        volume = None
        for node in nodes:
            if node.type == "VOLUME_PRINCIPLED":
                volume = node
                break

        if volume is None:
            volume = nodes.new("ShaderNodeVolumePrincipled")
            volume.location = (output.location.x - 200, output.location.y - 200)

        # Configure volume properties
        volume.inputs["Density"].default_value = density
        volume.inputs["Anisotropy"].default_value = anisotropy
        volume.inputs["Color"].default_value = scatter_color

        # Connect volume to output (this only affects the Volume slot, not Surface)
        links.new(volume.outputs["Volume"], output.inputs["Volume"])

        logger.info(f"World atmosphere configured with density {density}")
        return world

    except Exception as e:
        logger.error(f"Failed to setup world atmosphere: {str(e)}")
        raise


def setup_hdri_lighting(
    sun_elevation: float = 30.0,
    sun_rotation: float = 225.0,
    sun_intensity: float = 1.0,
    sun_size: float = 0.545,
    air_density: float = 1.0,
    visible_to_camera: bool = False,
    camera_background: tuple = None,
):
    """Set up HDRI-style sky lighting using Blender's Nishita sky model.

    Creates realistic sky lighting that contributes to ambient illumination
    without being visible in the final render (by default).

    The Nishita sky model provides physically-based atmospheric scattering
    for natural-looking outdoor lighting.

    Args:
        sun_elevation: Sun elevation angle in degrees (0=horizon, 90=overhead)
        sun_rotation: Sun rotation/azimuth in degrees (0=front, 180=back)
        sun_intensity: Multiplier for sun intensity (default: 1.0)
        sun_size: Angular diameter of sun disc in degrees (default: 0.545 = real sun).
                  Larger values create softer shadows, smaller values create sharper shadows.
        air_density: Atmospheric density (default: 1.0, higher=hazier)
        visible_to_camera: If False, sky is invisible but still lights scene
        camera_background: RGB tuple for background color when sky is invisible.
                          Default None = use transparent (black behind scene).
                          Use (0.9, 0.9, 0.9) for light gray if using atmosphere.

    Returns:
        bpy.types.World: The configured world object
    """
    logger = logging.getLogger(__name__)
    logger.info("Setting up HDRI sky lighting...")

    try:
        world = bpy.context.scene.world
        if world is None:
            world = bpy.data.worlds.new("World")
            bpy.context.scene.world = world

        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links

        # Clear existing nodes
        nodes.clear()

        # Create node network
        output = nodes.new("ShaderNodeOutputWorld")
        background = nodes.new("ShaderNodeBackground")
        sky_texture = nodes.new("ShaderNodeTexSky")

        # Position nodes
        output.location = (400, 0)
        background.location = (200, 0)
        sky_texture.location = (0, 0)

        # Configure sky model for realistic atmospheric scattering
        # Blender 5.0+ renamed NISHITA to MULTIPLE_SCATTERING/SINGLE_SCATTERING
        # Try modern names first, then fall back to older versions
        sky_configured = False

        # Blender 5.0+: MULTIPLE_SCATTERING (best quality, multiple atmospheric scattering)
        if not sky_configured:
            try:
                sky_texture.sky_type = "MULTIPLE_SCATTERING"
                sky_texture.sun_elevation = radians(sun_elevation)
                sky_texture.sun_rotation = radians(sun_rotation)
                sky_texture.sun_intensity = sun_intensity
                sky_texture.sun_size = radians(sun_size)  # Angular size in radians
                sky_texture.air_density = air_density
                sky_configured = True
                logger.info(f"  Using MULTIPLE_SCATTERING sky (elevation={sun_elevation}°, rotation={sun_rotation}°, size={sun_size}°)")
            except (TypeError, AttributeError):
                pass

        # Blender 2.90-4.x: NISHITA
        if not sky_configured:
            try:
                sky_texture.sky_type = "NISHITA"
                sky_texture.sun_elevation = radians(sun_elevation)
                sky_texture.sun_rotation = radians(sun_rotation)
                sky_texture.sun_intensity = sun_intensity
                sky_texture.sun_size = radians(sun_size)  # Angular size in radians
                sky_texture.air_density = air_density
                sky_configured = True
                logger.info(f"  Using NISHITA sky (elevation={sun_elevation}°, rotation={sun_rotation}°, size={sun_size}°)")
            except (TypeError, AttributeError):
                pass

        # Fallback: HOSEK_WILKIE (older but widely supported)
        if not sky_configured:
            sky_texture.sky_type = "HOSEK_WILKIE"
            sky_texture.sun_elevation = radians(sun_elevation)
            sky_texture.sun_rotation = radians(sun_rotation)
            sky_texture.turbidity = 2.5  # Atmospheric haziness
            logger.info(f"  Using HOSEK_WILKIE sky (elevation={sun_elevation}°, rotation={sun_rotation}°)")

        # Connect nodes
        links.new(sky_texture.outputs["Color"], background.inputs["Color"])
        links.new(background.outputs["Background"], output.inputs["Surface"])

        # Configure ray visibility - sky lights scene but isn't visible to camera
        if not visible_to_camera:
            # In Cycles, we can use a Light Path node to make the background
            # invisible to camera rays while still contributing to lighting
            light_path = nodes.new("ShaderNodeLightPath")
            mix_shader = nodes.new("ShaderNodeMixShader")

            light_path.location = (0, -200)
            mix_shader.location = (400, 0)

            if camera_background is not None:
                # Use solid color for camera rays (works with atmosphere/volume)
                camera_bg = nodes.new("ShaderNodeBackground")
                camera_bg.location = (200, -200)
                camera_bg.inputs["Color"].default_value = (*camera_background, 1.0)
                camera_bg.inputs["Strength"].default_value = 1.0

                # Reconnect: if camera ray, use solid bg; otherwise use sky
                links.new(light_path.outputs["Is Camera Ray"], mix_shader.inputs["Fac"])
                links.new(background.outputs["Background"], mix_shader.inputs[1])
                links.new(camera_bg.outputs["Background"], mix_shader.inputs[2])
                links.new(mix_shader.outputs["Shader"], output.inputs["Surface"])

                logger.info(f"✓ HDRI sky lighting configured (camera sees {camera_background})")
            else:
                # Use transparent for camera rays (original behavior)
                transparent = nodes.new("ShaderNodeBsdfTransparent")
                transparent.location = (200, -200)

                # Reconnect: if camera ray, use transparent; otherwise use background
                links.new(light_path.outputs["Is Camera Ray"], mix_shader.inputs["Fac"])
                links.new(background.outputs["Background"], mix_shader.inputs[1])
                links.new(transparent.outputs["BSDF"], mix_shader.inputs[2])
                links.new(mix_shader.outputs["Shader"], output.inputs["Surface"])

                logger.info("✓ HDRI sky lighting configured (invisible to camera)")
        else:
            logger.info("✓ HDRI sky lighting configured (visible)")

        return world

    except Exception as e:
        logger.error(f"Failed to setup HDRI lighting: {str(e)}")
        raise
