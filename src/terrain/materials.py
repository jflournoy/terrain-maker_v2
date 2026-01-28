"""
Material and shader operations for Blender terrain visualization.

This module contains functions for creating and configuring Blender materials,
shaders, and background planes for terrain rendering.
"""

import logging
import numpy as np
from typing import Optional, Dict, Tuple

import bpy

logger = logging.getLogger(__name__)

# =============================================================================
# ROAD COLOR PRESETS
# =============================================================================

# Named color presets for roads (polished stone/mineral appearance, dielectric)
ROAD_COLORS = {
    "obsidian": (0.02, 0.02, 0.02),      # Near-black volcanic glass
    "azurite": (0.04, 0.09, 0.16),       # Deep blue mineral (#0A1628)
    "azurite-light": (0.06, 0.15, 0.25), # Richer azurite (#0F2540)
    "malachite": (0.02, 0.08, 0.05),     # Deep green copper mineral
    "hematite": (0.08, 0.06, 0.06),      # Dark iron red-gray
}

# =============================================================================
# BASE MATERIAL PRESETS
# =============================================================================

# Named material presets for mesh bases (edge extrusion, backgrounds, etc.)
BASE_MATERIALS = {
    "clay": (0.5, 0.48, 0.45),         # Matte gray clay (default)
    "obsidian": (0.02, 0.02, 0.02),    # Dark volcanic glass (dielectric)
    "chrome": (0.9, 0.9, 0.92),        # Metallic chrome
    "plastic": (0.95, 0.95, 0.95),     # Glossy white plastic
    "gold": (1.0, 0.766, 0.336),       # Metallic gold
    "ivory": (0.95, 0.93, 0.88),       # Off-white with warm tone
}

# =============================================================================
# UNIFIED COLOR PRESETS (all colors available everywhere)
# =============================================================================

# Combined color presets from both ROAD_COLORS and BASE_MATERIALS.
# All color-accepting options (--road-color, --edge-base-material, --test-material)
# can use any of these colors.
ALL_COLORS = {**ROAD_COLORS, **BASE_MATERIALS}

# =============================================================================
# TERRAIN MATERIAL PRESETS
# =============================================================================

# Named material presets for terrain surface rendering.
# These control how vertex colors (elevation/score colormaps) are displayed.
# All presets are dielectric (non-metallic) to preserve color accuracy.
TERRAIN_MATERIALS = {
    # === DIELECTRIC PRESETS (preserve color accuracy) ===
    "matte": {
        # Pure diffuse - colors appear exactly as computed, no specular
        # Best for: 3D print preview, technical visualization
        "roughness": 1.0,
        "metallic": 0.0,
        "specular_ior_level": 0.0,
    },
    "eggshell": {
        # Very subtle sheen - almost matte but with slight depth
        # Best for: print-quality renders, subtle topography
        "roughness": 0.85,
        "metallic": 0.0,
        "specular_ior_level": 0.2,
    },
    "satin": {
        # Soft highlights reveal terrain form while preserving color accuracy
        # Best for: general-purpose visualization, presentations
        "roughness": 0.7,
        "metallic": 0.0,
        "specular_ior_level": 0.3,
    },
    "ceramic": {
        # Glossy surface like a ceramic sculpture or museum model
        # Best for: presentation renders, architectural model aesthetic
        "roughness": 0.35,
        "metallic": 0.0,
        "specular_ior_level": 0.5,
        "ior": 1.5,
    },
    "lacquered": {
        # Very glossy like varnished wood relief map
        # Best for: artistic renders, vintage map aesthetic
        "roughness": 0.25,
        "metallic": 0.0,
        "specular_ior_level": 0.5,
        "ior": 1.45,
    },
    "clearcoat": {
        # Matte color underneath glossy clear layer - automotive paint look
        # Best for: premium presentation renders
        "roughness": 0.6,
        "metallic": 0.0,
        "specular_ior_level": 0.3,
        "coat_weight": 1.0,
        "coat_roughness": 0.1,
    },
    "velvet": {
        # Soft fabric-like appearance with gentle edge highlights
        # Best for: organic/natural terrain visualization
        "roughness": 0.8,
        "metallic": 0.0,
        "specular_ior_level": 0.15,
        "sheen_weight": 0.3,
        "sheen_roughness": 0.5,
    },
    # === MATERIAL-STYLE PRESETS (from color materials) ===
    # These apply the shader properties of named materials while preserving vertex colors.
    # Note: metallic materials will shift colors toward metallic reflection behavior.
    "clay": {
        # Matte gray clay look - same as matte but named for consistency
        "roughness": 1.0,
        "metallic": 0.0,
        "specular_ior_level": 0.0,
    },
    "plastic": {
        # Glossy plastic finish
        "roughness": 0.2,
        "metallic": 0.0,
        "specular_ior_level": 0.5,
    },
    "ivory": {
        # Slightly glossy with warm character
        "roughness": 0.3,
        "metallic": 0.0,
        "specular_ior_level": 0.4,
    },
    "obsidian": {
        # Glossy volcanic glass - very smooth dielectric
        "roughness": 0.03,
        "metallic": 0.0,
        "specular_ior_level": 0.5,
        "ior": 1.5,
    },
    "chrome": {
        # Mirror-like metallic finish (will shift vertex colors toward metallic)
        "roughness": 0.05,
        "metallic": 1.0,
        "specular_ior_level": 0.5,
    },
    "gold": {
        # Metallic gold finish (will shift vertex colors toward metallic gold tones)
        "roughness": 0.1,
        "metallic": 1.0,
        "specular_ior_level": 0.5,
    },
    "mineral": {
        # Polished stone/mineral appearance - good generic glossy dielectric
        "roughness": 0.15,
        "metallic": 0.0,
        "specular_ior_level": 0.5,
        "ior": 1.55,
    },
}

# Default terrain material preset
DEFAULT_TERRAIN_MATERIAL = "satin"


# =============================================================================
# HELP TEXT GENERATORS
# =============================================================================


def get_all_colors_help() -> str:
    """Generate help text listing all available color presets (unified)."""
    return ", ".join(ALL_COLORS.keys())


def get_all_colors_choices() -> list[str]:
    """Return list of all valid color preset names for argparse choices."""
    return list(ALL_COLORS.keys())


def get_terrain_materials_help() -> str:
    """Generate help text listing all available terrain material presets."""
    return ", ".join(TERRAIN_MATERIALS.keys())


def get_terrain_materials_choices() -> list[str]:
    """Return list of valid terrain material preset names for argparse choices."""
    return list(TERRAIN_MATERIALS.keys())


# Legacy aliases for backward compatibility
def get_road_colors_help() -> str:
    """Generate help text listing all available color presets."""
    return get_all_colors_help()


def get_base_materials_help() -> str:
    """Generate help text listing all available color presets."""
    return get_all_colors_help()


def get_road_colors_choices() -> list[str]:
    """Return list of all valid color preset names for argparse choices."""
    return get_all_colors_choices()


def get_base_materials_choices() -> list[str]:
    """Return list of all valid color preset names for argparse choices."""
    return get_all_colors_choices()


def get_terrain_material_params(material: str) -> Dict:
    """
    Get terrain material parameters by preset name.

    Args:
        material: Preset name (case-insensitive). One of:
            - "matte": Pure diffuse, no specular (3D print preview)
            - "eggshell": Very subtle sheen (print-quality renders)
            - "satin": Soft highlights (general-purpose, default)
            - "ceramic": Glossy museum model look
            - "lacquered": Very glossy varnished wood look
            - "clearcoat": Glossy clear layer over matte color
            - "velvet": Soft fabric-like with edge highlights

    Returns:
        Dict of Principled BSDF parameters.

    Raises:
        ValueError: If material name is not recognized.

    Examples:
        >>> params = get_terrain_material_params("satin")
        >>> params["roughness"]
        0.7
    """
    material_lower = material.lower()
    if material_lower not in TERRAIN_MATERIALS:
        valid = ", ".join(TERRAIN_MATERIALS.keys())
        raise ValueError(f"Unknown terrain material: {material}. Valid options: {valid}")
    return TERRAIN_MATERIALS[material_lower].copy()


def get_color(color: str | Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Resolve a color preset name to RGB color tuple.

    Accepts either a preset color name (case-insensitive) or an RGB tuple.
    Color names map to predefined RGB values from ALL_COLORS (combined
    road colors and base materials).

    For terrain material presets (like "satin", "matte") that don't have
    associated colors, returns clay gray (0.5, 0.48, 0.45) as a neutral default.

    Args:
        color: Either a preset name from ALL_COLORS or an RGB tuple (0-1 range).
               Available presets: {get_all_colors_help()}

    Returns:
        RGB tuple (0-1 range) representing the color.

    Raises:
        ValueError: If color is a string but not in ALL_COLORS or TERRAIN_MATERIALS.

    Examples:
        >>> get_color("clay")
        (0.5, 0.48, 0.45)

        >>> get_color("azurite")  # Road color preset
        (0.04, 0.09, 0.16)

        >>> get_color("GOLD")  # Case-insensitive
        (1.0, 0.766, 0.336)

        >>> get_color("satin")  # Terrain material - returns clay gray
        (0.5, 0.48, 0.45)

        >>> get_color((0.6, 0.55, 0.5))  # Custom RGB
        (0.6, 0.55, 0.5)
    """
    if isinstance(color, str):
        color_lower = color.lower()
        if color_lower in ALL_COLORS:
            return ALL_COLORS[color_lower]
        # For terrain material presets without colors (satin, matte, etc.),
        # return clay gray as neutral default
        if color_lower in TERRAIN_MATERIALS:
            return ALL_COLORS["clay"]  # (0.5, 0.48, 0.45)
        raise ValueError(
            f"Unknown color preset: {color}. "
            f"Valid options: {get_all_colors_help()}"
        )
    else:
        # Assume it's already an RGB tuple - pass through
        return color


# Legacy alias for backward compatibility
def get_base_material_color(material: str | Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Resolve a color preset name to RGB. Alias for get_color()."""
    return get_color(material)


def apply_colormap_material(
    material: bpy.types.Material,
    terrain_material: Optional[str] = None,
) -> None:
    """
    Create a physically-based material for terrain visualization using vertex colors.

    Uses pure Principled BSDF for proper lighting response - no emission.
    Terrain responds realistically to sun direction and casts proper shadows.

    Args:
        material: Blender material to configure
        terrain_material: Optional preset name for material appearance. One of:
            - "matte": Pure diffuse, no specular (3D print preview)
            - "eggshell": Very subtle sheen (print-quality renders)
            - "satin": Soft highlights (general-purpose, recommended)
            - "ceramic": Glossy museum model look
            - "lacquered": Very glossy varnished wood look
            - "clearcoat": Glossy clear layer over matte color
            - "velvet": Soft fabric-like with edge highlights
            If None, uses DEFAULT_TERRAIN_MATERIAL ("satin").
    """
    # Get material parameters from preset
    preset_name = terrain_material or DEFAULT_TERRAIN_MATERIAL
    try:
        params = get_terrain_material_params(preset_name)
    except ValueError:
        logger.warning(f"Unknown terrain material '{terrain_material}', using {DEFAULT_TERRAIN_MATERIAL}")
        params = get_terrain_material_params(DEFAULT_TERRAIN_MATERIAL)
        preset_name = DEFAULT_TERRAIN_MATERIAL

    logger.info(f"Setting up {preset_name} material nodes for {material.name}")

    # Clear existing nodes
    material.node_tree.nodes.clear()
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    try:
        # Create shader nodes - pure Principled BSDF (no emission)
        output = nodes.new("ShaderNodeOutputMaterial")
        principled = nodes.new("ShaderNodeBsdfPrincipled")
        vertex_color = nodes.new("ShaderNodeVertexColor")

        # Position nodes
        output.location = (400, 300)
        principled.location = (200, 300)
        vertex_color.location = (0, 300)

        # Set vertex color layer
        vertex_color.layer_name = "TerrainColors"

        # Apply material parameters from preset
        principled.inputs["Roughness"].default_value = params["roughness"]
        principled.inputs["Metallic"].default_value = params["metallic"]
        principled.inputs["Specular IOR Level"].default_value = params["specular_ior_level"]

        # Optional parameters
        if "ior" in params:
            principled.inputs["IOR"].default_value = params["ior"]
        if "coat_weight" in params:
            principled.inputs["Coat Weight"].default_value = params["coat_weight"]
        if "coat_roughness" in params:
            principled.inputs["Coat Roughness"].default_value = params["coat_roughness"]
        if "sheen_weight" in params:
            principled.inputs["Sheen Weight"].default_value = params["sheen_weight"]
        if "sheen_roughness" in params:
            principled.inputs["Sheen Roughness"].default_value = params["sheen_roughness"]

        # Create connections
        logger.debug("Creating node connections")
        # Vertex color drives base color directly
        links.new(vertex_color.outputs["Color"], principled.inputs["Base Color"])

        # Connect to output
        links.new(principled.outputs["BSDF"], output.inputs["Surface"])

        logger.info(f"Material setup completed: {preset_name} (roughness={params['roughness']})")

    except Exception as e:
        logger.error(f"Error setting up material: {str(e)}")
        raise


def apply_water_shader(
    material: bpy.types.Material,
    water_color: Tuple[float, float, float] = (0.0, 0.153, 0.298),
    terrain_material: Optional[str] = None,
) -> None:
    """
    Apply water shader to material, coloring water areas based on vertex alpha channel.
    Uses alpha channel to mix between water color and elevation colors.
    Water pixels (alpha=1.0) render as water color; land pixels (alpha=0.0) show elevation colors.

    Args:
        material: Blender material to configure
        water_color: RGB tuple for water (default: University of Michigan blue #00274C)
        terrain_material: Optional preset name for land material appearance.
            See apply_colormap_material() for available presets.
            If None, uses DEFAULT_TERRAIN_MATERIAL.
    """
    # Get material parameters from preset
    preset_name = terrain_material or DEFAULT_TERRAIN_MATERIAL
    try:
        params = get_terrain_material_params(preset_name)
    except ValueError:
        logger.warning(f"Unknown terrain material '{terrain_material}', using {DEFAULT_TERRAIN_MATERIAL}")
        params = get_terrain_material_params(DEFAULT_TERRAIN_MATERIAL)
        preset_name = DEFAULT_TERRAIN_MATERIAL

    logger.info(f"Setting up water shader ({preset_name} terrain) for {material.name}")

    # Clear existing nodes
    material.node_tree.nodes.clear()
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    try:
        # Create nodes - pure Principled BSDF (no emission)
        output = nodes.new("ShaderNodeOutputMaterial")
        principled = nodes.new("ShaderNodeBsdfPrincipled")
        vertex_color = nodes.new("ShaderNodeVertexColor")
        mix_rgb = nodes.new("ShaderNodeMixRGB")  # Mix colors based on alpha

        # Position nodes
        output.location = (600, 300)
        principled.location = (400, 300)
        mix_rgb.location = (200, 300)
        vertex_color.location = (0, 300)

        # Set vertex color layer
        vertex_color.layer_name = "TerrainColors"

        # Configure water color
        water_rgba = (*water_color, 1.0)

        # Apply terrain material parameters from preset
        principled.inputs["Roughness"].default_value = params["roughness"]
        principled.inputs["Metallic"].default_value = params["metallic"]
        principled.inputs["Specular IOR Level"].default_value = params["specular_ior_level"]

        # Optional parameters
        if "ior" in params:
            principled.inputs["IOR"].default_value = params["ior"]
        if "coat_weight" in params:
            principled.inputs["Coat Weight"].default_value = params["coat_weight"]
        if "coat_roughness" in params:
            principled.inputs["Coat Roughness"].default_value = params["coat_roughness"]
        if "sheen_weight" in params:
            principled.inputs["Sheen Weight"].default_value = params["sheen_weight"]
        if "sheen_roughness" in params:
            principled.inputs["Sheen Roughness"].default_value = params["sheen_roughness"]

        # Configure mix RGB for blending land and water colors
        # Factor from alpha: 0 = land (use vertex color), 1 = water (use water color)
        mix_rgb.inputs["Color1"].default_value = water_rgba  # Water color
        mix_rgb.inputs["Color2"].default_value = (
            0.5,
            0.5,
            0.5,
            1.0,
        )  # Default (overridden by vertex color)

        # Create connections
        logger.debug("Creating node connections with water shader (no emission)")

        # Use alpha channel from vertex color to control water/land mixing
        # Alpha > 0.5 = water, Alpha < 0.5 = land
        links.new(vertex_color.outputs["Alpha"], mix_rgb.inputs["Fac"])

        # Use vertex color as the land color input (Color2)
        links.new(vertex_color.outputs["Color"], mix_rgb.inputs["Color2"])

        # Use mixed color for principled shader base color
        links.new(mix_rgb.outputs["Color"], principled.inputs["Base Color"])

        # Connect directly to output (no emission mixing)
        links.new(principled.outputs["BSDF"], output.inputs["Surface"])

        logger.info(f"Water shader setup completed: {preset_name} terrain")

    except Exception as e:
        logger.error(f"Error setting up water shader: {str(e)}")
        raise


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
    logger.info("Creating background plane...")

    if terrain_obj is None:
        raise ValueError("Terrain object cannot be None")

    default_material_params = {
        "base_color": (1, 1, 1, 1),
        "emission_color": (1, 1, 1, 1),
        "emission_strength": 0.1,  # Subtle fill light
        "roughness": 0.5,  # Mid roughness - diffuse reflection
        "metallic": 0.0,
        "ior": 1.0,
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
                (terrain_center[0] - half_size, terrain_center[1] + half_size, depth),
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
            output = nodes.new("ShaderNodeOutputMaterial")
            principled = nodes.new("ShaderNodeBsdfPrincipled")

            # Configure material properties using merged params
            principled.inputs["Base Color"].default_value = params["base_color"]
            principled.inputs["Emission Color"].default_value = params["emission_color"]
            principled.inputs["Emission Strength"].default_value = params["emission_strength"]
            principled.inputs["Roughness"].default_value = params["roughness"]
            principled.inputs["Metallic"].default_value = params["metallic"]
            principled.inputs["IOR"].default_value = params["ior"]

            # Position nodes
            output.location = (300, 0)
            principled.location = (0, 0)

            # Connect nodes
            mat.node_tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])

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


def apply_terrain_with_obsidian_roads(
    material: bpy.types.Material,
    terrain_style: Optional[str] = None,
    road_color: str | Tuple[float, float, float] = "obsidian",
    terrain_material: Optional[str] = None,
) -> None:
    """
    Create a material with glossy roads and terrain colors/test material.

    Reads from two vertex color layers:
    - "TerrainColors": The actual terrain colors (used for non-road areas)
    - "RoadMask": R channel marks road pixels (R > 0.5 = road)

    Roads render with glossy dielectric properties (like polished stone).
    Non-road areas use either vertex colors or a test material.

    Args:
        material: Blender material to configure
        terrain_style: Optional test material for terrain ("chrome", "clay", etc.)
                      If None, uses vertex colors with terrain_material preset.
        road_color: Road color - either a preset name from ROAD_COLORS
                   ("obsidian", "azurite", "azurite-light", "malachite", "hematite")
                   or an RGB tuple (0-1 range). Default: "obsidian" (near-black).
        terrain_material: Optional preset name for terrain material appearance when
                         using vertex colors (terrain_style=None). One of:
                         "matte", "eggshell", "satin", "ceramic", "lacquered",
                         "clearcoat", "velvet". Default: DEFAULT_TERRAIN_MATERIAL.
    """
    # Resolve road color (accepts any color from ALL_COLORS or RGB tuple)
    if isinstance(road_color, str):
        if road_color.lower() in ALL_COLORS:
            road_rgb = ALL_COLORS[road_color.lower()]
            road_color_name = road_color.lower()
        else:
            logger.warning(f"Unknown color preset '{road_color}', using obsidian")
            road_rgb = ALL_COLORS["obsidian"]
            road_color_name = "obsidian"
    else:
        road_rgb = road_color
        road_color_name = f"RGB{road_color}"

    logger.info(f"Setting up terrain + {road_color_name} roads material for {material.name}")
    if terrain_style:
        logger.info(f"  Terrain style: {terrain_style}")
    else:
        logger.info("  Terrain style: vertex colors (score-based)")

    material.node_tree.nodes.clear()
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    try:
        # === OUTPUT ===
        output = nodes.new("ShaderNodeOutputMaterial")
        output.location = (800, 0)

        # === ROAD MASK INPUT ===
        road_mask = nodes.new("ShaderNodeVertexColor")
        road_mask.layer_name = "RoadMask"
        road_mask.location = (-400, -200)

        # Separate RGB to extract just the R channel for mix factor
        # (Color output averages RGB → ~0.33 for R=1,G=0,B=0, we need 1.0)
        separate_rgb = nodes.new("ShaderNodeSeparateColor")
        separate_rgb.location = (-200, -200)

        # === MIXER (roads vs terrain) ===
        mix_shader = nodes.new("ShaderNodeMixShader")
        mix_shader.location = (600, 0)

        # === ROAD SHADER ===
        # Use material properties from TERRAIN_MATERIALS if available, otherwise default to mineral
        road_shader = nodes.new("ShaderNodeBsdfPrincipled")
        road_shader.location = (200, -200)
        road_shader.inputs["Base Color"].default_value = (*road_rgb, 1.0)

        # Get shader properties from TERRAIN_MATERIALS if the color has a matching material preset
        if road_color_name in TERRAIN_MATERIALS:
            road_params = TERRAIN_MATERIALS[road_color_name]
            road_shader.inputs["Roughness"].default_value = road_params["roughness"]
            road_shader.inputs["Metallic"].default_value = road_params["metallic"]
            road_shader.inputs["Specular IOR Level"].default_value = road_params["specular_ior_level"]
            if "ior" in road_params:
                road_shader.inputs["IOR"].default_value = road_params["ior"]
            logger.info(f"  Road material: {road_color_name} (metallic={road_params['metallic']})")
        else:
            # Default: polished mineral appearance for colors without specific material presets
            road_shader.inputs["Roughness"].default_value = 0.15  # Polished stone finish
            road_shader.inputs["Metallic"].default_value = 0.0    # Dielectric
            road_shader.inputs["IOR"].default_value = 1.55        # Mineral-like
            road_shader.inputs["Specular IOR Level"].default_value = 0.6  # Good reflectivity
            logger.info(f"  Road material: mineral (default)")

        # === TERRAIN SHADER ===
        if terrain_style and terrain_style.lower() != "none":
            # Use test material for terrain
            terrain_shader = nodes.new("ShaderNodeBsdfPrincipled")
            terrain_shader.location = (200, 200)
            _configure_principled_for_style(terrain_shader, terrain_style)
        else:
            # Use vertex colors with terrain material preset
            terrain_colors = nodes.new("ShaderNodeVertexColor")
            terrain_colors.layer_name = "TerrainColors"
            terrain_colors.location = (-400, 200)

            # Get terrain material parameters from preset
            preset_name = terrain_material or DEFAULT_TERRAIN_MATERIAL
            try:
                params = get_terrain_material_params(preset_name)
            except ValueError:
                logger.warning(f"Unknown terrain material '{terrain_material}', using {DEFAULT_TERRAIN_MATERIAL}")
                params = get_terrain_material_params(DEFAULT_TERRAIN_MATERIAL)
                preset_name = DEFAULT_TERRAIN_MATERIAL

            # Principled BSDF with terrain material preset
            terrain_principled = nodes.new("ShaderNodeBsdfPrincipled")
            terrain_principled.location = (0, 300)
            terrain_principled.inputs["Roughness"].default_value = params["roughness"]
            terrain_principled.inputs["Metallic"].default_value = params["metallic"]
            terrain_principled.inputs["Specular IOR Level"].default_value = params["specular_ior_level"]

            # Optional parameters
            if "ior" in params:
                terrain_principled.inputs["IOR"].default_value = params["ior"]
            if "coat_weight" in params:
                terrain_principled.inputs["Coat Weight"].default_value = params["coat_weight"]
            if "coat_roughness" in params:
                terrain_principled.inputs["Coat Roughness"].default_value = params["coat_roughness"]
            if "sheen_weight" in params:
                terrain_principled.inputs["Sheen Weight"].default_value = params["sheen_weight"]
            if "sheen_roughness" in params:
                terrain_principled.inputs["Sheen Roughness"].default_value = params["sheen_roughness"]

            # Connect terrain vertex colors directly to principled shader
            links.new(terrain_colors.outputs["Color"], terrain_principled.inputs["Base Color"])

            terrain_shader = terrain_principled
            logger.info(f"  Terrain material: {preset_name}")

        # === CONNECT EVERYTHING ===
        # Road mask R channel controls mixing (R=1 → road, R=0 → terrain)
        links.new(road_mask.outputs["Color"], separate_rgb.inputs["Color"])
        links.new(separate_rgb.outputs["Red"], mix_shader.inputs[0])  # Use R channel as mix factor

        # Input 1 = terrain (when mask is 0)
        # Input 2 = road (when mask is 1)
        links.new(terrain_shader.outputs["BSDF"], mix_shader.inputs[1])
        links.new(road_shader.outputs["BSDF"], mix_shader.inputs[2])

        # Connect to output
        links.new(mix_shader.outputs["Shader"], output.inputs["Surface"])

        logger.info(f"✓ Terrain + {road_color_name} roads material applied")

    except Exception as e:
        logger.error(f"Error setting up terrain + roads material: {str(e)}")
        raise


def _configure_principled_for_style(shader_node, style: str) -> None:
    """
    Configure a Principled BSDF node for a specific test material style.

    Supports all colors from ALL_COLORS. Some materials have special shader
    settings (chrome/gold are metallic, obsidian is glossy glass, etc.),
    while others use the color with standard dielectric settings.

    Args:
        shader_node: Blender Principled BSDF shader node to configure.
        style: Material/color name (case-insensitive) from ALL_COLORS.
    """
    style_lower = style.lower()

    # Materials with special shader settings
    if style_lower == "obsidian":
        # Obsidian is volcanic glass (dielectric), NOT metal
        # Reflects via Fresnel like glass, dark from iron/magnesium impurities
        shader_node.inputs["Base Color"].default_value = (0.02, 0.02, 0.02, 1.0)
        shader_node.inputs["Roughness"].default_value = 0.03  # Very smooth but not perfect mirror
        shader_node.inputs["Metallic"].default_value = 0.0    # Glass, not metal
        shader_node.inputs["IOR"].default_value = 1.5         # Glass-like
        shader_node.inputs["Specular IOR Level"].default_value = 0.5  # Standard dielectric
    elif style_lower == "chrome":
        shader_node.inputs["Base Color"].default_value = (0.9, 0.9, 0.92, 1.0)
        shader_node.inputs["Roughness"].default_value = 0.05
        shader_node.inputs["Metallic"].default_value = 1.0
    elif style_lower == "clay":
        shader_node.inputs["Base Color"].default_value = (0.5, 0.48, 0.45, 1.0)
        shader_node.inputs["Roughness"].default_value = 1.0
        shader_node.inputs["Metallic"].default_value = 0.0
        shader_node.inputs["Specular IOR Level"].default_value = 0.0
    elif style_lower == "plastic":
        shader_node.inputs["Base Color"].default_value = (0.95, 0.95, 0.95, 1.0)
        shader_node.inputs["Roughness"].default_value = 0.2
        shader_node.inputs["Metallic"].default_value = 0.0
        shader_node.inputs["Specular IOR Level"].default_value = 0.5
    elif style_lower == "gold":
        shader_node.inputs["Base Color"].default_value = (1.0, 0.766, 0.336, 1.0)
        shader_node.inputs["Roughness"].default_value = 0.1
        shader_node.inputs["Metallic"].default_value = 1.0
    elif style_lower == "ivory":
        # Off-white with warm tone, slightly glossy
        shader_node.inputs["Base Color"].default_value = (0.95, 0.93, 0.88, 1.0)
        shader_node.inputs["Roughness"].default_value = 0.3
        shader_node.inputs["Metallic"].default_value = 0.0
        shader_node.inputs["Specular IOR Level"].default_value = 0.4
    elif style_lower in ALL_COLORS:
        # Use color from ALL_COLORS with polished stone/mineral appearance
        # (dielectric, slightly glossy - good for road colors like azurite, malachite)
        rgb = ALL_COLORS[style_lower]
        shader_node.inputs["Base Color"].default_value = (*rgb, 1.0)
        shader_node.inputs["Roughness"].default_value = 0.15  # Polished finish
        shader_node.inputs["Metallic"].default_value = 0.0    # Dielectric
        shader_node.inputs["IOR"].default_value = 1.55        # Mineral-like
        shader_node.inputs["Specular IOR Level"].default_value = 0.5
    else:
        # Unknown style - default to gray
        shader_node.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1.0)
        shader_node.inputs["Roughness"].default_value = 0.5


# Keep for backwards compatibility
def apply_glassy_road_material(material: bpy.types.Material) -> None:
    """Deprecated: Use apply_terrain_with_obsidian_roads() instead."""
    apply_terrain_with_obsidian_roads(material, terrain_style=None)


# =============================================================================
# TEST MATERIALS - For visualization testing without vertex colors
# =============================================================================


def apply_test_material(material: bpy.types.Material, style: str) -> None:
    """
    Apply a test material to the entire terrain mesh.

    Test materials ignore vertex colors and apply a uniform material style
    for testing lighting, shadows, and mesh geometry.

    Args:
        material: Blender material to configure
        style: Material style name - any color from ALL_COLORS
            ({get_all_colors_help()}) or "terrain" for normal vertex colors.

    Raises:
        ValueError: If style is not recognized
    """
    style_lower = style.lower()

    if style_lower == "terrain":
        apply_colormap_material(material)
    elif style_lower in ALL_COLORS:
        _apply_test_material_generic(material, style_lower)
    else:
        raise ValueError(
            f"Unknown test material style: {style}. "
            f"Valid options: {get_all_colors_help()}, terrain"
        )


def _apply_test_material_generic(material: bpy.types.Material, style: str) -> None:
    """Apply a test material using the unified color/style system."""
    logger.info(f"Applying {style} test material to {material.name}")

    material.node_tree.nodes.clear()
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")

    principled.location = (0, 0)
    output.location = (300, 0)

    # Configure shader based on style (handles special cases and generic colors)
    _configure_principled_for_style(principled, style)

    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    logger.info(f"✓ {style.capitalize()} test material applied")


def _apply_test_material_obsidian(material: bpy.types.Material) -> None:
    """Obsidian volcanic glass - dark dielectric with glass-like reflections."""
    logger.info(f"Applying obsidian test material to {material.name}")

    material.node_tree.nodes.clear()
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")

    principled.location = (0, 0)
    output.location = (300, 0)

    # Obsidian is volcanic glass (dielectric), NOT metal
    # Dark color from iron/magnesium impurities, reflects via Fresnel like glass
    principled.inputs["Base Color"].default_value = (0.02, 0.02, 0.02, 1.0)
    principled.inputs["Roughness"].default_value = 0.03  # Very smooth but not perfect mirror
    principled.inputs["Metallic"].default_value = 0.0    # Glass, not metal
    principled.inputs["IOR"].default_value = 1.5         # Glass-like
    principled.inputs["Specular IOR Level"].default_value = 0.5  # Standard dielectric

    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    logger.info("✓ Obsidian test material applied")


def _apply_test_material_chrome(material: bpy.types.Material) -> None:
    """Metallic chrome - highly reflective silver metal."""
    logger.info(f"Applying chrome test material to {material.name}")

    material.node_tree.nodes.clear()
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")

    principled.location = (0, 0)
    output.location = (300, 0)

    # Bright chrome metal
    principled.inputs["Base Color"].default_value = (0.9, 0.9, 0.92, 1.0)
    principled.inputs["Roughness"].default_value = 0.05  # Slight roughness for realism
    principled.inputs["Metallic"].default_value = 1.0

    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    logger.info("✓ Chrome test material applied")


def _apply_test_material_clay(material: bpy.types.Material) -> None:
    """Matte clay - diffuse gray, no reflections, shows form clearly."""
    logger.info(f"Applying clay test material to {material.name}")

    material.node_tree.nodes.clear()
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")

    principled.location = (0, 0)
    output.location = (300, 0)

    # Neutral gray clay
    principled.inputs["Base Color"].default_value = (0.5, 0.48, 0.45, 1.0)
    principled.inputs["Roughness"].default_value = 1.0  # Fully matte
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Specular IOR Level"].default_value = 0.0  # No specular

    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    logger.info("✓ Clay test material applied")


def _apply_test_material_plastic(material: bpy.types.Material) -> None:
    """Glossy white plastic - shows highlights and shadows well."""
    logger.info(f"Applying plastic test material to {material.name}")

    material.node_tree.nodes.clear()
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")

    principled.location = (0, 0)
    output.location = (300, 0)

    # White glossy plastic
    principled.inputs["Base Color"].default_value = (0.95, 0.95, 0.95, 1.0)
    principled.inputs["Roughness"].default_value = 0.2  # Smooth but not mirror
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Specular IOR Level"].default_value = 0.5

    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    logger.info("✓ Plastic test material applied")


def _apply_test_material_gold(material: bpy.types.Material) -> None:
    """Metallic gold - warm reflective metal."""
    logger.info(f"Applying gold test material to {material.name}")

    material.node_tree.nodes.clear()
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")

    principled.location = (0, 0)
    output.location = (300, 0)

    # Gold metal color
    principled.inputs["Base Color"].default_value = (1.0, 0.766, 0.336, 1.0)
    principled.inputs["Roughness"].default_value = 0.1
    principled.inputs["Metallic"].default_value = 1.0

    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    logger.info("✓ Gold test material applied")


def _apply_test_material_ivory(material: bpy.types.Material) -> None:
    """Ivory - off-white with warm tone, slightly glossy."""
    logger.info(f"Applying ivory test material to {material.name}")

    material.node_tree.nodes.clear()
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")

    principled.location = (0, 0)
    output.location = (300, 0)

    # Ivory - warm off-white
    principled.inputs["Base Color"].default_value = (0.95, 0.93, 0.88, 1.0)
    principled.inputs["Roughness"].default_value = 0.3  # Slightly glossy
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Specular IOR Level"].default_value = 0.4

    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    logger.info("✓ Ivory test material applied")
