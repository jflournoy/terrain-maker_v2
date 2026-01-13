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

# Named color presets for roads (same glossy metallic properties, different colors)
ROAD_COLORS = {
    "obsidian": (0.02, 0.02, 0.02),      # Near-black glass
    "azurite": (0.04, 0.09, 0.16),       # Deep blue mineral (#0A1628)
    "azurite-light": (0.06, 0.15, 0.25), # Richer azurite (#0F2540)
    "malachite": (0.02, 0.08, 0.05),     # Deep green copper mineral
    "hematite": (0.08, 0.06, 0.06),      # Dark iron red-gray
}


def apply_colormap_material(material: bpy.types.Material) -> None:
    """
    Create a physically-based material for terrain visualization using vertex colors.

    Uses pure Principled BSDF for proper lighting response - no emission.
    Terrain responds realistically to sun direction and casts proper shadows.

    Args:
        material: Blender material to configure
    """
    logger.info(f"Setting up material nodes for {material.name}")

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

        # Configure principled shader for natural terrain appearance
        # Roughness 0.8 = mostly matte with subtle highlights
        principled.inputs["Roughness"].default_value = 0.8
        principled.inputs["Metallic"].default_value = 0.0
        principled.inputs["Specular IOR Level"].default_value = 0.3  # Subtle specular

        # Create connections
        logger.debug("Creating node connections")
        # Vertex color drives base color directly
        links.new(vertex_color.outputs["Color"], principled.inputs["Base Color"])

        # Connect to output
        links.new(principled.outputs["BSDF"], output.inputs["Surface"])

        logger.info("Material setup completed successfully (pure Principled BSDF)")

    except Exception as e:
        logger.error(f"Error setting up material: {str(e)}")
        raise


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
    logger.info(f"Setting up water shader for {material.name}")

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

        # Set principled shader - no emission, proper lighting response
        principled.inputs["Roughness"].default_value = 0.6  # Water is slightly smoother
        principled.inputs["Metallic"].default_value = 0.0
        principled.inputs["Specular IOR Level"].default_value = 0.5

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

        logger.info("Water shader setup completed successfully")

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
) -> None:
    """
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
    """
    # Resolve road color
    if isinstance(road_color, str):
        if road_color.lower() in ROAD_COLORS:
            road_rgb = ROAD_COLORS[road_color.lower()]
            road_color_name = road_color.lower()
        else:
            logger.warning(f"Unknown road color preset '{road_color}', using obsidian")
            road_rgb = ROAD_COLORS["obsidian"]
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

        # === MATTE ROAD SHADER (brushed stone appearance) ===
        road_shader = nodes.new("ShaderNodeBsdfPrincipled")
        road_shader.location = (200, -200)
        road_shader.inputs["Base Color"].default_value = (*road_rgb, 1.0)
        road_shader.inputs["Roughness"].default_value = 0.5  # Semi-matte finish
        road_shader.inputs["Metallic"].default_value = 0.8   # Slightly less metallic
        road_shader.inputs["IOR"].default_value = 1.5
        road_shader.inputs["Specular IOR Level"].default_value = 0.5

        # === TERRAIN SHADER ===
        if terrain_style and terrain_style.lower() != "none":
            # Use test material for terrain
            terrain_shader = nodes.new("ShaderNodeBsdfPrincipled")
            terrain_shader.location = (200, 200)
            _configure_principled_for_style(terrain_shader, terrain_style)
        else:
            # Use vertex colors with pure Principled BSDF for terrain
            terrain_colors = nodes.new("ShaderNodeVertexColor")
            terrain_colors.layer_name = "TerrainColors"
            terrain_colors.location = (-400, 200)

            # Pure Principled BSDF - no emission for proper lighting response
            terrain_principled = nodes.new("ShaderNodeBsdfPrincipled")
            terrain_principled.location = (0, 300)
            terrain_principled.inputs["Roughness"].default_value = 0.8
            terrain_principled.inputs["Metallic"].default_value = 0.0
            terrain_principled.inputs["Specular IOR Level"].default_value = 0.3

            # Connect terrain vertex colors directly to principled shader
            links.new(terrain_colors.outputs["Color"], terrain_principled.inputs["Base Color"])

            terrain_shader = terrain_principled

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
    """Configure a Principled BSDF node for a specific test material style."""
    style_lower = style.lower()

    if style_lower == "obsidian":
        shader_node.inputs["Base Color"].default_value = (0.02, 0.02, 0.02, 1.0)
        shader_node.inputs["Roughness"].default_value = 0.0
        shader_node.inputs["Metallic"].default_value = 1.0
        shader_node.inputs["IOR"].default_value = 1.5
        shader_node.inputs["Specular IOR Level"].default_value = 1.0
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
    else:
        # Default to gray
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
        style: Material style name - one of:
            - "obsidian": Glossy black glass (metallic, mirror-smooth)
            - "chrome": Metallic chrome with reflections
            - "clay": Matte gray clay (diffuse, no reflections)
            - "plastic": Glossy white plastic
            - "gold": Metallic gold with warm tones
            - "terrain": Normal terrain with vertex colors (default)

    Raises:
        ValueError: If style is not recognized
    """
    style_lower = style.lower()

    if style_lower == "obsidian":
        _apply_test_material_obsidian(material)
    elif style_lower == "chrome":
        _apply_test_material_chrome(material)
    elif style_lower == "clay":
        _apply_test_material_clay(material)
    elif style_lower == "plastic":
        _apply_test_material_plastic(material)
    elif style_lower == "gold":
        _apply_test_material_gold(material)
    elif style_lower == "terrain":
        apply_colormap_material(material)
    else:
        raise ValueError(
            f"Unknown test material style: {style}. "
            f"Valid options: obsidian, chrome, clay, plastic, gold, terrain"
        )


def _apply_test_material_obsidian(material: bpy.types.Material) -> None:
    """Glossy black obsidian glass - metallic, mirror-smooth."""
    logger.info(f"Applying obsidian test material to {material.name}")

    material.node_tree.nodes.clear()
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")

    principled.location = (0, 0)
    output.location = (300, 0)

    # Dark obsidian glass
    principled.inputs["Base Color"].default_value = (0.02, 0.02, 0.02, 1.0)
    principled.inputs["Roughness"].default_value = 0.0
    principled.inputs["Metallic"].default_value = 1.0
    principled.inputs["IOR"].default_value = 1.5
    principled.inputs["Specular IOR Level"].default_value = 1.0

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
