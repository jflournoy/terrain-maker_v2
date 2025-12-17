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
        output = nodes.new("ShaderNodeOutputMaterial")
        principled = nodes.new("ShaderNodeBsdfPrincipled")
        emission = nodes.new("ShaderNodeEmission")
        mix_shader = nodes.new("ShaderNodeMixShader")
        vertex_color = nodes.new("ShaderNodeVertexColor")

        # Position nodes
        output.location = (600, 300)
        mix_shader.location = (400, 300)
        principled.location = (200, 400)
        emission.location = (200, 200)
        vertex_color.location = (0, 300)

        # Set vertex color layer
        vertex_color.layer_name = "TerrainColors"

        # Configure principled shader for reflectance
        principled.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1.0)
        principled.inputs["Roughness"].default_value = 0.8

        # Configure emission to use vertex colors
        # Use vertex color directly as emission color with moderate strength
        emission.inputs["Strength"].default_value = 1.5

        # Create connections
        logger.debug("Creating node connections")
        # Vertex color drives both emission and base color
        links.new(vertex_color.outputs["Color"], emission.inputs["Color"])
        links.new(vertex_color.outputs["Color"], principled.inputs["Base Color"])

        # Mix between principled shader (reflected light) and emission (self-illuminated)
        # Use mostly emission with some reflected light for better color visibility
        links.new(principled.outputs["BSDF"], mix_shader.inputs[1])
        links.new(emission.outputs["Emission"], mix_shader.inputs[2])

        # Set mix factor to favor emission (70% emission, 30% principled)
        # This ensures vertex colors are visible and properly colored
        mix_shader.inputs[0].default_value = 0.3

        # Connect to output
        links.new(mix_shader.outputs["Shader"], output.inputs["Surface"])

        logger.info("Material setup completed successfully")

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
        # Create nodes
        output = nodes.new("ShaderNodeOutputMaterial")
        principled = nodes.new("ShaderNodeBsdfPrincipled")
        emission = nodes.new("ShaderNodeEmission")
        mix_shader = nodes.new("ShaderNodeMixShader")
        vertex_color = nodes.new("ShaderNodeVertexColor")
        mix_rgb = nodes.new("ShaderNodeMixRGB")  # Mix colors based on alpha

        # Position nodes
        output.location = (800, 300)
        mix_shader.location = (600, 300)
        mix_rgb.location = (400, 300)
        principled.location = (200, 400)
        emission.location = (200, 200)
        vertex_color.location = (0, 300)

        # Set vertex color layer
        vertex_color.layer_name = "TerrainColors"

        # Configure water color
        water_rgba = (*water_color, 1.0)

        # Configure emission shader
        emission.inputs["Strength"].default_value = 1.5

        # Set principled shader
        principled.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1.0)
        principled.inputs["Roughness"].default_value = 0.6  # Water is slightly smoother

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
        logger.debug("Creating node connections with water shader")

        # Use alpha channel from vertex color to control water/land mixing
        # Alpha > 0.5 = water, Alpha < 0.5 = land
        links.new(vertex_color.outputs["Alpha"], mix_rgb.inputs["Fac"])

        # Use vertex color as the land color input (Color2)
        links.new(vertex_color.outputs["Color"], mix_rgb.inputs["Color2"])

        # Use mixed color for both emission and principled
        links.new(mix_rgb.outputs["Color"], emission.inputs["Color"])
        links.new(mix_rgb.outputs["Color"], principled.inputs["Base Color"])

        # Mix between principled and emission
        links.new(principled.outputs["BSDF"], mix_shader.inputs[1])
        links.new(emission.outputs["Emission"], mix_shader.inputs[2])
        mix_shader.inputs[0].default_value = 0.3

        # Connect to output
        links.new(mix_shader.outputs["Shader"], output.inputs["Surface"])

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
        "emission_strength": 0.35,
        "roughness": 0.0,
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
