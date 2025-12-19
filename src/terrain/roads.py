"""
Road network visualization for terrain rendering.

This module provides functions to render road networks as colored lines on terrain,
with colors sampled from the terrain colormap and blended to maintain visibility.

Usage:
    from src.terrain.roads import add_roads_to_scene
    from examples.detroit_roads import get_roads

    # Fetch road data
    bbox = (42.3, -83.5, 42.5, -82.8)
    roads_geojson = get_roads(bbox)

    # Add roads to Blender scene
    add_roads_to_scene(terrain, roads_geojson, color_blend_factor=0.7)
"""

import logging
import numpy as np
from typing import Dict, Any, Optional

import bpy

logger = logging.getLogger(__name__)


# =============================================================================
# COLOR OPERATIONS
# =============================================================================


def blend_color_for_road(
    terrain_color_rgb: tuple,
    darken_factor: float = 0.7,
) -> tuple:
    """
    Darken a terrain color for road overlay visibility.

    Args:
        terrain_color_rgb: (R, G, B) tuple with values 0-1
        darken_factor: 0.7 = 30% darker, 0.5 = 50% darker, 0.9 = 10% darker

    Returns:
        (R, G, B) darkened color tuple
    """
    if len(terrain_color_rgb) >= 3:
        return tuple(c * darken_factor for c in terrain_color_rgb[:3])
    return terrain_color_rgb


def sample_color_at_mesh_coord(
    terrain,
    mesh_x: float,
    mesh_y: float,
) -> Optional[tuple]:
    """
    Sample terrain color at a mesh coordinate.

    Args:
        terrain: Terrain object with computed colors
        mesh_x: X coordinate in mesh space
        mesh_y: Y coordinate in mesh space

    Returns:
        (R, G, B) color tuple normalized to 0-1, or None if out of bounds
    """
    if not hasattr(terrain, "colors") or terrain.colors is None:
        logger.warning("Terrain colors not computed, using white")
        return (1.0, 1.0, 1.0)

    # Get the mesh vertex indices
    if not hasattr(terrain, "vertices") or terrain.vertices is None:
        return None

    # Find closest vertex to mesh coordinate
    vertices = terrain.vertices
    distances = np.sqrt((vertices[:, 0] - mesh_x) ** 2 + (vertices[:, 1] - mesh_y) ** 2)
    closest_idx = np.argmin(distances)

    # Get color from closest vertex
    if closest_idx < len(terrain.colors):
        color_rgba = terrain.colors[closest_idx]
        # Convert from 0-1 or 0-255 range
        if color_rgba[0] > 1.0:
            color_rgb = tuple(c / 255.0 for c in color_rgba[:3])
        else:
            color_rgb = tuple(color_rgba[:3])
        return color_rgb

    return None


# =============================================================================
# ROAD RENDERING
# =============================================================================


def create_road_curve(
    road_feature: Dict[str, Any],
    terrain,
    color_blend_factor: float = 0.7,
    curve_thickness: float = 3.0,
) -> Optional[bpy.types.Object]:
    """
    Create a Blender curve object for a single road.

    Args:
        road_feature: GeoJSON Feature with LineString geometry
        terrain: Terrain object with mesh and colors
        color_blend_factor: How much to darken road colors (0.5-0.9)
        curve_thickness: Bevel depth for road visibility

    Returns:
        Blender curve object, or None if failed
    """
    try:
        # Extract road geometry
        geometry = road_feature.get("geometry", {})
        if geometry.get("type") != "LineString":
            return None

        coordinates = geometry.get("coordinates", [])
        if len(coordinates) < 2:
            return None

        properties = road_feature.get("properties", {})
        road_name = properties.get("name", "road")
        road_id = properties.get("osm_id", 0)

        # Convert WGS84 to mesh coordinates and sample colors
        mesh_coords = []
        road_colors = []

        for lon, lat in coordinates:
            try:
                # Transform to mesh coordinates
                mesh_x, mesh_y, mesh_z = terrain.geo_to_mesh_coords([lon], [lat])
                mesh_x = mesh_x[0]
                mesh_y = mesh_y[0]

                # Sample terrain color at this point
                terrain_color = sample_color_at_mesh_coord(terrain, mesh_x, mesh_y)
                if terrain_color is None:
                    terrain_color = (1.0, 1.0, 1.0)

                # Darken color for road visibility
                road_color = blend_color_for_road(terrain_color, color_blend_factor)

                mesh_coords.append((mesh_x, mesh_y, mesh_z))
                road_colors.append(road_color)

            except Exception as e:
                logger.debug(f"Failed to process coordinate ({lon}, {lat}): {e}")
                continue

        if len(mesh_coords) < 2:
            return None

        # Create curve data
        curve_name = f"Road_{road_id}"
        curve_data = bpy.data.curves.new(curve_name, "CURVE")
        curve_data.dimensions = "3D"
        curve_data.resolution_u = 12
        curve_data.bevel_depth = curve_thickness
        curve_data.use_path = False

        # Create curve object
        curve_obj = bpy.data.objects.new(curve_name, curve_data)

        # Create spline
        spline = curve_data.splines.new("POLY")
        spline.points.add(len(mesh_coords) - 1)

        # Add coordinates to spline
        for point, coord in zip(spline.points, mesh_coords):
            point.co = (*coord, 1.0)

        # Create material with road color
        if len(road_colors) > 0:
            avg_color = tuple(np.mean(road_colors, axis=0))
            mat = create_road_material(f"RoadMat_{road_id}", avg_color)
            curve_obj.data.materials.append(mat)

        # Link to scene
        bpy.context.scene.collection.objects.link(curve_obj)

        logger.debug(f"Created road curve: {curve_name} with {len(mesh_coords)} points")
        return curve_obj

    except Exception as e:
        logger.warning(f"Failed to create curve for road {road_id}: {e}")
        return None


def create_road_material(
    material_name: str,
    color_rgb: tuple,
    emission_strength: float = 0.15,
) -> bpy.types.Material:
    """
    Create a Blender material for road visualization.

    Args:
        material_name: Name for the material
        color_rgb: (R, G, B) color tuple with values 0-1
        emission_strength: Emission strength (0.0-1.0)

    Returns:
        Blender material object
    """
    mat = bpy.data.materials.new(material_name)
    mat.use_nodes = True

    # Clear default nodes
    mat.node_tree.nodes.clear()

    # Create new nodes
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Add Principled BSDF
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*color_rgb, 1.0)

    # Try to set optional emission properties (compatible with different Blender versions)
    try:
        bsdf.inputs["Emission"].default_value = (*color_rgb, 1.0)
    except (KeyError, RuntimeError):
        pass  # Emission input not available in this Blender version

    try:
        bsdf.inputs["Emission Strength"].default_value = emission_strength
    except (KeyError, RuntimeError):
        pass  # Emission Strength input not available in this Blender version

    # Set other properties if available
    try:
        bsdf.inputs["Metallic"].default_value = 0.0
    except (KeyError, RuntimeError):
        pass

    try:
        bsdf.inputs["Roughness"].default_value = 0.4
    except (KeyError, RuntimeError):
        pass

    # Add output node
    output = nodes.new(type="ShaderNodeOutputMaterial")

    # Connect
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    return mat


# =============================================================================
# PUBLIC API
# =============================================================================


def add_roads_to_scene(
    terrain,
    roads_geojson: Dict[str, Any],
    color_blend_factor: float = 0.7,
    curve_thickness: float = 3.0,
) -> list[bpy.types.Object]:
    """
    Add road network to Blender scene as colored curves.

    Roads are rendered as Bezier curves with colors sampled from the terrain
    colormap and darkened by the blend factor for visibility.

    Args:
        terrain: Terrain object with mesh, vertices, and computed colors
        roads_geojson: GeoJSON FeatureCollection with road LineStrings
        color_blend_factor: Darkening factor for road colors (0.5-0.9, default 0.7)
        curve_thickness: Bevel depth for road curves (default 3.0 mesh units)

    Returns:
        List of created Blender curve objects

    Raises:
        ValueError: If terrain doesn't have mesh or colors computed
    """
    if not hasattr(terrain, "vertices") or terrain.vertices is None:
        raise ValueError("Terrain mesh not created. Call create_mesh() first.")

    if not hasattr(terrain, "colors") or terrain.colors is None:
        raise ValueError("Terrain colors not computed. Call compute_colors() first.")

    if not hasattr(terrain, "geo_to_mesh_coords"):
        raise ValueError("Terrain missing geo_to_mesh_coords method.")

    logger.info("Adding roads to scene...")

    # Validate blend factor
    if not 0.5 <= color_blend_factor <= 0.9:
        logger.warning(
            f"color_blend_factor {color_blend_factor} outside recommended range [0.5, 0.9], "
            "clamping to valid range"
        )
        color_blend_factor = max(0.5, min(0.9, color_blend_factor))

    road_curves = []

    features = roads_geojson.get("features", [])
    logger.info(f"  Processing {len(features)} road segments...")

    for idx, feature in enumerate(features):
        try:
            curve_obj = create_road_curve(
                feature,
                terrain,
                color_blend_factor=color_blend_factor,
                curve_thickness=curve_thickness,
            )
            if curve_obj is not None:
                road_curves.append(curve_obj)

            if (idx + 1) % 100 == 0:
                logger.debug(f"  Processed {idx + 1}/{len(features)} roads...")

        except Exception as e:
            logger.debug(f"Failed to process road {idx}: {e}")
            continue

    logger.info(f"  Created {len(road_curves)} road curves")

    return road_curves
