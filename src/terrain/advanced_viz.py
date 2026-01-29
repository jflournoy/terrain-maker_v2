"""
Advanced terrain visualization features.

This module provides specialized visualization capabilities for terrain data:
- Drive-time isochrone curves (3D transportation analysis)
- Slope calculation using Horn's method
- 3D legend generation for Blender scenes

Migrated from legacy helpers.py with improvements.
"""

import logging
import numpy as np
import geopandas as gpd
import shapely
import shapely.geometry
from shapely.validation import make_valid
import bpy
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm import tqdm

logger = logging.getLogger(__name__)


def horn_slope(dem, window_size=3):  # pylint: disable=unused-argument
    """
    Calculate slope using Horn's method with NaN handling (GPU-accelerated).

    Horn's method is a standard GIS technique for calculating terrain slope
    using a 3x3 Sobel-like kernel. This implementation properly handles NaN
    values through interpolation.

    Uses PyTorch GPU acceleration when available (7x speedup on CUDA).

    Args:
        dem (np.ndarray): Input DEM array (2D)
        window_size (int): Reserved for future use (currently fixed at 3)

    Returns:
        np.ndarray: Slope magnitude array (same shape as input)

    Examples:
        >>> dem = np.random.rand(100, 100) * 1000  # Random elevation
        >>> slopes = horn_slope(dem)
        >>> print(f"Slope range: {slopes.min():.2f} to {slopes.max():.2f}")
    """
    from src.terrain.gpu_ops import gpu_horn_slope

    logger.info("Computing Horn slope for DEM shape: %s", dem.shape)
    logger.info("Input DEM value range: %.2f to %.2f", np.nanmin(dem), np.nanmax(dem))
    logger.info("Input NaN count: %d", np.sum(np.isnan(dem)))

    slope = gpu_horn_slope(dem)

    logger.info("Output slope value range: %.2f to %.2f", np.nanmin(slope), np.nanmax(slope))
    logger.info("Output NaN count: %d", np.sum(np.isnan(slope)))

    return slope


def create_values_legend(  # pylint: disable=unused-argument
    terrain_obj,
    values,
    mpp=30,
    *,
    colormap_name="mako_r",
    n_samples=10,
    label="Value",
    units="",
    scale=0.2,
    position_offset=(5, 0, 0),
):
    """
    Create a 3D legend bar in the Blender scene.

    Generates a vertical bar with color gradient and text labels showing
    the value scale for the terrain visualization.

    Args:
        terrain_obj: Blender terrain mesh object (for positioning reference)
        values (np.ndarray): Value array to create legend for
        mpp (float): Meters per pixel (default: 30)
        colormap_name (str): Matplotlib colormap name (default: 'mako_r')
        n_samples (int): Number of labels on legend (default: 10)
        label (str): Legend title (default: 'Value')
        units (str): Unit string (e.g., 'meters', 'mm') (default: '')
        scale (float): Legend bar scale factor (default: 0.2)
        position_offset (tuple): (x, y, z) offset from terrain (default: (5, 0, 0))

    Returns:
        tuple: (legend_object, text_objects_list)

    Examples:
        >>> legend_obj, labels = create_values_legend(
        ...     terrain_mesh, elevation_data,
        ...     label='Elevation', units='meters', n_samples=5
        ... )
        >>> print(f"Created legend with {len(labels)} labels")
    """
    logger.info("Creating legend for %s with %d samples", label, n_samples)

    # Get value samples for labels
    valid_values = values[~np.isnan(values)]
    percentiles = np.linspace(5, 95, n_samples)
    samples = np.percentile(valid_values, percentiles)

    # Create the legend bar mesh
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
    legend_obj = bpy.context.active_object
    legend_obj.name = f"{label}_Legend"

    # Scale it to be a vertical bar
    legend_obj.scale = (scale, scale, scale * 5)

    # Position it relative to terrain
    bounds = terrain_obj.bound_box
    terrain_width = max(b[0] for b in bounds) - min(b[0] for b in bounds)

    legend_obj.location = (
        terrain_obj.location.x + terrain_width / 2 + position_offset[0],
        terrain_obj.location.y + position_offset[1],
        terrain_obj.location.z + position_offset[2],
    )

    # Apply colormap material
    mat = bpy.data.materials.new(name=f"{label}_Legend_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    # Create nodes for vertical gradient
    texture_coord = nodes.new("ShaderNodeTexCoord")
    separate_xyz = nodes.new("ShaderNodeSeparateXYZ")
    color_ramp = nodes.new("ShaderNodeValToRGB")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    output = nodes.new("ShaderNodeOutputMaterial")

    # Set up gradient colors from colormap
    cmap = plt.colormaps.get_cmap(colormap_name)
    for i, sample in enumerate(samples):
        if i < len(color_ramp.color_ramp.elements):
            color_ramp.color_ramp.elements[i].position = i / (len(samples) - 1)
            color = cmap(i / (len(samples) - 1))
            color_ramp.color_ramp.elements[i].color = (*color[:3], 1.0)
        else:
            elem = color_ramp.color_ramp.elements.new(i / (len(samples) - 1))
            color = cmap(i / (len(samples) - 1))
            elem.color = (*color[:3], 1.0)

    # Connect nodes
    mat.node_tree.links.new(texture_coord.outputs["Object"], separate_xyz.inputs[0])
    mat.node_tree.links.new(separate_xyz.outputs["Z"], color_ramp.inputs[0])
    mat.node_tree.links.new(color_ramp.outputs["Color"], principled.inputs["Base Color"])
    mat.node_tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])

    # Assign material
    legend_obj.data.materials.append(mat)

    # Create text labels
    text_objects = []
    for i, sample in enumerate(samples):
        z_pos = (i / (len(samples) - 1) - 0.5) * scale * 10

        bpy.ops.object.text_add(
            location=(
                legend_obj.location.x + scale * 1.5,
                legend_obj.location.y,
                legend_obj.location.z + z_pos,
            )
        )
        text_obj = bpy.context.active_object
        text_obj.name = f"{label}_Label_{i}"

        # Set text content
        text_obj.data.body = f"{sample:.1f} {units}"
        text_obj.data.size = scale * 0.5
        text_obj.data.align_x = "LEFT"

        text_objects.append(text_obj)

    logger.info("Created legend with %d labels", len(text_objects))
    return legend_obj, text_objects
