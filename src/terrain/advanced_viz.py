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
    Calculate slope using Horn's method with NaN handling.

    Horn's method is a standard GIS technique for calculating terrain slope
    using a 3x3 Sobel-like kernel. This implementation properly handles NaN
    values through interpolation.

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
    logger.info("Computing Horn slope for DEM shape: %s", dem.shape)
    logger.info("Input DEM value range: %.2f to %.2f", np.nanmin(dem), np.nanmax(dem))
    logger.info("Input NaN count: %d", np.sum(np.isnan(dem)))

    # Save the original NaN mask
    nan_mask = np.isnan(dem)

    # Fill NaN values with interpolation
    dem_filled = dem.copy()
    mask = np.isnan(dem_filled)
    if np.any(mask):
        dem_filled[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), dem_filled[~mask])

    logger.debug(
        "After filling NaNs - value range: %.2f to %.2f",
        np.nanmin(dem_filled),
        np.nanmax(dem_filled),
    )

    # Calculate gradients using Horn's method (3x3 Sobel-like kernels)
    dx = ndimage.convolve(dem_filled, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0)
    dy = ndimage.convolve(dem_filled, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8.0)

    logger.debug("Gradient ranges - dx: %.2f to %.2f", np.nanmin(dx), np.nanmax(dx))
    logger.debug("Gradient ranges - dy: %.2f to %.2f", np.nanmin(dy), np.nanmax(dy))

    # Calculate slope magnitude
    slope = np.hypot(dx, dy)

    # Restore NaN values to their original locations
    slope[nan_mask] = np.nan

    logger.info("Output slope value range: %.2f to %.2f", np.nanmin(slope), np.nanmax(slope))
    logger.info("Output NaN count: %d", np.sum(np.isnan(slope)))

    return slope


def load_drive_time_data(
    dem_data, utm_transform, meters_per_pixel, buffer_size, simplify_tolerance
):
    """
    Load and process drive-time polygon data for terrain visualization.

    Loads GeoJSON drive-time isochrones, projects them to match DEM coordinates,
    and smooths geometries for clean visualization.

    Args:
        dem_data (np.ndarray): Processed DEM data array
        utm_transform (Affine): Affine transform from UTM to pixel coordinates
        meters_per_pixel (float): Spatial resolution in meters
        buffer_size (float): Buffer size for geometry smoothing (percentage)
        simplify_tolerance (float): Simplification tolerance (percentage)

    Returns:
        geopandas.GeoDataFrame: Processed drive-time polygons in pixel coordinates

    Notes:
        - Expects a file named '1_to_5_hr_drive_times.geojson' in current directory
        - Projects to UTM Zone 17N (EPSG:32617) for Detroit area
        - Applies smoothing via buffer operations

    Examples:
        >>> drive_time = load_drive_time_data(
        ...     dem_data, utm_transform, mpp=30, buffer_size=10, simplify_tolerance=5
        ... )
        >>> print(f"Loaded {len(drive_time)} drive-time zones")
    """
    logger.info("Loading drive time data...")
    drive_time = gpd.read_file("1_to_5_hr_drive_times.geojson")

    # Fix any invalid geometries
    logger.info("Validating geometries...")
    drive_time["geometry"] = drive_time.geometry.apply(make_valid)

    # Project to UTM Zone 17N to match DEM
    logger.info("Projecting to UTM...")
    drive_time = drive_time.to_crs("EPSG:32617")

    # Transform from UTM coordinates to pixel coordinates using the affine transform
    def transform_to_pixels(geom):
        """Transform geometry from UTM coordinates to pixel coordinates.

        Converts UTM coordinates to pixel coordinates using the affine transform.
        Handles both Polygon and MultiPolygon geometries recursively.

        Args:
            geom: Shapely geometry in UTM coordinates (Polygon or MultiPolygon)

        Returns:
            Shapely geometry with coordinates transformed to pixel space
        """
        if geom.geom_type == "MultiPolygon":
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
    drive_time["geometry"] = drive_time.geometry.apply(transform_to_pixels)

    # Smooth the geometries
    buffer_px = meters_per_pixel * buffer_size / 100
    simplify_px = meters_per_pixel * simplify_tolerance / 100

    logger.info("Smoothing geometries...")
    drive_time["geometry"] = (
        drive_time.geometry.simplify(tolerance=simplify_px, preserve_topology=True)
        .buffer(buffer_px, join_style=2, cap_style=3)
        .buffer(-buffer_px, join_style=2, cap_style=3)
        .buffer(buffer_px / 2, join_style=1, cap_style=3)
        .buffer(-buffer_px / 2, join_style=1, cap_style=3)
    )

    logger.info("Processed %d drive time polygons", len(drive_time))
    logger.debug("Drive time bounds: %s", drive_time.total_bounds)

    return drive_time


def create_drive_time_curves(  # pylint: disable=unused-argument
    drive_time, terrain_obj, processed_dem, height_offset=1.0, bevel_depth=0.02
):
    """
    Create 3D glowing curves in Blender representing drive-time isochrones.

    Generates 3D curves that follow the terrain surface with emission shaders
    that glow on specific viewing angles. Great for transportation analysis
    visualization.

    Args:
        drive_time (geopandas.GeoDataFrame): Processed drive-time polygons
        terrain_obj: Blender terrain mesh object (for reference)
        processed_dem (np.ndarray): DEM data for coordinate centering
        height_offset (float): Height above terrain in Blender units (default: 1.0)
        bevel_depth (float): Thickness of the curve (default: 0.02)

    Returns:
        list: List of created Blender curve objects

    Notes:
        - Uses 'inferno' colormap for gradient coloring
        - Applies edge-emission shader for directional glow effect
        - Curves are positioned relative to DEM center

    Examples:
        >>> curves = create_drive_time_curves(
        ...     drive_time, terrain_mesh, dem_data, height_offset=0.5
        ... )
        >>> print(f"Created {len(curves)} curves")
    """
    logger.info("Creating drive time boundary curves...")

    mean_x = processed_dem.shape[1] / 2
    mean_y = processed_dem.shape[0] / 2
    logger.info("DEM coordinate means x,y: %s, %s", mean_x, mean_y)

    curves = []

    # Create inferno colormap for the number of unique polygons
    n_polygons = len(drive_time)
    cmap = plt.colormaps.get_cmap("inferno")

    start = 0.8
    end = 0.1
    colors = cmap(np.linspace(start, end, n_polygons))

    logger.info("Created %d colors from inferno colormap", n_polygons)

    for idx, zone in enumerate(tqdm(drive_time.geometry, desc="Processing drive time zones")):
        # Get color for this polygon
        color = colors[idx]

        # Create material for this specific curve
        mat = bpy.data.materials.new(name=f"DriveTimeMaterial_{idx}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        # Create shader nodes for edge-emission effect
        principled = nodes.new("ShaderNodeBsdfPrincipled")
        geometry = nodes.new("ShaderNodeNewGeometry")
        vector_math = nodes.new("ShaderNodeVectorMath")
        vector_math.operation = "CROSS_PRODUCT"

        # Create vector for UP
        combine_xyz = nodes.new("ShaderNodeCombineXYZ")
        combine_xyz.inputs[2].default_value = 1.0  # Z=1 for UP vector

        # Set up principled shader
        principled.inputs["Base Color"].default_value = (*color[:3], 1.0)
        principled.inputs["Metallic"].default_value = 0.0
        principled.inputs["Roughness"].default_value = 1.0
        principled.inputs["Emission Color"].default_value = (*color[:3], 1.0)
        principled.inputs["Emission Strength"].default_value = 0

        # Calculate right-hand direction using cross product of tangent and UP
        links.new(geometry.outputs["Tangent"], vector_math.inputs[0])
        links.new(combine_xyz.outputs[0], vector_math.inputs[1])

        # Compare with normal using dot product for directional emission
        dot = nodes.new("ShaderNodeVectorMath")
        dot.operation = "DOT_PRODUCT"
        links.new(vector_math.outputs[0], dot.inputs[0])
        links.new(geometry.outputs["Normal"], dot.inputs[1])

        map_range = nodes.new("ShaderNodeMapRange")
        map_range.inputs["From Min"].default_value = -1.0
        map_range.inputs["From Max"].default_value = 1.0
        map_range.inputs["To Min"].default_value = 0.0
        map_range.inputs["To Max"].default_value = 1.0

        emit = (350.0, 350.0, 350.0, 1.0)
        no_emit = (0.0, 0.0, 0.0, 0.0)

        # Use color ramp to control emission based on viewing angle
        color_ramp = nodes.new("ShaderNodeValToRGB")
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

        links.new(dot.outputs["Value"], map_range.inputs["Value"])
        links.new(map_range.outputs["Result"], color_ramp.inputs[0])
        links.new(color_ramp.outputs["Color"], principled.inputs["Emission Strength"])

        output = nodes.new("ShaderNodeOutputMaterial")
        links.new(principled.outputs["BSDF"], output.inputs["Surface"])

        # Handle both Polygon and MultiPolygon geometries
        boundaries = (
            [zone.exterior.coords]
            if zone.geom_type == "Polygon"
            else [poly.exterior.coords for poly in zone.geoms]
        )

        for boundary_idx, boundary in enumerate(boundaries):
            # Create the curve data
            curve_data = bpy.data.curves.new(f"DriveTimeCurve_{idx}_{boundary_idx}", "CURVE")
            curve_data.dimensions = "3D"
            curve_data.resolution_u = 12
            curve_data.bevel_depth = bevel_depth

            # Create the curve object
            curve_obj = bpy.data.objects.new(f"DriveTime_{idx}_{boundary_idx}", curve_data)

            # Create a new spline in the curve
            spline = curve_data.splines.new("POLY")

            # Get coordinates and scale/center them
            coords = [
                ((x - mean_x) / 100, (y - mean_y) / 100, height_offset) for x, y in list(boundary)
            ]

            # Set the number of points
            spline.points.add(len(coords) - 1)  # -1 because one point is created by default

            # Assign the coordinates
            for point, coord in zip(spline.points, coords):
                point.co = (*coord, 1)  # The fourth component is weight, usually 1

            # Assign material
            curve_obj.data.materials.append(mat)

            # Link to scene
            bpy.context.scene.collection.objects.link(curve_obj)
            curves.append(curve_obj)

    logger.info("Created %d drive time curve objects", len(curves))
    return curves


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
