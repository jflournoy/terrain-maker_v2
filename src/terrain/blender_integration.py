"""
Blender integration for terrain visualization.

This module contains Blender-specific code for creating and configuring
terrain meshes, materials, and rendering.
"""

import numpy as np

import bpy


def apply_vertex_colors(mesh_obj, vertex_colors, y_valid=None, x_valid=None, logger=None):
    """
    Apply colors to an existing Blender mesh.

    Accepts colors in either vertex-space (n_vertices, 3/4) or grid-space (height, width, 3/4).
    When grid-space colors are provided with y_valid/x_valid indices, colors are extracted
    for each vertex using those coordinates.

    Args:
        mesh_obj (bpy.types.Object): The Blender mesh object to apply colors to
        vertex_colors (np.ndarray): Colors in one of two formats:
            - Vertex-space: shape (n_vertices, 3) or (n_vertices, 4)
            - Grid-space: shape (height, width, 3) or (height, width, 4)
        y_valid (np.ndarray, optional): Y indices for grid-space colors
        x_valid (np.ndarray, optional): X indices for grid-space colors
        logger (logging.Logger, optional): Logger for progress messages
    """
    mesh = mesh_obj.data

    # Get or create color layer
    if len(mesh.vertex_colors) == 0:
        color_layer = mesh.vertex_colors.new(name="TerrainColors")
    else:
        color_layer = mesh.vertex_colors[0]

    if len(color_layer.data) == 0:
        if logger:
            logger.warning("Mesh has no color data")
        return

    # Check if colors are grid-space (3D) or vertex-space (2D)
    if vertex_colors.ndim == 3 and y_valid is not None and x_valid is not None:
        # Grid-space colors: extract colors for each vertex using indices
        colors_for_vertices = vertex_colors[y_valid, x_valid]
        if logger:
            logger.debug(f"Extracted {len(colors_for_vertices)} vertex colors from grid")
    else:
        # Already vertex-space colors
        colors_for_vertices = vertex_colors
        if logger:
            logger.debug(f"Using {len(colors_for_vertices)} vertex-space colors")

    # Normalize colors to 0-1 range if they're uint8
    colors_normalized = colors_for_vertices.astype(np.float32)
    if colors_normalized.max() > 1.0:
        colors_normalized = colors_normalized / 255.0

    # Ensure colors are RGBA (add alpha channel if needed)
    if colors_normalized.shape[-1] == 3:
        alpha = np.ones((colors_normalized.shape[0], 1), dtype=np.float32)
        colors_normalized = np.concatenate([colors_normalized, alpha], axis=1)

    # For each polygon loop, get vertex and set color
    for poly in mesh.polygons:
        for loop_idx in poly.loop_indices:
            vertex_idx = mesh.loops[loop_idx].vertex_index

            # Apply color if vertex index is in range
            if vertex_idx < len(colors_normalized):
                color_layer.data[loop_idx].color = colors_normalized[vertex_idx]

    if logger:
        logger.debug(f"✓ Applied colors to {len(color_layer.data)} loops")


def apply_road_mask(mesh_obj, road_mask, y_valid, x_valid, logger=None):
    """
    Apply a road mask as a separate vertex color layer for material detection.

    Creates a "RoadMask" vertex color layer where road vertices have R=1.0
    and non-road vertices have R=0.0. This allows the material shader to
    detect roads without changing the terrain colors.

    Args:
        mesh_obj (bpy.types.Object): The Blender mesh object
        road_mask (np.ndarray): 2D boolean or float array (height, width) where >0.5 = road
        y_valid (np.ndarray): Y indices mapping vertices to grid positions
        x_valid (np.ndarray): X indices mapping vertices to grid positions
        logger (logging.Logger, optional): Logger for progress messages
    """
    mesh = mesh_obj.data

    # Create road mask layer using vertex_colors API for ShaderNodeVertexColor compatibility
    # This matches how TerrainColors is created for consistent shader access
    try:
        road_layer = mesh.vertex_colors.new(name="RoadMask")
    except Exception as e:
        if logger:
            logger.warning(f"Failed to create road mask layer: {e}")
        return

    if len(road_layer.data) == 0:
        if logger:
            logger.warning("Mesh has no color data for road mask")
        return

    # Debug: check road mask statistics
    if logger:
        road_pixels = np.sum(road_mask > 0.5)
        logger.info(f"Road mask stats: shape={road_mask.shape}, road_pixels={road_pixels}, max={road_mask.max():.2f}")

    n_positions = len(y_valid)
    n_loops = len(road_layer.data)
    road_count = 0

    # First, initialize ALL loops to non-road (prevents undefined values)
    for i in range(n_loops):
        road_layer.data[i].color = (0.0, 0.0, 0.0, 1.0)

    # Then mark road vertices
    for poly in mesh.polygons:
        for loop_idx in poly.loop_indices:
            vertex_idx = mesh.loops[loop_idx].vertex_index

            if vertex_idx < n_positions:
                y, x = y_valid[vertex_idx], x_valid[vertex_idx]

                # Check bounds and road mask value
                if 0 <= y < road_mask.shape[0] and 0 <= x < road_mask.shape[1]:
                    if road_mask[y, x] > 0.5:
                        # Mark as road: R=1, G=0, B=0, A=1
                        road_layer.data[loop_idx].color = (1.0, 0.0, 0.0, 1.0)
                        road_count += 1

    # Update mesh to apply changes
    mesh.update()

    if logger:
        logger.info(f"✓ Applied road mask to {road_count}/{n_loops} vertex loops")


def apply_vertex_positions(
    mesh_obj,
    new_positions: np.ndarray,
    logger=None,
) -> None:
    """
    Apply new 3D positions to mesh vertices.

    Useful for applying smoothed vertex coordinates to an existing mesh,
    e.g., after road smoothing or terrain filtering.

    Args:
        mesh_obj: Blender mesh object to modify
        new_positions: Array of shape (n_vertices, 3) with new [x, y, z] positions
        logger: Optional logger for progress messages

    Raises:
        ValueError: If new_positions shape doesn't match mesh vertex count

    Example:
        >>> # Smooth road vertices and apply to mesh
        >>> from src.terrain.roads import smooth_road_vertices
        >>>
        >>> vertices = np.array([v.co[:] for v in mesh.data.vertices])
        >>> smoothed = smooth_road_vertices(vertices, road_mask, y_valid, x_valid)
        >>> apply_vertex_positions(mesh, smoothed)
    """
    mesh = mesh_obj.data
    n_vertices = len(mesh.vertices)

    if new_positions.shape[0] != n_vertices:
        raise ValueError(
            f"Position array size {new_positions.shape[0]} doesn't match "
            f"mesh vertex count {n_vertices}"
        )

    if new_positions.shape[1] != 3:
        raise ValueError(f"Expected (n, 3) positions, got shape {new_positions.shape}")

    # Apply new positions to all vertices
    for i, v in enumerate(mesh.vertices):
        v.co = new_positions[i]

    # Update mesh to recalculate normals etc.
    mesh.update()

    if logger:
        logger.info(f"✓ Applied new positions to {n_vertices} vertices")


def create_blender_mesh(
    vertices,
    faces,
    colors=None,
    y_valid=None,
    x_valid=None,
    name="TerrainMesh",
    logger=None,
):
    """
    Create a Blender mesh object from vertices and faces.

    Creates a new Blender mesh datablock, populates it with geometry data,
    optionally applies vertex colors, and creates a material with colormap shader.

    Args:
        vertices (np.ndarray): Array of (n, 3) vertex positions
        faces (list): List of tuples defining face connectivity
        colors (np.ndarray, optional): Array of RGB/RGBA colors (height, width, channels)
        y_valid (np.ndarray, optional): Array of y indices for vertex colors
        x_valid (np.ndarray, optional): Array of x indices for vertex colors
        name (str): Name for the mesh and object (default: "TerrainMesh")
        logger (logging.Logger, optional): Logger for progress messages

    Returns:
        bpy.types.Object: The created terrain mesh object

    Raises:
        RuntimeError: If Blender is not available or mesh creation fails
    """
    if logger:
        logger.info(
            f"Creating Blender mesh with {len(vertices)} vertices and {len(faces)} faces..."
        )

    try:
        # Create mesh datablock
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(vertices.tolist(), [], faces)
        mesh.update(calc_edges=True)

        # Apply colors if provided
        if colors is not None and y_valid is not None and x_valid is not None:
            if logger:
                logger.info("Applying vertex colors with optimized method...")

            color_layer = mesh.vertex_colors.new(name="TerrainColors")

            if len(color_layer.data) > 0:
                # Create color data array (Blender expects normalized 0-1 floats)
                color_data = np.zeros((len(color_layer.data), 4), dtype=np.float32)

                # Normalize colors to 0-1 range if they're uint8
                colors_normalized = colors.astype(np.float32)
                if colors_normalized.max() > 1.0:
                    colors_normalized = colors_normalized / 255.0

                # Ensure colors are RGBA (add alpha channel if needed)
                if colors_normalized.shape[-1] == 3:
                    alpha = np.ones(
                        (colors_normalized.shape[0], colors_normalized.shape[1], 1),
                        dtype=np.float32,
                    )
                    colors_normalized = np.concatenate([colors_normalized, alpha], axis=-1)

                # Get number of original positions (before boundary extension)
                n_positions = len(y_valid)

                # For each polygon loop, get vertex and set color
                for poly in mesh.polygons:
                    for loop_idx in poly.loop_indices:
                        vertex_idx = mesh.loops[loop_idx].vertex_index

                        # Only apply colors to top vertices
                        if vertex_idx < n_positions:
                            y, x = y_valid[vertex_idx], x_valid[vertex_idx]

                            # Check bounds
                            if (
                                0 <= y < colors_normalized.shape[0]
                                and 0 <= x < colors_normalized.shape[1]
                            ):
                                color_data[loop_idx] = colors_normalized[y, x]

                # Batch assign all colors at once
                try:
                    color_layer.data.foreach_set("color", color_data.flatten())
                except Exception as e:
                    if logger:
                        logger.warning(f"Batch color assignment failed: {e}")
                    # Fallback to slower per-loop assignment
                    for i, color in enumerate(color_data):
                        color_layer.data[i].color = color

        # Create object and link to scene
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(obj)

        # Create and assign material
        from src.terrain.core import apply_colormap_material

        mat = bpy.data.materials.new(name=f"{name}Material")
        mat.use_nodes = True
        obj.data.materials.append(mat)
        apply_colormap_material(mat)

        if logger:
            logger.info(f"Terrain mesh '{name}' created successfully")

        return obj

    except Exception as e:
        if logger:
            logger.error(f"Error creating terrain mesh: {str(e)}")
        raise
