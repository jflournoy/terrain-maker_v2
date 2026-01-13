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

    Uses Blender's foreach_set for ~100x faster bulk operations.

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

    n_loops = len(color_layer.data)
    if n_loops == 0:
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

    # FAST PATH: Use foreach_get/foreach_set for bulk operations
    # Get all loop->vertex mappings at once
    loop_vertex_indices = np.zeros(n_loops, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_vertex_indices)

    # Clamp indices to valid range
    max_color_idx = len(colors_normalized) - 1
    loop_vertex_indices = np.clip(loop_vertex_indices, 0, max_color_idx)

    # Build flat color array for all loops (RGBA per loop)
    loop_colors = colors_normalized[loop_vertex_indices].flatten()

    # Apply all colors at once
    color_layer.data.foreach_set("color", loop_colors)

    if logger:
        logger.debug(f"✓ Applied colors to {n_loops} loops (vectorized)")


def apply_road_mask(mesh_obj, road_mask, y_valid, x_valid, logger=None):
    """
    Apply a road mask as a separate vertex color layer for material detection.

    Creates a "RoadMask" vertex color layer where road vertices have R=1.0
    and non-road vertices have R=0.0. This allows the material shader to
    detect roads without changing the terrain colors.

    Uses Blender's foreach_set for ~100x faster bulk operations.

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

    n_loops = len(road_layer.data)
    if n_loops == 0:
        if logger:
            logger.warning("Mesh has no color data for road mask")
        return

    # Debug: check road mask statistics
    if logger:
        road_pixels = np.sum(road_mask > 0.5)
        logger.info(f"Road mask stats: shape={road_mask.shape}, road_pixels={road_pixels}, max={road_mask.max():.2f}")

    n_positions = len(y_valid)

    # FAST PATH: Use foreach_get/foreach_set for bulk operations
    # Get all loop->vertex mappings at once
    loop_vertex_indices = np.zeros(n_loops, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_vertex_indices)

    # Build road mask values for each vertex
    # First, create vertex-level road mask by sampling grid at valid positions
    vertex_road_values = np.zeros(n_positions, dtype=np.float32)

    # Vectorized bounds check and road mask lookup
    y_in_bounds = (y_valid >= 0) & (y_valid < road_mask.shape[0])
    x_in_bounds = (x_valid >= 0) & (x_valid < road_mask.shape[1])
    in_bounds = y_in_bounds & x_in_bounds

    # Sample road mask for in-bounds vertices
    vertex_road_values[in_bounds] = road_mask[y_valid[in_bounds], x_valid[in_bounds]]

    # Convert to binary (>0.5 = road)
    vertex_is_road = (vertex_road_values > 0.5).astype(np.float32)

    # Clamp loop indices to valid vertex range
    loop_vertex_indices = np.clip(loop_vertex_indices, 0, n_positions - 1)

    # Map vertex road values to loops
    loop_road_values = vertex_is_road[loop_vertex_indices]

    # Build RGBA color array: road = (1,0,0,1), non-road = (0,0,0,1)
    loop_colors = np.zeros((n_loops, 4), dtype=np.float32)
    loop_colors[:, 0] = loop_road_values  # R = road value
    loop_colors[:, 3] = 1.0  # A = 1

    # Apply all colors at once
    road_layer.data.foreach_set("color", loop_colors.flatten())

    # Update mesh to apply changes
    mesh.update()

    road_count = int(np.sum(loop_road_values > 0.5))
    if logger:
        logger.info(f"✓ Applied road mask to {road_count}/{n_loops} vertex loops (vectorized)")


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
