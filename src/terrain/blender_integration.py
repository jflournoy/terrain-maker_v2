"""
Blender integration for terrain visualization.

This module contains Blender-specific code for creating and configuring
terrain meshes, materials, and rendering.
"""

import numpy as np

import bpy


def apply_vertex_colors(mesh_obj, vertex_colors, y_valid=None, x_valid=None, logger=None):
    """
    Apply vertex-space colors directly to an existing Blender mesh.

    This function applies colors that are already in vertex space (one color per vertex)
    rather than grid space. Useful for re-applying colors after modification or when
    using dual colormaps that produce vertex-space colors.

    Args:
        mesh_obj (bpy.types.Object): The Blender mesh object to apply colors to
        vertex_colors (np.ndarray): Vertex-space colors, shape (n_vertices, 3) or (n_vertices, 4)
        y_valid (np.ndarray, optional): Not used, kept for API compatibility
        x_valid (np.ndarray, optional): Not used, kept for API compatibility
        logger (logging.Logger, optional): Logger for progress messages

    Note:
        This function assumes vertex_colors are already in the correct order matching
        the mesh vertices. For grid-space colors, use create_blender_mesh instead.
    """
    if logger:
        logger.debug(f"Applying {len(vertex_colors)} vertex colors to mesh...")

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

    # Normalize colors to 0-1 range if they're uint8
    colors_normalized = vertex_colors.astype(np.float32)
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
