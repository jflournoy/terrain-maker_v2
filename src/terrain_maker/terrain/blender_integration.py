"""
Blender integration for terrain visualization.

This module contains Blender-specific code for creating and configuring
terrain meshes, materials, and rendering.
"""

import numpy as np

from terrain_maker.terrain.mesh_operations import vertex_colors_rgba

import bpy


def apply_vertex_colors(mesh_obj, vertex_colors, y_valid=None, x_valid=None, n_surface_vertices=None, logger=None):
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
        n_surface_vertices (int, optional): Number of surface vertices. If provided,
            boundary vertices (index >= n_surface_vertices) will be skipped to preserve
            their existing colors (e.g., two-tier edge colors). Default: None (apply to all)
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
        raise ValueError("Cannot apply vertex colors to a mesh with no faces (no loops)")

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

    # Colors cover the surface vertices (the first n of the mesh); vertices after them
    # (boundary skirt) keep the color they already have. Clamping their indices onto the
    # last surface vertex used to paint the whole skirt that one color.
    n_vertices = len(mesh.vertices)
    n_surface = n_surface_vertices if n_surface_vertices is not None else len(colors_normalized)
    if len(colors_normalized) > n_vertices:
        raise ValueError(
            f"{len(colors_normalized)} colors for a mesh with {n_vertices} vertices"
        )
    if n_surface > len(colors_normalized):
        raise ValueError(
            f"n_surface_vertices={n_surface} but only {len(colors_normalized)} colors were given"
        )

    color_data_flat = np.zeros(n_loops * 4, dtype=np.float32)
    color_layer.data.foreach_get("color", color_data_flat)
    color_data = color_data_flat.reshape((n_loops, 4))

    surface_mask = loop_vertex_indices < n_surface
    color_data[surface_mask] = colors_normalized[loop_vertex_indices[surface_mask]]
    color_layer.data.foreach_set("color", color_data.flatten())
    if logger:
        logger.debug(
            f"✓ Applied colors to {int(np.sum(surface_mask))} surface loops, "
            f"kept {int(np.sum(~surface_mask))} boundary loops"
        )


def apply_ring_colors(mesh_obj, ring_mask, y_valid, x_valid, ring_color=(0.15, 0.15, 0.15), logger=None):
    """
    Apply a solid color to vertices within a ring mask.

    Modifies the existing TerrainColors vertex color layer, setting RGB values
    for vertices that fall within the ring mask to the specified color.
    This creates a dark outline around areas of interest (e.g., park zones).

    Uses Blender's foreach_get/foreach_set for efficient bulk operations.

    Args:
        mesh_obj (bpy.types.Object): The Blender mesh object
        ring_mask (np.ndarray): 2D boolean array (height, width) where True = apply ring color
        y_valid (np.ndarray): Y indices mapping vertices to grid positions
        x_valid (np.ndarray): X indices mapping vertices to grid positions
        ring_color (tuple): RGB color (0-1 range) to apply to ring vertices. Default: dark gray.
        logger (logging.Logger, optional): Logger for progress messages
    """
    mesh = mesh_obj.data

    # Get existing color layer
    if len(mesh.vertex_colors) == 0:
        if logger:
            logger.warning("No vertex colors to modify for ring colors")
        return

    color_layer = mesh.vertex_colors[0]  # TerrainColors
    n_loops = len(color_layer.data)

    if n_loops == 0:
        if logger:
            logger.warning("Mesh has no color data for ring colors")
        return

    # Get current colors
    current_colors = np.zeros(n_loops * 4, dtype=np.float32)
    color_layer.data.foreach_get("color", current_colors)
    current_colors = current_colors.reshape(-1, 4)

    # Build loop-to-vertex mapping (each face has 3 or 4 loops)
    # For triangulated meshes, each face has 3 vertices = 3 loops
    mesh.calc_loop_triangles()
    loop_to_vertex = np.zeros(n_loops, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_to_vertex)

    # Count how many loops will be modified
    modified_count = 0

    for loop_idx in range(n_loops):
        vert_idx = loop_to_vertex[loop_idx]

        # Skip if vertex index is out of range for our mapping
        if vert_idx >= len(y_valid):
            continue

        y = int(y_valid[vert_idx])
        x = int(x_valid[vert_idx])

        # Clamp to valid indices
        y = max(0, min(y, ring_mask.shape[0] - 1))
        x = max(0, min(x, ring_mask.shape[1] - 1))

        if ring_mask[y, x]:
            current_colors[loop_idx, 0] = ring_color[0]
            current_colors[loop_idx, 1] = ring_color[1]
            current_colors[loop_idx, 2] = ring_color[2]
            # Keep alpha unchanged
            modified_count += 1

    # Apply modified colors back
    color_layer.data.foreach_set("color", current_colors.flatten())

    if logger:
        logger.info(f"✓ Applied ring color to {modified_count:,} loops")


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
    if len(mesh.loops) == 0:
        raise ValueError("Cannot apply a road mask to a mesh with no faces (no loops)")
    try:
        road_layer = mesh.vertex_colors.new(name="RoadMask")
    except Exception as e:
        raise RuntimeError(f"Could not create the RoadMask color layer: {e}") from e

    n_loops = len(road_layer.data)

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

    # Every surface vertex must fall inside the mask; one that does not means the mask is
    # not on the mesh's grid, and reading it as "no road" would hide that
    in_bounds = ((y_valid >= 0) & (y_valid < road_mask.shape[0])
                 & (x_valid >= 0) & (x_valid < road_mask.shape[1]))
    if not np.all(in_bounds):
        raise ValueError(
            f"{int(np.sum(~in_bounds))} mesh vertices fall outside the road mask "
            f"(mask shape {road_mask.shape}); align the mask to the mesh grid first"
        )
    vertex_road_values[:] = road_mask[y_valid, x_valid]

    # Convert to binary (>0.5 = road)
    vertex_is_road = (vertex_road_values > 0.5).astype(np.float32)

    # Vertices past the surface grid (boundary skirt) are never road
    loop_road_values = np.zeros(n_loops, dtype=np.float32)
    on_surface = loop_vertex_indices < n_positions
    loop_road_values[on_surface] = vertex_is_road[loop_vertex_indices[on_surface]]

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
        >>> from terrain_maker.terrain.roads import smooth_road_vertices
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
    boundary_colors=None,
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
            for surface vertices
        y_valid (np.ndarray, optional): Array of y indices for vertex colors
        x_valid (np.ndarray, optional): Array of x indices for vertex colors
        boundary_colors (np.ndarray, optional): Array of RGB colors (n_boundary, 3)
            for boundary vertices in two-tier mode
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

        # Colors are computed per vertex (vectorized), then expanded to face loops
        if (colors is not None and y_valid is not None) or boundary_colors is not None:
            if logger:
                logger.info("Applying vertex colors...")
            color_layer = mesh.vertex_colors.new(name="TerrainColors")
            n_loops = len(color_layer.data)
            if n_loops > 0:
                per_vertex = vertex_colors_rgba(
                    len(vertices), colors, y_valid, x_valid, boundary_colors
                )
                loop_vertices = np.zeros(n_loops, dtype=np.int64)
                mesh.loops.foreach_get("vertex_index", loop_vertices)
                color_layer.data.foreach_set("color", per_vertex[loop_vertices].ravel())

        # Create object and link to scene
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(obj)

        # Create and assign material
        from terrain_maker.terrain.core import apply_colormap_material

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
