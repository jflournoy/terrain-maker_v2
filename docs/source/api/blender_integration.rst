Blender Integration Module
==========================

Low-level Blender mesh operations for terrain visualization.

This module contains Blender-specific code for creating and configuring terrain meshes,
applying vertex colors, and managing mesh data. Uses Blender's high-performance bulk
operations (``foreach_set``, ``foreach_get``) for ~100x faster vertex manipulation.

Vertex Colors
-------------

.. autofunction:: src.terrain.blender_integration.apply_vertex_colors

   Apply colors to mesh vertices with automatic grid-to-vertex mapping.
   Used in :doc:`../examples/combined_render`.

   **Supports two color formats:**

   1. **Vertex-space colors**: shape ``(n_vertices, 3)`` or ``(n_vertices, 4)``
      - One color per vertex
      - Direct application (fastest)

   2. **Grid-space colors**: shape ``(height, width, 3)`` or ``(height, width, 4)``
      - Colors in DEM grid coordinates
      - Requires ``y_valid``/``x_valid`` indices to extract per-vertex colors
      - Handles downsampled meshes automatically

   **Preserves boundary colors:**

   Use ``n_surface_vertices`` parameter to preserve edge colors when applying
   terrain colors (used with two-tier edges).

   Example::

       from src.terrain.blender_integration import apply_vertex_colors

       # Vertex-space colors (simple)
       apply_vertex_colors(mesh_obj, vertex_colors)

       # Grid-space colors (with mapping)
       apply_vertex_colors(
           mesh_obj,
           grid_colors,
           y_valid=terrain.y_valid,
           x_valid=terrain.x_valid
       )

       # Preserve edge colors (two-tier mode)
       apply_vertex_colors(
           mesh_obj,
           terrain_colors,
           y_valid=terrain.y_valid,
           x_valid=terrain.x_valid,
           n_surface_vertices=len(terrain.y_valid)
       )

.. autofunction:: src.terrain.blender_integration.apply_ring_colors

   Apply solid color to ring/boundary vertices.

   Used to color two-tier edge extrusions with base materials.

   Example::

       apply_ring_colors(
           mesh_obj,
           ring_mask,
           y_valid, x_valid,
           ring_color=(0.5, 0.48, 0.45)  # Clay
       )

.. autofunction:: src.terrain.blender_integration.apply_road_mask

   Apply road mask to color road vertices.

   Marks road vertices with a distinctive color for shader-based material detection.

   Example::

       apply_road_mask(
           mesh_obj,
           road_mask,
           y_valid, x_valid
       )

Vertex Positions
----------------

.. autofunction:: src.terrain.blender_integration.apply_vertex_positions

   Update mesh vertex positions in place.

   Used for modifying terrain geometry after mesh creation (smoothing, offsetting, etc.).

   Example::

       # Offset vertices by height
       new_positions = old_positions.copy()
       new_positions[:, 2] += height_offset

       apply_vertex_positions(mesh_obj, new_positions)

Mesh Creation
-------------

.. autofunction:: src.terrain.blender_integration.create_blender_mesh

   Create Blender mesh object from vertices and faces.

   Low-level mesh creation. Most users should use
   :meth:`~src.terrain.core.Terrain.create_mesh` instead.

   Example::

       mesh_obj = create_blender_mesh(
           vertices,
           faces,
           name="Terrain"
       )

Performance Notes
-----------------

This module uses Blender's bulk operations for maximum performance:

- ``foreach_set`` / ``foreach_get``: ~100x faster than Python loops
- Flat numpy arrays: Direct memory access without Python overhead
- Vectorized operations: Process all vertices simultaneously

**Typical performance:**

- Applying colors to 1M vertices: ~0.1 seconds (vs 10+ seconds with loops)
- Creating mesh with 1M faces: ~0.5 seconds
