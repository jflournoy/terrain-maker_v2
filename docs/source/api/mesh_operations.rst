Mesh Operations Module
======================

Low-level mesh generation and boundary operations.

This module contains functions for creating and manipulating terrain meshes,
extracted from the core Terrain class for better modularity and testability.

Optimized with Numba JIT compilation for performance-critical loops.

Core Mesh Generation
---------------------

.. autofunction:: src.terrain.mesh_operations.generate_vertex_positions

   Generate 3D vertex positions from DEM data.

   Converts 2D elevation grid to 3D mesh vertices with scaling.

   Example::

       from src.terrain.mesh_operations import generate_vertex_positions

       vertices = generate_vertex_positions(
           dem_data,
           valid_mask=~np.isnan(dem_data),
           scale_factor=100.0,
           height_scale=30.0
       )

.. autofunction:: src.terrain.mesh_operations.generate_faces

   Generate triangle faces for mesh with batch processing.

   Uses Numba JIT compilation for ~10x speedup on large meshes.

   Example::

       from src.terrain.mesh_operations import generate_faces

       faces = generate_faces(
           height=dem.shape[0],
           width=dem.shape[1],
           coord_to_index=coord_map,
           batch_size=10000
       )

.. autofunction:: src.terrain.mesh_operations._generate_faces_numba

   Numba-accelerated face generation (internal).

   **Performance:** ~10x faster than pure Python for 1M+ faces.

Boundary Detection
------------------

.. autofunction:: src.terrain.mesh_operations.find_boundary_points

   Find boundary points using morphological operations.

   Identifies edge pixels where valid DEM data meets invalid/NaN regions.

   Example::

       from src.terrain.mesh_operations import find_boundary_points

       valid_mask = ~np.isnan(dem_data)
       boundary = find_boundary_points(valid_mask)
       print(f"Found {len(boundary)} boundary points")

Boundary Extension
------------------

.. autofunction:: src.terrain.mesh_operations.create_boundary_extension

   Create two-tier edge extrusion for terrain mesh.

   **Main function for edge generation** - supports multiple modes:

   - Single-tier vs two-tier
   - Morphological vs rectangle edge detection
   - Standard vs fractional edge positioning
   - Catmull-Rom curve smoothing

   Example::

       from src.terrain.mesh_operations import create_boundary_extension

       # Basic two-tier edge
       edge_data = create_boundary_extension(
           processed_dem,
           terrain.y_valid,
           terrain.x_valid,
           valid_mask,
           scale_factor=100.0,
           height_scale=30.0,
           two_tier=True,
           edge_color=(0.5, 0.48, 0.45),  # Clay
           base_color=(0.3, 0.3, 0.3)
       )

       # With Catmull-Rom smoothing
       edge_data = create_boundary_extension(
           ...,
           use_catmull_rom=True,
           catmull_rom_subdivisions=20
       )

       # With fractional edges (smooth projection curves)
       edge_data = create_boundary_extension(
           ...,
           use_fractional_edges=True,
           use_rectangle_edges=True
       )

   See :doc:`../examples/combined_render` for usage examples.

Catmull-Rom Curve Fitting
--------------------------

.. autofunction:: src.terrain.mesh_operations.catmull_rom_curve

   Evaluate Catmull-Rom spline at parameter t.

   **Catmull-Rom properties:**

   - Passes through all control points
   - C¹ continuous (smooth first derivative)
   - Local control (changing p1/p2 only affects nearby curve)

.. autofunction:: src.terrain.mesh_operations.fit_catmull_rom_boundary_curve

   Fit smooth Catmull-Rom curve through boundary points.

   Eliminates pixel-grid staircase pattern from morphological boundaries.

   Example::

       from src.terrain.mesh_operations import fit_catmull_rom_boundary_curve

       # Smooth boundary with 20 points per segment
       smooth_boundary = fit_catmull_rom_boundary_curve(
           boundary_points,
           subdivisions=20,
           closed_loop=True
       )

Boundary Processing
-------------------

.. autofunction:: src.terrain.mesh_operations.smooth_boundary_points

   Apply moving average smoothing to boundary.

.. autofunction:: src.terrain.mesh_operations.deduplicate_boundary_points

   Remove duplicate consecutive points.

.. autofunction:: src.terrain.mesh_operations.sort_boundary_points_angular

   Sort boundary points by angular position (for closed loops).

.. autofunction:: src.terrain.mesh_operations.sort_boundary_points

   Sort boundary points for ordered traversal.

Rectangle Edge Generation
--------------------------

.. autofunction:: src.terrain.mesh_operations.generate_rectangle_edge_pixels

   Generate edge pixels using rectangle sampling (fast alternative to morphological).

   **Performance:** ~150x faster than morphological edge detection.

.. autofunction:: src.terrain.mesh_operations.generate_rectangle_edge_vertices

   Generate edge vertices from rectangle edge pixels.

.. autofunction:: src.terrain.mesh_operations.generate_transform_aware_rectangle_edges

   Generate rectangle edges with geographic transform awareness.

.. autofunction:: src.terrain.mesh_operations.generate_transform_aware_rectangle_edges_fractional

   Generate fractional rectangle edges preserving projection curvature.

   **Fractional edges:** Preserve subtle curves from WGS84→UTM transformation.

.. autofunction:: src.terrain.mesh_operations.diagnose_rectangle_edge_coverage

   Diagnostic function to analyze rectangle edge coverage.

Performance Optimizations
--------------------------

**Numba JIT compilation:**

- Face generation: ~10x speedup
- Boundary processing: ~5x speedup
- Requires: ``numba`` package (``pip install numba``)
- Automatic fallback to Python if numba unavailable

**Typical performance:**

+---------------------------+------------------+----------------+
| Operation                 | Array Size       | Time           |
+===========================+==================+================+
| Generate vertices         | 1M points        | ~50ms          |
| Generate faces (Numba)    | 2M triangles     | ~200ms         |
| Find boundary (morph)     | 4096²            | ~500ms         |
| Find boundary (rectangle) | 4096²            | ~3ms (150x)    |
| Catmull-Rom fit           | 1000 pts, sub=20 | ~100ms         |
+---------------------------+------------------+----------------+

**Memory usage:**

- Vertices: 12 bytes per vertex (3 float32s)
- Faces: 12 bytes per triangle (3 int32s)
- 1M vertices + 2M triangles ≈ 36MB

See Also
--------

- :doc:`core` - Uses these functions for mesh creation
- :doc:`blender_integration` - Applies vertex colors to generated meshes
- :doc:`../examples/combined_render` - Example usage with two-tier edges
