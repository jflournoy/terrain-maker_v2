"""Mesh generation operations for terrain visualization.

This module contains functions for creating and manipulating terrain meshes,
extracted from the core Terrain class for better modularity and testability.

Performance optimizations:
- Numba JIT compilation for hot loops (face generation)
- Vectorized NumPy operations where possible

The implementation lives in terrain_maker.terrain.mesh; this module re-exports it
so existing imports keep working.
"""

from terrain_maker.terrain._numba_compat import NUMBA_AVAILABLE  # noqa: F401
from terrain_maker.terrain.mesh.grid import (  # noqa: F401
    MeshData,
    _generate_faces_numba,
    _to_unit_rgba,
    find_boundary_points,
    generate_faces,
    generate_vertex_positions,
    vertex_colors_rgba,
)
from terrain_maker.terrain.mesh.boundary import (  # noqa: F401
    catmull_rom_curve,
    deduplicate_boundary_points,
    fit_catmull_rom_boundary_curve,
    smooth_boundary_points,
    sort_boundary_points,
    sort_boundary_points_angular,
)
from terrain_maker.terrain.mesh.rectangle_edges import (  # noqa: F401
    _select_rectangle_boundary,
    diagnose_rectangle_edge_coverage,
    generate_rectangle_edge_pixels,
    generate_rectangle_edge_vertices,
    generate_transform_aware_rectangle_edges,
    generate_transform_aware_rectangle_edges_fractional,
)
from terrain_maker.terrain.mesh.skirt import (  # noqa: F401
    _SkirtInputs,
    _build_single_tier_skirt,
    _build_two_tier_skirt,
    _interpolated_surface_color,
    _log_interpolation_diagnostics,
    _log_two_tier_face_stats,
    _make_position_lookup,
    _skirt_quad,
    _two_tier_colors,
    _two_tier_faces,
    _two_tier_vertices,
    _wraparound_gap_too_large,
    _z_along_original_boundary,
    create_boundary_extension,
)
