"""Flow accumulation module for hydrological analysis.

Implements D8 flow routing algorithm with precipitation weighting.
Based on flow-spec.md requirements.

Supports two backends:
- "pysheds": Uses pysheds library for core hydrology (recommended)
- "custom": Uses custom numba-accelerated implementation

The implementation lives in terrain_maker.terrain.hydrology; this module re-exports it
so existing imports keep working.
"""

from terrain_maker.terrain._numba_compat import NUMBA_AVAILABLE  # noqa: F401
from terrain_maker.terrain.hydrology.flow import PYSHEDS_AVAILABLE, PyshedsGrid  # noqa: F401
from terrain_maker.terrain.hydrology.flow_cache import (  # noqa: F401
    _get_cache_key_params,
    _get_dem_mtime,
    _load_from_cache,
    _save_to_cache,
    _validate_cache,
)
from terrain_maker.terrain.hydrology.routing import (  # noqa: F401
    D8_DIRECTIONS,
    D8_OFFSETS,
    _compute_flow_direction_jit,
    _fix_coastal_flow_directions,
    _fix_coastal_flow_directions_jit,
    compute_flow_direction,
    identify_outlets,
)
from terrain_maker.terrain.hydrology.accumulation import (  # noqa: F401
    _compute_drainage_area_jit,
    _compute_upstream_rainfall_jit,
    compute_discharge_potential,
    compute_drainage_area,
    compute_upstream_rainfall,
)
from terrain_maker.terrain.hydrology.breaching import (  # noqa: F401
    _apply_breach,
    _breach_sinks_parallel_batch,
    _cluster_sinks_checkerboard,
    _dijkstra_single_sink,
    _find_breach_path_dijkstra,
    _find_breach_path_dijkstra_jit,
    _identify_sinks,
    _identify_sinks_jit,
    _reconstruct_path,
    breach_depressions_constrained,
)
from terrain_maker.terrain.hydrology.conditioning import (  # noqa: F401
    _compute_flat_gradient_bfs,
    _compute_flat_gradient_bfs_jit,
    _fill_depressions,
    _fill_small_sinks,
    _resolve_flats,
    condition_dem,
    condition_dem_spec,
    detect_endorheic_basins,
    detect_ocean_mask,
    priority_flood_fill_epsilon,
)
from terrain_maker.terrain.hydrology.flow import (  # noqa: F401
    _accumulate_flow,
    _build_conditioning_masks,
    _cell_size_m,
    _condition_and_route,
    _downsample_to_max_cells,
    _load_aligned_precipitation,
    _route_lakes_and_find_inlets,
    _write_flow_outputs,
    _write_geotiff,
    compute_flow_with_basins,
    flow_accumulation,
)
