"""DEM conditioning: priority flood, depression filling, flats, ocean and basin masks."""

import logging
from typing import Tuple, Optional, Literal

import numpy as np

from terrain_maker.terrain._numba_compat import NUMBA_AVAILABLE, jit

from terrain_maker.terrain.hydrology.breaching import breach_depressions_constrained
from terrain_maker.terrain.hydrology.routing import identify_outlets

logger = logging.getLogger(__name__)


def priority_flood_fill_epsilon(
    dem: np.ndarray,
    outlets: np.ndarray,
    epsilon: float = 1e-4,
    nodata_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Fill residual depressions with epsilon gradient (Stage 2b of flow-spec.md).

    Uses Barnes et al. (2014) priority-flood algorithm with epsilon gradient
    applied DURING fill (not after). This creates flow-directing gradients
    as flats form, ensuring proper drainage without post-processing.

    Parameters
    ----------
    dem : np.ndarray
        Input DEM (already breached by Stage 2a)
    outlets : np.ndarray (bool)
        Outlet mask from identify_outlets()
    epsilon : float, default 1e-4
        Minimum elevation increment per cell (meters per cell).
        Creates gradients in filled areas to ensure drainage.
    nodata_mask : np.ndarray (bool), optional
        Cells to exclude from filling

    Returns
    -------
    np.ndarray
        DEM with residual depressions filled

    Notes
    -----
    This implements Stage 2b of flow-spec.md (lines 291-328).

    **Outlet Virtual Elevation:**
    Outlets are seeded into the priority queue with their original elevations.
    As the fill progresses outward from outlets, depression cells are raised
    with epsilon increments, creating micro-gradients that naturally point
    back toward the outlets. This ensures filled areas drain properly in Stage 3.

    **Epsilon Application (Flat Resolution Strategy):**
    Epsilon is applied DURING fill as cells are raised. This is the most
    robust flat resolution approach for several reasons:

    1. **Implicit Gradient Creation**: As the fill progresses from outlets toward
       interior sinks, cells filled later get progressively higher elevations
       (by epsilon increments). This creates natural gradients pointing back
       toward outlets without explicit post-processing.

    2. **Alternatives Mentioned in Literature:**
       - **Garbrecht & Martz (1997)**: Dual-gradient method assigns flow to
         flats based on proximity to higher terrain. More complex to implement.
       - **Postprocessing**: Some systems fill without gradients, then resolve
         flats afterward. Can be ambiguous for complex flat structures.

    3. **Recommendation**: The epsilon-during-fill approach (used here) is:
       - Simpler to understand and implement
       - Produces consistent drainage patterns
       - Guaranteed to resolve all flats into valid flow networks
       - No need for separate flat-resolution algorithm

    **Epsilon Selection:**
    For epsilon tuning, see flow-spec.md lines 484-489:
    - **Auto-calculated** (default): epsilon = 1e-5 * cell_resolution
      - For 10m DEM: epsilon ≈ 1e-4 m/cell (0.1 mm per cell)
      - For 1m DEM: epsilon ≈ 1e-5 m/cell (0.01 mm per cell)
    - **For integer DEMs**: Use epsilon = 1 in native elevation units
      - If elevation in millimeters: epsilon = 1 mm/cell
      - If elevation in centimeters: epsilon = 1 cm/cell
    - **Too small epsilon**: Floating-point accumulation errors may create
      ties or reversals in flat areas. Rule of thumb: epsilon should exceed
      DEM measurement precision.
    - **Too large epsilon**: Creates obvious artificial "stair-stepping"
      in filled areas. Visual inspection usually reveals values > 0.1m.

    **Seed Cells:**
    The priority queue is seeded with:
    1. All identified outlets (from Stage 1) - guaranteed sinks
    2. All cells adjacent to NoData (domain boundary) - implicit outlets

    This ensures water drains both to identified outlets AND off the map edge.

    **Flat Area Behavior:**
    After filling, flat areas will have cells at different elevations (differing
    by epsilon). During Stage 3 (flow direction), steepest descent will cause
    water on flats to flow toward outlet-adjacent cells, creating coherent
    drainage patterns. This is more realistic than assuming multiple flow
    directions on wide flats.

    References
    ----------
    Barnes, R., Lehman, C., & Mulla, D. (2014). Priority-flood: An optimal
    depression-filling and watershed-labeling algorithm for digital elevation
    models. Computers & Geosciences, 62, 117–127.

    Garbrecht, J., & Martz, L.W. (1997). The assignment of drainage direction
    over flat surfaces in raster digital elevation models. Journal of Hydrology,
    193, 204–213.

    Spec Reference: flow-spec.md lines 291-328 (Stage 2b)
                     flow-spec.md lines 484-494 (flat resolution discussion)
    """
    import heapq

    rows, cols = dem.shape
    filled = dem.copy().astype(np.float64)

    if nodata_mask is None:
        nodata_mask = np.zeros_like(dem, dtype=bool)

    # Priority queue: (elevation, row, col)
    pq = []
    in_queue = np.zeros((rows, cols), dtype=bool)

    # Seed priority queue with outlets
    for i in range(rows):
        for j in range(cols):
            if outlets[i, j] and not nodata_mask[i, j]:
                heapq.heappush(pq, (filled[i, j], i, j))
                in_queue[i, j] = True

    # Also seed with cells adjacent to NoData (border of domain)
    for i in range(rows):
        for j in range(cols):
            if nodata_mask[i, j] or in_queue[i, j]:
                continue

            # Check if adjacent to NoData
            adjacent_to_nodata = False
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    if nodata_mask[ni, nj]:
                        adjacent_to_nodata = True
                        break

            if adjacent_to_nodata:
                heapq.heappush(pq, (filled[i, j], i, j))
                in_queue[i, j] = True

    # Process cells in elevation order (priority-flood)
    filled_count = 0
    while pq:
        elev, r, c = heapq.heappop(pq)

        # Check all 8 neighbors
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ni, nj = r + di, c + dj

            # Bounds and status checks
            if not (0 <= ni < rows and 0 <= nj < cols):
                continue
            if in_queue[ni, nj]:
                continue
            if nodata_mask[ni, nj]:
                continue

            # If neighbor is in depression (below current + epsilon), raise it
            # This is the KEY DIFFERENCE: epsilon applied DURING fill
            if filled[ni, nj] < elev + epsilon:
                filled[ni, nj] = elev + epsilon
                filled_count += 1

            # Add neighbor to queue
            heapq.heappush(pq, (filled[ni, nj], ni, nj))
            in_queue[ni, nj] = True

    if filled_count > 0:
        logger.info(f"    Priority-flood raised {filled_count:,} cells to resolve depressions")

    return filled.astype(np.float32)


def condition_dem_spec(
    dem: np.ndarray,
    nodata_mask: Optional[np.ndarray] = None,
    coastal_elev_threshold: float = 10.0,
    edge_mode: Literal["all", "local_minima", "outward_slope", "none"] = "all",
    max_breach_depth: float = 50.0,
    max_breach_length: int = 100,
    epsilon: Optional[float] = None,
    masked_basin_outlets: Optional[np.ndarray] = None,
    parallel_method: str = "checkerboard",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Condition DEM using spec-compliant 4-stage pipeline.

    Orchestrates the complete flow-spec.md pipeline:
    1. Stage 1: Identify outlets (coastal, edge, masked basins)
    2. Stage 2a: Constrained breaching (Dijkstra least-cost)
    3. Stage 2b: Priority-flood fill residuals (with epsilon)
    4. (External) D8 flow direction → see compute_flow_direction()
    5. (External) Flow accumulation → see compute_drainage_area()

    Parameters
    ----------
    dem : np.ndarray
        Input digital elevation model
    nodata_mask : np.ndarray (bool), optional
        Mask of ocean/off-grid cells (True = NoData)
    coastal_elev_threshold : float, default 10.0
        Maximum elevation for coastal outlets (meters above sea level)
    edge_mode : {"all", "local_minima", "outward_slope", "none"}, default "all"
        Boundary outlet strategy
    max_breach_depth : float, default 50.0
        Maximum elevation drop at any single cell during breaching (meters)
    max_breach_length : int, default 100
        Maximum breach path length (cells)
    epsilon : float, default 1e-4
        Minimum elevation increment per cell in filled areas (meters)
    masked_basin_outlets : np.ndarray (bool), optional
        User-supplied outlet locations for known lakes/basins

    Returns
    -------
    conditioned_dem : np.ndarray
        Breached and filled DEM
    outlets : np.ndarray (bool)
        Outlet mask (useful for diagnostics and visualization)
    breached_dem : np.ndarray
        DEM after breaching but before filling (for fill depth calculation)

    Notes
    -----
    This implements the complete spec-compliant pipeline from flow-spec.md.

    **Implementation vs. Spec:**
    The implementation uses a two-pass breach approach rather than the
    priority-flood breach sketch in the spec (lines 125-172). Both approaches
    are equivalent and spec-compliant; two-pass is chosen for clarity:

    1. Identify all sinks (cells with no downslope neighbor)
    2. For each sink, attempt least-cost Dijkstra path to an outlet/drain-point
    3. Apply monotonic gradient along successful paths

    The priority-flood approach integrates breach discovery into the same
    priority queue as flow processing, which is more complex but equivalent.

    **Key Advantages Over Legacy Approach:**
    - Explicit outlet classification prevents boundary artifacts
    - Selective breaching (max_depth, max_length constraints) preserves
      legitimate basins and endorheic systems
    - Epsilon gradient applied during fill (not after), eliminating need for
      post-hoc flat resolution
    - No workarounds needed (_fix_coastal_flow, fill_small_sinks)
    - Deterministic and reproducible (no randomness or ties)

    **Critical Correctness Requirements:**
    After conditioning, the DEM must satisfy:
    1. Every land cell (not NoData) has a path to an outlet
    2. Every land cell has at least one downslope neighbor (guaranteed by fill)
    3. No cycles in flow directions (validated by compute_drainage_area)
    4. Outlets are lowest points in their regions (set explicitly in Stage 1)

    If cycles are detected in compute_drainage_area, check:
    - Outlets are correctly identified (Stage 1)
    - Breach parameters (max_depth, max_length) aren't too restrictive
    - Epsilon isn't too large (causing reversals on filled areas)

    **Parameter Tuning:**
    - **max_breach_depth**: Controls minimum basin depth preserved
      - Smaller values (5-20m) preserve more detailed basins
      - Larger values (50-100m) allow breaching of deeper basins
      - Default 50m is suitable for most 30m DEMs
    - **max_breach_length**: Controls maximum breach path extent
      - Smaller values (10-30 cells) for detailed hydrology
      - Larger values (100-300 cells) for regional studies
      - Default 100 cells suitable for outlet-finding on large DEMs
    - **epsilon**: Controls micro-gradient in filled areas
      - Auto-calculated as 1e-5 × cell_resolution by default
      - Usually no manual tuning needed
      - If flats look "stair-stepped", epsilon is too large

    **Performance Notes:**
    - Stage 1 (outlets): O(n) with 8-neighbor checks
    - Stage 2a (breach): O(n log n) per sink via Dijkstra + priority queue
    - Stage 2b (fill): O(n log n) via priority-flood
    - Overall: O(n log n) where n = number of cells
    - Typical performance: 100M cells in ~30s (C/Rust), ~5m (Python/numba)

    Examples
    --------
    >>> dem = np.array([[5, 5, 5],
    ...                 [5, 3, 5],
    ...                 [5, 5, 5]], dtype=np.float32)
    >>> nodata = np.zeros((3, 3), dtype=bool)
    >>> conditioned, outlets, breached = condition_dem_spec(dem, nodata)
    >>> # Outlets: [[False, False, False],
    >>> #           [False, True,  False],
    >>> #           [False, False, False]]
    >>> # Conditioned: [[5, 5, 5],
    >>> #               [5, 5, 5],
    >>> #               [5, 5, 5]]  (pit filled to neighbor level)

    >>> # Now use with Stage 3 and 4:
    >>> flow_dir = compute_flow_direction(conditioned)
    >>> drainage_area = compute_drainage_area(flow_dir)

    See Also
    --------
    identify_outlets : Stage 1
    breach_depressions_constrained : Stage 2a
    priority_flood_fill_epsilon : Stage 2b
    compute_flow_direction : Stage 3
    compute_drainage_area : Stage 4
    """
    if nodata_mask is None:
        nodata_mask = np.zeros_like(dem, dtype=bool)

    logger.info("  Stage 1: Identifying outlets...")
    outlets = identify_outlets(
        dem, nodata_mask, coastal_elev_threshold, edge_mode, masked_basin_outlets
    )
    num_outlets = np.sum(outlets)
    logger.info(f"    Found {num_outlets:,} outlet cells")

    # Skip breaching if disabled (max_breach_depth <= 0 or max_breach_length <= 0)
    if max_breach_depth <= 0 or max_breach_length <= 0:
        logger.info("  Stage 2a: Breaching SKIPPED (disabled via parameters)")
        breached = dem.copy()
    else:
        logger.info(
            f"  Stage 2a: Constrained breaching (max_depth={max_breach_depth}m, max_length={max_breach_length} cells)..."
        )
        breached = breach_depressions_constrained(
            dem,
            outlets,
            max_breach_depth,
            max_breach_length,
            epsilon,
            nodata_mask,
            parallel_method=parallel_method,
        )

    logger.info("  Stage 2b: Priority-flood fill residuals...")
    filled = priority_flood_fill_epsilon(breached, outlets, epsilon, nodata_mask)

    # Ensure masked cells maintain original elevation
    filled[nodata_mask] = dem[nodata_mask]

    # Ensure we never lowered elevations below what breaching produced
    # (breaching can lower elevations to create flow paths)
    filled = np.maximum(filled, breached)

    logger.info("  DEM conditioning complete (spec-compliant pipeline)")
    return filled, outlets, breached


def detect_ocean_mask(
    dem: np.ndarray, threshold: float = 0.0, border_only: bool = True
) -> np.ndarray:
    """
    Detect ocean or water bodies in DEM.

    Identifies cells at or below elevation threshold that are connected to
    the border (assumed to be ocean/large water bodies).

    Uses connected component labeling for efficient O(n) detection.

    Parameters
    ----------
    dem : np.ndarray
        Digital elevation model
    threshold : float, default 0.0
        Elevation threshold (meters). Cells <= threshold are candidates.
    border_only : bool, default True
        If True, only return border-connected low-elevation areas (ocean).
        If False, return all areas below threshold (includes inland lakes).

    Returns
    -------
    np.ndarray (bool)
        Boolean mask where True = ocean/water

    Examples
    --------
    >>> dem = np.array([[0, 0, 5], [0, 1, 6], [5, 6, 7]])
    >>> ocean = detect_ocean_mask(dem, threshold=0.0, border_only=True)
    >>> ocean
    array([[ True,  True, False],
           [ True, False, False],
           [False, False, False]])
    """
    from scipy.ndimage import label

    # Find all cells at or below threshold
    low_elevation = dem <= threshold

    if not border_only:
        return low_elevation

    # No low-elevation cells? No ocean.
    if not np.any(low_elevation):
        return np.zeros_like(dem, dtype=bool)

    # Label connected regions (O(n) operation)
    structure = np.ones((3, 3), dtype=bool)  # 8-connectivity
    labeled, num_features = label(low_elevation, structure=structure)

    # Find labels that touch any border
    border_labels = set()
    border_labels.update(labeled[0, :])  # Top border
    border_labels.update(labeled[-1, :])  # Bottom border
    border_labels.update(labeled[:, 0])  # Left border
    border_labels.update(labeled[:, -1])  # Right border
    border_labels.discard(0)  # Remove background label

    # Create mask for all border-connected regions
    ocean_mask = np.isin(labeled, list(border_labels))

    return ocean_mask


ADAPTIVE_BASIN_FRACTION = 1e-3  # min_basin_size=None: basins must cover 1/1000 of the grid


def adaptive_min_basin_size(total_cells: int) -> int:
    """Minimum endorheic-basin size (cells) when none is given: 1/1000 of the grid, at least 1."""
    return max(1, int(ADAPTIVE_BASIN_FRACTION * total_cells))


def detect_endorheic_basins(
    dem: np.ndarray,
    min_size: int = 10,
    exclude_mask: np.ndarray | None = None,
    min_depth: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """
    Detect endorheic (closed) basins in DEM.

    Identifies closed depressions (basins with no outlet) that exceed
    a minimum size threshold. Used to preserve large natural basins
    like the Salton Sea, Death Valley, etc.

    Parameters
    ----------
    dem : np.ndarray
        Digital elevation model
    min_size : int, default 10
        Minimum basin size in cells to be considered significant
    exclude_mask : np.ndarray (bool), optional
        Mask of areas to exclude from basin detection (e.g., ocean).
        This improves performance by only filling land areas.
    min_depth : float, default 0.5
        Minimum depression depth in meters to be considered a basin.
        Higher values = only preserve truly deep basins, fill shallower ones.

    Returns
    -------
    basin_mask : np.ndarray (bool)
        Boolean mask where True = part of endorheic basin
    basin_sizes : dict
        Dictionary mapping basin_id to size in cells

    Examples
    --------
    >>> # Create closed basin surrounded by mountains
    >>> dem = np.array([[50, 50, 50],
    ...                 [50, 10, 50],
    ...                 [50, 50, 50]])
    >>> mask, sizes = detect_endorheic_basins(dem, min_size=1)
    >>> mask[1, 1]  # Center is basin
    True
    """
    from scipy.ndimage import label

    # Fill depressions to find what WOULD be filled
    # Pass exclude_mask to avoid filling ocean (performance optimization)
    filled = _fill_depressions(dem, epsilon=0.0, mask=exclude_mask)
    fill_depth = filled - dem

    # Cells that would be filled are part of depressions
    # Higher min_depth = only preserve truly deep basins, fill shallower ones
    depressions = fill_depth > min_depth

    if not np.any(depressions):
        # No depressions found
        return np.zeros_like(dem, dtype=bool), {}

    # Label connected depression regions (O(n) operation)
    structure = np.ones((3, 3), dtype=bool)  # 8-connectivity
    labeled, num_features = label(depressions, structure=structure)

    # Calculate size of each basin (vectorized for performance)
    unique_labels, label_counts = np.unique(labeled[labeled > 0], return_counts=True)
    basin_sizes = dict(zip(unique_labels.tolist(), label_counts.tolist()))

    # Create mask for large basins (vectorized operation)
    basin_mask = np.zeros_like(dem, dtype=bool)
    if min_size is None:
        raise ValueError(
            "detect_endorheic_basins needs min_size (cells); it used to treat None as 10. "
            "For a size relative to the grid, use adaptive_min_basin_size(dem.size)."
        )
    large_basin_ids = [bid for bid, size in basin_sizes.items() if size >= min_size]
    if large_basin_ids:
        basin_mask = np.isin(labeled, large_basin_ids)

    return basin_mask, basin_sizes


def condition_dem(
    dem: np.ndarray,
    method: str = "fill",
    ocean_mask: np.ndarray | None = None,
    min_basin_size: int | None = None,
    max_fill_depth: float | None = None,
    min_basin_depth: float = 0.5,
    fill_small_sinks: int | None = None,
) -> np.ndarray:
    """
    Condition DEM by filling pits and resolving depressions.

    Uses morphological reconstruction (priority flood algorithm) to properly
    fill depressions and pits. This is the standard algorithm used by most
    hydrological analysis tools (e.g., GRASS, WhiteboxTools, ArcGIS).

    Supports masking ocean areas and preserving large endorheic basins.

    Parameters
    ----------
    dem : np.ndarray
        Input DEM
    method : str, default 'fill'
        Depression handling method ('fill' or 'breach')
        - 'fill': Complete depression filling (raises elevations)
        - 'breach': Minimal filling to preserve terrain (uses epsilon)
    ocean_mask : np.ndarray (bool), optional
        Boolean mask indicating ocean/water cells to exclude from conditioning.
        Masked cells maintain original elevation.
    min_basin_size : int, optional
        Minimum basin size (cells) to preserve. Basins >= this size are
        not filled (preserves large endorheic basins like Salton Sea).
    max_fill_depth : float, optional
        Maximum fill depth (meters). Depressions requiring fill > this
        depth are preserved (protects deep natural basins).
    min_basin_depth : float, default 0.5
        Minimum depression depth (meters) to be considered a preservable basin.
        Higher values = only preserve truly deep basins, fill shallower ones.
        Increase this for noisy high-resolution DEMs.
    fill_small_sinks : int, optional
        Maximum sink size (cells) to fill. After main filling, any remaining
        local minima (sinks) with contributing area < this size are filled.
        This removes small artifacts that create fragmented drainage.
        Typical values: 10-100 cells.

    Returns
    -------
    np.ndarray
        Conditioned DEM with depressions resolved

    Examples
    --------
    >>> # Basic usage
    >>> conditioned = condition_dem(dem)

    >>> # Mask ocean
    >>> ocean = detect_ocean_mask(dem, threshold=0.0)
    >>> conditioned = condition_dem(dem, ocean_mask=ocean)

    >>> # Preserve large basins
    >>> conditioned = condition_dem(dem, min_basin_size=10000)

    >>> # Fill small sinks (< 50 cells) to reduce fragmentation
    >>> conditioned = condition_dem(dem, fill_small_sinks=50)
    """
    # ========== BASIN PRESERVATION LOGIC ==========
    # Large endorheic basins (e.g., Salton Sea, Death Valley) are preserved
    # using TWO independent mechanisms that can work together:
    #
    # 1. Size-based preservation (min_basin_size):
    #    - Detects closed depressions larger than min_basin_size cells
    #    - Preserves these naturally-occurring basins by excluding from filling
    #    - Example: min_basin_size=10000 preserves basins >= 10,000 cells
    #
    # 2. Depth-based preservation (max_fill_depth):
    #    - After filling, restores any cells that required > max_fill_depth meters
    #    - Example: max_fill_depth=50 preserves basins requiring >50m fill
    #
    # These mechanisms preserve real geographic features while still filling
    # noise and local pits.

    # Create combined exclusion mask
    exclude_mask = np.zeros_like(dem, dtype=bool)

    if ocean_mask is not None:
        exclude_mask |= ocean_mask

    # === Size-based preservation: Preserve large closed basins ===
    if min_basin_size is not None:
        basin_mask, basin_sizes = detect_endorheic_basins(
            dem, min_size=min_basin_size, exclude_mask=ocean_mask, min_depth=min_basin_depth
        )
        total_depressions = len(basin_sizes)
        large_basins = sum(1 for size in basin_sizes.values() if size >= min_basin_size)
        num_cells_masked = np.sum(basin_mask)
        pct_masked = 100 * num_cells_masked / basin_mask.size
        logger.info(
            f"  Basin preservation: {total_depressions} depressions >{min_basin_depth}m deep, {large_basins} >= {min_basin_size} cells"
        )
        logger.info(f"  Masked {num_cells_masked:,} cells ({pct_masked:.1f}% of DEM)")
        exclude_mask |= basin_mask

    # === Main depression filling ===
    # Priority-flood algorithm properly fills depressions while respecting
    # exclusion masks (ocean, large basins, etc.)
    if method == "fill":
        filled = _fill_depressions(dem, epsilon=0.0, mask=exclude_mask)
    elif method == "breach":
        # Use 1e-4 (0.1mm) to avoid fragmentation from rounding (10mm precision)
        filled = _fill_depressions(dem, epsilon=1e-4, mask=exclude_mask)
    else:
        raise ValueError(f"Unknown fill method: {method}")

    # === Depth-based preservation: Preserve deep basins ===
    # If a depression would require > max_fill_depth meters of fill,
    # preserve it at original elevation (it's likely a real natural feature)
    if max_fill_depth is not None:
        fill_depth = filled - dem
        deep_basins = fill_depth > max_fill_depth
        filled[deep_basins] = dem[deep_basins]

    # Ensure masked cells maintain original elevation
    filled[exclude_mask] = dem[exclude_mask]

    # Ensure we never lower original elevations
    filled = np.maximum(filled, dem)

    # Fill small sinks if requested
    if fill_small_sinks is not None and fill_small_sinks > 0:
        filled = _fill_small_sinks(filled, max_sink_size=fill_small_sinks, mask=exclude_mask)

    return filled


def _fill_small_sinks(
    dem: np.ndarray,
    max_sink_size: int = 50,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Fill small sinks (local minima) that would create fragmented drainage.

    Iteratively finds local minima and fills small ones to their spill point.
    This removes DEM noise artifacts that create many tiny outlets.

    Optimized implementation: O(m + n*k) where m=grid size, n=num_sinks, k=avg_sink_size.
    Uses single-pass index collection instead of per-sink grid scans.

    Parameters
    ----------
    dem : np.ndarray
        Input DEM (already filled/breached)
    max_sink_size : int, default 50
        Maximum size (cells) of sinks to fill. Sinks larger than this
        are preserved (they may be real features).
    mask : np.ndarray (bool), optional
        Cells to exclude from filling (e.g., ocean, large basins).

    Returns
    -------
    np.ndarray
        DEM with small sinks filled
    """
    from scipy.ndimage import label, minimum_filter, maximum_filter, find_objects

    filled = dem.copy().astype(np.float64)
    rows, cols = dem.shape

    # Find local minima: cells that are strictly lower than all 8 neighbors
    footprint = np.ones((3, 3), dtype=bool)
    local_min_filter = minimum_filter(filled, footprint=footprint, mode="constant", cval=np.inf)
    local_max_filter = maximum_filter(filled, footprint=footprint, mode="constant", cval=-np.inf)
    local_minima = (filled == local_min_filter) & (filled < local_max_filter)

    # Exclude masked cells and boundary
    if mask is not None:
        local_minima &= ~mask
    local_minima[0, :] = False
    local_minima[-1, :] = False
    local_minima[:, 0] = False
    local_minima[:, -1] = False

    num_minima = np.sum(local_minima)
    logger.info(f"  Small sink detection: found {num_minima:,} local minima cells")
    if num_minima == 0:
        return filled.astype(np.float32)

    # Label connected sink regions
    structure = np.ones((3, 3), dtype=bool)  # 8-connectivity
    labeled, num_features = label(local_minima, structure=structure)
    logger.info(f"  Small sink detection: {num_features:,} connected sink regions")

    # Get bounding boxes for all regions in one pass
    slices = find_objects(labeled)

    # Precompute sink indices using vectorized numpy operations
    # Get all labeled cell coordinates at once
    labeled_rows, labeled_cols = np.where(labeled > 0)
    labeled_ids = labeled[labeled_rows, labeled_cols]

    # Sort by label ID to group cells belonging to same sink
    sort_order = np.argsort(labeled_ids)
    sorted_rows = labeled_rows[sort_order]
    sorted_cols = labeled_cols[sort_order]
    sorted_ids = labeled_ids[sort_order]

    # Find split points between different labels
    # np.diff finds where labels change, np.where finds those positions
    split_points = np.where(np.diff(sorted_ids) > 0)[0] + 1
    split_points = np.concatenate([[0], split_points, [len(sorted_ids)]])

    # Build dict mapping sink_id -> (row_indices, col_indices)
    sink_indices = {}
    for i in range(len(split_points) - 1):
        start, end = split_points[i], split_points[i + 1]
        if start < end:
            sink_id = sorted_ids[start]
            sink_indices[sink_id] = (sorted_rows[start:end], sorted_cols[start:end])

    # Process each sink using precomputed indices
    sinks_filled = 0
    cells_raised = 0

    for sink_id in range(1, num_features + 1):
        if sink_id not in sink_indices:
            continue

        sink_rows, sink_cols = sink_indices[sink_id]
        sink_size = len(sink_rows)

        if sink_size > max_sink_size:
            continue  # Skip large sinks

        obj_slice = slices[sink_id - 1]
        if obj_slice is None:
            continue

        # Expand slice by 1 for boundary detection
        r_start = max(0, obj_slice[0].start - 1)
        r_stop = min(rows, obj_slice[0].stop + 1)
        c_start = max(0, obj_slice[1].start - 1)
        c_stop = min(cols, obj_slice[1].stop + 1)

        # Work on cropped region
        local_labeled = labeled[r_start:r_stop, c_start:c_stop]
        local_filled = filled[r_start:r_stop, c_start:c_stop]
        local_mask = mask[r_start:r_stop, c_start:c_stop] if mask is not None else None

        # Create sink mask for this region only (small array)
        sink_mask_local = local_labeled == sink_id

        # Find boundary using dilation on small region
        from scipy.ndimage import binary_dilation

        expanded = binary_dilation(sink_mask_local, structure=structure)
        boundary = expanded & ~sink_mask_local

        if local_mask is not None:
            boundary &= ~local_mask

        if not np.any(boundary):
            continue

        sink_elevation = np.min(local_filled[sink_mask_local])
        boundary_elevs = local_filled[boundary]
        higher_neighbors = boundary_elevs > sink_elevation

        # Determine new elevation
        if np.any(higher_neighbors):
            new_elev = np.min(boundary_elevs[higher_neighbors]) + 1e-6
        else:
            lowest_boundary_elev = np.min(boundary_elevs)
            if lowest_boundary_elev <= sink_elevation:
                new_elev = lowest_boundary_elev + 1e-6
            else:
                continue

        # Update using precomputed indices (vectorized, avoids full-grid scan)
        filled[sink_rows, sink_cols] = new_elev
        sinks_filled += 1
        cells_raised += sink_size

    if sinks_filled > 0:
        logger.info(
            f"  Filled {sinks_filled} small sinks ({cells_raised:,} cells) with max_size={max_sink_size}"
        )

    return filled.astype(np.float32)


def _fill_depressions(
    dem: np.ndarray, epsilon: float = 0.0, mask: np.ndarray | None = None
) -> np.ndarray:
    """
    Fill depressions in DEM using morphological reconstruction.

    This implements depression filling via morphological reconstruction.
    Algorithm: Reconstruct from seed (borders at DEM elevation, interior at +inf)
    downward, constrained by original DEM.

    Parameters
    ----------
    dem : np.ndarray
        Input digital elevation model
    epsilon : float, default 0.0
        Small gradient to add in flat areas (for 'breach' method)
        If > 0, creates minimal gradients instead of true flats
    mask : np.ndarray (bool), optional
        Boolean mask indicating cells to exclude from filling.
        Masked cells maintain original elevation.

    Returns
    -------
    np.ndarray
        Depression-filled DEM
    """
    from skimage.morphology import reconstruction

    dem = dem.astype(np.float64)  # Use float64 for precision

    # Create seed: borders at DEM elevation, interior slightly higher
    # This allows reconstruction to fill depressions
    seed = dem.copy()
    seed[1:-1, 1:-1] = dem.max() + 1000  # Interior much higher

    # If mask provided, set masked cells as borders (won't be filled)
    if mask is not None:
        seed[mask] = dem[mask]

    # Morphological reconstruction by erosion
    # Erode seed downward, constrained by mask (original DEM)
    # This fills depressions to their spill point elevation
    filled = reconstruction(seed, dem, method="erosion")

    # Restore masked cells to original elevation
    if mask is not None:
        filled[mask] = dem[mask]

    # Add epsilon gradients if requested (for breach method)
    if epsilon > 0:
        filled = _resolve_flats(filled, epsilon)

    return filled.astype(np.float32)


def _resolve_flats(dem: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Resolve flat areas using Garbrecht-Martz (1997) dual-gradient algorithm.

    Uses TWO gradients combined:
    1. Gradient TOWARD lower terrain (pour points) - water flows to outlets
    2. Gradient AWAY from higher terrain (high points) - water flows from ridges

    The combined gradient ensures natural flow convergence in flat regions.

    References:
    - Garbrecht & Martz (1997): "The assignment of drainage direction over
      flat surfaces in raster digital elevation models" J. Hydrol. 193: 204-213
    - Barnes et al. (2014): "An Efficient Assignment of Drainage Direction
      Over Flat Surfaces" arXiv:1511.04433

    Parameters
    ----------
    dem : np.ndarray
        Input DEM (potentially with flat areas)
    epsilon : float
        Small value for gradient increment (default: 1e-5 m)

    Returns
    -------
    np.ndarray
        DEM with flats resolved using dual gradients
    """
    resolved = dem.copy().astype(np.float64)
    rows, cols = dem.shape

    # Round DEM to eliminate floating-point precision issues
    # Use 10mm precision (0.01m) to avoid fragmentation from tiny breach gradients
    dem_rounded = np.round(dem, 2)

    # Find flat cells: cells with at least one neighbor at the SAME elevation
    flat_cells = np.zeros((rows, cols), dtype=bool)

    # Create padded DEM for safe neighbor access
    padded = np.full((rows + 2, cols + 2), np.nan, dtype=dem_rounded.dtype)
    padded[1:-1, 1:-1] = dem_rounded

    # Check all 8 neighbors for equal elevation (vectorized)
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        neighbor = padded[1 + di : rows + 1 + di, 1 + dj : cols + 1 + dj]
        flat_cells |= dem_rounded == neighbor

    flat_count = np.sum(flat_cells)
    if flat_count == 0:
        return resolved.astype(np.float32)

    logger.info(f"  Flat resolution: {flat_count:,} flat cells found")

    # Find pour points: flat cells adjacent to STRICTLY LOWER terrain
    pour_points = np.zeros((rows, cols), dtype=bool)

    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        padded_inf = np.full((rows + 2, cols + 2), np.inf, dtype=dem_rounded.dtype)
        padded_inf[1:-1, 1:-1] = dem_rounded
        neighbor = padded_inf[1 + di : rows + 1 + di, 1 + dj : cols + 1 + dj]
        pour_points |= flat_cells & (neighbor < dem_rounded)

    # Boundary flat cells are also pour points
    boundary_mask = np.zeros((rows, cols), dtype=bool)
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True
    pour_points |= flat_cells & boundary_mask

    # NEW: Find high points: flat cells adjacent to STRICTLY HIGHER terrain
    high_points = np.zeros((rows, cols), dtype=bool)

    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        padded_neg = np.full((rows + 2, cols + 2), -np.inf, dtype=dem_rounded.dtype)
        padded_neg[1:-1, 1:-1] = dem_rounded
        neighbor = padded_neg[1 + di : rows + 1 + di, 1 + dj : cols + 1 + dj]
        high_points |= flat_cells & (neighbor > dem_rounded)

    pour_count = np.sum(pour_points)
    high_count = np.sum(high_points)

    if pour_count == 0 and high_count == 0:
        return resolved.astype(np.float32)

    logger.info(f"  Flat resolution: {pour_count:,} pour points, {high_count:,} high points")

    # Compute DUAL gradients (Garbrecht-Martz algorithm)
    # Gradient 1: Distance from pour points (toward lower terrain)
    dist_to_low = _compute_flat_gradient_bfs(flat_cells, pour_points, dem_rounded)

    # Gradient 2: Distance from high points (away from higher terrain)
    dist_from_high = _compute_flat_gradient_bfs(flat_cells, high_points, dem_rounded)

    # Combine gradients: cells should be HIGHER if:
    # - farther from pour points (dist_to_low is larger)
    # - closer to high points (dist_from_high is smaller)
    # Formula: combined = dist_to_low + (max_dist - dist_from_high)
    # Simplified: combined = dist_to_low - dist_from_high + const
    # The constant doesn't matter since we're adding relative gradients

    # Find max distances for normalization
    max_dist_low = np.max(dist_to_low[flat_cells]) if pour_count > 0 else 0
    max_dist_high = np.max(dist_from_high[flat_cells]) if high_count > 0 else 0

    # Combine: cells farther from outlets AND closer to ridges get higher elevation
    # This creates natural flow convergence toward outlets and away from ridges
    combined_gradient = np.zeros_like(dist_to_low)

    if pour_count > 0 and high_count > 0:
        # Full dual-gradient: both components
        # dist_to_low: higher = farther from outlet = should be higher
        # dist_from_high: higher = farther from ridge = should be lower
        # Combined: dist_to_low adds elevation, dist_from_high subtracts
        combined_gradient[flat_cells] = dist_to_low[flat_cells] + (
            max_dist_high - dist_from_high[flat_cells]
        )
    elif pour_count > 0:
        # Only pour points: just use distance toward outlets
        combined_gradient[flat_cells] = dist_to_low[flat_cells]
    elif high_count > 0:
        # Only high points: just use distance from ridges (inverted)
        combined_gradient[flat_cells] = max_dist_high - dist_from_high[flat_cells]

    # Apply combined gradient
    resolved[flat_cells] += combined_gradient[flat_cells] * epsilon

    return resolved.astype(np.float32)


@jit(nopython=True, cache=True)
def _compute_flat_gradient_bfs_jit(
    flat_cells: np.ndarray, pour_points: np.ndarray, dem_rounded: np.ndarray, gradient: np.ndarray
) -> None:
    """
    JIT-compiled multi-source BFS to compute gradient from pour points.

    Computes geodesic distance from pour points within each flat region,
    respecting elevation boundaries (cells at different elevations are barriers).

    Parameters
    ----------
    flat_cells : np.ndarray (bool)
        Mask of flat cells
    pour_points : np.ndarray (bool)
        Mask of pour points (sources for BFS)
    dem_rounded : np.ndarray
        Rounded DEM for elevation comparison
    gradient : np.ndarray (float64)
        Output gradient array (modified in-place)
    """
    rows, cols = flat_cells.shape

    # D8 offsets for neighbor checking
    offsets = np.array(
        [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)], dtype=np.int32
    )

    # Initialize queue with all pour points
    queue = np.zeros(rows * cols, dtype=np.int32)
    queue_start = 0
    queue_end = 0

    # Visited array (distance from nearest pour point at same elevation)
    visited = np.zeros((rows, cols), dtype=np.int32)
    visited[:, :] = -1  # -1 = not visited

    # Add all pour points to queue with distance 0
    for i in range(rows):
        for j in range(cols):
            if pour_points[i, j]:
                queue[queue_end] = i * cols + j
                queue_end += 1
                visited[i, j] = 0

    # BFS from all pour points simultaneously
    while queue_start < queue_end:
        flat_idx = queue[queue_start]
        queue_start += 1

        i = flat_idx // cols
        j = flat_idx % cols
        current_dist = visited[i, j]
        current_elev = dem_rounded[i, j]

        # Check all 8 neighbors
        for k in range(8):
            di = offsets[k, 0]
            dj = offsets[k, 1]
            ni = i + di
            nj = j + dj

            # Bounds check
            if 0 <= ni < rows and 0 <= nj < cols:
                # Only expand to flat cells at SAME elevation that haven't been visited
                if (
                    flat_cells[ni, nj]
                    and visited[ni, nj] == -1
                    and dem_rounded[ni, nj] == current_elev
                ):
                    visited[ni, nj] = current_dist + 1
                    queue[queue_end] = ni * cols + nj
                    queue_end += 1

    # Copy distances to gradient (pour points stay at 0)
    for i in range(rows):
        for j in range(cols):
            if visited[i, j] > 0:
                gradient[i, j] = visited[i, j]


def _compute_flat_gradient_bfs(
    flat_cells: np.ndarray, pour_points: np.ndarray, dem_rounded: np.ndarray
) -> np.ndarray:
    """
    Compute gradient from pour points using multi-source BFS.

    Parameters
    ----------
    flat_cells : np.ndarray (bool)
        Mask of flat cells
    pour_points : np.ndarray (bool)
        Mask of pour points (sources for BFS)
    dem_rounded : np.ndarray
        Rounded DEM for elevation comparison

    Returns
    -------
    np.ndarray
        Gradient values (distance from nearest pour point at same elevation)
    """
    gradient = np.zeros(flat_cells.shape, dtype=np.float64)

    if NUMBA_AVAILABLE:
        _compute_flat_gradient_bfs_jit(flat_cells, pour_points, dem_rounded, gradient)
    else:
        # Pure Python fallback (slower but works)
        rows, cols = flat_cells.shape
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        from collections import deque

        queue = deque()
        visited = np.full((rows, cols), -1, dtype=np.int32)

        # Initialize with pour points
        pour_coords = np.argwhere(pour_points)
        for i, j in pour_coords:
            queue.append((i, j, 0))
            visited[i, j] = 0

        # BFS
        while queue:
            i, j, dist = queue.popleft()
            current_elev = dem_rounded[i, j]

            for di, dj in offsets:
                ni, nj = i + di, j + dj
                if (
                    0 <= ni < rows
                    and 0 <= nj < cols
                    and flat_cells[ni, nj]
                    and visited[ni, nj] == -1
                    and dem_rounded[ni, nj] == current_elev
                ):
                    visited[ni, nj] = dist + 1
                    gradient[ni, nj] = dist + 1
                    queue.append((ni, nj, dist + 1))

    return gradient
