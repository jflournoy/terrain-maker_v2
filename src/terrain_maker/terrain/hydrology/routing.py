"""D8 flow directions (ESRI power-of-2 encoding), coastal fixes and outlets."""

import logging
from typing import Optional, Literal

import numpy as np

from terrain_maker.terrain._numba_compat import NUMBA_AVAILABLE, jit

logger = logging.getLogger(__name__)


# ==============================================================================
# D8 FLOW DIRECTION ENCODING (ESRI ArcGIS Convention)
# ==============================================================================
#
# This module uses ESRI's standard D8 power-of-2 encoding for compatibility
# with GIS workflows. Each direction is assigned a unique power of 2, allowing
# bitwise operations and multi-directional flow calculations.
#
# D8 Neighbor Geometry:
#   8  4  2
#  16  x  1
#  32 64 128
#
# Direction codes and their meanings:
#   1 = East (→)     : (0, +1), distance = 1
#   2 = Northeast ↗ : (-1, +1), distance = sqrt(2)
#   4 = North (↑)    : (-1, 0), distance = 1
#   8 = Northwest ↖ : (-1, -1), distance = sqrt(2)
#  16 = West (←)    : (0, -1), distance = 1
#  32 = Southwest ↙ : (+1, -1), distance = sqrt(2)
#  64 = South (↓)   : (+1, 0), distance = 1
# 128 = Southeast ↘ : (+1, +1), distance = sqrt(2)
#
# Reference: ArcGIS D8 Direction (https://pro.arcgis.com/...)
#           flow-spec.md, Section 1 (Data Structures)

# D8 flow direction encoding: (row_offset, col_offset) -> direction_code
# Directions follow ESRI's power-of-2 encoding (standard in ArcGIS/GRASS)
D8_DIRECTIONS = {
    (0, 1): 1,  # East
    (-1, 1): 2,  # Northeast (diagonal, distance = sqrt(2))
    (-1, 0): 4,  # North
    (-1, -1): 8,  # Northwest (diagonal, distance = sqrt(2))
    (0, -1): 16,  # West
    (1, -1): 32,  # Southwest (diagonal, distance = sqrt(2))
    (1, 0): 64,  # South
    (1, 1): 128,  # Southeast (diagonal, distance = sqrt(2))
}


# Reverse mapping for flow routing: direction_code -> (row_offset, col_offset)
D8_OFFSETS = {v: k for k, v in D8_DIRECTIONS.items()}


@jit(nopython=True, cache=True)
def _compute_flow_direction_jit(dem: np.ndarray, flow_dir: np.ndarray) -> None:
    """
    JIT-compiled flow direction computation (numba accelerated).

    Modifies flow_dir in-place for maximum performance.

    Parameters
    ----------
    dem : np.ndarray
        Digital elevation model
    flow_dir : np.ndarray
        Output array for flow directions (modified in-place)
    """
    rows, cols = dem.shape

    # D8 direction offsets and codes (explicit for numba)
    offsets = np.array(
        [
            (0, 1),  # East: 1
            (-1, 1),  # Northeast: 2
            (-1, 0),  # North: 4
            (-1, -1),  # Northwest: 8
            (0, -1),  # West: 16
            (1, -1),  # Southwest: 32
            (1, 0),  # South: 64
            (1, 1),  # Southeast: 128
        ],
        dtype=np.int32,
    )

    codes = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)

    # Pre-compute distance factors for diagonal vs cardinal neighbors
    distances = np.array([1.0, 1.414, 1.0, 1.414, 1.0, 1.414, 1.0, 1.414], dtype=np.float32)

    # For each cell, find steepest downslope neighbor
    for i in range(rows):
        for j in range(cols):
            max_slope = 0.0
            best_dir = 0

            current_elev = dem[i, j]

            # Check all 8 neighbors
            # Priority order for tie-breaking: S, E, W, N, SE, SW, NE, NW
            # This ensures consistent flow direction when slopes are equal
            priority_order = np.array([6, 0, 4, 2, 7, 5, 1, 3], dtype=np.int32)

            for p in range(8):
                k = priority_order[p]
                di, dj = offsets[k]
                ni = i + di
                nj = j + dj

                # Check bounds
                if 0 <= ni < rows and 0 <= nj < cols:
                    neighbor_elev = dem[ni, nj]
                    slope = (current_elev - neighbor_elev) / distances[k]

                    # Use >= for first priority direction, > for others
                    # This ensures consistent tie-breaking
                    if slope > max_slope:
                        max_slope = slope
                        best_dir = codes[k]

            # Handle pits and boundary outlets
            # A pit is an interior cell where all neighbors are higher
            # Pits become outlets (flow_dir = 0) to avoid cycles
            if best_dir == 0:
                # No downslope neighbor found - this is a pit or boundary
                # Mark as outlet (flow_dir = 0)
                best_dir = 0

            flow_dir[i, j] = best_dir


def compute_flow_direction(dem: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """
    Compute D8 flow direction from DEM (Stage 3 of flow-spec.md).

    Assigns each cell the direction of steepest descent, using ESRI's
    power-of-2 D8 encoding for compatibility with GIS workflows.

    Parameters
    ----------
    dem : np.ndarray
        Digital elevation model (2D array, already conditioned by Stages 1-2)
    mask : np.ndarray (bool), optional
        Boolean mask indicating cells to exclude from flow computation.
        Masked cells (ocean, lakes, etc.) will have flow_dir = 0 (no flow).

    Returns
    -------
    np.ndarray (uint8)
        Flow direction encoded as D8 using ESRI power-of-2 convention:
        - 0 = outlet or masked cell (no downstream flow)
        - 1,2,4,8,16,32,64,128 = ESRI D8 codes (see module docstring)

    Notes
    -----
    This implements Stage 3 of flow-spec.md (lines 347-390).

    D8 Direction Encoding (ESRI ArcGIS):
      8  4  2
     16  x  1
     32 64 128

    Direction codes represent powers of 2 for bitwise compatibility:
    - 1=East, 2=NE, 4=North, 8=NW, 16=West, 32=SW, 64=South, 128=SE

    Distances:
    - Orthogonal (1,4,16,64): distance = 1
    - Diagonal (2,8,32,128): distance = sqrt(2)

    Flow Direction Selection:
    Each cell flows toward the neighbor with maximum slope (elevation drop
    per unit distance). If no neighbor is lower, the cell is an outlet
    (flow_dir = 0). This should not occur after proper DEM conditioning
    (Stage 2) unless the cell is an outlet or masked.

    See Also
    --------
    identify_outlets : Stage 1 (outlet identification)
    breach_depressions_constrained : Stage 2a (depression breaching)
    priority_flood_fill_epsilon : Stage 2b (depression filling)
    compute_drainage_area : Stage 4 (flow accumulation)
    """
    rows, cols = dem.shape
    flow_dir = np.zeros((rows, cols), dtype=np.uint8)

    # Use JIT-compiled version if available (10-20x faster)
    if NUMBA_AVAILABLE:
        _compute_flow_direction_jit(dem, flow_dir)
        # Apply mask after computation
        if mask is not None:
            flow_dir[mask] = 0

        # CRITICAL FIX: Force land cells adjacent to masked cells (ocean/sinks)
        # to flow toward the masked cell, even if there's no downslope.
        # This ensures watersheds properly drain to ocean/water bodies.
        if mask is not None:
            _fix_coastal_flow_directions(flow_dir, mask)

        return flow_dir

    # Fallback: Pure Python implementation
    for i in range(rows):
        for j in range(cols):
            max_slope = 0.0
            best_dir = 0

            current_elev = dem[i, j]

            # Check all 8 neighbors
            for (di, dj), direction_code in D8_DIRECTIONS.items():
                ni, nj = i + di, j + dj

                # Check bounds
                if 0 <= ni < rows and 0 <= nj < cols:
                    neighbor_elev = dem[ni, nj]
                    slope = (current_elev - neighbor_elev) / np.sqrt(di**2 + dj**2)

                    if slope > max_slope:
                        max_slope = slope
                        best_dir = direction_code

            # Handle pits and boundary outlets
            # A pit is an interior cell where all neighbors are higher
            # Pits become outlets (flow_dir = 0) to avoid cycles
            if best_dir == 0:
                # No downslope neighbor found - this is a pit or boundary
                # Mark as outlet (flow_dir = 0)
                best_dir = 0

            flow_dir[i, j] = best_dir

    # Apply mask
    if mask is not None:
        flow_dir[mask] = 0

    return flow_dir


@jit(nopython=True, cache=True)
def _fix_coastal_flow_directions_jit(flow_dir: np.ndarray, mask: np.ndarray) -> int:
    """
    JIT-compiled coastal flow direction fix.

    Only fixes pit cells (flow_dir == 0) adjacent to masked cells.
    Does not override valid flow directions.

    Parameters
    ----------
    flow_dir : np.ndarray (uint8)
        Flow direction grid (modified in-place)
    mask : np.ndarray (bool)
        Mask of ocean/sink cells

    Returns
    -------
    int
        Number of cells fixed
    """
    rows, cols = flow_dir.shape

    # D8 offsets and codes (explicit arrays for numba)
    offsets = np.array(
        [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)], dtype=np.int32
    )
    codes = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)

    fixed_count = 0

    for i in range(rows):
        for j in range(cols):
            # Skip if already masked (ocean/sink)
            if mask[i, j]:
                continue

            # Only fix cells that are pits (flow_dir == 0)
            # Don't override valid flow directions
            if flow_dir[i, j] != 0:
                continue

            # Check if any neighbor is masked (ocean/sink)
            for k in range(8):
                di = offsets[k, 0]
                dj = offsets[k, 1]
                ni = i + di
                nj = j + dj

                if 0 <= ni < rows and 0 <= nj < cols and mask[ni, nj]:
                    # Found adjacent ocean/sink - point flow toward it
                    flow_dir[i, j] = codes[k]
                    fixed_count += 1
                    break

    return fixed_count


def _fix_coastal_flow_directions(flow_dir: np.ndarray, mask: np.ndarray) -> None:
    """
    Fix pit cells adjacent to masked cells (ocean/sinks) to flow toward them.

    Only modifies cells with flow_dir == 0 (pits). This ensures coastal pits
    drain to ocean while preserving valid inland flow directions.

    This prevents coastal pits from fragmenting the drainage network.

    Parameters
    ----------
    flow_dir : np.ndarray
        Flow direction grid (modified in-place)
    mask : np.ndarray (bool)
        Mask of ocean/sink cells
    """
    if NUMBA_AVAILABLE:
        fixed_count = _fix_coastal_flow_directions_jit(flow_dir, mask)
    else:
        # Pure Python fallback
        rows, cols = flow_dir.shape
        D8_OFFSETS = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]
        D8_CODES = [1, 2, 4, 8, 16, 32, 64, 128]

        fixed_count = 0
        for i in range(rows):
            for j in range(cols):
                if mask[i, j]:
                    continue
                # Only fix cells that are pits (flow_dir == 0)
                if flow_dir[i, j] != 0:
                    continue
                for (di, dj), code in zip(D8_OFFSETS, D8_CODES):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols and mask[ni, nj]:
                        flow_dir[i, j] = code
                        fixed_count += 1
                        break

    logger.info(f"  Fixed {fixed_count:,} coastal cells to flow toward ocean/sinks")


def identify_outlets(
    dem: np.ndarray,
    nodata_mask: np.ndarray,
    coastal_elev_threshold: float = 10.0,
    edge_mode: Literal["all", "local_minima", "outward_slope", "none"] = "all",
    masked_basin_outlets: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Identify drainage outlet cells (Stage 1 of flow-spec.md).

    Classifies cells that act as drainage termini where water leaves the system.
    Implements three outlet types: coastal, edge, and masked basin outlets.

    Parameters
    ----------
    dem : np.ndarray
        Digital elevation model
    nodata_mask : np.ndarray (bool)
        Mask of ocean/off-grid cells (True = NoData)
    coastal_elev_threshold : float, default 10.0
        Maximum elevation for coastal outlets (meters above sea level).
        Prevents high cliffs adjacent to ocean from being spurious outlets.
    edge_mode : {"all", "local_minima", "outward_slope", "none"}, default "all"
        Boundary outlet strategy:
        - "all": All boundary cells are outlets (safest, prevents artificial basins)
        - "local_minima": Only boundary cells that are local minima
        - "outward_slope": Boundary cells with interior neighbors sloping toward them
        - "none": No edge outlets (for islands fully surrounded by coastline)
    masked_basin_outlets : np.ndarray (bool), optional
        User-supplied outlet locations for known lakes/basins

    Returns
    -------
    np.ndarray (bool)
        Boolean mask where True = outlet cell

    Notes
    -----
    This implements Stage 1 of flow-spec.md (lines 42-108).

    Edge mode "all" is the safest default - it ensures no artificial endorheic
    basins form at boundaries. The cost is some fragmentation of edge drainage
    networks, but this is usually preferable to missed outlets.

    Examples
    --------
    >>> dem = np.array([[5, 5, 5],
    ...                 [5, 3, 5],
    ...                 [5, 5, 5]], dtype=np.float32)
    >>> nodata = np.array([[False, False, False],
    ...                    [True,  False, False],
    ...                    [False, False, False]])
    >>> outlets = identify_outlets(dem, nodata, coastal_elev_threshold=10.0)
    >>> outlets[1, 1]  # Low coastal cell adjacent to ocean
    True
    """
    rows, cols = dem.shape
    outlets = np.zeros((rows, cols), dtype=bool)

    # --- Coastal outlets ---
    # Land cells adjacent to NoData AND elevation <= threshold
    for i in range(rows):
        for j in range(cols):
            # Skip if already NoData
            if nodata_mask[i, j]:
                continue

            # Check if adjacent to NoData (8-connected)
            adjacent_to_nodata = False
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    if nodata_mask[ni, nj]:
                        adjacent_to_nodata = True
                        break

            # Mark as coastal outlet if low enough
            if adjacent_to_nodata and dem[i, j] <= coastal_elev_threshold:
                outlets[i, j] = True

    # --- Edge outlets ---
    if edge_mode == "all":
        # All boundary cells are outlets
        outlets[0, :] = True  # Top edge
        outlets[-1, :] = True  # Bottom edge
        outlets[:, 0] = True  # Left edge
        outlets[:, -1] = True  # Right edge
        # Don't override NoData cells
        outlets[nodata_mask] = False

    elif edge_mode == "local_minima":
        # Only boundary cells that are local minima among edge neighbors
        # Top edge
        for j in range(cols):
            if nodata_mask[0, j]:
                continue
            is_min = True
            # Check edge neighbors (left, right, below)
            for dj in [-1, 0, 1]:
                nj = j + dj
                if 0 <= nj < cols and nj != j:
                    if not nodata_mask[0, nj] and dem[0, nj] < dem[0, j]:
                        is_min = False
                        break
            # Check interior neighbor (below)
            if rows > 1 and not nodata_mask[1, j]:
                if dem[1, j] < dem[0, j]:
                    is_min = False
            if is_min:
                outlets[0, j] = True

        # Bottom edge
        for j in range(cols):
            if nodata_mask[-1, j]:
                continue
            is_min = True
            for dj in [-1, 0, 1]:
                nj = j + dj
                if 0 <= nj < cols and nj != j:
                    if not nodata_mask[-1, nj] and dem[-1, nj] < dem[-1, j]:
                        is_min = False
                        break
            # Check interior neighbor (above)
            if rows > 1 and not nodata_mask[-2, j]:
                if dem[-2, j] < dem[-1, j]:
                    is_min = False
            if is_min:
                outlets[-1, j] = True

        # Left edge
        for i in range(rows):
            if nodata_mask[i, 0]:
                continue
            is_min = True
            for di in [-1, 0, 1]:
                ni = i + di
                if 0 <= ni < rows and ni != i:
                    if not nodata_mask[ni, 0] and dem[ni, 0] < dem[i, 0]:
                        is_min = False
                        break
            # Check interior neighbor (right)
            if cols > 1 and not nodata_mask[i, 1]:
                if dem[i, 1] < dem[i, 0]:
                    is_min = False
            if is_min:
                outlets[i, 0] = True

        # Right edge
        for i in range(rows):
            if nodata_mask[i, -1]:
                continue
            is_min = True
            for di in [-1, 0, 1]:
                ni = i + di
                if 0 <= ni < rows and ni != i:
                    if not nodata_mask[ni, -1] and dem[ni, -1] < dem[i, -1]:
                        is_min = False
                        break
            # Check interior neighbor (left)
            if cols > 1 and not nodata_mask[i, -2]:
                if dem[i, -2] < dem[i, -1]:
                    is_min = False
            if is_min:
                outlets[i, -1] = True

    elif edge_mode == "outward_slope":
        # Boundary cells with interior neighbors sloping more steeply toward edge
        # than toward any other neighbor
        # Top edge
        for j in range(cols):
            if nodata_mask[0, j]:
                continue
            if rows > 1 and not nodata_mask[1, j]:
                # Interior neighbor below
                slope_to_edge = (dem[1, j] - dem[0, j]) / 1.0
                # Check if this is steepest slope from interior cell
                max_slope_elsewhere = -np.inf
                for di, dj in [(-1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    ni, nj = 1 + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        if not nodata_mask[ni, nj]:
                            dist = np.sqrt(di**2 + dj**2)
                            slope = (dem[1, j] - dem[ni, nj]) / dist
                            max_slope_elsewhere = max(max_slope_elsewhere, slope)
                if slope_to_edge > max_slope_elsewhere:
                    outlets[0, j] = True

        # Similar logic for other edges (bottom, left, right)
        # Bottom edge
        for j in range(cols):
            if nodata_mask[-1, j]:
                continue
            if rows > 1 and not nodata_mask[-2, j]:
                slope_to_edge = (dem[-2, j] - dem[-1, j]) / 1.0
                max_slope_elsewhere = -np.inf
                for di, dj in [(1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    ni, nj = -2 + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        if not nodata_mask[ni, nj]:
                            dist = np.sqrt(di**2 + dj**2)
                            slope = (dem[-2, j] - dem[ni, nj]) / dist
                            max_slope_elsewhere = max(max_slope_elsewhere, slope)
                if slope_to_edge > max_slope_elsewhere:
                    outlets[-1, j] = True

        # Left edge
        for i in range(rows):
            if nodata_mask[i, 0]:
                continue
            if cols > 1 and not nodata_mask[i, 1]:
                slope_to_edge = (dem[i, 1] - dem[i, 0]) / 1.0
                max_slope_elsewhere = -np.inf
                for di, dj in [(0, -1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    ni, nj = i + di, 1 + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        if not nodata_mask[ni, nj]:
                            dist = np.sqrt(di**2 + dj**2)
                            slope = (dem[i, 1] - dem[ni, nj]) / dist
                            max_slope_elsewhere = max(max_slope_elsewhere, slope)
                if slope_to_edge > max_slope_elsewhere:
                    outlets[i, 0] = True

        # Right edge
        for i in range(rows):
            if nodata_mask[i, -1]:
                continue
            if cols > 1 and not nodata_mask[i, -2]:
                slope_to_edge = (dem[i, -2] - dem[i, -1]) / 1.0
                max_slope_elsewhere = -np.inf
                for di, dj in [(0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    ni, nj = i + di, -2 + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        if not nodata_mask[ni, nj]:
                            dist = np.sqrt(di**2 + dj**2)
                            slope = (dem[i, -2] - dem[ni, nj]) / dist
                            max_slope_elsewhere = max(max_slope_elsewhere, slope)
                if slope_to_edge > max_slope_elsewhere:
                    outlets[i, -1] = True

    elif edge_mode == "none":
        # No edge outlets (for islands)
        pass

    # --- Masked basin outlets ---
    if masked_basin_outlets is not None:
        outlets |= masked_basin_outlets

    return outlets
