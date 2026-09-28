"""Sink detection and least-cost constrained breaching of depressions."""

import logging
from typing import Tuple, Optional

import numpy as np

from terrain_maker.terrain._numba_compat import NUMBA_AVAILABLE, jit, prange

logger = logging.getLogger(__name__)


@jit(nopython=True, parallel=True, cache=True)
def _identify_sinks_jit(
    dem: np.ndarray,
    outlets: np.ndarray,
    nodata_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    JIT-compiled parallel sink identification (10-20x faster than pure Python).

    Uses numba parallel execution to check cells concurrently across multiple CPU cores.

    Returns
    -------
    sink_rows, sink_cols, sink_elevs : np.ndarray
        Arrays of sink coordinates and elevations, sorted by elevation
    """
    rows, cols = dem.shape

    # First pass: count sinks per row (parallel)
    row_sink_counts = np.zeros(rows, dtype=np.int32)

    for i in prange(rows):
        count = 0
        for j in range(cols):
            # Skip outlets and nodata
            if outlets[i, j] or nodata_mask[i, j]:
                continue

            # Check if any neighbor is lower
            has_downslope = False
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        if not nodata_mask[ni, nj] and dem[ni, nj] < dem[i, j]:
                            has_downslope = True
                            break
                if has_downslope:
                    break

            if not has_downslope:
                count += 1

        row_sink_counts[i] = count

    # Calculate total sinks and row offsets
    total_sinks = np.sum(row_sink_counts)
    row_offsets = np.zeros(rows + 1, dtype=np.int32)
    for i in range(rows):
        row_offsets[i + 1] = row_offsets[i] + row_sink_counts[i]

    # Allocate output arrays
    sink_rows = np.empty(total_sinks, dtype=np.int32)
    sink_cols = np.empty(total_sinks, dtype=np.int32)
    sink_elevs = np.empty(total_sinks, dtype=np.float64)

    # Second pass: fill sink arrays (parallel)
    for i in prange(rows):
        offset = row_offsets[i]
        local_count = 0

        for j in range(cols):
            # Skip outlets and nodata
            if outlets[i, j] or nodata_mask[i, j]:
                continue

            # Check if any neighbor is lower
            has_downslope = False
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        if not nodata_mask[ni, nj] and dem[ni, nj] < dem[i, j]:
                            has_downslope = True
                            break
                if has_downslope:
                    break

            if not has_downslope:
                sink_rows[offset + local_count] = i
                sink_cols[offset + local_count] = j
                sink_elevs[offset + local_count] = dem[i, j]
                local_count += 1

    # Sort by elevation (argsort)
    sort_indices = np.argsort(sink_elevs)
    return sink_rows[sort_indices], sink_cols[sort_indices], sink_elevs[sort_indices]


def _identify_sinks(
    dem: np.ndarray,
    outlets: np.ndarray,
    nodata_mask: Optional[np.ndarray] = None,
) -> list:
    """
    Identify all sink cells (cells with no downslope neighbor).

    Returns sinks sorted by elevation (lowest first) to reduce cascading breaches.

    Parameters
    ----------
    dem : np.ndarray
        Digital elevation model
    outlets : np.ndarray (bool)
        Outlet mask from identify_outlets()
    nodata_mask : np.ndarray (bool), optional
        Cells to exclude

    Returns
    -------
    list of tuples
        List of (row, col, elevation) for each sink, sorted by elevation ascending
    """
    if nodata_mask is None:
        nodata_mask = np.zeros_like(dem, dtype=bool)

    # Use JIT-compiled version if available
    if NUMBA_AVAILABLE:
        sink_rows, sink_cols, sink_elevs = _identify_sinks_jit(dem, outlets, nodata_mask)
        return [(int(r), int(c), float(e)) for r, c, e in zip(sink_rows, sink_cols, sink_elevs)]

    # Fallback to pure Python
    rows, cols = dem.shape
    sinks = []

    for i in range(rows):
        for j in range(cols):
            # Skip outlets and nodata
            if outlets[i, j] or nodata_mask[i, j]:
                continue

            # Check if any neighbor is lower
            has_downslope = False
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    if not nodata_mask[ni, nj] and dem[ni, nj] < dem[i, j]:
                        has_downslope = True
                        break

            if not has_downslope:
                sinks.append((i, j, dem[i, j]))

    # Sort by elevation (process lowest first to avoid cascading)
    sinks.sort(key=lambda x: x[2])
    return sinks


def _reconstruct_path(parent_map: dict, start_r: int, start_c: int, end_r: int, end_c: int) -> list:
    """
    Reconstruct breach path from Dijkstra parent map.

    Parameters
    ----------
    parent_map : dict
        Mapping from (row, col) to (parent_row, parent_col)
    start_r, start_c : int
        Starting cell (sink)
    end_r, end_c : int
        Ending cell (drain point)

    Returns
    -------
    list of tuples
        Path from sink to drain as [(row, col), ...] in order
    """
    path = []
    r, c = end_r, end_c

    while (r, c) != (None, None):
        path.append((r, c))
        if r == start_r and c == start_c:
            break
        r, c = parent_map.get((r, c), (None, None))

    path.reverse()
    return path


@jit(nopython=True, cache=True)
def _find_breach_path_dijkstra_jit(
    dem: np.ndarray,
    start_row: int,
    start_col: int,
    outlets: np.ndarray,
    resolved: np.ndarray,
    max_depth: float,
    max_length: int,
) -> Tuple[bool, np.ndarray, np.ndarray]:
    """
    JIT-compiled Dijkstra breach path finder (5-10x faster).

    Returns
    -------
    found : bool
        Whether a valid path was found
    path_rows, path_cols : np.ndarray
        Path coordinates (empty if not found)
    """
    start_elev = dem[start_row, start_col]
    rows, cols = dem.shape

    # Use arrays instead of dicts for JIT compatibility
    visited = np.zeros((rows, cols), dtype=np.bool_)
    parent_r = np.full((rows, cols), -1, dtype=np.int32)
    parent_c = np.full((rows, cols), -1, dtype=np.int32)

    # Simple priority queue (using lists, heapq not fully supported in nopython)
    # Store: (cost, length, row, col, parent_row, parent_col)
    # We'll use a simplified approach: expand lowest cost first
    max_queue_size = min(10000, rows * cols)  # Reasonable limit
    queue_costs = np.full(max_queue_size, np.inf, dtype=np.float64)
    queue_lengths = np.zeros(max_queue_size, dtype=np.int32)
    queue_rows = np.zeros(max_queue_size, dtype=np.int32)
    queue_cols = np.zeros(max_queue_size, dtype=np.int32)
    queue_parent_r = np.zeros(max_queue_size, dtype=np.int32)
    queue_parent_c = np.zeros(max_queue_size, dtype=np.int32)
    queue_size = 1
    queue_costs[0] = 0.0
    queue_lengths[0] = 0
    queue_rows[0] = start_row
    queue_cols[0] = start_col
    queue_parent_r[0] = -1
    queue_parent_c[0] = -1

    end_r, end_c = -1, -1
    found = False

    while queue_size > 0:
        # Find minimum cost item (simple linear search for now)
        min_idx = 0
        min_cost = queue_costs[0]
        for i in range(1, queue_size):
            if queue_costs[i] < min_cost:
                min_cost = queue_costs[i]
                min_idx = i

        # Pop item
        cost = queue_costs[min_idx]
        length = queue_lengths[min_idx]
        r = queue_rows[min_idx]
        c = queue_cols[min_idx]
        pr = queue_parent_r[min_idx]
        pc = queue_parent_c[min_idx]

        # Remove from queue (swap with last)
        queue_size -= 1
        if min_idx < queue_size:
            queue_costs[min_idx] = queue_costs[queue_size]
            queue_lengths[min_idx] = queue_lengths[queue_size]
            queue_rows[min_idx] = queue_rows[queue_size]
            queue_cols[min_idx] = queue_cols[queue_size]
            queue_parent_r[min_idx] = queue_parent_r[queue_size]
            queue_parent_c[min_idx] = queue_parent_c[queue_size]

        # Skip if already visited
        if visited[r, c]:
            continue

        visited[r, c] = True
        parent_r[r, c] = pr
        parent_c[r, c] = pc

        # Check termination: reached outlet
        if outlets[r, c]:
            # Verify outlet is not deeper than max_depth below sink
            # (to prevent breaching sink by more than max_depth)
            sink_lowering = start_elev - dem[r, c]
            if sink_lowering <= max_depth:
                end_r, end_c = r, c
                found = True
                break

        # Check termination: reached resolved cell at or below start elevation
        if resolved[r, c] and dem[r, c] <= start_elev:
            end_r, end_c = r, c
            found = True
            break

        # Check length constraint
        if length >= max_length:
            continue

        # Explore neighbors
        for di in range(-1, 2):
            for dj in range(-1, 2):
                if di == 0 and dj == 0:
                    continue

                ni, nj = r + di, c + dj

                # Bounds check
                if not (0 <= ni < rows and 0 <= nj < cols):
                    continue
                if visited[ni, nj]:
                    continue

                # Cost to breach through this neighbor
                breach_depth_here = max(0.0, dem[ni, nj] - start_elev)

                # Check depth constraint
                if breach_depth_here > max_depth:
                    continue

                new_cost = cost + breach_depth_here
                new_length = length + 1

                # Add to queue if space available
                if queue_size < max_queue_size:
                    queue_costs[queue_size] = new_cost
                    queue_lengths[queue_size] = new_length
                    queue_rows[queue_size] = ni
                    queue_cols[queue_size] = nj
                    queue_parent_r[queue_size] = r
                    queue_parent_c[queue_size] = c
                    queue_size += 1

    if not found:
        return False, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    # Reconstruct path
    path_list_r = []
    path_list_c = []
    curr_r, curr_c = end_r, end_c

    while curr_r != -1:
        path_list_r.append(curr_r)
        path_list_c.append(curr_c)
        if curr_r == start_row and curr_c == start_col:
            break
        next_r = parent_r[curr_r, curr_c]
        next_c = parent_c[curr_r, curr_c]
        curr_r, curr_c = next_r, next_c

    # Convert to arrays and reverse
    path_r = np.array(path_list_r[::-1], dtype=np.int32)
    path_c = np.array(path_list_c[::-1], dtype=np.int32)

    return True, path_r, path_c


@jit(nopython=True, cache=True)
def _dijkstra_single_sink(
    dem: np.ndarray,
    start_row: int,
    start_col: int,
    outlets: np.ndarray,
    max_depth: float,
    max_length: int,
    path_r_out: np.ndarray,
    path_c_out: np.ndarray,
) -> int:
    """
    Single sink Dijkstra worker for parallel processing.

    Similar to _find_breach_path_dijkstra_jit but:
    1. Ignores `resolved` array (for Phase 1 parallelism)
    2. Writes path to pre-allocated output arrays
    3. Returns path length (0 if not found)

    This allows calling from within prange loops.
    """
    start_elev = dem[start_row, start_col]
    rows, cols = dem.shape

    # Use arrays for JIT compatibility
    visited = np.zeros((rows, cols), dtype=np.bool_)
    parent_r = np.full((rows, cols), -1, dtype=np.int32)
    parent_c = np.full((rows, cols), -1, dtype=np.int32)

    # Simple priority queue
    max_queue_size = min(10000, rows * cols)
    queue_costs = np.full(max_queue_size, np.inf, dtype=np.float64)
    queue_lengths = np.zeros(max_queue_size, dtype=np.int32)
    queue_rows = np.zeros(max_queue_size, dtype=np.int32)
    queue_cols = np.zeros(max_queue_size, dtype=np.int32)
    queue_parent_r = np.zeros(max_queue_size, dtype=np.int32)
    queue_parent_c = np.zeros(max_queue_size, dtype=np.int32)
    queue_size = 1
    queue_costs[0] = 0.0
    queue_lengths[0] = 0
    queue_rows[0] = start_row
    queue_cols[0] = start_col
    queue_parent_r[0] = -1
    queue_parent_c[0] = -1

    end_r, end_c = -1, -1
    found = False

    while queue_size > 0:
        # Find minimum cost item
        min_idx = 0
        min_cost = queue_costs[0]
        for i in range(1, queue_size):
            if queue_costs[i] < min_cost:
                min_cost = queue_costs[i]
                min_idx = i

        # Pop item
        cost = queue_costs[min_idx]
        length = queue_lengths[min_idx]
        r = queue_rows[min_idx]
        c = queue_cols[min_idx]
        pr = queue_parent_r[min_idx]
        pc = queue_parent_c[min_idx]

        # Remove from queue (swap with last)
        queue_size -= 1
        if min_idx < queue_size:
            queue_costs[min_idx] = queue_costs[queue_size]
            queue_lengths[min_idx] = queue_lengths[queue_size]
            queue_rows[min_idx] = queue_rows[queue_size]
            queue_cols[min_idx] = queue_cols[queue_size]
            queue_parent_r[min_idx] = queue_parent_r[queue_size]
            queue_parent_c[min_idx] = queue_parent_c[queue_size]

        if visited[r, c]:
            continue

        visited[r, c] = True
        parent_r[r, c] = pr
        parent_c[r, c] = pc

        # Termination: reached outlet
        if outlets[r, c]:
            # Verify outlet is not deeper than max_depth below sink
            # (to prevent breaching sink by more than max_depth)
            sink_lowering = start_elev - dem[r, c]
            if sink_lowering <= max_depth:
                end_r, end_c = r, c
                found = True
                break

        # Length constraint
        if length >= max_length:
            continue

        # Explore neighbors
        for di in range(-1, 2):
            for dj in range(-1, 2):
                if di == 0 and dj == 0:
                    continue

                ni, nj = r + di, c + dj

                if not (0 <= ni < rows and 0 <= nj < cols):
                    continue
                if visited[ni, nj]:
                    continue

                breach_depth_here = max(0.0, dem[ni, nj] - start_elev)

                if breach_depth_here > max_depth:
                    continue

                new_cost = cost + breach_depth_here
                new_length = length + 1

                if queue_size < max_queue_size:
                    queue_costs[queue_size] = new_cost
                    queue_lengths[queue_size] = new_length
                    queue_rows[queue_size] = ni
                    queue_cols[queue_size] = nj
                    queue_parent_r[queue_size] = r
                    queue_parent_c[queue_size] = c
                    queue_size += 1

    if not found:
        return 0  # No path found

    # Reconstruct path into output arrays
    path_idx = 0
    curr_r, curr_c = end_r, end_c
    max_path_len = len(path_r_out)

    while curr_r != -1 and path_idx < max_path_len:
        path_r_out[path_idx] = curr_r
        path_c_out[path_idx] = curr_c
        path_idx += 1
        if curr_r == start_row and curr_c == start_col:
            break
        next_r = parent_r[curr_r, curr_c]
        next_c = parent_c[curr_r, curr_c]
        curr_r, curr_c = next_r, next_c

    # Reverse path in-place (sink -> outlet becomes sink -> outlet order)
    for i in range(path_idx // 2):
        j = path_idx - 1 - i
        path_r_out[i], path_r_out[j] = path_r_out[j], path_r_out[i]
        path_c_out[i], path_c_out[j] = path_c_out[j], path_c_out[i]

    return path_idx


@jit(nopython=True, parallel=True, cache=True)
def _breach_sinks_parallel_batch(
    dem: np.ndarray,
    sinks_r: np.ndarray,
    sinks_c: np.ndarray,
    outlets: np.ndarray,
    max_depth: float,
    max_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find breach paths for a batch of sinks in parallel using prange.

    This is Phase 1 of two-phase parallel processing:
    - Uses outlets only (no resolved array) for termination
    - Each sink's Dijkstra is independent
    - Sinks should be spatially distant (checkerboard partitioning)

    Parameters
    ----------
    dem : np.ndarray
        Digital elevation model
    sinks_r, sinks_c : np.ndarray
        Row and column coordinates of sinks to process
    outlets : np.ndarray (bool)
        Outlet mask
    max_depth : float
        Maximum breach depth per cell
    max_length : int
        Maximum path length

    Returns
    -------
    found : np.ndarray (bool)
        Whether path was found for each sink
    paths_r, paths_c : np.ndarray (n_sinks, max_length+1)
        Path coordinates for each sink (-1 for unused slots)
    path_lengths : np.ndarray (int)
        Actual path length for each sink
    """
    n_sinks = len(sinks_r)
    max_path_len = max_length + 1

    # Pre-allocate outputs
    found = np.zeros(n_sinks, dtype=np.bool_)
    paths_r = np.full((n_sinks, max_path_len), -1, dtype=np.int32)
    paths_c = np.full((n_sinks, max_path_len), -1, dtype=np.int32)
    path_lengths = np.zeros(n_sinks, dtype=np.int32)

    # Process each sink in parallel
    for i in prange(n_sinks):
        sink_r = sinks_r[i]
        sink_c = sinks_c[i]

        # Run Dijkstra for this sink
        path_len = _dijkstra_single_sink(
            dem, sink_r, sink_c, outlets, max_depth, max_length, paths_r[i], paths_c[i]
        )

        found[i] = path_len > 0
        path_lengths[i] = path_len

    return found, paths_r, paths_c, path_lengths


def _cluster_sinks_checkerboard(
    sinks: list,
    grid_size: int,
    dem_shape: Tuple[int, int],
) -> Tuple[list, list]:
    """
    Cluster sinks into two batches using checkerboard pattern.

    Divides the DEM into grid cells of size `grid_size`. Sinks in
    "black" cells (even row+col) go in batch 1, "white" cells (odd)
    in batch 2. This ensures sinks in the same batch are at least
    `grid_size` cells apart diagonally.

    Parameters
    ----------
    sinks : list
        List of (row, col, elev, depth) tuples
    grid_size : int
        Size of grid cells (should be >= max_breach_length)
    dem_shape : tuple
        Shape of DEM (rows, cols)

    Returns
    -------
    batch_black, batch_white : list
        Two lists of sinks for parallel processing
    """
    batch_black = []  # Even grid cells
    batch_white = []  # Odd grid cells

    for sink in sinks:
        r, c, elev, depth = sink
        grid_r = r // grid_size
        grid_c = c // grid_size

        if (grid_r + grid_c) % 2 == 0:
            batch_black.append(sink)
        else:
            batch_white.append(sink)

    return batch_black, batch_white


def _find_breach_path_dijkstra(
    dem: np.ndarray,
    start_row: int,
    start_col: int,
    outlets: np.ndarray,
    resolved: np.ndarray,
    max_depth: float,
    max_length: int,
) -> Optional[list]:
    """
    Find least-cost breach path from sink to draining cell using Dijkstra.

    Cost metric: Total elevation that must be removed along the path.

    Termination conditions:
    1. Reached an outlet cell
    2. Reached a resolved cell with elevation <= start_elev
    3. No more cells to explore within constraints (breach failed)

    Parameters
    ----------
    dem : np.ndarray
        Digital elevation model
    start_row, start_col : int
        Sink cell coordinates
    outlets : np.ndarray (bool)
        Outlet mask
    resolved : np.ndarray (bool)
        Tracks which cells already have drainage paths
    max_depth : float
        Maximum elevation drop allowed at any single cell (meters)
    max_length : int
        Maximum path length (cells)

    Returns
    -------
    list of tuples or None
        Breach path [(row, col), ...] from sink to drain, or None if failed
    """
    # Use JIT-compiled version if available
    if NUMBA_AVAILABLE:
        found, path_r, path_c = _find_breach_path_dijkstra_jit(
            dem, start_row, start_col, outlets, resolved, max_depth, max_length
        )
        if found:
            return [(int(r), int(c)) for r, c in zip(path_r, path_c)]
        else:
            return None

    # Fallback to pure Python with heapq
    import heapq

    start_elev = dem[start_row, start_col]
    rows, cols = dem.shape

    # Priority queue: (cost, length, row, col, parent_row, parent_col)
    pq = [(0, 0, start_row, start_col, None, None)]
    visited = {}  # {(row, col): cost}
    parent_map = {}  # {(row, col): (parent_row, parent_col)}

    while pq:
        cost, length, r, c, pr, pc = heapq.heappop(pq)

        # Skip if already visited with lower cost
        if (r, c) in visited:
            continue
        visited[(r, c)] = cost
        parent_map[(r, c)] = (pr, pc)

        # Check termination: reached outlet
        if outlets[r, c]:
            # Verify outlet is not deeper than max_depth below sink
            # (to prevent breaching sink by more than max_depth)
            sink_lowering = start_elev - dem[r, c]
            if sink_lowering <= max_depth:
                return _reconstruct_path(parent_map, start_row, start_col, r, c)

        # Check termination: reached resolved cell at or below start elevation
        if resolved[r, c] and dem[r, c] <= start_elev:
            return _reconstruct_path(parent_map, start_row, start_col, r, c)

        # Check length constraint
        if length >= max_length:
            continue  # Don't expand further from this cell

        # Explore neighbors
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ni, nj = r + di, c + dj

            # Bounds check
            if not (0 <= ni < rows and 0 <= nj < cols):
                continue
            if (ni, nj) in visited:
                continue

            # Cost to breach through this neighbor
            # Must carve down to start_elev (or below)
            breach_depth_here = max(0, dem[ni, nj] - start_elev)

            # Check depth constraint
            if breach_depth_here > max_depth:
                continue  # Would exceed max breach depth at this cell

            new_cost = cost + breach_depth_here
            new_length = length + 1

            heapq.heappush(pq, (new_cost, new_length, ni, nj, r, c))

    # No viable path found within constraints
    return None


def _apply_breach(dem: np.ndarray, path: list, epsilon: float) -> None:
    """
    Carve breach path into DEM with monotonic gradient.

    Works backward from drain point to sink, ensuring each cell is epsilon
    lower than the previous cell in the path.

    Only lowers cells, never raises them.

    Parameters
    ----------
    dem : np.ndarray
        Digital elevation model (modified in-place)
    path : list of tuples
        Breach path [(row, col), ...] from sink to drain
    epsilon : float
        Small gradient for breach paths (meters per cell)
    """
    n = len(path)
    if n < 2:
        return

    # Start from drain point (end of path)
    drain_r, drain_c = path[-1]
    base_elev = dem[drain_r, drain_c]

    # Work backward toward sink
    for i in range(n - 2, -1, -1):
        r, c = path[i]
        # Each cell should be epsilon higher than the next cell in path
        required_elev = base_elev + epsilon * (n - 1 - i)
        # Only lower cells, never raise them
        if dem[r, c] > required_elev:
            dem[r, c] = required_elev


def breach_depressions_constrained(
    dem: np.ndarray,
    outlets: np.ndarray,
    max_breach_depth: float = 50.0,
    max_breach_length: int = 100,
    epsilon: float = 1e-4,
    nodata_mask: Optional[np.ndarray] = None,
    parallel_method: str = "checkerboard",
) -> np.ndarray:
    """
    Remove depressions via constrained breaching (Stage 2a of flow-spec.md).

    Uses Lindsay (2016) constrained least-cost breaching algorithm.
    Implements two-pass approach:
    1. Identify all sinks (cells with no downslope neighbor)
    2. For each sink, attempt Dijkstra breach within constraints

    Parameters
    ----------
    dem : np.ndarray
        Input digital elevation model
    outlets : np.ndarray (bool)
        Outlet mask from identify_outlets()
    max_breach_depth : float, default 50.0
        Maximum elevation drop allowed at any single cell (meters)
    max_breach_length : int, default 100
        Maximum breach path length (cells)
    epsilon : float, default 1e-4
        Small gradient for breach paths (meters per cell)
    nodata_mask : np.ndarray (bool), optional
        Cells to exclude from breaching
    parallel_method : {"checkerboard", "iterative"}, default "checkerboard"
        Parallelization strategy for breach path finding:
        - "checkerboard": Two-phase parallel processing using checkerboard partitioning.
          Fast but can only terminate at outlets (misses chaining opportunities).
        - "iterative": Iterative refinement that runs multiple rounds, allowing each
          round's breaches to become termination targets for the next round. Finds
          more breaches via chaining but may be slower.

    Returns
    -------
    np.ndarray
        DEM with depressions breached where possible

    Notes
    -----
    This implements Stage 2a of flow-spec.md (lines 115-289).

    **Outlet Virtual Elevation:**
    Outlets (identified in Stage 1) act as guaranteed sinks in the breach algorithm.
    Breach paths MUST terminate at an outlet or at a cell that can drain naturally.
    Outlets are marked in the `resolved` set before breaching starts, ensuring
    they are always considered valid drainage targets.

    **Breach Path Termination:**
    Breach paths from sinks can terminate at:
    1. An outlet cell (guaranteed sink)
    2. A cell lower than the sink (natural downslope)
    3. A cell that has already been resolved (has a path to an outlet)

    This ensures the final flow network has no artificial endorheic basins.

    **Residual Sinks:**
    Sinks that cannot be breached within constraints (max_breach_depth, max_breach_length)
    are left for Stage 2b (priority-flood fill) to handle. These are typically:
    - Large legitimate basins (lakes, endorheic systems)
    - Depressions too deep/long to breach efficiently

    References
    ----------
    Lindsay, J.B. (2016). Efficient hybrid breaching-filling sink removal
    methods for flow path enforcement in digital elevation models.
    Hydrological Processes, 30, 846–857.

    Spec Reference: flow-spec.md lines 42-108 (outlet identification),
    115-289 (constrained breach)
    """
    breached = dem.copy().astype(np.float64)
    rows, cols = dem.shape

    if nodata_mask is None:
        nodata_mask = np.zeros_like(dem, dtype=bool)

    # Track which cells have drainage paths
    resolved = outlets.copy()

    logger.info("  Stage 2a: Identifying sinks...")

    # Show threading info
    if NUMBA_AVAILABLE:
        try:
            from numba import get_num_threads

            num_threads = get_num_threads()
            logger.info(f"    Numba: {num_threads} threads available for parallel operations")
        except:
            pass

    sinks = _identify_sinks(breached, outlets, nodata_mask)
    logger.info(f"    Found {len(sinks):,} sink cells")

    if len(sinks) == 0:
        logger.info("    No sinks to breach")
        return breached.astype(np.float32)

    # Precompute fill depths to identify shallow sinks (optimization)
    logger.info("    Computing sink depths...")
    from skimage.morphology import reconstruction

    filled_preview = breached.copy().astype(np.float64)
    seed = filled_preview.copy()
    seed[1:-1, 1:-1] = filled_preview.max() + 1000
    if nodata_mask is not None:
        seed[nodata_mask] = filled_preview[nodata_mask]
    filled_preview = reconstruction(seed, filled_preview, method="erosion")

    # Calculate depth for each sink
    sink_depths = []
    for sink_r, sink_c, sink_elev in sinks:
        depth = filled_preview[sink_r, sink_c] - sink_elev
        sink_depths.append(depth)

    # Filter to significant sinks (> 1.0m depth)
    # Shallow sinks will be handled by Stage 2b (priority-flood fill)
    depth_threshold = 1.0  # meters
    significant_sinks = [
        (sink_r, sink_c, sink_elev, depth)
        for (sink_r, sink_c, sink_elev), depth in zip(sinks, sink_depths)
        if depth > depth_threshold
    ]

    shallow_skipped = len(sinks) - len(significant_sinks)
    logger.info(f"    Skipping {shallow_skipped:,} shallow sinks (<{depth_threshold}m deep)")
    logger.info(
        f"    Processing {len(significant_sinks):,} significant sinks (>{depth_threshold}m deep)"
    )

    logger.info(
        f"  Stage 2a: Attempting constrained breaching (max_depth={max_breach_depth}m, max_length={max_breach_length} cells)..."
    )

    total_sinks = len(significant_sinks)
    breached_count = 0
    failed_count = 0
    already_resolved_count = 0

    # Estimate if parallel is useful: only helps when sinks are near outlets
    # With large max_breach_length and large DEM, most sinks are interior
    # and parallel (outlets-only) will find ~0 breaches - wasted effort
    avg_dim = (rows + cols) / 2
    parallel_useful = max_breach_length < avg_dim / 4  # Sinks likely near outlets

    if not parallel_useful and NUMBA_AVAILABLE and parallel_method != "iterative":
        logger.info(
            f"    Note: max_breach_length ({max_breach_length}) is large relative to DEM ({rows}x{cols})"
        )
        logger.info(f"    Using serial JIT (better for interior sinks that chain together)")

    # Iterative refinement parallel processing
    if parallel_method == "iterative" and NUMBA_AVAILABLE and total_sinks > 0:
        # Iterative refinement: run parallel batches repeatedly, updating terminals each round
        # This allows chaining: sinks resolved in round N become terminals for round N+1
        try:
            from numba import get_num_threads

            num_threads = get_num_threads()
            logger.info(
                f"    Using iterative refinement parallel breaching ({num_threads} CPU cores)"
            )
        except:
            logger.info(f"    Using iterative refinement parallel breaching")

        grid_size = 2 * max_breach_length
        remaining_sinks = list(significant_sinks)
        iteration = 0
        max_iterations = 100  # Safety limit

        while remaining_sinks and iteration < max_iterations:
            iteration += 1
            iteration_breached = 0

            # Current terminals = outlets + all resolved cells from previous iterations
            terminals = outlets | resolved

            # Cluster remaining sinks using checkerboard pattern
            batch_black, batch_white = _cluster_sinks_checkerboard(
                remaining_sinks, grid_size, (rows, cols)
            )

            logger.info(f"    Iteration {iteration}: {len(remaining_sinks):,} sinks remaining...")

            newly_resolved_sinks = []

            for batch, phase_name in [(batch_black, "black"), (batch_white, "white")]:
                if not batch:
                    continue

                # Prepare arrays for batch
                batch_r = np.array([s[0] for s in batch], dtype=np.int32)
                batch_c = np.array([s[1] for s in batch], dtype=np.int32)

                # Run parallel Dijkstra with current terminals
                found, paths_r, paths_c, path_lengths = _breach_sinks_parallel_batch(
                    breached, batch_r, batch_c, terminals, max_breach_depth, max_breach_length
                )

                # Apply successful breaches
                for i in range(len(batch)):
                    sink_r, sink_c = batch_r[i], batch_c[i]

                    if resolved[sink_r, sink_c]:
                        already_resolved_count += 1
                        newly_resolved_sinks.append((sink_r, sink_c))
                        continue

                    if found[i]:
                        path_len = path_lengths[i]
                        path = [(int(paths_r[i, j]), int(paths_c[i, j])) for j in range(path_len)]
                        _apply_breach(breached, path, epsilon)
                        for r, c in path:
                            resolved[r, c] = True
                        breached_count += 1
                        iteration_breached += 1
                        newly_resolved_sinks.append((sink_r, sink_c))

            # Remove resolved sinks from remaining list
            resolved_set = set(newly_resolved_sinks)
            remaining_sinks = [s for s in remaining_sinks if (s[0], s[1]) not in resolved_set]

            logger.info(
                f"      Breached {iteration_breached:,} sinks this iteration, {len(remaining_sinks):,} remaining"
            )

            # If no progress, stop iterating
            if iteration_breached == 0:
                logger.info(f"    No new breaches in iteration {iteration}, stopping refinement")
                break

        # Count remaining as failed
        failed_count = len(remaining_sinks)
        if failed_count > 0:
            logger.info(
                f"    {failed_count:,} sinks could not be breached (will be filled in Stage 2b)"
            )

    # Use two-phase parallel processing with numba prange (checkerboard method)
    elif (
        parallel_method == "checkerboard"
        and NUMBA_AVAILABLE
        and total_sinks > 100
        and parallel_useful
    ):
        # Two-phase parallel processing: uses numba prange for true multi-core parallelism
        try:
            from numba import get_num_threads

            num_threads = get_num_threads()
            logger.info(f"    Using two-phase parallel breaching ({num_threads} CPU cores)")
        except:
            logger.info(f"    Using two-phase parallel breaching")

        # Cluster sinks using checkerboard pattern (grid_size = 2 * max_breach_length)
        # Sinks in same batch are guaranteed to be far enough apart that paths won't overlap
        grid_size = 2 * max_breach_length
        batch_black, batch_white = _cluster_sinks_checkerboard(
            significant_sinks, grid_size, (rows, cols)
        )
        logger.info(
            f"    Checkerboard clustering: {len(batch_black):,} black cells, {len(batch_white):,} white cells"
        )

        # Sub-batch size for progress reporting (process in chunks)
        # Use 1% intervals like serial version (~100 updates per phase)
        sub_batch_size = max(10, total_sinks // 100)

        def process_batch_with_progress(
            batch, phase_name, start_breached, start_failed, start_resolved
        ):
            """Process a batch in sub-batches with progress reporting."""
            nonlocal breached, resolved
            batch_breached = start_breached
            batch_failed = start_failed
            batch_resolved = start_resolved
            n = len(batch)

            for chunk_start in range(0, n, sub_batch_size):
                chunk_end = min(chunk_start + sub_batch_size, n)
                chunk = batch[chunk_start:chunk_end]

                # Prepare arrays for this chunk
                chunk_r = np.array([s[0] for s in chunk], dtype=np.int32)
                chunk_c = np.array([s[1] for s in chunk], dtype=np.int32)

                # Run parallel Dijkstra on chunk
                found, paths_r, paths_c, path_lengths = _breach_sinks_parallel_batch(
                    breached, chunk_r, chunk_c, outlets, max_breach_depth, max_breach_length
                )

                # Apply successful breaches
                for i in range(len(chunk)):
                    sink_r, sink_c = chunk_r[i], chunk_c[i]
                    if resolved[sink_r, sink_c]:
                        batch_resolved += 1
                        continue

                    if found[i]:
                        path_len = path_lengths[i]
                        path = [(int(paths_r[i, j]), int(paths_c[i, j])) for j in range(path_len)]
                        _apply_breach(breached, path, epsilon)
                        for r, c in path:
                            resolved[r, c] = True
                        batch_breached += 1
                    else:
                        batch_failed += 1

                # Progress report
                processed = chunk_end
                percent = 100.0 * processed / n
                logger.info(
                    f"      {phase_name}: {percent:5.1f}% ({processed:,}/{n:,}) - {batch_breached:,} breached, {batch_failed:,} failed"
                )

            return batch_breached, batch_failed, batch_resolved

        # Phase 1: Process "black" cells in parallel sub-batches
        if len(batch_black) > 0:
            logger.info(f"    Phase 1: Processing {len(batch_black):,} sinks...")
            breached_count, failed_count, already_resolved_count = process_batch_with_progress(
                batch_black, "Phase 1", breached_count, failed_count, already_resolved_count
            )

        # Phase 2: Process "white" cells in parallel sub-batches
        if len(batch_white) > 0:
            logger.info(f"    Phase 2: Processing {len(batch_white):,} sinks...")
            breached_count, failed_count, already_resolved_count = process_batch_with_progress(
                batch_white, "Phase 2", breached_count, failed_count, already_resolved_count
            )

        # Note: Failed sinks are left for Stage 2b (priority-flood) to handle
        # Priority-flood is O(n log n) and much faster than re-running Dijkstra

    else:
        # Serial processing for small sink counts or when numba unavailable
        if NUMBA_AVAILABLE:
            logger.info(f"    Using serial JIT-compiled Dijkstra (small sink count)")
        else:
            logger.info(f"    Using serial pure-Python Dijkstra (numba unavailable)")

        # Progress reporting
        if total_sinks > 0:
            progress_interval = max(1, total_sinks // 100)  # Report every 1%
            logger.info(f"    Progress (showing every 1%):")

        for idx, (sink_r, sink_c, sink_elev, depth) in enumerate(significant_sinks):
            # Progress reporting
            if total_sinks > 0 and idx % progress_interval == 0:
                percent = 100.0 * idx / total_sinks
                logger.info(
                    f"      {percent:5.1f}% ({idx:,} / {total_sinks:,} sinks, {breached_count:,} breached, {failed_count:,} failed)"
                )

            # Skip if already resolved by previous breach
            if resolved[sink_r, sink_c]:
                already_resolved_count += 1
                continue

            # Attempt to find breach path
            path = _find_breach_path_dijkstra(
                breached, sink_r, sink_c, outlets, resolved, max_breach_depth, max_breach_length
            )

            if path is not None:
                # Apply breach
                _apply_breach(breached, path, epsilon)

                # Mark all cells in path as resolved
                for r, c in path:
                    resolved[r, c] = True

                breached_count += 1
            else:
                # Could not breach within constraints
                failed_count += 1

        if total_sinks > 0:
            logger.info(
                f"      100.0% ({total_sinks:,} / {total_sinks:,} sinks, {breached_count:,} breached, {failed_count:,} failed)"
            )

    logger.info(
        f"    Results: {breached_count:,} breached, {already_resolved_count:,} already resolved, {failed_count:,} failed"
    )
    logger.info(
        f"    Total sinks handled: {shallow_skipped:,} shallow (skipped) + {breached_count:,} breached + {already_resolved_count:,} resolved = {shallow_skipped + breached_count + already_resolved_count:,}"
    )
    logger.info(f"    Remaining for Stage 2b: {failed_count + shallow_skipped:,} sinks")

    return breached.astype(np.float32)
