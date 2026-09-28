"""Drainage area, upstream rainfall and discharge from D8 directions."""

import logging

import numpy as np

from terrain_maker.terrain._numba_compat import NUMBA_AVAILABLE, jit

from terrain_maker.terrain.hydrology.routing import D8_OFFSETS

logger = logging.getLogger(__name__)


@jit(nopython=True, cache=True)
def _compute_drainage_area_jit(flow_dir: np.ndarray, drainage_area: np.ndarray) -> np.bool_:
    """
    JIT-compiled drainage area computation (numba accelerated).

    Uses array-based topological sorting (Kahn's algorithm) for massive speedup
    over dict/set approach. Implements Stage 4 of flow-spec.md with cycle detection.

    Parameters
    ----------
    flow_dir : np.ndarray
        D8 flow direction grid (ESRI codes: 0,1,2,4,8,16,32,64,128)
    drainage_area : np.ndarray
        Output array (modified in-place), initialized to 1.0

    Returns
    -------
    bool
        True if a cycle was detected (should raise error in caller),
        False if topological sort completed successfully
    """
    rows, cols = flow_dir.shape

    # D8 direction offsets (ESRI convention)
    # Maps direction codes 1,2,4,8,16,32,64,128 to (dy, dx) offsets
    offsets = np.array(
        [
            (0, 1),  # 1: East
            (-1, 1),  # 2: Northeast
            (-1, 0),  # 4: North
            (-1, -1),  # 8: Northwest
            (0, -1),  # 16: West
            (1, -1),  # 32: Southwest
            (1, 0),  # 64: South
            (1, 1),  # 128: Southeast
        ],
        dtype=np.int32,
    )

    codes = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)

    # Count how many cells flow INTO each cell (in_degree in topological sort)
    contributor_count = np.zeros((rows, cols), dtype=np.int32)

    for i in range(rows):
        for j in range(cols):
            direction = flow_dir[i, j]
            if direction > 0:
                # Find which offset corresponds to this direction
                for k in range(8):
                    if codes[k] == direction:
                        di, dj = offsets[k]
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            contributor_count[ni, nj] += 1
                        break

    # Create queue of cells to process (cells with 0 contributors = ridgelines/outlets)
    # Use flat indexing for efficiency
    queue_size = 0
    queue = np.zeros(rows * cols, dtype=np.int32)

    for i in range(rows):
        for j in range(cols):
            if contributor_count[i, j] == 0:
                queue[queue_size] = i * cols + j
                queue_size += 1

    # Track cells processed for cycle detection
    cells_processed = 0
    initial_queue_size = queue_size

    # Process cells in topological order (Kahn's algorithm)
    queue_pos = 0
    while queue_pos < queue_size:
        # Dequeue
        flat_idx = queue[queue_pos]
        queue_pos += 1
        cells_processed += 1

        i = flat_idx // cols
        j = flat_idx % cols

        # Get receiver
        direction = flow_dir[i, j]
        if direction > 0:
            # Find offset for this direction
            for k in range(8):
                if codes[k] == direction:
                    di, dj = offsets[k]
                    ni, nj = i + di, j + dj

                    if 0 <= ni < rows and 0 <= nj < cols:
                        # Accumulate drainage area
                        drainage_area[ni, nj] += drainage_area[i, j]

                        # Decrement contributor count
                        contributor_count[ni, nj] -= 1

                        # If all contributors processed, add to queue
                        if contributor_count[ni, nj] == 0:
                            queue[queue_size] = ni * cols + nj
                            queue_size += 1
                    break
        # Note: Cells with direction==0 (outlets/ocean) are processed but don't
        # send flow anywhere. They act as sinks that accumulate drainage from
        # upstream but have no outflow.

    # Cycle detection: count total non-masked cells that should have been processed
    # (all cells have valid flow_dir or are outlets with direction=0)
    total_cells_with_flow = 0
    for i in range(rows):
        for j in range(cols):
            if flow_dir[i, j] >= 0:  # Valid flow direction or outlet
                total_cells_with_flow += 1

    # If not all cells were processed, there's a cycle
    # Note: cells_processed includes initial queue, so compare to total
    cycle_detected = cells_processed < total_cells_with_flow

    return cycle_detected


def compute_drainage_area(flow_dir: np.ndarray) -> np.ndarray:
    """
    Compute drainage area (unweighted flow accumulation).

    Stage 4 of flow-spec.md pipeline. Uses topological sort (Kahn's algorithm) to
    traverse the flow network and accumulate drainage areas.

    Parameters
    ----------
    flow_dir : np.ndarray
        D8 flow direction grid (ESRI codes: 0,1,2,4,8,16,32,64,128)
        where 0 indicates outlet/nodata (no downstream flow)

    Returns
    -------
    np.ndarray
        Number of cells draining through each pixel (including itself)

    Raises
    ------
    RuntimeError
        If a cycle is detected in the flow network, indicating a bug in DEM conditioning.
        Cycles should never occur after proper Stage 2 (DEM conditioning).

    Notes
    -----
    Implementation matches flow-spec.md Stage 4 (lines 392-438).
    Uses topological sort with cycle detection to ensure each cell's
    contributions are computed before it contributes to its receiver.

    If cycle detection fires, check that:
    - DEM was properly conditioned (Stage 2a: breaching, 2b: filling)
    - All outlets are properly identified (Stage 1)
    - No invalid flow directions exist (Stage 3)
    """
    rows, cols = flow_dir.shape
    drainage_area = np.ones((rows, cols), dtype=np.float32)

    # Use JIT-compiled version if available (50-100x faster)
    if NUMBA_AVAILABLE:
        cycle_detected = _compute_drainage_area_jit(flow_dir, drainage_area)
        if cycle_detected:
            raise RuntimeError(
                "Cycle detected in flow network! DEM conditioning failed. "
                "Check that outlets are properly identified (Stage 1) and "
                "DEM is properly conditioned (Stage 2a: breach, 2b: fill)."
            )
        return drainage_area

    # Fallback: Pure Python implementation with dict/set
    # Build flow network: for each cell, track which cells flow INTO it
    contributors = {}  # contributors[cell] = list of cells that flow to it
    receivers = {}  # receivers[cell] = cell it flows to

    for i in range(rows):
        for j in range(cols):
            direction = flow_dir[i, j]
            if direction > 0 and direction in D8_OFFSETS:
                di, dj = D8_OFFSETS[direction]
                receiver = (i + di, j + dj)
                if 0 <= receiver[0] < rows and 0 <= receiver[1] < cols:
                    receivers[(i, j)] = receiver
                    # Track that (i,j) contributes to receiver
                    if receiver not in contributors:
                        contributors[receiver] = []
                    contributors[receiver].append((i, j))

    # Process cells using topological sort (Kahn's algorithm)
    # Cells with no contributors (in_degree=0) are processed first
    processed = set()
    to_process = []

    # Find all cells with no contributors (ridgelines/peaks/outlets)
    for i in range(rows):
        for j in range(cols):
            if (i, j) not in contributors:
                to_process.append((i, j))

    # Process in topological order
    while to_process:
        cell = to_process.pop(0)
        if cell in processed:
            continue

        i, j = cell
        processed.add(cell)

        # Add this cell's drainage area to its receiver
        receiver = receivers.get(cell)
        if receiver is not None:
            ri, rj = receiver
            drainage_area[ri, rj] += drainage_area[i, j]

            # Check if all contributors to receiver are now processed
            receiver_contributors = contributors.get(receiver, [])
            if all(c in processed for c in receiver_contributors):
                to_process.append(receiver)

    # Cycle detection: ensure all land cells were processed
    total_land_cells = np.sum((flow_dir >= 0).astype(int))  # cells with valid flow or outlet
    if len(processed) < total_land_cells:
        unprocessed = total_land_cells - len(processed)
        raise RuntimeError(
            f"Cycle detected in flow network! {unprocessed} cells never reached in_degree 0. "
            "This indicates a bug in DEM conditioning (Stage 2). "
            "Check outlets (Stage 1) and breaching/filling (Stage 2a/2b)."
        )

    return drainage_area


@jit(nopython=True, cache=True)
def _compute_upstream_rainfall_jit(
    flow_dir: np.ndarray, precipitation: np.ndarray, upstream_rainfall: np.ndarray
) -> None:
    """
    JIT-compiled upstream rainfall computation (numba accelerated).

    Uses array-based topological sorting for massive speedup over dict/set approach.

    Parameters
    ----------
    flow_dir : np.ndarray
        D8 flow direction grid
    precipitation : np.ndarray
        Annual precipitation (mm/year)
    upstream_rainfall : np.ndarray
        Output array (modified in-place), initialized to precipitation values
    """
    rows, cols = flow_dir.shape

    # D8 direction offsets
    offsets = np.array(
        [
            (0, 1),  # 1: East
            (-1, 1),  # 2: Northeast
            (-1, 0),  # 4: North
            (-1, -1),  # 8: Northwest
            (0, -1),  # 16: West
            (1, -1),  # 32: Southwest
            (1, 0),  # 64: South
            (1, 1),  # 128: Southeast
        ],
        dtype=np.int32,
    )

    codes = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)

    # Count contributors
    contributor_count = np.zeros((rows, cols), dtype=np.int32)

    for i in range(rows):
        for j in range(cols):
            direction = flow_dir[i, j]
            if direction > 0:
                for k in range(8):
                    if codes[k] == direction:
                        di, dj = offsets[k]
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            contributor_count[ni, nj] += 1
                        break

    # Initialize queue with ridgelines (cells with 0 contributors)
    queue_size = 0
    queue = np.zeros(rows * cols, dtype=np.int32)

    for i in range(rows):
        for j in range(cols):
            if contributor_count[i, j] == 0:
                queue[queue_size] = i * cols + j
                queue_size += 1

    # Process in topological order
    queue_pos = 0
    while queue_pos < queue_size:
        flat_idx = queue[queue_pos]
        queue_pos += 1

        i = flat_idx // cols
        j = flat_idx % cols

        # Get receiver
        direction = flow_dir[i, j]
        if direction > 0:
            for k in range(8):
                if codes[k] == direction:
                    di, dj = offsets[k]
                    ni, nj = i + di, j + dj

                    if 0 <= ni < rows and 0 <= nj < cols:
                        # Accumulate upstream rainfall
                        upstream_rainfall[ni, nj] += upstream_rainfall[i, j]

                        # Decrement contributor count
                        contributor_count[ni, nj] -= 1

                        # If ready, add to queue
                        if contributor_count[ni, nj] == 0:
                            queue[queue_size] = ni * cols + nj
                            queue_size += 1
                    break


def compute_upstream_rainfall(flow_dir: np.ndarray, precipitation: np.ndarray) -> np.ndarray:
    """
    Compute precipitation-weighted flow accumulation.

    Parameters
    ----------
    flow_dir : np.ndarray
        D8 flow direction grid
    precipitation : np.ndarray
        Annual precipitation (mm/year)

    Returns
    -------
    np.ndarray
        Total upstream precipitation (mm·m²) at each pixel
        Represents cumulative rainfall from entire upstream area
    """
    rows, cols = flow_dir.shape
    upstream_rainfall = precipitation.copy().astype(np.float32)

    # Use JIT-compiled version if available (50-100x faster)
    if NUMBA_AVAILABLE:
        _compute_upstream_rainfall_jit(flow_dir, precipitation, upstream_rainfall)
        return upstream_rainfall

    # Fallback: Pure Python implementation with dict/set
    # Build flow network
    contributors = {}
    receivers = {}

    for i in range(rows):
        for j in range(cols):
            direction = flow_dir[i, j]
            if direction > 0 and direction in D8_OFFSETS:
                di, dj = D8_OFFSETS[direction]
                receiver = (i + di, j + dj)
                if 0 <= receiver[0] < rows and 0 <= receiver[1] < cols:
                    receivers[(i, j)] = receiver
                    if receiver not in contributors:
                        contributors[receiver] = []
                    contributors[receiver].append((i, j))

    # Topological sort: process from ridgelines to outlets
    processed = set()
    to_process = []

    # Find cells with no contributors (ridgelines)
    for i in range(rows):
        for j in range(cols):
            if (i, j) not in contributors:
                to_process.append((i, j))

    # Process in topological order
    while to_process:
        cell = to_process.pop(0)
        if cell in processed:
            continue

        i, j = cell
        processed.add(cell)

        # Add this cell's upstream rainfall to its receiver
        receiver = receivers.get(cell)
        if receiver is not None:
            ri, rj = receiver
            upstream_rainfall[ri, rj] += upstream_rainfall[i, j]

            # Check if receiver is ready to process
            receiver_contributors = contributors.get(receiver, [])
            if all(c in processed for c in receiver_contributors):
                to_process.append(receiver)

    return upstream_rainfall


def compute_discharge_potential(
    drainage_area: np.ndarray,
    upstream_rainfall: np.ndarray,
) -> np.ndarray:
    """
    Compute discharge potential combining drainage area and rainfall.

    Discharge potential represents where the largest water flows occur,
    combining topographic convergence (drainage area) with climate (rainfall).
    Higher values indicate locations where both large watersheds AND high
    precipitation combine to produce significant discharge.

    Parameters
    ----------
    drainage_area : np.ndarray
        Drainage area in cells (from compute_drainage_area)
    upstream_rainfall : np.ndarray
        Upstream rainfall accumulation (from compute_upstream_rainfall)

    Returns
    -------
    np.ndarray
        Discharge potential index: drainage_area × (upstream_rainfall / mean_rainfall)
        Units are dimensionless, scaled relative to mean rainfall

    Notes
    -----
    The formula normalizes upstream rainfall by mean to produce a dimensionless
    multiplier. This means:
    - Discharge potential = drainage_area when rainfall is uniform
    - Cells with above-average rainfall have higher discharge potential
    - Cells with below-average rainfall have lower discharge potential

    This is useful for identifying where actual river discharge would be highest,
    accounting for both watershed size and precipitation patterns.

    Examples
    --------
    >>> drainage_area = np.array([[1, 2], [4, 8]], dtype=np.float32)
    >>> upstream_rainfall = np.array([[100, 200], [400, 800]], dtype=np.float32)
    >>> discharge = compute_discharge_potential(drainage_area, upstream_rainfall)
    >>> discharge.shape
    (2, 2)
    """
    # Handle edge cases
    upstream_valid = upstream_rainfall[upstream_rainfall > 0]
    if len(upstream_valid) == 0:
        # No valid rainfall data - return drainage area as-is
        return drainage_area.astype(np.float32)

    mean_rainfall = np.mean(upstream_valid)

    # Compute discharge potential
    # Formula: drainage × (rainfall / mean_rainfall)
    discharge = drainage_area.astype(np.float32) * (upstream_rainfall / mean_rainfall)

    # Ensure zeros stay zero (avoid NaN from 0/0)
    discharge[upstream_rainfall == 0] = 0

    return discharge
