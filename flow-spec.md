# Hydrological Flow Routing: Code Specification

## Overview

This document specifies a self-contained hydrological flow routing system operating on a Digital Elevation Model (DEM) raster grid. The pipeline has four stages:

1. **Outlet identification** — classify cells that serve as drainage termini
2. **DEM conditioning** — breach sinks (with constrained depth/length), then minimally fill residuals
3. **Flow direction** — assign D8 flow directions
4. **Flow accumulation** — compute upstream contributing area

All algorithms operate on a 2D grid of elevation values where NoData cells represent ocean, lakes, or off-map areas.

---

## 1. Data Structures

```
Grid:
    nrows, ncols: int
    elevation: float[nrows][ncols]
    nodata: bool[nrows][ncols]         # true for ocean, off-map, masked areas
    outlet: bool[nrows][ncols]         # true for drainage termini
    flow_dir: int[nrows][ncols]        # D8 direction (0-7 or -1 for outlet/nodata)
    flow_acc: int[nrows][ncols]        # contributing area (cell count)
    resolved: bool[nrows][ncols]       # used during conditioning

# D8 neighbor encoding (0 = East, counter-clockwise)
#   5  6  7
#   4  x  0
#   3  2  1
#
# dx = [1, 1, 0, -1, -1, -1, 0, 1]
# dy = [0, 1, 1,  1,  0, -1, -1, -1]
# dist = [1, sqrt(2), 1, sqrt(2), 1, sqrt(2), 1, sqrt(2)]

NEIGHBORS: list of (dx, dy, distance) for 8 directions
```

---

## 2. Stage 1: Outlet Identification

### Goal

Identify cells that act as drainage termini. Water reaching an outlet cell leaves the system.

### Classification Rules

A land cell (not NoData) is an outlet if:

- **Coastal outlet:** The cell is adjacent (8-connected) to an ocean/NoData cell AND the cell's elevation is below a coastal elevation threshold (e.g., 10m above sea level). This prevents high cliffs adjacent to ocean from being spurious outlets.
- **Edge outlet:** The cell is on the DEM boundary (row 0, row nrows-1, col 0, col ncols-1) AND it is a local minimum along its edge segment, OR it has at least one interior neighbor with steeper slope toward the edge than toward any other neighbor.
- **Masked basin outlet:** The cell is adjacent to a pre-masked interior NoData region (known lake, endorheic basin). These are optional user-supplied masks.

### Pseudocode

```
function identify_outlets(grid, coastal_elev_threshold, edge_mode):

    for each cell (r, c) where not grid.nodata[r][c]:

        # --- Coastal outlets ---
        if any neighbor (nr, nc) is nodata AND grid.elevation[r][c] < coastal_elev_threshold:
            grid.outlet[r][c] = true
            continue

        # --- Edge outlets ---
        if cell is on grid boundary:
            if edge_mode == "all":
                grid.outlet[r][c] = true
            elif edge_mode == "local_minima":
                # Check if this cell is a local minimum among edge neighbors
                is_min = true
                for each edge-adjacent neighbor (nr, nc):
                    if grid.elevation[nr][nc] < grid.elevation[r][c]:
                        is_min = false
                if is_min:
                    grid.outlet[r][c] = true
            elif edge_mode == "outward_slope":
                # Check if any interior neighbor slopes toward this edge cell
                # more steeply than toward any other neighbor
                for each interior neighbor (nr, nc):
                    slope_to_edge = (grid.elevation[nr][nc] - grid.elevation[r][c]) / dist
                    max_slope_elsewhere = max slope from (nr, nc) to any other neighbor
                    if slope_to_edge > max_slope_elsewhere:
                        grid.outlet[r][c] = true
                        break

        # --- Masked basin outlets (optional) ---
        if any neighbor (nr, nc) is nodata AND is_interior(nr, nc):
            # Interior nodata = pre-masked basin
            grid.outlet[r][c] = true
```

### Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `coastal_elev_threshold` | Max elevation for coastal outlet | 10m |
| `edge_mode` | Strategy for boundary outlets | `"all"` or `"local_minima"` |

### Notes

- `edge_mode = "all"` is the safest default — it ensures no artificial endorheic basins form at boundaries. The cost is some fragmentation of edge drainage networks, but this is usually preferable to missed outlets.
- For domains fully surrounded by coastline (islands), edge outlets may be unnecessary.
- Outlets should be assigned a virtual elevation of `-infinity` (or a very low sentinel) for the conditioning stage so they always act as sinks in the priority queue.

---

## 3. Stage 2: DEM Conditioning — Constrained Breach + Residual Fill

This is the most complex stage. It has two sub-steps:

### 3.1 Constrained Least-Cost Breaching (Lindsay 2016)

#### Goal

For each depression (pit), carve the least-cost path to an already-resolved (draining) cell, subject to maximum breach depth and length constraints. This removes spurious sinks while preserving large legitimate basins.

#### Algorithm: Priority-Flood Breach

This combines pit identification and breach resolution in a single priority-queue sweep starting from outlets and working inward by ascending elevation.

```
function breach_depressions(grid, max_breach_depth, max_breach_length, epsilon):

    # Priority queue: (elevation, row, col), min-heap by elevation
    PQ = MinHeap()

    # Initialize: seed queue with all outlet cells
    for each cell (r, c) where grid.outlet[r][c]:
        grid.resolved[r][c] = true
        PQ.push( (grid.elevation[r][c], r, c) )

    # Also seed with nodata-adjacent cells that aren't outlets
    # (they border the domain edge implicitly)

    # Process cells in order of ascending elevation
    while PQ is not empty:
        (elev, r, c) = PQ.pop()

        for each neighbor (nr, nc, dist) of (r, c):
            if grid.resolved[nr][nc] or grid.nodata[nr][nc]:
                continue

            n_elev = grid.elevation[nr][nc]

            if n_elev >= elev:
                # Neighbor is higher or equal — it drains naturally to (r,c)
                # (or will after epsilon enforcement)
                grid.resolved[nr][nc] = true
                PQ.push( (n_elev, nr, nc) )

            else:
                # Neighbor is LOWER than current cell — it's in a depression
                # We need to find a breach path OUT of this depression
                breach_path = find_breach_path(grid, nr, nc,
                                               max_breach_depth,
                                               max_breach_length)

                if breach_path is not None:
                    # Apply the breach: lower cells along path
                    apply_breach(grid, breach_path, epsilon)
                    grid.resolved[nr][nc] = true
                    PQ.push( (grid.elevation[nr][nc], nr, nc) )
                else:
                    # Cannot breach within constraints — mark for filling later
                    # Push with ORIGINAL elevation; fill step will handle it
                    grid.resolved[nr][nc] = true
                    PQ.push( (n_elev, nr, nc) )
```

**Note:** The above is a simplified sketch. The actual Lindsay (2016) algorithm integrates breaching directly into the priority-flood framework more tightly. An alternative (and possibly cleaner) implementation is a two-pass approach:

#### Alternative: Two-Pass Breach

```
function breach_depressions_two_pass(grid, max_breach_depth, max_breach_length, epsilon):

    # Pass 1: Identify all sink cells (cells with no downslope neighbor)
    sinks = []
    for each cell (r, c) where not grid.nodata[r][c] and not grid.outlet[r][c]:
        has_downslope = false
        for each neighbor (nr, nc, dist):
            if grid.nodata[nr][nc]: continue
            if grid.elevation[nr][nc] < grid.elevation[r][c]:
                has_downslope = true
                break
        if not has_downslope:
            sinks.append( (r, c) )

    # Sort sinks by elevation (process lowest first to avoid cascading issues)
    sort sinks by grid.elevation ascending

    # Pass 2: For each sink, attempt breach
    for each (r, c) in sinks:
        # Skip if already resolved by a previous breach
        if cell_has_downslope_path_to_outlet(grid, r, c):
            continue

        breach_path = find_breach_path(grid, r, c,
                                       max_breach_depth,
                                       max_breach_length)
        if breach_path is not None:
            apply_breach(grid, breach_path, epsilon)
```

#### find_breach_path — Dijkstra Least-Cost Path

```
function find_breach_path(grid, start_r, start_c, max_depth, max_length):
    # Find least-cost path from sink to a cell that can drain
    # (an outlet, or a cell lower than the sink with a path to an outlet)
    #
    # Cost metric: total elevation that must be removed along the path

    start_elev = grid.elevation[start_r][start_c]

    # Dijkstra from the sink cell
    # State: (cumulative_cost, path_length, row, col, parent)
    DPQ = MinHeap()
    DPQ.push( (0, 0, start_r, start_c, None) )
    visited = {}
    parent_map = {}

    while DPQ is not empty:
        (cost, length, r, c, parent) = DPQ.pop()

        if (r, c) in visited:
            continue
        visited[(r, c)] = cost
        parent_map[(r, c)] = parent

        # Check termination: have we reached a cell that drains?
        # A cell "drains" if it's resolved (has a path to an outlet)
        # and its elevation is <= start_elev (so water flows downhill out)
        if grid.resolved[r][c] and grid.elevation[r][c] <= start_elev:
            return reconstruct_path(parent_map, start_r, start_c, r, c)

        if grid.outlet[r][c]:
            return reconstruct_path(parent_map, start_r, start_c, r, c)

        # Constraint checks
        if length >= max_length:
            continue  # Don't expand further from this cell

        for each neighbor (nr, nc, dist) of (r, c):
            if (nr, nc) in visited or grid.nodata[nr][nc]:
                continue

            # Cost to breach through this neighbor:
            # If neighbor is higher than the start sink, we must carve it
            # down to at most start_elev (minus epsilon per step)
            target_elev = start_elev  # The breach path must be <= sink elev
            breach_depth_here = max(0, grid.elevation[nr][nc] - target_elev)

            if breach_depth_here > max_depth:
                continue  # Would exceed max breach depth at this cell

            new_cost = cost + breach_depth_here
            new_length = length + 1

            DPQ.push( (new_cost, new_length, nr, nc, (r, c)) )

    # No viable breach path found within constraints
    return None
```

#### apply_breach — Carve the Path

```
function apply_breach(grid, path, epsilon):
    # path is a list of (r, c) from sink to drain-point
    # Enforce monotonically decreasing elevation along the path

    # Work backward from the drain-point (end of path) to the sink (start)
    # The drain-point elevation is the baseline
    n = len(path)
    drain_r, drain_c = path[n - 1]
    base_elev = grid.elevation[drain_r][drain_c]

    for i from (n - 2) down to 0:
        r, c = path[i]
        required_elev = base_elev + epsilon * (n - 1 - i)
        # Only lower cells, never raise them
        if grid.elevation[r][c] > required_elev:
            grid.elevation[r][c] = required_elev
```

### 3.2 Residual Fill — Priority-Flood (Barnes et al. 2014)

After breaching, some sinks may remain (those that couldn't be breached within constraints). Fill these minimally.

```
function priority_flood_fill(grid, epsilon):
    # Standard priority-flood with epsilon gradient to resolve flats

    PQ = MinHeap()
    in_queue = bool[nrows][ncols] initialized to false

    # Seed with all outlet cells and nodata-adjacent land cells
    for each cell (r, c) where grid.outlet[r][c]:
        PQ.push( (grid.elevation[r][c], r, c) )
        in_queue[r][c] = true

    # Also seed with all edge cells (if they are outlets)
    # and cells adjacent to nodata
    for each cell (r, c) on grid boundary or adjacent to nodata:
        if not grid.nodata[r][c] and not in_queue[r][c]:
            PQ.push( (grid.elevation[r][c], r, c) )
            in_queue[r][c] = true

    while PQ is not empty:
        (elev, r, c) = PQ.pop()

        for each neighbor (nr, nc, dist) of (r, c):
            if in_queue[nr][nc] or grid.nodata[nr][nc]:
                continue

            in_queue[nr][nc] = true

            if grid.elevation[nr][nc] < elev + epsilon:
                # This cell is in a depression — raise it
                grid.elevation[nr][nc] = elev + epsilon

            PQ.push( (grid.elevation[nr][nc], nr, nc) )
```

### Parameters

| Parameter | Description | Typical Value | Notes |
|-----------|-------------|---------------|-------|
| `max_breach_depth` | Max elevation drop at any single cell | 5–50m | Domain-dependent |
| `max_breach_length` | Max cells in breach path | 10–100 cells | Resolution-dependent |
| `epsilon` | Minimum elevation drop per cell | 1e-5 to 1e-3 | Must exceed float precision noise |

### Notes on Legitimate Basins

Basins that are NOT breached (because they exceed constraints) will be filled. This creates flat areas. Options:

- **Pre-mask known basins** as NoData before running the pipeline. This is the cleanest approach for known lakes or endorheic basins.
- **Post-hoc identification:** After filling, any flat area larger than a threshold is likely a real basin. These can be masked and the pipeline re-run.
- **Tuning constraints:** If too many ridges are being breached, tighten `max_breach_depth` and `max_breach_length`. If too many small sinks survive to the fill step, loosen them.

---

## 4. Stage 3: D8 Flow Direction

After conditioning, every non-outlet, non-NoData cell should have at least one lower neighbor (guaranteed by the fill step).

```
function compute_d8_flow_direction(grid):

    for each cell (r, c):
        if grid.nodata[r][c]:
            grid.flow_dir[r][c] = -1
            continue

        if grid.outlet[r][c]:
            grid.flow_dir[r][c] = -1  # Terminal; or point toward ocean/edge
            continue

        max_slope = -infinity
        best_dir = -1

        for dir in 0..7:
            nr = r + dy[dir]
            nc = c + dx[dir]

            if out_of_bounds(nr, nc):
                continue
            if grid.nodata[nr][nc] and not grid.outlet[r][c]:
                continue

            slope = (grid.elevation[r][c] - grid.elevation[nr][nc]) / dist[dir]

            if slope > max_slope:
                max_slope = slope
                best_dir = dir

        grid.flow_dir[r][c] = best_dir

        # Safety check: if best_dir is still -1, something went wrong
        # in conditioning — this cell has no downslope neighbor
        if best_dir == -1:
            log_warning("Unresolved flat at", r, c)
```

---

## 5. Stage 4: Flow Accumulation

Compute upstream contributing area for each cell by traversing the flow network.

```
function compute_flow_accumulation(grid):

    # Initialize all cells to 1 (each cell contributes itself)
    for each cell (r, c):
        if grid.nodata[r][c]:
            grid.flow_acc[r][c] = 0
        else:
            grid.flow_acc[r][c] = 1

    # Compute in-degree for each cell
    in_degree = int[nrows][ncols] initialized to 0

    for each cell (r, c) where flow_dir[r][c] >= 0:
        (nr, nc) = downstream_cell(r, c, grid.flow_dir[r][c])
        in_degree[nr][nc] += 1

    # Topological sort via queue (Kahn's algorithm)
    queue = Queue()
    for each cell (r, c) where not grid.nodata[r][c]:
        if in_degree[r][c] == 0:
            queue.push( (r, c) )  # Headwater cells (ridgelines)

    while queue is not empty:
        (r, c) = queue.pop()
        dir = grid.flow_dir[r][c]

        if dir < 0:
            continue  # Outlet or nodata

        (nr, nc) = downstream_cell(r, c, dir)
        grid.flow_acc[nr][nc] += grid.flow_acc[r][c]

        in_degree[nr][nc] -= 1
        if in_degree[nr][nc] == 0:
            queue.push( (nr, nc) )
```

### Notes

- This is O(n) where n is the number of cells — each cell is visited exactly once.
- If any cell never reaches in-degree 0 during traversal, there's a cycle in the flow network, which indicates a bug in the conditioning step.
- To extract stream networks, threshold: `stream[r][c] = (flow_acc[r][c] > threshold)`.

---

## 6. Full Pipeline

```
function hydrological_flow_routing(dem_path, params):

    grid = load_dem(dem_path)

    # Stage 1: Outlets
    identify_outlets(grid,
                     coastal_elev_threshold = params.coastal_threshold,
                     edge_mode = params.edge_mode)

    # Stage 2a: Breach
    breach_depressions(grid,
                       max_breach_depth = params.max_breach_depth,
                       max_breach_length = params.max_breach_length,
                       epsilon = params.epsilon)

    # Stage 2b: Fill residuals
    priority_flood_fill(grid, epsilon = params.epsilon)

    # Stage 3: Flow direction
    compute_d8_flow_direction(grid)

    # Stage 4: Flow accumulation
    compute_flow_accumulation(grid)

    return grid
```

---

## 7. Implementation Considerations

### Memory

For large DEMs (e.g., 10k × 10k = 100M cells), each grid layer at 4 bytes is ~400MB. Minimize concurrent arrays. The `resolved`, `in_queue`, and `outlet` booleans can be packed into a single bitfield or status byte per cell.

### Priority Queue Performance

Both breaching and filling are dominated by priority queue operations. Use a binary heap at minimum. For very large grids, a hierarchical bucket queue (exploiting the fact that elevations are often quantized to integer mm or cm) can be significantly faster.

### Epsilon Selection

- Too small: floating point accumulation errors create ties or reversals
- Too large: distorts elevations in large filled areas
- Rule of thumb: `epsilon = 1e-5 * cell_resolution` (e.g., 1e-4 for 10m DEM)
- For integer DEMs, use `epsilon = 1` in the native units (e.g., 1mm if elevation is in mm)

### Flat Resolution

After filling, flat areas may span many cells. The epsilon gradient imposed during priority-flood provides implicit flat resolution — cells filled later get slightly higher elevations, creating a gradient toward the outlet. This is usually sufficient for D8 but can produce unrealistic parallel drainage on wide flats. For better flat handling, see Garbrecht & Martz (1997) or Barnes et al. (2014) "An Efficient Assignment of Drainage Direction Over Flat Surfaces."

### Parallelism

- Flow accumulation is inherently sequential (topological order dependency)
- D8 direction is embarrassingly parallel (each cell independent)
- Priority-flood and breaching are harder to parallelize; domain decomposition with halo exchange is possible but complex. For most applications, the serial versions are fast enough on modern hardware (100M cells in ~30 seconds in C/Rust, ~5 minutes in Python with numpy).

---

## References

- Barnes, R., Lehman, C., & Mulla, D. (2014). Priority-flood: An optimal depression-filling and watershed-labeling algorithm for digital elevation models. *Computers & Geosciences*, 62, 117–127.
- Garbrecht, J., & Martz, L.W. (1997). The assignment of drainage direction over flat surfaces in raster digital elevation models. *Journal of Hydrology*, 193, 204–213.
- Lindsay, J.B. (2016). Efficient hybrid breaching-filling sink removal methods for flow path enforcement in digital elevation models. *Hydrological Processes*, 30, 846–857.
- Lindsay, J.B., & Creed, I.F. (2005). Removal of artifact depressions from digital elevation models. *Hydrological Processes*, 19, 3113–3126.
- O'Callaghan, J.F., & Mark, D.M. (1984). The extraction of drainage networks from digital elevation data. *Computer Vision, Graphics, and Image Processing*, 28, 323–344.
- Tarboton, D.G. (1997). A new method for the determination of flow directions and upslope areas in grid digital elevation models. *Water Resources Research*, 33, 309–319.

