Gridded Data Module
===================

Memory-efficient loading and processing of large gridded datasets.

This module provides automatic tiling, memory monitoring, and caching for processing
large external datasets (SNODAS snow data, temperature grids, precipitation, etc.).

Prevents out-of-memory errors on large DEMs by automatically splitting processing
into tiles with configurable memory limits.

Core Classes
------------

.. autoclass:: src.terrain.gridded_data.GriddedDataLoader
   :members:
   :undoc-members:

   Main class for loading and processing gridded data with automatic tiling.

   **Key features:**

   - Transparent automatic tiling for large datasets
   - Memory monitoring with failsafe (prevents OOM/thrashing)
   - Per-step and merged result caching
   - Smart aggregation (spatial concatenation, statistical averaging)

   **Usage pattern:**

   1. Define data bounds and output shape
   2. Add pipeline steps (load, process, transform)
   3. Execute with automatic tiling if needed
   4. Get cached results on subsequent runs

   Example::

       from src.terrain.gridded_data import GriddedDataLoader, TiledDataConfig

       # Configure tiling
       config = TiledDataConfig(
           max_output_pixels=4096 * 4096,  # ~16M pixels = ~64MB
           target_tile_outputs=2000,       # 2000x2000 tiles
           enable_memory_monitoring=True,
           max_memory_percent=85.0
       )

       # Create loader
       loader = GriddedDataLoader(
           bounds=(minx, miny, maxx, maxy),
           output_shape=(height, width),
           config=config,
           cache_dir='.gridded_cache'
       )

       # Add pipeline steps
       loader.add_step('load', load_snow_data_fn)
       loader.add_step('process', compute_score_fn)
       loader.add_step('smooth', smooth_fn)

       # Execute (automatically tiles if needed)
       result = loader.execute()

   See :doc:`../examples/sledding` for real-world usage.

Configuration
-------------

.. autoclass:: src.terrain.gridded_data.TiledDataConfig
   :members:
   :undoc-members:

   Configuration for automatic tiling behavior.

   **Key parameters:**

   - ``max_output_pixels``: Tile if output exceeds this (default: 16M pixels)
   - ``target_tile_outputs``: Target tile size (default: 2000×2000)
   - ``max_memory_percent``: Abort if RAM usage exceeds (default: 85%)
   - ``max_swap_percent``: Abort if swap usage exceeds (default: 50%)
   - ``enable_memory_monitoring``: Enable safety checks (default: True)

   Example::

       # Conservative settings (low memory systems)
       config = TiledDataConfig(
           max_output_pixels=2048 * 2048,  # 4M pixels
           target_tile_outputs=1000,       # 1000x1000 tiles
           max_memory_percent=70.0
       )

       # Aggressive settings (high memory systems)
       config = TiledDataConfig(
           max_output_pixels=8192 * 8192,  # 64M pixels
           target_tile_outputs=4000,       # 4000x4000 tiles
           max_memory_percent=90.0
       )

Memory Monitoring
-----------------

.. autoclass:: src.terrain.gridded_data.MemoryMonitor
   :members:
   :undoc-members:

   Monitors system memory and aborts processing if limits exceeded.

   **What it monitors:**

   - RAM usage (percent of total)
   - Swap usage (percent of total)
   - Available memory (absolute bytes)

   **When it aborts:**

   - RAM usage > ``max_memory_percent`` (default: 85%)
   - Swap usage > ``max_swap_percent`` (default: 50%)

   **Requires:** ``psutil`` package (``pip install psutil``)

   Example::

       from src.terrain.gridded_data import MemoryMonitor, TiledDataConfig

       config = TiledDataConfig(max_memory_percent=85.0)
       monitor = MemoryMonitor(config)

       # Start monitoring in background thread
       monitor.start()

       # Do expensive processing...
       process_large_data()

       # Stop monitoring
       monitor.stop()

.. autoexception:: src.terrain.gridded_data.MemoryLimitExceeded

   Raised when memory usage exceeds configured limits.

   Prevents system thrashing/OOM by aborting early.

Tile Specification
------------------

.. autoclass:: src.terrain.gridded_data.TileSpecGridded
   :members:
   :undoc-members:

   Specification for a single tile with geographic extent.

   **Attributes:**

   - ``src_slice``: Slice into source data (with halo padding)
   - ``out_slice``: Slice into output array
   - ``extent``: Geographic bounds (minx, miny, maxx, maxy)
   - ``target_shape``: Output shape for this tile (height, width)

Utility Functions
-----------------

.. autofunction:: src.terrain.gridded_data.downsample_for_viz

   Downsample large grids for faster visualization.

.. autofunction:: src.terrain.gridded_data.create_mock_snow_data

   Create mock SNODAS-like data for testing.

   Example::

       mock_data = create_mock_snow_data(shape=(1000, 1000))
       # Returns dict with 'swe', 'depth', 'density' arrays

Performance Notes
-----------------

**Memory efficiency:**

- Processes tiles independently (only one tile in memory)
- Automatic garbage collection between tiles
- Memory-mapped caching for large results

**Typical tiling overhead:**

- No tiling: ~0ms overhead
- Tiled (4 tiles): ~100-200ms overhead (cache lookups, tile merging)
- Tiled (16 tiles): ~500-1000ms overhead

**When tiling triggers:**

- Output pixels > ``max_output_pixels`` (default: 16M)
- Example: 4096×4096 DEM triggers tiling by default

**Cache effectiveness:**

- First run: Full computation + caching
- Subsequent runs: ~100x faster (cache hits only)
- Cache invalidation: Automatic on parameter/data changes
