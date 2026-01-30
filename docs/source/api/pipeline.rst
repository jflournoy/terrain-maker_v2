Pipeline Module
===============

Dependency graph execution for terrain rendering pipeline.

This module provides a lightweight task dependency system with automatic caching,
staleness detection, and execution planning. No external build tools required -
all logic stays in Python.

Useful for complex multi-view renders where intermediate results (DEM loading,
transforms, mesh creation) can be cached and reused across different camera angles.

TerrainPipeline
---------------

.. autoclass:: src.terrain.pipeline.TerrainPipeline
   :members:
   :undoc-members:

   Dependency graph executor for terrain rendering.

   **Pipeline tasks:**

   1. ``load_dem``: Load and merge SRTM tiles (cached)
   2. ``apply_transforms``: Reproject, flip, downsample (cached)
   3. ``detect_water``: Identify water bodies (cached)
   4. ``create_mesh``: Build Blender geometry (cached, reused across views)
   5. ``render_view``: Output to PNG (cached per view/render params)

   **Features:**

   - Automatic staleness detection via hashing
   - Dry-run execution plans (``explain()``)
   - Reusable cached outputs across multiple views
   - Per-layer caching support

   Example::

       from src.terrain.pipeline import TerrainPipeline

       # Create pipeline
       pipeline = TerrainPipeline(
           dem_dir='data/dem/detroit',
           cache_enabled=True
       )

       # Show execution plan (dry run)
       pipeline.explain('render_view')
       # Output:
       # Task load_dem: CACHED
       # Task apply_transforms: CACHED
       # Task detect_water: CACHED
       # Task create_mesh: CACHED
       # Task render_view: NEEDS COMPUTATION

       # Execute pipeline
       result = pipeline.execute('render_view')

       # Get cache statistics
       stats = pipeline.cache_stats()
       print(f"Cache hits: {stats['hits']}")
       print(f"Cache misses: {stats['misses']}")

       # Clear cache for specific task
       pipeline.clear_cache(task='render_view')

       # Clear all caches
       pipeline.clear_cache()

   **Multi-view rendering:**

   Cache mesh once, render multiple views::

       pipeline = TerrainPipeline(dem_dir='data/dem')

       # First view: mesh cached
       pipeline.set_params('render_view', {'direction': 'south'})
       pipeline.execute('render_view')

       # Second view: reuses cached mesh (faster!)
       pipeline.set_params('render_view', {'direction': 'north'})
       pipeline.execute('render_view')

       # Third view: reuses cached mesh again
       pipeline.set_params('render_view', {'direction': 'east'})
       pipeline.execute('render_view')

TaskState
---------

.. autoclass:: src.terrain.pipeline.TaskState
   :members:
   :undoc-members:

   Represents execution state of a pipeline task.

   **Attributes:**

   - ``name``: Task identifier
   - ``depends_on``: List of task dependencies
   - ``params``: Task parameters (for cache key)
   - ``cached``: Whether result is cached
   - ``computed``: Whether task was executed this run
   - ``result``: Cached or computed result
   - ``cache_key``: Hash for cache lookup

Pipeline Design Patterns
------------------------

**Pattern 1: Multi-view renders**

Cache expensive mesh creation, render multiple camera angles::

    pipeline = TerrainPipeline('data/dem')

    for direction in ['north', 'south', 'east', 'west']:
        pipeline.set_params('render_view', {
            'direction': direction,
            'output': f'render_{direction}.png'
        })
        pipeline.execute('render_view')

**Pattern 2: Parameter sweeps**

Test different DEM smoothing parameters::

    pipeline = TerrainPipeline('data/dem')

    for strength in [0.5, 1.0, 1.5, 2.0]:
        pipeline.set_params('apply_transforms', {
            'smooth_strength': strength
        })
        pipeline.execute('render_view')
        # Mesh and render are regenerated, DEM loading is cached

**Pattern 3: Development workflow**

Fast iteration on render settings::

    pipeline = TerrainPipeline('data/dem', cache_enabled=True)

    # First run: everything computed
    pipeline.execute('render_view')  # ~60s

    # Tweak lighting
    pipeline.set_params('render_view', {'sun_energy': 1.5})
    pipeline.execute('render_view')  # ~5s (reuses mesh)

    # Tweak camera
    pipeline.set_params('render_view', {'camera_distance': 2.0})
    pipeline.execute('render_view')  # ~5s (reuses mesh)

Cache Integration
-----------------

TerrainPipeline integrates with:

- :class:`~terrain.cache.DEMCache` - DEM loading cache
- :class:`~terrain.mesh_cache.MeshCache` - Mesh geometry cache

All caches use hash-based validation for automatic staleness detection.

**Cache invalidation:**

- Parameter changes invalidate dependent tasks
- Source file changes invalidate all downstream tasks
- Manual invalidation via ``clear_cache()``

Performance Notes
-----------------

**Typical speedups (multi-view renders):**

- First view: ~60s (full pipeline)
- Subsequent views: ~5s (cached mesh, only render changes)
- 10x speedup for camera/lighting iterations

**Cache overhead:**

- Hash computation: ~10-50ms per task
- Cache lookup: ~5-20ms per task
- Total overhead: ~100-500ms for full pipeline

**When to use TerrainPipeline:**

- Multi-view renders (same DEM, different cameras)
- Parameter sweeps (testing smoothing, colors, etc.)
- Development iteration (fast render feedback)

**When NOT to use:**

- Single render (overhead > savings)
- Rapidly changing DEMs
- Memory-constrained systems (caches use disk space)
