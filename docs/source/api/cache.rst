Cache Module
============

Caching system for efficient terrain processing pipeline.

This module provides three caching systems to avoid reloading and reprocessing expensive
operations: DEM loading, transforms, and full pipeline stages.

All caches use hash-based validation to automatically invalidate when source data changes.

DEMCache
--------

.. autoclass:: src.terrain.cache.DEMCache
   :members:
   :undoc-members:

   Caches loaded and merged DEM data with hash validation.

   **What it caches:**

   - Merged DEM arrays (.npz files)
   - Affine transforms
   - Source file metadata (paths, modification times)

   **When cache is invalidated:**

   - Files are added/removed from source directory
   - Files are modified (based on mtime)
   - Directory path changes

   Example::

       from src.terrain.cache import DEMCache

       cache = DEMCache(cache_dir='.dem_cache', enabled=True)

       # Load with caching
       dem, transform = cache.get_or_load(
           directory_path='data/hgt',
           pattern='*.hgt',
           load_func=lambda: load_dem_files('data/hgt')
       )

TransformCache
--------------

.. autoclass:: src.terrain.cache.TransformCache
   :members:
   :undoc-members:

   Caches transformed raster data (reprojection, smoothing, etc.).

   **What it caches:**

   - Transformed DEM arrays
   - Transform parameters (for validation)
   - Intermediate processing results

   **Cache key components:**

   - Transform name (e.g., 'reproject', 'smooth')
   - Parameter dictionary (all args)
   - Input data hash

   Example::

       from src.terrain.cache import TransformCache

       cache = TransformCache(cache_dir='.transform_cache')

       # Cache expensive transform
       result = cache.get_or_compute(
           'wavelet_denoise',
           compute_fn=lambda: wavelet_denoise_dem(dem, wavelet='db4'),
           params={'wavelet': 'db4', 'levels': 3}
       )

   See :doc:`transforms` for usage with transform functions.

PipelineCache
-------------

.. autoclass:: src.terrain.cache.PipelineCache
   :members:
   :undoc-members:

   High-level caching for full terrain processing pipelines.

   **What it caches:**

   - Complete pipeline stage outputs
   - Multi-step processing chains
   - Score computations

   **Pipeline stages:**

   - ``load``: DEM loading and merging
   - ``transform``: Geographic transforms (reproject, flip, downsample)
   - ``smooth``: DEM smoothing operations
   - ``scores``: Score grid computations
   - ``water``: Water body detection
   - ``mesh``: Final mesh generation

   Example::

       from src.terrain.cache import PipelineCache

       cache = PipelineCache(cache_dir='.pipeline_cache')

       # Cache pipeline stage
       smoothed_dem = cache.get_or_run_stage(
           stage='smooth',
           compute_fn=lambda: run_smoothing_pipeline(dem),
           params={
               'wavelet': True,
               'adaptive': True,
               'bilateral': True
           }
       )

   Used in :doc:`../examples/combined_render` to avoid reprocessing expensive operations.

Cache Management
----------------

**Cache directory structure:**

```
.dem_cache/
├── dem_abc123.npz         # Cached DEM data
└── dem_abc123.json        # Metadata

.transform_cache/
├── reproject_def456.npz   # Cached transform results
└── reproject_def456.json  # Transform params

.pipeline_cache/
├── stage_smooth_ghi789.npz
└── stage_smooth_ghi789.json
```

**Clearing cache:**

```python
# Clear all caches
import shutil
shutil.rmtree('.dem_cache')
shutil.rmtree('.transform_cache')
shutil.rmtree('.pipeline_cache')
```

**Cache validation:**

All caches automatically validate:
- Source file changes (modification time)
- Parameter changes
- Data shape/dtype changes

Invalid cache entries are automatically regenerated.

Performance Notes
-----------------

**Typical speedups:**

- DEM loading: 50-100x faster (0.1s vs 5-10s for large merges)
- Transforms: 10-50x faster (depends on complexity)
- Pipeline stages: 100-1000x faster for multi-stage pipelines

**Cache overhead:**

- Hash computation: ~10-50ms per cache lookup
- Disk I/O: ~50-200ms for large DEMs
- Memory: Minimal (uses memory-mapped arrays when possible)

**When to use caching:**

- Development/iteration (same data, different parameters)
- Batch processing (same DEM, multiple visualizations)
- Testing (avoid reloading data between test runs)

**When NOT to use caching:**

- One-off renders (cache overhead > compute time)
- Rapidly changing source data
- Limited disk space
