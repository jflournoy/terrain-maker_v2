# Caching in Terrain Maker

Terrain Maker uses a **three-tier caching system** to optimize performance and avoid redundant computations:

1. **Subsystem Caching** - Cache expensive I/O and transformation operations
2. **Pipeline Task Caching** - Cache task outputs based on input hashes
3. **Data Caching** - Cache preprocessed data (SNODAS snow data)

## Tier 1: Subsystem Caching

### DEMCache
Caches loaded and merged DEM (Digital Elevation Model) data to avoid reloading from disk.

**Location:** `src/terrain/cache.py`

**How it works:**
- Uses SHA256 hashing of source file metadata (paths + modification times)
- Stores DEM arrays as `.npz` files with metadata JSON
- Automatically invalidates cache when source files change

**Example:**
```python
from src.terrain.cache import DEMCache
from pathlib import Path

cache = DEMCache(cache_dir=Path(".dem_cache"), enabled=True)

# Compute hash of source files
source_hash = cache.compute_source_hash(
    directory_path="data/dem/detroit",
    pattern="*.hgt",
    recursive=False
)

# Load from cache or compute and save
cache_path = cache.get_cache_path(source_hash, cache_name="detroit")
if cache_path.exists():
    # Load from cache
    cached_data = np.load(cache_path)["dem"]
else:
    # Load and process, then cache
    dem = load_dem_files(dem_dir)
    cache.save_dem_cache(dem, source_hash)
```

**When to use:**
- Loading DEM files from disk
- Merging multiple DEM tiles
- Any expensive DEM preprocessing

### MeshCache
Caches generated 3D mesh files to avoid recomputing Blender meshes.

**Location:** `src/terrain/mesh_cache.py`

**How it works:**
- Uses SHA256 hashing of DEM hash + mesh parameters
- Stores cached meshes as `.blend` files
- Preserves mesh parameters and vertex data

**When to use:**
- Creating 3D meshes for rendering
- Blender mesh generation with specific parameters

## Tier 2: Task/Workflow Caching

### SnowAnalysisPipeline (Generic Dependency Graph Caching)

Provides a **reusable caching framework** for sequential computations with explicit dependencies.

**Location:** `src/snow/analysis.py` (class: `SnowAnalysisPipeline`)

**How it works:**
- Define custom tasks with SHA256 hashing of inputs
- Implements a dependency graph to track task relationships
- Automatically caches task results as `.npz` files with metadata
- Invalidates cache when inputs change (based on hash comparison)

**Key features:**
- **Input-based cache invalidation:** Different inputs = different hash = cache miss
- **Flexible task definitions:** Users define their own tasks and dependencies
- **Metadata logging:** Timestamps and execution parameters stored for debugging
- **Optional caching:** Toggle with `caching_enabled=False` for testing
- **Force rebuild:** `force_rebuild=True` bypasses cache when needed
- **Cache statistics:** Query cache size and entry counts

**Usage Example - Define Your Own Tasks:**

```python
from src.snow.analysis import SnowAnalysisPipeline
from pathlib import Path
import numpy as np

# Step 1: Subclass to define custom task implementations
class MyAnalysisPipeline(SnowAnalysisPipeline):
    def _execute_task(self, task_name, **kwargs):
        if task_name == "load_data":
            # Your custom load logic
            data = load_my_data(**kwargs)
            return {"data": data}

        elif task_name == "process_data":
            # Your custom processing logic
            raw_data = kwargs.get("data")
            processed = process_my_data(raw_data, **kwargs)
            return {"processed": processed}

        elif task_name == "analyze":
            # Your custom analysis logic
            processed = kwargs.get("processed")
            results = analyze_my_data(processed, **kwargs)
            return {"results": results}

        raise ValueError(f"Unknown task: {task_name}")

# Step 2: Create and use your pipeline
pipeline = MyAnalysisPipeline(
    cache_dir=Path(".my_analysis_cache"),
    caching_enabled=True  # Enable caching
)

# Step 3: Execute tasks - caching happens automatically
raw_data = np.random.rand(100, 100)

# First run - computes and caches result
result1 = pipeline.execute(task="load_data", data=raw_data)

# Second run with same inputs - loads from cache instantly
result2 = pipeline.execute(task="load_data", data=raw_data)
# Same hash → cache hit → no recomputation

# Different inputs trigger recomputation
different_data = np.random.rand(100, 100)
result3 = pipeline.execute(task="load_data", data=different_data)
# Different hash → cache miss → recomputed and cached

# Force recomputation (bypass cache)
result4 = pipeline.execute(task="load_data", data=raw_data, force_rebuild=True)

# Monitor caching
stats = pipeline.get_cache_stats()
print(f"Cache size: {stats['total_cache_size_mb']:.2f} MB")
print(f"Cached entries: {stats['num_cached_results']}")

# Clear cache between experiments
pipeline.clear_cache()
```

**When to use:**
- Sequential workflows where one step depends on previous outputs
- Computationally expensive analyses you'll run multiple times
- Iterative development - avoid recomputing stable intermediate results
- Multi-step data pipelines with expensive processing

## Tier 3: Data Caching

### SNODAS Data Caching
`SnowAnalysis` caches processed SNODAS snow data to avoid reprocessing the same geographic extents.

**Location:** `src/snow/analysis.py` (class: `SnowAnalysis`)

**How it works:**
- Uses MD5 hashing of extent tuple + SNODAS root directory
- Caches snow statistics as `.npz` files
- Automatically detects when extent or data directory changes

**When to use:**
- Processing SNODAS snow data for multiple analyses
- Rerunning analysis on same geographic region

## Choosing Which Cache to Use

| Use Case | Recommended Cache | Why |
|----------|-------------------|-----|
| Loading raw DEM files repeatedly | DEMCache | Avoids re-reading and merging files |
| Creating multiple 3D meshes with same DEM | MeshCache | Avoids expensive Blender mesh generation |
| Custom sequential analysis workflow | SnowAnalysisPipeline (subclass) | Generic caching + dependency tracking |
| Re-analyzing same SNODAS extent | SnowAnalysis (auto) | Built into snow processing workflow |
| One-off computation | None needed | Skip caching overhead |

## Performance Impact

**Typical cache benefits:**

- **First run (no cache):** Full computation time
- **Subsequent runs (cache hit):** 100-1000× faster
  - DEM loading: 10-100× faster
  - Mesh generation: 100-1000× faster
  - Task caching: 10-100× faster (avoids numpy operations)

**Cache overhead:**

- Cache checking: <10ms per task
- .npz I/O: 100-500ms (depends on data size)
- Metadata logging: <1ms per task

**Example timing:**

```
First run (no cache):
- load_dem: 2.1s
- compute_snow_stats: 3.4s
- calculate_scores: 1.5s
- Total: 7.0s

Second run (cache hit):
- load_dem: 0.2s (cache load)
- compute_snow_stats: 0.3s (cache load)
- calculate_scores: 0.4s (cache load)
- Total: 0.9s
```

## Disabling Caching

Caches can be disabled for development or testing:

```python
# Disable pipeline caching
pipeline = MyAnalysisPipeline(caching_enabled=False)

# Forces recomputation every run
result = pipeline.execute(task="my_task", force_rebuild=True)
```

## Cache Storage Locations

| Cache | Default Location | Configurable? |
|-------|------------------|---------------|
| DEMCache | `.dem_cache/` | Via constructor |
| MeshCache | `.mesh_cache/` | Via constructor |
| SnowAnalysisPipeline | `.snow_analysis_cache/` | Via `cache_dir` parameter |

## Debugging Cache Issues

### View cache contents:
```bash
ls -lah .snow_analysis_cache/
cat .snow_analysis_cache/load_dem_*_meta.json | jq
```

### Clear all caches:
```python
pipeline.clear_cache()
import shutil
shutil.rmtree(".dem_cache")
shutil.rmtree(".mesh_cache")
```

### Force recomputation:
```python
pipeline.execute(task="compute_snow_stats", force_rebuild=True, ...)
```

### Check if cache was used:
```python
stats_before = pipeline.get_cache_stats()
result = pipeline.execute(task="compute_snow_stats", ...)
stats_after = pipeline.get_cache_stats()

if stats_before['num_cached_results'] == stats_after['num_cached_results']:
    print("Cache was used (no new files)")
else:
    print("New computation (cache was missed)")
```

## Best Practices

1. **Use appropriate cache tier:** Don't use pipeline caching for one-time operations
2. **Monitor cache size:** Large cache directories can accumulate over time
3. **Clear old caches periodically:** Manual cleanup of `.cache` directories
4. **Use consistent parameters:** Different parameters create different cache entries
5. **Test without cache first:** Verify computation works before relying on caching
6. **Document cache dependencies:** If task outputs are cached, document input assumptions

## Further Reading

- See [examples/snow_analysis_pipeline_example.py](../examples/snow_analysis_pipeline_example.py) for detailed usage examples
- See [SNOW_SLEDDING.md](./SNOW_SLEDDING.md) for snow analysis workflow that could benefit from caching
- Terrain Maker's internal caching: [DEMCache](../src/terrain/cache.py), [MeshCache](../src/terrain/mesh_cache.py)
