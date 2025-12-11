# Caching in Terrain Maker

Terrain Maker uses a **two-tier internal caching system** to optimize performance and avoid redundant computations:

1. **Subsystem Caching** - Cache expensive I/O and transformation operations (DEMCache, MeshCache)
2. **Data Caching** - Cache preprocessed SNODAS snow data

## Internal Caching Patterns

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

## Tier 2: Gridded Data Pipeline Caching

### GriddedDataLoader
Generic loader for external gridded datasets (SNODAS, temperature, precipitation, etc.) with pipeline-based caching.

**Location:** `src/terrain/gridded_data.py` (class: `GriddedDataLoader`)

**How it works:**
- User defines processing pipeline as sequence of (name, function, kwargs) tuples
- Each pipeline step is cached independently as `.npz` file
- Cache key includes: step name + function code hash + kwargs + upstream dependencies
- **Smart invalidation:** If step N changes, steps N, N+1, N+2... are invalidated (but earlier steps are reused)

**Pipeline Caching Example:**

```python
from src.terrain.gridded_data import GriddedDataLoader

# Define processing steps
def load_data(source, extent, target_shape):
    # Load and crop data (expensive I/O)
    return {"raw": load_and_crop(source, extent, target_shape)}

def compute_stats(input_data):
    # Compute statistics (fast numpy)
    return {"mean": input_data["raw"].mean(axis=0)}

# Build pipeline
pipeline = [
    ("load", load_data, {}),
    ("stats", compute_stats, {}),
]

# Run with caching
loader = GriddedDataLoader(terrain, cache_dir=Path(".cache"))
result = loader.run_pipeline(
    data_source="/path/to/data",
    pipeline=pipeline,
    cache_name="my_analysis"
)
```

**Caching Behavior - Smart Invalidation:**

First run (no cache): All steps execute
```
Step 1: load → Cached
Step 2: stats → Cached
```

Second run (all cached): All steps reuse cache
```
Step 1: load → Cache HIT
Step 2: stats → Cache HIT
```

Third run (after changing step 2): Only step 2 recomputes
```
Step 1: load → Cache HIT (reused!)
Step 2: stats → RECOMPUTED (code changed)
```

**When to use:**
- Processing external gridded datasets (SNODAS, temperature, precipitation)
- Multi-step data transformation pipelines
- Iterative development on analysis parameters
- Example: See `examples/detroit_snow_sledding.py` for SNODAS pipeline

## How Terrain Maker Uses Caching Internally

| Use Case | Caching Mechanism | How It Works |
|----------|-------------------|--------------|
| Loading raw DEM files repeatedly | DEMCache | Avoids re-reading and merging files from disk |
| Creating multiple 3D meshes with same DEM | MeshCache | Avoids expensive Blender mesh generation |
| Processing SNODAS or other gridded data | GriddedDataLoader | Caches each pipeline step independently |
| One-off computation | No caching | Skip caching overhead |

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

## Cache Storage Locations

| Cache | Default Location | Configurable? |
|-------|------------------|---------------|
| DEMCache | `.dem_cache/` | Via constructor |
| MeshCache | `.mesh_cache/` | Via constructor |
| GriddedDataLoader | `.gridded_data_cache/` | Via constructor |

## Debugging Cache Issues

### View cache contents:
```bash
ls -lah .dem_cache/
ls -lah .mesh_cache/
ls -lah .gridded_data_cache/
```

### Clear caches:
```bash
rm -rf .dem_cache
rm -rf .mesh_cache
rm -rf .gridded_data_cache
```

## Best Practices

1. **Monitor cache size:** Large cache directories can accumulate over time
2. **Clear old caches periodically:** Remove `.dem_cache`, `.mesh_cache`, and `.gridded_data_cache` directories when restarting analysis
3. **Understand cache invalidation:**
   - DEMCache: Uses source file metadata (paths + modification times)
   - MeshCache: Uses DEM hash + mesh parameters
   - GriddedDataLoader: Uses function source code hash + kwargs + upstream dependencies (smart invalidation)
4. **Use for expensive operations:** Caching is most beneficial for:
   - DEM loading and merging
   - 3D mesh generation
   - External gridded data processing (SNODAS, precipitation, temperature)
5. **Pipeline parameter tuning:** GriddedDataLoader's smart invalidation makes it ideal for iterative parameter adjustment - upstream steps stay cached while only modified steps recompute

## Further Reading

- **Cache implementations:**
  - `src/terrain/cache.py` - DEMCache for DEM caching
  - `src/terrain/mesh_cache.py` - MeshCache for 3D mesh caching
  - `src/terrain/gridded_data.py` - GriddedDataLoader with pipeline caching

- **Example usage:** See `examples/detroit_snow_sledding.py` for a real-world example using GriddedDataLoader for SNODAS snow data processing
