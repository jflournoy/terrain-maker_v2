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

## Tier 2: Data Caching

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

## How Terrain Maker Uses Caching Internally

| Use Case | Caching Mechanism | How It Works |
|----------|-------------------|--------------|
| Loading raw DEM files repeatedly | DEMCache | Avoids re-reading and merging files from disk |
| Creating multiple 3D meshes with same DEM | MeshCache | Avoids expensive Blender mesh generation |
| Re-analyzing same SNODAS extent | SnowAnalysis (auto) | Built into snow processing workflow |
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

## Debugging Cache Issues

### View cache contents:
```bash
ls -lah .dem_cache/
ls -lah .mesh_cache/
```

### Clear caches:
```bash
rm -rf .dem_cache
rm -rf .mesh_cache
```

## Best Practices

1. **Monitor cache size:** Large cache directories can accumulate over time
2. **Clear old caches periodically:** Remove `.dem_cache` and `.mesh_cache` directories when restarting analysis
3. **Understand cache invalidation:** DEMCache uses source file metadata (paths + mod times), MeshCache uses DEM hash + mesh parameters
4. **Use for expensive operations:** Caching is most beneficial for DEM loading and mesh generation

## Further Reading

- **Internal cache implementations:** [DEMCache](../src/terrain/cache.py), [MeshCache](../src/terrain/mesh_cache.py), [SnowAnalysis](../src/snow/analysis.py)
- **Example usage:** See [examples/detroit_snow_sledding.py](../examples/detroit_snow_sledding.py) for a real-world example using Terrain Maker's caching system
