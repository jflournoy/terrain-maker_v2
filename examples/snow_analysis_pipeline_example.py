"""
Example: Using SnowAnalysisPipeline with Dependency Caching

This example demonstrates how to use the SnowAnalysisPipeline class
to run sequential snow analysis computations with automatic caching.

The pipeline provides:
- Task dependency tracking and execution ordering
- Hash-based cache invalidation
- Force rebuild capabilities
- Cache statistics and management
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.snow.analysis import SnowAnalysisPipeline

# ============================================================================
# EXAMPLE 1: Basic Pipeline Usage
# ============================================================================

def example_basic_usage():
    """Basic example of creating and running a pipeline."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Pipeline Usage")
    print("=" * 70)

    # Create a pipeline with a custom cache directory
    cache_dir = Path("./demo_snow_cache")
    pipeline = SnowAnalysisPipeline(cache_dir=cache_dir)

    # Create mock DEM data
    dem = np.random.rand(50, 50) * 3000  # Elevation in meters

    # Execute the full pipeline
    print("\nExecuting full pipeline...")
    result = pipeline.execute_all(
        dem=dem,
        extent=(0, 0, 50000, 50000),  # 50km x 50km region
        snodas_dir="/path/to/snodas"
    )

    print(f"Pipeline result keys: {result.keys()}")
    print(f"Sledding score shape: {result['sledding_score'].shape}")


# ============================================================================
# EXAMPLE 2: Task Dependencies
# ============================================================================

def example_task_dependencies():
    """Show how to explore task dependencies."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Understanding Task Dependencies")
    print("=" * 70)

    pipeline = SnowAnalysisPipeline()

    # Explain dependencies for each task
    for task in ["load_dem", "compute_snow_stats", "calculate_scores"]:
        explanation = pipeline.explain(task)
        print(f"\n{task}:")
        print(f"  {explanation}")


# ============================================================================
# EXAMPLE 3: Caching Behavior
# ============================================================================

def example_caching_behavior():
    """Demonstrate cache hits and misses."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Cache Hits and Misses")
    print("=" * 70)

    cache_dir = Path("./demo_snow_cache_v2")
    pipeline = SnowAnalysisPipeline(cache_dir=cache_dir)

    dem = np.random.rand(30, 30) * 2000
    extent = (0, 0, 30000, 30000)

    # First run - miss, computes result
    print("\nFirst run (cache miss, recompute)...")
    result1 = pipeline.execute(
        task="compute_snow_stats",
        dem=dem,
        extent=extent,
        snodas_dir="/path/to/snodas"
    )
    print(f"  Computed snow stats with shape: {result1['median_max_depth'].shape}")

    # Second run - hit, loads from cache
    print("\nSecond run (cache hit, load from cache)...")
    result2 = pipeline.execute(
        task="compute_snow_stats",
        dem=dem,
        extent=extent,
        snodas_dir="/path/to/snodas"
    )
    print(f"  Loaded from cache: {np.array_equal(result1['median_max_depth'], result2['median_max_depth'])}")

    # Change inputs - miss, recomputes
    print("\nThird run with different DEM (cache miss, recompute)...")
    dem_different = np.random.rand(30, 30) * 2500  # Different elevations
    result3 = pipeline.execute(
        task="compute_snow_stats",
        dem=dem_different,
        extent=extent,
        snodas_dir="/path/to/snodas"
    )
    print(f"  Different input triggered recomputation")


# ============================================================================
# EXAMPLE 4: Cache Management
# ============================================================================

def example_cache_management():
    """Show how to manage cache."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Cache Management")
    print("=" * 70)

    cache_dir = Path("./demo_snow_cache_v3")
    pipeline = SnowAnalysisPipeline(cache_dir=cache_dir)

    # Run some computations
    dem = np.random.rand(20, 20) * 1500
    pipeline.execute_all(
        dem=dem,
        extent=(0, 0, 20000, 20000),
        snodas_dir="/path/to/snodas"
    )

    # Check cache stats
    print("\nCache Statistics:")
    stats = pipeline.get_cache_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Clear cache
    print("\nClearing cache...")
    pipeline.clear_cache()
    stats_after = pipeline.get_cache_stats()
    print(f"  Cache files after clear: {stats_after['num_cached_results']}")


# ============================================================================
# EXAMPLE 5: Force Rebuild
# ============================================================================

def example_force_rebuild():
    """Show how to force recomputation despite cache."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Force Rebuild")
    print("=" * 70)

    cache_dir = Path("./demo_snow_cache_v4")
    pipeline = SnowAnalysisPipeline(cache_dir=cache_dir)

    dem = np.random.rand(25, 25) * 1800
    extent = (0, 0, 25000, 25000)

    # First run - computes
    print("\nFirst run (compute)...")
    result1 = pipeline.execute(
        task="compute_snow_stats",
        dem=dem,
        extent=extent,
        snodas_dir="/path/to/snodas"
    )

    # Second run with force_rebuild=True
    print("\nSecond run with force_rebuild=True...")
    result2 = pipeline.execute(
        task="compute_snow_stats",
        dem=dem,
        extent=extent,
        snodas_dir="/path/to/snodas",
        force_rebuild=True  # Bypass cache even though inputs are identical
    )
    print(f"  Results are identical: {np.array_equal(result1['median_max_depth'], result2['median_max_depth'])}")
    print(f"  But computation was redone (force_rebuild=True)")


# ============================================================================
# EXAMPLE 6: Using Without Caching
# ============================================================================

def example_without_caching():
    """Show how to run pipeline without caching."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Pipeline Without Caching")
    print("=" * 70)

    # Create pipeline with caching disabled
    pipeline = SnowAnalysisPipeline(caching_enabled=False)

    dem = np.random.rand(20, 20) * 1500

    print("\nRunning pipeline with caching_enabled=False...")
    result = pipeline.execute_all(
        dem=dem,
        extent=(0, 0, 20000, 20000),
        snodas_dir="/path/to/snodas"
    )

    stats = pipeline.get_cache_stats()
    print(f"  Cache enabled: {stats['cache_enabled']}")
    print(f"  Cached results: {stats['num_cached_results']}")


if __name__ == "__main__":
    """Run all examples."""
    try:
        example_basic_usage()
        example_task_dependencies()
        example_caching_behavior()
        example_cache_management()
        example_force_rebuild()
        example_without_caching()

        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
