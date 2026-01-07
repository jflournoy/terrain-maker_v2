"""
Tests for target-style pipeline caching with proper invalidation.

TDD RED phase - these tests define the caching behavior we want:
1. Cache hits when inputs unchanged
2. Cache misses when ANY upstream parameter changes
3. Downstream targets invalidated when upstream changes
4. Hash-based verification prevents stale cache usage
"""

import pytest
import numpy as np
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch


class TestPipelineCacheBasics:
    """Tests for basic PipelineCache functionality."""

    def test_pipeline_cache_can_be_imported(self):
        """Test that PipelineCache can be imported."""
        from src.terrain.cache import PipelineCache

        assert PipelineCache is not None

    def test_pipeline_cache_init_creates_directory(self):
        """Test that PipelineCache creates cache directory."""
        from src.terrain.cache import PipelineCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "pipeline_cache"
            cache = PipelineCache(cache_dir=cache_dir)

            assert cache_dir.exists()
            assert cache.cache_dir == cache_dir

    def test_pipeline_cache_disabled_mode(self):
        """Test that disabled cache doesn't create directory or cache."""
        from src.terrain.cache import PipelineCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "should_not_exist"
            cache = PipelineCache(cache_dir=cache_dir, enabled=False)

            assert not cache_dir.exists()
            assert cache.enabled is False


class TestTargetDefinition:
    """Tests for defining pipeline targets with dependencies."""

    def test_define_target_with_no_dependencies(self):
        """Test defining a root target (no upstream dependencies)."""
        from src.terrain.cache import PipelineCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(cache_dir=tmpdir)

            # Define a root target (like DEM loading)
            cache.define_target(
                name="dem_loaded",
                params={"directory": "/data/dem", "pattern": "*.hgt"},
            )

            assert "dem_loaded" in cache.targets
            assert cache.targets["dem_loaded"]["dependencies"] == []

    def test_define_target_with_dependencies(self):
        """Test defining a target that depends on another."""
        from src.terrain.cache import PipelineCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(cache_dir=tmpdir)

            # Define chain: dem_loaded -> reprojected
            cache.define_target(
                name="dem_loaded",
                params={"directory": "/data/dem"},
            )
            cache.define_target(
                name="reprojected",
                params={"target_crs": "EPSG:32617"},
                dependencies=["dem_loaded"],
            )

            assert cache.targets["reprojected"]["dependencies"] == ["dem_loaded"]

    def test_define_target_with_multiple_dependencies(self):
        """Test target with multiple upstream dependencies."""
        from src.terrain.cache import PipelineCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(cache_dir=tmpdir)

            # colors depends on both dem and scores
            cache.define_target(name="dem", params={})
            cache.define_target(name="scores", params={})
            cache.define_target(
                name="colors",
                params={"colormap": "viridis"},
                dependencies=["dem", "scores"],
            )

            assert set(cache.targets["colors"]["dependencies"]) == {"dem", "scores"}


class TestCacheKeyComputation:
    """Tests for computing cache keys from target params and dependencies."""

    def test_cache_key_deterministic(self):
        """Test that same params produce same cache key."""
        from src.terrain.cache import PipelineCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(cache_dir=tmpdir)

            cache.define_target(name="test", params={"a": 1, "b": "hello"})

            key1 = cache.compute_target_key("test")
            key2 = cache.compute_target_key("test")

            assert key1 == key2
            assert isinstance(key1, str)
            assert len(key1) == 64  # SHA256 hex

    def test_cache_key_changes_with_params(self):
        """Test that different params produce different keys."""
        from src.terrain.cache import PipelineCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(cache_dir=tmpdir)

            cache.define_target(name="test", params={"resolution": 30})
            key1 = cache.compute_target_key("test")

            # Update params
            cache.define_target(name="test", params={"resolution": 60})
            key2 = cache.compute_target_key("test")

            assert key1 != key2

    def test_cache_key_incorporates_upstream_keys(self):
        """CRITICAL: Cache key must incorporate ALL upstream keys."""
        from src.terrain.cache import PipelineCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(cache_dir=tmpdir)

            # Chain: A -> B -> C
            cache.define_target(name="A", params={"val": 1})
            cache.define_target(name="B", params={"val": 2}, dependencies=["A"])
            cache.define_target(name="C", params={"val": 3}, dependencies=["B"])

            key_c_original = cache.compute_target_key("C")

            # Change A's params - C's key MUST change
            cache.define_target(name="A", params={"val": 999})

            key_c_after = cache.compute_target_key("C")

            assert key_c_original != key_c_after, (
                "Changing upstream 'A' must invalidate downstream 'C'"
            )

    def test_cache_key_handles_file_mtimes(self):
        """Test that file modification times are incorporated into keys."""
        from src.terrain.cache import PipelineCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(cache_dir=tmpdir)

            # Create a test file
            test_file = Path(tmpdir) / "test.hgt"
            test_file.write_text("initial")

            cache.define_target(
                name="dem",
                params={"directory": tmpdir, "pattern": "*.hgt"},
                file_inputs=[test_file],
            )

            key1 = cache.compute_target_key("dem")

            # Modify file (wait to ensure different mtime)
            time.sleep(0.01)
            test_file.write_text("modified")

            key2 = cache.compute_target_key("dem")

            assert key1 != key2, "File modification must invalidate cache"


class TestCacheHitMiss:
    """Tests for cache hit/miss behavior."""

    def test_cache_miss_on_first_run(self):
        """Test that first run is always a cache miss."""
        from src.terrain.cache import PipelineCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(cache_dir=tmpdir)

            cache.define_target(name="test", params={"val": 1})

            result = cache.get_cached("test")
            assert result is None

    def test_cache_hit_after_save(self):
        """Test that saved data can be retrieved."""
        from src.terrain.cache import PipelineCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(cache_dir=tmpdir)

            cache.define_target(name="test", params={"val": 1})

            # Save some data
            data = np.random.rand(100, 100).astype(np.float32)
            cache.save_target("test", data)

            # Should hit cache
            result = cache.get_cached("test")
            assert result is not None
            np.testing.assert_array_almost_equal(result, data)

    def test_cache_miss_after_param_change(self):
        """Test that changing params causes cache miss."""
        from src.terrain.cache import PipelineCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(cache_dir=tmpdir)

            cache.define_target(name="test", params={"val": 1})
            data = np.random.rand(50, 50).astype(np.float32)
            cache.save_target("test", data)

            # Change params
            cache.define_target(name="test", params={"val": 2})

            # Should miss cache
            result = cache.get_cached("test")
            assert result is None

    def test_cache_miss_when_upstream_changes(self):
        """CRITICAL: Upstream changes must cause downstream cache miss."""
        from src.terrain.cache import PipelineCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(cache_dir=tmpdir)

            # Define chain: A -> B
            cache.define_target(name="A", params={"val": 1})
            cache.define_target(name="B", params={"val": 2}, dependencies=["A"])

            # Cache both
            data_b = np.ones((10, 10), dtype=np.float32)
            cache.save_target("B", data_b)

            # Verify cache hit
            assert cache.get_cached("B") is not None

            # Change A's params
            cache.define_target(name="A", params={"val": 999})

            # B must miss cache even though B's own params didn't change
            result = cache.get_cached("B")
            assert result is None, (
                "Changing upstream 'A' must invalidate downstream 'B'"
            )


class TestCacheWithMetadata:
    """Tests for caching with additional metadata (transform, etc.)."""

    def test_save_and_load_with_transform(self):
        """Test saving/loading numpy array with Affine transform."""
        from src.terrain.cache import PipelineCache
        from rasterio import Affine

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(cache_dir=tmpdir)

            cache.define_target(name="dem", params={"resolution": 30})

            # Save array with transform
            data = np.random.rand(100, 100).astype(np.float32)
            transform = Affine(30.0, 0, -83.5, 0, -30.0, 43.0)

            cache.save_target("dem", data, metadata={"transform": transform})

            # Load and verify
            result, meta = cache.get_cached("dem", return_metadata=True)
            assert result is not None
            np.testing.assert_array_almost_equal(result, data)
            assert "transform" in meta
            assert meta["transform"] == transform

    def test_save_multiple_arrays(self):
        """Test saving multiple arrays for one target."""
        from src.terrain.cache import PipelineCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(cache_dir=tmpdir)

            cache.define_target(name="colors", params={})

            # Save multiple arrays
            colors = np.random.rand(1000, 4).astype(np.float32)
            water_mask = np.random.randint(0, 2, (100, 100)).astype(np.uint8)

            cache.save_target(
                "colors",
                {"colors": colors, "water_mask": water_mask},
            )

            # Load
            result = cache.get_cached("colors")
            assert "colors" in result
            assert "water_mask" in result
            np.testing.assert_array_almost_equal(result["colors"], colors)
            np.testing.assert_array_equal(result["water_mask"], water_mask)


class TestCacheDisabledMode:
    """Tests for disabled cache mode (always rebuilds)."""

    def test_disabled_cache_never_saves(self):
        """Test that disabled cache doesn't save anything."""
        from src.terrain.cache import PipelineCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(cache_dir=tmpdir, enabled=False)

            cache.define_target(name="test", params={})
            data = np.ones((10, 10), dtype=np.float32)
            cache.save_target("test", data)

            # Should return None
            result = cache.get_cached("test")
            assert result is None

    def test_disabled_cache_no_files_created(self):
        """Test that disabled cache creates no cache files."""
        from src.terrain.cache import PipelineCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache = PipelineCache(cache_dir=cache_dir, enabled=False)

            cache.define_target(name="test", params={})
            data = np.ones((10, 10), dtype=np.float32)
            cache.save_target("test", data)

            # Directory shouldn't exist
            assert not cache_dir.exists()


class TestDeepDependencyChain:
    """Tests for deep dependency chains (4+ levels)."""

    def test_deep_chain_invalidation(self):
        """Test that changes propagate through deep chains."""
        from src.terrain.cache import PipelineCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(cache_dir=tmpdir)

            # Chain: A -> B -> C -> D -> E
            cache.define_target(name="A", params={"v": 1})
            cache.define_target(name="B", params={"v": 2}, dependencies=["A"])
            cache.define_target(name="C", params={"v": 3}, dependencies=["B"])
            cache.define_target(name="D", params={"v": 4}, dependencies=["C"])
            cache.define_target(name="E", params={"v": 5}, dependencies=["D"])

            # Cache E
            data = np.ones((5, 5), dtype=np.float32)
            cache.save_target("E", data)
            assert cache.get_cached("E") is not None

            # Change A (root)
            cache.define_target(name="A", params={"v": 999})

            # E must miss cache
            assert cache.get_cached("E") is None, (
                "Root change must invalidate entire chain"
            )

    def test_diamond_dependency(self):
        """Test diamond dependency pattern: A -> B,C -> D."""
        from src.terrain.cache import PipelineCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(cache_dir=tmpdir)

            #     A
            #    / \
            #   B   C
            #    \ /
            #     D
            cache.define_target(name="A", params={"v": 1})
            cache.define_target(name="B", params={"v": 2}, dependencies=["A"])
            cache.define_target(name="C", params={"v": 3}, dependencies=["A"])
            cache.define_target(name="D", params={"v": 4}, dependencies=["B", "C"])

            # Cache D
            data = np.ones((5, 5), dtype=np.float32)
            cache.save_target("D", data)
            assert cache.get_cached("D") is not None

            # Change A
            cache.define_target(name="A", params={"v": 999})

            # D must miss cache
            assert cache.get_cached("D") is None


class TestCacheCleanup:
    """Tests for cache cleanup and management."""

    def test_clear_target(self):
        """Test clearing a specific target's cache."""
        from src.terrain.cache import PipelineCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(cache_dir=tmpdir)

            cache.define_target(name="A", params={})
            cache.define_target(name="B", params={})

            cache.save_target("A", np.ones((5, 5)))
            cache.save_target("B", np.ones((5, 5)))

            # Clear only A
            cache.clear_target("A")

            assert cache.get_cached("A") is None
            assert cache.get_cached("B") is not None

    def test_clear_all(self):
        """Test clearing entire cache."""
        from src.terrain.cache import PipelineCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(cache_dir=tmpdir)

            cache.define_target(name="A", params={})
            cache.define_target(name="B", params={})

            cache.save_target("A", np.ones((5, 5)))
            cache.save_target("B", np.ones((5, 5)))

            cache.clear_all()

            assert cache.get_cached("A") is None
            assert cache.get_cached("B") is None


class TestRealisticTerrainScenarios:
    """Tests for realistic terrain rendering scenarios."""

    def test_terrain_pipeline_cache_workflow(self):
        """Test a realistic terrain rendering pipeline with caching."""
        from src.terrain.cache import PipelineCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(cache_dir=tmpdir)

            # Define terrain pipeline targets
            cache.define_target(
                name="dem_loaded",
                params={"directory": "/data/dem", "pattern": "*.hgt"},
            )
            cache.define_target(
                name="dem_reprojected",
                params={"src_crs": "EPSG:4326", "dst_crs": "EPSG:32617"},
                dependencies=["dem_loaded"],
            )
            cache.define_target(
                name="dem_transformed",
                params={"flip": "horizontal", "scale": 0.0001},
                dependencies=["dem_reprojected"],
            )
            cache.define_target(
                name="colors_computed",
                params={"colormap": "mako", "gamma": 0.5},
                dependencies=["dem_transformed"],
            )
            cache.define_target(
                name="mesh_created",
                params={"height_scale": 30.0, "boundary_extension": True},
                dependencies=["colors_computed"],
            )

            # Simulate first run: all cache misses
            assert cache.get_cached("dem_loaded") is None
            assert cache.get_cached("dem_reprojected") is None
            assert cache.get_cached("dem_transformed") is None
            assert cache.get_cached("colors_computed") is None
            assert cache.get_cached("mesh_created") is None

            # Save results
            cache.save_target("dem_loaded", np.ones((100, 100)))
            cache.save_target("dem_reprojected", np.ones((100, 100)))
            cache.save_target("dem_transformed", np.ones((100, 100)))
            cache.save_target("colors_computed", np.ones((100, 100, 3)))
            cache.save_target("mesh_created", np.ones((1000, 3)))

            # Second run: all cache hits
            assert cache.get_cached("dem_loaded") is not None
            assert cache.get_cached("dem_reprojected") is not None
            assert cache.get_cached("dem_transformed") is not None
            assert cache.get_cached("colors_computed") is not None
            assert cache.get_cached("mesh_created") is not None

    def test_colormap_change_invalidates_mesh(self):
        """Test that changing colormap invalidates downstream mesh cache."""
        from src.terrain.cache import PipelineCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(cache_dir=tmpdir)

            # Define minimal pipeline: colors -> mesh
            cache.define_target(
                name="colors",
                params={"colormap": "mako", "gamma": 0.5},
            )
            cache.define_target(
                name="mesh",
                params={"height_scale": 30.0},
                dependencies=["colors"],
            )

            # Save mesh
            cache.save_target("mesh", np.ones((1000, 3)))
            assert cache.get_cached("mesh") is not None

            # Change colormap (user picks different visualization)
            cache.define_target(
                name="colors",
                params={"colormap": "viridis", "gamma": 1.0},  # Changed!
            )

            # Mesh should now be invalidated
            assert cache.get_cached("mesh") is None

    def test_render_settings_dont_invalidate_mesh(self):
        """Test that render-only settings don't invalidate upstream caches."""
        from src.terrain.cache import PipelineCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(cache_dir=tmpdir)

            # Define pipeline with separate render target
            cache.define_target(
                name="mesh",
                params={"height_scale": 30.0},
            )
            cache.define_target(
                name="render",
                params={"width": 1920, "height": 1080, "samples": 2048},
                dependencies=["mesh"],
            )

            # Save mesh
            cache.save_target("mesh", np.ones((1000, 3)))

            # Change render settings (should NOT invalidate mesh)
            cache.define_target(
                name="render",
                params={"width": 3840, "height": 2160, "samples": 8192},  # Changed!
                dependencies=["mesh"],
            )

            # Mesh should still be cached
            assert cache.get_cached("mesh") is not None

    def test_height_scale_invalidates_mesh_only(self):
        """Test that height scale change invalidates mesh but not colors."""
        from src.terrain.cache import PipelineCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(cache_dir=tmpdir)

            cache.define_target(
                name="colors",
                params={"colormap": "mako"},
            )
            cache.define_target(
                name="mesh",
                params={"height_scale": 30.0},
                dependencies=["colors"],
            )

            # Save both
            cache.save_target("colors", np.ones((100, 100, 3)))
            cache.save_target("mesh", np.ones((1000, 3)))

            # Change height scale
            cache.define_target(
                name="mesh",
                params={"height_scale": 50.0},  # Changed!
                dependencies=["colors"],
            )

            # Colors still cached, mesh invalidated
            assert cache.get_cached("colors") is not None
            assert cache.get_cached("mesh") is None

    def test_road_toggle_invalidates_downstream(self):
        """Test that enabling/disabling roads invalidates mesh."""
        from src.terrain.cache import PipelineCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(cache_dir=tmpdir)

            # Pipeline: colors -> roads -> mesh
            cache.define_target(name="colors", params={})
            cache.define_target(
                name="roads",
                params={"enabled": False, "road_types": []},
                dependencies=["colors"],
            )
            cache.define_target(
                name="mesh",
                params={},
                dependencies=["roads"],
            )

            # Save everything
            cache.save_target("colors", np.ones((100, 100, 3)))
            cache.save_target("roads", np.zeros((100, 100)))
            cache.save_target("mesh", np.ones((1000, 3)))

            # Enable roads
            cache.define_target(
                name="roads",
                params={"enabled": True, "road_types": ["motorway"]},  # Changed!
                dependencies=["colors"],
            )

            # Colors still valid, roads and mesh invalidated
            assert cache.get_cached("colors") is not None
            assert cache.get_cached("roads") is None
            assert cache.get_cached("mesh") is None


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_get_cached_undefined_target(self):
        """Test getting cache for undefined target returns None."""
        from src.terrain.cache import PipelineCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(cache_dir=tmpdir)

            result = cache.get_cached("nonexistent")
            assert result is None

    def test_circular_dependency_detection(self):
        """Test that circular dependencies are detected."""
        from src.terrain.cache import PipelineCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(cache_dir=tmpdir)

            cache.define_target(name="A", params={}, dependencies=["C"])
            cache.define_target(name="B", params={}, dependencies=["A"])

            # This would create cycle: A -> B -> C -> A
            with pytest.raises(ValueError, match="(?i)circular"):
                cache.define_target(name="C", params={}, dependencies=["B"])

    def test_numpy_array_params(self):
        """Test that numpy arrays in params are handled correctly."""
        from src.terrain.cache import PipelineCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(cache_dir=tmpdir)

            bbox = np.array([-83.5, 42.0, -82.5, 43.0])
            cache.define_target(name="test", params={"bbox": bbox})

            # Should compute key without error
            key = cache.compute_target_key("test")
            assert isinstance(key, str)

    def test_empty_params(self):
        """Test target with empty params."""
        from src.terrain.cache import PipelineCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = PipelineCache(cache_dir=tmpdir)

            cache.define_target(name="test", params={})

            key = cache.compute_target_key("test")
            assert isinstance(key, str)

            # Should still be able to save/load
            data = np.ones((5, 5))
            cache.save_target("test", data)
            result = cache.get_cached("test")
            assert result is not None
