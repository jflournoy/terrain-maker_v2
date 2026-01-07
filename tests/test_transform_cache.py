"""
Tests for TransformCache with dependency tracking.

TDD RED phase - these tests define the caching interface.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path


class TestTransformCacheBasics:
    """Tests for basic TransformCache functionality."""

    def test_transform_cache_can_be_imported(self):
        """Test that TransformCache can be imported."""
        from src.terrain.cache import TransformCache

        assert TransformCache is not None

    def test_transform_cache_init_creates_directory(self):
        """Test that TransformCache creates cache directory."""
        from src.terrain.cache import TransformCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "transform_cache"
            cache = TransformCache(cache_dir=cache_dir)

            assert cache_dir.exists()
            assert cache.cache_dir == cache_dir

    def test_transform_cache_disabled_skips_directory(self):
        """Test that disabled cache skips directory creation."""
        from src.terrain.cache import TransformCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "should_not_exist"
            cache = TransformCache(cache_dir=cache_dir, enabled=False)

            assert not cache_dir.exists()
            assert cache.enabled is False


class TestTransformCacheHashing:
    """Tests for cache key computation with dependency tracking."""

    def test_compute_transform_hash_basic(self):
        """Test computing hash from transform parameters."""
        from src.terrain.cache import TransformCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TransformCache(cache_dir=tmpdir)

            # Hash should be deterministic for same params
            hash1 = cache.compute_transform_hash(
                upstream_hash="abc123",
                transform_name="reproject",
                params={"target_crs": "EPSG:32617", "resolution": 30.0},
            )
            hash2 = cache.compute_transform_hash(
                upstream_hash="abc123",
                transform_name="reproject",
                params={"target_crs": "EPSG:32617", "resolution": 30.0},
            )

            assert hash1 == hash2
            assert isinstance(hash1, str)
            assert len(hash1) == 64  # SHA256 hex length

    def test_compute_transform_hash_different_params(self):
        """Test that different params produce different hashes."""
        from src.terrain.cache import TransformCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TransformCache(cache_dir=tmpdir)

            hash1 = cache.compute_transform_hash(
                upstream_hash="abc123",
                transform_name="smooth",
                params={"sigma_spatial": 3.0},
            )
            hash2 = cache.compute_transform_hash(
                upstream_hash="abc123",
                transform_name="smooth",
                params={"sigma_spatial": 5.0},
            )

            assert hash1 != hash2

    def test_compute_transform_hash_different_upstream(self):
        """Test that different upstream hash produces different result."""
        from src.terrain.cache import TransformCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TransformCache(cache_dir=tmpdir)

            hash1 = cache.compute_transform_hash(
                upstream_hash="abc123",
                transform_name="smooth",
                params={"sigma_spatial": 3.0},
            )
            hash2 = cache.compute_transform_hash(
                upstream_hash="xyz789",  # Different upstream
                transform_name="smooth",
                params={"sigma_spatial": 3.0},
            )

            assert hash1 != hash2

    def test_compute_transform_hash_handles_numpy_params(self):
        """Test that numpy arrays in params are handled correctly."""
        from src.terrain.cache import TransformCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TransformCache(cache_dir=tmpdir)

            # Should not raise error with numpy array params
            hash1 = cache.compute_transform_hash(
                upstream_hash="abc123",
                transform_name="custom",
                params={"bbox": np.array([-83.5, 42.0, -82.5, 43.0])},
            )

            assert isinstance(hash1, str)


class TestTransformCacheSaveLoad:
    """Tests for saving and loading cached transforms."""

    def test_save_transform_creates_file(self):
        """Test that save_transform creates cache file."""
        from src.terrain.cache import TransformCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TransformCache(cache_dir=tmpdir)

            data = np.random.rand(100, 100).astype(np.float32)
            cache_key = "test_hash_123"

            cache.save_transform(cache_key, data, transform_name="test")

            # Should create .npz file
            cache_path = cache.get_cache_path(cache_key, "test")
            assert cache_path.exists()

    def test_load_transform_returns_data(self):
        """Test that load_transform returns saved data."""
        from src.terrain.cache import TransformCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TransformCache(cache_dir=tmpdir)

            original = np.random.rand(100, 100).astype(np.float32)
            cache_key = "test_hash_456"

            cache.save_transform(cache_key, original, transform_name="test")
            loaded = cache.load_transform(cache_key, transform_name="test")

            assert loaded is not None
            np.testing.assert_array_almost_equal(loaded, original)

    def test_load_transform_cache_miss_returns_none(self):
        """Test that load_transform returns None on cache miss."""
        from src.terrain.cache import TransformCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TransformCache(cache_dir=tmpdir)

            result = cache.load_transform("nonexistent_hash", transform_name="test")

            assert result is None

    def test_save_transform_with_metadata(self):
        """Test saving transform with additional metadata."""
        from src.terrain.cache import TransformCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TransformCache(cache_dir=tmpdir)

            data = np.random.rand(50, 50).astype(np.float32)
            cache_key = "test_hash_789"
            metadata = {
                "upstream_hash": "parent_abc",
                "transform_name": "smooth",
                "params": {"sigma_spatial": 3.0},
            }

            cache.save_transform(cache_key, data, transform_name="smooth", metadata=metadata)

            # Metadata should be saved
            meta_path = cache.get_metadata_path(cache_key, "smooth")
            assert meta_path.exists()

    def test_disabled_cache_returns_none(self):
        """Test that disabled cache always returns None."""
        from src.terrain.cache import TransformCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TransformCache(cache_dir=tmpdir, enabled=False)

            data = np.random.rand(50, 50).astype(np.float32)
            cache.save_transform("test_key", data, transform_name="test")

            # Should return None even for "saved" data
            result = cache.load_transform("test_key", transform_name="test")
            assert result is None


class TestTransformCacheDependencyChain:
    """Tests for dependency chain tracking and invalidation."""

    def test_register_dependency_chain(self):
        """Test registering a chain of dependent transforms."""
        from src.terrain.cache import TransformCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TransformCache(cache_dir=tmpdir)

            # Register: dem_source -> reprojected -> smoothed -> water_mask
            cache.register_dependency("reprojected", upstream="dem_source")
            cache.register_dependency("smoothed", upstream="reprojected")
            cache.register_dependency("water_mask", upstream="smoothed")

            # Should track dependencies
            deps = cache.get_dependency_chain("water_mask")
            assert deps == ["dem_source", "reprojected", "smoothed", "water_mask"]

    def test_get_full_cache_key(self):
        """Test computing full cache key from dependency chain."""
        from src.terrain.cache import TransformCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TransformCache(cache_dir=tmpdir)

            # Set up transforms with params
            cache.register_transform(
                "reprojected",
                upstream="dem_source",
                params={"target_crs": "EPSG:32617"},
            )
            cache.register_transform(
                "smoothed",
                upstream="reprojected",
                params={"sigma_spatial": 3.0},
            )

            # Full key should incorporate entire chain
            key1 = cache.get_full_cache_key("smoothed", source_hash="abc123")
            key2 = cache.get_full_cache_key("smoothed", source_hash="abc123")

            assert key1 == key2

            # Different source should produce different key
            key3 = cache.get_full_cache_key("smoothed", source_hash="xyz789")
            assert key1 != key3

    def test_invalidate_downstream(self):
        """Test that changing upstream invalidates downstream caches."""
        from src.terrain.cache import TransformCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TransformCache(cache_dir=tmpdir)

            # Register chain
            cache.register_dependency("reprojected", upstream="dem_source")
            cache.register_dependency("smoothed", upstream="reprojected")

            # Save some data
            data1 = np.random.rand(50, 50).astype(np.float32)
            data2 = np.random.rand(50, 50).astype(np.float32)

            cache.save_transform("key_reprojected", data1, transform_name="reprojected")
            cache.save_transform("key_smoothed", data2, transform_name="smoothed")

            # Invalidate reprojected should also invalidate smoothed
            deleted = cache.invalidate_downstream("reprojected")

            assert deleted >= 2  # Both reprojected and smoothed


class TestTransformCacheIntegration:
    """Integration tests for realistic caching scenarios."""

    def test_cache_terrain_transform_pipeline(self):
        """Test caching a realistic terrain transform pipeline."""
        from src.terrain.cache import TransformCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TransformCache(cache_dir=tmpdir)

            # Simulate terrain pipeline
            source_hash = "dem_files_hash_abc"

            # Step 1: Reprojection
            reproject_key = cache.compute_transform_hash(
                upstream_hash=source_hash,
                transform_name="reproject",
                params={"target_crs": "EPSG:32617", "resolution": 30.0},
            )
            reprojected_dem = np.random.rand(1000, 1000).astype(np.float32)
            cache.save_transform(reproject_key, reprojected_dem, transform_name="reproject")

            # Step 2: Smoothing (depends on reprojection)
            smooth_key = cache.compute_transform_hash(
                upstream_hash=reproject_key,
                transform_name="smooth",
                params={"sigma_spatial": 3.0, "sigma_intensity": 0.05},
            )
            smoothed_dem = np.random.rand(1000, 1000).astype(np.float32)
            cache.save_transform(smooth_key, smoothed_dem, transform_name="smooth")

            # Step 3: Water detection (depends on smoothed)
            water_key = cache.compute_transform_hash(
                upstream_hash=smooth_key,
                transform_name="water_detection",
                params={"slope_threshold": 0.5},
            )
            water_mask = np.random.randint(0, 2, (1000, 1000)).astype(np.uint8)
            cache.save_transform(water_key, water_mask, transform_name="water_detection")

            # Verify all cached
            assert cache.load_transform(reproject_key, "reproject") is not None
            assert cache.load_transform(smooth_key, "smooth") is not None
            assert cache.load_transform(water_key, "water_detection") is not None

    def test_cache_hit_on_second_run(self):
        """Test that second run with same params gets cache hit."""
        from src.terrain.cache import TransformCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TransformCache(cache_dir=tmpdir)

            # First run - compute and cache
            key = cache.compute_transform_hash(
                upstream_hash="source_123",
                transform_name="expensive_op",
                params={"param1": 42},
            )

            result1 = cache.load_transform(key, "expensive_op")
            assert result1 is None  # Cache miss

            # Simulate expensive computation
            computed = np.ones((100, 100), dtype=np.float32) * 42
            cache.save_transform(key, computed, transform_name="expensive_op")

            # Second run - should hit cache
            result2 = cache.load_transform(key, "expensive_op")
            assert result2 is not None
            np.testing.assert_array_equal(result2, computed)
