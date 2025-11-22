"""
Test suite for DEM caching module.

Tests the DEMCache class for hash validation, save/load operations,
invalidation, and cache management.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from src.terrain.cache import DEMCache
from rasterio import Affine


class TestDEMCacheHashComputation(unittest.TestCase):
    """Test DEM source hash computation."""

    def setUp(self):
        """Create temporary directory with test files."""
        self.test_dir = tempfile.mkdtemp()
        self.cache_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directories."""
        shutil.rmtree(self.test_dir)
        shutil.rmtree(self.cache_dir)

    def test_compute_source_hash_returns_string(self):
        """Test that compute_source_hash returns a valid hash string."""
        # Create test files
        Path(self.test_dir, "dem1.hgt").touch()
        Path(self.test_dir, "dem2.hgt").touch()

        cache = DEMCache(cache_dir=Path(self.cache_dir))
        hash_val = cache.compute_source_hash(self.test_dir, pattern='*.hgt')

        self.assertIsInstance(hash_val, str)
        self.assertEqual(len(hash_val), 64)  # SHA256 hash is 64 hex chars

    def test_compute_source_hash_is_deterministic(self):
        """Test that same files produce same hash."""
        Path(self.test_dir, "dem1.hgt").touch()
        Path(self.test_dir, "dem2.hgt").touch()

        cache = DEMCache(cache_dir=Path(self.cache_dir))
        hash1 = cache.compute_source_hash(self.test_dir, pattern='*.hgt')
        hash2 = cache.compute_source_hash(self.test_dir, pattern='*.hgt')

        self.assertEqual(hash1, hash2)

    def test_compute_source_hash_changes_with_file_count(self):
        """Test that hash changes when files are added."""
        Path(self.test_dir, "dem1.hgt").touch()

        cache = DEMCache(cache_dir=Path(self.cache_dir))
        hash1 = cache.compute_source_hash(self.test_dir, pattern='*.hgt')

        # Add another file
        Path(self.test_dir, "dem2.hgt").touch()
        hash2 = cache.compute_source_hash(self.test_dir, pattern='*.hgt')

        self.assertNotEqual(hash1, hash2)

    def test_compute_source_hash_raises_on_empty_directory(self):
        """Test that empty directory raises ValueError."""
        cache = DEMCache(cache_dir=Path(self.cache_dir))

        with self.assertRaises(ValueError):
            cache.compute_source_hash(self.test_dir, pattern='*.hgt')

    def test_compute_source_hash_raises_on_no_matches(self):
        """Test that no matching files raises ValueError."""
        Path(self.test_dir, "dem1.txt").touch()

        cache = DEMCache(cache_dir=Path(self.cache_dir))

        with self.assertRaises(ValueError):
            cache.compute_source_hash(self.test_dir, pattern='*.hgt')


class TestDEMCacheSaveLoad(unittest.TestCase):
    """Test DEM cache save and load operations."""

    def setUp(self):
        """Create temporary cache directory."""
        self.cache_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.cache_dir)

    def test_save_cache_creates_files(self):
        """Test that save_cache creates .npz and metadata files."""
        cache = DEMCache(cache_dir=Path(self.cache_dir), enabled=True)

        dem = np.array([[100, 110], [120, 130]], dtype=np.float32)
        transform = Affine(1, 0, 0, 0, -1, 100)
        source_hash = "test_hash_12345"

        cache_path, metadata_path = cache.save_cache(dem, transform, source_hash)

        self.assertIsNotNone(cache_path)
        self.assertIsNotNone(metadata_path)
        self.assertTrue(cache_path.exists())
        self.assertTrue(metadata_path.exists())

    def test_save_cache_disabled_returns_none(self):
        """Test that disabled cache returns None."""
        cache = DEMCache(cache_dir=Path(self.cache_dir), enabled=False)

        dem = np.array([[100, 110]], dtype=np.float32)
        transform = Affine(1, 0, 0, 0, -1, 100)
        source_hash = "test_hash"

        cache_path, metadata_path = cache.save_cache(dem, transform, source_hash)

        self.assertIsNone(cache_path)
        self.assertIsNone(metadata_path)

    def test_load_cache_returns_dem_and_transform(self):
        """Test that load_cache returns DEM array and transform."""
        cache = DEMCache(cache_dir=Path(self.cache_dir), enabled=True)

        dem_orig = np.array([[100, 110], [120, 130]], dtype=np.float32)
        transform_orig = Affine(1, 0, 0, 0, -1, 100)
        source_hash = "test_hash_load"

        # Save first
        cache.save_cache(dem_orig, transform_orig, source_hash)

        # Load
        result = cache.load_cache(source_hash)

        self.assertIsNotNone(result)
        dem_loaded, transform_loaded = result

        np.testing.assert_array_equal(dem_loaded, dem_orig)
        self.assertEqual(transform_loaded, transform_orig)

    def test_load_cache_returns_none_on_miss(self):
        """Test that load_cache returns None for missing cache."""
        cache = DEMCache(cache_dir=Path(self.cache_dir), enabled=True)

        result = cache.load_cache("nonexistent_hash")

        self.assertIsNone(result)

    def test_load_cache_disabled_returns_none(self):
        """Test that disabled cache always returns None."""
        cache = DEMCache(cache_dir=Path(self.cache_dir), enabled=False)

        result = cache.load_cache("any_hash")

        self.assertIsNone(result)


class TestDEMCacheClear(unittest.TestCase):
    """Test DEM cache clearing operations."""

    def setUp(self):
        """Create temporary cache directory."""
        self.cache_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.cache_dir)

    def test_clear_cache_removes_files(self):
        """Test that clear_cache removes cached files."""
        cache = DEMCache(cache_dir=Path(self.cache_dir), enabled=True)

        # Create some cached files
        dem = np.array([[100, 110]], dtype=np.float32)
        transform = Affine(1, 0, 0, 0, -1, 100)
        cache.save_cache(dem, transform, "hash1")
        cache.save_cache(dem, transform, "hash2")

        # Verify files exist
        cache_files_before = list(Path(self.cache_dir).glob("*.npz"))
        self.assertGreater(len(cache_files_before), 0)

        # Clear cache
        deleted = cache.clear_cache()

        self.assertGreater(deleted, 0)
        cache_files_after = list(Path(self.cache_dir).glob("*.npz"))
        self.assertEqual(len(cache_files_after), 0)

    def test_clear_cache_disabled_returns_zero(self):
        """Test that disabled cache clear returns 0."""
        cache = DEMCache(cache_dir=Path(self.cache_dir), enabled=False)

        deleted = cache.clear_cache()

        self.assertEqual(deleted, 0)


class TestDEMCacheStats(unittest.TestCase):
    """Test DEM cache statistics."""

    def setUp(self):
        """Create temporary cache directory."""
        self.cache_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.cache_dir)

    def test_get_cache_stats_returns_dict(self):
        """Test that get_cache_stats returns a dictionary."""
        cache = DEMCache(cache_dir=Path(self.cache_dir), enabled=True)

        stats = cache.get_cache_stats()

        self.assertIsInstance(stats, dict)
        self.assertIn('cache_dir', stats)
        self.assertIn('enabled', stats)
        self.assertIn('cache_files', stats)
        self.assertIn('total_size_mb', stats)

    def test_get_cache_stats_empty_cache(self):
        """Test stats for empty cache."""
        cache = DEMCache(cache_dir=Path(self.cache_dir), enabled=True)

        stats = cache.get_cache_stats()

        self.assertEqual(stats['cache_files'], 0)
        self.assertEqual(stats['total_size_mb'], 0)

    def test_get_cache_stats_with_files(self):
        """Test stats with cached files."""
        cache = DEMCache(cache_dir=Path(self.cache_dir), enabled=True)

        dem = np.array([[100, 110], [120, 130]], dtype=np.float32)
        transform = Affine(1, 0, 0, 0, -1, 100)
        cache.save_cache(dem, transform, "hash1")

        stats = cache.get_cache_stats()

        self.assertGreater(stats['cache_files'], 0)
        self.assertGreater(stats['total_size_mb'], 0)


class TestDEMCacheIntegration(unittest.TestCase):
    """Integration tests for DEM cache workflow."""

    def setUp(self):
        """Create temporary directories."""
        self.test_dir = tempfile.mkdtemp()
        self.cache_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir)
        shutil.rmtree(self.cache_dir)

    def test_full_workflow_load_save_load(self):
        """Test complete workflow: save and load DEM cache."""
        # Create test files
        Path(self.test_dir, "dem1.hgt").touch()
        Path(self.test_dir, "dem2.hgt").touch()

        cache = DEMCache(cache_dir=Path(self.cache_dir), enabled=True)

        # Compute hash
        source_hash = cache.compute_source_hash(self.test_dir, pattern='*.hgt')

        # Create and save DEM
        dem_orig = np.random.randn(100, 100).astype(np.float32)
        transform_orig = Affine(10, 0, 0, 0, -10, 1000)
        cache.save_cache(dem_orig, transform_orig, source_hash)

        # Load from cache
        result = cache.load_cache(source_hash)
        dem_loaded, transform_loaded = result

        np.testing.assert_array_almost_equal(dem_loaded, dem_orig)
        self.assertEqual(transform_loaded, transform_orig)

    def test_invalidation_on_file_modification(self):
        """Test that hash changes when source files are modified."""
        Path(self.test_dir, "dem.hgt").touch()

        cache = DEMCache(cache_dir=Path(self.cache_dir))
        hash1 = cache.compute_source_hash(self.test_dir, pattern='*.hgt')

        # Simulate file modification by updating mtime
        import time
        time.sleep(0.1)
        Path(self.test_dir, "dem.hgt").touch()

        hash2 = cache.compute_source_hash(self.test_dir, pattern='*.hgt')

        self.assertNotEqual(hash1, hash2)


class TestDEMCacheEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def setUp(self):
        """Create temporary directories."""
        self.test_dir = tempfile.mkdtemp()
        self.cache_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir)
        shutil.rmtree(self.cache_dir)

    def test_save_load_with_nan_values(self):
        """Test that DEM with NaN values preserves them correctly."""
        cache = DEMCache(cache_dir=Path(self.cache_dir), enabled=True)

        dem_orig = np.array([[100, np.nan], [np.nan, 130]], dtype=np.float32)
        transform = Affine(1, 0, 0, 0, -1, 100)
        source_hash = "hash_with_nan"

        cache.save_cache(dem_orig, transform, source_hash)
        result = cache.load_cache(source_hash)

        dem_loaded, _ = result
        # Check that NaN locations match
        np.testing.assert_array_equal(np.isnan(dem_loaded), np.isnan(dem_orig))
        # Check that non-NaN values match
        np.testing.assert_array_equal(dem_loaded[~np.isnan(dem_loaded)],
                                      dem_orig[~np.isnan(dem_orig)])

    def test_save_load_large_dem(self):
        """Test caching of larger DEM array."""
        cache = DEMCache(cache_dir=Path(self.cache_dir), enabled=True)

        # Create large DEM (simulating real data)
        dem_orig = np.random.randn(1000, 1000).astype(np.float32) * 1000 + 500
        transform = Affine(30, 0, 0, 0, -30, 10000)
        source_hash = "hash_large"

        cache.save_cache(dem_orig, transform, source_hash)
        result = cache.load_cache(source_hash)

        dem_loaded, transform_loaded = result
        np.testing.assert_array_almost_equal(dem_loaded, dem_orig, decimal=4)
        self.assertEqual(transform_loaded, transform)

    def test_cache_stats_file_list(self):
        """Test that cache stats include file list."""
        cache = DEMCache(cache_dir=Path(self.cache_dir), enabled=True)

        dem = np.array([[100, 110]], dtype=np.float32)
        transform = Affine(1, 0, 0, 0, -1, 100)
        cache.save_cache(dem, transform, "hash1")

        stats = cache.get_cache_stats()

        self.assertIn('files', stats)
        # We save both .npz and .json files
        self.assertEqual(len(stats['files']), 2)
        for file_info in stats['files']:
            self.assertIn('name', file_info)
            self.assertIn('size_mb', file_info)


if __name__ == '__main__':
    unittest.main()
