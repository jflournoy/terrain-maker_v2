"""
Test suite for mesh caching module.

Tests the MeshCache class for hash computation, save/load operations,
and cache management.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from src.terrain.mesh_cache import MeshCache


class TestMeshCacheHashComputation(unittest.TestCase):
    """Test mesh parameter hash computation."""

    def setUp(self):
        """Create temporary cache directory."""
        self.cache_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.cache_dir)

    def test_compute_mesh_hash_returns_string(self):
        """Test that compute_mesh_hash returns a valid hash string."""
        cache = MeshCache(cache_dir=Path(self.cache_dir))

        mesh_params = {
            'scale_factor': 100.0,
            'height_scale': 4.0,
            'center_model': True,
            'boundary_extension': True
        }
        dem_hash = "dem_hash_value"

        hash_val = cache.compute_mesh_hash(dem_hash, mesh_params)

        self.assertIsInstance(hash_val, str)
        self.assertEqual(len(hash_val), 64)  # SHA256

    def test_compute_mesh_hash_is_deterministic(self):
        """Test that same parameters produce same hash."""
        cache = MeshCache(cache_dir=Path(self.cache_dir))

        mesh_params = {
            'scale_factor': 100.0,
            'height_scale': 4.0,
            'center_model': True,
            'boundary_extension': True
        }
        dem_hash = "dem_hash_123"

        hash1 = cache.compute_mesh_hash(dem_hash, mesh_params)
        hash2 = cache.compute_mesh_hash(dem_hash, mesh_params)

        self.assertEqual(hash1, hash2)

    def test_compute_mesh_hash_differs_with_parameters(self):
        """Test that different parameters produce different hash."""
        cache = MeshCache(cache_dir=Path(self.cache_dir))

        dem_hash = "dem_hash_123"

        mesh_params1 = {
            'scale_factor': 100.0,
            'height_scale': 4.0,
            'center_model': True,
            'boundary_extension': True
        }

        mesh_params2 = {
            'scale_factor': 100.0,
            'height_scale': 5.0,  # Different
            'center_model': True,
            'boundary_extension': True
        }

        hash1 = cache.compute_mesh_hash(dem_hash, mesh_params1)
        hash2 = cache.compute_mesh_hash(dem_hash, mesh_params2)

        self.assertNotEqual(hash1, hash2)

    def test_compute_mesh_hash_differs_with_dem_hash(self):
        """Test that different DEM hashes produce different mesh hashes."""
        cache = MeshCache(cache_dir=Path(self.cache_dir))

        mesh_params = {
            'scale_factor': 100.0,
            'height_scale': 4.0,
            'center_model': True,
            'boundary_extension': True
        }

        hash1 = cache.compute_mesh_hash("dem_hash_1", mesh_params)
        hash2 = cache.compute_mesh_hash("dem_hash_2", mesh_params)

        self.assertNotEqual(hash1, hash2)

    def test_compute_mesh_hash_handles_array_parameters(self):
        """Test that array parameters are handled correctly."""
        cache = MeshCache(cache_dir=Path(self.cache_dir))

        mesh_params = {
            'scale_factor': 100.0,
            'height_scale': 4.0,
            'water_mask': np.array([True, False, True])
        }

        # Should not raise an error
        hash_val = cache.compute_mesh_hash("dem_hash", mesh_params)

        self.assertIsInstance(hash_val, str)


class TestMeshCachePaths(unittest.TestCase):
    """Test cache path generation."""

    def setUp(self):
        """Create temporary cache directory."""
        self.cache_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.cache_dir)

    def test_get_cache_path_returns_path(self):
        """Test that get_cache_path returns a valid Path."""
        cache = MeshCache(cache_dir=Path(self.cache_dir))

        path = cache.get_cache_path("test_hash")

        self.assertIsInstance(path, Path)
        self.assertTrue(str(path).endswith('.blend'))

    def test_get_cache_path_contains_hash(self):
        """Test that cache path contains the hash."""
        cache = MeshCache(cache_dir=Path(self.cache_dir))
        test_hash = "abc123def456"

        path = cache.get_cache_path(test_hash)

        self.assertIn(test_hash, str(path))

    def test_get_metadata_path_returns_path(self):
        """Test that get_metadata_path returns a valid Path."""
        cache = MeshCache(cache_dir=Path(self.cache_dir))

        path = cache.get_metadata_path("test_hash")

        self.assertIsInstance(path, Path)
        self.assertTrue(str(path).endswith('.json'))

    def test_get_metadata_path_contains_hash(self):
        """Test that metadata path contains the hash."""
        cache = MeshCache(cache_dir=Path(self.cache_dir))
        test_hash = "xyz789abc123"

        path = cache.get_metadata_path(test_hash)

        self.assertIn(test_hash, str(path))


class TestMeshCacheSaveLoad(unittest.TestCase):
    """Test mesh cache save and load operations."""

    def setUp(self):
        """Create temporary cache and blend directories."""
        self.cache_dir = tempfile.mkdtemp()
        self.blend_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.cache_dir)
        shutil.rmtree(self.blend_dir)

    def test_save_cache_with_existing_blend_file(self):
        """Test save_cache with an existing blend file."""
        # Create a mock blend file
        blend_file = Path(self.blend_dir) / "test.blend"
        blend_file.write_text("mock blend content")

        cache = MeshCache(cache_dir=Path(self.cache_dir), enabled=True)

        mesh_params = {
            'scale_factor': 100.0,
            'height_scale': 4.0,
            'center_model': True,
            'boundary_extension': True
        }
        mesh_hash = "mesh_hash_123"

        cache_path, metadata_path = cache.save_cache(
            blend_file,
            mesh_hash,
            mesh_params
        )

        self.assertIsNotNone(cache_path)
        self.assertIsNotNone(metadata_path)
        self.assertTrue(cache_path.exists())
        self.assertTrue(metadata_path.exists())

    def test_save_cache_disabled_returns_none(self):
        """Test that disabled cache returns None."""
        blend_file = Path(self.blend_dir) / "test.blend"
        blend_file.write_text("mock")

        cache = MeshCache(cache_dir=Path(self.cache_dir), enabled=False)

        mesh_params = {'scale_factor': 100.0}
        cache_path, metadata_path = cache.save_cache(
            blend_file,
            "hash",
            mesh_params
        )

        self.assertIsNone(cache_path)
        self.assertIsNone(metadata_path)

    def test_load_cache_returns_path(self):
        """Test that load_cache returns cached blend file path."""
        # Create and save a blend file
        blend_file = Path(self.blend_dir) / "test.blend"
        blend_file.write_text("mock blend content")

        cache = MeshCache(cache_dir=Path(self.cache_dir), enabled=True)
        mesh_params = {'scale_factor': 100.0}
        mesh_hash = "mesh_hash_load"

        cache.save_cache(blend_file, mesh_hash, mesh_params)

        # Load from cache
        result = cache.load_cache(mesh_hash)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, Path)
        self.assertTrue(result.exists())

    def test_load_cache_returns_none_on_miss(self):
        """Test that load_cache returns None for missing cache."""
        cache = MeshCache(cache_dir=Path(self.cache_dir), enabled=True)

        result = cache.load_cache("nonexistent_hash")

        self.assertIsNone(result)

    def test_load_cache_disabled_returns_none(self):
        """Test that disabled cache always returns None."""
        cache = MeshCache(cache_dir=Path(self.cache_dir), enabled=False)

        result = cache.load_cache("any_hash")

        self.assertIsNone(result)


class TestMeshCacheClear(unittest.TestCase):
    """Test mesh cache clearing operations."""

    def setUp(self):
        """Create temporary cache and blend directories."""
        self.cache_dir = tempfile.mkdtemp()
        self.blend_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.cache_dir)
        shutil.rmtree(self.blend_dir)

    def test_clear_cache_removes_blend_files(self):
        """Test that clear_cache removes cached blend files."""
        cache = MeshCache(cache_dir=Path(self.cache_dir), enabled=True)

        # Create some cached blend files
        blend_file1 = Path(self.blend_dir) / "blend1.blend"
        blend_file2 = Path(self.blend_dir) / "blend2.blend"
        blend_file1.write_text("mock")
        blend_file2.write_text("mock")

        mesh_params = {'scale_factor': 100.0}
        cache.save_cache(blend_file1, "hash1", mesh_params)
        cache.save_cache(blend_file2, "hash2", mesh_params)

        # Verify files exist
        blend_files_before = list(Path(self.cache_dir).glob("*.blend"))
        self.assertGreater(len(blend_files_before), 0)

        # Clear cache
        deleted = cache.clear_cache()

        self.assertGreater(deleted, 0)
        blend_files_after = list(Path(self.cache_dir).glob("*.blend"))
        self.assertEqual(len(blend_files_after), 0)

    def test_clear_cache_disabled_returns_zero(self):
        """Test that disabled cache clear returns 0."""
        cache = MeshCache(cache_dir=Path(self.cache_dir), enabled=False)

        deleted = cache.clear_cache()

        self.assertEqual(deleted, 0)


class TestMeshCacheStats(unittest.TestCase):
    """Test mesh cache statistics."""

    def setUp(self):
        """Create temporary cache and blend directories."""
        self.cache_dir = tempfile.mkdtemp()
        self.blend_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.cache_dir)
        shutil.rmtree(self.blend_dir)

    def test_get_cache_stats_returns_dict(self):
        """Test that get_cache_stats returns a dictionary."""
        cache = MeshCache(cache_dir=Path(self.cache_dir), enabled=True)

        stats = cache.get_cache_stats()

        self.assertIsInstance(stats, dict)
        self.assertIn('cache_dir', stats)
        self.assertIn('enabled', stats)
        self.assertIn('blend_files', stats)
        self.assertIn('total_size_mb', stats)

    def test_get_cache_stats_empty_cache(self):
        """Test stats for empty cache."""
        cache = MeshCache(cache_dir=Path(self.cache_dir), enabled=True)

        stats = cache.get_cache_stats()

        self.assertEqual(stats['blend_files'], 0)
        self.assertEqual(stats['total_size_mb'], 0)

    def test_get_cache_stats_with_files(self):
        """Test stats with cached blend files."""
        cache = MeshCache(cache_dir=Path(self.cache_dir), enabled=True)

        # Create and save a blend file
        blend_file = Path(self.blend_dir) / "test.blend"
        blend_file.write_text("mock blend content with some data")

        mesh_params = {'scale_factor': 100.0}
        cache.save_cache(blend_file, "hash1", mesh_params)

        stats = cache.get_cache_stats()

        self.assertGreater(stats['blend_files'], 0)
        self.assertGreater(stats['total_size_mb'], 0)


class TestMeshCacheIntegration(unittest.TestCase):
    """Integration tests for mesh cache workflow."""

    def setUp(self):
        """Create temporary directories."""
        self.cache_dir = tempfile.mkdtemp()
        self.blend_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.cache_dir)
        shutil.rmtree(self.blend_dir)

    def test_full_workflow_save_load_clear(self):
        """Test complete workflow: save, load, and clear mesh cache."""
        cache = MeshCache(cache_dir=Path(self.cache_dir), enabled=True)

        # Create blend file
        blend_file = Path(self.blend_dir) / "test.blend"
        blend_file.write_text("mock blend content")

        # Compute hash and save
        mesh_params = {
            'scale_factor': 100.0,
            'height_scale': 4.0,
            'center_model': True
        }
        dem_hash = "dem_hash_workflow"
        mesh_hash = cache.compute_mesh_hash(dem_hash, mesh_params)

        cache.save_cache(blend_file, mesh_hash, mesh_params)

        # Load from cache
        cached_path = cache.load_cache(mesh_hash)
        self.assertIsNotNone(cached_path)

        # Clear cache
        deleted = cache.clear_cache()
        self.assertGreater(deleted, 0)

        # Verify it's gone
        result = cache.load_cache(mesh_hash)
        self.assertIsNone(result)

    def test_multiple_mesh_variants_different_hashes(self):
        """Test that different mesh parameters create different cache entries."""
        cache = MeshCache(cache_dir=Path(self.cache_dir), enabled=True)

        dem_hash = "dem_hash"

        # Create two different mesh variants
        params1 = {'scale_factor': 100.0, 'height_scale': 4.0}
        params2 = {'scale_factor': 100.0, 'height_scale': 5.0}

        hash1 = cache.compute_mesh_hash(dem_hash, params1)
        hash2 = cache.compute_mesh_hash(dem_hash, params2)

        # Save both
        blend_file = Path(self.blend_dir) / "test.blend"
        blend_file.write_text("mock content")

        cache.save_cache(blend_file, hash1, params1)
        cache.save_cache(blend_file, hash2, params2)

        # Both should be loadable
        self.assertIsNotNone(cache.load_cache(hash1))
        self.assertIsNotNone(cache.load_cache(hash2))


if __name__ == '__main__':
    unittest.main()
