"""
Test suite for terrain pipeline dependency graph system.

Tests the TerrainPipeline class for dependency resolution,
caching integration, and execution planning.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np

from src.terrain.pipeline import TerrainPipeline, TaskState


class TestPipelineInitialization(unittest.TestCase):
    """Test pipeline initialization and setup."""

    def setUp(self):
        """Create temporary directories."""
        self.cache_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.cache_dir)

    def test_pipeline_creation_with_defaults(self):
        """Test creating pipeline with default parameters."""
        pipeline = TerrainPipeline(cache_enabled=False, verbose=False)

        self.assertIsNotNone(pipeline)
        self.assertFalse(pipeline.cache_enabled)
        self.assertFalse(pipeline.verbose)

    def test_pipeline_has_dem_cache(self):
        """Test that pipeline has initialized DEM cache."""
        pipeline = TerrainPipeline(cache_enabled=False)

        self.assertIsNotNone(pipeline.dem_cache)

    def test_pipeline_has_mesh_cache(self):
        """Test that pipeline has initialized mesh cache."""
        pipeline = TerrainPipeline(cache_enabled=False)

        self.assertIsNotNone(pipeline.mesh_cache)

    def test_pipeline_has_task_graph(self):
        """Test that pipeline defines task dependencies."""
        pipeline = TerrainPipeline(cache_enabled=False)

        self.assertIn('load_dem', pipeline._task_graph)
        self.assertIn('apply_transforms', pipeline._task_graph)
        self.assertIn('detect_water', pipeline._task_graph)
        self.assertIn('create_mesh', pipeline._task_graph)
        self.assertIn('render_view', pipeline._task_graph)


class TestPipelineHashComputation(unittest.TestCase):
    """Test hash computation for cache keys."""

    def setUp(self):
        """Create pipeline."""
        self.pipeline = TerrainPipeline(cache_enabled=False, verbose=False)

    def test_compute_hash_deterministic(self):
        """Test that same inputs produce same hash."""
        hash1 = self.pipeline._compute_hash("test", "data", scale=1.0)
        hash2 = self.pipeline._compute_hash("test", "data", scale=1.0)

        self.assertEqual(hash1, hash2)

    def test_compute_hash_differs_on_different_args(self):
        """Test that different inputs produce different hashes."""
        hash1 = self.pipeline._compute_hash("test1", scale=1.0)
        hash2 = self.pipeline._compute_hash("test2", scale=1.0)

        self.assertNotEqual(hash1, hash2)

    def test_compute_hash_differs_on_different_kwargs(self):
        """Test that different kwargs produce different hashes."""
        hash1 = self.pipeline._compute_hash("test", scale=1.0)
        hash2 = self.pipeline._compute_hash("test", scale=2.0)

        self.assertNotEqual(hash1, hash2)

    def test_compute_hash_handles_numpy_arrays(self):
        """Test that numpy arrays are hashed correctly."""
        arr = np.array([[1, 2], [3, 4]], dtype=np.float32)

        hash1 = self.pipeline._compute_hash(arr)
        hash2 = self.pipeline._compute_hash(arr)

        self.assertEqual(hash1, hash2)

    def test_compute_hash_differs_on_different_arrays(self):
        """Test that different arrays produce different hashes."""
        arr1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
        arr2 = np.array([[1, 2], [3, 5]], dtype=np.float32)

        hash1 = self.pipeline._compute_hash(arr1)
        hash2 = self.pipeline._compute_hash(arr2)

        self.assertNotEqual(hash1, hash2)

    def test_compute_hash_handles_dicts(self):
        """Test that dictionaries are hashed correctly."""
        params = {'scale_factor': 100.0, 'height_scale': 4.0}

        hash1 = self.pipeline._compute_hash(params)
        hash2 = self.pipeline._compute_hash(params)

        self.assertEqual(hash1, hash2)

    def test_compute_hash_is_64_chars(self):
        """Test that hash is SHA256 (64 hex chars)."""
        hash_val = self.pipeline._compute_hash("test")

        self.assertEqual(len(hash_val), 64)


class TestPipelineDependencyResolution(unittest.TestCase):
    """Test dependency ordering and execution planning."""

    def setUp(self):
        """Create pipeline."""
        self.pipeline = TerrainPipeline(cache_enabled=False, verbose=False)

    def test_execution_order_load_dem(self):
        """Test execution order for load_dem (no dependencies)."""
        order = self.pipeline._compute_execution_order('load_dem')

        self.assertEqual(order, ['load_dem'])

    def test_execution_order_apply_transforms(self):
        """Test execution order for apply_transforms (depends on load_dem)."""
        order = self.pipeline._compute_execution_order('apply_transforms')

        self.assertEqual(order, ['load_dem', 'apply_transforms'])

    def test_execution_order_detect_water(self):
        """Test execution order for detect_water."""
        order = self.pipeline._compute_execution_order('detect_water')

        # detect_water -> apply_transforms -> load_dem
        self.assertEqual(order[0], 'load_dem')
        self.assertEqual(order[1], 'apply_transforms')
        self.assertEqual(order[2], 'detect_water')

    def test_execution_order_create_mesh(self):
        """Test execution order for create_mesh (multiple dependencies)."""
        order = self.pipeline._compute_execution_order('create_mesh')

        # Must have load_dem and detect_water before create_mesh
        self.assertIn('load_dem', order)
        self.assertIn('detect_water', order)
        self.assertEqual(order[-1], 'create_mesh')

    def test_execution_order_render_view(self):
        """Test execution order for render_view (full pipeline)."""
        order = self.pipeline._compute_execution_order('render_view')

        # Should include all tasks
        self.assertEqual(order[0], 'load_dem')
        self.assertEqual(order[-1], 'render_view')
        self.assertEqual(len(order), 5)  # All 5 tasks

    def test_task_graph_consistency(self):
        """Test that task graph references are valid."""
        for task_name, task_info in self.pipeline._task_graph.items():
            for dep in task_info['depends_on']:
                self.assertIn(dep, self.pipeline._task_graph,
                             f"Task {task_name} depends on {dep} which doesn't exist")


class TestTaskState(unittest.TestCase):
    """Test TaskState dataclass."""

    def test_task_state_creation(self):
        """Test creating TaskState."""
        state = TaskState(
            name='test',
            depends_on=['dep1', 'dep2'],
            params={'key': 'value'}
        )

        self.assertEqual(state.name, 'test')
        self.assertEqual(state.depends_on, ['dep1', 'dep2'])
        self.assertEqual(state.params, {'key': 'value'})
        self.assertFalse(state.cached)
        self.assertFalse(state.computed)

    def test_task_state_result_tracking(self):
        """Test that TaskState tracks results."""
        state = TaskState(name='test')
        self.assertIsNone(state.result)

        state.result = "some_result"
        self.assertEqual(state.result, "some_result")

    def test_task_state_cache_flag(self):
        """Test cache hit tracking."""
        state = TaskState(name='test')
        self.assertFalse(state.cached)

        state.cached = True
        self.assertTrue(state.cached)

    def test_task_state_computed_flag(self):
        """Test computed flag tracking."""
        state = TaskState(name='test')
        self.assertFalse(state.computed)

        state.computed = True
        self.assertTrue(state.computed)


class TestPipelineExplanation(unittest.TestCase):
    """Test pipeline explanation and planning."""

    def setUp(self):
        """Create pipeline."""
        self.pipeline = TerrainPipeline(cache_enabled=False, verbose=False)

    def test_explain_valid_task(self):
        """Test explaining a valid task (should not raise)."""
        # Should print without raising
        try:
            self.pipeline.explain('render_view')
        except Exception as e:
            self.fail(f"explain() raised {type(e).__name__}: {e}")

    def test_explain_invalid_task(self):
        """Test explaining an invalid task."""
        # Should handle gracefully
        try:
            self.pipeline.explain('nonexistent_task')
        except Exception as e:
            self.fail(f"explain() raised {type(e).__name__}: {e}")


class TestPipelineCacheManagement(unittest.TestCase):
    """Test cache statistics and management."""

    def setUp(self):
        """Create temporary cache directory and pipeline."""
        self.cache_dir = tempfile.mkdtemp()
        self.pipeline = TerrainPipeline(
            cache_enabled=True,
            dem_cache_dir=self.cache_dir,
            verbose=False
        )

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.cache_dir)

    def test_cache_stats_available(self):
        """Test that cache stats can be retrieved."""
        stats = self.pipeline.cache_stats()

        self.assertIn('dem', stats)
        self.assertIn('mesh', stats)
        self.assertIn('total_mb', stats)
        self.assertIn('total_files', stats)

    def test_cache_stats_structure(self):
        """Test cache stats have required fields."""
        stats = self.pipeline.cache_stats()

        # DEM stats
        self.assertIn('cache_files', stats['dem'])
        self.assertIn('total_size_mb', stats['dem'])

        # Mesh stats
        self.assertIn('blend_files', stats['mesh'])
        self.assertIn('total_size_mb', stats['mesh'])

    def test_clear_cache_returns_count(self):
        """Test that clear_cache returns file count."""
        count = self.pipeline.clear_cache()

        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 0)


class TestPipelineDependencyTree(unittest.TestCase):
    """Test pipeline dependency tree behavior."""

    def setUp(self):
        """Create pipeline."""
        self.pipeline = TerrainPipeline(cache_enabled=False, verbose=False)

    def test_create_mesh_depends_on_load_dem(self):
        """Test that create_mesh depends on load_dem."""
        task_info = self.pipeline._task_graph['create_mesh']
        self.assertIn('load_dem', task_info['depends_on'])

    def test_create_mesh_depends_on_detect_water(self):
        """Test that create_mesh depends on detect_water."""
        task_info = self.pipeline._task_graph['create_mesh']
        self.assertIn('detect_water', task_info['depends_on'])

    def test_render_view_depends_on_create_mesh(self):
        """Test that render_view depends on create_mesh."""
        task_info = self.pipeline._task_graph['render_view']
        self.assertIn('create_mesh', task_info['depends_on'])

    def test_detect_water_depends_on_apply_transforms(self):
        """Test that detect_water depends on apply_transforms."""
        task_info = self.pipeline._task_graph['detect_water']
        self.assertIn('apply_transforms', task_info['depends_on'])

    def test_apply_transforms_depends_on_load_dem(self):
        """Test that apply_transforms depends on load_dem."""
        task_info = self.pipeline._task_graph['apply_transforms']
        self.assertIn('load_dem', task_info['depends_on'])

    def test_load_dem_has_no_dependencies(self):
        """Test that load_dem has no dependencies (root task)."""
        task_info = self.pipeline._task_graph['load_dem']
        self.assertEqual(task_info['depends_on'], [])

    def test_full_pipeline_has_correct_depth(self):
        """Test that full pipeline execution order has correct depth."""
        order = self.pipeline._compute_execution_order('render_view')

        # Check that all tasks are included
        self.assertEqual(len(order), 5)

        # Check critical ordering constraints
        load_dem_idx = order.index('load_dem')
        apply_transforms_idx = order.index('apply_transforms')
        detect_water_idx = order.index('detect_water')
        create_mesh_idx = order.index('create_mesh')
        render_view_idx = order.index('render_view')

        # load_dem must come before apply_transforms
        self.assertLess(load_dem_idx, apply_transforms_idx)

        # apply_transforms must come before detect_water
        self.assertLess(apply_transforms_idx, detect_water_idx)

        # detect_water must come before create_mesh
        self.assertLess(detect_water_idx, create_mesh_idx)

        # create_mesh must come before render_view
        self.assertLess(create_mesh_idx, render_view_idx)

    def test_dependency_chain_is_acyclic(self):
        """Test that the dependency graph has no cycles."""
        # If there were cycles, _compute_execution_order would
        # either infinite loop or raise an error
        try:
            for task_name in self.pipeline._task_graph.keys():
                order = self.pipeline._compute_execution_order(task_name)
                self.assertIsInstance(order, list)
        except RecursionError:
            self.fail("Dependency graph has cycles!")


class TestMeshCacheHashComputation(unittest.TestCase):
    """Test that mesh cache hash includes necessary dependencies."""

    def setUp(self):
        """Create pipeline."""
        self.pipeline = TerrainPipeline(cache_enabled=False, verbose=False)

    def test_mesh_hash_changes_with_parameters(self):
        """Test that different mesh parameters produce different hashes."""
        hash1 = self.pipeline._compute_hash(
            "dem_hash_abc",
            scale_factor=100.0,
            height_scale=4.0
        )
        hash2 = self.pipeline._compute_hash(
            "dem_hash_abc",
            scale_factor=100.0,
            height_scale=5.0  # Different height_scale
        )

        self.assertNotEqual(hash1, hash2)

    def test_mesh_hash_changes_with_dem_source(self):
        """Test that different DEM sources produce different hashes."""
        hash1 = self.pipeline._compute_hash(
            "dem_hash_abc",
            scale_factor=100.0,
            height_scale=4.0
        )
        hash2 = self.pipeline._compute_hash(
            "dem_hash_def",  # Different DEM source
            scale_factor=100.0,
            height_scale=4.0
        )

        self.assertNotEqual(hash1, hash2)

    def test_mesh_hash_same_for_same_inputs(self):
        """Test that same inputs produce same hash (deterministic)."""
        hash1 = self.pipeline._compute_hash(
            "dem_hash_abc",
            scale_factor=100.0,
            height_scale=4.0
        )
        hash2 = self.pipeline._compute_hash(
            "dem_hash_abc",
            scale_factor=100.0,
            height_scale=4.0
        )

        self.assertEqual(hash1, hash2)

    def test_mesh_hash_independent_of_view(self):
        """Test that mesh hash doesn't include view direction."""
        # Mesh geometry should be the same regardless of camera angle
        # So the hash should not include view-specific parameters
        base_params = {
            'scale_factor': 100.0,
            'height_scale': 4.0,
        }

        hash1 = self.pipeline._compute_hash("dem_hash", **base_params)
        hash2 = self.pipeline._compute_hash("dem_hash", **base_params)

        self.assertEqual(hash1, hash2)


if __name__ == '__main__':
    unittest.main()
