"""
Tests for snow analysis pipeline with dependency graph caching.

This tests the integration of the caching system into snow analysis,
allowing users to cache intermediate computations in sequential pipelines.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.snow.analysis import SnowAnalysisPipeline


class TestSnowAnalysisPipelineBasics:
    """Test basic pipeline creation and configuration."""

    def test_pipeline_creation(self):
        """Test that a snow analysis pipeline can be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            pipeline = SnowAnalysisPipeline(cache_dir=cache_dir)

            assert pipeline is not None
            assert pipeline.cache_dir == cache_dir

    def test_pipeline_with_caching_disabled(self):
        """Test pipeline can be created with caching disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            pipeline = SnowAnalysisPipeline(cache_dir=cache_dir, caching_enabled=False)

            assert pipeline.caching_enabled is False

    def test_pipeline_has_tasks(self):
        """Test that pipeline defines required tasks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            pipeline = SnowAnalysisPipeline(cache_dir=cache_dir)

            # Should have tasks for snow analysis pipeline
            expected_tasks = {"load_dem", "compute_snow_stats", "calculate_scores"}
            assert hasattr(pipeline, "tasks")
            assert set(pipeline.tasks.keys()).issuperset(expected_tasks)


class TestSnowAnalysisPipelineDependencies:
    """Test dependency graph and execution order."""

    def test_pipeline_defines_dependencies(self):
        """Test that tasks have defined dependencies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            pipeline = SnowAnalysisPipeline(cache_dir=cache_dir)

            # compute_snow_stats depends on load_dem
            assert pipeline.tasks["compute_snow_stats"]["dependencies"] == ["load_dem"]

            # calculate_scores depends on both previous tasks
            score_deps = pipeline.tasks["calculate_scores"]["dependencies"]
            assert "compute_snow_stats" in score_deps

    def test_pipeline_explain_shows_dependencies(self):
        """Test that pipeline can explain task dependencies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            pipeline = SnowAnalysisPipeline(cache_dir=cache_dir)

            explanation = pipeline.explain("calculate_scores")
            assert isinstance(explanation, str)
            assert "depends on" in explanation.lower()


class TestSnowAnalysisPipelineCaching:
    """Test caching behavior and cache invalidation."""

    def test_cache_hit_returns_cached_data(self):
        """Test that running pipeline twice with same inputs uses cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            pipeline = SnowAnalysisPipeline(cache_dir=cache_dir)

            # Mock inputs
            dem = np.random.rand(10, 10)
            extent = (0, 0, 10, 10)

            # First run
            result1 = pipeline.execute(
                task="compute_snow_stats",
                dem=dem,
                extent=extent,
                snodas_dir="/fake/path"
            )

            # Second run with same inputs
            result2 = pipeline.execute(
                task="compute_snow_stats",
                dem=dem,
                extent=extent,
                snodas_dir="/fake/path"
            )

            # Results should be identical (from cache)
            assert result1 is not None
            assert result2 is not None
            # Both should come from cache (execution count should be 1, not 2)
            assert pipeline._execution_count("compute_snow_stats") == 1

    def test_cache_miss_triggers_recomputation(self):
        """Test that changing inputs invalidates cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            pipeline = SnowAnalysisPipeline(cache_dir=cache_dir)

            dem1 = np.random.rand(10, 10)
            dem2 = np.random.rand(10, 10)  # Different data

            # First run
            pipeline.execute(
                task="compute_snow_stats",
                dem=dem1,
                extent=(0, 0, 10, 10),
                snodas_dir="/fake/path"
            )
            count_after_first = pipeline._execution_count("compute_snow_stats")

            # Second run with different DEM
            pipeline.execute(
                task="compute_snow_stats",
                dem=dem2,
                extent=(0, 0, 10, 10),
                snodas_dir="/fake/path"
            )
            count_after_second = pipeline._execution_count("compute_snow_stats")

            # Should have executed twice due to different input
            assert count_after_second > count_after_first

    def test_cache_directory_is_created(self):
        """Test that cache directory is created automatically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache" / "nested" / "path"
            pipeline = SnowAnalysisPipeline(cache_dir=cache_dir)

            # Cache dir should be created
            assert cache_dir.exists()

    def test_cache_metadata_stored(self):
        """Test that cache metadata is stored for debugging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            pipeline = SnowAnalysisPipeline(cache_dir=cache_dir)

            # Execute a task
            dem = np.random.rand(10, 10)
            pipeline.execute(
                task="compute_snow_stats",
                dem=dem,
                extent=(0, 0, 10, 10),
                snodas_dir="/fake/path"
            )

            # Check that metadata exists
            metadata_files = list(cache_dir.glob("*_meta.json"))
            assert len(metadata_files) > 0


class TestSnowAnalysisPipelineExecution:
    """Test pipeline execution and result handling."""

    def test_execute_single_task(self):
        """Test executing a single task in pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            pipeline = SnowAnalysisPipeline(cache_dir=cache_dir)

            dem = np.random.rand(10, 10)
            result = pipeline.execute(
                task="compute_snow_stats",
                dem=dem,
                extent=(0, 0, 10, 10),
                snodas_dir="/fake/path"
            )

            assert result is not None

    def test_execute_full_pipeline(self):
        """Test executing the full pipeline from DEM to scores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            pipeline = SnowAnalysisPipeline(cache_dir=cache_dir)

            dem = np.random.rand(10, 10)

            # Execute full pipeline
            result = pipeline.execute_all(
                dem=dem,
                extent=(0, 0, 10, 10),
                snodas_dir="/fake/path"
            )

            assert result is not None
            assert "sledding_score" in result or "scores" in result.keys()

    def test_execute_with_force_rebuild(self):
        """Test force_rebuild flag bypasses cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            pipeline = SnowAnalysisPipeline(cache_dir=cache_dir)

            dem = np.random.rand(10, 10)

            # First execution
            pipeline.execute(
                task="compute_snow_stats",
                dem=dem,
                extent=(0, 0, 10, 10),
                snodas_dir="/fake/path"
            )
            count_first = pipeline._execution_count("compute_snow_stats")

            # Second execution with force_rebuild
            pipeline.execute(
                task="compute_snow_stats",
                dem=dem,
                extent=(0, 0, 10, 10),
                snodas_dir="/fake/path",
                force_rebuild=True
            )
            count_second = pipeline._execution_count("compute_snow_stats")

            # Should have executed twice despite same inputs
            assert count_second > count_first


class TestSnowAnalysisPipelineInterface:
    """Test user-friendly interface for the pipeline."""

    def test_pipeline_is_easy_to_use(self):
        """Test that pipeline provides simple interface for users."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"

            # Simple one-liner should work
            pipeline = SnowAnalysisPipeline(cache_dir=cache_dir)
            assert hasattr(pipeline, "execute")
            assert hasattr(pipeline, "execute_all")
            assert callable(pipeline.execute)
            assert callable(pipeline.execute_all)

    def test_pipeline_defaults_to_reasonable_cache_location(self):
        """Test that pipeline can use default cache location."""
        # Should work without specifying cache_dir
        pipeline = SnowAnalysisPipeline()
        assert pipeline.cache_dir is not None

    def test_clear_cache_method(self):
        """Test that users can clear cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            pipeline = SnowAnalysisPipeline(cache_dir=cache_dir)

            # Should have clear_cache method
            assert hasattr(pipeline, "clear_cache")
            assert callable(pipeline.clear_cache)

    def test_get_cache_stats(self):
        """Test that users can query cache statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            pipeline = SnowAnalysisPipeline(cache_dir=cache_dir)

            # Should have cache stats method
            assert hasattr(pipeline, "get_cache_stats")

            # Stats should be a dict
            stats = pipeline.get_cache_stats()
            assert isinstance(stats, dict)


class TestSnowAnalysisPipelineHashComputation:
    """Test that hash-based cache invalidation works correctly."""

    def test_same_inputs_produce_same_hash(self):
        """Test that identical inputs produce identical hashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            pipeline = SnowAnalysisPipeline(cache_dir=cache_dir)

            dem = np.array([[1, 2], [3, 4]])
            extent = (0, 0, 10, 10)

            hash1 = pipeline._compute_task_hash(
                "compute_snow_stats",
                dem=dem,
                extent=extent,
                snodas_dir="/path/a"
            )
            hash2 = pipeline._compute_task_hash(
                "compute_snow_stats",
                dem=dem,
                extent=extent,
                snodas_dir="/path/a"
            )

            assert hash1 == hash2

    def test_different_inputs_produce_different_hashes(self):
        """Test that different inputs produce different hashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            pipeline = SnowAnalysisPipeline(cache_dir=cache_dir)

            dem1 = np.array([[1, 2], [3, 4]])
            dem2 = np.array([[1, 2], [3, 5]])  # Different
            extent = (0, 0, 10, 10)

            hash1 = pipeline._compute_task_hash(
                "compute_snow_stats",
                dem=dem1,
                extent=extent,
                snodas_dir="/path/a"
            )
            hash2 = pipeline._compute_task_hash(
                "compute_snow_stats",
                dem=dem2,
                extent=extent,
                snodas_dir="/path/a"
            )

            assert hash1 != hash2
