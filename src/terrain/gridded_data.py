"""
Generic gridded data loader with pipeline caching for terrain visualization.

Handles loading external gridded datasets (SNODAS, temperature, precipitation, etc.),
processing through user-defined pipelines, and caching each step independently.
"""

import hashlib
import inspect
import json
import logging
from pathlib import Path
from typing import Callable, Dict, Any, Tuple, List
import numpy as np

logger = logging.getLogger(__name__)


class GriddedDataLoader:
    """
    Load and cache external gridded data with pipeline processing.

    This class provides a general framework for:
    - Loading gridded data from arbitrary formats
    - Processing data through multi-step pipelines
    - Caching each pipeline step independently
    - Smart cache invalidation based on step dependencies

    Pipeline format: List of (name, function, kwargs) tuples

    Example:
        >>> def load_data(source, extent, target_shape):
        ...     # Load and crop data
        ...     return {"raw": data_array}
        >>>
        >>> def compute_stats(input_data):
        ...     # Compute statistics from previous step
        ...     raw = input_data["raw"]
        ...     return {"mean": raw.mean(), "std": raw.std()}
        >>>
        >>> pipeline = [
        ...     ("load", load_data, {}),
        ...     ("stats", compute_stats, {}),
        ... ]
        >>>
        >>> loader = GriddedDataLoader(terrain, cache_dir=Path(".cache"))
        >>> result = loader.run_pipeline(
        ...     data_source="/path/to/data",
        ...     pipeline=pipeline,
        ...     cache_name="my_analysis"
        ... )
    """

    def __init__(self, terrain, cache_dir: Path = None):
        """
        Initialize gridded data loader.

        Args:
            terrain: Terrain object (provides extent and resolution)
            cache_dir: Directory for caching (default: .gridded_data_cache)
        """
        self.terrain = terrain
        self.cache_dir = cache_dir or Path(".gridded_data_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"GriddedDataLoader initialized at: {self.cache_dir}")

    def run_pipeline(
        self,
        data_source: Any,
        pipeline: List[Tuple[str, Callable, Dict]],
        cache_name: str,
        force_reprocess: bool = False,
    ) -> Any:
        """
        Execute a processing pipeline with caching at each step.

        Args:
            data_source: Data source (directory, file list, URL, etc.)
            pipeline: List of (step_name, function, kwargs) tuples
                      Each function receives previous step's output as first arg
            cache_name: Base name for cache files
            force_reprocess: Force reprocessing all steps even if cached

        Returns:
            Output of final pipeline step
        """
        logger.info(f"Running pipeline '{cache_name}' with {len(pipeline)} steps")

        current_data = data_source
        upstream_cache_key = self._compute_source_cache_key(data_source)

        for i, (step_name, step_func, step_kwargs) in enumerate(pipeline):
            logger.info(f"  Step {i+1}/{len(pipeline)}: {step_name}")

            # Execute step with caching
            current_data, step_cache_key = self._execute_step(
                step_name=step_name,
                func=step_func,
                input_data=current_data,
                kwargs=step_kwargs,
                upstream_cache_key=upstream_cache_key,
                cache_name=cache_name,
                force_reprocess=force_reprocess,
            )

            # Update upstream key for next step
            upstream_cache_key = step_cache_key

        logger.info(f"Pipeline '{cache_name}' completed")
        return current_data

    def _execute_step(
        self,
        step_name: str,
        func: Callable,
        input_data: Any,
        kwargs: Dict,
        upstream_cache_key: str,
        cache_name: str,
        force_reprocess: bool,
    ) -> Tuple[Any, str]:
        """
        Execute a single pipeline step with caching.

        Args:
            step_name: Name of this step
            func: Function to execute
            input_data: Output from previous step (or data_source for first step)
            kwargs: Additional arguments for func
            upstream_cache_key: Cache key from previous step
            cache_name: Base cache name
            force_reprocess: Force recomputation

        Returns:
            Tuple of (step_output, step_cache_key)
        """
        # Compute cache key for this step
        step_cache_key = self._compute_step_cache_key(
            step_name, func, kwargs, upstream_cache_key
        )

        cache_file = self.cache_dir / f"{cache_name}_{step_name}_{step_cache_key[:16]}.npz"

        # Try to load from cache
        if not force_reprocess and cache_file.exists():
            logger.debug(f"    Cache hit: {cache_file.name}")
            step_output = self._load_step_cache(cache_file)
            if step_output is not None:
                return step_output, step_cache_key

        # Execute step
        logger.debug(f"    Executing: {func.__name__}")

        # Inject terrain parameters if needed
        if self._needs_terrain_params(func):
            extent = self.terrain.dem_bounds
            pixel_size = 1.0 / 120.0  # TODO: make configurable
            minx, miny, maxx, maxy = extent
            target_width = int(round((maxx - minx) / pixel_size))
            target_height = int(round((maxy - miny) / pixel_size))

            kwargs = {
                **kwargs,
                "extent": extent,
                "target_shape": (target_height, target_width),
            }

        step_output = func(input_data, **kwargs)

        # Save to cache
        self._save_step_cache(cache_file, step_output)
        logger.debug(f"    Cached: {cache_file.name}")

        return step_output, step_cache_key

    def _compute_source_cache_key(self, data_source: Any) -> str:
        """Compute cache key for data source."""
        source_str = str(data_source)
        extent = self.terrain.dem_bounds
        cache_str = f"{source_str}:{extent}"
        return hashlib.sha256(cache_str.encode()).hexdigest()

    def _compute_step_cache_key(
        self, step_name: str, func: Callable, kwargs: Dict, upstream_key: str
    ) -> str:
        """
        Compute cache key for a pipeline step.

        Includes:
        - Step name
        - Function source code hash
        - Step kwargs
        - Upstream cache key (dependency tracking)
        """
        try:
            # Hash function source code
            func_source = inspect.getsource(func)
            func_hash = hashlib.sha256(func_source.encode()).hexdigest()
        except (OSError, TypeError):
            # Fallback for built-in functions or functions without source
            func_hash = hashlib.sha256(func.__name__.encode()).hexdigest()

        # Hash kwargs (convert to sorted json for stability)
        kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)

        # Combine all components
        cache_str = f"{step_name}:{func_hash}:{kwargs_str}:{upstream_key}"
        return hashlib.sha256(cache_str.encode()).hexdigest()

    def _needs_terrain_params(self, func: Callable) -> bool:
        """Check if function signature expects extent/target_shape parameters."""
        try:
            sig = inspect.signature(func)
            params = sig.parameters.keys()
            return "extent" in params or "target_shape" in params
        except (ValueError, TypeError):
            return False

    def _save_step_cache(self, cache_file: Path, data: Any):
        """Save step output to cache."""
        try:
            if isinstance(data, dict):
                # Save dict of arrays
                np.savez_compressed(cache_file, **data)
            elif isinstance(data, np.ndarray):
                # Save single array
                np.savez_compressed(cache_file, data=data)
            else:
                # Pickle for other types
                np.savez_compressed(cache_file, data=np.array(data, dtype=object))
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _load_step_cache(self, cache_file: Path) -> Any:
        """Load step output from cache."""
        try:
            with np.load(cache_file, allow_pickle=True) as npz:
                if len(npz.files) == 1 and "data" in npz.files:
                    # Single array or pickled object
                    data = npz["data"]
                    return data.item() if data.dtype == object else data
                else:
                    # Dict of arrays
                    return {k: npz[k] for k in npz.files}
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
