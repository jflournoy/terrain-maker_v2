"""
Generic gridded data loader with pipeline caching for terrain visualization.

Handles loading external gridded datasets (SNODAS, temperature, precipitation, etc.),
processing through user-defined pipelines, and caching each step independently.

Features:
- Transparent automatic tiling for large datasets
- Memory monitoring with failsafe to prevent OOM/thrashing
- Per-step and merged result caching
- Smart aggregation (concatenation for spatial data, averaging for statistics)
"""

import gc
import hashlib
import inspect
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Any, Tuple, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class MemoryLimitExceeded(Exception):
    """Raised when memory usage exceeds configured limits."""

    pass


@dataclass
class TiledDataConfig:
    """Configuration for automatic tiling in GriddedDataLoader."""

    max_output_pixels: int = 4096 * 4096
    """Maximum output pixels before triggering tiling (default: ~16M = ~64MB for float32)."""

    target_tile_outputs: int = 2000
    """Target output pixels per tile dimension (default: 2000x2000)."""

    halo: int = 0
    """Halo size for operations needing boundary overlap (default: 0 for gridded data)."""

    enable_tile_cache: bool = True
    """Cache individual tiles (vs only final merged result)."""

    aggregation_strategy: str = "auto"
    """How to merge tiles: 'concatenate', 'mean', 'weighted_mean', 'auto' (default)."""

    max_memory_percent: float = 85.0
    """Maximum RAM usage percent before aborting (default: 85%)."""

    max_swap_percent: float = 50.0
    """Maximum swap usage percent before aborting (default: 50%)."""

    memory_check_interval: float = 5.0
    """Seconds between memory checks (default: 5s)."""

    enable_memory_monitoring: bool = True
    """Enable memory monitoring failsafe (default: True)."""


@dataclass
class TileSpecGridded:
    """Tile specification with geographic extent for gridded data."""

    src_slice: Tuple[slice, slice]
    """Slice into source DEM (with halo)."""

    out_slice: Tuple[slice, slice]
    """Slice into output arrays."""

    extent: Tuple[float, float, float, float]
    """Geographic extent (minx, miny, maxx, maxy)."""

    target_shape: Tuple[int, int]
    """Target output shape for this tile (height, width)."""


class MemoryMonitor:
    """Monitor system memory and abort processing if limits exceeded."""

    def __init__(self, config: TiledDataConfig):
        """
        Initialize memory monitor.

        Args:
            config: TiledDataConfig with memory thresholds
        """
        try:
            import psutil  # pylint: disable=import-outside-toplevel

            self.psutil = psutil
            self.available = True
        except ImportError:
            logger.warning("psutil not installed - memory monitoring disabled")
            self.available = False

        self.max_memory_percent = config.max_memory_percent
        self.max_swap_percent = config.max_swap_percent
        self.check_interval = config.memory_check_interval
        self.last_check = 0
        self.enabled = config.enable_memory_monitoring and self.available

    def check_memory(self, force: bool = False) -> None:
        """
        Check memory usage and raise MemoryLimitExceeded if over threshold.

        Args:
            force: Force check even if check_interval hasn't elapsed

        Raises:
            MemoryLimitExceeded: If memory or swap usage exceeds limits
        """
        if not self.enabled:
            return

        current_time = time.time()

        if not force and (current_time - self.last_check) < self.check_interval:
            return

        self.last_check = current_time

        # Check RAM usage
        memory = self.psutil.virtual_memory()
        if memory.percent > self.max_memory_percent:
            raise MemoryLimitExceeded(
                f"Memory usage {memory.percent:.1f}% exceeds limit "
                f"{self.max_memory_percent}%"
            )

        # Check swap usage
        swap = self.psutil.swap_memory()
        if swap.percent > self.max_swap_percent:
            raise MemoryLimitExceeded(
                f"Swap usage {swap.percent:.1f}% exceeds limit "
                f"{self.max_swap_percent}%"
            )

        logger.debug(f"Memory: {memory.percent:.1f}% RAM, {swap.percent:.1f}% swap")


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

    def __init__(
        self,
        terrain,
        cache_dir: Path = None,
        auto_tile: bool = True,
        tile_config: Optional[TiledDataConfig] = None,
    ):
        """
        Initialize gridded data loader.

        Args:
            terrain: Terrain object (provides extent and resolution)
            cache_dir: Directory for caching (default: .gridded_data_cache)
            auto_tile: Enable automatic tiling when outputs exceed memory threshold
                      (default: True)
            tile_config: TiledDataConfig for tiling behavior (uses defaults if None)
        """
        self.terrain = terrain
        self.cache_dir = cache_dir or Path(".gridded_data_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.auto_tile = auto_tile
        self.tile_config = tile_config or TiledDataConfig()
        logger.debug(f"GriddedDataLoader initialized at: {self.cache_dir}")
        logger.debug(f"Auto-tiling: {self.auto_tile}")

    def run_pipeline(
        self,
        data_source: Any,
        pipeline: List[Tuple[str, Callable, Dict]],
        cache_name: str,
        force_reprocess: bool = False,
    ) -> Any:
        """
        Execute a processing pipeline with caching at each step.

        Features:
        - Transparent automatic tiling for large outputs
        - Memory monitoring with failsafe
        - Per-step and merged result caching

        Args:
            data_source: Data source (directory, file list, URL, etc.)
            pipeline: List of (step_name, function, kwargs) tuples
                      Each function receives previous step's output as first arg
            cache_name: Base name for cache files
            force_reprocess: Force reprocessing all steps even if cached

        Returns:
            Output of final pipeline step

        Raises:
            MemoryLimitExceeded: If memory limits exceeded during tiling
        """
        logger.info(f"Running pipeline '{cache_name}' with {len(pipeline)} steps")

        # Check if tiling is needed
        if self.auto_tile and self._should_tile(data_source, pipeline, cache_name):
            # Get terrain extent
            extent = self.terrain.dem_bounds

            # Determine target shape from first step output (cached from _should_tile)
            try:
                step_name, step_func, step_kwargs = pipeline[0]
                step_output, _ = self._execute_step(
                    step_name=step_name,
                    func=step_func,
                    input_data=data_source,
                    kwargs=step_kwargs,
                    upstream_cache_key=self._compute_source_cache_key(data_source),
                    cache_name=cache_name,
                    force_reprocess=False,  # Use cached result
                )
                target_shape = self._get_output_shape(step_output)
                if not target_shape or len(target_shape) < 2:
                    logger.warning("Could not determine output shape, skipping tiling")
                    return self._run_pipeline_non_tiled(
                        data_source, pipeline, cache_name, force_reprocess
                    )
                target_shape = tuple(target_shape[:2])  # Use first 2 dimensions
            except Exception as e:
                logger.warning(f"Error determining target shape for tiling: {e}, skipping tiling")
                return self._run_pipeline_non_tiled(
                    data_source, pipeline, cache_name, force_reprocess
                )

            # Create tile specifications
            tile_specs = self._create_tile_specs(target_shape, extent)

            # Execute tiled pipeline
            return self._execute_tiled_pipeline(
                data_source, pipeline, cache_name, tile_specs, force_reprocess, target_shape, extent
            )
        else:
            # No tiling needed, use standard pipeline execution
            return self._run_pipeline_non_tiled(
                data_source, pipeline, cache_name, force_reprocess
            )

    def _run_pipeline_non_tiled(
        self,
        data_source: Any,
        pipeline: List[Tuple[str, Callable, Dict]],
        cache_name: str,
        force_reprocess: bool,
    ) -> Any:
        """
        Execute pipeline without tiling (original logic).

        Args:
            data_source: Data source
            pipeline: Pipeline to execute
            cache_name: Cache name
            force_reprocess: Force reprocessing flag

        Returns:
            Output of final pipeline step
        """
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

    def _should_tile(
        self,
        data_source: Any,
        pipeline: List[Tuple[str, Callable, Dict]],
        cache_name: str,
    ) -> bool:
        """
        Detect if pipeline output would exceed memory threshold via dry-run.

        Executes first pipeline step to determine output shape and size.
        Leverages caching to avoid redundant computation.

        Args:
            data_source: Data source
            pipeline: Pipeline to analyze
            cache_name: Cache name for steps

        Returns:
            True if output size exceeds max_output_pixels threshold
        """
        if not self.auto_tile or not pipeline:
            return False

        try:
            step_name, step_func, step_kwargs = pipeline[0]

            # Execute first step (may hit cache)
            step_output, _ = self._execute_step(
                step_name=step_name,
                func=step_func,
                input_data=data_source,
                kwargs=step_kwargs,
                upstream_cache_key=self._compute_source_cache_key(data_source),
                cache_name=cache_name,
                force_reprocess=False,
            )

            # Determine output size
            total_pixels = self._get_array_pixel_count(step_output)

            if total_pixels > self.tile_config.max_output_pixels:
                logger.info(
                    f"Auto-tiling triggered: output size {total_pixels:,} pixels "
                    f"exceeds threshold {self.tile_config.max_output_pixels:,}"
                )
                return True

            logger.debug(
                f"Auto-tiling not needed: output size {total_pixels:,} pixels "
                f"within threshold {self.tile_config.max_output_pixels:,}"
            )
            return False

        except Exception as e:
            logger.debug(f"Error detecting tiling need: {e}, defaulting to False")
            return False

    def _get_array_pixel_count(self, data: Any) -> int:
        """
        Get total pixel count from array or dict of arrays.

        Args:
            data: Array, dict of arrays, or other data

        Returns:
            Total number of pixels, or 0 if not array data
        """
        if isinstance(data, np.ndarray):
            return data.size

        if isinstance(data, dict):
            total = 0
            for v in data.values():
                if isinstance(v, np.ndarray):
                    total += v.size
            return total

        return 0

    def _create_tile_specs(
        self,
        target_shape: Tuple[int, int],
        extent: Tuple[float, float, float, float],
    ) -> List[TileSpecGridded]:
        """
        Create tile specifications for gridded data processing.

        Divides target shape into regular tiles based on target_tile_outputs.
        Each tile gets geographic extent and output slice information.

        Args:
            target_shape: Target output shape (height, width)
            extent: Geographic extent (minx, miny, maxx, maxy)

        Returns:
            List of TileSpecGridded with tile layout
        """
        target_h, target_w = target_shape
        minx, miny, maxx, maxy = extent
        tile_size = self.tile_config.target_tile_outputs

        tiles = []

        # Calculate number of tiles
        tiles_h = (target_h + tile_size - 1) // tile_size
        tiles_w = (target_w + tile_size - 1) // tile_size

        logger.debug(f"Creating tile grid: {tiles_h}x{tiles_w} tiles")

        for tile_row in range(tiles_h):
            for tile_col in range(tiles_w):
                # Compute tile bounds in output space
                row_start = tile_row * tile_size
                row_end = min(row_start + tile_size, target_h)
                col_start = tile_col * tile_size
                col_end = min(col_start + tile_size, target_w)

                # Compute geographic extent for this tile
                pixel_size_y = (maxy - miny) / target_h
                pixel_size_x = (maxx - minx) / target_w

                tile_minx = minx + col_start * pixel_size_x
                tile_miny = miny + row_start * pixel_size_y
                tile_maxx = minx + col_end * pixel_size_x
                tile_maxy = miny + row_end * pixel_size_y

                tile_extent = (tile_minx, tile_miny, tile_maxx, tile_maxy)
                tile_target_shape = (row_end - row_start, col_end - col_start)

                tile_spec = TileSpecGridded(
                    src_slice=(slice(row_start, row_end), slice(col_start, col_end)),
                    out_slice=(slice(row_start, row_end), slice(col_start, col_end)),
                    extent=tile_extent,
                    target_shape=tile_target_shape,
                )

                tiles.append(tile_spec)
                logger.debug(
                    f"  Tile [{tile_row},{tile_col}]: "
                    f"output slice [{row_start}:{row_end}, {col_start}:{col_end}], "
                    f"shape {tile_target_shape}"
                )

        return tiles

    def _create_tile_source(
        self,
        data_source: Any,
        tile_spec: TileSpecGridded,
    ) -> Dict[str, Any]:
        """
        Create tile-specific data source with extent and target_shape.

        Extracts tile extent and target shape for this tile so that pipeline
        functions can process just the tile (not full extent).

        Args:
            data_source: Original data source
            tile_spec: Specification for this tile

        Returns:
            Dict with 'extent' and 'target_shape' for this tile, preserving
            original data_source structure/path
        """
        return {
            "data_source": data_source,
            "extent": tile_spec.extent,
            "target_shape": tile_spec.target_shape,
        }

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

    def _execute_tiled_pipeline(
        self,
        data_source: Any,
        pipeline: List[Tuple[str, Callable, Dict]],
        cache_name: str,
        tile_specs: List[TileSpecGridded],
        force_reprocess: bool,
        target_shape: Tuple[int, int],
        extent: Tuple[float, float, float, float],
    ) -> Any:
        """
        Execute pipeline with tiling and memory monitoring.

        Processes each tile independently through the full pipeline,
        with memory checks before each tile to prevent OOM/thrashing.

        Args:
            data_source: Original data source
            pipeline: Pipeline to execute
            cache_name: Cache name for steps
            tile_specs: List of TileSpecGridded specifications
            force_reprocess: Force reprocessing all tiles
            target_shape: Target output shape for final result
            extent: Geographic extent for entire dataset

        Returns:
            Aggregated results from all tiles

        Raises:
            MemoryLimitExceeded: If memory limits exceeded before completion
        """
        logger.info(f"Executing tiled pipeline with {len(tile_specs)} tiles")

        # Initialize memory monitor
        monitor = MemoryMonitor(self.tile_config)

        # Check memory before starting
        try:
            monitor.check_memory(force=True)
        except MemoryLimitExceeded as e:
            logger.error(f"Memory limit exceeded before starting: {e}")
            raise

        tile_outputs = []
        tile_shapes = {}  # Track output shapes for aggregation

        # Process each tile
        for i, tile_spec in enumerate(tile_specs):
            # Check memory before processing this tile
            try:
                monitor.check_memory()
            except MemoryLimitExceeded as e:
                logger.error(f"Memory limit exceeded before tile {i+1}/{len(tile_specs)}: {e}")
                logger.error(
                    f"Processed {i}/{len(tile_specs)} tiles successfully. "
                    f"Consider reducing tile size or freeing memory."
                )
                raise

            logger.info(f"Processing tile {i+1}/{len(tile_specs)}: shape {tile_spec.target_shape}")

            # Create tile-specific data source
            tile_source = self._create_tile_source(data_source, tile_spec)

            # Execute full pipeline for this tile
            try:
                tile_output = self._execute_tile_pipeline(
                    tile_source,
                    pipeline,
                    f"{cache_name}_tile{i}",
                    force_reprocess,
                )
            except Exception as e:
                logger.error(f"Error processing tile {i+1}: {e}")
                raise

            tile_outputs.append(tile_output)

            # Track output shapes for later aggregation
            tile_shapes[i] = self._get_output_shape(tile_output)

            # Force garbage collection after each tile
            gc.collect()

        # Check memory before aggregation
        try:
            monitor.check_memory(force=True)
        except MemoryLimitExceeded as e:
            logger.error(f"Memory limit exceeded before aggregation: {e}")
            raise

        # Aggregate tiles into final result
        merged = self._aggregate_tiles(tile_outputs, tile_specs, target_shape)

        logger.info(f"Tiled pipeline completed, aggregated result shape: {self._get_output_shape(merged)}")

        return merged

    def _execute_tile_pipeline(
        self,
        tile_source: Dict[str, Any],
        pipeline: List[Tuple[str, Callable, Dict]],
        cache_name: str,
        force_reprocess: bool,
    ) -> Any:
        """
        Execute full pipeline for a single tile.

        Similar to run_pipeline but with tile-specific caching.

        Args:
            tile_source: Tile-specific data source (includes extent/target_shape)
            pipeline: Pipeline to execute
            cache_name: Cache name for this tile's steps
            force_reprocess: Force reprocessing

        Returns:
            Output of final pipeline step for this tile
        """
        current_data = tile_source
        upstream_cache_key = hashlib.sha256(
            f"{tile_source['extent']}:{tile_source['target_shape']}".encode()
        ).hexdigest()

        for step_name, step_func, step_kwargs in pipeline:
            # Execute step with tile-specific cache
            current_data, upstream_cache_key = self._execute_step(
                step_name=step_name,
                func=step_func,
                input_data=current_data,
                kwargs=step_kwargs,
                upstream_cache_key=upstream_cache_key,
                cache_name=cache_name,
                force_reprocess=force_reprocess,
            )

        return current_data

    def _get_output_shape(self, data: Any) -> Tuple[int, ...]:
        """Get shape of array or dict of arrays."""
        if isinstance(data, np.ndarray):
            return data.shape

        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, np.ndarray):
                    return v.shape
        return ()

    def _aggregate_tiles(
        self,
        tile_outputs: List[Any],
        tile_specs: List[TileSpecGridded],
        target_shape: Tuple[int, int],
    ) -> Any:
        """
        Aggregate tile outputs into final result.

        Auto-detects aggregation strategy based on output type and shape.

        Args:
            tile_outputs: List of outputs from each tile
            tile_specs: List of tile specifications
            target_shape: Target output shape for final result

        Returns:
            Aggregated result matching original expected output shape
        """
        if not tile_outputs:
            return None

        # Auto-detect aggregation strategy
        if self.tile_config.aggregation_strategy == "auto":
            strategy = self._determine_aggregation(tile_outputs)
        else:
            strategy = self.tile_config.aggregation_strategy

        logger.debug(f"Aggregating {len(tile_outputs)} tiles using strategy: {strategy}")

        if strategy == "concatenate":
            return self._concatenate_spatial(tile_outputs, tile_specs, target_shape)
        elif strategy == "mean":
            return self._average_statistics(tile_outputs)
        elif strategy == "weighted_mean":
            return self._weighted_average_statistics(tile_outputs, tile_specs)
        else:
            logger.warning(f"Unknown aggregation strategy '{strategy}', using first tile")
            return tile_outputs[0]

    def _determine_aggregation(self, tile_outputs: List[Any]) -> str:
        """
        Auto-detect aggregation strategy from tile outputs.

        Args:
            tile_outputs: List of outputs from tiles

        Returns:
            Strategy name: 'concatenate' for 2D+ spatial data, 'mean' for scalars/1D
        """
        if not tile_outputs:
            return "first"

        first_output = tile_outputs[0]

        if isinstance(first_output, dict):
            # Check first array in dict
            for v in first_output.values():
                if isinstance(v, np.ndarray):
                    return "concatenate" if v.ndim >= 2 else "mean"
            return "first"

        elif isinstance(first_output, np.ndarray):
            return "concatenate" if first_output.ndim >= 2 else "mean"

        else:
            return "first"  # Non-array data

    def _concatenate_spatial(
        self,
        tile_outputs: List[Any],
        tile_specs: List[TileSpecGridded],
        target_shape: Tuple[int, int],
    ) -> Any:
        """
        Concatenate spatial arrays using tile specifications.

        Args:
            tile_outputs: List of tile outputs
            tile_specs: List of tile specifications
            target_shape: Target output shape

        Returns:
            Assembled spatial arrays with target_shape
        """
        first_output = tile_outputs[0]

        if isinstance(first_output, dict):
            # Merge each key separately
            result = {}
            for key in first_output.keys():
                arrays = [output[key] for output in tile_outputs if isinstance(output.get(key), np.ndarray)]
                if arrays:
                    result[key] = self._assemble_grid(arrays, tile_specs, target_shape)
            return result
        else:
            return self._assemble_grid(tile_outputs, tile_specs, target_shape)

    def _average_statistics(self, tile_outputs: List[Any]) -> Any:
        """
        Average statistical outputs across tiles.

        Args:
            tile_outputs: List of tile outputs

        Returns:
            Averaged result
        """
        if not tile_outputs:
            return None

        first_output = tile_outputs[0]

        if isinstance(first_output, dict):
            result = {}
            for key in first_output.keys():
                values = [output[key] for output in tile_outputs if key in output]
                if values:
                    if isinstance(values[0], np.ndarray):
                        result[key] = np.mean(values, axis=0)
                    else:
                        result[key] = np.mean(values)
            return result
        elif isinstance(first_output, np.ndarray):
            return np.mean(tile_outputs, axis=0)
        else:
            return np.mean(tile_outputs)

    def _weighted_average_statistics(
        self,
        tile_outputs: List[Any],
        tile_specs: List[TileSpecGridded],
    ) -> Any:
        """
        Weighted average across tiles based on tile sizes.

        Larger tiles get higher weights in the average.

        Args:
            tile_outputs: List of tile outputs
            tile_specs: List of tile specifications

        Returns:
            Weighted average result
        """
        if not tile_outputs:
            return None

        # Compute weights based on tile size
        tile_sizes = np.array([np.prod(spec.target_shape) for spec in tile_specs])
        weights = tile_sizes / tile_sizes.sum()

        first_output = tile_outputs[0]

        if isinstance(first_output, dict):
            result = {}
            for key in first_output.keys():
                values = [output[key] for output in tile_outputs if key in output]
                if values:
                    if isinstance(values[0], np.ndarray):
                        # Weighted average for each array
                        weighted = sum(v * w for v, w in zip(values, weights))
                        result[key] = weighted
                    else:
                        result[key] = sum(v * w for v, w in zip(values, weights))
            return result
        elif isinstance(first_output, np.ndarray):
            return sum(v * w for v, w in zip(tile_outputs, weights))
        else:
            return sum(v * w for v, w in zip(tile_outputs, weights))

    def _assemble_grid(
        self,
        arrays: List[np.ndarray],
        tile_specs: List[TileSpecGridded],
        target_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Assemble tiled arrays into final grid using tile specifications.

        Args:
            arrays: List of tile arrays
            tile_specs: List of tile specifications
            target_shape: Target output shape

        Returns:
            Final assembled array with target_shape
        """
        if not arrays or not tile_specs:
            return None

        # Initialize output array with first array's dtype
        dtype = arrays[0].dtype
        output = np.zeros(target_shape, dtype=dtype)

        # Place each tile into its position
        for array, spec in zip(arrays, tile_specs):
            output[spec.out_slice] = array

        return output

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
