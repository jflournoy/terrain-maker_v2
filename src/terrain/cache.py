"""
DEM caching module for efficient terrain visualization pipeline.

Implements .npz-based caching with hash validation to avoid reloading
and reprocessing expensive DEM merging operations.
"""

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from rasterio import Affine
import time

logger = logging.getLogger(__name__)


class DEMCache:
    """
    Manages caching of loaded and merged DEM data with hash validation.

    The cache stores:
    - DEM array as .npz file
    - Metadata including file hash, timestamp, and file list

    Attributes:
        cache_dir: Directory where cache files are stored
        enabled: Whether caching is enabled
    """

    def __init__(self, cache_dir: Optional[Path] = None, enabled: bool = True):
        """
        Initialize DEM cache.

        Args:
            cache_dir: Directory for cache files. If None, uses .dem_cache/ in project root
            enabled: Whether caching is enabled (default: True)
        """
        if cache_dir is None:
            cache_dir = Path.cwd() / ".dem_cache"

        self.cache_dir = Path(cache_dir)
        self.enabled = enabled

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"DEM cache initialized at: {self.cache_dir}")

    def compute_source_hash(
        self, directory_path: str, pattern: str, recursive: bool = False
    ) -> str:
        """
        Compute hash of source DEM files based on paths and modification times.

        This ensures the cache is invalidated if:
        - Files are added/removed
        - Files are modified
        - Directory path changes

        Args:
            directory_path: Path to DEM directory
            pattern: File pattern (e.g., "*.hgt")
            recursive: Whether search is recursive

        Returns:
            SHA256 hash of source file metadata
        """
        from pathlib import Path
        import glob

        directory = Path(directory_path)

        # Get matching files
        glob_func = directory.rglob if recursive else directory.glob
        dem_files = sorted(glob_func(pattern))

        if not dem_files:
            raise ValueError(f"No files matching '{pattern}' found in {directory}")

        # Build metadata string including paths and mtimes
        metadata_parts = [
            str(directory.resolve()),  # Absolute path
            pattern,
            str(recursive),
        ]

        for file_path in dem_files:
            # Include file path and modification time
            mtime = file_path.stat().st_mtime
            metadata_parts.append(f"{file_path}:{mtime}")

        # Compute hash
        metadata_str = "|".join(metadata_parts)
        hash_obj = hashlib.sha256(metadata_str.encode())
        return hash_obj.hexdigest()

    def get_cache_path(self, source_hash: str, cache_name: str = "dem") -> Path:
        """
        Get the path for a cache file.

        Args:
            source_hash: Hash of source files
            cache_name: Name of cache item (default: "dem")

        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{cache_name}_{source_hash}.npz"

    def get_metadata_path(self, source_hash: str, cache_name: str = "dem") -> Path:
        """
        Get the path for cache metadata file.

        Args:
            source_hash: Hash of source files
            cache_name: Name of cache item (default: "dem")

        Returns:
            Path to metadata file
        """
        return self.cache_dir / f"{cache_name}_{source_hash}_meta.json"

    def save_cache(
        self, dem_array: np.ndarray, transform: Affine, source_hash: str, cache_name: str = "dem"
    ) -> Tuple[Path, Path]:
        """
        Save DEM array and transform to cache.

        Args:
            dem_array: Merged DEM array
            transform: Affine transform
            source_hash: Hash of source files
            cache_name: Name of cache item (default: "dem")

        Returns:
            Tuple of (cache_file_path, metadata_file_path)
        """
        if not self.enabled:
            return None, None

        cache_path = self.get_cache_path(source_hash, cache_name)
        metadata_path = self.get_metadata_path(source_hash, cache_name)

        start_time = time.time()

        # Save DEM array with transform as list (Affine constructor: a, b, c, d, e, f)
        transform_list = [
            transform.a,
            transform.b,
            transform.c,
            transform.d,
            transform.e,
            transform.f,
        ]
        np.savez_compressed(
            cache_path, dem=dem_array, transform_data=np.array(transform_list, dtype=np.float64)
        )

        # Save metadata
        metadata = {
            "source_hash": source_hash,
            "dem_shape": dem_array.shape,
            "dem_dtype": str(dem_array.dtype),
            "dem_min": float(np.nanmin(dem_array)),
            "dem_max": float(np.nanmax(dem_array)),
            "cache_time": time.time(),
            "transform": transform_list,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        elapsed = time.time() - start_time
        logger.info(f"Cached DEM to {cache_path.name} ({elapsed:.2f}s)")
        logger.debug(f"Cache size: {cache_path.stat().st_size / (1024*1024):.1f} MB")

        return cache_path, metadata_path

    def load_cache(
        self, source_hash: str, cache_name: str = "dem"
    ) -> Optional[Tuple[np.ndarray, Affine]]:
        """
        Load cached DEM data.

        Args:
            source_hash: Hash of source files
            cache_name: Name of cache item (default: "dem")

        Returns:
            Tuple of (dem_array, transform) or None if cache doesn't exist
        """
        if not self.enabled:
            return None

        cache_path = self.get_cache_path(source_hash, cache_name)

        if not cache_path.exists():
            logger.debug(f"Cache miss: {cache_path.name}")
            return None

        try:
            start_time = time.time()

            # Load cache
            cache_data = np.load(cache_path, allow_pickle=True)
            dem_array = cache_data["dem"]
            transform_data = cache_data["transform_data"]

            # Reconstruct Affine transform from stored components (a, b, c, d, e, f)
            if isinstance(transform_data, np.ndarray):
                transform_values = tuple(transform_data)
            else:
                transform_values = tuple(transform_data)

            transform = Affine(*transform_values)

            elapsed = time.time() - start_time
            logger.info(f"Loaded DEM from cache ({elapsed:.2f}s)")
            logger.debug(f"Cache file: {cache_path.name}")
            logger.debug(f"DEM shape: {dem_array.shape}, dtype: {dem_array.dtype}")

            return dem_array, transform

        except Exception as e:
            logger.warning(f"Failed to load cache {cache_path.name}: {e}")
            logger.debug("Cache will be regenerated")
            return None

    def clear_cache(self, cache_name: str = "dem") -> int:
        """
        Clear all cached files for a given cache name.

        Args:
            cache_name: Name of cache item to clear

        Returns:
            Number of files deleted
        """
        if not self.enabled:
            return 0

        deleted_count = 0

        # Find all cache files matching the pattern
        for cache_file in self.cache_dir.glob(f"{cache_name}_*"):
            try:
                cache_file.unlink()
                deleted_count += 1
                logger.debug(f"Deleted: {cache_file.name}")
            except Exception as e:
                logger.warning(f"Failed to delete {cache_file.name}: {e}")

        logger.info(f"Cleared {deleted_count} cache files for '{cache_name}'")
        return deleted_count

    def get_cache_stats(self) -> dict:
        """
        Get statistics about cached files.

        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "cache_dir": str(self.cache_dir),
            "enabled": self.enabled,
            "cache_files": 0,
            "total_size_mb": 0,
            "files": [],
        }

        if not self.cache_dir.exists():
            return stats

        for cache_file in self.cache_dir.glob("*"):
            if cache_file.is_file():
                size_bytes = cache_file.stat().st_size
                stats["cache_files"] += 1
                stats["total_size_mb"] += size_bytes / (1024 * 1024)
                stats["files"].append(
                    {
                        "name": cache_file.name,
                        "size_mb": size_bytes / (1024 * 1024),
                        "mtime": cache_file.stat().st_mtime,
                    }
                )

        return stats


class TransformCache:
    """
    Cache for transform pipeline results with dependency tracking.

    Tracks chains of transforms (reproject -> smooth -> water_detect) and
    computes cache keys that incorporate the full dependency chain, ensuring
    downstream caches are invalidated when upstream params change.

    Attributes:
        cache_dir: Directory where cache files are stored
        enabled: Whether caching is enabled
        dependencies: Graph of transform dependencies
        transforms: Registered transforms with their parameters
    """

    def __init__(self, cache_dir: Optional[Path] = None, enabled: bool = True):
        """
        Initialize transform cache.

        Args:
            cache_dir: Directory for cache files. If None, uses .transform_cache/
            enabled: Whether caching is enabled (default: True)
        """
        if cache_dir is None:
            cache_dir = Path.cwd() / ".transform_cache"

        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self.dependencies: dict[str, str] = {}  # child -> parent
        self.transforms: dict[str, dict] = {}  # name -> {upstream, params}

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Transform cache initialized at: {self.cache_dir}")

    def compute_transform_hash(
        self,
        upstream_hash: str,
        transform_name: str,
        params: dict,
    ) -> str:
        """
        Compute cache key from upstream hash and transform parameters.

        The key incorporates:
        - Upstream cache key (propagating the full dependency chain)
        - Transform name
        - All transform parameters (sorted for determinism)

        Args:
            upstream_hash: Hash of upstream data/transform
            transform_name: Name of this transform (e.g., "reproject", "smooth")
            params: Transform parameters dict

        Returns:
            SHA256 hash string (64 chars)
        """
        # Convert params to deterministic string representation
        def serialize_value(v):
            """Convert a parameter value to a deterministic string for hashing.

            Handles numpy arrays specially by including shape, dtype, and content hash.
            Other values are converted to their string representation.

            Args:
                v: Parameter value (can be ndarray, scalar, string, etc.)

            Returns:
                Deterministic string representation of the value
            """
            if isinstance(v, np.ndarray):
                return f"ndarray:{v.shape}:{v.dtype}:{hash(v.tobytes())}"
            return str(v)

        sorted_params = sorted(
            (k, serialize_value(v)) for k, v in params.items()
        )

        metadata_parts = [
            upstream_hash,
            transform_name,
            json.dumps(sorted_params, sort_keys=True),
        ]

        metadata_str = "|".join(metadata_parts)
        hash_obj = hashlib.sha256(metadata_str.encode())
        return hash_obj.hexdigest()

    def get_cache_path(self, cache_key: str, transform_name: str) -> Path:
        """
        Get path for cache file.

        Args:
            cache_key: Cache key hash
            transform_name: Name of transform

        Returns:
            Path to cache .npz file
        """
        return self.cache_dir / f"{transform_name}_{cache_key[:16]}.npz"

    def get_metadata_path(self, cache_key: str, transform_name: str) -> Path:
        """
        Get path for metadata file.

        Args:
            cache_key: Cache key hash
            transform_name: Name of transform

        Returns:
            Path to metadata .json file
        """
        return self.cache_dir / f"{transform_name}_{cache_key[:16]}_meta.json"

    def save_transform(
        self,
        cache_key: str,
        data: np.ndarray,
        transform_name: str,
        metadata: Optional[dict] = None,
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Save transform result to cache.

        Args:
            cache_key: Cache key hash
            data: Transform result array
            transform_name: Name of transform
            metadata: Optional additional metadata

        Returns:
            Tuple of (cache_path, metadata_path) or (None, None) if disabled
        """
        if not self.enabled:
            return None, None

        cache_path = self.get_cache_path(cache_key, transform_name)
        meta_path = self.get_metadata_path(cache_key, transform_name)

        start_time = time.time()

        # Save data
        np.savez_compressed(cache_path, data=data)

        # Save metadata
        meta = {
            "cache_key": cache_key,
            "transform_name": transform_name,
            "shape": data.shape,
            "dtype": str(data.dtype),
            "cache_time": time.time(),
        }
        if metadata:
            meta.update(metadata)

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        elapsed = time.time() - start_time
        logger.debug(f"Cached {transform_name} ({elapsed:.2f}s)")

        return cache_path, meta_path

    def load_transform(
        self,
        cache_key: str,
        transform_name: str,
    ) -> Optional[np.ndarray]:
        """
        Load transform result from cache.

        Args:
            cache_key: Cache key hash
            transform_name: Name of transform

        Returns:
            Cached array or None if cache miss/disabled
        """
        if not self.enabled:
            return None

        cache_path = self.get_cache_path(cache_key, transform_name)

        if not cache_path.exists():
            logger.debug(f"Cache miss: {transform_name}")
            return None

        try:
            cache_data = np.load(cache_path)
            data = cache_data["data"]
            logger.debug(f"Cache hit: {transform_name}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_path}: {e}")
            return None

    def register_dependency(self, child: str, upstream: str) -> None:
        """
        Register a dependency between transforms.

        Args:
            child: Name of dependent transform
            upstream: Name of upstream transform it depends on
        """
        self.dependencies[child] = upstream

    def register_transform(
        self,
        name: str,
        upstream: str,
        params: dict,
    ) -> None:
        """
        Register a transform with its parameters.

        Args:
            name: Transform name
            upstream: Name of upstream dependency
            params: Transform parameters
        """
        self.register_dependency(name, upstream)
        self.transforms[name] = {
            "upstream": upstream,
            "params": params,
        }

    def get_dependency_chain(self, transform_name: str) -> list[str]:
        """
        Get full dependency chain for a transform.

        Args:
            transform_name: Name of transform

        Returns:
            List of transform names from root to target
        """
        chain = [transform_name]
        current = transform_name

        while current in self.dependencies:
            parent = self.dependencies[current]
            chain.insert(0, parent)
            current = parent

        return chain

    def get_full_cache_key(self, transform_name: str, source_hash: str) -> str:
        """
        Compute full cache key incorporating dependency chain.

        Args:
            transform_name: Target transform name
            source_hash: Hash of original source data

        Returns:
            Cache key hash
        """
        chain = self.get_dependency_chain(transform_name)

        # Start with source hash
        current_hash = source_hash

        # Walk chain, computing hash at each step
        for name in chain[1:]:  # Skip source
            if name in self.transforms:
                params = self.transforms[name]["params"]
                current_hash = self.compute_transform_hash(
                    current_hash, name, params
                )

        return current_hash

    def invalidate_downstream(self, transform_name: str) -> int:
        """
        Invalidate all caches downstream of a transform.

        Args:
            transform_name: Name of transform whose downstream should be invalidated

        Returns:
            Number of cache files deleted
        """
        if not self.enabled:
            return 0

        # Find all downstream transforms
        to_invalidate = {transform_name}
        changed = True
        while changed:
            changed = False
            for child, parent in self.dependencies.items():
                if parent in to_invalidate and child not in to_invalidate:
                    to_invalidate.add(child)
                    changed = True

        # Delete cache files for all downstream transforms
        deleted = 0
        for name in to_invalidate:
            for cache_file in self.cache_dir.glob(f"{name}_*"):
                try:
                    cache_file.unlink()
                    deleted += 1
                    logger.debug(f"Invalidated: {cache_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete {cache_file}: {e}")

        logger.info(f"Invalidated {deleted} downstream cache files")
        return deleted


class PipelineCache:
    """
    Target-style caching for terrain processing pipelines.

    Like a build system (Make, Bazel), this cache:
    - Tracks targets with defined parameters and dependencies
    - Computes cache keys that incorporate the FULL dependency chain
    - Ensures downstream targets are invalidated when upstream changes
    - Supports file inputs with mtime tracking

    Example:
        cache = PipelineCache()
        cache.define_target("dem_loaded", params={"path": "/data"})
        cache.define_target("reprojected", params={"crs": "EPSG:32617"},
                           dependencies=["dem_loaded"])

        # First run: cache miss
        if cache.get_cached("reprojected") is None:
            data = expensive_operation()
            cache.save_target("reprojected", data)

        # Second run (same params): cache hit
        # If dem_loaded params change: cache miss (invalidated)

    Attributes:
        cache_dir: Directory where cache files are stored
        enabled: Whether caching is enabled
        targets: Dict of target definitions {name: {params, dependencies, file_inputs}}
    """

    def __init__(self, cache_dir: Optional[Path] = None, enabled: bool = True):
        """
        Initialize pipeline cache.

        Args:
            cache_dir: Directory for cache files. If None, uses .pipeline_cache/
            enabled: Whether caching is enabled (default: True)
        """
        if cache_dir is None:
            cache_dir = Path.cwd() / ".pipeline_cache"

        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self.targets: dict[str, dict] = {}  # name -> {params, dependencies, file_inputs}

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Pipeline cache initialized at: {self.cache_dir}")

    def _serialize_value(self, v) -> str:
        """Serialize a value to a deterministic string representation."""
        if isinstance(v, np.ndarray):
            return f"ndarray:{v.shape}:{v.dtype}:{hash(v.tobytes())}"
        if isinstance(v, Affine):
            return f"Affine:{v.a},{v.b},{v.c},{v.d},{v.e},{v.f}"
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, (list, tuple)):
            return json.dumps([self._serialize_value(x) for x in v])
        if isinstance(v, dict):
            return json.dumps({k: self._serialize_value(val) for k, val in sorted(v.items())})
        return str(v)

    def _has_circular_dependency(self, target_name: str, dependencies: list[str]) -> bool:
        """Check if adding these dependencies would create a cycle."""
        # Build a set of all ancestors of each dependency
        visited = set()
        to_check = list(dependencies)

        while to_check:
            dep = to_check.pop()
            if dep == target_name:
                return True
            if dep in visited:
                continue
            visited.add(dep)
            if dep in self.targets:
                to_check.extend(self.targets[dep].get("dependencies", []))

        return False

    def define_target(
        self,
        name: str,
        params: dict,
        dependencies: Optional[list[str]] = None,
        file_inputs: Optional[list[Path]] = None,
    ) -> None:
        """
        Define a pipeline target with its parameters and dependencies.

        Args:
            name: Unique name for this target
            params: Parameters that affect the target's output
            dependencies: List of upstream target names this depends on
            file_inputs: List of file paths whose mtimes should be tracked

        Raises:
            ValueError: If adding this target would create a circular dependency
        """
        dependencies = dependencies or []
        file_inputs = file_inputs or []

        # Check for circular dependencies
        if self._has_circular_dependency(name, dependencies):
            raise ValueError(
                f"Circular dependency detected: adding dependencies {dependencies} "
                f"to target '{name}' would create a cycle"
            )

        self.targets[name] = {
            "params": params,
            "dependencies": dependencies,
            "file_inputs": [Path(f) for f in file_inputs],
        }

    def _compute_file_inputs_hash(self, file_inputs: list[Path]) -> str:
        """Compute hash component from file modification times."""
        if not file_inputs:
            return ""

        parts = []
        for f in sorted(file_inputs):
            if f.exists():
                mtime = f.stat().st_mtime
                parts.append(f"{f}:{mtime}")
            else:
                parts.append(f"{f}:MISSING")

        return "|".join(parts)

    def compute_target_key(self, target_name: str) -> str:
        """
        Compute cache key for a target, incorporating all upstream dependencies.

        The key is a SHA256 hash that changes if:
        - Target's own params change
        - Any upstream target's params change
        - Any file inputs are modified

        Args:
            target_name: Name of the target

        Returns:
            64-character hex SHA256 hash, or empty string if target undefined
        """
        if target_name not in self.targets:
            return ""

        target = self.targets[target_name]

        # Recursively get upstream keys
        upstream_keys = []
        for dep in target["dependencies"]:
            upstream_key = self.compute_target_key(dep)
            upstream_keys.append(f"{dep}:{upstream_key}")

        # Build hash input
        parts = [
            target_name,
            json.dumps(
                [(k, self._serialize_value(v)) for k, v in sorted(target["params"].items())],
                sort_keys=True,
            ),
            "|".join(sorted(upstream_keys)),
            self._compute_file_inputs_hash(target.get("file_inputs", [])),
        ]

        metadata_str = "||".join(parts)
        hash_obj = hashlib.sha256(metadata_str.encode())
        return hash_obj.hexdigest()

    def _get_cache_path(self, target_name: str, cache_key: str) -> Path:
        """Get path for cache file."""
        return self.cache_dir / f"{target_name}_{cache_key[:16]}.npz"

    def _get_metadata_path(self, target_name: str, cache_key: str) -> Path:
        """Get path for metadata file."""
        return self.cache_dir / f"{target_name}_{cache_key[:16]}_meta.pkl"

    def save_target(
        self,
        target_name: str,
        data,
        metadata: Optional[dict] = None,
    ) -> Optional[Path]:
        """
        Save target output to cache.

        Args:
            target_name: Name of the target
            data: numpy array, or dict of arrays to cache
            metadata: Optional additional metadata (can include Affine transforms)

        Returns:
            Path to cache file, or None if disabled
        """
        if not self.enabled:
            return None

        if target_name not in self.targets:
            logger.warning(f"Cannot save undefined target: {target_name}")
            return None

        cache_key = self.compute_target_key(target_name)
        cache_path = self._get_cache_path(target_name, cache_key)
        meta_path = self._get_metadata_path(target_name, cache_key)

        start_time = time.time()

        # Handle dict of arrays vs single array
        if isinstance(data, dict):
            np.savez_compressed(cache_path, **data)
        else:
            np.savez_compressed(cache_path, data=data)

        # Save metadata with pickle (supports Affine, etc.)
        meta = {
            "cache_key": cache_key,
            "target_name": target_name,
            "cache_time": time.time(),
            "is_dict": isinstance(data, dict),
        }
        if metadata:
            meta.update(metadata)

        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

        elapsed = time.time() - start_time
        logger.debug(f"Cached {target_name} ({elapsed:.2f}s)")

        return cache_path

    def get_cached(
        self,
        target_name: str,
        return_metadata: bool = False,
    ):
        """
        Get cached target output if available.

        Args:
            target_name: Name of the target
            return_metadata: If True, return (data, metadata) tuple

        Returns:
            Cached data (array or dict of arrays), or None if cache miss.
            If return_metadata=True, returns (data, metadata) or (None, None)
        """
        if not self.enabled:
            return (None, None) if return_metadata else None

        if target_name not in self.targets:
            return (None, None) if return_metadata else None

        cache_key = self.compute_target_key(target_name)
        cache_path = self._get_cache_path(target_name, cache_key)
        meta_path = self._get_metadata_path(target_name, cache_key)

        if not cache_path.exists():
            logger.debug(f"Cache miss: {target_name}")
            return (None, None) if return_metadata else None

        try:
            # Load metadata first to check if dict
            meta = {}
            if meta_path.exists():
                with open(meta_path, "rb") as f:
                    meta = pickle.load(f)

            # Load data
            cache_data = np.load(cache_path, allow_pickle=True)

            if meta.get("is_dict", False):
                # Return dict of arrays
                data = {key: cache_data[key] for key in cache_data.files}
            else:
                data = cache_data["data"]

            logger.debug(f"Cache hit: {target_name}")

            if return_metadata:
                return data, meta
            return data

        except Exception as e:
            logger.warning(f"Failed to load cache for {target_name}: {e}")
            return (None, None) if return_metadata else None

    def clear_target(self, target_name: str) -> int:
        """
        Clear cache files for a specific target.

        Args:
            target_name: Name of target to clear

        Returns:
            Number of files deleted
        """
        if not self.enabled or not self.cache_dir.exists():
            return 0

        deleted = 0
        for f in self.cache_dir.glob(f"{target_name}_*"):
            try:
                f.unlink()
                deleted += 1
            except Exception as e:
                logger.warning(f"Failed to delete {f}: {e}")

        logger.debug(f"Cleared {deleted} cache files for '{target_name}'")
        return deleted

    def clear_all(self) -> int:
        """
        Clear all cache files.

        Returns:
            Number of files deleted
        """
        if not self.enabled or not self.cache_dir.exists():
            return 0

        deleted = 0
        for f in self.cache_dir.glob("*"):
            if f.is_file():
                try:
                    f.unlink()
                    deleted += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {f}: {e}")

        logger.info(f"Cleared {deleted} cache files")
        return deleted
