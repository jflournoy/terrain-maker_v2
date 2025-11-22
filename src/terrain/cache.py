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
        self,
        directory_path: str,
        pattern: str,
        recursive: bool = False
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
        self,
        dem_array: np.ndarray,
        transform: Affine,
        source_hash: str,
        cache_name: str = "dem"
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
        transform_list = [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f]
        np.savez_compressed(
            cache_path,
            dem=dem_array,
            transform_data=np.array(transform_list, dtype=np.float64)
        )

        # Save metadata
        metadata = {
            "source_hash": source_hash,
            "dem_shape": dem_array.shape,
            "dem_dtype": str(dem_array.dtype),
            "dem_min": float(np.nanmin(dem_array)),
            "dem_max": float(np.nanmax(dem_array)),
            "cache_time": time.time(),
            "transform": transform_list
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        elapsed = time.time() - start_time
        logger.info(f"Cached DEM to {cache_path.name} ({elapsed:.2f}s)")
        logger.debug(f"Cache size: {cache_path.stat().st_size / (1024*1024):.1f} MB")

        return cache_path, metadata_path

    def load_cache(
        self,
        source_hash: str,
        cache_name: str = "dem"
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
            "files": []
        }

        if not self.cache_dir.exists():
            return stats

        for cache_file in self.cache_dir.glob("*"):
            if cache_file.is_file():
                size_bytes = cache_file.stat().st_size
                stats["cache_files"] += 1
                stats["total_size_mb"] += size_bytes / (1024 * 1024)
                stats["files"].append({
                    "name": cache_file.name,
                    "size_mb": size_bytes / (1024 * 1024),
                    "mtime": cache_file.stat().st_mtime
                })

        return stats
