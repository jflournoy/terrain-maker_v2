"""
Mesh caching module for Blender .blend file reuse.

Caches mesh generation results so render-only passes don't need to regenerate
the mesh. Useful for iterating on camera angles and render settings without
waiting for mesh generation.
"""

import hashlib
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MeshCache:
    """
    Manages caching of generated Blender mesh files.

    The cache stores:
    - .blend files (Blender scene with generated mesh)
    - Metadata including generation parameters and hash

    Attributes:
        cache_dir: Directory where mesh cache is stored
        enabled: Whether caching is enabled
    """

    def __init__(self, cache_dir: Optional[Path] = None, enabled: bool = True):
        """
        Initialize mesh cache.

        Args:
            cache_dir: Directory for cache files. If None, uses .mesh_cache/ in project root
            enabled: Whether caching is enabled (default: True)
        """
        if cache_dir is None:
            cache_dir = Path.cwd() / ".mesh_cache"

        self.cache_dir = Path(cache_dir)
        self.enabled = enabled

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.debug("Mesh cache initialized at: %s", self.cache_dir)

    def compute_mesh_hash(self, dem_hash: str, mesh_params: dict) -> str:
        """
        Compute hash of mesh generation parameters.

        This ensures the cache is invalidated if mesh generation parameters change:
        - DEM data (via dem_hash)
        - scale_factor
        - height_scale
        - center_model
        - boundary_extension
        - water_mask applied

        Args:
            dem_hash: Hash of DEM data
            mesh_params: Dictionary of mesh generation parameters

        Returns:
            SHA256 hash of mesh parameters
        """
        # Create sorted representation of parameters for consistent hashing
        param_str = f"{dem_hash}:"
        for key in sorted(mesh_params.keys()):
            value = mesh_params[key]
            if isinstance(value, (np.ndarray, list)):
                # For arrays, use shape and dtype
                if isinstance(value, np.ndarray):
                    param_str += f"{key}={value.shape}_{value.dtype}:"
                else:
                    param_str += f"{key}={len(value)}:"
            else:
                param_str += f"{key}={value}:"

        hash_obj = hashlib.sha256(param_str.encode())
        return hash_obj.hexdigest()

    def get_cache_path(self, mesh_hash: str, cache_name: str = "mesh") -> Path:
        """
        Get the path for a mesh cache file.

        Args:
            mesh_hash: Hash of mesh parameters
            cache_name: Name of cache item (default: "mesh")

        Returns:
            Path to .blend file
        """
        return self.cache_dir / f"{cache_name}_{mesh_hash}.blend"

    def get_metadata_path(self, mesh_hash: str, cache_name: str = "mesh") -> Path:
        """
        Get the path for mesh metadata file.

        Args:
            mesh_hash: Hash of mesh parameters
            cache_name: Name of cache item (default: "mesh")

        Returns:
            Path to metadata file
        """
        return self.cache_dir / f"{cache_name}_{mesh_hash}_meta.json"

    def save_cache(
        self, blend_file: Path, mesh_hash: str, mesh_params: dict, cache_name: str = "mesh"
    ) -> Tuple[Path, Path]:
        """
        Cache a generated mesh by copying the .blend file.

        Args:
            blend_file: Path to source .blend file
            mesh_hash: Hash of mesh parameters
            mesh_params: Dictionary of mesh generation parameters
            cache_name: Name of cache item (default: "mesh")

        Returns:
            Tuple of (cached_blend_path, metadata_path)
        """
        if not self.enabled:
            return None, None

        if not blend_file.exists():
            logger.warning("Blend file not found: %s", blend_file)
            return None, None

        cache_path = self.get_cache_path(mesh_hash, cache_name)
        metadata_path = self.get_metadata_path(mesh_hash, cache_name)

        start_time = time.time()

        # Copy blend file to cache
        try:
            shutil.copy2(blend_file, cache_path)
        except (OSError, IOError) as e:
            logger.warning("Failed to copy blend file: %s", e)
            return None, None

        # Save metadata
        metadata = {
            "mesh_hash": mesh_hash,
            "mesh_params": mesh_params,
            "cache_time": time.time(),
            "source_file": str(blend_file),
            "cache_file": str(cache_path),
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        elapsed = time.time() - start_time
        cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
        logger.info(
            "Cached mesh to %s (%.1f MB, %.2fs)", cache_path.name, cache_size_mb, elapsed
        )

        return cache_path, metadata_path

    def load_cache(self, mesh_hash: str, cache_name: str = "mesh") -> Optional[Path]:
        """
        Load cached mesh .blend file.

        Args:
            mesh_hash: Hash of mesh parameters
            cache_name: Name of cache item (default: "mesh")

        Returns:
            Path to cached .blend file or None if not found
        """
        if not self.enabled:
            return None

        cache_path = self.get_cache_path(mesh_hash, cache_name)

        if not cache_path.exists():
            logger.debug("Mesh cache miss: %s", cache_path.name)
            return None

        try:
            start_time = time.time()
            elapsed = time.time() - start_time

            cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
            logger.info("Loaded mesh from cache (%.1f MB, %.2fs)", cache_size_mb, elapsed)
            logger.debug("Cache file: %s", cache_path.name)

            return cache_path

        except (OSError, IOError) as e:
            logger.warning("Failed to load mesh cache %s: %s", cache_path.name, e)
            return None

    def clear_cache(self, cache_name: str = "mesh") -> int:
        """
        Clear cached mesh files for a specific cache name.

        Args:
            cache_name: Name of cache item to clear (default: "mesh")

        Returns:
            Number of files deleted
        """
        if not self.enabled:
            return 0

        deleted_count = 0

        # Find .blend files matching the cache_name
        pattern = f"{cache_name}_*.blend"
        for blend_file in self.cache_dir.glob(pattern):
            try:
                blend_file.unlink()
                deleted_count += 1
                logger.debug("Deleted: %s", blend_file.name)
            except (OSError, IOError) as e:
                logger.warning("Failed to delete %s: %s", blend_file.name, e)

        # Also delete associated metadata files
        meta_pattern = f"{cache_name}_*_meta.json"
        for meta_file in self.cache_dir.glob(meta_pattern):
            try:
                meta_file.unlink()
                deleted_count += 1
                logger.debug("Deleted: %s", meta_file.name)
            except (OSError, IOError) as e:
                logger.warning("Failed to delete %s: %s", meta_file.name, e)

        logger.info("Cleared %d mesh cache files for %s", deleted_count, cache_name)
        return deleted_count

    def get_cache_stats(self) -> dict:
        """
        Get statistics about cached mesh files.

        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "cache_dir": str(self.cache_dir),
            "enabled": self.enabled,
            "blend_files": 0,
            "total_size_mb": 0,
            "files": [],
        }

        if not self.cache_dir.exists():
            return stats

        for blend_file in self.cache_dir.glob("*.blend"):
            if blend_file.is_file():
                size_bytes = blend_file.stat().st_size
                stats["blend_files"] += 1
                stats["total_size_mb"] += size_bytes / (1024 * 1024)
                stats["files"].append(
                    {
                        "name": blend_file.name,
                        "size_mb": size_bytes / (1024 * 1024),
                        "mtime": blend_file.stat().st_mtime,
                    }
                )

        return stats
