"""Fail-fast memory checks for operations that allocate several arrays the size of the input.

A full-resolution DEM can have 10^8+ cells; a distance transform or per-pixel KDTree query
on it needs tens of bytes per cell and gets the process killed by the OS with no message.
check_memory estimates the peak before the allocation and raises a clear error instead.

The budget is half the currently available RAM, or TERRAIN_MAKER_MEMORY_LIMIT_GB if set.
"""

import logging
import os

logger = logging.getLogger(__name__)

_GB = 1024**3

# Approximate peak bytes per input cell (2D), including scipy's internal buffers
EDT_DISTANCES = 24  # float64 distances + int32 feature transform
EDT_INDICES = 32  # int64 (2, H, W) indices + internal feature transform
EDT_DISTANCES_AND_INDICES = 48
KDTREE_GRID_QUERY = 48  # int64 (N, 2) grid coords, float copy, distances, indices


class ArrayTooLargeError(MemoryError):
    """An operation would need more memory than the configured budget."""


def available_memory_bytes():
    """Currently available RAM in bytes, or None if it can't be determined."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


def _limit_bytes():
    configured = os.environ.get("TERRAIN_MAKER_MEMORY_LIMIT_GB")
    if configured:
        return float(configured) * _GB
    available = available_memory_bytes()
    return None if available is None else available / 2


def check_memory(n_cells, bytes_per_cell, operation):
    """Raise ArrayTooLargeError if operation on n_cells would exceed the memory budget."""
    needed = n_cells * bytes_per_cell
    limit = _limit_bytes()
    if limit is not None and needed > limit:
        raise ArrayTooLargeError(
            f"{operation} on {n_cells:,} cells needs ~{needed / _GB:.1f} GB, over the "
            f"{limit / _GB:.1f} GB budget. Downsample first (work at flow or mesh "
            f"resolution rather than the full-resolution DEM), or raise the budget with "
            f"TERRAIN_MAKER_MEMORY_LIMIT_GB."
        )
    if needed > _GB:
        logger.info(f"{operation}: ~{needed / _GB:.1f} GB for {n_cells:,} cells")
