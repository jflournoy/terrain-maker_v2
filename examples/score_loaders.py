"""
Shared score loading utilities for example scripts.

Provides standardized loading of pre-computed score grids, parks data,
and mock score generation. Used by detroit_dual_render.py and
detroit_combined_render.py.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from affine import Affine

from src.terrain.data_loading import load_score_grid

logger = logging.getLogger(__name__)


def load_sledding_scores(output_dir: Path) -> Tuple[Optional[np.ndarray], Optional[Affine]]:
    """Load sledding scores from detroit_snow_sledding.py output.

    Uses load_score_grid() for standardized NPZ loading with transform metadata.

    Args:
        output_dir: Directory containing score output files.

    Returns:
        Tuple of (score_array, transform_affine). Transform may be None if
        file was saved without transform metadata (backward compatibility).
    """
    # Check possible locations in priority order
    possible_paths = [
        output_dir / "sledding" / "sledding_scores.npz",  # New location
        output_dir / "sledding_scores.npz",  # Old flat structure
        Path("examples/output/sledding_scores.npz"),  # Legacy hardcoded location
    ]

    for score_path in possible_paths:
        if score_path.exists():
            try:
                score, score_transform = load_score_grid(
                    score_path,
                    data_keys=["score", "sledding_score", "data"]
                )
                logger.info(f"Loaded sledding scores from {score_path}")
                if score_transform:
                    logger.info(f"  Transform: origin=({score_transform.c:.4f}, {score_transform.f:.4f})")
                return score, score_transform
            except Exception as e:
                logger.warning(f"Failed to load {score_path}: {e}")
                continue

    logger.warning("Sledding scores not found, will use mock data")
    return None, None


def load_xc_skiing_scores(output_dir: Path) -> Tuple[Optional[np.ndarray], Optional[Affine]]:
    """Load XC skiing scores from detroit_xc_skiing.py output.

    Uses load_score_grid() for standardized NPZ loading with transform metadata.

    Args:
        output_dir: Directory containing score output files.

    Returns:
        Tuple of (score_array, transform_affine). Transform may be None if
        file was saved without transform metadata (backward compatibility).
    """
    score_path = output_dir / "xc_skiing_scores.npz"
    if score_path.exists():
        try:
            score, score_transform = load_score_grid(
                score_path,
                data_keys=["score", "xc_score", "data"]
            )
            logger.info(f"Loaded XC skiing scores from {score_path}")
            if score_transform:
                logger.info(f"  Transform: origin=({score_transform.c:.4f}, {score_transform.f:.4f})")
            return score, score_transform
        except Exception as e:
            logger.warning(f"Failed to load {score_path}: {e}")

    logger.warning("XC skiing scores not found, will use mock data")
    return None, None


def load_xc_skiing_parks(output_dir: Path) -> Optional[list[dict]]:
    """Load scored parks from detroit_xc_skiing.py output.

    Args:
        output_dir: Directory containing xc_skiing_parks.json.

    Returns:
        List of park dicts with name/lat/lon/score, or None if not found.
    """
    parks_path = output_dir / "xc_skiing_parks.json"
    if parks_path.exists():
        logger.info(f"Loading parks from {parks_path}")
        with open(parks_path, 'r') as f:
            return json.load(f)

    logger.warning("Parks not found")
    return None


def load_xc_skiing_components(output_dir: Path) -> Optional[dict[str, np.ndarray]]:
    """Load individual XC skiing raw input grids from .npz output.

    Raw input grids are saved with 'raw_' prefix by detroit_xc_skiing.py.
    These are the pre-transform values (snow depth in mm, coverage ratio,
    consistency score) -- not the transformed [0,1] component scores.

    Falls back to transformed component scores (component_ prefix) if raw
    inputs are not available (older .npz files).

    Args:
        output_dir: Directory containing xc_skiing_scores.npz

    Returns:
        Dict mapping component names to raw input arrays, or None if not available.
        Keys: "snow_depth", "snow_coverage", "snow_consistency"
    """
    score_path = output_dir / "xc_skiing_scores.npz"
    if not score_path.exists():
        logger.warning("XC skiing scores file not found: %s", score_path)
        return None

    data = np.load(score_path)
    components = {}

    # Prefer raw inputs (pre-transform values)
    for key in ["snow_depth", "snow_coverage", "snow_consistency"]:
        raw_key = f"raw_{key}"
        if raw_key in data:
            components[key] = data[raw_key]

    if components:
        logger.info(
            "Loaded %d raw input grids from %s", len(components), score_path
        )
        return components

    # Fallback: use transformed component scores from older .npz files
    for key in ["snow_depth", "snow_coverage", "snow_consistency"]:
        npz_key = f"component_{key}"
        if npz_key in data:
            components[key] = data[npz_key]

    if not components:
        logger.warning(
            "No component scores found in %s. "
            "Re-run detroit_xc_skiing.py to generate component data.",
            score_path,
        )
        return None

    logger.warning(
        "Loaded %d transformed component scores (raw inputs not available). "
        "Re-run detroit_xc_skiing.py to generate raw input data.",
        len(components),
    )
    return components


def generate_mock_scores(shape: Tuple[int, int]) -> np.ndarray:
    """Generate mock score grid for testing.

    Args:
        shape: (height, width) tuple for the output array.

    Returns:
        Random float32 array with values in [0.3, 0.9].
    """
    np.random.seed(42)
    score = np.random.uniform(0.3, 0.9, shape).astype(np.float32)
    return score
