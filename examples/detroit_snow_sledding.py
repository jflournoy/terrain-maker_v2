#!/usr/bin/env python3
"""
Detroit Snow & Sledding Analysis Example.

This example demonstrates integrating terrain elevation data with snow analysis
to identify optimal sledding locations. It produces visual outputs at multiple
stages of the pipeline:

1. DEM Visualization - Shows the Detroit elevation model
2. Snow Depth - Displays snow depth distribution
3. Sledding Score - Highlights areas with best sledding potential
4. 3D Render - Creates 3D terrain mesh with snow overlay (requires Blender)

Usage:
    # Run specific step with mock data (fast)
    python examples/detroit_snow_sledding.py --step dem --mock-data

    # Run all steps with real data
    python examples/detroit_snow_sledding.py --all-steps

    # Specify output directory
    python examples/detroit_snow_sledding.py --output-dir ./outputs --step score

    # Get help
    python examples/detroit_snow_sledding.py --help
"""

import sys
import argparse
import logging
import hashlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import tempfile
import tarfile
import rasterio

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.terrain.core import Terrain
from src.terrain.data_loading import load_dem_files
from src.terrain.gridded_data import downsample_for_viz
from src.snow import load_snodas_stats

# Configure logging with both console and file output
LOG_FILE = Path(__file__).parent / "detroit_snow_sledding.log"

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Clear any existing handlers
logger.handlers = []

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logger.addHandler(console_handler)

# File handler (persists past crashes)
file_handler = logging.FileHandler(LOG_FILE, mode='w')  # 'w' = overwrite each run
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s"))
logger.addHandler(file_handler)

# Also capture root logger for library logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
    handlers=[file_handler]
)

# Note: downsample_for_viz imported from src.terrain.gridded_data
# Note: load_snodas_stats imported from src.snow

try:
    import bpy

    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False


# =============================================================================
# Slope Statistics Caching
# =============================================================================


def compute_cached_slope_statistics(dem, dem_transform, dem_crs, target_shape, target_transform, target_crs, cache_dir=None):
    """
    Compute slope statistics with caching based on DEM content hash.

    Args:
        dem: DEM array
        dem_transform: DEM affine transform
        dem_crs: DEM coordinate reference system
        target_shape: Target output shape
        target_transform: Target affine transform
        target_crs: Target coordinate reference system
        cache_dir: Cache directory (default: .slope_cache)

    Returns:
        Slope statistics object
    """
    if cache_dir is None:
        cache_dir = Path(".slope_cache")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Compute cache key from DEM content
    dem_hash = hashlib.sha256(dem.tobytes()).hexdigest()[:16]
    target_hash = hashlib.sha256(str((target_shape, target_crs)).encode()).hexdigest()[:16]
    cache_key = f"{dem_hash}_{target_hash}"
    cache_file = cache_dir / f"slope_stats_{cache_key}.npz"

    # Try to load from cache
    if cache_file.exists():
        logger.info(f"Loading cached slope statistics: {cache_file.name}")
        try:
            with np.load(cache_file, allow_pickle=True) as npz:
                # Reconstruct the slope stats object
                from dataclasses import dataclass

                @dataclass
                class SlopeStats:
                    slope_mean: np.ndarray
                    slope_max: np.ndarray
                    slope_min: np.ndarray
                    slope_std: np.ndarray
                    slope_p95: np.ndarray
                    roughness: np.ndarray
                    aspect_sin: np.ndarray
                    aspect_cos: np.ndarray

                    @property
                    def dominant_aspect(self) -> np.ndarray:
                        """Compute dominant aspect from vector-averaged sin/cos components."""
                        return np.degrees(np.arctan2(self.aspect_sin, self.aspect_cos)) % 360

                    @property
                    def aspect_strength(self) -> np.ndarray:
                        """Compute aspect strength (consistency of slope direction)."""
                        return np.sqrt(self.aspect_sin**2 + self.aspect_cos**2)

                stats = SlopeStats(
                    slope_mean=npz['slope_mean'],
                    slope_max=npz['slope_max'],
                    slope_min=npz['slope_min'],
                    slope_std=npz['slope_std'],
                    slope_p95=npz['slope_p95'],
                    roughness=npz['roughness'],
                    aspect_sin=npz['aspect_sin'],
                    aspect_cos=npz['aspect_cos'],
                )
                logger.info("✓ Cache hit for slope statistics")
                return stats
        except Exception as e:
            logger.warning(f"Failed to load slope stats cache: {e}")

    # Compute slope statistics
    logger.info("Computing slope statistics from DEM (tiled processing)...")
    from src.snow.slope_statistics import compute_tiled_slope_statistics

    slope_stats = compute_tiled_slope_statistics(
        dem=dem,
        dem_transform=dem_transform,
        dem_crs=dem_crs,
        target_shape=target_shape,
        target_transform=target_transform,
        target_crs=target_crs,
    )

    # Save to cache
    try:
        np.savez_compressed(
            cache_file,
            slope_mean=slope_stats.slope_mean,
            slope_max=slope_stats.slope_max,
            slope_min=slope_stats.slope_min,
            slope_std=slope_stats.slope_std,
            slope_p95=slope_stats.slope_p95,
            roughness=slope_stats.roughness,
            aspect_sin=slope_stats.aspect_sin,
            aspect_cos=slope_stats.aspect_cos,
        )
        logger.debug(f"Cached slope statistics to {cache_file.name}")
    except Exception as e:
        logger.warning(f"Failed to save slope stats cache: {e}")

    return slope_stats


# =============================================================================
# Visualization Helpers
# =============================================================================

# Note: load_snodas_stats imported from src.snow


def visualize_dem(dem: np.ndarray, output_path: Path):
    """
    Create DEM visualization.

    Args:
        dem: Digital elevation model array
        output_path: Where to save the visualization
    """
    logger.info(f"Creating DEM visualization: {output_path}")

    fig, ax = plt.subplots(figsize=(12, 9))

    # Use viridis colormap for elevation
    im = ax.imshow(dem, cmap="viridis", aspect="auto")
    ax.set_title("Detroit Digital Elevation Model", fontsize=16, fontweight="bold")
    ax.set_xlabel("Longitude (pixels)")
    ax.set_ylabel("Latitude (pixels)")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Elevation (meters)", rotation=270, labelpad=20)

    # Add statistics text
    stats_text = f"Min: {np.nanmin(dem):.1f}m\nMax: {np.nanmax(dem):.1f}m\nMean: {np.nanmean(dem):.1f}m"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ DEM visualization saved: {output_path}")


def visualize_snow_depth(snow_depth: np.ndarray, output_path: Path):
    """
    Create snow depth visualization.

    Args:
        snow_depth: Snow depth array (cm)
        output_path: Where to save the visualization
    """
    logger.info(f"Creating snow depth visualization: {output_path}")

    fig, ax = plt.subplots(figsize=(12, 9))

    # Use viridis colormap for snow depth
    # Mask zero values for better visualization
    snow_depth_masked = np.ma.masked_where(snow_depth == 0, snow_depth)

    im = ax.imshow(snow_depth_masked, cmap="viridis", aspect="auto", vmin=0, vmax=np.nanmax(snow_depth))
    ax.set_title("Snow Depth Distribution", fontsize=16, fontweight="bold")
    ax.set_xlabel("Longitude (pixels)")
    ax.set_ylabel("Latitude (pixels)")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Snow Depth (cm)", rotation=270, labelpad=20)

    # Add statistics text
    valid_depth = snow_depth[snow_depth > 0]
    if len(valid_depth) > 0:
        stats_text = (
            f"Min: {np.nanmin(valid_depth):.1f}cm\n"
            f"Max: {np.nanmax(valid_depth):.1f}cm\n"
            f"Mean: {np.nanmean(valid_depth):.1f}cm\n"
            f"Coverage: {(len(valid_depth) / snow_depth.size) * 100:.1f}%"
        )
    else:
        stats_text = "No snow detected"

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Snow depth visualization saved: {output_path}")


def visualize_sledding_score(score: np.ndarray, output_path: Path, scale: str = "linear", gamma: float = None):
    """
    Create sledding score visualization.

    Args:
        score: Sledding suitability score (0-1)
        output_path: Where to save the visualization
        scale: Scaling method - "linear", "log", "percentile", "sqrt", "power"
        gamma: Gamma value for power norm (used with scale="power")
    """
    scale_desc = f"{scale} scale" if scale != "power" else f"power scale (γ={gamma})"
    logger.info(f"Creating sledding score visualization ({scale_desc}): {output_path}")

    fig, ax = plt.subplots(figsize=(12, 9))

    # Prepare data based on scaling method
    if scale == "power":
        # Custom power norm with specified gamma
        from matplotlib.colors import PowerNorm
        display_data = score
        norm = PowerNorm(gamma=gamma, vmin=0, vmax=1)
        vmin, vmax = None, None
        cbar_label = f"Sledding Score (γ={gamma:.1f}, 0=poor, 1=excellent)"
    elif scale == "log":
        # True logarithmic scale using SymLogNorm (handles zeros gracefully)
        # Linear near zero (linthresh=0.01), logarithmic elsewhere
        from matplotlib.colors import SymLogNorm
        display_data = score
        norm = SymLogNorm(linthresh=0.01, vmin=0, vmax=1, base=10)
        vmin, vmax = None, None
        cbar_label = "Sledding Score (log scale, 0=poor, 1=excellent)"
    elif scale == "percentile":
        # Map scores to percentile ranks
        scores_flat = score.flatten()
        scores_clean = scores_flat[~np.isnan(scores_flat)]
        display_data = np.zeros_like(score)
        valid_mask = ~np.isnan(score)
        display_data[valid_mask] = (
            np.searchsorted(np.sort(scores_clean), score[valid_mask]) / len(scores_clean)
        )
        display_data[~valid_mask] = np.nan
        norm = None
        vmin, vmax = 0, 1
        cbar_label = "Percentile Rank (0=worst, 1=best)"
    elif scale == "sqrt":
        # Square root scale (spreads out low values)
        # Use PowerNorm to transform colors but show original values on colorbar
        from matplotlib.colors import PowerNorm
        display_data = score
        norm = PowerNorm(gamma=0.5, vmin=0, vmax=1)  # gamma=0.5 is sqrt
        vmin, vmax = None, None
        cbar_label = "Sledding Score (sqrt scale, 0=poor, 1=excellent)"
    else:  # linear
        display_data = score
        norm = None
        vmin, vmax = 0, 1
        cbar_label = "Sledding Score (0=poor, 1=excellent)"

    # Use viridis colormap (yellow = good sledding, purple = poor)
    if norm:
        im = ax.imshow(display_data, cmap="viridis", aspect="auto", norm=norm)
    else:
        im = ax.imshow(display_data, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)

    # Set title based on scale type
    if scale == "power":
        title = f"Sledding Suitability Score (Power Scale, γ={gamma:.1f})"
    else:
        title = f"Sledding Suitability Score ({scale.capitalize()} Scale)"
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Longitude (pixels)")
    ax.set_ylabel("Latitude (pixels)")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, rotation=270, labelpad=20)

    # Add statistics and interpretation text
    stats_text = (
        f"Min: {np.nanmin(score):.2f}\n"
        f"Max: {np.nanmax(score):.2f}\n"
        f"Mean: {np.nanmean(score):.2f}\n"
        f"\n"
        f"Yellow: Great sledding\n"
        f"Green: Moderate\n"
        f"Purple: Poor"
    )

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Sledding score visualization saved: {output_path}")


def save_sledding_score_histogram(score: np.ndarray, output_path: Path):
    """
    Create histogram of sledding score distribution.

    Args:
        score: Sledding suitability score (0-1)
        output_path: Where to save the histogram
    """
    logger.info(f"Creating sledding score histogram: {output_path}")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Flatten and remove NaNs
    scores_flat = score.flatten()
    scores_clean = scores_flat[~np.isnan(scores_flat)]

    # Create histogram with percentile lines
    ax.hist(scores_clean, bins=50, color="steelblue", alpha=0.7, edgecolor="black")

    # Calculate percentiles
    p25 = np.percentile(scores_clean, 25)
    p50 = np.percentile(scores_clean, 50)
    p75 = np.percentile(scores_clean, 75)

    # Add percentile lines
    ax.axvline(p25, color="orange", linestyle="--", linewidth=2, label=f"25th percentile: {p25:.3f}")
    ax.axvline(p50, color="red", linestyle="--", linewidth=2, label=f"50th percentile: {p50:.3f}")
    ax.axvline(p75, color="green", linestyle="--", linewidth=2, label=f"75th percentile: {p75:.3f}")

    ax.set_xlabel("Sledding Score", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency (pixels)", fontsize=12, fontweight="bold")
    ax.set_title("Sledding Score Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Sledding score histogram saved: {output_path}")


def save_sledding_score_percentiles(score: np.ndarray, output_path: Path):
    """
    Create percentile map showing where each location falls in the score distribution.

    Args:
        score: Sledding suitability score (0-1)
        output_path: Where to save the percentile map
    """
    logger.info(f"Creating sledding score percentile map: {output_path}")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Flatten to get percentiles
    scores_flat = score.flatten()
    scores_clean = scores_flat[~np.isnan(scores_flat)]

    # Create percentile ranks (0-100)
    percentile_map = np.zeros_like(score)
    valid_mask = ~np.isnan(score)
    percentile_map[valid_mask] = (
        np.searchsorted(np.sort(scores_clean), score[valid_mask]) / len(scores_clean) * 100
    )
    percentile_map[~valid_mask] = np.nan

    # Use turbo colormap for percentiles (0=red, 100=blue)
    im = ax.imshow(percentile_map, cmap="turbo", aspect="equal", interpolation="nearest", vmin=0, vmax=100)
    ax.set_title("Sledding Score Percentiles", fontweight="bold", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add colorbar with percentile labels
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, ticks=[0, 25, 50, 75, 100])
    cbar.ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    cbar.set_label("Percentile Rank", rotation=270, labelpad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Sledding score percentile map saved: {output_path}")


def save_sledding_score_filtered(score: np.ndarray, output_path: Path, threshold: float = 0.7):
    """
    Create filtered map showing only excellent sledding locations (score > threshold).

    Args:
        score: Sledding suitability score (0-1)
        output_path: Where to save the filtered map
        threshold: Score threshold for excellent locations (default: 0.7)
    """
    logger.info(f"Creating filtered sledding score map (threshold > {threshold}): {output_path}")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create filtered score: excellent locations retain value, others become 0
    excellent_spots = score > threshold
    filtered_score = np.where(excellent_spots, score, 0)

    # Count excellent pixels
    excellent_count = np.sum(excellent_spots)
    total_pixels = np.sum(~np.isnan(score))
    excellent_pct = (excellent_count / total_pixels * 100) if total_pixels > 0 else 0

    # Use RdYlGn colormap with vmin/vmax to highlight only the excellent range (0.7-1.0)
    im = ax.imshow(filtered_score, cmap="RdYlGn", aspect="equal", interpolation="nearest", vmin=0.7, vmax=1.0)
    ax.set_title(f"Excellent Sledding Locations (Score > {threshold})", fontweight="bold", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Sledding Score", rotation=270, labelpad=20)

    # Add annotation showing how many excellent pixels found
    ax.text(
        0.5, -0.12,
        f"{excellent_count:,} excellent pixels ({excellent_pct:.1f}% of area)",
        transform=ax.transAxes,
        ha='center',
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Sledding score filtered map saved: {output_path}")


def save_slope_stat_panels(slope_stats, output_dir: Path):
    """
    Save individual slope statistics panels to separate PNG files.

    Args:
        slope_stats: SlopeStatistics object from compute_tiled_slope_statistics()
        output_dir: Directory to save individual panel PNG files
    """
    logger.info(f"Saving individual slope statistics panels to {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Panel 1: Slope Mean
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(slope_stats.slope_mean, cmap="viridis", aspect="equal", interpolation="nearest")
    ax.set_title("Slope Mean (°)", fontweight="bold", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"{np.nanmin(slope_stats.slope_mean):.1f}-{np.nanmax(slope_stats.slope_mean):.1f}°", rotation=270, labelpad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "mean.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Saved: {output_dir / 'mean.png'}")

    # Panel 2: Slope Max
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(slope_stats.slope_max, cmap="inferno", aspect="equal", interpolation="nearest")
    ax.set_title("Slope Max (°) - Cliff Detection", fontweight="bold", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"{np.nanmin(slope_stats.slope_max):.1f}-{np.nanmax(slope_stats.slope_max):.1f}°", rotation=270, labelpad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "max.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Saved: {output_dir / 'max.png'}")

    # Panel 3: Slope Min
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(slope_stats.slope_min, cmap="cividis", aspect="equal", interpolation="nearest")
    ax.set_title("Slope Min (°) - Flat Spots", fontweight="bold", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"{np.nanmin(slope_stats.slope_min):.1f}-{np.nanmax(slope_stats.slope_min):.1f}°", rotation=270, labelpad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "min.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Saved: {output_dir / 'min.png'}")

    # Panel 4: Slope Std Dev
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(slope_stats.slope_std, cmap="magma", aspect="equal", interpolation="nearest")
    ax.set_title("Slope Std Dev (°) - Terrain Consistency", fontweight="bold", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"{np.nanmin(slope_stats.slope_std):.1f}-{np.nanmax(slope_stats.slope_std):.1f}°", rotation=270, labelpad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "std.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Saved: {output_dir / 'std.png'}")

    # Panel 5: Slope P95
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(slope_stats.slope_p95, cmap="plasma", aspect="equal", interpolation="nearest")
    ax.set_title("Slope P95 (°) - 95th Percentile", fontweight="bold", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"{np.nanmin(slope_stats.slope_p95):.1f}-{np.nanmax(slope_stats.slope_p95):.1f}°", rotation=270, labelpad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "p95.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Saved: {output_dir / 'p95.png'}")

    # Panel 6: Roughness
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(slope_stats.roughness, cmap="inferno", aspect="equal", interpolation="nearest")
    ax.set_title("Roughness (m elev std)", fontweight="bold", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"{np.nanmin(slope_stats.roughness):.1f}-{np.nanmax(slope_stats.roughness):.1f}m", rotation=270, labelpad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "roughness.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Saved: {output_dir / 'roughness.png'}")

    # Panel 7: Aspect (dominant slope direction)
    from matplotlib.colors import Normalize
    fig, ax = plt.subplots(figsize=(10, 8))
    aspect_deg = slope_stats.dominant_aspect

    # Use twilight colormap with 180° offset: North (0°) = light, South (180°) = dark
    # Add 180° to flip the coordinate system so North appears light and South appears dark
    norm = Normalize(vmin=0, vmax=360)
    aspect_colors = plt.cm.twilight(norm((aspect_deg + 180) % 360))

    # Fade to gray for low slopes
    slope_fade_threshold = 3.0
    slope_fade = np.clip(slope_stats.slope_mean / slope_fade_threshold, 0, 1)
    gray = 0.5
    for i in range(3):
        aspect_colors[:, :, i] = gray + slope_fade * (aspect_colors[:, :, i] - gray)

    im = ax.imshow(aspect_colors, aspect="equal", interpolation="nearest")
    ax.set_title("Dominant Aspect (°) - N=light, S=dark", fontweight="bold", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.twilight, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, ticks=[0, 90, 180, 270, 360])
    cbar.ax.set_yticklabels(['N', 'E', 'S', 'W', 'N'])
    plt.tight_layout()
    plt.savefig(output_dir / "aspect.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Saved: {output_dir / 'aspect.png'}")

    # Panel 8: Aspect Strength
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(slope_stats.aspect_strength, cmap="viridis", aspect="equal", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("Aspect Strength (0=varied, 1=uniform)", fontweight="bold", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_dir / "aspect_strength.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Saved: {output_dir / 'aspect_strength.png'}")


def save_slope_penalty_panels(slope_stats, output_dir: Path):
    """
    Save individual slope penalty panels to separate PNG files.

    Args:
        slope_stats: SlopeStatistics object
        output_dir: Directory to save individual panel PNG files
    """
    from src.scoring import trapezoidal, dealbreaker, terrain_consistency

    logger.info(f"Saving individual slope penalty panels to {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute base trapezoidal score
    slope_score = trapezoidal(
        slope_stats.slope_mean,
        sweet_range=(5, 15),
        ramp_range=(3, 25),
    )

    # Compute cliff penalty
    cliff_penalty = dealbreaker(
        slope_stats.slope_p95,
        threshold=25,
        falloff=10,
    )
    cliff_mask = slope_stats.slope_p95 > 25

    # Compute terrain consistency
    consistency_penalty = terrain_consistency(
        slope_stats.roughness,
        slope_stats.slope_std,
        roughness_threshold=30,
        slope_std_threshold=10,
    )
    rough_mask = (slope_stats.roughness > 15) | (slope_stats.slope_std > 5)

    # Final slope score
    final_score = slope_score * cliff_penalty * consistency_penalty
    score_reduction = slope_score - final_score

    # Panel 1: Base score
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(slope_score, cmap="viridis", aspect="equal", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("Base Slope Score", fontweight="bold", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_dir / "base_score.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Saved: {output_dir / 'base_score.png'}")

    # Panel 2: Cliff penalty
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cliff_penalty, cmap="plasma", aspect="equal", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("Cliff Penalty (p95 > 25°)", fontweight="bold", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8)
    cliff_pixels = np.sum(cliff_mask)
    ax.text(0.5, -0.15, f"{cliff_pixels} pixels penalized",
            transform=ax.transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7))
    plt.tight_layout()
    plt.savefig(output_dir / "cliff_penalty.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Saved: {output_dir / 'cliff_penalty.png'}")

    # Panel 3: Terrain consistency
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(consistency_penalty, cmap="plasma", aspect="equal", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("Terrain Consistency", fontweight="bold", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8)
    rough_pixels = np.sum(rough_mask)
    ax.text(0.5, -0.15, f"{rough_pixels} pixels with variation",
            transform=ax.transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7))
    plt.tight_layout()
    plt.savefig(output_dir / "terrain_consistency.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Saved: {output_dir / 'terrain_consistency.png'}")

    # Panel 4: Combined penalty
    combined_penalty = cliff_penalty * consistency_penalty
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(combined_penalty, cmap="inferno", aspect="equal", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("Combined Penalties", fontweight="bold", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_dir / "combined_penalty.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Saved: {output_dir / 'combined_penalty.png'}")

    # Panel 5: Final score
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(final_score, cmap="viridis", aspect="equal", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("Final Slope Score", fontweight="bold", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_dir / "final_score.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Saved: {output_dir / 'final_score.png'}")

    # Panel 6: Score reduction
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(score_reduction, cmap="magma", aspect="equal", interpolation="nearest", vmin=0)
    ax.set_title("Score Reduction", fontweight="bold", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_dir / "score_reduction.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"✓ Saved: {output_dir / 'score_reduction.png'}")


def save_score_component_panels(
    slope_stats,
    snow_stats: dict,
    final_score: np.ndarray,
    output_dir: Path,
    scorer=None,
):
    """
    Save individual score component panels for improved sledding scoring.

    Visualizes the new scoring system components:
    - Trapezoid functions for sweet spots (snow, slope)
    - Deal breaker conditions
    - Coverage with diminishing returns
    - Synergy bonuses
    - Final multiplicative score

    Args:
        slope_stats: SlopeStatistics object
        snow_stats: Dictionary with snow statistics
        final_score: Final sledding score array
        output_dir: Directory to save individual panel PNG files
        scorer: Unused (kept for API compatibility)
    """
    from src.terrain.scoring import (
        trapezoid_score,
        sledding_deal_breakers,
        coverage_diminishing_returns,
        sledding_synergy_bonus,
    )

    logger.info(f"Saving improved sledding score component panels to {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert inputs to scoring units
    snow_depth_mm = snow_stats["median_max_depth"]
    snow_depth_inches = snow_depth_mm * 0.0393701  # mm to inches
    coverage_months = snow_stats["mean_snow_day_ratio"] * 12.0  # fraction of year to months
    slope_degrees = slope_stats.slope_mean
    roughness = slope_stats.roughness

    # Compute trapezoid scores
    snow_trapezoid = trapezoid_score(
        snow_depth_inches,
        min_value=1.0,  # Marginal on grass (~25mm)
        optimal_min=4.0,  # Good coverage (~100mm)
        optimal_max=12.0,  # Excellent range (~300mm)
        max_value=20.0,  # Too much for little kids (~500mm)
    )

    slope_trapezoid = trapezoid_score(
        slope_degrees,
        min_value=1.0,  # Allow gentler slopes
        optimal_min=6.0,
        optimal_max=12.0,
        max_value=20.0,
    )

    # Compute coverage with diminishing returns
    coverage_score = coverage_diminishing_returns(coverage_months)

    # Compute deal breakers
    is_deal_breaker = sledding_deal_breakers(
        slope=slope_degrees,
        roughness=roughness,
        coverage_months=coverage_months,
    )

    # Compute synergy bonuses
    synergy = sledding_synergy_bonus(
        slope=slope_degrees,
        snow_depth=snow_depth_inches,
        coverage_months=coverage_months,
        roughness=roughness,
    )

    # Compute base score (multiplicative)
    base_score = snow_trapezoid * slope_trapezoid * coverage_score
    base_score[is_deal_breaker] = 0.0  # Apply deal breakers

    # Save equation to markdown file
    equation_text = (
        f"# Improved Sledding Score Formula\n\n"
        f"**Method**: Trapezoid Functions + Deal Breakers + Synergy Bonuses\n\n"
        f"## Sweet Spot Scoring (Trapezoid Functions)\n\n"
        f"Each factor is scored using trapezoid functions with optimal ranges:\n\n"
        f"```\n"
        f"Snow Depth:   1\" ━━━ [4-12\"] ━━━ 20\"  (inches, 25-100-300-500mm)\n"
        f"Slope:        1° ━━━ [6-12°] ━━━ 20°  (degrees)\n"
        f"Coverage:     Diminishing returns: 1 - exp(-months/2)\n"
        f"```\n\n"
        f"## Deal Breakers (Hard Zeros)\n\n"
        f"```\n"
        f"Slope > 40°  OR  Roughness > 6m  OR  Coverage < 0.5 months  →  Score = 0\n"
        f"```\n\n"
        f"## Base Score (Multiplicative)\n\n"
        f"```\n"
        f"BASE = Snow_Trapezoid × Slope_Trapezoid × Coverage_Score\n"
        f"```\n\n"
        f"## Synergy Bonuses (Hierarchical)\n\n"
        f"```\n"
        f"1. Perfect Combo (slope + snow + 3+ mo):      +30%\n"
        f"2. Consistent (4+ mo + good slope + low var): +15%\n"
        f"3. Moderate Slope + Flat Runout:              +20%\n"
        f"4. Low Variability + Perfect Slope:           +10%\n"
        f"```\n\n"
        f"## Final Score\n\n"
        f"```\n"
        f"FINAL = BASE × SYNERGY_BONUS  (range: 0 to ~1.5)\n"
        f"```\n"
    )

    equation_path = output_dir / "equation.md"
    with open(equation_path, 'w') as f:
        f.write(equation_text)
    logger.info(f"✓ Saved: {equation_path}")

    # Helper function to save individual panels
    def save_single_panel(data, filename, title, cmap, norm=None, vmin=0, vmax=1):
        fig, ax = plt.subplots(figsize=(10, 8))
        if norm:
            im = ax.imshow(data, cmap=cmap, aspect="equal", interpolation="nearest", norm=norm)
        else:
            im = ax.imshow(data, cmap=cmap, aspect="equal", interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontweight="bold", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"✓ Saved: {output_dir / filename}")

    # Save raw inputs
    save_single_panel(snow_depth_inches, "raw_snow_depth_inches.png",
                     "Raw: Snow Depth (inches)", "viridis",
                     vmin=0, vmax=min(30, np.nanpercentile(snow_depth_inches, 99)))
    save_single_panel(slope_degrees, "raw_slope_degrees.png",
                     "Raw: Slope (degrees)", "cividis",
                     vmin=0, vmax=min(25, np.nanpercentile(slope_degrees, 99)))
    save_single_panel(coverage_months, "raw_coverage_months.png",
                     "Raw: Coverage (months)", "plasma",
                     vmin=0, vmax=6)
    save_single_panel(roughness, "raw_roughness.png",
                     "Raw: Roughness (meters)", "magma",
                     vmin=0, vmax=min(10, np.nanpercentile(roughness, 99)))

    # Save trapezoid scores
    save_single_panel(snow_trapezoid, "snow_trapezoid_score.png",
                     "Snow Trapezoid Score (1-4-12-20\")", "viridis")
    save_single_panel(slope_trapezoid, "slope_trapezoid_score.png",
                     "Slope Trapezoid Score (1-6-12-20°)", "viridis")

    # Save coverage score
    save_single_panel(coverage_score, "coverage_score.png",
                     "Coverage Score (Diminishing Returns)", "viridis")

    # Save deal breaker mask
    save_single_panel((~is_deal_breaker).astype(float), "valid_terrain_mask.png",
                     "Valid Terrain (No Deal Breakers)", "RdYlGn",
                     vmin=0, vmax=1)

    # Save synergy bonuses
    save_single_panel(synergy, "synergy_bonus.png",
                     "Synergy Bonus Multiplier", "inferno",
                     vmin=1.0, vmax=min(1.5, np.nanmax(synergy)))

    # Save base score
    save_single_panel(base_score, "base_score.png",
                     "Base Score (Before Synergy)", "viridis",
                     vmin=0, vmax=1)

    # Save final score
    save_single_panel(final_score, "final_score.png",
                     "Final Score (With Synergies)", "viridis",
                     vmin=0, vmax=min(1.5, np.nanmax(final_score)))


def visualize_sledding_factors(
    snow_stats: dict,
    sledding_score: np.ndarray,
    output_path: Path,
    slope_score: np.ndarray = None,
    slope_deg: np.ndarray = None,
    depth_score: np.ndarray = None,
    coverage_score: np.ndarray = None,
    variability_score: np.ndarray = None,
):
    """
    Create multi-panel visualization showing all factors in sledding score calculation.

    Args:
        snow_stats: Dictionary with snow statistics
        sledding_score: Final sledding score array
        output_path: Where to save the visualization
        slope_score: Slope suitability score (0-1)
        slope_deg: Raw slope in degrees
        depth_score: Pre-computed depth score from SnowAnalysis
        coverage_score: Pre-computed coverage score from SnowAnalysis
        variability_score: Pre-computed variability/consistency score from SnowAnalysis
    """
    logger.info(f"Creating sledding factors visualization: {output_path}")

    # Use pre-computed scores if provided, otherwise raise error (DRY - don't duplicate logic)
    if depth_score is None or coverage_score is None or variability_score is None:
        raise ValueError(
            "depth_score, coverage_score, and variability_score must be provided. "
            "Get these from SnowAnalysis after calling calculate_sledding_score()."
        )

    # Create 3x3 grid for 9 panels (or 2x5 for 10 if we have slope)
    has_slope = slope_score is not None and slope_deg is not None
    if has_slope:
        fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    else:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    panel_idx = 0

    # Panel: Slope (degrees) - raw
    # Use adaptive vmax based on data range (95th percentile) for better visibility
    if has_slope:
        ax = axes[panel_idx]
        actual_max = np.nanmax(slope_deg)
        slope_vmax = min(15, np.nanpercentile(slope_deg, 99))  # Cap display at 15° for visibility
        im = ax.imshow(slope_deg, cmap="cividis", aspect="equal", interpolation="nearest", vmin=0, vmax=slope_vmax)
        ax.set_title(f"Slope (actual max={actual_max:.1f}°)", fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8)
        panel_idx += 1

    # Panel: Median Max Snow Depth (raw)
    ax = axes[panel_idx]
    im = ax.imshow(snow_stats["median_max_depth"], cmap="viridis", aspect="equal", interpolation="nearest")
    ax.set_title("Median Max Snow Depth (mm)", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    panel_idx += 1

    # Panel: Snow Coverage Ratio (raw)
    ax = axes[panel_idx]
    im = ax.imshow(snow_stats["mean_snow_day_ratio"], cmap="plasma", aspect="equal", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("Snow Day Ratio (persistence)", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    panel_idx += 1

    # Panel: Interseason CV (raw)
    ax = axes[panel_idx]
    im = ax.imshow(snow_stats["interseason_cv"], cmap="magma", aspect="equal", interpolation="nearest")
    ax.set_title("Interseason CV (year-to-year)", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    panel_idx += 1

    # Panel: Intraseason CV (raw)
    ax = axes[panel_idx]
    im = ax.imshow(snow_stats["mean_intraseason_cv"], cmap="magma", aspect="equal", interpolation="nearest")
    ax.set_title("Mean Intraseason CV (within-winter)", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    panel_idx += 1

    # Panel: Slope Score (normalized, 25% weight)
    if has_slope:
        ax = axes[panel_idx]
        im = ax.imshow(slope_score, cmap="plasma", aspect="equal", interpolation="nearest", vmin=0, vmax=1)
        ax.set_title("Slope Score (25% weight)", fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8)
        panel_idx += 1

    # Panel: Depth Score (normalized, 30% weight)
    ax = axes[panel_idx]
    im = ax.imshow(depth_score, cmap="plasma", aspect="equal", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("Depth Score (30% weight)", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    panel_idx += 1

    # Panel: Coverage Score (normalized, 30% weight)
    ax = axes[panel_idx]
    im = ax.imshow(coverage_score, cmap="plasma", aspect="equal", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("Coverage Score (30% weight)", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    panel_idx += 1

    # Panel: Variability Score (normalized, 15% weight)
    ax = axes[panel_idx]
    im = ax.imshow(variability_score, cmap="plasma", aspect="equal", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("Consistency Score (15% weight)", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    panel_idx += 1

    # Panel: Final Sledding Score
    ax = axes[panel_idx]
    im = ax.imshow(sledding_score, cmap="viridis", aspect="equal", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("Final Sledding Score", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Remove axis ticks for cleaner look
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    # Add overall title with formula
    if has_slope:
        formula = "Score = 0.25×Slope + 0.30×Depth + 0.30×Coverage + 0.15×Consistency"
    else:
        formula = "Score = 0.30×Depth + 0.30×Coverage + 0.15×Consistency (no slope)"
    fig.suptitle(
        f"Sledding Score Components\n{formula}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Sledding factors visualization saved: {output_path}")


def render_3d_with_snow(terrain: Terrain, output_path: Path):
    """
    Create 3D render with snow overlay using Blender.

    Args:
        terrain: Terrain object with snow data layers
        output_path: Where to save the render
    """
    if not BLENDER_AVAILABLE:
        logger.warning("Blender not available - skipping 3D render")
        return

    logger.info(f"Creating 3D render with snow overlay: {output_path}")

    # pylint: disable=import-outside-toplevel
    from src.terrain.core import (
        clear_scene,
        position_camera_relative,
        setup_light,
        setup_render_settings,
        render_scene_to_file,
    )

    # Clear Blender scene
    clear_scene()

    # Apply transforms if not already applied
    if not terrain.data_layers["dem"].get("transformed", False):
        logger.info("Applying identity transforms to terrain for mesh creation")
        # For mock data without real transforms, just mark the original data as transformed
        terrain.data_layers["dem"]["transformed_data"] = terrain.data_layers["dem"]["data"]
        # Use crs from metadata or default to EPSG:4326
        terrain.data_layers["dem"]["transformed_crs"] = terrain.data_layers["dem"].get("crs", "EPSG:4326")
        # Mark as transformed
        terrain.data_layers["dem"]["transformed"] = True

    # Compute colors if not already computed
    if terrain.vertex_colors is None:
        logger.info("Computing vertex colors")
        terrain.compute_colors()

    # Create mesh with sledding score coloring
    mesh_obj = terrain.create_mesh(
        scale_factor=100,
        height_scale=2.0,
        center_model=True,
        boundary_extension=0.1,
    )

    # Setup camera
    position_camera_relative(
        mesh_obj,
        direction="south",
        distance=0.264,
        elevation=0.396,
    )

    # Setup lighting and rendering
    setup_light(angle=2, energy=3)
    setup_render_settings(use_gpu=True, samples=512, use_denoising=False)

    # Render
    render_scene_to_file(
        output_path=output_path,
        width=960,
        height=720,
        file_format="PNG",
        color_mode="RGBA",
        compression=90,
        save_blend_file=False,
    )

    logger.info(f"✓ 3D render saved: {output_path}")


def run_step_dem(output_dir: Path, use_mock: bool):
    """Run DEM visualization step."""
    logger.info("\n" + "=" * 70)
    logger.info("Step 1: DEM Visualization")
    logger.info("=" * 70)

    if use_mock:
        # Create mock DEM with realistic Detroit geographic extent
        logger.info("Using mock DEM data")
        from rasterio.transform import Affine

        # Detroit bounding box (approximate)
        detroit_bounds = (-83.3, 42.2, -82.9, 42.5)
        dem_height, dem_width = 500, 500
        dem = np.random.rand(dem_height, dem_width) * 100 + 150  # 150-250m elevation

        # Create proper geographic transform
        transform = Affine.translation(detroit_bounds[0], detroit_bounds[3]) * Affine.scale(
            (detroit_bounds[2] - detroit_bounds[0]) / dem_width,
            -(detroit_bounds[3] - detroit_bounds[1]) / dem_height
        )
    else:
        # Load real DEM
        dem_dir = project_root / "data" / "dem" / "detroit"
        if not dem_dir.exists():
            logger.warning(f"DEM directory not found: {dem_dir}")
            logger.info("Falling back to mock data")
            from rasterio.transform import Affine

            detroit_bounds = (-83.3, 42.2, -82.9, 42.5)
            dem_height, dem_width = 500, 500
            dem = np.random.rand(dem_height, dem_width) * 100 + 150
            transform = Affine.translation(detroit_bounds[0], detroit_bounds[3]) * Affine.scale(
                (detroit_bounds[2] - detroit_bounds[0]) / dem_width,
                -(detroit_bounds[3] - detroit_bounds[1]) / dem_height
            )
        else:
            try:
                # Load all HGT files in directory (110 files covering full extent)
                # This defines the full geographic bounds for SNODAS processing
                dem, transform = load_dem_files(dem_dir, pattern="*.hgt")
            except ValueError as e:
                logger.warning(f"Failed to load DEM files: {e}")
                logger.info("Falling back to mock data")
                from rasterio.transform import Affine

                detroit_bounds = (-83.3, 42.2, -82.9, 42.5)
                dem_height, dem_width = 500, 500
                dem = np.random.rand(dem_height, dem_width) * 100 + 150
                transform = Affine.translation(detroit_bounds[0], detroit_bounds[3]) * Affine.scale(
                    (detroit_bounds[2] - detroit_bounds[0]) / dem_width,
                    -(detroit_bounds[3] - detroit_bounds[1]) / dem_height
                )

    # Visualize (downsample if needed for memory efficiency)
    dem_viz, stride = downsample_for_viz(dem)
    if stride > 1:
        logger.info(f"Downsampling DEM for visualization: {dem.shape} -> {dem_viz.shape} (stride={stride})")
    (output_dir / "01_raw").mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "01_raw" / "dem.png"
    visualize_dem(dem_viz, output_path)

    return dem, transform


def run_step_snow(output_dir: Path, dem: np.ndarray, use_mock: bool, terrain=None, snodas_dir=None):
    """Run snow depth visualization step."""
    logger.info("\n" + "=" * 70)
    logger.info("Step 2: Snow Depth Analysis")
    logger.info("=" * 70)

    snow_stats = load_snodas_stats(
        terrain=terrain,
        snodas_dir=snodas_dir,
        cache_dir=Path("snow_analysis_cache"),
        cache_name="snodas",
        mock_data=use_mock,
        mock_shape=dem.shape,
    )

    # Visualize snow depth (downsample if needed for memory efficiency)
    snow_data = snow_stats["median_max_depth"]
    snow_viz, stride = downsample_for_viz(snow_data)
    if stride > 1:
        logger.info(f"Downsampling snow data for visualization: {snow_data.shape} -> {snow_viz.shape} (stride={stride})")
    (output_dir / "01_raw").mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "01_raw" / "snow_depth.png"
    visualize_snow_depth(snow_viz, output_path)

    return snow_stats


def run_step_score(output_dir: Path, dem: np.ndarray, snow_stats: dict, transform=None):
    """Run sledding score calculation and visualization step."""
    logger.info("\n" + "=" * 70)
    logger.info("Step 3: Sledding Score Calculation")
    logger.info("=" * 70)

    # Create Terrain object
    if transform is None:
        from rasterio.transform import Affine

        transform = Affine.identity()

    terrain = Terrain(dem, transform, dem_crs="EPSG:4326")

    # Import improved scoring system
    from src.scoring.configs.sledding import compute_improved_sledding_score

    # Calculate slope statistics from DEM (with caching)
    target_shape = snow_stats["median_max_depth"].shape
    slope_stats = compute_cached_slope_statistics(
        dem=dem,
        dem_transform=transform,
        dem_crs="EPSG:4326",
        target_shape=target_shape,
        target_transform=transform,
        target_crs="EPSG:4326",
        cache_dir=Path(".slope_cache"),
    )

    logger.info(f"Slope stats computed: mean={np.mean(slope_stats.slope_mean):.1f}°, "
               f"max={np.max(slope_stats.slope_max):.1f}°, "
               f"p95={np.mean(slope_stats.slope_p95):.1f}°")

    # Compute improved sledding score with trapezoid functions and synergies
    logger.info("Computing improved sledding score with sweet spots and synergies...")
    sledding_score = compute_improved_sledding_score(slope_stats, snow_stats)

    # Get component scores is no longer needed with the new system
    component_scores = None  # Legacy support

    # Mask invalid values
    sledding_score = np.ma.masked_invalid(sledding_score)

    logger.info("Sledding score stats:")
    logger.info(f"  Range: {np.ma.min(sledding_score):.3f} to {np.ma.max(sledding_score):.3f}")
    logger.info(f"  Mean: {np.ma.mean(sledding_score):.3f}")
    high_score_pixels = np.ma.sum(sledding_score > 0.7)
    logger.info(
        f"  Pixels > 0.7: {high_score_pixels} "
        f"({high_score_pixels / sledding_score.count() * 100:.1f}%)"
    )

    # Add sledding score to terrain
    terrain.add_data_layer(
        "sledding_score",
        sledding_score,
        transform,
        "EPSG:4326",
        target_layer="dem",
    )

    logger.info(f"Terrain layers: {list(terrain.data_layers.keys())}")

    # Save sledding score grid as .npz for use in other examples (like detroit_dual_render.py)
    # Include transform so loaders can properly georeference the data
    # Save to output_dir/sledding/ to match XC skiing pattern
    data_dir = output_dir / "sledding"
    data_dir.mkdir(parents=True, exist_ok=True)
    score_path = data_dir / "sledding_scores.npz"

    # Calculate the correct transform for the score grid shape
    # The score grid has different dimensions than the DEM, so we need to
    # compute pixel sizes based on the geographic extent and score grid shape
    from rasterio.transform import Affine
    dem_height, dem_width = dem.shape
    score_rows, score_cols = sledding_score.shape

    # Get geographic bounds from DEM transform
    minx = transform.c  # x origin (west)
    maxy = transform.f  # y origin (north)
    maxx = minx + transform.a * dem_width  # east
    miny = maxy + transform.e * dem_height  # south (transform.e is negative)

    # Create transform for score grid dimensions
    score_transform = Affine(
        (maxx - minx) / score_cols,  # pixel width
        0,
        minx,  # x origin
        0,
        -(maxy - miny) / score_rows,  # pixel height (negative for north-up)
        maxy,  # y origin
    )
    logger.info(f"Score grid transform: pixel size = {score_transform.a:.6f}° x {abs(score_transform.e):.6f}°")

    # Save transform as tuple (a, b, c, d, e, f) for Affine reconstruction
    transform_tuple = (score_transform.a, score_transform.b, score_transform.c,
                       score_transform.d, score_transform.e, score_transform.f)
    np.savez_compressed(score_path, score=sledding_score, transform=transform_tuple, crs="EPSG:4326")
    logger.info(f"✓ Saved sledding scores to {score_path} (with transform metadata)")

    # Create output directory for visualizations
    (output_dir / "05_final").mkdir(parents=True, exist_ok=True)

    # Visualize sledding score with multiple scales (downsample if needed for memory efficiency)
    score_viz, stride = downsample_for_viz(sledding_score)
    if stride > 1:
        logger.info(f"Downsampling score for visualization: {sledding_score.shape} -> {score_viz.shape} (stride={stride})")

    # Linear scale (default, baseline for comparison)
    output_path = output_dir / "05_final" / "sledding_score_linear.png"
    visualize_sledding_score(score_viz, output_path, scale="linear")

    # Percentile scale (shows relative ranking, handles discontinuity from deal breakers)
    output_path_pct = output_dir / "05_final" / "sledding_score_percentile.png"
    visualize_sledding_score(score_viz, output_path_pct, scale="percentile")

    # True logarithmic scale (SymLogNorm - handles zeros, strong compression)
    output_path_log = output_dir / "05_final" / "sledding_score_log.png"
    visualize_sledding_score(score_viz, output_path_log, scale="log")

    # Power norm scales with gamma from 0.5 to 0.9 by 0.1 increments
    # Lower gamma = more compression of low values (sqrt-like)
    # Higher gamma = closer to linear (less compression)
    for gamma in [0.5, 0.6, 0.7, 0.8, 0.9]:
        output_path_power = output_dir / "05_final" / f"sledding_score_gamma_{gamma:.1f}.png"
        visualize_sledding_score(score_viz, output_path_power, scale="power", gamma=gamma)

    # Save histogram of sledding scores
    histogram_path = output_dir / "05_final" / "sledding_score_histogram.png"
    save_sledding_score_histogram(score_viz, histogram_path)

    # Save percentile map of sledding scores
    percentile_path = output_dir / "05_final" / "sledding_score_percentiles.png"
    save_sledding_score_percentiles(score_viz, percentile_path)

    # Save filtered map showing only excellent locations (score > 0.7)
    filtered_path = output_dir / "05_final" / "sledding_score_excellent.png"
    save_sledding_score_filtered(score_viz, filtered_path, threshold=0.7)

    # Downsample snow_stats arrays to match
    if stride > 1:
        snow_stats_viz = {
            k: v[::stride, ::stride] if hasattr(v, '__getitem__') and len(v.shape) == 2 else v
            for k, v in snow_stats.items()
        }
    else:
        snow_stats_viz = snow_stats

    # Visualize slope statistics in detail
    if slope_stats is not None:
        logger.info("Generating slope statistics visualizations...")

        # Downsample slope stats for visualization
        slope_stats_viz = slope_stats
        if stride > 1:
            # Create downsampled version of slope stats
            from dataclasses import replace
            slope_stats_viz = replace(
                slope_stats_viz,
                slope_mean=slope_stats_viz.slope_mean[::stride, ::stride],
                slope_max=slope_stats_viz.slope_max[::stride, ::stride],
                slope_min=slope_stats_viz.slope_min[::stride, ::stride],
                slope_std=slope_stats_viz.slope_std[::stride, ::stride],
                slope_p95=slope_stats_viz.slope_p95[::stride, ::stride],
                roughness=slope_stats_viz.roughness[::stride, ::stride],
                aspect_sin=slope_stats_viz.aspect_sin[::stride, ::stride],
                aspect_cos=slope_stats_viz.aspect_cos[::stride, ::stride],
            )

        # Save individual slope statistics panels
        slope_stats_dir = output_dir / "02_slope_stats"
        save_slope_stat_panels(slope_stats_viz, slope_stats_dir)

        # Save individual slope penalty panels
        penalty_dir = output_dir / "03_slope_penalties"
        save_slope_penalty_panels(slope_stats_viz, penalty_dir)

        # Save individual score component panels
        component_dir = output_dir / "04_score_components"
        save_score_component_panels(
            slope_stats=slope_stats_viz,
            snow_stats=snow_stats_viz,
            final_score=score_viz,
            output_dir=component_dir,
        )
    else:
        logger.warning("No slope_stats available - skipping slope statistics visualizations")

    return terrain, sledding_score


def run_step_render(output_dir: Path, terrain: Terrain):
    """Run 3D rendering step."""
    logger.info("\n" + "=" * 70)
    logger.info("Step 4: 3D Rendering")
    logger.info("=" * 70)

    if not BLENDER_AVAILABLE:
        logger.warning("Blender not available - skipping 3D render")
        logger.info("Install Blender Python module (bpy) to enable 3D rendering")
        return

    # Set color mapping to blend sledding score
    from src.terrain.core import elevation_colormap

    def blend_elevation_and_sledding(dem_data, sledding_score_data):
        """Blend elevation colors with sledding score."""
        dem_colors = elevation_colormap(dem_data, cmap_name="viridis")

        # Use sledding score to modulate saturation
        # High score = more vibrant, low score = more muted
        dem_colors[:, :, :3] = dem_colors[:, :, :3] * (0.5 + 0.5 * sledding_score_data[:, :, np.newaxis])

        return dem_colors

    terrain.set_color_mapping(blend_elevation_and_sledding, source_layers=["dem", "sledding_score"])

    # Render
    (output_dir / "06_render").mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "06_render" / "render_3d.png"
    render_3d_with_snow(terrain, output_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Detroit Snow & Sledding Analysis Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run DEM visualization with mock data
  python examples/detroit_snow_sledding.py --step dem --mock-data

  # Run all steps with real data
  python examples/detroit_snow_sledding.py --all-steps

  # Run specific step with custom output directory
  python examples/detroit_snow_sledding.py --output-dir ./my_outputs --step score
        """,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/images"),
        help="Output directory for visualizations (default: docs/images/)",
    )

    parser.add_argument(
        "--step",
        choices=["dem", "snow", "score", "render"],
        help="Run a specific step (dem, snow, score, render)",
    )

    parser.add_argument(
        "--all-steps", action="store_true", help="Run all visualization steps"
    )

    parser.add_argument(
        "--mock-data",
        action="store_true",
        help="Use mock data instead of loading real DEM/SNODAS data",
    )

    parser.add_argument(
        "--snodas-dir",
        type=Path,
        help="Path to SNODAS data directory (for real snow data)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.step and not args.all_steps:
        parser.error("Must specify either --step or --all-steps")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "=" * 70)
    logger.info("Detroit Snow & Sledding Analysis")
    logger.info("=" * 70)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Using mock data: {args.mock_data}")

    # State variables for multi-step execution
    dem = None
    transform = None
    snow_stats = None
    terrain = None

    # Run requested steps
    steps_to_run = []
    if args.all_steps:
        # All steps = visualization steps only (not 3D render)
        # Use --step render separately for 3D rendering
        steps_to_run = ["dem", "snow", "score"]
    else:
        steps_to_run = [args.step]

    for step in steps_to_run:
        if step == "dem":
            dem, transform = run_step_dem(args.output_dir, args.mock_data)

        elif step == "snow":
            if dem is None:
                dem, transform = run_step_dem(args.output_dir, args.mock_data)
            # Create terrain early for SNODAS processing
            if terrain is None and not args.mock_data:
                from rasterio.transform import Affine
                if transform is None:
                    transform = Affine.identity()
                terrain = Terrain(dem, transform, dem_crs="EPSG:4326")
            snow_stats = run_step_snow(args.output_dir, dem, args.mock_data, terrain, args.snodas_dir)

        elif step == "score":
            if dem is None:
                dem, transform = run_step_dem(args.output_dir, args.mock_data)
            if snow_stats is None:
                # Create terrain early for SNODAS processing
                if terrain is None and not args.mock_data:
                    from rasterio.transform import Affine
                    if transform is None:
                        transform = Affine.identity()
                    terrain = Terrain(dem, transform, dem_crs="EPSG:4326")
                snow_stats = run_step_snow(args.output_dir, dem, args.mock_data, terrain, args.snodas_dir)
            terrain, _ = run_step_score(args.output_dir, dem, snow_stats, transform)

        elif step == "render":
            if dem is None:
                dem, transform = run_step_dem(args.output_dir, args.mock_data)
            if snow_stats is None:
                # Create terrain early for SNODAS processing
                if terrain is None and not args.mock_data:
                    from rasterio.transform import Affine
                    if transform is None:
                        transform = Affine.identity()
                    terrain = Terrain(dem, transform, dem_crs="EPSG:4326")
                snow_stats = run_step_snow(args.output_dir, dem, args.mock_data, terrain, args.snodas_dir)
            # Ensure sledding_score layer exists (required for 3D rendering)
            if terrain is None or "sledding_score" not in terrain.data_layers:
                terrain, _ = run_step_score(args.output_dir, dem, snow_stats, transform)
            run_step_render(args.output_dir, terrain)

    logger.info("\n" + "=" * 70)
    logger.info("✓ Analysis complete!")
    logger.info(f"Outputs saved to: {args.output_dir}")
    logger.info("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\n[✗] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n[✗] Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
