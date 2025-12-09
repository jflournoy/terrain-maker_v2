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
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.terrain.core import Terrain
from src.terrain.data_loading import load_dem_files
from src.snow.analysis import SnowAnalysis

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


def downsample_for_viz(arr: np.ndarray, max_dim: int = 2000) -> tuple[np.ndarray, int]:
    """
    Downsample array using stride slicing for cheap visualization.

    Args:
        arr: Input array to downsample
        max_dim: Maximum dimension size for output

    Returns:
        Tuple of (downsampled_array, stride_used)
    """
    max_shape = max(arr.shape)
    if max_shape <= max_dim:
        return arr, 1

    stride = max(1, max_shape // max_dim)
    downsampled = arr[::stride, ::stride]
    return downsampled, stride

try:
    import bpy

    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False


def create_mock_snow_data(shape: tuple) -> dict:
    """
    Create mock snow data for testing.

    Args:
        shape: Shape of the snow data arrays

    Returns:
        Dictionary with mock snow statistics
    """
    logger.info("Creating mock snow data for testing...")

    # Create realistic-looking mock data
    np.random.seed(42)

    # Snow depth (0-300cm, concentrated in certain areas)
    median_max_depth = np.random.gamma(2, 30, shape).astype(np.float32)
    median_max_depth = np.clip(median_max_depth, 0, 300)

    # Snow coverage ratio (0-1, mostly high in winter)
    mean_snow_day_ratio = np.random.beta(8, 2, shape).astype(np.float32)

    # Variability (0-1, lower is more consistent)
    interseason_cv = np.random.beta(2, 8, shape).astype(np.float32) * 0.5
    mean_intraseason_cv = np.random.beta(2, 8, shape).astype(np.float32) * 0.3

    return {
        "median_max_depth": median_max_depth,
        "mean_snow_day_ratio": mean_snow_day_ratio,
        "interseason_cv": interseason_cv,
        "mean_intraseason_cv": mean_intraseason_cv,
    }


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


def visualize_sledding_score(score: np.ndarray, output_path: Path):
    """
    Create sledding score visualization.

    Args:
        score: Sledding suitability score (0-1)
        output_path: Where to save the visualization
    """
    logger.info(f"Creating sledding score visualization: {output_path}")

    fig, ax = plt.subplots(figsize=(12, 9))

    # Use viridis colormap (yellow = good sledding, purple = poor)
    im = ax.imshow(score, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_title("Sledding Suitability Score", fontsize=16, fontweight="bold")
    ax.set_xlabel("Longitude (pixels)")
    ax.set_ylabel("Latitude (pixels)")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Sledding Score (0=poor, 1=excellent)", rotation=270, labelpad=20)

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


def visualize_slope_statistics(
    slope_stats,
    output_path: Path,
):
    """
    Create multi-panel visualization of all slope statistics.

    Shows: mean, max, min, std, p95, roughness, aspect, and derived penalties.

    Args:
        slope_stats: SlopeStatistics object from compute_tiled_slope_statistics()
        output_path: Where to save the visualization
    """
    logger.info(f"Creating slope statistics visualization: {output_path}")

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    panel_idx = 0

    # Panel 1: Slope Mean
    ax = axes[panel_idx]
    im = ax.imshow(slope_stats.slope_mean, cmap="viridis", aspect="equal", interpolation="nearest")
    ax.set_title(f"Slope Mean (°)\nrange: {np.nanmin(slope_stats.slope_mean):.1f}-{np.nanmax(slope_stats.slope_mean):.1f}°", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    panel_idx += 1

    # Panel 2: Slope Max (cliff detection)
    ax = axes[panel_idx]
    im = ax.imshow(slope_stats.slope_max, cmap="inferno", aspect="equal", interpolation="nearest")
    ax.set_title(f"Slope Max (°)\nrange: {np.nanmin(slope_stats.slope_max):.1f}-{np.nanmax(slope_stats.slope_max):.1f}°", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    panel_idx += 1

    # Panel 3: Slope Min
    ax = axes[panel_idx]
    im = ax.imshow(slope_stats.slope_min, cmap="cividis", aspect="equal", interpolation="nearest")
    ax.set_title(f"Slope Min (°)\nrange: {np.nanmin(slope_stats.slope_min):.1f}-{np.nanmax(slope_stats.slope_min):.1f}°", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    panel_idx += 1

    # Panel 4: Slope Std Dev
    ax = axes[panel_idx]
    im = ax.imshow(slope_stats.slope_std, cmap="magma", aspect="equal", interpolation="nearest")
    ax.set_title(f"Slope Std Dev (°)\nrange: {np.nanmin(slope_stats.slope_std):.1f}-{np.nanmax(slope_stats.slope_std):.1f}°", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    panel_idx += 1

    # Panel 5: Slope P95
    ax = axes[panel_idx]
    im = ax.imshow(slope_stats.slope_p95, cmap="plasma", aspect="equal", interpolation="nearest")
    ax.set_title(f"Slope P95 (°)\nrange: {np.nanmin(slope_stats.slope_p95):.1f}-{np.nanmax(slope_stats.slope_p95):.1f}°", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    panel_idx += 1

    # Panel 6: Roughness
    ax = axes[panel_idx]
    im = ax.imshow(slope_stats.roughness, cmap="inferno", aspect="equal", interpolation="nearest")
    ax.set_title(f"Roughness (m elev std)\nrange: {np.nanmin(slope_stats.roughness):.1f}-{np.nanmax(slope_stats.roughness):.1f}m", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    panel_idx += 1

    # Panel 7: Aspect (dominant slope direction)
    # Custom visualization: fade to gray for low slopes (aspect doesn't matter when flat)
    ax = axes[panel_idx]
    aspect_deg = slope_stats.dominant_aspect

    # Create a shifted twilight colormap so North (0°) maps to dark center
    # twilight: dark at 0.5, light at 0 and 1
    # We shift the colormap by 0.5 so that input 0 → twilight(0.5) = dark
    from matplotlib.colors import Normalize, ListedColormap
    base_cmap = plt.cm.twilight
    n_colors = 256
    shifted_colors = [base_cmap((i / n_colors + 0.5) % 1.0) for i in range(n_colors)]
    twilight_north_dark = ListedColormap(shifted_colors)

    # Use original aspect values (N=0, E=90, S=180, W=270)
    norm = Normalize(vmin=0, vmax=360)
    aspect_colors = twilight_north_dark(norm(aspect_deg))

    # Fade to gray for low slopes (below ~3°, aspect is meaningless)
    slope_fade_threshold = 3.0  # degrees
    slope_fade = np.clip(slope_stats.slope_mean / slope_fade_threshold, 0, 1)

    # Blend with neutral gray based on slope
    gray = 0.5
    for i in range(3):  # RGB channels only
        aspect_colors[:, :, i] = gray + slope_fade * (aspect_colors[:, :, i] - gray)

    im = ax.imshow(aspect_colors, aspect="equal", interpolation="nearest")
    ax.set_title(f"Dominant Aspect (°)\nN=dark, S=light | Faded where flat", fontweight="bold")

    # Colorbar with standard compass labels (N=0, E=90, S=180, W=270)
    sm = plt.cm.ScalarMappable(cmap=twilight_north_dark, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, ticks=[0, 90, 180, 270, 360])
    cbar.ax.set_yticklabels(['N', 'E', 'S', 'W', 'N'])
    panel_idx += 1

    # Panel 8: Aspect Strength (consistency)
    ax = axes[panel_idx]
    strength = slope_stats.aspect_strength
    im = ax.imshow(strength, cmap="viridis", aspect="equal", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title(f"Aspect Strength\n(0=varied, 1=uniform)", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Remove axis ticks for cleaner look
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "Slope Statistics: Full-Resolution Analysis with Tiling\n"
        "Max slope detects hidden cliffs; Min slope finds flat spots; Std/P95 measure terrain consistency",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Slope statistics visualization saved: {output_path}")


def visualize_slope_penalties(
    slope_stats,
    output_path: Path,
):
    """
    Create visualization showing how penalties affect the slope score.

    Uses the scoring transforms from src.scoring to ensure consistency
    with the actual scoring calculations.

    Args:
        slope_stats: SlopeStatistics object
        output_path: Where to save the visualization
    """
    from src.scoring import trapezoidal, dealbreaker, terrain_consistency

    logger.info(f"Creating slope penalties visualization: {output_path}")

    # Compute base trapezoidal score using the scoring transform
    slope_deg = slope_stats.slope_mean
    slope_score = trapezoidal(
        slope_deg,
        sweet_range=(5, 15),
        ramp_range=(3, 25),
    )

    # Compute cliff penalty using dealbreaker on p95 (not max - max is noisy)
    cliff_penalty = dealbreaker(
        slope_stats.slope_p95,
        threshold=25,
        falloff=10,
    )
    cliff_mask = slope_stats.slope_p95 > 25

    # Compute terrain consistency (combined roughness + slope_std)
    consistency_penalty = terrain_consistency(
        slope_stats.roughness,
        slope_stats.slope_std,
        roughness_threshold=30,
        slope_std_threshold=10,
    )
    rough_mask = (slope_stats.roughness > 15) | (slope_stats.slope_std > 5)

    # Final slope score
    final_score = slope_score * cliff_penalty * consistency_penalty

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Panel 1: Base trapezoidal score
    ax = axes[0]
    im = ax.imshow(slope_score, cmap="viridis", aspect="equal", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("Base Trapezoidal Score\n(from slope mean)", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel 2: Cliff penalty (based on p95, not max - less noise sensitive)
    ax = axes[1]
    im = ax.imshow(cliff_penalty, cmap="plasma", aspect="equal", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title(f"Cliff Penalty\n(dealbreaker if p95 > 25°)", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    # Add threshold line indication
    cliff_pixels = np.sum(cliff_mask)
    ax.text(0.5, 0.05, f"{cliff_pixels} pixels penalized",
            transform=ax.transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7))

    # Panel 3: Terrain consistency (combined roughness + slope_std)
    ax = axes[2]
    im = ax.imshow(consistency_penalty, cmap="plasma", aspect="equal", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title(f"Terrain Consistency\n(RMS of roughness + slope_std)", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)
    rough_pixels = np.sum(rough_mask)
    ax.text(0.5, 0.05, f"{rough_pixels} pixels with variation",
            transform=ax.transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7))

    # Panel 4: Combined penalties
    ax = axes[3]
    combined_penalty = cliff_penalty * consistency_penalty
    im = ax.imshow(combined_penalty, cmap="inferno", aspect="equal", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("Combined Penalties\n(cliff × consistency)", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel 5: Final score
    ax = axes[4]
    im = ax.imshow(final_score, cmap="viridis", aspect="equal", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("Final Slope Score\n(base × penalties)", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel 6: Score reduction map
    ax = axes[5]
    score_reduction = slope_score - final_score
    im = ax.imshow(score_reduction, cmap="magma", aspect="equal", interpolation="nearest", vmin=0)
    ax.set_title("Score Reduction\n(points lost to penalties)", fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "Slope Score Penalties: How Terrain Hazards Reduce Sledding Suitability\n"
        "Cliffs and rough terrain reduce the score even if average slope is ideal",
        fontsize=14,
        fontweight="bold",
        y=1.00,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Slope penalties visualization saved: {output_path}")


def visualize_all_score_components(
    slope_stats,
    snow_stats: dict,
    component_scores: dict,
    final_score: np.ndarray,
    output_path: Path,
    scorer=None,
):
    """
    Comprehensive visualization showing scoring components with equation.

    Creates a 4x4 grid with larger, more readable panels:
    - Row 1: Additive components (slope, depth, coverage, consistency)
    - Row 2: More additive + multiplicative (aspect, runout, cliff, terrain)
    - Row 3: Raw inputs for context
    - Row 4: Combination steps + equation

    Args:
        slope_stats: SlopeStatistics object
        snow_stats: Dictionary with snow statistics
        component_scores: Dict from scorer.get_component_scores()
        final_score: Final sledding score array
        output_path: Where to save the visualization
        scorer: Optional ScoreCombiner to generate formula from config
    """
    from src.scoring import trapezoidal, dealbreaker, linear, snow_consistency, terrain_consistency
    from src.scoring.configs import DEFAULT_SLEDDING_SCORER
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import Normalize, ListedColormap

    logger.info(f"Creating comprehensive score components visualization: {output_path}")

    # Use provided scorer or default
    if scorer is None:
        scorer = DEFAULT_SLEDDING_SCORER

    # Compute all transformations
    slope_score = trapezoidal(slope_stats.slope_mean, sweet_range=(5, 15), ramp_range=(3, 25))
    cliff_penalty = dealbreaker(slope_stats.slope_p95, threshold=25, falloff=10)
    terrain_cons = terrain_consistency(slope_stats.roughness, slope_stats.slope_std)
    depth_score = trapezoidal(snow_stats["median_max_depth"], sweet_range=(150, 500), ramp_range=(50, 1000))
    coverage_score = linear(snow_stats["mean_snow_day_ratio"], value_range=(0, 1), power=0.5)
    # Combined snow consistency using RMS of inter/intra-season CVs
    consistency_score = snow_consistency(
        snow_stats["interseason_cv"],
        snow_stats["mean_intraseason_cv"],
    )

    # Aspect bonus: northness × strength
    northness = np.cos(np.radians(slope_stats.dominant_aspect))
    aspect_bonus_raw = northness * slope_stats.aspect_strength * 0.05
    aspect_score = linear(aspect_bonus_raw, value_range=(-0.05, 0.05))

    # Runout bonus
    runout_bonus = np.where(slope_stats.slope_min < 5, 1.0, 0.0)

    # Get weights from scorer config
    weights = {comp.name: comp.weight for comp in scorer.components if comp.role == "additive"}
    w_slope = weights.get("slope_mean", 0.30)
    w_depth = weights.get("snow_depth", 0.15)
    w_coverage = weights.get("snow_coverage", 0.25)
    w_consistency = weights.get("snow_consistency", 0.20)
    w_aspect = weights.get("aspect_bonus", 0.05)
    w_runout = weights.get("runout_bonus", 0.05)

    # Combination using config weights
    additive_sum = (
        w_slope * slope_score +
        w_depth * depth_score +
        w_coverage * coverage_score +
        w_consistency * consistency_score +
        w_aspect * aspect_score +
        w_runout * runout_bonus
    )
    multiplicative = cliff_penalty * terrain_cons
    combined_final = additive_sum * multiplicative

    # Compute percentiles for the final score
    percentiles = [25, 50, 75, 90, 95]
    pct_values = {p: np.nanpercentile(combined_final, p) for p in percentiles}

    # Create figure with 5 rows (equation row + 4 data rows)
    fig = plt.figure(figsize=(20, 28))
    gs = GridSpec(5, 4, figure=fig, height_ratios=[0.4, 1, 1, 1, 1], hspace=0.35, wspace=0.25)

    # =========================================================================
    # ROW 0: EQUATION (generated from scorer config)
    # =========================================================================
    ax_eq = fig.add_subplot(gs[0, :])
    ax_eq.axis('off')

    # Generate formula from scorer configuration
    additive_parts = []
    multiplicative_parts = []
    for comp in scorer.components:
        name = comp.name.replace("_", " ").title()
        if comp.role == "additive":
            weight_pct = int(comp.weight * 100)
            additive_parts.append(f"{comp.weight:.2f} × {name}")
        else:
            multiplicative_parts.append(name)

    additive_formula = "  +  ".join(additive_parts)
    multiplicative_formula = "  ×  ".join(multiplicative_parts)

    # Build compact formula for final line
    additive_compact = " + ".join(
        f"{comp.weight:.2f}×{comp.name.replace('_', '').title()[:6]}"
        for comp in scorer.components if comp.role == "additive"
    )
    mult_compact = " × ".join(
        comp.name.replace("_", "").title()[:8]
        for comp in scorer.components if comp.role == "multiplicative"
    )

    equation_text = (
        f"SLEDDING SCORE FORMULA (from {scorer.name})\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"ADDITIVE (weighted sum = base score):\n"
        f"    {additive_formula}\n\n"
        f"MULTIPLICATIVE (penalties applied to base score):\n"
        f"    × {multiplicative_formula}\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"FINAL = ({additive_compact}) × {mult_compact}"
    )
    ax_eq.text(0.5, 0.5, equation_text, transform=ax_eq.transAxes,
               fontsize=11, fontfamily='monospace', ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

    # =========================================================================
    # ROW 1: ADDITIVE COMPONENTS (main 4)
    # =========================================================================
    row1_data = [
        (slope_score, f"Slope Score ({int(w_slope*100)}%)\ntrapezoidal 5-15°", "viridis"),
        (depth_score, f"Depth Score ({int(w_depth*100)}%)\ntrapezoidal 150-500mm", "viridis"),
        (coverage_score, f"Coverage Score ({int(w_coverage*100)}%)\nlinear √(ratio)", "viridis"),
        (consistency_score, f"Snow Reliability ({int(w_consistency*100)}%)\nRMS(inter+intra CV)", "viridis"),
    ]

    for i, (data, title, cmap) in enumerate(row1_data):
        ax = fig.add_subplot(gs[1, i])
        im = ax.imshow(data, cmap=cmap, aspect="equal", interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(title, fontsize=11, fontweight="bold", color="darkgreen")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.8)

    # =========================================================================
    # ROW 2: ADDITIVE (remaining) + MULTIPLICATIVE
    # =========================================================================
    row2_data = [
        (aspect_score, f"Aspect Score ({int(w_aspect*100)}%)\nnorth bonus", "viridis", "darkgreen"),
        (runout_bonus, f"Runout Bonus ({int(w_runout*100)}%)\nmin slope < 5°", "viridis", "darkgreen"),
        (cliff_penalty, "Cliff Penalty (×)\ndealbreaker p95 > 25°", "plasma", "darkred"),
        (terrain_cons, "Terrain Consistency (×)\nextreme roughness only", "plasma", "darkred"),
    ]

    for i, (data, title, cmap, color) in enumerate(row2_data):
        ax = fig.add_subplot(gs[2, i])
        im = ax.imshow(data, cmap=cmap, aspect="equal", interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(title, fontsize=11, fontweight="bold", color=color)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.8)

    # =========================================================================
    # ROW 3: KEY RAW INPUTS
    # =========================================================================
    row3_data = [
        (slope_stats.slope_mean, "Raw: Slope Mean (°)", "cividis", None),
        (snow_stats["median_max_depth"], "Raw: Snow Depth (mm)", "viridis", None),
        (snow_stats["interseason_cv"], "Raw: Interseason CV\n(year-to-year)", "magma", None),
        (snow_stats["mean_intraseason_cv"], "Raw: Intraseason CV\n(within-winter)", "magma", None),
    ]

    for i, (data, title, cmap, vlim) in enumerate(row3_data):
        ax = fig.add_subplot(gs[3, i])
        vmin, vmax = vlim if vlim else (np.nanmin(data), np.nanmax(data))
        im = ax.imshow(data, cmap=cmap, aspect="equal", interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11, fontweight="bold", color="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.8)

    # =========================================================================
    # ROW 4: COMBINATION STEPS
    # =========================================================================
    from matplotlib.colors import TwoSlopeNorm

    # For final score, use TwoSlopeNorm centered at 0.7 to emphasize good scores
    # This gives more color resolution to scores > 0.7 (the excellent sledding range)
    final_norm = TwoSlopeNorm(vmin=0, vcenter=0.7, vmax=1)

    row4_data = [
        (additive_sum, "Additive Sum\n(weighted components)", "viridis", None),
        (multiplicative, "Multiplicative Product\n(cliff × terrain)", "inferno", None),
        (combined_final, "FINAL SCORE\n(additive × mult)", "viridis", final_norm),
        (additive_sum - combined_final, "Score Reduction\n(lost to penalties)", "magma", None),
    ]

    for i, (data, title, cmap, norm) in enumerate(row4_data):
        ax = fig.add_subplot(gs[4, i])
        if norm is None:
            # Default 0-1 for most panels
            if "Reduction" in title:
                im = ax.imshow(data, cmap=cmap, aspect="equal", interpolation="nearest",
                               vmin=0, vmax=max(0.01, np.nanmax(data)))
            else:
                im = ax.imshow(data, cmap=cmap, aspect="equal", interpolation="nearest",
                               vmin=0, vmax=1)
        else:
            # Use custom norm for final score
            im = ax.imshow(data, cmap=cmap, aspect="equal", interpolation="nearest", norm=norm)
        color = "darkblue" if "FINAL" in title else "black"
        ax.set_title(title, fontsize=11, fontweight="bold", color=color)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        if "FINAL" in title:
            # Add percentile tick marks on colorbar
            # Use actual percentile values as custom ticks
            pct_ticks = [pct_values[p] for p in percentiles]

            # Set colorbar ticks to show both standard values and percentiles
            all_ticks = sorted(set([0, 0.7, 1] + pct_ticks))
            cbar.set_ticks(all_ticks)
            tick_labels = []
            for t in all_ticks:
                if t == 0.7:
                    tick_labels.append("0.7 ←")
                elif t in pct_ticks:
                    p_idx = pct_ticks.index(t)
                    tick_labels.append(f"{t:.2f} (P{percentiles[p_idx]})")
                else:
                    tick_labels.append(f"{t:.1f}")
            cbar.set_ticklabels(tick_labels)
            cbar.ax.tick_params(labelsize=8)

    # Add row labels on the left side
    fig.text(0.02, 0.78, "ADDITIVE\n(+)", fontsize=12, fontweight="bold", color="darkgreen",
             rotation=90, va='center', ha='center')
    fig.text(0.02, 0.58, "ADDITIVE/MULT\n(+ / ×)", fontsize=12, fontweight="bold", color="purple",
             rotation=90, va='center', ha='center')
    fig.text(0.02, 0.38, "RAW\nINPUTS", fontsize=12, fontweight="bold", color="gray",
             rotation=90, va='center', ha='center')
    fig.text(0.02, 0.15, "FINAL\nCOMBINE", fontsize=12, fontweight="bold", color="darkblue",
             rotation=90, va='center', ha='center')

    fig.suptitle(
        "Sledding Suitability Score: Component Breakdown",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Comprehensive score components visualization saved: {output_path}")


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
    output_path = output_dir / "01_raw_dem.png"
    visualize_dem(dem_viz, output_path)

    return dem, transform


def run_step_snow(output_dir: Path, dem: np.ndarray, use_mock: bool, terrain=None, snodas_dir=None):
    """Run snow depth visualization step."""
    logger.info("\n" + "=" * 70)
    logger.info("Step 2: Snow Depth Analysis")
    logger.info("=" * 70)

    if use_mock:
        # Create mock snow data
        logger.info("Using mock snow data")
        snow_stats = create_mock_snow_data(dem.shape)
    else:
        # Try to load real SNODAS data
        if snodas_dir and snodas_dir.exists() and terrain:
            logger.info(f"Loading real SNODAS data from: {snodas_dir}")
            try:
                from src.snow.analysis import SnowAnalysis

                snow = SnowAnalysis(terrain=terrain, snodas_root_dir=snodas_dir)
                _, _, failed = snow.process_snow_data()

                # Extract stats from snow object
                snow_stats = snow.stats
                logger.info(f"✓ Loaded SNODAS data ({len(failed)} files failed)")
            except Exception as e:
                logger.warning(f"Failed to load SNODAS data: {e}")
                logger.info("Falling back to mock data")
                snow_stats = create_mock_snow_data(dem.shape)
        else:
            if not snodas_dir:
                logger.warning("No SNODAS directory specified (use --snodas-dir)")
            elif not snodas_dir.exists():
                logger.warning(f"SNODAS directory not found: {snodas_dir}")
            else:
                logger.warning("Terrain object not available for SNODAS processing")
            logger.info("Falling back to mock data")
            snow_stats = create_mock_snow_data(dem.shape)

    # Visualize snow depth (downsample if needed for memory efficiency)
    snow_data = snow_stats["median_max_depth"]
    snow_viz, stride = downsample_for_viz(snow_data)
    if stride > 1:
        logger.info(f"Downsampling snow data for visualization: {snow_data.shape} -> {snow_viz.shape} (stride={stride})")
    output_path = output_dir / "01_raw_snow_depth.png"
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

    # Create SnowAnalysis instance
    snow = SnowAnalysis(terrain=terrain)

    # Inject mock snow statistics and metadata
    snow.stats = snow_stats
    snow.metadata = {
        "transform": transform,
        "crs": "EPSG:4326",
        "data_units": "cm"
    }

    # Calculate sledding score - this adds to terrain.data_layers
    sledding_score = snow.calculate_sledding_score()

    logger.info(
        f"Sledding score range: {np.nanmin(sledding_score):.3f} - {np.nanmax(sledding_score):.3f}"
    )
    logger.info(f"Terrain layers: {list(terrain.data_layers.keys())}")

    # Visualize sledding score (downsample if needed for memory efficiency)
    score_viz, stride = downsample_for_viz(sledding_score)
    if stride > 1:
        logger.info(f"Downsampling score for visualization: {sledding_score.shape} -> {score_viz.shape} (stride={stride})")
    output_path = output_dir / "05_final_sledding.png"
    visualize_sledding_score(score_viz, output_path)

    # Visualize all factors that go into the sledding score
    # Downsample snow_stats arrays to match
    if stride > 1:
        snow_stats_viz = {
            k: v[::stride, ::stride] if hasattr(v, '__getitem__') and len(v.shape) == 2 else v
            for k, v in snow_stats.items()
        }
    else:
        snow_stats_viz = snow_stats

    # Get all computed scores from SnowAnalysis (calculated during sledding score)
    # This ensures DRY - scoring logic lives only in SnowAnalysis
    slope_score_viz = None
    slope_deg_viz = None
    depth_score_viz = None
    coverage_score_viz = None
    variability_score_viz = None

    if hasattr(snow, 'slope_score') and snow.slope_score is not None:
        slope_score_viz = snow.slope_score[::stride, ::stride] if stride > 1 else snow.slope_score
    if hasattr(snow, 'slope_deg') and snow.slope_deg is not None:
        slope_deg_viz = snow.slope_deg[::stride, ::stride] if stride > 1 else snow.slope_deg
    if hasattr(snow, 'depth_score') and snow.depth_score is not None:
        depth_score_viz = snow.depth_score[::stride, ::stride] if stride > 1 else snow.depth_score
    if hasattr(snow, 'coverage_score') and snow.coverage_score is not None:
        coverage_score_viz = snow.coverage_score[::stride, ::stride] if stride > 1 else snow.coverage_score
    if hasattr(snow, 'variability_score') and snow.variability_score is not None:
        variability_score_viz = snow.variability_score[::stride, ::stride] if stride > 1 else snow.variability_score

    # Safety check: ensure arrays match the visualization target shape
    target_viz_shape = score_viz.shape
    logger.info(f"Visualization target shape: {target_viz_shape}")

    def resize_if_needed(arr, name):
        if arr is None:
            return None
        logger.info(f"{name} viz shape: {arr.shape}")
        if arr.shape != target_viz_shape:
            logger.warning(f"Shape mismatch! Resizing {name} from {arr.shape} to {target_viz_shape}")
            from scipy.ndimage import zoom
            zoom_factors = (target_viz_shape[0] / arr.shape[0],
                          target_viz_shape[1] / arr.shape[1])
            return zoom(arr, zoom_factors, order=1)
        return arr

    # Visualize slope statistics in detail
    if hasattr(snow, 'slope_stats') and snow.slope_stats is not None:
        logger.info("Generating slope statistics visualizations...")

        # Downsample slope stats for visualization
        slope_stats_viz = snow.slope_stats
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

        # Visualize all slope statistics
        slope_stats_path = output_dir / "02_stats_slope.png"
        visualize_slope_statistics(slope_stats_viz, slope_stats_path)

        # Visualize how penalties affect the score
        penalties_path = output_dir / "03_penalty_slope.png"
        visualize_slope_penalties(slope_stats_viz, penalties_path)

        # NEW: Comprehensive raw → transformed → combined visualization
        comprehensive_path = output_dir / "04_all_components.png"
        visualize_all_score_components(
            slope_stats=slope_stats_viz,
            snow_stats=snow_stats_viz,
            component_scores=snow.component_scores if hasattr(snow, 'component_scores') else {},
            final_score=score_viz,
            output_path=comprehensive_path,
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
    output_path = output_dir / "06_render_3d.png"
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
        default=Path("examples"),
        help="Output directory for visualizations (default: examples/)",
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
