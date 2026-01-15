#!/usr/bin/env python3
"""
Detroit Combined Sledding/XC Skiing Terrain Rendering Example.

This example renders a single terrain showing sledding suitability scores with
XC skiing locations highlighted using dual colormaps in Blender.

Features:
- Single terrain with dual colormaps blending sledding and XC skiing data
- Base terrain: Boreal-Mako colormap (forest green → blue → pale mint) for sledding
- Overlay zones: Rocket colormap highlighting XC skiing scores near park clusters
- Proximity-based coloring with automatic park clustering (10km radius)
- Detects and colors water bodies blue (slope-based detection)
- Sunset-style lighting with warm/cool color contrast
- Applies geographic transforms (WGS84 → UTM) for proper coordinate handling
- Intelligently limits mesh density to prevent OOM with large DEMs

Requirements:
- Blender Python API available (bpy)
- Pre-computed sledding scores from detroit_snow_sledding.py
- Pre-computed XC skiing scores + parks from detroit_xc_skiing.py

Output:
- docs/images/combined_render/sledding_with_xc_parks_3d.png (640×360 preview)
- docs/images/combined_render/sledding_with_xc_parks_3d_histogram.png (RGB histogram)
- docs/images/combined_render/sledding_with_xc_parks_3d_luminance.png (B&W histogram)
- docs/images/combined_render/sledding_with_xc_parks_3d_print.png (print quality @ DPI)
- docs/images/combined_render/sledding_with_xc_parks_3d.blend (Blender file)

Usage:
    # Run with computed scores (standard quality)
    python examples/detroit_combined_render.py

    # Run at print quality (default: 10x8 inches @ 300 DPI)
    python examples/detroit_combined_render.py --print-quality

    # Custom print size: 12x9 inches @ 150 DPI
    python examples/detroit_combined_render.py --print-quality --print-width 12 --print-height 9 --print-dpi 150

    # Large print with GPU memory-saving options (essential for limited VRAM)
    python examples/detroit_combined_render.py --print-quality --auto-tile --tile-size 1024

    # Test different height scales from south view
    python examples/detroit_combined_render.py --camera-direction south --height-scale 20
    python examples/detroit_combined_render.py --camera-direction south --height-scale 40

    # Add a background plane (eggshell white, no shadows)
    python examples/detroit_combined_render.py --background

    # Background with custom color and shadows
    python examples/detroit_combined_render.py --background --background-color "#E8E4D9" --background-shadow

    # Run with mock data
    python examples/detroit_combined_render.py --mock-data --no-render

    # Specify output directory
    python examples/detroit_combined_render.py --output-dir ./renders

    # Two-tier edge extrusion with default clay base
    python examples/detroit_combined_render.py --two-tier-edge

    # Two-tier edge with gold base material
    python examples/detroit_combined_render.py --two-tier-edge --edge-base-material gold

    # Two-tier edge with custom RGB base color
    python examples/detroit_combined_render.py --two-tier-edge --edge-base-material "0.6,0.55,0.5"

    # Two-tier edge with sharp transition (no color blending)
    python examples/detroit_combined_render.py --two-tier-edge --edge-base-material ivory --no-edge-blend-colors
"""

import sys
import argparse
import logging
import json
import gc
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

import bpy

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.terrain.core import (
    Terrain,
    elevation_colormap,
    clear_scene,
    setup_camera,
    position_camera_relative,
    setup_two_point_lighting,
    setup_render_settings,
    render_scene_to_file,
    print_render_settings_report,
    reproject_raster,
    flip_raster,
    scale_elevation,
    feature_preserving_smooth,
    setup_hdri_lighting,
    setup_world_atmosphere,
)
from src.terrain.color_mapping import boreal_mako_cmap
from src.terrain.transforms import (
    smooth_score_data,
    despeckle_scores,
    despeckle_dem,
    wavelet_denoise_dem,
    slope_adaptive_smooth,
    remove_bumps,
    cached_reproject,
    upscale_scores,
)
from src.terrain.scene_setup import create_background_plane
from src.terrain.blender_integration import apply_vertex_colors, apply_road_mask, apply_vertex_positions
from src.terrain.materials import apply_terrain_with_obsidian_roads, apply_test_material
from src.terrain.data_loading import load_dem_files, load_filtered_hgt_files, load_score_grid
from src.terrain.gridded_data import MemoryMonitor, TiledDataConfig, MemoryLimitExceeded
from src.terrain.roads import add_roads_layer, smooth_road_vertices, offset_road_vertices, smooth_road_mask
from src.terrain.water import identify_water_by_slope
from src.terrain.cache import PipelineCache
from src.terrain.diagnostics import (
    generate_full_wavelet_diagnostics,
    generate_full_adaptive_smooth_diagnostics,
    generate_bump_removal_diagnostics,
    generate_upscale_diagnostics,
    generate_rgb_histogram,
    generate_luminance_histogram,
)
from examples.detroit_roads import get_roads_tiled
from affine import Affine
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
from PIL import Image

# Configure logging
LOG_FILE = Path(__file__).parent / "detroit_combined_render.log"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers = []

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE, mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s"))
logger.addHandler(file_handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
    handlers=[file_handler]
)


# =============================================================================
# COLOR COMPRESSION FORMULA
# =============================================================================

def compress_colormap_score(score, transition):
    """
    Apply two-segment compression formula to preserve colormap extremes.

    This formula compresses the low end of the score range while expanding
    the high end to ensure pale/white colors remain visible even with high
    transition values.

    The formula:
    - [0, transition] maps to [0, 0.27] (compress low scores around transition)
    - [transition, 1.0] maps to [0.27, 1.0] (expand high scores to show full colormap)

    Args:
        score: Numpy array or scalar of score values [0, 1]
        transition: Transition point where green→blue occurs [0, 1]

    Returns:
        Compressed score in range [0, 1]
    """
    # Handle both scalar and array inputs
    is_scalar = np.isscalar(score)
    score_arr = np.asarray(score)

    # Two-segment piecewise formula
    result = np.zeros_like(score_arr, dtype=np.float64)

    # Segment 1: [0, transition] → [0, 0.27]
    mask_low = score_arr <= transition
    result[mask_low] = score_arr[mask_low] * (0.27 / transition)

    # Segment 2: [transition, 1.0] → [0.27, 1.0]
    mask_high = score_arr > transition
    result[mask_high] = 0.27 + (score_arr[mask_high] - transition) * (1.0 - 0.27) / (1.0 - transition)

    # Clip to ensure we stay in [0, 1]
    result = np.clip(result, 0.0, 1.0)

    # Return scalar if input was scalar
    return float(result) if is_scalar else result


# =============================================================================
# CONFIGURATION
# =============================================================================

# Render dimensions for vertex count calculation
# Use conservative 960x720 (proven to work in detroit_elevation_real)
# Much safer for dual mesh rendering than full 1920x1080
RENDER_WIDTH = 1920
RENDER_HEIGHT = 1080

# =============================================================================
# LOADING OUTPUTS
# =============================================================================

def load_sledding_scores(output_dir: Path) -> Tuple[Optional[np.ndarray], Optional[Affine]]:
    """Load sledding scores from detroit_snow_sledding.py output.

    Uses load_score_grid() for standardized NPZ loading with transform metadata.

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
    """Load scored parks from detroit_xc_skiing.py output."""
    parks_path = output_dir / "xc_skiing_parks.json"
    if parks_path.exists():
        logger.info(f"Loading parks from {parks_path}")
        with open(parks_path, 'r') as f:
            return json.load(f)

    logger.warning("Parks not found")
    return None


def generate_mock_scores(shape: Tuple[int, int]) -> np.ndarray:
    """Generate mock score grid."""
    np.random.seed(42)
    score = np.random.uniform(0.3, 0.9, shape).astype(np.float32)
    return score


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Detroit Combined Terrain Rendering (Sledding with XC Skiing Parks)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Render with pre-computed scores (1920x1080, 2048 samples)
  python examples/detroit_combined_render.py

  # Render at print quality (3000x2400 @ 300 DPI, 4096 samples)
  python examples/detroit_combined_render.py --print-quality

  # Test height scales from south view (for finding best vertical exaggeration)
  python examples/detroit_combined_render.py --camera-direction south --height-scale 15
  python examples/detroit_combined_render.py --camera-direction south --height-scale 25
  python examples/detroit_combined_render.py --camera-direction south --height-scale 35

  # Add background plane (eggshell white, no shadows)
  python examples/detroit_combined_render.py --background

  # Background with custom color and drop shadows
  python examples/detroit_combined_render.py --background --background-color "#E8E4D9" --background-shadow

  # Specify output directory for results
  python examples/detroit_combined_render.py --output-dir ./renders
        """,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/images/combined_render"),
        help="Output directory (default: docs/images/combined_render/)",
    )

    parser.add_argument(
        "--scores-dir",
        type=Path,
        default=Path("docs/images"),
        help="Directory containing pre-computed scores (default: docs/images/)",
    )

    parser.add_argument(
        "--mock-data",
        action="store_true",
        help="Use mock data for all inputs (skips loading real DEM)",
    )

    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Create scene but don't render (for debugging in Blender UI)",
    )

    parser.add_argument(
        "--print-quality",
        action="store_true",
        help="Render at print quality with configurable DPI and size (default: 10x8 inches @ 300 DPI)",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "jpeg", "PNG", "JPEG"],
        default="png",
        help="Output format: png (lossless, no profile) or jpeg (lossy, better for print). Default: png",
    )

    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality 0-100 (default: 95). Only used with --format jpeg.",
    )

    parser.add_argument(
        "--embed-profile",
        action="store_true",
        help="Embed sRGB ICC color profile in output image (requires ImageMagick). "
             "Recommended for print to ensure colors are interpreted correctly.",
    )

    parser.add_argument(
        "--print-dpi",
        type=int,
        default=300,
        help="DPI for print quality renders (default: 300). Only used with --print-quality.",
    )

    parser.add_argument(
        "--print-width",
        type=float,
        default=10.0,
        help="Width in inches for print quality renders (default: 10.0). Only used with --print-quality.",
    )

    parser.add_argument(
        "--print-height",
        type=float,
        default=8.0,
        help="Height in inches for print quality renders (default: 8.0). Only used with --print-quality.",
    )

    parser.add_argument(
        "--background",
        action="store_true",
        help="Enable background plane in render",
    )

    parser.add_argument(
        "--background-color",
        type=str,
        default="#F5F5F0",
        help="Background plane color as hex string (default: #F5F5F0 eggshell white)",
    )

    parser.add_argument(
        "--background-distance",
        type=float,
        default=50.0,
        help="Distance below terrain to place background plane (default: 50.0 units, use 0 for drop shadows)",
    )

    parser.add_argument(
        "--background-flat",
        action="store_true",
        help="Use flat color (emission) for background that ignores lighting. "
             "Without this, background responds to scene lighting and dark colors appear lighter.",
    )

    parser.add_argument(
        "--height-scale",
        type=float,
        default=30.0,
        help="Vertical exaggeration for terrain (default: 30.0, try 10-50 range)",
    )

    parser.add_argument(
        "--camera-direction",
        type=str,
        default="above",
        choices=["above", "south", "north", "east", "west", "northeast", "northwest", "southeast", "southwest"],
        help="Camera viewing direction (default: above)",
    )

    parser.add_argument(
        "--ortho-scale",
        type=float,
        default=0.9,
        help="Orthographic camera scale (default: 0.9, smaller=zoomed in)",
    )

    parser.add_argument(
        "--camera-elevation",
        type=float,
        default=0.3,
        help="Camera height as fraction of mesh diagonal (default: 0.3, higher=more overhead view)",
    )

    # Lighting options
    parser.add_argument(
        "--sun-azimuth",
        type=float,
        default=225.0,
        help="Direction sun comes FROM in degrees (default: 225=SW, 0=N, 90=E, 180=S, 270=W)",
    )

    parser.add_argument(
        "--sun-elevation",
        type=float,
        default=30.0,
        help="Sun angle above horizon in degrees (default: 30, 0=horizon, 90=overhead)",
    )

    parser.add_argument(
        "--sun-energy",
        type=float,
        default=7.0,
        help="Sun light strength/energy (default: 7.0)",
    )

    parser.add_argument(
        "--sun-angle",
        type=float,
        default=1.0,
        help="Sun light angular size in degrees (default: 1.0, smaller=sharper shadows)",
    )

    parser.add_argument(
        "--fill-azimuth",
        type=float,
        default=45.0,
        help="Direction fill light comes FROM in degrees (default: 45=NE, opposite sun)",
    )

    parser.add_argument(
        "--fill-elevation",
        type=float,
        default=60.0,
        help="Fill light angle above horizon in degrees (default: 60)",
    )

    parser.add_argument(
        "--fill-energy",
        type=float,
        default=0.0,
        help="Fill light strength/energy (default: 0 = no fill light)",
    )

    parser.add_argument(
        "--fill-angle",
        type=float,
        default=3.0,
        help="Fill light angular size in degrees (default: 3.0, softer than sun)",
    )

    parser.add_argument(
        "--sky-intensity",
        type=float,
        default=1.0,
        help="HDRI sky intensity multiplier (default: 1.0). Lower values dim the sky. "
             "Try 0.3-0.5 for test materials like clay to avoid overexposure.",
    )

    parser.add_argument(
        "--air-density",
        type=float,
        default=1.0,
        help="Atmospheric density for sky scattering (default: 1.0). "
             "Lower values (0.3-0.5) reduce haze for clearer renders. "
             "Set to 0 for no atmospheric scattering.",
    )

    parser.add_argument(
        "--roads",
        action="store_true",
        default=False,
        help="Include interstate and state roads (default: False, use --roads to enable)",
    )

    parser.add_argument(
        "--no-roads",
        action="store_false",
        dest="roads",
        help="Disable road rendering",
    )

    parser.add_argument(
        "--road-types",
        nargs="+",
        default=["motorway", "trunk", "primary"],
        help="OSM highway types to render (default: motorway trunk primary)",
    )

    parser.add_argument(
        "--road-width",
        type=int,
        default=3,
        help="Road width in pixels (default: 3, try 5-10 for thicker roads)",
    )

    parser.add_argument(
        "--test-material",
        type=str,
        choices=["none", "obsidian", "chrome", "clay", "plastic", "gold"],
        default="none",
        help="Test material for entire terrain (ignores vertex colors): none=normal terrain colors, "
             "obsidian=glossy black, chrome=reflective metal, clay=matte gray, plastic=glossy white, gold=metallic gold",
    )

    parser.add_argument(
        "--road-color",
        type=str,
        default="azurite",
        help="Road color preset: obsidian (black), azurite (deep blue), azurite-light (richer blue), "
             "malachite (deep green), hematite (dark iron gray). Default: azurite",
    )

    # Terrain smoothing options
    parser.add_argument(
        "--smooth",
        action="store_true",
        default=False,
        help="Apply feature-preserving terrain smoothing (removes DEM noise, preserves ridges)",
    )

    parser.add_argument(
        "--smooth-spatial",
        type=float,
        default=3.0,
        help="Terrain smoothing spatial sigma in pixels (default: 3.0, larger = more smoothing)",
    )

    parser.add_argument(
        "--smooth-intensity",
        type=float,
        default=None,
        help="Terrain smoothing intensity sigma (default: auto = 5%% of elevation range)",
    )

    parser.add_argument(
        "--road-smoothing",
        action="store_true",
        default=False,
        help="Smooth road vertex elevations to reduce bumpiness (default: False)",
    )

    parser.add_argument(
        "--road-smoothing-radius",
        type=int,
        default=2,
        help="Smoothing radius for road vertices (default: 2, larger = smoother)",
    )

    parser.add_argument(
        "--road-offset",
        type=float,
        default=0.0,
        help="Fixed Z offset for road vertices. Positive = raise, negative = lower. "
             "Alternative to --road-smoothing. Default: 0.0 (no offset)",
    )

    parser.add_argument(
        "--road-antialias",
        type=float,
        default=0.0,
        help="Anti-alias road edges with Gaussian blur (sigma in pixels). "
             "Reduces jagged stair-step edges. Typical values: 0.5-2.0. "
             "0 = disabled (default), 1.0 = standard anti-aliasing.",
    )

    # Score smoothing options
    parser.add_argument(
        "--smooth-scores",
        action="store_true",
        default=False,
        help="Apply bilateral smoothing to score data to reduce blockiness from low-res SNODAS",
    )

    parser.add_argument(
        "--smooth-scores-spatial",
        type=float,
        default=5.0,
        help="Score smoothing spatial sigma in pixels (default: 5.0, larger = more smoothing)",
    )

    parser.add_argument(
        "--smooth-scores-intensity",
        type=float,
        default=None,
        help="Score smoothing intensity sigma (default: auto = 15%% of score range)",
    )

    # Score despeckle options (median filter - removes isolated outliers)
    parser.add_argument(
        "--despeckle-scores",
        action="store_true",
        default=False,
        help="Apply median filter to remove isolated speckles from score data. "
             "Better than bilateral smoothing for removing isolated low-score pixels "
             "in high-score regions (common with upsampled SNODAS data).",
    )

    parser.add_argument(
        "--despeckle-kernel",
        type=int,
        default=3,
        help="Despeckle kernel size (default: 3 for 3x3). Larger kernels remove "
             "larger speckle clusters. Common values: 3, 5, 7.",
    )

    # Score upscaling options (AI super-resolution)
    parser.add_argument(
        "--upscale-scores",
        action="store_true",
        default=False,
        help="Upscale score data using AI super-resolution to reduce blockiness. "
             "Uses Real-ESRGAN if available, otherwise bilateral upscaling.",
    )

    parser.add_argument(
        "--upscale-factor",
        type=int,
        default=4,
        choices=[2, 4, 8, 16],
        help="Score upscaling factor (default: 4). Higher = smoother but slower. "
             "Ignored if --upscale-to-dem is set.",
    )

    parser.add_argument(
        "--upscale-to-dem",
        action="store_true",
        default=False,
        help="Automatically calculate upscale factor to match SNODAS resolution "
             "to the downsampled DEM resolution. Overrides --upscale-factor.",
    )

    parser.add_argument(
        "--upscale-method",
        type=str,
        default="auto",
        choices=["auto", "esrgan", "bilateral", "bicubic"],
        help="Upscaling method: auto (try ESRGAN then bilateral), esrgan (AI), "
             "bilateral (edge-preserving), bicubic (simple). Default: auto",
    )

    # DEM despeckle (uniform noise removal for elevation)
    parser.add_argument(
        "--despeckle-dem",
        action="store_true",
        default=False,
        help="Apply median filter to remove isolated elevation noise/speckles. "
             "More uniform than --smooth (bilateral) which can look patchy. "
             "Good for DEM sensor noise or small artifacts.",
    )

    parser.add_argument(
        "--despeckle-dem-kernel",
        type=int,
        default=3,
        help="DEM despeckle kernel size (default: 3 for 3x3). "
             "3 = removes single-pixel noise, 5 = removes 2x2 artifacts, "
             "7 = stronger smoothing (may affect small terrain features).",
    )

    # Colormap transition point
    parser.add_argument(
        "--colormap-transition",
        type=float,
        default=0.27,
        help="Score value where colormap transitions from boreal green to mako blue (default: 0.27). "
             "Lower values (0.15-0.25) = transition at low scores (very little dark boreal green). "
             "Higher values (0.35-0.70) = transition at high scores (a lot of dark boreal green). "
             "Useful for adjusting visual snow score threshold.",
    )

    parser.add_argument(
        "--purple-position",
        type=float,
        default=0.6,
        help="Position of purple ribbon in boreal_mako colormap (0.0-1.0, default: 0.6). "
             "The purple band marks a score threshold (e.g., 0.5 for mid-range, 0.7 for high scores). "
             "Use with --colormap-viz-only to preview different positions.",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Gamma correction for score colormap (default: 0.5). "
             "Lower values (0.3-0.4) = brighter, expand low scores. "
             "Higher values (0.6-0.8) = darker, compress low scores. "
             "Affects where scores map to the colormap gradient.",
    )

    parser.add_argument(
        "--colormap-viz-only",
        action="store_true",
        default=False,
        help="Create matplotlib visualizations of score data with different colormap transitions "
             "(0.15, 0.27, 0.40, 0.50, 0.70) and exit before mesh creation. Fast way to preview "
             "how the colormap looks on your data without rendering. Saves to output_dir/colormap_viz/",
    )

    # Wavelet denoising (frequency-aware, structure-preserving)
    parser.add_argument(
        "--wavelet-denoise",
        action="store_true",
        default=False,
        help="Apply wavelet denoising to DEM. Smarter than --despeckle-dem because "
             "it separates terrain structure (ridges, valleys) from high-frequency noise. "
             "Requires PyWavelets: pip install PyWavelets",
    )

    parser.add_argument(
        "--wavelet-type",
        type=str,
        default="db4",
        choices=["db4", "haar", "sym4", "coif2"],
        help="Wavelet type (default: db4). db4=smooth terrain, haar=sharp edges, "
             "sym4=balanced, coif2=very smooth.",
    )

    parser.add_argument(
        "--wavelet-levels",
        type=int,
        default=3,
        help="Wavelet decomposition levels (default: 3). Each level halves resolution. "
             "2=fine detail, 3=balanced, 4=aggressive smoothing.",
    )

    parser.add_argument(
        "--wavelet-sigma",
        type=float,
        default=2.0,
        help="Noise threshold multiplier (default: 2.0). Higher = more denoising. "
             "1.5=light, 2.0=standard, 3.0=aggressive.",
    )

    parser.add_argument(
        "--wavelet-diagnostics",
        action="store_true",
        default=False,
        help="Export diagnostic plots showing wavelet denoising effect on terrain. "
             "Creates before/after comparison and coefficient analysis plots. "
             "Requires --wavelet-denoise to be enabled.",
    )

    parser.add_argument(
        "--diagnostic-dir",
        type=str,
        default=None,
        help="Directory to save diagnostic plots (default: OUTPUT_DIR/diagnostics). "
             "Created automatically if doesn't exist.",
    )

    # Morphological bump removal
    parser.add_argument(
        "--remove-bumps",
        type=int,
        default=None,
        metavar="SIZE",
        help="Remove bumps (buildings, trees) via morphological opening. SIZE is kernel radius "
             "in pixels. Removes features up to ~SIZE*2 pixels across. For downsampled DEMs: "
             "1=subtle, 2=small buildings, 3=medium, 5=large. This ONLY removes local maxima, "
             "never creates new bumps. More targeted than adaptive-smooth.",
    )
    parser.add_argument(
        "--remove-bumps-strength",
        type=float,
        default=1.0,
        metavar="STRENGTH",
        help="Strength of bump removal effect (default: 1.0). Range 0.0-1.0 where "
             "0.0=no effect, 0.3=subtle, 0.5=half effect, 1.0=full effect. "
             "Use with --remove-bumps 1 for fine control over small adjustments.",
    )

    # Slope-adaptive smoothing (smooths flat areas, preserves hills)
    parser.add_argument(
        "--adaptive-smooth",
        action="store_true",
        default=False,
        help="Apply slope-adaptive smoothing: strong smoothing in flat areas (e.g., buildings "
             "appearing as bumps), minimal smoothing on slopes/hills. Better than uniform "
             "smoothing for removing flat-area artifacts while preserving terrain structure.",
    )

    parser.add_argument(
        "--adaptive-slope-threshold",
        type=float,
        default=2.0,
        help="Slope threshold in degrees (default: 2.0). Areas with slope below this "
             "get full smoothing. 1.0=aggressive, 2.0=balanced, 5.0=smooth gentle slopes too.",
    )

    parser.add_argument(
        "--adaptive-smooth-sigma",
        type=float,
        default=5.0,
        help="Gaussian blur sigma in pixels for flat areas (default: 5.0). "
             "3.0=light, 5.0=moderate, 10.0=strong smoothing.",
    )

    parser.add_argument(
        "--adaptive-transition",
        type=float,
        default=1.0,
        help="Transition width in degrees (default: 1.0). How quickly smoothing "
             "fades off above threshold. 0.5=sharp, 1.0=smooth, 2.0=very gradual.",
    )

    parser.add_argument(
        "--adaptive-edge-threshold",
        type=float,
        default=None,
        help="Edge preservation threshold in meters (default: None=disabled). "
             "Preserves sharp boundaries like lake edges. Areas with local elevation "
             "range exceeding this threshold won't be smoothed. Recommended: 5-10m.",
    )

    parser.add_argument(
        "--vertex-multiplier",
        type=float,
        default=None,  # Set based on quality mode below
        help="Vertices per pixel multiplier (auto: 0.5 for preview, 2.5 for print). Higher = more detail, slower. "
             "Example: 3.0 gives ~6x more vertices than default 0.5 preview mode.",
    )

    parser.add_argument(
        "--downsample-method",
        type=str,
        default="average",
        choices=["average", "lanczos", "cubic", "bilinear"],
        help="DEM downsampling method (default: average). "
             "average: Area averaging - best for DEMs, no overshoot. "
             "lanczos: Sharp with good anti-aliasing. "
             "cubic: Cubic spline interpolation. "
             "bilinear: Safe fallback.",
    )

    # Atmosphere/fog options (EXPERIMENTAL - currently causes black renders)
    parser.add_argument(
        "--atmosphere",
        action="store_true",
        default=False,
        help="[EXPERIMENTAL] Enable atmospheric fog effect (currently broken - causes black renders)",
    )

    parser.add_argument(
        "--atmosphere-density",
        type=float,
        default=0.0002,
        help="[EXPERIMENTAL] Fog density (default: 0.0002). Currently causes black renders.",
    )

    # HDRI sky lighting (invisible but adds ambient light)
    parser.add_argument(
        "--hdri-lighting",
        action="store_true",
        default=True,
        help="Enable HDRI sky lighting for realistic ambient illumination (default: True)",
    )

    parser.add_argument(
        "--no-hdri-lighting",
        action="store_false",
        dest="hdri_lighting",
        help="Disable HDRI sky lighting",
    )

    # GPU memory-saving options for large renders
    parser.add_argument(
        "--auto-tile",
        action="store_true",
        default=False,
        help="Enable automatic tiling to reduce GPU VRAM usage (essential for large prints)",
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=2048,
        help="Tile size in pixels when --auto-tile is enabled (default: 2048). "
             "Smaller tiles = less VRAM but slower. Try 512-1024 for limited VRAM.",
    )

    parser.add_argument(
        "--persistent-data",
        action="store_true",
        default=False,
        help="Keep scene data in VRAM between frames (useful for multiple renders)",
    )

    # Pipeline caching options
    parser.add_argument(
        "--cache",
        action="store_true",
        default=False,
        help="Enable pipeline caching to speed up repeated renders. "
             "Caches DEM transforms, color computation, etc. Cache is invalidated "
             "when upstream parameters change.",
    )

    parser.add_argument(
        "--no-cache",
        action="store_false",
        dest="cache",
        help="Disable pipeline caching (default)",
    )

    # Two-tier edge extrusion arguments
    parser.add_argument(
        "--two-tier-edge",
        action="store_true",
        default=False,
        help="Enable two-tier edge extrusion with clean base material (default: False)",
    )

    parser.add_argument(
        "--edge-mid-depth",
        type=float,
        default=None,
        help="Depth of middle tier for two-tier edge (default: auto = base_depth * 0.25). "
             "Only used when --two-tier-edge is enabled.",
    )

    parser.add_argument(
        "--edge-base-material",
        type=str,
        default="clay",
        help="Material for base layer (default: clay). Options: clay, obsidian, chrome, "
             "plastic, gold, ivory, or RGB tuple like '0.6,0.55,0.5'. "
             "Only used when --two-tier-edge is enabled.",
    )

    parser.add_argument(
        "--no-edge-blend-colors",
        action="store_false",
        dest="edge_blend_colors",
        default=True,
        help="Disable color blending to mid tier (sharp transition at mid tier). "
             "By default, surface colors blend to the mid tier.",
    )

    parser.add_argument(
        "--smooth-boundary",
        action="store_true",
        default=False,
        help="Smooth boundary points to eliminate stair-step edges (optional refinement). "
             "Useful when two-tier-edge is enabled to create smoother transitions.",
    )

    parser.add_argument(
        "--smooth-boundary-window",
        type=int,
        default=5,
        help="Window size for boundary smoothing (default: 5). Larger values = more smoothing. "
             "Only used when --smooth-boundary is enabled.",
    )

    parser.add_argument(
        "--clear-cache",
        action="store_true",
        default=False,
        help="Clear all cached data before running",
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".pipeline_cache"),
        help="Directory for pipeline cache files (default: .pipeline_cache/)",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Parse edge_base_material (could be a material name or RGB tuple string)
    if args.two_tier_edge:
        if "," in args.edge_base_material:
            # Parse as RGB tuple
            try:
                rgb_values = [float(x.strip()) for x in args.edge_base_material.split(",")]
                if len(rgb_values) != 3:
                    raise ValueError("RGB tuple must have exactly 3 values")
                if not all(0 <= v <= 1 for v in rgb_values):
                    raise ValueError("RGB values must be in range 0-1")
                args.edge_base_material = tuple(rgb_values)
            except (ValueError, IndexError) as e:
                print(f"Error parsing --edge-base-material RGB tuple: {e}")
                print("Format should be: '0.6,0.55,0.5' (three values 0-1)")
                sys.exit(1)
        # else: leave as string material name

    # Set resolution and quality based on mode
    if args.print_quality:
        render_width = int(args.print_width * args.print_dpi)
        render_height = int(args.print_height * args.print_dpi)
        render_samples = 4096  # High quality for print (with denoising)
        quality_mode = "PRINT"
        default_vertex_mult = 2.5  # High detail for print
    else:   
        # FAST preview mode - optimized for quick iteration
        render_width = 640  # Low res for speed
        render_height = 360
        render_samples = 64  # Minimal samples - biggest speed gain
        quality_mode = "PREVIEW"
        default_vertex_mult = 0.5  # Low detail for speed

    # Apply vertex multiplier default if not explicitly set
    if args.vertex_multiplier is None:
        args.vertex_multiplier = default_vertex_mult

    # Rebuild boreal_mako colormap with specified purple position
    # Always rebuild to ensure consistency between visualization and rendering
    from src.terrain.color_mapping import _build_boreal_mako_cmap
    import matplotlib
    custom_boreal_mako = _build_boreal_mako_cmap(purple_position=args.purple_position)
    matplotlib.colormaps.register(custom_boreal_mako, force=True)
    if args.purple_position != 0.6:
        logger.info(f"Using boreal_mako colormap with purple ribbon at position {args.purple_position}")
    else:
        logger.info(f"Using boreal_mako colormap with default purple position (0.6)")

    logger.info("\n" + "=" * 70)
    logger.info("Detroit Combined Terrain Rendering (Sledding + XC Parks)")
    logger.info("=" * 70)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Scores directory: {args.scores_dir}")
    logger.info(f"Quality mode: {quality_mode}")
    if args.print_quality:
        logger.info(f"  Resolution: {render_width}×{render_height} ({args.print_width}×{args.print_height} inches @ {args.print_dpi} DPI)")
    else:
        logger.info(f"  Resolution: {render_width}×{render_height} (screen)")
    logger.info(f"  Samples: {render_samples:,}")
    logger.info(f"  Vertex multiplier: {args.vertex_multiplier}")
    if args.background:
        logger.info(f"Background plane: ENABLED")
        logger.info(f"  Color: {args.background_color}")
        logger.info(f"  Distance below terrain: {args.background_distance} units")

    # Initialize pipeline cache
    cache = PipelineCache(cache_dir=args.cache_dir, enabled=args.cache)
    if args.cache:
        logger.info(f"Pipeline caching: ENABLED (dir: {args.cache_dir})")
        if args.clear_cache:
            deleted = cache.clear_all()
            logger.info(f"  Cleared {deleted} cached files")
    else:
        logger.info("Pipeline caching: DISABLED (use --cache to enable)")

    # Define pipeline targets with their parameters
    # This allows cache keys to change when parameters change
    dem_dir = Path("data/dem/detroit")
    dem_params = {
        "directory": str(dem_dir),
        "min_latitude": 41,
        "mock_data": args.mock_data,
    }

    transform_params = {
        "src_crs": "EPSG:4326",
        "dst_crs": "EPSG:32617",
        "flip": "horizontal",
        "scale_factor": 0.0001,
        "target_vertices": int(np.floor(render_width * render_height * args.vertex_multiplier)),
        "smooth": args.smooth,
        "smooth_spatial": args.smooth_spatial,
        "smooth_intensity": args.smooth_intensity,
        "despeckle_dem": args.despeckle_dem,
        "despeckle_dem_kernel": args.despeckle_dem_kernel if args.despeckle_dem else None,
        "wavelet_denoise": args.wavelet_denoise,
        "wavelet_type": args.wavelet_type if args.wavelet_denoise else None,
        "wavelet_levels": args.wavelet_levels if args.wavelet_denoise else None,
        "wavelet_sigma": args.wavelet_sigma if args.wavelet_denoise else None,
        "adaptive_smooth": args.adaptive_smooth,
        "adaptive_slope_threshold": args.adaptive_slope_threshold if args.adaptive_smooth else None,
        "adaptive_smooth_sigma": args.adaptive_smooth_sigma if args.adaptive_smooth else None,
        "adaptive_transition": args.adaptive_transition if args.adaptive_smooth else None,
        "adaptive_edge_threshold": args.adaptive_edge_threshold if args.adaptive_smooth else None,
        "remove_bumps": args.remove_bumps,
        "remove_bumps_strength": args.remove_bumps_strength if args.remove_bumps else None,
    }

    color_params = {
        "colormap": "boreal_mako",
        "purple_position": args.purple_position,
        "gamma": args.gamma,
        "smooth_scores": args.smooth_scores,
        "smooth_scores_spatial": args.smooth_scores_spatial if args.smooth_scores else None,
        "despeckle_scores": args.despeckle_scores,
        "despeckle_kernel": args.despeckle_kernel if args.despeckle_scores else None,
        "roads_enabled": args.roads,
        "road_types": tuple(args.road_types) if args.roads else (),
        "road_width": args.road_width if args.roads else 0,
        "road_antialias": args.road_antialias if args.roads else 0,
    }

    mesh_params = {
        "height_scale": args.height_scale,
        "scale_factor": 100,
        "center_model": True,
        "boundary_extension": True,
        "two_tier_edge": args.two_tier_edge,
        "edge_mid_depth": args.edge_mid_depth,
        "edge_base_material": args.edge_base_material,
        "edge_blend_colors": args.edge_blend_colors,
        "smooth_boundary": args.smooth_boundary,
        "smooth_boundary_window": args.smooth_boundary_window if args.smooth_boundary else 5,
    }

    # Register targets with cache (defines dependency graph)
    cache.define_target("dem_loaded", params=dem_params)
    cache.define_target("dem_transformed", params=transform_params, dependencies=["dem_loaded"])
    cache.define_target("colors_computed", params=color_params, dependencies=["dem_transformed"])
    cache.define_target("mesh_created", params=mesh_params, dependencies=["colors_computed"])

    # Load data
    logger.info("\n" + "=" * 70)
    logger.info("[1/5] Loading Data")
    logger.info("=" * 70)

    # Load DEM
    if args.mock_data:
        logger.info("Generating mock DEM...")
        dem = np.random.randint(150, 250, (1024, 1024)).astype(np.float32)
        # Create a WGS84 transform (lat/lon) for Detroit area
        # Detroit is approximately at -83.05 lon, 42.35 lat
        # ~0.0001 degrees per pixel (~10m resolution at this latitude)
        # Note: Negative scale on Y because rasters are typically north-up
        transform = Affine.translation(-83.2, 42.5) * Affine.scale(0.0001, -0.0001)
        dem_crs = "EPSG:4326"  # WGS84 (lat/lon)
    else:
        dem_dir = Path("data/dem/detroit")
        if dem_dir.exists():
            # Load HGT files filtered to northern tiles (N41 and above)
            # This focuses on areas with better snow coverage (Detroit metro and north)
            # Tiles range from N37-N46; loading N41+ removes the southern ~40% of extent
            dem, transform = load_filtered_hgt_files(
                dem_dir,
                min_latitude=41,  # Load N41 and above (N41, N42, N43, N44, N45, N46)
            )
            dem_crs = "EPSG:4326"  # Real data is typically WGS84
            logger.info(f"  (focusing on Detroit metro and northern areas with better snow)")
        else:
            logger.info("Generating mock DEM (DEM directory not found)...")
            dem = np.random.randint(150, 250, (1024, 1024)).astype(np.float32)
            # Create a WGS84 transform for Detroit area
            transform = Affine.translation(-83.2, 42.5) * Affine.scale(0.0001, -0.0001)
            dem_crs = "EPSG:4326"

    # Load sledding scores
    # When using --mock-data, always generate mock scores to ensure dimensions match mock DEM
    if args.mock_data:
        logger.info("Generating mock sledding scores (mock mode)...")
        sledding_scores = generate_mock_scores(dem.shape)
        # Mock scores cover same extent as DEM - let library calculate transform automatically
        score_transform = None
    else:   
        sledding_scores, loaded_transform = load_sledding_scores(args.scores_dir)
        if sledding_scores is None:
            logger.error("Sledding scores not found. Run detroit_snow_sledding.py first.")
            return 1

        # Use loaded transform if available (new format), otherwise fall back to calculation
        if loaded_transform is not None:
            score_transform = loaded_transform
            logger.info("Using transform from score file (automatic georeferencing)")
        else:
            # Legacy fallback: calculate transform based on DEM extent
            logger.info("No transform in score file, calculating from DEM extent (legacy mode)")
            score_height, score_width = sledding_scores.shape
            dem_height, dem_width = dem.shape
            score_pixel_width = transform.a * dem_width / score_width
            score_pixel_height = transform.e * dem_height / score_height
            score_transform = Affine.translation(transform.c, transform.f) * Affine.scale(
                score_pixel_width,
                score_pixel_height
            )

        logger.info(f"Score shape: {sledding_scores.shape}, DEM shape: {dem.shape}")

    # Load XC skiing scores
    if args.mock_data:
        logger.info("Generating mock XC skiing scores (mock mode)...")
        xc_scores = generate_mock_scores(dem.shape)
        # Mock scores cover same extent as DEM - let library calculate transform automatically
        xc_transform = None
    else:
        xc_scores, xc_loaded_transform = load_xc_skiing_scores(args.scores_dir / "xc_skiing")
        if xc_scores is None:
            logger.error("XC skiing scores not found. Run detroit_xc_skiing.py first.")
            return 1

        # Use loaded transform if available, otherwise use same as sledding
        if xc_loaded_transform is not None:
            xc_transform = xc_loaded_transform
            logger.info("Using transform from XC score file (automatic georeferencing)")
        else:
            # Legacy fallback: use sledding transform (assumes same extent)
            xc_transform = score_transform
            logger.info("No transform in XC score file, using sledding transform")

    # Despeckle scores BEFORE upscaling (removes isolated outliers at native resolution)
    if args.despeckle_scores and not args.mock_data:
        logger.info(f"Despeckle scores with kernel size {args.despeckle_kernel}...")
        sledding_scores = despeckle_scores(sledding_scores, kernel_size=args.despeckle_kernel)
        xc_scores = despeckle_scores(xc_scores, kernel_size=args.despeckle_kernel)
        logger.info("Despeckled scores (before upscaling)")

    # Upscale scores if requested (AI super-resolution to reduce blockiness)
    if args.upscale_scores and not args.mock_data:
        # Calculate upscale factor automatically if --upscale-to-dem is set
        if args.upscale_to_dem and score_transform is not None:
            # Calculate target DEM dimensions after downsampling
            target_vertices = int(np.floor(render_width * render_height * args.vertex_multiplier))
            dem_aspect = dem.shape[1] / dem.shape[0]  # width/height
            target_height = int(np.sqrt(target_vertices / dem_aspect))
            target_width = int(target_height * dem_aspect)

            # Calculate DEM geographic extent (in degrees)
            dem_extent_x = abs(transform.a) * dem.shape[1]  # degrees longitude
            dem_extent_y = abs(transform.e) * dem.shape[0]  # degrees latitude

            # DEM effective pixel size after downsampling (in degrees)
            dem_pixel_deg_x = dem_extent_x / target_width
            dem_pixel_deg_y = dem_extent_y / target_height

            # SNODAS pixel size (in degrees) - use absolute values
            snodas_pixel_deg_x = abs(score_transform.a)
            snodas_pixel_deg_y = abs(score_transform.e)

            # Calculate ratio (use average of x and y)
            ratio_x = snodas_pixel_deg_x / dem_pixel_deg_x
            ratio_y = snodas_pixel_deg_y / dem_pixel_deg_y
            ratio = (ratio_x + ratio_y) / 2

            # Pick smallest power of 2 that is >= ratio
            if ratio <= 1:
                computed_factor = 1  # No upscaling needed
            elif ratio <= 2:
                computed_factor = 2
            elif ratio <= 4:
                computed_factor = 4
            elif ratio <= 8:
                computed_factor = 8
            else:
                computed_factor = 16

            logger.info(f"Auto-calculating upscale factor (--upscale-to-dem):")
            logger.info(f"  DEM effective resolution: {target_width}×{target_height} pixels")
            logger.info(f"  DEM pixel size: {dem_pixel_deg_x*111000:.0f}m × {dem_pixel_deg_y*111000:.0f}m")
            logger.info(f"  SNODAS pixel size: {snodas_pixel_deg_x*111000:.0f}m × {snodas_pixel_deg_y*111000:.0f}m")
            logger.info(f"  Resolution ratio: {ratio:.1f}x → upscale factor: {computed_factor}x")

            if computed_factor == 1:
                logger.info("  SNODAS resolution already matches DEM, skipping upscale")
                args.upscale_scores = False  # Skip upscaling
            else:
                args.upscale_factor = computed_factor

        if args.upscale_scores:  # Check again in case we disabled it above
            logger.info(f"Upscaling scores by {args.upscale_factor}x using {args.upscale_method}...")

            # Store originals for diagnostics
            sledding_scores_original = sledding_scores.copy()
            xc_scores_original = xc_scores.copy()

            # Upscale sledding scores
            sledding_scores = upscale_scores(
                sledding_scores,
                scale=args.upscale_factor,
                method=args.upscale_method,
            )
            # Update transform to reflect new resolution
            if score_transform is not None:
                score_transform = Affine(
                    score_transform.a / args.upscale_factor,  # pixel width
                    score_transform.b,
                    score_transform.c,  # origin unchanged
                    score_transform.d,
                    score_transform.e / args.upscale_factor,  # pixel height
                    score_transform.f,  # origin unchanged
                )

            # Upscale XC scores
            xc_scores = upscale_scores(
                xc_scores,
                scale=args.upscale_factor,
                method=args.upscale_method,
            )
            # Update transform to reflect new resolution
            if xc_transform is not None:
                xc_transform = Affine(
                    xc_transform.a / args.upscale_factor,
                    xc_transform.b,
                    xc_transform.c,
                    xc_transform.d,
                    xc_transform.e / args.upscale_factor,
                    xc_transform.f,
                )

            logger.info(f"Upscaled scores: sledding {sledding_scores.shape}, XC {xc_scores.shape}")

            # Always generate upscale diagnostics
            # Use diagnostic_dir if set, otherwise use output_dir/diagnostics
            if args.diagnostic_dir:
                diagnostic_dir = Path(args.diagnostic_dir)
            else:
                diagnostic_dir = args.output_dir / "diagnostics"
            diagnostic_dir.mkdir(parents=True, exist_ok=True)

            # Determine which method was actually used
            # (auto falls back to bilateral if esrgan not installed)
            used_method = args.upscale_method
            if args.upscale_method == "auto":
                try:
                    import realesrgan  # noqa: F401
                    used_method = "esrgan"
                except ImportError:
                    used_method = "bilateral"

            # Generate sledding upscale diagnostics
            generate_upscale_diagnostics(
                sledding_scores_original,
                sledding_scores,
                diagnostic_dir,
                prefix="sledding_upscale",
                scale=args.upscale_factor,
                method=used_method,
                cmap="plasma",
            )

            # Generate XC upscale diagnostics
            generate_upscale_diagnostics(
                xc_scores_original,
                xc_scores,
                diagnostic_dir,
                prefix="xc_upscale",
                scale=args.upscale_factor,
                method=used_method,
                cmap="viridis",
            )
            logger.info(f"Saved upscale diagnostics to {diagnostic_dir}")

    # Load parks for XC skiing markers
    parks = None
    if not args.mock_data:
        parks = load_xc_skiing_parks(args.scores_dir / "xc_skiing")
        if parks:
            logger.info(f"Loaded {len(parks)} parks for markers")

    # Load road data
    road_data = None
    road_bbox = None
    if args.roads:
        logger.info("Loading road data from OpenStreetMap...")
        try:
            # Compute bbox from DEM transform to match actual rendered area
            dem_height, dem_width = dem.shape

            # Get corners from transform
            west = transform.c
            north = transform.f
            east = west + dem_width * transform.a
            south = north + dem_height * transform.e

            # Ensure correct ordering
            if south > north:
                south, north = north, south
            if west > east:
                west, east = east, west

            road_bbox = (south, west, north, east)
            logger.info(f"  DEM bbox: lat [{south:.2f}, {north:.2f}], lon [{west:.2f}, {east:.2f}]")

            # Use get_roads_tiled() - handles tiling and retries automatically
            road_data = get_roads_tiled(road_bbox, args.road_types)

            if road_data and road_data.get("features"):
                logger.info(f"  Loaded {len(road_data['features'])} road segments (obsidian material)")
            else:
                logger.warning("  No roads found or fetch failed")
                road_data = None
        except Exception as e:
            logger.warning(f"Failed to load road data: {e}")
            road_data = None

    logger.info(f"Loaded: DEM {dem.shape}, sledding scores {sledding_scores.shape}, XC scores {xc_scores.shape}")

    # Create Blender scene
    logger.info("\n" + "=" * 70)
    logger.info("Creating Blender Scene")
    logger.info("=" * 70)

    clear_scene()

    # Initialize memory monitor
    memory_config = TiledDataConfig()  # Defaults: 85% RAM, 50% swap
    monitor = MemoryMonitor(memory_config)

    if monitor.enabled:
        logger.info("Memory monitoring enabled")
        monitor.check_memory(force=True)
    else:
        logger.warning("Memory monitoring disabled (psutil not available)")

    # Calculate target vertices for mesh creation
    # Match render resolution for optimal detail
    target_vertices = int(np.floor(render_width * render_height * args.vertex_multiplier))
    logger.info(f"Target vertices: {target_vertices:,} ({quality_mode} resolution, {args.vertex_multiplier}x multiplier)")

    # Create single terrain mesh with dual colormaps
    # Base: Boreal-Mako colormap for sledding suitability scores (with gamma=0.5)
    # Overlay: Rocket colormap for XC skiing scores near parks
    logger.info("\n[2/4] Creating Combined Terrain Mesh (Dual Colormap)...")
    logger.info(f"  Base: Sledding scores with gamma={args.gamma} (boreal_mako colormap - forest green → blue → mint)")
    logger.info("  Overlay: XC skiing scores near parks (rocket colormap)")

    # For combined rendering, we need a custom terrain creation process
    # to add both sledding and XC skiing scores as separate layers
    logger.debug("Creating terrain with DEM...")
    terrain_combined = Terrain(dem, transform, dem_crs=dem_crs)

    # Add standard transforms (cached_reproject saves ~24s on subsequent runs)
    terrain_combined.add_transform(cached_reproject(src_crs=dem_crs, dst_crs="EPSG:32617"))
    terrain_combined.add_transform(flip_raster(axis="horizontal"))
    # Note: scale_elevation is added AFTER adaptive_smooth so slope computation uses real elevations

    # Configure downsampling FIRST (all smoothing runs on downsampled data for memory efficiency)
    if target_vertices:
        logger.debug(f"Configuring for target vertices: {target_vertices:,} (method: {args.downsample_method})")
        terrain_combined.configure_for_target_vertices(
            target_vertices=target_vertices, method=args.downsample_method
        )

    # =========================================================================
    # PHASE 1: Apply geometry transforms (reproject, flip, downsample)
    # Water detection happens AFTER this phase, BEFORE any smoothing
    # =========================================================================

    # Configure diagnostic modes for transform visualization
    adaptive_diagnostic_mode = args.adaptive_smooth
    bump_diagnostic_mode = args.remove_bumps is not None
    wavelet_diagnostic_mode = args.wavelet_diagnostics and args.wavelet_denoise
    if args.wavelet_diagnostics and not args.wavelet_denoise:
        logger.warning("--wavelet-diagnostics requires --wavelet-denoise to be enabled. Ignoring.")

    # Debug: report which modes are active
    logger.info(f"DEBUG: Transform modes: adaptive={adaptive_diagnostic_mode}, bump={bump_diagnostic_mode}, wavelet={wavelet_diagnostic_mode}")

    # Apply geometry transforms to DEM (with caching)
    if args.cache:
        transform_key = cache.compute_target_key("dem_transformed")
        cached_data, cached_meta = cache.get_cached("dem_transformed", return_metadata=True)
        if cached_data is not None and cached_meta:
            logger.info(f"  Cache HIT: dem_transformed (key: {transform_key[:12]}...) - restoring from cache")
            # Restore cached transform result including Affine transform
            terrain_combined.data_layers["dem"]["transformed_data"] = cached_data
            terrain_combined.data_layers["dem"]["transformed"] = True
            # Restore the Affine transform and CRS (needed for coordinate conversions)
            if "affine_transform" in cached_meta:
                terrain_combined.data_layers["dem"]["transformed_transform"] = cached_meta["affine_transform"]
            if "crs" in cached_meta:
                terrain_combined.data_layers["dem"]["transformed_crs"] = cached_meta["crs"]
            # Clear transforms since they're "applied"
            terrain_combined.transforms = []
            logger.debug("✓ Skipped apply_transforms() - loaded from cache")
        else:
            logger.info(f"  Cache MISS: dem_transformed (key: {transform_key[:12]}...) - computing")
            terrain_combined.apply_transforms()
            # Save transformed DEM with Affine transform for future runs
            dem_layer = terrain_combined.data_layers["dem"]
            cache.save_target(
                "dem_transformed",
                dem_layer["transformed_data"],
                metadata={
                    "affine_transform": dem_layer.get("transformed_transform"),
                    "crs": dem_layer.get("transformed_crs"),
                },
            )
            logger.debug("✓ Saved dem_transformed to cache")
    else:
        logger.debug("Applying transforms to DEM...")
        terrain_combined.apply_transforms()

    # Debug: verify DEM state after geometry transforms (before smoothing)
    dem_after_geom = terrain_combined.data_layers["dem"]["transformed_data"]
    logger.info(f"DEM after geometry transforms: shape={dem_after_geom.shape}, "
                f"min={np.nanmin(dem_after_geom):.4f}, max={np.nanmax(dem_after_geom):.4f}, "
                f"std={np.nanstd(dem_after_geom):.4f}")

    # =========================================================================
    # WATER DETECTION: Detect water on pre-smoothed DEM (before any smoothing)
    # This ensures accurate slope-based detection without smoothing artifacts
    # =========================================================================
    logger.debug("Detecting water bodies on pre-smoothed DEM...")
    dem_for_water = terrain_combined.data_layers["dem"]["transformed_data"]
    water_mask = identify_water_by_slope(
        dem_for_water,
        slope_threshold=0.01,  # Very flat areas only (water is slope ≈ 0)
        fill_holes=True,
    )
    water_pixel_count = np.sum(water_mask) if water_mask is not None else 0
    logger.info(f"Water detection: {water_pixel_count:,} pixels detected as water")

    # =========================================================================
    # PHASE 2: Apply smoothing transforms (after water detection)
    # These operations may artificially flatten areas, so water is detected first
    # =========================================================================

    # Apply feature-preserving smoothing
    if args.smooth:
        logger.info(f"Applying feature-preserving smoothing (spatial={args.smooth_spatial}, intensity={args.smooth_intensity or 'auto'})")
        smooth_transform = feature_preserving_smooth(
            sigma_spatial=args.smooth_spatial,
            sigma_intensity=args.smooth_intensity,
        )
        dem_layer = terrain_combined.data_layers["dem"]
        smoothed_dem, _, _ = smooth_transform(dem_layer["transformed_data"])
        dem_layer["transformed_data"] = smoothed_dem

    # Apply DEM despeckle (uniform noise removal via median filter)
    if args.despeckle_dem:
        logger.info(f"Applying DEM despeckle (kernel_size={args.despeckle_dem_kernel})")
        despeckle_transform = despeckle_dem(kernel_size=args.despeckle_dem_kernel)
        dem_layer = terrain_combined.data_layers["dem"]
        despeckled_dem, _, _ = despeckle_transform(dem_layer["transformed_data"])
        dem_layer["transformed_data"] = despeckled_dem

    # Apply wavelet denoising (normal mode - not diagnostic)
    if args.wavelet_denoise and not wavelet_diagnostic_mode:
        logger.info(
            f"Applying wavelet denoising (wavelet={args.wavelet_type}, levels={args.wavelet_levels}, "
            f"sigma={args.wavelet_sigma})"
        )
        wavelet_transform = wavelet_denoise_dem(
            wavelet=args.wavelet_type,
            levels=args.wavelet_levels,
            threshold_sigma=args.wavelet_sigma,
            preserve_structure=True,
        )
        dem_layer = terrain_combined.data_layers["dem"]
        denoised_dem, _, _ = wavelet_transform(dem_layer["transformed_data"])
        dem_layer["transformed_data"] = denoised_dem

    # Apply wavelet denoising with diagnostics (if requested)
    if wavelet_diagnostic_mode:
        logger.info(
            f"Wavelet diagnostic mode: capturing before/after state "
            f"(wavelet={args.wavelet_type}, levels={args.wavelet_levels}, sigma={args.wavelet_sigma})"
        )

        # Get the transformed DEM (pre-wavelet)
        dem_layer = terrain_combined.data_layers["dem"]
        pre_wavelet_dem = dem_layer["transformed_data"].copy()

        # Apply wavelet denoising manually
        wavelet_transform = wavelet_denoise_dem(
            wavelet=args.wavelet_type,
            levels=args.wavelet_levels,
            threshold_sigma=args.wavelet_sigma,
            preserve_structure=True,
        )
        post_wavelet_dem, _, _ = wavelet_transform(pre_wavelet_dem)

        # Update the terrain with the denoised DEM
        dem_layer["transformed_data"] = post_wavelet_dem

        # Generate diagnostic plots
        diagnostic_dir = Path(args.diagnostic_dir) if args.diagnostic_dir else args.output_dir / "diagnostics"
        logger.info(f"Generating wavelet diagnostic plots in {diagnostic_dir}")

        comparison_path, coefficients_path = generate_full_wavelet_diagnostics(
            original=pre_wavelet_dem,
            denoised=post_wavelet_dem,
            output_dir=diagnostic_dir,
            prefix="dem_wavelet",
            wavelet=args.wavelet_type,
            levels=args.wavelet_levels,
            threshold_sigma=args.wavelet_sigma,
        )

        logger.info(f"✓ Saved wavelet comparison: {comparison_path}")
        if coefficients_path:
            logger.info(f"✓ Saved coefficient analysis: {coefficients_path}")

    # Apply adaptive smoothing with diagnostics (if requested)
    if adaptive_diagnostic_mode:
        edge_str = f", edge={args.adaptive_edge_threshold}m" if args.adaptive_edge_threshold else ""
        logger.info(
            f"Adaptive smooth diagnostic mode: capturing before/after state "
            f"(threshold={args.adaptive_slope_threshold}°, sigma={args.adaptive_smooth_sigma}, "
            f"transition={args.adaptive_transition}°{edge_str})"
        )

        # Get the transformed DEM (pre-smooth, already downsampled)
        dem_layer = terrain_combined.data_layers["dem"]
        pre_smooth_dem = dem_layer["transformed_data"].copy()
        dem_affine = dem_layer.get("transformed_transform")

        pixel_size = abs(dem_affine.a) if dem_affine is not None else None
        if pixel_size is not None:
            logger.info(f"  Pixel size (downsampled): {pixel_size:.1f}m")

        # Apply adaptive smoothing manually
        adaptive_transform = slope_adaptive_smooth(
            slope_threshold=args.adaptive_slope_threshold,
            smooth_sigma=args.adaptive_smooth_sigma,
            transition_width=args.adaptive_transition,
            edge_threshold=args.adaptive_edge_threshold,
        )
        post_smooth_dem, _, _ = adaptive_transform(pre_smooth_dem, dem_affine)

        # Apply scale_elevation after smoothing (same as normal mode)
        scale_transform = scale_elevation(scale_factor=0.0001)
        scaled_dem, _, _ = scale_transform(post_smooth_dem)

        # Update the terrain with the smoothed and scaled DEM
        dem_layer["transformed_data"] = scaled_dem

        # Debug: verify adaptive smooth saved correctly
        verify_dem = terrain_combined.data_layers["dem"]["transformed_data"]
        logger.info(f"DEBUG: After adaptive smooth save: std={np.nanstd(verify_dem):.6f}, "
                    f"same_array={verify_dem is scaled_dem}, "
                    f"pre_smooth_std={np.nanstd(pre_smooth_dem):.4f}, post_smooth_std={np.nanstd(post_smooth_dem):.4f}")

        # Generate diagnostic plots (using unscaled data for meaningful elevation values)
        diagnostic_dir = Path(args.diagnostic_dir) if args.diagnostic_dir else args.output_dir / "diagnostics"
        logger.info(f"Generating adaptive smooth diagnostic plots in {diagnostic_dir}")

        spatial_path, histogram_path = generate_full_adaptive_smooth_diagnostics(
            original=pre_smooth_dem,
            smoothed=post_smooth_dem,
            output_dir=diagnostic_dir,
            prefix="dem_adaptive_smooth",
            slope_threshold=args.adaptive_slope_threshold,
            smooth_sigma=args.adaptive_smooth_sigma,
            transition_width=args.adaptive_transition,
            pixel_size=pixel_size,
            edge_threshold=args.adaptive_edge_threshold,
        )

        logger.info(f"✓ Saved adaptive smooth comparison: {spatial_path}")
        logger.info(f"✓ Saved adaptive smooth histogram: {histogram_path}")

    # Apply bump removal with diagnostics (if requested)
    if bump_diagnostic_mode:
        strength_str = f", strength={args.remove_bumps_strength}" if args.remove_bumps_strength < 1.0 else ""
        logger.info(f"Bump removal diagnostic mode: capturing before/after state (kernel={args.remove_bumps}{strength_str})")

        # Get the transformed DEM (pre-bump-removal, already downsampled)
        dem_layer = terrain_combined.data_layers["dem"]
        pre_bump_dem = dem_layer["transformed_data"].copy()

        # Check if data is already scaled (from adaptive smooth mode)
        already_scaled = adaptive_diagnostic_mode  # adaptive mode applies scale_elevation

        # If already scaled, unscale for proper bump removal then rescale
        if already_scaled:
            logger.info("  Note: DEM already scaled from adaptive smooth - unscaling for bump removal")
            pre_bump_dem_unscaled = pre_bump_dem / 0.0001
            bump_transform = remove_bumps(kernel_size=args.remove_bumps, strength=args.remove_bumps_strength)
            post_bump_dem_unscaled, _, _ = bump_transform(pre_bump_dem_unscaled, None)
            # Scale back
            scaled_dem = post_bump_dem_unscaled * 0.0001
            post_bump_dem = post_bump_dem_unscaled  # For diagnostics (unscaled)
        else:
            # Apply bump removal to unscaled data
            bump_transform = remove_bumps(kernel_size=args.remove_bumps, strength=args.remove_bumps_strength)
            post_bump_dem, _, _ = bump_transform(pre_bump_dem, None)

            # Apply scale_elevation after bump removal
            scale_transform = scale_elevation(scale_factor=0.0001)
            scaled_dem, _, _ = scale_transform(post_bump_dem)

        # Update the terrain with the processed DEM
        dem_layer["transformed_data"] = scaled_dem

        # Debug: verify bump removal saved correctly
        verify_dem = terrain_combined.data_layers["dem"]["transformed_data"]
        logger.info(f"DEBUG: After bump removal save: std={np.nanstd(verify_dem):.6f}, "
                    f"pre_bump_std={np.nanstd(pre_bump_dem):.6f}, post_bump_std={np.nanstd(post_bump_dem):.6f}")

        # Generate diagnostic plots (using unscaled data for meaningful elevation values)
        diagnostic_dir = Path(args.diagnostic_dir) if args.diagnostic_dir else args.output_dir / "diagnostics"
        logger.info(f"Generating bump removal diagnostic plots in {diagnostic_dir}")

        # Use unscaled data for diagnostics
        if already_scaled:
            diag_original = pre_bump_dem / 0.0001  # Unscale for meaningful elevation values
            diag_after = post_bump_dem  # Already unscaled above
        else:
            diag_original = pre_bump_dem
            diag_after = post_bump_dem

        diag_path = generate_bump_removal_diagnostics(
            original=diag_original,
            after_removal=diag_after,
            output_dir=diagnostic_dir,
            prefix="dem_bump_removal",
            kernel_size=args.remove_bumps,
        )

        logger.info(f"✓ Saved bump removal diagnostics: {diag_path}")

    # Apply scale_elevation in normal mode (diagnostic modes handle it themselves)
    if not adaptive_diagnostic_mode and not bump_diagnostic_mode:
        logger.debug("Applying scale_elevation to DEM...")
        scale_transform = scale_elevation(scale_factor=0.0001)
        dem_layer = terrain_combined.data_layers["dem"]
        scaled_dem, _, _ = scale_transform(dem_layer["transformed_data"])
        dem_layer["transformed_data"] = scaled_dem

    # Debug: verify DEM state before mesh creation
    dem_for_debug = terrain_combined.data_layers["dem"]["transformed_data"]
    logger.info(f"DEM state before mesh: shape={dem_for_debug.shape}, "
                f"min={np.nanmin(dem_for_debug):.4f}, max={np.nanmax(dem_for_debug):.4f}, "
                f"std={np.nanstd(dem_for_debug):.4f}")

    # Add sledding scores as base layer
    logger.debug("Adding sledding scores layer...")
    if score_transform is not None:
        terrain_combined.add_data_layer(
            "sledding",
            sledding_scores,
            score_transform,
            dem_crs,
            target_layer="dem",
        )
    else:
        terrain_combined.add_data_layer(
            "sledding",
            sledding_scores,
            same_extent_as="dem",
        )

    # Add XC skiing scores as overlay layer
    logger.debug("Adding XC skiing scores layer...")
    if xc_transform is not None:
        terrain_combined.add_data_layer(
            "xc_skiing",
            xc_scores,
            xc_transform,
            dem_crs,
            target_layer="dem",
        )
    else:
        terrain_combined.add_data_layer(
            "xc_skiing",
            xc_scores,
            same_extent_as="dem",
        )

    # Apply score smoothing if requested (reduces blockiness from low-res SNODAS data)
    if args.smooth_scores:
        logger.info(
            f"Smoothing score data (spatial={args.smooth_scores_spatial}, "
            f"intensity={args.smooth_scores_intensity or 'auto'})"
        )
        # Smooth sledding scores
        sledding_data = terrain_combined.data_layers["sledding"]["data"]
        terrain_combined.data_layers["sledding"]["data"] = smooth_score_data(
            sledding_data,
            sigma_spatial=args.smooth_scores_spatial,
            sigma_intensity=args.smooth_scores_intensity,
        )
        # Smooth XC skiing scores
        xc_data = terrain_combined.data_layers["xc_skiing"]["data"]
        terrain_combined.data_layers["xc_skiing"]["data"] = smooth_score_data(
            xc_data,
            sigma_spatial=args.smooth_scores_spatial,
            sigma_intensity=args.smooth_scores_intensity,
        )
        logger.info("✓ Score data smoothed")

    # Note: despeckle_scores is now applied earlier (before upscaling) for better results

    # === Colormap Visualization Mode ===
    # If --colormap-viz-only is set, create matplotlib visualizations and exit
    if args.colormap_viz_only:
        logger.info("Colormap visualization mode - creating matplotlib plots...")

        # Create output directory for visualizations
        viz_dir = args.output_dir / "colormap_viz"
        viz_dir.mkdir(parents=True, exist_ok=True)

        # Get the sledding scores (downsampled, already processed)
        sledding_scores = terrain_combined.data_layers["sledding"]["data"]

        logger.info(f"Sledding scores shape: {sledding_scores.shape}")
        logger.info(f"Score range: [{np.nanmin(sledding_scores):.3f}, {np.nanmax(sledding_scores):.3f}]")
        logger.info(f"Score mean: {np.nanmean(sledding_scores):.3f}")

        # Score distribution
        bins = [0, 0.2, 0.4, 0.40, 0.55, 0.6, 0.8, 1.0]
        hist, _ = np.histogram(sledding_scores[~np.isnan(sledding_scores)], bins=bins)
        logger.info("Score distribution:")
        for i in range(len(bins)-1):
            pct = 100 * hist[i] / np.sum(~np.isnan(sledding_scores))
            marker = " ← PURPLE ZONE (WIDER)" if bins[i] == 0.40 else ""
            logger.info(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}): {hist[i]:6d} ({pct:5.1f}%){marker}")

        # Create simple visualization: normalize → gamma → colormap
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        fig.suptitle(f'Detroit Sledding Scores with Boreal-Mako Colormap\n(normalized, then gamma={args.gamma})',
                     fontsize=14, fontweight='bold')

        # Normalize scores to 0-1.0 range (max score becomes 1.0)
        max_score = np.nanmax(sledding_scores)
        logger.info(f"Max score before normalization: {max_score:.3f}")
        normalized_scores = sledding_scores / max_score

        # Apply gamma correction to normalized scores
        gamma_corrected_scores = np.power(normalized_scores, args.gamma)

        # Apply colormap (get from registry to respect --purple-position)
        import matplotlib
        boreal_mako_from_registry = matplotlib.colormaps.get_cmap("boreal_mako")
        im = ax.imshow(gamma_corrected_scores, cmap=boreal_mako_from_registry,
                      vmin=0, vmax=1.0,
                      origin='lower', aspect='auto')

        ax.set_title(f'Normalized scores with gamma={args.gamma} applied', fontsize=12, fontweight='bold')
        ax.axis('off')

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)
        cbar.set_label('Gamma-Corrected Score (normalized score ^ 0.5)', fontsize=11, fontweight='bold')

        plt.tight_layout()

        output_path = viz_dir / "scores_map.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Saved: {output_path}")
        plt.close()

        # Create histogram plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle(f'Detroit Score Distribution (normalized, then gamma={args.gamma})',
                     fontsize=14, fontweight='bold')

        valid_gamma = gamma_corrected_scores[~np.isnan(gamma_corrected_scores)]

        # Histogram (showing gamma-corrected values)
        counts, bins, patches = ax.hist(valid_gamma.flatten(), bins=100,
                                       color='steelblue', alpha=0.7,
                                       edgecolor='black', linewidth=0.3)

        ax.set_xlabel('Gamma-Corrected Score (normalized score ^ 0.5)', fontsize=11)
        ax.set_ylabel('Pixel Count', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xlim(0, 1.0)

        plt.tight_layout()

        output_path = viz_dir / "scores_histograms.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Saved: {output_path}")
        plt.close()

        logger.info("")
        logger.info("="*70)
        logger.info("Colormap visualizations created successfully!")
        logger.info("="*70)
        logger.info(f"Output directory: {viz_dir}")
        logger.info("")
        logger.info("Exiting before mesh creation (--colormap-viz-only mode)")
        return 0

    # Create mesh for proximity calculations
    logger.debug("Creating temporary mesh for proximity calculations...")
    mesh_temp = terrain_combined.create_mesh(
        scale_factor=100,
        height_scale=args.height_scale,
        center_model=True,
        boundary_extension=True,
        water_mask=None,
        two_tier_edge=args.two_tier_edge,
        edge_mid_depth=args.edge_mid_depth,
        edge_base_material=args.edge_base_material,
        edge_blend_colors=args.edge_blend_colors,
        smooth_boundary=args.smooth_boundary,
        smooth_boundary_window=args.smooth_boundary_window if args.smooth_boundary else 5,
    )

    if mesh_temp is None:
        logger.error("Failed to create temporary mesh")
        return 1

    # Compute proximity mask for parks if available
    if parks:
        logger.info(f"Computing proximity mask for {len(parks)} parks...")
        park_lons = np.array([p["lon"] for p in parks])
        park_lats = np.array([p["lat"] for p in parks])
        park_mask = terrain_combined.compute_proximity_mask(
            park_lons,
            park_lats,
            radius_meters=2_500,
            cluster_threshold_meters=500,
        )
        logger.debug(f"Proximity mask: {np.sum(park_mask)} vertices in park zones")

    # Add roads as data layer if requested
    # Roads are added BEFORE color mapping so they can be part of the multi-overlay system
    if args.roads and road_data and road_bbox:
        logger.info("\nAdding roads as data layer...")
        try:
            # Use the same bbox that was used to fetch roads (computed from DEM)
            add_roads_layer(
                terrain=terrain_combined,
                roads_geojson=road_data,
                bbox=road_bbox,
                resolution=30.0,  # 30m pixels
                road_width_pixels=args.road_width,
            )

            # Apply road anti-aliasing if requested (smooths jagged edges)
            if args.road_antialias > 0 and "roads" in terrain_combined.data_layers:
                logger.info(f"Anti-aliasing road edges (sigma={args.road_antialias})...")
                road_data_layer = terrain_combined.data_layers["roads"]["data"]
                smoothed_roads = smooth_road_mask(road_data_layer, sigma=args.road_antialias)
                terrain_combined.data_layers["roads"]["data"] = smoothed_roads
                logger.info("✓ Road anti-aliasing applied")

        except Exception as e:
            logger.warning(f"Failed to add roads layer: {e}")

    # Road smoothing is now applied after mesh creation (on vertices, not DEM)
    # This avoids the coordinate alignment issues that plagued the old approach

    # Set up color mapping with overlays
    # Note: Roads are NOT included in color overlays - they keep terrain colors
    # but get glassy material applied via separate RoadMask vertex color layer
    if args.roads and road_data and road_bbox and parks:
        # With parks: base sledding + XC skiing overlay (no road color overlay)
        logger.info("Setting multi-overlay color mapping:")
        logger.info(f"  Base: Sledding scores with gamma={args.gamma} (boreal_mako colormap)")
        logger.info("  Overlay: XC skiing scores near parks (rocket colormap)")
        logger.info("  Roads: Keep terrain color, apply glassy material via mask")

        overlays = [
            {
                "colormap": lambda score: elevation_colormap(
                    score, cmap_name="rocket", min_elev=0.0, max_elev=1.0
                ),
                "source_layers": ["xc_skiing"],
                "priority": 20,
                "mask": park_mask,  # Only apply near parks
            },
        ]

        terrain_combined.set_multi_color_mapping(
            base_colormap=lambda score: elevation_colormap(
                np.power(score / np.nanmax(score), args.gamma), cmap_name="boreal_mako", min_elev=0.0, max_elev=1.0
            ),
            base_source_layers=["sledding"],
            overlays=overlays,
        )
    elif args.roads and road_data and road_bbox:
        # Roads but no parks: just base sledding (no overlays, roads get glassy material)
        logger.info("Setting color mapping:")
        logger.info(f"  Base: Sledding scores with gamma={args.gamma} (boreal_mako colormap)")
        logger.info("  Roads: Keep terrain color, apply glassy material via mask")
        terrain_combined.set_color_mapping(
            lambda score: elevation_colormap(np.power(score / np.nanmax(score), args.gamma), cmap_name="boreal_mako", min_elev=0.0, max_elev=1.0),
            source_layers=["sledding"],
        )
    else:
        # No roads - use original blended or standard color mapping
        if parks:
            logger.info("Setting blended color mapping:")
            logger.info(f"  Base: Sledding scores with gamma={args.gamma} (boreal_mako colormap)")
            logger.info("  Overlay: XC skiing scores near parks (rocket colormap)")
            terrain_combined.set_blended_color_mapping(
                base_colormap=lambda score: elevation_colormap(
                    np.power(score / np.nanmax(score), args.gamma), cmap_name="boreal_mako", min_elev=0.0, max_elev=1.0
                ),
                base_source_layers=["sledding"],
                overlay_colormap=lambda score: elevation_colormap(
                    score, cmap_name="rocket", min_elev=0.0, max_elev=1.0
                ),
                overlay_source_layers=["xc_skiing"],
                overlay_mask=park_mask,
            )
        else:
            logger.info(f"No parks available - using sledding scores with gamma={args.gamma} (boreal_mako colormap)")
            terrain_combined.set_color_mapping(
                lambda score: elevation_colormap(np.power(score / np.nanmax(score), args.gamma), cmap_name="boreal_mako", min_elev=0.0, max_elev=1.0),
                source_layers=["sledding"],
            )

    # NOTE: Water mask was detected earlier (before smoothing) for accurate slope detection
    # The water_mask variable is already defined and ready to use

    # Compute colors (with caching)
    if args.cache:
        color_key = cache.compute_target_key("colors_computed")
        cached_colors = cache.get_cached("colors_computed")
        if cached_colors is not None:
            logger.info(f"  Cache HIT: colors_computed (key: {color_key[:12]}...) - restoring from cache")
            terrain_combined.colors = cached_colors
            logger.debug("✓ Skipped compute_colors() - loaded from cache")
        else:
            logger.info(f"  Cache MISS: colors_computed (key: {color_key[:12]}...) - computing")
            terrain_combined.compute_colors(water_mask=water_mask)
            cache.save_target("colors_computed", terrain_combined.colors)
            logger.debug("✓ Saved colors_computed to cache")
    else:
        logger.debug("Computing colors with water detection...")
        terrain_combined.compute_colors(water_mask=water_mask)

    # Remove temporary mesh
    bpy.data.objects.remove(mesh_temp, do_unlink=True)

    # Create final mesh
    logger.debug("Creating final combined mesh...")
    logger.info(f"Two-tier edge settings: enabled={args.two_tier_edge}, mid_depth={args.edge_mid_depth}, "
                f"base_material={args.edge_base_material}, blend_colors={args.edge_blend_colors}")
    logger.info(f"Boundary smoothing: enabled={args.smooth_boundary}, window_size={args.smooth_boundary_window}")
    mesh_combined = terrain_combined.create_mesh(
        scale_factor=100,
        height_scale=args.height_scale,
        center_model=True,
        boundary_extension=True,
        water_mask=None,  # Already applied
        two_tier_edge=args.two_tier_edge,
        edge_mid_depth=args.edge_mid_depth,
        edge_base_material=args.edge_base_material,
        edge_blend_colors=args.edge_blend_colors,
        smooth_boundary=args.smooth_boundary,
        smooth_boundary_window=args.smooth_boundary_window if args.smooth_boundary else 5,
    )

    if mesh_combined is None:
        logger.error("Failed to create combined terrain mesh")
        return 1

    # Apply vertex colors
    logger.debug("Applying vertex colors...")

    # Diagnostic: Check for purple colors in the vertex colors
    colors_rgb = terrain_combined.colors[:, :3]  # Get RGB channels only
    is_purple = (colors_rgb[:, 0] > colors_rgb[:, 1]) & (colors_rgb[:, 0] > colors_rgb[:, 2])  # R > G and R > B
    num_purple = np.sum(is_purple)
    if num_purple > 0:
        logger.info(f"  ✓ Found {num_purple:,} purple vertices ({100*num_purple/len(colors_rgb):.2f}% of mesh)")
        purple_avg = colors_rgb[is_purple].mean(axis=0)
        logger.info(f"    Average purple color: R={purple_avg[0]:.3f} G={purple_avg[1]:.3f} B={purple_avg[2]:.3f}")
    else:
        logger.warning(f"  ⚠ No purple vertices found in mesh colors (expected with --purple-position {args.purple_position})")

    # Apply vertex colors while preserving two-tier edge boundary colors
    n_surface_vertices = len(terrain_combined.y_valid) if hasattr(terrain_combined, 'y_valid') else None
    apply_vertex_colors(mesh_combined, terrain_combined.colors, terrain_combined.y_valid, terrain_combined.x_valid, n_surface_vertices=n_surface_vertices)

    # Apply road mask and material if roads are enabled
    # Roads are ALWAYS obsidian, terrain uses vertex colors or test material
    has_roads = args.roads and road_data and "roads" in terrain_combined.data_layers
    road_layer_data = None

    if has_roads:
        logger.info("Applying road mask...")
        road_layer_data = terrain_combined.data_layers["roads"]["data"]
        apply_road_mask(mesh_combined, road_layer_data, terrain_combined.y_valid, terrain_combined.x_valid, logger)

        # Apply road smoothing if requested (smooths vertex Z coords along roads)
        if args.road_smoothing:
            logger.info(f"Smoothing road vertex elevations (radius={args.road_smoothing_radius})...")
            mesh_data = mesh_combined.data
            vertices = np.array([v.co[:] for v in mesh_data.vertices])

            smoothed_vertices = smooth_road_vertices(
                vertices=vertices,
                road_mask=road_layer_data,
                y_valid=terrain_combined.y_valid,
                x_valid=terrain_combined.x_valid,
                smoothing_radius=args.road_smoothing_radius,
            )

            apply_vertex_positions(mesh_combined, smoothed_vertices, logger)

        # Apply road offset (can be combined with smoothing or used alone)
        if args.road_offset != 0.0:
            logger.info(f"Applying road Z offset: {args.road_offset:+.1f} units...")
            mesh_data = mesh_combined.data
            vertices = np.array([v.co[:] for v in mesh_data.vertices])

            offset_vertices = offset_road_vertices(
                vertices=vertices,
                road_mask=road_layer_data,
                y_valid=terrain_combined.y_valid,
                x_valid=terrain_combined.x_valid,
                offset=args.road_offset,
            )

            apply_vertex_positions(mesh_combined, offset_vertices, logger)

    # Apply material based on roads and test_material settings
    # Roads use configurable color (default azurite); terrain uses vertex colors or test material
    if mesh_combined.data.materials:
        if has_roads:
            # Use mixed material: glossy roads + terrain (vertex colors or test material)
            terrain_style = args.test_material if args.test_material != "none" else None
            logger.info(f"Applying material: {args.road_color} roads + {terrain_style or 'vertex colors'} terrain")
            apply_terrain_with_obsidian_roads(
                mesh_combined.data.materials[0],
                terrain_style=terrain_style,
                road_color=args.road_color,
            )
        elif args.test_material != "none":
            # No roads, but test material requested
            logger.info(f"Applying test material: {args.test_material}")
            apply_test_material(mesh_combined.data.materials[0], args.test_material)
        # else: keep default colormap material (already applied by create_mesh)

    logger.info("✓ Combined terrain mesh created successfully")

    # Free the original DEM array from memory (it's no longer needed)
    # The Terrain objects have their own downsampled copies
    del dem
    gc.collect()
    logger.info("Freed original DEM from memory")

    # Setup camera and lighting
    logger.info("\n[3/4] Setting up Camera & Lighting...")
    logger.info(f"  Camera direction: {args.camera_direction}")
    logger.info(f"  Height scale: {args.height_scale}")
    logger.info(f"  Ortho scale: {args.ortho_scale}")
    logger.info(f"  Camera elevation: {args.camera_elevation}")
    camera = position_camera_relative(
        mesh_obj=mesh_combined,
        direction=args.camera_direction,
        camera_type="ORTHO",
        ortho_scale=args.ortho_scale,
        elevation=args.camera_elevation,
    )
    # Setup HDRI sky lighting (invisible but adds realistic ambient illumination)
    # HDRI sky includes a physically-simulated sun, so we use that instead of explicit lights
    if args.hdri_lighting:
        logger.info("Setting up HDRI sky lighting (provides sun + ambient)...")
        # When atmosphere is enabled, use a light gray background so fog is visible
        # (transparent background + volume = black scene)
        camera_bg = (0.88, 0.88, 0.86) if args.atmosphere else None
        setup_hdri_lighting(
            sun_elevation=args.sun_elevation,
            sun_rotation=args.sun_azimuth,
            sun_intensity=args.sun_energy / 7.0,  # Sun disc brightness
            sun_size=args.sun_angle,  # Controls shadow softness (larger = softer)
            air_density=args.air_density,  # Atmospheric scattering (lower = clearer)
            visible_to_camera=False,
            camera_background=camera_bg,
            sky_strength=args.sky_intensity,  # Overall sky ambient brightness
        )
        if args.sky_intensity != 1.0:
            logger.info(f"  Sky intensity: {args.sky_intensity} (ambient dimmed)")
        if args.air_density != 1.0:
            logger.info(f"  Air density: {args.air_density} (atmospheric scattering)")
        # Only create fill light if requested (HDRI sky provides main sun)
        lights = setup_two_point_lighting(
            sun_azimuth=args.sun_azimuth,
            sun_elevation=args.sun_elevation,
            sun_energy=0,  # Skip explicit sun - HDRI sky provides it
            sun_angle=args.sun_angle,
            fill_azimuth=args.fill_azimuth,
            fill_elevation=args.fill_elevation,
            fill_energy=args.fill_energy,
            fill_angle=args.fill_angle,
        )
    else:
        # No HDRI - use explicit sun light
        # Set world to BLACK to prevent Blender's default gray world from providing ambient light
        world = bpy.context.scene.world
        if world is None:
            world = bpy.data.worlds.new("World")
            bpy.context.scene.world = world
        world.use_nodes = True
        world.node_tree.nodes.clear()
        output = world.node_tree.nodes.new("ShaderNodeOutputWorld")
        background = world.node_tree.nodes.new("ShaderNodeBackground")
        background.inputs["Color"].default_value = (0, 0, 0, 1)  # Pure black
        background.inputs["Strength"].default_value = 0.0  # Zero emission
        world.node_tree.links.new(background.outputs["Background"], output.inputs["Surface"])
        logger.info("Set world background to black (no ambient light)")

        lights = setup_two_point_lighting(
            sun_azimuth=args.sun_azimuth,
            sun_elevation=args.sun_elevation,
            sun_energy=args.sun_energy,
            sun_angle=args.sun_angle,
            fill_azimuth=args.fill_azimuth,
            fill_elevation=args.fill_elevation,
            fill_energy=args.fill_energy,
            fill_angle=args.fill_angle,
        )

    # Setup atmospheric fog if requested
    if args.atmosphere:
        logger.info(f"Setting up atmospheric fog (density={args.atmosphere_density})...")
        setup_world_atmosphere(
            density=args.atmosphere_density,
        )

    # Create background plane if requested
    if args.background:
        logger.info(f"Creating background plane...")
        logger.info(f"  Color: {args.background_color}")
        logger.info(f"  Distance below terrain: {args.background_distance} units")
        background_plane = create_background_plane(
            camera=camera,
            mesh_or_meshes=mesh_combined,
            distance_below=args.background_distance,
            color=args.background_color,
            receive_shadows=not args.background_flat,  # Flat color ignores shadows
            flat_color=args.background_flat,
        )
        logger.info("✓ Background plane created successfully")

    logger.info("✓ Scene created successfully")

    # Determine output format (defined outside render block for summary)
    output_format = args.format.upper()
    format_ext = "jpg" if output_format == "JPEG" else "png"
    color_mode = "RGB" if output_format == "JPEG" else "RGBA"

    # Render if requested
    if not args.no_render:
        logger.info(f"\n[4/4] Rendering to {output_format}...")
        logger.info("=" * 70)

        # Configure render settings
        setup_render_settings(
            use_gpu=True,
            samples=render_samples,
            use_denoising=True,
            use_persistent_data=args.persistent_data,
            use_auto_tile=args.auto_tile,
            tile_size=args.tile_size,
        )
        memory_opts = []
        if args.auto_tile:
            memory_opts.append(f"auto-tile {args.tile_size}px")
        if args.persistent_data:
            memory_opts.append("persistent data")
        memory_info = f", {', '.join(memory_opts)}" if memory_opts else ""
        logger.info(f"Render settings configured (GPU, {render_samples:,} samples, denoising on{memory_info})")

        # Set output filename based on quality mode
        if args.print_quality:
            output_filename = f"sledding_with_xc_parks_3d_print.{format_ext}"
        else:
            output_filename = f"sledding_with_xc_parks_3d.{format_ext}"

        output_path = args.output_dir / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Rendering to {output_path} ({render_width}x{render_height})...")

        result = render_scene_to_file(
            str(output_path),
            width=render_width,
            height=render_height,
            file_format=output_format,
            color_mode=color_mode,
            compression=args.jpeg_quality if output_format == "JPEG" else 90,
        )

        if result:
            logger.info(f"✓ Render saved: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")

            # Embed sRGB color profile if requested (for print)
            if args.embed_profile:
                logger.info("Embedding sRGB ICC color profile...")
                try:
                    import subprocess
                    # Find sRGB profile (common locations)
                    profile_paths = [
                        "/usr/share/color/icc/colord/sRGB.icc",
                        "/usr/share/color/icc/sRGB.icc",
                        "/usr/share/color/icc/OpenICC/sRGB.icc",
                        "/usr/share/color/icc/ghostscript/srgb.icc",
                    ]
                    profile_path = None
                    for p in profile_paths:
                        if Path(p).exists():
                            profile_path = p
                            break

                    if profile_path:
                        # Use ImageMagick to embed the profile
                        cmd = ["convert", str(output_path), "-profile", profile_path, str(output_path)]
                        result_proc = subprocess.run(cmd, capture_output=True, text=True)
                        if result_proc.returncode == 0:
                            logger.info(f"✓ Embedded sRGB profile from {profile_path}")
                        else:
                            logger.warning(f"Failed to embed profile: {result_proc.stderr}")
                    else:
                        logger.warning("sRGB ICC profile not found. Install colord or icc-profiles package.")
                        logger.info("  Alternative: convert image.jpg -profile /path/to/sRGB.icc output.jpg")
                except FileNotFoundError:
                    logger.warning("ImageMagick not found. Install with: sudo apt install imagemagick")
                    logger.info("  Alternative: convert image.jpg -profile /path/to/sRGB.icc output.jpg")
                except Exception as e:
                    logger.warning(f"Failed to embed color profile: {e}")

            # Generate RGB histogram
            histogram_filename = output_path.stem + "_histogram.png"
            histogram_path = output_path.parent / histogram_filename
            generate_rgb_histogram(output_path, histogram_path)

            # Generate luminance (B&W) histogram
            lum_histogram_filename = output_path.stem + "_luminance.png"
            lum_histogram_path = output_path.parent / lum_histogram_filename
            generate_luminance_histogram(output_path, lum_histogram_path)

        # Print actual Blender settings used for this render
        print_render_settings_report(logger)

    logger.info("\n" + "=" * 70)
    logger.info("✓ Detroit Combined Terrain Rendering Complete!")
    logger.info("=" * 70)
    logger.info("\nSummary:")
    logger.info(f"  ✓ Loaded DEM and terrain scores")
    logger.info(f"  ✓ Created combined terrain mesh ({len(mesh_combined.data.vertices)} vertices)")
    logger.info(f"    - Base colormap: boreal_mako (forest green → blue → mint) for sledding scores (gamma={args.gamma})")
    logger.info(f"    - Overlay colormap: rocket for XC skiing scores near parks")
    logger.info(f"    - 10km zones around {len(parks) if parks else 0} park locations")
    logger.info(f"  ✓ Applied geographic transforms (WGS84 → UTM, flip, scale)")
    logger.info(f"  ✓ Detected and colored water bodies blue")
    logger.info(f"  ✓ Set up orthographic camera and lighting")
    if args.background:
        logger.info(f"  ✓ Created background plane ({args.background_color})")
    if not args.no_render:
        logger.info(f"  ✓ Rendered {render_width}×{render_height} {output_format} with {render_samples:,} samples")
        if args.embed_profile:
            logger.info(f"    (with embedded sRGB ICC profile)")
        logger.info(f"  ✓ Generated RGB + luminance histograms")
        if args.print_quality:
            logger.info(f"    ({args.print_width}×{args.print_height} inches @ {args.print_dpi} DPI - PRINT QUALITY)")
    logger.info(f"\nOutput directory: {args.output_dir}")
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
