#!/usr/bin/env python3
"""
Detroit Combined Sledding/XC Skiing Terrain Rendering Example.

This example renders a single terrain showing sledding suitability scores with
XC skiing locations highlighted using dual colormaps in Blender.

Features:
- Single terrain with dual colormaps blending sledding and XC skiing data
- Base terrain: Mako colormap (purple → yellow) showing sledding suitability
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
- docs/images/combined_render/sledding_with_xc_parks_3d.png (1920×1080, 2048 samples)
- docs/images/combined_render/sledding_with_xc_parks_3d_histogram.png (RGB histogram)
- docs/images/combined_render/sledding_with_xc_parks_3d_print.png (3000×2400 @ 300 DPI, 8192 samples)
- docs/images/combined_render/sledding_with_xc_parks_3d_print_histogram.png (RGB histogram)
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
from src.terrain.transforms import smooth_score_data, despeckle_scores
from src.terrain.scene_setup import create_background_plane
from src.terrain.blender_integration import apply_vertex_colors, apply_road_mask, apply_vertex_positions
from src.terrain.materials import apply_terrain_with_obsidian_roads, apply_test_material
from src.terrain.data_loading import load_dem_files, load_filtered_hgt_files, load_score_grid
from src.terrain.gridded_data import MemoryMonitor, TiledDataConfig, MemoryLimitExceeded
from src.terrain.roads import add_roads_layer, smooth_road_vertices, offset_road_vertices
from src.terrain.water import identify_water_by_slope
from src.terrain.cache import PipelineCache
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
# BLENDER RENDERING
# =============================================================================


def add_skiing_bumps_to_mesh(
    mesh_obj: 'bpy.types.Object',
    terrain: 'Terrain',
    parks: list,
    park_radius: float = 2500.0,
) -> None:
    """
    Add sphere markers at ski parks using average terrain height and color.

    Creates UV spheres positioned at each park location. The sphere center
    is placed at the average Z coordinate of all terrain vertices within the
    park radius, and colored with the average color of those vertices.

    Args:
        mesh_obj: Blender mesh object (terrain) to sample heights/colors from
        terrain: Terrain object with coordinate conversion
        parks: List of park dicts with 'lon', 'lat', and 'skiing_score' fields
        park_radius: Radius of spheres in meters (default: 2500m)
    """
    if not parks:
        logger.warning("No parks provided, skipping skiing bumps")
        return

    import bpy
    from mathutils import Vector, Color

    mesh = mesh_obj.data
    logger.info(f"Adding sphere markers for {len(parks)} parks (radius: {park_radius}m)...")

    # Get mesh parameters
    if not hasattr(terrain, 'model_params'):
        logger.error("Terrain object missing model_params - was create_mesh() called?")
        return

    scale_factor = terrain.model_params.get('scale_factor', 100.0)

    # Get pixel size for meter-to-mesh conversion
    dem_info = terrain.data_layers.get("dem", {})
    transformed_transform = dem_info.get("transformed_transform")
    if transformed_transform is None:
        logger.error("Transformed DEM transform not found")
        return

    pixel_size_meters = abs(transformed_transform.a)
    meters_per_mesh_unit = scale_factor * pixel_size_meters
    mesh_radius = park_radius / meters_per_mesh_unit
    mesh_radius_sq = mesh_radius ** 2

    logger.info(f"Coordinate conversion: 1 mesh unit = {meters_per_mesh_unit:.1f}m, "
                f"sphere radius = {mesh_radius:.3f} mesh units")

    # Extract park coordinates and terrain vertices
    park_lons = np.array([park['lon'] for park in parks])
    park_lats = np.array([park['lat'] for park in parks])

    logger.info(f"Extracting {len(mesh.vertices)} vertex positions...")
    vert_x = np.array([v.co.x for v in mesh.vertices])
    vert_y = np.array([v.co.y for v in mesh.vertices])
    vert_z = np.array([v.co.z for v in mesh.vertices])

    # Convert park coordinates to mesh space
    logger.info(f"Converting {len(park_lons)} park coordinates...")
    park_x_arr, park_y_arr, _ = terrain.geo_to_mesh_coords(park_lons, park_lats)

    # Filter parks within mesh bounds
    mesh_minx, mesh_maxx = vert_x.min(), vert_x.max()
    mesh_miny, mesh_maxy = vert_y.min(), vert_y.max()
    in_bounds = (
        (park_x_arr >= mesh_minx) & (park_x_arr <= mesh_maxx) &
        (park_y_arr >= mesh_miny) & (park_y_arr <= mesh_maxy) &
        ~np.isnan(park_x_arr) & ~np.isnan(park_y_arr)
    )

    logger.info(f"Creating spheres for {np.sum(in_bounds)} parks within bounds...")

    spheres_created = 0

    # For each park, create a sphere at average height with average color
    for park_x, park_y in zip(park_x_arr[in_bounds], park_y_arr[in_bounds]):
        # Find vertices within park radius
        dx = vert_x - park_x
        dy = vert_y - park_y
        dist_sq = dx*dx + dy*dy
        in_radius = dist_sq <= mesh_radius_sq

        if np.any(in_radius):
            # Calculate average Z and color
            avg_z = np.mean(vert_z[in_radius])

            # Create UV sphere at park location
            bpy.ops.mesh.primitive_uv_sphere_add(
                radius=mesh_radius,
                location=(park_x, park_y, avg_z),
            )
            sphere = bpy.context.active_object
            sphere.name = f"SkiPark_Sphere"

            # Color the sphere with average terrain color
            if hasattr(mesh, 'vertex_colors') and len(mesh.vertex_colors) > 0:
                # Get average color from terrain vertices
                colors = []
                for idx in np.where(in_radius)[0]:
                    if idx < len(mesh.vertex_colors):
                        color = mesh.vertex_colors[idx].color
                        colors.append(color)

                if colors:
                    avg_color = tuple(np.mean(colors, axis=0))
                    # Create material with average color
                    mat = bpy.data.materials.new(name="SkiParkMaterial")
                    mat.use_nodes = True
                    bsdf = mat.node_tree.nodes["Principled BSDF"]
                    bsdf.inputs[0].default_value = avg_color  # Base color

                    sphere.data.materials.append(mat)

            spheres_created += 1

            if spheres_created % 100 == 0:
                logger.info(f"  Created {spheres_created} spheres...")

    logger.info(f"✓ Created {spheres_created} sphere markers for ski parks")


def generate_rgb_histogram(image_path: Path, output_path: Path) -> Optional[Path]:
    """
    Generate and save an RGB histogram of the rendered image.

    Creates a figure with histograms for each color channel (R, G, B)
    overlaid on the same axes with transparency.

    Args:
        image_path: Path to the rendered PNG image
        output_path: Path to save the histogram image

    Returns:
        Path to saved histogram image, or None if failed
    """
    try:
        # Load image
        img = Image.open(image_path)
        img_array = np.array(img)

        logger.info(f"Generating RGB histogram for {image_path.name}...")
        logger.info(f"  Image shape: {img_array.shape}")

        # Handle RGBA vs RGB
        if img_array.ndim == 3 and img_array.shape[2] >= 3:
            r_channel = img_array[:, :, 0].flatten()
            g_channel = img_array[:, :, 1].flatten()
            b_channel = img_array[:, :, 2].flatten()
        else:
            logger.warning("Image is not RGB/RGBA, skipping histogram")
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot histograms with transparency
        bins = 256
        alpha = 0.5

        ax.hist(r_channel, bins=bins, range=(0, 255), color='red',
                alpha=alpha, label='Red', density=True)
        ax.hist(g_channel, bins=bins, range=(0, 255), color='green',
                alpha=alpha, label='Green', density=True)
        ax.hist(b_channel, bins=bins, range=(0, 255), color='blue',
                alpha=alpha, label='Blue', density=True)

        # Style
        ax.set_xlabel('Pixel Value', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'RGB Histogram: {image_path.name}', fontsize=14)
        ax.legend(loc='upper right')
        ax.set_xlim(0, 255)
        ax.grid(True, alpha=0.3)

        # Add stats annotation
        stats_text = (
            f"R: μ={np.mean(r_channel):.1f}, σ={np.std(r_channel):.1f}\n"
            f"G: μ={np.mean(g_channel):.1f}, σ={np.std(g_channel):.1f}\n"
            f"B: μ={np.mean(b_channel):.1f}, σ={np.std(b_channel):.1f}"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Save
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)

        logger.info(f"✓ RGB histogram saved: {output_path}")
        return output_path

    except Exception as e:
        logger.warning(f"Failed to generate RGB histogram: {e}")
        return None


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

  # Render at print quality (3000x2400 @ 300 DPI, 8192 samples)
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
        "--skiing-bumps",
        action="store_true",
        help="Add dome bumps at XC skiing park locations to visualize their locations",
    )

    parser.add_argument(
        "--bump-radius",
        type=float,
        default=2500.0,
        help="Radius of hemisphere bumps in meters (peak height = radius, default: 2500m, try 1000-5000)",
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

    parser.add_argument(
        "--vertex-multiplier",
        type=float,
        default=2.5,
        help="Vertices per pixel multiplier (default: 2.5). Higher = more detail, slower render. "
             "Example: 3.0 gives ~20%% more vertices than default.",
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

    # Set resolution and quality based on mode
    if args.print_quality:
        render_width = int(args.print_width * args.print_dpi)
        render_height = int(args.print_height * args.print_dpi)
        render_samples = 8192  # High quality for print
        quality_mode = "PRINT"
    else:
        render_width = 1920  # Standard HD
        render_height = 1080
        render_samples = 2048  # Good quality for screen
        quality_mode = "SCREEN"

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
    }

    color_params = {
        "colormap": "mako",
        "gamma": 0.5,
        "smooth_scores": args.smooth_scores,
        "smooth_scores_spatial": args.smooth_scores_spatial if args.smooth_scores else None,
        "despeckle_scores": args.despeckle_scores,
        "despeckle_kernel": args.despeckle_kernel if args.despeckle_scores else None,
        "roads_enabled": args.roads,
        "road_types": tuple(args.road_types) if args.roads else (),
        "road_width": args.road_width if args.roads else 0,
    }

    mesh_params = {
        "height_scale": args.height_scale,
        "scale_factor": 100,
        "center_model": True,
        "boundary_extension": True,
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
    # Base: Mako colormap for sledding suitability scores (with gamma=0.5)
    # Overlay: Rocket colormap for XC skiing scores near parks
    logger.info("\n[2/4] Creating Combined Terrain Mesh (Dual Colormap)...")
    logger.info("  Base: Sledding scores with gamma=0.5 (mako colormap - purple to yellow)")
    logger.info("  Overlay: XC skiing scores near parks (rocket colormap)")

    # For combined rendering, we need a custom terrain creation process
    # to add both sledding and XC skiing scores as separate layers
    logger.debug("Creating terrain with DEM...")
    terrain_combined = Terrain(dem, transform, dem_crs=dem_crs)

    # Add standard transforms
    terrain_combined.add_transform(reproject_raster(src_crs=dem_crs, dst_crs="EPSG:32617"))
    terrain_combined.add_transform(flip_raster(axis="horizontal"))
    terrain_combined.add_transform(scale_elevation(scale_factor=0.0001))

    # Configure downsampling (BEFORE smoothing - smoothing on 5M pixels is 200x faster than 1B)
    if target_vertices:
        logger.debug(f"Configuring for target vertices: {target_vertices:,}")
        terrain_combined.configure_for_target_vertices(target_vertices=target_vertices)

    # Apply feature-preserving smoothing AFTER downsample (runs on ~5M pixels, not 1B)
    if args.smooth:
        logger.info(f"Adding feature-preserving smoothing (spatial={args.smooth_spatial}, intensity={args.smooth_intensity or 'auto'})")
        terrain_combined.add_transform(
            feature_preserving_smooth(
                sigma_spatial=args.smooth_spatial,
                sigma_intensity=args.smooth_intensity,
            )
        )

    # Apply transforms to DEM
    logger.debug("Applying transforms to DEM...")
    terrain_combined.apply_transforms()

    # Log cache status for transforms
    if args.cache:
        transform_key = cache.compute_target_key("dem_transformed")
        cached_transform = cache.get_cached("dem_transformed")
        if cached_transform is not None:
            logger.info(f"  Cache HIT: dem_transformed (key: {transform_key[:12]}...)")
        else:
            logger.info(f"  Cache MISS: dem_transformed (key: {transform_key[:12]}..., will cache on save)")
            # Save transformed DEM state for future runs
            cache.save_target("dem_transformed", terrain_combined.dem)

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

    # Apply despeckle if requested (removes isolated speckles via median filter)
    if args.despeckle_scores:
        logger.info(f"Despeckle score data (kernel_size={args.despeckle_kernel})")
        # Despeckle sledding scores
        sledding_data = terrain_combined.data_layers["sledding"]["data"]
        terrain_combined.data_layers["sledding"]["data"] = despeckle_scores(
            sledding_data,
            kernel_size=args.despeckle_kernel,
        )
        # Despeckle XC skiing scores
        xc_data = terrain_combined.data_layers["xc_skiing"]["data"]
        terrain_combined.data_layers["xc_skiing"]["data"] = despeckle_scores(
            xc_data,
            kernel_size=args.despeckle_kernel,
        )
        logger.info("✓ Score data despeckled")

    # Create mesh for proximity calculations
    logger.debug("Creating temporary mesh for proximity calculations...")
    mesh_temp = terrain_combined.create_mesh(
        scale_factor=100,
        height_scale=args.height_scale,
        center_model=True,
        boundary_extension=True,
        water_mask=None,
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
        logger.info("  Base: Sledding scores with gamma=0.5 (mako colormap)")
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
                np.power(score, 0.5), cmap_name="mako", min_elev=0.0, max_elev=np.power(1.5, 0.5)
            ),
            base_source_layers=["sledding"],
            overlays=overlays,
        )
    elif args.roads and road_data and road_bbox:
        # Roads but no parks: just base sledding (no overlays, roads get glassy material)
        logger.info("Setting color mapping:")
        logger.info("  Base: Sledding scores with gamma=0.5 (mako colormap)")
        logger.info("  Roads: Keep terrain color, apply glassy material via mask")
        terrain_combined.set_color_mapping(
            lambda score: elevation_colormap(np.power(score, 0.5), cmap_name="mako", min_elev=0.0, max_elev=np.power(1.5, 0.5)),
            source_layers=["sledding"],
        )
    else:
        # No roads - use original blended or standard color mapping
        if parks:
            logger.info("Setting blended color mapping:")
            logger.info("  Base: Sledding scores with gamma=0.5 (mako colormap)")
            logger.info("  Overlay: XC skiing scores near parks (rocket colormap)")
            terrain_combined.set_blended_color_mapping(
                base_colormap=lambda score: elevation_colormap(
                    np.power(score, 0.5), cmap_name="mako", min_elev=0.0, max_elev=np.power(1.5, 0.5)
                ),
                base_source_layers=["sledding"],
                overlay_colormap=lambda score: elevation_colormap(
                    score, cmap_name="rocket", min_elev=0.0, max_elev=1.0
                ),
                overlay_source_layers=["xc_skiing"],
                overlay_mask=park_mask,
            )
        else:
            logger.info("No parks available - using sledding scores with gamma=0.5 (mako colormap)")
            terrain_combined.set_color_mapping(
                lambda score: elevation_colormap(np.power(score, 0.5), cmap_name="mako", min_elev=0.0, max_elev=np.power(1.5, 0.5)),
                source_layers=["sledding"],
            )

    # Detect water on downsampled DEM (fast - ~170x faster than highres detection)
    # Unscale DEM back to meters before slope detection (was scaled by 0.0001)
    logger.debug("Detecting water bodies on downsampled DEM...")
    dem_data = terrain_combined.data_layers["dem"]["transformed_data"]
    unscaled_dem = dem_data / 0.0001  # Back to meters
    water_mask = identify_water_by_slope(
        unscaled_dem,
        slope_threshold=0.01,  # Very flat areas only (water is slope ≈ 0)
        fill_holes=True,
    )

    # Compute colors
    logger.debug("Computing colors with water detection...")
    terrain_combined.compute_colors(water_mask=water_mask)

    # Log cache status for colors
    if args.cache:
        color_key = cache.compute_target_key("colors_computed")
        cached_colors = cache.get_cached("colors_computed")
        if cached_colors is not None:
            logger.info(f"  Cache HIT: colors_computed (key: {color_key[:12]}...)")
        else:
            logger.info(f"  Cache MISS: colors_computed (key: {color_key[:12]}..., caching)")
            cache.save_target("colors_computed", terrain_combined.colors)

    # Remove temporary mesh
    bpy.data.objects.remove(mesh_temp, do_unlink=True)

    # Create final mesh
    logger.debug("Creating final combined mesh...")
    mesh_combined = terrain_combined.create_mesh(
        scale_factor=100,
        height_scale=args.height_scale,
        center_model=True,
        boundary_extension=True,
        water_mask=None,  # Already applied
    )

    if mesh_combined is None:
        logger.error("Failed to create combined terrain mesh")
        return 1

    # Apply vertex colors
    logger.debug("Applying vertex colors...")
    apply_vertex_colors(mesh_combined, terrain_combined.colors, terrain_combined.y_valid, terrain_combined.x_valid)

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

    # Add skiing bumps if requested
    if args.skiing_bumps and parks:
        logger.info("\nAdding skiing bumps...")
        add_skiing_bumps_to_mesh(
            mesh_combined,
            terrain_combined,
            parks,
            park_radius=args.bump_radius,
        )

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
            sun_intensity=args.sun_energy / 7.0,  # Scale relative to default energy=7
            sun_size=args.sun_angle,  # Controls shadow softness (larger = softer)
            visible_to_camera=False,
            camera_background=camera_bg,
        )
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

    # Render if requested
    if not args.no_render:
        logger.info("\n[4/4] Rendering to PNG...")
        logger.info("=" * 70)

        # Configure render settings
        setup_render_settings(
            use_gpu=True,
            samples=render_samples,
            use_denoising=False,
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
        logger.info(f"Render settings configured (GPU, {render_samples:,} samples, denoising off{memory_info})")

        # Set output filename based on quality mode
        if args.print_quality:
            output_filename = "sledding_with_xc_parks_3d_print.png"
        else:
            output_filename = "sledding_with_xc_parks_3d.png"

        output_path = args.output_dir / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Rendering to {output_path} ({render_width}x{render_height})...")

        result = render_scene_to_file(
            str(output_path),
            width=render_width,
            height=render_height,
        )

        if result:
            logger.info(f"✓ Render saved: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")

            # Generate RGB histogram
            histogram_filename = output_path.stem + "_histogram.png"
            histogram_path = output_path.parent / histogram_filename
            generate_rgb_histogram(output_path, histogram_path)

        # Print actual Blender settings used for this render
        print_render_settings_report(logger)

    logger.info("\n" + "=" * 70)
    logger.info("✓ Detroit Combined Terrain Rendering Complete!")
    logger.info("=" * 70)
    logger.info("\nSummary:")
    logger.info(f"  ✓ Loaded DEM and terrain scores")
    logger.info(f"  ✓ Created combined terrain mesh ({len(mesh_combined.data.vertices)} vertices)")
    logger.info(f"    - Base colormap: mako (purple → yellow) for sledding scores (gamma=0.5)")
    logger.info(f"    - Overlay colormap: rocket for XC skiing scores near parks")
    logger.info(f"    - 10km zones around {len(parks) if parks else 0} park locations")
    if args.skiing_bumps and parks:
        logger.info(f"    - Added half-sphere bumps at {len(parks)} park locations (colored by skiing score)")
    logger.info(f"  ✓ Applied geographic transforms (WGS84 → UTM, flip, scale)")
    logger.info(f"  ✓ Detected and colored water bodies blue")
    logger.info(f"  ✓ Set up orthographic camera and lighting")
    if args.background:
        logger.info(f"  ✓ Created background plane ({args.background_color})")
    if not args.no_render:
        logger.info(f"  ✓ Rendered {render_width}×{render_height} PNG with {render_samples:,} samples")
        logger.info(f"  ✓ Generated RGB histogram for color analysis")
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
