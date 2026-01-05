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
- docs/images/combined_render/sledding_with_xc_parks_3d_print.png (3000×2400 @ 300 DPI, 8192 samples)
- docs/images/combined_render/sledding_with_xc_parks_3d.blend (Blender file)

Usage:
    # Run with computed scores (standard quality)
    python examples/detroit_combined_render.py

    # Run at print quality (10x8 inches @ 300 DPI)
    python examples/detroit_combined_render.py --print-quality

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
    setup_light,
    setup_render_settings,
    render_scene_to_file,
    reproject_raster,
    flip_raster,
    scale_elevation,
    feature_preserving_smooth,
)
from src.terrain.scene_setup import create_background_plane
from src.terrain.blender_integration import apply_vertex_colors, apply_road_mask
from src.terrain.materials import apply_terrain_with_obsidian_roads, apply_test_material
from src.terrain.data_loading import load_dem_files, load_score_grid
from src.terrain.gridded_data import MemoryMonitor, TiledDataConfig, MemoryLimitExceeded
from src.terrain.roads import add_roads_layer, smooth_road_vertices
from src.terrain.water import identify_water_by_slope
from examples.detroit_roads import get_roads_tiled
from affine import Affine

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


def calculate_mesh_width(mesh_obj) -> float:
    """
    Calculate mesh width from bound_box (X dimension).

    Args:
        mesh_obj: Blender mesh object

    Returns:
        Width in Blender units
    """
    bounds = mesh_obj.bound_box
    xs = [v[0] for v in bounds]
    return max(xs) - min(xs)


def calculate_dual_terrain_spacing(
    width_left: float,
    width_right: float,
    gap_ratio: float = 0.10,
    min_gap: float = 0.5,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Calculate symmetric positions for two terrains with dynamic spacing.

    Args:
        width_left: Width of left terrain mesh
        width_right: Width of right terrain mesh
        gap_ratio: Gap as fraction of combined width (default: 10%)
        min_gap: Minimum gap in Blender units (default: 0.5)

    Returns:
        Tuple of (left_position, right_position) as (x, y, z) tuples
    """
    combined_width = width_left + width_right
    gap = max(combined_width * gap_ratio, min_gap)

    left_x = -(width_left / 2 + gap / 2)
    right_x = (width_right / 2 + gap / 2)

    return (left_x, 0, 0), (right_x, 0, 0)


def validate_spacing(
    gap: float,
    width_left: float,
    width_right: float,
    min_gap_warn: float = 0.3,
    max_gap_warn: float = 5.0,
) -> None:
    """
    Validate spacing and log warnings if suboptimal.

    Args:
        gap: Calculated gap between meshes
        width_left: Width of left mesh
        width_right: Width of right mesh
        min_gap_warn: Warn if gap below this threshold (default: 0.3)
        max_gap_warn: Warn if gap above this threshold (default: 5.0)
    """
    if gap < min_gap_warn:
        logger.warning(
            f"Terrain spacing is tight: gap={gap:.2f} < {min_gap_warn:.2f}. "
            f"Meshes may appear to overlap from certain angles."
        )

    if gap > max_gap_warn:
        logger.warning(
            f"Terrain spacing is wide: gap={gap:.2f} > {max_gap_warn:.2f}. "
            f"Consider adjusting camera to frame both terrains."
        )

    logger.info(
        f"Terrain spacing: left_width={width_left:.2f}, "
        f"right_width={width_right:.2f}, gap={gap:.2f}"
    )


def setup_dual_camera(
    mesh_left,
    mesh_right,
    direction: str = "above",
    distance: float = 3.5,
    elevation: float = 5,
    camera_type: str = "ORTHO",
    ortho_scale: float = 1.5,
) -> Optional[object]:
    """
    Setup camera to view both terrains side-by-side.

    Uses position_camera_relative with multi-mesh support to automatically
    compute a combined bounding box and position the camera to see both terrains.

    Args:
        mesh_left: Left terrain object
        mesh_right: Right terrain object
        direction: Cardinal direction for camera (default: 'above')
        distance: Distance multiplier relative to combined mesh diagonal
        elevation: Height as fraction of mesh diagonal
        camera_type: 'ORTHO' or 'PERSP' (default: 'ORTHO')
        ortho_scale: Orthographic scale multiplier (default: 1.5)

    Returns:
        Camera object positioned to view both terrains
    """
    logger.info("Setting up dual camera...")

    # Collect meshes (filter out None values)
    meshes = [m for m in [mesh_left, mesh_right] if m is not None]

    if not meshes:
        logger.error("No valid meshes provided for camera setup")
        return None

    # Use position_camera_relative with list of meshes
    # This automatically computes combined bounding box and positions camera
    camera = position_camera_relative(
        meshes,
        direction=direction,
        distance=distance,
        elevation=elevation,
        camera_type=camera_type,
        ortho_scale=ortho_scale,
        sun_angle=0,  # We setup lighting separately
        sun_energy=0,
    )

    logger.info(f"✓ Camera positioned {direction} of both terrains ({camera_type})")
    return camera


def setup_lighting(
    sun_azimuth: float = 225.0,
    sun_elevation: float = 30.0,
    sun_energy: float = 7.0,
    sun_angle: float = 1.0,
    fill_azimuth: float = 45.0,
    fill_elevation: float = 60.0,
    fill_energy: float = 1.0,
    fill_angle: float = 3.0,
) -> list:
    """Setup Blender lighting with intuitive sun positioning.

    Creates a primary sun for shadows and a fill light for softer contrast.

    Args:
        sun_azimuth: Direction sun comes FROM in degrees (0=N, 90=E, 180=S, 270=W)
        sun_elevation: Sun angle above horizon in degrees (0=horizon, 90=overhead)
        sun_energy: Sun light strength/energy
        sun_angle: Sun angular size in degrees (smaller=sharper shadows)
        fill_azimuth: Direction fill light comes FROM in degrees
        fill_elevation: Fill light angle above horizon in degrees
        fill_energy: Fill light strength/energy
        fill_angle: Fill light angular size in degrees (softer than sun)

    Returns:
        List of light objects
    """
    logger.info(f"Setting up lighting (sun: azimuth={sun_azimuth}°, elevation={sun_elevation}°)...")

    lights = []

    # Primary sun - uses intuitive azimuth/elevation positioning
    sun_light = setup_light(
        angle=sun_angle,  # Sharper shadows (smaller angle = harder shadows)
        energy=sun_energy,
        azimuth=sun_azimuth,
        elevation=sun_elevation,
    )
    # Set warm sunset color (golden/orange)
    sun_light.data.color = (1.0, 0.85, 0.6)  # Warm golden
    lights.append(sun_light)

    # Fill light - cooler blue from opposite side for contrast
    fill_light = setup_light(
        angle=fill_angle,  # Softer
        energy=fill_energy,
        azimuth=fill_azimuth,
        elevation=fill_elevation,
    )
    # Cool blue fill to contrast with warm sun
    fill_light.data.color = (0.7, 0.8, 1.0)  # Cool blue
    lights.append(fill_light)

    logger.info("✓ Lighting setup complete")
    return lights


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


def render_dual_terrain(
    output_path: Path,
    width: int = 1920,
    height: int = 1080,
) -> bool:
    """
    Render the scene to file using library function.

    Args:
        output_path: Output PNG path
        width: Render width
        height: Render height

    Returns:
        True if successful
    """
    logger.info(f"Rendering to {output_path} ({width}x{height})...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use library function to render
    try:
        render_scene_to_file(
            str(output_path),
            width=width,
            height=height,
        )

        if output_path.exists():
            logger.info(f"✓ Render saved: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
            return True
        else:
            logger.warning("Render output not created")
            return False
    except Exception as e:
        logger.error(f"Render failed: {e}")
        return False


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
        help="Render at print quality: 10x8 inches @ 300 DPI (3000x2400 px) with 8192 samples",
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
        default=1.0,
        help="Fill light strength/energy (default: 1.0)",
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

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set resolution and quality based on mode
    if args.print_quality:
        render_width = 3000  # 10 inches @ 300 DPI
        render_height = 2400  # 8 inches @ 300 DPI
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
    logger.info(f"  Resolution: {render_width}×{render_height} ({render_width/300:.1f}×{render_height/300:.1f} inches @ 300 DPI)")
    logger.info(f"  Samples: {render_samples:,}")
    if args.background:
        logger.info(f"Background plane: ENABLED")
        logger.info(f"  Color: {args.background_color}")
        logger.info(f"  Distance below terrain: {args.background_distance} units")

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
            # Filter HGT files to load only northern tiles (N41 and above)
            # This focuses on areas with better snow coverage (Detroit metro and north)
            # Tiles range from N37-N46; loading N41+ removes the southern ~40% of extent
            import re
            import rasterio
            from rasterio.merge import merge
            from tqdm import tqdm

            min_latitude = 41  # Load N41 and above (N41, N42, N43, N44, N45, N46)

            all_hgt_files = sorted(dem_dir.glob("*.hgt"))
            logger.info(f"Found {len(all_hgt_files)} total HGT files in {dem_dir}")

            # Filter by latitude from filename (e.g., N42W083.hgt -> lat=42)
            filtered_files = []
            for f in all_hgt_files:
                match = re.match(r'N(\d+)W\d+\.hgt', f.name)
                if match:
                    lat = int(match.group(1))
                    if lat >= min_latitude:
                        filtered_files.append(f)

            logger.info(f"Loading {len(filtered_files)} HGT files with latitude >= N{min_latitude}")
            logger.info(f"  (focusing on Detroit metro and northern areas with better snow)")

            if not filtered_files:
                raise ValueError(f"No HGT files found with latitude >= N{min_latitude}")

            # Open filtered files
            dem_datasets = []
            with tqdm(filtered_files, desc="Opening filtered DEM files") as pbar:
                for file in pbar:
                    try:
                        ds = rasterio.open(file)
                        dem_datasets.append(ds)
                        pbar.set_postfix({"opened": len(dem_datasets)})
                    except Exception as e:
                        logger.warning(f"Failed to open {file}: {e}")
                        continue

            if not dem_datasets:
                raise ValueError("No valid DEM files could be opened")

            logger.info(f"Successfully opened {len(dem_datasets)} DEM files")

            # Merge datasets
            try:
                with rasterio.Env():
                    merged_dem, transform = merge(dem_datasets)
                    dem = merged_dem[0]  # Extract first band
                    dem_crs = "EPSG:4326"  # Real data is typically WGS84

                    logger.info(f"Successfully merged filtered DEMs:")
                    logger.info(f"  Shape: {dem.shape}")
                    logger.info(f"  Value range: {np.nanmin(dem):.2f} to {np.nanmax(dem):.2f}")
                    logger.info(f"  Transform: {transform}")
            finally:
                # Clean up
                for ds in dem_datasets:
                    ds.close()
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
    target_vertices = int(np.floor(render_width * render_height * 2.5))
    logger.info(f"Target vertices: {target_vertices:,} ({quality_mode} resolution)")

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

            for i, v in enumerate(mesh_data.vertices):
                v.co = smoothed_vertices[i]

            mesh_data.update()
            logger.info("✓ Road vertex elevations smoothed")

    # Apply material based on roads and test_material settings
    # Roads are always obsidian; terrain uses vertex colors or test material
    if mesh_combined.data.materials:
        if has_roads:
            # Use mixed material: obsidian roads + terrain (vertex colors or test material)
            terrain_style = args.test_material if args.test_material != "none" else None
            logger.info(f"Applying material: obsidian roads + {terrain_style or 'vertex colors'} terrain")
            apply_terrain_with_obsidian_roads(mesh_combined.data.materials[0], terrain_style=terrain_style)
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
    lights = setup_lighting(
        sun_azimuth=args.sun_azimuth,
        sun_elevation=args.sun_elevation,
        sun_energy=args.sun_energy,
        sun_angle=args.sun_angle,
        fill_azimuth=args.fill_azimuth,
        fill_elevation=args.fill_elevation,
        fill_energy=args.fill_energy,
        fill_angle=args.fill_angle,
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
            receive_shadows=True,  # Always receive shadows; distance controls shadow appearance
        )
        logger.info("✓ Background plane created successfully")

    logger.info("✓ Scene created successfully")

    # Render if requested
    if not args.no_render:
        logger.info("\n[4/4] Rendering to PNG...")
        logger.info("=" * 70)

        # Configure render settings
        setup_render_settings(use_gpu=True, samples=render_samples, use_denoising=False)
        logger.info(f"Render settings configured (GPU, {render_samples:,} samples, denoising off)")

        # Set output filename based on quality mode
        if args.print_quality:
            output_filename = "sledding_with_xc_parks_3d_print.png"
        else:
            output_filename = "sledding_with_xc_parks_3d.png"

        output_path = args.output_dir / output_filename
        render_dual_terrain(output_path, width=render_width, height=render_height)

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
        if args.print_quality:
            logger.info(f"    ({render_width/300:.1f}×{render_height/300:.1f} inches @ 300 DPI - PRINT QUALITY)")
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
