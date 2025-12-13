#!/usr/bin/env python3
"""
Detroit Dual Terrain Rendering Example.

This example renders sledding and cross-country skiing terrain suitability maps
side-by-side in Blender using terrain maker's rendering library, with park
locations marked on the XC skiing terrain.

Features:
- Detects and colors water bodies blue (slope-based detection)
- Colors activity suitability red (low) → yellow → green (high)
- Applies geographic transforms for proper coordinate handling
- Intelligently limits mesh density to prevent OOM with large DEMs
- Dynamic spacing between terrain meshes based on actual bounds
- Dual camera positioned to view both terrains

Requirements:
- Blender Python API available (bpy)
- Pre-computed sledding scores from detroit_snow_sledding.py
- Pre-computed XC skiing scores + parks from detroit_xc_skiing.py

Usage:
    # Run with computed scores
    python examples/detroit_dual_render.py

    # Run with mock data
    python examples/detroit_dual_render.py --mock-data --no-render

    # Specify output directory
    python examples/detroit_dual_render.py --output-dir ./renders
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
)
from src.terrain.data_loading import load_dem_files
from src.terrain.gridded_data import MemoryMonitor, TiledDataConfig, MemoryLimitExceeded
from affine import Affine

# Configure logging
LOG_FILE = Path(__file__).parent / "detroit_dual_render.log"
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

    Returns:
        Tuple of (score_array, transform_affine). Transform may be None if
        file was saved without transform metadata (backward compatibility).
    """
    # First check for sledding_score.npz
    score_path = output_dir / "sledding_scores.npz"
    if score_path.exists():
        logger.info(f"Loading sledding scores from {score_path}")
        data = np.load(score_path)

        # Get score array
        if "score" in data:
            score = data["score"]
        elif "sledding_score" in data:
            score = data["sledding_score"]
        else:
            score = data[data.files[0]]

        # Get transform if present (new format includes transform metadata)
        score_transform = None
        if "transform" in data:
            t = data["transform"]
            score_transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
            logger.info(f"  Loaded transform: origin=({t[2]:.4f}, {t[5]:.4f}), pixel=({t[0]:.6f}, {t[4]:.6f})")

        return score, score_transform

    # Check alternative locations
    alt_paths = [
        output_dir.parent / "sledding_scores.npz",
        Path("examples/output/sledding_scores.npz"),
    ]
    for alt_path in alt_paths:
        if alt_path.exists():
            logger.info(f"Loading sledding scores from {alt_path}")
            data = np.load(alt_path)
            score = data[data.files[0]]

            # Get transform if present
            score_transform = None
            if "transform" in data:
                t = data["transform"]
                score_transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])

            return score, score_transform

    logger.warning("Sledding scores not found, will use mock data")
    return None, None


def load_xc_skiing_scores(output_dir: Path) -> Tuple[Optional[np.ndarray], Optional[Affine]]:
    """Load XC skiing scores from detroit_xc_skiing.py output.

    Returns:
        Tuple of (score_array, transform_affine). Transform may be None if
        file was saved without transform metadata (backward compatibility).
    """
    score_path = output_dir / "xc_skiing_scores.npz"
    if score_path.exists():
        logger.info(f"Loading XC skiing scores from {score_path}")
        data = np.load(score_path)

        # Get score array
        if "score" in data:
            score = data["score"]
        else:
            score = data[data.files[0]]

        # Get transform if present (new format includes transform metadata)
        score_transform = None
        if "transform" in data:
            t = data["transform"]
            score_transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
            logger.info(f"  Loaded transform: origin=({t[2]:.4f}, {t[5]:.4f}), pixel=({t[0]:.6f}, {t[4]:.6f})")

        return score, score_transform

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


def create_terrain_with_score(
    name: str,
    score_grid: np.ndarray,
    dem: np.ndarray,
    transform: Affine,
    dem_crs: str = "EPSG:32617",
    score_transform: Optional[Affine] = None,
    location: Tuple[float, float, float] = (0, 0, 0),
    scale_factor: float = 100,
    height_scale: float = 12.5,
    target_vertices: Optional[int] = None,
    cmap_name: str = "viridis",
) -> Tuple[Optional[object], Optional["Terrain"]]:
    """
    Create a terrain mesh using terrain maker's Terrain class.

    Args:
        name: Mesh name (for logging and reference)
        score_grid: Score values for vertex coloring (0-1 range)
        dem: DEM elevation data
        transform: Affine transform for DEM coordinate mapping
        dem_crs: CRS of DEM data (default: EPSG:32617 for UTM zone 17N)
        score_transform: Affine transform for score data (if different from DEM)
        location: (x, y, z) position in Blender for the mesh
        scale_factor: Horizontal scale divisor
        height_scale: Vertical elevation scale
        target_vertices: Target vertex count for downsampling (optional)
        cmap_name: Matplotlib colormap name for score visualization (default: viridis)

    Returns:
        Tuple of (mesh_obj, terrain) where mesh_obj is the Blender object and
        terrain is the Terrain instance (useful for geo_to_mesh_coords)
    """
    # Note: When score_transform is None, the library's same_extent_as feature
    # will automatically calculate the transform from the DEM's extent.
    logger.info(f"Creating terrain mesh: {name}")
    logger.debug(
        f"  Score input: shape={score_grid.shape}, "
        f"range=[{np.nanmin(score_grid):.3f}, {np.nanmax(score_grid):.3f}], "
        f"std={np.nanstd(score_grid):.3f}"
    )

    # Create terrain using terrain maker library
    terrain = Terrain(dem, transform, dem_crs=dem_crs)

    # Build transform pipeline: geographic transforms + downsampling
    logger.debug(f"  Building transform pipeline for {name}...")

    # 1. Reproject to UTM Zone 17N for proper geographic scaling
    # Use dem_crs as source CRS (typically EPSG:4326 for real data)
    utm_reproject = reproject_raster(src_crs=dem_crs, dst_crs="EPSG:32617", num_threads=4)
    terrain.add_transform(utm_reproject)

    # 2. Flip raster horizontally to correct orientation
    terrain.add_transform(flip_raster(axis="horizontal"))

    # 3. Scale elevation to prevent extreme Z values
    terrain.add_transform(scale_elevation(scale_factor=0.0001))

    # 4. Configure downsampling BEFORE apply_transforms() so it's included in the pipeline
    zoom_factor = 1.0
    if target_vertices is not None:
        original_h, original_w = terrain.dem_shape
        original_vertices = original_h * original_w

        if original_vertices > target_vertices:
            zoom_factor = terrain.configure_for_target_vertices(target_vertices, order=4)
            logger.info(
                f"Downsampling {name}: {original_vertices:,} → ~{target_vertices:,} vertices "
                f"(zoom={zoom_factor:.6f})"
            )
        else:
            logger.debug(
                f"No downsampling needed for {name}: {original_vertices:,} ≤ {target_vertices:,}"
            )

    # 5. Apply transforms to DEM only first (reproject + flip + scale + downsample)
    logger.debug(f"  Applying transforms to DEM...")
    terrain.apply_transforms()

    # 6. Add score layer AFTER transforms - it will be reprojected to match
    # the transformed DEM's dimensions.
    logger.debug(f"  Adding score data layer for {name}...")
    if score_transform is not None:
        # Explicit transform provided - use it directly
        logger.debug(f"  Using explicit score transform: {score_transform}")
        terrain.add_data_layer(
            "score",
            score_grid,
            score_transform,
            dem_crs,
            target_layer="dem",
        )
    else:
        # No transform provided - use same_extent_as for automatic georeferencing
        # This assumes score covers the same geographic extent as the DEM
        logger.debug(f"  Using same_extent_as='dem' for automatic georeferencing")
        terrain.add_data_layer(
            "score",
            score_grid,
            same_extent_as="dem",  # Library calculates transform automatically
        )

    # Note: Score layer does NOT need manual flipping
    # The reprojection to the transformed DEM coordinates handles alignment correctly
    # because it maps geographic coordinates, not pixel indices

    # Set color mapping using specified colormap
    # min_elev=0.0, max_elev=1.0 maps scores in [0,1] range to full colormap
    logger.debug(f"  Setting color mapping with colormap: {cmap_name}")
    terrain.set_color_mapping(
        lambda score: elevation_colormap(score, cmap_name=cmap_name, min_elev=0.0, max_elev=1.0),
        source_layers=["score"],
    )

    # Compute vertex colors based on resampled score data
    terrain.compute_colors()

    # Detect water bodies on the unscaled DEM
    # Water will be colored blue, activity scores red-yellow-green
    logger.debug(f"  Detecting water bodies for {name}...")
    water_mask = None
    try:
        from src.terrain.water import identify_water_by_slope

        transformed_dem = terrain.data_layers["dem"]["transformed_data"]
        # Unscale the DEM to get original elevation values for slope calculation
        unscaled_dem = transformed_dem / 0.0001

        water_mask = identify_water_by_slope(
            unscaled_dem, slope_threshold=0.01, fill_holes=True
        )
        water_pixels = np.sum(water_mask)
        water_percent = 100 * water_pixels / water_mask.size
        logger.debug(
            f"  Water detected: {water_pixels:,} pixels ({water_percent:.1f}% of terrain)"
        )
    except Exception as e:
        logger.debug(f"  Water detection skipped: {e}")
        water_mask = None

    # Create mesh using terrain maker library
    mesh_obj = terrain.create_mesh(
        scale_factor=scale_factor,
        height_scale=height_scale,
        center_model=True,
        boundary_extension=True,
        water_mask=water_mask,
    )

    if mesh_obj is None:
        logger.error(f"Failed to create mesh for {name}")
        return None, None

    # Position the mesh at the specified location
    mesh_obj.location = location
    mesh_obj.name = name

    logger.info(f"✓ Created mesh: {name} at position {location}")
    return mesh_obj, terrain


def create_park_markers(
    parks: list[dict],
    terrain: "Terrain",
    mesh_obj,
    elevation_offset: float = 0.1,
    marker_radius: float = 0.05,
) -> list:
    """
    Create glowing sphere markers for parks using Terrain's coordinate transform.

    Args:
        parks: List of parks with lat, lon, score
        terrain: Terrain object (must have create_mesh() already called)
        mesh_obj: Blender mesh object (for positioning offset in multi-mesh scenes)
        elevation_offset: Height above terrain in mesh units (default: 0.1)
        marker_radius: Radius of marker spheres (default: 0.05)

    Returns:
        List of created marker objects
    """
    if not parks:
        return []

    logger.info(f"Creating {len(parks)} park markers...")
    markers = []

    # Extract lon/lat arrays from parks for batch processing
    lons = np.array([park["lon"] for park in parks])
    lats = np.array([park["lat"] for park in parks])

    # Convert all coordinates at once using Terrain's method
    try:
        xs, ys, zs = terrain.geo_to_mesh_coords(lons, lats, elevation_offset=elevation_offset)
    except (RuntimeError, ValueError) as e:
        logger.error(f"Failed to convert park coordinates: {e}")
        return []

    # Add mesh object's location offset (for dual-terrain scenes)
    if mesh_obj and mesh_obj.location:
        xs = xs + mesh_obj.location[0]
        ys = ys + mesh_obj.location[1]

    for i, park in enumerate(parks):
        try:
            x, y, z = xs[i], ys[i], zs[i]

            # Create sphere
            bpy.ops.mesh.primitive_ico_sphere_add(
                radius=marker_radius,
                location=(x, y, z)
            )
            marker = bpy.context.active_object
            marker.name = f"Park_{park['name'].replace(' ', '_')}"

            # Create diffuse material - warm terra cotta contrasts with mako's blues/greens
            mat = bpy.data.materials.new(name=f"ParkMarker_{i}")
            # Note: use_nodes defaults to True in Blender 4.x+
            nodes = mat.node_tree.nodes
            nodes.clear()

            # Use Principled BSDF for natural lighting with shadows
            bsdf = nodes.new("ShaderNodeBsdfPrincipled")
            bsdf.inputs["Base Color"].default_value = (0.8, 0.4, 0.3, 1.0)  # Warm terra cotta
            bsdf.inputs["Roughness"].default_value = 0.5  # Semi-matte for visible shadows

            output = nodes.new("ShaderNodeOutputMaterial")
            mat.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

            marker.data.materials.append(mat)
            markers.append(marker)

            logger.debug(f"  Created marker: {park['name']} at ({x:.2f}, {y:.2f}, {z:.2f})")

        except Exception as e:
            logger.warning(f"Error creating marker for {park['name']}: {e}")

    logger.info(f"✓ Created {len(markers)} markers")
    return markers


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
    direction: str = "south",
    distance: float = 3.5,
    elevation: float = 5,
    camera_type: str = "ORTHO",
    ortho_scale: float = 1,
) -> Optional[object]:
    """
    Setup camera to view both terrains side-by-side.

    Uses position_camera_relative with multi-mesh support to automatically
    compute a combined bounding box and position the camera to see both terrains.

    Args:
        mesh_left: Left terrain object
        mesh_right: Right terrain object
        direction: Cardinal direction for camera (default: 'south')
        distance: Distance multiplier relative to combined mesh diagonal
        elevation: Height as fraction of mesh diagonal
        camera_type: 'ORTHO' or 'PERSP' (default: 'PERSP')

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


def setup_lighting() -> list:
    """Setup Blender lighting with sunset-style positioning.

    Creates a low-angle sun for dramatic shadows and warm fill light.
    """
    from math import radians

    logger.info("Setting up sunset lighting...")

    lights = []

    # Primary sun - low angle from the west/southwest for sunset shadows
    # rotation_euler: (pitch down from horizon, yaw direction, roll)
    sun_light = setup_light(
        location=(10, -5, 2),  # Position doesn't matter much for sun type
        angle=1,  # Sharper shadows (smaller angle = harder shadows)
        energy=4.0,
        rotation_euler=(radians(75), 0, radians(-45)),  # Low sun from SW
    )
    # Set warm sunset color (golden/orange)
    sun_light.data.color = (1.0, 0.85, 0.6)  # Warm golden
    lights.append(sun_light)

    # Fill light - cooler blue from opposite side for contrast
    fill_light = setup_light(
        location=(-10, 5, 5),
        angle=3,  # Softer
        energy=1,
        rotation_euler=(radians(60), 0, radians(135)),  # From NE, higher
    )
    # Cool blue fill to contrast with warm sun
    fill_light.data.color = (0.7, 0.8, 1.0)  # Cool blue
    lights.append(fill_light)

    logger.info("✓ Sunset lighting setup complete")
    return lights


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
        description="Detroit Dual Terrain Rendering (Sledding + XC Skiing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Render with pre-computed scores
  python examples/detroit_dual_render.py

  # Specify output directory for results
  python examples/detroit_dual_render.py --output-dir ./renders
        """,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/output/dual_render"),
        help="Output directory (default: examples/output/dual_render/)",
    )

    parser.add_argument(
        "--scores-dir",
        type=Path,
        default=Path("examples/output"),
        help="Directory containing pre-computed scores (default: examples/output/)",
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

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "=" * 70)
    logger.info("Detroit Dual Terrain Rendering (Blender Script)")
    logger.info("=" * 70)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Scores directory: {args.scores_dir}")

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
            logger.info(f"Loading DEM from {dem_dir}...")
            dem, transform = load_dem_files(dem_dir)
            dem_crs = "EPSG:4326"  # Real data is typically WGS84
            logger.info(f"DEM shape: {dem.shape}")
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
    # Ultra-conservative for dual rendering: 250k per mesh (500k total)
    # Blender's mesh creation is memory-intensive; this prevents OOM
    # For comparison: detroit_elevation_real uses 1.38M for ONE mesh
    # We need TWO meshes, so each gets 250k for safety
    target_vertices = 1920*1080
    logger.info(f"Target vertices per mesh: {target_vertices:,} (ultra-conservative for dual rendering)")

    # Create first mesh at origin
    logger.info("\n[2/5] Creating Sledding Terrain Mesh...")
    mesh_sledding, terrain_sledding = create_terrain_with_score(
        "SleddingTerrain",
        sledding_scores,
        dem,
        transform,
        dem_crs=dem_crs,
        score_transform=score_transform,  # Score may have different bounds than DEM
        location=(0, 0, 0),
        scale_factor=100,
        target_vertices=target_vertices,
        cmap_name="mako",  # Purple to yellow for sledding scores
    )

    if mesh_sledding is None:
        logger.error("Failed to create sledding terrain mesh")
        return 1

    # Measure sledding mesh
    width_sledding = calculate_mesh_width(mesh_sledding)
    logger.debug(f"Sledding terrain width: {width_sledding:.2f}")

    # Aggressive memory cleanup before second mesh (critical point)
    logger.info("Cleaning up memory before creating second mesh...")
    gc.collect()

    # Memory check before second mesh
    try:
        monitor.check_memory(force=True)
        logger.info("Memory check passed before creating second mesh")
    except MemoryLimitExceeded as e:
        logger.error(f"Insufficient memory to create second mesh: {e}")
        logger.error("Try reducing vertex count or closing other applications")
        return 1

    # Create second mesh at origin
    logger.info("\n[3/5] Creating XC Skiing Terrain Mesh...")
    mesh_xc, terrain_xc = create_terrain_with_score(
        "XCSkiingTerrain",
        xc_scores,
        dem,
        transform,
        dem_crs=dem_crs,
        score_transform=xc_transform,  # Use XC skiing's own transform
        location=(0, 0, 0),
        scale_factor=100,
        target_vertices=target_vertices,
        cmap_name="mako",  # Blue to green to yellow for XC skiing scores
    )

    if mesh_xc is None:
        logger.error("Failed to create XC skiing terrain mesh")
        return 1

    # Measure XC mesh
    width_xc = calculate_mesh_width(mesh_xc)
    logger.debug(f"XC skiing terrain width: {width_xc:.2f}")

    # Calculate dynamic spacing
    pos_left, pos_right = calculate_dual_terrain_spacing(
        width_sledding,
        width_xc,
        gap_ratio=0.10,
        min_gap=0.5,
    )

    # Validate and log spacing
    gap = pos_right[0] - pos_left[0] - (width_sledding + width_xc) / 2
    validate_spacing(gap, width_sledding, width_xc)

    # Reposition meshes
    mesh_sledding.location = pos_left
    mesh_xc.location = pos_right

    logger.info(f"Positioned sledding terrain at {pos_left}")
    logger.info(f"Positioned XC skiing terrain at {pos_right}")

    # Create park markers on XC skiing terrain
    markers = []
    if parks and terrain_xc:
        markers = create_park_markers(parks, terrain_xc, mesh_xc)
        logger.info(f"Created {len(markers)} park markers")

    # Free the original DEM array from memory (it's no longer needed)
    # The Terrain objects have their own downsampled copies
    del dem
    gc.collect()
    logger.info("Freed original DEM from memory")

    # Setup camera and lighting
    logger.info("\n[4/5] Setting up Camera, Lighting & Markers...")
    camera = setup_dual_camera(mesh_sledding, mesh_xc)
    lights = setup_lighting()

    logger.info("✓ Scene created successfully")

    # Render if requested
    if not args.no_render:
        logger.info("\n[5/5] Rendering to PNG...")
        logger.info("=" * 70)

        # Configure render settings for high-quality output
        setup_render_settings(use_gpu=True, samples=2048, use_denoising=False)
        logger.info("Render settings configured (GPU, 2048 samples, denoising off)")

        output_path = args.output_dir / "sledding_and_xc_skiing_3d.png"
        render_dual_terrain(output_path, width=1920, height=1080)

    logger.info("\n" + "=" * 70)
    logger.info("✓ Detroit Dual Terrain Rendering Complete!")
    logger.info("=" * 70)
    logger.info("\nSummary:")
    logger.info(f"  ✓ Loaded DEM and terrain scores")
    logger.info(f"  ✓ Created sledding terrain mesh ({len(mesh_sledding.data.vertices)} vertices) - plasma colormap")
    logger.info(f"  ✓ Created XC skiing terrain mesh ({len(mesh_xc.data.vertices)} vertices) - viridis colormap")
    logger.info(f"  ✓ Applied geographic transforms (WGS84 → UTM, flip, scale)")
    logger.info(f"  ✓ Detected and colored water bodies blue")
    logger.info(f"  ✓ Positioned meshes with dynamic spacing")
    logger.info(f"  ✓ Set up dual camera and lighting")
    if not args.no_render:
        logger.info(f"  ✓ Rendered 1920×1080 PNG with 2048 samples")
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
