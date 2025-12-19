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
)
from src.terrain.scene_setup import create_background_plane
from src.terrain.blender_integration import apply_vertex_colors
from src.terrain.data_loading import load_dem_files
from src.terrain.gridded_data import MemoryMonitor, TiledDataConfig, MemoryLimitExceeded
from src.terrain.roads import add_roads_to_scene
from examples.detroit_roads import get_roads
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

    Returns:
        Tuple of (score_array, transform_affine). Transform may be None if
        file was saved without transform metadata (backward compatibility).
    """
    # First check for sledding/sledding_scores.npz (new location matching XC skiing pattern)
    score_path = output_dir / "sledding" / "sledding_scores.npz"
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

    # Check alternative locations (backward compatibility)
    alt_paths = [
        output_dir / "sledding_scores.npz",  # Old flat structure
        Path("examples/output/sledding_scores.npz"),  # Legacy hardcoded location
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
    height_scale: float = 30.0,
    target_vertices: Optional[int] = None,
    cmap_name: str = "viridis",
    parks: Optional[list[dict]] = None,
    park_radius_meters: float = 10_000,
    park_cluster_threshold: float = 500,
    base_cmap_name: str = "michigan",
    overlay_cmap_name: str = "rocket",
) -> Tuple[Optional[object], Optional["Terrain"]]:
    """
    Create a terrain mesh using terrain maker's Terrain class.

    Supports dual colormaps for proximity-based visualization (e.g., elevation
    for general terrain, scores near parks).

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
        cmap_name: Colormap name for single-colormap mode (default: viridis)
        parks: Optional list of parks for dual colormap mode. If provided,
            enables dual colormaps with score overlay near parks.
        park_radius_meters: Radius around parks to show score overlay (default: 10000m = 10km)
        park_cluster_threshold: Distance threshold for merging nearby parks (default: 500m)
        base_cmap_name: Base colormap for general terrain (default: michigan)
        overlay_cmap_name: Overlay colormap for park zones (default: rocket)

    Returns:
        Tuple of (mesh_obj, terrain) where mesh_obj is the Blender object and
        terrain is the Terrain instance (useful for proximity_mask calculations)
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

    # Apply colormapping (single or dual depending on parks availability)
    if parks:
        # Dual colormap mode: elevation base + score overlay near parks
        logger.info(f"  Using dual colormaps for {name}:")
        logger.info(f"    Base: {base_cmap_name} (elevation)")
        logger.info(f"    Overlay: {overlay_cmap_name} (scores near parks)")

        # Create mesh first (needed for proximity mask computation)
        logger.debug(f"  Creating mesh for proximity calculations...")
        mesh_obj_temp = terrain.create_mesh(
            scale_factor=scale_factor,
            height_scale=height_scale,
            center_model=True,
            boundary_extension=True,
            water_mask=None,  # Will apply water coloring separately
        )

        if mesh_obj_temp is None:
            logger.error(f"Failed to create temporary mesh for {name}")
            return None, None

        # Compute proximity mask for park zones
        park_lons = np.array([p["lon"] for p in parks])
        park_lats = np.array([p["lat"] for p in parks])

        logger.info(f"  Computing proximity mask ({len(parks)} parks)...")
        park_mask = terrain.compute_proximity_mask(
            park_lons,
            park_lats,
            radius_meters=park_radius_meters,
            cluster_threshold_meters=park_cluster_threshold,
        )

        # Set dual colormaps
        terrain.set_blended_color_mapping(
            base_colormap=lambda elev: elevation_colormap(
                elev, cmap_name=base_cmap_name, min_elev=0.0, max_elev=None
            ),
            base_source_layers=["dem"],
            overlay_colormap=lambda score: elevation_colormap(
                score, cmap_name=overlay_cmap_name, min_elev=0.0, max_elev=1.0
            ),
            overlay_source_layers=["score"],
            overlay_mask=park_mask,
        )

        logger.debug(f"  Dual colormap configured: {np.sum(park_mask)} vertices in overlay zones")
        # Will compute colors with water detection below

    else:
        # Single colormap mode: score-based coloring for entire terrain
        logger.debug(f"  Setting color mapping with colormap: {cmap_name}")
        terrain.set_color_mapping(
            lambda score: elevation_colormap(score, cmap_name=cmap_name, min_elev=0.0, max_elev=1.0),
            source_layers=["score"],
        )

    # Detect water bodies on HIGH-RES DEM (before downsampling)
    # Water will be colored University of Michigan blue (#00274C) in both modes
    # NOTE: This must be called AFTER apply_transforms() so the library can access
    # both the original high-res DEM and the transformed/downsampled version
    logger.debug(f"  Detecting water bodies on high-resolution DEM for {name}...")
    water_mask = terrain.detect_water_highres(
        slope_threshold=0.01,  # Low threshold for nearly-flat water
        fill_holes=True,
        scale_factor=0.0001,  # Match the scale_elevation transform
    )
    # If water detection fails, the method will raise an exception
    # We don't silently ignore water detection errors

    # Compute colors with water detection
    # In blended mode, water detection happens internally
    # In single mode, it happens in create_mesh()
    if parks:
        logger.debug(f"  Computing blended colors with water detection...")
        terrain.compute_colors(water_mask=water_mask)
        logger.debug(f"  ✓ Blended colors with water detection computed")

        # Remove temporary mesh and create final mesh
        logger.debug(f"  Removing temporary mesh and creating final mesh...")
        bpy.data.objects.remove(mesh_obj_temp, do_unlink=True)

        # Create final mesh with colors already applied (including water)
        mesh_obj = terrain.create_mesh(
            scale_factor=scale_factor,
            height_scale=height_scale,
            center_model=True,
            boundary_extension=True,
            water_mask=None,  # Water already applied in compute_colors()
        )

        if mesh_obj is None:
            logger.error(f"Failed to create final mesh for {name}")
            return None, None

        # Explicitly apply the computed vertex colors to the mesh
        logger.debug(f"  Applying vertex colors to mesh...")
        apply_vertex_colors(mesh_obj, terrain.colors, terrain.y_valid, terrain.x_valid)
        logger.debug(f"  ✓ Vertex colors applied")

    else:
        # Single colormap mode: compute colors first, then create mesh with water detection
        terrain.compute_colors()

        mesh_obj = terrain.create_mesh(
            scale_factor=scale_factor,
            height_scale=height_scale,
            center_model=True,
            boundary_extension=True,
            water_mask=water_mask,  # Water detection handled in create_mesh()
        )

        if mesh_obj is None:
            logger.error(f"Failed to create mesh for {name}")
            return None, None

    # Position the mesh at the specified location
    mesh_obj.location = location
    mesh_obj.name = name

    logger.info(f"✓ Created mesh: {name} at position {location}")
    return mesh_obj, terrain


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
    sun_angle_pitch: float = 75.0,
    sun_angle_direction: float = -45.0,
    sun_energy: float = 7.0,
    sun_angle: float = 1.0,
    fill_angle_pitch: float = 60.0,
    fill_angle_direction: float = 135.0,
    fill_energy: float = 1.0,
    fill_angle: float = 3.0,
) -> list:
    """Setup Blender lighting with sunset-style positioning.

    Creates a low-angle sun for dramatic shadows and warm fill light.

    Args:
        sun_angle_pitch: Sun pitch angle in degrees (0=horizon, 90=overhead)
        sun_angle_direction: Sun direction in degrees (0=north, 90=east, -45=southwest)
        sun_energy: Sun light strength/energy
        sun_angle: Sun angular size in degrees (smaller=sharper shadows)
        fill_angle_pitch: Fill light pitch angle in degrees
        fill_angle_direction: Fill light direction in degrees
        fill_energy: Fill light strength/energy
        fill_angle: Fill light angular size in degrees (softer than sun)

    Returns:
        List of light objects
    """
    from math import radians

    logger.info("Setting up sunset lighting...")

    lights = []

    # Primary sun - low angle from the west/southwest for sunset shadows
    # rotation_euler: (pitch down from horizon, yaw direction, roll)
    sun_light = setup_light(
        location=(10, -5, 2),  # Position doesn't matter much for sun type
        angle=sun_angle,  # Sharper shadows (smaller angle = harder shadows)
        energy=sun_energy,
        rotation_euler=(radians(sun_angle_pitch), 0, radians(sun_angle_direction)),
    )
    # Set warm sunset color (golden/orange)
    sun_light.data.color = (1.0, 0.85, 0.6)  # Warm golden
    lights.append(sun_light)

    # Fill light - cooler blue from opposite side for contrast
    fill_light = setup_light(
        location=(-10, 5, 5),
        angle=fill_angle,  # Softer
        energy=fill_energy,
        rotation_euler=(radians(fill_angle_pitch), 0, radians(fill_angle_direction)),
    )
    # Cool blue fill to contrast with warm sun
    fill_light.data.color = (0.7, 0.8, 1.0)  # Cool blue
    lights.append(fill_light)

    logger.info("✓ Sunset lighting setup complete")
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

    # Lighting options
    parser.add_argument(
        "--sun-angle-pitch",
        type=float,
        default=75.0,
        help="Sun light pitch angle in degrees (default: 75.0, 0=horizon, 90=overhead)",
    )

    parser.add_argument(
        "--sun-angle-direction",
        type=float,
        default=-45.0,
        help="Sun light direction angle in degrees (default: -45.0, 0=north, 90=east, -45=southwest)",
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
        "--fill-angle-pitch",
        type=float,
        default=60.0,
        help="Fill light pitch angle in degrees (default: 60.0)",
    )

    parser.add_argument(
        "--fill-angle-direction",
        type=float,
        default=135.0,
        help="Fill light direction angle in degrees (default: 135.0, northeast)",
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
        default=True,
        help="Include interstate and state roads (default: True)",
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
        "--road-color-blend",
        type=float,
        default=0.7,
        help="Road color darkening factor, 0.5-0.9 (default: 0.7)",
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
    if args.roads:
        logger.info("Loading road data from OpenStreetMap...")
        try:
            # Calculate road bbox from DEM extent with 0.1 degree padding
            min_lat = transform.f + (dem.shape[0] * transform.e)  # Bottom
            max_lat = transform.f  # Top
            min_lon = transform.c  # Left
            max_lon = transform.c + (dem.shape[1] * transform.a)  # Right

            # Normalize lat/lon order (south, west, north, east)
            road_bbox = (
                min(min_lat, max_lat) - 0.1,  # south
                min(min_lon, max_lon) - 0.1,  # west
                max(min_lat, max_lat) + 0.1,  # north
                max(min_lon, max_lon) + 0.1,  # east
            )

            road_data = get_roads(road_bbox, args.road_types)
            if road_data and road_data.get("features"):
                logger.info(f"  Loaded {len(road_data['features'])} road segments")
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

    # Configure downsampling
    if target_vertices:
        logger.debug(f"Configuring for target vertices: {target_vertices:,}")
        terrain_combined.configure_for_target_vertices(target_vertices=target_vertices)

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

        # Set blended color mapping: sledding (mako) base + XC skiing (rocket) overlay
        logger.info("Setting blended color mapping:")
        logger.info("  Base: Sledding scores with gamma=0.5 (mako colormap)")
        logger.info("  Overlay: XC skiing scores near parks (rocket colormap)")
        terrain_combined.set_blended_color_mapping(
            base_colormap=lambda score: elevation_colormap(
                np.power(score, 0.5), cmap_name="mako", min_elev=0.0, max_elev=np.power(1.5, 0.5)  # Power scale gamma=0.5
            ),
            base_source_layers=["sledding"],
            overlay_colormap=lambda score: elevation_colormap(
                score, cmap_name="rocket", min_elev=0.0, max_elev=1.0  # XC skiing 0-1 range
            ),
            overlay_source_layers=["xc_skiing"],
            overlay_mask=park_mask,
        )
    else:
        # No parks - just use sledding scores with gamma=0.5
        logger.info("No parks available - using sledding scores with gamma=0.5 (mako colormap)")
        terrain_combined.set_color_mapping(
            lambda score: elevation_colormap(np.power(score, 0.5), cmap_name="mako", min_elev=0.0, max_elev=np.power(1.5, 0.5)),
            source_layers=["sledding"],
        )

    # Detect water
    logger.debug("Detecting water bodies...")
    water_mask = terrain_combined.detect_water_highres(
        slope_threshold=0.01,
        fill_holes=True,
        scale_factor=0.0001,
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

    # Add roads to scene if requested
    if args.roads and road_data:
        logger.info("\nAdding roads to terrain...")
        try:
            add_roads_to_scene(
                terrain_combined,
                road_data,
                color_blend_factor=args.road_color_blend,
            )
            logger.info(f"✓ Roads added successfully")
        except Exception as e:
            logger.warning(f"Failed to add roads: {e}")

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
    camera = position_camera_relative(
        mesh_obj=mesh_combined,
        direction=args.camera_direction,
        camera_type="ORTHO",
        ortho_scale=args.ortho_scale,
    )
    lights = setup_lighting(
        sun_angle_pitch=args.sun_angle_pitch,
        sun_angle_direction=args.sun_angle_direction,
        sun_energy=args.sun_energy,
        sun_angle=args.sun_angle,
        fill_angle_pitch=args.fill_angle_pitch,
        fill_angle_direction=args.fill_angle_direction,
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
