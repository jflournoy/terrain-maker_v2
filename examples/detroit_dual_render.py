#!/usr/bin/env python3
"""
Detroit Dual Terrain Rendering Example.

This example renders sledding and cross-country skiing terrain suitability maps
side-by-side in Blender using terrain maker's rendering library, with park
locations marked on the XC skiing terrain.

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
    render_scene_to_file,
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
RENDER_WIDTH = 960
RENDER_HEIGHT = 720

# =============================================================================
# LOADING OUTPUTS
# =============================================================================

def load_sledding_scores(output_dir: Path) -> Optional[np.ndarray]:
    """Load sledding scores from detroit_snow_sledding.py output."""
    # First check for sledding_score.npz
    score_path = output_dir / "sledding_scores.npz"
    if score_path.exists():
        logger.info(f"Loading sledding scores from {score_path}")
        data = np.load(score_path)
        if "score" in data:
            return data["score"]
        elif "sledding_score" in data:
            return data["sledding_score"]
        else:
            # Return first array found
            return data[data.files[0]]

    # Check alternative locations
    alt_paths = [
        output_dir.parent / "sledding_scores.npz",
        Path("examples/output/sledding_scores.npz"),
    ]
    for alt_path in alt_paths:
        if alt_path.exists():
            logger.info(f"Loading sledding scores from {alt_path}")
            data = np.load(alt_path)
            return data[data.files[0]]

    logger.warning("Sledding scores not found, will use mock data")
    return None


def load_xc_skiing_scores(output_dir: Path) -> Optional[np.ndarray]:
    """Load XC skiing scores from detroit_xc_skiing.py output."""
    score_path = output_dir / "xc_skiing_scores.npz"
    if score_path.exists():
        logger.info(f"Loading XC skiing scores from {score_path}")
        data = np.load(score_path)
        if "score" in data:
            return data["score"]
        else:
            return data[data.files[0]]

    logger.warning("XC skiing scores not found, will use mock data")
    return None


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
    location: Tuple[float, float, float] = (0, 0, 0),
    scale_factor: float = 100,
    height_scale: float = 2.0,
    target_vertices: Optional[int] = None,
) -> Optional[object]:
    """
    Create a terrain mesh using terrain maker's Terrain class.

    Args:
        name: Mesh name (for logging and reference)
        score_grid: Score values for vertex coloring (0-1 range)
        dem: DEM elevation data
        transform: Affine transform for coordinate mapping
        dem_crs: CRS of DEM data (default: EPSG:32617 for UTM zone 17N)
        location: (x, y, z) position in Blender for the mesh
        scale_factor: Horizontal scale divisor
        height_scale: Vertical elevation scale
        target_vertices: Target vertex count for downsampling (optional)

    Returns:
        Blender object positioned at specified location
    """
    logger.info(f"Creating terrain mesh: {name}")

    # Create terrain using terrain maker library
    terrain = Terrain(dem, transform, dem_crs=dem_crs)

    # Add score grid as data layer (use same CRS as DEM)
    terrain.add_data_layer(
        "score",
        score_grid,
        transform,
        dem_crs,
        target_layer="dem",
    )

    # Set color mapping: red (low) to green (high)
    # Uses elevation_colormap with RdYlGn colormap for score visualization
    terrain.set_color_mapping(
        lambda score: elevation_colormap(score, cmap_name="RdYlGn", min_elev=0.0, max_elev=1.0),
        source_layers=["score"],
    )

    # Compute vertex colors based on score data
    terrain.compute_colors()

    # Apply transforms to prepare DEM for mesh creation
    # Add identity transform to mark DEM as transformed
    terrain.add_transform(lambda data: data)
    terrain.apply_transforms()

    # Configure vertex count limiting if specified (prevents OOM with large DEMs)
    if target_vertices is not None:
        original_h, original_w = terrain.dem_shape
        original_vertices = original_h * original_w

        if original_vertices > target_vertices:
            zoom = terrain.configure_for_target_vertices(target_vertices, order=4)
            logger.info(
                f"Downsampling {name}: {original_vertices:,} → ~{target_vertices:,} vertices "
                f"(zoom={zoom:.6f})"
            )
        else:
            logger.debug(
                f"No downsampling needed for {name}: {original_vertices:,} ≤ {target_vertices:,}"
            )

    # Create mesh using terrain maker library
    mesh_obj = terrain.create_mesh(
        scale_factor=scale_factor,
        height_scale=height_scale,
        center_model=True,
        boundary_extension=True,
    )

    if mesh_obj is None:
        logger.error(f"Failed to create mesh for {name}")
        return None

    # Position the mesh at the specified location
    mesh_obj.location = location
    mesh_obj.name = name

    logger.info(f"✓ Created mesh: {name} at position {location}")
    return mesh_obj


def create_park_markers(
    parks: list[dict],
    dem: np.ndarray,
    transform: Affine,
    terrain_obj,
    scale_factor: float = 100,
    height_scale: float = 2.0,
) -> list:
    """
    Create glowing sphere markers for parks.

    Args:
        parks: List of parks with lat, lon, score, pixel_coords
        dem: DEM for elevation lookup
        transform: Affine transform
        terrain_obj: Blender object (for positioning relative to)
        scale_factor: Horizontal scale
        height_scale: Vertical scale

    Returns:
        List of created marker objects
    """
    if not parks:
        return []

    logger.info(f"Creating {len(parks)} park markers...")
    markers = []

    for i, park in enumerate(parks):
        try:
            row, col = park["pixel_coords"]

            # Get elevation at park location
            if 0 <= row < dem.shape[0] and 0 <= col < dem.shape[1]:
                elevation = dem[int(row), int(col)]
            else:
                elevation = dem.mean()

            # Convert to Blender world coordinates
            x = (col - dem.shape[1] / 2) / scale_factor
            y = (row - dem.shape[0] / 2) / scale_factor
            z = elevation * height_scale + 0.2  # Slightly above terrain

            # Add offset for terrain_obj position if it exists
            if terrain_obj and terrain_obj.location:
                x += terrain_obj.location[0]
                y += terrain_obj.location[1]

            # Create sphere
            bpy.ops.mesh.primitive_ico_sphere_add(
                radius=0.05,
                location=(x, y, z)
            )
            marker = bpy.context.active_object
            marker.name = f"Park_{park['name'].replace(' ', '_')}"

            # Create emission material (color by score)
            score = park.get("score", 0.5)
            color_r = 1.0 - score  # Red for low score
            color_g = score  # Green for high score
            color_b = 0.0

            mat = bpy.data.materials.new(name=f"ParkMarker_{i}")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            nodes.clear()

            # Create shader network
            emission = nodes.new("ShaderNodeEmission")
            emission.inputs["Color"].default_value = (color_r, color_g, color_b, 1.0)
            emission.inputs["Strength"].default_value = 3.0

            output = nodes.new("ShaderNodeOutputMaterial")
            mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])

            marker.data.materials.append(mat)
            markers.append(marker)

            logger.debug(f"  Created marker: {park['name']} at ({x:.2f}, {y:.2f}, {z:.2f}), score={score:.3f}")

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
    distance: float = 5,
    height: float = 3,
) -> Optional[object]:
    """
    Setup camera to view both terrains side-by-side.

    Uses terrain maker's library functions for camera setup.

    Args:
        mesh_left: Left terrain object
        mesh_right: Right terrain object
        distance: Distance from center
        height: Camera height

    Returns:
        Camera object
    """
    logger.info("Setting up dual camera...")

    # Calculate center between meshes
    if mesh_left and mesh_right:
        center_x = (mesh_left.location[0] + mesh_right.location[0]) / 2
        center_y = (mesh_left.location[1] + mesh_right.location[1]) / 2
    else:
        center_x = 0
        center_y = 0

    # Use library function to setup camera
    # Position relative to center point
    camera_angle = (0.5, 0, 0)  # Tilt angle: 0.5 radians tilt on X axis
    camera_location = (center_x, center_y - distance, height)
    camera = setup_camera(
        camera_angle=camera_angle,
        camera_location=camera_location,
        scale=1.0,
        focal_length=50,
        camera_type="PERSP",
    )

    logger.info("✓ Camera positioned")
    return camera


def setup_lighting() -> list:
    """Setup Blender lighting using library functions.

    Creates primary and fill lights for the scene.
    """
    logger.info("Setting up lighting...")

    lights = []

    # Primary sun light from library function
    sun_light = setup_light(
        location=(2, 2, 3),
        angle=2,
        energy=2.0,
        rotation_euler=(0, 0, 0),
    )
    lights.append(sun_light)

    # Fill light for shadow detail
    fill_light = setup_light(
        location=(-2, -2, 2),
        angle=1,
        energy=1.0,
        rotation_euler=(0, 0, 0),
    )
    lights.append(fill_light)

    logger.info("✓ Lighting setup complete")
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
    logger.info("Loading Data")
    logger.info("=" * 70)

    # Load DEM
    if args.mock_data:
        logger.info("Generating mock DEM...")
        dem = np.random.randint(150, 250, (1024, 1024)).astype(np.float32)
        # Create a proper UTM transform (Detroit is in UTM zone 17N)
        # Each pixel represents ~10m x ~10m in projected coordinates
        transform = Affine.translation(300000, 4700000) * Affine.scale(10, -10)
    else:
        dem_dir = Path("data/dem/detroit")
        if dem_dir.exists():
            logger.info(f"Loading DEM from {dem_dir}...")
            dem, transform = load_dem_files(dem_dir)
        else:
            logger.info("Generating mock DEM (DEM directory not found)...")
            dem = np.random.randint(150, 250, (1024, 1024)).astype(np.float32)
            # Create a proper UTM transform (Detroit is in UTM zone 17N)
            transform = Affine.translation(300000, 4700000) * Affine.scale(10, -10)

    # Load sledding scores
    sledding_scores = load_sledding_scores(args.scores_dir)
    if sledding_scores is None:
        if args.mock_data:
            logger.info("Generating mock sledding scores...")
            sledding_scores = generate_mock_scores(dem.shape)
        else:
            logger.error("Sledding scores not found. Run detroit_snow_sledding.py first.")
            return 1

    # Load XC skiing scores
    xc_scores = load_xc_skiing_scores(args.scores_dir / "xc_skiing")
    if xc_scores is None:
        if args.mock_data:
            logger.info("Generating mock XC skiing scores...")
            xc_scores = generate_mock_scores(dem.shape)
        else:
            logger.error("XC skiing scores not found. Run detroit_xc_skiing.py first.")
            return 1

    # Load parks
    parks = load_xc_skiing_parks(args.scores_dir / "xc_skiing")
    if parks is None:
        parks = []

    logger.info(f"Loaded: DEM {dem.shape}, sledding scores {sledding_scores.shape}, XC scores {xc_scores.shape}, {len(parks)} parks")

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
    # Conservative: 500k max per mesh to prevent OOM with dual rendering
    # This ensures safe memory usage for both meshes combined
    target_vertices = 500000
    logger.info(f"Target vertices per mesh: {target_vertices:,}")

    # Create first mesh at origin
    logger.info("Creating first terrain mesh (sledding)...")
    mesh_sledding = create_terrain_with_score(
        "SleddingTerrain",
        sledding_scores,
        dem,
        transform,
        location=(0, 0, 0),
        scale_factor=100,
        height_scale=2.0,
        target_vertices=target_vertices,
    )

    if mesh_sledding is None:
        logger.error("Failed to create sledding terrain mesh")
        return 1

    # Measure sledding mesh
    width_sledding = calculate_mesh_width(mesh_sledding)
    logger.debug(f"Sledding terrain width: {width_sledding:.2f}")

    # Memory check before second mesh (critical point)
    try:
        monitor.check_memory(force=True)
        logger.info("Memory check passed before creating second mesh")
    except MemoryLimitExceeded as e:
        logger.error(f"Insufficient memory to create second mesh: {e}")
        logger.error("Try reducing DEM resolution or closing other applications")
        return 1

    # Create second mesh at origin
    logger.info("Creating second terrain mesh (XC skiing)...")
    mesh_xc = create_terrain_with_score(
        "XCSkiingTerrain",
        xc_scores,
        dem,
        transform,
        location=(0, 0, 0),
        scale_factor=100,
        height_scale=2.0,
        target_vertices=target_vertices,
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

    # Add park markers
    markers = create_park_markers(parks, dem, transform, mesh_xc)

    # Free the original DEM array from memory (it's no longer needed)
    # The Terrain objects have their own downsampled copies
    del dem
    gc.collect()
    logger.info("Freed original DEM from memory")

    # Setup camera and lighting
    camera = setup_dual_camera(mesh_sledding, mesh_xc)
    lights = setup_lighting()

    logger.info("✓ Scene created successfully")

    # Render if requested
    if not args.no_render:
        logger.info("\n" + "=" * 70)
        logger.info("Rendering")
        logger.info("=" * 70)

        output_path = args.output_dir / "sledding_and_xc_skiing_3d.png"
        render_dual_terrain(output_path, width=1920, height=1080)

    logger.info("\n" + "=" * 70)
    logger.info("✓ Complete!")
    logger.info(f"Output directory: {args.output_dir}")
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
