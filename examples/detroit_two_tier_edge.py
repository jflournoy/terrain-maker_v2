"""
Two-Tier Edge Extrusion Example.

Demonstrates the two-tier edge extrusion feature which creates a professional
appearance with a small colored edge near the surface and a larger uniform base
below. This is useful for 3D printing and architectural visualizations.

Two-tier mode provides visual separation between terrain colors and a clean
material base, creating models that look polished and intentional.

Requirements:
- Blender Python API available (bpy)
- SRTM elevation data (automatically downloaded)

Usage:
    # Single-tier (default - current behavior)
    python examples/detroit_two_tier_edge.py --mode single

    # Two-tier with default clay base
    python examples/detroit_two_tier_edge.py --mode two-tier-clay

    # Two-tier with gold base
    python examples/detroit_two_tier_edge.py --mode two-tier-gold

    # Two-tier with sharp color transition
    python examples/detroit_two_tier_edge.py --mode two-tier-sharp

    # Two-tier with custom RGB color
    python examples/detroit_two_tier_edge.py --mode two-tier-custom

    # All comparisons side-by-side
    python examples/detroit_two_tier_edge.py --mode all
"""

import argparse
import logging
import os
from pathlib import Path
import bpy

import numpy as np

from src.terrain.core import Terrain
from src.terrain.loader import load_dem_files

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
SRTM_TILES_DIR = Path("data/srtm")
OUTPUT_DIR = Path("renders/two_tier_edge")

# Detroit area bounds (meters in UTM Zone 17N)
# These are approximate - adjust based on your needs
DETROIT_BBOX = {
    "west": 282000,    # UTM zone 17N
    "south": 4660000,
    "east": 320000,
    "north": 4700000,
}


def clear_scene():
    """Remove all mesh objects from Blender scene."""
    for obj in list(bpy.data.objects):
        if obj.type == "MESH":
            bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh, do_unlink=True)


def create_terrain_mesh(mode: str) -> tuple:
    """
    Create terrain mesh with specified edge extrusion mode.

    Args:
        mode: One of 'single', 'clay', 'gold', 'sharp', 'custom'

    Returns:
        Tuple of (terrain_object, terrain, mode_description)
    """
    logger.info(f"Loading DEM data...")

    # Load DEM files
    dem_data, transform = load_dem_files(SRTM_TILES_DIR, pattern="*.hgt")
    terrain = Terrain(dem_data, transform)

    # Configure terrain
    logger.info(f"Configuring terrain for {mode} mode...")
    terrain.add_transform(lambda d, t: (d, t, None))
    terrain.apply_transforms()

    # Create mesh based on mode
    if mode == "single":
        logger.info("Creating single-tier edge (default behavior)...")
        terrain_obj = terrain.create_mesh(
            base_depth=-0.2,
            boundary_extension=True,
            scale_factor=100.0,
            height_scale=1.0,
            center_model=True,
            two_tier_edge=False,  # Single-tier (default)
        )
        description = "Single-Tier Edge (Default)"

    elif mode == "clay":
        logger.info("Creating two-tier edge with clay base...")
        terrain_obj = terrain.create_mesh(
            base_depth=-0.2,
            boundary_extension=True,
            scale_factor=100.0,
            height_scale=1.0,
            center_model=True,
            two_tier_edge=True,                # Enable two-tier
            edge_base_material="clay",         # Clay material (default)
            edge_blend_colors=True,            # Blend surface colors to mid tier
        )
        description = "Two-Tier with Clay Base (Default)"

    elif mode == "gold":
        logger.info("Creating two-tier edge with gold base...")
        terrain_obj = terrain.create_mesh(
            base_depth=-0.2,
            boundary_extension=True,
            scale_factor=100.0,
            height_scale=1.0,
            center_model=True,
            two_tier_edge=True,
            edge_base_material="gold",         # Gold material preset
            edge_blend_colors=True,
        )
        description = "Two-Tier with Gold Base"

    elif mode == "sharp":
        logger.info("Creating two-tier edge with sharp color transition...")
        terrain_obj = terrain.create_mesh(
            base_depth=-0.2,
            boundary_extension=True,
            scale_factor=100.0,
            height_scale=1.0,
            center_model=True,
            two_tier_edge=True,
            edge_base_material="obsidian",     # Black base for sharp contrast
            edge_blend_colors=False,           # Sharp transition (no blending)
        )
        description = "Two-Tier with Sharp Transition (Obsidian)"

    elif mode == "custom":
        logger.info("Creating two-tier edge with custom RGB color...")
        terrain_obj = terrain.create_mesh(
            base_depth=-0.2,
            boundary_extension=True,
            scale_factor=100.0,
            height_scale=1.0,
            center_model=True,
            two_tier_edge=True,
            edge_base_material=(0.55, 0.52, 0.48),  # Custom warm gray
            edge_blend_colors=True,
        )
        description = "Two-Tier with Custom RGB (Warm Gray)"

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return terrain_obj, terrain, description


def setup_scene_and_render(terrain_obj, terrain, mode: str, description: str):
    """Set up lighting, camera, and render the scene."""
    logger.info(f"Setting up scene for {description}...")

    # Set up world background
    bpy.context.scene.world.use_nodes = True
    bg = bpy.context.scene.world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.9, 0.9, 0.95, 1.0)  # Light gray sky

    # Position camera
    logger.info("Positioning camera...")
    terrain.position_camera_relative(
        direction="north",
        distance=3.0,
        height_ratio=0.4,
        target_offset_y=0.2,
    )

    # Set up lighting
    logger.info("Setting up lighting...")
    light_data = bpy.data.lights.new(name="SunLight", type="SUN")
    light_data.use_nodes = True
    light_data.energy = 2.0

    light_obj = bpy.data.objects.new(name="SunLight", object_data=light_data)
    bpy.context.scene.collection.objects.link(light_obj)

    # Position sun light from northwest, above
    light_obj.location = (-3, -3, 4)
    light_obj.rotation_euler = (0.6, 0.3, -0.5)

    # Render settings
    logger.info("Configuring render settings...")
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.render.samples = 128
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080

    # Output
    output_dir = OUTPUT_DIR / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"detroit_two_tier_{mode}.png"

    bpy.context.scene.render.filepath = str(output_file)

    logger.info(f"Rendering to {output_file}...")
    bpy.ops.render.render(write_still=True)

    logger.info(f"✓ Rendered: {output_file}")
    return output_file


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Two-Tier Edge Extrusion Example"
    )
    parser.add_argument(
        "--mode",
        default="clay",
        choices=["single", "clay", "gold", "sharp", "custom", "all"],
        help="Extrusion mode to demonstrate",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR),
        help="Output directory for renders",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    modes_to_render = (
        ["single", "clay", "gold", "sharp", "custom"]
        if args.mode == "all"
        else [args.mode]
    )

    for mode in modes_to_render:
        logger.info(f"\n{'='*60}")
        logger.info(f"Rendering: {mode}")
        logger.info(f"{'='*60}")

        try:
            # Clear scene
            clear_scene()

            # Create mesh
            terrain_obj, terrain, description = create_terrain_mesh(mode)

            if terrain_obj is None:
                logger.error(f"Failed to create terrain for mode: {mode}")
                continue

            # Render
            output_file = setup_scene_and_render(terrain_obj, terrain, mode, description)
            logger.info(f"✓ Successfully rendered: {output_file}")

        except Exception as e:
            logger.error(f"Error rendering {mode}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    logger.info(f"\n{'='*60}")
    logger.info("All renders complete!")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
