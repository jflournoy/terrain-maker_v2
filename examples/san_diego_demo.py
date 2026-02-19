#!/usr/bin/env python3
"""
San Diego Terrain Demo: Complete Pipeline from Download to Render.

Shows how easy it is to go from nothing to a beautiful 3D terrain render:
1. Download SRTM data from NASA (one function call)
2. Load DEM files (one function call)
3. Create terrain and add transforms (few lines)
4. Render with nice colors, lighting, and background (library handles it all)

Requirements:
- Blender Python API (bpy)
- NASA Earthdata credentials in .env file

Usage:
    python examples/san_diego_demo.py
    python examples/san_diego_demo.py --skip-download  # if you already have data
    python examples/san_diego_demo.py --no-render      # just setup scene
"""

import sys
import os
from pathlib import Path

import bpy

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file for NASA credentials
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.terrain.dem_downloader import download_dem_by_bbox
from src.terrain.data_loading import load_dem_files
from src.terrain.core import (
    Terrain,
    elevation_colormap,
    clear_scene,
    position_camera_relative,
    setup_render_settings,
    render_scene_to_file,
    reproject_raster,
    flip_raster,
    scale_elevation,
)
from src.terrain.scene_setup import create_background_plane, setup_hdri_lighting
from src.terrain.materials import apply_colormap_material


def main():
    import argparse
    parser = argparse.ArgumentParser(description="San Diego Terrain Demo")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("examples/output"))
    parser.add_argument("--dem-dir", type=Path, default=Path("data/san_diego_dem"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.dem_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download DEM (if needed)
    if not args.skip_download:
        print("Downloading San Diego DEM tiles from NASA...")
        bbox = (32.5, -117.6, 33.5, -116.0)  # San Diego County
        download_dem_by_bbox(
            bbox=bbox,
            output_dir=str(args.dem_dir),
            username=os.environ.get("EARTHDATA_USERNAME"),
            password=os.environ.get("EARTHDATA_PASSWORD"),
        )
        print("✓ Download complete")

    # Step 2: Load DEM (automatically extracts from ZIP if needed)
    print(f"\nLoading DEM from {args.dem_dir}...")
    dem, transform = load_dem_files(args.dem_dir)
    print(f"✓ Loaded DEM: {dem.shape} pixels")

    # Step 3: Create terrain with transforms
    print("\nCreating terrain...")
    clear_scene()

    terrain = Terrain(dem, transform, dem_crs="EPSG:4326")

    # Add transforms: optimized reproject + downsample, flip, scale
    target_vertices = 10_000_000
    dem_h, dem_w = dem.shape
    zoom_factor = np.sqrt(target_vertices / (dem_h * dem_w))

    # Combined downsampling + reprojection (saves 40-50s on large DEMs)
    terrain.add_transform(downsample_then_reproject(
        src_crs="EPSG:4326",
        dst_crs="EPSG:32611",
        downsample_zoom_factor=zoom_factor,
    ))
    terrain.add_transform(flip_raster(axis="horizontal"))
    terrain.add_transform(scale_elevation(scale_factor=0.0001))
    terrain.apply_transforms()

    # Set colormap (plasma from viridis family)
    terrain.set_color_mapping(
        lambda elev: elevation_colormap(elev, cmap_name="plasma"),
        source_layers=["dem"],
    )

    # Detect water (San Diego Bay)
    water_mask = terrain.detect_water_highres(
        slope_threshold=0.0000000000000001,
        fill_holes=False,
        scale_factor=0.0001,
    )

    # Create mesh with smooth fractional edges (now library defaults)
    terrain.compute_colors()
    mesh = terrain.create_mesh(
        scale_factor=100,
        height_scale=8.0,
        center_model=True,
        boundary_extension=True,
        water_mask=water_mask, 
        base_depth=1.0
    )
    print(f"✓ Created mesh: {len(mesh.data.vertices):,} vertices")

    # Apply eggshell material for subtle sheen
    apply_colormap_material(mesh.data.materials[0], terrain_material="eggshell")

    # Step 4: Setup scene (camera, lights, background)
    print("\nSetting up scene...")

    # Camera
    camera = position_camera_relative(
        mesh,
        direction="south-southwest",
        camera_type="PERSP",
        focal_length=50,
        distance=1.0,
        elevation=1.0,
    )

    # Sky lighting (realistic atmospheric illumination)
    setup_hdri_lighting(
        sun_elevation=15.0,      # Mid-afternoon sun
        sun_rotation=225.0,      # From southwest
        sun_intensity=0.05,       # Natural brightness
        air_density=0.05,         # Clear atmosphere
        visible_to_camera=False, # Sky lights scene but isn't visible
        sky_strength=1.75
    )

    # Background plane
    create_background_plane(
        camera=camera,
        mesh_or_meshes=mesh,
        distance_below=0.0,
        color="#000000",
        roughness=0.20,
        size_multiplier=10,
        receive_shadows=True,
    )
    print("✓ Scene ready")

    # Step 5: Render
    if not args.no_render:
        print("\nRendering...")
        setup_render_settings(use_gpu=True, samples=2048, use_denoising=True)

        output_path = args.output_dir / "san_diego_demo.png"
        render_scene_to_file(
            str(output_path),
            width=72*10,
            height=72*8,
            file_format="PNG",
            color_mode="RGB")

        print(f"✓ Rendered to {output_path}")
    else:
        print("\nSkipped render (--no-render)")

    print("\n✓ Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
