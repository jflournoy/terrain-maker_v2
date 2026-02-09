#!/usr/bin/env python3
"""
San Diego Flow Accumulation Demo: Hydrological Analysis + 3D Visualization.

Builds on san_diego_demo.py to add flow accumulation analysis:
1. Download SRTM DEM from NASA (same as san_diego_demo.py)
2. Download real WorldClim precipitation data (global dataset, covers USA + Mexico)
3. Compute flow direction, drainage area, and upstream rainfall
4. Visualize results in 3D with Blender rendering
5. Color terrain by upstream rainfall to show water accumulation

Shows integration of flow accumulation with terrain visualization using real climate data.

NOTE: Uses WorldClim instead of PRISM because San Diego DEM extends into Mexico.
PRISM only covers continental USA, but WorldClim provides global coverage at ~4.5km resolution.

Requirements:
- Blender Python API (bpy)
- NASA Earthdata credentials in .env file
- Internet connection for WorldClim data download (~10MB, one-time)

Usage:
    python examples/san_diego_flow_demo.py
    python examples/san_diego_flow_demo.py --skip-download  # if you already have DEM
    python examples/san_diego_flow_demo.py --precip path/to/precip.tif  # use custom precip data
"""

import sys
import os
from pathlib import Path
import numpy as np
import rasterio
from rasterio import Affine

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
from src.terrain.flow_accumulation import flow_accumulation
from src.terrain.precipitation_downloader import download_precipitation
from src.terrain.color_mapping import elevation_colormap


def visualize_flow_with_matplotlib(result: dict, output_dir: Path) -> None:
    """
    Create 2D matplotlib visualizations of flow results.

    Parameters
    ----------
    result : dict
        Flow accumulation results
    output_dir : Path
        Directory to save plots
    """
    import matplotlib.pyplot as plt

    print("\nCreating 2D visualizations...")

    drainage_area = result["drainage_area"]
    upstream_rainfall = result["upstream_rainfall"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Drainage Area (log scale)
    ax = axes[0]
    drainage_log = np.log10(drainage_area + 1)
    im1 = ax.imshow(drainage_log, cmap="Blues", interpolation="bilinear")
    ax.set_title("Drainage Area (log10 cells)", fontsize=14, fontweight="bold")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    plt.colorbar(im1, ax=ax, label="log10(cells)")

    # Plot 2: Upstream Rainfall (log scale)
    ax = axes[1]
    rainfall_log = np.log10(upstream_rainfall + 1)
    im2 = ax.imshow(rainfall_log, cmap="viridis", interpolation="bilinear")
    ax.set_title("Upstream Rainfall (log10 mm·m²)", fontsize=14, fontweight="bold")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    plt.colorbar(im2, ax=ax, label="log10(mm·m²)")

    plt.tight_layout()

    output_path = output_dir / "flow_analysis_2d.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved 2D visualization to {output_path}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="San Diego Flow Accumulation Demo")
    parser.add_argument("--skip-download", action="store_true", help="Skip DEM download")
    parser.add_argument("--no-render", action="store_true", help="Skip 3D rendering")
    parser.add_argument(
        "--precip", type=Path, default=None, help="Path to precipitation GeoTIFF (optional)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("examples/output"), help="Output directory"
    )
    parser.add_argument(
        "--dem-dir", type=Path, default=Path("data/san_diego_dem"), help="DEM directory"
    )
    parser.add_argument(
        "--color-by",
        type=str,
        choices=["elevation", "drainage", "rainfall"],
        default="rainfall",
        help="Color terrain by: elevation, drainage area, or upstream rainfall",
    )
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

    # Step 2: Load DEM
    print(f"\nLoading DEM from {args.dem_dir}...", flush=True)
    dem, transform = load_dem_files(args.dem_dir)
    print(f"✓ Loaded DEM: {dem.shape} pixels", flush=True)

    # Extract actual DEM bounds from transform
    # IMPORTANT: DEM download fetches full SRTM tiles (1°×1°), so actual extent
    # is larger than requested bbox. We need precipitation to cover the full DEM.
    height, width = dem.shape
    left = transform.c
    top = transform.f
    right = transform.c + transform.a * width
    bottom = transform.f + transform.e * height
    dem_bbox = (bottom, left, top, right)  # (min_lat, min_lon, max_lat, max_lon)
    print(f"  DEM bounds: lat [{dem_bbox[0]:.4f}, {dem_bbox[2]:.4f}], "
          f"lon [{dem_bbox[1]:.4f}, {dem_bbox[3]:.4f}]", flush=True)

    # Step 3: Prepare precipitation data
    print("DEBUG: About to check precipitation...", flush=True)
    print(f"DEBUG: precip={args.precip}, exists={args.precip.exists() if args.precip else 'N/A'}", flush=True)
    if args.precip and args.precip.exists():
        print(f"\nUsing precipitation data from {args.precip}", flush=True)
        precip_path = args.precip
    else:
        # Download WorldClim precipitation data (global dataset covering USA + Mexico)
        # Use actual DEM bounds to ensure full coverage
        # NOTE: Using WorldClim instead of PRISM because San Diego DEM extends into Mexico.
        # PRISM only covers continental USA, missing ~26.5% of this DEM (south of border).
        print("\nDownloading real WorldClim precipitation data...")
        print(f"  Using DEM bounds: {dem_bbox}")
        print(f"  WorldClim covers USA + Mexico (global dataset)")
        precip_path = download_precipitation(
            bbox=dem_bbox,
            output_dir=str(args.output_dir),
            dataset="worldclim",  # Global coverage (not just USA)
            use_real_data=True,
        )

    # Step 4: Save merged DEM for flow accumulation
    print("\nPreparing DEM for flow accumulation...", flush=True)
    print("DEBUG: Step 4 starting", flush=True)
    flow_data_dir = args.output_dir / "flow_data"
    flow_data_dir.mkdir(parents=True, exist_ok=True)
    merged_dem_path = flow_data_dir / "merged_dem.tif"

    # Save merged DEM as GeoTIFF
    print("DEBUG: Importing rasterio...", flush=True)
    import rasterio
    from rasterio import Affine

    # Check if file already exists - skip if so
    if merged_dem_path.exists():
        print(f"DEBUG: Merged DEM already exists, skipping write", flush=True)
    else:
        print(f"DEBUG: Writing DEM {dem.shape} to {merged_dem_path}...", flush=True)
        height, width = dem.shape
        with rasterio.open(
            merged_dem_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=dem.dtype,
            crs="EPSG:4326",
            transform=transform,
            compress="lzw",
        ) as dst:
            dst.write(dem, 1)

    print(f"✓ Saved merged DEM: {merged_dem_path}", flush=True)

    # Step 5: Compute flow accumulation
    print("\nComputing flow accumulation...", flush=True)
    print(f"DEBUG: Step 5 starting - calling flow_accumulation()", flush=True)
    flow_output_dir = args.output_dir / "flow_outputs"

    # Use adaptive resolution: compute flow at 3x render resolution for speed
    # Render target: 10M vertices, so flow computed at ~30M cells (vs 77.7M full DEM)
    TARGET_RENDER_VERTICES = 10_000_000

    print(f"DEBUG: DEM path: {merged_dem_path}", flush=True)
    print(f"DEBUG: Precip path: {precip_path}", flush=True)
    print(f"DEBUG: Output dir: {flow_output_dir}", flush=True)

    flow_result = flow_accumulation(
        dem_path=str(merged_dem_path),
        precipitation_path=str(precip_path),
        output_dir=str(flow_output_dir),
        fill_method="breach",
        target_vertices=TARGET_RENDER_VERTICES,  # Auto-downsample for performance
        min_basin_size=2500,  # Conservative: preserve basins > ~20 km² (scaled by downsample²)
    )

    # Print summary
    metadata = flow_result["metadata"]
    print(f"\n{'=' * 60}")
    print("FLOW ACCUMULATION RESULTS")
    print(f"{'=' * 60}")
    if metadata["downsampling_applied"]:
        print(f"  Original DEM: {metadata['original_shape'][0]}×{metadata['original_shape'][1]} "
              f"({metadata['original_shape'][0] * metadata['original_shape'][1]:,} cells)")
        print(f"  Flow computed at: {metadata['downsampled_shape'][0]}×{metadata['downsampled_shape'][1]} "
              f"({metadata['downsampled_shape'][0] * metadata['downsampled_shape'][1]:,} cells)")
        print(f"  Downsample factor: {metadata['downsample_factor']:.2f}x")
    print(f"  Total area: {metadata['total_area_km2']:.1f} km²")
    print(f"  Max drainage area: {metadata['max_drainage_area_km2']:.2f} km²")
    print(f"  Max upstream water: {metadata['max_upstream_rainfall_m3']:.0f} m³/year")
    print(f"{'=' * 60}\n")

    # Create 2D visualizations
    visualize_flow_with_matplotlib(flow_result, args.output_dir)

    # Step 6: Create 3D terrain visualization
    print("\nCreating 3D terrain...")
    clear_scene()

    # Load DEM (library will automatically align layers to match after transforms)
    print(f"Loading DEM from {args.dem_dir}...")
    terrain = Terrain(dem, transform, dem_crs="EPSG:4326")

    # Add transforms: reproject to UTM, flip, scale, downsample
    terrain.add_transform(reproject_raster("EPSG:4326", "EPSG:32611", num_threads=4))
    terrain.add_transform(flip_raster(axis="horizontal"))
    terrain.add_transform(scale_elevation(scale_factor=0.0001))
    terrain.configure_for_target_vertices(10_000_000, method="average")

    # Add flow accumulation results as data layers
    print("Adding flow data layers...")
    drainage_area = flow_result["drainage_area"]
    upstream_rainfall = flow_result["upstream_rainfall"]

    # Use log scale for better visualization
    drainage_log = np.log10(drainage_area + 1)
    rainfall_log = np.log10(upstream_rainfall + 1)

    # Add as data layers (will be resampled to match terrain)
    terrain.add_data_layer(
        "drainage_area_log",
        drainage_log.astype(np.float32),
        transform,
        crs="EPSG:4326",
        target_layer="dem",
    )

    terrain.add_data_layer(
        "upstream_rainfall_log",
        rainfall_log.astype(np.float32),
        transform,
        crs="EPSG:4326",
        target_layer="dem",
    )

    # Apply transforms BEFORE water detection
    # This ensures water detection happens on downsampled terrain
    terrain.apply_transforms()

    # Set colormap based on user choice
    if args.color_by == "elevation":
        print("Coloring by elevation...")
        terrain.set_color_mapping(
            lambda elev: elevation_colormap(elev, cmap_name="terrain"),
            source_layers=["dem"],
        )
    elif args.color_by == "drainage":
        print("Coloring by drainage area (log scale)...")
        terrain.set_color_mapping(
            lambda drain: elevation_colormap(drain, cmap_name="Blues"),
            source_layers=["drainage_area_log"],
        )
    else:  # rainfall
        print("Coloring by upstream rainfall (log scale)...")
        terrain.set_color_mapping(
            lambda rain: elevation_colormap(rain, cmap_name="viridis"),
            source_layers=["upstream_rainfall_log"],
        )

    # Detect water AFTER transforms (on downsampled terrain)
    # This ensures water mask matches the final terrain dimensions
    water_mask = terrain.detect_water_highres(
        slope_threshold=0.0000000000000001,
        fill_holes=False,
        scale_factor=0.0001,
    )

    # Create mesh
    terrain.compute_colors()
    mesh = terrain.create_mesh(
        scale_factor=100,
        height_scale=8.0,
        center_model=True,
        boundary_extension=True,
        water_mask=water_mask,
        base_depth=1.0,
    )
    print(f"✓ Created mesh: {len(mesh.data.vertices):,} vertices")

    # Apply material
    apply_colormap_material(mesh.data.materials[0], terrain_material="eggshell")

    # Step 7: Setup scene
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

    # Sky lighting
    setup_hdri_lighting(
        sun_elevation=15.0,
        sun_rotation=225.0,
        sun_intensity=0.05,
        air_density=0.05,
        visible_to_camera=False,
        sky_strength=1.75,
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

    # Step 8: Render
    if not args.no_render:
        print("\nRendering...")
        setup_render_settings(use_gpu=True, samples=2048, use_denoising=True)

        output_filename = f"san_diego_flow_{args.color_by}.png"
        output_path = args.output_dir / output_filename
        render_scene_to_file(
            str(output_path), width=72 * 10, height=72 * 8, file_format="PNG", color_mode="RGB"
        )

        print(f"✓ Rendered to {output_path}")
    else:
        print("\nSkipped render (--no-render)")

    print("\n✓ Done!")
    print(f"\nOutput files:")
    print(f"  2D visualization: {args.output_dir / 'flow_analysis_2d.png'}")
    print(f"  3D render: {args.output_dir / f'san_diego_flow_{args.color_by}.png'}")
    print(f"  Flow GeoTIFFs: {flow_output_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
