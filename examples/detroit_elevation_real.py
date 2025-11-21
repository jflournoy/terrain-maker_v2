#!/usr/bin/env python3
"""
Detroit Real Elevation Visualization

This example demonstrates loading real SRTM elevation data for the Detroit
metro area and creating a 3D terrain visualization with Blender.

Data Source:
    - SRTM 90m Digital Elevation Model tiles (.hgt format)
    - Downloaded from NASA Earth Explorer
    - Location: ../../../geotiff-rayshade/detroit/

Workflow:
    1. Load and merge all SRTM HGT tiles (full coverage)
    2. Initialize Terrain object
    3. Apply 10:1 downsampling to reduce vertex count
    4. Set up Viridis color mapping (elevation-based)
    5. Generate Blender mesh
    6. Render to PNG

Output:
    - PNG saved to: examples/detroit_elevation_real.png (1920×1440)
    - Blender file: examples/detroit_elevation_real.blend

Usage:
    python examples/detroit_elevation_real.py
"""

import sys
from pathlib import Path
from math import radians

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.terrain.core import (
    Terrain, load_dem_files, scale_elevation, flip_raster, reproject_raster,
    elevation_colormap, clear_scene, setup_camera_and_light,
    setup_render_settings, render_scene_to_file
)

try:
    import bpy
except ImportError:
    bpy = None

# SRTM tiles directory
SRTM_TILES_DIR = Path(__file__).parent.parent / "data" / "dem" / "detroit"



def main():
    """Run the Detroit real elevation visualization."""
    print("=" * 70)
    print("Detroit Real Elevation Visualization")
    print("=" * 70)

    # Clear Blender scene to remove default objects
    try:
        import bpy
        clear_scene()
        print("✓ Blender scene cleared")
    except ImportError:
        print("⚠️  Blender not available - skipping scene clear")

    # Step 1: Load DEM
    print("[1/6] Loading SRTM tiles...")
    try:
        dem_data, transform = load_dem_files(SRTM_TILES_DIR, pattern='*.hgt')
    except Exception as e:
        print(f"\n✗ Failed to load DEM: {e}")
        return 1

    # Step 2: Initialize Terrain
    print("\n[2/6] Initializing Terrain object...")
    terrain = Terrain(dem_data, transform)
    print(f"      Terrain initialized")
    print(f"      DEM shape: {terrain.dem_shape}")

    # Step 3: Apply transforms (downsample to reduce mesh complexity)
    print("\n[3/6] Applying transforms...")

    print(f"      Original DEM shape: {terrain.dem_shape}")

    # Configure downsampling to target approximately 500,000 vertices
    # This automatically calculates the optimal zoom_factor
    target_vertices = 1_000_000
    zoom = terrain.configure_for_target_vertices(target_vertices, order=4)
    print(f"      Configured for {target_vertices:,} target vertices")
    print(f"      Calculated zoom_factor: {zoom:.6f}")

    # Reproject from WGS84 to UTM Zone 17N for proper geographic scaling
    utm_reproject = reproject_raster(
        src_crs='EPSG:4326',      # WGS84 (source: SRTM data)
        dst_crs='EPSG:32617',     # UTM Zone 17N (Detroit area)
        num_threads=4
    )
    terrain.transforms.append(utm_reproject)
    # Flip DEM data to correct north-south orientation
    terrain.transforms.append(flip_raster(axis='horizontal'))
    # Add elevation scaling to enhance height features
    terrain.transforms.append(scale_elevation(scale_factor=0.0001))
    terrain.apply_transforms()

    # Check downsampled size
    transformed_dem = terrain.data_layers['dem']['transformed_data']
    print(f"      Downsampled DEM shape: {transformed_dem.shape}")
    print(f"      Actual vertices: {transformed_dem.shape[0] * transformed_dem.shape[1]:,}")
    print(f"      Transforms applied successfully")

    # Step 4: Set up color mapping
    print("\n[4/6] Setting up color mapping...")
    # Use class method for elevation-based Viridis colormap
    terrain.set_color_mapping(
        lambda dem: elevation_colormap(dem, cmap_name='viridis'),
        source_layers=['dem']
    )
    print(f"      Color mapping configured (Viridis colormap)")

    # Step 5: Create Blender mesh
    print("\n[5/6] Creating Blender mesh...")
    try:
        mesh_obj = terrain.create_mesh(
            scale_factor=100.0,  
            height_scale=1.0,  
            center_model=True,
            boundary_extension=True
        )

        if mesh_obj is None:
            print(f"      ✗ Mesh creation failed!")
            return 1

        print(f"      ✓ Mesh created successfully!")
        print(f"      Vertices: {len(mesh_obj.data.vertices)}")
        print(f"      Polygons: {len(mesh_obj.data.polygons)}")

    except Exception as e:
        print(f"      ✗ Error creating mesh: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 6: Render to PNG
    print("\n[6/6] Setting up camera and rendering to PNG...")

    # Tuned camera parameters for Detroit elevation terrain
    camera_location = (0, -5, 350)
    camera_rotation_deg = (0.0, 0.0, 0.0)
    camera_angle = (
        radians(camera_rotation_deg[0]),
        radians(camera_rotation_deg[1]),
        radians(camera_rotation_deg[2])
    )

    # Set up camera and light using class method
    camera, light = setup_camera_and_light(
        camera_angle=camera_angle,
        camera_location=camera_location,
        camera_type='ORTHO',
        scale=20,
        sun_angle=2,
        sun_energy=3,
        focal_length=30
    )

    # Configure render settings using class method
    setup_render_settings(
        use_gpu=True,
        samples=32,
        use_denoising=False
    )

    print(f"      Camera: TUNED VIEW")
    print(f"      Location: {camera_location}")
    print(f"      Rotation: {camera_rotation_deg}")
    print(f"      Samples: 32")
    print(f"      Rendering...")

    # Render scene to file using class method
    output_path = Path(__file__).parent / "detroit_elevation_real.png"
    render_file = render_scene_to_file(
        output_path=output_path,
        width=1920,
        height=1440,
        file_format='PNG',
        color_mode='RGBA',
        compression=90,
        save_blend_file=True
    )

    if render_file:
        file_size_mb = render_file.stat().st_size / (1024 * 1024)
        print(f"      ✓ Rendered successfully!")
        print(f"      File: {render_file.name}")
        print(f"      Size: {file_size_mb:.1f} MB")

    # Summary
    print("\n" + "=" * 70)
    print("Detroit Real Elevation Visualization Complete!")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  ✓ Loaded and merged all SRTM tiles (full coverage)")
    print(f"  ✓ Downsampled 10:1 to reduce vertex count")
    print(f"  ✓ Created Terrain object with real elevation data")
    print(f"  ✓ Applied transforms (downsampling + processing)")
    print(f"  ✓ Configured Viridis elevation-based color mapping")
    print(f"  ✓ Generated Blender mesh with {len(mesh_obj.data.vertices)} vertices")
    if render_file:
        print(f"  ✓ Rendered to PNG: {render_file}")
    print(f"\nThe terrain visualization is ready!")

    return 0


if __name__ == '__main__':
    exit(main())
