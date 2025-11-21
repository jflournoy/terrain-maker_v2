#!/usr/bin/env python3
"""
Detroit Real Elevation Visualization - Terrain Maker Example

This example demonstrates how EASY it is to create stunning 3D terrain
visualizations with real-world geographic data using Terrain Maker.

With just a few lines of Python, you can:
  ✓ Load real SRTM elevation data from local tiles
  ✓ Apply intelligent downsampling (no complex config needed!)
  ✓ Reproject to proper geographic coordinates automatically
  ✓ Create publication-quality 3D visualizations
  ✓ Render with professional Blender integration
  ✓ Generate multiple camera views with intelligent framing

Data Source:
    - SRTM 90m Digital Elevation Model tiles (.hgt format)
    - Downloaded from NASA Earth Explorer
    - Location: data/dem/detroit/

What Makes This Easy:
    ✓ Load tiles with one function call
    ✓ Configure mesh density with target vertices (not magic numbers)
    ✓ Cardinal direction camera positioning with view-specific framing
    ✓ Built-in color mapping from elevation data
    ✓ Automatic coordinate system handling

Multiple Views:
    The script supports command-line arguments to easily generate renders
    from different camera angles with optimal framing for each view:

    --view Options:
        north, south, east, west      Cardinal directions
        northeast, northwest, southeast, southwest  Diagonal views
        above                         Overhead perspective
                                      (Note: rotation may vary due to camera up-vector ambiguity)
                                      Use --camera-type ORTHO for stable overhead views

    Example Commands:
        npm run py:example:detroit-north
        npm run py:example:detroit-south
        npm run py:example:detroit-east
        npm run py:example:detroit-west
        npm run py:example:detroit-above

Output:
    - PNG saved to: examples/detroit_elevation_{view}.png (960×720)
    - Blender file: examples/detroit_elevation_{view}.blend

Usage:
    python examples/detroit_elevation_real.py [OPTIONS]

Options:
    --view, -v {north,south,east,west,...}  Camera view direction (default: north)
    --output, -o FILE                        Custom output filename
    --camera-type, -c {PERSP,ORTHO}          Camera projection (default: PERSP)
    --distance, -d FLOAT                     Camera distance multiplier (default: 0.264)
    --elevation, -e FLOAT                    Camera elevation multiplier (default: 0.396)
    --focal-length, -f FLOAT                 Focal length in mm (default: 15)

Advanced Usage:
    Create a dramatic southwest view with orthographic projection:
        python examples/detroit_elevation_real.py --view southwest --camera-type ORTHO

    Generate a custom output file:
        python examples/detroit_elevation_real.py --view north --output my_render.png

    Adjust camera distance for a wider view:
        python examples/detroit_elevation_real.py --view north --distance 0.5
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.terrain.core import (
    Terrain, load_dem_files, scale_elevation, flip_raster, reproject_raster,
    elevation_colormap, clear_scene, position_camera_relative, setup_light,
    setup_render_settings, render_scene_to_file
)

try:
    import bpy
except ImportError:
    bpy = None

# SRTM tiles directory
SRTM_TILES_DIR = Path(__file__).parent.parent / "data" / "dem" / "detroit"

# View-specific camera target offsets for better framing
# Adjusts the focus point based on view direction to keep terrain centered
VIEW_TARGET_OFFSETS = {
    'north': (0, 2, 0),
    'south': (0, -1.5, 0),
    'east': (1.5, 0, 0),
    'west': (-1.5, 0, 0),
    'above': (0, 0, 0),  # Overhead view - center is natural target
}


def parse_args():
    """Parse command line arguments for camera view configuration."""
    parser = argparse.ArgumentParser(
        description='Detroit Real Elevation Visualization with configurable camera views'
    )
    parser.add_argument(
        '--view', '-v',
        choices=['north', 'south', 'east', 'west',
                 'northeast', 'northwest', 'southeast', 'southwest', 'above'],
        default='north',
        help='Camera direction/view (default: north)'
    )
    parser.add_argument(
        '--distance', '-d',
        type=float,
        default=0.264,  # .33/1.25
        help='Distance multiplier from mesh diagonal (default: 0.264)'
    )
    parser.add_argument(
        '--elevation', '-e',
        type=float,
        default=0.396,  # .33/1.25*1.5
        help='Camera elevation as fraction of mesh diagonal (default: 0.396)'
    )
    parser.add_argument(
        '--camera-type', '-c',
        choices=['PERSP', 'ORTHO'],
        default='PERSP',
        help='Camera type: PERSP (perspective) or ORTHO (orthographic) (default: PERSP)'
    )
    parser.add_argument(
        '--focal-length', '-f',
        type=float,
        default=15,
        help='Focal length in mm for perspective camera (default: 15)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output filename (default: detroit_elevation_{view}.png)'
    )
    return parser.parse_args()


def main():
    """Run the Detroit real elevation visualization."""
    args = parse_args()

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

    WIDTH = 960
    HEIGHT = 720

    # Configure downsampling to target approximately 500,000 vertices
    # This automatically calculates the optimal zoom_factor
    target_vertices = WIDTH*HEIGHT*2
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
    # Use beautiful Mako colormap for elevation visualization
    # (Mako = perceptually uniform, great for elevation data)
    terrain.set_color_mapping(
        lambda dem: elevation_colormap(dem, cmap_name='mako'),
        source_layers=['dem']
    )
    print(f"      Color mapping configured (Mako colormap)")

    # Step 5: Create Blender mesh
    print("\n[5/6] Creating Blender mesh...")
    try:
        mesh_obj = terrain.create_mesh(
            scale_factor=100.0,  
            height_scale=4.0,  
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

    # Get camera target offset for this view direction (or default to center)
    look_at = VIEW_TARGET_OFFSETS.get(args.view, (0, 0, 0))

    # Position camera using cardinal directions
    # This is much more intuitive than manual coordinate specification
    camera = position_camera_relative(
        mesh_obj,
        look_at=look_at,
        direction=args.view,
        distance=args.distance,
        elevation=args.elevation,
        camera_type=args.camera_type,
        focal_length=args.focal_length,
    )

    # Create sun light for terrain illumination
    light = setup_light(
        angle=2,            # Narrow light cone
        energy=3            # Brightness
    )

    # Configure render settings using class method
    setup_render_settings(
        use_gpu=True,
        samples=2048,
        use_denoising=False
    )

    print(f"      Camera: {args.view.title()}-facing cardinal view")
    print(f"      Direction: {args.view}, distance: {args.distance:.3f}x, elevation: {args.elevation:.3f}x")
    print(f"      Type: {args.camera_type}")
    print(f"      Samples: 2048")
    print(f"      Rendering...")

    # Render scene to file using class method
    if args.output:
        output_path = Path(__file__).parent / args.output
    else:
        output_path = Path(__file__).parent / f"detroit_elevation_{args.view}.png"
    render_file = render_scene_to_file(
        output_path=output_path,
        width=WIDTH,
        height=HEIGHT,
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
    print(f"  ✓ Configured downsampling to target vertex count intelligently")
    print(f"  ✓ Applied geographic coordinate reprojection (WGS84 → UTM)")
    print(f"  ✓ Created Terrain object with real elevation data")
    print(f"  ✓ Applied transforms (reproject + flip + scale)")
    print(f"  ✓ Configured beautiful Mako elevation-based color mapping")
    print(f"  ✓ Generated Blender mesh with {len(mesh_obj.data.vertices)} vertices")
    if render_file:
        print(f"  ✓ Rendered to PNG: {render_file}")
    print(f"\nThat's it! Professional terrain visualization in just a few lines of Python!")

    return 0


if __name__ == '__main__':
    exit(main())
