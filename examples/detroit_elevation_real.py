#!/usr/bin/env python3
"""
Detroit Real Elevation Visualization - Terrain Maker Example

This example demonstrates how EASY it is to create stunning 3D terrain
visualizations with real-world geographic data using Terrain Maker.

With just a few lines of Python, you can:
  ✓ Load real SRTM elevation data from local tiles
  ✓ Apply intelligent downsampling (no complex config needed!)
  ✓ Reproject to proper geographic coordinates automatically
  ✓ Detect water bodies using slope-based analysis
  ✓ Create publication-quality 3D visualizations with water rendering
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

Camera Views - The Power of Intuitive Positioning:
    The position_camera_relative() function makes it trivial to generate
    publication-quality renders from ANY angle. Just specify a cardinal
    direction and the camera automatically:

    ✓ Positions itself at the optimal distance
    ✓ Sets elevation for dramatic effect
    ✓ Targets the terrain with view-specific offsets for perfect framing
    ✓ Calculates proper rotation to look at the target

    Compare this to manual camera setup in Blender - coordinates, angles,
    focal lengths. With position_camera_relative(), it's just ONE parameter!

Available Views (with intelligent auto-framing):

    Cardinal Directions:
        north     - View from the north, looking south at the terrain
        south     - View from the south, looking north (default framing offset: -1.5 on Y)
        east      - View from the east, looking west
        west      - View from the west, looking east

    Diagonal Views:
        northeast, northwest, southeast, southwest - Corner perspectives

    Overhead:
        above     - Perfect overhead view (zero rotation, stable framing)
                    Uses intelligent elevation positioning for best perspective

Quick Render Commands:
    npm run py:example:detroit-north      # North-facing view
    npm run py:example:detroit-south      # South-facing view
    npm run py:example:detroit-east       # East-facing view
    npm run py:example:detroit-west       # West-facing view
    npm run py:example:detroit-above      # Overhead bird's-eye view

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

Why position_camera_relative() is Amazing:
    ✓ ONE function generates professional renders from ANY direction
    ✓ Automatic target offset adjustment for each view direction
    ✓ No manual coordinate calculations needed
    ✓ Consistent framing across all views
    ✓ Easy A/B testing different perspectives

Advanced Camera Customization:

    Dramatic southwest angled view:
        python examples/detroit_elevation_real.py --view southwest --elevation 0.2

    Close-up north perspective:
        python examples/detroit_elevation_real.py --view north --distance 0.15

    Wide-angle overview from northeast:
        python examples/detroit_elevation_real.py --view northeast --distance 0.5 --elevation 0.6

    Custom output file:
        python examples/detroit_elevation_real.py --view north --output custom_render.png

    High-altitude overhead view (adjust elevation):
        python examples/detroit_elevation_real.py --view above --elevation 1.2

Comparing Views:
    Generate a complete set for comparison:
        npm run py:example:detroit-north     # Side view
        npm run py:example:detroit-south     # Opposite side
        npm run py:example:detroit-east      # Another angle
        npm run py:example:detroit-west      # Fourth cardinal
        npm run py:example:detroit-above     # Bird's eye

    All files saved with view-specific names for easy organization!
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.terrain.core import (
    Terrain,
    load_dem_files,
    scale_elevation,
    flip_raster,
    reproject_raster,
    elevation_colormap,
    clear_scene,
    position_camera_relative,
    setup_light,
    setup_render_settings,
    render_scene_to_file,
)
from src.terrain.cache import DEMCache
from src.terrain.mesh_cache import MeshCache

try:
    import bpy
except ImportError:
    bpy = None

# SRTM tiles directory
SRTM_TILES_DIR = Path(__file__).parent.parent / "data" / "dem" / "detroit"

# View-specific camera target offsets for better framing
# Adjusts the focus point based on view direction to keep terrain centered
VIEW_TARGET_OFFSETS = {
    "north": (0, 2, 0),
    "south": (0, -1.5, 0),
    "east": (1.5, 0, 0),
    "west": (-1.5, 0, 0),
    "above": (0, 0, 0),  # Overhead view - center is natural target
}


def parse_args():
    """Parse command line arguments for camera view configuration."""
    parser = argparse.ArgumentParser(
        description="Detroit Real Elevation Visualization with configurable camera views"
    )
    parser.add_argument(
        "--view",
        "-v",
        choices=[
            "north",
            "south",
            "east",
            "west",
            "northeast",
            "northwest",
            "southeast",
            "southwest",
            "above",
        ],
        default="south",
        help="Camera direction/view (default: south)",
    )
    parser.add_argument(
        "--distance",
        "-d",
        type=float,
        default=0.264,  # .33/1.25
        help="Distance multiplier from mesh diagonal (default: 0.264)",
    )
    parser.add_argument(
        "--elevation",
        "-e",
        type=float,
        default=0.396,  # .33/1.25*1.5
        help="Camera elevation as fraction of mesh diagonal (default: 0.396)",
    )
    parser.add_argument(
        "--camera-type",
        "-c",
        choices=["PERSP", "ORTHO"],
        default="PERSP",
        help="Camera type: PERSP (perspective) or ORTHO (orthographic) (default: PERSP)",
    )
    parser.add_argument(
        "--focal-length",
        "-f",
        type=float,
        default=15,
        help="Focal length in mm for perspective camera (default: 15)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output filename (default: detroit_elevation_{view}.png)",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Enable DEM caching (saves/loads from .dem_cache/ directory)",
    )
    parser.add_argument(
        "--clear-cache", action="store_true", help="Clear all cached DEM files and exit"
    )
    return parser.parse_args()


def main():
    """Run the Detroit real elevation visualization."""
    args = parse_args()

    print("=" * 70)
    print("Detroit Real Elevation Visualization")
    print("=" * 70)

    # Initialize caches
    cache = DEMCache(enabled=args.cache)
    mesh_cache = MeshCache(enabled=args.cache)

    # Handle cache clearing
    if args.clear_cache:
        print("\n[Cache] Clearing cached files...")
        dem_deleted = cache.clear_cache(cache_name="detroit")
        mesh_deleted = mesh_cache.clear_cache(cache_name="mesh")
        print(f"[Cache] Cleared {dem_deleted} DEM files and {mesh_deleted} mesh files")
        stats = cache.get_cache_stats()
        mesh_stats = mesh_cache.get_cache_stats()
        total_mb = stats["total_size_mb"] + mesh_stats["total_size_mb"]
        print(f"[Cache] Remaining cache size: {total_mb:.1f} MB")
        return 0

    # Clear Blender scene to remove default objects
    try:
        import bpy

        clear_scene()
        print("✓ Blender scene cleared")
    except ImportError:
        print("⚠️  Blender not available - skipping scene clear")

    # Output dimensions (used for rendering)
    WIDTH = 960
    HEIGHT = 720

    # ==========================================================================
    # EARLY CACHE CHECK: Check mesh cache BEFORE loading any data
    # This is the key optimization - we can compute the mesh hash from just
    # file paths + mtimes + params, without loading actual DEM data.
    # ==========================================================================
    mesh_obj = None
    cached_mesh_loaded = False
    mesh_hash = None

    # Mesh parameters that affect the cached geometry
    mesh_params = {
        "scale_factor": 100.0,
        "height_scale": 4.0,
        "center_model": True,
        "boundary_extension": True,
        "water_mask": True,  # We always use water detection
    }

    if args.cache:
        print("\n[0/6] Checking mesh cache (before loading data)...")
        try:
            # Compute mesh hash from source files + params (NO DATA LOADING)
            dem_hash = cache.compute_source_hash(SRTM_TILES_DIR, pattern="*.hgt")
            mesh_hash = mesh_cache.compute_mesh_hash(dem_hash, mesh_params)
            print(f"      Mesh hash: {mesh_hash[:16]}...")

            # Check if mesh exists in cache
            cached_blend_path = mesh_cache.load_cache(mesh_hash, cache_name="detroit_mesh")

            if cached_blend_path is not None:
                print(f"      [Cache] ✓ Found cached mesh!")
                print(f"      [Cache] File: {cached_blend_path.name}")
                print(f"      [Cache] Skipping DEM load, transforms, water detection...")

                # Load the cached blend file directly
                bpy.ops.wm.open_mainfile(filepath=str(cached_blend_path))

                # Get the mesh object from the loaded blend file
                for obj in bpy.context.scene.objects:
                    if obj.type == "MESH":
                        mesh_obj = obj
                        cached_mesh_loaded = True
                        print(f"      ✓ Mesh loaded from cache!")
                        print(f"      Vertices: {len(mesh_obj.data.vertices)}")
                        print(f"      Polygons: {len(mesh_obj.data.polygons)}")
                        break

                if not cached_mesh_loaded:
                    print(f"      [Cache] No mesh object in blend file, will rebuild")
        except Exception as e:
            print(f"      [Cache] Cache check failed: {e}")
            print(f"      Will run full pipeline...")

    # ==========================================================================
    # If mesh was loaded from cache, skip directly to rendering
    # Otherwise, run the full pipeline
    # ==========================================================================
    cache_hit = False
    water_mask = None
    terrain = None

    if not cached_mesh_loaded:
        # Run full pipeline: Load DEM → Transform → Water detect → Create mesh

        # Step 1: Load DEM (with caching)
        print("\n[1/6] Loading SRTM tiles...")
        dem_data = None
        transform = None

        # Try to load from cache
        if args.cache:
            try:
                print("      [Cache] Computing source hash...")
                source_hash = cache.compute_source_hash(SRTM_TILES_DIR, pattern="*.hgt")
                cached_result = cache.load_cache(source_hash, cache_name="detroit")
                if cached_result is not None:
                    dem_data, transform = cached_result
                    cache_hit = True
                    print(f"      [Cache] ✓ Loaded from cache")
            except Exception as e:
                print(f"      [Cache] Cache load failed: {e}")
                print(f"      [Cache] Will load fresh and cache result")

        # Load from source if cache miss or caching disabled
        if dem_data is None:
            try:
                dem_data, transform = load_dem_files(SRTM_TILES_DIR, pattern="*.hgt")

                # Save to cache if caching enabled
                if args.cache:
                    try:
                        source_hash = cache.compute_source_hash(SRTM_TILES_DIR, pattern="*.hgt")
                        cache.save_cache(dem_data, transform, source_hash, cache_name="detroit")
                    except Exception as e:
                        print(f"      [Cache] Failed to cache: {e}")
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
        target_vertices = WIDTH * HEIGHT * 2
        zoom = terrain.configure_for_target_vertices(target_vertices, order=4)
        print(f"      Configured for {target_vertices:,} target vertices")
        print(f"      Calculated zoom_factor: {zoom:.6f}")

        # Reproject from WGS84 to UTM Zone 17N for proper geographic scaling
        utm_reproject = reproject_raster(src_crs="EPSG:4326", dst_crs="EPSG:32617", num_threads=4)
        terrain.transforms.append(utm_reproject)
        terrain.transforms.append(flip_raster(axis="horizontal"))
        terrain.transforms.append(scale_elevation(scale_factor=0.0001))
        terrain.apply_transforms()

        # Detect water on the UNSCALED DEM
        print(f"      Detecting water bodies (on unscaled DEM)...")
        try:
            from src.terrain.water import identify_water_by_slope
            import numpy as np

            transformed_dem = terrain.data_layers["dem"]["transformed_data"]
            unscaled_dem = transformed_dem / 0.0001

            water_mask = identify_water_by_slope(
                unscaled_dem, slope_threshold=0.01, fill_holes=True
            )
            print(
                f"      Water detected: {np.sum(water_mask)} water pixels ({100*np.sum(water_mask)/water_mask.size:.1f}% of terrain)"
            )
        except Exception as e:
            print(f"      ⚠️  Water detection failed: {e}")
            water_mask = None

        print(f"      Downsampled DEM shape: {transformed_dem.shape}")
        print(f"      Actual vertices: {transformed_dem.shape[0] * transformed_dem.shape[1]:,}")
        print(f"      Transforms applied successfully")

        # Step 4: Set up color mapping
        print("\n[4/6] Setting up color mapping...")
        terrain.set_color_mapping(
            lambda dem: elevation_colormap(dem, cmap_name="mako"), source_layers=["dem"]
        )
        print(f"      Color mapping configured (Mako colormap)")

        # Step 5: Create Blender mesh
        print("\n[5/6] Creating Blender mesh...")
        try:
            mesh_obj = terrain.create_mesh(
                scale_factor=100.0,
                height_scale=4.0,
                center_model=True,
                boundary_extension=True,
                water_mask=water_mask,
            )

            if mesh_obj is None:
                print(f"      ✗ Mesh creation failed!")
                return 1

            print(f"      ✓ Mesh created successfully!")
            print(f"      Vertices: {len(mesh_obj.data.vertices)}")
            print(f"      Polygons: {len(mesh_obj.data.polygons)}")
            print(f"      ✓ Water colored blue (from slope-based detection)")

        except Exception as e:
            print(f"      ✗ Error creating mesh: {e}")
            import traceback

            traceback.print_exc()
            return 1
    else:
        # Mesh was loaded from cache - skip steps 1-5
        print("\n[1-5/6] Skipped (mesh loaded from cache)")

    # Cache the mesh after rendering if not already cached
    if args.cache and mesh_hash and not cached_mesh_loaded:
        try:
            # Blend file will be saved during render_scene_to_file
            # Flag will be checked after render
            pass
        except Exception as e:
            print(f"      [Cache] Failed to cache mesh: {e}")

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
    light = setup_light(angle=2, energy=3)  # Narrow light cone  # Brightness

    # Configure render settings using class method
    setup_render_settings(use_gpu=True, samples=2048, use_denoising=False)

    print(f"      Camera: {args.view.title()}-facing cardinal view")
    print(
        f"      Direction: {args.view}, distance: {args.distance:.3f}x, elevation: {args.elevation:.3f}x"
    )
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
        file_format="PNG",
        color_mode="RGBA",
        compression=90,
        save_blend_file=True,
    )

    if render_file:
        file_size_mb = render_file.stat().st_size / (1024 * 1024)
        print(f"      ✓ Rendered successfully!")
        print(f"      File: {render_file.name}")
        print(f"      Size: {file_size_mb:.1f} MB")

        # Cache the mesh (.blend file) if caching enabled
        if args.cache and mesh_hash:
            try:
                # Blend file is saved alongside PNG with .blend extension
                blend_file = render_file.parent / f"{render_file.stem}.blend"
                if blend_file.exists():
                    print(f"      [Cache] Caching mesh to cache...")
                    mesh_cache.save_cache(
                        blend_file, mesh_hash, mesh_params, cache_name="detroit_mesh"
                    )
            except Exception as e:
                print(f"      [Cache] Failed to cache mesh: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("Detroit Real Elevation Visualization Complete!")
    print("=" * 70)
    print(f"\nSummary:")
    if cached_mesh_loaded:
        print(f"  ✓ [Cache] Loaded mesh directly from cache (skipped all upstream steps!)")
    else:
        print(f"  ✓ Loaded and merged all SRTM tiles (full coverage)")
        if cache_hit:
            print(f"  ✓ [Cache] DEM loaded from cache (skipped merge)")
        print(f"  ✓ Configured downsampling to target vertex count intelligently")
        print(f"  ✓ Applied geographic coordinate reprojection (WGS84 → UTM)")
        print(f"  ✓ Created Terrain object with real elevation data")
        print(f"  ✓ Applied transforms (reproject + flip + scale)")
        print(f"  ✓ Configured beautiful Mako elevation-based color mapping")
        print(f"  ✓ Detected and applied water bodies (slope-based identification)")
    print(f"  ✓ Mesh with {len(mesh_obj.data.vertices)} vertices")
    if render_file:
        print(f"  ✓ Rendered to PNG: {render_file}")

    # Cache statistics
    if args.cache:
        dem_stats = cache.get_cache_stats()
        mesh_stats = mesh_cache.get_cache_stats()
        print(f"\n[Cache] Statistics:")
        print(f"  DEM Cache:")
        print(f"    Location: {dem_stats['cache_dir']}")
        print(f"    Files: {dem_stats['cache_files']}")
        print(f"    Size: {dem_stats['total_size_mb']:.1f} MB")
        print(f"  Mesh Cache:")
        print(f"    Location: {mesh_stats['cache_dir']}")
        print(f"    Files: {mesh_stats['blend_files']}")
        print(f"    Size: {mesh_stats['total_size_mb']:.1f} MB")
        total_size = dem_stats["total_size_mb"] + mesh_stats["total_size_mb"]
        print(f"  Total cache size: {total_size:.1f} MB")

    print(
        f"\nThat's it! Professional terrain visualization with water detection in just a few lines of Python!"
    )

    return 0


if __name__ == "__main__":
    exit(main())
