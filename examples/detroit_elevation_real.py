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
    1. Load and merge SRTM HGT tiles
    2. Crop to Detroit metro area (42.25-42.45°N, 83.25-82.75°W)
    3. Initialize Terrain object
    4. Apply transforms
    5. Set up color mapping (elevation-based)
    6. Generate Blender mesh
    7. Render to PNG

Output:
    - PNG saved to: examples/detroit_elevation_real.png (1920×1440)
    - Blender file: examples/detroit_elevation_real.blend

Usage:
    python examples/detroit_elevation_real.py
"""

import sys
from pathlib import Path
from math import radians
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.windows import from_bounds
from affine import Affine

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.terrain.core import Terrain

# Detroit metro bounds (rough bounding box)
DETROIT_BOUNDS = {
    'north': 42.45,
    'south': 42.25,
    'east': -82.75,
    'west': -83.25,
}

# SRTM tiles directory
SRTM_TILES_DIR = Path(__file__).parent.parent.parent / "geotiff-rayshade" / "detroit"


def load_detroit_dem():
    """
    Load and merge SRTM tiles covering Detroit metro area.

    Returns:
        tuple: (dem_data, transform) - elevation array and affine transform
    """
    print("[1/6] Loading SRTM tiles...")

    if not SRTM_TILES_DIR.exists():
        raise FileNotFoundError(f"SRTM tiles directory not found: {SRTM_TILES_DIR}")

    # Find all HGT files
    hgt_files = sorted(SRTM_TILES_DIR.glob("*.hgt"))

    if not hgt_files:
        raise FileNotFoundError(f"No .hgt files found in {SRTM_TILES_DIR}")

    print(f"      Found {len(hgt_files)} SRTM tiles")

    # Open all datasets
    datasets = []
    for hgt_file in hgt_files:
        try:
            ds = rasterio.open(hgt_file)
            datasets.append(ds)
        except Exception as e:
            print(f"      ⚠️  Could not open {hgt_file.name}: {e}")
            continue

    if not datasets:
        raise ValueError("Could not open any SRTM tiles")

    print(f"      Successfully opened {len(datasets)} tiles")
    print(f"      Merging tiles...")

    # Merge all datasets
    merged_dem, merged_transform = merge(datasets)

    # Extract first band (SRTM is single band)
    dem = merged_dem[0].astype(np.float32)

    print(f"      Merged DEM shape: {dem.shape}")
    print(f"      Elevation range: {np.nanmin(dem):.1f} - {np.nanmax(dem):.1f} meters")

    # Crop to Detroit metro bounds
    print(f"      Cropping to Detroit metro bounds...")

    # Convert bounds to pixel window
    window = from_bounds(
        DETROIT_BOUNDS['west'], DETROIT_BOUNDS['south'],
        DETROIT_BOUNDS['east'], DETROIT_BOUNDS['north'],
        merged_transform
    )

    # Read cropped data
    dem_cropped = dem[
        int(window.row_off):int(window.row_off + window.height),
        int(window.col_off):int(window.col_off + window.width)
    ]

    # Calculate new transform for cropped data
    ul_x = merged_transform.c + window.col_off * merged_transform.a
    ul_y = merged_transform.f + window.row_off * merged_transform.e
    transform_cropped = Affine(
        merged_transform.a, merged_transform.b, ul_x,
        merged_transform.d, merged_transform.e, ul_y
    )

    print(f"      Cropped DEM shape: {dem_cropped.shape}")
    print(f"      Cropped elevation range: {np.nanmin(dem_cropped):.1f} - {np.nanmax(dem_cropped):.1f} meters")

    # Close all datasets
    for ds in datasets:
        ds.close()

    return dem_cropped, transform_cropped


def create_color_function():
    """
    Create a color mapping function for elevation-based coloring.

    Maps low elevations to blue (water), medium to green (land),
    and high to brown/gray (hills).
    """
    def elevation_to_rgb(dem):
        """Map DEM elevation to RGB colors."""
        # Handle NaN values
        valid_mask = ~np.isnan(dem)
        dem_valid = dem[valid_mask]

        if len(dem_valid) == 0:
            return np.zeros(dem.shape + (3,), dtype=np.uint8)

        # Normalize elevation to 0-1 range
        dem_min, dem_max = dem_valid.min(), dem_valid.max()
        normalized = np.zeros_like(dem)
        normalized[valid_mask] = (dem[valid_mask] - dem_min) / (dem_max - dem_min + 1e-8)

        # Create RGB colors based on elevation
        red = np.zeros_like(normalized)
        green = np.zeros_like(normalized)
        blue = np.zeros_like(normalized)

        # Low elevations (0-0.33) - blueish green (water/low ground)
        low_mask = (normalized < 0.33) & valid_mask
        red[low_mask] = 0.1
        green[low_mask] = 0.5
        blue[low_mask] = 0.8

        # Mid elevations (0.33-0.66) - green (meadow/land)
        mid_mask = ((normalized >= 0.33) & (normalized < 0.66)) & valid_mask
        red[mid_mask] = 0.3
        green[mid_mask] = 0.7
        blue[mid_mask] = 0.2

        # High elevations (0.66-1.0) - brown/tan (hills)
        high_mask = (normalized >= 0.66) & valid_mask
        red[high_mask] = 0.7
        green[high_mask] = 0.6
        blue[high_mask] = 0.4

        # Set invalid areas to dark gray
        invalid_mask = ~valid_mask
        red[invalid_mask] = 0.3
        green[invalid_mask] = 0.3
        blue[invalid_mask] = 0.3

        # Stack to create RGB and convert to uint8
        rgb = np.stack([red, green, blue], axis=-1)
        return (rgb * 255).astype(np.uint8)

    return elevation_to_rgb


def render_to_png(mesh_obj, output_path=None):
    """Render the Blender scene to PNG."""
    try:
        import bpy
    except ImportError:
        print("\n⚠️  Blender/bpy not available - skipping PNG render")
        return None

    if output_path is None:
        output_path = Path(__file__).parent / "detroit_elevation_real.png"
    else:
        output_path = Path(output_path)

    output_path = output_path.resolve()
    print(f"\n[6/6] Setting up camera and rendering to PNG...")

    try:
        # Tuned camera parameters for Detroit elevation terrain
        camera_location = (7.3589, -6.9257, 4.9583)
        camera_rotation_deg = (63.559, 0, 46.692)
        camera_angle = (
            radians(camera_rotation_deg[0]),
            radians(camera_rotation_deg[1]),
            radians(camera_rotation_deg[2])
        )
        camera_scale = 20.0

        # Delete any existing cameras
        for obj in list(bpy.data.objects):
            if obj.type == 'CAMERA':
                bpy.data.objects.remove(obj, do_unlink=True)

        # Delete any existing lights
        for obj in list(bpy.data.objects):
            if obj.type == 'LIGHT':
                bpy.data.objects.remove(obj, do_unlink=True)

        # Create camera
        cam_data = bpy.data.cameras.new("Camera")
        cam_data.lens = 50
        cam_obj = bpy.data.objects.new("Camera", cam_data)
        bpy.context.scene.collection.objects.link(cam_obj)

        cam_obj.location = camera_location
        cam_obj.rotation_euler = camera_angle
        cam_data.type = 'PERSP'
        cam_data.ortho_scale = camera_scale

        bpy.context.scene.camera = cam_obj

        # Create sun light
        sun = bpy.data.lights.new(name="Sun", type='SUN')
        sun_obj = bpy.data.objects.new("Sun", sun)
        bpy.context.scene.collection.objects.link(sun_obj)

        sun_obj.location = (10, 10, 10)
        sun_obj.rotation_euler = (radians(45), radians(45), radians(0))
        sun.angle = 2
        sun.energy = 3

        # Configure render settings
        bpy.context.scene.render.filepath = str(output_path)
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'
        bpy.context.scene.render.image_settings.compression = 90

        bpy.context.scene.render.resolution_x = 1920
        bpy.context.scene.render.resolution_y = 1440
        bpy.context.scene.render.resolution_percentage = 100

        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.samples = 32

        print(f"      Camera: TUNED VIEW")
        print(f"      Location: {camera_location}")
        print(f"      Rotation: {camera_rotation_deg}")
        print(f"      Samples: 32")
        print(f"      Rendering...")

        # Render
        bpy.ops.render.render(write_still=True)

        if output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"      ✓ Rendered successfully!")
            print(f"      File: {output_path.name}")
            print(f"      Size: {file_size_mb:.1f} MB")
        else:
            print(f"      ✗ Render file not created")
            return None

        # Export Blender file
        blend_path = output_path.parent / "detroit_elevation_real.blend"
        try:
            bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
            print(f"      ✓ Saved Blender file: {blend_path.name}")
        except Exception as e:
            print(f"      ⚠️  Could not save Blender file: {e}")

        return output_path

    except Exception as e:
        print(f"      ✗ Render failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run the Detroit real elevation visualization."""
    print("=" * 70)
    print("Detroit Real Elevation Visualization")
    print("=" * 70)

    # Step 1: Load DEM
    try:
        dem_data, transform = load_detroit_dem()
    except Exception as e:
        print(f"\n✗ Failed to load DEM: {e}")
        return 1

    # Step 2: Initialize Terrain
    print("\n[2/6] Initializing Terrain object...")
    terrain = Terrain(dem_data, transform)
    print(f"      Terrain initialized")
    print(f"      DEM shape: {terrain.dem_shape}")

    # Step 3: Apply transforms
    print("\n[3/6] Applying coordinate transforms...")

    def identity_transform(data, trans):
        """Identity transform (no-op for demonstration)."""
        return data, trans, None

    terrain.transforms.append(identity_transform)
    terrain.apply_transforms()
    print(f"      Transforms applied successfully")

    # Step 4: Set up color mapping
    print("\n[4/6] Setting up color mapping...")
    color_func = create_color_function()
    terrain.set_color_mapping(color_func, source_layers=['dem'])
    print(f"      Color mapping configured")

    # Step 5: Create Blender mesh
    print("\n[5/6] Creating Blender mesh...")
    try:
        mesh_obj = terrain.create_mesh(
            scale_factor=120.0,
            height_scale=0.007,  # Reduced height exaggeration
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
    render_file = render_to_png(mesh_obj)

    # Summary
    print("\n" + "=" * 70)
    print("Detroit Real Elevation Visualization Complete!")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  ✓ Loaded SRTM tiles covering Detroit metro area")
    print(f"  ✓ Cropped to Detroit bounds ({DETROIT_BOUNDS['south']:.2f}N-{DETROIT_BOUNDS['north']:.2f}N, "
          f"{DETROIT_BOUNDS['west']:.2f}E-{DETROIT_BOUNDS['east']:.2f}E)")
    print(f"  ✓ Created Terrain object with real elevation data")
    print(f"  ✓ Applied transforms")
    print(f"  ✓ Configured elevation-based color mapping")
    print(f"  ✓ Generated Blender mesh with {len(mesh_obj.data.vertices)} vertices")
    if render_file:
        print(f"  ✓ Rendered to PNG: {render_file}")
    print(f"\nThe terrain visualization is ready!")

    return 0


if __name__ == '__main__':
    exit(main())
