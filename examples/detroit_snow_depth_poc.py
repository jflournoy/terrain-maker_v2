#!/usr/bin/env python3
"""
Detroit Snow Depth Visualization - Proof of Concept

This example demonstrates the terrain-maker library workflow:
1. Load DEM data (synthetic Detroit elevation)
2. Create a terrain object
3. Apply coordinate transforms
4. Set up color mapping
5. Generate a Blender mesh
6. Render to PNG

Output:
    - PNG saved to: examples/detroit_terrain.png (1920×1440)

Usage:
    python examples/detroit_snow_depth_poc.py
"""

import sys
from pathlib import Path
from math import radians

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from affine import Affine

# Import terrain-maker library
from src.terrain.core import Terrain, setup_camera_and_light


def create_detroit_dem():
    """Create synthetic Detroit DEM data.

    Detroit is relatively flat with elevations 160-320 meters.
    This creates a synthetic DEM with gentle rolling terrain.
    """
    # Create base elevation
    height, width = 100, 100
    dem = np.ones((height, width), dtype=np.float32) * 250  # Base elevation

    # Add gentle rolling hills
    x = np.linspace(0, 4 * np.pi, width)
    y = np.linspace(0, 4 * np.pi, height)
    X, Y = np.meshgrid(x, y)

    # Add wave pattern (simulating terrain variation)
    variation = 30 * (np.sin(X / 4) + np.cos(Y / 4))
    dem = dem + variation.astype(np.float32)

    # Clip to realistic Detroit elevation range
    dem = np.clip(dem, 160, 320).astype(np.float32)

    return dem


def create_color_function():
    """Create a color mapping function for elevation-based coloring.

    Maps low elevations to blue (water), medium to green (land),
    and high to brown/gray (mountains).
    """
    def elevation_to_rgb(dem):
        """Map DEM elevation to RGB colors.

        Args:
            dem: (H, W) elevation data

        Returns:
            (H, W, 3) RGB color array (uint8)
        """
        # Normalize elevation to 0-1 range
        dem_min, dem_max = dem.min(), dem.max()
        normalized = (dem - dem_min) / (dem_max - dem_min + 1e-8)

        # Create RGB colors based on elevation
        red = np.zeros_like(normalized)
        green = np.zeros_like(normalized)
        blue = np.zeros_like(normalized)

        # Low elevations (160-230m) - blueish green (water/low ground)
        low_mask = normalized < 0.33
        red[low_mask] = 0.1
        green[low_mask] = 0.5
        blue[low_mask] = 0.7

        # Mid elevations (230-280m) - green (meadow/land)
        mid_mask = (normalized >= 0.33) & (normalized < 0.66)
        red[mid_mask] = 0.3
        green[mid_mask] = 0.6
        blue[mid_mask] = 0.2

        # High elevations (280-320m) - brown (hills)
        high_mask = normalized >= 0.66
        red[high_mask] = 0.6
        green[high_mask] = 0.4
        blue[high_mask] = 0.2

        # Stack to create RGB and convert to uint8
        rgb = np.stack([red, green, blue], axis=-1)
        return (rgb * 255).astype(np.uint8)

    return elevation_to_rgb


def render_to_png(output_path=None):
    """Render the Blender scene to PNG.

    Args:
        output_path: Path to save the rendered PNG (default: examples/detroit_terrain.png)

    Returns:
        Path to the rendered file, or None if rendering failed
    """
    try:
        import bpy
    except ImportError:
        print("\n⚠️  Blender/bpy not available - skipping PNG render")
        print("    (POC still successful - mesh was created)")
        return None

    if output_path is None:
        output_path = Path(__file__).parent / "detroit_terrain.png"
    else:
        output_path = Path(output_path)

    output_path = output_path.resolve()
    print(f"\n[6/6] Setting up camera and rendering to PNG...")

    try:
        # Simple top-down view: camera directly above mesh looking straight down
        # This is easy to debug - if the mesh exists, we'll see it
        camera_location = (0, 0, 150)  # Directly above origin at z=150
        camera_angle = (radians(90), radians(0), radians(0))  # 90° pitch = looking straight down
        camera_scale = 250  # Zoom level

        setup_camera_and_light(
            camera_angle=camera_angle,
            camera_location=camera_location,
            scale=camera_scale,
            sun_angle=45,  # Sun from above too
            sun_energy=2,
            focal_length=50
        )

        # Set basic render output settings
        bpy.context.scene.render.filepath = str(output_path)
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'
        bpy.context.scene.render.image_settings.compression = 90

        # Set render resolution
        bpy.context.scene.render.resolution_x = 1920
        bpy.context.scene.render.resolution_y = 1440
        bpy.context.scene.render.resolution_percentage = 100

        # Set render engine and samples
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.samples = 32

        print(f"      Camera: TOP-DOWN VIEW at {camera_location}")
        print(f"      Looking: straight down (90° pitch)")
        print(f"      Samples: 32")
        print(f"      Rendering...")

        # Render and save
        bpy.ops.render.render(write_still=True)

        if output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"      ✓ Rendered successfully!")
            print(f"      File: {output_path.name}")
            print(f"      Size: {file_size_mb:.1f} MB")
            return output_path
        else:
            print(f"      ✗ Render file not created")
            return None

    except Exception as e:
        print(f"      ✗ Render failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run the Detroit snow depth visualization proof of concept."""
    print("=" * 60)
    print("Detroit Snow Depth Visualization - Proof of Concept")
    print("=" * 60)

    # Step 1: Create synthetic DEM
    print("\n[1/6] Creating synthetic Detroit DEM...")
    dem_data = create_detroit_dem()
    print(f"      DEM shape: {dem_data.shape}")
    print(f"      Elevation range: {dem_data.min():.1f} - {dem_data.max():.1f} meters")

    # Step 2: Initialize Terrain object
    print("\n[2/6] Initializing Terrain object...")
    transform = Affine.identity()
    terrain = Terrain(dem_data, transform)
    print(f"      Terrain initialized with {len(terrain.data_layers)} layer(s)")
    print(f"      Available layers: {list(terrain.data_layers.keys())}")

    # Step 3: Apply transforms
    print("\n[3/6] Applying coordinate transforms...")

    # Register an identity transform (no-op, just for demonstration)
    def identity_transform(data, trans):
        """Identity transform (returns data unchanged)."""
        return data, trans, None

    terrain.transforms.append(identity_transform)
    terrain.apply_transforms()
    print(f"      Transforms applied successfully")
    print(f"      Transformed layer status: {terrain.data_layers['dem']['transformed']}")

    # Step 4: Set up color mapping
    print("\n[4/6] Setting up color mapping...")
    color_func = create_color_function()
    terrain.set_color_mapping(color_func, source_layers=['dem'])
    print(f"      Color mapping configured for layer 'dem'")
    print(f"      Color function: {color_func.__name__}")

    # Step 5: Create Blender mesh
    print("\n[5/6] Creating Blender mesh...")
    mesh_obj = terrain.create_mesh(
        scale_factor=100.0,
        center_model=True,
        boundary_extension=True
    )

    if mesh_obj is not None:
        print(f"      Mesh created successfully!")
        print(f"      Vertices: {len(mesh_obj.data.vertices)}")
        print(f"      Polygons: {len(mesh_obj.data.polygons)}")
        print(f"      Mesh object: {mesh_obj.name}")
        print(f"      Mesh data: {mesh_obj.data.name}")
    else:
        print(f"      ERROR: Mesh creation failed!")
        return 1

    # Step 6: Render to PNG
    render_file = render_to_png()

    # Success summary
    print("\n" + "=" * 60)
    print("Proof of Concept Complete!")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  ✓ Loaded synthetic Detroit DEM (100x100 pixels)")
    print(f"  ✓ Created Terrain object with data layer")
    print(f"  ✓ Applied transforms to DEM data")
    print(f"  ✓ Configured elevation-based color mapping")
    print(f"  ✓ Generated Blender mesh with {len(mesh_obj.data.vertices)} vertices")
    if render_file:
        print(f"  ✓ Rendered to PNG: {render_file}")
    print(f"\nThe terrain visualization is ready!")
    print(f"The mesh can be further customized or exported to 3D file formats")
    print(f"(OBJ, FBX, GLTF, etc.)")

    return 0


if __name__ == '__main__':
    exit(main())
