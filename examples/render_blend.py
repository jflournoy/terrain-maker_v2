#!/usr/bin/env python3
"""
Render an existing Blender file.

Re-renders a .blend file saved from a previous terrain-maker run,
allowing you to adjust render settings without recomputing data.

Usage:
    python examples/render_blend.py san_diego_flow_rainfall.blend
    python examples/render_blend.py san_diego_flow_rainfall.blend --samples 512 --resolution 1920x1080
    python examples/render_blend.py san_diego_flow_rainfall.blend -o new_render.png --samples 1024
"""

import argparse
import sys
from pathlib import Path

import bpy


def main():
    parser = argparse.ArgumentParser(
        description="Render an existing Blender file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick preview render
  python examples/render_blend.py scene.blend --samples 64

  # High quality render
  python examples/render_blend.py scene.blend --samples 2048 --resolution 3840x2160

  # Render to specific output
  python examples/render_blend.py scene.blend -o final_render.png

  # Use CPU if GPU runs out of memory
  python examples/render_blend.py scene.blend --device CPU --samples 256
        """,
    )
    parser.add_argument(
        "blend_file",
        type=Path,
        help="Path to .blend file to render",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output image path (default: <blend_name>_render.png)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Render samples (default: use scene settings)",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=None,
        help="Resolution as WIDTHxHEIGHT (e.g., 1920x1080)",
    )
    parser.add_argument(
        "--device",
        choices=["GPU", "CPU"],
        default="GPU",
        help="Render device (default: GPU)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Tile size in pixels (smaller = less GPU memory, default: auto)",
    )

    args = parser.parse_args()

    # Validate input
    if not args.blend_file.exists():
        # Try looking in examples/output
        alt_path = Path("examples/output") / args.blend_file
        if alt_path.exists():
            args.blend_file = alt_path
        else:
            print(f"Error: Blend file not found: {args.blend_file}")
            sys.exit(1)

    # Determine output path
    if args.output is None:
        args.output = args.blend_file.with_suffix("").parent / (args.blend_file.stem + "_render.png")

    print(f"Loading: {args.blend_file}")
    bpy.ops.wm.open_mainfile(filepath=str(args.blend_file))

    # Parse resolution if provided
    width, height = None, None
    if args.resolution:
        try:
            width, height = map(int, args.resolution.lower().split("x"))
        except ValueError:
            print(f"Error: Invalid resolution format: {args.resolution}")
            print("  Use WIDTHxHEIGHT format, e.g., 1920x1080")
            sys.exit(1)

    # Update render settings
    scene = bpy.context.scene

    if width and height:
        scene.render.resolution_x = width
        scene.render.resolution_y = height
        print(f"Resolution: {width}x{height}")
    else:
        print(f"Resolution: {scene.render.resolution_x}x{scene.render.resolution_y}")

    if args.samples:
        scene.cycles.samples = args.samples
    print(f"Samples: {scene.cycles.samples}")

    # Set device
    scene.cycles.device = args.device
    if args.device == "GPU":
        # Enable GPU compute
        prefs = bpy.context.preferences.addons["cycles"].preferences
        # Try OPTIX first, fall back to CUDA
        for device_type in ["OPTIX", "CUDA", "HIP", "METAL"]:
            try:
                prefs.compute_device_type = device_type
                prefs.get_devices()
                gpu_found = False
                for device in prefs.devices:
                    if device.type != "CPU":
                        device.use = True
                        gpu_found = True
                if gpu_found:
                    print(f"Device: GPU ({device_type})")
                    break
            except Exception:
                continue
    else:
        print(f"Device: CPU")

    # Set tile size (smaller = less memory usage)
    if args.tile_size:
        scene.cycles.tile_size = args.tile_size
        print(f"Tile size: {args.tile_size}px")

    # Set output path
    scene.render.filepath = str(args.output)
    scene.render.image_settings.file_format = "PNG"

    print(f"Output: {args.output}")
    print()
    print("Rendering...")

    # Render
    bpy.ops.render.render(write_still=True)

    print(f"\nRendered: {args.output}")


if __name__ == "__main__":
    main()
