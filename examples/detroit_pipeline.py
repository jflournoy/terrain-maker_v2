#!/usr/bin/env python3
"""
Example: Using the TerrainPipeline for declarative dependency management.

This demonstrates the Python-native DAG system for terrain rendering.
Instead of manual cache checking and dependency tracking, the pipeline
automatically handles:
- Task dependencies
- Cache key computation
- Execution planning
- Staleness detection

Usage:
    # Show execution plan (dry run)
    python examples/detroit_pipeline.py --explain south

    # Render a single view
    python examples/detroit_pipeline.py --view south

    # Render all views (mesh built once, reused for all)
    python examples/detroit_pipeline.py --all

    # Clear cache and rebuild
    python examples/detroit_pipeline.py --clear

    # Force rebuild even if cache exists (without clearing other cache)
    python examples/detroit_pipeline.py --view south --force

    # With caching disabled
    python examples/detroit_pipeline.py --no-cache --view south

Examples:
    # Show what would execute to build the south view
    python examples/detroit_pipeline.py --explain south

    # Render all 5 cardinal views with mesh reuse
    python examples/detroit_pipeline.py --all --cache

    # Render with different water detection threshold
    python examples/detroit_pipeline.py --view north --slope-threshold 0.05

    # High-quality render with more samples
    python examples/detroit_pipeline.py --view south --samples 4096
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.terrain.pipeline import TerrainPipeline

try:
    import bpy
except ImportError:
    bpy = None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Detroit terrain visualization using dependency pipeline"
    )

    # Main commands
    parser.add_argument(
        "--view",
        "-v",
        choices=["north", "south", "east", "west", "above"],
        default="south",
        help="Camera view direction (default: south)",
    )
    parser.add_argument(
        "--all", action="store_true", help="Render all views (mesh built once, reused for all)"
    )
    parser.add_argument(
        "--explain", type=str, metavar="TASK", help="Show execution plan for a task without running"
    )

    # Cache options
    parser.add_argument(
        "--cache", action="store_true", default=True, help="Enable caching (default: enabled)"
    )
    parser.add_argument("--no-cache", action="store_false", dest="cache", help="Disable caching")
    parser.add_argument("--clear", action="store_true", help="Clear cache and rebuild from scratch")
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force rebuild even if cache exists (keeps other cached data)",
    )

    # Pipeline parameters
    parser.add_argument(
        "--target-vertices",
        type=int,
        default=1382400,
        help="Mesh density target (default: WIDTH*HEIGHT*2 = 1382400)",
    )
    parser.add_argument(
        "--slope-threshold",
        type=float,
        default=0.01,
        help="Water detection threshold (default: 0.01)",
    )

    # Render parameters
    parser.add_argument(
        "--samples", type=int, default=2048, help="Render samples for quality (default: 2048)"
    )
    parser.add_argument(
        "--width", type=int, default=960, help="Output width in pixels (default: 960)"
    )
    parser.add_argument(
        "--height", type=int, default=720, help="Output height in pixels (default: 720)"
    )
    parser.add_argument(
        "--distance", type=float, default=0.264, help="Camera distance multiplier (default: 0.264)"
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=0.396,
        help="Camera elevation multiplier (default: 0.396)",
    )

    return parser.parse_args()


def main():
    """Run the pipeline."""
    args = parse_args()

    print("\n" + "=" * 70)
    print("Detroit Terrain Visualization - Dependency Pipeline")
    print("=" * 70 + "\n")

    # Initialize pipeline
    dem_dir = Path(__file__).parent.parent / "data" / "dem" / "detroit"
    pipeline = TerrainPipeline(
        dem_dir=dem_dir, cache_enabled=args.cache, force_rebuild=args.force, verbose=True
    )

    # Handle cache clearing
    if args.clear:
        print("[Cache] Clearing all cached files...\n")
        deleted = pipeline.clear_cache()
        print(f"[Cache] Cleared {deleted} files\n")

    # Show execution plan if requested
    if args.explain:
        pipeline.explain(args.explain)
        return 0

    # Render single view
    if not args.all:
        print(f"[Pipeline] Building: render_{args.view}")
        print(f"  View: {args.view}")
        print(f"  Resolution: {args.width}×{args.height}")
        print(f"  Samples: {args.samples}")
        cache_status = "enabled"
        if not args.cache:
            cache_status = "disabled"
        elif args.force:
            cache_status = "enabled (force rebuild)"
        print(f"  Cache: {cache_status}\n")

        # Build the render task
        # In a real implementation, this would call pipeline.build('render_view')
        # For now, show the concept:
        pipeline._log(f"\n[→] Loading DEM")
        pipeline._log(f"[→] Applying transforms (target_vertices={args.target_vertices:,})")
        pipeline._log(f"[→] Detecting water (slope_threshold={args.slope_threshold})")
        pipeline._log(f"[→] Creating mesh (reuse from cache if available)")
        pipeline._log(f"[→] Rendering {args.view} view")

        # Show cache stats
        stats = pipeline.cache_stats()
        if stats["total_files"] > 0:
            print(f"\n[Cache] Stats:")
            print(
                f"  DEM cache: {stats['dem']['cache_files']} files, {stats['dem']['total_size_mb']:.1f} MB"
            )
            print(
                f"  Mesh cache: {stats['mesh']['blend_files']} files, {stats['mesh']['total_size_mb']:.1f} MB"
            )
            print(f"  Total: {stats['total_files']} files, {stats['total_mb']:.1f} MB\n")

        return 0

    # Render all views
    if args.all:
        print("[Pipeline] Building: render_all_views")
        print("  This efficiently reuses the mesh across all views\n")

        views = ["north", "south", "east", "west", "above"]
        results = pipeline.render_all_views(views=views, use_cache=args.cache)

        print(f"\n[✓] Rendered {len(results)} views:")
        for view, path in results.items():
            print(f"  {view:8} → {path.name}")

        # Show final cache stats
        print("\n[Cache] Final stats:")
        stats = pipeline.cache_stats()
        print(
            f"  DEM cache: {stats['dem']['cache_files']} files, {stats['dem']['total_size_mb']:.1f} MB"
        )
        print(
            f"  Mesh cache: {stats['mesh']['blend_files']} files, {stats['mesh']['total_size_mb']:.1f} MB"
        )
        print(f"  Total: {stats['total_files']} files, {stats['total_mb']:.1f} MB\n")

        return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code or 0)
    except KeyboardInterrupt:
        print("\n[✗] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[✗] Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
