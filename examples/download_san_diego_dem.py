#!/usr/bin/env python3
"""
Quick demo: Download SRTM elevation data for San Diego County.

This demonstrates the DEM downloader by fetching real SRTM tiles from NASA Earthdata.

Requirements:
    - NASA Earthdata account (free): https://urs.earthdata.nasa.gov/users/new
    - Set environment variables EARTHDATA_USERNAME and EARTHDATA_PASSWORD

Usage:
    python examples/download_san_diego_dem.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load credentials from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("💡 Tip: Install python-dotenv to use .env files: pip install python-dotenv")

from src.terrain.dem_downloader import (
    download_dem_by_bbox,
    display_bbox_on_map,
    calculate_required_srtm_tiles,
)


def main():
    """Download SRTM data for San Diego County."""

    # San Diego County approximate bounding box
    # Covers from coast to inland mountains
    bbox = (
        32.5,    # south (near Mexican border)
        -117.6,  # west (Pacific coast)
        33.5,    # north (Orange County border)
        -116.0,  # east (desert/mountains)
    )

    print("=" * 60)
    print("SRTM Download Demo - San Diego County")
    print("=" * 60)

    # Calculate which tiles we need
    tiles = calculate_required_srtm_tiles(bbox)
    print(f"\n📍 Bounding Box: {bbox}")
    print(f"   South: {bbox[0]}°, West: {bbox[1]}°")
    print(f"   North: {bbox[2]}°, East: {bbox[3]}°")
    print(f"\n📦 Required SRTM tiles: {tiles}")
    print(f"   ({len(tiles)} tiles total)")

    # Create visualization
    output_dir = Path("data/san_diego_dem")
    map_file = output_dir / "san_diego_bbox.html"
    output_dir.mkdir(parents=True, exist_ok=True)

    display_bbox_on_map(bbox, output_file=str(map_file))
    print(f"\n🗺️  Created bbox visualization: {map_file}")
    print("   Open this file in a browser to see the area!")

    # Get credentials from environment (.env file or exports)
    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")

    if not username or not password:
        print("\n⚠️  NASA Earthdata credentials not found!")
        print("\n   Option 1: Use .env file (recommended)")
        print("   1. Copy .env.example to .env")
        print("   2. Add your credentials to .env")
        print("\n   Option 2: Export environment variables")
        print("   export EARTHDATA_USERNAME='your_username'")
        print("   export EARTHDATA_PASSWORD='your_password'")
        print("\n   Get free account: https://urs.earthdata.nasa.gov/users/new")
        print("\n   Skipping download for now...")
        return

    # Download tiles
    print(f"\n⬇️  Downloading SRTM tiles to: {output_dir}")
    print("   This may take a few minutes...")

    files = download_dem_by_bbox(
        bbox=bbox,
        output_dir=str(output_dir),
        username=username,
        password=password,
    )

    print(f"\n✅ Download complete!")
    print(f"   Downloaded {len(files)} tiles:")
    for file in files:
        if file.exists():
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"   - {file.name} ({size_mb:.1f} MB)")

    # Summary
    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. Open the HTML map to visualize the area")
    print("  2. Use these HGT files with terrain-maker!")
    print("=" * 60)


if __name__ == "__main__":
    main()
