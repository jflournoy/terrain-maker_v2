#!/usr/bin/env python3
"""
Quick test: Download a single SRTM tile.

Downloads just one tile for downtown San Diego to test credentials.
Much faster than the full San Diego County download.

Usage:
    python examples/download_single_tile_test.py
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
    print("💡 Tip: Install python-dotenv to use .env files")

from src.terrain.dem_downloader import download_dem_by_bbox, display_bbox_on_map


def main():
    """Download a single SRTM tile for testing."""

    # Small bbox covering just downtown San Diego (1 tile: N32W117)
    bbox = (32.7, -117.2, 32.8, -117.1)

    print("=" * 60)
    print("SRTM Single Tile Test - Downtown San Diego")
    print("=" * 60)
    print(f"\n📍 Bbox: {bbox}")
    print("   This should download just 1 tile: N32W117")

    output_dir = Path("data/test_srtm")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualization
    map_file = output_dir / "test_bbox.html"
    display_bbox_on_map(bbox, output_file=str(map_file))
    print(f"\n🗺️  Map: {map_file}")

    # Get credentials
    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")

    if not username or not password:
        print("\n⚠️  Credentials not found!")
        print("   1. Copy .env.example to .env")
        print("   2. Add your NASA Earthdata credentials")
        return

    print(f"\n⬇️  Downloading to: {output_dir}")

    files = download_dem_by_bbox(
        bbox=bbox,
        output_dir=str(output_dir),
        username=username,
        password=password,
    )

    if files:
        print(f"\n✅ Success! Downloaded {len(files)} tile(s):")
        for file in files:
            if file.exists():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"   - {file.name} ({size_mb:.1f} MB)")
    else:
        print("\n❌ No files downloaded - check credentials or try again")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
