#!/usr/bin/env python3
"""
Road Network Data Fetcher CLI.

Command-line interface for fetching road network data from OpenStreetMap.
The actual fetching logic is in the library: src/terrain/roads.py

Usage:
    # Fetch roads for San Diego
    python examples/detroit_roads.py --bbox 32.5 -117.6 33.5 -116.0

    # Fetch only interstates
    python examples/detroit_roads.py --bbox 32.5 -117.6 33.5 -116.0 --road-types motorway

    # Save to specific file
    python examples/detroit_roads.py --bbox 42.3 -83.5 42.5 -82.8 --output roads.geojson

Library usage:
    from src.terrain.roads import get_roads, get_roads_tiled

    # Fetch roads for a bounding box
    bbox = (42.3, -83.5, 42.5, -82.8)  # (south, west, north, east)
    roads = get_roads(bbox, road_types=['motorway', 'trunk', 'primary'])

    # For large areas, use tiled fetching
    large_bbox = (37.0, -95.0, 45.0, -85.0)
    roads = get_roads_tiled(large_bbox, tile_size=2.0)
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from library
from src.terrain.roads import get_roads, get_roads_tiled


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch road data from OpenStreetMap Overpass API"
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=[42.25, -83.5, 42.5, -82.8],
        metavar=("SOUTH", "WEST", "NORTH", "EAST"),
        help="Bounding box: south west north east (default: Detroit area)",
    )
    parser.add_argument(
        "--road-types",
        nargs="+",
        default=["motorway", "trunk", "primary"],
        help="Road types to fetch (default: motorway trunk primary)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output GeoJSON file (optional)",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force fresh fetch, skip cache",
    )
    parser.add_argument(
        "--tiled",
        action="store_true",
        help="Use tiled fetching for large areas",
    )
    parser.add_argument(
        "--tile-size",
        type=float,
        default=2.0,
        help="Tile size in degrees for tiled fetching (default: 2.0)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    # Fetch roads
    bbox = tuple(args.bbox)

    if args.tiled:
        roads = get_roads_tiled(
            bbox,
            args.road_types,
            tile_size=args.tile_size,
            force_refresh=args.force_refresh,
        )
    else:
        roads = get_roads(bbox, args.road_types, force_refresh=args.force_refresh)

    if roads and args.output:
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(roads, f, indent=2)
        print(f"Saved {len(roads['features'])} roads to {output_path}")
    elif roads:
        print(f"Fetched {len(roads['features'])} road segments")
