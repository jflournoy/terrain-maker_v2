#!/usr/bin/env python3
"""
Seville, Spain - Overhead Terrain Map with Roads, Railways, and Water Bodies.

Renders a publication-quality overhead terrain map of the Seville metropolitan
area using real SRTM elevation data, OpenStreetMap roads and railways, and
HydroLAKES water body data.

Features:
- SRTM elevation data downloaded automatically via NASA Earthdata
- Turbo colormap for elevation (perceptually uniform rainbow alternative)
- Water bodies from HydroLAKES (global lake/reservoir dataset) in complementary blue
- Major roads (motorway, trunk, primary) as dark overlay
- Railway lines (rail, light_rail) as distinct overlay
- Overhead camera view for map-like presentation
- Print-quality output: 10x8 inches at 150 DPI (1500x1200 pixels)

Data Sources:
- SRTM 1-arc-second (~30m) DEM tiles from NASA Earthdata
- HydroLAKES v10 polygons (local shapefile) for water bodies
- OpenStreetMap Overpass API for roads and railways

Requirements:
- Blender Python API available (bpy)
- NASA Earthdata credentials in .env file, or as env vars
  (EARTHDATA_USERNAME / EARTHDATA_PASSWORD)
- HydroLAKES shapefile in data/hydrolakes/HydroLAKES_polys_v10_shp/
- Internet connection for OSM data (cached after first fetch)

Usage:
    python examples/seville_terrain.py

    # Skip rendering (data pipeline only)
    python examples/seville_terrain.py --no-render

    # Custom output directory
    python examples/seville_terrain.py --output-dir ./renders

    # Adjust height exaggeration
    python examples/seville_terrain.py --height-scale 8
"""

import sys
import argparse
import hashlib
import json
import logging
import math
import time
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

import bpy
import requests
from mathutils import Vector
from rasterio import Affine

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
    setup_hdri_lighting,
    setup_render_settings,
    render_scene_to_file,
)
from src.terrain.roads import (
    get_roads_tiled,
    add_roads_layer,
    rasterize_roads_to_layer,
)
from src.terrain.scene_setup import frame_camera_to_objects
from src.terrain.water_bodies import download_water_bodies, rasterize_lakes_to_mask
from src.terrain.dem_downloader import download_dem_by_bbox

logger = logging.getLogger(__name__)

# =============================================================================
# SEVILLE CONFIGURATION
# =============================================================================

SEVILLE_BBOX = (37.25, -6.15, 37.55, -5.80)
SEVILLE_CITY_BBOX = (37.34, -6.02, 37.42, -5.92)
SEVILLE_DEM_DIR = Path(__file__).parent.parent / "data" / "dem" / "seville"
SEVILLE_UTM_CRS = "EPSG:32630"  # UTM Zone 30N

WIDTH = 10 * 150   # 1500
HEIGHT = 8 * 150   # 1200

# Overlay layers: (layer_name, RGB color, priority — lower = drawn first)
OVERLAY_SPEC = [
    ("water",    (40, 100, 180), 1),   # Cool blue complementing warm turbo
    ("roads",    (10, 10, 10),   5),   # Near-black
    ("railways", (60, 60, 60),   10),  # Dark gray
]


# =============================================================================
# OSM RAILWAY FETCHING
#
# Roads use the library's get_roads_tiled() + add_roads_layer().
# Railways need a custom Overpass query — reuse rasterize_roads_to_layer()
# for the LineString rasterization.
# =============================================================================


def _fetch_overpass_cached(query, cache_key, max_age_days=30, retries=3):
    """Fetch from Overpass API with local caching and retry on 429/504."""
    cache_dir = Path.cwd() / "data" / "cache" / "osm"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_key}.json"
    meta_file = cache_dir / f"{cache_key}_meta.json"

    if cache_file.exists() and meta_file.exists():
        try:
            with open(meta_file) as f:
                meta = json.load(f)
            created = datetime.fromisoformat(meta["created_at"])
            if (datetime.now() - created) < timedelta(days=max_age_days):
                with open(cache_file) as f:
                    return json.load(f)
        except Exception:
            pass

    print("    Querying Overpass API...")
    for attempt in range(retries):
        try:
            resp = requests.post(
                "https://overpass-api.de/api/interpreter",
                data={"data": query},
                timeout=150,
            )
        except requests.exceptions.Timeout:
            wait = 20 * (attempt + 1)
            print(f"    Timeout, waiting {wait}s (attempt {attempt + 1}/{retries})...")
            time.sleep(wait)
            continue

        if resp.status_code in (429, 504):
            wait = 20 * (attempt + 1)
            print(f"    {resp.status_code} error, waiting {wait}s (attempt {attempt + 1}/{retries})...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        break
    else:
        raise RuntimeError(f"Overpass API failed after {retries} retries (last status: {resp.status_code})")

    data = resp.json()

    with open(cache_file, "w") as f:
        json.dump(data, f)
    with open(meta_file, "w") as f:
        json.dump({"created_at": datetime.now().isoformat()}, f)
    return data


def _fetch_railways_tile(bbox):
    """Fetch railway LineStrings from OSM for a single tile bbox."""
    south, west, north, east = bbox
    bb = f"{south},{west},{north},{east}"
    query = (
        f'[out:json][timeout:120];'
        f'(way["railway"="rail"]({bb});way["railway"="light_rail"]({bb}););'
        f'out geom;'
    )
    cache_key = hashlib.sha256(f"railways_{bb}".encode()).hexdigest()[:16]
    data = _fetch_overpass_cached(query, cache_key)

    features = []
    for el in data.get("elements", []):
        if el.get("type") != "way" or "geometry" not in el:
            continue
        coords = [[n["lon"], n["lat"]] for n in el["geometry"]]
        if len(coords) < 2:
            continue
        tags = el.get("tags", {})
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {
                "osm_id": el["id"],
                "railway": tags.get("railway", ""),
                "name": tags.get("name", ""),
            },
        })
    return {"type": "FeatureCollection", "features": features}


def fetch_railways_tiled(bbox):
    """Fetch railway data with automatic tiling to avoid Overpass timeouts."""
    south, west, north, east = bbox
    tile_size = 1.0
    lat_tiles = max(1, math.ceil((north - south) / tile_size))
    lon_tiles = max(1, math.ceil((east - west) / tile_size))
    total = lat_tiles * lon_tiles
    print(f"    Fetching railways in {lat_tiles}x{lon_tiles} = {total} tile(s)...")

    all_features = []
    for i, (lat_idx, lon_idx) in enumerate(
        (la, lo) for la in range(lat_tiles) for lo in range(lon_tiles)
    ):
        t_south = south + lat_idx * tile_size
        t_west = west + lon_idx * tile_size
        result = _fetch_railways_tile((
            t_south, t_west,
            min(t_south + tile_size, north),
            min(t_west + tile_size, east),
        ))
        all_features.extend(result.get("features", []))
        if i + 1 < total:
            time.sleep(5)

    return {"type": "FeatureCollection", "features": all_features}


# =============================================================================
# PANEL BUILDER — shared pipeline for main and city inset
# =============================================================================


def _solid_overlay(color):
    """Return a colormap function that paints active pixels a single RGB color."""
    def _cmap(grid):
        out = np.zeros((*grid.shape, 3), dtype=np.uint8)
        out[grid > 0.5] = color
        return out
    return _cmap


def _build_overlays(terrain):
    """Build overlay dicts from OVERLAY_SPEC for layers present in terrain."""
    return [
        {"colormap": _solid_overlay(color), "source_layers": [name],
         "threshold": 0.5, "priority": priority}
        for name, color, priority in OVERLAY_SPEC
        if name in terrain.data_layers
    ]


def _add_seville_transforms(terrain):
    """Append the standard Seville reproject → flip → scale pipeline."""
    terrain.add_transform(reproject_raster(
        src_crs="EPSG:4326", dst_crs=SEVILLE_UTM_CRS, num_threads=4))
    terrain.add_transform(flip_raster(axis="horizontal"))
    terrain.add_transform(scale_elevation(scale_factor=0.0001))


def _add_water_layer(terrain, dem_bbox, lakes_geojson):
    """Detect water (slope + HydroLAKES), add combined mask as layer.

    Returns the combined boolean water mask.
    """
    water = terrain.detect_water_highres(slope_threshold=0.01)
    n_lakes = len(lakes_geojson.get("features", [])) if lakes_geojson else 0

    if n_lakes > 0:
        mask, tf = rasterize_lakes_to_mask(lakes_geojson, dem_bbox, resolution=0.001)
        if np.any(mask):
            terrain.add_data_layer(
                "hydrolakes", (mask > 0).astype(np.float32),
                tf, "EPSG:4326", target_layer="dem",
            )
            hl = terrain.data_layers["hydrolakes"]
            water = water | (hl.get("transformed_data", hl.get("data")) > 0.5)

    terrain.add_data_layer("water", water.astype(np.float32), same_extent_as="dem")
    return water


def _add_infrastructure_layers(terrain, dem_bbox, roads, railways,
                               road_width=5, rail_width=2):
    """Add roads and railways as data layers."""
    if roads and roads.get("features"):
        add_roads_layer(terrain, roads, dem_bbox, road_width_pixels=road_width)

    if railways and railways.get("features"):
        grid, tf = rasterize_roads_to_layer(
            railways, dem_bbox, resolution=30.0, road_width_pixels=rail_width)
        if np.any(grid):
            terrain.add_data_layer(
                "railways", grid.astype(np.float32),
                tf, "EPSG:4326", target_layer="dem",
            )


def _build_panel(dem_data, dem_transform, target_vertices,
                 roads, railways, lakes_geojson, height_scale,
                 road_width=5, rail_width=2, label="panel"):
    """Full pipeline: Terrain → transforms → layers → color → mesh.

    Parameters control resolution-sensitive behavior:
    - target_vertices: vertex budget (higher = denser mesh)
    - road_width / rail_width: pixel widths scaled to panel resolution

    Returns (terrain, mesh_obj, water_mask).
    """
    terrain = Terrain(dem_data, dem_transform)
    terrain.configure_for_target_vertices(target_vertices, method="average")
    _add_seville_transforms(terrain)
    terrain.apply_transforms()

    transformed = terrain.data_layers["dem"]["transformed_data"]
    dem_bbox = terrain.get_bbox_wgs84()
    print(f"  [{label}] {transformed.shape} ({transformed.size:,} vertices), "
          f"bbox lat [{dem_bbox[0]:.4f}, {dem_bbox[2]:.4f}]")

    water = _add_water_layer(terrain, dem_bbox, lakes_geojson)
    _add_infrastructure_layers(terrain, dem_bbox, roads, railways,
                               road_width=road_width, rail_width=rail_width)

    overlays = _build_overlays(terrain)
    terrain.set_multi_color_mapping(
        base_colormap=lambda dem: elevation_colormap(dem, cmap_name="turbo"),
        base_source_layers=["dem"],
        overlays=overlays,
    )
    print(f"  [{label}] Turbo base + {len(overlays)} overlay(s)")

    mesh = terrain.create_mesh(
        scale_factor=100.0,
        height_scale=height_scale,
        center_model=True,
        boundary_extension=True,
        water_mask=water,
    )
    if mesh is not None:
        print(f"  [{label}] {len(mesh.data.vertices):,} verts, "
              f"{len(mesh.data.polygons):,} polys")
    return terrain, mesh, water


def _crop_dem(dem_data, dem_transform, bbox):
    """Crop DEM array to a geographic bbox. Returns (cropped_data, cropped_transform)."""
    south, west, north, east = bbox
    inv = ~dem_transform
    col_min, row_min = inv * (west, north)
    col_max, row_max = inv * (east, south)
    r0 = int(max(0, row_min))
    r1 = int(min(dem_data.shape[0], row_max))
    c0 = int(max(0, col_min))
    c1 = int(min(dem_data.shape[1], col_max))
    origin_x, origin_y = dem_transform * (c0, r0)
    cropped_transform = Affine(
        dem_transform.a, dem_transform.b, origin_x,
        dem_transform.d, dem_transform.e, origin_y,
    )
    return dem_data[r0:r1, c0:c1].copy(), cropped_transform


def _position_inset_below(main_mesh, inset_mesh, gap_fraction=0.1):
    """Scale inset to match main mesh width, position below with gap."""
    main_corners = [main_mesh.matrix_world @ Vector(c) for c in main_mesh.bound_box]
    main_min_x = min(c.x for c in main_corners)
    main_max_x = max(c.x for c in main_corners)
    main_min_y = min(c.y for c in main_corners)
    main_max_y = max(c.y for c in main_corners)
    main_width = main_max_x - main_min_x
    main_height = main_max_y - main_min_y

    inset_corners = [Vector(c) for c in inset_mesh.bound_box]
    inset_width = max(c.x for c in inset_corners) - min(c.x for c in inset_corners)

    scale = main_width / inset_width if inset_width > 0 else 1.0
    inset_mesh.scale = (scale, scale, scale)
    bpy.context.view_layer.objects.active = inset_mesh
    inset_mesh.select_set(True)
    bpy.ops.object.transform_apply(scale=True)

    inset_corners = [Vector(c) for c in inset_mesh.bound_box]
    inset_max_y = max(c.y for c in inset_corners)

    gap = main_height * gap_fraction
    inset_mesh.location.x = (main_min_x + main_max_x) / 2
    inset_mesh.location.y = main_min_y - gap - inset_max_y
    inset_mesh.location.z = 0
    print(f"  Inset scaled {scale:.1f}x, gap={gap:.2f}")


def _frame_camera_with_aspect(meshes, width, height, padding=1.1):
    """Frame orthographic camera to meshes, corrected for render aspect ratio."""
    camera = frame_camera_to_objects(meshes, padding=padding)
    corners = [obj.matrix_world @ Vector(c)
               for obj in meshes for c in obj.bound_box]
    scene_w = max(c.x for c in corners) - min(c.x for c in corners)
    scene_h = max(c.y for c in corners) - min(c.y for c in corners)
    aspect = width / height
    camera.data.ortho_scale = max(scene_w, scene_h * aspect) * padding
    print(f"  Camera: ortho_scale={camera.data.ortho_scale:.1f}, "
          f"scene={scene_w:.1f}x{scene_h:.1f}, aspect={aspect:.2f}")
    return camera


# =============================================================================
# MAIN
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Seville terrain map with roads, railways, and water bodies"
    )
    parser.add_argument("--no-render", action="store_true", help="Data pipeline only")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--height-scale", type=float, default=6.0)
    parser.add_argument("--samples", type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 70)
    print("Seville, Spain - Overhead Terrain Map")
    print(f"  Output: {WIDTH}x{HEIGHT} px (10x8\" @ 150 DPI)")
    print("=" * 70)

    # ---- 1. Ensure DEM tiles exist ------------------------------------------
    print("\n[1/5] Checking DEM tiles...")
    SEVILLE_DEM_DIR.mkdir(parents=True, exist_ok=True)
    hgt_files = list(SEVILLE_DEM_DIR.glob("*.hgt")) + list(SEVILLE_DEM_DIR.glob("*.zip"))
    if not hgt_files:
        print("  No tiles found, downloading...")
        downloaded = download_dem_by_bbox(SEVILLE_BBOX, str(SEVILLE_DEM_DIR))
        if not downloaded:
            raise RuntimeError(
                f"Could not download DEM tiles. Add Earthdata credentials to .env "
                f"or place SRTM .hgt files in {SEVILLE_DEM_DIR}"
            )
        print(f"  Downloaded {len(downloaded)} tile(s)")
    else:
        print(f"  Found {len(hgt_files)} tile(s)")

    # ---- 2. Load DEM --------------------------------------------------------
    print("\n[2/5] Loading DEM...")
    try:
        dem_data, dem_transform = load_dem_files(SEVILLE_DEM_DIR, pattern="*.hgt")
    except Exception:
        dem_data, dem_transform = load_dem_files(SEVILLE_DEM_DIR, pattern="*.zip")
    print(f"  Shape: {dem_data.shape}, range: "
          f"{np.nanmin(dem_data):.0f}\u2013{np.nanmax(dem_data):.0f} m")

    # ---- 3. Fetch shared data -----------------------------------------------
    # Roads, railways, and lakes are fetched once and reused by both panels.
    # Use the full DEM bbox (SRTM tiles are 1x1 degree, typically larger than
    # the target bbox) so all data covers both the main and city panels.
    print("\n[3/5] Fetching shared data layers...")

    # Compute DEM WGS84 extent directly from the original Affine transform.
    # The raw DEM is already in EPSG:4326, so no reprojection needed.
    h, w = dem_data.shape
    x0, y0 = dem_transform.c, dem_transform.f
    x1 = x0 + dem_transform.a * w
    y1 = y0 + dem_transform.e * h
    dem_bbox = (min(y0, y1), min(x0, x1), max(y0, y1), max(x0, x1))
    print(f"  DEM extent: lat [{dem_bbox[0]:.4f}, {dem_bbox[2]:.4f}], "
          f"lon [{dem_bbox[1]:.4f}, {dem_bbox[3]:.4f}]")

    # HydroLAKES
    water_dir = Path(args.output_dir or ".") / "water_bodies"
    water_dir.mkdir(parents=True, exist_ok=True)
    geojson_path = download_water_bodies(
        bbox=dem_bbox, output_dir=str(water_dir),
        data_source="hydrolakes", min_area_km2=0.01,
    )
    with open(geojson_path) as f:
        lakes_geojson = json.load(f)
    print(f"  HydroLAKES: {len(lakes_geojson.get('features', []))} features")

    # Roads
    roads = get_roads_tiled(
        dem_bbox, road_types=["motorway", "trunk", "primary"],
        tile_size=1.0, retry_count=3, retry_delay=15.0)
    n_roads = len(roads["features"]) if roads and roads.get("features") else 0
    print(f"  Roads: {n_roads} segments")

    # Railways
    railways = fetch_railways_tiled(dem_bbox)
    print(f"  Railways: {len(railways.get('features', []))} segments")

    # ---- 4. Build panels ----------------------------------------------------
    print("\n[4/5] Building terrain panels...")

    # Main panel (data pipeline only — mesh requires bpy)
    main_target = WIDTH * HEIGHT * 2
    if args.no_render:
        # Just validate data pipeline, skip mesh creation
        terrain = Terrain(dem_data, dem_transform)
        terrain.configure_for_target_vertices(main_target, method="average")
        _add_seville_transforms(terrain)
        terrain.apply_transforms()
        _add_water_layer(terrain, terrain.get_bbox_wgs84(), lakes_geojson)
        print("  --no-render: data pipeline complete")
        return 0

    # Clear scene before creating any Blender meshes
    try:
        clear_scene()
    except Exception:
        pass

    # Main panel: broad area, moderate vertex density
    _, main_mesh, _ = _build_panel(
        dem_data, dem_transform, main_target,
        roads, railways, lakes_geojson, args.height_scale,
        road_width=5, rail_width=2, label="main",
    )
    if main_mesh is None:
        raise RuntimeError("Main mesh creation failed")

    # City inset: smaller crop, higher vertex density per unit area.
    # target = WIDTH * HEIGHT gives ~1 vertex per render pixel on the smaller area.
    city_dem, city_tf = _crop_dem(dem_data, dem_transform, SEVILLE_CITY_BBOX)
    print(f"  City DEM crop: {city_dem.shape} ({city_dem.size:,} pixels)")

    city_target = WIDTH * HEIGHT
    _, city_mesh, _ = _build_panel(
        city_dem, city_tf, city_target,
        roads, railways, lakes_geojson, args.height_scale,
        road_width=2, rail_width=1, label="city",
    )

    # ---- 5. Scene setup + render --------------------------------------------
    print("\n[5/5] Rendering...")

    all_meshes = [main_mesh]
    if city_mesh is not None:
        _position_inset_below(main_mesh, city_mesh)
        all_meshes.append(city_mesh)
    else:
        print("  WARNING: City inset mesh creation failed, rendering main only")

    _frame_camera_with_aspect(all_meshes, WIDTH, HEIGHT)

    setup_hdri_lighting(
        sun_elevation=12.0,     # Low sun — golden hour / sunset
        sun_rotation=240.0,     # Southwest — typical Seville evening sun
        sun_intensity=0.25,
        air_density=0.1,
    )
    setup_render_settings(use_gpu=True, samples=args.samples, use_denoising=True)

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent
    output_path = output_dir / "seville_terrain.png"
    print(f"  Rendering {WIDTH}x{HEIGHT} to {output_path}...")

    render_scene_to_file(
        output_path=output_path, width=WIDTH, height=HEIGHT,
        file_format="PNG", color_mode="RGBA", compression=90,
        save_blend_file=True,
    )
    print(f"\nDone: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
