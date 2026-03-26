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
import logging
import json
import hashlib
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import requests

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
    setup_hdri_lighting,
    setup_render_settings,
    render_scene_to_file,
)
from src.terrain.roads import (
    get_roads_tiled,
    add_roads_layer,
    rasterize_roads_to_layer,
)
from src.terrain.water_bodies import download_water_bodies, rasterize_lakes_to_mask
from src.terrain.dem_downloader import download_dem_by_bbox

try:
    import bpy
except ImportError:
    bpy = None

logger = logging.getLogger(__name__)

# =============================================================================
# SEVILLE CONFIGURATION
# =============================================================================

# Seville metropolitan area bounding box (south, west, north, east)
SEVILLE_BBOX = (37.25, -6.15, 37.55, -5.80)
# Tight bbox around Seville city center for the inset panel
SEVILLE_CITY_BBOX = (37.34, -6.02, 37.42, -5.92)
SEVILLE_DEM_DIR = Path(__file__).parent.parent / "data" / "dem" / "seville"
SEVILLE_UTM_CRS = "EPSG:32630"  # UTM Zone 30N

# Output: 10x8 inches at 150 DPI
WIDTH = 10 * 150   # 1500
HEIGHT = 8 * 150   # 1200

# Water color — cool blue complementing warm turbo colormap
WATER_COLOR = (40, 100, 180)


# =============================================================================
# OSM RAILWAY FETCHING (Overpass API with caching + retry)
#
# Roads use the library's get_roads_tiled() + add_roads_layer().
# Water uses the library's download_water_bodies() (HydroLAKES).
# Railways need a custom Overpass query — reuse rasterize_roads_to_layer()
# for the LineString rasterization.
# =============================================================================


def _fetch_overpass_cached(
    query: str, cache_key: str, max_age_days: int = 30, retries: int = 3
) -> Dict[str, Any]:
    """Fetch from Overpass API with local caching and retry on 429/504."""
    import time as _time

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

    print(f"    Querying Overpass API...")
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
            _time.sleep(wait)
            continue

        if resp.status_code in (429, 504):
            wait = 20 * (attempt + 1)
            print(f"    {resp.status_code} error, waiting {wait}s (attempt {attempt + 1}/{retries})...")
            _time.sleep(wait)
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


def _fetch_railways_tile(bbox: Tuple[float, float, float, float]) -> Dict[str, Any]:
    """Fetch railway LineStrings from OSM for a single tile bbox."""
    south, west, north, east = bbox
    bb = f"{south},{west},{north},{east}"
    query = f'[out:json][timeout:120];(way["railway"="rail"]({bb});way["railway"="light_rail"]({bb}););out geom;'
    h = hashlib.sha256(f"railways_{bb}".encode()).hexdigest()[:16]
    data = _fetch_overpass_cached(query, h)

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


def fetch_railways_tiled(bbox: Tuple[float, float, float, float]) -> Dict[str, Any]:
    """Fetch railway data with automatic tiling to avoid Overpass timeouts."""
    import math
    import time as _time

    south, west, north, east = bbox
    # Railways are sparse — use 1° tiles (much fewer requests than roads)
    tile_size = 1.0
    lat_tiles = max(1, math.ceil((north - south) / tile_size))
    lon_tiles = max(1, math.ceil((east - west) / tile_size))
    total = lat_tiles * lon_tiles
    print(f"    Fetching railways in {lat_tiles}x{lon_tiles} = {total} tile(s)...")

    all_features = []
    fetched = 0
    for lat_idx in range(lat_tiles):
        for lon_idx in range(lon_tiles):
            t_south = south + lat_idx * tile_size
            t_north = min(t_south + tile_size, north)
            t_west = west + lon_idx * tile_size
            t_east = min(t_west + tile_size, east)
            result = _fetch_railways_tile((t_south, t_west, t_north, t_east))
            all_features.extend(result.get("features", []))
            fetched += 1
            # Longer delay between tiles to respect Overpass rate limits
            if fetched < total:
                _time.sleep(5)

    return {"type": "FeatureCollection", "features": all_features}


# =============================================================================
# COLORMAP FUNCTIONS (thin wrappers for set_multi_color_mapping overlays)
# =============================================================================


def water_colormap(water_grid):
    """Map water pixels to complementary blue."""
    colors = np.zeros((*water_grid.shape, 3), dtype=np.uint8)
    colors[water_grid > 0.5] = WATER_COLOR
    return colors


def seville_road_colormap(road_grid, score=None):
    """Map road pixels to black."""
    colors = np.zeros((*road_grid.shape, 3), dtype=np.uint8)
    colors[road_grid > 0.5] = (10, 10, 10)
    return colors


def railway_colormap(railway_grid):
    """Map railway pixels to dark gray."""
    colors = np.zeros((*railway_grid.shape, 3), dtype=np.uint8)
    colors[railway_grid > 0.5] = (60, 60, 60)
    return colors


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
    bbox = SEVILLE_BBOX

    print("=" * 70)
    print("Seville, Spain - Overhead Terrain Map")
    print(f"  Output: {WIDTH}x{HEIGHT} px (10x8\" @ 150 DPI)")
    print("=" * 70)

    # ---- 1. Ensure DEM tiles exist ------------------------------------------
    print("\n[1/6] Checking DEM tiles...")
    SEVILLE_DEM_DIR.mkdir(parents=True, exist_ok=True)
    hgt_files = list(SEVILLE_DEM_DIR.glob("*.hgt")) + list(SEVILLE_DEM_DIR.glob("*.zip"))
    if not hgt_files:
        print("  No tiles found, downloading...")
        downloaded = download_dem_by_bbox(bbox, str(SEVILLE_DEM_DIR))
        if not downloaded:
            raise RuntimeError(
                f"Could not download DEM tiles. Add Earthdata credentials to .env "
                f"or place SRTM .hgt files in {SEVILLE_DEM_DIR}"
            )
        print(f"  Downloaded {len(downloaded)} tile(s)")
    else:
        print(f"  Found {len(hgt_files)} tile(s)")

    # ---- 2. Load DEM --------------------------------------------------------
    print("\n[2/6] Loading DEM...")
    try:
        dem_data, transform = load_dem_files(SEVILLE_DEM_DIR, pattern="*.hgt")
    except Exception:
        dem_data, transform = load_dem_files(SEVILLE_DEM_DIR, pattern="*.zip")
    print(f"  Shape: {dem_data.shape}, range: {np.nanmin(dem_data):.0f}–{np.nanmax(dem_data):.0f} m")

    # ---- 3. Create Terrain + transforms -------------------------------------
    print("\n[3/6] Initializing Terrain...")
    terrain = Terrain(dem_data, transform)

    target_vertices = WIDTH * HEIGHT * 2
    terrain.configure_for_target_vertices(target_vertices, method="average")
    terrain.transforms.append(reproject_raster(src_crs="EPSG:4326", dst_crs=SEVILLE_UTM_CRS, num_threads=4))
    terrain.transforms.append(flip_raster(axis="horizontal"))
    terrain.transforms.append(scale_elevation(scale_factor=0.0001))
    terrain.apply_transforms()

    transformed_dem = terrain.data_layers["dem"]["transformed_data"]
    print(f"  Downsampled: {transformed_dem.shape} ({transformed_dem.size:,} vertices)")

    # Get the actual DEM extent in WGS84 — this is the full area we need data for
    # (SRTM tiles are 1°×1° so the DEM is typically larger than the target bbox)
    dem_bbox = terrain.get_bbox_wgs84()
    print(f"  DEM extent: lat [{dem_bbox[0]:.4f}, {dem_bbox[2]:.4f}], "
          f"lon [{dem_bbox[1]:.4f}, {dem_bbox[3]:.4f}]")

    # ---- 4. Add data layers -------------------------------------------------
    print("\n[4/6] Adding data layers...")

    # Water — combine slope-based detection (catches ocean) with HydroLAKES (inland lakes)
    print("  [Water - slope detection + HydroLAKES]")

    # Slope-based: detects ocean and large flat water surfaces
    slope_water = terrain.detect_water_highres(slope_threshold=0.01)
    print(f"    Slope-detected: {np.sum(slope_water):,} pixels (ocean + flat water)")

    # HydroLAKES: inland lakes and reservoirs from shapefile
    water_dir = Path(args.output_dir or ".") / "water_bodies"
    water_dir.mkdir(parents=True, exist_ok=True)
    geojson_path = download_water_bodies(
        bbox=dem_bbox,
        output_dir=str(water_dir),
        data_source="hydrolakes",
        min_area_km2=0.01,
    )
    with open(geojson_path) as f:
        lakes_geojson = json.load(f)
    n_lakes = len(lakes_geojson.get("features", []))
    if n_lakes > 0:
        print(f"    HydroLAKES: {n_lakes} lake/reservoir features")
        lake_mask, lake_transform = rasterize_lakes_to_mask(lakes_geojson, dem_bbox, resolution=0.001)
        if np.any(lake_mask):
            terrain.add_data_layer(
                "hydrolakes", (lake_mask > 0).astype(np.float32),
                lake_transform, "EPSG:4326", target_layer="dem",
            )
            # Merge HydroLAKES into slope mask
            lake_data = terrain.data_layers["hydrolakes"]
            lake_aligned = lake_data.get("transformed_data", lake_data.get("data"))
            slope_water = slope_water | (lake_aligned > 0.5)
    else:
        print("    No HydroLAKES features in DEM extent")

    # Combined water mask as a data layer (for colormap overlay)
    water_mask = slope_water
    # Store combined mask using same_extent_as to match the transformed DEM grid
    terrain.add_data_layer(
        "water", water_mask.astype(np.float32), same_extent_as="dem",
    )
    print(f"    Combined water mask: {np.sum(water_mask):,} pixels ({100*np.mean(water_mask):.1f}%)")

    # Roads — library handles tiled fetch, rasterize, and add_data_layer
    print("  [Roads - OSM]")
    roads = get_roads_tiled(dem_bbox, road_types=["motorway", "trunk", "primary"],
                           tile_size=1.0, retry_count=3, retry_delay=15.0)
    if roads and roads.get("features"):
        print(f"    {len(roads['features'])} segments")
        add_roads_layer(terrain, roads, dem_bbox, road_width_pixels=5)
    else:
        print("    No road data (edge tiles may have none — skipping)")

    # Railways — custom Overpass fetch, reuse rasterize_roads_to_layer for LineStrings
    print("  [Railways - OSM]")
    railways = fetch_railways_tiled(dem_bbox)
    if railways.get("features"):
        print(f"    {len(railways['features'])} segments")
        rail_grid, rail_transform = rasterize_roads_to_layer(
            railways, dem_bbox, resolution=30.0, road_width_pixels=2
        )
        terrain.add_data_layer(
            "railways", rail_grid.astype(np.float32),
            rail_transform, "EPSG:4326", target_layer="dem",
        )
    else:
        print("    No railway data (edge tiles may have none — skipping)")

    # ---- 5. Color mapping ---------------------------------------------------
    print("\n[5/6] Configuring colors...")

    overlays = []
    # Water always first priority (drawn on top of terrain)
    if "water" in terrain.data_layers:
        overlays.append({"colormap": water_colormap, "source_layers": ["water"],
                         "threshold": 0.5, "priority": 1})
    if "roads" in terrain.data_layers:
        overlays.append({"colormap": seville_road_colormap, "source_layers": ["roads"],
                         "threshold": 0.5, "priority": 5})
    if "railways" in terrain.data_layers:
        overlays.append({"colormap": railway_colormap, "source_layers": ["railways"],
                         "threshold": 0.5, "priority": 10})

    terrain.set_multi_color_mapping(
        base_colormap=lambda dem: elevation_colormap(dem, cmap_name="turbo"),
        base_source_layers=["dem"],
        overlays=overlays,
    )
    print(f"  Turbo base + {len(overlays)} overlay(s)")

    # ---- 6. Mesh + Render ---------------------------------------------------
    print("\n[6/7] Creating main mesh...")

    if args.no_render:
        print("  --no-render: data pipeline complete")
        return 0

    try:
        clear_scene()
    except Exception:
        pass

    mesh_obj = terrain.create_mesh(
        scale_factor=100.0,
        height_scale=args.height_scale,
        center_model=True,
        boundary_extension=True,
        water_mask=water_mask,
    )
    if mesh_obj is None:
        raise RuntimeError("Mesh creation failed")
    print(f"  Main mesh: {len(mesh_obj.data.vertices):,} verts, {len(mesh_obj.data.polygons):,} polys")

    # ---- 7. Seville city inset ----------------------------------------------
    print("\n[7/7] Creating Seville city inset...")

    # Crop the ORIGINAL full-resolution DEM to Seville city bbox (before any downsampling)
    from rasterio import Affine as _Affine
    city_south, city_west, city_north, city_east = SEVILLE_CITY_BBOX
    dem_transform_orig = terrain.data_layers["dem"]["transform"]
    inv_t = ~dem_transform_orig
    col_min, row_min = inv_t * (city_west, city_north)
    col_max, row_max = inv_t * (city_east, city_south)
    r0, r1 = int(max(0, row_min)), int(min(dem_data.shape[0], row_max))
    c0, c1 = int(max(0, col_min)), int(min(dem_data.shape[1], col_max))
    city_dem = dem_data[r0:r1, c0:c1].copy()
    city_origin_x, city_origin_y = dem_transform_orig * (c0, r0)
    city_transform = _Affine(
        dem_transform_orig.a, dem_transform_orig.b, city_origin_x,
        dem_transform_orig.d, dem_transform_orig.e, city_origin_y,
    )
    print(f"  City DEM crop: {city_dem.shape} ({city_dem.size:,} pixels) "
          f"from rows [{r0}:{r1}], cols [{c0}:{c1}]")

    # High resolution for city — at least 1 vertex per render pixel.
    # The crop is small, so request enough vertices that the library keeps full res
    # or only lightly downsamples.
    city_terrain = Terrain(city_dem, city_transform)
    city_min_vertices = WIDTH * HEIGHT  # 1 vertex per pixel in the output
    city_terrain.configure_for_target_vertices(city_min_vertices, method="average")
    city_terrain.transforms.append(reproject_raster(src_crs="EPSG:4326", dst_crs=SEVILLE_UTM_CRS, num_threads=4))
    city_terrain.transforms.append(flip_raster(axis="horizontal"))
    city_terrain.transforms.append(scale_elevation(scale_factor=0.0001))
    city_terrain.apply_transforms()

    city_transformed = city_terrain.data_layers["dem"]["transformed_data"]
    print(f"  City mesh: {city_transformed.shape} ({city_transformed.size:,} vertices, "
          f"target >={city_min_vertices:,})")

    city_dem_bbox = city_terrain.get_bbox_wgs84()

    # Water — slope + HydroLAKES
    city_water = city_terrain.detect_water_highres(slope_threshold=0.01)
    if n_lakes > 0:
        city_lake_mask, city_lake_tf = rasterize_lakes_to_mask(lakes_geojson, city_dem_bbox, resolution=0.001)
        if np.any(city_lake_mask):
            city_terrain.add_data_layer(
                "hydrolakes", (city_lake_mask > 0).astype(np.float32),
                city_lake_tf, "EPSG:4326", target_layer="dem",
            )
            cl = city_terrain.data_layers["hydrolakes"]
            city_water = city_water | (cl.get("transformed_data", cl.get("data")) > 0.5)
    city_terrain.add_data_layer("water", city_water.astype(np.float32), same_extent_as="dem")

    # Roads — thinner lines proportional to the higher-res inset
    if roads and roads.get("features"):
        add_roads_layer(city_terrain, roads, city_dem_bbox, road_width_pixels=2)

    # Railways
    if railways and railways.get("features"):
        city_rail_grid, city_rail_tf = rasterize_roads_to_layer(
            railways, city_dem_bbox, resolution=30.0, road_width_pixels=1
        )
        if np.any(city_rail_grid):
            city_terrain.add_data_layer(
                "railways", city_rail_grid.astype(np.float32),
                city_rail_tf, "EPSG:4326", target_layer="dem",
            )

    # Color mapping
    city_overlays = []
    if "water" in city_terrain.data_layers:
        city_overlays.append({"colormap": water_colormap, "source_layers": ["water"],
                              "threshold": 0.5, "priority": 1})
    if "roads" in city_terrain.data_layers:
        city_overlays.append({"colormap": seville_road_colormap, "source_layers": ["roads"],
                              "threshold": 0.5, "priority": 5})
    if "railways" in city_terrain.data_layers:
        city_overlays.append({"colormap": railway_colormap, "source_layers": ["railways"],
                              "threshold": 0.5, "priority": 10})

    if city_overlays:
        city_terrain.set_multi_color_mapping(
            base_colormap=lambda dem: elevation_colormap(dem, cmap_name="turbo"),
            base_source_layers=["dem"],
            overlays=city_overlays,
        )
    else:
        city_terrain.set_color_mapping(
            lambda dem: elevation_colormap(dem, cmap_name="turbo"),
            source_layers=["dem"],
        )

    city_mesh = city_terrain.create_mesh(
        scale_factor=100.0,
        height_scale=args.height_scale,
        center_model=True,
        boundary_extension=True,
        water_mask=city_water,
    )
    if city_mesh is None:
        print("  WARNING: City inset mesh creation failed, rendering main only")
    else:
        print(f"  City mesh: {len(city_mesh.data.vertices):,} verts, {len(city_mesh.data.polygons):,} polys")

        # Scale city mesh width to match main mesh width, position below
        import bpy as _bpy
        from mathutils import Vector

        # Get main mesh world-space bounds
        main_corners = [mesh_obj.matrix_world @ Vector(c) for c in mesh_obj.bound_box]
        main_min_x = min(c.x for c in main_corners)
        main_max_x = max(c.x for c in main_corners)
        main_min_y = min(c.y for c in main_corners)
        main_max_y = max(c.y for c in main_corners)
        main_width = main_max_x - main_min_x
        main_height = main_max_y - main_min_y
        main_cx = (main_min_x + main_max_x) / 2

        # Get city mesh local bounds (still at origin, not yet moved)
        city_corners = [Vector(c) for c in city_mesh.bound_box]
        city_width = max(c.x for c in city_corners) - min(c.x for c in city_corners)

        # Scale uniformly so city width matches main width
        scale = main_width / city_width if city_width > 0 else 1.0
        city_mesh.scale = (scale, scale, scale)
        _bpy.context.view_layer.objects.active = city_mesh
        city_mesh.select_set(True)
        _bpy.ops.object.transform_apply(scale=True)

        # Recalculate city bounds after scaling (still local, centered at origin)
        city_corners = [Vector(c) for c in city_mesh.bound_box]
        city_min_y_local = min(c.y for c in city_corners)
        city_max_y_local = max(c.y for c in city_corners)
        city_h = city_max_y_local - city_min_y_local

        # Place so city's top edge is below main's bottom edge with gap
        gap = main_height * 0.1
        # We want: city_max_y_world = main_min_y - gap
        # city_max_y_world = city_mesh.location.y + city_max_y_local
        # So: city_mesh.location.y = main_min_y - gap - city_max_y_local
        city_mesh.location.x = main_cx
        city_mesh.location.y = main_min_y - gap - city_max_y_local
        city_mesh.location.z = 0
        print(f"  Scaled {scale:.1f}x, positioned below main (gap={gap:.2f}, city_h={city_h:.2f})")

    # ---- Render -------------------------------------------------------------
    print("\n  Rendering...")

    # Frame camera to show both meshes, accounting for render aspect ratio
    from src.terrain.scene_setup import frame_camera_to_objects
    from mathutils import Vector as _Vector
    all_meshes = [mesh_obj]
    if city_mesh is not None:
        all_meshes.append(city_mesh)
    camera = frame_camera_to_objects(all_meshes, padding=1.1)

    # frame_camera_to_objects uses max(extent_x, extent_y) which doesn't
    # account for the render aspect ratio. With two meshes stacked vertically,
    # the scene is taller than wide — adjust ortho_scale for our 10:8 aspect.
    aspect = WIDTH / HEIGHT  # 1.25
    all_corners = []
    for obj in all_meshes:
        all_corners.extend(obj.matrix_world @ _Vector(c) for c in obj.bound_box)
    scene_w = max(c.x for c in all_corners) - min(c.x for c in all_corners)
    scene_h = max(c.y for c in all_corners) - min(c.y for c in all_corners)
    # ortho_scale = horizontal view width. Vertical view = ortho_scale / aspect.
    # Need: ortho_scale >= scene_w AND ortho_scale / aspect >= scene_h
    padding = 1.1
    camera.data.ortho_scale = max(scene_w, scene_h * aspect) * padding
    print(f"  Camera: ortho_scale={camera.data.ortho_scale:.1f}, "
          f"scene={scene_w:.1f}x{scene_h:.1f}, aspect={aspect:.2f}")

    # Nishita sky model with sunset sun angle for warm, dramatic lighting
    setup_hdri_lighting(
        sun_elevation=12.0,     # Low sun — golden hour / sunset
        sun_rotation=240.0,     # Southwest — typical Seville evening sun
        sun_intensity=.25,
        air_density=.1,        # Slightly hazy for warm sunset glow
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
