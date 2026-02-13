#!/usr/bin/env python3
"""
San Diego Flow Accumulation Demo: Hydrological Analysis + 3D Visualization.

Builds on san_diego_demo.py to add flow accumulation analysis:
1. Download SRTM DEM from NASA (same as san_diego_demo.py)
2. Download real WorldClim precipitation data (global dataset, covers USA + Mexico)
3. Compute flow direction, drainage area, and upstream rainfall
4. Visualize results in 3D with Blender rendering
5. Color terrain by upstream rainfall to show water accumulation

Shows integration of flow accumulation with terrain visualization using real climate data.

NOTE: Uses WorldClim instead of PRISM because San Diego DEM extends into Mexico.
PRISM only covers continental USA, but WorldClim provides global coverage.
Default is WorldClim 30-second (~1km) resolution for high-quality precipitation data.

Requirements:
- Blender Python API (bpy)
- NASA Earthdata credentials in .env file
- Internet connection for WorldClim data download (~1GB for 30s, ~10MB for 2.5m)

Usage:
    python examples/san_diego_flow_demo.py
    python examples/san_diego_flow_demo.py --skip-download  # if you already have DEM
    python examples/san_diego_flow_demo.py --precip path/to/precip.tif  # use custom precip data
    python examples/san_diego_flow_demo.py --max-breach-depth 50 --max-breach-length 200  # tune breaching
"""

import sys
import os
from pathlib import Path
import numpy as np
import rasterio
from rasterio import Affine

import bpy

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file for NASA credentials
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from src.terrain.dem_downloader import download_dem_by_bbox
from src.terrain.data_loading import load_dem_files
from src.terrain.core import (
    Terrain,
    clear_scene,
    position_camera_relative,
    setup_render_settings,
    render_scene_to_file,
    reproject_raster,
    flip_raster,
    scale_elevation,
)
from src.terrain.scene_setup import create_background_plane, setup_hdri_lighting
from src.terrain.materials import apply_colormap_material
from src.terrain.flow_accumulation import (
    flow_accumulation,
    compute_flow_direction,
    compute_drainage_area,
    compute_upstream_rainfall,
    compute_discharge_potential,
    condition_dem,
    detect_ocean_mask,
)
from src.terrain.water_bodies import (
    download_water_bodies,
    rasterize_lakes_to_mask,
    identify_outlet_cells,
)
from src.terrain.precipitation_downloader import download_precipitation
from src.terrain.color_mapping import elevation_colormap
from src.terrain.visualization.flow_diagnostics import (
    create_flow_diagnostics,
    plot_stream_overlay,
)


def get_cmap_name(cmap_spec: str) -> str:
    """Return colormap name, handling reversed colormaps.

    Matplotlib natively supports '_r' suffix for reversed colormaps:
    - "viridis" -> normal viridis
    - "viridis_r" -> reversed viridis

    This function just passes through the name - matplotlib handles the rest.
    """
    return cmap_spec


def main():
    import argparse

    parser = argparse.ArgumentParser(description="San Diego Flow Accumulation Demo")
    parser.add_argument("--skip-download", action="store_true", help="Skip DEM download")
    parser.add_argument("--no-render", action="store_true", help="Skip 3D rendering")
    parser.add_argument(
        "--precip", type=Path, default=None, help="Path to precipitation GeoTIFF (optional)"
    )
    parser.add_argument(
        "--precip-dataset",
        type=str,
        choices=["worldclim_30s", "worldclim", "prism"],
        default="worldclim_30s",
        help="Precipitation dataset: worldclim_30s (~1km, default), worldclim (~4.5km), prism (USA only)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("examples/output"), help="Output directory"
    )
    parser.add_argument(
        "--dem-dir", type=Path, default=Path("data/san_diego_dem"), help="DEM directory"
    )
    parser.add_argument(
        "--color-by",
        type=str,
        choices=["elevation", "drainage", "rainfall", "discharge"],
        default="rainfall",
        help="Color base terrain by: elevation, drainage area, upstream rainfall, or discharge potential",
    )
    parser.add_argument(
        "--stream-overlay",
        action="store_true",
        help="Enable stream network overlay on top of base map",
    )
    parser.add_argument(
        "--stream-color-by",
        type=str,
        choices=["drainage", "rainfall", "discharge"],
        default="discharge",
        help="Color stream overlay by: drainage area, upstream rainfall, or discharge potential (default: discharge)",
    )
    parser.add_argument(
        "--stream-percentile",
        type=float,
        default=95,
        help="Percentile threshold for stream extraction (default: 95 = top 5%%)",
    )
    parser.add_argument(
        "--base-cmap",
        type=str,
        default=None,
        help="Colormap for base terrain. Use '_r' suffix to reverse (e.g., 'viridis_r'). "
             "Options: viridis, plasma, inferno, magma, cividis, terrain, modern_terrain. "
             "Default depends on --color-by.",
    )
    parser.add_argument(
        "--stream-cmap",
        type=str,
        default="plasma",
        help="Colormap for stream overlay. Use '_r' suffix to reverse (e.g., 'plasma_r'). "
             "Options: viridis, plasma, inferno, magma, cividis. Default: plasma.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="south-southwest",
        help="Camera direction: north, south, east, west, northeast, southeast, southwest, "
             "northwest, above, above-tilted. Default: south-southwest.",
    )
    parser.add_argument(
        "--variable-width",
        action="store_true",
        help="Scale stream line width by metric value (thicker = higher value)",
    )
    parser.add_argument(
        "--max-stream-width",
        type=int,
        default=3,
        help="Maximum stream width in pixels when --variable-width is enabled (default: 3)",
    )
    parser.add_argument(
        "--no-water-bodies",
        action="store_true",
        help="Disable water body integration (lakes/reservoirs)",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        choices=["synthetic", "nhd", "hydrolakes"],
        default="hydrolakes",
        help="Water body data source: synthetic, nhd (USA), or hydrolakes (global)",
    )
    parser.add_argument(
        "--min-lake-area",
        type=float,
        default=0.1,
        help="Minimum lake area in km² to include (default: 0.1)",
    )
    # Resolution/quality args for prototyping
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast prototyping mode: 500K vertices, 128 samples, 480p render",
    )
    parser.add_argument(
        "--target-vertices",
        type=int,
        default=None,
        help="Target vertex count for mesh. Overrides --vertices-per-pixel calculation.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Render samples (default: 2048, --fast: 128)",
    )
    parser.add_argument(
        "--render-scale",
        type=float,
        default=None,
        help="Render resolution scale (default: 1.0, --fast: 0.5). Overridden by --width/--height.",
    )
    parser.add_argument(
        "--height-scale",
        type=float,
        default=None,
        help="Terrain height exaggeration (default: 8.0, --fast: 4.0)",
    )
    # New resolution arguments
    parser.add_argument(
        "--width",
        type=float,
        default=None,
        help="Render width in inches. Pixels = width × DPI. Overrides --render-scale.",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=None,
        help="Render height in inches. Pixels = height × DPI. Overrides --render-scale.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=72,
        help="DPI for output (default: 72). Render pixels = inches × DPI.",
    )
    parser.add_argument(
        "--vertices-per-pixel",
        type=float,
        default=None,
        help="Vertices per render pixel. Auto-calculates --target-vertices from render dimensions. "
             "Example: 10.0 means 10 vertices per pixel (e.g., 10\"×8\" @ 72 DPI = 4.1M vertices). "
             "Higher = more mesh detail. Overridden by explicit --target-vertices.",
    )
    # Flow algorithm parameters
    parser.add_argument(
        "--max-breach-depth",
        type=float,
        default=25.0,
        help="Max vertical breach per cell in meters (default: 25.0)",
    )
    parser.add_argument(
        "--max-breach-length",
        type=int,
        default=150,
        help="Max breach path length in cells (default: 150)",
    )
    parser.add_argument(
        "--parallel-method",
        type=str,
        choices=["checkerboard", "iterative"],
        default="checkerboard",
        help="Parallel breaching method: 'checkerboard' (fast, outlets-only) or 'iterative' (slower, enables chaining)",
    )
    parser.add_argument(
        "--breach",
        action="store_true",
        help="Enable breaching step (disabled by default). Breaching is expensive and often unnecessary.",
    )
    # Basin detection parameters
    parser.add_argument(
        "--min-basin-depth",
        type=float,
        default=5.0,
        help="Min depth (meters) to be considered an endorheic basin (default: 5.0). Lower = preserve shallower basins.",
    )
    parser.add_argument(
        "--min-basin-size",
        type=int,
        default=None,
        help="Min size (cells) to be considered an endorheic basin. Default: adaptive (1/1000 of total cells).",
    )
    args = parser.parse_args()

    # Base render dimensions (10x8 inches)
    base_width_in, base_height_in = 10.0, 8.0

    # Determine render dimensions in inches, then convert to pixels
    # Priority: explicit --width/--height > --render-scale > --fast defaults
    if args.width is not None or args.height is not None:
        # Use explicit dimensions in inches, defaulting to base aspect ratio if only one specified
        if args.width is not None and args.height is not None:
            width_in, height_in = args.width, args.height
        elif args.width is not None:
            width_in = args.width
            height_in = width_in * base_height_in / base_width_in
        else:  # args.height is not None
            height_in = args.height
            width_in = height_in * base_width_in / base_height_in
        args.render_scale = 1.0  # Not used when explicit dimensions provided
    else:
        # Use --render-scale (or defaults) applied to base dimensions
        if args.render_scale is None:
            args.render_scale = 0.5 if args.fast else 1.0
        width_in = base_width_in * args.render_scale
        height_in = base_height_in * args.render_scale

    # Convert inches to pixels using DPI
    render_width = int(width_in * args.dpi)
    render_height = int(height_in * args.dpi)

    # Store computed render dimensions
    args.render_width = render_width
    args.render_height = render_height
    args.width_in = width_in
    args.height_in = height_in

    # Calculate target vertices from vertices-per-pixel if specified
    if args.target_vertices is None:
        if args.vertices_per_pixel is not None:
            total_pixels = render_width * render_height
            args.target_vertices = int(total_pixels * args.vertices_per_pixel)
            print(f"Calculated target vertices: {args.target_vertices:,} "
                  f"({width_in:.1f}\"×{height_in:.1f}\" @ {args.dpi} DPI = {render_width}×{render_height}px × {args.vertices_per_pixel} vpp)")
        else:
            # Use defaults
            args.target_vertices = 1_500_000 if args.fast else 10_000_000

    # Apply remaining --fast defaults
    if args.fast:
        if args.samples is None:
            args.samples = 128
        if args.height_scale is None:
            args.height_scale = 4.0
    else:
        if args.samples is None:
            args.samples = 2048
        if args.height_scale is None:
            args.height_scale = 8.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.dem_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download DEM (if needed)
    if not args.skip_download:
        print("Downloading San Diego DEM tiles from NASA...")
        bbox = (32.5, -117.6, 33.5, -116.0)  # San Diego County
        download_dem_by_bbox(
            bbox=bbox,
            output_dir=str(args.dem_dir),
            username=os.environ.get("EARTHDATA_USERNAME"),
            password=os.environ.get("EARTHDATA_PASSWORD"),
        )
        print("✓ Download complete")

    # Step 2: Load DEM
    print(f"\nLoading DEM from {args.dem_dir}...", flush=True)
    dem, transform = load_dem_files(args.dem_dir)
    print(f"✓ Loaded DEM: {dem.shape} pixels", flush=True)

    # Extract actual DEM bounds from transform
    # IMPORTANT: DEM download fetches full SRTM tiles (1°×1°), so actual extent
    # is larger than requested bbox. We need precipitation to cover the full DEM.
    height, width = dem.shape
    left = transform.c
    top = transform.f
    right = transform.c + transform.a * width
    bottom = transform.f + transform.e * height
    dem_bbox = (bottom, left, top, right)  # (min_lat, min_lon, max_lat, max_lon)
    print(f"  DEM bounds: lat [{dem_bbox[0]:.4f}, {dem_bbox[2]:.4f}], "
          f"lon [{dem_bbox[1]:.4f}, {dem_bbox[3]:.4f}]", flush=True)

    # Step 3: Prepare precipitation data
    print("DEBUG: About to check precipitation...", flush=True)
    print(f"DEBUG: precip={args.precip}, exists={args.precip.exists() if args.precip else 'N/A'}", flush=True)

    precip_path = None

    # First, check if user provided explicit path
    if args.precip and args.precip.exists():
        print(f"\nUsing precipitation data from {args.precip}", flush=True)
        precip_path = args.precip
    else:
        # Check for existing WorldClim file in output directory
        existing_precip = list(args.output_dir.glob("wc2.1_*.tif"))
        if existing_precip:
            print(f"\nFound existing precipitation data: {existing_precip[0]}", flush=True)
            precip_path = existing_precip[0]
        else:
            # Download WorldClim precipitation data (global dataset covering USA + Mexico)
            # Use actual DEM bounds to ensure full coverage
            # NOTE: Using WorldClim instead of PRISM because San Diego DEM extends into Mexico.
            # PRISM only covers continental USA, missing ~26.5% of this DEM (south of border).
            print("\nDownloading real WorldClim precipitation data...")
            print(f"  Using DEM bounds: {dem_bbox}")
            print(f"  WorldClim covers USA + Mexico (global dataset)")
            precip_path = download_precipitation(
                bbox=dem_bbox,
                output_dir=str(args.output_dir),
                dataset="worldclim",  # Global coverage (not just USA)
                use_real_data=True,
            )

    # Step 4: Save merged DEM for flow accumulation
    print("\nPreparing DEM for flow accumulation...", flush=True)
    print("DEBUG: Step 4 starting", flush=True)
    flow_data_dir = args.output_dir / "flow_data"
    flow_data_dir.mkdir(parents=True, exist_ok=True)
    merged_dem_path = flow_data_dir / "merged_dem.tif"

    # Save merged DEM as GeoTIFF
    print("DEBUG: Importing rasterio...", flush=True)
    import rasterio
    from rasterio import Affine

    # Check if file already exists - skip if so
    if merged_dem_path.exists():
        print(f"DEBUG: Merged DEM already exists, skipping write", flush=True)
    else:
        print(f"DEBUG: Writing DEM {dem.shape} to {merged_dem_path}...", flush=True)
        height, width = dem.shape
        with rasterio.open(
            merged_dem_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=dem.dtype,
            crs="EPSG:4326",
            transform=transform,
            compress="lzw",
        ) as dst:
            dst.write(dem, 1)

    print(f"✓ Saved merged DEM: {merged_dem_path}", flush=True)

    # Step 5: Load water bodies BEFORE flow computation
    # Water bodies affect DEM conditioning and flow routing
    lake_mask = None
    lake_outlets = None

    if not args.no_water_bodies:
        print("\nLoading water bodies (HydroLAKES)...")
        try:
            import json

            water_bodies_dir = args.output_dir / "water_bodies"
            water_bodies_dir.mkdir(parents=True, exist_ok=True)

            geojson_path = download_water_bodies(
                bbox=dem_bbox,
                output_dir=str(water_bodies_dir),
                data_source=args.data_source,
                min_area_km2=args.min_lake_area,
            )

            with open(geojson_path, "r") as f:
                lakes_geojson = json.load(f)

            num_features = len(lakes_geojson.get("features", []))
            if num_features > 0:
                print(f"  Loaded {num_features} water bodies from {args.data_source}")

                # Rasterize lakes to mask matching DEM shape
                resolution = abs(transform.a)  # pixel width in degrees
                lake_mask_raw, lake_transform = rasterize_lakes_to_mask(lakes_geojson, dem_bbox, resolution)

                # Resample to match DEM shape if needed
                if lake_mask_raw.shape != dem.shape:
                    from scipy.ndimage import zoom
                    scale_y = dem.shape[0] / lake_mask_raw.shape[0]
                    scale_x = dem.shape[1] / lake_mask_raw.shape[1]
                    lake_mask = zoom(lake_mask_raw, (scale_y, scale_x), order=0)
                else:
                    lake_mask = lake_mask_raw

                print(f"  Lake mask shape: {lake_mask.shape}, {np.sum(lake_mask > 0):,} lake cells")

                # Identify outlet cells from HydroLAKES pour points
                outlets_dict = {}
                for idx, feature in enumerate(lakes_geojson["features"], start=1):
                    props = feature.get("properties", {})
                    # HydroLAKES uses Pour_long/Pour_lat
                    if "Pour_long" in props and "Pour_lat" in props:
                        outlets_dict[idx] = (props["Pour_long"], props["Pour_lat"])

                if outlets_dict:
                    lake_outlets_raw = identify_outlet_cells(lake_mask_raw, outlets_dict, lake_transform)
                    if lake_outlets_raw.shape != dem.shape:
                        from scipy.ndimage import zoom
                        scale_y = dem.shape[0] / lake_outlets_raw.shape[0]
                        scale_x = dem.shape[1] / lake_outlets_raw.shape[1]
                        lake_outlets = zoom(lake_outlets_raw.astype(np.uint8), (scale_y, scale_x), order=0).astype(bool)
                    else:
                        lake_outlets = lake_outlets_raw
                    print(f"  Lake outlets: {np.sum(lake_outlets):,} cells")
            else:
                print("  No water bodies found in bounding box")
        except Exception as e:
            print(f"  Warning: Could not load water bodies: {e}")
            import traceback
            traceback.print_exc()
            lake_mask = None
            lake_outlets = None

    # Step 6: Detect ocean mask for diagnostics
    # (Basin detection happens inside flow_accumulation() with CLI args)
    ocean_mask = detect_ocean_mask(dem, threshold=0.0, border_only=True)

    # Step 7: Compute flow accumulation with tuned parameters
    print("\nComputing flow accumulation...", flush=True)
    flow_output_dir = args.output_dir / "flow_outputs"

    # Use adaptive resolution based on target vertices
    print(f"  Target vertices: {args.target_vertices:,} ({'--fast mode' if args.fast else 'full quality'})")

    # Tuned parameters from validate_flow_with_water_bodies.py:
    # --backend spec --edge-mode all --coastal-elev-threshold -20
    # Basin preservation via detect_basins parameter (validated approach)
    flow_result = flow_accumulation(
        dem_path=str(merged_dem_path),
        precipitation_path=str(precip_path),
        output_dir=str(flow_output_dir),
        # Tuned algorithm parameters
        backend="spec",
        edge_mode="all",
        coastal_elev_threshold=-20.0,  # Allow outlets below sea level (Salton Sea)
        # Breaching constraints (configurable via CLI arguments)
        # Breaching disabled by default (expensive), use --breach to enable
        max_breach_depth=args.max_breach_depth if args.breach else 0.0,
        max_breach_length=args.max_breach_length if args.breach else 0,
        parallel_method=args.parallel_method,
        # Basin preservation (configurable via --min-basin-depth and --min-basin-size)
        detect_basins=True,       # Automatically detect and preserve endorheic basins
        min_basin_size=args.min_basin_size,  # None = adaptive (1/1000 of total cells)
        min_basin_depth=args.min_basin_depth,  # Require significant depth to avoid small depressions
        # Precipitation upscaling (ESRGAN before ocean masking)
        upscale_precip=True,      # Upscale precipitation to DEM resolution
        upscale_factor=4,         # 4x upscaling
        upscale_method="auto",    # Try ESRGAN, fall back to bilateral
        # Water body integration
        lake_mask=lake_mask,
        lake_outlets=lake_outlets,
        # Resolution
        target_vertices=args.target_vertices,
    )

    # DEBUG: Check breached_dem status
    print("\nDEBUG: Checking breached_dem...")
    print(f"  breached_dem is None: {flow_result.get('breached_dem') is None}")
    if flow_result.get('breached_dem') is not None:
        bd = flow_result.get('breached_dem')
        cd = flow_result['conditioned_dem']
        print(f"  breached_dem shape: {bd.shape}, dtype: {bd.dtype}")
        print(f"  conditioned_dem shape: {cd.shape}, dtype: {cd.dtype}")
        print(f"  min(breached): {bd.min():.2f}, max(breached): {bd.max():.2f}")
        print(f"  min(conditioned): {cd.min():.2f}, max(conditioned): {cd.max():.2f}")
        fill_depth = cd - bd
        print(f"  min(conditioned - breached): {fill_depth.min():.2f}, max: {fill_depth.max():.2f}")
        negative_count = np.sum(fill_depth < 0)
        if negative_count > 0:
            print(f"  WARNING: {negative_count:,} cells with negative fill depth!")
            print(f"  Negative value range: {fill_depth[fill_depth < 0].min():.2f} to {fill_depth[fill_depth < 0].max():.2f}")
    else:
        print("  WARNING: breached_dem is None - breach depth plot will not be created!")

    # Print summary
    metadata = flow_result["metadata"]
    print(f"\n{'=' * 60}")
    print("FLOW ACCUMULATION RESULTS")
    print(f"{'=' * 60}")
    if metadata["downsampling_applied"]:
        print(f"  Original DEM: {metadata['original_shape'][0]}×{metadata['original_shape'][1]} "
              f"({metadata['original_shape'][0] * metadata['original_shape'][1]:,} cells)")
        print(f"  Flow computed at: {metadata['downsampled_shape'][0]}×{metadata['downsampled_shape'][1]} "
              f"({metadata['downsampled_shape'][0] * metadata['downsampled_shape'][1]:,} cells)")
        print(f"  Downsample factor: {metadata['downsample_factor']:.2f}x")
    print(f"  Total area: {metadata['total_area_km2']:.1f} km²")
    print(f"  Max drainage area: {metadata['max_drainage_area_km2']:.2f} km²")
    print(f"  Max upstream water: {metadata['max_upstream_rainfall_m3']:.0f} m³/year")
    print(f"{'=' * 60}\n")

    # Create aligned data for both diagnostics and 3D rendering using Terrain object
    print("\nPreparing aligned data layers...")

    # Use flow resolution DEM as base for alignment (ensures diagnostics match flow computation resolution)
    flow_dem = flow_result["conditioned_dem"] if not metadata["downsampling_applied"] else flow_result["conditioned_dem"]
    flow_transform = Affine(*flow_result["metadata"]["transform"])
    flow_crs = flow_result["metadata"]["crs"]

    # Create Terrain object for data alignment (at flow resolution)
    terrain_align = Terrain(flow_dem, flow_transform, dem_crs=flow_crs)

    # Add original-resolution data layers - Terrain will handle alignment via target_layer
    # This replaces manual scipy.zoom resampling with the library's alignment mechanism

    # Load precipitation data with proper masking and transform
    with rasterio.open(precip_path) as src:
        precip_data = src.read(1)
        precip_transform = src.transform
        precip_crs = src.crs

        # Mask invalid values (WorldClim can have extreme negative values like -3.4e38)
        # Valid precipitation range: 0 to 20,000 mm/year (reasonable global max)
        precip_data = np.where((precip_data >= 0) & (precip_data < 20000), precip_data, 0)
        print(f"  Precipitation range: [{np.min(precip_data):.1f}, {np.max(precip_data):.1f}] mm/year")

    # Add all data layers with original transforms - library handles reprojection/resampling
    terrain_align.add_data_layer("dem_original", dem, transform, crs="EPSG:4326", target_layer="dem")
    terrain_align.add_data_layer("ocean_mask", ocean_mask.astype(np.float32), transform, crs="EPSG:4326", target_layer="dem")
    terrain_align.add_data_layer("precip", precip_data, precip_transform, crs=str(precip_crs), target_layer="dem")

    # Note: basin_mask is not available here (detected internally by flow_accumulation)
    # Diagnostics will use basin_mask=None

    if lake_mask is not None:
        terrain_align.add_data_layer("lake_mask", lake_mask.astype(np.float32), transform, crs="EPSG:4326", target_layer="dem")

    if lake_outlets is not None:
        terrain_align.add_data_layer("lake_outlets", lake_outlets.astype(np.float32), transform, crs="EPSG:4326", target_layer="dem")

    # Apply transforms to align all data
    terrain_align.apply_transforms()

    # Extract aligned data for diagnostics (all at flow resolution)
    print("\nCreating diagnostic visualizations...")
    dem_aligned = terrain_align.data_layers["dem_original"]["data"]
    ocean_mask_aligned = terrain_align.data_layers["ocean_mask"]["data"].astype(bool)
    precip_aligned = terrain_align.data_layers["precip"]["data"]
    basin_mask_aligned = terrain_align.data_layers["basin_mask"]["data"].astype(bool) if "basin_mask" in terrain_align.data_layers else None
    lake_mask_aligned = terrain_align.data_layers["lake_mask"]["data"].astype(np.uint16) if "lake_mask" in terrain_align.data_layers else None
    lake_outlets_aligned = terrain_align.data_layers["lake_outlets"]["data"].astype(bool) if "lake_outlets" in terrain_align.data_layers else None

    # Generate all flow diagnostic plots with aligned data
    # Note: num_basins not available (basin detection happens inside flow_accumulation)
    num_basins = 0

    # Get breached DEM if available (spec backend only)
    # breached_dem is already at flow resolution, no alignment needed
    breached_dem_for_diag = flow_result.get("breached_dem")

    create_flow_diagnostics(
        dem=dem_aligned,
        dem_conditioned=flow_result["conditioned_dem"],
        ocean_mask=ocean_mask_aligned,
        flow_dir=flow_result["flow_direction"],
        drainage_area=flow_result["drainage_area"],
        upstream_rainfall=flow_result["upstream_rainfall"],
        precip=precip_aligned,
        output_dir=args.output_dir,
        lake_mask=lake_mask_aligned,
        lake_outlets=lake_outlets_aligned,
        basin_mask=basin_mask_aligned,
        breached_dem=breached_dem_for_diag,
        num_basins=num_basins,
        is_real_precip=True,  # Using real WorldClim data
    )

    # Step 5b: Generate stream overlay visualizations
    # These show streams colored by metric values, overlaid on base maps
    print("\nGenerating stream overlay visualizations...")

    # Compute discharge potential for diagnostics (same as 3D rendering)
    diag_discharge = compute_discharge_potential(
        flow_result["drainage_area"],
        flow_result["upstream_rainfall"],
    )

    # Overlay 1: Discharge-colored streams on discharge base (same metric)
    width_suffix = "_varwidth" if args.variable_width else ""
    plot_stream_overlay(
        base_data=diag_discharge,
        stream_threshold_data=flow_result["drainage_area"],
        stream_color_data=diag_discharge,
        output_path=args.output_dir / f"12_stream_overlay_discharge_on_discharge{width_suffix}.png",
        base_cmap="viridis",
        stream_cmap="plasma",
        percentile=args.stream_percentile,
        stream_alpha=1.0,  # Opaque streams
        base_label="Discharge Potential",
        stream_label="Discharge Potential",
        title="Discharge-Colored Streams on Discharge Map",
        lake_mask=lake_mask_aligned,
        variable_width=args.variable_width,
        max_width=args.max_stream_width,
    )

    # Overlay 2: Same as above but with semi-transparent streams
    plot_stream_overlay(
        base_data=diag_discharge,
        stream_threshold_data=flow_result["drainage_area"],
        stream_color_data=diag_discharge,
        output_path=args.output_dir / f"12b_stream_overlay_discharge_semitransparent{width_suffix}.png",
        base_cmap="viridis",
        stream_cmap="plasma",
        percentile=args.stream_percentile,
        stream_alpha=0.7,  # Semi-transparent for blending
        base_label="Discharge Potential",
        stream_label="Discharge Potential",
        title="Discharge Streams (Semi-Transparent) on Discharge Map",
        lake_mask=lake_mask_aligned,
        variable_width=args.variable_width,
        max_width=args.max_stream_width,
    )

    # Overlay 3: Discharge-colored streams on elevation (topo + flow)
    plot_stream_overlay(
        base_data=dem_aligned,
        stream_threshold_data=flow_result["drainage_area"],
        stream_color_data=diag_discharge,
        output_path=args.output_dir / f"13_stream_overlay_discharge_on_elevation{width_suffix}.png",
        base_cmap="terrain",
        stream_cmap="plasma",
        percentile=args.stream_percentile,
        stream_alpha=1.0,
        base_label="Elevation (m)",
        stream_label="Discharge Potential",
        title="Discharge-Colored Streams on Elevation",
        lake_mask=lake_mask_aligned,
        base_log_scale=False,  # Elevation doesn't need log scale
        variable_width=args.variable_width,
        max_width=args.max_stream_width,
    )

    # Overlay 4: Upstream rainfall-colored streams on drainage area
    plot_stream_overlay(
        base_data=flow_result["drainage_area"],
        stream_threshold_data=flow_result["drainage_area"],
        stream_color_data=flow_result["upstream_rainfall"],
        output_path=args.output_dir / f"14_stream_overlay_rainfall_on_drainage{width_suffix}.png",
        base_cmap="viridis",
        stream_cmap="cividis",
        percentile=args.stream_percentile,
        stream_alpha=1.0,
        base_label="Drainage Area (cells)",
        stream_label="Upstream Rainfall",
        title="Rainfall-Colored Streams on Drainage Area",
        lake_mask=lake_mask_aligned,
        variable_width=args.variable_width,
        max_width=args.max_stream_width,
    )

    # Overlay 5: Discharge-colored streams on drainage area
    plot_stream_overlay(
        base_data=flow_result["drainage_area"],
        stream_threshold_data=flow_result["drainage_area"],
        stream_color_data=diag_discharge,
        output_path=args.output_dir / f"15_stream_overlay_discharge_on_drainage{width_suffix}.png",
        base_cmap="viridis",
        stream_cmap="plasma",
        percentile=args.stream_percentile,
        stream_alpha=1.0,
        base_label="Drainage Area (cells)",
        stream_label="Discharge Potential",
        title="Discharge-Colored Streams on Drainage Area",
        lake_mask=lake_mask_aligned,
        variable_width=args.variable_width,
        max_width=args.max_stream_width,
    )

    # Overlay 6: Variable width example (always enabled) - discharge on drainage
    plot_stream_overlay(
        base_data=flow_result["drainage_area"],
        stream_threshold_data=flow_result["drainage_area"],
        stream_color_data=diag_discharge,
        output_path=args.output_dir / "16_stream_overlay_variable_width.png",
        base_cmap="viridis",
        stream_cmap="plasma",
        percentile=args.stream_percentile,
        stream_alpha=1.0,
        base_label="Drainage Area (cells)",
        stream_label="Discharge Potential",
        title="Variable Width Streams (width ∝ discharge)",
        lake_mask=lake_mask_aligned,
        variable_width=True,  # Always enabled for this example
        max_width=args.max_stream_width,
    )

    # Step 6: Create 3D terrain visualization
    print("\nCreating 3D terrain...")
    clear_scene()

    # Load DEM (ground truth for 3D rendering)
    print(f"Loading DEM from {args.dem_dir}...")
    # Get CRS from flow metadata (flow was computed from this DEM, so CRS should match)
    dem_crs = metadata.get("crs", "EPSG:4326")
    terrain = Terrain(dem, transform, dem_crs=dem_crs)

    # Add transforms: reproject to UTM, flip, scale, downsample
    terrain.add_transform(reproject_raster("EPSG:4326", "EPSG:32611", num_threads=4))
    terrain.add_transform(flip_raster(axis="horizontal"))
    terrain.add_transform(scale_elevation(scale_factor=0.0001))
    terrain.configure_for_target_vertices(args.target_vertices, method="average")

    # Add flow accumulation results as data layers
    print("Adding flow data layers...")

    # If flow data was downsampled, resample to match DEM using geographically-aware reprojection
    # This ensures geographic metadata (transform, CRS) is preserved correctly
    if metadata["downsampling_applied"]:
        print(f"  Resampling flow data from {metadata['downsampled_shape']} to {dem.shape}...")
        from rasterio.warp import reproject, Resampling

        # Extract flow data at downsampled resolution
        drainage_area_ds = flow_result["drainage_area"]
        upstream_rainfall_ds = flow_result["upstream_rainfall"]
        flow_transform_ds = Affine(*flow_result["metadata"]["transform"])

        # Prepare output arrays at DEM resolution
        drainage_area = np.zeros(dem.shape, dtype=np.float32)
        upstream_rainfall = np.zeros(dem.shape, dtype=np.float32)

        # Get CRS from metadata (don't hardcode - handle any projection)
        flow_crs = metadata.get("crs", "EPSG:4326")

        # Reproject using rasterio (geographically-aware, respects transforms)
        # This handles different resolution, alignment, AND projection
        reproject(
            source=drainage_area_ds,
            destination=drainage_area,
            src_transform=flow_transform_ds,
            src_crs=flow_crs,  # Use actual CRS from flow computation
            dst_transform=transform,
            dst_crs=dem_crs,  # Use actual DEM CRS
            resampling=Resampling.bilinear,
        )

        reproject(
            source=upstream_rainfall_ds,
            destination=upstream_rainfall,
            src_transform=flow_transform_ds,
            src_crs=flow_crs,
            dst_transform=transform,
            dst_crs=dem_crs,
            resampling=Resampling.bilinear,
        )

        # Use original DEM transform for resampled data
        flow_transform = transform
        print(f"  ✓ Reprojected with geographic awareness (src: {flow_transform_ds}, dst: {transform})")
    else:
        # No downsampling - use flow data as-is
        drainage_area = flow_result["drainage_area"]
        upstream_rainfall = flow_result["upstream_rainfall"]
        flow_transform = Affine(*flow_result["metadata"]["transform"])

    # Compute discharge potential
    discharge_potential = compute_discharge_potential(drainage_area, upstream_rainfall)

    # Use log scale for better visualization
    drainage_log = np.log10(drainage_area + 1)
    rainfall_log = np.log10(upstream_rainfall + 1)
    discharge_log = np.log10(discharge_potential + 1)

    # Debug: Print drainage value range
    print(f"  Drainage area range: {drainage_area.min():.1f} - {drainage_area.max():.1f}")
    print(f"  Drainage log range: {drainage_log.min():.3f} - {drainage_log.max():.3f}")
    print(f"  Non-zero drainage cells: {np.sum(drainage_area > 0):,}")

    # Apply transforms to DEM only first (reproject, flip, scale, downsample)
    # Color data layers will be added AFTER so they don't get scaled by scale_elevation
    terrain.apply_transforms()

    # Now add color data layers using same_extent_as to match transformed DEM
    # These won't go through scale_elevation since transforms already applied
    print("Adding color data layers (post-transform)...")
    terrain.add_data_layer(
        "drainage_area_log",
        drainage_log.astype(np.float32),
        flow_transform,
        crs=dem_crs,
        target_layer="dem",  # Align to transformed DEM grid
    )

    terrain.add_data_layer(
        "upstream_rainfall_log",
        rainfall_log.astype(np.float32),
        flow_transform,
        crs=dem_crs,
        target_layer="dem",
    )

    terrain.add_data_layer(
        "discharge_potential_log",
        discharge_log.astype(np.float32),
        flow_transform,
        crs=dem_crs,
        target_layer="dem",
    )

    # Create stream mask and stream discharge layer for 3D overlay
    # Extract streams as top percentile of drainage area
    valid_drainage = drainage_area[drainage_area > 0]
    if len(valid_drainage) > 0:
        stream_threshold = np.percentile(valid_drainage, args.stream_percentile)
        stream_mask = drainage_area >= stream_threshold
        # Stream discharge: discharge values only at stream pixels, 0 elsewhere
        stream_discharge = np.where(stream_mask, discharge_log, 0).astype(np.float32)
        terrain.add_data_layer(
            "stream_discharge",
            stream_discharge,
            flow_transform,
            crs=dem_crs,
            target_layer="dem",
        )
        print(f"  Stream cells (top {100-args.stream_percentile:.0f}%): {np.sum(stream_mask):,}")
    else:
        stream_mask = np.zeros_like(drainage_area, dtype=bool)
        print("  Warning: No valid drainage data for stream extraction")

    # Add water masks for 3D rendering (ocean + lakes)
    print("Adding water masks...")
    terrain.add_data_layer(
        "ocean_mask",
        ocean_mask.astype(np.float32),
        transform,
        crs=dem_crs,
        target_layer="dem",
    )
    print(f"  Ocean mask: {np.sum(ocean_mask):,} cells")

    if lake_mask is not None:
        terrain.add_data_layer(
            "lake_mask",
            (lake_mask > 0).astype(np.float32),
            transform,
            crs=dem_crs,
            target_layer="dem",
        )
        print(f"  Lake mask: {np.sum(lake_mask > 0):,} cells")

    # Create combined water mask from aligned data layers
    water_mask_combined = terrain.data_layers["ocean_mask"]["data"].astype(bool)
    if "lake_mask" in terrain.data_layers:
        water_mask_combined = water_mask_combined | terrain.data_layers["lake_mask"]["data"].astype(bool)
    print(f"  Combined water mask: {np.sum(water_mask_combined):,} cells")

    # Set colormap based on user choice
    # Determine default base colormap if not specified
    if args.base_cmap is None:
        default_cmaps = {
            "elevation": "modern_terrain",
            "drainage": "viridis_r",  # reversed so high values are dark
            "discharge": "plasma_r",
            "rainfall": "viridis_r",
        }
        args.base_cmap = default_cmaps[args.color_by]

    # Get colormap names (use _r suffix for reversed, e.g., viridis_r)
    base_cmap_name = get_cmap_name(args.base_cmap)
    stream_cmap_name = get_cmap_name(args.stream_cmap)

    if args.color_by == "elevation":
        base_colormap = lambda elev: elevation_colormap(elev, cmap_name=base_cmap_name)
        base_layer = "dem"
        base_label = "elevation"
    elif args.color_by == "drainage":
        base_colormap = lambda drain: elevation_colormap(drain, cmap_name=base_cmap_name, min_elev=0.1)
        base_layer = "drainage_area_log"
        base_label = "drainage area (log)"
    elif args.color_by == "discharge":
        base_colormap = lambda discharge: elevation_colormap(discharge, cmap_name=base_cmap_name)
        base_layer = "discharge_potential_log"
        base_label = "discharge potential (log)"
    else:  # rainfall
        base_colormap = lambda rain: elevation_colormap(rain, cmap_name=base_cmap_name)
        base_layer = "upstream_rainfall_log"
        base_label = "upstream rainfall (log)"

    print(f"Using colormaps: base={args.base_cmap} ({base_cmap_name}), stream={args.stream_cmap} ({stream_cmap_name})")

    # Apply color mapping - with or without stream overlay
    if args.stream_overlay:
        # Determine stream color layer based on --stream-color-by
        if args.stream_color_by == "drainage":
            stream_layer = "drainage_area_log"
            stream_label = "drainage"
        elif args.stream_color_by == "rainfall":
            stream_layer = "upstream_rainfall_log"
            stream_label = "rainfall"
        else:  # discharge
            stream_layer = "discharge_potential_log"
            stream_label = "discharge"

        print(f"Coloring by {base_label} + stream overlay ({stream_label}, top {100-args.stream_percentile:.0f}%)...")

        # Create stream-specific data layer (stream values only where streams exist)
        stream_data = terrain.data_layers[stream_layer]["data"]
        stream_mask_data = terrain.data_layers["stream_discharge"]["data"] > 0
        stream_values = np.where(stream_mask_data, stream_data, 0).astype(np.float32)

        # Add the stream-colored layer
        terrain.data_layers["stream_colored"] = {
            "data": stream_values,
            "transformed_data": stream_values,
            "transform": terrain.data_layers[stream_layer].get("transform"),
            "crs": terrain.data_layers[stream_layer].get("crs"),
        }

        def stream_colormap(data):
            return elevation_colormap(data, cmap_name=stream_cmap_name)

        terrain.set_multi_color_mapping(
            base_colormap=base_colormap,
            base_source_layers=[base_layer],
            overlays=[
                {
                    "colormap": stream_colormap,
                    "source_layers": ["stream_colored"],
                    "threshold": 0.01,
                    "priority": 10,
                },
            ],
        )
    else:
        print(f"Coloring by {base_label}...")
        terrain.set_color_mapping(base_colormap, source_layers=[base_layer])

    # Create mesh
    terrain.compute_colors()

    # Debug: Check color distribution
    print(f"\n  Color stats after compute_colors():")
    print(f"    Unique colors: {len(np.unique(terrain.colors, axis=0))}")
    print(f"    Color range R: {terrain.colors[:, 0].min():.3f} - {terrain.colors[:, 0].max():.3f}")
    print(f"    Color range G: {terrain.colors[:, 1].min():.3f} - {terrain.colors[:, 1].max():.3f}")
    print(f"    Color range B: {terrain.colors[:, 2].min():.3f} - {terrain.colors[:, 2].max():.3f}")
    print(f"    Available layers: {list(terrain.data_layers.keys())}")

    mesh = terrain.create_mesh(
        scale_factor=100,
        height_scale=args.height_scale,
        center_model=True,
        boundary_extension=True,
        base_depth=1.0,
        water_mask=water_mask_combined,  # Apply water coloring (ocean + lakes)
    )
    print(f"✓ Created mesh: {len(mesh.data.vertices):,} vertices")

    # Apply material
    apply_colormap_material(mesh.data.materials[0], terrain_material="eggshell")

    # Step 7: Setup scene
    print("\nSetting up scene...")

    # Camera - use orthographic for top-down views, perspective otherwise
    is_above = args.camera in ("above", "above-tilted")
    camera = position_camera_relative(
        mesh,
        direction=args.camera,
        camera_type="ORTHO" if is_above else "PERSP",
        focal_length=50,
        distance=1.0 if not is_above else 2.0,
        elevation=1.0 if not is_above else 3.0,  # High elevation for overhead view
        ortho_scale=1.1 if is_above else 1.2,
    )
    print(f"  Camera: {args.camera} ({'orthographic' if is_above else 'perspective'})")

    # Sky lighting
    setup_hdri_lighting(
        sun_elevation=15.0,
        sun_rotation=225.0,
        sun_intensity=0.04,  # Dimmed slightly for --fast mode
        air_density=0.05,
        visible_to_camera=False,
        sky_strength=1.75,
    )

    # Background plane
    create_background_plane(
        camera=camera,
        mesh_or_meshes=mesh,
        distance_below=0.0,
        color="#000000",
        roughness=0.20,
        size_multiplier=10,
        receive_shadows=True,
    )
    print("✓ Scene ready")

    # Step 8: Render
    if not args.no_render:
        # Use pre-calculated render dimensions
        render_width = args.render_width
        render_height = args.render_height

        print(f"\nRendering ({args.width_in:.1f}\"×{args.height_in:.1f}\" @ {args.dpi} DPI = {render_width}×{render_height}px, {args.samples} samples)...")
        setup_render_settings(use_gpu=True, samples=args.samples, use_denoising=True)

        output_filename = f"san_diego_flow_{args.color_by}.png"
        output_path = args.output_dir / output_filename
        render_scene_to_file(
            str(output_path), width=render_width, height=render_height, file_format="PNG", color_mode="RGB"
        )

        print(f"✓ Rendered to {output_path}")
    else:
        print("\nSkipped render (--no-render)")

    print("\n✓ Done!")
    print(f"\nOutput files:")
    print(f"  Diagnostic plots: {args.output_dir}/01_*.png through 11_*.png")
    print(f"  3D render: {args.output_dir / f'san_diego_flow_{args.color_by}.png'}")
    print(f"  Flow GeoTIFFs: {flow_output_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
