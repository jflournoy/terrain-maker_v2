"""
Water body handling for flow accumulation.

Provides functions to:
- Download lake/reservoir polygons from NHD (USA) or HydroLAKES (global)
- Identify lake outlets (pour points)
- Rasterize lake polygons to mask matching DEM grid
- Route flow through lakes to their outlets

Data Sources:
- NHD: https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer
- HydroLAKES: https://www.hydrosheds.org/products/hydrolakes
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import numpy as np
from rasterio import Affine
import json
import hashlib


def rasterize_lakes_to_mask(
    lakes_geojson: Dict[str, Any],
    bbox: Tuple[float, float, float, float],
    resolution: float = 0.001,
) -> Tuple[np.ndarray, Affine]:
    """
    Rasterize lake polygons to a labeled mask with geographic transform.

    Uses rasterio.features.rasterize for robust handling of complex polygons,
    interior rings (islands), and MultiPolygon geometries.

    Parameters
    ----------
    lakes_geojson : dict
        GeoJSON FeatureCollection with lake polygons
    bbox : tuple
        Bounding box (south, west, north, east) in degrees
    resolution : float
        Cell size in degrees (default: 0.001 ≈ 100m)

    Returns
    -------
    mask : np.ndarray (uint16)
        Labeled mask where 0 = no lake, 1+ = lake ID
    transform : Affine
        Geographic transform (pixel to coordinate)
    """
    from rasterio.features import rasterize as rio_rasterize
    from shapely.geometry import shape

    south, west, north, east = bbox

    # Calculate grid dimensions
    n_rows = int(np.ceil((north - south) / resolution))
    n_cols = int(np.ceil((east - west) / resolution))

    # Create Affine transform (top-left origin, Y decreasing)
    transform = Affine(resolution, 0, west, 0, -resolution, north)

    # Build list of (geometry, label) pairs for rasterio
    shapes = []
    features = lakes_geojson.get("features", [])
    for idx, feature in enumerate(features, start=1):
        geom_dict = feature.get("geometry")
        if not geom_dict:
            continue
        try:
            geom = shape(geom_dict)
            if geom.is_empty:
                continue
            shapes.append((geom, idx))
        except Exception:
            continue

    if not shapes:
        mask = np.zeros((n_rows, n_cols), dtype=np.uint16)
        return mask, transform

    # Rasterize all polygons at once — handles holes, MultiPolygon, complex coastlines
    mask = rio_rasterize(
        shapes,
        out_shape=(n_rows, n_cols),
        transform=transform,
        dtype=np.uint16,
        fill=0,
    )

    return mask, transform


def identify_outlet_cells(
    lake_mask: np.ndarray,
    outlets: Dict[int, Tuple[float, float]],
    transform: Affine,
) -> np.ndarray:
    """
    Convert outlet points to raster cell locations.

    Parameters
    ----------
    lake_mask : np.ndarray
        Labeled lake mask (0 = no lake, N = lake ID)
    outlets : dict
        Mapping of lake_id -> (lon, lat) outlet coordinates
    transform : Affine
        Geographic transform

    Returns
    -------
    outlet_mask : np.ndarray (bool)
        Boolean mask where True = outlet cell
    """
    outlet_mask = np.zeros(lake_mask.shape, dtype=bool)
    inv_transform = ~transform

    for lake_id, (lon, lat) in outlets.items():
        # Convert geographic coords to pixel coords
        col, row = inv_transform * (lon, lat)
        row, col = int(row), int(col)

        # Check bounds
        if 0 <= row < lake_mask.shape[0] and 0 <= col < lake_mask.shape[1]:
            outlet_mask[row, col] = True

    return outlet_mask


def create_lake_flow_routing(
    lake_mask: np.ndarray,
    outlet_mask: np.ndarray,
    dem: np.ndarray,
) -> np.ndarray:
    """
    Create flow direction grid that routes all lake cells toward outlet.

    Uses BFS from outlet to assign converging flow directions.
    For endorheic lakes (no outlet), all cells get flow_dir=0.

    Parameters
    ----------
    lake_mask : np.ndarray
        Labeled lake mask (0 = no lake, N = lake ID)
    outlet_mask : np.ndarray (bool)
        Boolean mask of outlet cells
    dem : np.ndarray
        Digital elevation model (used for tie-breaking if needed)

    Returns
    -------
    flow_dir : np.ndarray (uint8)
        D8 flow direction for lake cells (0 elsewhere)
    """
    from scipy.ndimage import label

    rows, cols = lake_mask.shape
    flow_dir = np.zeros((rows, cols), dtype=np.uint8)

    # D8 direction codes and offsets
    # Direction encodes where water flows TO
    D8_OFFSETS = {
        1: (0, 1),    # E
        2: (-1, 1),   # NE
        4: (-1, 0),   # N
        8: (-1, -1),  # NW
        16: (0, -1),  # W
        32: (1, -1),  # SW
        64: (1, 0),   # S
        128: (1, 1),  # SE
    }

    # Reverse: what direction points TO a cell
    REVERSE_DIR = {
        1: 16,   # E -> W
        2: 32,   # NE -> SW
        4: 64,   # N -> S
        8: 128,  # NW -> SE
        16: 1,   # W -> E
        32: 2,   # SW -> NE
        64: 4,   # S -> N
        128: 8,  # SE -> NW
    }

    # Process each unique lake
    unique_lakes = np.unique(lake_mask)
    unique_lakes = unique_lakes[unique_lakes > 0]

    for lake_id in unique_lakes:
        this_lake = lake_mask == lake_id

        # Find outlet(s) for this lake
        lake_outlets = outlet_mask & this_lake

        if not np.any(lake_outlets):
            # Endorheic lake - all cells are terminal sinks
            flow_dir[this_lake] = 0
            continue

        # BFS from outlet to assign flow directions that converge toward outlet
        # Each cell's flow direction points toward its BFS parent (toward outlet)
        visited = np.zeros_like(this_lake, dtype=bool)
        queue = []

        # Initialize with outlet cells
        outlet_rows, outlet_cols = np.where(lake_outlets)
        for r, c in zip(outlet_rows, outlet_cols):
            visited[r, c] = True
            flow_dir[r, c] = 0  # Outlet is terminal
            queue.append((r, c))

        # BFS
        head = 0
        while head < len(queue):
            r, c = queue[head]
            head += 1

            # Check all 8 neighbors
            for direction, (dr, dc) in D8_OFFSETS.items():
                nr, nc = r + dr, c + dc

                if (0 <= nr < rows and 0 <= nc < cols and
                        this_lake[nr, nc] and not visited[nr, nc]):
                    visited[nr, nc] = True
                    # Neighbor's flow points toward current cell (toward outlet)
                    flow_dir[nr, nc] = REVERSE_DIR[direction]
                    queue.append((nr, nc))

    return flow_dir


def _trace_flows_to_lake(
    flow_dir: np.ndarray,
    lake_mask: np.ndarray,
    start_r: int,
    start_c: int,
    lake_id: int,
    max_steps: int = 100,
) -> bool:
    """Trace flow path from a cell and check if it re-enters the given lake.

    Parameters
    ----------
    flow_dir : np.ndarray
        D8 flow direction grid
    lake_mask : np.ndarray
        Labeled lake mask
    start_r, start_c : int
        Starting cell (should be outside the lake)
    lake_id : int
        Lake ID to check for re-entry
    max_steps : int
        Maximum steps to trace before giving up

    Returns
    -------
    bool
        True if the flow path re-enters the lake (would create a cycle)
    """
    from src.terrain.flow_accumulation import D8_OFFSETS

    rows, cols = flow_dir.shape
    r, c = start_r, start_c

    for _ in range(max_steps):
        d = flow_dir[r, c]
        if d == 0:
            return False  # Reached terminal — no cycle
        if d not in D8_OFFSETS:
            return False  # Invalid direction — no cycle

        dr, dc = D8_OFFSETS[d]
        r, c = r + dr, c + dc

        if not (0 <= r < rows and 0 <= c < cols):
            return False  # Left the grid — no cycle

        # Check if we re-entered the lake
        if lake_id > 0 and lake_mask[r, c] == lake_id:
            return True
        # For boolean masks, any lake cell counts
        if lake_id == 0 and lake_mask[r, c] > 0:
            return True

    return False  # Didn't re-enter within max_steps


def find_lake_spillways(
    lake_mask: np.ndarray,
    dem: np.ndarray,
) -> dict:
    """Find natural spillway for each lake based on DEM topology.

    Scans all boundary cells of each lake (lake cells with at least one
    non-lake neighbor) and finds where the surrounding rim is lowest.
    The spillway is the (boundary_cell, non-lake_neighbor) pair with
    the lowest non-lake neighbor elevation — this is where the lake
    would naturally overflow.

    Works for both natural lakes and reservoirs: the dam location
    typically corresponds to the lowest rim point because dams are
    built across valleys.

    Parameters
    ----------
    lake_mask : np.ndarray
        Labeled lake mask (0 = no lake, N = lake ID)
    dem : np.ndarray
        Digital elevation model

    Returns
    -------
    dict
        Mapping lake_id -> (spillway_row, spillway_col, direction_code)
        where direction_code is the D8 direction from the spillway cell
        to its lowest non-lake neighbor.
    """
    from src.terrain.flow_accumulation import D8_DIRECTIONS

    rows, cols = lake_mask.shape
    lake_ids = np.unique(lake_mask[lake_mask > 0])
    spillways = {}

    for lake_id in lake_ids:
        best_neighbor_elev = np.inf
        best_spill = None  # (row, col, direction_code)

        # Find all cells belonging to this lake
        lake_rows, lake_cols = np.where(lake_mask == lake_id)

        for r, c in zip(lake_rows, lake_cols):
            # Check all 8 neighbors
            for (dr, dc), direction_code in D8_DIRECTIONS.items():
                nr, nc = r + dr, c + dc

                # Bounds check
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue

                # Skip cells in the same lake
                if lake_mask[nr, nc] == lake_id:
                    continue

                # This is a non-lake neighbor — candidate rim point
                if dem[nr, nc] < best_neighbor_elev:
                    best_neighbor_elev = dem[nr, nc]
                    best_spill = (r, c, direction_code)

        if best_spill is not None:
            spillways[lake_id] = best_spill

    return spillways


def compute_outlet_downstream_directions(
    flow_dir: np.ndarray,
    lake_mask: np.ndarray,
    outlet_mask: np.ndarray,
    dem: np.ndarray,
    basin_mask: Optional[np.ndarray] = None,
    spillways: Optional[dict] = None,
) -> np.ndarray:
    """
    Compute flow direction FROM lake outlets TO downstream terrain.

    For through-draining lakes, the outlet gets a flow direction pointing
    to the lowest adjacent non-lake cell whose flow path does NOT
    re-enter the lake (which would create a cycle). This turns lakes
    into links in river chains rather than terminal sinks.

    When a spillway dict is provided (from find_lake_spillways), the
    precomputed spillway direction is used as a fallback if no strictly
    lower non-lake neighbor is found. This handles the common case where
    the DEM has the lake surface at or above the rim elevation.

    Endorheic outlets (inside basins) remain terminal (flow_dir=0).

    Parameters
    ----------
    flow_dir : np.ndarray (uint8)
        D8 flow direction grid (outlet cells typically have 0)
    lake_mask : np.ndarray
        Labeled lake mask (0 = no lake, N = lake ID)
    outlet_mask : np.ndarray (bool)
        Boolean mask of outlet cells
    dem : np.ndarray
        Digital elevation model
    basin_mask : np.ndarray (bool), optional
        Boolean mask of endorheic basin cells. Outlets inside basins
        remain terminal.
    spillways : dict, optional
        Dict from find_lake_spillways(): lake_id -> (row, col, direction).
        Used as fallback when no strictly lower neighbor exists.

    Returns
    -------
    flow_dir : np.ndarray (uint8)
        Copy of input with outlet cells updated to point downstream
    """
    from src.terrain.flow_accumulation import D8_DIRECTIONS, D8_OFFSETS

    rows, cols = flow_dir.shape
    result = flow_dir.copy()

    if basin_mask is None:
        basin_mask = np.zeros((rows, cols), dtype=bool)

    outlet_rows, outlet_cols = np.where(outlet_mask)

    for r, c in zip(outlet_rows, outlet_cols):
        # Endorheic outlets stay terminal
        if basin_mask[r, c]:
            continue

        # Skip outlets that already have a flow direction (e.g., basin
        # outlets that kept their terrain flow, or outlets already
        # connected by a previous routing step)
        if flow_dir[r, c] != 0:
            continue

        # Collect candidate neighbors: lower, non-lake, sorted by elevation
        this_lake_id = lake_mask[r, c]
        candidates = []

        for (dr, dc), direction_code in D8_DIRECTIONS.items():
            nr, nc = r + dr, c + dc

            # Bounds check
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            # Skip cells in the same lake
            if lake_mask[nr, nc] == this_lake_id and this_lake_id > 0:
                continue

            # Only consider cells lower than the outlet
            if dem[nr, nc] < dem[r, c]:
                candidates.append((dem[nr, nc], direction_code, nr, nc))

        # Sort by elevation (lowest first) — prefer steepest descent
        candidates.sort(key=lambda x: x[0])

        # Pick the lowest candidate whose flow path doesn't re-enter the lake
        best_dir = 0
        for _elev, direction_code, nr, nc in candidates:
            if not _trace_flows_to_lake(
                flow_dir, lake_mask, nr, nc, this_lake_id
            ):
                best_dir = direction_code
                break

        # Fallback: use precomputed spillway direction if no strictly lower
        # neighbor was found. The spillway is the lowest rim point — the
        # DEM may have the lake surface at or above the rim elevation
        # (common for reservoirs where DEM represents water surface).
        if best_dir == 0 and spillways is not None and this_lake_id in spillways:
            spill_r, spill_c, spill_dir = spillways[this_lake_id]
            # Only use if this outlet IS the spillway cell
            if r == spill_r and c == spill_c:
                # Verify the spillway direction doesn't create a cycle
                if spill_dir in D8_OFFSETS:
                    sdr, sdc = D8_OFFSETS[spill_dir]
                    snr, snc = r + sdr, c + sdc
                    if (0 <= snr < rows and 0 <= snc < cols
                            and not _trace_flows_to_lake(
                                flow_dir, lake_mask, snr, snc, this_lake_id
                            )):
                        best_dir = spill_dir

        result[r, c] = best_dir

    return result


def download_water_bodies(
    bbox: Tuple[float, float, float, float],
    output_dir: str,
    data_source: str = "hydrolakes",
    min_area_km2: float = 0.1,
    force_download: bool = False,
) -> Path:
    """
    Download lake polygons and outlets for bounding box.

    Parameters
    ----------
    bbox : tuple
        Bounding box (south, west, north, east) in degrees
    output_dir : str
        Directory to save downloaded data
    data_source : str
        Data source: "nhd" (USA only) or "hydrolakes" (global)
    min_area_km2 : float
        Minimum lake area to include (km²)
    force_download : bool
        If True, re-download even if cached

    Returns
    -------
    Path
        Path to downloaded GeoJSON file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create cache key from bbox
    bbox_str = f"{bbox[0]:.4f}_{bbox[1]:.4f}_{bbox[2]:.4f}_{bbox[3]:.4f}"
    cache_key = hashlib.md5(f"{data_source}_{bbox_str}".encode()).hexdigest()[:12]
    cache_file = output_path / f"water_bodies_{data_source}_{cache_key}.geojson"

    if cache_file.exists() and not force_download:
        return cache_file

    if data_source == "nhd":
        waterbodies, flowlines = download_nhd_water_bodies(bbox, str(output_path))
        # Combine into single GeoJSON with outlet info
        geojson = _merge_nhd_with_outlets(waterbodies, flowlines)
    elif data_source == "hydrolakes":
        geojson = download_hydrolakes(bbox, str(output_path))
    else:
        raise ValueError(f"Unknown data source: {data_source}")

    # Filter by area
    if min_area_km2 > 0:
        geojson = _filter_by_area(geojson, min_area_km2)

    # Save to cache
    with open(cache_file, "w") as f:
        json.dump(geojson, f)

    return cache_file


def download_nhd_water_bodies(
    bbox: Tuple[float, float, float, float],
    output_dir: str,
) -> Tuple[Dict, Dict]:
    """
    Download NHDWaterbody and NHDFlowline from USGS REST API.

    Parameters
    ----------
    bbox : tuple
        Bounding box (south, west, north, east)
    output_dir : str
        Output directory

    Returns
    -------
    waterbodies : dict
        GeoJSON of lake polygons
    flowlines : dict
        GeoJSON of stream lines
    """
    import requests

    base_url = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer"

    # NHD layer IDs (from service metadata)
    # Layer 12: Waterbody - Large Scale (detailed polygons)
    # Layer 6: Flowline - Large Scale (stream lines)
    waterbody_layer = 12
    flowline_layer = 6

    south, west, north, east = bbox
    geometry = f"{west},{south},{east},{north}"

    params = {
        "geometry": geometry,
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "outSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "f": "geojson",
    }

    # Download waterbodies
    wb_url = f"{base_url}/{waterbody_layer}/query"
    response = requests.get(wb_url, params=params, timeout=60)
    response.raise_for_status()
    waterbodies = response.json()

    # Download flowlines
    fl_url = f"{base_url}/{flowline_layer}/query"
    response = requests.get(fl_url, params=params, timeout=60)
    response.raise_for_status()
    flowlines = response.json()

    return waterbodies, flowlines


def download_hydrolakes(
    bbox: Tuple[float, float, float, float],
    output_dir: str,
) -> Dict:
    """
    Get HydroLAKES data for bounding box.

    Note: HydroLAKES is distributed as a shapefile that must be downloaded
    once from hydrosheds.org. This function filters the local data to bbox.

    For initial implementation, returns empty FeatureCollection with
    instructions for manual download.

    Parameters
    ----------
    bbox : tuple
        Bounding box (south, west, north, east)
    output_dir : str
        Output directory

    Returns
    -------
    dict
        GeoJSON FeatureCollection with lake polygons and pour_point properties
    """
    # Check for local HydroLAKES shapefile
    # First check in project data directory, then in output directory
    hydrolakes_paths = [
        Path("data/hydrolakes/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10.shp"),
        Path(output_dir) / "HydroLAKES_polys_v10.shp",
    ]

    hydrolakes_path = None
    for path in hydrolakes_paths:
        if path.exists():
            hydrolakes_path = path
            break

    if hydrolakes_path is None:
        # Return empty with instructions
        print("HydroLAKES shapefile not found")
        print("Expected at: data/hydrolakes/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10.shp")
        print("Or download from: https://www.hydrosheds.org/products/hydrolakes")
        print("Extract HydroLAKES_polys_v10.shp to data/hydrolakes/HydroLAKES_polys_v10_shp/")
        return {"type": "FeatureCollection", "features": []}

    # Filter shapefile to bbox using geopandas
    try:
        import geopandas as gpd
        from shapely.geometry import box

        # Convert bbox from (south, west, north, east) to geopandas format (minx, miny, maxx, maxy)
        south, west, north, east = bbox
        bbox_geopandas = (west, south, east, north)

        # Load lake polygons
        gdf = gpd.read_file(hydrolakes_path, bbox=bbox_geopandas)

        # Load pour points shapefile to get outlet coordinates
        points_path = Path(hydrolakes_path).parent / "HydroLAKES_points_v10.shp"
        if points_path.exists():
            try:
                gdf_points = gpd.read_file(points_path, bbox=bbox_geopandas)

                # Create a mapping of Hylak_id -> outlet coordinates
                outlets_dict = {}
                for idx, row in gdf_points.iterrows():
                    lake_id = row.get('Hylak_id')
                    geom = row.geometry
                    if lake_id is not None and geom is not None:
                        outlets_dict[lake_id] = (geom.x, geom.y)

                # Add outlet coordinates to polygon features
                for idx, row in gdf.iterrows():
                    lake_id = row.get('Hylak_id')
                    if lake_id in outlets_dict:
                        lon, lat = outlets_dict[lake_id]
                        row['pour_point'] = [lon, lat]
                        row['Pour_long'] = lon
                        row['Pour_lat'] = lat
            except Exception as e:
                print(f"Warning: Could not load HydroLAKES pour points: {e}")

        # Convert to GeoJSON
        geojson = json.loads(gdf.to_json())

        # Add pour_point from attributes if available
        for feature in geojson.get("features", []):
            props = feature.get("properties", {})
            if "pour_point" in props:
                # Already added from points file
                pass
            elif "Pour_lat" in props and "Pour_long" in props:
                # HydroLAKES pour point fields
                props["pour_point"] = [props["Pour_long"], props["Pour_lat"]]

        return geojson

    except ImportError:
        print("geopandas required for HydroLAKES filtering")
        return {"type": "FeatureCollection", "features": []}


def identify_lake_outlets_from_nhd(
    waterbodies: Dict,
    flowlines: Dict,
) -> Dict[int, Tuple[float, float]]:
    """
    Identify outlet points by finding where flowlines exit lake polygons.

    Parameters
    ----------
    waterbodies : dict
        GeoJSON of lake polygons
    flowlines : dict
        GeoJSON of stream lines

    Returns
    -------
    dict
        Mapping of lake_id -> (lon, lat) outlet coordinates
    """
    try:
        from shapely.geometry import shape, Point
        from shapely.ops import nearest_points
    except ImportError:
        print("shapely required for NHD outlet detection")
        return {}

    outlets = {}

    # Build lookup of lake polygons
    lakes = {}
    for feature in waterbodies.get("features", []):
        lake_id = feature.get("properties", {}).get("OBJECTID", id(feature))
        geom = shape(feature.get("geometry"))
        lakes[lake_id] = geom

    # Find flowlines that intersect each lake
    for lake_id, lake_poly in lakes.items():
        boundary = lake_poly.boundary
        outlet_point = None
        min_elev = float("inf")

        for fl_feature in flowlines.get("features", []):
            fl_geom = shape(fl_feature.get("geometry"))

            if lake_poly.intersects(fl_geom):
                # Find intersection point on lake boundary
                intersection = boundary.intersection(fl_geom)
                if intersection.is_empty:
                    continue

                # Get a point from intersection
                if intersection.geom_type == "Point":
                    point = intersection
                elif intersection.geom_type == "MultiPoint":
                    point = list(intersection.geoms)[0]
                else:
                    # Use centroid for lines
                    point = intersection.centroid

                # Check if this is the outlet (would need elevation check ideally)
                # For now, just use the first intersection found
                if outlet_point is None:
                    outlet_point = point

        if outlet_point is not None:
            outlets[lake_id] = (outlet_point.x, outlet_point.y)

    return outlets


def _merge_nhd_with_outlets(
    waterbodies: Dict,
    flowlines: Dict,
) -> Dict:
    """Merge NHD waterbodies with outlet information from flowlines."""
    outlets = identify_lake_outlets_from_nhd(waterbodies, flowlines)

    # Add outlet info to waterbody features
    for feature in waterbodies.get("features", []):
        lake_id = feature.get("properties", {}).get("OBJECTID", id(feature))
        if lake_id in outlets:
            feature["properties"]["outlet"] = outlets[lake_id]

    return waterbodies


def _filter_by_area(geojson: Dict, min_area_km2: float) -> Dict:
    """Filter GeoJSON features by area."""
    try:
        from shapely.geometry import shape
    except ImportError:
        return geojson

    filtered_features = []
    for feature in geojson.get("features", []):
        geom = shape(feature.get("geometry"))
        # Approximate area in km² (rough, assumes lat/lon)
        area_deg2 = geom.area
        area_km2 = area_deg2 * 111 * 111  # Very rough approximation

        if area_km2 >= min_area_km2:
            filtered_features.append(feature)

    return {"type": "FeatureCollection", "features": filtered_features}


def identify_lake_inlets(
    lake_mask: np.ndarray,
    dem: np.ndarray,
    outlet_mask: Optional[np.ndarray] = None,
) -> Dict[int, list]:
    """
    Identify inlet cells for each lake (where water enters from surrounding terrain).

    For each unique lake, finds cells on the lake boundary that are adjacent to
    lower-elevation non-lake cells. These represent where water naturally flows
    into the lake from surrounding terrain.

    Parameters
    ----------
    lake_mask : np.ndarray
        Labeled lake mask (0 = no lake, 1+ = lake ID)
    dem : np.ndarray
        Conditioned digital elevation model
    outlet_mask : np.ndarray, optional
        Boolean outlet mask to exclude outlet cells from inlet identification

    Returns
    -------
    dict
        Mapping of lake_id -> list of (row, col) inlet cell coordinates
    """
    inlets = {}

    # D8 neighbor offsets (in row, col format)
    neighbors = [
        (-1, 0), (-1, 1),  # N, NE
        (0, 1), (1, 1),    # E, SE
        (1, 0), (1, -1),   # S, SW
        (0, -1), (-1, -1)  # W, NW
    ]

    # Find unique lake IDs
    lake_ids = np.unique(lake_mask[lake_mask > 0])

    for lake_id in lake_ids:
        inlet_cells = []
        lake_cells = np.where(lake_mask == lake_id)

        if len(lake_cells[0]) == 0:
            continue

        # Check each lake cell for adjacent non-lake cells with lower elevation
        for row, col in zip(lake_cells[0], lake_cells[1]):
            lake_elev = dem[row, col]

            # Check all 8 neighbors
            for drow, dcol in neighbors:
                nrow, ncol = row + drow, col + dcol

                # Check bounds
                if 0 <= nrow < dem.shape[0] and 0 <= ncol < dem.shape[1]:
                    # Skip if it's another lake cell
                    if lake_mask[nrow, ncol] > 0:
                        continue

                    # Skip if it's the outlet
                    if outlet_mask is not None and outlet_mask[row, col]:
                        continue

                    neighbor_elev = dem[nrow, ncol]

                    # This is an inlet if neighbor is lower (water flows into lake)
                    # or equal elevation (neutral flow boundary)
                    if neighbor_elev <= lake_elev + 0.1:  # Small tolerance for numerical precision
                        inlet_cells.append((row, col))
                        break  # One inlet per lake cell is enough

        if inlet_cells:
            inlets[lake_id] = inlet_cells

    return inlets
