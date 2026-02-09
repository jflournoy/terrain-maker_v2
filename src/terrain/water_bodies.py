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
from typing import Dict, Tuple, Optional, Any, List
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
    south, west, north, east = bbox

    # Calculate grid dimensions
    n_rows = int(np.ceil((north - south) / resolution))
    n_cols = int(np.ceil((east - west) / resolution))

    # Create Affine transform (top-left origin, Y decreasing)
    transform = Affine(resolution, 0, west, 0, -resolution, north)

    # Initialize mask
    mask = np.zeros((n_rows, n_cols), dtype=np.uint16)

    # Rasterize each lake polygon
    features = lakes_geojson.get("features", [])

    for idx, feature in enumerate(features, start=1):
        geometry = feature.get("geometry", {})
        if geometry.get("type") != "Polygon":
            continue

        coords = geometry.get("coordinates", [[]])
        if not coords or not coords[0]:
            continue

        # Get polygon exterior ring
        ring = coords[0]

        # Rasterize polygon using scanline algorithm
        _rasterize_polygon(mask, ring, idx, transform)

    return mask, transform


def _rasterize_polygon(
    mask: np.ndarray,
    ring: List[List[float]],
    label: int,
    transform: Affine,
) -> None:
    """
    Rasterize a single polygon to the mask using scanline algorithm.

    Parameters
    ----------
    mask : np.ndarray
        Output mask (modified in-place)
    ring : list
        List of [lon, lat] coordinates forming polygon exterior
    label : int
        Label value to assign to polygon cells
    transform : Affine
        Geographic transform
    """
    rows, cols = mask.shape
    inv_transform = ~transform

    # Convert polygon vertices to pixel coordinates
    pixel_coords = []
    for lon, lat in ring:
        col, row = inv_transform * (lon, lat)
        pixel_coords.append((int(row), int(col)))

    if len(pixel_coords) < 3:
        return

    # Get bounding box of polygon in pixel coords
    min_row = max(0, min(p[0] for p in pixel_coords))
    max_row = min(rows - 1, max(p[0] for p in pixel_coords))
    min_col = max(0, min(p[1] for p in pixel_coords))
    max_col = min(cols - 1, max(p[1] for p in pixel_coords))

    # For each row, find intersections with polygon edges
    for row in range(min_row, max_row + 1):
        intersections = []

        n = len(pixel_coords)
        for i in range(n):
            p1 = pixel_coords[i]
            p2 = pixel_coords[(i + 1) % n]

            # Check if edge crosses this row
            if (p1[0] <= row < p2[0]) or (p2[0] <= row < p1[0]):
                if p1[0] != p2[0]:
                    # Calculate x intersection
                    col = p1[1] + (row - p1[0]) * (p2[1] - p1[1]) / (p2[0] - p1[0])
                    intersections.append(col)

        # Sort intersections and fill between pairs
        intersections.sort()
        for i in range(0, len(intersections) - 1, 2):
            start_col = max(min_col, int(intersections[i]))
            end_col = min(max_col, int(intersections[i + 1]))
            mask[row, start_col:end_col + 1] = label


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
    hydrolakes_path = Path(output_dir) / "HydroLAKES_polys_v10.shp"

    if not hydrolakes_path.exists():
        # Return empty with instructions
        print(f"HydroLAKES shapefile not found at {hydrolakes_path}")
        print("Download from: https://www.hydrosheds.org/products/hydrolakes")
        print("Extract HydroLAKES_polys_v10.shp to the output directory")
        return {"type": "FeatureCollection", "features": []}

    # Filter shapefile to bbox using geopandas
    try:
        import geopandas as gpd
        from shapely.geometry import box

        gdf = gpd.read_file(hydrolakes_path, bbox=bbox)

        # Convert to GeoJSON
        geojson = json.loads(gdf.to_json())

        # Add pour_point from HydroLAKES attributes
        for feature in geojson.get("features", []):
            props = feature.get("properties", {})
            if "Pour_lat" in props and "Pour_long" in props:
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
