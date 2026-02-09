# Using Real Water Body Data

This guide shows how to use **real water body data** from NHD (USA) or HydroLAKES (global) instead of synthetic lakes.

## Quick Start

```bash
# Use NHD data (USA only - very detailed)
python examples/validate_flow_with_nhd_data.py --data-source nhd

# Use HydroLAKES data (global coverage)
python examples/validate_flow_with_nhd_data.py --data-source hydrolakes

# Adjust minimum lake size
python examples/validate_flow_with_nhd_data.py --data-source nhd --min-area-km2 0.1
```

## Data Sources

### 1. NHD (National Hydrography Dataset) - USA Only

**Pros:**
- Very detailed and accurate
- Downloaded automatically via REST API
- Includes flowlines to identify outlets
- Updated regularly

**Cons:**
- USA only
- Requires internet connection
- API can be slow for large areas

**Usage:**
```python
from src.terrain.water_bodies import download_water_bodies

geojson_path = download_water_bodies(
    bbox=(south, west, north, east),  # WGS84 degrees
    output_dir="data/water_bodies",
    data_source="nhd",
    min_area_km2=0.01,  # Filter small ponds
)
```

### 2. HydroLAKES - Global Coverage

**Pros:**
- Global coverage
- Includes pour points (outlets) in attributes
- High-quality lake boundaries
- Single download covers entire world

**Cons:**
- Requires manual download first (374 MB)
- Less detailed than NHD in USA
- No automatic updates

**Setup:**
1. Download HydroLAKES from: https://www.hydrosheds.org/products/hydrolakes
2. Extract `HydroLAKES_polys_v10.shp` to `data/water_bodies/`
3. Install geopandas: `uv pip install geopandas`

**Usage:**
```python
from src.terrain.water_bodies import download_water_bodies

geojson_path = download_water_bodies(
    bbox=(south, west, north, east),
    output_dir="data/water_bodies",  # Must contain HydroLAKES_polys_v10.shp
    data_source="hydrolakes",
    min_area_km2=0.1,
)
```

## Example: San Diego Flow with NHD Data

```bash
# Run with NHD data (automatically downloads for San Diego region)
python examples/validate_flow_with_nhd_data.py \
  --data-source nhd \
  --bigness medium \
  --min-area-km2 0.05 \
  --output examples/output/flow_nhd_san_diego/
```

**What this does:**
1. Loads San Diego DEM (medium 500×500 subset)
2. Downloads NHD water bodies via REST API
3. Filters lakes >= 0.05 km² (5 hectares)
4. Rasterizes lake polygons to match DEM
5. Identifies lake outlets from NHD flowlines
6. Computes flow accumulation with lake routing
7. Generates 11 validation plots

**Expected output:**
- Real reservoir and lake shapes (irregular, not synthetic squares)
- Outlets at actual dam/spillway locations
- Proper flow convergence toward outlets

## Installation Requirements

### For NHD (automatic download):
```bash
# Already included in base dependencies
pip install requests rasterio numpy
```

### For HydroLAKES (requires geopandas):
```bash
# Install geopandas for shapefile reading
uv pip install geopandas

# Or add to pyproject.toml:
# [project.optional-dependencies]
# hydrolakes = ["geopandas>=0.12.0"]
```

## API Reference

### `download_water_bodies()`

```python
def download_water_bodies(
    bbox: Tuple[float, float, float, float],
    output_dir: str,
    data_source: str = "nhd",
    min_area_km2: float = 0.1,
    force_download: bool = False,
) -> Path
```

**Parameters:**
- `bbox`: Bounding box as (south, west, north, east) in WGS84 degrees
- `output_dir`: Directory for cached GeoJSON files
- `data_source`: "nhd" or "hydrolakes"
- `min_area_km2`: Minimum lake area in km²
- `force_download`: If True, re-download even if cached

**Returns:**
- Path to GeoJSON file with lake polygons and outlets

**Caching:**
- Creates cache files like `water_bodies_nhd_<hash>.geojson`
- Subsequent calls with same bbox use cached data
- Set `force_download=True` to refresh

### `rasterize_lakes_to_mask()`

```python
def rasterize_lakes_to_mask(
    lakes_geojson: Dict[str, Any],
    bbox: Tuple[float, float, float, float],
    resolution: float = 0.001,
) -> Tuple[np.ndarray, Affine]
```

**Parameters:**
- `lakes_geojson`: GeoJSON FeatureCollection with lake polygons
- `bbox`: Bounding box (south, west, north, east)
- `resolution`: Cell size in degrees (default: 0.001 ≈ 100m)

**Returns:**
- `mask`: Labeled mask (uint16) where 0 = no lake, N = lake ID
- `transform`: Affine transform (pixel to coordinate)

### `identify_outlet_cells()`

```python
def identify_outlet_cells(
    lake_mask: np.ndarray,
    outlets: Dict[int, Tuple[float, float]],
    transform: Affine,
) -> np.ndarray
```

**Parameters:**
- `lake_mask`: Labeled lake mask from rasterize_lakes_to_mask()
- `outlets`: Dict mapping lake_id -> (lon, lat) outlet coordinates
- `transform`: Geographic transform

**Returns:**
- Boolean mask where True = outlet cell

## Complete Integration Example

```python
from pathlib import Path
import rasterio
from src.terrain.water_bodies import (
    download_water_bodies,
    rasterize_lakes_to_mask,
    identify_outlet_cells,
)
from src.terrain.flow_accumulation import flow_accumulation

# 1. Load DEM
with rasterio.open("data/dem.tif") as src:
    dem = src.read(1)
    dem_transform = src.transform

    # Calculate bounding box
    rows, cols = dem.shape
    west, north = dem_transform * (0, 0)
    east, south = dem_transform * (cols, rows)
    bbox = (south, west, north, east)

# 2. Download water bodies
geojson_path = download_water_bodies(
    bbox=bbox,
    output_dir="data/water_bodies",
    data_source="nhd",
    min_area_km2=0.1,
)

# 3. Load GeoJSON
import json
with open(geojson_path) as f:
    lakes_geojson = json.load(f)

# 4. Extract outlets
outlets_dict = {}
for idx, feature in enumerate(lakes_geojson["features"], start=1):
    outlet = feature["properties"].get("outlet")
    if outlet:
        outlets_dict[idx] = tuple(outlet)

# 5. Rasterize lakes
resolution = abs(dem_transform.a)
lake_mask, lake_transform = rasterize_lakes_to_mask(
    lakes_geojson, bbox, resolution=resolution
)

# 6. Identify outlet cells
lake_outlets = identify_outlet_cells(lake_mask, outlets_dict, lake_transform)

# 7. Run flow accumulation with lakes
results = flow_accumulation(
    dem_path="data/dem.tif",
    precipitation_path="data/precip.tif",
    lake_mask=lake_mask,
    lake_outlets=lake_outlets,
    backend="custom",
)

# 8. Get results
flow_dir = results["flow_direction"]
drainage_area = results["drainage_area"]
upstream_rainfall = results["upstream_rainfall"]
```

## Troubleshooting

### NHD API Errors

**Error:** `requests.exceptions.HTTPError: 500 Server Error`

**Solution:** NHD API can be unreliable. Try:
```python
# Add retry logic
import time
for attempt in range(3):
    try:
        geojson_path = download_water_bodies(bbox, "data", "nhd")
        break
    except Exception as e:
        if attempt < 2:
            print(f"Retry {attempt+1}/3...")
            time.sleep(5)
        else:
            raise
```

### HydroLAKES Not Found

**Error:** `HydroLAKES shapefile not found`

**Solution:**
1. Download from https://www.hydrosheds.org/products/hydrolakes
2. Extract all files (not just .shp - need .shx, .dbf, .prj too)
3. Place in `output_dir/HydroLAKES_polys_v10.shp`

### No Outlets Found

**Error:** `Outlet cells: 0`

**Possible causes:**
1. **Out of bounds:** Outlets outside DEM extent
2. **Wrong projection:** Outlet coordinates in wrong CRS
3. **Missing data:** Lake features don't have outlet attributes

**Solution:**
```python
# Check outlet coordinates
for lake_id, (lon, lat) in outlets_dict.items():
    print(f"Lake {lake_id}: outlet at ({lon:.4f}, {lat:.4f})")

    # Check if in bounds
    west, north = dem_transform * (0, 0)
    east, south = dem_transform * (dem.shape[1], dem.shape[0])

    in_bounds = west <= lon <= east and south <= lat <= north
    print(f"  In bounds: {in_bounds}")
```

### Shape Mismatch

**Error:** `ValueError: operands could not be broadcast together`

**Solution:** The script automatically resamples lake masks to match DEM shape:
```python
if lake_mask.shape != dem.shape:
    from scipy.ndimage import zoom
    scale_y = dem.shape[0] / lake_mask.shape[0]
    scale_x = dem.shape[1] / lake_mask.shape[1]
    lake_mask = zoom(lake_mask, (scale_y, scale_x), order=0)
```

## Performance Tips

### Large Areas

For large regions (> 1000 km²):

1. **Increase min_area_km2** to filter small lakes:
```bash
python examples/validate_flow_with_nhd_data.py \
  --data-source nhd \
  --min-area-km2 1.0  # Only lakes >= 1 km²
```

2. **Downsample DEM** before flow computation:
```bash
python examples/validate_flow_with_nhd_data.py \
  --bigness full \
  --target-size 1000  # Downsample to 1000x1000
```

3. **Use coarser resolution** for rasterization:
```python
# Match DEM resolution instead of oversampling
resolution = abs(dem_transform.a) * 2  # 2x DEM pixel size
lake_mask, _ = rasterize_lakes_to_mask(geojson, bbox, resolution)
```

### Caching

Water body downloads are automatically cached:
```
data/water_bodies/
├── water_bodies_nhd_a1b2c3d4.geojson      # Cached NHD data
├── water_bodies_hydrolakes_e5f6g7h8.geojson  # Cached HydroLAKES
└── HydroLAKES_polys_v10.shp               # Original shapefile
```

Cache keys are based on data_source + bbox hash, so:
- Same region = uses cache
- Different region = new download
- Force refresh with `force_download=True`

## Comparison: Synthetic vs Real Data

| Feature | Synthetic Lakes | NHD (USA) | HydroLAKES (Global) |
|---------|----------------|-----------|---------------------|
| **Coverage** | Any region | USA only | Global |
| **Setup** | None | Auto-download | Manual download |
| **Accuracy** | Approximation | Very accurate | Accurate |
| **Lake shapes** | Irregular (from DEM) | Surveyed polygons | Surveyed polygons |
| **Outlets** | Lowest boundary point | From flowlines | From attributes |
| **Dependencies** | None | requests | geopandas |
| **Speed** | Fast | Slow (API) | Fast (cached) |
| **Use case** | Testing, development | Production (USA) | Production (global) |

## Next Steps

1. **Run with real data:** Try [validate_flow_with_nhd_data.py](validate_flow_with_nhd_data.py)
2. **Integrate into your pipeline:** Use the integration example above
3. **Contribute improvements:** Submit PRs for better outlet detection
4. **Add new data sources:** Extend water_bodies.py with regional datasets

## References

- **NHD API docs:** https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer
- **HydroLAKES:** https://www.hydrosheds.org/products/hydrolakes
- **Source code:** [src/terrain/water_bodies.py](../src/terrain/water_bodies.py)
- **Validation script:** [validate_flow_with_nhd_data.py](validate_flow_with_nhd_data.py)
