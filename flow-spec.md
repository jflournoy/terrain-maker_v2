# Flow Accumulation Feature Specification
## For Elevation Map Blender Render Library

### Overview
Add hydrological flow accumulation analysis to generate two key outputs from DEM data:
1. **Local annual rainfall** - precipitation at each pixel
2. **Upslope contributing rainfall** - total water from upstream drainage area

### Inputs

#### Required
- `dem_path` (str): Path to DEM raster file (GeoTIFF, any standard format)
- `precipitation_path` (str): Path to annual precipitation raster (mm/year)
  - Must have same extent/resolution as DEM, or be resampled to match

#### Optional Parameters
- `output_dir` (str): Directory for output files. Default: same as input DEM
- `flow_algorithm` (str): Flow routing method. Default: 'd8'
  - Options: 'd8', 'dinf' (D-infinity for multi-directional flow)
- `fill_method` (str): Depression filling approach. Default: 'breach'
  - Options: 'breach', 'fill'
  - 'breach' is recommended for realistic flow paths
- `cell_size` (float): Override DEM resolution in meters. Default: extract from raster metadata

### Core Processing Steps

#### 1. Preprocessing
```
Load DEM and precipitation rasters
Verify spatial alignment (extent, resolution, CRS)
Resample precipitation to match DEM if needed
```

#### 2. Hydrological Conditioning
```
Fill pits (small single-cell depressions)
Breach or fill depressions (larger flat areas)
Resolve flats (assign flow direction in flat areas)
```

#### 3. Flow Direction Calculation
```
Apply D8 algorithm:
  - For each cell, determine steepest downslope neighbor
  - Encode flow direction (typically 1,2,4,8,16,32,64,128 for 8 neighbors)
```

#### 4. Flow Accumulation
```
Unweighted accumulation:
  - Count number of upstream cells draining through each pixel
  - Units: cells or area (cells × cell_size²)

Weighted accumulation (precipitation-weighted):
  - Sum precipitation values from all upstream cells
  - Formula: Σ(precipitation_i × cell_area) for all upstream cells i
  - Units: mm·m² (millimeters × square meters)
  - Equivalent to cubic meters of water per year when multiplied by cell_size²/1000
```

### Outputs

#### Required Output Files
All outputs should be saved as GeoTIFF with same CRS/extent as input DEM:

1. **`flow_direction.tif`**
   - Flow direction raster (D8 encoding)
   - Data type: uint8
   - Values: 1,2,4,8,16,32,64,128 or nodata

2. **`flow_accumulation_area.tif`**
   - Unweighted drainage area
   - Data type: float32
   - Units: cells or square meters (specify in metadata)
   
3. **`flow_accumulation_rainfall.tif`** 
   - Precipitation-weighted accumulation
   - Data type: float32
   - Units: mm·m² (document in metadata)
   - This is the key output: total upstream rainfall

4. **`dem_conditioned.tif`**
   - Depression-filled/breached DEM
   - Data type: float32
   - Units: meters

#### Optional Outputs
5. **`stream_network.tif`** or **`stream_network.shp`**
   - Boolean raster or vector polylines
   - Threshold: cells where flow_accumulation_area > threshold
   - Recommended threshold: 0.5-1.0 km² drainage area

### Return Values (Python API)

```python
{
    'flow_direction': numpy.ndarray or rasterio object,
    'drainage_area': numpy.ndarray,  # cells draining to each pixel
    'upstream_rainfall': numpy.ndarray,  # mm·m² of total rainfall
    'conditioned_dem': numpy.ndarray,
    'metadata': {
        'cell_size_m': float,
        'drainage_area_units': 'cells' or 'square_meters',
        'total_area_km2': float,
        'max_drainage_area_km2': float,
        'max_upstream_rainfall_m3': float,  # converted from mm·m²
        'algorithm': str,
        'fill_method': str
    },
    'files': {
        'flow_direction': 'path/to/flow_direction.tif',
        'drainage_area': 'path/to/flow_accumulation_area.tif',
        'upstream_rainfall': 'path/to/flow_accumulation_rainfall.tif',
        'conditioned_dem': 'path/to/dem_conditioned.tif'
    }
}
```

### Example Usage

```python
from elevation_renderer import flow_accumulation

# Basic usage
results = flow_accumulation(
    dem_path='san_diego_dem.tif',
    precipitation_path='prism_annual_precip.tif'
)

# Access arrays
upstream_rainfall = results['upstream_rainfall']  # mm·m² at each pixel

# Convert to annual water volume in cubic meters
cell_area = results['metadata']['cell_size_m'] ** 2
annual_water_m3 = upstream_rainfall * cell_area / 1000

# Advanced usage with options
results = flow_accumulation(
    dem_path='san_diego_dem.tif',
    precipitation_path='prism_annual_precip.tif',
    output_dir='./hydro_outputs',
    flow_algorithm='dinf',  # Multi-directional flow
    fill_method='breach',
    generate_streams=True,
    stream_threshold_km2=1.0
)
```

### Implementation Recommendations

#### Library Choice
Use **pysheds** or **whitebox** for Python implementation:

**pysheds** (recommended for Python-native workflow):
```python
from pysheds.grid import Grid

grid = Grid.from_raster(dem_path)
dem = grid.read_raster(dem_path)

# Conditioning
pit_filled = grid.fill_pits(dem)
flooded = grid.fill_depressions(pit_filled)
inflated = grid.resolve_flats(flooded)

# Flow direction
fdir = grid.flowdir(inflated, routing='d8')

# Accumulation
acc_area = grid.accumulation(fdir, routing='d8')

# Weighted accumulation
precip = grid.read_raster(precip_path)
acc_rainfall = grid.accumulation(fdir, weights=precip, routing='d8')
```

**whitebox** (faster for large datasets, requires binary download):
```python
import whitebox
wbt = whitebox.WhiteboxTools()

wbt.breach_depressions_least_cost(dem=dem_path, output='dem_breached.tif')
wbt.d8_pointer(dem='dem_breached.tif', output='flow_dir.tif')
wbt.d8_flow_accumulation(
    input='dem_breached.tif',
    output='flow_accum_area.tif'
)
wbt.d8_flow_accumulation(
    input='dem_breached.tif',
    output='flow_accum_rainfall.tif',
    weight=precip_path
)
```

#### Performance Considerations
- For DEM > 10,000 × 10,000 pixels: use tiled processing or whitebox
- For interactive/web applications: pre-compute and cache results
- Memory usage: ~5× input DEM size (storing intermediate arrays)

#### Validation
Outputs should satisfy:
1. All flow directions point downhill (except pits/outlets)
2. Flow accumulation = 1 at ridgelines/peaks
3. Flow accumulation increases monotonically downstream
4. Weighted accumulation ≥ local precipitation × cell area
5. Mass balance: Σ(precipitation × area) = Σ(outlet flows)

### Edge Cases and Error Handling

#### Input Validation
```python
# Check spatial alignment
assert dem.crs == precip.crs, "CRS mismatch"
assert dem.bounds == precip.bounds, "Extent mismatch"
if dem.shape != precip.shape:
    # Resample precipitation to match DEM
    precip_resampled = resample(precip, dem.shape, method='bilinear')
```

#### NoData Handling
- Preserve nodata pixels throughout pipeline
- Flow cannot cross nodata boundaries
- Set accumulation to nodata at nodata pixels

#### Flat Areas
- Resolve with epsilon gradients or direction rules
- Document approach in metadata

#### Sinks/Depressions
- Log count and location of filled/breached depressions
- Optionally output depression map for quality control

### Integration with Blender

#### Potential Blender Uses
1. **Geometry displacement**: Use flow accumulation to modulate terrain roughness
2. **Texture generation**: Map upstream rainfall to shader nodes
3. **Particle systems**: Emit water particles along high-accumulation paths
4. **Vegetation distribution**: Dense vegetation in high-accumulation areas
5. **Animation**: Simulate water flow along drainage network

#### Blender-Specific Output Format
Optionally export as:
- **OpenEXR** with multiple channels (elevation, flow_dir, accumulation_area, upstream_rainfall)
- **Normalized textures** (0-1 range) for direct shader input
- **32-bit float** for maximum precision

```python
# Example: Export for Blender
def export_for_blender(results, output_path):
    """
    Export multi-channel EXR for Blender displacement/shading
    """
    import OpenEXR
    import Imath
    
    # Normalize arrays to 0-1 for shader use
    rainfall_norm = normalize(results['upstream_rainfall'])
    area_norm = normalize(results['drainage_area'])
    
    # Create multi-channel EXR
    header = OpenEXR.Header(width, height)
    header['channels'] = {
        'elevation': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        'drainage_area': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
        'upstream_rainfall': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    }
    # ... write EXR
```

### Testing

#### Unit Tests
- Test with synthetic DEMs (plane, cone, valley)
- Verify flow accumulation = analytical solutions
- Test precipitation weighting with uniform rainfall

#### Integration Tests  
- Process real-world DEM (e.g., SRTM tile)
- Compare against reference implementation
- Validate outputs against known watersheds

#### Example Test Case
```python
def test_uniform_rainfall():
    """Flow accumulation with uniform rain should equal area × rainfall"""
    dem = create_synthetic_valley()  # 100×100 pixels, 30m resolution
    precip = np.full_like(dem, 500.0)  # 500 mm/year everywhere
    
    results = flow_accumulation(dem, precip)
    
    # At outlet, weighted accumulation should equal total basin rainfall
    outlet_idx = find_outlet(results['flow_direction'])
    drainage_area_cells = results['drainage_area'][outlet_idx]
    upstream_rainfall = results['upstream_rainfall'][outlet_idx]
    
    expected = drainage_area_cells * 500 * (30**2)  # cells × mm/year × m²/cell
    assert np.isclose(upstream_rainfall, expected, rtol=0.01)
```

### Documentation Requirements

#### User-Facing Docs
- Explain difference between drainage area and upstream rainfall
- Provide interpretation guide (what do values mean?)
- Include visualization examples (linear vs log scale)
- Real-world use cases (watershed delineation, flood modeling, ecology)

#### API Documentation
- Docstrings for all functions
- Type hints for all parameters
- Example code for common workflows

#### Metadata
Each output GeoTIFF should include:
```
TIFFTAG_IMAGEDESCRIPTION: "Precipitation-weighted flow accumulation"
TIFFTAG_SOFTWARE: "elevation_renderer v1.0"
Custom tags:
  - ALGORITHM: "d8"
  - FILL_METHOD: "breach"
  - UNITS: "mm·m²" or "cells"
  - CREATED: ISO timestamp
  - SOURCE_DEM: input filename
  - SOURCE_PRECIP: input filename
```

### Future Enhancements

1. **Multi-directional flow** (D-infinity, MFD algorithms)
2. **Kinematic routing** (time-of-concentration, hydrograph generation)
3. **Evapotranspiration** (reduce effective precipitation by ET rates)
4. **Soil infiltration** (account for permeability, saturation)
5. **Snowmelt modeling** (elevation-dependent snow accumulation/melt)
6. **Stream power index** (slope × upstream area for erosion modeling)
7. **Topographic wetness index** (ln(area / tan(slope)))
8. **Interactive visualization** (WebGL-based flow path tracing)

---

## Example Output Interpretation

For a pixel at coordinates (r, c):
- `dem[r,c]` = 450m elevation
- `precipitation[r,c]` = 600 mm/year local rainfall
- `drainage_area[r,c]` = 2500 cells upstream
- `upstream_rainfall[r,c]` = 1,500,000 mm·m²

**Interpretation:**
- This pixel is at 450m elevation
- It receives 600mm of rain locally per year
- 2,500 cells (2500 × 900m² = 2.25 km²) drain through this point
- The total water from upstream is 1.5M mm·m² = 1,350 m³/year
- This is equivalent to a stream flow of ~0.043 L/s average annual flow
- In a storm, this represents the contributing area for peak discharge calculations

This information is critical for:
- Flood risk assessment (how much water can accumulate?)
- Riparian habitat modeling (perennial vs ephemeral streams)
- Infrastructure planning (culvert sizing, road crossings)
- Erosion potential (where does water concentrate?)