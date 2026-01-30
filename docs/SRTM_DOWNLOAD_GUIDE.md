# SRTM Data Download Guide

The automated SRTM downloader is a work in progress. Here are current options for getting SRTM elevation data:

## Option 1: Manual Download (Recommended for now)

### OpenTopography (Easiest, no account needed)

1. Go to https://portal.opentopography.org/raster?opentopoID=OTSRTM.082015.4326.1
2. Click "Select a Region"
3. Draw your bounding box on the map
4. Click "Submit"
5. Download the GeoTIFF file
6. Use with terrain-maker:

```python
from src.terrain.data_loading import load_dem_files

# Load your downloaded GeoTIFF
dem, transform = load_dem_files("path/to/downloaded_data", pattern="*.tif")
```

### NASA Earthdata (More options, requires free account)

1. Create free account: https://urs.earthdata.nasa.gov/users/new
2. Go to: https://search.earthdata.nasa.gov/
3. Search for "SRTM"
4. Select your area of interest
5. Download tiles
6. Extract `.hgt` files from ZIP
7. Use with terrain-maker:

```python
from src.terrain.data_loading import load_dem_files

# Load HGT files
dem, transform = load_dem_files("data/srtm_tiles", pattern="*.hgt")
```

## Option 2: Use Our Downloader (Work in Progress)

The `dem_downloader.py` module provides:
- ✅ Bounding box calculation
- ✅ Interactive map visualization
- ✅ SRTM tile naming
- ⚠️  Downloading (authentication issues with NASA Earthdata)

Current status: NASA Earthdata changed their authentication system. We're working on updating the downloader to use the new API.

## Option 3: Command Line (Advanced)

If you're comfortable with command line tools:

```bash
# Using wget with NASA credentials
wget --user=USERNAME --password=PASSWORD \
  https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/N32W117.SRTMGL1.hgt.zip

# Extract
unzip N32W117.SRTMGL1.hgt.zip
```

## San Diego Example

For San Diego County, you need these tiles:
- N32W118, N32W117, N32W116 (southern row)
- N33W118, N33W117, N33W116 (northern row)

Download from OpenTopography or NASA Earthdata, then:

```python
from src.terrain.core import Terrain
from src.terrain.data_loading import load_dem_files

# Load the downloaded tiles
dem, transform = load_dem_files("data/san_diego_srtm")

# Create terrain
terrain = Terrain(dem, transform)
```

## We're Working On It!

The automated downloader will be updated soon. For now, manual download is reliable and takes just a few minutes.
