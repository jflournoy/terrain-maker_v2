"""
Precipitation data downloader module.

Downloads and manages precipitation data from various sources (PRISM, WorldClim, etc.)
for hydrological analysis and terrain visualization.

Supports:
- PRISM (Parameter-elevation Regressions on Independent Slopes Model)
- WorldClim (global climate data)
- CHELSA (high-resolution climate data)
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import rasterio
from rasterio import Affine
from rasterio.transform import from_bounds
import requests
import zipfile
import io


# Dataset metadata
DATASETS = {
    "prism": {
        "name": "PRISM Climate Group",
        "resolution": "4km (~30 arc-seconds)",
        "temporal_coverage": "1981-2010 normals",
        "url": "https://prism.oregonstate.edu",
        "description": "High-quality spatial climate data for the US",
        "coverage": "Continental USA only",
    },
    "worldclim": {
        "name": "WorldClim",
        "resolution": "4.5km (2.5 arc-minutes)",
        "temporal_coverage": "1970-2000",
        "url": "https://www.worldclim.org",
        "description": "Global climate data (covers USA, Mexico, and worldwide)",
        "coverage": "Global",
    },
    "chelsa": {
        "name": "CHELSA",
        "resolution": "1km (30 arc-seconds)",
        "temporal_coverage": "1979-2013",
        "url": "https://chelsa-climate.org",
        "description": "High-resolution climate data for mountainous regions",
    },
}


def download_precipitation(
    bbox: Tuple[float, float, float, float],
    output_dir: str,
    dataset: str = "prism",
    force_download: bool = False,
    use_real_data: bool = True,
) -> Path:
    """
    Download precipitation data for bounding box.

    Parameters
    ----------
    bbox : tuple
        Bounding box (min_lat, min_lon, max_lat, max_lon) in WGS84
    output_dir : str
        Directory to save downloaded data
    dataset : str, default 'prism'
        Dataset to download ('prism', 'worldclim', 'chelsa')
    force_download : bool, default False
        Force re-download even if cached
    use_real_data : bool, default True
        If True, download real data from servers. If False, generate synthetic data for testing.

    Returns
    -------
    Path
        Path to downloaded precipitation GeoTIFF

    Raises
    ------
    ValueError
        If bbox is invalid or dataset not supported
    """
    # Validate bbox
    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
        raise ValueError("bbox must be a tuple/list with 4 values (min_lat, min_lon, max_lat, max_lon)")

    min_lat, min_lon, max_lat, max_lon = bbox

    # Validate bbox order
    if min_lat >= max_lat or min_lon >= max_lon:
        raise ValueError(
            f"Invalid bbox: coordinates must be (min_lat, min_lon, max_lat, max_lon). "
            f"Got ({min_lat}, {min_lon}, {max_lat}, {max_lon})"
        )

    # Validate dataset
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(DATASETS.keys())}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate output filename
    bbox_str = f"{min_lat}_{min_lon}_{max_lat}_{max_lon}".replace(".", "p").replace("-", "m")
    output_file = output_path / f"{dataset}_annual_precip_{bbox_str}.tif"

    # Check cache
    if output_file.exists() and not force_download:
        print(f"Using cached precipitation data: {output_file}")
        return output_file

    # Download data (dataset-specific)
    if dataset == "prism":
        data, transform = get_prism_annual_precip(bbox, output_dir=str(output_path), use_real_data=use_real_data)
    elif dataset == "worldclim":
        if use_real_data:
            data, transform = download_real_worldclim_annual(bbox, output_dir=str(output_path))
        else:
            data, transform = _create_synthetic_precipitation(bbox)
    elif dataset == "chelsa":
        # Stub for CHELSA (would implement actual download)
        data, transform = _create_synthetic_precipitation(bbox)
    else:
        raise ValueError(f"Dataset {dataset} not yet implemented")

    # Save as GeoTIFF
    _write_precipitation_geotiff(output_file, data, transform, crs="EPSG:4326")

    print(f"✓ Downloaded precipitation data to {output_file}")
    return output_file


def download_real_worldclim_annual(
    bbox: Tuple[float, float, float, float], output_dir: str
) -> Tuple[np.ndarray, Affine]:
    """
    Download real WorldClim annual precipitation data.

    WorldClim provides global climate data at ~4.5km resolution (2.5 arc-minutes).
    This function downloads BIO12 (annual precipitation) for the specified bbox.

    Uses 2.5m resolution instead of 30s for reasonable download size (~10MB vs 9.7GB).

    Parameters
    ----------
    bbox : tuple
        Bounding box (min_lat, min_lon, max_lat, max_lon)
    output_dir : str
        Output directory

    Returns
    -------
    np.ndarray
        Precipitation data (mm/year)
    Affine
        Geographic transform

    Raises
    ------
    Exception
        If download fails or network is unavailable
    """
    min_lat, min_lon, max_lat, max_lon = bbox

    # WorldClim 2.1 BIO12 (annual precipitation) at 2.5 minutes (~4.5km)
    # Using 2.5m resolution (comparable to PRISM's 4km) - much smaller than 30s!
    # File size: ~5-10MB vs 9.7GB for 30s resolution
    base_url = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_2.5m_bio.zip"

    print(f"Downloading WorldClim annual precipitation from {base_url}...")
    print(f"  (2.5 minute resolution, ~10MB, should take <30 seconds)")

    try:
        # Download the ZIP file
        response = requests.get(base_url, timeout=300)  # 5 min timeout for large file
        response.raise_for_status()

        # Extract ZIP contents
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            # Find BIO12 (annual precipitation) TIF file
            bio12_file = None
            for name in zf.namelist():
                if 'bio_12' in name.lower() and name.endswith('.tif'):
                    bio12_file = name
                    break

            if not bio12_file:
                raise ValueError(f"Could not find BIO12 (annual precip) in ZIP. Files: {zf.namelist()}")

            # Extract to temp directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            tif_path = output_path / bio12_file

            with open(tif_path, 'wb') as f:
                f.write(zf.read(bio12_file))

            print(f"✓ Downloaded WorldClim BIO12 to {tif_path}")

            # Read the TIF file using rasterio
            with rasterio.open(tif_path) as src:
                # Read full raster
                full_data = src.read(1).astype(np.float32)
                full_transform = src.transform
                full_crs = src.crs

                # Calculate pixel coordinates for bbox
                from rasterio.transform import rowcol

                # Get row/col for bbox corners
                min_row, min_col = rowcol(full_transform, min_lon, max_lat)  # Upper-left
                max_row, max_col = rowcol(full_transform, max_lon, min_lat)  # Lower-right

                # Clamp to raster bounds
                min_row = max(0, min_row)
                min_col = max(0, min_col)
                max_row = min(full_data.shape[0], max_row)
                max_col = min(full_data.shape[1], max_col)

                # Extract subset
                subset_data = full_data[min_row:max_row, min_col:max_col].copy()

                # Create transform for subset
                subset_transform = full_transform * Affine.translation(min_col, min_row)

                print(f"✓ Extracted subset: {subset_data.shape} pixels")
                print(f"  Precipitation range: {subset_data.min():.1f} to {subset_data.max():.1f} mm/year")

                return subset_data, subset_transform

    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error downloading WorldClim data: {e}") from e
    except Exception as e:
        raise Exception(f"Failed to download WorldClim data: {e}") from e


def download_real_prism_annual(
    bbox: Tuple[float, float, float, float], output_dir: str
) -> Tuple[np.ndarray, Affine]:
    """
    Download real PRISM annual precipitation data from FTP server.

    IMPORTANT: PRISM only covers continental USA. For areas outside USA
    (e.g., Mexico, Canada), use WorldClim instead via download_real_worldclim_annual().

    Parameters
    ----------
    bbox : tuple
        Bounding box (min_lat, min_lon, max_lat, max_lon)
    output_dir : str
        Output directory

    Returns
    -------
    np.ndarray
        Precipitation data (mm/year)
    Affine
        Geographic transform

    Raises
    ------
    Exception
        If download fails or network is unavailable
    """
    min_lat, min_lon, max_lat, max_lon = bbox

    # PRISM annual normals 1991-2020 (4km resolution)
    # New URL structure as of 2025: https://ftp.prism.oregonstate.edu/normals/us/4km/ppt/monthly/
    base_url = "https://ftp.prism.oregonstate.edu/normals/us/4km/ppt/monthly/prism_ppt_us_25m_2020_avg_30y.zip"

    print(f"Downloading PRISM annual precipitation from {base_url}...")

    try:
        # Download the ZIP file
        response = requests.get(base_url, timeout=60)
        response.raise_for_status()

        # Extract ZIP contents
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            # Find the TIF file (PRISM now uses GeoTIFF format)
            tif_file = None
            for name in zf.namelist():
                if name.endswith('.tif') and not name.endswith('.aux.xml'):
                    tif_file = name
                    break

            if not tif_file:
                raise ValueError(f"Could not find TIF file in ZIP. Files: {zf.namelist()}")

            # Extract TIF file to temp directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            tif_path = output_path / tif_file

            with open(tif_path, 'wb') as f:
                f.write(zf.read(tif_file))

            print(f"✓ Downloaded PRISM data to {tif_path}")

            # Read the TIF file using rasterio
            with rasterio.open(tif_path) as src:
                # Read full raster
                full_data = src.read(1).astype(np.float32)
                full_transform = src.transform
                full_crs = src.crs

                # Calculate pixel coordinates for bbox
                # Transform from geographic to pixel coordinates
                from rasterio.transform import rowcol

                # Get row/col for bbox corners
                min_row, min_col = rowcol(full_transform, min_lon, max_lat)  # Upper-left
                max_row, max_col = rowcol(full_transform, max_lon, min_lat)  # Lower-right

                # Clamp to raster bounds
                min_row = max(0, min_row)
                min_col = max(0, min_col)
                max_row = min(full_data.shape[0], max_row)
                max_col = min(full_data.shape[1], max_col)

                # Extract subset
                subset_data = full_data[min_row:max_row, min_col:max_col].copy()

                # Create transform for subset
                # Transform maps from pixel coordinates to geographic coordinates
                # For a subset starting at (min_row, min_col), we need to adjust the transform
                subset_transform = full_transform * Affine.translation(min_col, min_row)

                return subset_data, subset_transform

    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error downloading PRISM data: {e}") from e
    except Exception as e:
        raise Exception(f"Failed to download PRISM data: {e}") from e


def get_prism_annual_precip(
    bbox: Tuple[float, float, float, float], output_dir: str, use_real_data: bool = False
) -> Tuple[np.ndarray, Affine]:
    """
    Download PRISM annual precipitation for bounding box.

    By default, creates synthetic orographic precipitation model for testing.
    Set use_real_data=True to download actual PRISM data from Oregon State servers.

    Parameters
    ----------
    bbox : tuple
        Bounding box (min_lat, min_lon, max_lat, max_lon)
    output_dir : str
        Output directory
    use_real_data : bool, default False
        If True, download real PRISM data. If False, generate synthetic data.

    Returns
    -------
    np.ndarray
        Precipitation data (mm/year)
    Affine
        Geographic transform
    """
    min_lat, min_lon, max_lat, max_lon = bbox

    # Use real PRISM data if requested
    if use_real_data:
        return download_real_prism_annual(bbox, output_dir)

    # Create synthetic precipitation data
    # Resolution: ~4km (PRISM resolution) = ~0.04 degrees
    resolution = 0.04

    # Calculate grid dimensions
    width = int((max_lon - min_lon) / resolution) + 1
    height = int((max_lat - min_lat) / resolution) + 1

    # Create orographic precipitation model
    # Base: 300mm/year at sea level
    # Gradient: wetter in north, drier in south (typical for US West Coast)
    base_precip = 300.0
    lat_gradient = np.linspace(1.5, 0.8, height).reshape(-1, 1)  # Wetter in north
    lon_gradient = np.linspace(0.9, 1.1, width).reshape(1, -1)  # Slight west-east gradient

    # Create precipitation grid
    precip = base_precip * lat_gradient * lon_gradient

    # Add some variability (simulate topographic effects)
    noise = np.random.RandomState(42).normal(1.0, 0.15, size=(height, width))
    precip = precip * noise

    # Clip to reasonable range
    precip = np.clip(precip, 100, 2000)  # 100-2000 mm/year

    # Create geographic transform
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)

    return precip.astype(np.float32), transform


def validate_precipitation_alignment(
    dem: Optional[np.ndarray],
    precip: Optional[np.ndarray],
    dem_shape: Optional[Tuple[int, int]] = None,
    dem_crs: Optional[str] = None,
    precip_crs: Optional[str] = None,
) -> None:
    """
    Validate that precipitation data aligns with DEM.

    Parameters
    ----------
    dem : np.ndarray, optional
        DEM array
    precip : np.ndarray, optional
        Precipitation array
    dem_shape : tuple, optional
        DEM shape if arrays not provided
    dem_crs : str, optional
        DEM CRS
    precip_crs : str, optional
        Precipitation CRS

    Raises
    ------
    ValueError
        If shapes or CRS don't match
    """
    # Validate shapes
    if dem is not None and precip is not None:
        if dem.shape != precip.shape:
            raise ValueError(
                f"Precipitation shape mismatch: DEM {dem.shape} vs Precip {precip.shape}"
            )
    elif dem_shape is not None and precip is not None:
        if precip.shape != dem_shape:
            raise ValueError(
                f"Precipitation shape mismatch: DEM {dem_shape} vs Precip {precip.shape}"
            )

    # Validate CRS
    if dem_crs is not None and precip_crs is not None:
        if dem_crs != precip_crs:
            raise ValueError(
                f"CRS mismatch: DEM {dem_crs} vs Precipitation {precip_crs}. "
                "Reproject one to match the other."
            )


def list_available_datasets(include_metadata: bool = False) -> Union[List[str], Dict]:
    """
    List available precipitation datasets.

    Parameters
    ----------
    include_metadata : bool, default False
        Include metadata for each dataset

    Returns
    -------
    list or dict
        List of dataset names, or dict with metadata if include_metadata=True
    """
    if include_metadata:
        return DATASETS.copy()
    else:
        return list(DATASETS.keys())


def _create_synthetic_precipitation(
    bbox: Tuple[float, float, float, float], resolution: float = 0.04
) -> Tuple[np.ndarray, Affine]:
    """
    Create synthetic precipitation data for testing.

    Parameters
    ----------
    bbox : tuple
        Bounding box (min_lat, min_lon, max_lat, max_lon)
    resolution : float
        Resolution in degrees (default 0.04 = ~4km)

    Returns
    -------
    np.ndarray
        Precipitation data (mm/year)
    Affine
        Geographic transform
    """
    min_lat, min_lon, max_lat, max_lon = bbox

    width = int((max_lon - min_lon) / resolution) + 1
    height = int((max_lat - min_lat) / resolution) + 1

    # Simple gradient model
    precip = np.linspace(400, 800, height * width).reshape(height, width).astype(np.float32)

    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)

    return precip, transform


def _write_precipitation_geotiff(
    path: Path, data: np.ndarray, transform: Affine, crs: str = "EPSG:4326"
) -> None:
    """
    Write precipitation data to GeoTIFF.

    Parameters
    ----------
    path : Path
        Output file path
    data : np.ndarray
        Precipitation data
    transform : Affine
        Geographic transform
    crs : str
        Coordinate reference system
    """
    height, width = data.shape

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(data, 1)
        # Add metadata
        dst.update_tags(
            1,
            units="mm/year",
            description="Annual precipitation",
        )
