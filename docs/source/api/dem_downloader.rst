DEM Downloader Module
=====================

Download SRTM Digital Elevation Model data for specified geographic areas.

This module provides functions to download SRTM elevation data using either
bounding box coordinates or place names, via NASA Earthdata authentication.

SRTM Data
---------

- **Source:** NASA Shuttle Radar Topography Mission
- **Coverage:** 60N to 56S latitude
- **Resolution:** 1 arc-second (~30m) for SRTM1, 3 arc-second (~90m) for SRTM3
- **Format:** HGT (Height) files, 1 x 1 tile grid

Usage Examples
--------------

Download by bounding box::

    from src.terrain.dem_downloader import download_dem_by_bbox

    bbox = (42.0, -83.5, 42.5, -83.0)  # Detroit area
    files = download_dem_by_bbox(
        bbox=bbox,
        output_dir="data/detroit_dem",
    )

Download by place name::

    from src.terrain.dem_downloader import download_dem_by_place_name

    files = download_dem_by_place_name(
        place_name="Detroit, MI",
        output_dir="data/detroit_dem",
    )

API Reference
-------------

.. automodule:: src.terrain.dem_downloader
   :members:
   :undoc-members:
   :show-inheritance:

See Also
--------

- :doc:`data_loading` - Loading downloaded DEM files
- :doc:`core` - Terrain class for working with DEM data
