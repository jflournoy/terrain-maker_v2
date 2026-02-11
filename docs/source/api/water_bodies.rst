Water Bodies Module
===================

The water bodies module provides functions for downloading, rasterizing, and routing flow through lakes and reservoirs.

Overview
--------

This module integrates water body data from two sources:

- **NHD (National Hydrography Dataset)** - US-only, high detail
- **HydroLAKES** - Global coverage, includes reservoir data

Water bodies are treated differently based on their context:

- **Lakes inside endorheic basins** act as sinks (terminal drainage)
- **Lakes outside basins** act as conduits (water flows through to outlet)

See :doc:`../guides/flow-routing` for detailed information on water body routing strategies.

Data Download
-------------

.. autofunction:: src.terrain.water_bodies.download_water_bodies

   Main download function that dispatches to NHD or HydroLAKES.

   Example::

       from src.terrain.water_bodies import download_water_bodies

       geojson_path = download_water_bodies(
           bbox=(32.5, -117.5, 33.5, -116.5),  # south, west, north, east
           output_dir="data/water_bodies",
           data_source="hydrolakes",
           min_area_km2=0.01,
       )

.. autofunction:: src.terrain.water_bodies.download_nhd_water_bodies

   Download from National Hydrography Dataset (US only).

.. autofunction:: src.terrain.water_bodies.download_hydrolakes

   Download from HydroLAKES global database.

Rasterization
-------------

.. autofunction:: src.terrain.water_bodies.rasterize_lakes_to_mask

   Convert GeoJSON lake polygons to a labeled raster mask.

   Example::

       import json
       from src.terrain.water_bodies import rasterize_lakes_to_mask

       with open("lakes.geojson") as f:
           lakes_geojson = json.load(f)

       lake_mask, transform = rasterize_lakes_to_mask(
           lakes_geojson,
           bbox=(32.5, -117.5, 33.5, -116.5),
           resolution=30.0,  # meters
       )
       # lake_mask: 2D array where each lake has unique ID (1, 2, 3, ...)

Outlet and Inlet Detection
--------------------------

.. autofunction:: src.terrain.water_bodies.identify_outlet_cells

   Identify outlet pixels from geographic outlet coordinates.

.. autofunction:: src.terrain.water_bodies.identify_lake_outlets_from_nhd

   Extract outlet coordinates from NHD flowline data.

.. autofunction:: src.terrain.water_bodies.identify_lake_inlets

   Find cells where streams flow INTO lakes.

   Example::

       from src.terrain.water_bodies import identify_lake_inlets

       inlets_dict = identify_lake_inlets(
           lake_mask,
           dem_conditioned,
           outlet_mask=lake_outlets,  # exclude outlets from inlets
       )
       # Returns {lake_id: [(row, col), ...], ...}

Flow Routing
------------

.. autofunction:: src.terrain.water_bodies.create_lake_flow_routing

   Create D8 flow directions for lake interiors using BFS from outlets.

   Example::

       from src.terrain.water_bodies import create_lake_flow_routing

       lake_flow = create_lake_flow_routing(
           lake_mask,
           lake_outlets,
           dem_conditioned,
       )
       # Returns D8 direction grid for lake cells (0 at outlets)

       # Merge with terrain flow
       flow_dir = np.where(lake_mask > 0, lake_flow, terrain_flow_dir)

Usage with Flow Accumulation
----------------------------

Complete workflow integrating water bodies with flow routing::

    from src.terrain.flow_accumulation import (
        condition_dem_spec,
        compute_flow_direction,
        compute_drainage_area,
        detect_ocean_mask,
        detect_endorheic_basins,
    )
    from src.terrain.water_bodies import (
        download_water_bodies,
        rasterize_lakes_to_mask,
        identify_outlet_cells,
        create_lake_flow_routing,
    )
    import json

    # Load DEM
    with rasterio.open("dem.tif") as src:
        dem = src.read(1)
        transform = src.transform

    # Detect ocean and basins
    ocean_mask = detect_ocean_mask(dem, threshold=0.0)
    basin_mask, basins = detect_endorheic_basins(dem, min_size=5000)

    # Download water bodies
    geojson_path = download_water_bodies(
        bbox=bbox,
        output_dir="water_bodies",
        data_source="hydrolakes",
    )

    with open(geojson_path) as f:
        lakes_geojson = json.load(f)

    # Rasterize lakes
    lake_mask, lake_transform = rasterize_lakes_to_mask(lakes_geojson, bbox)

    # Extract outlets
    outlets_dict = {
        idx: feature["properties"]["outlet"]
        for idx, feature in enumerate(lakes_geojson["features"], 1)
        if "outlet" in feature["properties"]
    }
    lake_outlets = identify_outlet_cells(lake_mask, outlets_dict, lake_transform)

    # Condition DEM (mask lakes inside basins, keep others as conduits)
    lakes_in_basins = lake_mask & basin_mask
    conditioning_mask = ocean_mask | basin_mask | lakes_in_basins

    dem_conditioned, outlets = condition_dem_spec(dem, nodata_mask=conditioning_mask)

    # Compute flow direction (terrain only)
    flow_dir_terrain = compute_flow_direction(dem_conditioned, mask=ocean_mask)

    # Apply lake routing (for lakes outside basins)
    lakes_outside = lake_mask & ~basin_mask
    lake_flow = create_lake_flow_routing(lakes_outside, lake_outlets, dem_conditioned)
    flow_dir = np.where(lakes_outside > 0, lake_flow, flow_dir_terrain)

    # Compute drainage area
    drainage_area = compute_drainage_area(flow_dir)

See Also
--------

- :doc:`../guides/flow-routing` - Comprehensive guide including water body handling
- :mod:`src.terrain.flow_accumulation` - Core flow routing functions
- :mod:`src.terrain.water` - Water detection from slope analysis
