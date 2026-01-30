Roads Module
============

Road network visualization using the data layer pipeline.

This module provides efficient road rendering by rasterizing OpenStreetMap road networks
as geographic data layers. Roads are automatically aligned with the DEM through the
transform pipeline, handling reprojection and downsampling correctly.

Much more efficient than creating individual Blender objects - rasterizes roads in ~5
seconds and automatically handles coordinate transformations.

Quick Start
-----------

Simple API for adding roads to terrain::

    from src.terrain.roads import add_roads_layer
    from examples.detroit_roads import get_roads

    # Get roads from OpenStreetMap
    roads_geojson = get_roads(bbox)

    # Add as a data layer (automatic alignment)
    add_roads_layer(
        terrain=terrain,
        roads_geojson=roads_geojson,
        bbox=bbox,  # (south, west, north, east) in WGS84
        road_types=['motorway', 'trunk', 'primary'],
        road_width=3,
        colormap_name="viridis"
    )

See :doc:`../examples/combined_render` for full example.

Core Functions
--------------

.. autofunction:: src.terrain.roads.add_roads_layer

   High-level function to rasterize and add roads as a data layer.
   Used in :doc:`../examples/combined_render`.

.. autofunction:: src.terrain.roads.rasterize_roads_to_layer

   Rasterize road geometries to a grid with proper geographic transform.

   Returns road grid and Affine transform in WGS84 (EPSG:4326).
   Use with :meth:`~src.terrain.core.Terrain.add_data_layer` to add to terrain.

   Example::

       road_grid, road_transform = rasterize_roads_to_layer(
           roads_geojson,
           bbox=(south, west, north, east),
           resolution=30.0,  # 30m pixels
           road_types=['motorway', 'trunk'],
           road_width=3
       )

       terrain.add_data_layer(
           "roads", road_grid, road_transform,
           "EPSG:4326", target_layer="dem"
       )

Vertex Smoothing
----------------

.. autofunction:: src.terrain.roads.smooth_road_vertices

   Smooth road elevations to eliminate sudden height changes.
   Used in :doc:`../examples/combined_render`.

   Example::

       smooth_road_vertices(
           terrain, road_mask,
           kernel_size=5,
           iterations=2
       )

.. autofunction:: src.terrain.roads.offset_road_vertices

   Lower road vertices to create embankment effect.

   Example::

       offset_road_vertices(
           terrain, road_mask,
           offset_meters=-2.0  # Lower by 2 meters
       )

DEM Smoothing
-------------

.. autofunction:: src.terrain.roads.smooth_dem_along_roads

   Smooth DEM elevation data along roads before mesh creation.

Mask Operations
---------------

.. autofunction:: src.terrain.roads.smooth_road_mask

   Smooth road mask to avoid harsh transitions.

   Example::

       smoothed_mask = smooth_road_mask(
           road_mask,
           kernel_size=5,
           threshold=0.3
       )

Colormaps
---------

.. autofunction:: src.terrain.roads.road_colormap

   Map roads to distinctive marker color for shader detection.

.. autofunction:: src.terrain.roads.get_viridis_colormap

   Get viridis colormap for road visualization.
