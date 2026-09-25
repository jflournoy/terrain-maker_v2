Data Loading Module
===================

Functions for loading geographic data from various sources.

DEM Loading
-----------

.. autofunction:: terrain_maker.terrain.data_loading.load_filtered_hgt_files

   Load SRTM HGT files with latitude filtering.

.. autofunction:: terrain_maker.terrain.data_loading.save_score_grid

.. autofunction:: terrain_maker.terrain.data_loading.find_score_file

Score Data
----------

.. autofunction:: terrain_maker.terrain.scoring.compute_sledding_score

   Used in :doc:`../examples/sledding`.

   Example::

       score = compute_sledding_score(
           terrain,
           depth_weight=0.4,
           coverage_weight=0.3,
           slope_weight=0.3
       )

Roads
-----

.. autofunction:: terrain_maker.terrain.roads.get_roads_tiled

   Fetch roads from OpenStreetMap via Overpass API.

.. autofunction:: terrain_maker.terrain.roads.add_roads_layer

   Add roads as a terrain data layer.
   Used in :doc:`../examples/combined_render`.

.. autofunction:: terrain_maker.terrain.roads.rasterize_roads_to_layer

.. autofunction:: terrain_maker.terrain.roads.smooth_road_vertices

.. autofunction:: terrain_maker.terrain.roads.offset_road_vertices

.. autofunction:: terrain_maker.terrain.roads.smooth_road_mask

   Anti-alias road edges.

Diagnostics
-----------

.. autofunction:: terrain_maker.terrain.diagnostics.generate_rgb_histogram

.. autofunction:: terrain_maker.terrain.diagnostics.generate_luminance_histogram

.. autofunction:: terrain_maker.terrain.diagnostics.plot_wavelet_diagnostics

.. autofunction:: terrain_maker.terrain.diagnostics.generate_upscale_diagnostics
