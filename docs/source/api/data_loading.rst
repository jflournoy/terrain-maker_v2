Data Loading Module
===================

Functions for loading geographic data from various sources.

DEM Loading
-----------

.. autofunction:: src.terrain.data_loading.load_filtered_hgt_files

   Load SRTM HGT files with latitude filtering.

.. autofunction:: src.terrain.data_loading.save_score_grid

.. autofunction:: src.terrain.data_loading.find_score_file

Score Data
----------

.. autofunction:: src.terrain.scoring.compute_sledding_score

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

.. autofunction:: src.terrain.roads.get_roads_tiled

   Fetch roads from OpenStreetMap via Overpass API.

.. autofunction:: src.terrain.roads.add_roads_layer

   Add roads as a terrain data layer.
   Used in :doc:`../examples/combined_render`.

.. autofunction:: src.terrain.roads.rasterize_roads_to_layer

.. autofunction:: src.terrain.roads.smooth_road_vertices

.. autofunction:: src.terrain.roads.offset_road_vertices

.. autofunction:: src.terrain.roads.smooth_road_mask

   Anti-alias road edges.

Diagnostics
-----------

.. autofunction:: src.terrain.diagnostics.generate_rgb_histogram

.. autofunction:: src.terrain.diagnostics.generate_luminance_histogram

.. autofunction:: src.terrain.diagnostics.plot_wavelet_diagnostics

.. autofunction:: src.terrain.diagnostics.generate_upscale_diagnostics
