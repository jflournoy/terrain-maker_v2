Visualization Module
====================

Diagnostic and debug visualizations for terrain processing.

This subpackage provides visualization tools for flow analysis, edge
extrusion debugging, bounds pipeline transformations, and linear feature
layers (streams, roads, trails).

Flow Diagnostics
----------------

Standardized visualizations for flow accumulation analysis: DEM views,
ocean masks, water bodies, drainage networks, and validation summaries.

.. automodule:: src.terrain.visualization.flow_diagnostics
   :members:
   :undoc-members:
   :show-inheritance:

Line Layers
-----------

Linear feature overlay creation (streams, roads, trails, power lines) from
raster data. Supports variable-width lines with smooth gaussian tapering.

.. automodule:: src.terrain.visualization.line_layers
   :members:
   :undoc-members:
   :show-inheritance:

Bounds Pipeline
---------------

Multi-stage coordinate transformation pipeline for bounds visualization.
Handles WGS84 to UTM projection, flipping, and downsampling to mesh grid.

.. automodule:: src.terrain.visualization.bounds_pipeline
   :members:
   :undoc-members:
   :show-inheritance:

Edge Debug
----------

Diagnostic plots for rectangle edge sampling and coordinate transformations.

.. automodule:: src.terrain.visualization.edge_debug
   :members:
   :undoc-members:
   :show-inheritance:

See Also
--------

- :doc:`diagnostics` - Terrain diagnostics
- :doc:`flow_accumulation` - Flow routing algorithms
- :doc:`flow_pipeline` - Flow computation pipeline
