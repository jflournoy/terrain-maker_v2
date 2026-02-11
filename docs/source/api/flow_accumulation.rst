Flow Accumulation Module
========================

The flow accumulation module provides hydrological flow routing algorithms for DEM data, implementing D8 flow direction and drainage area computation.

Overview
--------

This module implements a complete hydrological flow routing pipeline based on peer-reviewed literature:

1. **Outlet Identification** - Classify cells that serve as drainage termini
2. **DEM Conditioning** - Breach sinks and fill residual depressions
3. **Flow Direction** - Assign D8 flow directions (ESRI encoding)
4. **Flow Accumulation** - Compute upstream contributing area

See :doc:`../guides/flow-routing` for detailed algorithm explanations and debugging strategies.

Main Functions
--------------

.. autofunction:: src.terrain.flow_accumulation.flow_accumulation

   Example::

       from src.terrain.flow_accumulation import flow_accumulation

       # Run complete flow pipeline
       result = flow_accumulation(
           dem_path="data/my_dem.tif",
           precip_path="data/precipitation.tif",  # optional
           backend="spec",
           max_breach_depth=25.0,
           max_breach_length=80,
       )

       flow_dir = result['flow_dir']
       drainage_area = result['drainage_area']
       upstream_rainfall = result['upstream_rainfall']

Flow Direction
--------------

.. autofunction:: src.terrain.flow_accumulation.compute_flow_direction

   Example::

       import numpy as np
       from src.terrain.flow_accumulation import compute_flow_direction

       # Create simple DEM
       dem = np.array([
           [10, 9, 8],
           [9, 8, 7],
           [8, 7, 6],
       ], dtype=np.float32)

       flow_dir = compute_flow_direction(dem)
       # Returns D8 direction codes (ESRI convention: 1=E, 64=S, 128=SE, etc.)

Drainage Area
-------------

.. autofunction:: src.terrain.flow_accumulation.compute_drainage_area

   Example::

       from src.terrain.flow_accumulation import (
           compute_flow_direction,
           compute_drainage_area,
       )

       flow_dir = compute_flow_direction(dem)
       drainage_area = compute_drainage_area(flow_dir)
       # Each cell contains count of upstream contributing cells

Upstream Rainfall
-----------------

.. autofunction:: src.terrain.flow_accumulation.compute_upstream_rainfall

   Example::

       from src.terrain.flow_accumulation import (
           compute_flow_direction,
           compute_upstream_rainfall,
       )

       flow_dir = compute_flow_direction(dem)
       precipitation = np.ones_like(dem) * 500  # 500mm everywhere
       upstream_rainfall = compute_upstream_rainfall(flow_dir, precipitation)
       # Each cell contains sum of precipitation from all upstream cells

DEM Conditioning
----------------

.. autofunction:: src.terrain.flow_accumulation.condition_dem_spec

   This is the recommended conditioning function, implementing the spec-compliant
   constrained breaching and priority-flood fill algorithm.

   Example::

       from src.terrain.flow_accumulation import condition_dem_spec

       dem_conditioned, outlets = condition_dem_spec(
           dem,
           nodata_mask=ocean_mask,
           coastal_elev_threshold=0.0,
           edge_mode='all',
           max_breach_depth=25.0,
           max_breach_length=80,
           epsilon=1e-4,
       )

.. autofunction:: src.terrain.flow_accumulation.condition_dem

   Legacy conditioning function with simpler fill-only approach.

.. autofunction:: src.terrain.flow_accumulation.breach_depressions_constrained

   Low-level breaching function for advanced use cases.

.. autofunction:: src.terrain.flow_accumulation.priority_flood_fill_epsilon

   Low-level priority-flood fill with epsilon gradient.

Outlet and Basin Detection
--------------------------

.. autofunction:: src.terrain.flow_accumulation.identify_outlets

.. autofunction:: src.terrain.flow_accumulation.detect_ocean_mask

   Example::

       from src.terrain.flow_accumulation import detect_ocean_mask

       ocean_mask = detect_ocean_mask(dem, threshold=0.0, border_only=True)

.. autofunction:: src.terrain.flow_accumulation.detect_endorheic_basins

   Example::

       from src.terrain.flow_accumulation import detect_endorheic_basins

       basin_mask, basins = detect_endorheic_basins(
           dem,
           min_size=5000,      # cells
           min_depth=1.0,      # meters
           exclude_mask=ocean_mask,
       )

D8 Direction Constants
----------------------

The module uses ESRI's standard D8 power-of-2 encoding::

    # D8 Neighbor Geometry:
    #   8  4  2
    #  16  x  1
    #  32 64 128

.. py:data:: src.terrain.flow_accumulation.D8_DIRECTIONS
   :type: dict

   Mapping from (row_offset, col_offset) to direction code.

.. py:data:: src.terrain.flow_accumulation.D8_OFFSETS
   :type: dict

   Reverse mapping from direction code to (row_offset, col_offset).

See Also
--------

- :doc:`../guides/flow-routing` - Comprehensive guide with algorithm details
- `flow-spec.md <https://github.com/your-repo/terrain-maker/blob/main/flow-spec.md>`_ - Full algorithm specification
