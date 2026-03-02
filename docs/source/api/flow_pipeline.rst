Flow Pipeline Module
====================

High-level flow computation pipeline with basin preservation.

This module provides a validated flow computation pattern that properly handles
ocean detection, endorheic basin detection, water body integration, and DEM
conditioning. It orchestrates the lower-level functions from
:doc:`flow_accumulation` and :doc:`water_bodies`.

Pipeline Steps
--------------

1. **Ocean Detection** - Identify ocean/sea-level cells
2. **Endorheic Basin Detection** - Find and preserve internally-draining basins
3. **Water Body Integration** - Lake routing with spillway identification
4. **DEM Conditioning** - Breach sinks and fill residual depressions
5. **Flow Direction** - D8 flow directions with combined masks
6. **Drainage Area** - Upstream contributing area computation
7. **Discharge Potential** - Optional rainfall-weighted flow accumulation

API Reference
-------------

.. automodule:: src.terrain.flow_pipeline
   :members:
   :undoc-members:
   :show-inheritance:

See Also
--------

- :doc:`flow_accumulation` - Low-level flow routing algorithms
- :doc:`water_bodies` - Lake and river identification
