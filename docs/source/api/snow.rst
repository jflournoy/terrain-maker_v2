Snow Analysis Module
====================

SNODAS snow data processing and slope statistics computation.

This package provides tools for loading, processing, and analyzing snow data
from NOAA's SNODAS (Snow Data Assimilation System) and computing terrain
slope statistics for scoring.

SNODAS Processing
-----------------

Pipeline for loading and computing snow statistics:

1. ``batch_process_snodas_data`` - Load and reproject SNODAS files
2. ``calculate_snow_statistics`` - Compute aggregated seasonal statistics
3. ``load_snodas_stats`` - High-level orchestrator with caching and fallback

.. automodule:: src.snow.snodas
   :members:
   :undoc-members:
   :show-inheritance:

Slope Statistics
----------------

Tiled slope computation at full DEM resolution with geographic transform-aware
aggregation. Ensures cliff faces within a pixel are captured rather than
hidden by downsampling.

.. automodule:: src.snow.slope_statistics
   :members:
   :undoc-members:
   :show-inheritance:

See Also
--------

- :doc:`scoring_combiner` - Score combination using snow statistics
- :doc:`gridded_data` - Memory-efficient tiled processing
