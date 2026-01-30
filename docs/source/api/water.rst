Water Module
============

Water body detection from elevation data using slope analysis.

This module identifies water bodies by detecting flat surfaces (low slope) in DEM data.
Water is characterized by near-zero slope, while terrain typically has higher slopes.

Water Detection
---------------

.. autofunction:: src.terrain.water.identify_water_by_slope

   Identify water bodies by detecting flat areas in DEM.
   Used in :doc:`../examples/combined_render` for lake/river detection.

   Example::

       from src.terrain.water import identify_water_by_slope

       # Detect water in DEM
       water_mask = identify_water_by_slope(
           dem_data,
           slope_threshold=0.1,  # Flatter than 0.1 = water
           fill_holes=True       # Smooth boundaries
       )

       # Use mask to color water blue
       colors[water_mask] = [0.02, 0.13, 0.25]  # Michigan blue

   **How it works:**

   1. Calculate slope magnitude using Horn's method (Sobel operators)
   2. Classify pixels below threshold as water
   3. Apply morphological operations to fill gaps and smooth boundaries

   **Typical slope thresholds:**

   - ``0.05``: Very strict, only extremely flat surfaces
   - ``0.1``: Default, good for most DEMs
   - ``0.2-0.5``: More permissive, captures wetlands and floodplains

   **Slope threshold depends on DEM resolution:**

   - High-res DEM (1-10m): Use lower threshold (0.05-0.1)
   - Low-res DEM (30m+): Use higher threshold (0.1-0.3)

Internal Functions
------------------

.. autofunction:: src.terrain.water._calculate_slope

   Calculate slope magnitude using Sobel operators (Horn's method).

.. autofunction:: src.terrain.water._smooth_water_mask

   Apply morphological operations to smooth water mask.
