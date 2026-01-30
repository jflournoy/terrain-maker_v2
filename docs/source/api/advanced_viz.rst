Advanced Visualization Module
=============================

Specialized visualization features for terrain data.

This module provides advanced visualization capabilities including drive-time isochrones,
slope calculation, and 3D legends for Blender scenes.

Migrated from legacy helpers.py with improvements.

Slope Calculation
-----------------

.. autofunction:: src.terrain.advanced_viz.horn_slope

   Calculate slope using Horn's method with GPU acceleration.

   **Horn's method:**

   - Standard GIS technique using 3×3 Sobel-like kernel
   - Properly handles NaN values through interpolation
   - GPU-accelerated (7x speedup on CUDA via PyTorch)

   Example::

       from src.terrain.advanced_viz import horn_slope

       # Calculate slopes
       slopes = horn_slope(dem_data)

       # Slopes in degrees (gradient magnitude)
       print(f"Slope range: {slopes.min():.2f}° to {slopes.max():.2f}°")

   **GPU acceleration:**

   Automatically uses GPU if available via :func:`~terrain.gpu_ops.gpu_horn_slope`.

   **NaN handling:**

   - Interpolates NaN values before slope computation
   - Returns NaN where interpolation fails
   - Logs NaN counts for debugging

3D Legends
----------

.. autofunction:: src.terrain.advanced_viz.create_values_legend

   Create 3D legend bar in Blender scene.

   Generates a vertical bar with color gradient and text labels showing
   the value scale for terrain visualization.

   Example::

       from src.terrain.advanced_viz import create_values_legend

       # Create legend for elevation
       legend_obj, labels = create_values_legend(
           terrain_obj=terrain_mesh,
           values=elevation_data,
           mpp=30.0,                    # Meters per pixel
           colormap_name='michigan',
           n_samples=10,
           label='Elevation',
           units='meters',
           scale=0.2,
           position_offset=(5, 0, 0)
       )

       print(f"Created legend with {len(labels)} labels")

   **Parameters:**

   - ``terrain_obj``: Reference mesh for positioning
   - ``values``: Data array for min/max range
   - ``mpp``: Meters per pixel (default: 30)
   - ``colormap_name``: Matplotlib colormap (default: 'mako_r')
   - ``n_samples``: Number of labels (default: 10)
   - ``label``: Legend title (default: 'Value')
   - ``units``: Unit string (e.g., 'meters')
   - ``scale``: Legend size multiplier (default: 0.2)
   - ``position_offset``: (x, y, z) offset from terrain

   **Returns:**

   Tuple of (legend_object, text_objects_list)

Legacy Features
---------------

This module was migrated from legacy ``helpers.py`` and may contain
additional features for:

- Drive-time isochrone curves (3D transportation analysis)
- Custom color gradients for legends
- Terrain-relative positioning

See source code for additional functionality.
