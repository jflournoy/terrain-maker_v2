Color Mapping Module
====================

Color mapping functions and custom colormaps for terrain visualization.

This module provides functions for mapping elevation and slope data to colors using
matplotlib colormaps, including custom perceptually uniform colormaps optimized for
terrain visualization.

Custom Colormaps
----------------

**michigan** - Michigan Natural Landscape Colormap
   Perceptually uniform progression through Michigan's natural features:

   - Great Lakes (deep blue) → Northern forests (evergreen) → Upland meadows → Sand dunes (tan)
   - Desaturated ~35% for subtle, sun-bleached appearance
   - Works well as base layer in dual colormaps

   Example::

       from src.terrain.color_mapping import elevation_colormap

       colors = elevation_colormap(
           dem_data,
           cmap_name='michigan',
           gamma=1.0
       )

**boreal_mako** - Boreal-Mako Winter Forest Colormap
   Perceptually uniform winter forest palette with purple ribbon for edge effect:

   - Dark boreal green (cool, blue-tinted forest)
   - Blue → purple ribbon → cyan → white
   - Monotonically increasing luminance (like viridis/mako)
   - Used in :doc:`../examples/combined_render` for sledding score visualization

   Access via the global ``boreal_mako_cmap`` variable::

       from src.terrain.color_mapping import boreal_mako_cmap

       # Use directly with matplotlib
       import matplotlib.pyplot as plt
       plt.imshow(data, cmap=boreal_mako_cmap)

       # Or use by name
       colors = elevation_colormap(data, cmap_name='boreal_mako')

.. autofunction:: src.terrain.color_mapping._build_boreal_mako_cmap

   Build boreal_mako colormap with configurable purple ribbon position.

   **Purple position parameter:**

   - Controls where the purple ribbon appears (0.0-1.0)
   - Default: 0.6 (creates edge effect for scores near upper range)
   - Used for visual emphasis in score visualizations

   Example::

       from src.terrain.color_mapping import _build_boreal_mako_cmap

       # Custom purple position
       custom_cmap = _build_boreal_mako_cmap(purple_position=0.7)

Color Mapping Functions
------------------------

.. autofunction:: src.terrain.color_mapping.elevation_colormap

   Map elevation or score data to RGB colors using matplotlib colormaps.

   Example::

       from src.terrain.color_mapping import elevation_colormap

       # Basic usage
       colors = elevation_colormap(dem_data, cmap_name='viridis')

       # With custom range and gamma correction
       colors = elevation_colormap(
           dem_data,
           cmap_name='michigan',
           min_elev=0,      # Override auto-min
           max_elev=500,    # Override auto-max
           gamma=1.2        # Brighten mid-tones
       )

       # Using custom colormaps
       colors = elevation_colormap(dem_data, cmap_name='boreal_mako')

   **Supported colormaps:**

   - Standard matplotlib: viridis, plasma, inferno, magma, cividis, terrain, gist_earth, turbo
   - Custom: michigan, boreal_mako

.. autofunction:: src.terrain.color_mapping.slope_colormap

   Map slope data to RGB colors.

   Example::

       from src.terrain.color_mapping import slope_colormap

       # Calculate slopes (example)
       dy, dx = np.gradient(dem_data)
       slopes = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

       # Color by slope
       colors = slope_colormap(
           slopes,
           cmap_name='terrain',
           min_slope=0,
           max_slope=45
       )
