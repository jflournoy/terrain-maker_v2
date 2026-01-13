Core Module
===========

The core module provides the main :class:`Terrain` class and essential functions for loading and processing DEM data.

Terrain Class
-------------

.. autoclass:: src.terrain.core.Terrain
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Examples

   Basic usage::

       from src.terrain.core import Terrain, load_dem_files

       dem, transform = load_dem_files('path/to/tiles')
       terrain = Terrain(dem, transform)
       terrain.apply_transforms()
       mesh = terrain.create_mesh()

   See :doc:`../examples/elevation` for a complete example.

Loading Functions
-----------------

.. autofunction:: src.terrain.core.load_dem_files

.. autofunction:: src.terrain.data_loading.load_filtered_hgt_files

   Example::

       dem, transform = load_filtered_hgt_files(
           'path/to/tiles',
           min_latitude=41  # Only load N41 and above
       )

Color Mapping
-------------

See :meth:`~src.terrain.core.Terrain.set_color_mapping` for single colormap usage.

See :meth:`~src.terrain.core.Terrain.set_blended_color_mapping` for dual-colormap visualization,
used in :doc:`../examples/combined_render`.

Data Layers
-----------

See :meth:`~src.terrain.core.Terrain.add_data_layer` for adding georeferenced data layers,
used in :doc:`../examples/sledding` for adding snow data.

See :meth:`~src.terrain.core.Terrain.compute_proximity_mask` for creating proximity zones,
used in :doc:`../examples/combined_render` for park proximity zones.
