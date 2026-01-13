Rendering Module
================

Blender scene setup and rendering functions.

Scene Setup
-----------

.. autofunction:: src.terrain.scene_setup.clear_scene

.. autofunction:: src.terrain.scene_setup.position_camera_relative

   Smart camera positioning using cardinal directions.
   Used in :doc:`../examples/elevation`.

   Example::

       camera = position_camera_relative(
           mesh,
           direction='south',
           elevation=0.3,
           camera_type='ORTHO',
           ortho_scale=7.0
       )

.. autofunction:: src.terrain.scene_setup.setup_hdri_lighting

   Nishita sky model for realistic ambient lighting.
   Used in :doc:`../examples/combined_render`.

.. autofunction:: src.terrain.scene_setup.setup_two_point_lighting

   Traditional sun + fill lighting.

.. autofunction:: src.terrain.scene_setup.create_background_plane

Rendering
---------

.. autofunction:: src.terrain.rendering.render_scene_to_file

   Example::

       result = render_scene_to_file(
           'output.png',
           width=1920,
           height=1080,
           file_format='PNG'
       )

.. autofunction:: src.terrain.rendering.setup_render_settings

Materials
---------

.. autofunction:: src.terrain.materials.apply_vertex_colors

.. autofunction:: src.terrain.materials.apply_terrain_with_obsidian_roads

   Used in :doc:`../examples/combined_render` for glossy road rendering.

.. autofunction:: src.terrain.materials.apply_test_material

   Debug materials: obsidian, chrome, clay, plastic, gold.

Water Detection
---------------

.. autofunction:: src.terrain.water.identify_water_by_slope

   Slope-based water detection. Used in :doc:`../examples/elevation`.

   Example::

       water_mask = identify_water_by_slope(
           dem,
           slope_threshold=0.01,
           fill_holes=True
       )
