Scene Setup Module
==================

Functions for configuring Blender scenes, cameras, lighting, and atmosphere.

This module provides all the tools needed to set up professional terrain visualizations
in Blender, from camera positioning to HDRI lighting.

Background Plane
----------------

.. autofunction:: src.terrain.scene_setup.create_background_plane

   Used in :doc:`../examples/combined_render` for clean backgrounds.

   Example::

       background = create_background_plane(
           camera, terrain_mesh,
           distance_below=50.0,
           color="#F5F5F0",
           receive_shadows=True
       )

Camera Setup
------------

.. autofunction:: src.terrain.scene_setup.setup_camera

   Basic camera creation and configuration.

.. autofunction:: src.terrain.scene_setup.position_camera_relative

   Smart camera positioning using cardinal directions (north, south, east, west, above).
   Used in :doc:`../examples/combined_render`.

   Example::

       position_camera_relative(
           camera, terrain_mesh,
           direction='south',
           distance_multiplier=2.0,
           elevation_angle=2.5,
           tilt_angle=0
       )

.. autofunction:: src.terrain.scene_setup.calculate_camera_frustum_size

   Calculate camera field of view for precise framing.

Lighting
--------

.. autofunction:: src.terrain.scene_setup.setup_two_point_lighting

   Traditional two-point lighting with sun and fill lights.

.. autofunction:: src.terrain.scene_setup.setup_hdri_lighting

   Environment-based HDRI lighting for photorealistic results.
   Used in :doc:`../examples/combined_render`.

   Example::

       setup_hdri_lighting(
           hdri_path='path/to/hdri.exr',
           strength=1.0,
           rotation=0.0
       )

.. autofunction:: src.terrain.scene_setup.setup_light

   Create individual Blender lights (sun, spot, area, point).

.. autofunction:: src.terrain.scene_setup.setup_world_atmosphere

   Add atmospheric scattering effects.

   Example::

       setup_world_atmosphere(
           density=0.0002,
           scatter_color=(1, 1, 1, 1),
           anisotropy=0.8
       )

Scene Management
----------------

.. autofunction:: src.terrain.scene_setup.clear_scene

   Remove all objects from the scene before setting up.

.. autofunction:: src.terrain.scene_setup.setup_camera_and_light

   Legacy combined camera+light setup (consider using individual functions instead).

Utilities
---------

.. autofunction:: src.terrain.scene_setup.create_matte_material

   Create matte materials for background objects.

.. autofunction:: src.terrain.scene_setup.hex_to_rgb

   Convert hex color strings to RGB tuples.

   Example::

       r, g, b = hex_to_rgb("#F5F5F0")  # Eggshell white
