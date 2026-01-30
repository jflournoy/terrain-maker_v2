Materials Module
================

Material presets and shader functions for terrain visualization.

This module provides color presets, material configurations, and shader functions
for creating professional-looking terrain renders with roads, water, and other features.

Color Presets
-------------

The module defines several color preset dictionaries:

**ROAD_COLORS** - Polished stone/mineral appearance for roads:

- ``obsidian``: Near-black volcanic glass
- ``azurite``: Deep blue mineral
- ``azurite-light``: Richer azurite
- ``malachite``: Deep green copper mineral
- ``hematite``: Dark iron red-gray

**BASE_MATERIALS** - Materials for mesh bases and backgrounds:

- ``clay``: Matte gray clay (default)
- ``obsidian``: Dark volcanic glass
- ``chrome``: Metallic chrome
- ``plastic``: Glossy white plastic
- ``gold``: Metallic gold
- ``ivory``: Off-white with warm tone

**ALL_COLORS** - Unified dictionary combining all presets.

Color Helper Functions
----------------------

.. autofunction:: src.terrain.materials.get_all_colors_choices

   Get list of all available color preset names.

.. autofunction:: src.terrain.materials.get_all_colors_help

   Get formatted help text listing all color presets.

.. autofunction:: src.terrain.materials.get_terrain_materials_choices

.. autofunction:: src.terrain.materials.get_terrain_materials_help

.. autofunction:: src.terrain.materials.get_road_colors_choices

.. autofunction:: src.terrain.materials.get_road_colors_help

.. autofunction:: src.terrain.materials.get_base_materials_choices

.. autofunction:: src.terrain.materials.get_base_materials_help

Color Lookup Functions
----------------------

.. autofunction:: src.terrain.materials.get_color

   Convert color preset name or RGB tuple to normalized RGB.

   Example::

       # Using preset name
       rgb = get_color("azurite")  # Returns (0.04, 0.09, 0.16)

       # Using RGB tuple
       rgb = get_color((0.5, 0.5, 0.5))  # Returns (0.5, 0.5, 0.5)

.. autofunction:: src.terrain.materials.get_base_material_color

   Get color for base material presets.

.. autofunction:: src.terrain.materials.get_terrain_material_params

   Get material parameters (color, roughness, metallic) for terrain materials.

Material Application Functions
-------------------------------

.. autofunction:: src.terrain.materials.apply_colormap_material

   Apply vertex color material to terrain mesh.

.. autofunction:: src.terrain.materials.apply_water_shader

   Apply water shader with custom color and glossiness.

.. autofunction:: src.terrain.materials.apply_terrain_with_obsidian_roads

   Apply vertex colors to terrain with glossy road material.
   Used in :doc:`../examples/combined_render` for road rendering.

   Example::

       apply_terrain_with_obsidian_roads(
           terrain_mesh,
           road_color="azurite",
           road_roughness=0.05
       )

.. autofunction:: src.terrain.materials.apply_glassy_road_material

   Apply glossy glass-like material to road surfaces.

.. autofunction:: src.terrain.materials.apply_test_material

   Apply test materials for development (obsidian, chrome, clay, plastic, gold, ivory).

   Example::

       apply_test_material(material, style="obsidian")
       apply_test_material(material, style="chrome")
