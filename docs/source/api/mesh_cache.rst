Mesh Cache Module
=================

Caching for Blender mesh files (.blend) to speed up render iteration.

This module caches generated Blender meshes so render-only passes don't need to
regenerate geometry. Useful for iterating on camera angles and render settings
without waiting for mesh generation.

MeshCache
---------

.. autoclass:: src.terrain.mesh_cache.MeshCache
   :members:
   :undoc-members:

   Manages caching of Blender .blend files with mesh geometry.

   **What it caches:**

   - Complete .blend files with generated terrain mesh
   - Mesh generation parameters (scale_factor, height_scale, etc.)
   - DEM hash (for automatic invalidation)

   **When cache is invalidated:**

   - DEM data changes
   - Mesh parameters change (scale, height, vertex count, etc.)
   - Manual cache clearing

   Example::

       from src.terrain.mesh_cache import MeshCache

       cache = MeshCache(cache_dir='.mesh_cache', enabled=True)

       # Try to load cached mesh
       blend_file = cache.get_cached_mesh(
           dem_hash='abc123',
           mesh_params={
               'scale_factor': 100,
               'height_scale': 30,
               'target_vertices': 100000
           }
       )

       if blend_file:
           # Load existing .blend file
           bpy.ops.wm.open_mainfile(filepath=str(blend_file))
       else:
           # Generate new mesh
           mesh = terrain.create_mesh(scale_factor=100, height_scale=30)

           # Save to cache
           cache.save_mesh(
               blend_file='scene.blend',
               dem_hash='abc123',
               mesh_params={...}
           )

Usage Patterns
--------------

**Pattern 1: Multi-view renders**

Generate mesh once, render multiple camera angles::

    cache = MeshCache(enabled=True)
    dem_hash = hashlib.md5(dem_data.tobytes()).hexdigest()

    mesh_params = {
        'scale_factor': 100,
        'height_scale': 30,
        'target_vertices': 100000
    }

    # First render: mesh generated and cached
    blend_file = cache.get_cached_mesh(dem_hash, mesh_params)
    if not blend_file:
        mesh = terrain.create_mesh(**mesh_params)
        cache.save_mesh('scene.blend', dem_hash, mesh_params)

    # Render view 1
    render_scene_to_file('output_south.png', direction='south')

    # Subsequent renders: reuse cached mesh (10-50x faster!)
    blend_file = cache.get_cached_mesh(dem_hash, mesh_params)
    bpy.ops.wm.open_mainfile(filepath=str(blend_file))
    render_scene_to_file('output_north.png', direction='north')

**Pattern 2: Render iteration**

Fast iteration on lighting/camera settings::

    cache = MeshCache(enabled=True)

    # Generate mesh once (slow)
    mesh = terrain.create_mesh()  # ~30s
    cache.save_mesh('scene.blend', dem_hash, mesh_params)

    # Iterate on lighting (fast)
    for sun_energy in [0.5, 1.0, 1.5, 2.0]:
        bpy.ops.wm.open_mainfile(filepath=cache_path)  # ~1s
        setup_lighting(sun_energy=sun_energy)          # ~0.1s
        render_scene_to_file(f'output_{sun_energy}.png')  # ~5s

**Pattern 3: Development workflow**

::

    cache = MeshCache(enabled=True)

    # Initial run: mesh generated
    # ... create_mesh() ...  # 30s

    # Tweak colors, re-render (mesh cached)
    # ... apply_colors() ...
    # ... render() ...       # 5s (no mesh generation)

    # Tweak lighting, re-render (mesh cached)
    # ... setup_lighting() ...
    # ... render() ...       # 5s (no mesh generation)

Performance Notes
-----------------

**Typical speedups:**

- Mesh generation: 10-60s (depending on vertex count)
- Cache load: 1-3s (loading .blend file)
- Speedup: 10-50x for render-only passes

**Cache overhead:**

- Hash computation: ~10-50ms
- .blend file copy: ~100-500ms (depends on mesh size)
- Disk usage: ~5-50MB per cached mesh

**When to use MeshCache:**

- Multi-view renders (same mesh, different angles)
- Render iteration (lighting, camera, background)
- Development workflow (frequent re-renders)

**When NOT to use:**

- Single render (overhead > savings)
- Rapidly changing DEM data
- Limited disk space

Integration with Pipeline
--------------------------

MeshCache integrates with :class:`~terrain.pipeline.TerrainPipeline`::

    from src.terrain.pipeline import TerrainPipeline

    pipeline = TerrainPipeline(
        dem_dir='data/dem',
        cache_enabled=True  # Enables MeshCache
    )

    # Mesh cached automatically
    pipeline.execute('render_view')

See Also
--------

- :doc:`cache` - DEM and transform caching
- :doc:`pipeline` - Full pipeline caching with mesh cache integration
