Transforms Module
=================

Transform functions for processing DEM and score data.

Geographic Transforms
---------------------

.. autofunction:: terrain_maker.terrain.transforms.reproject_raster

   Used in :doc:`../examples/elevation` to convert WGS84 to UTM.

.. autofunction:: terrain_maker.terrain.transforms.flip_raster

.. autofunction:: terrain_maker.terrain.transforms.scale_elevation

.. autofunction:: terrain_maker.terrain.transforms.downsample_raster

Smoothing Transforms
--------------------

.. autofunction:: terrain_maker.terrain.transforms.feature_preserving_smooth

   Bilateral filter that preserves ridges and edges.

.. autofunction:: terrain_maker.terrain.transforms.wavelet_denoise_dem

   Frequency-aware denoising. Used in :doc:`../examples/combined_render`.

   Example::

       terrain.add_transform(wavelet_denoise_dem(
           wavelet='db4',
           levels=3,
           threshold_sigma=2.0
       ))

.. autofunction:: terrain_maker.terrain.transforms.slope_adaptive_smooth

   Smooths flat areas (buildings) while preserving hills.
   Used in :doc:`../examples/combined_render`.

.. autofunction:: terrain_maker.terrain.transforms.remove_bumps

   Morphological opening to remove local maxima.

.. autofunction:: terrain_maker.terrain.transforms.despeckle_dem

   Median filter for isolated outliers.

Score Data Transforms
---------------------

.. autofunction:: terrain_maker.terrain.transforms.smooth_score_data

   Reduces blockiness in low-resolution SNODAS data.

.. autofunction:: terrain_maker.terrain.transforms.despeckle_scores

.. autofunction:: terrain_maker.terrain.transforms.upscale_scores

   AI super-resolution for score data.

Caching
-------

.. autoclass:: terrain_maker.terrain.cache.TransformCache
   :members:

   Example::

       from terrain_maker.terrain.cache import TransformCache

       cache = TransformCache(cache_dir='.cache')
       result = cache.get_or_compute(
           'my_transform',
           compute_fn=lambda: expensive_operation(),
           params={'key': 'value'}
       )
