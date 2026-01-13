Transforms Module
=================

Transform functions for processing DEM and score data.

Geographic Transforms
---------------------

.. autofunction:: src.terrain.transforms.reproject_raster

   Used in :doc:`../examples/elevation` to convert WGS84 to UTM.

.. autofunction:: src.terrain.transforms.flip_raster

.. autofunction:: src.terrain.transforms.scale_elevation

.. autofunction:: src.terrain.transforms.downsample_raster

Smoothing Transforms
--------------------

.. autofunction:: src.terrain.transforms.feature_preserving_smooth

   Bilateral filter that preserves ridges and edges.

.. autofunction:: src.terrain.transforms.wavelet_denoise_dem

   Frequency-aware denoising. Used in :doc:`../examples/combined_render`.

   Example::

       terrain.add_transform(wavelet_denoise_dem(
           wavelet='db4',
           levels=3,
           threshold_sigma=2.0
       ))

.. autofunction:: src.terrain.transforms.slope_adaptive_smooth

   Smooths flat areas (buildings) while preserving hills.
   Used in :doc:`../examples/combined_render`.

.. autofunction:: src.terrain.transforms.remove_bumps

   Morphological opening to remove local maxima.

.. autofunction:: src.terrain.transforms.despeckle_dem

   Median filter for isolated outliers.

Score Data Transforms
---------------------

.. autofunction:: src.terrain.transforms.smooth_score_data

   Reduces blockiness in low-resolution SNODAS data.

.. autofunction:: src.terrain.transforms.despeckle_scores

.. autofunction:: src.terrain.transforms.upscale_scores

   AI super-resolution for score data.

Caching
-------

.. autoclass:: src.terrain.cache.TransformCache
   :members:

   Example::

       from src.terrain.cache import TransformCache

       cache = TransformCache(cache_dir='.cache')
       result = cache.get_or_compute(
           'my_transform',
           compute_fn=lambda: expensive_operation(),
           params={'key': 'value'}
       )
