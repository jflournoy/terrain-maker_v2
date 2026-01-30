Diagnostics Module
==================

Diagnostic plotting utilities for debugging terrain processing.

This module provides visualization functions to understand and debug terrain transforms,
particularly wavelet denoising, slope-adaptive smoothing, bump removal, road elevation,
and upscaling operations.

All functions generate multi-panel matplotlib figures saved to disk.

Wavelet Denoising Diagnostics
------------------------------

.. autofunction:: src.terrain.diagnostics.plot_wavelet_diagnostics

   Generate diagnostic plots showing wavelet denoising effects.

   **Creates:**

   - Original DEM panel
   - Denoised DEM panel
   - Difference map (noise removed)
   - Cross-section profile comparison

   Example::

       from src.terrain.diagnostics import plot_wavelet_diagnostics

       plot_wavelet_diagnostics(
           original=dem_before,
           denoised=dem_after,
           output_path='diagnostics/wavelet.png',
           title_prefix='Wavelet Denoising (db4)',
           cmap='terrain'
       )

.. autofunction:: src.terrain.diagnostics.plot_wavelet_coefficients

   Visualize wavelet decomposition coefficients.

.. autofunction:: src.terrain.diagnostics.generate_full_wavelet_diagnostics

   Generate comprehensive wavelet diagnostic report.

Adaptive Smoothing Diagnostics
-------------------------------

.. autofunction:: src.terrain.diagnostics.plot_adaptive_smooth_diagnostics

   Generate diagnostic plots for slope-adaptive smoothing.

   **Creates:**

   - Original DEM
   - Smoothed DEM
   - Difference map
   - Slope mask (shows smoothed vs preserved areas)
   - Cross-section profiles

   Example::

       from src.terrain.diagnostics import plot_adaptive_smooth_diagnostics

       plot_adaptive_smooth_diagnostics(
           original=dem_before,
           smoothed=dem_after,
           slope_mask=slope_mask,
           output_path='diagnostics/adaptive_smooth.png',
           slope_threshold=2.0
       )

.. autofunction:: src.terrain.diagnostics.plot_adaptive_smooth_histogram

   Histogram showing slope distribution and smoothing threshold.

.. autofunction:: src.terrain.diagnostics.generate_full_adaptive_smooth_diagnostics

   Generate comprehensive adaptive smoothing diagnostic report.

Bump Removal Diagnostics
-------------------------

.. autofunction:: src.terrain.diagnostics.plot_bump_removal_diagnostics

   Visualize morphological bump removal effects.

   **Creates:**

   - Original DEM with bumps
   - DEM after bump removal
   - Difference map (bumps removed)
   - Histogram of height changes

   Example::

       from src.terrain.diagnostics import plot_bump_removal_diagnostics

       plot_bump_removal_diagnostics(
           original=dem_before,
           processed=dem_after,
           output_path='diagnostics/bump_removal.png',
           kernel_size=5
       )

.. autofunction:: src.terrain.diagnostics.generate_bump_removal_diagnostics

   Generate comprehensive bump removal diagnostic report.

Upscaling Diagnostics
----------------------

.. autofunction:: src.terrain.diagnostics.plot_upscale_diagnostics

   Visualize AI super-resolution upscaling results.

   **Creates:**

   - Low-resolution input
   - High-resolution output
   - Difference from nearest-neighbor
   - Sharpness metrics

   Example::

       from src.terrain.diagnostics import plot_upscale_diagnostics

       plot_upscale_diagnostics(
           low_res=score_input,
           high_res=score_upscaled,
           output_path='diagnostics/upscale.png',
           scale_factor=4
       )

.. autofunction:: src.terrain.diagnostics.generate_upscale_diagnostics

   Generate comprehensive upscaling diagnostic report.

Road Elevation Diagnostics
---------------------------

.. autofunction:: src.terrain.diagnostics.plot_road_elevation_diagnostics

   Visualize DEM smoothing effects along roads.

   **Creates:**

   - Original DEM elevations along roads
   - Smoothed DEM elevations
   - Height differences
   - Road network overlay

   Example::

       from src.terrain.diagnostics import plot_road_elevation_diagnostics

       plot_road_elevation_diagnostics(
           original_dem=dem_before,
           smoothed_dem=dem_after,
           road_mask=road_mask,
           output_path='diagnostics/road_elevation.png'
       )

.. autofunction:: src.terrain.diagnostics.generate_road_elevation_diagnostics

   Generate comprehensive road elevation diagnostic report.

.. autofunction:: src.terrain.diagnostics.plot_road_vertex_z_diagnostics

   Visualize vertex-level road elevation changes.

Pipeline Visualization
----------------------

.. autofunction:: src.terrain.diagnostics.plot_processing_pipeline

   Visualize multi-stage processing pipeline.

   Shows before/after for each processing step in sequence.

   Example::

       from src.terrain.diagnostics import plot_processing_pipeline

       plot_processing_pipeline(
           stages=[
               ('Original', dem_original),
               ('Wavelet Denoised', dem_wavelet),
               ('Adaptive Smoothed', dem_adaptive),
               ('Final', dem_final)
           ],
           output_path='diagnostics/pipeline.png'
       )

Histogram Generation
--------------------

.. autofunction:: src.terrain.diagnostics.generate_rgb_histogram

   Generate RGB channel histograms for rendered images.

   Useful for validating color balance and dynamic range.

   Example::

       from src.terrain.diagnostics import generate_rgb_histogram

       generate_rgb_histogram(
           image_path='render_output.png',
           output_path='diagnostics/rgb_histogram.png'
       )

.. autofunction:: src.terrain.diagnostics.generate_luminance_histogram

   Generate grayscale luminance histogram.

   Useful for checking exposure and contrast.

   Example::

       from src.terrain.diagnostics import generate_luminance_histogram

       generate_luminance_histogram(
           image_path='render_output.png',
           output_path='diagnostics/luminance_histogram.png'
       )

Usage Patterns
--------------

**Pattern 1: Debug single transform**

::

    from src.terrain.diagnostics import plot_wavelet_diagnostics

    # Apply transform
    denoised = wavelet_denoise_dem(dem, wavelet='db4')

    # Generate diagnostics
    plot_wavelet_diagnostics(
        original=dem,
        denoised=denoised,
        output_path='debug_wavelet.png'
    )

**Pattern 2: Full pipeline diagnostics**

::

    from src.terrain.diagnostics import plot_processing_pipeline

    stages = []
    stages.append(('Original', dem))

    # Step 1: Wavelet denoise
    dem = wavelet_denoise_dem(dem)
    stages.append(('Wavelet', dem))

    # Step 2: Adaptive smooth
    dem = slope_adaptive_smooth(dem)
    stages.append(('Adaptive', dem))

    # Step 3: Bump removal
    dem = remove_bumps(dem)
    stages.append(('Bumps Removed', dem))

    # Visualize entire pipeline
    plot_processing_pipeline(stages, 'pipeline.png')

**Pattern 3: Batch diagnostics**

::

    from src.terrain.diagnostics import generate_full_wavelet_diagnostics

    # Comprehensive diagnostic report (multiple plots)
    generate_full_wavelet_diagnostics(
        original=dem_before,
        denoised=dem_after,
        output_dir='diagnostics/wavelet/'
    )

See Also
--------

- :doc:`transforms` - Transform functions to diagnose
- :doc:`../guides/diagnostics` - Diagnostics usage guide
