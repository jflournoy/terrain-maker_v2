Diagnostics Module
==================

Histograms for checking scores and rendered images.

Each function writes a matplotlib figure to disk.

Score Histograms
----------------

.. autofunction:: terrain_maker.terrain.diagnostics.generate_score_histogram

   Raw scores beside the same scores after the normalization used for coloring
   (and optionally the print-safe colormap), with bars colored by the colormap.

   Example::

       from terrain_maker.terrain.diagnostics import generate_score_histogram

       generate_score_histogram(
           raw_scores=scores,
           transformed_scores=normalization.apply(scores),
           output_path="histograms/scores_as_rendered.png",
           cmap_name="boreal_mako",
           title="Base scores as rendered",
       )

Render Histograms
-----------------

.. autofunction:: terrain_maker.terrain.diagnostics.generate_rgb_histogram

   RGB channel histograms of a rendered image, for checking color balance and range.

.. autofunction:: terrain_maker.terrain.diagnostics.generate_luminance_histogram

   Luminance histogram of a rendered image, for checking exposure and contrast.

See Also
--------

- :doc:`../guides/diagnostics` - What the combined render writes and how to read it
