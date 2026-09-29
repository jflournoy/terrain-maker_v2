# Diagnostic Histograms

Terrain Maker writes a few histograms that make it possible to check a render against the
numbers behind it, without opening Blender.

## Score histograms

`generate_score_histogram()` plots raw scores next to the same scores after the transform
that maps them onto the colormap (normalization, optional stretch, gamma), with bars colored
by the colormap itself. Pass `title` to say which scores are shown.

`examples/detroit_combined_render.py` writes two, both through the same normalization as
the rendered colors:

| File | Shows |
| --- | --- |
| `histograms/scores_before_masking.png` | Base scores before lakes are blanked and near-zero scores are floored |
| `histograms/scores_as_rendered.png` | Base scores exactly as colored in the render |

Comparing the two shows what lake masking and the score floor changed.

## Render histograms

After a render, the combined render also writes, next to the image:

- `<image>_histogram.png`: RGB channel histograms (`generate_rgb_histogram()`)
- `<image>_luminance.png`: luminance histogram (`generate_luminance_histogram()`)

These check color balance, exposure and clipping, which matters most for print output.

## API reference

See the [diagnostics module reference](../api/diagnostics.rst).
