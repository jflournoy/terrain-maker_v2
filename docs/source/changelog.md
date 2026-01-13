# Changelog

## v1.0 (2026-01-12)

### Features

- **Core**
  - `Terrain` class for DEM processing
  - Multi-layer data system with auto-reprojection
  - Configurable mesh density via `configure_for_target_vertices()`

- **Transforms**
  - Wavelet denoising (`wavelet_denoise_dem`)
  - Slope-adaptive smoothing (`slope_adaptive_smooth`)
  - Morphological bump removal (`remove_bumps`)
  - Score upscaling with AI super-resolution

- **Rendering**
  - HDRI sky lighting with Nishita model
  - Cardinal direction camera positioning
  - Orthographic and perspective cameras
  - Print-quality output with ICC profiles

- **Roads**
  - OpenStreetMap road fetching
  - Glossy road materials
  - Road vertex smoothing

- **Diagnostics**
  - RGB/luminance histograms
  - Wavelet coefficient analysis
  - Upscale comparison plots

### Performance

- Numba JIT for 2-3x faster mesh generation
- Transform caching system
- Memory-efficient tiled processing
