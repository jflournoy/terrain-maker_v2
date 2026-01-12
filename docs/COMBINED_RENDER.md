# Combined Render Example

The `detroit_combined_render.py` script is our most comprehensive example, demonstrating all major features of terrain-maker in a single pipeline. It combines sledding and cross-country skiing suitability scores into a dual-colormap visualization with roads, print-quality rendering, and advanced diagnostics.

## Quick Start

```bash
# Fast preview (640x360, ~10 seconds)
python examples/detroit_combined_render.py

# Print quality (4K+ resolution, ~5 minutes)
python examples/detroit_combined_render.py --print-quality

# With roads and smoothing
python examples/detroit_combined_render.py --roads --smooth --wavelet-denoise
```

## Output Files

| File | Description |
|------|-------------|
| `sledding_with_xc_parks_3d.png` | Preview render (640x360) |
| `sledding_with_xc_parks_3d_print.png` | Print render (18x12 inches @ 300 DPI) |
| `*_histogram.png` | RGB channel histogram |
| `*_luminance.png` | Grayscale luminance histogram |
| `diagnostics/` | Optional diagnostic plots |

## Feature Categories

### 1. Quality Modes

#### Preview Mode (Default)

```bash
python examples/detroit_combined_render.py
```

- Resolution: 640x360 pixels
- Samples: 64 (with denoising)
- Vertex multiplier: 0.5x
- Render time: ~10 seconds

#### Print Quality

```bash
python examples/detroit_combined_render.py --print-quality \
    --print-dpi 300 \
    --print-width 18 \
    --print-height 12
```

- Resolution: 5400x3600 pixels (18x12 inches @ 300 DPI)
- Samples: 4096 (with denoising)
- Vertex multiplier: 2.5x
- Render time: ~5 minutes

#### Custom Resolution

```bash
# Higher detail mesh for print
python examples/detroit_combined_render.py --print-quality \
    --vertex-multiplier 4.0 \
    --print-dpi 600
```

### 2. Output Format

```bash
# PNG (default, lossless, with alpha)
python examples/detroit_combined_render.py --format png

# JPEG (smaller files, for print)
python examples/detroit_combined_render.py --format jpeg \
    --jpeg-quality 95 \
    --embed-profile  # Embed sRGB ICC for print shops
```

### 3. Dual Colormap System

The combined render uses two colormaps blended by proximity to parks:

```
Base layer:    Sledding scores → Mako colormap (purple → yellow)
Overlay layer: XC skiing scores → Rocket colormap (near parks only)
```

Parks create 2.5km proximity zones where XC skiing scores show through. This creates a visualization showing:

- General sledding terrain suitability (everywhere)
- XC skiing conditions (highlighted near parks)

### 4. Road Rendering

```bash
# Enable roads (interstate and primary highways)
python examples/detroit_combined_render.py --roads

# Customize road types (OSM highway tags)
python examples/detroit_combined_render.py --roads \
    --road-types motorway trunk primary secondary

# Adjust road appearance
python examples/detroit_combined_render.py --roads \
    --road-width 5 \
    --road-color azurite \
    --road-antialias 1.0
```

**Road color presets:**

- `obsidian` - Glossy black
- `azurite` - Deep blue (default)
- `azurite-light` - Richer blue
- `malachite` - Deep green
- `hematite` - Dark iron gray

**Road smoothing options:**

```bash
# Smooth road vertex elevations (reduces bumpiness)
--road-smoothing --road-smoothing-radius 2

# Fixed Z offset for roads
--road-offset 0.5  # Raise roads slightly above terrain
```

### 5. DEM Smoothing Pipeline

Multiple smoothing options can be combined for optimal terrain appearance:

#### Feature-Preserving Bilateral Smoothing

```bash
python examples/detroit_combined_render.py --smooth \
    --smooth-spatial 3.0 \
    --smooth-intensity 10.0
```

Preserves ridges and edges while smoothing flat areas.

#### Wavelet Denoising

```bash
python examples/detroit_combined_render.py --wavelet-denoise \
    --wavelet-type db4 \
    --wavelet-levels 3 \
    --wavelet-sigma 2.0 \
    --wavelet-diagnostics  # Export before/after comparison
```

Separates terrain structure from high-frequency noise. Best for sensor noise.

#### DEM Despeckle (Median Filter)

```bash
python examples/detroit_combined_render.py --despeckle-dem \
    --despeckle-dem-kernel 3
```

Removes isolated outlier pixels. More uniform than bilateral smoothing.

#### Slope-Adaptive Smoothing

```bash
python examples/detroit_combined_render.py --adaptive-smooth \
    --adaptive-slope-threshold 2.0 \
    --adaptive-smooth-sigma 5.0 \
    --adaptive-transition 1.0 \
    --adaptive-edge-threshold 5.0
```

Strong smoothing in flat areas (removes building bumps), minimal on slopes.

#### Morphological Bump Removal

```bash
python examples/detroit_combined_render.py --remove-bumps 2 \
    --remove-bumps-strength 0.7
```

Removes local maxima (buildings, trees) via morphological opening.

### 6. Score Data Enhancement

SNODAS snow data is lower resolution than the DEM. These options improve score visualization:

#### Score Smoothing (Bilateral)

```bash
python examples/detroit_combined_render.py --smooth-scores \
    --smooth-scores-spatial 5.0 \
    --smooth-scores-intensity 0.15
```

Reduces blockiness from low-res SNODAS data.

#### Score Despeckle

```bash
python examples/detroit_combined_render.py --despeckle-scores \
    --despeckle-kernel 3
```

Removes isolated speckles before upscaling.

#### AI Super-Resolution Upscaling

```bash
# Automatic upscale factor to match DEM resolution
python examples/detroit_combined_render.py --upscale-scores \
    --upscale-to-dem

# Or specify factor manually
python examples/detroit_combined_render.py --upscale-scores \
    --upscale-factor 4 \
    --upscale-method auto  # esrgan, bilateral, bicubic
```

### 7. Camera and Lighting

#### Camera Positioning

```bash
# Cardinal directions
python examples/detroit_combined_render.py \
    --camera-direction south \
    --camera-elevation 0.3 \
    --ortho-scale 7.0

# Options: north, south, east, west, northeast, northwest, etc.
```

#### HDRI Sky Lighting (Default)

```bash
python examples/detroit_combined_render.py \
    --sun-azimuth 225 \    # SW direction
    --sun-elevation 30 \    # 30 degrees above horizon
    --sun-energy 7.0 \
    --air-density 0.5 \     # Less haze
    --sky-intensity 1.0
```

#### Two-Point Lighting

```bash
python examples/detroit_combined_render.py \
    --fill-energy 2.0 \     # Enable fill light
    --fill-azimuth 45 \     # NE direction
    --fill-elevation 60
```

### 8. Background Plane

```bash
python examples/detroit_combined_render.py --background \
    --background-color white \
    --background-distance 0.5 \
    --background-flat  # No shadows on background
```

**Color options:** `white`, `light_gray`, `paper`, `cream`, hex (`#F5F5F5`)

### 9. Test Materials

Debug lighting and shading without vertex colors:

```bash
python examples/detroit_combined_render.py \
    --test-material clay  # Matte gray

# Options: none, obsidian, chrome, clay, plastic, gold
```

### 10. Memory Management

For large prints on limited GPU VRAM:

```bash
python examples/detroit_combined_render.py --print-quality \
    --auto-tile \
    --tile-size 1024 \
    --persistent-data
```

### 11. Pipeline Caching

Speed up repeated renders by caching intermediate results:

```bash
# Enable caching
python examples/detroit_combined_render.py --cache

# Clear and rebuild cache
python examples/detroit_combined_render.py --cache --clear-cache

# Custom cache directory
python examples/detroit_combined_render.py --cache \
    --cache-dir /tmp/terrain_cache
```

Cache invalidates automatically when upstream parameters change.

### 12. Diagnostics

```bash
# Export all diagnostic plots
python examples/detroit_combined_render.py \
    --wavelet-denoise --wavelet-diagnostics \
    --adaptive-smooth \
    --diagnostic-dir ./diagnostics

# Skip rendering, just generate diagnostics
python examples/detroit_combined_render.py --no-render \
    --wavelet-denoise --wavelet-diagnostics
```

**Diagnostic outputs:**

- `dem_wavelet_comparison.png` - Before/after wavelet denoising
- `dem_wavelet_coefficients.png` - Wavelet coefficient analysis
- `dem_adaptive_smooth_*.png` - Slope-based smoothing visualization
- `dem_bump_removal_*.png` - Bump removal effect
- `sledding_upscale_*.png` - Score upscaling comparison
- `*_histogram.png` - RGB channel histogram
- `*_luminance.png` - Luminance histogram

## Complete Example

Production-ready print with all enhancements:

```bash
python examples/detroit_combined_render.py \
    --print-quality \
    --print-dpi 300 \
    --print-width 24 \
    --print-height 18 \
    --format jpeg \
    --jpeg-quality 95 \
    --embed-profile \
    --roads \
    --road-color azurite \
    --road-width 3 \
    --road-antialias 1.0 \
    --wavelet-denoise \
    --wavelet-type db4 \
    --wavelet-levels 3 \
    --adaptive-smooth \
    --adaptive-slope-threshold 2.0 \
    --upscale-scores \
    --upscale-to-dem \
    --despeckle-scores \
    --background \
    --background-color white \
    --camera-direction south \
    --sun-azimuth 225 \
    --air-density 0.5 \
    --cache \
    --auto-tile \
    --tile-size 2048 \
    --output-dir ./output/print
```

## CLI Reference

### Basic Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir` | `docs/images/combined_render/roads` | Output directory |
| `--scores-dir` | `examples/output` | Score data directory |
| `--mock-data` | False | Use mock data (no real files needed) |
| `--no-render` | False | Skip rendering (scene setup only) |

### Quality Options

| Flag | Default | Description |
|------|---------|-------------|
| `--print-quality` | False | High-res print mode |
| `--print-dpi` | 300 | Print resolution |
| `--print-width` | 18 | Print width in inches |
| `--print-height` | 12 | Print height in inches |
| `--vertex-multiplier` | auto | Mesh detail multiplier |
| `--downsample-method` | `average` | DEM resampling: average, lanczos, cubic, bilinear |

### Format Options

| Flag | Default | Description |
|------|---------|-------------|
| `--format` | `png` | Output format: png, jpeg |
| `--jpeg-quality` | 95 | JPEG quality (1-100) |
| `--embed-profile` | False | Embed sRGB ICC profile |

### Camera Options

| Flag | Default | Description |
|------|---------|-------------|
| `--camera-direction` | `south` | View direction |
| `--camera-elevation` | 0.3 | Height as fraction of diagonal |
| `--ortho-scale` | 7.0 | Orthographic camera scale |
| `--height-scale` | 30.0 | Vertical exaggeration |

### Lighting Options

| Flag | Default | Description |
|------|---------|-------------|
| `--sun-azimuth` | 225 | Sun direction (degrees from N) |
| `--sun-elevation` | 30 | Sun angle above horizon |
| `--sun-energy` | 7.0 | Sun brightness |
| `--sun-angle` | 1.0 | Sun size (shadow softness) |
| `--fill-energy` | 0.0 | Fill light (0=disabled) |
| `--sky-intensity` | 1.0 | Ambient sky brightness |
| `--air-density` | 1.0 | Atmospheric scattering |
| `--hdri-lighting/--no-hdri-lighting` | True | HDRI sky system |

### Background Options

| Flag | Default | Description |
|------|---------|-------------|
| `--background` | False | Enable background plane |
| `--background-color` | `white` | Plane color |
| `--background-distance` | 0.5 | Distance below terrain |
| `--background-flat` | False | Disable shadows on background |

### Road Options

| Flag | Default | Description |
|------|---------|-------------|
| `--roads/--no-roads` | False | Enable road rendering |
| `--road-types` | motorway trunk primary | OSM highway types |
| `--road-width` | 3 | Road width in pixels |
| `--road-color` | `azurite` | Road color preset |
| `--road-antialias` | 0.0 | Edge smoothing sigma |
| `--road-smoothing` | False | Smooth road elevations |
| `--road-smoothing-radius` | 2 | Smoothing neighborhood |
| `--road-offset` | 0.0 | Fixed Z offset for roads |

### DEM Smoothing Options

| Flag | Default | Description |
|------|---------|-------------|
| `--smooth` | False | Bilateral smoothing |
| `--smooth-spatial` | 3.0 | Spatial sigma |
| `--smooth-intensity` | auto | Intensity sigma |
| `--despeckle-dem` | False | Median filter denoising |
| `--despeckle-dem-kernel` | 3 | Kernel size |
| `--wavelet-denoise` | False | Wavelet denoising |
| `--wavelet-type` | `db4` | Wavelet family |
| `--wavelet-levels` | 3 | Decomposition levels |
| `--wavelet-sigma` | 2.0 | Noise threshold |
| `--adaptive-smooth` | False | Slope-adaptive smoothing |
| `--adaptive-slope-threshold` | 2.0 | Slope threshold (degrees) |
| `--adaptive-smooth-sigma` | 5.0 | Smoothing sigma |
| `--adaptive-transition` | 1.0 | Transition width |
| `--adaptive-edge-threshold` | None | Edge preservation (meters) |
| `--remove-bumps` | None | Morphological opening size |
| `--remove-bumps-strength` | 1.0 | Effect strength (0-1) |

### Score Enhancement Options

| Flag | Default | Description |
|------|---------|-------------|
| `--smooth-scores` | False | Bilateral score smoothing |
| `--smooth-scores-spatial` | 5.0 | Spatial sigma |
| `--smooth-scores-intensity` | auto | Intensity sigma |
| `--despeckle-scores` | False | Median filter for scores |
| `--despeckle-kernel` | 3 | Kernel size |
| `--upscale-scores` | False | AI super-resolution |
| `--upscale-factor` | 4 | Scale factor (2, 4, 8, 16) |
| `--upscale-to-dem` | False | Auto-calculate factor |
| `--upscale-method` | `auto` | Method: auto, esrgan, bilateral, bicubic |

### Cache Options

| Flag | Default | Description |
|------|---------|-------------|
| `--cache/--no-cache` | False | Enable pipeline caching |
| `--clear-cache` | False | Clear cache before run |
| `--cache-dir` | `.pipeline_cache` | Cache directory |

### Memory Options

| Flag | Default | Description |
|------|---------|-------------|
| `--auto-tile` | False | GPU tile rendering |
| `--tile-size` | 2048 | Tile size in pixels |
| `--persistent-data` | False | Keep data in VRAM |

### Diagnostic Options

| Flag | Default | Description |
|------|---------|-------------|
| `--wavelet-diagnostics` | False | Export wavelet plots |
| `--diagnostic-dir` | output/diagnostics | Diagnostic output path |
| `--test-material` | `none` | Debug material |

## Pipeline Architecture

```
1. LOAD DATA
   ├── DEM tiles (SRTM HGT files, filtered to N41+)
   ├── Sledding scores (from snow analysis pipeline)
   ├── XC skiing scores (from skiing analysis pipeline)
   ├── Park locations (GeoJSON)
   └── Road data (OpenStreetMap via Overpass API)

2. PREPROCESS SCORES (before terrain transforms)
   ├── Despeckle scores (median filter)
   └── Upscale scores (AI super-resolution)

3. TRANSFORM DEM (with caching)
   ├── Reproject (WGS84 → UTM Zone 17N)
   ├── Flip horizontal (for correct orientation)
   └── Downsample (target vertex count)

4. DETECT WATER (on pre-smoothed DEM)
   └── Slope-based water detection (slope ≈ 0)

5. SMOOTH DEM (after water detection)
   ├── Feature-preserving bilateral
   ├── Despeckle (median filter)
   ├── Wavelet denoising
   ├── Slope-adaptive smoothing
   ├── Bump removal (morphological)
   └── Scale elevation (0.0001 factor)

6. ADD DATA LAYERS
   ├── Sledding scores (aligned to DEM)
   ├── XC skiing scores (aligned to DEM)
   ├── Roads (rasterized to DEM grid)
   └── Optional: Smooth score layers

7. COMPUTE COLORS
   ├── Base: Sledding → Mako colormap (gamma=0.5)
   ├── Overlay: XC skiing → Rocket colormap
   ├── Mask: Park proximity (2.5km radius)
   └── Water: University of Michigan blue

8. CREATE MESH
   ├── Generate vertices from DEM
   ├── Apply vertex colors
   ├── Apply road mask (for material)
   ├── Smooth road vertices (optional)
   └── Apply materials (roads get glossy shader)

9. SETUP SCENE
   ├── Position camera (cardinal direction)
   ├── HDRI sky lighting (sun + ambient)
   ├── Fill light (optional)
   └── Background plane (optional)

10. RENDER
    ├── Configure GPU + denoising
    ├── Render to PNG/JPEG
    ├── Embed ICC profile (optional)
    └── Generate histograms
```

## See Also

- [EXAMPLES.md](EXAMPLES.md) - Other example scripts
- [API_REFERENCE.md](API_REFERENCE.md) - Full API documentation
- [QUICK_START.md](QUICK_START.md) - Getting started guide

---

**Last Updated:** 2026-01-12
