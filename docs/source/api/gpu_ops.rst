GPU Operations Module
=====================

GPU-accelerated terrain processing operations using PyTorch.

This module provides GPU-accelerated versions of common terrain processing operations.
Functions automatically use CUDA when available, falling back to CPU otherwise.

All functions accept numpy arrays and return numpy arrays for easy integration.

**Requirements:** PyTorch with CUDA support (``pip install torch``)

Slope Calculation
-----------------

.. autofunction:: src.terrain.gpu_ops.gpu_horn_slope

   Calculate slope magnitude using Horn's method with GPU acceleration.

   **Performance:**

   - GPU (CUDA): ~7x faster than CPU
   - Typical: 100ms for 4096×4096 DEM on GPU vs 700ms on CPU

   **How it works:**

   - Uses PyTorch's ``F.conv2d`` for efficient convolution
   - Sobel-like kernels for gradient estimation
   - Handles NaN values via interpolation

   Example::

       from src.terrain.gpu_ops import gpu_horn_slope

       # Calculate slopes (auto-detects GPU)
       slopes = gpu_horn_slope(dem_data)

       # Results identical to scipy.ndimage implementation
       print(f"Max slope: {slopes.max():.2f}°")

   Used by :func:`~terrain.advanced_viz.horn_slope`.

Filtering Operations
--------------------

.. autofunction:: src.terrain.gpu_ops.gpu_gaussian_blur

   GPU-accelerated Gaussian blur using separable convolution.

   **Performance:**

   - GPU: ~10-20x faster than scipy.ndimage for large arrays
   - Uses separable kernels for efficiency (2 passes vs 2D kernel)

   Example::

       from src.terrain.gpu_ops import gpu_gaussian_blur

       # Smooth DEM with sigma=2.0
       smoothed = gpu_gaussian_blur(dem_data, sigma=2.0)

.. autofunction:: src.terrain.gpu_ops.gpu_median_filter

   GPU-accelerated median filter for noise removal.

   **Performance:**

   - GPU: ~5-10x faster than scipy.ndimage
   - Uses PyTorch's ``unfold`` + median for efficiency

   Example::

       from src.terrain.gpu_ops import gpu_median_filter

       # Remove salt-and-pepper noise
       cleaned = gpu_median_filter(dem_data, kernel_size=3)

.. autofunction:: src.terrain.gpu_ops.gpu_max_filter

   GPU-accelerated maximum filter (morphological dilation).

   **Performance:**

   - GPU: ~15-25x faster than scipy.ndimage
   - Uses ``max_pool2d`` for optimal GPU utilization

   Example::

       from src.terrain.gpu_ops import gpu_max_filter

       # Morphological dilation
       dilated = gpu_max_filter(dem_data, kernel_size=5)

.. autofunction:: src.terrain.gpu_ops.gpu_min_filter

   GPU-accelerated minimum filter (morphological erosion).

   **Performance:**

   - GPU: ~15-25x faster than scipy.ndimage
   - Uses ``max_pool2d`` on negated data

   Example::

       from src.terrain.gpu_ops import gpu_min_filter

       # Morphological erosion
       eroded = gpu_min_filter(dem_data, kernel_size=5)

Device Management
-----------------

.. autofunction:: src.terrain.gpu_ops._get_device

   Get best available device (CUDA > CPU).

   Automatically detects CUDA availability.

   Example::

       from src.terrain.gpu_ops import _get_device

       device = _get_device()
       print(f"Using device: {device}")  # "cuda" or "cpu"

Performance Notes
-----------------

**GPU vs CPU speedups:**

+------------------+------------+---------------+
| Operation        | Array Size | GPU Speedup   |
+==================+============+===============+
| Horn slope       | 4096²      | 7x            |
| Gaussian blur    | 4096²      | 10-20x        |
| Median filter    | 4096²      | 5-10x         |
| Max/Min filter   | 4096²      | 15-25x        |
+------------------+------------+---------------+

**Memory requirements:**

- GPU operations require ~3-4x array size in VRAM
- 4096×4096 float32 array: ~64MB → ~200MB VRAM needed
- Most operations fall back to CPU if VRAM insufficient

**When to use GPU ops:**

- Large arrays (>2048×2048)
- Repeated operations (amortize data transfer)
- Batch processing
- Real-time/interactive applications

**When NOT to use:**

- Small arrays (<1024×1024) - overhead > speedup
- Limited VRAM
- CPU-only systems (auto-fallback but no benefit)

Integration Example
-------------------

Integrate GPU ops into terrain pipeline::

    from src.terrain.gpu_ops import gpu_horn_slope, gpu_gaussian_blur
    from src.terrain.transforms import slope_adaptive_smooth

    # Load DEM
    dem = load_dem_files('data/hgt')

    # GPU-accelerated slope calculation
    slopes = gpu_horn_slope(dem)

    # GPU-accelerated smoothing
    dem_smooth = gpu_gaussian_blur(dem, sigma=2.0)

    # Use in pipeline
    terrain = Terrain(dem_smooth, transform)
    # ...

See Also
--------

- :doc:`advanced_viz` - Uses gpu_horn_slope for slope calculation
- :doc:`transforms` - CPU-based transform functions
