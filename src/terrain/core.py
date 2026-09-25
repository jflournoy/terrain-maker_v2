from __future__ import annotations

import json
from pathlib import Path

import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from typing import Optional, Dict, Any, Callable
import functools
import inspect

# Terrain's methods are grouped by concern into these mixins
from src.terrain._terrain_color import TerrainColorMixin
from src.terrain._terrain_mesh import TerrainMeshMixin
from src.terrain._terrain_proximity import TerrainProximityMixin
from src.terrain._terrain_water import TerrainWaterMixin

# Output handling is configured once for the whole package in _logging.py
logger = logging.getLogger(__name__)


def calculate_target_vertices(
    width: int,
    height: int,
    multiplier: float = 2.0,
) -> int:
    """
    Calculate target vertex count for optimal mesh density at a given render resolution.

    This helper calculates an appropriate number of vertices for terrain meshes
    based on the intended output resolution. Using ~2 vertices per output pixel
    ensures good detail without excessive geometry.

    Args:
        width: Render width in pixels
        height: Render height in pixels
        multiplier: Vertices per pixel (default: 2.0).
                   Higher values = more detail but slower renders.
                   - 1.0: Minimum detail (1 vertex per pixel)
                   - 2.0: Good balance for most renders (recommended)
                   - 3.0+: High detail for print or zoomed views

    Returns:
        int: Target vertex count for terrain mesh creation

    Example:
        ```python
        # For 1920x1080 render
        target = calculate_target_vertices(1920, 1080)  # ~4.1M vertices

        # For print quality (3000x2400 @ 300 DPI)
        target = calculate_target_vertices(3000, 2400, multiplier=2.5)  # ~18M vertices

        # Use with Terrain
        terrain.configure_for_target_vertices(target_vertices=target)
        ```
    """
    return int(width * height * multiplier)


# Functions that live in other modules but are also importable from core,
# e.g. `from src.terrain.core import setup_camera`. Resolved lazily (PEP 562)
# so importing core doesn't load every submodule up front.
_REEXPORTS = {
    "clear_scene": "src.terrain.scene_setup",
    "setup_camera": "src.terrain.scene_setup",
    "setup_light": "src.terrain.scene_setup",
    "setup_camera_and_light": "src.terrain.scene_setup",
    "setup_two_point_lighting": "src.terrain.scene_setup",
    "position_camera_relative": "src.terrain.scene_setup",
    "setup_world_atmosphere": "src.terrain.scene_setup",
    "setup_hdri_lighting": "src.terrain.scene_setup",
    "setup_render_settings": "src.terrain.rendering",
    "render_scene_to_file": "src.terrain.rendering",
    "get_render_settings_report": "src.terrain.rendering",
    "print_render_settings_report": "src.terrain.rendering",
    "apply_colormap_material": "src.terrain.materials",
    "apply_water_shader": "src.terrain.materials",
    "create_background_plane": "src.terrain.materials",
    "load_dem_files": "src.terrain.data_loading",
    "load_filtered_hgt_files": "src.terrain.data_loading",
    "reproject_raster": "src.terrain.transforms",
    "smooth_raster": "src.terrain.transforms",
    "flip_raster": "src.terrain.transforms",
    "scale_elevation": "src.terrain.transforms",
    "feature_preserving_smooth": "src.terrain.transforms",
    "smooth_score_data": "src.terrain.transforms",
    "slope_colormap": "src.terrain.color_mapping",
    "elevation_colormap": "src.terrain.color_mapping",
}


def __getattr__(name):
    module = _REEXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)


def __dir__():
    return sorted(set(globals()) | set(_REEXPORTS))


def downsample_raster(zoom_factor=0.1, method="average", nodata_value=np.nan, optimized=True):
    """
    Create a raster downsampling transform function with specified parameters.

    For large compression ratios (>100:1), automatically uses two-pass downsampling
    for improved performance (expected ~35x speedup on billion-pixel DEMs).

    Args:
        zoom_factor: Scaling factor for downsampling (default: 0.1)
        method: Downsampling method (default: "average")
            - "average": Area averaging - best for DEMs, no overshoot
            - "lanczos": Lanczos resampling - sharp, minimal aliasing
            - "cubic": Cubic spline interpolation
            - "bilinear": Bilinear interpolation - safe fallback
        nodata_value: Value to treat as no data (default: np.nan)
        optimized: Use two-pass optimization for large compressions (default: True)

    Returns:
        function: A transform function that downsamples raster data
    """
    from src.terrain.transforms import (
        downsample_raster as _downsample_raster,
        downsample_raster_optimized as _downsample_raster_optimized,
    )

    if optimized:
        return _downsample_raster_optimized(zoom_factor, method, nodata_value)
    else:
        return _downsample_raster(zoom_factor, method, nodata_value)


def transform_wrapper(transform_func):
    """
    Standardize transform function interface with consistent output

    Args:
        transform_func: The original transform function to wrap

    Returns:
        A wrapped function with consistent signature and return format
    """

    @functools.wraps(transform_func)
    def wrapped_transform(data: np.ndarray, transform: rasterio.Affine = None) -> tuple:
        """
        Standardized transform wrapper with consistent signature

        Args:
            data: Input numpy array to transform
            transform: Optional affine transform

        Returns:
            Tuple of (transformed_data, transform, [crs]) where CRS is optional
        """
        # Inspect function signature to determine how to call
        sig = inspect.signature(transform_func)
        params = list(sig.parameters.keys())

        try:
            # Initialize result variables
            transformed_data = None
            final_transform = transform
            crs = None

            # Case 1: Transform takes only data
            if len(params) == 1 and params[0] == "data":
                transformed_data = transform_func(data)

            # Case 2: Transform takes (data, transform)
            elif len(params) == 2 and params[0] == "data" and params[1] == "transform":
                result = transform_func(data, transform)

                # Handle different return types
                if isinstance(result, tuple):
                    if len(result) == 3:
                        # When transform returns (data, transform, crs)
                        transformed_data, final_transform, crs = result
                    elif len(result) == 2:
                        # When transform returns (data, transform)
                        transformed_data, final_transform = result
                    else:
                        transformed_data = result[0]
                else:
                    transformed_data = result

            # Case 3: More complex signature or other parameters
            else:
                result = transform_func(data, transform)

                # Handle different return types
                if isinstance(result, tuple):
                    if len(result) == 3:
                        # When transform returns (data, transform, crs)
                        transformed_data, final_transform, crs = result
                    elif len(result) == 2:
                        # When transform returns (data, transform)
                        transformed_data, final_transform = result
                    else:
                        transformed_data = result[0]
                else:
                    transformed_data = result

            # Return standardized format: always a 3-tuple with optional None for crs
            return transformed_data, final_transform, crs

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error in transform {transform_func.__name__}: {e}")
            raise

    return wrapped_transform


class TerrainCache:
    """
    Cache manager for terrain data processing results.

    Handles persistent storage and retrieval of transformed terrain data layers
    as GeoTIFF files with geographic metadata. Supports loading and saving with
    coordinate reference system (CRS) and custom metadata.

    Attributes:
        cache_dir (Path): Root directory for cached GeoTIFF files.
        logger (logging.Logger): Logger instance for cache operations.

    Examples:
        >>> cache = TerrainCache('my_cache_dir')
        >>> cache.save('dem_transformed', dem_array, transform, 'EPSG:32617')
        >>> data, transform, crs = cache.load('dem_transformed')
    """

    def __init__(self, cache_dir: str = "terrain_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def get_target_path(self, target_name: str) -> Path:
        """Get path for a specific target"""
        return self.cache_dir / f"{target_name}.tif"

    def exists(self, target_name: str) -> bool:
        """Check if target exists"""
        return self.get_target_path(target_name).exists()

    def save(self, target_name: str, data: np.ndarray, transform, crs="EPSG:4326", metadata=None):
        """Save data as GeoTIFF with CRS and metadata"""
        path = self.get_target_path(target_name)
        self.logger.info(f"Saving {target_name}")

        # Save main raster data
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs=crs,
            transform=transform,
            compress="lzw",
        ) as dst:
            dst.write(data, 1)

            # Save metadata if provided
            if metadata:
                # Convert any non-string values to strings for GDAL metadata
                string_metadata = {k: str(v) for k, v in metadata.items()}
                dst.update_tags(**string_metadata)

        # For more complex metadata that can't be stored in GDAL tags
        if metadata:
            # Save to a companion JSON file
            meta_path = path.with_suffix(".json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f)

    def load(self, target_name: str) -> Optional[Dict[str, Any]]:
        """Load GeoTIFF and metadata if it exists"""
        path = self.get_target_path(target_name)
        if not path.exists():
            return None

        self.logger.info(f"Loading {target_name}")

        # Load raster data
        with rasterio.open(path) as src:
            data = src.read(1)
            transform = src.transform
            crs = src.crs
            # Get basic metadata from tags
            basic_metadata = src.tags()

        # Try to load companion metadata file
        meta_path = path.with_suffix(".json")
        full_metadata = basic_metadata.copy()
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    full_metadata.update(json.load(f))
            except (json.JSONDecodeError, OSError) as e:
                self.logger.warning(f"Failed to load metadata file {meta_path}: {e}")

        return {"data": data, "transform": transform, "crs": crs, "metadata": full_metadata}


class Terrain(TerrainColorMixin, TerrainMeshMixin, TerrainProximityMixin, TerrainWaterMixin):
    """
    Core class for managing Digital Elevation Model (DEM) data and terrain operations.

    Handles loading, transforming, and visualizing terrain data from raster sources.
    Supports coordinate reprojection, downsampling, color mapping, and 3D mesh generation
    for Blender visualization. Uses efficient caching to avoid recomputation of transforms.

    Attributes:
        dem_shape (tuple): Shape of DEM array as (height, width).
        dem_transform (rasterio.Affine): Affine transform for geographic coordinates.
        data_layers (dict): Dictionary of data layers (DEM, overlays, derived data).
        transforms (list): List of transform functions to apply.
        vertices (np.ndarray): Vertex positions for generated mesh.
        vertex_colors (np.ndarray): RGBA colors for mesh vertices.

    Examples:
        >>> dem_data = np.random.rand(100, 100) * 1000
        >>> transform = rasterio.Affine.identity()
        >>> terrain = Terrain(dem_data, transform, dem_crs='EPSG:4326')
        >>> terrain.apply_transforms()
        >>> mesh = terrain.create_mesh(scale_factor=100.0)
    """

    def __init__(
        self,
        dem_data: np.ndarray,
        dem_transform: rasterio.Affine,
        dem_crs: str = "EPSG:4326",
        cache_dir: str = "terrain_cache",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize terrain from DEM data.

        Args:
            dem_data (np.ndarray): DEM array of shape (height, width) containing elevation values.
                Integer types are converted to float32. Must be 2D.
            dem_transform (rasterio.Affine): Affine transform mapping pixel coordinates to
                geographic coordinates.
            dem_crs (str): Coordinate reference system in EPSG format (default: 'EPSG:4326').
            cache_dir (str): Directory for caching computations (default: 'terrain_cache').
            logger (logging.Logger, optional): Logger instance for diagnostic output.

        Raises:
            TypeError: If dem_data is not a numpy array or has unsupported dtype.
            ValueError: If dem_data is not 2D.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("Initializing Terrain Cache")
        self.cache = TerrainCache(cache_dir)
        self.logger.info(f"Cache directory contents: {list(self.cache.cache_dir.glob('**/*'))}")

        self.logger.info("Initializing Terrain...")

        # Validate input
        if not isinstance(dem_data, np.ndarray):
            raise TypeError("dem_data must be a numpy array")
        if dem_data.ndim != 2:
            raise ValueError(f"dem_data must be 2D, got shape {dem_data.shape}")

        # Convert to float32 if needed
        if np.issubdtype(dem_data.dtype, np.integer):
            self.logger.info(f"Converting DEM data from {dem_data.dtype} to float32")
            dem_data = dem_data.astype(np.float32)
        elif not np.issubdtype(dem_data.dtype, np.floating):
            raise TypeError(f"Unsupported DEM data type: {dem_data.dtype}")

        # Store original DEM data and transform
        self.dem_transform = dem_transform

        self.dem_bounds = rasterio.transform.array_bounds(
            dem_data.shape[0], dem_data.shape[1], dem_transform
        )

        # Calculate resolution in meters
        self.resolution = (abs(dem_transform[0]) * 111320, abs(dem_transform[4]) * 111320)
        self.dem_shape = dem_data.shape

        # Initialize list of transforms and data layers
        self.transforms = []
        self.data_layers = {}

        # Track cumulative transform metadata (e.g., elevation scale factors)
        # This allows algorithms to compensate for prior transforms
        self.transform_metadata = {
            "elevation_scale": 1.0,  # Cumulative scale factor applied to elevation
        }

        # Initialize containers for processed data
        self.processed_dem = None  # Will hold transformed DEM data
        self.vertices = None  # Will hold final vertex positions
        self.faces = None  # Will hold face indices
        self.vertex_colors = None  # Will hold vertex colors

        self.add_data_layer("dem", dem_data, dem_transform, dem_crs)

        self.logger.info(f"Terrain initialized with DEM data:")
        self.logger.info(f"  Shape: {dem_data.shape}")
        self.logger.info(f"  Resolution: {self.resolution[0]:.2f}m x {self.resolution[1]:.2f}m")
        self.logger.info(f"  Value range: {np.nanmin(dem_data):.2f} to {np.nanmax(dem_data):.2f}")

    def visualize_dem(
        self,
        layer: str = "dem",
        use_transformed: bool = False,
        title: str = None,
        cmap: str = "terrain",
        percentile_clip: bool = True,
        clip_percentiles: tuple = (1, 99),
        max_pixels: int = 500_000,
        show_histogram: bool = True,
    ) -> None:
        """
        Create diagnostic visualization of any terrain data layer.

        Args:
            layer: Name of data layer to visualize (default: 'dem')
            use_transformed: Whether to use transformed or original data (default: False)
            title: Plot title (default: auto-generated based on layer)
            cmap: Matplotlib colormap
            percentile_clip: Whether to clip extreme values
            clip_percentiles: Tuple of (min, max) percentiles to clip (default: (1, 99))
            max_pixels: Maximum number of pixels for subsampling
            show_histogram: Whether to show the histogram panel (default: True)
        """
        self.logger.info(f"Creating visualization for layer '{layer}'")

        # Validate requested layer exists
        if layer not in self.data_layers:
            available_layers = list(self.data_layers.keys())
            raise ValueError(f"Layer '{layer}' not found. Available layers: {available_layers}")

        layer_info = self.data_layers[layer]

        # Determine which data to use (transformed or original)
        if use_transformed and not layer_info.get("transformed", False):
            self.logger.warning(
                f"Transformed data requested for layer '{layer}' but not available. Using original."
            )
            use_transformed = False

        if use_transformed:
            plot_data = layer_info["transformed_data"]
            data_transform = layer_info["transformed_transform"]
            data_crs = layer_info["transformed_crs"]
            self.logger.info(f"Using transformed data for layer '{layer}'")
        else:
            plot_data = layer_info["data"]
            data_transform = layer_info["transform"]
            data_crs = layer_info["crs"]
            self.logger.info(f"Using original data for layer '{layer}'")

        # Generate title if not provided
        if title is None:
            transform_status = "Transformed" if use_transformed else "Original"
            title = f"{transform_status} {layer.capitalize()} Layer Visualization"

        # Remove NaN for calculations
        valid_data = plot_data[~np.isnan(plot_data)]
        if len(valid_data) == 0:
            self.logger.error(f"Layer '{layer}' contains no valid data (all NaN)")
            return

        # Logging basic statistics
        self.logger.info("Data Statistics:")
        self.logger.info(f"  Shape: {plot_data.shape}")
        self.logger.info(f"  Min Value: {valid_data.min():.4f}")
        self.logger.info(f"  Max Value: {valid_data.max():.4f}")
        self.logger.info(f"  Mean Value: {valid_data.mean():.4f}")
        self.logger.info(f"  Median Value: {np.median(valid_data):.4f}")

        # NaN analysis
        nan_percentage = np.isnan(plot_data).mean() * 100
        self.logger.info(f"  NaN Percentage: {nan_percentage:.2f}%")

        # Subsampling to prevent memory issues
        def sample_array(arr):
            """Downsample array for visualization if it exceeds max_pixels limit."""
            total_pixels = arr.size
            if total_pixels <= max_pixels:
                return arr

            sample_rate = int(np.sqrt(total_pixels / max_pixels))
            self.logger.info(f"Subsampling with rate 1/{sample_rate} for visualization")
            return arr[::sample_rate, ::sample_rate]

        sampled_data = sample_array(plot_data)

        # Determine color scaling
        if percentile_clip:
            min_percentile, max_percentile = clip_percentiles
            vmin = np.percentile(valid_data, min_percentile)
            vmax = np.percentile(valid_data, max_percentile)
            self.logger.info(
                f"Clipping to {min_percentile}-{max_percentile} percentiles: [{vmin:.4f}, {vmax:.4f}]"
            )
        else:
            vmin, vmax = valid_data.min(), valid_data.max()

        # Determine plot layout
        if show_histogram:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle(title, fontsize=16)

            # Main data heatmap
            im = ax1.imshow(sampled_data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax1.set_title(f"{layer.capitalize()} Visualization")
            ax1.set_xlabel("Column Index")
            ax1.set_ylabel("Row Index")
            plt.colorbar(im, ax=ax1, shrink=0.8, label=layer.capitalize())

            # Value distribution histogram
            ax2.hist(valid_data, bins=50, color="skyblue", alpha=0.7)
            ax2.set_title(f"{layer.capitalize()} Distribution")
            ax2.set_xlabel("Value")
            ax2.set_ylabel("Frequency")

            # Add grid lines
            ax1.grid(False)
            ax2.grid(True, alpha=0.3)

        else:
            # Simple single plot with just the heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            fig.suptitle(title, fontsize=16)

            im = ax.imshow(sampled_data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(f"{layer.capitalize()} Visualization")
            ax.set_xlabel("Column Index")
            ax.set_ylabel("Row Index")
            plt.colorbar(im, ax=ax, shrink=0.8, label=layer.capitalize())

        # Add data source and transform info in footer
        transform_str = f"Transform: [{data_transform[0]:.6f}, {data_transform[1]:.6f}, {data_transform[2]:.6f}, {data_transform[3]:.6f}, {data_transform[4]:.6f}, {data_transform[5]:.6f}]"
        plt.figtext(0.5, 0.01, f"CRS: {data_crs} | {transform_str}", ha="center", fontsize=8)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.05)  # Make room for footer
        plt.show()

        self.logger.info("Visualization complete.")

    def add_transform(self, transform_func):
        """
        Add a transform function to the processing pipeline.

        Args:
            transform_func (callable): Function that transforms DEM data. Should accept
                (dem_array: np.ndarray) and return transformed np.ndarray.

        Returns:
            None: Modifies internal transforms list in place.

        Examples:
            >>> terrain.add_transform(lambda dem: gaussian_filter(dem, sigma=2))
            >>> terrain.apply_transforms()
        """
        wrapped_transform = transform_wrapper(transform_func)

        self.transforms.append(wrapped_transform)
        self.logger.info(f"Added transform: {transform_func.__name__}")

    def add_data_layer(
        self,
        name: str,
        data: np.ndarray,
        transform: Optional[rasterio.Affine] = None,
        crs: Optional[str] = None,
        target_crs: Optional[str] = None,
        target_layer: Optional[str] = None,
        same_extent_as: Optional[str] = None,
        resampling: Resampling = Resampling.bilinear,
        nodata: Optional[float] = None,
    ) -> None:
        """
        Add a data layer, optionally reprojecting to match another layer.

        Stores data with geographic metadata (CRS and transform). Can automatically
        reproject and resample to match an existing layer's grid for multi-layer analysis.

        Args:
            name (str): Unique name for this data layer (e.g., 'dem', 'elevation', 'slope').
            data (np.ndarray): 2D array of data values, shape (height, width).
            transform (rasterio.Affine, optional): Affine transform mapping pixel to geographic
                coords. Required unless same_extent_as is specified.
            crs (str, optional): Coordinate reference system in EPSG format (e.g., 'EPSG:4326').
                Required unless same_extent_as is specified (inherits from reference layer).
            target_crs (str, optional): Target CRS to reproject to. If None and target_layer
                specified, uses target layer's CRS. If None and no target, uses input crs.
            target_layer (str, optional): Name of existing layer to match grid and CRS.
                If specified, data is automatically reprojected and resampled to align.
            same_extent_as (str, optional): Name of existing layer whose geographic extent
                this data covers. When specified, transform and CRS are automatically
                calculated from the reference layer's bounds and the data's shape.
                This is useful when score grids or overlays cover the same area as the
                DEM but at different resolutions. Implies target_layer if not specified.
            resampling (rasterio.enums.Resampling): Resampling method for reprojection
                (default: Resampling.bilinear). See rasterio docs for options.
            nodata (float, optional): No-data sentinel value used during reprojection.
                Source pixels with this value are treated as missing and won't influence
                bilinear interpolation neighbours. Destination pixels with no source
                coverage are filled with this value. Defaults to ``np.nan`` for
                floating-point arrays and ``0`` for integer arrays.

        Returns:
            None: Modifies internal data_layers dictionary.

        Raises:
            KeyError: If target_layer or same_extent_as layer doesn't exist.
            ValueError: If neither transform nor same_extent_as is provided.

        Examples:
            >>> # Add elevation data with native CRS
            >>> terrain.add_data_layer('dem', dem_array, transform, 'EPSG:4326')

            >>> # Add overlay data, reproject to match DEM
            >>> terrain.add_data_layer('landcover', lc_array, lc_transform, 'EPSG:3857',
            ...                        target_layer='dem')

            >>> # Add score data that covers the same extent as DEM (automatic transform)
            >>> terrain.add_data_layer('score', score_array, same_extent_as='dem')

            >>> # Use nearest-neighbor for categorical data
            >>> terrain.add_data_layer('zones', zone_array, zone_transform, 'EPSG:4326',
            ...                        target_layer='dem', resampling=Resampling.nearest)
        """
        self.logger.info(f"Adding data layer '{name}'")

        # Handle same_extent_as: calculate transform from reference layer's bounds
        if same_extent_as is not None:
            if same_extent_as not in self.data_layers:
                raise KeyError(f"Reference layer '{same_extent_as}' not found for same_extent_as")

            ref_info = self.data_layers[same_extent_as]

            # Use ORIGINAL extent and CRS (before transforms) since the source data
            # typically covers the same geographic area as the original reference layer.
            # The reprojection will handle coordinate transformation.
            ref_data = ref_info["data"]
            ref_transform = ref_info["transform"]
            ref_crs = ref_info["crs"]

            # Calculate geographic bounds of reference layer
            ref_height, ref_width = ref_data.shape
            # Top-left corner
            x_origin = ref_transform.c
            y_origin = ref_transform.f
            # Bottom-right corner
            x_end = x_origin + ref_transform.a * ref_width
            y_end = y_origin + ref_transform.e * ref_height

            # Calculate pixel size for source data to cover same extent
            src_height, src_width = data.shape
            pixel_width = (x_end - x_origin) / src_width
            pixel_height = (y_end - y_origin) / src_height

            # Create transform for source data
            transform = rasterio.Affine(pixel_width, 0, x_origin, 0, pixel_height, y_origin)
            crs = ref_crs

            self.logger.info(
                f"Calculated transform from '{same_extent_as}' extent: "
                f"origin=({x_origin:.4f}, {y_origin:.4f}), "
                f"pixel=({pixel_width:.6f}, {pixel_height:.6f})"
            )

            # If target_layer not specified, use same_extent_as as target
            if target_layer is None:
                target_layer = same_extent_as

        # Validate that transform and crs are provided (either directly or via same_extent_as)
        if transform is None:
            raise ValueError("transform is required (or use same_extent_as to calculate automatically)")
        if crs is None:
            raise ValueError("crs is required (or use same_extent_as to inherit from reference layer)")

        # Store target_layer reference for post-transform alignment
        target_layer_ref = target_layer

        # Determine target CRS and transform
        if target_layer is not None:
            if target_layer not in self.data_layers:
                raise KeyError(f"Target layer '{target_layer}' not found")

            target_info = self.data_layers[target_layer]

            # Use transformed data dimensions if transforms have been applied,
            # otherwise fall back to original dimensions. This ensures data layers
            # added after downsampling are automatically resampled to match the
            # actual mesh dimensions.
            if target_info.get("transformed", False) and "transformed_data" in target_info:
                target_shape = target_info["transformed_data"].shape
                target_transform = target_info.get(
                    "transformed_transform", target_info["transform"]
                )
                target_crs = target_info.get("transformed_crs", target_info["crs"])
                self.logger.debug(
                    f"Using transformed target shape {target_shape} for layer alignment"
                )
            else:
                target_shape = target_info["data"].shape
                target_transform = target_info["transform"]
                target_crs = target_info["crs"]

        elif target_crs is not None:
            # If target_crs provided but no reference layer, we need a reference layer
            if not self.data_layers:
                raise ValueError("Cannot determine target grid without reference layer")

            # Use first layer as reference for grid
            reference_layer = next(iter(self.data_layers.values()))

            # Use transformed dimensions if available
            if reference_layer.get("transformed", False) and "transformed_data" in reference_layer:
                target_transform = reference_layer.get(
                    "transformed_transform", reference_layer["transform"]
                )
                target_shape = reference_layer["transformed_data"].shape
            else:
                target_transform = reference_layer["transform"]
                target_shape = reference_layer["data"].shape

        else:
            # If no target specified, keep original
            self.data_layers[name] = {
                "data": data,
                "transform": transform,
                "crs": crs,
                "transformed": False,
                "target_layer": None,
            }
            self.logger.info(f"Added layer '{name}' with original CRS {crs}")
            return

        # Resolve nodata value: explicit arg > auto-detect from dtype
        if nodata is None:
            nodata_value = np.nan if np.issubdtype(data.dtype, np.floating) else 0
        else:
            nodata_value = nodata

        # Create target array and reproject if needed
        # Note: Affine.__ne__ returns array, so use tuple comparison instead
        transforms_differ = (crs != target_crs) or (tuple(transform) != tuple(target_transform))
        if transforms_differ:
            self.logger.info(f"Reprojecting from {crs} to {target_crs}")
            self.logger.info(f"Transforms: {transform} to {target_transform}")
            # Initialise with nodata so uncovered destination pixels don't become 0
            aligned_data = np.full(target_shape, nodata_value, dtype=data.dtype)

            try:
                reproject(
                    data,
                    aligned_data,
                    src_transform=transform,
                    src_crs=crs,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    resampling=resampling,
                    src_nodata=nodata_value,
                    dst_nodata=nodata_value,
                )

                # Store reprojected data
                self.data_layers[name] = {
                    "data": aligned_data,
                    "transform": target_transform,
                    "crs": target_crs,
                    "original_data": data,
                    "original_transform": transform,
                    "original_crs": crs,
                    "transformed": False,
                    "target_layer": target_layer_ref,
                }

                self.logger.info(f"Successfully added layer '{name}' (reprojected):")
                self.logger.info(f"  Shape: {aligned_data.shape}")
                valid_pixels = aligned_data[~np.isnan(aligned_data)] if np.issubdtype(aligned_data.dtype, np.floating) else aligned_data.ravel()
                if valid_pixels.size > 0:
                    self.logger.info(
                        f"  Value range: {valid_pixels.min():.2f} to {valid_pixels.max():.2f}"
                    )
                else:
                    self.logger.info("  Value range: all nodata (no valid pixels after reprojection)")

            except Exception as e:
                self.logger.error(f"Failed to reproject layer '{name}': {str(e)}")
                raise
        else:
            # No reprojection needed
            self.data_layers[name] = {
                "data": data,
                "transform": transform,
                "crs": crs,
                "transformed": False,
                "target_layer": target_layer_ref,
            }
            self.logger.info(f"Added layer '{name}' (no reprojection needed)")

    def get_bbox_wgs84(self, layer: str = "dem") -> tuple[float, float, float, float]:
        """
        Get bounding box in WGS84 coordinates (EPSG:4326).

        Returns bbox in standard format used by OSM, web mapping APIs, etc.
        Handles reprojection from any source CRS back to WGS84.

        Args:
            layer: Layer name to get bbox for (default: "dem")

        Returns:
            Tuple of (south, west, north, east) in WGS84 degrees

        Raises:
            KeyError: If specified layer doesn't exist

        Example:
            >>> terrain = Terrain(dem_data, transform, dem_crs="EPSG:32617")  # UTM
            >>> south, west, north, east = terrain.get_bbox_wgs84()
            >>> print(f"Bounds: {south:.4f}°N to {north:.4f}°N, {west:.4f}°E to {east:.4f}°E")
        """
        from rasterio.warp import transform_bounds

        if layer not in self.data_layers:
            raise KeyError(f"Layer '{layer}' not found")

        layer_info = self.data_layers[layer]

        # Use transformed data/transform if available, otherwise original
        if layer_info.get("transformed") and "transformed_data" in layer_info:
            data = layer_info["transformed_data"]
            layer_transform = layer_info.get("transformed_transform", layer_info["transform"])
            layer_crs = layer_info.get("transformed_crs", layer_info["crs"])
        else:
            data = layer_info["data"]
            layer_transform = layer_info["transform"]
            layer_crs = layer_info["crs"]

        # Calculate bounds from transform and shape
        height, width = data.shape
        # Top-left corner
        x_origin = layer_transform.c
        y_origin = layer_transform.f
        # Bottom-right corner
        x_end = x_origin + layer_transform.a * width
        y_end = y_origin + layer_transform.e * height

        # Normalize bounds (west, south, east, north)
        west = min(x_origin, x_end)
        east = max(x_origin, x_end)
        south = min(y_origin, y_end)
        north = max(y_origin, y_end)

        # If already WGS84, just return
        if layer_crs in ("EPSG:4326", "epsg:4326", None):
            return (south, west, north, east)

        # Transform bounds to WGS84
        transformed_bounds = transform_bounds(
            layer_crs, "EPSG:4326", west, south, east, north
        )

        # transform_bounds returns (west, south, east, north)
        t_west, t_south, t_east, t_north = transformed_bounds

        return (t_south, t_west, t_north, t_east)

    def compute_data_layer(
        self,
        name: str,
        source_layer: str,
        compute_func: Callable[[np.ndarray], np.ndarray],
        transformed: bool = False,
        cache_key: Optional[str] = None,
    ) -> np.ndarray:
        """
        Compute a new data layer from an existing one using a transformation function.

        Allows creating derived layers (e.g., slope, aspect, hillshade) from existing data.
        Results are stored as new layer and optionally cached.

        Args:
            name (str): Name for the computed layer.
            source_layer (str): Name of existing source layer to compute from.
            compute_func (Callable): Function that accepts source array (np.ndarray)
                and returns computed array (np.ndarray). Can return same or different shape.
            transformed (bool): If True, use already-transformed source data; if False,
                use original source data (default: False).
            cache_key (str, optional): Custom cache identifier. If None, auto-generated
                from layer name and function name.

        Returns:
            np.ndarray: The computed layer data array.

        Raises:
            KeyError: If source_layer doesn't exist.
            ValueError: If transformed=True but source hasn't been transformed.

        Examples:
            >>> # Compute slope from DEM using scipy
            >>> from scipy.ndimage import sobel
            >>> slope = terrain.compute_data_layer(
            ...     'slope', 'dem',
            ...     lambda dem: np.sqrt(sobel(dem, axis=0)**2 + sobel(dem, axis=1)**2)
            ... )

            >>> # Compute hill-shade visualization
            >>> from scipy.ndimage import gaussian_filter
            >>> hillshade = terrain.compute_data_layer(
            ...     'hillshade', 'dem',
            ...     lambda dem: np.clip(gaussian_filter(dem, 2) * 0.5, 0, 1)
            ... )

            >>> # Compute from transformed (downsampled) data
            >>> downsampled_slope = terrain.compute_data_layer(
            ...     'slope_downsampled', 'dem',
            ...     lambda dem: np.gradient(dem)[0],
            ...     transformed=True
            ... )
        """
        self.logger.info(f"Computing layer '{name}' from '{source_layer}'")

        # Verify source layer exists
        if source_layer not in self.data_layers:
            raise KeyError(f"Source layer '{source_layer}' not found")

        source_layer_info = self.data_layers[source_layer]

        # Check if transformed data is requested but not available
        if transformed and not source_layer_info.get("transformed", False):
            raise ValueError(f"Source layer '{source_layer}' has not been transformed")

        # Get appropriate source data and metadata
        if transformed:
            source_data = source_layer_info["transformed_data"]
            source_transform = source_layer_info["transformed_transform"]
            source_crs = source_layer_info["transformed_crs"]
        else:
            source_data = source_layer_info["data"]
            source_transform = source_layer_info["transform"]
            source_crs = source_layer_info["crs"]

        # Generate cache key if not provided
        if cache_key is None:
            transform_suffix = "_transformed" if transformed else ""
            cache_key = f"{name}_{source_layer}{transform_suffix}_{compute_func.__name__}"

        # Try to load from cache
        cached = self.cache.load(cache_key)

        if cached is None:
            self.logger.info(f"Computing {name} from {source_layer}")
            try:
                # Apply the computation function
                computed_data = compute_func(source_data)

                # Cache the result with source metadata
                self.cache.save(
                    cache_key,
                    computed_data,
                    transform=source_transform,
                    crs=source_crs,
                    metadata={"source_layer": source_layer, "transformed": transformed},
                )

            except Exception as e:
                self.logger.error(f"Failed to compute layer '{name}': {str(e)}")
                raise
        else:
            self.logger.info(f"Loaded cached computation for '{name}'")
            computed_data = cached["data"]
            # Use cached metadata if available
            source_transform = cached.get("transform", source_transform)
            source_crs = cached.get("crs", source_crs)

        # Add the computed layer with correct transform and CRS
        self.add_data_layer(name, computed_data, source_transform, source_crs)

        return computed_data

    def apply_transforms(self, cache=False):
        """
        Apply all transforms to all data layers with optional caching.

        Processes each data layer through the transform pipeline. Results are cached
        to avoid recomputation. Transforms are applied in order.

        Args:
            cache (bool): Whether to cache results (default: False).

        Returns:
            None: Updates internal data_layers with 'transformed_data' for each layer.

        Examples:
            >>> terrain.add_transform(flip_raster(axis='horizontal'))
            >>> terrain.apply_transforms(cache=True)
            >>> dem_data = terrain.data_layers['dem']['transformed_data']
        """
        if not self.transforms:
            self.logger.warning("No transforms to apply")
            return

        # Process all data layers with individual caches
        with tqdm(total=len(self.data_layers), desc="Processing data layers") as pbar:
            for name, layer in self.data_layers.items():
                # Skip already transformed layers
                if layer.get("transformed", False):
                    self.logger.debug(f"Layer {name} already transformed, skipping")
                    pbar.update(1)
                    continue

                # Create target name from transform sequence
                layer_target = f"{name}_{'_'.join(t.__name__ for t in self.transforms)}"
                cached_layer = self.cache.load(layer_target)

                if cached_layer is None:
                    import time as _time
                    self.logger.info(f"Cache miss for {layer_target}, computing transforms...")
                    layer_data = layer["data"].copy()
                    current_transform = layer["transform"]
                    current_crs = layer["crs"]

                    # Track transforms that were applied
                    applied_transforms = []
                    n_transforms = len(self.transforms)
                    pipeline_start = _time.time()

                    for i, transform_func in enumerate(self.transforms, 1):
                        transform_name = transform_func.__name__
                        input_shape = layer_data.shape
                        n_pixels = layer_data.size

                        self.logger.info(
                            f"  [{i}/{n_transforms}] Applying {transform_name}... "
                            f"(input: {input_shape[0]}×{input_shape[1]} = {n_pixels/1e6:.1f}M pixels)"
                        )
                        transform_start = _time.time()

                        # Apply transform (wrapper always returns 3-tuple)
                        try:
                            layer_data, current_transform, new_crs = transform_func(
                                layer_data, current_transform
                            )

                            # Update CRS if the transform provided a new one
                            if new_crs is not None:
                                current_crs = new_crs

                            applied_transforms.append(transform_name)

                            # Track elevation scale factor if this transform has one
                            if hasattr(transform_func, '_elevation_scale_factor'):
                                scale = transform_func._elevation_scale_factor
                                self.transform_metadata["elevation_scale"] *= scale
                                self.logger.debug(
                                    f"  Updated elevation_scale: {self.transform_metadata['elevation_scale']}"
                                )

                            transform_elapsed = _time.time() - transform_start
                            output_shape = layer_data.shape
                            self.logger.info(
                                f"  [{i}/{n_transforms}] ✓ {transform_name} complete in {transform_elapsed:.1f}s "
                                f"→ {output_shape[0]}×{output_shape[1]}"
                            )
                        except Exception as e:
                            self.logger.error(
                                f"Failed applying transform {transform_name}: {str(e)}"
                            )
                            raise

                    pipeline_elapsed = _time.time() - pipeline_start
                    self.logger.info(
                        f"  All {n_transforms} transforms complete in {pipeline_elapsed:.1f}s total"
                    )

                    # Save result with comprehensive metadata
                    metadata = {
                        "transforms": applied_transforms,
                        "original_shape": layer["data"].shape,
                        "transformed_shape": layer_data.shape,
                        "original_crs": layer["crs"],
                        "final_crs": current_crs,
                    }

                    if cache:
                        self.cache.save(
                            layer_target,
                            layer_data,
                            transform=current_transform,
                            crs=current_crs,
                            metadata=metadata,
                        )

                    # Update layer info
                    self.data_layers[name].update(
                        {
                            "transformed_data": layer_data,
                            "transformed_transform": current_transform,
                            "transformed_crs": current_crs,
                            "transformed": True,
                            "transform_metadata": metadata,
                        }
                    )
                else:
                    self.logger.info(f"Cache hit for {layer_target}")
                    self.data_layers[name].update(
                        {
                            "transformed_data": cached_layer["data"],
                            "transformed_transform": cached_layer["transform"],
                            "transformed_crs": cached_layer.get("crs", layer["crs"]),
                            "transformed": True,
                            "transform_metadata": cached_layer.get("metadata", {}),
                        }
                    )

                pbar.update(1)

        self.logger.info("Transforms applied successfully")

        # Post-processing: Align layers with target_layer to match target's final shape
        self._align_layers_to_targets()

    def _align_layers_to_targets(self):
        """
        Post-processing step after apply_transforms() to align layers with their targets.

        For layers added with target_layer parameter, this resamples them to match
        the target layer's FINAL transformed shape. This ensures layers with different
        source resolutions all end up at the same final resolution.

        This is called automatically at the end of apply_transforms().
        """
        from rasterio.warp import reproject, Resampling

        layers_to_align = []
        for name, layer in self.data_layers.items():
            target_ref = layer.get("target_layer")
            if target_ref and layer.get("transformed", False):
                layers_to_align.append((name, target_ref))

        if not layers_to_align:
            return

        self.logger.info(f"Aligning {len(layers_to_align)} layers to their targets...")

        for name, target_ref in layers_to_align:
            if target_ref not in self.data_layers:
                self.logger.warning(
                    f"Target layer '{target_ref}' not found for layer '{name}', skipping alignment"
                )
                continue

            layer = self.data_layers[name]
            target = self.data_layers[target_ref]

            if not target.get("transformed", False):
                self.logger.warning(
                    f"Target layer '{target_ref}' not transformed yet, skipping alignment for '{name}'"
                )
                continue

            # Get current and target shapes
            current_shape = layer["transformed_data"].shape
            target_shape = target["transformed_data"].shape

            if current_shape == target_shape:
                self.logger.debug(f"Layer '{name}' already matches target shape {target_shape}")
                continue

            self.logger.info(
                f"Resampling '{name}' from {current_shape} to match '{target_ref}' {target_shape}"
            )

            # Resample to match target shape
            aligned_data = np.zeros(target_shape, dtype=layer["transformed_data"].dtype)

            try:
                reproject(
                    layer["transformed_data"],
                    aligned_data,
                    src_transform=layer["transformed_transform"],
                    src_crs=layer["transformed_crs"],
                    dst_transform=target["transformed_transform"],
                    dst_crs=target["transformed_crs"],
                    resampling=Resampling.bilinear,
                )

                # Update layer with aligned data
                layer["transformed_data"] = aligned_data
                layer["transformed_transform"] = target["transformed_transform"]
                layer["transformed_crs"] = target["transformed_crs"]

                self.logger.info(f"✓ Aligned '{name}' to match '{target_ref}'")

            except Exception as e:
                self.logger.error(f"Failed to align '{name}' to '{target_ref}': {str(e)}")
                # Don't raise - continue with other layers

    def configure_for_target_vertices(
        self, target_vertices: int, method: str = "average"
    ) -> float:
        """
        Configure downsampling to achieve approximately target_vertices.

        This method calculates the appropriate zoom_factor to achieve a desired
        vertex count for mesh generation. It provides a more intuitive API than
        manually calculating zoom_factor from the original DEM shape.

        Args:
            target_vertices: Desired vertex count for final mesh (e.g., 500_000)
            method: Downsampling method (default: "average")
                - "average": Area averaging - best for DEMs, no overshoot
                - "lanczos": Lanczos resampling - sharp, minimal aliasing
                - "cubic": Cubic spline interpolation
                - "bilinear": Bilinear interpolation - safe fallback

        Returns:
            Calculated zoom_factor that was added to transforms

        Raises:
            ValueError: If target_vertices is invalid

        Example:
            terrain = Terrain(dem, transform)
            zoom = terrain.configure_for_target_vertices(500_000, method="average")
            print(f"Calculated zoom_factor: {zoom:.4f}")
            terrain.apply_transforms()
            mesh = terrain.create_mesh(scale_factor=400.0)
        """
        if not isinstance(target_vertices, int) or target_vertices <= 0:
            raise ValueError(f"target_vertices must be a positive integer, got {target_vertices}")

        original_h, original_w = self.dem_shape
        original_vertices = original_h * original_w

        if target_vertices > original_vertices:
            self.logger.warning(
                f"Target vertices ({target_vertices:,}) exceeds source vertices "
                f"({original_vertices:,}). Using original resolution (zoom_factor=1.0)."
            )
            zoom_factor = 1.0
        else:
            # Calculate zoom_factor: vertices = (H * zoom) * (W * zoom)
            # So: zoom_factor = sqrt(target_vertices / (H * W))
            zoom_factor = np.sqrt(target_vertices / original_vertices)

        self.logger.info(
            f"Configuring for {target_vertices:,} target vertices\n"
            f"  Original DEM: {original_h} × {original_w} ({original_vertices:,} vertices)\n"
            f"  Calculated zoom_factor: {zoom_factor:.6f}\n"
            f"  Downsampling method: {method}\n"
            f"  Resulting grid: {int(original_h * zoom_factor)} × {int(original_w * zoom_factor)}"
        )

        # Add downsampling transform to the pipeline
        self.transforms.append(downsample_raster(zoom_factor=zoom_factor, method=method))

        return zoom_factor

                # Keep alpha unchanged
