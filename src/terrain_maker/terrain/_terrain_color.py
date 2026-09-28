"""Vertex color mapping: single, blended and multi-overlay colormaps."""

from __future__ import annotations


import numpy as np
import logging

from terrain_maker.terrain.water import shoreline_water_colors
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable

from scipy.ndimage import zoom

# Output handling is configured once for the whole package in _logging.py
logger = logging.getLogger(__name__)


def _as_rgba(colors):
    """Append an opaque alpha channel to RGB colors (255 for uint8, else 1)."""
    if colors.shape[-1] != 3:
        return colors
    alpha = np.full(
        colors.shape[:2] + (1,), 255 if colors.dtype == np.uint8 else 1, dtype=colors.dtype
    )
    return np.concatenate([colors, alpha], axis=-1)


@dataclass
class ColormapLayer:
    """A colormap applied to named data layers."""

    colormap: Callable
    sources: list
    kwargs: dict = field(default_factory=dict)

    def compute(self, terrain, label):
        arrays = [terrain._layer_array(layer) for layer in self.sources]
        try:
            return _as_rgba(np.asarray(self.colormap(*arrays, **self.kwargs)))
        except Exception as e:
            terrain.logger.error(f"Error computing {label} colors: {str(e)}")
            raise


@dataclass
class SingleColorMapping:
    """One colormap everywhere, with an optional mask that makes pixels transparent."""

    colormap: Callable
    sources: list
    kwargs: dict
    mask_func: Optional[Callable] = None
    mask_sources: list = field(default_factory=list)
    mask_kwargs: dict = field(default_factory=dict)
    mask_threshold: Optional[float] = None
    mode = "standard"

    def compute(self, terrain):
        colors = np.asarray(
            ColormapLayer(self.colormap, self.sources, self.kwargs).compute(terrain, "base")
        )
        if colors.ndim != 3 or colors.shape[-1] not in (3, 4):
            raise ValueError(
                "Color mapping must return an (H, W, 3) or (H, W, 4) array; "
                f"got shape {colors.shape}. Wrap values in a colormap such as "
                "elevation_colormap()."
            )
        if self.mask_func:
            arrays = [terrain._layer_array(layer) for layer in self.mask_sources]
            try:
                mask = self.mask_func(*arrays, **self.mask_kwargs)
            except Exception as e:
                terrain.logger.error(f"Error computing mask: {str(e)}")
                raise
            if self.mask_threshold is not None:
                mask = mask >= self.mask_threshold
            colors[..., 3] = np.where(mask, 0.0, 1.0)
        return colors


@dataclass
class BlendedColorMapping:
    """Overlay colormap where overlay_mask is True, base colormap elsewhere."""

    base: ColormapLayer
    overlay: ColormapLayer
    overlay_mask: np.ndarray
    mode = "blended"

    def compute(self, terrain):
        base = self.base.compute(terrain, "base")
        overlay = self.overlay.compute(terrain, "overlay")
        mask = _grid_mask(terrain, self.overlay_mask, base.shape[:2])
        if mask is None:
            raise ValueError(
                f"overlay_mask shape {self.overlay_mask.shape} matches neither the color grid "
                f"{base.shape[:2]} nor the mesh vertices (call create_mesh() first)"
            )
        terrain.logger.info(f"Blended colors: {np.sum(mask)} overlay pixels")
        return np.where(mask[..., None], overlay, base)


@dataclass
class MultiOverlayColorMapping:
    """Base colormap with overlays applied in priority order, each by mask or threshold."""

    base: ColormapLayer
    overlays: list
    mode = "multi-overlay"

    def compute(self, terrain):
        colors = self.base.compute(terrain, "base").copy()
        for i, spec in enumerate(self.overlays):
            layer = ColormapLayer(
                spec["colormap"], spec["source_layers"], spec.get("colormap_kwargs", {})
            )
            overlay = layer.compute(terrain, f"overlay {i}")
            mask = None
            if spec.get("mask") is not None:
                mask = _grid_mask(terrain, np.asarray(spec["mask"]), colors.shape[:2])
                if mask is None:
                    terrain.logger.warning(
                        f"Overlay {i}: mask shape {np.shape(spec['mask'])} doesn't match grid "
                        f"shape {colors.shape[:2]}. Falling back to threshold-based mask."
                    )
            if mask is None:
                mask = _threshold_mask(
                    terrain._layer_array(spec["source_layers"][0]), spec.get("threshold", 0.5)
                )
            colors[mask] = overlay[mask]
            terrain.logger.info(f"Overlay {i}: {np.sum(mask)} grid pixels")
        return colors


def _grid_mask(terrain, mask, grid_shape):
    """A boolean mask as a grid: grid masks pass through, vertex masks are scattered
    to their pixels (extra entries for skirt vertices are ignored). None if neither fits."""
    if mask.shape == tuple(grid_shape):
        return mask.astype(bool)
    y_valid = getattr(terrain, "y_valid", None)
    if mask.ndim == 1 and y_valid is not None and len(mask) >= len(y_valid):
        grid = np.zeros(grid_shape, dtype=bool)
        grid[y_valid, terrain.x_valid] = mask[: len(y_valid)]
        return grid
    return None


def _threshold_mask(values, threshold):
    """Pixels at or above threshold (strictly above for 0, so sparse layers skip zeros)."""
    above = values > 0.0 if threshold == 0.0 else values >= threshold
    return above & ~np.isnan(values)


class TerrainColorMixin:
    """Terrain methods: vertex color mapping: single, blended and multi-overlay colormaps."""

    def set_color_mapping(
        self,
        color_func: Callable[[np.ndarray, ...], np.ndarray],
        source_layers: list[str],
        *,
        color_kwargs: Optional[Dict[str, Any]] = None,
        mask_func: Optional[Callable[[np.ndarray, ...], np.ndarray]] = None,
        mask_layers: Optional[list[str] | str] = None,
        mask_kwargs: Optional[Dict[str, Any]] = None,
        mask_threshold: Optional[float] = None,
    ) -> None:
        """
        Set up how to map data layers to colors (RGB) and optionally a mask/alpha channel.

        Allows flexible color mapping by applying a function to one or more data layers.
        Optionally applies a separate mask function for transparency/alpha channel control.
        Color mapping is applied during mesh creation with `compute_colors()`.

        Args:
            color_func (Callable): Function that accepts N data arrays (one per source_layers)
                and returns colored array of shape (H, W, 3) for RGB or (H, W, 4) for RGBA.
                Values should be in range [0, 1] for 8-bit output.
            source_layers (list[str]): Names of data layers to pass to color_func, in order.
                E.g., ['dem'] for single layer or ['red', 'green', 'blue'] for composite.
            color_kwargs (dict, optional): Additional keyword arguments passed to color_func.
            mask_func (Callable, optional): Function producing alpha/mask values (0-1) for
                transparency. Takes layer arrays as input. If omitted, fully opaque.
            mask_layers (list[str] | str, optional): Layer names for mask_func. If None,
                uses source_layers. Single string converted to list.
            mask_kwargs (dict, optional): Additional keyword arguments for mask_func.
            mask_threshold (float, optional): If mask_func is threshold-based, convenience
                parameter for threshold value (implementation-dependent).

        Returns:
            None: Modifies internal color mapping configuration.

        Raises:
            ValueError: If source_layers or mask_layers refer to non-existent layers.

        Examples:
            >>> # Single-layer elevation with viridis colormap
            >>> from matplotlib.cm import viridis
            >>> terrain.set_color_mapping(
            ...     lambda dem: viridis(dem / dem.max()),
            ...     ['dem']
            ... )

            >>> # RGB composite from three layers
            >>> terrain.set_color_mapping(
            ...     lambda r, g, b: np.stack([r, g, b], axis=-1),
            ...     ['red_band', 'green_band', 'blue_band']
            ... )

            >>> # Elevation with water transparency mask
            >>> terrain.set_color_mapping(
            ...     lambda dem: elevation_colormap(dem),
            ...     ['dem'],
            ...     mask_func=lambda dem: (dem > 0).astype(float),
            ...     mask_layers=['dem']
            ... )

            >>> # Hillshade with elevation colors and slope transparency
            >>> terrain.set_color_mapping(
            ...     lambda dem: dem_colors,
            ...     ['dem'],
            ...     mask_func=lambda dem: 1 - np.clip(np.gradient(dem)[0], 0, 1),
            ...     mask_layers=['dem']
            ... )
        """
        missing_layers = [name for name in source_layers if name not in self.data_layers]
        if missing_layers:
            raise ValueError(f"Source layers not found: {missing_layers}")
        if mask_func:
            if mask_layers is None:
                mask_layers = source_layers
            elif isinstance(mask_layers, str):
                mask_layers = [mask_layers]
            missing_mask_layers = [name for name in mask_layers if name not in self.data_layers]
            if missing_mask_layers:
                raise ValueError(f"Mask layers not found: {missing_mask_layers}")
        else:
            mask_layers = []

        self._color_spec = SingleColorMapping(
            colormap=color_func,
            sources=list(source_layers),
            kwargs=color_kwargs or {},
            mask_func=mask_func,
            mask_sources=list(mask_layers),
            mask_kwargs=mask_kwargs or {},
            mask_threshold=mask_threshold,
        )

        self.logger.info(f"Color function: {color_func.__name__}")
        self.logger.info(f"Color source layers: {source_layers}")
        if color_kwargs:
            self.logger.info(f"Color kwargs: {color_kwargs}")
        if mask_func:
            self.logger.info(f"Mask function: {mask_func.__name__}")
            self.logger.info(f"Mask source layers: {mask_layers}")
            if mask_kwargs:
                self.logger.info(f"Mask kwargs: {mask_kwargs}")
            if mask_threshold is not None:
                self.logger.info(f"Mask threshold: {mask_threshold}")

    def set_blended_color_mapping(
        self,
        base_colormap: Callable[[np.ndarray], np.ndarray],
        base_source_layers: list[str],
        overlay_colormap: Callable[[np.ndarray], np.ndarray],
        overlay_source_layers: list[str],
        overlay_mask: np.ndarray,
        *,
        base_color_kwargs: Optional[Dict[str, Any]] = None,
        overlay_color_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Apply two different colormaps based on a spatial mask (hard transition).

        Uses base colormap for most of terrain and overlay colormap for masked zones.
        This is useful for showing different data types in different regions, such as
        elevation colors for general terrain but suitability scores near parks.

        Args:
            base_colormap: Function mapping base layer(s) to RGB/RGBA. Takes N arrays
                (one per base_source_layers) and returns (H, W, 3) or (H, W, 4).
            base_source_layers: Layer names to pass to base_colormap, e.g., ['dem'].
            overlay_colormap: Function mapping overlay layer(s) to RGB/RGBA. Takes N
                arrays (one per overlay_source_layers) and returns (H, W, 3) or (H, W, 4).
            overlay_source_layers: Layer names to pass to overlay_colormap, e.g., ['score'].
            overlay_mask: Boolean array of shape (num_vertices,) indicating where to use
                overlay colormap. True = overlay, False = base. Use compute_proximity_mask()
                to create this mask.
            base_color_kwargs: Optional kwargs passed to base_colormap.
            overlay_color_kwargs: Optional kwargs passed to overlay_colormap.

        Returns:
            None: Modifies internal color mapping to use blended approach.

        Raises:
            ValueError: If source layers don't exist or overlay_mask has wrong shape.
            RuntimeError: If create_mesh() hasn't been called yet (needed for mask validation).

        Example:
            >>> from terrain_maker.terrain.color_mapping import elevation_colormap
            >>> # Compute proximity mask for park zones
            >>> park_mask = terrain.compute_proximity_mask(
            ...     park_lons, park_lats, radius_meters=1000
            ... )
            >>> # Set dual colormaps
            >>> terrain.set_blended_color_mapping(
            ...     base_colormap=lambda elev: elevation_colormap(
            ...         elev, cmap_name="gist_earth"
            ...     ),
            ...     base_source_layers=["dem"],
            ...     overlay_colormap=lambda score: elevation_colormap(
            ...         score, cmap_name="cool", min_elev=0, max_elev=1
            ...     ),
            ...     overlay_source_layers=["score"],
            ...     overlay_mask=park_mask
            ... )
            >>> terrain.compute_colors()  # Apply the blended mapping
        """
        all_layers = set(base_source_layers) | set(overlay_source_layers)
        missing_layers = [name for name in all_layers if name not in self.data_layers]
        if missing_layers:
            raise ValueError(f"Source layers not found: {missing_layers}")
        overlay_mask = np.asarray(overlay_mask)
        if overlay_mask.ndim not in (1, 2):
            raise ValueError(
                f"overlay_mask must be 1D (vertex-space) or 2D (grid-space). "
                f"Got shape {overlay_mask.shape}."
            )

        self._color_spec = BlendedColorMapping(
            base=ColormapLayer(base_colormap, list(base_source_layers), base_color_kwargs or {}),
            overlay=ColormapLayer(
                overlay_colormap, list(overlay_source_layers), overlay_color_kwargs or {}
            ),
            overlay_mask=overlay_mask,
        )

        self.logger.info("Blended color mapping configured:")
        self.logger.info(f"  Base colormap: {base_colormap.__name__} on {base_source_layers}")
        self.logger.info(
            f"  Overlay colormap: {overlay_colormap.__name__} on {overlay_source_layers}"
        )
        mask_sum = np.sum(overlay_mask)
        mask_type = "grid-space" if overlay_mask.ndim == 2 else "vertex-space"
        self.logger.info(
            f"  Overlay mask ({mask_type}): {mask_sum}/{overlay_mask.size} elements "
            f"({100.0 * mask_sum / overlay_mask.size:.1f}%)"
        )

    def set_multi_color_mapping(
        self,
        base_colormap: Callable[[np.ndarray, ...], np.ndarray],
        base_source_layers: list[str],
        overlays: list[dict],
        *,
        base_color_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Apply multiple data layers as overlays with different colormaps and priority.

        This enables flexible data visualization where multiple geographic features
        (roads, trails, land use, power lines, etc.) can each be colored independently
        based on their own colormaps and source layers.

        Args:
            base_colormap: Function mapping base layer(s) to RGB/RGBA. Takes N arrays
                (one per base_source_layers) and returns (H, W, 3) or (H, W, 4).
            base_source_layers: Layer names for base_colormap, e.g., ['dem'].
            overlays: List of overlay specifications. Each overlay dict contains
                ``colormap`` (function returning H,W,3/4), ``source_layers`` (list of
                layer names), ``colormap_kwargs`` (optional dict), ``threshold`` (value
                above which overlay applies, default 0.5), and ``priority`` (lower =
                higher priority).
            base_color_kwargs: Optional kwargs passed to base_colormap.

        Returns:
            None: Modifies internal color mapping to use multi-overlay approach.

        Raises:
            ValueError: If source layers don't exist or overlay specs are invalid.

        Example:
            >>> # Base elevation colors with roads and trails overlays
            >>> terrain.set_multi_color_mapping(
            ...     base_colormap=lambda elev: elevation_colormap(elev, "michigan"),
            ...     base_source_layers=["dem"],
            ...     overlays=[
            ...         {
            ...             "colormap": lambda roads: colormap_roads(roads),
            ...             "source_layers": ["roads"],
            ...             "priority": 10,  # High priority roads show on top
            ...         },
            ...         {
            ...             "colormap": lambda trails: colormap_trails(trails),
            ...             "source_layers": ["trails"],
            ...             "priority": 20,  # Lower priority
            ...         },
            ...     ]
            ... )
            >>> terrain.compute_colors()
        """
        all_layers = set(base_source_layers)
        for overlay in overlays:
            all_layers.update(overlay.get("source_layers", []))
        missing_layers = [name for name in all_layers if name not in self.data_layers]
        if missing_layers:
            raise ValueError(f"Source layers not found: {missing_layers}")
        for i, overlay in enumerate(overlays):
            for key in ("colormap", "source_layers", "priority"):
                if key not in overlay:
                    raise ValueError(f"Overlay {i} missing required '{key}' key")

        # Lower priority number = applied first
        sorted_overlays = sorted(overlays, key=lambda x: x["priority"])
        self._color_spec = MultiOverlayColorMapping(
            base=ColormapLayer(base_colormap, list(base_source_layers), base_color_kwargs or {}),
            overlays=sorted_overlays,
        )

        self.logger.info("Multi-overlay color mapping configured:")
        self.logger.info(f"  Base colormap: {base_colormap.__name__} on {base_source_layers}")
        self.logger.info(f"  Number of overlays: {len(overlays)}")
        for i, overlay in enumerate(sorted_overlays):
            mask_info = ", has_mask=True" if "mask" in overlay else ""
            threshold = overlay.get("threshold", "default")
            self.logger.info(
                f"    Overlay {i} (priority {overlay['priority']}): "
                f"{overlay['colormap'].__name__} on {overlay['source_layers']}"
                f" [threshold={threshold}{mask_info}]"
            )

    def compute_colors(self, water_mask=None):
        """
        Compute colors using color_func and optionally mask_func.

        Supports three modes (set with set_color_mapping, set_blended_color_mapping or
        set_multi_color_mapping). Every mode produces the same layout: an (H, W, 4) RGBA
        grid aligned with the transformed DEM, stored as self.colors. Mesh vertices are
        colored from it at (y_valid, x_valid).

        Args:
            water_mask (np.ndarray, optional): Boolean water mask in grid space (height × width).
                Water pixels are recolored with a shoreline-to-deep blue gradient.

        Returns:
            np.ndarray: RGBA color grid of shape (H, W, 4).
        """
        spec = getattr(self, "_color_spec", None)
        if spec is None:
            raise ValueError("Color mapping not set. Call set_color_mapping() first.")
        self.logger.info(f"Computing colors ({spec.mode})...")
        colors = spec.compute(self)
        if water_mask is not None:
            self._apply_water_gradient(colors, water_mask)
        self.colors = colors
        self.logger.info(f"Colors computed successfully with shape {colors.shape}")
        return colors

    def _apply_water_gradient(self, colors, water_mask):
        """Recolor water in a colors grid with a shoreline-to-deep blue gradient (in place).

        Only mesh-vertex pixels are recolored once the mesh exists; before that, every
        water pixel is.
        """
        if water_mask.shape != colors.shape[:2]:
            # Colors can come from a layer at a different resolution (e.g. a score layer)
            self.logger.warning(
                f"Water mask shape {water_mask.shape} does not match colors shape "
                f"{colors.shape[:2]}. Resampling water mask to match colors."
            )
            water_mask = zoom(
                water_mask.astype(np.float32),
                zoom=(
                    colors.shape[0] / water_mask.shape[0],
                    colors.shape[1] / water_mask.shape[1],
                ),
                order=0,  # nearest neighbor keeps it boolean
                prefilter=False,
            ).astype(np.bool_)

        ys, xs = getattr(self, "y_valid", None), getattr(self, "x_valid", None)
        if ys is None or xs is None:
            ys, xs = np.nonzero(water_mask)
        indices, water_colors = shoreline_water_colors(water_mask, ys, xs)
        colors[ys[indices], xs[indices], :3] = water_colors
        self.logger.info(f"Water colored with depth gradient ({len(indices)} pixels)")

    def _layer_array(self, layer):
        """A layer's data after transforms, or its original data if none ran."""
        info = self.data_layers[layer]
        return info["transformed_data"] if info.get("transformed") else info["data"]
