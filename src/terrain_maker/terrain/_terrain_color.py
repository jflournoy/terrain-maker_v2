"""Vertex color mapping: single, blended and multi-overlay colormaps."""

from __future__ import annotations


import numpy as np
import logging
from typing import Optional, Dict, Any, Callable


# Output handling is configured once for the whole package in _logging.py
logger = logging.getLogger(__name__)


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
        # Validate source_layers exist
        missing_layers = [name for name in source_layers if name not in self.data_layers]
        if missing_layers:
            raise ValueError(f"Source layers not found: {missing_layers}")

        # Default kwargs dicts
        if color_kwargs is None:
            color_kwargs = {}
        if mask_kwargs is None:
            mask_kwargs = {}

        # Handle mask layer defaults
        if mask_func:
            if mask_layers is None:
                mask_layers = source_layers
            elif isinstance(mask_layers, str):
                mask_layers = [mask_layers]

            # Validate mask_layers exist
            missing_mask_layers = [name for name in mask_layers if name not in self.data_layers]
            if missing_mask_layers:
                raise ValueError(f"Mask layers not found: {missing_mask_layers}")
        else:
            mask_layers = []

        # Store mapping setup
        self.color_mapping = color_func
        self.color_sources = list(source_layers)
        self.color_kwargs = color_kwargs

        self.mask_func = mask_func
        self.mask_sources = mask_layers
        self.mask_kwargs = mask_kwargs
        self.mask_threshold = mask_threshold

        # Logging
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
        # Validate source_layers exist
        all_layers = set(base_source_layers) | set(overlay_source_layers)
        missing_layers = [name for name in all_layers if name not in self.data_layers]
        if missing_layers:
            raise ValueError(f"Source layers not found: {missing_layers}")

        # Validate overlay_mask (can be grid-space or vertex-space)
        # Actual shape validation happens in _compute_blended_colors()
        overlay_mask = np.asarray(overlay_mask)

        # Check if it's a valid 1D or 2D array
        if overlay_mask.ndim not in (1, 2):
            raise ValueError(
                f"overlay_mask must be 1D (vertex-space) or 2D (grid-space). "
                f"Got shape {overlay_mask.shape}."
            )

        # Default kwargs
        if base_color_kwargs is None:
            base_color_kwargs = {}
        if overlay_color_kwargs is None:
            overlay_color_kwargs = {}

        # Store blended color mapping configuration
        self.color_mapping_mode = "blended"
        self.base_colormap = base_colormap
        self.base_color_sources = list(base_source_layers)
        self.base_color_kwargs = base_color_kwargs
        self.overlay_colormap = overlay_colormap
        self.overlay_color_sources = list(overlay_source_layers)
        self.overlay_color_kwargs = overlay_color_kwargs
        self.overlay_mask = overlay_mask

        self.logger.info("Blended color mapping configured:")
        self.logger.info(f"  Base colormap: {base_colormap.__name__} on {base_source_layers}")
        self.logger.info(f"  Overlay colormap: {overlay_colormap.__name__} on {overlay_source_layers}")

        # Log mask info (grid-space or vertex-space)
        mask_sum = np.sum(overlay_mask)
        mask_size = overlay_mask.size
        mask_type = "grid-space" if overlay_mask.ndim == 2 else "vertex-space"
        self.logger.info(
            f"  Overlay mask ({mask_type}): {mask_sum}/{mask_size} elements "
            f"({100.0 * mask_sum / mask_size:.1f}%)"
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
        # Validate all source_layers exist
        all_layers = set(base_source_layers)
        for overlay in overlays:
            all_layers.update(overlay.get("source_layers", []))

        missing_layers = [name for name in all_layers if name not in self.data_layers]
        if missing_layers:
            raise ValueError(f"Source layers not found: {missing_layers}")

        # Validate overlay specs
        for i, overlay in enumerate(overlays):
            if "colormap" not in overlay:
                raise ValueError(f"Overlay {i} missing required 'colormap' key")
            if "source_layers" not in overlay:
                raise ValueError(f"Overlay {i} missing required 'source_layers' key")
            if "priority" not in overlay:
                raise ValueError(f"Overlay {i} missing required 'priority' key")

        # Sort overlays by priority (lower number = higher priority = applied first)
        sorted_overlays = sorted(overlays, key=lambda x: x["priority"])

        # Default kwargs
        if base_color_kwargs is None:
            base_color_kwargs = {}

        # Store multi-overlay configuration
        self.color_mapping_mode = "multi_overlay"
        self.base_colormap = base_colormap
        self.base_color_sources = list(base_source_layers)
        self.base_color_kwargs = base_color_kwargs
        self.overlays = sorted_overlays

        self.logger.info("Multi-overlay color mapping configured:")
        self.logger.info(f"  Base colormap: {base_colormap.__name__} on {base_source_layers}")
        self.logger.info(f"  Number of overlays: {len(overlays)}")
        for i, overlay in enumerate(sorted_overlays):
            has_mask = "mask" in overlay
            mask_info = f", has_mask={has_mask}" if has_mask else ""
            threshold = overlay.get("threshold", "default")
            self.logger.info(
                f"    Overlay {i} (priority {overlay['priority']}): "
                f"{overlay['colormap'].__name__} on {overlay['source_layers']}"
                f" [threshold={threshold}{mask_info}]"
            )

    def compute_colors(self, water_mask=None):
        """
        Compute colors using color_func and optionally mask_func.

        Supports three modes:
        - Standard: Single colormap applied to all vertices
        - Blended: Two colormaps blended based on proximity mask
        - Multi-overlay: Multiple overlays with different colormaps and priority

        Args:
            water_mask (np.ndarray, optional): Boolean water mask in grid space (height × width).
                For blended mode, water pixels will be colored blue in the final vertex colors.
                For standard and multi-overlay modes, water detection is handled in create_mesh().

        Returns:
            np.ndarray: RGBA color array.
        """
        # Check if multi-overlay mode
        if hasattr(self, "color_mapping_mode") and self.color_mapping_mode == "multi_overlay":
            return self._compute_multi_overlay_colors(water_mask=water_mask)

        # Check if blended mode
        if hasattr(self, "color_mapping_mode") and self.color_mapping_mode == "blended":
            return self._compute_blended_colors(water_mask=water_mask)

        # Standard single colormap mode
        if not hasattr(self, "color_mapping") or not hasattr(self, "color_sources"):
            raise ValueError("Color mapping not set. Call set_color_mapping() first.")

        self.logger.info("Computing colors...")

        # Prepare color data arrays
        color_arrays = [
            (
                self.data_layers[layer]["transformed_data"]
                if self.data_layers[layer].get("transformed")
                else self.data_layers[layer]["data"]
            )
            for layer in self.color_sources
        ]

        # Compute base colors
        try:
            colors = self.color_mapping(*color_arrays, **self.color_kwargs)
        except Exception as e:
            self.logger.error(f"Error computing colors: {str(e)}")
            raise

        colors = np.asarray(colors)
        if colors.ndim != 3 or colors.shape[-1] not in (3, 4):
            raise ValueError(
                "Color mapping must return an (H, W, 3) or (H, W, 4) array; "
                f"got shape {colors.shape}. Wrap values in a colormap such as "
                "elevation_colormap()."
            )

        # Ensure RGBA
        if colors.shape[-1] == 3:
            # Create alpha channel with appropriate max value for the data type
            if colors.dtype == np.uint8:
                alpha_channel = np.full(colors.shape[:2] + (1,), 255, dtype=colors.dtype)
            else:
                alpha_channel = np.ones(colors.shape[:2] + (1,), dtype=colors.dtype)
            colors = np.concatenate([colors, alpha_channel], axis=-1)

        # Apply mask if provided
        if self.mask_func:
            mask_arrays = [
                (
                    self.data_layers[layer]["transformed_data"]
                    if self.data_layers[layer].get("transformed")
                    else self.data_layers[layer]["data"]
                )
                for layer in self.mask_sources
            ]

            try:
                mask = self.mask_func(*mask_arrays, **self.mask_kwargs)
            except Exception as e:
                self.logger.error(f"Error computing mask: {str(e)}")
                raise

            # Apply threshold if provided
            if self.mask_threshold is not None:
                mask = mask >= self.mask_threshold

            # Update alpha channel based on mask
            colors[..., 3] = np.where(mask, 0.0, 1.0)

        self.colors = colors

        self.logger.info(f"Colors computed successfully with shape {colors.shape}")

        return colors

    def _compute_blended_colors(self, water_mask=None):
        """
        Compute colors using blended colormap mode (internal method).

        Computes base colors for entire DEM, overlay colors for overlay zones,
        and blends them according to overlay_mask at the vertex level.

        Args:
            water_mask (np.ndarray, optional): Boolean water mask in grid space (height × width).
                If provided, water pixels will be colored blue in the final vertex colors.

        Returns:
            np.ndarray: RGBA color array with blended colors.
        """
        self.logger.info("Computing blended colors...")

        # Get transformed DEM data to determine grid shape
        dem_data = self.data_layers["dem"]["transformed_data"]
        height, width = dem_data.shape

        # Compute base colors for all pixels
        base_arrays = [
            (
                self.data_layers[layer]["transformed_data"]
                if self.data_layers[layer].get("transformed")
                else self.data_layers[layer]["data"]
            )
            for layer in self.base_color_sources
        ]

        try:
            base_colors_grid = self.base_colormap(*base_arrays, **self.base_color_kwargs)
        except Exception as e:
            self.logger.error(f"Error computing base colors: {str(e)}")
            raise

        # Compute overlay colors for all pixels
        overlay_arrays = [
            (
                self.data_layers[layer]["transformed_data"]
                if self.data_layers[layer].get("transformed")
                else self.data_layers[layer]["data"]
            )
            for layer in self.overlay_color_sources
        ]

        try:
            overlay_colors_grid = self.overlay_colormap(*overlay_arrays, **self.overlay_color_kwargs)
        except Exception as e:
            self.logger.error(f"Error computing overlay colors: {str(e)}")
            raise

        # Ensure both are RGBA
        for colors_grid, name in [(base_colors_grid, "base"), (overlay_colors_grid, "overlay")]:
            if colors_grid.shape[-1] == 3:
                if colors_grid.dtype == np.uint8:
                    alpha = np.full(colors_grid.shape[:2] + (1,), 255, dtype=colors_grid.dtype)
                else:
                    alpha = np.ones(colors_grid.shape[:2] + (1,), dtype=colors_grid.dtype)
                if name == "base":
                    base_colors_grid = np.concatenate([colors_grid, alpha], axis=-1)
                else:
                    overlay_colors_grid = np.concatenate([colors_grid, alpha], axis=-1)

        # Map grid colors to vertex colors using y_valid, x_valid
        # These map vertex index to (row, col) in the grid
        base_vertex_colors = base_colors_grid[self.y_valid, self.x_valid]
        overlay_vertex_colors = overlay_colors_grid[self.y_valid, self.x_valid]

        # Handle boundary vertices if they exist
        # Boundary vertices (from boundary_extension) don't map to grid pixels
        # Use nearest valid pixel color (or default color)
        num_surface_vertices = len(self.y_valid)
        # vertices may not be set yet if compute_colors() is called early
        num_total_vertices = len(self.vertices) if self.vertices is not None else num_surface_vertices

        if num_total_vertices > num_surface_vertices:
            # Has boundary vertices - pad with default color
            self.logger.info(
                f"  Padding colors for {num_total_vertices - num_surface_vertices} boundary vertices"
            )
            # Use mean color for boundary (or could use edge colors)
            default_color = np.mean(base_vertex_colors, axis=0).astype(base_vertex_colors.dtype)

            # Extend vertex colors arrays
            base_vertex_colors = np.vstack(
                [base_vertex_colors, np.tile(default_color, (num_total_vertices - num_surface_vertices, 1))]
            )
            overlay_vertex_colors = np.vstack(
                [
                    overlay_vertex_colors,
                    np.tile(default_color, (num_total_vertices - num_surface_vertices, 1)),
                ]
            )

        # Convert overlay_mask to vertex-space if it's grid-space
        if self.overlay_mask.ndim == 2:
            # Grid-space mask: convert to vertex-space using y_valid, x_valid
            self.logger.info(f"  Converting grid-space mask to vertex-space...")
            overlay_mask_vertex = self.overlay_mask[self.y_valid, self.x_valid]

            # Pad with False for boundary vertices if they exist
            if num_total_vertices > num_surface_vertices:
                padding = np.zeros(num_total_vertices - num_surface_vertices, dtype=bool)
                overlay_mask_vertex = np.concatenate([overlay_mask_vertex, padding])
        else:
            # Already vertex-space
            overlay_mask_vertex = self.overlay_mask

        # Blend colors using overlay_mask: True = overlay, False = base
        colors = np.where(
            overlay_mask_vertex[:, None],  # Broadcast to (N, 1) for RGBA channels
            overlay_vertex_colors,
            base_vertex_colors,
        )

        num_overlay = np.sum(overlay_mask_vertex)
        num_base = len(overlay_mask_vertex) - num_overlay
        self.logger.info(
            f"Blended colors computed: {num_overlay} overlay vertices, {num_base} base vertices"
        )

        # Apply water coloring if water mask provided
        if water_mask is not None:
            self.logger.info(f"Applying water coloring to blended vertex colors...")
            self.logger.debug(f"  Water mask shape: {water_mask.shape}")
            self.logger.debug(f"  DEM shape: {dem_data.shape}")
            self.logger.debug(f"  Num surface vertices: {num_surface_vertices}")
            self.logger.debug(f"  y_valid range: {self.y_valid.min()}-{self.y_valid.max()}")
            self.logger.debug(f"  x_valid range: {self.x_valid.min()}-{self.x_valid.max()}")

            # Create shoreline vignette for water bodies (cartographic style)
            # Compute distance transform to measure distance from water edges (shores)
            from scipy.ndimage import distance_transform_edt

            water_distances = distance_transform_edt(water_mask)

            # Define gradient colors for shoreline vignette
            edge_color = np.array([25, 85, 125], dtype=np.float32)  # Light blue (shore)
            center_color = np.array([15, 50, 85], dtype=np.float32)  # Dark blue (interior)

            # Map water mask from grid space to vertex space (vectorized)
            # Only for surface vertices (not boundary vertices)
            water_at_vertices = water_mask[self.y_valid, self.x_valid]
            water_vertex_indices = np.where(water_at_vertices)[0]

            # Get raw pixel distances for all water vertices
            water_y = self.y_valid[water_vertex_indices]
            water_x = self.x_valid[water_vertex_indices]
            water_pixel_distances = water_distances[water_y, water_x]

            # Cartographic shoreline vignette style (vintage map aesthetic)
            # Gradient only in shoreline band; interior water is uniform dark
            shoreline_width_pixels = 12

            # t=1 means dark (interior), t=0 means light (at shore edge)
            # Start with all water as interior (dark)
            t = np.ones_like(water_pixel_distances)

            # Apply gradient only within shoreline band
            in_shoreline_band = water_pixel_distances < shoreline_width_pixels
            t[in_shoreline_band] = water_pixel_distances[in_shoreline_band] / shoreline_width_pixels

            # Power curve for smoother transition
            t = np.power(t, 0.5)[:, np.newaxis]
            water_colors = edge_color * (1 - t) + center_color * t

            # Apply gradient colors to water vertices
            surface_colors = colors[:num_surface_vertices]
            surface_colors[water_vertex_indices, :3] = water_colors.astype(np.uint8)
            water_vertex_count = len(water_vertex_indices)

            self.logger.info(f"Water colored blue ({water_vertex_count} vertices)")

        self.colors = colors

        return colors

    def _compute_multi_overlay_colors(self, water_mask=None):
        """
        Compute colors using multi-overlay mode (internal method).

        Combines base colormap with multiple overlays. For each grid pixel, applies the
        first overlay (by priority) whose source data is non-zero and non-NaN. Falls back
        to base colormap if no overlays match.

        Args:
            water_mask (np.ndarray, optional): Boolean water mask in grid space (height × width).
                If provided, water pixels will be colored blue in the final vertex colors.

        Returns:
            np.ndarray: RGBA color array with multi-overlay colors.
        """
        self.logger.info("Computing multi-overlay colors...")

        # Get transformed DEM data to determine grid shape (fall back to original if not transformed)
        dem_layer = self.data_layers["dem"]
        if "transformed_data" in dem_layer:
            dem_data = dem_layer["transformed_data"]
        else:
            dem_data = dem_layer["data"]
        height, width = dem_data.shape

        # Compute base colors for all pixels
        base_arrays = [
            (
                self.data_layers[layer]["transformed_data"]
                if self.data_layers[layer].get("transformed")
                else self.data_layers[layer]["data"]
            )
            for layer in self.base_color_sources
        ]

        try:
            base_colors_grid = self.base_colormap(*base_arrays, **self.base_color_kwargs)
        except Exception as e:
            self.logger.error(f"Error computing base colors: {str(e)}")
            raise

        # Ensure base colors are RGBA
        if base_colors_grid.shape[-1] == 3:
            if base_colors_grid.dtype == np.uint8:
                alpha = np.full(base_colors_grid.shape[:2] + (1,), 255, dtype=base_colors_grid.dtype)
            else:
                alpha = np.ones(base_colors_grid.shape[:2] + (1,), dtype=base_colors_grid.dtype)
            base_colors_grid = np.concatenate([base_colors_grid, alpha], axis=-1)

        # Initialize result grid with base colors
        result_colors_grid = np.copy(base_colors_grid)

        # Apply overlays in priority order
        for overlay_idx, overlay in enumerate(self.overlays):
            overlay_colormap = overlay["colormap"]
            overlay_sources = overlay["source_layers"]
            overlay_kwargs = overlay.get("colormap_kwargs", {})
            priority = overlay["priority"]

            # Get overlay source data
            overlay_arrays = [
                (
                    self.data_layers[layer]["transformed_data"]
                    if self.data_layers[layer].get("transformed")
                    else self.data_layers[layer]["data"]
                )
                for layer in overlay_sources
            ]

            try:
                overlay_colors_grid = overlay_colormap(*overlay_arrays, **overlay_kwargs)
            except Exception as e:
                self.logger.error(f"Error computing overlay {overlay_idx} colors: {str(e)}")
                raise

            # Ensure overlay colors are RGBA
            if overlay_colors_grid.shape[-1] == 3:
                if overlay_colors_grid.dtype == np.uint8:
                    alpha = np.full(overlay_colors_grid.shape[:2] + (1,), 255, dtype=overlay_colors_grid.dtype)
                else:
                    alpha = np.ones(overlay_colors_grid.shape[:2] + (1,), dtype=overlay_colors_grid.dtype)
                overlay_colors_grid = np.concatenate([overlay_colors_grid, alpha], axis=-1)

            # Create mask for where this overlay applies
            # Option 1: Explicit mask provided (e.g., park_mask for proximity-based overlays)
            # Option 2: Threshold-based mask from first source layer value
            explicit_mask = overlay.get("mask", None)
            use_explicit_mask = False

            if explicit_mask is not None:
                # Explicit mask provided - can be grid-space or vertex-space
                explicit_mask = np.asarray(explicit_mask)
                grid_shape = overlay_arrays[0].shape
                has_mesh = hasattr(self, "y_valid") and self.y_valid is not None

                if explicit_mask.shape == grid_shape:
                    # Already grid-space - preferred form
                    overlay_mask = explicit_mask.astype(bool)
                    use_explicit_mask = True
                    self.logger.info(
                        f"Overlay {overlay_idx}: using grid-space mask directly, "
                        f"{np.sum(overlay_mask)} grid pixels"
                    )
                elif has_mesh and explicit_mask.shape == (len(self.y_valid),):
                    # Convert vertex mask to grid mask (exact match)
                    overlay_mask = np.zeros(grid_shape, dtype=bool)
                    overlay_mask[self.y_valid, self.x_valid] = explicit_mask
                    use_explicit_mask = True
                    self.logger.info(
                        f"Overlay {overlay_idx}: converted vertex mask to grid mask, "
                        f"{np.sum(overlay_mask)} grid pixels"
                    )
                elif has_mesh and len(explicit_mask.shape) == 1 and len(explicit_mask) >= len(self.y_valid):
                    # Vertex-space mask but might include boundary vertices
                    self.logger.info(
                        f"Overlay {overlay_idx}: mask has {len(explicit_mask)} entries, "
                        f"using first {len(self.y_valid)} for surface vertices"
                    )
                    overlay_mask = np.zeros(grid_shape, dtype=bool)
                    overlay_mask[self.y_valid, self.x_valid] = explicit_mask[:len(self.y_valid)]
                    use_explicit_mask = True
                    self.logger.info(
                        f"Overlay {overlay_idx}: converted vertex mask to grid mask, "
                        f"{np.sum(overlay_mask)} grid pixels"
                    )
                else:
                    self.logger.warning(
                        f"Overlay {overlay_idx}: mask shape {explicit_mask.shape} doesn't match "
                        f"grid shape {grid_shape}. Falling back to threshold-based mask."
                    )

            if not use_explicit_mask:
                # Use threshold-based mask from source layer values
                overlay_mask_data = overlay_arrays[0]
                threshold = overlay.get("threshold", 0.5)

                # Special case: threshold=0.0 should exclude zeros (use > not >=)
                # This handles sparse layers like stream networks where non-feature pixels are 0
                if threshold == 0.0:
                    overlay_mask = (overlay_mask_data > 0.0) & ~np.isnan(overlay_mask_data)
                else:
                    overlay_mask = (overlay_mask_data >= threshold) & ~np.isnan(overlay_mask_data)

                self.logger.info(
                    f"Overlay {overlay_idx}: using threshold={threshold}, "
                    f"{np.sum(overlay_mask)} grid pixels"
                )

            # Apply overlay colors where mask is True
            result_colors_grid[overlay_mask] = overlay_colors_grid[overlay_mask]

            self.logger.debug(
                f"Overlay {overlay_idx} (priority {priority}): "
                f"applied to {np.sum(overlay_mask)} grid pixels"
            )

        # Store grid-level colors (create_mesh will handle vertex mapping and water coloring)
        # This matches the pattern used by standard mode compute_colors()
        self.colors = result_colors_grid

        self.logger.info(f"Multi-overlay colors computed: {len(self.overlays)} overlays applied")
        self.logger.info(f"Colors stored as grid-space array: {result_colors_grid.shape}")
        return result_colors_grid
