"""Water body detection from slope."""

from __future__ import annotations


import numpy as np
import logging




# Output handling is configured once for the whole package in _logging.py
logger = logging.getLogger(__name__)


class TerrainWaterMixin:
    """Terrain methods: water body detection from slope."""

    def detect_water_highres(
        self,
        slope_threshold: float = 0.01,
        fill_holes: bool = True,
        scale_factor: float = 0.0001,
    ) -> np.ndarray:
        """
        Detect water bodies on high-resolution DEM before downsampling.

        This method properly handles water detection by:
        1. Applying all NON-downsampling transforms to DEM (reproject, flip, etc.)
        2. Detecting water on the high-resolution transformed DEM
        3. Downsampling the water mask to match the final terrain resolution

        This prevents false water detection that occurs when detecting on downsampled DEMs.

        Args:
            slope_threshold: Maximum slope magnitude to classify as water (default: 0.01)
            fill_holes: Whether to apply morphological hole filling (default: True)
            scale_factor: Elevation scale factor to unscale before detection (default: 0.0001)

        Returns:
            np.ndarray: Boolean water mask matching transformed_data shape

        Raises:
            ValueError: If transforms haven't been applied yet
            ImportError: If water detection module is not available

        Example::

            terrain = Terrain(dem, transform, dem_crs="EPSG:4326")
            terrain.add_transform(reproject_raster(src_crs="EPSG:4326", dst_crs="EPSG:32617"))
            terrain.add_transform(flip_raster(axis="horizontal"))
            terrain.add_transform(scale_elevation(scale_factor=0.0001))
            terrain.configure_for_target_vertices(target_vertices=1_000_000)
            terrain.apply_transforms()  # Includes downsampling

            # Detect water on high-res BEFORE it was downsampled
            water_mask = terrain.detect_water_highres(slope_threshold=0.01)
            # water_mask shape matches downsampled terrain

        Note:
            This method requires that:
            - Transforms have been applied (apply_transforms() called)
            - The DEM layer exists in data_layers
            - Water detection module (src.terrain.water) is available
        """
        if "dem" not in self.data_layers:
            raise ValueError("DEM layer not found. Cannot detect water without DEM.")

        if not self.data_layers["dem"].get("transformed", False):
            raise ValueError(
                "Transforms have not been applied yet. Call apply_transforms() first."
            )

        from src.terrain.water import identify_water_by_slope
        from scipy.ndimage import zoom

        self.logger.info("Detecting water bodies on high-resolution DEM...")

        # Get the original DEM
        original_dem = self.data_layers["dem"]["data"]
        original_transform = self.data_layers["dem"]["transform"]
        original_crs = self.data_layers["dem"]["crs"]

        # Apply all non-downsampling transforms
        highres_dem = original_dem.copy()
        current_transform = original_transform
        current_crs = original_crs

        for transform_func in self.transforms:
            # Skip downsampling transforms - we want high-res
            if "downsample" in transform_func.__name__.lower():
                self.logger.debug(f"Skipping {transform_func.__name__} for high-res water detection")
                continue

            try:
                highres_dem, current_transform, new_crs = transform_func(
                    highres_dem, current_transform
                )
                if new_crs is not None:
                    current_crs = new_crs
            except Exception as e:
                self.logger.error(
                    f"Failed applying transform {transform_func.__name__} for water detection: {e}"
                )
                raise

        self.logger.info(f"  High-res DEM shape: {highres_dem.shape}")

        # Unscale elevation if scale_factor was applied
        if scale_factor is not None and scale_factor != 1.0:
            unscaled_dem = highres_dem / scale_factor
        else:
            unscaled_dem = highres_dem

        # Detect water on high-res DEM
        water_mask_highres = identify_water_by_slope(
            unscaled_dem,
            slope_threshold=slope_threshold,
            fill_holes=fill_holes,
        )

        water_pixels_highres = np.sum(water_mask_highres)
        water_percent_highres = 100 * water_pixels_highres / water_mask_highres.size
        self.logger.info(
            f"  High-res water detection: {water_pixels_highres:,} pixels "
            f"({water_percent_highres:.1f}%)"
        )

        # Now downsample the water mask to match the FINAL transformed DEM
        # (after ALL transforms including adaptive downsampling)
        # We need to apply ALL transforms to get the actual final shape
        final_dem = original_dem.copy()
        final_transform = original_transform
        final_crs = original_crs

        for transform_func in self.transforms:
            try:
                final_dem, final_transform, new_crs = transform_func(
                    final_dem, final_transform
                )
                if new_crs is not None:
                    final_crs = new_crs
            except Exception as e:
                self.logger.error(
                    f"Failed applying transform {transform_func.__name__} for final shape: {e}"
                )
                raise

        target_shape = final_dem.shape
        self.logger.debug(
            f"  Final DEM shape after all transforms: {target_shape}"
        )

        if water_mask_highres.shape == target_shape:
            self.logger.info(f"  Water mask already matches target shape {target_shape}")
            return water_mask_highres

        # Calculate zoom factor for downsampling the mask
        zoom_y = target_shape[0] / water_mask_highres.shape[0]
        zoom_x = target_shape[1] / water_mask_highres.shape[1]

        self.logger.info(
            f"  Downsampling water mask: {water_mask_highres.shape} → {target_shape}"
        )

        # Downsample water mask using nearest-neighbor (order=0) to preserve boolean nature
        water_mask_downsampled = zoom(
            water_mask_highres.astype(np.float32),
            zoom=(zoom_y, zoom_x),
            order=0,  # Nearest neighbor for boolean mask
            prefilter=False,
        ).astype(np.bool_)

        water_pixels_final = np.sum(water_mask_downsampled)
        water_percent_final = 100 * water_pixels_final / water_mask_downsampled.size
        self.logger.info(
            f"  Final water mask: {water_pixels_final:,} pixels "
            f"({water_percent_final:.1f}%)"
        )

        return water_mask_downsampled

    def detect_water(
        self,
        slope_threshold: float = 0.01,
        fill_holes: bool = True,
    ) -> np.ndarray:
        """
        Detect water bodies on the transformed (downsampled) DEM.

        This is a fast, simple water detection method that works on the already-
        transformed DEM. It automatically handles elevation scale factor unscaling.

        For higher accuracy at the cost of speed, use detect_water_highres() instead,
        which detects water on the full-resolution DEM before downsampling.

        Args:
            slope_threshold: Maximum slope magnitude to classify as water (default: 0.01).
                            Water bodies have nearly zero slope.
            fill_holes: Whether to apply morphological hole filling (default: True)

        Returns:
            np.ndarray: Boolean water mask matching transformed_data shape

        Raises:
            ValueError: If transforms haven't been applied yet

        Example:
            ```python
            terrain = Terrain(dem, transform)
            terrain.apply_transforms()
            water_mask = terrain.detect_water()
            terrain.compute_colors(water_mask=water_mask)
            ```
        """
        if "dem" not in self.data_layers:
            raise ValueError("DEM layer not found. Cannot detect water without DEM.")

        if not self.data_layers["dem"].get("transformed", False):
            raise ValueError(
                "Transforms have not been applied yet. Call apply_transforms() first."
            )

        from src.terrain.water import identify_water_by_slope

        self.logger.info("Detecting water bodies on transformed DEM...")

        # Get the transformed DEM
        dem_data = self.data_layers["dem"]["transformed_data"]

        # Auto-detect elevation scale factor from model_params or estimate from data range
        scale_factor = self.model_params.get("elevation_scale", None)

        if scale_factor is None:
            # Estimate: if max elevation is << 1, it was probably scaled
            max_elev = np.nanmax(dem_data)
            if max_elev < 1.0:
                # Likely scaled by 0.0001 (max ~0.03 for 300m terrain)
                scale_factor = 0.0001
                self.logger.debug(f"Auto-detected scale factor: {scale_factor}")
            else:
                scale_factor = 1.0  # Already in meters

        # Unscale to meters for proper slope calculation
        if scale_factor != 1.0:
            unscaled_dem = dem_data / scale_factor
            self.logger.debug(f"Unscaled DEM from {np.nanmax(dem_data):.4f} to {np.nanmax(unscaled_dem):.1f}m")
        else:
            unscaled_dem = dem_data

        # Detect water
        water_mask = identify_water_by_slope(
            unscaled_dem,
            slope_threshold=slope_threshold,
            fill_holes=fill_holes,
        )

        water_pixels = np.sum(water_mask)
        water_percent = 100 * water_pixels / water_mask.size
        self.logger.info(f"  Water detected: {water_pixels:,} pixels ({water_percent:.1f}%)")

        return water_mask
