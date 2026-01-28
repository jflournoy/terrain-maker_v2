"""
Transformation pipeline for bounds visualization.

Encapsulates the multi-stage transformation from WGS84 original DEM through
reprojection, flipping, and downsampling to final mesh grid.

IMPORTANT: The WGS84 → UTM transformation is NON-LINEAR (Transverse Mercator
projection involves trigonometric relationships). A rectangular boundary in
WGS84 becomes CURVED when projected to UTM. This module uses pyproj to handle
the proper projection math, not linear approximations.

This module provides reusable components for both visualization and testing.
"""

import numpy as np
from typing import Tuple, List, Optional

try:
    from pyproj import Transformer as PyProjTransformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False


class SimpleAffine:
    """Minimal affine transform for mapping between coordinate spaces.

    Represents the affine transform equation:
        x_world = c + pixel_x * a + pixel_y * b
        y_world = f + pixel_x * d + pixel_y * e

    Where (c, f) is the top-left corner, (a, e) are pixel scales,
    and (b, d) handle rotation/skew.
    """

    def __init__(self, c: float, d: float, e: float, f: float, a: float, b: float):
        """Initialize affine transform coefficients.

        Args:
            c: X-coordinate of top-left corner (easting/longitude)
            d: Pixel spacing in x direction for y (usually 0 for aligned grid)
            e: Y-coordinate scale (usually negative for top-to-bottom scanning)
            f: Y-coordinate of top-left corner (northing/latitude)
            a: Pixel spacing in x direction (meters or degrees per pixel)
            b: Pixel spacing in y direction for x (usually 0 for aligned grid)
        """
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.a = a
        self.b = b

    def map_pixel_to_world(self, y: int, x: int) -> Tuple[float, float]:
        """Map grid pixel coordinates to world coordinates.

        Args:
            y: Row index (0 = top)
            x: Column index (0 = left)

        Returns:
            Tuple of (world_x, world_y) coordinates
        """
        world_x = self.c + x * self.a + y * self.b
        world_y = self.f + x * self.d + y * self.e
        return world_x, world_y


class TransformationPipeline:
    """Manages the transformation pipeline from WGS84 to final mesh grid.

    Pipeline stages:
    1. Original: WGS84 coordinates, full resolution
    2. Reprojected: UTM, distorted by NON-LINEAR projection (Transverse Mercator)
    3. Flipped: Same shape as reprojected, but horizontally flipped
    4. Final: Downsampled mesh grid (20×49 pixels for Detroit)

    IMPORTANT: The WGS84 → UTM transformation is non-linear. A rectangular
    boundary in WGS84 becomes curved in UTM space due to the Transverse
    Mercator projection's trigonometric relationships.
    """

    def __init__(
        self,
        original_shape: Tuple[int, int],
        distortion_factor: float,
        target_vertices: int,
        bbox_wgs84: Optional[Tuple[float, float, float, float]] = None,
        bbox_utm: Optional[Tuple[float, float, float, float]] = None,
        src_crs: str = "EPSG:4326",
        dst_crs: str = "EPSG:32617",
    ):
        """Initialize transformation pipeline.

        Args:
            original_shape: (height, width) of original DEM in WGS84
            distortion_factor: Height compression ratio from projection (e.g., 0.87)
                Used for shape estimation; actual projection uses pyproj
            target_vertices: Target number of vertices for final mesh
            bbox_wgs84: Optional (min_lon, min_lat, max_lon, max_lat) in WGS84
            bbox_utm: Optional (min_easting, min_northing, max_easting, max_northing) in UTM
            src_crs: Source coordinate reference system (default: EPSG:4326 / WGS84)
            dst_crs: Destination coordinate reference system (default: EPSG:32617 / UTM 17N)
        """
        self.original_shape = original_shape
        self.distortion_factor = distortion_factor
        self.target_vertices = target_vertices
        self.bbox_wgs84 = bbox_wgs84
        self.bbox_utm = bbox_utm
        self.src_crs = src_crs
        self.dst_crs = dst_crs

        # Compute derived shapes
        self._compute_shapes()

    def _compute_shapes(self) -> None:
        """Compute grid shapes at each transformation stage.

        Note: The distortion_factor is used to estimate the reprojected shape.
        This is an approximation - the actual WGS84→UTM transformation is non-linear
        and a WGS84 rectangle becomes a curved quadrilateral in UTM space.
        """
        height, width = self.original_shape

        # Stage 2: Reprojection distortion (estimated)
        # Add 1 to ensure boundary pixels aren't filtered at exact edge
        self.reprojected_shape = (
            int(height * self.distortion_factor) + 1,
            int(width / self.distortion_factor) + 1,
        )

        # Stage 3: Flipped (shape unchanged)
        self.flipped_shape = self.reprojected_shape

        # Stage 4: Final mesh grid downsampling
        rep_height, rep_width = self.reprojected_shape
        aspect_ratio = rep_height / rep_width

        # Solve: height * width = target_vertices, height/width = aspect_ratio
        final_height = int(np.sqrt(self.target_vertices * aspect_ratio))
        final_width = int(np.sqrt(self.target_vertices / aspect_ratio))

        self.final_shape = (final_height, final_width)

        # Downsampling factors (from reprojected to final)
        self.downsample_factors = (
            rep_height / final_height,
            rep_width / final_width,
        )

    def get_shape(self, stage: str) -> Tuple[int, int]:
        """Get grid shape at a transformation stage.

        Args:
            stage: One of 'original', 'reprojected', 'flipped', 'final'

        Returns:
            (height, width) tuple for that stage
        """
        if stage == 'original':
            return self.original_shape
        elif stage == 'reprojected':
            return self.reprojected_shape
        elif stage == 'flipped':
            return self.flipped_shape
        elif stage == 'final':
            return self.final_shape
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def get_affine(self, stage: str) -> SimpleAffine:
        """Get affine transform for a stage.

        Args:
            stage: One of 'original', 'reprojected', 'final'

        Returns:
            SimpleAffine transform for mapping pixels to world coordinates
        """
        if stage == 'original':
            if self.bbox_wgs84 is None:
                raise ValueError("bbox_wgs84 required to compute original affine")
            min_lon, min_lat, max_lon, max_lat = self.bbox_wgs84
            height, width = self.original_shape
            scale_x = (max_lon - min_lon) / width
            scale_y = -(max_lat - min_lat) / height
            return SimpleAffine(
                c=min_lon, f=max_lat,
                a=scale_x, e=scale_y,
                b=0, d=0
            )
        elif stage == 'reprojected':
            if self.bbox_utm is None:
                raise ValueError("bbox_utm required to compute reprojected affine")
            min_easting, min_northing, max_easting, max_northing = self.bbox_utm
            height, width = self.reprojected_shape
            scale_x = (max_easting - min_easting) / width
            scale_y = -(max_northing - min_northing) / height
            return SimpleAffine(
                c=min_easting, f=max_northing,
                a=scale_x, e=scale_y,
                b=0, d=0
            )
        else:
            raise ValueError(f"No affine for stage: {stage}")


class EdgeTransformer:
    """Transforms edge pixels through the transformation pipeline.

    Applies the sequence of transformations (reprojection, flip, downsample)
    to edge pixel coordinates.

    IMPORTANT: The WGS84 → UTM reprojection uses pyproj for proper non-linear
    Transverse Mercator projection. This means rectangular edges in WGS84
    become curved in UTM space.
    """

    def __init__(self, pipeline: TransformationPipeline, use_pyproj: bool = True):
        """Initialize with a transformation pipeline.

        Args:
            pipeline: TransformationPipeline instance
            use_pyproj: If True, use pyproj for proper non-linear projection.
                If False, fall back to linear approximation (for testing).
        """
        self.pipeline = pipeline
        self.use_pyproj = use_pyproj and HAS_PYPROJ

        # Create pyproj transformer if available and enabled
        self._proj_transformer = None
        if self.use_pyproj and pipeline.bbox_wgs84 is not None:
            self._proj_transformer = PyProjTransformer.from_crs(
                pipeline.src_crs, pipeline.dst_crs, always_xy=True
            )

    def transform_stage(
        self,
        pixels: List[Tuple[int, int]],
        from_stage: str,
        to_stage: str,
    ) -> List[Tuple[int, int]]:
        """Transform pixels from one stage to the next.

        Args:
            pixels: List of (y, x) pixel coordinates
            from_stage: Source stage name
            to_stage: Destination stage name

        Returns:
            List of (y, x) coordinates in destination stage
        """
        # Map stage transitions to transformation methods
        stage_order = ['original', 'reprojected', 'flipped', 'final']
        from_idx = stage_order.index(from_stage)
        to_idx = stage_order.index(to_stage)

        if from_idx >= to_idx:
            raise ValueError(f"Cannot transform backward: {from_stage} → {to_stage}")

        result = pixels
        for i in range(from_idx, to_idx):
            current_stage = stage_order[i]
            next_stage = stage_order[i + 1]
            result = self._apply_single_transform(result, current_stage, next_stage)

        return result

    def _apply_single_transform(
        self,
        pixels: List[Tuple[int, int]],
        from_stage: str,
        to_stage: str,
    ) -> List[Tuple[int, int]]:
        """Apply a single transformation step.

        Args:
            pixels: Input pixel coordinates
            from_stage: Source stage
            to_stage: Destination stage

        Returns:
            Transformed pixel coordinates
        """
        if from_stage == 'original' and to_stage == 'reprojected':
            # Apply projection distortion
            return self._transform_original_to_reprojected(pixels)
        elif from_stage == 'reprojected' and to_stage == 'flipped':
            # Apply horizontal flip
            return self._transform_reprojected_to_flipped(pixels)
        elif from_stage == 'flipped' and to_stage == 'final':
            # Apply downsampling
            return self._transform_flipped_to_final(pixels)
        else:
            raise ValueError(f"Unknown transformation: {from_stage} → {to_stage}")

    def _transform_original_to_reprojected(self, pixels: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Apply WGS84 → UTM projection (non-linear Transverse Mercator).

        If pyproj is available and bbox_wgs84/bbox_utm are provided, uses proper
        coordinate transformation. Otherwise falls back to linear approximation.
        """
        # Use proper projection if available
        if self._proj_transformer is not None and self.pipeline.bbox_wgs84 is not None:
            return self._transform_with_pyproj(pixels)

        # Fallback: linear approximation (for backward compatibility)
        distortion_factor = self.pipeline.distortion_factor
        return [
            (int(y * distortion_factor), int(x / distortion_factor))
            for y, x in pixels
        ]

    def _transform_with_pyproj(self, pixels: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Transform pixels using proper coordinate projection via pyproj.

        Flow: pixel → WGS84 geographic → UTM geographic → pixel in reprojected grid

        This captures the NON-LINEAR behavior of the Transverse Mercator projection,
        where straight lines in WGS84 become curved in UTM.

        Note: A WGS84 rectangle becomes a curved quadrilateral in UTM space.
        Edge pixels may map to positions that don't align with the rectangular
        reprojected grid boundaries.
        """
        if self.pipeline.bbox_wgs84 is None or self.pipeline.bbox_utm is None:
            raise ValueError("bbox_wgs84 and bbox_utm required for pyproj transformation")

        # Get affines for pixel ↔ geographic coordinate conversion
        affine_wgs84 = self.pipeline.get_affine('original')
        affine_utm = self.pipeline.get_affine('reprojected')

        rep_h, rep_w = self.pipeline.reprojected_shape

        result = []
        for y_orig, x_orig in pixels:
            # 1. Pixel → WGS84 geographic coordinates
            lon, lat = affine_wgs84.map_pixel_to_world(y_orig, x_orig)

            # 2. WGS84 → UTM (non-linear Transverse Mercator projection!)
            easting, northing = self._proj_transformer.transform(lon, lat)

            # 3. UTM geographic → pixel in reprojected grid
            # Need to invert the UTM affine: pixel = (geo - offset) / scale
            x_reproj = (easting - affine_utm.c) / affine_utm.a
            y_reproj = (northing - affine_utm.f) / affine_utm.e

            # Round to integer pixel coordinates
            y_int, x_int = int(round(y_reproj)), int(round(x_reproj))

            # Clamp to valid bounds (preserves edge pixels that land near boundary)
            y_int = max(0, min(rep_h - 1, y_int))
            x_int = max(0, min(rep_w - 1, x_int))

            result.append((y_int, x_int))

        return result

    def _transform_with_pyproj_fractional(self, pixels: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        """Transform pixels using pyproj, returning FRACTIONAL coordinates without clamping.

        This preserves the true curved boundary from the non-linear projection,
        even when coordinates extend beyond the rectangular grid bounds.

        Returns:
            List of (y, x) tuples with fractional precision, unclamped.
        """
        if self.pipeline.bbox_wgs84 is None or self.pipeline.bbox_utm is None:
            raise ValueError("bbox_wgs84 and bbox_utm required for pyproj transformation")

        affine_wgs84 = self.pipeline.get_affine('original')
        affine_utm = self.pipeline.get_affine('reprojected')

        result = []
        for y_orig, x_orig in pixels:
            # 1. Pixel → WGS84 geographic coordinates
            lon, lat = affine_wgs84.map_pixel_to_world(y_orig, x_orig)

            # 2. WGS84 → UTM (non-linear Transverse Mercator projection!)
            easting, northing = self._proj_transformer.transform(lon, lat)

            # 3. UTM geographic → fractional pixel in reprojected grid (NO clamping)
            x_reproj = (easting - affine_utm.c) / affine_utm.a
            y_reproj = (northing - affine_utm.f) / affine_utm.e

            result.append((y_reproj, x_reproj))

        return result

    def _transform_reprojected_to_flipped(self, pixels: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Apply horizontal flip (mirror left-right)."""
        _, width = self.pipeline.reprojected_shape
        return [
            (y, width - 1 - x)
            for y, x in pixels
        ]

    def _transform_flipped_to_final(self, pixels: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Downsample to final mesh grid (integer coordinates).

        Uses clamping instead of filtering to preserve edge pixels that land
        at or slightly beyond the grid boundary (due to rounding).

        Note: For smooth edge extrusion, use transform_to_mesh_space() instead
        to get fractional coordinates.
        """
        downsample_y, downsample_x = self.pipeline.downsample_factors
        final_height, final_width = self.pipeline.final_shape

        result = []
        for y, x in pixels:
            # Downsample with rounding
            y_final = int(np.round(y / downsample_y))
            x_final = int(np.round(x / downsample_x))

            # Clamp to valid grid bounds (preserves boundary pixels)
            y_final = max(0, min(final_height - 1, y_final))
            x_final = max(0, min(final_width - 1, x_final))

            result.append((y_final, x_final))

        # Remove duplicates created by rounding/clamping
        return list(set(result))

    def transform_to_mesh_space(
        self,
        pixels: List[Tuple[int, int]],
        from_stage: str = 'original',
        clamp: bool = True,
    ) -> List[Tuple[float, float]]:
        """Transform edge pixels to mesh coordinate space with FRACTIONAL precision.

        Unlike transform_full_pipeline() which snaps to integer grid cells,
        this method preserves fractional coordinates for smooth edge extrusion.

        Args:
            pixels: Edge pixel coordinates (y, x) in the source stage
            from_stage: Source stage ('original', 'reprojected', or 'flipped')
            clamp: If True, clamp coordinates to [0, dim-1]. If False, return
                true projected coordinates which may exceed grid bounds due to
                non-linear projection curvature.

        Returns:
            List of (y, x) coordinates in mesh space with fractional precision.
            If clamp=True, values range from 0.0 to (height-1) and 0.0 to (width-1).
            If clamp=False, values may exceed these bounds.
        """
        downsample_y, downsample_x = self.pipeline.downsample_factors
        final_height, final_width = self.pipeline.final_shape
        _, rep_width = self.pipeline.reprojected_shape

        # When clamp=False and from_stage='original', use fractional pyproj to preserve
        # the true curved boundary from the non-linear projection
        if not clamp and from_stage == 'original' and self._proj_transformer is not None:
            # Get fractional reprojected coordinates (preserves curvature)
            reproj_frac = self._transform_with_pyproj_fractional(pixels)

            # Apply flip (mirror x around center)
            flipped_frac = [(y, rep_width - 1 - x) for y, x in reproj_frac]

            # Scale to mesh space
            result = []
            for y, x in flipped_frac:
                y_mesh = y / downsample_y
                x_mesh = x / downsample_x
                result.append((y_mesh, x_mesh))

            return result

        # Standard path: use integer transformation (with clamping at intermediate stages)
        if from_stage == 'original':
            flipped = self.transform_stage(pixels, 'original', 'flipped')
        elif from_stage == 'reprojected':
            flipped = self.transform_stage(pixels, 'reprojected', 'flipped')
        elif from_stage == 'flipped':
            flipped = pixels
        else:
            raise ValueError(f"Unknown stage: {from_stage}")

        result = []
        for y, x in flipped:
            # Scale to mesh coordinates (fractional)
            y_mesh = y / downsample_y
            x_mesh = x / downsample_x

            if clamp:
                # Clamp to valid range [0, dim-1]
                y_mesh = max(0.0, min(float(final_height - 1), y_mesh))
                x_mesh = max(0.0, min(float(final_width - 1), x_mesh))

            result.append((y_mesh, x_mesh))

        return result

    def transform_full_pipeline(self, pixels: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Apply full transformation from original to final.

        Args:
            pixels: Edge pixel coordinates in original WGS84 space

        Returns:
            Edge pixels in final mesh grid coordinates
        """
        return self.transform_stage(pixels, 'original', 'final')
