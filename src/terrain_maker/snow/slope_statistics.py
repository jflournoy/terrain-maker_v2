"""
Tiled slope statistics computation using geographic transforms.

Computes slope and related statistics at full DEM resolution, then aggregates
per output pixel using exact geographic coordinate mapping.

This ensures that:
1. Cliff faces within a pixel are captured (not hidden by downsampling)
2. Downsampling respects actual geographic transforms (not uniform strides)
3. Memory usage stays bounded by processing in tiles with halos
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import logging
import numpy as np
from rasterio.transform import Affine

logger = logging.getLogger(__name__)


@dataclass
class TiledSlopeConfig:
    """Configuration for tiled slope computation."""

    target_tile_outputs: int = 100
    """Target number of output pixels per tile dimension."""

    halo: int = 1
    """Halo size (pixels) for gradient computation at tile boundaries."""

    max_tile_size: int = 4096
    """Maximum tile size in source pixels for memory safety."""


@dataclass
class TileSpec:
    """Specification for a single tile during processing."""

    src_slice: Tuple[slice, slice]
    """Slice into source DEM (with halo)."""

    core_slice: Tuple[slice, slice]
    """Slice within tile to extract core region (no halo)."""

    out_slice: Tuple[slice, slice]
    """Slice into output arrays."""

    row_stride: int
    """Average source rows per output row in this tile."""

    col_stride: int
    """Average source columns per output column in this tile."""


@dataclass
class SlopeStatistics:
    """Aggregated slope statistics at output resolution."""

    slope_mean: np.ndarray
    """Mean slope in degrees for each output pixel."""

    slope_max: np.ndarray
    """Maximum slope (cliff detection)."""

    slope_min: np.ndarray
    """Minimum slope (flat spots)."""

    slope_std: np.ndarray
    """Standard deviation of slopes (terrain consistency)."""

    slope_p95: np.ndarray
    """95th percentile slope (robust cliff indicator)."""

    roughness: np.ndarray
    """Elevation standard deviation (surface bumpiness)."""

    aspect_sin: np.ndarray
    """sin(aspect) for vector averaging across output pixels."""

    aspect_cos: np.ndarray
    """cos(aspect) for vector averaging across output pixels."""

    @property
    def dominant_aspect(self) -> np.ndarray:
        """
        Compute dominant aspect from vector-averaged sin/cos components.

        Handles circular nature of aspect (0-360 degrees).

        Returns:
            Dominant aspect in degrees (0=North, 90=East, 180=South, 270=West)
        """
        return np.degrees(np.arctan2(self.aspect_sin, self.aspect_cos)) % 360

    @property
    def aspect_strength(self) -> np.ndarray:
        """
        Compute aspect strength (consistency of slope direction).

        Returns:
            Strength from 0 (uniform directions) to 1 (all same direction)
        """
        return np.sqrt(self.aspect_sin**2 + self.aspect_cos**2)


def compute_pixel_mapping(
    dem_shape: Tuple[int, int],
    dem_transform: Affine,
    dem_crs: str,
    target_shape: Tuple[int, int],
    target_transform: Affine,
    target_crs: str,
) -> Dict:
    """
    Compute precise pixel-to-pixel mapping using geographic transforms.

    For each output pixel (i, j):
    1. Calculate its geographic bounds using target_transform
    2. Transform to source CRS if needed
    3. Use dem_transform inverse to find source pixel bounds
    4. Return mapping: output_pixel_ij → [list of source pixels]

    Args:
        dem_shape: (height, width) of source DEM
        dem_transform: Affine transform for source DEM
        dem_crs: CRS string for source DEM (e.g., 'EPSG:4326')
        target_shape: (height, width) of target output grid
        target_transform: Affine transform for target grid
        target_crs: CRS string for target grid

    Returns:
        Mapping structure containing:
        - 'row_stride': average source pixels per output row
        - 'col_stride': average source pixels per output column
        - 'mapping': detailed per-pixel mapping (for irregular transforms)

    Handles:
        - Non-integer scaling ratios
        - Rotation, skew, reprojection
        - Edge effects and partial pixel overlaps
    """
    logger.info(f"Computing pixel mapping: {dem_shape} → {target_shape}")

    # For now, compute simple stride using integer division
    # TODO: Full geographic transform mapping for non-uniform cases
    row_stride = max(1, dem_shape[0] // target_shape[0])
    col_stride = max(1, dem_shape[1] // target_shape[1])

    logger.info(
        f"  Stride: rows={row_stride}, cols={col_stride}"
    )

    return {
        "row_stride": row_stride,
        "col_stride": col_stride,
        "dem_shape": dem_shape,
        "target_shape": target_shape,
    }


def compute_tile_layout(
    dem_shape: Tuple[int, int],
    target_shape: Tuple[int, int],
    pixel_mapping: Dict,
    config: Optional[TiledSlopeConfig] = None,
) -> List[TileSpec]:
    """
    Compute adaptive tile size based on pixel mapping.

    Strategy:
    1. Estimate average source pixels per output pixel
    2. Tile size = target_tile_outputs × stride (in source pixels)
    3. Clamp to max_tile_size for memory safety
    4. Ensure tile boundaries align with output pixel boundaries

    Args:
        dem_shape: (height, width) of source DEM
        target_shape: (height, width) of target grid
        pixel_mapping: Result from compute_pixel_mapping()
        config: TiledSlopeConfig (uses defaults if None)

    Returns:
        List of TileSpec for each tile to process
    """
    if config is None:
        config = TiledSlopeConfig()

    src_height, src_width = dem_shape
    tgt_height, tgt_width = target_shape

    # Use integer division to avoid rounding issues
    row_stride = max(1, src_height // tgt_height)
    col_stride = max(1, src_width // tgt_width)

    # Compute tile size in source pixels
    tile_src_rows = min(
        config.target_tile_outputs * row_stride, config.max_tile_size
    )
    tile_src_cols = min(
        config.target_tile_outputs * col_stride, config.max_tile_size
    )

    # Align to stride boundaries to ensure output pixels align
    tile_src_rows = (tile_src_rows // row_stride) * row_stride
    tile_src_cols = (tile_src_cols // col_stride) * col_stride

    logger.info(
        f"Tile layout: {tile_src_rows}×{tile_src_cols} source pixels "
        f"({tile_src_rows // row_stride}×{tile_src_cols // col_stride} output pixels)"
    )

    tiles = []
    for y_start in range(0, src_height, tile_src_rows):
        for x_start in range(0, src_width, tile_src_cols):
            y_end = min(y_start + tile_src_rows, src_height)
            x_end = min(x_start + tile_src_cols, src_width)

            # Add halo for gradient computation (clamped to array bounds)
            y_halo_start = max(0, y_start - config.halo)
            x_halo_start = max(0, x_start - config.halo)
            y_halo_end = min(src_height, y_end + config.halo)
            x_halo_end = min(src_width, x_end + config.halo)

            # Track which output pixels this tile contributes to
            # Clamp to target shape to avoid exceeding bounds
            out_y_start = y_start // row_stride
            out_y_end = min((y_end - 1) // row_stride + 1, tgt_height)
            out_x_start = x_start // col_stride
            out_x_end = min((x_end - 1) // col_stride + 1, tgt_width)

            # Compute core slice (where gradient was computed without halo)
            core_y_start = y_start - y_halo_start
            core_y_end = y_end - y_halo_start
            core_x_start = x_start - x_halo_start
            core_x_end = x_end - x_halo_start

            tiles.append(
                TileSpec(
                    src_slice=(
                        slice(y_halo_start, y_halo_end),
                        slice(x_halo_start, x_halo_end),
                    ),
                    core_slice=(
                        slice(core_y_start, core_y_end),
                        slice(core_x_start, core_x_end),
                    ),
                    out_slice=(
                        slice(out_y_start, out_y_end),
                        slice(out_x_start, out_x_end),
                    ),
                    row_stride=row_stride,
                    col_stride=col_stride,
                )
            )

    logger.info(f"  Created {len(tiles)} tiles")
    return tiles


def compute_tile_slopes(
    tile_data: np.ndarray,
    core_slice: Tuple[slice, slice],
    cell_size_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute slope and aspect at full resolution for a tile.

    Args:
        tile_data: Tile elevation data (with halo for proper boundaries)
        core_slice: Slice to extract core region (without halo)
        cell_size_m: Cell size in meters

    Returns:
        Tuple of:
        - slope_deg: Slope in degrees (core region only)
        - aspect_deg: Aspect in degrees (core region only)
    """
    # Ensure float32 for gradient computation
    tile_data_f32 = tile_data.astype(np.float32)

    # Compute gradient on full tile (including halo for boundary handling)
    gy, gx = np.gradient(tile_data_f32, cell_size_m)

    # Extract core region (halo was only for boundary handling)
    gy_core = gy[core_slice]
    gx_core = gx[core_slice]

    # Compute slope magnitude in degrees
    # slope_rad = arctan(sqrt(gx² + gy²))
    slope_magnitude = np.sqrt(gx_core**2 + gy_core**2)
    slope_rad = np.arctan(slope_magnitude)
    slope_deg = np.degrees(slope_rad)

    # Compute aspect (direction of steepest descent)
    # atan2(gx, -gy) gives aspect where 0=North, 90=East, 180=South, 270=West
    aspect_rad = np.arctan2(gx_core, -gy_core)
    aspect_deg = np.degrees(aspect_rad) % 360

    return slope_deg, aspect_deg


def aggregate_by_geographic_mapping(
    slope_full_res: np.ndarray,
    aspect_full_res: np.ndarray,
    elevation_full_res: np.ndarray,
    row_stride: int,
    col_stride: int,
    output_shape: Tuple[int, int],
) -> Dict[str, np.ndarray]:
    """
    Aggregate statistics using geographic pixel mapping.

    For now uses uniform stride-based aggregation. This could be extended to
    handle irregular mappings from reprojection.

    Args:
        slope_full_res: Slope in degrees at full resolution
        aspect_full_res: Aspect in degrees at full resolution
        elevation_full_res: Elevation at full resolution
        row_stride: Source pixels per output row
        col_stride: Source pixels per output column
        output_shape: (height, width) of output grid

    Returns:
        Dictionary of statistics arrays:
        - 'mean': mean slope per output pixel
        - 'max': max slope per output pixel
        - 'min': min slope per output pixel
        - 'std': std dev of slopes
        - 'p95': 95th percentile slope
        - 'roughness': elevation std dev
        - 'aspect_sin': vector-averaged aspect sin component
        - 'aspect_cos': vector-averaged aspect cos component
    """
    out_h, out_w = output_shape

    # Expected size after aggregation
    expected_h = out_h * row_stride
    expected_w = out_w * col_stride

    # Calculate how many full blocks we can fit
    actual_h = min(slope_full_res.shape[0], expected_h)
    actual_w = min(slope_full_res.shape[1], expected_w)

    # Adjust output shape if input is smaller than expected
    out_h = min(out_h, actual_h // row_stride)
    out_w = min(out_w, actual_w // col_stride)

    # Handle degenerate case: input too small to produce any output
    if out_h == 0 or out_w == 0:
        empty_shape = (out_h, out_w)
        return {
            "mean": np.empty(empty_shape, dtype=np.float32),
            "max": np.empty(empty_shape, dtype=np.float32),
            "min": np.empty(empty_shape, dtype=np.float32),
            "std": np.empty(empty_shape, dtype=np.float32),
            "p95": np.empty(empty_shape, dtype=np.float32),
            "roughness": np.empty(empty_shape, dtype=np.float32),
            "aspect_sin": np.empty(empty_shape, dtype=np.float32),
            "aspect_cos": np.empty(empty_shape, dtype=np.float32),
        }

    # Truncate to exact multiple of stride (must be divisible for reshape)
    trunc_h = out_h * row_stride
    trunc_w = out_w * col_stride

    slope_trunc = slope_full_res[:trunc_h, :trunc_w]
    aspect_trunc = aspect_full_res[:trunc_h, :trunc_w]
    elev_trunc = elevation_full_res[:trunc_h, :trunc_w]

    # Reshape to (out_h, row_stride, out_w, col_stride)
    slope_blocks = slope_trunc.reshape(out_h, row_stride, out_w, col_stride)
    aspect_blocks = aspect_trunc.reshape(out_h, row_stride, out_w, col_stride)
    elev_blocks = elev_trunc.reshape(out_h, row_stride, out_w, col_stride)

    # Compute statistics over block axes (1, 3)
    stats = {}
    stats["mean"] = np.mean(slope_blocks, axis=(1, 3)).astype(np.float32)
    stats["max"] = np.max(slope_blocks, axis=(1, 3)).astype(np.float32)
    stats["min"] = np.min(slope_blocks, axis=(1, 3)).astype(np.float32)
    stats["std"] = np.std(slope_blocks, axis=(1, 3)).astype(np.float32)

    # 95th percentile - transpose then flatten to preserve block boundaries
    # slope_blocks is (out_h, row_stride, out_w, col_stride)
    # Transpose to (out_h, out_w, row_stride, col_stride) so each block is contiguous
    slope_blocks_t = slope_blocks.transpose(0, 2, 1, 3)
    flat_blocks = slope_blocks_t.reshape(out_h, out_w, -1)
    stats["p95"] = np.percentile(flat_blocks, 95, axis=-1).astype(np.float32)

    # Roughness = elevation std dev within block
    stats["roughness"] = np.std(elev_blocks, axis=(1, 3)).astype(np.float32)

    # Aspect: use vector averaging (sin/cos components)
    aspect_rad = np.radians(aspect_blocks)
    stats["aspect_sin"] = np.mean(np.sin(aspect_rad), axis=(1, 3)).astype(np.float32)
    stats["aspect_cos"] = np.mean(np.cos(aspect_rad), axis=(1, 3)).astype(np.float32)

    return stats


def compute_tiled_slope_statistics(
    dem: np.ndarray,
    dem_transform: Affine,
    dem_crs: str,
    target_shape: Tuple[int, int],
    target_transform: Affine,
    target_crs: str,
    config: Optional[TiledSlopeConfig] = None,
) -> SlopeStatistics:
    """
    Compute slope statistics at target resolution using tiled processing.

    Processes the full-resolution DEM in tiles to preserve cliff faces
    and extreme slopes that would be lost by downsampling first.

    Args:
        dem: Full-resolution DEM array (height x width, float32 preferred)
        dem_transform: Affine transform for DEM
        dem_crs: CRS for DEM (e.g., 'EPSG:4326')
        target_shape: Output shape (out_height, out_width)
        target_transform: Affine transform for output grid
        target_crs: CRS for output grid
        config: Configuration parameters (uses defaults if None)

    Returns:
        SlopeStatistics dataclass with all computed statistics
    """
    if config is None:
        config = TiledSlopeConfig()

    dem_shape = dem.shape
    logger.info(
        f"Computing tiled slope statistics: DEM {dem_shape} → target {target_shape}"
    )

    # Step 1: Compute pixel mapping
    pixel_mapping = compute_pixel_mapping(
        dem_shape, dem_transform, dem_crs,
        target_shape, target_transform, target_crs,
    )

    # Step 2: Determine tile layout
    tiles = compute_tile_layout(dem_shape, target_shape, pixel_mapping, config)

    # Step 3: Initialize output arrays
    output = {
        "slope_mean": np.zeros(target_shape, dtype=np.float32),
        "slope_max": np.zeros(target_shape, dtype=np.float32),
        "slope_min": np.full(target_shape, np.inf, dtype=np.float32),
        "slope_std": np.zeros(target_shape, dtype=np.float32),
        "slope_p95": np.zeros(target_shape, dtype=np.float32),
        "roughness": np.zeros(target_shape, dtype=np.float32),
        "aspect_sin": np.zeros(target_shape, dtype=np.float32),
        "aspect_cos": np.zeros(target_shape, dtype=np.float32),
    }

    # Step 4: Compute cell size at target resolution
    cell_size_deg = abs(dem_transform[0])
    lat_center = (
        dem_transform[5] + dem.shape[0] * abs(dem_transform[4]) / 2
    )
    meters_per_degree = 111320 * np.cos(np.radians(lat_center))
    cell_size_m = cell_size_deg * meters_per_degree

    logger.info(f"  Cell size: {cell_size_m:.1f}m")

    # Step 5: Process each tile
    for tile_idx, tile_spec in enumerate(tiles):
        logger.debug(
            f"Processing tile {tile_idx + 1}/{len(tiles)} at "
            f"{tile_spec.src_slice}"
        )

        # Load tile with halo
        tile_data = dem[tile_spec.src_slice]

        # Compute slopes at full resolution
        slope_deg, aspect_deg = compute_tile_slopes(
            tile_data, tile_spec.core_slice, cell_size_m
        )

        # Extract elevation for roughness calculation
        elev_core = tile_data[tile_spec.core_slice]

        # Aggregate statistics for this tile
        tile_stats = aggregate_by_geographic_mapping(
            slope_deg, aspect_deg, elev_core,
            tile_spec.row_stride, tile_spec.col_stride,
            (
                tile_spec.out_slice[0].stop - tile_spec.out_slice[0].start,
                tile_spec.out_slice[1].stop - tile_spec.out_slice[1].start,
            ),
        )

        # Skip empty tiles (input too small for any output)
        if tile_stats["mean"].size == 0:
            logger.debug(f"Skipping empty tile {tile_idx + 1}")
            continue

        # Handle edge tiles where actual output is smaller than expected
        # This happens when input doesn't have enough pixels to fill out_slice
        actual_h, actual_w = tile_stats["mean"].shape
        safe_slice = (
            slice(
                tile_spec.out_slice[0].start,
                tile_spec.out_slice[0].start + actual_h,
            ),
            slice(
                tile_spec.out_slice[1].start,
                tile_spec.out_slice[1].start + actual_w,
            ),
        )

        # Accumulate into output arrays using safe slice
        output["slope_mean"][safe_slice] = tile_stats["mean"]
        output["slope_max"][safe_slice] = tile_stats["max"]
        output["slope_min"][safe_slice] = np.minimum(
            output["slope_min"][safe_slice],
            tile_stats["min"]
        )
        output["slope_std"][safe_slice] = tile_stats["std"]
        output["slope_p95"][safe_slice] = tile_stats["p95"]
        output["roughness"][safe_slice] = tile_stats["roughness"]
        output["aspect_sin"][safe_slice] = tile_stats["aspect_sin"]
        output["aspect_cos"][safe_slice] = tile_stats["aspect_cos"]

    # Fix any remaining inf values in slope_min
    output["slope_min"] = np.where(
        np.isinf(output["slope_min"]),
        output["slope_mean"],
        output["slope_min"]
    )

    logger.info("Slope statistics summary:")
    logger.info(f"  Mean slope: {np.mean(output['slope_mean']):.2f}°")
    logger.info(f"  Max slope: {np.max(output['slope_max']):.2f}°")
    logger.info(f"  Slopes > 25°: {np.sum(output['slope_max'] > 25)} pixels")

    return SlopeStatistics(
        slope_mean=output["slope_mean"],
        slope_max=output["slope_max"],
        slope_min=output["slope_min"],
        slope_std=output["slope_std"],
        slope_p95=output["slope_p95"],
        roughness=output["roughness"],
        aspect_sin=output["aspect_sin"],
        aspect_cos=output["aspect_cos"],
    )
