"""
Flow diagnostic visualization functions.

This module provides standardized visualizations for flow accumulation analysis,
including DEM views, ocean masks, water bodies, drainage networks, and validation summaries.

Example:
    from src.terrain.visualization.flow_diagnostics import create_flow_diagnostics

    # Generate all diagnostic plots
    create_flow_diagnostics(
        dem=dem,
        dem_conditioned=dem_conditioned,
        ocean_mask=ocean_mask,
        flow_dir=flow_dir,
        drainage_area=drainage_area,
        upstream_rainfall=upstream_rainfall,
        precip=precip,
        output_dir=Path("output/diagnostics"),
        lake_mask=lake_mask,
        lake_outlets=lake_outlets,
    )
"""

from pathlib import Path
from typing import Optional

import matplotlib
from src.terrain.flow_accumulation import compute_flow_direction, compute_drainage_area
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Colormap constants - all perceptually uniform, colorblind-friendly
FLOW_COLORMAPS = {
    "elevation": "viridis",
    "binary": "cividis",
    "fill": "magma",
    "direction": "plasma",
    "drainage": "viridis",
    "streams": "inferno",
    "precip": "cividis",
    "rainfall": "plasma",
}

# D8 flow direction codes to (dx, dy) vectors
D8_VECTORS = {
    1: (1, 0),      # East
    2: (1, -1),     # SE
    4: (0, -1),     # South
    8: (-1, -1),    # SW
    16: (-1, 0),    # West
    32: (-1, 1),    # NW
    64: (0, 1),     # North
    128: (1, 1),    # NE
}


def save_flow_plot(
    data: np.ndarray,
    title: str,
    output_path: Path,
    cmap: str,
    label: str = "",
    log_scale: bool = False,
    mask: Optional[np.ndarray] = None,
    overlay_data: Optional[np.ndarray] = None,
    overlay_cmap: Optional[str] = None,
    overlay_alpha: float = 0.7,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: tuple = (12, 10),
    dpi: int = 150,
    pixel_perfect: bool = True,
) -> Path:
    """
    Save a single flow diagnostic plot to file.

    Parameters
    ----------
    data : np.ndarray
        2D array of data to plot
    title : str
        Plot title
    output_path : Path
        Output file path
    cmap : str
        Matplotlib colormap name
    label : str, optional
        Colorbar label
    log_scale : bool, optional
        Apply log10 transformation
    mask : np.ndarray, optional
        Boolean mask for data (True = mask out)
    overlay_data : np.ndarray, optional
        Boolean array for overlay
    overlay_cmap : str, optional
        Colormap for overlay
    overlay_alpha : float, optional
        Alpha for overlay (default: 0.7)
    vmin, vmax : float, optional
        Colorbar limits
    figsize : tuple, optional
        Figure size in inches
    dpi : int, optional
        Output resolution
    pixel_perfect : bool, optional
        If True, save pixel-perfect raw image + annotated thumbnail (default: True)

    Returns
    -------
    Path
        Path to saved file (raw version if pixel_perfect=True)
    """
    output_path = Path(output_path)

    # Handle masking and prepare plot data
    plot_data = data.copy().astype(float)
    if mask is not None:
        plot_data = np.ma.masked_where(mask, plot_data)

    # Apply log scale
    display_label = label
    if log_scale and np.any(plot_data > 0):
        plot_data = np.log10(plot_data + 1)
        display_label = f"log10({label})" if label else "log10"

    # Get colormap
    cmap_obj = plt.get_cmap(cmap)

    # Normalize data for colormap
    if vmin is None:
        vmin = np.nanmin(plot_data) if not isinstance(plot_data, np.ma.MaskedArray) else np.nanmin(plot_data.compressed())
    if vmax is None:
        vmax = np.nanmax(plot_data) if not isinstance(plot_data, np.ma.MaskedArray) else np.nanmax(plot_data.compressed())

    if vmax > vmin:
        norm_data = (plot_data - vmin) / (vmax - vmin)
    else:
        norm_data = np.zeros_like(plot_data)

    # Handle masked arrays
    if isinstance(plot_data, np.ma.MaskedArray):
        norm_data = np.ma.filled(norm_data, 0.0)

    # Apply colormap to get RGB
    rgb_image = cmap_obj(norm_data)[:, :, :3]  # Drop alpha

    # Add overlay if provided (blend on top)
    if overlay_data is not None and overlay_cmap is not None:
        overlay_cmap_obj = plt.get_cmap(overlay_cmap)
        overlay_mask = overlay_data.astype(bool)
        overlay_rgb = overlay_cmap_obj(np.ones_like(overlay_data, dtype=float))[:, :, :3]
        for c in range(3):
            rgb_image[:, :, c] = np.where(
                overlay_mask,
                overlay_alpha * overlay_rgb[:, :, c] + (1 - overlay_alpha) * rgb_image[:, :, c],
                rgb_image[:, :, c]
            )

    if pixel_perfect:
        # 1. Save RAW pixel-perfect version
        plt.imsave(output_path, rgb_image)
        print(f"  Saved raw (pixel-perfect): {output_path.name} [{data.shape[0]}×{data.shape[1]} pixels]")

        # 2. Save ANNOTATED thumbnail
        thumb_path = output_path.with_stem(output_path.stem + "_annotated")
        fig, ax = plt.subplots(figsize=figsize)

        # Show the composite image
        ax.imshow(rgb_image, origin='upper')

        # Add colorbar using ScalarMappable
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=display_label, shrink=0.8)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

        plt.tight_layout()
        plt.savefig(thumb_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"  Saved annotated thumbnail: {thumb_path.name}")
    else:
        # Legacy mode: single matplotlib figure
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(plot_data, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label=display_label, shrink=0.8)

        if overlay_data is not None and overlay_cmap is not None:
            overlay_masked = np.ma.masked_where(~overlay_data, overlay_data.astype(float))
            ax.imshow(overlay_masked, cmap=overlay_cmap, alpha=overlay_alpha)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {output_path.name}")

    return output_path


def plot_dem(
    dem: np.ndarray,
    output_path: Path,
    title: Optional[str] = None,
) -> Path:
    """Plot original DEM elevation."""
    if title is None:
        title = f"Original DEM ({dem.shape[0]}x{dem.shape[1]})"
    return save_flow_plot(
        dem, title, output_path,
        cmap=FLOW_COLORMAPS["elevation"],
        label="Elevation (m)"
    )


def plot_ocean_mask(
    ocean_mask: np.ndarray,
    output_path: Path,
) -> Path:
    """Plot ocean mask with coverage percentage."""
    ocean_pct = 100 * np.sum(ocean_mask) / ocean_mask.size
    return save_flow_plot(
        ocean_mask.astype(float),
        f"Ocean Mask ({ocean_pct:.1f}% ocean)",
        output_path,
        cmap=FLOW_COLORMAPS["binary"],
        label="Ocean (1) / Land (0)"
    )


def plot_water_bodies(
    dem: np.ndarray,
    lake_mask: np.ndarray,
    output_path: Path,
    flow_dir: Optional[np.ndarray] = None,
    lake_outlets: Optional[np.ndarray] = None,
    lake_inlets: Optional[np.ndarray] = None,
) -> Path:
    """
    Plot water bodies with optional flow arrows for outlets/inlets.

    Shows DEM as background with lakes overlaid in blue.
    If flow_dir is provided, draws quiver arrows showing:
    - Red arrows: outlets (water leaving lakes)
    - Green arrows: inlets (water entering lakes)

    Saves pixel-perfect raw image + annotated thumbnail with arrows.
    """
    output_path = Path(output_path)
    num_lakes = len(np.unique(lake_mask[lake_mask > 0]))
    lake_pct = 100 * np.sum(lake_mask > 0) / lake_mask.size

    # Create pixel-perfect composite image
    cmap_obj = plt.get_cmap(FLOW_COLORMAPS["elevation"])

    # Normalize DEM for colormap
    dem_min, dem_max = np.nanmin(dem), np.nanmax(dem)
    if dem_max > dem_min:
        dem_norm = (dem - dem_min) / (dem_max - dem_min)
    else:
        dem_norm = np.zeros_like(dem)

    # Base image from DEM
    rgb_image = cmap_obj(dem_norm)[:, :, :3]

    # Overlay lakes in blue (same color as ocean in 3D render)
    lake_pixels = lake_mask > 0
    lake_color = np.array([25, 85, 125]) / 255.0  # Medium blue matching ocean
    lake_alpha = 0.7
    for c in range(3):
        rgb_image[:, :, c] = np.where(
            lake_pixels,
            lake_alpha * lake_color[c] + (1 - lake_alpha) * rgb_image[:, :, c],
            rgb_image[:, :, c]
        )

    # 1. Save RAW pixel-perfect version
    plt.imsave(output_path, rgb_image)
    print(f"  Saved raw (pixel-perfect): {output_path.name} [{dem.shape[0]}×{dem.shape[1]} pixels]")

    # 2. Save ANNOTATED thumbnail with arrows
    thumb_path = output_path.with_stem(output_path.stem + "_annotated")
    fig, ax = plt.subplots(figsize=(12, 10))

    # Show the composite image
    ax.imshow(rgb_image, origin='upper')

    # Draw outlet arrows (red, pointing outward)
    if flow_dir is not None and lake_outlets is not None and np.any(lake_outlets):
        outlet_rows, outlet_cols = np.where(lake_outlets)
        outlet_dirs = flow_dir[outlet_rows, outlet_cols]

        outlet_u = np.zeros(len(outlet_rows), dtype=float)
        outlet_v = np.zeros(len(outlet_rows), dtype=float)

        for i, d8_code in enumerate(outlet_dirs):
            if d8_code in D8_VECTORS:
                outlet_u[i], outlet_v[i] = D8_VECTORS[int(d8_code)]

        ax.quiver(
            outlet_cols, outlet_rows, outlet_u, outlet_v,
            color="red", scale=80, scale_units="xy", width=0.001,
            label=f"Outlets ({len(outlet_rows)})"
        )

    # Draw inlet arrows (green, pointing inward - reversed direction)
    if flow_dir is not None and lake_inlets is not None and np.any(lake_inlets):
        inlet_rows, inlet_cols = np.where(lake_inlets)
        inlet_dirs = flow_dir[inlet_rows, inlet_cols]

        inlet_u = np.zeros(len(inlet_rows), dtype=float)
        inlet_v = np.zeros(len(inlet_rows), dtype=float)

        for i, d8_code in enumerate(inlet_dirs):
            if d8_code in D8_VECTORS:
                u, v = D8_VECTORS[int(d8_code)]
                inlet_u[i], inlet_v[i] = -u, -v  # Reverse for inlets

        ax.quiver(
            inlet_cols, inlet_rows, inlet_u, inlet_v,
            color="green", scale=80, scale_units="xy", width=0.001,
            label=f"Inlets ({len(inlet_rows)})"
        )

    # Add colorbar using ScalarMappable
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap=FLOW_COLORMAPS["elevation"], norm=Normalize(vmin=dem_min, vmax=dem_max))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Elevation (m)", shrink=0.7)

    ax.set_title(
        f"Water Bodies ({num_lakes} lakes, {lake_pct:.2f}% of area)",
        fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    if (lake_outlets is not None and np.any(lake_outlets)) or \
       (lake_inlets is not None and np.any(lake_inlets)):
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(thumb_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved annotated thumbnail: {thumb_path.name}")

    return output_path


def plot_endorheic_basins(
    dem: np.ndarray,
    basin_mask: np.ndarray,
    output_path: Path,
) -> Path:
    """Plot endorheic basins overlaid on DEM. Pixel-perfect output."""
    output_path = Path(output_path)
    basin_coverage = 100 * np.sum(basin_mask) / basin_mask.size

    # Create pixel-perfect composite
    cmap_obj = plt.get_cmap(FLOW_COLORMAPS["elevation"])

    # Normalize DEM for colormap
    dem_min, dem_max = np.nanmin(dem), np.nanmax(dem)
    if dem_max > dem_min:
        dem_norm = (dem - dem_min) / (dem_max - dem_min)
    else:
        dem_norm = np.zeros_like(dem)

    # Base image from DEM
    rgb_image = cmap_obj(dem_norm)[:, :, :3]

    # Overlay basins in purple
    basin_color = np.array([0.6, 0.2, 0.8])  # Purple
    basin_alpha = 0.7
    for c in range(3):
        rgb_image[:, :, c] = np.where(
            basin_mask,
            basin_alpha * basin_color[c] + (1 - basin_alpha) * rgb_image[:, :, c],
            rgb_image[:, :, c]
        )

    # 1. Save RAW pixel-perfect version
    plt.imsave(output_path, rgb_image)
    print(f"  Saved raw (pixel-perfect): {output_path.name} [{dem.shape[0]}×{dem.shape[1]} pixels]")

    # 2. Save ANNOTATED thumbnail
    thumb_path = output_path.with_stem(output_path.stem + "_annotated")
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(rgb_image, origin='upper')

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap=FLOW_COLORMAPS["elevation"], norm=Normalize(vmin=dem_min, vmax=dem_max))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Elevation (m)", shrink=0.8)

    ax.set_title(
        f"Endorheic Basins ({basin_coverage:.2f}% of area)",
        fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    plt.tight_layout()
    plt.savefig(thumb_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved annotated thumbnail: {thumb_path.name}")

    return output_path


def plot_conditioned_dem(
    dem_conditioned: np.ndarray,
    output_path: Path,
) -> Path:
    """Plot conditioned (depression-filled) DEM."""
    return save_flow_plot(
        dem_conditioned,
        "Conditioned DEM (depression-filled)",
        output_path,
        cmap=FLOW_COLORMAPS["elevation"],
        label="Elevation (m)"
    )


def plot_breach_depth(
    dem: np.ndarray,
    breached_dem: np.ndarray,
    ocean_mask: np.ndarray,
    output_path: Path,
) -> Path:
    """Plot breach depth (how much elevation was lowered during breaching).

    Breach depth shows where and how much the DEM was lowered to create flow paths.
    Positive values indicate breaching occurred (elevation was lowered).
    """
    breach_depth = dem - breached_dem  # How much we lowered elevations
    breach_depth[ocean_mask] = 0

    # Ensure strictly non-negative (clip any numerical errors to zero)
    breach_depth = np.maximum(breach_depth, 0)

    # Compute statistics (only where breaching occurred)
    breached_mask = breach_depth > 0.01  # Breached cells (>1cm threshold)
    if np.any(breached_mask):
        breach_max = np.max(breach_depth[breached_mask])
        breach_mean = np.mean(breach_depth[breached_mask])
        breach_count = np.sum(breached_mask)
        title = f"Breach Depth (max={breach_max:.1f}m, mean={breach_mean:.2f}m, {breach_count:,} cells)"
    else:
        title = "Breach Depth (no breaching occurred)"

    return save_flow_plot(
        breach_depth,
        title,
        output_path,
        cmap=FLOW_COLORMAPS["fill"],  # Same colormap as fill depth
        label="m",
        log_scale=True
    )


def plot_fill_depth(
    dem: np.ndarray,
    dem_conditioned: np.ndarray,
    ocean_mask: np.ndarray,
    output_path: Path,
    breached_dem: Optional[np.ndarray] = None,
) -> Path:
    """Plot depression fill depth.

    Fill depth represents how much water was added to fill depressions.
    If breached_dem is provided, computes fill depth as (conditioned - breached),
    which correctly shows only the FILLING operation (not breaching).
    Otherwise falls back to (conditioned - original), which includes both operations.
    """
    if breached_dem is not None:
        # Correct: fill depth = how much we raised AFTER breaching
        fill_depth = dem_conditioned - breached_dem
    else:
        # Fallback: net change from original (includes both breaching and filling)
        fill_depth = dem_conditioned - dem

    fill_depth[ocean_mask] = 0

    # Ensure strictly non-negative (clip any numerical errors to zero)
    fill_depth = np.maximum(fill_depth, 0)

    fill_max = np.max(fill_depth)
    return save_flow_plot(
        fill_depth,
        f"Depression Fill Depth (max={fill_max:.1f}m)",
        output_path,
        cmap=FLOW_COLORMAPS["fill"],
        label="m",
        log_scale=True
    )


def plot_flow_direction(
    flow_dir: np.ndarray,
    output_path: Path,
) -> Path:
    """Plot D8 flow direction codes."""
    flow_display = flow_dir.astype(float)
    flow_display[flow_dir == 0] = np.nan
    return save_flow_plot(
        flow_display,
        "Flow Direction (D8 codes)",
        output_path,
        cmap=FLOW_COLORMAPS["direction"],
        label="D8 code (1-128)"
    )


def plot_drainage_area(
    drainage_area: np.ndarray,
    output_path: Path,
    lake_mask: Optional[np.ndarray] = None,
) -> Path:
    """Plot drainage area (log scale) with optional lake overlay."""
    lake_overlay = (lake_mask > 0) if lake_mask is not None else None
    return save_flow_plot(
        drainage_area,
        "Drainage Area (log scale)",
        output_path,
        cmap=FLOW_COLORMAPS["drainage"],
        label="cells",
        log_scale=True,
        overlay_data=lake_overlay,
        overlay_cmap="Blues"
    )


def plot_drainage_area_comparison(
    dem: np.ndarray,
    dem_conditioned: np.ndarray,
    ocean_mask: np.ndarray,
    output_path: Path,
) -> Path:
    """
    Create comparison plot of drainage area from raw vs conditioned DEM.

    Shows three panels:
    - Drainage area from raw DEM (left)
    - Drainage area from conditioned DEM (middle)
    - Absolute difference between them (right)

    This verifies that conditioning (breaching + filling) affects flow routing.
    """
    # Compute drainage areas
    flow_dir_raw = compute_flow_direction(dem, mask=ocean_mask)
    drainage_raw = compute_drainage_area(flow_dir_raw)

    flow_dir_cond = compute_flow_direction(dem_conditioned, mask=ocean_mask)
    drainage_cond = compute_drainage_area(flow_dir_cond)

    drainage_diff = np.abs(drainage_raw - drainage_cond)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Drainage Area Comparison: Raw vs Conditioned DEM", fontsize=14, fontweight="bold")

    # Plot 1: Drainage from raw DEM
    im1 = axes[0].imshow(np.log10(drainage_raw + 1), cmap=FLOW_COLORMAPS["drainage"], origin='upper')
    axes[0].set_title("Drainage Area from Raw DEM (log10)")
    axes[0].set_ylabel("Row")
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label("log10(cells)")

    # Plot 2: Drainage from conditioned DEM
    im2 = axes[1].imshow(np.log10(drainage_cond + 1), cmap=FLOW_COLORMAPS["drainage"], origin='upper')
    axes[1].set_title("Drainage Area from Conditioned DEM (log10)")
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label("log10(cells)")

    # Plot 3: Difference
    im3 = axes[2].imshow(drainage_diff, cmap="hot", origin='upper')
    axes[2].set_title("Drainage Area Difference (|raw - conditioned|)")
    axes[2].set_ylabel("Row")
    cbar3 = plt.colorbar(im3, ax=axes[2])
    cbar3.set_label("cells")

    # Add statistics text
    diff_count = np.sum(drainage_diff > 0)
    diff_pct = 100 * diff_count / dem.size
    fig.text(0.5, 0.02,
             f"Cells with difference: {diff_count:,} ({diff_pct:.2f}%) | "
             f"Max difference: {drainage_diff.max():.1f} cells | "
             f"Mean difference: {drainage_diff.mean():.1f} cells",
             ha='center', fontsize=10)

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def plot_stream_network(
    dem: np.ndarray,
    drainage_area: np.ndarray,
    output_path: Path,
    lake_mask: Optional[np.ndarray] = None,
    percentile: float = 95,
    base_dim: float = 0.4,
) -> Path:
    """
    Plot stream network extracted from drainage area. Pixel-perfect output.

    Streams are defined as cells with drainage area >= percentile threshold.

    Parameters
    ----------
    base_dim : float
        Dim factor for base DEM (0.0 = black, 1.0 = full brightness). Default 0.4.
    """
    output_path = Path(output_path)

    # Use top percentile drainage area threshold for stream visualization
    stream_threshold = np.percentile(drainage_area[drainage_area > 0], percentile)
    streams = drainage_area >= stream_threshold
    stream_count = np.sum(streams)

    # Create pixel-perfect composite
    cmap_obj = plt.get_cmap(FLOW_COLORMAPS["elevation"])

    # Normalize DEM for colormap
    dem_min, dem_max = np.nanmin(dem), np.nanmax(dem)
    if dem_max > dem_min:
        dem_norm = (dem - dem_min) / (dem_max - dem_min)
    else:
        dem_norm = np.zeros_like(dem)

    # Base image from DEM (dimmed to make streams visible)
    rgb_image = cmap_obj(dem_norm)[:, :, :3] * base_dim

    # Overlay streams (bright cyan on dimmed background)
    stream_color = np.array([0.0, 0.9, 1.0])  # Bright cyan for streams
    stream_alpha = 1.0  # Fully opaque streams
    for c in range(3):
        rgb_image[:, :, c] = np.where(
            streams,
            stream_color[c],  # Full brightness streams on dimmed background
            rgb_image[:, :, c]
        )

    # Overlay lakes in blue (same color as ocean in 3D render)
    if lake_mask is not None:
        lake_pixels = lake_mask > 0
        lake_color = np.array([25, 85, 125]) / 255.0  # Medium blue matching ocean
        lake_alpha = 0.5
        for c in range(3):
            rgb_image[:, :, c] = np.where(
                lake_pixels,
                lake_alpha * lake_color[c] + (1 - lake_alpha) * rgb_image[:, :, c],
                rgb_image[:, :, c]
            )

    # 1. Save RAW pixel-perfect version
    plt.imsave(output_path, rgb_image)
    print(f"  Saved raw (pixel-perfect): {output_path.name} [{dem.shape[0]}×{dem.shape[1]} pixels, {stream_count:,} stream cells]")

    # 2. Save ANNOTATED thumbnail
    thumb_path = output_path.with_stem(output_path.stem + "_annotated")
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(rgb_image, origin='upper')

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap=FLOW_COLORMAPS["elevation"], norm=Normalize(vmin=dem_min, vmax=dem_max))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Elevation (m)", shrink=0.8)

    ax.set_title(
        f"Stream Network with Lakes (top {100-percentile:.1f}%, {stream_count:,} cells)",
        fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    plt.tight_layout()
    plt.savefig(thumb_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved annotated thumbnail: {thumb_path.name}")

    return output_path


def plot_stream_overlay(
    base_data: np.ndarray,
    stream_threshold_data: np.ndarray,
    stream_color_data: np.ndarray,
    output_path: Path,
    base_cmap: str = "viridis",
    stream_cmap: str = "plasma",
    percentile: float = 95,
    stream_alpha: float = 1.0,
    base_label: str = "Base Metric",
    stream_label: str = "Stream Metric",
    title: str = "Stream Network Overlay",
    lake_mask: Optional[np.ndarray] = None,
    base_log_scale: bool = True,
    stream_log_scale: bool = True,
    variable_width: bool = False,
    min_width: int = 1,
    max_width: int = 4,
    base_dim: float = 0.5,
) -> Path:
    """
    Plot stream network colored by a metric, overlaid on a base map.

    Creates visualization with streams extracted from one metric and colored
    by another (or the same) metric, over a full-coverage base map.

    Parameters
    ----------
    base_data : np.ndarray
        Data for background map coloring (e.g., discharge_potential, elevation)
    stream_threshold_data : np.ndarray
        Data for extracting streams (usually drainage_area for percentile threshold)
    stream_color_data : np.ndarray
        Data for coloring stream pixels (e.g., discharge_potential, upstream_rainfall)
    output_path : Path
        Output file path
    base_cmap : str
        Matplotlib colormap for base map (default: viridis)
    stream_cmap : str
        Matplotlib colormap for streams (default: plasma)
    percentile : float
        Percentile threshold for stream extraction (default: 95 = top 5%)
    stream_alpha : float
        Stream transparency: 1.0 = opaque, 0.7 = semi-transparent (default: 1.0)
    base_label : str
        Colorbar label for base map
    stream_label : str
        Label describing stream metric (used in title)
    title : str
        Plot title
    lake_mask : np.ndarray, optional
        Lake mask for overlay (integer IDs or boolean)
    base_log_scale : bool
        Apply log10 to base data (default: True)
    stream_log_scale : bool
        Apply log10 to stream color data (default: True)
    variable_width : bool
        Scale stream line width by metric value (default: False)
    min_width : int
        Minimum stream width in pixels when variable_width=True (default: 1)
    max_width : int
        Maximum stream width in pixels when variable_width=True (default: 4)
    base_dim : float
        Dim factor for base map (0.0 = black, 1.0 = full brightness). Default 0.5.
        Dimming the base makes streams more visible.

    Returns
    -------
    Path
        Path to saved raw pixel-perfect file
    """
    output_path = Path(output_path)

    # Extract stream mask using percentile threshold
    valid_threshold = stream_threshold_data[stream_threshold_data > 0]
    if len(valid_threshold) == 0:
        print(f"  Warning: No valid stream data, skipping {output_path.name}")
        return output_path
    stream_threshold = np.percentile(valid_threshold, percentile)
    stream_mask = stream_threshold_data >= stream_threshold

    # Prepare base data (log scale if requested)
    base_plot = base_data.copy().astype(np.float32)
    if base_log_scale and np.any(base_plot > 0):
        base_plot = np.log10(base_plot + 1)

    # Prepare stream color data (log scale if requested)
    stream_plot = stream_color_data.copy().astype(np.float32)
    if stream_log_scale and np.any(stream_plot > 0):
        stream_plot = np.log10(stream_plot + 1)

    # Variable width: expand stream pixels based on metric value with smooth tapering
    if variable_width and np.any(stream_mask):
        from scipy.ndimage import maximum_filter, gaussian_filter, distance_transform_edt

        # Normalize stream values to determine radius (1 to max_width)
        stream_vals = stream_plot[stream_mask]
        val_min, val_max = np.min(stream_vals), np.max(stream_vals)

        if val_max > val_min:
            # Compute per-pixel target width based on metric value
            # Higher values = wider streams
            width_map = np.zeros_like(stream_plot)
            normalized_vals = (stream_plot[stream_mask] - val_min) / (val_max - val_min)
            width_map[stream_mask] = min_width + normalized_vals * (max_width - min_width)

            # Smooth the width map along streams for gradual tapering
            # Use distance-weighted diffusion constrained to stream neighborhood
            # First, expand slightly to allow smoothing across stream pixels
            expanded_for_smooth = maximum_filter(width_map, size=3)
            smooth_mask = expanded_for_smooth > 0

            # Gaussian smooth the width values (only affects stream region)
            smoothed_width = gaussian_filter(width_map, sigma=max_width)

            # Keep only within expanded stream region and normalize
            smoothed_width = np.where(smooth_mask, smoothed_width, 0)
            # Re-normalize to preserve width range
            smooth_stream_vals = smoothed_width[smooth_mask & (smoothed_width > 0)]
            if len(smooth_stream_vals) > 0:
                sw_min, sw_max = np.min(smooth_stream_vals), np.max(smooth_stream_vals)
                if sw_max > sw_min:
                    smoothed_width[smooth_mask] = min_width + (smoothed_width[smooth_mask] - sw_min) / (sw_max - sw_min) * (max_width - min_width)

            # Create expanded stream by applying variable dilation based on smoothed width
            expanded_values = np.zeros_like(stream_plot)
            expanded_values[stream_mask] = stream_plot[stream_mask]

            # Process by radius (from largest to smallest so big rivers dominate)
            for radius in range(max_width, min_width - 1, -1):
                # Find stream pixels that should have at least this radius
                radius_mask = stream_mask & (smoothed_width >= radius - 0.5)

                if np.any(radius_mask):
                    # Create a temporary grid with just these pixels' values
                    radius_values = np.zeros_like(stream_plot)
                    radius_values[radius_mask] = stream_plot[radius_mask]

                    # Expand using maximum filter
                    size = 2 * radius + 1
                    expanded_radius = maximum_filter(radius_values, size=size)

                    # Update expanded values (larger values win)
                    update_mask = expanded_radius > expanded_values
                    expanded_values[update_mask] = expanded_radius[update_mask]

            # Update stream_mask and stream_plot to use expanded versions
            stream_mask = expanded_values > 0
            stream_plot = np.where(stream_mask, expanded_values, stream_plot)

            print(f"    Variable width applied: {min_width}-{max_width}px with smooth tapering")

    # Get colormaps
    base_cmap_obj = plt.get_cmap(base_cmap)
    stream_cmap_obj = plt.get_cmap(stream_cmap)

    # Normalize base data to 0-1
    base_min, base_max = np.nanmin(base_plot), np.nanmax(base_plot)
    if base_max > base_min:
        base_norm = (base_plot - base_min) / (base_max - base_min)
    else:
        base_norm = np.zeros_like(base_plot)

    # Create base RGB image (dimmed to make streams more visible)
    base_rgb = base_cmap_obj(base_norm)[:, :, :3] * base_dim  # Drop alpha, apply dim

    # Normalize stream data to 0-1 (only for stream pixels)
    stream_values = stream_plot[stream_mask]
    if len(stream_values) > 0:
        stream_min, stream_max = np.nanmin(stream_values), np.nanmax(stream_values)
        if stream_max > stream_min:
            stream_norm = (stream_plot - stream_min) / (stream_max - stream_min)
        else:
            stream_norm = np.zeros_like(stream_plot)
    else:
        stream_norm = np.zeros_like(stream_plot)

    # Create stream RGBA (colored by metric, not binary)
    stream_rgba = stream_cmap_obj(stream_norm)

    # Composite: blend streams over base
    composite = base_rgb.copy()
    # Apply stream pixels with alpha blending
    for c in range(3):
        composite[:, :, c] = np.where(
            stream_mask,
            stream_alpha * stream_rgba[:, :, c] + (1 - stream_alpha) * base_rgb[:, :, c],
            base_rgb[:, :, c]
        )

    # Overlay lakes in blue if provided (same color as ocean in 3D render)
    if lake_mask is not None:
        lake_pixels = lake_mask > 0
        # Use same blue as ocean in 3D render (edge_color from core.py create_mesh)
        lake_color = np.array([25, 85, 125]) / 255.0  # Medium blue matching ocean
        lake_alpha = 0.7
        for c in range(3):
            composite[:, :, c] = np.where(
                lake_pixels,
                lake_alpha * lake_color[c] + (1 - lake_alpha) * composite[:, :, c],
                composite[:, :, c]
            )

    # 1. Save RAW pixel-perfect version
    plt.imsave(output_path, composite)
    stream_count = np.sum(stream_mask)
    print(f"  Saved raw (pixel-perfect): {output_path.name} [{base_data.shape[0]}×{base_data.shape[1]} pixels, {stream_count:,} stream cells]")

    # 2. Save ANNOTATED thumbnail
    thumb_path = output_path.with_stem(output_path.stem + "_annotated")
    fig, ax = plt.subplots(figsize=(10, 8))

    # Show composite
    ax.imshow(composite, origin='upper')

    # Add colorbars for both base and stream
    # Create dummy ScalarMappables for colorbars
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    base_label_full = f"log10({base_label})" if base_log_scale else base_label
    stream_label_full = f"log10({stream_label})" if stream_log_scale else stream_label

    # Base colorbar (left side)
    sm_base = ScalarMappable(cmap=base_cmap, norm=Normalize(vmin=base_min, vmax=base_max))
    cbar_base = plt.colorbar(sm_base, ax=ax, location='left', shrink=0.6, pad=0.02)
    cbar_base.set_label(base_label_full, fontsize=9)

    # Stream colorbar (right side)
    if len(stream_values) > 0:
        sm_stream = ScalarMappable(cmap=stream_cmap, norm=Normalize(vmin=stream_min, vmax=stream_max))
        cbar_stream = plt.colorbar(sm_stream, ax=ax, location='right', shrink=0.6, pad=0.02)
        cbar_stream.set_label(f"Streams: {stream_label_full}", fontsize=9)

    alpha_desc = "opaque" if stream_alpha >= 1.0 else f"α={stream_alpha:.1f}"
    ax.set_title(
        f"{title}\n(top {100-percentile:.0f}% streams, {stream_count:,} cells, {alpha_desc})",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("Column", fontsize=9)
    ax.set_ylabel("Row", fontsize=9)

    plt.tight_layout()
    plt.savefig(thumb_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved annotated thumbnail: {thumb_path.name}")

    return output_path


def plot_precipitation(
    precip: np.ndarray,
    output_path: Path,
    is_real: bool = False,
) -> Path:
    """Plot precipitation data."""
    if is_real:
        title = "WorldClim Annual Precipitation"
        label = "mm/year"
    else:
        title = "Synthetic Precipitation (elevation-based)"
        label = "mm/year (synthetic)"
    return save_flow_plot(
        precip, title, output_path,
        cmap=FLOW_COLORMAPS["precip"],
        label=label
    )


def plot_upstream_rainfall(
    upstream_rainfall: np.ndarray,
    output_path: Path,
    lake_mask: Optional[np.ndarray] = None,
) -> Path:
    """Plot upstream accumulated rainfall (log scale)."""
    lake_overlay = (lake_mask > 0) if lake_mask is not None else None
    return save_flow_plot(
        upstream_rainfall,
        "Upstream Rainfall",
        output_path,
        cmap=FLOW_COLORMAPS["rainfall"],
        label="mm·m²",
        log_scale=True,
        overlay_data=lake_overlay,
        overlay_cmap="Blues"
    )


def plot_discharge_potential(
    drainage_area: np.ndarray,
    upstream_rainfall: np.ndarray,
    output_path: Path,
    lake_mask: Optional[np.ndarray] = None,
    log_scale: bool = True,
) -> Path:
    """
    Plot discharge potential (drainage × rainfall-weighted).

    Combines drainage area with upstream rainfall to show where
    the biggest flows occur.
    """
    mean_rainfall = np.mean(upstream_rainfall[upstream_rainfall > 0]) \
        if np.any(upstream_rainfall > 0) else 1.0
    discharge_potential = drainage_area.astype(np.float32) * (upstream_rainfall / mean_rainfall)

    scale_label = "Log Scale" if log_scale else "Linear Scale"
    lake_overlay = (lake_mask > 0) if lake_mask is not None else None

    return save_flow_plot(
        discharge_potential,
        f"Discharge Potential - {scale_label} (drainage × rainfall)",
        output_path,
        cmap=FLOW_COLORMAPS["rainfall"],
        label="Discharge Index",
        log_scale=log_scale,
        overlay_data=lake_overlay,
        overlay_cmap="Blues"
    )


def plot_validation_summary(
    output_path: Path,
    dem_shape: tuple,
    ocean_cells: int,
    max_drainage: float,
    max_upstream: float,
    num_lakes: int = 0,
    num_outlets: int = 0,
    num_basins: int = 0,
    basin_cells: int = 0,
    cycles: int = 0,
    sample_size: int = 1000,
    mass_balance: float = 100.0,
    drainage_violations: int = 0,
    is_real_precip: bool = False,
) -> Path:
    """
    Plot validation summary as a text table.

    Parameters are all the statistics to display in the summary.
    """
    status = "PASS" if cycles == 0 and mass_balance > 95 and drainage_violations == 0 else "FAIL"
    if cycles == 0 and mass_balance > 95 and drainage_violations > 0:
        status = "WARN"

    violation_text = f"{drainage_violations:>8,}"
    if drainage_violations > 0:
        violation_text = f"{drainage_violations:>8,} !"

    total_cells = dem_shape[0] * dem_shape[1]
    ocean_pct = 100 * ocean_cells / total_cells

    summary_text = f"""
    ========================================================
             FLOW VALIDATION SUMMARY
    ========================================================

      Status:        {status:>8}

      Cycles:        {cycles:>8} / {sample_size:<8}
      Mass Balance:  {mass_balance:>8.1f}%
      Drain. Viol.:  {violation_text}

    --------------------------------------------------------
      Water Bodies & Basins
    --------------------------------------------------------

      Lakes:         {num_lakes:>8}
      Outlets:       {num_outlets:>8}
      Endor. Basins: {num_basins:>8}  ({basin_cells:,} cells)

    --------------------------------------------------------
      DEM Statistics
    --------------------------------------------------------

      Size:          {dem_shape[0]:>4} x {dem_shape[1]:<4} pixels
      Total Cells:   {total_cells:>12,}
      Ocean Cells:   {ocean_cells:>12,} ({ocean_pct:.1f}%)

      Max Drainage:  {max_drainage:>12,.0f} cells
      Max Upstream:  {max_upstream:>12,.0f} mm·m²

      Precip Type:   {'Real (WorldClim)' if is_real_precip else 'Synthetic'}

    ========================================================
    """

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")

    color = "green" if status == "PASS" else ("orange" if status == "WARN" else "red")
    ax.text(
        0.5, 0.5, summary_text,
        transform=ax.transAxes,
        fontsize=10,
        fontfamily="monospace",
        verticalalignment="center",
        horizontalalignment="center",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor=color, linewidth=3)
    )

    plt.tight_layout()
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")
    return output_path


def create_flow_diagnostics(
    dem: np.ndarray,
    dem_conditioned: np.ndarray,
    ocean_mask: np.ndarray,
    flow_dir: np.ndarray,
    drainage_area: np.ndarray,
    upstream_rainfall: np.ndarray,
    precip: np.ndarray,
    output_dir: Path,
    lake_mask: Optional[np.ndarray] = None,
    lake_outlets: Optional[np.ndarray] = None,
    lake_inlets: Optional[np.ndarray] = None,
    basin_mask: Optional[np.ndarray] = None,
    breached_dem: Optional[np.ndarray] = None,
    num_basins: int = 0,
    is_real_precip: bool = False,
    cycles: int = 0,
    sample_size: int = 1000,
    mass_balance: float = 100.0,
    drainage_violations: int = 0,
) -> Path:
    """
    Generate all flow diagnostic visualizations.

    Creates a comprehensive set of plots for flow accumulation analysis:
    1. Original DEM
    2. Ocean mask
    3. Water bodies (if lake_mask provided)
    3b. Endorheic basins (if basin_mask provided)
    4. Conditioned DEM
    5. Fill depth
    6. Flow direction
    7. Drainage area
    8. Stream network with lakes
    9. Precipitation
    10. Upstream rainfall
    10b. Discharge potential (log scale)
    10c. Discharge potential (linear scale)
    11. Validation summary

    Parameters
    ----------
    dem : np.ndarray
        Original DEM
    dem_conditioned : np.ndarray
        Depression-filled DEM
    ocean_mask : np.ndarray
        Boolean ocean mask
    flow_dir : np.ndarray
        D8 flow direction codes
    drainage_area : np.ndarray
        Accumulated drainage area (cells)
    upstream_rainfall : np.ndarray
        Accumulated upstream rainfall
    precip : np.ndarray
        Precipitation data
    output_dir : Path
        Directory for output images
    lake_mask : np.ndarray, optional
        Lake mask (integer IDs)
    lake_outlets : np.ndarray, optional
        Boolean mask of lake outlet cells
    lake_inlets : np.ndarray, optional
        Boolean mask of lake inlet cells
    basin_mask : np.ndarray, optional
        Boolean mask of endorheic basins
    num_basins : int, optional
        Number of endorheic basins
    is_real_precip : bool, optional
        Whether precipitation is real (WorldClim) or synthetic
    cycles : int, optional
        Number of flow cycles detected (validation)
    sample_size : int, optional
        Sample size for cycle detection
    mass_balance : float, optional
        Mass balance percentage
    drainage_violations : int, optional
        Number of drainage violations

    Returns
    -------
    Path
        Output directory path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating flow diagnostics in: {output_dir}")

    # 1. Original DEM
    plot_dem(dem, output_dir / "01_dem_original.png")

    # 2. Ocean Mask
    plot_ocean_mask(ocean_mask, output_dir / "02_ocean_mask.png")

    # 3. Water Bodies (if available)
    if lake_mask is not None and np.any(lake_mask > 0):
        plot_water_bodies(
            dem, lake_mask, output_dir / "03_water_bodies.png",
            flow_dir=flow_dir,
            lake_outlets=lake_outlets,
            lake_inlets=lake_inlets
        )

    # 3b. Endorheic Basins (if available)
    if basin_mask is not None and np.any(basin_mask):
        plot_endorheic_basins(dem, basin_mask, output_dir / "03b_endorheic_basins.png")

    # 4. Conditioned DEM
    plot_conditioned_dem(dem_conditioned, output_dir / "04_dem_conditioned.png")

    # 4b. Breach Depth (if breached_dem available)
    if breached_dem is not None:
        plot_breach_depth(dem, breached_dem, ocean_mask, output_dir / "04b_breach_depth.png")

    # 5. Fill Depth
    plot_fill_depth(dem, dem_conditioned, ocean_mask, output_dir / "05_fill_depth.png", breached_dem)

    # 6. Flow Direction
    plot_flow_direction(flow_dir, output_dir / "06_flow_direction.png")

    # 7. Drainage Area
    plot_drainage_area(drainage_area, output_dir / "07_drainage_area.png", lake_mask)

    # 7b. Drainage Area Comparison (Raw vs Conditioned)
    plot_drainage_area_comparison(dem, dem_conditioned, ocean_mask, output_dir / "07b_drainage_area_comparison.png")

    # 8. Stream Network (multiple percentiles)
    # 8a. Default (95th percentile = top 5%)
    plot_stream_network(
        dem, drainage_area, output_dir / "08a_stream_network_95pct.png",
        lake_mask=lake_mask, percentile=95
    )
    # 8b. 97.5th percentile (top 2.5%)
    plot_stream_network(
        dem, drainage_area, output_dir / "08b_stream_network_97.5pct.png",
        lake_mask=lake_mask, percentile=97.5
    )
    # 8c. 99th percentile (top 1%)
    plot_stream_network(
        dem, drainage_area, output_dir / "08c_stream_network_99pct.png",
        lake_mask=lake_mask, percentile=99
    )

    # 9. Precipitation
    plot_precipitation(precip, output_dir / "09_precipitation.png", is_real=is_real_precip)

    # 10. Upstream Rainfall
    plot_upstream_rainfall(
        upstream_rainfall, output_dir / "10_upstream_rainfall.png",
        lake_mask=lake_mask
    )

    # 10b. Discharge Potential (log scale)
    plot_discharge_potential(
        drainage_area, upstream_rainfall,
        output_dir / "10b_discharge_potential_log.png",
        lake_mask=lake_mask, log_scale=True
    )

    # 10c. Discharge Potential (linear scale)
    plot_discharge_potential(
        drainage_area, upstream_rainfall,
        output_dir / "10c_discharge_potential_linear.png",
        lake_mask=lake_mask, log_scale=False
    )

    # 11. Validation Summary
    num_lakes = len(np.unique(lake_mask[lake_mask > 0])) if lake_mask is not None else 0
    num_outlets = int(np.sum(lake_outlets)) if lake_outlets is not None else 0
    basin_cells = int(np.sum(basin_mask)) if basin_mask is not None else 0

    plot_validation_summary(
        output_dir / "11_validation_summary.png",
        dem_shape=dem.shape,
        ocean_cells=int(np.sum(ocean_mask)),
        max_drainage=float(np.max(drainage_area)),
        max_upstream=float(np.max(upstream_rainfall)),
        num_lakes=num_lakes,
        num_outlets=num_outlets,
        num_basins=num_basins,
        basin_cells=basin_cells,
        cycles=cycles,
        sample_size=sample_size,
        mass_balance=mass_balance,
        drainage_violations=drainage_violations,
        is_real_precip=is_real_precip,
    )

    print(f"  Generated {14 if lake_mask is not None else 12} diagnostic images")
    return output_dir


def vectorize_stream_network(stream_mask: np.ndarray, simplify_tolerance: float = 2.0) -> list:
    """
    Convert stream raster to vector polylines using topology-aware path extraction.

    Uses skan library for proper skeleton-to-vector conversion that preserves
    stream network topology and handles branch points correctly.

    Args:
        stream_mask: Boolean array of stream pixels
        simplify_tolerance: Douglas-Peucker tolerance for simplification (pixels).
                          Set to 0 to disable simplification.

    Returns:
        List of polylines, each as (N, 2) array of [y, x] coordinates.
        Each polyline represents a stream segment between junctions/endpoints.

    Example:
        >>> stream_mask = stream_raster > 0
        >>> polylines = vectorize_stream_network(stream_mask, simplify_tolerance=2.0)
        >>> print(f"Extracted {len(polylines)} stream segments")
    """
    try:
        from skan import Skeleton
        from skimage.morphology import skeletonize
    except ImportError:
        print("WARNING: skan or scikit-image not available for stream vectorization")
        print("  Install with: pip install skan scikit-image")
        return []

    if not np.any(stream_mask):
        return []

    # Skeletonize to 1-pixel centerlines
    skeleton = skeletonize(stream_mask)

    # Use skan to extract paths (handles topology correctly)
    skel_obj = Skeleton(skeleton)

    # Get path coordinates using skan's coordinate extraction
    # skel_obj.coordinates is a (N, ndim) array of all skeleton pixel coords
    # skel_obj.paths gives us the path structure
    polylines = []

    for path_idx in range(skel_obj.n_paths):
        # Get indices of pixels in this path
        path_indices = skel_obj.path(path_idx)

        if len(path_indices) < 3:  # Skip very short segments
            continue

        # Extract (y, x) coordinates for this path
        coords = skel_obj.coordinates[path_indices]  # Shape: (N, 2) where cols are [y, x]

        # Simplify using Douglas-Peucker if requested
        if simplify_tolerance > 0:
            from skimage.measure import approximate_polygon
            coords_simplified = approximate_polygon(coords.astype(float), tolerance=simplify_tolerance)
            if len(coords_simplified) >= 2:  # Need at least 2 points for a line
                polylines.append(coords_simplified)
        else:
            polylines.append(coords.astype(float))

    return polylines


def polyline_to_variable_width_polygon(polyline: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """
    Convert a polyline to a variable-width polygon.

    Creates a filled polygon outline by offsetting perpendicular to the polyline
    at each point based on the width at that point.

    Args:
        polyline: (N, 2) array of [y, x] coordinates
        widths: (N,) array of half-widths at each point

    Returns:
        (M, 2) array of polygon vertices [y, x]
    """
    if len(polyline) < 2:
        return np.array([])

    # Calculate tangent vectors along the polyline
    tangents = np.zeros_like(polyline)
    tangents[:-1] = polyline[1:] - polyline[:-1]
    tangents[-1] = tangents[-2]  # Use previous tangent for last point

    # Normalize tangents
    lengths = np.sqrt(np.sum(tangents**2, axis=1))
    lengths[lengths == 0] = 1.0  # Avoid division by zero
    tangents = tangents / lengths[:, np.newaxis]

    # Calculate perpendicular vectors (rotate tangent 90°)
    # For 2D: perp = [-dy, dx]
    perpendiculars = np.column_stack([-tangents[:, 1], tangents[:, 0]])

    # Create left and right edges
    left_edge = polyline + perpendiculars * widths[:, np.newaxis]
    right_edge = polyline - perpendiculars * widths[:, np.newaxis]

    # Combine to form polygon: left edge forward + right edge backward
    polygon = np.vstack([left_edge, right_edge[::-1]])

    return polygon


def plot_vectorized_streams(
    stream_raster: np.ndarray,
    base_data: np.ndarray,
    output_path: Path,
    base_cmap: str = "terrain",
    base_label: str = "Elevation",
    title: str = "Vectorized Stream Network",
    simplify_tolerance: float = 2.0,
    base_log_scale: bool = False,
    variable_width: bool = True,
    max_width: float = 3.0,
) -> Path:
    """
    Plot vectorized stream network overlaid on base map (TEMPORARY/EXPERIMENTAL).

    Creates diagnostic showing:
    1. Left: Original stream raster
    2. Right: Vectorized polylines overlaid on base map

    Args:
        stream_raster: Stream metric values (0 = no stream)
        base_data: Base map data (e.g., elevation, drainage)
        output_path: Where to save plot
        base_cmap: Matplotlib colormap for base map
        base_label: Label for base map colorbar
        title: Plot title
        simplify_tolerance: Douglas-Peucker tolerance (pixels)
        base_log_scale: Apply log scale to base data

    Returns:
        Path to saved plot
    """
    output_path = Path(output_path)

    # Create stream mask
    stream_mask = stream_raster > 0
    num_stream_pixels = np.sum(stream_mask)

    if num_stream_pixels == 0:
        print(f"  Warning: No stream pixels, skipping {output_path.name}")
        return output_path

    # Extract polylines using topology-aware vectorization
    print(f"  Vectorizing {num_stream_pixels:,} stream pixels...")
    polylines = vectorize_stream_network(stream_mask, simplify_tolerance=simplify_tolerance)

    if len(polylines) == 0:
        print(f"  Warning: No polylines extracted, skipping {output_path.name}")
        return output_path

    total_points = sum(len(p) for p in polylines)
    print(f"  Extracted {len(polylines):,} stream segments ({total_points:,} total points)")

    # Render polylines with variable width to raster (1 data pixel = 1 image pixel)
    from skimage.draw import polygon as draw_polygon
    polyline_raster = np.zeros_like(stream_mask, dtype=np.uint8)

    print(f"  Creating variable-width polygons (max width: {max_width:.1f}px)...")
    skipped = 0
    drawn = 0

    for idx, polyline in enumerate(polylines):
        # Ensure polyline is 2D (N, 2)
        if polyline.ndim != 2 or polyline.shape[1] != 2:
            skipped += 1
            continue

        if len(polyline) < 2:
            skipped += 1
            continue

        # Get width values along this polyline from stream_raster
        # Sample stream metric values at polyline points
        coords_int = np.round(polyline).astype(int)
        coords_int[:, 0] = np.clip(coords_int[:, 0], 0, stream_raster.shape[0] - 1)
        coords_int[:, 1] = np.clip(coords_int[:, 1], 0, stream_raster.shape[1] - 1)

        metric_values = stream_raster[coords_int[:, 0], coords_int[:, 1]]

        if not variable_width or np.all(metric_values == 0):
            # No variable width or no metric data - use constant width
            widths = np.ones(len(polyline)) * 0.5  # 1px width (0.5 radius)
        else:
            # Normalize metric values to [0, max_width/2] for half-widths
            min_val, max_val = metric_values.min(), metric_values.max()
            if max_val > min_val:
                widths = (metric_values - min_val) / (max_val - min_val) * (max_width / 2)
                widths = np.clip(widths, 0.5, max_width / 2)  # At least 0.5px radius
            else:
                widths = np.ones(len(polyline)) * 0.5

        # Convert polyline to variable-width polygon
        polygon_coords = polyline_to_variable_width_polygon(polyline, widths)

        if len(polygon_coords) == 0:
            skipped += 1
            continue

        # Rasterize polygon
        polygon_int = np.round(polygon_coords).astype(int)
        polygon_int[:, 0] = np.clip(polygon_int[:, 0], 0, stream_mask.shape[0] - 1)
        polygon_int[:, 1] = np.clip(polygon_int[:, 1], 0, stream_mask.shape[1] - 1)

        rr, cc = draw_polygon(polygon_int[:, 0], polygon_int[:, 1], shape=stream_mask.shape)
        polyline_raster[rr, cc] = 255

        drawn += 1

    # Save as simple PNG: 1 data pixel = 1 image pixel
    from PIL import Image
    img = Image.fromarray(polyline_raster, mode='L')
    img.save(output_path)

    print(f"  ✓ Rasterized {drawn:,}/{len(polylines):,} polylines ({skipped:,} skipped) → {np.sum(polyline_raster > 0):,} pixels")
    return output_path
