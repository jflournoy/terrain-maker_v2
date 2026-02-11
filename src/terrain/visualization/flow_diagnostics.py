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
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: tuple = (12, 10),
    dpi: int = 150,
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
    vmin, vmax : float, optional
        Colorbar limits
    figsize : tuple, optional
        Figure size in inches
    dpi : int, optional
        Output resolution

    Returns
    -------
    Path
        Path to saved file
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Handle masking
    plot_data = data.copy().astype(float)
    if mask is not None:
        plot_data = np.ma.masked_where(mask, plot_data)

    # Apply log scale
    if log_scale and np.any(plot_data > 0):
        plot_data = np.log10(plot_data + 1)
        label = f"log10({label})" if label else "log10"

    im = ax.imshow(plot_data, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label=label, shrink=0.8)

    # Add overlay if provided
    if overlay_data is not None and overlay_cmap is not None:
        overlay_masked = np.ma.masked_where(~overlay_data, overlay_data.astype(float))
        ax.imshow(overlay_masked, cmap=overlay_cmap, alpha=0.7)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    plt.tight_layout()
    output_path = Path(output_path)
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
    """
    num_lakes = len(np.unique(lake_mask[lake_mask > 0]))
    lake_pct = 100 * np.sum(lake_mask > 0) / lake_mask.size

    fig, ax = plt.subplots(figsize=(12, 10))

    # Show elevation as background
    im_dem = ax.imshow(dem, cmap=FLOW_COLORMAPS["elevation"], alpha=0.6)

    # Overlay lakes in blue
    lake_display = np.ma.masked_where(lake_mask == 0, lake_mask.astype(float))
    ax.imshow(lake_display, cmap="Blues", alpha=0.7, vmin=0, vmax=max(num_lakes, 1))

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

    plt.colorbar(im_dem, ax=ax, label="Elevation (m)", shrink=0.7)
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
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")
    return output_path


def plot_endorheic_basins(
    dem: np.ndarray,
    basin_mask: np.ndarray,
    output_path: Path,
) -> Path:
    """Plot endorheic basins overlaid on DEM."""
    basin_coverage = 100 * np.sum(basin_mask) / basin_mask.size

    fig, ax = plt.subplots(figsize=(12, 10))

    # Show elevation as background
    im_dem = ax.imshow(dem, cmap=FLOW_COLORMAPS["elevation"], alpha=0.6)

    # Overlay basins in purple
    basin_display = np.ma.masked_where(~basin_mask, basin_mask.astype(float))
    ax.imshow(basin_display, cmap="Purples", alpha=0.7, vmin=0, vmax=1)

    plt.colorbar(im_dem, ax=ax, label="Elevation (m)", shrink=0.8)
    ax.set_title(
        f"Endorheic Basins ({basin_coverage:.2f}% of area)",
        fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    plt.tight_layout()
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")
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


def plot_stream_network(
    dem: np.ndarray,
    drainage_area: np.ndarray,
    output_path: Path,
    lake_mask: Optional[np.ndarray] = None,
    percentile: float = 95,
) -> Path:
    """
    Plot stream network extracted from drainage area.

    Streams are defined as cells with drainage area >= percentile threshold.
    """
    # Use top percentile drainage area threshold for stream visualization
    stream_threshold = np.percentile(drainage_area[drainage_area > 0], percentile)
    streams = drainage_area >= stream_threshold

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(dem, cmap=FLOW_COLORMAPS["elevation"], alpha=0.6)

    # Overlay streams
    stream_overlay = np.ma.masked_where(~streams, streams.astype(float))
    ax.imshow(stream_overlay, cmap=FLOW_COLORMAPS["streams"], alpha=0.9)

    # Overlay lakes in blue
    if lake_mask is not None:
        lake_overlay = np.ma.masked_where(
            lake_mask == 0, np.ones_like(lake_mask, dtype=float)
        )
        ax.imshow(lake_overlay, cmap="Blues", alpha=0.5)

    stream_count = np.sum(streams)
    plt.colorbar(im, ax=ax, label="Elevation (m)", shrink=0.8)
    ax.set_title(
        f"Stream Network with Lakes (top {100-percentile:.0f}%, {stream_count:,} cells)",
        fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    plt.tight_layout()
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")
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

    # 8. Stream Network
    plot_stream_network(
        dem, drainage_area, output_dir / "08_stream_network_with_lakes.png",
        lake_mask=lake_mask
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

    print(f"  Generated {13 if lake_mask is not None else 11} diagnostic images")
    return output_dir
