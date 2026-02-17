"""
Diagnostic plotting utilities for terrain processing.

Provides visualization functions to understand and debug terrain transforms,
particularly wavelet denoising, slope-adaptive smoothing, and other processing steps.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


def plot_wavelet_diagnostics(
    original: np.ndarray,
    denoised: np.ndarray,
    output_path: Path,
    title_prefix: str = "Wavelet Denoising",
    nodata_value: float = np.nan,
    profile_row: Optional[int] = None,
    cmap: str = "terrain",
) -> Path:
    """
    Generate diagnostic plots showing wavelet denoising effects.

    Creates a multi-panel figure showing:
    - Original DEM
    - Denoised DEM
    - Difference (noise removed)
    - Cross-section profile comparison

    Args:
        original: Original DEM before denoising
        denoised: DEM after wavelet denoising
        output_path: Path to save the diagnostic plot
        title_prefix: Prefix for plot titles
        nodata_value: Value treated as no data
        profile_row: Row index for cross-section (default: middle row)
        cmap: Colormap for elevation visualization

    Returns:
        Path to saved diagnostic plot
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    # Handle nodata
    if np.isnan(nodata_value):
        orig_masked = np.ma.masked_invalid(original)
        deno_masked = np.ma.masked_invalid(denoised)
    else:
        orig_masked = np.ma.masked_equal(original, nodata_value)
        deno_masked = np.ma.masked_equal(denoised, nodata_value)

    # Compute difference (noise removed)
    difference = original - denoised
    if np.isnan(nodata_value):
        diff_masked = np.ma.masked_invalid(difference)
    else:
        diff_masked = np.ma.masked_where(
            (original == nodata_value) | (denoised == nodata_value), difference
        )

    # Statistics
    valid_diff = difference[~np.isnan(difference)] if np.isnan(nodata_value) else difference[
        (original != nodata_value) & (denoised != nodata_value)
    ]
    diff_std = np.std(valid_diff)
    diff_mean = np.mean(valid_diff)
    diff_max = np.max(np.abs(valid_diff))

    # Profile row (default to middle)
    if profile_row is None:
        profile_row = original.shape[0] // 2

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"{title_prefix} Diagnostics", fontsize=14, fontweight="bold")

    # Shared elevation range
    vmin = min(np.nanmin(original), np.nanmin(denoised))
    vmax = max(np.nanmax(original), np.nanmax(denoised))

    # 1. Original DEM
    ax1 = axes[0, 0]
    im1 = ax1.imshow(orig_masked, cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.axhline(y=profile_row, color="red", linestyle="--", alpha=0.7, linewidth=1)
    ax1.set_title("Original DEM")
    ax1.set_xlabel("Column")
    ax1.set_ylabel("Row")
    plt.colorbar(im1, ax=ax1, label="Elevation (m)")

    # 2. Denoised DEM
    ax2 = axes[0, 1]
    im2 = ax2.imshow(deno_masked, cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.axhline(y=profile_row, color="red", linestyle="--", alpha=0.7, linewidth=1)
    ax2.set_title("Denoised DEM")
    ax2.set_xlabel("Column")
    ax2.set_ylabel("Row")
    plt.colorbar(im2, ax=ax2, label="Elevation (m)")

    # 3. Difference (noise removed) - diverging colormap centered at 0
    ax3 = axes[1, 0]
    # Use symmetric range centered at 0
    diff_range = max(abs(np.nanmin(valid_diff)), abs(np.nanmax(valid_diff)))
    if diff_range > 0:
        norm = TwoSlopeNorm(vmin=-diff_range, vcenter=0, vmax=diff_range)
    else:
        norm = None
    im3 = ax3.imshow(diff_masked, cmap="RdBu_r", norm=norm)
    ax3.set_title(f"Noise Removed (σ={diff_std:.3f}m, max={diff_max:.3f}m)")
    ax3.set_xlabel("Column")
    ax3.set_ylabel("Row")
    plt.colorbar(im3, ax=ax3, label="Δ Elevation (m)")

    # 4. Cross-section profile comparison
    ax4 = axes[1, 1]
    orig_profile = original[profile_row, :]
    deno_profile = denoised[profile_row, :]

    # Mask invalid values for plotting
    if np.isnan(nodata_value):
        valid_mask = ~np.isnan(orig_profile) & ~np.isnan(deno_profile)
    else:
        valid_mask = (orig_profile != nodata_value) & (deno_profile != nodata_value)

    x = np.arange(len(orig_profile))

    ax4.plot(
        x[valid_mask],
        orig_profile[valid_mask],
        label="Original",
        alpha=0.7,
        linewidth=1,
    )
    ax4.plot(
        x[valid_mask],
        deno_profile[valid_mask],
        label="Denoised",
        alpha=0.9,
        linewidth=1.5,
    )
    ax4.set_title(f"Cross-Section Profile (Row {profile_row})")
    ax4.set_xlabel("Column")
    ax4.set_ylabel("Elevation (m)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add text with statistics
    stats_text = (
        f"Statistics:\n"
        f"  Mean diff: {diff_mean:.4f} m\n"
        f"  Std diff: {diff_std:.4f} m\n"
        f"  Max |diff|: {diff_max:.4f} m"
    )
    ax4.text(
        0.02,
        0.98,
        stats_text,
        transform=ax4.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
    )

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved wavelet diagnostics to {output_path}")
    return output_path


def plot_wavelet_coefficients(
    original: np.ndarray,
    output_path: Path,
    wavelet: str = "db4",
    levels: int = 3,
    threshold_sigma: float = 2.0,
    nodata_value: float = np.nan,
) -> Path:
    """
    Generate diagnostic plots showing wavelet coefficient distributions.

    Creates a figure showing:
    - Coefficient histograms before/after thresholding
    - Decomposition levels visualization
    - Threshold line for each level

    Args:
        original: Original DEM
        output_path: Path to save the diagnostic plot
        wavelet: Wavelet type used for decomposition
        levels: Number of decomposition levels
        threshold_sigma: Sigma multiplier for thresholding
        nodata_value: Value treated as no data

    Returns:
        Path to saved diagnostic plot
    """
    try:
        import pywt
    except ImportError:
        logger.warning("PyWavelets not available, skipping coefficient plot")
        return None

    import matplotlib.pyplot as plt

    # Handle nodata - fill with local mean for decomposition
    data = original.copy().astype(np.float64)
    if np.isnan(nodata_value):
        mask = np.isnan(data)
    else:
        mask = data == nodata_value

    if np.any(mask):
        valid_mean = np.nanmean(original)
        data[mask] = valid_mean

    # Wavelet decomposition
    coeffs = pywt.wavedec2(data, wavelet, level=levels)

    # Estimate noise from finest level HH band
    hh = coeffs[-1][2]
    sigma = np.median(np.abs(hh)) / 0.6745
    threshold = threshold_sigma * sigma

    # Create figure
    fig, axes = plt.subplots(2, levels + 1, figsize=(4 * (levels + 1), 8))
    fig.suptitle(
        f"Wavelet Decomposition Analysis ({wavelet}, {levels} levels, σ={threshold_sigma})",
        fontsize=12,
        fontweight="bold",
    )

    # Row 1: Approximation and detail coefficients
    # Approximation (coarsest level)
    ax_approx = axes[0, 0]
    im = ax_approx.imshow(coeffs[0], cmap="viridis", aspect="auto")
    ax_approx.set_title(f"Approximation (cA{levels})")
    ax_approx.axis("off")
    plt.colorbar(im, ax=ax_approx, fraction=0.046)

    # Detail coefficients at each level
    for i, (cH, cV, cD) in enumerate(coeffs[1:]):
        level = levels - i
        ax = axes[0, i + 1]

        # Show combined detail magnitude
        detail_mag = np.sqrt(cH**2 + cV**2 + cD**2)
        im = ax.imshow(detail_mag, cmap="inferno", aspect="auto")
        ax.set_title(f"Detail Level {level}\n(combined |cH,cV,cD|)")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Row 2: Histograms of detail coefficients
    all_coeffs_flat = []
    for i, (cH, cV, cD) in enumerate(coeffs[1:]):
        level = levels - i
        ax = axes[1, i + 1]

        # Flatten all detail coefficients for this level
        flat = np.concatenate([cH.flatten(), cV.flatten(), cD.flatten()])
        all_coeffs_flat.extend(flat)

        # Histogram
        ax.hist(flat, bins=100, density=True, alpha=0.7, color=f"C{i}")
        ax.axvline(-threshold, color="red", linestyle="--", label=f"±threshold")
        ax.axvline(threshold, color="red", linestyle="--")
        ax.set_title(f"Level {level} Coefficients\n(threshold={threshold:.2f})")
        ax.set_xlabel("Coefficient Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

        # Count how many coefficients would be zeroed
        zeroed = np.sum(np.abs(flat) < threshold)
        total = len(flat)
        ax.text(
            0.98,
            0.98,
            f"Zeroed: {100*zeroed/total:.1f}%",
            transform=ax.transAxes,
            fontsize=9,
            ha="right",
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    # First subplot in row 2: overall stats
    ax_stats = axes[1, 0]
    ax_stats.axis("off")

    stats_text = (
        f"Noise Estimation:\n"
        f"  σ_noise = {sigma:.4f}\n"
        f"  threshold = {threshold_sigma} × σ = {threshold:.4f}\n\n"
        f"Wavelet: {wavelet}\n"
        f"Levels: {levels}\n\n"
        f"Soft thresholding:\n"
        f"  |c| < threshold → 0\n"
        f"  |c| ≥ threshold → sign(c)×(|c|-threshold)"
    )
    ax_stats.text(
        0.1,
        0.9,
        stats_text,
        transform=ax_stats.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.9},
    )

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved wavelet coefficient analysis to {output_path}")
    return output_path


def plot_processing_pipeline(
    stages: dict[str, np.ndarray],
    output_path: Path,
    title: str = "Processing Pipeline",
    cmap: str = "terrain",
    nodata_value: float = np.nan,
) -> Path:
    """
    Generate comparison plots showing multiple processing stages.

    Useful for visualizing an entire processing pipeline with
    multiple transforms (raw → despeckle → wavelet → smooth, etc.)

    Args:
        stages: Dictionary mapping stage names to DEM arrays
            Example: {"Raw": raw_dem, "Despeckled": despeckled, "Final": final}
        output_path: Path to save the diagnostic plot
        title: Plot title
        cmap: Colormap for elevation
        nodata_value: Value treated as no data

    Returns:
        Path to saved diagnostic plot
    """
    import matplotlib.pyplot as plt

    n_stages = len(stages)
    if n_stages == 0:
        logger.warning("No stages provided for pipeline plot")
        return None

    # Calculate grid layout
    cols = min(3, n_stages)
    rows = (n_stages + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Compute global elevation range for consistent coloring
    vmin = float("inf")
    vmax = float("-inf")
    for data in stages.values():
        if np.isnan(nodata_value):
            valid = data[~np.isnan(data)]
        else:
            valid = data[data != nodata_value]
        if len(valid) > 0:
            vmin = min(vmin, np.min(valid))
            vmax = max(vmax, np.max(valid))

    # Plot each stage
    for idx, (name, data) in enumerate(stages.items()):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        if np.isnan(nodata_value):
            masked = np.ma.masked_invalid(data)
        else:
            masked = np.ma.masked_equal(data, nodata_value)

        im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, label="Elevation (m)")

    # Hide unused subplots
    for idx in range(n_stages, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis("off")

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved pipeline comparison to {output_path}")
    return output_path


def generate_full_wavelet_diagnostics(
    original: np.ndarray,
    denoised: np.ndarray,
    output_dir: Path,
    prefix: str = "wavelet",
    wavelet: str = "db4",
    levels: int = 3,
    threshold_sigma: float = 2.0,
    nodata_value: float = np.nan,
) -> Tuple[Path, Path]:
    """
    Generate all wavelet diagnostic plots.

    Creates both the before/after comparison and the coefficient analysis.

    Args:
        original: Original DEM before denoising
        denoised: DEM after wavelet denoising
        output_dir: Directory to save diagnostic plots
        prefix: Filename prefix for saved plots
        wavelet: Wavelet type used for denoising
        levels: Number of decomposition levels
        threshold_sigma: Sigma multiplier used for thresholding
        nodata_value: Value treated as no data

    Returns:
        Tuple of (comparison_plot_path, coefficient_plot_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate comparison plot
    comparison_path = plot_wavelet_diagnostics(
        original,
        denoised,
        output_dir / f"{prefix}_comparison.png",
        title_prefix=f"Wavelet Denoising ({wavelet}, σ={threshold_sigma})",
        nodata_value=nodata_value,
    )

    # Generate coefficient analysis
    coefficients_path = plot_wavelet_coefficients(
        original,
        output_dir / f"{prefix}_coefficients.png",
        wavelet=wavelet,
        levels=levels,
        threshold_sigma=threshold_sigma,
        nodata_value=nodata_value,
    )

    return comparison_path, coefficients_path


def plot_adaptive_smooth_diagnostics(
    original: np.ndarray,
    smoothed: np.ndarray,
    output_path: Path,
    slope_threshold: float = 2.0,
    smooth_sigma: float = 5.0,
    transition_width: float = 1.0,
    title_prefix: str = "Slope-Adaptive Smoothing",
    nodata_value: float = np.nan,
    profile_row: Optional[int] = None,
    cmap: str = "terrain",
    pixel_size: Optional[float] = None,
    edge_threshold: Optional[float] = None,
    edge_window: int = 5,
) -> Path:
    """
    Generate diagnostic plots showing slope-adaptive smoothing effects.

    Creates a multi-panel figure showing:
    - Original DEM
    - Computed slope map
    - Smoothing weight mask (where smoothing is applied)
    - Smoothed DEM
    - Difference (noise removed)
    - Cross-section profile comparison

    Args:
        original: Original DEM before smoothing
        smoothed: DEM after slope-adaptive smoothing
        output_path: Path to save the diagnostic plot
        slope_threshold: Slope threshold in degrees used for smoothing
        smooth_sigma: Gaussian sigma used for smoothing
        transition_width: Width of sigmoid transition zone
        title_prefix: Prefix for plot titles
        nodata_value: Value treated as no data
        profile_row: Row index for cross-section (default: middle row)
        cmap: Colormap for elevation visualization
        pixel_size: Pixel size in meters (e.g., 30.0 for SRTM data).
            Required for accurate slope calculation.
        edge_threshold: Edge preservation threshold in meters (default: None).
            If set, shows which areas are protected due to sharp elevation changes.
        edge_window: Window size for edge detection (default: 5).

    Returns:
        Path to saved diagnostic plot
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    # Handle nodata
    if np.isnan(nodata_value):
        orig_masked = np.ma.masked_invalid(original)
        smooth_masked = np.ma.masked_invalid(smoothed)
        valid_mask = ~np.isnan(original)
    else:
        orig_masked = np.ma.masked_equal(original, nodata_value)
        smooth_masked = np.ma.masked_equal(smoothed, nodata_value)
        valid_mask = original != nodata_value

    # Compute slope from original (same as transform does)
    data_for_slope = original.copy()
    if np.isnan(nodata_value):
        nodata_mask = np.isnan(data_for_slope)
    else:
        nodata_mask = data_for_slope == nodata_value

    if np.any(nodata_mask):
        valid_mean = np.nanmean(original) if np.isnan(nodata_value) else np.mean(
            original[~nodata_mask]
        )
        data_for_slope[nodata_mask] = valid_mean

    # Compute slope using Sobel gradients
    dy = ndimage.sobel(data_for_slope, axis=0, mode="reflect") / 8.0
    dx = ndimage.sobel(data_for_slope, axis=1, mode="reflect") / 8.0

    # Account for pixel size to get actual slope in degrees
    # Without pixel size, slopes will be wildly overestimated
    if pixel_size is not None and pixel_size > 0:
        dx = dx / pixel_size
        dy = dy / pixel_size
    else:
        import warnings

        warnings.warn(
            "No pixel_size provided - slope values may be incorrect. "
            "Pass pixel_size (e.g., 30.0 for SRTM data) for accurate results."
        )

    gradient_magnitude = np.sqrt(dx**2 + dy**2)
    slope_degrees = np.degrees(np.arctan(gradient_magnitude))

    # Compute smoothing weight mask (sigmoid function)
    # Clip exponent to avoid overflow (exp(30) ~ 1e13, exp(700) overflows)
    k = 4.0 / transition_width
    exponent = np.clip(k * (slope_degrees - slope_threshold), -30, 30)
    smoothing_weight_slope = 1.0 / (1.0 + np.exp(exponent))

    # Compute edge protection mask if edge_threshold is provided
    edge_weight = None
    edge_protected_pct = 0
    if edge_threshold is not None:
        from scipy.ndimage import maximum_filter, minimum_filter

        # Compute local elevation range (max - min in window)
        local_max = maximum_filter(data_for_slope, size=edge_window, mode="reflect")
        local_min = minimum_filter(data_for_slope, size=edge_window, mode="reflect")
        local_range = local_max - local_min

        # Create edge mask: 1.0 near edges, 0.0 elsewhere
        edge_k = 4.0 / (edge_threshold * 0.5)  # Transition over half the threshold
        edge_exponent = np.clip(-edge_k * (local_range - edge_threshold), -30, 30)
        edge_weight = 1.0 / (1.0 + np.exp(edge_exponent))

        # Final smoothing weight combines slope and edge protection
        smoothing_weight = smoothing_weight_slope * (1.0 - edge_weight)

        # Stats
        edge_protected_pct = 100 * np.mean(edge_weight[valid_mask] > 0.5) if np.any(valid_mask) else 0
    else:
        smoothing_weight = smoothing_weight_slope

    # Compute difference
    difference = original - smoothed
    if np.isnan(nodata_value):
        diff_masked = np.ma.masked_invalid(difference)
        valid_diff = difference[~np.isnan(difference)]
    else:
        diff_masked = np.ma.masked_where(
            (original == nodata_value) | (smoothed == nodata_value), difference
        )
        valid_diff = difference[valid_mask]

    # Statistics
    diff_std = np.std(valid_diff)
    diff_mean = np.mean(valid_diff)
    diff_max = np.max(np.abs(valid_diff))

    # Slope statistics
    valid_slopes = slope_degrees[valid_mask] if np.any(~valid_mask) else slope_degrees.flatten()
    slope_mean = np.mean(valid_slopes)
    slope_std = np.std(valid_slopes)
    flat_fraction = np.mean(valid_slopes < slope_threshold) * 100

    # Compute quantitative smoothing metrics (needed for plot labels)
    original_std = np.std(original[valid_mask])
    smoothed_std = np.std(smoothed[valid_mask])
    std_reduction_pct = 100 * (original_std - smoothed_std) / original_std if original_std > 0 else 0

    # Elevation range for context
    vmin_early = np.nanmin(original)
    vmax_early = np.nanmax(original)
    total_relief = vmax_early - vmin_early

    # Mean change metrics
    mean_abs_change = np.mean(np.abs(valid_diff))
    pct_of_relief = 100 * mean_abs_change / total_relief if total_relief > 0 else 0

    # Mean change in smoothed areas only (where weight > 0.5)
    smooth_mask_flat = smoothing_weight[valid_mask] > 0.5
    if np.any(smooth_mask_flat):
        mean_change_in_smoothed = np.mean(np.abs(valid_diff[smooth_mask_flat]))
        smoothed_area_pct = 100 * np.sum(smooth_mask_flat) / len(smooth_mask_flat)
    else:
        mean_change_in_smoothed = 0
        smoothed_area_pct = 0

    # Profile row (default to middle)
    if profile_row is None:
        profile_row = original.shape[0] // 2

    # Compute Gaussian-blurred version for visualization
    from scipy.ndimage import gaussian_filter
    blurred_dem = gaussian_filter(data_for_slope, sigma=smooth_sigma, mode="reflect")

    # Compute local elevation range for edge visualization (even if edge_threshold not set)
    from scipy.ndimage import maximum_filter, minimum_filter
    local_max = maximum_filter(data_for_slope, size=edge_window, mode="reflect")
    local_min = minimum_filter(data_for_slope, size=edge_window, mode="reflect")
    local_range = local_max - local_min

    # Create expanded figure with 4x3 subplots for step-by-step diagnostics
    fig = plt.figure(figsize=(20, 22))

    # Title with parameters
    edge_str = f", edge={edge_threshold}m" if edge_threshold is not None else ""
    fig.suptitle(
        f"{title_prefix}\n"
        f"Parameters: slope_threshold={slope_threshold}°, σ={smooth_sigma}px, "
        f"transition={transition_width}°{edge_str}, pixel_size={pixel_size:.1f}m" if pixel_size else
        f"{title_prefix}\n"
        f"Parameters: slope_threshold={slope_threshold}°, σ={smooth_sigma}px, "
        f"transition={transition_width}°{edge_str}",
        fontsize=14,
        fontweight="bold",
    )

    # Shared elevation range
    vmin = min(np.nanmin(original), np.nanmin(smoothed))
    vmax = max(np.nanmax(original), np.nanmax(smoothed))

    # =========================================================================
    # ROW 1: INPUT ANALYSIS
    # =========================================================================

    # 1. Original DEM
    ax1 = fig.add_subplot(4, 3, 1)
    im1 = ax1.imshow(orig_masked, cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.axhline(y=profile_row, color="red", linestyle="--", alpha=0.7, linewidth=1)
    ax1.set_title("① Original DEM", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Column")
    ax1.set_ylabel("Row")
    plt.colorbar(im1, ax=ax1, label="Elevation (m)", shrink=0.8)

    # 2. Slope map
    ax2 = fig.add_subplot(4, 3, 2)
    slope_masked = np.ma.masked_where(~valid_mask, slope_degrees) if np.any(~valid_mask) else slope_degrees
    slope_vmax = max(np.percentile(valid_slopes, 99), slope_threshold * 3, 1.0)
    im2 = ax2.imshow(slope_masked, cmap="viridis", vmin=0, vmax=slope_vmax)
    ax2.axhline(y=profile_row, color="red", linestyle="--", alpha=0.7, linewidth=1)
    ax2.set_title(f"② Slope Map\nmean={slope_mean:.2f}°, {flat_fraction:.1f}% < {slope_threshold}°",
                  fontsize=11, fontweight="bold")
    ax2.set_xlabel("Column")
    ax2.set_ylabel("Row")
    cbar2 = plt.colorbar(im2, ax=ax2, label="Slope (degrees)", shrink=0.8)
    # Mark threshold on colorbar
    if slope_vmax > 0:
        cbar2.ax.axhline(y=slope_threshold / slope_vmax, color="red", linewidth=2)

    # 3. Local elevation range (for edge detection context)
    ax3 = fig.add_subplot(4, 3, 3)
    range_masked = np.ma.masked_where(~valid_mask, local_range) if np.any(~valid_mask) else local_range
    range_vmax = max(np.percentile(local_range[valid_mask], 99), edge_threshold * 2 if edge_threshold else 10)
    im3 = ax3.imshow(range_masked, cmap="magma", vmin=0, vmax=range_vmax)
    ax3.axhline(y=profile_row, color="white", linestyle="--", alpha=0.7, linewidth=1)
    if edge_threshold is not None:
        ax3.set_title(f"③ Local Elevation Range\n(window={edge_window}px, threshold={edge_threshold}m)",
                      fontsize=11, fontweight="bold")
    else:
        ax3.set_title(f"③ Local Elevation Range\n(window={edge_window}px, no edge threshold)",
                      fontsize=11, fontweight="bold")
    ax3.set_xlabel("Column")
    ax3.set_ylabel("Row")
    cbar3 = plt.colorbar(im3, ax=ax3, label="Range (m)", shrink=0.8)
    if edge_threshold is not None and range_vmax > 0:
        cbar3.ax.axhline(y=edge_threshold / range_vmax, color="cyan", linewidth=2)

    # =========================================================================
    # ROW 2: WEIGHT CONSTRUCTION (step by step)
    # =========================================================================

    # 4. Slope-based smoothing weight (before edge protection)
    ax4 = fig.add_subplot(4, 3, 4)
    slope_weight_masked = np.ma.masked_where(~valid_mask, smoothing_weight_slope) if np.any(~valid_mask) else smoothing_weight_slope
    im4 = ax4.imshow(slope_weight_masked, cmap="plasma", vmin=0, vmax=1)
    ax4.axhline(y=profile_row, color="white", linestyle="--", alpha=0.7, linewidth=1)
    slope_smooth_pct = 100 * np.mean(smoothing_weight_slope[valid_mask] > 0.5)
    ax4.set_title(f"④ Slope-Based Weight\n(before edge protection, {slope_smooth_pct:.1f}% smoothed)",
                  fontsize=11, fontweight="bold")
    ax4.set_xlabel("Column")
    ax4.set_ylabel("Row")
    plt.colorbar(im4, ax=ax4, label="Weight (0=preserve, 1=smooth)", shrink=0.8)

    # 5. Edge protection mask (or placeholder if not enabled)
    ax5 = fig.add_subplot(4, 3, 5)
    if edge_weight is not None:
        edge_masked = np.ma.masked_where(~valid_mask, edge_weight) if np.any(~valid_mask) else edge_weight
        im5 = ax5.imshow(edge_masked, cmap="cool", vmin=0, vmax=1)
        ax5.set_title(f"⑤ Edge Protection Mask\n({edge_protected_pct:.1f}% protected, threshold={edge_threshold}m)",
                      fontsize=11, fontweight="bold")
        plt.colorbar(im5, ax=ax5, label="Protection (0=none, 1=full)", shrink=0.8)
    else:
        # Show what WOULD be protected at various thresholds
        hypothetical_edge = 1.0 / (1.0 + np.exp(-4.0 / 5.0 * (local_range - 5.0)))
        hypo_masked = np.ma.masked_where(~valid_mask, hypothetical_edge) if np.any(~valid_mask) else hypothetical_edge
        im5 = ax5.imshow(hypo_masked, cmap="cool", vmin=0, vmax=1, alpha=0.5)
        ax5.set_title(f"⑤ Edge Protection\n(disabled - showing hypothetical at 5m)",
                      fontsize=11, fontweight="bold", color="gray")
        plt.colorbar(im5, ax=ax5, label="Hypothetical protection", shrink=0.8)
    ax5.axhline(y=profile_row, color="white", linestyle="--", alpha=0.7, linewidth=1)
    ax5.set_xlabel("Column")
    ax5.set_ylabel("Row")

    # 6. Final combined smoothing weight
    ax6 = fig.add_subplot(4, 3, 6)
    weight_masked = np.ma.masked_where(~valid_mask, smoothing_weight) if np.any(~valid_mask) else smoothing_weight
    im6 = ax6.imshow(weight_masked, cmap="plasma", vmin=0, vmax=1)
    ax6.axhline(y=profile_row, color="white", linestyle="--", alpha=0.7, linewidth=1)
    final_smooth_pct = 100 * np.mean(smoothing_weight[valid_mask] > 0.5)
    if edge_weight is not None:
        ax6.set_title(f"⑥ Final Smoothing Weight\n(after edge protection, {final_smooth_pct:.1f}% smoothed)",
                      fontsize=11, fontweight="bold")
        # Add edge contours to show protected boundaries
        ax6.contour(edge_weight, levels=[0.5], colors=["cyan"], linewidths=1.0, linestyles="--")
    else:
        ax6.set_title(f"⑥ Final Smoothing Weight\n({final_smooth_pct:.1f}% smoothed)",
                      fontsize=11, fontweight="bold")
    ax6.set_xlabel("Column")
    ax6.set_ylabel("Row")
    plt.colorbar(im6, ax=ax6, label="Weight (0=preserve, 1=smooth)", shrink=0.8)

    # =========================================================================
    # ROW 3: RESULTS
    # =========================================================================

    # 7. Gaussian blur applied (intermediate result)
    ax7 = fig.add_subplot(4, 3, 7)
    blur_masked = np.ma.masked_where(~valid_mask, blurred_dem) if np.any(~valid_mask) else blurred_dem
    im7 = ax7.imshow(blur_masked, cmap=cmap, vmin=vmin, vmax=vmax)
    ax7.axhline(y=profile_row, color="red", linestyle="--", alpha=0.7, linewidth=1)
    blur_std = np.std(blurred_dem[valid_mask])
    ax7.set_title(f"⑦ Gaussian Blur (σ={smooth_sigma}px)\nstd: {original_std:.2f}m → {blur_std:.2f}m",
                  fontsize=11, fontweight="bold")
    ax7.set_xlabel("Column")
    ax7.set_ylabel("Row")
    plt.colorbar(im7, ax=ax7, label="Elevation (m)", shrink=0.8)

    # 8. Final smoothed DEM
    ax8 = fig.add_subplot(4, 3, 8)
    im8 = ax8.imshow(smooth_masked, cmap=cmap, vmin=vmin, vmax=vmax)
    ax8.axhline(y=profile_row, color="red", linestyle="--", alpha=0.7, linewidth=1)
    ax8.set_title(f"⑧ Final Smoothed DEM\nstd: {original_std:.2f}m → {smoothed_std:.2f}m ({std_reduction_pct:.2f}% reduction)",
                  fontsize=11, fontweight="bold")
    ax8.set_xlabel("Column")
    ax8.set_ylabel("Row")
    plt.colorbar(im8, ax=ax8, label="Elevation (m)", shrink=0.8)

    # 9. Difference map
    ax9 = fig.add_subplot(4, 3, 9)
    nonzero_diff = valid_diff[np.abs(valid_diff) > 1e-6]
    if len(nonzero_diff) > 0:
        diff_p05 = np.percentile(nonzero_diff, 5)
        diff_p95 = np.percentile(nonzero_diff, 95)
        diff_range = max(abs(diff_p05), abs(diff_p95))
        nonzero_std = np.std(nonzero_diff)
        if nonzero_std > 0:
            diff_range = min(diff_range, nonzero_std * 3)
        if diff_range < 0.1 and diff_max > 0.1:
            diff_range = min(diff_max, 1.0)
    else:
        diff_range = max(diff_max, 0.1)
    if diff_range > 0:
        norm = TwoSlopeNorm(vmin=-diff_range, vcenter=0, vmax=diff_range)
    else:
        norm = None
    im9 = ax9.imshow(diff_masked, cmap="RdBu_r", norm=norm)
    ax9.axhline(y=profile_row, color="black", linestyle="--", alpha=0.7, linewidth=1)
    ax9.set_title(f"⑨ Elevation Change (Original - Smoothed)\nσ={diff_std:.3f}m, max |Δ|={diff_max:.3f}m",
                  fontsize=11, fontweight="bold")
    ax9.set_xlabel("Column")
    ax9.set_ylabel("Row")
    plt.colorbar(im9, ax=ax9, label="Δ Elevation (m)", shrink=0.8)

    # =========================================================================
    # ROW 4: ANALYSIS (cross-sections and stats)
    # =========================================================================

    # 10. Cross-section elevation profile (spans 2 columns)
    ax10 = fig.add_subplot(4, 3, 10)
    orig_profile = original[profile_row, :]
    smooth_profile = smoothed[profile_row, :]
    blur_profile = blurred_dem[profile_row, :]

    if np.isnan(nodata_value):
        valid_profile_mask = ~np.isnan(orig_profile) & ~np.isnan(smooth_profile)
    else:
        valid_profile_mask = (orig_profile != nodata_value) & (smooth_profile != nodata_value)

    x = np.arange(len(orig_profile))

    ax10.plot(x[valid_profile_mask], orig_profile[valid_profile_mask],
              label="Original", alpha=0.7, linewidth=1.0, color="C0")
    ax10.plot(x[valid_profile_mask], blur_profile[valid_profile_mask],
              label=f"Full blur (σ={smooth_sigma})", alpha=0.5, linewidth=1.0, color="gray", linestyle=":")
    ax10.plot(x[valid_profile_mask], smooth_profile[valid_profile_mask],
              label="Adaptive smooth", alpha=0.9, linewidth=1.5, color="C1")
    ax10.set_title(f"⑩ Elevation Profile (Row {profile_row})", fontsize=11, fontweight="bold")
    ax10.set_xlabel("Column")
    ax10.set_ylabel("Elevation (m)")
    ax10.legend(loc="upper right", fontsize=9)
    ax10.grid(True, alpha=0.3)

    # 11. Cross-section weights profile
    ax11 = fig.add_subplot(4, 3, 11)
    slope_profile = slope_degrees[profile_row, :]
    slope_weight_profile = smoothing_weight_slope[profile_row, :]
    final_weight_profile = smoothing_weight[profile_row, :]

    ax11_twin = ax11.twinx()

    # Slope on primary axis
    ax11.plot(x[valid_profile_mask], slope_profile[valid_profile_mask],
              label="Slope", alpha=0.7, linewidth=1.0, color="C2")
    ax11.axhline(y=slope_threshold, color="C2", linestyle="--", alpha=0.5, linewidth=1)
    ax11.set_ylabel("Slope (degrees)", color="C2")
    ax11.tick_params(axis="y", labelcolor="C2")
    ax11.set_ylim(0, slope_vmax)

    # Weights on secondary axis
    ax11_twin.fill_between(x[valid_profile_mask], 0, slope_weight_profile[valid_profile_mask],
                           alpha=0.2, color="purple", label="Slope weight")
    ax11_twin.plot(x[valid_profile_mask], final_weight_profile[valid_profile_mask],
                   label="Final weight", alpha=0.9, linewidth=1.5, color="orange")
    ax11_twin.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax11_twin.set_ylabel("Smoothing Weight", color="orange")
    ax11_twin.tick_params(axis="y", labelcolor="orange")
    ax11_twin.set_ylim(0, 1.1)

    ax11.set_title(f"⑪ Slope & Weight Profile (Row {profile_row})", fontsize=11, fontweight="bold")
    ax11.set_xlabel("Column")

    # Combined legend
    lines1, labels1 = ax11.get_legend_handles_labels()
    lines2, labels2 = ax11_twin.get_legend_handles_labels()
    ax11.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax11.grid(True, alpha=0.3)

    # 12. Statistics summary panel
    ax12 = fig.add_subplot(4, 3, 12)
    ax12.axis("off")

    edge_stats = ""
    if edge_threshold is not None:
        edge_stats = (
            f"\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"EDGE PROTECTION\n"
            f"  Threshold:        {edge_threshold:.1f} m\n"
            f"  Window size:      {edge_window} px\n"
            f"  Area protected:   {edge_protected_pct:.1f}%\n"
            f"  Effect: {slope_smooth_pct:.1f}% → {final_smooth_pct:.1f}% smoothed"
        )

    stats_text = (
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"INPUT TERRAIN\n"
        f"  Shape:            {original.shape[0]} × {original.shape[1]} px\n"
        f"  Pixel size:       {pixel_size:.1f} m\n" if pixel_size else
        f"  Shape:            {original.shape[0]} × {original.shape[1]} px\n"
    )
    stats_text += (
        f"  Elevation range:  {vmin:.1f} - {vmax:.1f} m ({vmax-vmin:.1f}m relief)\n"
        f"  Original std:     {original_std:.2f} m\n"
        f"\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"SLOPE ANALYSIS\n"
        f"  Mean slope:       {slope_mean:.2f}° ± {slope_std:.2f}°\n"
        f"  Flat area:        {flat_fraction:.1f}% (< {slope_threshold}°)\n"
        f"  Steep area:       {100-flat_fraction:.1f}% (≥ {slope_threshold}°)\n"
        f"\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"SMOOTHING EFFECT\n"
        f"  Std reduction:    {std_reduction_pct:.2f}%\n"
        f"  Mean |change|:    {mean_abs_change:.4f} m\n"
        f"  Max |change|:     {diff_max:.4f} m\n"
        f"  % of relief:      {pct_of_relief:.3f}%\n"
        f"  Area smoothed:    {smoothed_area_pct:.1f}% (weight > 0.5)\n"
        f"  Mean Δ in smooth: {mean_change_in_smoothed:.4f} m"
        f"{edge_stats}"
    )

    ax12.text(
        0.05, 0.95, stats_text,
        transform=ax12.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightyellow", "alpha": 0.9, "edgecolor": "gray"},
    )
    ax12.set_title("⑫ Summary Statistics", fontsize=11, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save with higher DPI for better clarity in expanded plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Log key metrics to console
    edge_log = f", {edge_protected_pct:.1f}% edge-protected" if edge_threshold is not None else ""
    logger.info(f"Adaptive smooth effect: std reduced {std_reduction_pct:.2f}%, "
                f"mean |Δ|={mean_abs_change:.3f}m ({pct_of_relief:.2f}% of {total_relief:.1f}m relief), "
                f"{smoothed_area_pct:.1f}% area smoothed{edge_log}")
    logger.info(f"Saved adaptive smoothing diagnostics to {output_path}")
    return output_path


def plot_adaptive_smooth_histogram(
    original: np.ndarray,
    smoothed: np.ndarray,
    output_path: Path,
    slope_threshold: float = 2.0,
    transition_width: float = 1.0,
    title_prefix: str = "Slope-Adaptive Smoothing",
    nodata_value: float = np.nan,
    pixel_size: Optional[float] = None,
) -> Path:
    """
    Generate histogram analysis of slope-adaptive smoothing.

    Shows distribution of slopes and how much smoothing was applied
    at different slope values.

    Args:
        original: Original DEM before smoothing
        smoothed: DEM after slope-adaptive smoothing
        output_path: Path to save the diagnostic plot
        slope_threshold: Slope threshold in degrees used for smoothing
        transition_width: Width of sigmoid transition zone
        title_prefix: Prefix for plot titles
        nodata_value: Value treated as no data
        pixel_size: Pixel size in meters (e.g., 30.0 for SRTM data).
            Required for accurate slope calculation.

    Returns:
        Path to saved diagnostic plot
    """
    import matplotlib.pyplot as plt

    # Handle nodata
    if np.isnan(nodata_value):
        valid_mask = ~np.isnan(original)
    else:
        valid_mask = original != nodata_value

    # Compute slope
    data_for_slope = original.copy()
    if np.any(~valid_mask):
        valid_mean = np.nanmean(original) if np.isnan(nodata_value) else np.mean(
            original[valid_mask]
        )
        data_for_slope[~valid_mask] = valid_mean

    dy = ndimage.sobel(data_for_slope, axis=0, mode="reflect") / 8.0
    dx = ndimage.sobel(data_for_slope, axis=1, mode="reflect") / 8.0

    # Account for pixel size to get actual slope in degrees
    if pixel_size is not None and pixel_size > 0:
        dx = dx / pixel_size
        dy = dy / pixel_size
    else:
        import warnings

        warnings.warn(
            "No pixel_size provided - slope values may be incorrect. "
            "Pass pixel_size (e.g., 30.0 for SRTM data) for accurate results."
        )

    gradient_magnitude = np.sqrt(dx**2 + dy**2)
    slope_degrees = np.degrees(np.arctan(gradient_magnitude))

    # Compute weight and difference
    # Clip exponent to avoid overflow (exp(30) ~ 1e13, exp(700) overflows)
    k = 4.0 / transition_width
    exponent = np.clip(k * (slope_degrees - slope_threshold), -30, 30)
    smoothing_weight = 1.0 / (1.0 + np.exp(exponent))
    difference = np.abs(original - smoothed)

    # Get valid data
    valid_slopes = slope_degrees[valid_mask]
    valid_weights = smoothing_weight[valid_mask]
    valid_diffs = difference[valid_mask]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"{title_prefix} - Distribution Analysis", fontsize=14, fontweight="bold")

    # 1. Slope histogram - auto-scale x-axis to show actual data range
    ax1 = axes[0, 0]
    # Ensure we show at least up to threshold, but scale to actual data
    slope_hist_max = max(np.percentile(valid_slopes, 99.5), slope_threshold * 2, 1.0)
    ax1.hist(valid_slopes, bins=100, density=True, alpha=0.7, color="steelblue", edgecolor="none",
             range=(0, slope_hist_max))
    ax1.axvline(slope_threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold ({slope_threshold}°)")
    ax1.axvline(slope_threshold - transition_width, color="orange", linestyle=":", alpha=0.7)
    ax1.axvline(slope_threshold + transition_width, color="orange", linestyle=":", alpha=0.7,
                label=f"Transition (±{transition_width}°)")
    ax1.set_xlabel("Slope (degrees)")
    ax1.set_ylabel("Density")
    ax1.set_title(f"Slope Distribution (median={np.median(valid_slopes):.3f}°)")
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, slope_hist_max)

    # 2. Weight histogram
    ax2 = axes[0, 1]
    ax2.hist(valid_weights, bins=50, density=True, alpha=0.7, color="green", edgecolor="none")
    ax2.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="50% threshold")
    ax2.set_xlabel("Smoothing Weight")
    ax2.set_ylabel("Density")
    ax2.set_title("Smoothing Weight Distribution")
    ax2.legend(fontsize=9)

    # Stats
    fully_smoothed = np.mean(valid_weights > 0.9) * 100
    partially_smoothed = np.mean((valid_weights > 0.1) & (valid_weights <= 0.9)) * 100
    preserved = np.mean(valid_weights <= 0.1) * 100
    ax2.text(
        0.98, 0.98,
        f"Fully smoothed (w>0.9): {fully_smoothed:.1f}%\n"
        f"Transition (0.1<w≤0.9): {partially_smoothed:.1f}%\n"
        f"Preserved (w≤0.1): {preserved:.1f}%",
        transform=ax2.transAxes, fontsize=9, ha="right", va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    # 3. Sigmoid function visualization - scale to actual data range
    ax3 = axes[1, 0]
    # Show sigmoid over the relevant range (up to where data actually exists)
    sigmoid_xmax = max(slope_hist_max, slope_threshold * 3, 1.0)
    slope_range = np.linspace(0, sigmoid_xmax, 200)
    weight_curve = 1.0 / (1.0 + np.exp(k * (slope_range - slope_threshold)))
    ax3.plot(slope_range, weight_curve, linewidth=2, color="C0")
    ax3.axvline(slope_threshold, color="red", linestyle="--", alpha=0.7, label=f"threshold={slope_threshold}°")
    ax3.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax3.fill_between(slope_range, 0, weight_curve, alpha=0.2, color="C0")
    # Mark actual median slope on the curve
    median_slope = np.median(valid_slopes)
    median_weight = 1.0 / (1.0 + np.exp(k * (median_slope - slope_threshold)))
    ax3.scatter([median_slope], [median_weight], color="orange", s=100, zorder=5,
                label=f"median slope={median_slope:.3f}°")
    ax3.set_xlabel("Slope (degrees)")
    ax3.set_ylabel("Smoothing Weight")
    ax3.set_title(f"Sigmoid Transfer Function (k={k:.2f})")
    ax3.set_xlim(0, sigmoid_xmax)
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)

    # 4. Change vs slope scatter/hexbin - use viridis, auto-scale axes
    ax4 = axes[1, 1]
    # Auto-scale to actual data range
    slope_extent_max = max(np.percentile(valid_slopes, 99.5), slope_threshold * 2, 1.0)
    diff_extent_max = max(np.percentile(valid_diffs, 99.5), 0.01)  # Ensure visible even if all near zero
    # Use hexbin for efficiency with large datasets
    hb = ax4.hexbin(
        valid_slopes,
        valid_diffs,
        gridsize=50,
        cmap="viridis",
        mincnt=1,
        extent=[0, slope_extent_max, 0, diff_extent_max],
    )
    ax4.axvline(slope_threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold")
    ax4.set_xlabel("Slope (degrees)")
    ax4.set_ylabel("|Change| (m)")
    ax4.set_title(f"Elevation Change vs Slope (max Δ={np.max(valid_diffs):.4f}m)")
    ax4.legend(fontsize=9)
    plt.colorbar(hb, ax=ax4, label="Count")

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved adaptive smoothing histogram to {output_path}")
    return output_path


def generate_full_adaptive_smooth_diagnostics(
    original: np.ndarray,
    smoothed: np.ndarray,
    output_dir: Path,
    prefix: str = "adaptive_smooth",
    slope_threshold: float = 2.0,
    smooth_sigma: float = 5.0,
    transition_width: float = 1.0,
    nodata_value: float = np.nan,
    pixel_size: Optional[float] = None,
    edge_threshold: Optional[float] = None,
    edge_window: int = 5,
) -> Tuple[Path, Path]:
    """
    Generate all slope-adaptive smoothing diagnostic plots.

    Creates both the spatial comparison and the histogram analysis.

    Args:
        original: Original DEM before smoothing
        smoothed: DEM after slope-adaptive smoothing
        output_dir: Directory to save diagnostic plots
        prefix: Filename prefix for saved plots
        slope_threshold: Slope threshold in degrees
        smooth_sigma: Gaussian sigma used for smoothing
        transition_width: Width of sigmoid transition zone
        nodata_value: Value treated as no data
        pixel_size: Pixel size in meters (e.g., 30.0 for SRTM data).
            Required for accurate slope calculation.
        edge_threshold: Edge preservation threshold in meters (default: None).
            If set, shows which areas are protected due to sharp elevation changes.
        edge_window: Window size for edge detection (default: 5).

    Returns:
        Tuple of (spatial_plot_path, histogram_plot_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate spatial comparison plot
    spatial_path = plot_adaptive_smooth_diagnostics(
        original,
        smoothed,
        output_dir / f"{prefix}_comparison.png",
        slope_threshold=slope_threshold,
        smooth_sigma=smooth_sigma,
        transition_width=transition_width,
        nodata_value=nodata_value,
        pixel_size=pixel_size,
        edge_threshold=edge_threshold,
        edge_window=edge_window,
    )

    # Generate histogram analysis
    histogram_path = plot_adaptive_smooth_histogram(
        original,
        smoothed,
        output_dir / f"{prefix}_histogram.png",
        slope_threshold=slope_threshold,
        transition_width=transition_width,
        nodata_value=nodata_value,
        pixel_size=pixel_size,
    )

    return spatial_path, histogram_path


def plot_bump_removal_diagnostics(
    original: np.ndarray,
    after_removal: np.ndarray,
    output_path: Path,
    kernel_size: int = 3,
    title_prefix: str = "Bump Removal",
    nodata_value: float = np.nan,
    cmap: str = "terrain",
) -> Path:
    """
    Generate diagnostic plot for morphological bump removal.

    Shows:
    - Original DEM
    - After bump removal
    - Bumps removed (difference) - always positive since only peaks are removed
    - Histogram of bump heights

    Args:
        original: Original DEM before bump removal
        after_removal: DEM after morphological opening
        output_path: Path to save the diagnostic plot
        kernel_size: Kernel size used for removal
        title_prefix: Prefix for plot titles
        nodata_value: Value treated as no data
        cmap: Colormap for elevation visualization

    Returns:
        Path to saved diagnostic plot
    """
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Handle nodata mask
    if np.isnan(nodata_value):
        valid_mask = ~np.isnan(original) & ~np.isnan(after_removal)
    else:
        valid_mask = (original != nodata_value) & (after_removal != nodata_value)

    # Compute difference (bumps removed - always >= 0)
    difference = original - after_removal
    valid_diff = difference[valid_mask]

    # Statistics
    bumps_affected = np.sum(valid_diff > 0.1)  # Pixels lowered by >0.1m
    total_valid = np.sum(valid_mask)
    bump_fraction = 100 * bumps_affected / total_valid if total_valid > 0 else 0
    max_bump = np.max(valid_diff)
    mean_bump = np.mean(valid_diff[valid_diff > 0.1]) if bumps_affected > 0 else 0

    # Create masked arrays for plotting
    orig_masked = np.ma.masked_where(~valid_mask, original)
    after_masked = np.ma.masked_where(~valid_mask, after_removal)
    diff_masked = np.ma.masked_where(~valid_mask, difference)

    # Shared elevation range
    vmin = min(np.nanmin(original), np.nanmin(after_removal))
    vmax = max(np.nanmax(original), np.nanmax(after_removal))

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"{title_prefix} (kernel={kernel_size}, {bump_fraction:.1f}% pixels affected)",
        fontsize=14,
        fontweight="bold",
    )

    # 1. Original DEM
    ax1 = axes[0, 0]
    im1 = ax1.imshow(orig_masked, cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.set_title("Original DEM")
    ax1.set_xlabel("Column")
    ax1.set_ylabel("Row")
    plt.colorbar(im1, ax=ax1, label="Elevation (m)")

    # 2. After bump removal
    ax2 = axes[0, 1]
    im2 = ax2.imshow(after_masked, cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.set_title("After Bump Removal")
    ax2.set_xlabel("Column")
    ax2.set_ylabel("Row")
    plt.colorbar(im2, ax=ax2, label="Elevation (m)")

    # 3. Bumps removed (difference) - use hot colormap, always positive
    ax3 = axes[1, 0]
    # Scale to show variation - use 95th percentile to avoid outliers
    if bumps_affected > 0:
        diff_p95 = np.percentile(valid_diff[valid_diff > 0], 95)
        diff_vmax = max(diff_p95, 1.0)  # At least 1m range
    else:
        diff_vmax = 1.0
    im3 = ax3.imshow(diff_masked, cmap="YlOrRd", vmin=0, vmax=diff_vmax)
    ax3.set_title(f"Bumps Removed (max={max_bump:.1f}m, mean={mean_bump:.1f}m)")
    ax3.set_xlabel("Column")
    ax3.set_ylabel("Row")
    plt.colorbar(im3, ax=ax3, label="Height Removed (m)")

    # 4. Histogram of bump heights
    ax4 = axes[1, 1]
    bump_heights = valid_diff[valid_diff > 0.1]
    if len(bump_heights) > 0:
        ax4.hist(bump_heights, bins=50, color="coral", edgecolor="darkred", alpha=0.7)
        ax4.axvline(mean_bump, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_bump:.1f}m")
        ax4.axvline(np.median(bump_heights), color="blue", linestyle="--", linewidth=2,
                    label=f"Median: {np.median(bump_heights):.1f}m")
        ax4.legend()
    ax4.set_title(f"Bump Height Distribution ({bumps_affected:,} pixels)")
    ax4.set_xlabel("Bump Height (m)")
    ax4.set_ylabel("Count")
    ax4.set_xlim(0, None)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def generate_bump_removal_diagnostics(
    original: np.ndarray,
    after_removal: np.ndarray,
    output_dir: Path,
    prefix: str = "bump_removal",
    kernel_size: int = 3,
    nodata_value: float = np.nan,
) -> Path:
    """
    Generate bump removal diagnostic plot.

    Args:
        original: Original DEM before bump removal
        after_removal: DEM after morphological opening
        output_dir: Directory to save diagnostic plot
        prefix: Filename prefix
        kernel_size: Kernel size used for removal
        nodata_value: Value treated as no data

    Returns:
        Path to saved diagnostic plot
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    return plot_bump_removal_diagnostics(
        original,
        after_removal,
        output_dir / f"{prefix}_comparison.png",
        kernel_size=kernel_size,
        nodata_value=nodata_value,
    )


def plot_upscale_diagnostics(
    original: np.ndarray,
    upscaled: np.ndarray,
    output_path: Path,
    scale: int = 4,
    method: str = "unknown",
    title_prefix: str = "Score Upscaling",
    nodata_value: float = np.nan,
    cmap: str = "viridis",
) -> Path:
    """
    Generate diagnostic plots showing score upscaling effects.

    Creates a multi-panel figure showing:
    - Original score grid
    - Upscaled score grid
    - Zoomed comparison of a region
    - Histograms of value distributions
    - Edge detail comparison

    Args:
        original: Original score grid before upscaling
        upscaled: Score grid after upscaling
        output_path: Path to save the diagnostic plot
        scale: Upscaling factor used
        method: Upscaling method name (for title)
        title_prefix: Prefix for plot titles
        nodata_value: Value treated as no data
        cmap: Colormap for score visualization

    Returns:
        Path to saved diagnostic plot
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Handle nodata mask
    if np.isnan(nodata_value):
        orig_valid = ~np.isnan(original)
        up_valid = ~np.isnan(upscaled)
    else:
        orig_valid = original != nodata_value
        up_valid = upscaled != nodata_value

    # Statistics
    orig_min = np.nanmin(original) if np.isnan(nodata_value) else np.min(original[orig_valid])
    orig_max = np.nanmax(original) if np.isnan(nodata_value) else np.max(original[orig_valid])
    up_min = np.nanmin(upscaled) if np.isnan(nodata_value) else np.min(upscaled[up_valid])
    up_max = np.nanmax(upscaled) if np.isnan(nodata_value) else np.max(upscaled[up_valid])

    # Shared color normalization
    vmin = min(orig_min, up_min)
    vmax = max(orig_max, up_max)
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Create figure with 3x2 layout
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        f"{title_prefix}\n"
        f"Method: {method}, Scale: {scale}x, "
        f"Shape: {original.shape} → {upscaled.shape}",
        fontsize=14,
        fontweight="bold",
    )

    # Mask arrays for plotting
    orig_masked = np.ma.masked_where(~orig_valid, original)
    up_masked = np.ma.masked_where(~up_valid, upscaled)

    # =========================================================================
    # ROW 1: Full images
    # =========================================================================

    # 1. Original score grid
    ax1 = fig.add_subplot(3, 2, 1)
    im1 = ax1.imshow(orig_masked, cmap=cmap, norm=norm, interpolation="nearest")
    ax1.set_title(f"① Original Scores ({original.shape[0]}×{original.shape[1]})",
                  fontsize=11, fontweight="bold")
    ax1.set_xlabel("Column")
    ax1.set_ylabel("Row")
    plt.colorbar(im1, ax=ax1, label="Score", shrink=0.8)

    # Find zoom region with high variance (more interesting for comparison)
    zoom_size = min(original.shape[0] // 4, original.shape[1] // 4, 20)
    zoom_size = max(zoom_size, 5)  # Minimum size

    # Compute local variance to find interesting region
    from scipy.ndimage import uniform_filter
    # Fill NaN for variance computation
    orig_filled = original.copy()
    if np.any(~orig_valid):
        orig_filled[~orig_valid] = np.nanmean(original)

    # Local mean and mean of squares
    local_mean = uniform_filter(orig_filled, size=zoom_size, mode='reflect')
    local_sqr_mean = uniform_filter(orig_filled**2, size=zoom_size, mode='reflect')
    local_var = local_sqr_mean - local_mean**2

    # Avoid edges (need room for zoom window)
    margin = zoom_size // 2 + 1
    local_var[:margin, :] = 0
    local_var[-margin:, :] = 0
    local_var[:, :margin] = 0
    local_var[:, -margin:] = 0

    # Find position with middling-high variance (avoid extremes like water edges)
    # Using 75th-85th percentile range to get interesting but not extreme regions
    valid_var = local_var[local_var > 0]
    if len(valid_var) > 0:
        low_threshold = np.percentile(valid_var, 75)
        high_threshold = np.percentile(valid_var, 85)
        # Find candidates in the middling-high variance range
        candidates = np.where((local_var >= low_threshold) & (local_var <= high_threshold))
        if len(candidates[0]) > 0:
            # Pick the candidate closest to image center (more likely to be interesting)
            center_y, center_x = original.shape[0] // 2, original.shape[1] // 2
            distances = np.sqrt((candidates[0] - center_y)**2 + (candidates[1] - center_x)**2)
            best_idx = np.argmin(distances)
            var_idx = (candidates[0][best_idx], candidates[1][best_idx])
        else:
            # Fallback to maximum variance if no candidates in range
            var_idx = np.unravel_index(np.argmax(local_var), local_var.shape)
    else:
        # Fallback to center if no valid variance
        var_idx = (original.shape[0] // 2, original.shape[1] // 2)

    zoom_y = var_idx[0] - zoom_size // 2
    zoom_x = var_idx[1] - zoom_size // 2

    # Clamp to valid range
    zoom_y = max(0, min(zoom_y, original.shape[0] - zoom_size))
    zoom_x = max(0, min(zoom_x, original.shape[1] - zoom_size))

    rect = plt.Rectangle((zoom_x - 0.5, zoom_y - 0.5), zoom_size, zoom_size,
                          fill=False, edgecolor="red", linewidth=2)
    ax1.add_patch(rect)

    # 2. Upscaled score grid
    ax2 = fig.add_subplot(3, 2, 2)
    im2 = ax2.imshow(up_masked, cmap=cmap, norm=norm, interpolation="nearest")
    ax2.set_title(f"② Upscaled Scores ({upscaled.shape[0]}×{upscaled.shape[1]})",
                  fontsize=11, fontweight="bold")
    ax2.set_xlabel("Column")
    ax2.set_ylabel("Row")
    plt.colorbar(im2, ax=ax2, label="Score", shrink=0.8)

    # Draw corresponding zoom region box (scaled)
    rect2 = plt.Rectangle((zoom_x * scale - 0.5, zoom_y * scale - 0.5),
                           zoom_size * scale, zoom_size * scale,
                           fill=False, edgecolor="red", linewidth=2)
    ax2.add_patch(rect2)

    # =========================================================================
    # ROW 2: Zoomed comparison
    # =========================================================================

    # 3. Zoomed original (with interpolation=nearest to show pixels)
    ax3 = fig.add_subplot(3, 2, 3)
    zoom_orig = original[zoom_y:zoom_y + zoom_size, zoom_x:zoom_x + zoom_size]
    im3 = ax3.imshow(zoom_orig, cmap=cmap, norm=norm, interpolation="nearest")
    ax3.set_title(f"③ Original (mid-high variance {zoom_size}×{zoom_size} region)",
                  fontsize=11, fontweight="bold")
    ax3.set_xlabel("Column")
    ax3.set_ylabel("Row")
    plt.colorbar(im3, ax=ax3, label="Score", shrink=0.8)
    # Add grid lines to show pixel boundaries
    ax3.set_xticks(np.arange(-0.5, zoom_size, 1), minor=True)
    ax3.set_yticks(np.arange(-0.5, zoom_size, 1), minor=True)
    ax3.grid(which="minor", color="white", linewidth=0.5, alpha=0.5)

    # 4. Zoomed upscaled (same region, but at higher resolution)
    ax4 = fig.add_subplot(3, 2, 4)
    zoom_up = upscaled[zoom_y * scale:(zoom_y + zoom_size) * scale,
                       zoom_x * scale:(zoom_x + zoom_size) * scale]
    im4 = ax4.imshow(zoom_up, cmap=cmap, norm=norm, interpolation="nearest")
    ax4.set_title(f"④ Upscaled (same region, {zoom_up.shape[0]}×{zoom_up.shape[1]} pixels)",
                  fontsize=11, fontweight="bold")
    ax4.set_xlabel("Column")
    ax4.set_ylabel("Row")
    plt.colorbar(im4, ax=ax4, label="Score", shrink=0.8)

    # =========================================================================
    # ROW 3: Histograms and statistics
    # =========================================================================

    # 5. Value distribution histograms
    ax5 = fig.add_subplot(3, 2, 5)
    orig_values = original[orig_valid].flatten()
    up_values = upscaled[up_valid].flatten()

    bins = np.linspace(vmin, vmax, 50)
    ax5.hist(orig_values, bins=bins, alpha=0.6, label=f"Original (n={len(orig_values):,})",
             color="C0", density=True)
    ax5.hist(up_values, bins=bins, alpha=0.6, label=f"Upscaled (n={len(up_values):,})",
             color="C1", density=True)
    ax5.set_xlabel("Score Value")
    ax5.set_ylabel("Density")
    ax5.set_title("⑤ Value Distribution Comparison", fontsize=11, fontweight="bold")
    ax5.legend(fontsize=9)

    # Add statistics
    orig_mean = np.mean(orig_values)
    orig_std = np.std(orig_values)
    up_mean = np.mean(up_values)
    up_std = np.std(up_values)
    ax5.axvline(orig_mean, color="C0", linestyle="--", alpha=0.8)
    ax5.axvline(up_mean, color="C1", linestyle="--", alpha=0.8)

    # 6. Statistics summary panel
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.axis("off")

    # Compute gradient magnitude to measure smoothness
    from scipy.ndimage import sobel
    orig_grad = np.sqrt(sobel(original, axis=0)**2 + sobel(original, axis=1)**2)
    up_grad = np.sqrt(sobel(upscaled, axis=0)**2 + sobel(upscaled, axis=1)**2)
    orig_roughness = np.mean(orig_grad[orig_valid])
    up_roughness = np.mean(up_grad[up_valid])
    roughness_change = 100 * (up_roughness - orig_roughness) / orig_roughness if orig_roughness > 0 else 0

    stats_text = (
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"UPSCALING PARAMETERS\n"
        f"  Method:          {method}\n"
        f"  Scale factor:    {scale}x\n"
        f"  Shape:           {original.shape} → {upscaled.shape}\n"
        f"  Pixel count:     {original.size:,} → {upscaled.size:,}\n"
        f"\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"VALUE STATISTICS\n"
        f"  Original range:  {orig_min:.2f} - {orig_max:.2f}\n"
        f"  Upscaled range:  {up_min:.2f} - {up_max:.2f}\n"
        f"  Original mean:   {orig_mean:.2f} ± {orig_std:.2f}\n"
        f"  Upscaled mean:   {up_mean:.2f} ± {up_std:.2f}\n"
        f"\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"SMOOTHNESS ANALYSIS\n"
        f"  Original gradient: {orig_roughness:.4f}\n"
        f"  Upscaled gradient: {up_roughness:.4f}\n"
        f"  Change:            {roughness_change:+.1f}%\n"
        f"  (negative = smoother transitions)\n"
    )

    ax6.text(
        0.05, 0.95, stats_text,
        transform=ax6.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightyellow", "alpha": 0.9, "edgecolor": "gray"},
    )
    ax6.set_title("⑥ Summary Statistics", fontsize=11, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved upscale diagnostics to {output_path}")
    return output_path


def generate_upscale_diagnostics(
    original: np.ndarray,
    upscaled: np.ndarray,
    output_dir: Path,
    prefix: str = "score_upscale",
    scale: int = 4,
    method: str = "unknown",
    nodata_value: float = np.nan,
    cmap: str = "viridis",
) -> Path:
    """
    Generate upscale diagnostic plot.

    Args:
        original: Original score grid before upscaling
        upscaled: Score grid after upscaling
        output_dir: Directory to save diagnostic plot
        prefix: Filename prefix
        scale: Upscaling factor used
        method: Upscaling method name
        nodata_value: Value treated as no data
        cmap: Colormap for score visualization

    Returns:
        Path to saved diagnostic plot
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    return plot_upscale_diagnostics(
        original,
        upscaled,
        output_dir / f"{prefix}_diagnostics.png",
        scale=scale,
        method=method,
        nodata_value=nodata_value,
        cmap=cmap,
    )


# =============================================================================
# ROAD ELEVATION DIAGNOSTICS
# =============================================================================


def plot_road_elevation_diagnostics(
    dem: np.ndarray,
    road_mask: np.ndarray,
    output_dir: Path,
    prefix: str = "road_elevation",
    kernel_radius: int = 3,
    nodata_value: float = np.nan,
    cmap: str = "terrain",
    dpi: int = 300,
) -> list[Path]:
    """
    Plot the max height difference between road pixels and surrounding terrain.

    For each road pixel, computes the max elevation of nearby non-road pixels
    and reports the difference (road - surrounding_max). Negative means the road
    sits below the highest neighboring terrain; positive means the road is
    higher than all surrounding non-road pixels.

    Saves six separate high-resolution plots:
    - {prefix}_dem_overlay.png: DEM with road overlay
    - {prefix}_diff_map.png: Spatial map of road-vs-surrounding max difference
    - {prefix}_positive_map.png: Roads above surrounding terrain (positive deviations)
    - {prefix}_negative_map.png: Roads below surrounding terrain (negative deviations)
    - {prefix}_histogram.png: Distribution of differences
    - {prefix}_stats.png: Summary statistics panel

    Args:
        dem: 2D elevation array (same shape as road_mask)
        road_mask: 2D array where >0.5 indicates road pixels
        output_dir: Directory to save the diagnostic plots
        prefix: Filename prefix for all output files
        kernel_radius: Radius of the neighborhood window for computing
            surrounding terrain max (default: 3 pixels)
        nodata_value: Value treated as no data in the DEM
        cmap: Colormap for elevation visualization
        dpi: Output resolution (default: 300)

    Returns:
        List of paths to saved diagnostic plots, empty if no roads
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    # Binary road mask
    is_road = road_mask > 0.5
    n_road = int(np.sum(is_road))

    if n_road == 0:
        logger.warning("No road pixels found, skipping road elevation diagnostic")
        return saved_paths

    # Valid DEM mask
    if np.isnan(nodata_value):
        valid_dem = ~np.isnan(dem)
    else:
        valid_dem = dem != nodata_value

    h, w = dem.shape
    ksize = 2 * kernel_radius + 1

    # Compute max of surrounding non-road pixels using maximum_filter.
    # Set road pixels to -inf so they lose the max competition, then run
    # the compiled C filter (orders of magnitude faster than generic_filter
    # with a Python callback on large grids like 3625x4809).
    from scipy.ndimage import maximum_filter

    dem_no_roads = dem.astype(np.float64).copy()
    dem_no_roads[is_road] = -np.inf
    dem_no_roads[~valid_dem] = -np.inf

    surrounding_max = maximum_filter(dem_no_roads, size=ksize, mode='reflect')

    # Where the max is still -inf, all neighbors were road/nodata
    surrounding_max[surrounding_max == -np.inf] = np.nan

    # Compute difference: road elevation - surrounding terrain max
    diff = np.full((h, w), np.nan, dtype=np.float64)
    road_and_valid = is_road & valid_dem & ~np.isnan(surrounding_max)
    diff[road_and_valid] = dem[road_and_valid] - surrounding_max[road_and_valid]

    valid_diffs = diff[road_and_valid]
    n_valid = len(valid_diffs)

    if n_valid == 0:
        logger.warning("No valid road-vs-surrounding differences computed")
        return saved_paths

    # Statistics
    median_diff = float(np.median(valid_diffs))
    mean_diff = float(np.mean(valid_diffs))
    std_diff = float(np.std(valid_diffs))
    pct_higher = float(100 * np.mean(valid_diffs > 0))
    pct_lower = float(100 * np.mean(valid_diffs < 0))
    max_above = float(np.max(valid_diffs))
    max_below = float(np.min(valid_diffs))
    p05 = float(np.percentile(valid_diffs, 5))
    p95 = float(np.percentile(valid_diffs, 95))

    # Shared data for spatial plots
    dem_masked = np.ma.masked_where(~valid_dem, dem)
    vmin_e = float(np.nanmin(dem[valid_dem]))
    vmax_e = float(np.nanmax(dem[valid_dem]))
    diff_at_roads = np.ma.masked_where(~road_and_valid, diff)
    abs_max = max(abs(p05), abs(p95), 0.01)
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    # --- Plot 1: DEM with road overlay ---
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    im1 = ax1.imshow(dem_masked, cmap=cmap, vmin=vmin_e, vmax=vmax_e)
    road_overlay = np.ma.masked_where(~is_road, np.ones((h, w)))
    ax1.imshow(road_overlay, cmap='Reds', vmin=0, vmax=1, alpha=0.6)
    ax1.set_title("DEM with Road Overlay", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Column")
    ax1.set_ylabel("Row")
    plt.colorbar(im1, ax=ax1, label="Elevation (m)")
    plt.tight_layout()
    p1 = output_dir / f"{prefix}_dem_overlay.png"
    fig1.savefig(p1, dpi=dpi, bbox_inches="tight")
    plt.close(fig1)
    saved_paths.append(p1)

    # --- Plot 2: Spatial difference map ---
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    im2 = ax2.imshow(diff_at_roads, cmap='RdBu_r', norm=norm)
    ax2.set_title(
        f"Road - Surrounding Max Elevation (m)\n"
        f"(kernel radius={kernel_radius}px, {n_valid:,} road pixels)",
        fontsize=14, fontweight="bold",
    )
    ax2.set_xlabel("Column")
    ax2.set_ylabel("Row")
    plt.colorbar(im2, ax=ax2, label="Difference (m)")
    plt.tight_layout()
    p2 = output_dir / f"{prefix}_diff_map.png"
    fig2.savefig(p2, dpi=dpi, bbox_inches="tight")
    plt.close(fig2)
    saved_paths.append(p2)

    # --- Plot 3: Positive deviations (road above surrounding terrain) ---
    pos_mask = road_and_valid & (diff > 0)
    n_pos = int(np.sum(pos_mask))
    fig_pos, ax_pos = plt.subplots(figsize=(10, 8))
    if n_pos > 0:
        pos_only = np.ma.masked_where(~pos_mask, diff)
        pos_max = float(np.max(diff[pos_mask]))
        im_pos = ax_pos.imshow(pos_only, cmap='Reds', vmin=0, vmax=max(pos_max, 0.01))
        plt.colorbar(im_pos, ax=ax_pos, label="Road above surrounding (m)")
    else:
        ax_pos.imshow(np.zeros((h, w)), cmap='Reds', vmin=0, vmax=1)
        ax_pos.text(0.5, 0.5, "No positive deviations",
                    transform=ax_pos.transAxes, ha='center', va='center',
                    fontsize=16, color='gray')
    ax_pos.set_title(
        f"Roads Above Surrounding Terrain\n"
        f"({n_pos:,} pixels, {pct_higher:.1f}% of road)",
        fontsize=14, fontweight="bold",
    )
    ax_pos.set_xlabel("Column")
    ax_pos.set_ylabel("Row")
    plt.tight_layout()
    p_pos = output_dir / f"{prefix}_positive_map.png"
    fig_pos.savefig(p_pos, dpi=dpi, bbox_inches="tight")
    plt.close(fig_pos)
    saved_paths.append(p_pos)

    # --- Plot 4: Negative deviations (road below surrounding terrain) ---
    neg_mask = road_and_valid & (diff < 0)
    n_neg = int(np.sum(neg_mask))
    fig_neg, ax_neg = plt.subplots(figsize=(10, 8))
    if n_neg > 0:
        neg_only = np.ma.masked_where(~neg_mask, -diff)  # flip sign so deeper = larger
        neg_max = float(np.max(-diff[neg_mask]))
        im_neg = ax_neg.imshow(neg_only, cmap='Blues', vmin=0, vmax=max(neg_max, 0.01))
        plt.colorbar(im_neg, ax=ax_neg, label="Road below surrounding (m)")
    else:
        ax_neg.imshow(np.zeros((h, w)), cmap='Blues', vmin=0, vmax=1)
        ax_neg.text(0.5, 0.5, "No negative deviations",
                    transform=ax_neg.transAxes, ha='center', va='center',
                    fontsize=16, color='gray')
    ax_neg.set_title(
        f"Roads Below Surrounding Terrain\n"
        f"({n_neg:,} pixels, {pct_lower:.1f}% of road)",
        fontsize=14, fontweight="bold",
    )
    ax_neg.set_xlabel("Column")
    ax_neg.set_ylabel("Row")
    plt.tight_layout()
    p_neg = output_dir / f"{prefix}_negative_map.png"
    fig_neg.savefig(p_neg, dpi=dpi, bbox_inches="tight")
    plt.close(fig_neg)
    saved_paths.append(p_neg)

    # --- Plot 5: Histogram ---
    fig3, ax3 = plt.subplots(figsize=(10, 7))
    ax3.hist(valid_diffs, bins=80, color='steelblue', edgecolor='none', alpha=0.8)
    ax3.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax3.axvline(median_diff, color='red', linestyle='--', linewidth=2,
                label=f'Median: {median_diff:.3f}m')
    ax3.axvline(mean_diff, color='orange', linestyle='--', linewidth=2,
                label=f'Mean: {mean_diff:.3f}m')
    ax3.set_xlabel("Road - Surrounding Max Elevation (m)", fontsize=12)
    ax3.set_ylabel("Count", fontsize=12)
    ax3.set_title(
        "Distribution of Road vs Max Surrounding Elevation Differences",
        fontsize=14, fontweight="bold",
    )
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    p3 = output_dir / f"{prefix}_histogram.png"
    fig3.savefig(p3, dpi=dpi, bbox_inches="tight")
    plt.close(fig3)
    saved_paths.append(p3)

    # --- Plot 4: Statistics panel ---
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    ax4.axis("off")

    stats_text = (
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"ROAD ELEVATION ANALYSIS\n"
        f"  Metric:              max height difference\n"
        f"  Total road pixels:   {n_road:,}\n"
        f"  Valid comparisons:   {n_valid:,}\n"
        f"  Kernel radius:       {kernel_radius} px\n"
        f"\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"DIFFERENCE (road - surrounding max)\n"
        f"  Median:    {median_diff:+.4f} m\n"
        f"  Mean:      {mean_diff:+.4f} m\n"
        f"  Std:       {std_diff:.4f} m\n"
        f"  P5-P95:    [{p05:+.3f}, {p95:+.3f}] m\n"
        f"  Max above: {max_above:+.3f} m\n"
        f"  Max below: {max_below:+.3f} m\n"
        f"\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"DIRECTION\n"
        f"  Roads higher:  {pct_higher:.1f}%\n"
        f"  Roads lower:   {pct_lower:.1f}%\n"
        f"  Roads equal:   {100 - pct_higher - pct_lower:.1f}%\n"
    )

    ax4.text(
        0.05, 0.95, stats_text,
        transform=ax4.transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightyellow",
              "alpha": 0.9, "edgecolor": "gray"},
    )
    ax4.set_title("Road Elevation Summary Statistics",
                  fontsize=14, fontweight="bold")
    plt.tight_layout()
    p4 = output_dir / f"{prefix}_stats.png"
    fig4.savefig(p4, dpi=dpi, bbox_inches="tight")
    plt.close(fig4)
    saved_paths.append(p4)

    logger.info(f"Road elevation diagnostic (max): median diff = {median_diff:+.4f}m, "
                f"{pct_higher:.1f}% higher, {pct_lower:.1f}% lower")
    for p in saved_paths:
        logger.info(f"  Saved: {p}")
    return saved_paths


def generate_road_elevation_diagnostics(
    dem: np.ndarray,
    road_mask: np.ndarray,
    output_dir: Path,
    prefix: str = "road_elevation",
    kernel_radius: int = 3,
    nodata_value: float = np.nan,
    dpi: int = 300,
) -> list[Path]:
    """
    Generate road elevation diagnostic plots (6 separate high-res files).

    Args:
        dem: 2D elevation array
        road_mask: 2D array where >0.5 indicates road pixels
        output_dir: Directory to save diagnostic plots
        prefix: Filename prefix
        kernel_radius: Radius for surrounding terrain window
        nodata_value: Value treated as no data
        dpi: Output resolution (default: 300)

    Returns:
        List of paths to saved diagnostic plots, empty if no roads
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    return plot_road_elevation_diagnostics(
        dem,
        road_mask,
        output_dir,
        prefix=prefix,
        kernel_radius=kernel_radius,
        nodata_value=nodata_value,
        dpi=dpi,
    )


# =============================================================================
# VERTEX-LEVEL ROAD ELEVATION DIAGNOSTICS
# =============================================================================


def plot_road_vertex_z_diagnostics(
    vertices: np.ndarray,
    road_mask: np.ndarray,
    y_valid: np.ndarray,
    x_valid: np.ndarray,
    output_dir: Path,
    prefix: str = "road_vertex_z",
    kernel_radius: int = 3,
    dpi: int = 300,
    label: str = "",
) -> list[Path]:
    """
    Plot road-vs-surrounding elevation using actual mesh vertex Z positions.

    Reconstructs a Z-height grid from vertex positions, then computes the max
    height of non-road neighbors for each road pixel. This reflects what the
    renderer actually sees, including any smoothing/offset already applied.

    Saves six separate high-resolution plots:
    - {prefix}_dem_overlay.png: Vertex Z grid with road overlay
    - {prefix}_diff_map.png: Spatial difference map (road Z - surrounding max Z)
    - {prefix}_positive_map.png: Roads above surrounding (positive deviations)
    - {prefix}_negative_map.png: Roads below surrounding (negative deviations)
    - {prefix}_histogram.png: Distribution of differences
    - {prefix}_stats.png: Summary statistics

    Args:
        vertices: (N, 3) array of mesh vertex positions (X, Y, Z)
        road_mask: 2D grid array where >0.5 indicates road pixels
        y_valid: 1D array mapping surface vertex index -> grid row
        x_valid: 1D array mapping surface vertex index -> grid column
        output_dir: Directory to save diagnostic plots
        prefix: Filename prefix for all output files
        kernel_radius: Radius for surrounding terrain window (default: 3)
        dpi: Output resolution (default: 300)
        label: Optional label for plot titles (e.g. "pre-smoothing")

    Returns:
        List of paths to saved diagnostic plots, empty if no roads
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from scipy.ndimage import maximum_filter

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    h, w = road_mask.shape
    is_road = road_mask > 0.5
    n_road_grid = int(np.sum(is_road))

    if n_road_grid == 0:
        logger.warning("No road pixels found, skipping vertex Z diagnostic")
        return saved_paths

    # Reconstruct Z grid from vertex positions
    # y_valid/x_valid map surface vertices (indices 0..len-1) to grid coords
    n_surface = len(y_valid)
    z_grid = np.full((h, w), np.nan, dtype=np.float64)
    vertex_z = vertices[:n_surface, 2]
    z_grid[y_valid, x_valid] = vertex_z

    valid_z = ~np.isnan(z_grid)
    ksize = 2 * kernel_radius + 1

    # Compute max Z of surrounding non-road pixels
    z_no_roads = z_grid.copy()
    z_no_roads[is_road] = -np.inf
    z_no_roads[~valid_z] = -np.inf

    surrounding_max = maximum_filter(z_no_roads, size=ksize, mode='reflect')
    surrounding_max[surrounding_max == -np.inf] = np.nan

    # Difference: road vertex Z - surrounding max Z
    diff = np.full((h, w), np.nan, dtype=np.float64)
    road_and_valid = is_road & valid_z & ~np.isnan(surrounding_max)
    diff[road_and_valid] = z_grid[road_and_valid] - surrounding_max[road_and_valid]

    valid_diffs = diff[road_and_valid]
    n_valid = len(valid_diffs)

    if n_valid == 0:
        logger.warning("No valid vertex Z road-vs-surrounding differences")
        return saved_paths

    # Statistics
    median_diff = float(np.median(valid_diffs))
    mean_diff = float(np.mean(valid_diffs))
    std_diff = float(np.std(valid_diffs))
    pct_higher = float(100 * np.mean(valid_diffs > 0))
    pct_lower = float(100 * np.mean(valid_diffs < 0))
    max_above = float(np.max(valid_diffs))
    max_below = float(np.min(valid_diffs))
    p05 = float(np.percentile(valid_diffs, 5))
    p95 = float(np.percentile(valid_diffs, 95))

    title_suffix = f" ({label})" if label else ""

    # Shared data
    z_masked = np.ma.masked_where(~valid_z, z_grid)
    vmin_z = float(np.nanmin(z_grid[valid_z]))
    vmax_z = float(np.nanmax(z_grid[valid_z]))
    diff_at_roads = np.ma.masked_where(~road_and_valid, diff)
    abs_max = max(abs(p05), abs(p95), 0.001)
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    # --- Plot 1: Vertex Z grid with road overlay ---
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    im1 = ax1.imshow(z_masked, cmap='terrain', vmin=vmin_z, vmax=vmax_z)
    road_overlay = np.ma.masked_where(~is_road, np.ones((h, w)))
    ax1.imshow(road_overlay, cmap='Reds', vmin=0, vmax=1, alpha=0.6)
    ax1.set_title(f"Mesh Vertex Z with Road Overlay{title_suffix}",
                  fontsize=14, fontweight="bold")
    ax1.set_xlabel("Column")
    ax1.set_ylabel("Row")
    plt.colorbar(im1, ax=ax1, label="Vertex Z (scene units)")
    plt.tight_layout()
    p1 = output_dir / f"{prefix}_dem_overlay.png"
    fig1.savefig(p1, dpi=dpi, bbox_inches="tight")
    plt.close(fig1)
    saved_paths.append(p1)

    # --- Plot 2: Spatial difference map ---
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    im2 = ax2.imshow(diff_at_roads, cmap='RdBu_r', norm=norm)
    ax2.set_title(
        f"Road Vertex Z - Surrounding Max Z{title_suffix}\n"
        f"(kernel={kernel_radius}px, {n_valid:,} road pixels)",
        fontsize=14, fontweight="bold",
    )
    ax2.set_xlabel("Column")
    ax2.set_ylabel("Row")
    plt.colorbar(im2, ax=ax2, label="Difference (scene units)")
    plt.tight_layout()
    p2 = output_dir / f"{prefix}_diff_map.png"
    fig2.savefig(p2, dpi=dpi, bbox_inches="tight")
    plt.close(fig2)
    saved_paths.append(p2)

    # --- Plot 3: Positive deviations ---
    pos_mask = road_and_valid & (diff > 0)
    n_pos = int(np.sum(pos_mask))
    fig_pos, ax_pos = plt.subplots(figsize=(10, 8))
    if n_pos > 0:
        pos_only = np.ma.masked_where(~pos_mask, diff)
        pos_max = float(np.max(diff[pos_mask]))
        im_pos = ax_pos.imshow(pos_only, cmap='Reds', vmin=0,
                               vmax=max(pos_max, 0.001))
        plt.colorbar(im_pos, ax=ax_pos,
                     label="Road above surrounding (scene units)")
    else:
        ax_pos.imshow(np.zeros((h, w)), cmap='Reds', vmin=0, vmax=1)
        ax_pos.text(0.5, 0.5, "No positive deviations",
                    transform=ax_pos.transAxes, ha='center', va='center',
                    fontsize=16, color='gray')
    ax_pos.set_title(
        f"Roads Above Surrounding Terrain{title_suffix}\n"
        f"({n_pos:,} pixels, {pct_higher:.1f}% of road)",
        fontsize=14, fontweight="bold",
    )
    ax_pos.set_xlabel("Column")
    ax_pos.set_ylabel("Row")
    plt.tight_layout()
    p_pos = output_dir / f"{prefix}_positive_map.png"
    fig_pos.savefig(p_pos, dpi=dpi, bbox_inches="tight")
    plt.close(fig_pos)
    saved_paths.append(p_pos)

    # --- Plot 4: Negative deviations ---
    neg_mask = road_and_valid & (diff < 0)
    n_neg = int(np.sum(neg_mask))
    fig_neg, ax_neg = plt.subplots(figsize=(10, 8))
    if n_neg > 0:
        neg_only = np.ma.masked_where(~neg_mask, -diff)
        neg_max = float(np.max(-diff[neg_mask]))
        im_neg = ax_neg.imshow(neg_only, cmap='Blues', vmin=0,
                               vmax=max(neg_max, 0.001))
        plt.colorbar(im_neg, ax=ax_neg,
                     label="Road below surrounding (scene units)")
    else:
        ax_neg.imshow(np.zeros((h, w)), cmap='Blues', vmin=0, vmax=1)
        ax_neg.text(0.5, 0.5, "No negative deviations",
                    transform=ax_neg.transAxes, ha='center', va='center',
                    fontsize=16, color='gray')
    ax_neg.set_title(
        f"Roads Below Surrounding Terrain{title_suffix}\n"
        f"({n_neg:,} pixels, {pct_lower:.1f}% of road)",
        fontsize=14, fontweight="bold",
    )
    ax_neg.set_xlabel("Column")
    ax_neg.set_ylabel("Row")
    plt.tight_layout()
    p_neg = output_dir / f"{prefix}_negative_map.png"
    fig_neg.savefig(p_neg, dpi=dpi, bbox_inches="tight")
    plt.close(fig_neg)
    saved_paths.append(p_neg)

    # --- Plot 5: Histogram ---
    fig3, ax3 = plt.subplots(figsize=(10, 7))
    ax3.hist(valid_diffs, bins=80, color='steelblue', edgecolor='none', alpha=0.8)
    ax3.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax3.axvline(median_diff, color='red', linestyle='--', linewidth=2,
                label=f'Median: {median_diff:.4f}')
    ax3.axvline(mean_diff, color='orange', linestyle='--', linewidth=2,
                label=f'Mean: {mean_diff:.4f}')
    ax3.set_xlabel("Road Vertex Z - Surrounding Max Z (scene units)", fontsize=12)
    ax3.set_ylabel("Count", fontsize=12)
    ax3.set_title(
        f"Distribution of Road Vertex Z Differences{title_suffix}",
        fontsize=14, fontweight="bold",
    )
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    p3 = output_dir / f"{prefix}_histogram.png"
    fig3.savefig(p3, dpi=dpi, bbox_inches="tight")
    plt.close(fig3)
    saved_paths.append(p3)

    # --- Plot 6: Statistics panel ---
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    ax4.axis("off")

    z_range = vmax_z - vmin_z

    stats_text = (
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"VERTEX Z ROAD ANALYSIS{title_suffix}\n"
        f"  Source:              mesh vertex positions\n"
        f"  Metric:              max height difference\n"
        f"  Road pixels:         {n_road_grid:,}\n"
        f"  Valid comparisons:   {n_valid:,}\n"
        f"  Kernel radius:       {kernel_radius} px\n"
        f"  Z range:             {vmin_z:.4f} - {vmax_z:.4f}\n"
        f"  Z relief:            {z_range:.4f}\n"
        f"\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"DIFFERENCE (road Z - surrounding max Z)\n"
        f"  Median:    {median_diff:+.6f}\n"
        f"  Mean:      {mean_diff:+.6f}\n"
        f"  Std:       {std_diff:.6f}\n"
        f"  P5-P95:    [{p05:+.5f}, {p95:+.5f}]\n"
        f"  Max above: {max_above:+.5f}\n"
        f"  Max below: {max_below:+.5f}\n"
        f"\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"DIRECTION\n"
        f"  Roads higher:  {pct_higher:.1f}%\n"
        f"  Roads lower:   {pct_lower:.1f}%\n"
        f"  Roads equal:   {100 - pct_higher - pct_lower:.1f}%\n"
    )

    ax4.text(
        0.05, 0.95, stats_text,
        transform=ax4.transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightyellow",
              "alpha": 0.9, "edgecolor": "gray"},
    )
    ax4.set_title(f"Vertex Z Road Summary{title_suffix}",
                  fontsize=14, fontweight="bold")
    plt.tight_layout()
    p4 = output_dir / f"{prefix}_stats.png"
    fig4.savefig(p4, dpi=dpi, bbox_inches="tight")
    plt.close(fig4)
    saved_paths.append(p4)

    logger.info(f"Vertex Z road diagnostic{title_suffix}: "
                f"median diff = {median_diff:+.6f}, "
                f"{pct_higher:.1f}% higher, {pct_lower:.1f}% lower")
    for p in saved_paths:
        logger.info(f"  Saved: {p}")
    return saved_paths


# =============================================================================
# RENDER IMAGE DIAGNOSTICS
# =============================================================================


def generate_rgb_histogram(
    image_path: Path,
    output_path: Path,
) -> Optional[Path]:
    """
    Generate and save an RGB histogram of a rendered image.

    Creates a figure with histograms for each color channel (R, G, B)
    overlaid on the same axes with transparency. Useful for analyzing
    color balance and distribution in rendered outputs.

    Args:
        image_path: Path to the rendered image (PNG, JPEG, etc.)
        output_path: Path to save the histogram image

    Returns:
        Path to saved histogram image, or None if failed
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    try:
        # Load image
        img = Image.open(image_path)
        img_array = np.array(img)

        logger.info(f"Generating RGB histogram for {image_path.name}...")
        logger.info(f"  Image shape: {img_array.shape}")

        # Handle RGBA vs RGB
        if img_array.ndim == 3 and img_array.shape[2] >= 3:
            r_channel = img_array[:, :, 0].flatten()
            g_channel = img_array[:, :, 1].flatten()
            b_channel = img_array[:, :, 2].flatten()
        else:
            logger.warning("Image is not RGB/RGBA, skipping histogram")
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot histograms with transparency
        bins = 256
        alpha = 0.5

        ax.hist(r_channel, bins=bins, range=(0, 255), color='red',
                alpha=alpha, label='Red')
        ax.hist(g_channel, bins=bins, range=(0, 255), color='green',
                alpha=alpha, label='Green')
        ax.hist(b_channel, bins=bins, range=(0, 255), color='blue',
                alpha=alpha, label='Blue')

        # Use log scale for Y-axis to show detail across wide range
        ax.set_yscale('log')

        # Style
        ax.set_xlabel('Pixel Value', fontsize=12)
        ax.set_ylabel('Pixel Count (log scale)', fontsize=12)
        ax.set_title(f'RGB Histogram: {image_path.name}', fontsize=14)
        ax.legend(loc='upper right')
        ax.set_xlim(0, 255)
        ax.grid(True, alpha=0.3)

        # Add stats annotation
        stats_text = (
            f"R: μ={np.mean(r_channel):.1f}, σ={np.std(r_channel):.1f}\n"
            f"G: μ={np.mean(g_channel):.1f}, σ={np.std(g_channel):.1f}\n"
            f"B: μ={np.mean(b_channel):.1f}, σ={np.std(b_channel):.1f}"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Save
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)

        logger.info(f"RGB histogram saved: {output_path}")
        return output_path

    except Exception as e:
        logger.warning(f"Failed to generate RGB histogram: {e}")
        return None


def generate_luminance_histogram(
    image_path: Path,
    output_path: Path,
) -> Optional[Path]:
    """
    Generate and save a luminance (B&W) histogram of a rendered image.

    Shows distribution of brightness values with annotations for pure black
    and pure white pixel counts. Useful for checking exposure and clipping
    in rendered outputs, especially for print preparation.

    Args:
        image_path: Path to the rendered image (PNG, JPEG, etc.)
        output_path: Path to save the histogram image

    Returns:
        Path to saved histogram image, or None if failed
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    try:
        # Load image
        img = Image.open(image_path)
        img_array = np.array(img)

        logger.info(f"Generating luminance histogram for {image_path.name}...")

        # Handle RGBA vs RGB - compute luminance using standard weights
        if img_array.ndim == 3 and img_array.shape[2] >= 3:
            # ITU-R BT.601 luminance formula
            luminance = (
                0.299 * img_array[:, :, 0] +
                0.587 * img_array[:, :, 1] +
                0.114 * img_array[:, :, 2]
            ).astype(np.uint8).flatten()
        elif img_array.ndim == 2:
            luminance = img_array.flatten()
        else:
            logger.warning("Unexpected image format, skipping luminance histogram")
            return None

        total_pixels = len(luminance)

        # Count extremes
        pure_black = np.sum(luminance == 0)
        pure_white = np.sum(luminance == 255)
        near_black = np.sum(luminance <= 5)  # 0-5
        near_white = np.sum(luminance >= 250)  # 250-255

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot histogram
        _, _, patches = ax.hist(
            luminance, bins=256, range=(0, 255),
            color='gray', edgecolor='none', alpha=0.8
        )

        # Highlight extremes
        for i, patch in enumerate(patches):
            if i <= 5:  # Near black
                patch.set_facecolor('black')
            elif i >= 250:  # Near white
                patch.set_facecolor('yellow')
                patch.set_edgecolor('orange')

        # Use log scale for Y-axis to show detail across wide range
        ax.set_yscale('log')

        # Style
        ax.set_xlabel('Luminance (0=black, 255=white)', fontsize=12)
        ax.set_ylabel('Pixel Count (log scale)', fontsize=12)
        ax.set_title(f'Luminance Histogram: {image_path.name}', fontsize=14)
        ax.set_xlim(0, 255)
        ax.grid(True, alpha=0.3)

        # Add vertical lines at extremes
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=255, color='orange', linestyle='--', alpha=0.5, linewidth=1)

        # Stats annotation
        black_pct = 100 * pure_black / total_pixels
        white_pct = 100 * pure_white / total_pixels
        near_black_pct = 100 * near_black / total_pixels
        near_white_pct = 100 * near_white / total_pixels

        stats_text = (
            f"Total pixels: {total_pixels:,}\n"
            f"─────────────────\n"
            f"Pure black (0):   {pure_black:,} ({black_pct:.2f}%)\n"
            f"Near black (≤5):  {near_black:,} ({near_black_pct:.2f}%)\n"
            f"─────────────────\n"
            f"Pure white (255): {pure_white:,} ({white_pct:.2f}%)\n"
            f"Near white (≥250): {near_white:,} ({near_white_pct:.2f}%)\n"
            f"─────────────────\n"
            f"Mean: {np.mean(luminance):.1f}\n"
            f"Median: {np.median(luminance):.1f}\n"
            f"Std: {np.std(luminance):.1f}"
        )
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        # Save
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)

        logger.info(f"Luminance histogram saved: {output_path}")
        logger.info(f"  Pure black: {pure_black:,} ({black_pct:.2f}%)")
        logger.info(f"  Pure white: {pure_white:,} ({white_pct:.2f}%)")
        return output_path

    except Exception as e:
        logger.warning(f"Failed to generate luminance histogram: {e}")
        return None


# =============================================================================
# SCORE DISTRIBUTION DIAGNOSTICS
# =============================================================================


def generate_score_histogram(
    raw_scores: np.ndarray,
    transformed_scores: np.ndarray,
    output_path: Path,
    cmap_name: str = "boreal_mako",
    transform_label: str = "Normalized + Gamma",
    rendered_max: Optional[float] = None,
    rendered_min_nonzero: Optional[float] = None,
    gamma: float = 1.0,
    normalize_scores: bool = False,
) -> Optional[Path]:
    """
    Generate a side-by-side histogram of raw and transformed scores with colormap-colored bars.

    Left panel shows the raw score distribution with bars colored by their eventual
    colormap color (applying the normalization + gamma pipeline to each bin center).
    Right panel shows the transformed score distribution with bars colored directly
    by their position on the colormap.

    Args:
        raw_scores: Raw score array (2D grid, may contain NaN).
        transformed_scores: Scores after normalization + gamma (2D grid, may contain NaN).
        output_path: Path to save the histogram image.
        cmap_name: Matplotlib colormap name for coloring bars.
        transform_label: Label describing the transformation (e.g. "Normalized, gamma=0.5").
        rendered_max: Max score in the rendered region (used for normalization reference lines).
        rendered_min_nonzero: Min nonzero score in the rendered region.
        gamma: Gamma value used in the transformation.
        normalize_scores: Whether --normalize-scores stretch was applied.

    Returns:
        Path to the saved histogram image, or None on failure.
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        cmap = matplotlib.colormaps.get_cmap(cmap_name)

        fig, (ax_raw, ax_trans) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle("Score Distribution: Raw vs Transformed", fontsize=14, fontweight="bold")

        # --- Left panel: Raw scores ---
        valid_raw = raw_scores[~np.isnan(raw_scores)]
        n_bins = 100
        counts_raw, bins_raw, patches_raw = ax_raw.hist(
            valid_raw.flatten(), bins=n_bins, edgecolor="black", linewidth=0.3
        )

        # Color each bar by its eventual colormap color
        for patch, bin_left, bin_right in zip(patches_raw, bins_raw[:-1], bins_raw[1:]):
            bin_center = (bin_left + bin_right) / 2.0
            # Apply the same transform pipeline the render uses
            if rendered_max and rendered_max > 0:
                val = bin_center / rendered_max
            else:
                raw_max = np.nanmax(valid_raw) if len(valid_raw) > 0 else 1.0
                val = bin_center / raw_max if raw_max > 0 else 0.0
            if normalize_scores and rendered_min_nonzero is not None and rendered_max:
                norm_min = rendered_min_nonzero / rendered_max
                if 1.0 > norm_min:
                    val = (val - norm_min) / (1.0 - norm_min)
            val = np.clip(val, 0.0, 1.0) ** gamma
            patch.set_facecolor(cmap(val))

        ax_raw.set_xlabel("Raw Score", fontsize=11)
        ax_raw.set_ylabel("Pixel Count", fontsize=11)
        ax_raw.set_title("Raw Scores (before normalization)", fontsize=12, fontweight="bold")
        ax_raw.grid(True, alpha=0.3, axis="y")

        # Reference lines and stats
        if rendered_max is not None:
            ax_raw.axvline(rendered_max, color="red", linestyle="--", alpha=0.7, label=f"max={rendered_max:.3f}")
        if rendered_min_nonzero is not None and rendered_min_nonzero > 0:
            ax_raw.axvline(rendered_min_nonzero, color="orange", linestyle="--", alpha=0.7,
                           label=f"min (nonzero)={rendered_min_nonzero:.3f}")
        if len(valid_raw) > 0:
            n_zero = np.sum(valid_raw == 0)
            pct_zero = 100.0 * n_zero / len(valid_raw)
            stats_text = (
                f"mean: {np.mean(valid_raw):.3f}\n"
                f"median: {np.median(valid_raw):.3f}\n"
                f"zero: {pct_zero:.1f}%"
            )
            ax_raw.text(
                0.98, 0.98, stats_text, transform=ax_raw.transAxes,
                fontsize=9, verticalalignment="top", horizontalalignment="right",
                fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
            )
        ax_raw.legend(fontsize=9, loc="upper left")

        # --- Right panel: Transformed scores ---
        valid_trans = transformed_scores[~np.isnan(transformed_scores)]
        counts_trans, bins_trans, patches_trans = ax_trans.hist(
            valid_trans.flatten(), bins=n_bins, edgecolor="black", linewidth=0.3
        )

        # Color each bar directly by its colormap position
        for patch, bin_left, bin_right in zip(patches_trans, bins_trans[:-1], bins_trans[1:]):
            bin_center = (bin_left + bin_right) / 2.0
            patch.set_facecolor(cmap(np.clip(bin_center, 0.0, 1.0)))

        norm_label = " + stretch" if normalize_scores else ""
        ax_trans.set_xlabel(f"Transformed Score (÷max, gamma={gamma}{norm_label})", fontsize=11)
        ax_trans.set_ylabel("Pixel Count", fontsize=11)
        ax_trans.set_title(transform_label, fontsize=12, fontweight="bold")
        ax_trans.grid(True, alpha=0.3, axis="y")
        ax_trans.set_xlim(0, 1.0)

        if len(valid_trans) > 0:
            stats_text = (
                f"mean: {np.mean(valid_trans):.3f}\n"
                f"median: {np.median(valid_trans):.3f}\n"
                f"range: [{np.min(valid_trans):.3f}, {np.max(valid_trans):.3f}]"
            )
            ax_trans.text(
                0.98, 0.98, stats_text, transform=ax_trans.transAxes,
                fontsize=9, verticalalignment="top", horizontalalignment="right",
                fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Score distribution histogram saved: {output_path}")
        return output_path

    except Exception as e:
        logger.warning(f"Failed to generate score histogram: {e}")
        return None
