"""
Diagnostic plotting utilities for terrain processing.

Provides visualization functions to understand and debug terrain transforms,
particularly wavelet denoising, slope-adaptive smoothing, and other processing steps.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# ROAD ELEVATION DIAGNOSTICS
# =============================================================================


# =============================================================================
# VERTEX-LEVEL ROAD ELEVATION DIAGNOSTICS
# =============================================================================


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
    print_cmap_name: Optional[str] = None,
    title: Optional[str] = None,
) -> Optional[Path]:
    """
    Generate a side-by-side histogram of raw and transformed scores with colormap-colored bars.

    Left panel shows the raw score distribution with bars colored by their eventual
    colormap color (applying the normalization + gamma pipeline to each bin center).
    Middle panel shows the transformed score distribution with bars colored directly
    by their position on the colormap.
    Optional right panel shows the same transformed distribution with the print-safe
    colormap for visual comparison.

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
        title: Figure title naming what the scores are (default: generic raw vs transformed).
        print_cmap_name: Optional print-safe colormap name. When provided, a third
            panel is added showing the transformed scores with print-safe colors.

    Returns:
        Path to the saved histogram image, or None on failure.
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        cmap = matplotlib.colormaps.get_cmap(cmap_name)

        n_panels = 3 if print_cmap_name else 2
        fig_width = 24 if print_cmap_name else 16
        fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, 7))
        ax_raw, ax_trans = axes[0], axes[1]
        title = title or "Score Distribution: Raw vs Transformed"
        if print_cmap_name:
            title += " vs Print-Safe"
        fig.suptitle(title, fontsize=14, fontweight="bold")

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

        # --- Third panel: Print-safe colormap (optional) ---
        if print_cmap_name:
            ax_print = axes[2]
            print_cmap = matplotlib.colormaps.get_cmap(print_cmap_name)

            counts_print, bins_print, patches_print = ax_print.hist(
                valid_trans.flatten(), bins=n_bins, edgecolor="black", linewidth=0.3
            )

            # Color each bar by its print-safe colormap position
            for patch, bin_left, bin_right in zip(patches_print, bins_print[:-1], bins_print[1:]):
                bin_center = (bin_left + bin_right) / 2.0
                patch.set_facecolor(print_cmap(np.clip(bin_center, 0.0, 1.0)))

            ax_print.set_xlabel(f"Transformed Score (print-safe)", fontsize=11)
            ax_print.set_ylabel("Pixel Count", fontsize=11)
            ax_print.set_title(f"Print-Safe ({print_cmap_name})", fontsize=12, fontweight="bold")
            ax_print.grid(True, alpha=0.3, axis="y")
            ax_print.set_xlim(0, 1.0)

            if len(valid_trans) > 0:
                stats_text = (
                    f"mean: {np.mean(valid_trans):.3f}\n"
                    f"median: {np.median(valid_trans):.3f}\n"
                    f"range: [{np.min(valid_trans):.3f}, {np.max(valid_trans):.3f}]"
                )
                ax_print.text(
                    0.98, 0.98, stats_text, transform=ax_print.transAxes,
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
