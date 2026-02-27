#!/usr/bin/env python3
"""
Diagnostic: visualize raw SNODAS data coverage and statistics.

Loads the cached SNODAS stats (from xc_skiing_cache/) and plots:
  - Valid data mask (where zeros = no SNODAS coverage)
  - Each raw statistic: median max depth, mean snow day ratio,
    interseason CV, mean intraseason CV
  - Derived snow_consistency score
  - Coverage vs consistency side-by-side comparison

Usage:
    uv run python examples/diagnose_snodas_coverage.py
    uv run python examples/diagnose_snodas_coverage.py --cache-file path/to/stats.npz
    uv run python examples/diagnose_snodas_coverage.py --output-dir ./debug_maps
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CACHE = None  # auto-detected from xc_skiing_cache/
DEFAULT_OUTPUT = Path("docs/images/xc_skiing/diagnostics")

# Geographic extent matching the Detroit DEM
EXTENT_LON = (-89.0, -78.0)   # west, east
EXTENT_LAT = (37.0, 47.0)      # south, north


def load_stats(cache_path: Path) -> dict:
    """Load raw SNODAS statistics from npz cache."""
    with np.load(cache_path, allow_pickle=False) as f:
        keys = [k for k in f.files if k not in ("metadata", "failed_files")]
        return {k: f[k] for k in keys}


def snow_consistency(interseason_cv, intraseason_cv,
                     interseason_threshold=1.5, intraseason_threshold=1.0):
    """Reproduce the snow_consistency() transform for diagnostics."""
    inter_norm = np.clip(interseason_cv / interseason_threshold, 0.0, 1.0)
    intra_norm = np.clip(intraseason_cv / intraseason_threshold, 0.0, 1.0)
    inconsistency = np.sqrt((inter_norm**2 + intra_norm**2) / 2)
    return 1.0 - inconsistency


def make_extent_args(shape):
    """Build imshow extent=(left, right, bottom, top) from known geo bounds."""
    return dict(
        extent=[EXTENT_LON[0], EXTENT_LON[1], EXTENT_LAT[0], EXTENT_LAT[1]],
        origin="upper",
        aspect="auto",
    )


def add_colorbar(fig, ax, im, label):
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label, fontsize=9)
    return cbar


def print_stats(name, arr, valid_mask):
    valid = arr[valid_mask]
    print(f"  {name}:")
    print(f"    shape={arr.shape}, zero_pixels={np.sum(~valid_mask):,} ({100*np.mean(~valid_mask):.1f}%)")
    if valid.size > 0:
        pcts = np.percentile(valid, [5, 25, 50, 75, 95])
        print(f"    non-zero range=[{valid.min():.4f}, {valid.max():.4f}]")
        print(f"    P05/25/50/75/95: {pcts[0]:.3f}/{pcts[1]:.3f}/{pcts[2]:.3f}/{pcts[3]:.3f}/{pcts[4]:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose SNODAS coverage and statistics")
    parser.add_argument("--cache-file", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.cache_file is None:
        candidates = sorted(Path("xc_skiing_cache").glob("xc_snodas_compute_stats_*.npz"))
        if not candidates:
            print("ERROR: No compute_stats cache found in xc_skiing_cache/", file=sys.stderr)
            print("Run detroit_xc_skiing.py first to generate the SNODAS cache.", file=sys.stderr)
            sys.exit(1)
        args.cache_file = candidates[-1]  # most recent
        print(f"Auto-detected cache: {args.cache_file}")

    if not args.cache_file.exists():
        print(f"ERROR: Cache file not found: {args.cache_file}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading: {args.cache_file}")
    stats = load_stats(args.cache_file)

    depth = stats["median_max_depth"].astype(np.float64)
    coverage = stats["mean_snow_day_ratio"]
    inter_cv = stats["interseason_cv"]
    intra_cv = stats["mean_intraseason_cv"]

    # Valid = any pixel where coverage > 0 (zero = no SNODAS data)
    valid = coverage > 0
    consistency = snow_consistency(inter_cv, intra_cv)
    # Mask zeros (no-data) for display
    depth_ma = np.where(valid, depth, np.nan)
    coverage_ma = np.where(valid, coverage, np.nan)
    inter_cv_ma = np.where(valid, inter_cv, np.nan)
    intra_cv_ma = np.where(valid, intra_cv, np.nan)
    consistency_ma = np.where(valid, consistency, np.nan)

    print("\nRaw statistics summary:")
    print_stats("median_max_depth (mm)", depth, valid)
    print_stats("mean_snow_day_ratio", coverage, valid)
    print_stats("interseason_cv", inter_cv, valid)
    print_stats("mean_intraseason_cv", intra_cv, valid)
    print_stats("snow_consistency (derived)", consistency, valid)

    ex = make_extent_args(depth.shape)

    # -----------------------------------------------------------------------
    # Figure 1: Coverage mask + raw statistics
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("SNODAS Data Coverage & Raw Statistics", fontsize=14, fontweight="bold")

    # Panel 1: Valid data mask
    ax = axes[0, 0]
    mask_display = valid.astype(np.float32)
    cmap_mask = mcolors.ListedColormap(["#d62728", "#2ca02c"])  # red=missing, green=valid
    im = ax.imshow(mask_display, cmap=cmap_mask, vmin=0, vmax=1, **ex)
    ax.set_title(f"SNODAS Coverage\n({100*valid.mean():.1f}% valid, {100*(~valid).mean():.1f}% missing)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2ca02c", label="Valid"), Patch(color="#d62728", label="Missing (no SNODAS)")],
              loc="lower right", fontsize=8)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Panel 2: Snow depth (mm)
    ax = axes[0, 1]
    im = ax.imshow(depth_ma, cmap="Blues", **ex)
    ax.set_title("Median Max Snow Depth (mm)")
    add_colorbar(fig, ax, im, "mm")
    ax.set_xlabel("Longitude")

    # Panel 3: Snow day ratio (coverage)
    ax = axes[0, 2]
    im = ax.imshow(coverage_ma, cmap="Blues", vmin=0, vmax=1, **ex)
    ax.set_title("Mean Snow Day Ratio\n(fraction of winter days with snow)")
    add_colorbar(fig, ax, im, "ratio 0–1")
    ax.set_xlabel("Longitude")

    # Panel 4: Interseason CV (year-to-year variability)
    ax = axes[1, 0]
    im = ax.imshow(inter_cv_ma, cmap="Reds", vmin=0, **ex)
    ax.set_title("Interseason CV\n(year-to-year variability; threshold=1.5)")
    add_colorbar(fig, ax, im, "CV")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Panel 5: Intraseason CV (within-winter variability)
    ax = axes[1, 1]
    im = ax.imshow(intra_cv_ma, cmap="Reds", vmin=0, **ex)
    ax.set_title("Mean Intraseason CV\n(within-winter variability; threshold=1.0)")
    add_colorbar(fig, ax, im, "CV")
    ax.set_xlabel("Longitude")

    # Panel 6: Derived consistency score
    ax = axes[1, 2]
    im = ax.imshow(consistency_ma, cmap="viridis", vmin=0, vmax=1, **ex)
    ax.set_title("Derived snow_consistency\n(1=consistent=good, 0=variable=bad)")
    add_colorbar(fig, ax, im, "score 0–1")
    ax.set_xlabel("Longitude")

    plt.tight_layout()
    out1 = args.output_dir / "snodas_coverage_and_stats.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Saved: {out1}")

    # -----------------------------------------------------------------------
    # Figure 2: CV distributions — understand why consistency is universally low
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("CV Distributions (valid pixels only)", fontsize=13)

    ax = axes[0]
    ax.hist(inter_cv[valid], bins=100, color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(1.5, color="red", linestyle="--", label="threshold (1.5)")
    ax.set_xlabel("Interseason CV")
    ax.set_ylabel("Pixel count")
    ax.set_title("Year-to-year CV\n(% above threshold: {:.1f}%)".format(
        100 * (inter_cv[valid] >= 1.5).mean()))
    ax.legend()

    ax = axes[1]
    ax.hist(intra_cv[valid], bins=100, color="coral", edgecolor="none", alpha=0.8)
    ax.axvline(1.0, color="red", linestyle="--", label="threshold (1.0)")
    ax.set_xlabel("Mean Intraseason CV")
    ax.set_title("Within-winter CV\n(% above threshold: {:.1f}%)".format(
        100 * (intra_cv[valid] >= 1.0).mean()))
    ax.legend()

    ax = axes[2]
    ax.hist(consistency[valid], bins=100, color="seagreen", edgecolor="none", alpha=0.8)
    ax.set_xlabel("snow_consistency score")
    ax.set_title("Derived consistency score\n(1=consistent, 0=variable)")
    ax.axvline(consistency[valid].mean(), color="red", linestyle="--",
               label=f"mean={consistency[valid].mean():.3f}")
    ax.legend()

    plt.tight_layout()
    out2 = args.output_dir / "snodas_cv_distributions.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {out2}")

    # -----------------------------------------------------------------------
    # Figure 3: Coverage percentage histogram
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Snow Day Coverage Distribution (valid pixels)", fontsize=13)

    ax = axes[0]
    ax.hist(coverage[valid], bins=60, color="steelblue", edgecolor="none", alpha=0.8)
    ax.set_xlabel("Mean snow day ratio (0–1)")
    ax.set_ylabel("Pixel count")
    ax.set_title(f"Coverage distribution\n"
                 f"min={coverage[valid].min():.3f}, median={np.median(coverage[valid]):.3f}, "
                 f"max={coverage[valid].max():.3f}")

    ax = axes[1]
    # sqrt-transformed (as used in scoring)
    cov_transformed = np.sqrt(coverage[valid])
    ax.hist(cov_transformed, bins=60, color="darkorange", edgecolor="none", alpha=0.8)
    ax.set_xlabel("sqrt(coverage) — component score")
    ax.set_title(f"After sqrt transform\n"
                 f"min={cov_transformed.min():.3f}, median={np.median(cov_transformed):.3f}, "
                 f"max={cov_transformed.max():.3f}")

    plt.tight_layout()
    out3 = args.output_dir / "snodas_coverage_distribution.png"
    fig.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {out3}")

    print(f"\n✓ All diagnostic maps saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
