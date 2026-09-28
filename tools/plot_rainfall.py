#!/usr/bin/env python3
"""Quick plot of PRISM rainfall data with proper geographic projection."""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import rasterio

def main():
    output_dir = Path("examples/output/flow-validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find precipitation file (WorldClim or PRISM)
    output_path = Path("examples/output")
    precip_files = list(output_path.glob("worldclim_annual_precip*.tif"))
    if not precip_files:
        precip_files = list(output_path.glob("prism_ppt_us_25m*.tif"))
    if not precip_files:
        precip_files = list(output_path.glob("prism_annual_precip*.tif"))

    if not precip_files:
        print("No precipitation data found in examples/output/")
        print("Run the San Diego flow demo first to download data.")
        return 1

    precip_path = precip_files[0]
    print(f"Loading rainfall data from {precip_path}...")

    with rasterio.open(precip_path) as src:
        precip = src.read(1).astype(np.float32)
        bounds = src.bounds

    # Mask nodata (negative values)
    precip = np.where(precip < 0, np.nan, precip)

    # Display in equirectangular (WGS84 native) with 'auto' aspect
    # This stretches to fill the figure without distorting the data itself
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot with geographic extent
    im = ax.imshow(
        precip,
        cmap="cividis",
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        aspect="auto",
        interpolation="bilinear",
        origin="upper",
    )

    ax.set_title("PRISM Annual Precipitation (30-year average)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude (WGS84)")
    ax.set_ylabel("Latitude (WGS84)")

    cbar = plt.colorbar(im, ax=ax, label="Precipitation (mm/year)")

    # Add grid for reference
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # Add stats
    valid = precip[~np.isnan(precip)]
    stats_text = f"Min: {valid.min():.0f} mm  |  Mean: {valid.mean():.0f} mm  |  Max: {valid.max():.0f} mm"
    fig.text(0.5, 0.02, stats_text, ha="center", fontsize=10)

    plt.tight_layout()

    output_path = output_dir / "rainfall_data.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved plot to {output_path}")
    plt.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
