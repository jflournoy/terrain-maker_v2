#!/usr/bin/env python3
"""Plot WorldClim 30-second (~1km) rainfall data for San Diego region."""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import rasterio

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.terrain.precipitation_downloader import download_precipitation


def main():
    # San Diego DEM bbox (extends into Mexico)
    bbox = (31.5, -117.5, 33.5, -115.5)  # (min_lat, min_lon, max_lat, max_lon)

    output_dir = Path("data/worldclim_30s")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading WorldClim 30-second precipitation data...")
    print(f"  Bbox: {bbox}")
    print(f"  Resolution: ~1km (30 arc-seconds)")

    # Download using worldclim_30s dataset
    precip_path = download_precipitation(
        bbox=bbox,
        output_dir=str(output_dir),
        dataset="worldclim_30s",
        use_real_data=True,
    )

    print(f"\nLoading data from {precip_path}...")

    with rasterio.open(precip_path) as src:
        precip = src.read(1).astype(np.float32)
        bounds = src.bounds
        transform = src.transform

    print(f"  Shape: {precip.shape}")
    print(f"  Resolution: {abs(transform.a):.6f}° ({abs(transform.a) * 111:.1f} km)")

    # Mask invalid values
    precip = np.where((precip >= 0) & (precip < 20000), precip, np.nan)

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 10))

    im = ax.imshow(
        precip,
        cmap="cividis",
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        aspect="auto",
        interpolation="bilinear",
        origin="upper",
    )

    ax.set_title("WorldClim 30s Annual Precipitation - San Diego Region\n(~1km resolution, extends into Mexico)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude (°E)", fontsize=12)
    ax.set_ylabel("Latitude (°N)", fontsize=12)

    cbar = plt.colorbar(im, ax=ax, label="Annual Precipitation (mm/year)", shrink=0.8)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # Add US-Mexico border line (approximate)
    border_lat = 32.5337
    ax.axhline(y=border_lat, color='red', linestyle='--', linewidth=2, alpha=0.7, label='US-Mexico Border')
    ax.legend(loc='upper right', fontsize=10)

    # Add stats
    valid = precip[~np.isnan(precip)]
    stats_text = (f"Data points: {valid.size:,}  |  "
                  f"Min: {valid.min():.0f} mm  |  "
                  f"Mean: {valid.mean():.0f} mm  |  "
                  f"Max: {valid.max():.0f} mm")
    fig.text(0.5, 0.02, stats_text, ha="center", fontsize=10)

    plt.tight_layout()

    output_path = Path("examples/output") / "san_diego_rainfall_30s.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved plot to {output_path}")
    plt.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
