#!/usr/bin/env python3
"""
Diagnostic: per-lake boundary elevation and flow direction detail.

For each lake, shows:
- Lake cells with DEM elevation mapped to color
- Boundary cell elevations (the rim)
- Flow direction arrows at and around the boundary
- Spillway location (lowest rim point) vs HydroLAKES pour point
- Whether the outlet is interior vs boundary

Usage:
    python examples/diagnose_lake_boundaries.py
    python examples/diagnose_lake_boundaries.py --top 20
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib import cm
import matplotlib.patheffects as pe
from src.terrain.water_bodies import (
    rasterize_lakes_to_mask,
    identify_outlet_cells,
    find_lake_spillways,
)
from src.terrain.flow_accumulation import D8_OFFSETS, D8_DIRECTIONS


# Arrow offsets for D8 directions (dx, dy in plot coords = col, -row)
D8_ARROWS = {
    1: (1, 0),      # East
    2: (1, 1),      # NE (plot y is inverted)
    4: (0, 1),      # North
    8: (-1, 1),     # NW
    16: (-1, 0),    # West
    32: (-1, -1),   # SW
    64: (0, -1),    # South
    128: (1, -1),   # SE
}


def load_cached_data(output_dir: Path):
    """Load cached flow outputs and water body data."""
    data = {}

    with rasterio.open(output_dir / "flow_outputs" / "flow_direction.tif") as src:
        data["flow_dir"] = src.read(1)
        data["transform"] = src.transform
        data["shape"] = data["flow_dir"].shape

    with rasterio.open(output_dir / "flow_outputs" / "flow_accumulation_area.tif") as src:
        data["drainage"] = src.read(1)

    with rasterio.open(output_dir / "flow_outputs" / "dem_conditioned.tif") as src:
        data["dem"] = src.read(1)

    wb_dir = output_dir / "water_bodies"
    geojsons = list(wb_dir.glob("*.geojson"))
    if geojsons:
        with open(geojsons[0]) as f:
            data["lakes_geojson"] = json.load(f)

    return data


def rasterize_lakes_at_flow_resolution(data):
    """Rasterize lakes to match flow direction grid."""
    transform = data["transform"]
    shape = data["shape"]

    left = transform.c
    top = transform.f
    right = left + transform.a * shape[1]
    bottom = top + transform.e * shape[0]
    bbox = (bottom, left, top, right)

    resolution = abs(transform.a)
    lake_mask_raw, lake_transform = rasterize_lakes_to_mask(
        data["lakes_geojson"], bbox, resolution
    )

    if lake_mask_raw.shape != shape:
        from scipy.ndimage import zoom
        scale_y = shape[0] / lake_mask_raw.shape[0]
        scale_x = shape[1] / lake_mask_raw.shape[1]
        lake_mask = zoom(lake_mask_raw, (scale_y, scale_x), order=0)
    else:
        lake_mask = lake_mask_raw

    # Identify HydroLAKES pour points
    outlets_dict = {}
    for idx, feature in enumerate(data["lakes_geojson"]["features"], start=1):
        props = feature.get("properties", {})
        if "Pour_long" in props and "Pour_lat" in props:
            outlets_dict[idx] = (props["Pour_long"], props["Pour_lat"])

    outlet_mask = np.zeros(shape, dtype=bool)
    if outlets_dict:
        outlet_mask_raw = identify_outlet_cells(
            lake_mask_raw, outlets_dict, lake_transform
        )
        if outlet_mask_raw.shape != shape:
            from scipy.ndimage import zoom
            scale_y = shape[0] / outlet_mask_raw.shape[0]
            scale_x = shape[1] / outlet_mask_raw.shape[1]
            outlet_mask = zoom(
                outlet_mask_raw.astype(np.uint8), (scale_y, scale_x), order=0
            ).astype(bool)
        else:
            outlet_mask = outlet_mask_raw

    return lake_mask, outlet_mask


def get_lake_boundary_info(lake_mask, dem, flow_dir, lake_id):
    """Get boundary cells, their elevations, and neighbor info for a lake."""
    lake_cells = np.where(lake_mask == lake_id)
    if len(lake_cells[0]) == 0:
        return None

    boundary_cells = []
    interior_cells = []

    for r, c in zip(lake_cells[0], lake_cells[1]):
        is_boundary = False
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < lake_mask.shape[0] and 0 <= nc < lake_mask.shape[1]:
                    if lake_mask[nr, nc] != lake_id:
                        is_boundary = True
                        break
            if is_boundary:
                break

        if is_boundary:
            boundary_cells.append((r, c))
        else:
            interior_cells.append((r, c))

    # Find lowest rim neighbor for each boundary cell
    rim_info = []
    for r, c in boundary_cells:
        lowest_neighbor_elev = np.inf
        lowest_neighbor = None
        for (dr, dc), direction_code in D8_DIRECTIONS.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < dem.shape[0] and 0 <= nc < dem.shape[1]:
                if lake_mask[nr, nc] != lake_id:
                    if dem[nr, nc] < lowest_neighbor_elev:
                        lowest_neighbor_elev = dem[nr, nc]
                        lowest_neighbor = (nr, nc, direction_code)
        rim_info.append({
            "cell": (r, c),
            "elev": dem[r, c],
            "lowest_rim_elev": lowest_neighbor_elev if lowest_neighbor else None,
            "lowest_rim_cell": lowest_neighbor[:2] if lowest_neighbor else None,
        })

    return {
        "boundary": boundary_cells,
        "interior": interior_cells,
        "rim_info": rim_info,
        "lake_elev_mean": np.mean([dem[r, c] for r, c in zip(lake_cells[0], lake_cells[1])]),
        "lake_cells_count": len(lake_cells[0]),
    }


def plot_lake_detail(ax, lake_mask, dem, flow_dir, drainage, outlet_mask,
                     spillway, lake_id, pad=5):
    """Plot a single lake with boundary elevation, flow arrows, and outlets."""
    lake_cells = np.where(lake_mask == lake_id)
    if len(lake_cells[0]) == 0:
        return

    # Bounding box with padding
    r_min = max(0, lake_cells[0].min() - pad)
    r_max = min(lake_mask.shape[0], lake_cells[0].max() + pad + 1)
    c_min = max(0, lake_cells[1].min() - pad)
    c_max = min(lake_mask.shape[1], lake_cells[1].max() + pad + 1)

    # Crop arrays
    dem_crop = dem[r_min:r_max, c_min:c_max]
    lake_crop = lake_mask[r_min:r_max, c_min:c_max]
    flow_crop = flow_dir[r_min:r_max, c_min:c_max]
    drain_crop = drainage[r_min:r_max, c_min:c_max]
    out_crop = outlet_mask[r_min:r_max, c_min:c_max]

    rows, cols = dem_crop.shape

    # Base: DEM elevation
    im = ax.imshow(
        dem_crop, cmap="terrain", interpolation="nearest",
        extent=[c_min - 0.5, c_max - 0.5, r_max - 0.5, r_min - 0.5],
    )

    # Overlay: lake cells with semi-transparent blue
    lake_overlay = np.ma.masked_where(lake_crop != lake_id, lake_crop)
    ax.imshow(
        lake_overlay, cmap="Blues", alpha=0.4, interpolation="nearest",
        extent=[c_min - 0.5, c_max - 0.5, r_max - 0.5, r_min - 0.5],
    )

    # Mark boundary cells: color by their lowest rim neighbor elevation
    info = get_lake_boundary_info(lake_mask, dem, flow_dir, lake_id)
    if info and info["rim_info"]:
        rim_elevs = [
            ri["lowest_rim_elev"] for ri in info["rim_info"]
            if ri["lowest_rim_elev"] is not None and np.isfinite(ri["lowest_rim_elev"])
        ]
        if rim_elevs:
            vmin, vmax = min(rim_elevs), max(rim_elevs)
            if vmin == vmax:
                vmin -= 1
                vmax += 1
            norm = Normalize(vmin=vmin, vmax=vmax)
            cmap = cm.RdYlGn_r  # Green=low (good spillway), Red=high

            for ri in info["rim_info"]:
                r, c = ri["cell"]
                if ri["lowest_rim_elev"] is not None and np.isfinite(ri["lowest_rim_elev"]):
                    color = cmap(norm(ri["lowest_rim_elev"]))
                    ax.plot(c, r, 's', color=color, markersize=4, markeredgecolor='k',
                            markeredgewidth=0.3)

    # Elevation text on boundary cells (if lake is small enough)
    if info and len(info["boundary"]) <= 80:
        for ri in info["rim_info"]:
            r, c = ri["cell"]
            if ri["lowest_rim_elev"] is not None and np.isfinite(ri["lowest_rim_elev"]):
                ax.text(c, r + 0.35, f"{ri['lowest_rim_elev']:.0f}",
                        fontsize=3, ha='center', va='top', color='white',
                        fontweight='bold',
                        path_effects=[
                            pe.withStroke(
                                linewidth=0.8, foreground='black'
                            )
                        ])

    # Flow arrows for non-lake cells near boundary
    arrow_stride = max(1, min(rows, cols) // 20)
    for r in range(r_min, r_max, max(1, arrow_stride)):
        for c in range(c_min, c_max, max(1, arrow_stride)):
            if lake_mask[r, c] == lake_id:
                continue  # Skip lake interior
            d = flow_dir[r, c]
            if d != 0 and d in D8_ARROWS:
                dx, dy = D8_ARROWS[d]
                ax.annotate(
                    '', xy=(c + dx * 0.35, r - dy * 0.35),
                    xytext=(c, r),
                    arrowprops=dict(
                        arrowstyle='->', color='gray', lw=0.5,
                        mutation_scale=5,
                    ),
                )

    # Mark HydroLAKES pour point (outlet_mask)
    out_rows, out_cols = np.where(out_crop)
    for or_, oc_ in zip(out_rows, out_cols):
        abs_r, abs_c = or_ + r_min, oc_ + c_min
        # Check if this outlet is on the boundary or interior
        is_on_boundary = any(
            (r, c) == (abs_r, abs_c)
            for r, c in (info["boundary"] if info else [])
        )
        marker = 'o' if is_on_boundary else 'X'
        color = 'lime' if is_on_boundary else 'red'
        label = 'Pour pt (boundary)' if is_on_boundary else 'Pour pt (INTERIOR)'
        ax.plot(abs_c, abs_r, marker, color=color, markersize=8,
                markeredgecolor='k', markeredgewidth=1.0, zorder=10)
        ax.text(abs_c + 0.6, abs_r - 0.6, label, fontsize=4, color=color,
                fontweight='bold', zorder=10,
                path_effects=[
                    pe.withStroke(
                        linewidth=0.8, foreground='black'
                    )
                ])

    # Mark spillway
    if spillway is not None:
        sr, sc, s_dir = spillway
        ax.plot(sc, sr, 'D', color='yellow', markersize=8,
                markeredgecolor='k', markeredgewidth=1.0, zorder=10)
        # Arrow from spillway to downstream
        if s_dir in D8_ARROWS:
            sdx, sdy = D8_ARROWS[s_dir]
            ax.annotate(
                '', xy=(sc + sdx * 0.8, sr - sdy * 0.8),
                xytext=(sc, sr),
                arrowprops=dict(
                    arrowstyle='->', color='yellow', lw=2.0,
                    mutation_scale=12,
                ),
                zorder=11,
            )
        # Spillway rim elevation
        if s_dir in D8_OFFSETS:
            sdr, sdc = D8_OFFSETS[s_dir]
            snr, snc = sr + sdr, sc + sdc
            if 0 <= snr < dem.shape[0] and 0 <= snc < dem.shape[1]:
                ax.text(sc + 0.6, sr + 0.6,
                        f"Spill→{dem[snr, snc]:.1f}m",
                        fontsize=4, color='yellow', fontweight='bold',
                        path_effects=[
                            pe.withStroke(
                                linewidth=0.8, foreground='black'
                            )
                        ])

    # Title
    lake_area = info["lake_cells_count"] if info else 0
    lake_elev = info["lake_elev_mean"] if info else 0
    outlet_r, outlet_c = np.where(outlet_mask & (lake_mask == lake_id))
    is_connected = False
    if len(outlet_r) > 0:
        is_connected = flow_dir[outlet_r[0], outlet_c[0]] != 0

    status = "CONNECTED" if is_connected else "TERMINAL"
    ax.set_title(
        f"Lake {lake_id}  |  {lake_area} cells  |  elev≈{lake_elev:.0f}m  |  {status}",
        fontsize=7, fontweight='bold',
        color='green' if is_connected else 'red',
    )

    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, shrink=0.6, label='DEM (m)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="examples/output")
    parser.add_argument("--top", type=int, default=20,
                        help="Number of lakes to show (sorted by size)")
    parser.add_argument("--min-cells", type=int, default=50,
                        help="Minimum lake cells to include")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("Loading cached flow data...")
    data = load_cached_data(output_dir)
    print(f"  Grid: {data['shape']}")

    print("Rasterizing lakes...")
    lake_mask, outlet_mask = rasterize_lakes_at_flow_resolution(data)
    lake_ids = np.unique(lake_mask[lake_mask > 0])
    print(f"  {len(lake_ids)} lakes rasterized")

    print("Computing spillways...")
    spillways = find_lake_spillways(lake_mask, data["dem"])
    print(f"  {len(spillways)} spillways found")

    # Sort lakes by cell count, filter by min size
    lake_sizes = []
    for lid in lake_ids:
        count = int(np.sum(lake_mask == lid))
        if count >= args.min_cells:
            lake_sizes.append((lid, count))
    lake_sizes.sort(key=lambda x: -x[1])
    lake_sizes = lake_sizes[:args.top]

    print(f"  Plotting {len(lake_sizes)} lakes (>= {args.min_cells} cells)")

    # Layout: grid of subplots
    n = len(lake_sizes)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 5, nrows * 4.5),
        squeeze=False,
    )

    for idx, (lake_id, count) in enumerate(lake_sizes):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        spillway = spillways.get(lake_id)

        # Adaptive padding based on lake size
        pad = max(3, min(15, int(np.sqrt(count) * 0.5)))

        plot_lake_detail(
            ax, lake_mask, data["dem"], data["flow_dir"], data["drainage"],
            outlet_mask, spillway, lake_id, pad=pad,
        )

    # Hide empty axes
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='lime', edgecolor='k', label='Pour pt (boundary)'),
        mpatches.Patch(facecolor='red', edgecolor='k', label='Pour pt (INTERIOR)'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='yellow',
                    markeredgecolor='k', markersize=8, label='Spillway (lowest rim)'),
        mpatches.Patch(facecolor='green', edgecolor='k', alpha=0.6,
                       label='Low rim elev'),
        mpatches.Patch(facecolor='red', edgecolor='k', alpha=0.6,
                       label='High rim elev'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
               fontsize=8, framealpha=0.9)

    fig.suptitle(
        "Per-Lake Boundary Elevation & Outlet Analysis\n"
        "Squares = boundary cells colored by lowest rim neighbor elevation\n"
        "Yellow diamond = spillway (lowest rim point)  |  Circle/X = HydroLAKES pour point",
        fontsize=10, fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])

    out_path = output_dir / "diagnostic_lake_boundaries.png"
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"\nSaved: {out_path}")
    plt.close(fig)

    # Print summary table
    print(f"\n{'Lake':>6} {'Cells':>7} {'LakeElev':>9} {'SpillRim':>9} "
          f"{'PourPt':>10} {'Status':>10}")
    print("-" * 62)

    for lake_id, count in lake_sizes:
        info = get_lake_boundary_info(lake_mask, data["dem"], data["flow_dir"], lake_id)
        spillway = spillways.get(lake_id)

        spill_rim = "N/A"
        if spillway:
            sr, sc, s_dir = spillway
            if s_dir in D8_OFFSETS:
                sdr, sdc = D8_OFFSETS[s_dir]
                snr, snc = sr + sdr, sc + sdc
                if 0 <= snr < data["dem"].shape[0] and 0 <= snc < data["dem"].shape[1]:
                    spill_rim = f"{data['dem'][snr, snc]:.1f}"

        # Check if pour point is interior or boundary
        outlet_r, outlet_c = np.where(outlet_mask & (lake_mask == lake_id))
        pour_loc = "N/A"
        if len(outlet_r) > 0:
            is_boundary = any(
                (r, c) == (outlet_r[0], outlet_c[0])
                for r, c in (info["boundary"] if info else [])
            )
            pour_loc = "boundary" if is_boundary else "INTERIOR"

        is_connected = False
        if len(outlet_r) > 0:
            is_connected = data["flow_dir"][outlet_r[0], outlet_c[0]] != 0

        status = "CONNECTED" if is_connected else "TERMINAL"

        print(f"{lake_id:>6} {count:>7} {info['lake_elev_mean']:>9.1f} "
              f"{spill_rim:>9} {pour_loc:>10} {status:>10}")

    print("\nDone!")


if __name__ == "__main__":
    main()
