#!/usr/bin/env python3
"""
Diagnostic: visualize lake-stream connections in flow network.

Loads cached flow outputs and water body data to check whether
lake outlets are connected to downstream terrain (issue #44).

Shows:
- Lake boundaries and outlet locations
- Stream network (high drainage area cells)
- Flow direction at outlets: connected (arrow) vs terminal (X)
- Drainage area upstream vs downstream of each lake

Usage:
    python examples/diagnose_lake_connections.py
    python examples/diagnose_lake_connections.py --output-dir examples/output
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
from src.terrain.water_bodies import (
    rasterize_lakes_to_mask,
    identify_outlet_cells,
    compute_outlet_downstream_directions,
)
from src.terrain.flow_accumulation import D8_OFFSETS, D8_DIRECTIONS


def load_cached_data(output_dir: Path):
    """Load cached flow outputs and water body data."""
    flow_dir_path = output_dir / "flow_outputs" / "flow_direction.tif"
    drainage_path = output_dir / "flow_outputs" / "flow_accumulation_area.tif"
    rainfall_path = output_dir / "flow_outputs" / "flow_accumulation_rainfall.tif"
    dem_path = output_dir / "flow_outputs" / "dem_conditioned.tif"
    water_path = output_dir / "water_bodies" / "water_bodies_hydrolakes_36438f2a5ecc.geojson"

    data = {}

    # Flow direction
    with rasterio.open(flow_dir_path) as src:
        data["flow_dir"] = src.read(1)
        data["transform"] = src.transform
        data["shape"] = data["flow_dir"].shape

    # Drainage area
    with rasterio.open(drainage_path) as src:
        data["drainage"] = src.read(1)

    # DEM
    with rasterio.open(dem_path) as src:
        data["dem"] = src.read(1)

    # Upstream rainfall
    if rainfall_path.exists():
        with rasterio.open(rainfall_path) as src:
            data["rainfall"] = src.read(1)

    # Water bodies
    if water_path.exists():
        with open(water_path) as f:
            data["lakes_geojson"] = json.load(f)
    else:
        # Try to find any water body file
        wb_dir = output_dir / "water_bodies"
        geojsons = list(wb_dir.glob("*.geojson")) if wb_dir.exists() else []
        if geojsons:
            with open(geojsons[0]) as f:
                data["lakes_geojson"] = json.load(f)

    return data


def rasterize_lakes_at_flow_resolution(data):
    """Rasterize lakes to match flow direction grid resolution."""
    transform = data["transform"]
    shape = data["shape"]

    # Extract bbox from transform and shape
    left = transform.c
    top = transform.f
    right = left + transform.a * shape[1]
    bottom = top + transform.e * shape[0]
    bbox = (bottom, left, top, right)

    resolution = abs(transform.a)
    lake_mask_raw, lake_transform = rasterize_lakes_to_mask(
        data["lakes_geojson"], bbox, resolution
    )

    # Resample if needed
    if lake_mask_raw.shape != shape:
        from scipy.ndimage import zoom

        scale_y = shape[0] / lake_mask_raw.shape[0]
        scale_x = shape[1] / lake_mask_raw.shape[1]
        lake_mask = zoom(lake_mask_raw, (scale_y, scale_x), order=0)
    else:
        lake_mask = lake_mask_raw

    # Identify outlets from HydroLAKES pour points
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

    return lake_mask, outlet_mask, outlets_dict


def analyze_outlet_connections(flow_dir, lake_mask, outlet_mask, dem):
    """Check each outlet: is it connected or terminal?"""
    outlet_rows, outlet_cols = np.where(outlet_mask)
    results = []

    for r, c in zip(outlet_rows, outlet_cols):
        lake_id = lake_mask[r, c]
        direction = flow_dir[r, c]

        info = {
            "row": r,
            "col": c,
            "lake_id": lake_id,
            "flow_dir": direction,
            "connected": direction != 0,
            "downstream_cell": None,
            "downstream_drainage": None,
            "outlet_drainage": None,
        }

        if lake_id > 0:
            # Count lake cells
            info["lake_cells"] = int(np.sum(lake_mask == lake_id))

        info["outlet_drainage"] = float(
            data["drainage"][r, c] if "drainage" in data else 0
        )

        if direction != 0 and direction in D8_OFFSETS:
            dr, dc = D8_OFFSETS[direction]
            nr, nc = r + dr, c + dc
            if 0 <= nr < flow_dir.shape[0] and 0 <= nc < flow_dir.shape[1]:
                info["downstream_cell"] = (nr, nc)
                info["downstream_drainage"] = float(data["drainage"][nr, nc])

        results.append(info)

    return results


def plot_lake_stream_connections(data, lake_mask, outlet_mask, outlet_info, output_path):
    """Create diagnostic plot showing lake-stream connections."""
    flow_dir = data["flow_dir"]
    drainage = data["drainage"]
    dem = data["dem"]
    shape = data["shape"]

    # Stream threshold: top 1% of drainage area
    stream_threshold = np.percentile(drainage[drainage > 1], 99)
    streams = drainage >= stream_threshold

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # --- Panel 1: Overview with lakes and streams ---
    ax = axes[0]
    ax.set_title("Lake-Stream Connections (Full Domain)", fontsize=14)

    # Background: DEM
    dem_display = dem.copy()
    dem_display[dem_display <= 0] = np.nan
    ax.imshow(dem_display, cmap="terrain", alpha=0.4)

    # Overlay: drainage area (log scale)
    drainage_display = drainage.copy().astype(float)
    drainage_display[drainage_display <= 1] = np.nan
    ax.imshow(
        drainage_display,
        cmap="Blues",
        norm=LogNorm(vmin=100, vmax=drainage.max()),
        alpha=0.5,
    )

    # Lakes: filled
    lake_overlay = np.ma.masked_where(lake_mask == 0, lake_mask)
    ax.imshow(lake_overlay, cmap="cool", alpha=0.6)

    # Streams: bright lines
    stream_overlay = np.zeros((*shape, 4))
    stream_overlay[streams] = [0, 0.5, 1, 0.8]  # Blue
    ax.imshow(stream_overlay)

    # Outlet markers
    connected = [o for o in outlet_info if o["connected"]]
    terminal = [o for o in outlet_info if not o["connected"]]

    if connected:
        ax.scatter(
            [o["col"] for o in connected],
            [o["row"] for o in connected],
            c="lime",
            s=30,
            marker="v",
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
            label=f"Connected outlets ({len(connected)})",
        )

    if terminal:
        ax.scatter(
            [o["col"] for o in terminal],
            [o["row"] for o in terminal],
            c="red",
            s=30,
            marker="x",
            linewidths=1.5,
            zorder=5,
            label=f"Terminal outlets ({len(terminal)})",
        )

    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    # --- Panel 2: Zoomed view of largest lakes ---
    ax2 = axes[1]
    ax2.set_title("Outlet Detail (Largest Lakes)", fontsize=14)

    # Find the lake with most outlet drainage
    if outlet_info:
        sorted_outlets = sorted(
            outlet_info, key=lambda o: o.get("outlet_drainage", 0), reverse=True
        )
        # Pick the top outlet with an actual lake
        focus = None
        for o in sorted_outlets:
            if o.get("lake_id", 0) > 0:
                focus = o
                break
        if focus is None:
            focus = sorted_outlets[0]

        # Zoom window around the focus outlet
        margin = 200
        r, c = focus["row"], focus["col"]
        r_min = max(0, r - margin)
        r_max = min(shape[0], r + margin)
        c_min = max(0, c - margin)
        c_max = min(shape[1], c + margin)

        # DEM crop
        dem_crop = dem[r_min:r_max, c_min:c_max].copy().astype(float)
        dem_crop[dem_crop <= 0] = np.nan
        ax2.imshow(
            dem_crop,
            cmap="terrain",
            alpha=0.4,
            extent=[c_min, c_max, r_max, r_min],
        )

        # Drainage crop
        drain_crop = drainage[r_min:r_max, c_min:c_max].copy().astype(float)
        drain_crop[drain_crop <= 1] = np.nan
        ax2.imshow(
            drain_crop,
            cmap="Blues",
            norm=LogNorm(vmin=100, vmax=drainage.max()),
            alpha=0.5,
            extent=[c_min, c_max, r_max, r_min],
        )

        # Lakes crop
        lake_crop = lake_mask[r_min:r_max, c_min:c_max]
        lake_overlay2 = np.ma.masked_where(lake_crop == 0, lake_crop)
        ax2.imshow(
            lake_overlay2,
            cmap="cool",
            alpha=0.6,
            extent=[c_min, c_max, r_max, r_min],
        )

        # Draw flow arrows at outlets in this window
        arrow_scale = 8
        for o in outlet_info:
            or_, oc = o["row"], o["col"]
            if r_min <= or_ < r_max and c_min <= oc < c_max:
                if o["connected"] and o["flow_dir"] in D8_OFFSETS:
                    dr, dc = D8_OFFSETS[o["flow_dir"]]
                    ax2.annotate(
                        "",
                        xy=(oc + dc * arrow_scale, or_ + dr * arrow_scale),
                        xytext=(oc, or_),
                        arrowprops=dict(
                            arrowstyle="->",
                            color="lime",
                            lw=2.5,
                        ),
                        zorder=10,
                    )
                    # Label with drainage
                    drain_text = f"D={o['outlet_drainage']:.0f}"
                    if o["downstream_drainage"] is not None:
                        drain_text += f"→{o['downstream_drainage']:.0f}"
                    ax2.text(
                        oc + 5,
                        or_ - 5,
                        drain_text,
                        fontsize=7,
                        color="white",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7),
                        zorder=11,
                    )
                else:
                    ax2.plot(
                        oc,
                        or_,
                        "rx",
                        markersize=12,
                        markeredgewidth=2.5,
                        zorder=10,
                    )
                    drain_text = f"D={o['outlet_drainage']:.0f} TERMINAL"
                    ax2.text(
                        oc + 5,
                        or_ - 5,
                        drain_text,
                        fontsize=7,
                        color="red",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7),
                        zorder=11,
                    )

        ax2.set_xlim(c_min, c_max)
        ax2.set_ylim(r_max, r_min)

        lake_name = f"Lake {focus['lake_id']}"
        ax2.set_title(
            f"Outlet Detail: {lake_name} "
            f"(drainage={focus['outlet_drainage']:.0f})",
            fontsize=12,
        )
    else:
        ax2.text(0.5, 0.5, "No outlets found", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_before_after(data, lake_mask, outlet_mask, output_path):
    """Show drainage area before vs after applying outlet downstream routing."""
    from src.terrain.flow_accumulation import compute_drainage_area

    flow_dir_original = data["flow_dir"]
    dem = data["dem"]
    drainage_original = data["drainage"]

    # Apply outlet downstream routing
    basin_mask = np.zeros(flow_dir_original.shape, dtype=bool)
    flow_dir_fixed = compute_outlet_downstream_directions(
        flow_dir_original, lake_mask, outlet_mask, dem, basin_mask=basin_mask
    )

    # Count changes
    changed = flow_dir_fixed != flow_dir_original
    n_changed = np.sum(changed)
    print(f"\nOutlet routing changed {n_changed} cells")

    # Show which outlets changed
    changed_outlets = changed & outlet_mask
    print(f"  Of which {np.sum(changed_outlets)} are outlet cells")

    # Recompute drainage with fixed flow directions
    print("Recomputing drainage area with connected outlets...")
    drainage_fixed = compute_drainage_area(flow_dir_fixed)

    # Find outlets and compare drainage
    outlet_rows, outlet_cols = np.where(outlet_mask)
    print(f"\nOutlet drainage comparison ({len(outlet_rows)} outlets):")
    print(f"{'Outlet':>12} {'Lake':>6} {'Before':>12} {'After':>12} {'Downstream':>12} {'Status':>10}")
    print("-" * 70)
    for r, c in zip(outlet_rows, outlet_cols):
        lake_id = lake_mask[r, c]
        d_before = drainage_original[r, c]
        d_after = drainage_fixed[r, c]

        # Check downstream
        d_downstream = "N/A"
        status = "TERMINAL"
        fdir = flow_dir_fixed[r, c]
        if fdir != 0 and fdir in D8_OFFSETS:
            dr, dc = D8_OFFSETS[fdir]
            nr, nc = r + dr, c + dc
            if 0 <= nr < dem.shape[0] and 0 <= nc < dem.shape[1]:
                d_downstream = f"{drainage_fixed[nr, nc]:.0f}"
                status = "CONNECTED" if drainage_fixed[nr, nc] > d_after else "STALE?"
        print(
            f"({r:4d},{c:4d}) {lake_id:6d} {d_before:12.0f} {d_after:12.0f} {d_downstream:>12} {status:>10}"
        )

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Stream threshold
    thresh = np.percentile(drainage_original[drainage_original > 1], 99)

    for ax, drain, title in [
        (axes[0], drainage_original, "Before: Outlet Routing"),
        (axes[1], drainage_fixed, "After: Outlet Routing"),
    ]:
        d = drain.copy().astype(float)
        d[d <= 1] = np.nan
        ax.imshow(d, cmap="Blues", norm=LogNorm(vmin=100, vmax=drain.max()))

        # Lake overlay
        lake_overlay = np.ma.masked_where(lake_mask == 0, lake_mask)
        ax.imshow(lake_overlay, cmap="cool", alpha=0.4)

        # Outlets
        ax.scatter(outlet_cols, outlet_rows, c="red", s=15, marker="o", zorder=5)
        ax.set_title(title, fontsize=14)

    # Panel 3: difference
    ax = axes[2]
    diff = drainage_fixed.astype(float) - drainage_original.astype(float)
    # Only show where there's a meaningful difference
    diff_display = diff.copy()
    diff_display[np.abs(diff_display) < 1] = np.nan
    vmax = max(np.nanmax(np.abs(diff_display)), 1)
    im = ax.imshow(
        diff_display, cmap="RdBu_r", vmin=-vmax, vmax=vmax
    )
    lake_overlay = np.ma.masked_where(lake_mask == 0, lake_mask)
    ax.imshow(lake_overlay, cmap="cool", alpha=0.3)
    ax.scatter(outlet_cols, outlet_rows, c="black", s=15, marker="o", zorder=5)
    ax.set_title("Difference (After - Before)", fontsize=14)
    plt.colorbar(im, ax=ax, label="Drainage area change", shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_path}")
    plt.close()


def debug_outlet_rejections(data, lake_mask, outlet_mask):
    """For top-drainage outlets, trace each candidate neighbor and show why rejected."""
    from src.terrain.water_bodies import _trace_flows_to_lake

    flow_dir = data["flow_dir"]
    dem = data["dem"]
    drainage = data["drainage"]
    rows, cols = flow_dir.shape

    outlet_rows, outlet_cols = np.where(outlet_mask)

    # Sort outlets by drainage (largest first)
    outlet_drainages = [(drainage[r, c], r, c) for r, c in zip(outlet_rows, outlet_cols)]
    outlet_drainages.sort(reverse=True)

    # Show top 5 largest-drainage outlets
    for d_val, r, c in outlet_drainages[:5]:
        lake_id = lake_mask[r, c]
        print(f"\nOutlet ({r},{c}) lake={lake_id} drainage={d_val:.0f} "
              f"dem={dem[r, c]:.1f} flow_dir={flow_dir[r, c]}")

        # Check all 8 neighbors
        for (dr, dc), direction_code in D8_DIRECTIONS.items():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            in_lake = lake_mask[nr, nc] == lake_id and lake_id > 0
            if in_lake:
                continue

            elev_diff = dem[nr, nc] - dem[r, c]
            if dem[nr, nc] >= dem[r, c]:
                print(f"  Dir {direction_code:3d} → ({nr},{nc}): "
                      f"HIGHER (elev={dem[nr, nc]:.1f}, diff={elev_diff:+.1f})")
                continue

            # It's lower — trace to check for cycle
            cycle = _trace_flows_to_lake(
                flow_dir, lake_mask, nr, nc, lake_id, max_steps=100
            )

            # Find how many steps until re-entry (for debug)
            if cycle:
                steps_to_reentry = _count_steps_to_lake(
                    flow_dir, lake_mask, nr, nc, lake_id
                )
                print(f"  Dir {direction_code:3d} → ({nr},{nc}): "
                      f"LOWER (elev={dem[nr, nc]:.1f}, diff={elev_diff:+.1f}) "
                      f"→ CYCLE at step {steps_to_reentry}")
            else:
                print(f"  Dir {direction_code:3d} → ({nr},{nc}): "
                      f"LOWER (elev={dem[nr, nc]:.1f}, diff={elev_diff:+.1f}) "
                      f"→ SAFE ✓")


def plot_outlet_neighbor_detail(data, lake_mask, outlet_mask, output_path, n_outlets=6):
    """Tight-zoom plot of each top outlet showing every non-lake neighbor.

    For each of the top-N outlets (by drainage), draws a small grid view
    with the outlet highlighted and each neighbor color-coded:
      green  = SAFE (lower, no cycle — this is the chosen downstream)
      yellow = CYCLE (lower, but flow re-enters the lake)
      orange = HIGHER (elevation >= outlet)
      cyan   = IN-LAKE (same lake, skipped)
    Each neighbor is annotated with elevation difference.
    """
    from src.terrain.water_bodies import _trace_flows_to_lake

    flow_dir = data["flow_dir"]
    dem = data["dem"]
    drainage = data["drainage"]
    rows, cols = flow_dir.shape

    outlet_rows, outlet_cols = np.where(outlet_mask)
    if len(outlet_rows) == 0:
        print("No outlets to plot.")
        return

    # Sort outlets by drainage (largest first)
    outlet_list = [
        (drainage[r, c], r, c)
        for r, c in zip(outlet_rows, outlet_cols)
    ]
    outlet_list.sort(reverse=True)
    outlet_list = outlet_list[:n_outlets]

    n = len(outlet_list)
    ncols_fig = min(n, 3)
    nrows_fig = (n + ncols_fig - 1) // ncols_fig
    fig, axes = plt.subplots(
        nrows_fig, ncols_fig, figsize=(7 * ncols_fig, 7 * nrows_fig),
        squeeze=False,
    )

    # Small margin around outlet for the zoom
    margin = 8

    for idx, (d_val, r, c) in enumerate(outlet_list):
        ax = axes[idx // ncols_fig][idx % ncols_fig]
        lake_id = lake_mask[r, c]

        r_min = max(0, r - margin)
        r_max = min(rows, r + margin + 1)
        c_min = max(0, c - margin)
        c_max = min(cols, c + margin + 1)

        # DEM crop as background
        dem_crop = dem[r_min:r_max, c_min:c_max].copy().astype(float)
        ax.imshow(
            dem_crop, cmap="terrain", alpha=0.5,
            extent=[c_min - 0.5, c_max - 0.5, r_max - 0.5, r_min - 0.5],
        )

        # Lake overlay
        lake_crop = lake_mask[r_min:r_max, c_min:c_max]
        lake_vis = np.ma.masked_where(lake_crop == 0, lake_crop)
        ax.imshow(
            lake_vis, cmap="cool", alpha=0.5,
            extent=[c_min - 0.5, c_max - 0.5, r_max - 0.5, r_min - 0.5],
        )

        # Grid lines so individual cells are visible
        for row_i in range(r_min, r_max + 1):
            ax.axhline(row_i - 0.5, color="gray", lw=0.3, alpha=0.5)
        for col_i in range(c_min, c_max + 1):
            ax.axvline(col_i - 0.5, color="gray", lw=0.3, alpha=0.5)

        # Mark outlet prominently
        ax.plot(c, r, "s", color="white", markersize=16, markeredgecolor="black",
                markeredgewidth=2, zorder=20)
        ax.text(c, r, "OUT", fontsize=6, ha="center", va="center",
                fontweight="bold", color="black", zorder=21)

        # Annotate each of the 8 neighbors
        status_colors = {
            "SAFE": "lime",
            "CYCLE": "gold",
            "HIGHER": "orangered",
            "IN-LAKE": "deepskyblue",
            "EDGE": "gray",
        }

        chosen_nr, chosen_nc = None, None
        for (dr, dc), direction_code in D8_DIRECTIONS.items():
            nr, nc = r + dr, c + dc

            # Bounds check
            if not (0 <= nr < rows and 0 <= nc < cols):
                ax.plot(nr, nc, "s", color="gray", markersize=10, alpha=0.3,
                        markeredgecolor="black", markeredgewidth=0.5, zorder=10)
                continue

            in_lake = (lake_mask[nr, nc] == lake_id and lake_id > 0)

            if in_lake:
                status = "IN-LAKE"
                label = "lake"
            elif dem[nr, nc] >= dem[r, c]:
                status = "HIGHER"
                elev_diff = dem[nr, nc] - dem[r, c]
                label = f"+{elev_diff:.1f}m"
            else:
                # Lower non-lake neighbor — check for cycle
                elev_diff = dem[nr, nc] - dem[r, c]
                cycle = _trace_flows_to_lake(
                    flow_dir, lake_mask, nr, nc, lake_id, max_steps=100
                )
                if cycle:
                    steps = _count_steps_to_lake(flow_dir, lake_mask, nr, nc, lake_id)
                    status = "CYCLE"
                    label = f"{elev_diff:+.1f}m\ncycle@{steps}"
                else:
                    status = "SAFE"
                    label = f"{elev_diff:+.1f}m\nOK"
                    if chosen_nr is None:
                        chosen_nr, chosen_nc = nr, nc

            color = status_colors[status]
            ax.plot(nr, nc, "s", color=color, markersize=14, alpha=0.8,
                    markeredgecolor="black", markeredgewidth=1, zorder=10)
            ax.text(nr + 0.0, nc + 0.0, "", fontsize=1, zorder=11)  # placeholder
            # Put label below the marker
            ax.text(nc, nr + 0.45, label, fontsize=5.5, ha="center", va="top",
                    color="black", fontweight="bold", zorder=12,
                    bbox=dict(boxstyle="round,pad=0.15", fc=color, alpha=0.7,
                              edgecolor="none"))

        # Draw arrow from outlet to chosen neighbor
        if chosen_nr is not None:
            ax.annotate(
                "", xy=(chosen_nc, chosen_nr),
                xytext=(c, r),
                arrowprops=dict(arrowstyle="-|>", color="lime", lw=3,
                                mutation_scale=20),
                zorder=15,
            )

        fdir_code = flow_dir[r, c]
        fdir_str = f"dir={fdir_code}" if fdir_code != 0 else "TERMINAL"
        ax.set_title(
            f"Outlet ({r},{c})  lake={lake_id}\n"
            f"elev={dem[r, c]:.1f}m  drain={d_val:.0f}  {fdir_str}",
            fontsize=10,
        )
        ax.set_xlim(c_min - 0.5, c_max - 0.5)
        ax.set_ylim(r_max - 0.5, r_min - 0.5)
        ax.set_aspect("equal")

    # Hide unused subplots
    for idx in range(n, nrows_fig * ncols_fig):
        axes[idx // ncols_fig][idx % ncols_fig].set_visible(False)

    # Legend
    legend_handles = [
        mpatches.Patch(color="lime", label="SAFE (lower, no cycle)"),
        mpatches.Patch(color="gold", label="CYCLE (lower, re-enters lake)"),
        mpatches.Patch(color="orangered", label="HIGHER (elev ≥ outlet)"),
        mpatches.Patch(color="deepskyblue", label="IN-LAKE (same lake)"),
        mpatches.Patch(facecolor="white", edgecolor="black",
                       linewidth=2, label="Outlet cell"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center",
        ncol=5, fontsize=9, frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def _count_steps_to_lake(flow_dir, lake_mask, start_r, start_c, lake_id):
    """Count steps until flow path re-enters the lake."""
    rows, cols = flow_dir.shape
    r, c = start_r, start_c
    for step in range(200):
        d = flow_dir[r, c]
        if d == 0 or d not in D8_OFFSETS:
            return -1  # Terminal, no re-entry
        dr, dc = D8_OFFSETS[d]
        r, c = r + dr, c + dc
        if not (0 <= r < rows and 0 <= c < cols):
            return -1
        if lake_id > 0 and lake_mask[r, c] == lake_id:
            return step + 1
    return -1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Diagnose lake-stream connections in flow network"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/output"),
        help="Directory with cached flow outputs",
    )
    args = parser.parse_args()

    print("Loading cached flow data...")
    data = load_cached_data(args.output_dir)
    print(f"  Flow direction: {data['flow_dir'].shape}")
    print(f"  Drainage area range: [{data['drainage'].min():.0f}, {data['drainage'].max():.0f}]")
    print(f"  DEM range: [{data['dem'].min():.1f}, {data['dem'].max():.1f}]")

    if "lakes_geojson" not in data:
        print("ERROR: No water body GeoJSON found")
        sys.exit(1)

    n_features = len(data["lakes_geojson"].get("features", []))
    print(f"  Water bodies: {n_features} features")

    print("\nRasterizing lakes at flow resolution...")
    lake_mask, outlet_mask, outlets_dict = rasterize_lakes_at_flow_resolution(data)
    print(f"  Lake cells: {np.sum(lake_mask > 0):,}")
    print(f"  Unique lakes: {len(np.unique(lake_mask[lake_mask > 0]))}")
    print(f"  Outlet cells: {np.sum(outlet_mask):,}")

    print("\nAnalyzing outlet connections (current cached data)...")
    outlet_info = analyze_outlet_connections(
        data["flow_dir"], lake_mask, outlet_mask, data["dem"]
    )

    connected = sum(1 for o in outlet_info if o["connected"])
    terminal = sum(1 for o in outlet_info if not o["connected"])
    print(f"  Connected: {connected}, Terminal: {terminal}")

    # Plot 1: Current state
    plot_lake_stream_connections(
        data,
        lake_mask,
        outlet_mask,
        outlet_info,
        args.output_dir / "diagnostic_lake_connections.png",
    )

    # Plot 2: Before/after outlet routing
    plot_before_after(
        data,
        lake_mask,
        outlet_mask,
        args.output_dir / "diagnostic_lake_before_after.png",
    )

    # Plot 3: Per-outlet neighbor detail
    plot_outlet_neighbor_detail(
        data,
        lake_mask,
        outlet_mask,
        args.output_dir / "diagnostic_outlet_neighbors.png",
    )

    # Debug: why are outlets terminal? Trace candidates for top lakes
    print("\n" + "=" * 70)
    print("DEBUG: Why are large-drainage outlets still terminal?")
    print("=" * 70)
    debug_outlet_rejections(data, lake_mask, outlet_mask)

    print("\nDone! Check the diagnostic plots.")
