#!/usr/bin/env python3
"""
Diagnostic: compare drainage/discharge with HydroLAKES outlets vs spillway outlets.

Loads cached flow data, then runs two scenarios:
1. BASELINE: HydroLAKES pour points as outlets (current behavior — mostly interior)
2. SPILLWAY: Snap outlets to spillway cells (always on lake boundary)

Shows before/after drainage area and upstream rainfall for both scenarios.

Usage:
    python examples/diagnose_spillway_drainage.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm, Normalize
from src.terrain.water_bodies import (
    rasterize_lakes_to_mask,
    identify_outlet_cells,
    find_lake_spillways,
    create_lake_flow_routing,
    compute_outlet_downstream_directions,
)
from src.terrain.flow_accumulation import (
    D8_OFFSETS,
    D8_DIRECTIONS,
    compute_drainage_area,
    compute_upstream_rainfall,
)


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
    rainfall_path = output_dir / "flow_outputs" / "flow_accumulation_rainfall.tif"
    if rainfall_path.exists():
        with rasterio.open(rainfall_path) as src:
            data["rainfall"] = src.read(1)
    wb_dir = output_dir / "water_bodies"
    geojsons = list(wb_dir.glob("*.geojson"))
    if geojsons:
        with open(geojsons[0]) as f:
            data["lakes_geojson"] = json.load(f)
    return data


def rasterize_lakes(data):
    """Rasterize lakes at flow grid resolution. Return mask and raw outlet info."""
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

    # HydroLAKES pour points
    outlets_dict = {}
    for idx, feature in enumerate(data["lakes_geojson"]["features"], start=1):
        props = feature.get("properties", {})
        if "Pour_long" in props and "Pour_lat" in props:
            outlets_dict[idx] = (props["Pour_long"], props["Pour_lat"])

    pour_outlet_mask = np.zeros(shape, dtype=bool)
    if outlets_dict:
        outlet_mask_raw = identify_outlet_cells(
            lake_mask_raw, outlets_dict, lake_transform
        )
        if outlet_mask_raw.shape != shape:
            from scipy.ndimage import zoom
            scale_y = shape[0] / outlet_mask_raw.shape[0]
            scale_x = shape[1] / outlet_mask_raw.shape[1]
            pour_outlet_mask = zoom(
                outlet_mask_raw.astype(np.uint8), (scale_y, scale_x), order=0
            ).astype(bool)
        else:
            pour_outlet_mask = outlet_mask_raw

    return lake_mask, pour_outlet_mask


def build_flow_with_outlets(base_flow_dir, lake_mask, outlet_mask, dem):
    """Build complete flow direction grid with lake routing + outlet connections."""
    flow_dir = base_flow_dir.copy()

    # BFS routing inside lakes toward outlets
    lake_flow = create_lake_flow_routing(lake_mask, outlet_mask, dem)
    flow_dir[lake_mask > 0] = lake_flow[lake_mask > 0]

    # Compute spillways for fallback
    spillways = find_lake_spillways(lake_mask, dem)

    # Connect outlets to downstream terrain
    flow_dir = compute_outlet_downstream_directions(
        flow_dir, lake_mask, outlet_mask, dem,
        basin_mask=None, spillways=spillways,
    )

    return flow_dir, spillways


def make_spillway_outlet_mask(lake_mask, spillways):
    """Create outlet mask using spillway cells instead of HydroLAKES pour points."""
    outlet_mask = np.zeros(lake_mask.shape, dtype=bool)
    for lake_id, (sr, sc, _direction) in spillways.items():
        outlet_mask[sr, sc] = True
    return outlet_mask


def count_connected(flow_dir, outlet_mask):
    """Count how many outlet cells have a non-zero flow direction."""
    outlet_rows, outlet_cols = np.where(outlet_mask)
    connected = 0
    terminal = 0
    for r, c in zip(outlet_rows, outlet_cols):
        if flow_dir[r, c] != 0:
            connected += 1
        else:
            terminal += 1
    return connected, terminal


def main():
    output_dir = Path("examples/output")

    print("=" * 70)
    print("DRAINAGE COMPARISON: HydroLAKES Pour Points vs Spillway Outlets")
    print("=" * 70)

    print("\nLoading cached flow data...")
    data = load_cached_data(output_dir)
    dem = data["dem"]
    base_flow_dir = data["flow_dir"]  # Terrain-only flow directions
    print(f"  Grid: {data['shape']}")

    print("Rasterizing lakes...")
    lake_mask, pour_outlet_mask = rasterize_lakes(data)
    lake_ids = np.unique(lake_mask[lake_mask > 0])
    print(f"  {len(lake_ids)} lakes")

    # ── Scenario 1: HydroLAKES Pour Points (baseline) ──────────────
    print("\n--- Scenario 1: HydroLAKES Pour Points (current) ---")
    flow_pour, spillways_pour = build_flow_with_outlets(
        base_flow_dir, lake_mask, pour_outlet_mask, dem
    )
    conn_pour, term_pour = count_connected(flow_pour, pour_outlet_mask)
    print(f"  Connected: {conn_pour}, Terminal: {term_pour}")

    print("  Computing drainage area...")
    drainage_pour = compute_drainage_area(flow_pour)
    print(f"  Max drainage: {drainage_pour.max():,.0f}")

    rainfall_pour = None
    if "rainfall" in data:
        # We need precipitation grid to recompute rainfall accumulation.
        # We don't have the raw precip grid cached separately, but we can
        # use a uniform proxy (1.0 everywhere) to show relative differences.
        print("  Computing upstream rainfall (uniform precip proxy)...")
        precip_proxy = np.ones(data["shape"], dtype=np.float32)
        rainfall_pour = compute_upstream_rainfall(flow_pour, precip_proxy)
        print(f"  Max upstream rainfall: {rainfall_pour.max():,.0f}")

    # ── Scenario 2: Spillway Outlets ───────────────────────────────
    print("\n--- Scenario 2: Spillway Outlets (proposed) ---")
    spillways_all = find_lake_spillways(lake_mask, dem)
    spill_outlet_mask = make_spillway_outlet_mask(lake_mask, spillways_all)
    print(f"  {np.sum(spill_outlet_mask)} spillway outlets placed")

    flow_spill, _ = build_flow_with_outlets(
        base_flow_dir, lake_mask, spill_outlet_mask, dem
    )
    conn_spill, term_spill = count_connected(flow_spill, spill_outlet_mask)
    print(f"  Connected: {conn_spill}, Terminal: {term_spill}")

    print("  Computing drainage area...")
    drainage_spill = compute_drainage_area(flow_spill)
    print(f"  Max drainage: {drainage_spill.max():,.0f}")

    rainfall_spill = None
    if rainfall_pour is not None:
        print("  Computing upstream rainfall (uniform precip proxy)...")
        rainfall_spill = compute_upstream_rainfall(flow_spill, precip_proxy)
        print(f"  Max upstream rainfall: {rainfall_spill.max():,.0f}")

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'':>20} {'Pour Points':>14} {'Spillways':>14} {'Change':>10}")
    print(f"  {'Connected outlets':>20} {conn_pour:>14} {conn_spill:>14} {conn_spill - conn_pour:>+10}")
    print(f"  {'Terminal outlets':>20} {term_pour:>14} {term_spill:>14} {term_spill - term_pour:>+10}")
    print(f"  {'Max drainage':>20} {drainage_pour.max():>14,.0f} {drainage_spill.max():>14,.0f}")
    if rainfall_pour is not None:
        print(f"  {'Max rainfall accum':>20} {rainfall_pour.max():>14,.0f} {rainfall_spill.max():>14,.0f}")

    # Cells where drainage increased
    drainage_diff = drainage_spill.astype(np.float64) - drainage_pour.astype(np.float64)
    increased = np.sum(drainage_diff > 0)
    decreased = np.sum(drainage_diff < 0)
    print(f"\n  Cells with increased drainage: {increased:,}")
    print(f"  Cells with decreased drainage: {decreased:,}")

    # ── Per-lake downstream comparison ───────────────────────────────
    # For each lake, find a "probe" cell ~20 steps downstream of the
    # spillway outlet in the spillway flow network, then compare drainage
    # at that same cell under both scenarios.

    def trace_downstream(flow_dir, start_r, start_c, steps=20):
        """Follow flow path for N steps, return final cell."""
        r, c = start_r, start_c
        for _ in range(steps):
            d = flow_dir[r, c]
            if d == 0 or d not in D8_OFFSETS:
                break
            dr, dc = D8_OFFSETS[d]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < flow_dir.shape[0] and 0 <= nc < flow_dir.shape[1]):
                break
            r, c = nr, nc
        return r, c

    print(f"\n{'Lake':>6} {'Cells':>7} {'SpillConn':>10} "
          f"{'DnPour':>10} {'DnSpill':>11} {'DnGain':>10} {'DnGain%':>8}")
    print("-" * 74)

    lake_stats = []
    for lid in lake_ids:
        count = int(np.sum(lake_mask == lid))
        if count < 20:
            continue

        spill_info = spillways_all.get(int(lid))
        if not spill_info:
            continue

        sr, sc, s_dir = spill_info
        spill_conn = flow_spill[sr, sc] != 0

        # Trace ~20 steps downstream from spillway in the spillway flow network
        probe_r, probe_c = trace_downstream(flow_spill, sr, sc, steps=20)

        # Skip if probe ended up inside another lake or at the spillway itself
        if (probe_r, probe_c) == (sr, sc):
            probe_r, probe_c = trace_downstream(flow_spill, sr, sc, steps=5)

        # Drainage at probe cell under both scenarios
        dn_pour = int(drainage_pour[probe_r, probe_c])
        dn_spill = int(drainage_spill[probe_r, probe_c])
        dn_gain = dn_spill - dn_pour
        dn_pct = (dn_gain / max(dn_pour, 1)) * 100

        lake_stats.append((
            lid, count, spill_conn,
            dn_pour, dn_spill, dn_gain, dn_pct,
            sr, sc, probe_r, probe_c,
        ))

    lake_stats.sort(key=lambda x: -x[5])  # Sort by downstream gain

    for lid, count, sc, dp, ds, dg, dpct, *_ in lake_stats:
        sc_str = "YES" if sc else "no"
        gain_str = f"+{dg:,}" if dg > 0 else f"{dg:,}"
        pct_str = f"+{dpct:.0f}%" if dpct > 0 else f"{dpct:.0f}%"
        print(f"{lid:>6} {count:>7} {sc_str:>10} "
              f"{dp:>10,} {ds:>11,} {gain_str:>10} {pct_str:>8}")

    # ── Plots ──────────────────────────────────────────────────────
    print("\nGenerating comparison plots...")

    # Plot 1: Side-by-side drainage area (log scale)
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    vmin_drain = 1
    vmax_drain = max(drainage_pour.max(), drainage_spill.max())
    drain_norm = LogNorm(vmin=vmin_drain, vmax=vmax_drain)

    im0 = axes[0].imshow(drainage_pour, norm=drain_norm, cmap='viridis')
    axes[0].set_title(f"Pour Point Outlets\n{conn_pour} connected / {term_pour} terminal",
                      fontsize=11, fontweight='bold')
    plt.colorbar(im0, ax=axes[0], shrink=0.7, label='Drainage Area (cells)')

    im1 = axes[1].imshow(drainage_spill, norm=drain_norm, cmap='viridis')
    axes[1].set_title(f"Spillway Outlets\n{conn_spill} connected / {term_spill} terminal",
                      fontsize=11, fontweight='bold')
    plt.colorbar(im1, ax=axes[1], shrink=0.7, label='Drainage Area (cells)')

    # Difference: log of ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(
            drainage_pour > 0,
            drainage_spill.astype(np.float64) / drainage_pour.astype(np.float64),
            1.0,
        )
        log_ratio = np.log10(np.clip(ratio, 0.01, 100.0))

    im2 = axes[2].imshow(log_ratio, cmap='RdBu', vmin=-2, vmax=2)
    axes[2].set_title("log₁₀(Spillway / PourPoint)\nBlue = spillway has more drainage",
                      fontsize=11, fontweight='bold')
    plt.colorbar(im2, ax=axes[2], shrink=0.7, label='log₁₀ ratio')

    # Mark lake outlines on all panels
    lake_boundary = np.zeros_like(lake_mask, dtype=bool)
    from scipy.ndimage import binary_dilation
    lake_binary = lake_mask > 0
    dilated = binary_dilation(lake_binary)
    lake_boundary = dilated & ~lake_binary
    for ax in axes:
        ax.contour(lake_boundary, levels=[0.5], colors='cyan', linewidths=0.3)

    fig.suptitle("Drainage Area: HydroLAKES Pour Points vs Spillway Outlets",
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    path1 = output_dir / "diagnostic_spillway_drainage.png"
    fig.savefig(path1, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path1}")
    plt.close(fig)

    # Plot 2: Upstream rainfall comparison (if available)
    if rainfall_pour is not None and rainfall_spill is not None:
        fig2, axes2 = plt.subplots(1, 3, figsize=(20, 7))

        vmax_rain = max(rainfall_pour.max(), rainfall_spill.max())
        rain_norm = LogNorm(vmin=1, vmax=vmax_rain)

        im0 = axes2[0].imshow(rainfall_pour, norm=rain_norm, cmap='plasma')
        axes2[0].set_title(f"Pour Point Outlets\nUpstream Rainfall Accumulation",
                           fontsize=11, fontweight='bold')
        plt.colorbar(im0, ax=axes2[0], shrink=0.7, label='Accumulated Rainfall')

        im1 = axes2[1].imshow(rainfall_spill, norm=rain_norm, cmap='plasma')
        axes2[1].set_title(f"Spillway Outlets\nUpstream Rainfall Accumulation",
                           fontsize=11, fontweight='bold')
        plt.colorbar(im1, ax=axes2[1], shrink=0.7, label='Accumulated Rainfall')

        with np.errstate(divide='ignore', invalid='ignore'):
            rain_ratio = np.where(
                rainfall_pour > 0,
                rainfall_spill / rainfall_pour,
                1.0,
            )
            rain_log_ratio = np.log10(np.clip(rain_ratio, 0.01, 100.0))

        im2 = axes2[2].imshow(rain_log_ratio, cmap='RdBu', vmin=-2, vmax=2)
        axes2[2].set_title("log₁₀(Spillway / PourPoint)\nBlue = spillway has more",
                           fontsize=11, fontweight='bold')
        plt.colorbar(im2, ax=axes2[2], shrink=0.7, label='log₁₀ ratio')

        for ax in axes2:
            ax.contour(lake_boundary, levels=[0.5], colors='cyan', linewidths=0.3)

        fig2.suptitle("Upstream Rainfall: HydroLAKES Pour Points vs Spillway Outlets",
                      fontsize=14, fontweight='bold')
        fig2.tight_layout(rect=[0, 0, 1, 0.95])

        path2 = output_dir / "diagnostic_spillway_rainfall.png"
        fig2.savefig(path2, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path2}")
        plt.close(fig2)

    # Plot 3: Per-lake side-by-side — pour-point vs spillway drainage
    # Pick lakes with biggest downstream drainage gain
    top_gains = sorted(lake_stats, key=lambda x: -x[5])[:6]

    if top_gains:
        fig3, axes3 = plt.subplots(3, 4, figsize=(22, 16))

        for idx, (lid, count, sc, dp, ds, dg, dpct,
                  sr, sc_col, probe_r, probe_c) in enumerate(top_gains):
            ax_pour = axes3[idx // 2 * 3 // 3 + idx % 2 if False else idx, 0] \
                if False else axes3[idx // 2, (idx % 2) * 2]
            # Simpler: row = idx//2, left_col = (idx%2)*2
            row_idx = idx // 2 if idx < 6 else 0
            col_offset = (idx % 2) * 2
            ax_pour = axes3[row_idx, col_offset]
            ax_spill = axes3[row_idx, col_offset + 1]

            lake_cells = np.where(lake_mask == lid)
            # Pad enough to include the probe point
            all_rows = list(lake_cells[0]) + [probe_r]
            all_cols = list(lake_cells[1]) + [probe_c]
            pad = max(10, int(np.sqrt(count) * 0.5))
            r_min = max(0, min(all_rows) - pad)
            r_max = min(lake_mask.shape[0], max(all_rows) + pad + 1)
            c_min = max(0, min(all_cols) - pad)
            c_max = min(lake_mask.shape[1], max(all_cols) + pad + 1)

            drain_crop_pour = drainage_pour[r_min:r_max, c_min:c_max].astype(float)
            drain_crop_spill = drainage_spill[r_min:r_max, c_min:c_max].astype(float)

            vmax_local = max(drain_crop_pour.max(), drain_crop_spill.max(), 10)
            shared_norm = LogNorm(vmin=1, vmax=vmax_local)

            # Lake boundary contour data
            lake_crop = lake_mask[r_min:r_max, c_min:c_max]
            lake_edge = np.zeros_like(lake_crop, dtype=bool)
            for r in range(lake_crop.shape[0]):
                for c in range(lake_crop.shape[1]):
                    if lake_crop[r, c] == lid:
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                nr, nc = r + dr, c + dc
                                if (0 <= nr < lake_crop.shape[0] and
                                        0 <= nc < lake_crop.shape[1] and
                                        lake_crop[nr, nc] != lid):
                                    lake_edge[r, c] = True
            edge_y = np.arange(r_min, r_max)
            edge_x = np.arange(c_min, c_max)
            EX, EY = np.meshgrid(edge_x, edge_y)

            extent = [c_min - 0.5, c_max - 0.5, r_max - 0.5, r_min - 0.5]

            # Get marker positions
            pr_, pc_ = np.where(pour_outlet_mask & (lake_mask == lid))
            has_pour = len(pr_) > 0

            # Helper: draw both markers + probe on an axis
            def annotate_panel(ax, drain_value, is_spill_panel):
                # Both markers on both panels for comparison
                if has_pour:
                    ax.plot(pc_[0], pr_[0], 'X', color='red', markersize=10,
                            markeredgecolor='white', markeredgewidth=1, zorder=10)
                ax.plot(sc_col, sr, 'D', color='yellow', markersize=12,
                        markeredgecolor='k', markeredgewidth=1.5, zorder=10)
                # Probe point
                ax.plot(probe_c, probe_r, 'o', color='magenta', markersize=10,
                        markeredgecolor='white', markeredgewidth=1.5, zorder=10)
                ax.annotate(
                    f'{drain_value:,}', (probe_c, probe_r),
                    xytext=(8, -8), textcoords='offset points',
                    fontsize=7, color='magenta', fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=1.5, foreground='black')],
                )

            # LEFT: Pour-point drainage
            im_p = ax_pour.imshow(
                drain_crop_pour, norm=shared_norm, cmap='viridis', extent=extent,
            )
            ax_pour.contour(EX, EY, lake_edge, levels=[0.5],
                            colors='cyan', linewidths=0.8)
            annotate_panel(ax_pour, dp, is_spill_panel=False)

            ax_pour.set_title(
                f"Lake {lid} — Pour Points\nDownstream probe: {dp:,}",
                fontsize=8, fontweight='bold', color='red',
            )
            plt.colorbar(im_p, ax=ax_pour, shrink=0.6, label='Drainage')

            # RIGHT: Spillway drainage
            im_s = ax_spill.imshow(
                drain_crop_spill, norm=shared_norm, cmap='viridis', extent=extent,
            )
            ax_spill.contour(EX, EY, lake_edge, levels=[0.5],
                             colors='cyan', linewidths=0.8)
            annotate_panel(ax_spill, ds, is_spill_panel=True)

            gain_str = f"+{dg:,}" if dg > 0 else f"{dg:,}"
            ax_spill.set_title(
                f"Lake {lid} — Spillway Outlets\n"
                f"Downstream probe: {ds:,} ({gain_str})",
                fontsize=8, fontweight='bold', color='green',
            )
            plt.colorbar(im_s, ax=ax_spill, shrink=0.6, label='Drainage')

        fig3.suptitle(
            "Downstream Drainage: Pour Points (left) vs Spillways (right)\n"
            "Red X = pour point  |  Yellow ◆ = spillway  |  "
            "Magenta ● = downstream probe (same cell, both scenarios)",
            fontsize=11, fontweight='bold',
        )
        fig3.tight_layout(rect=[0, 0, 1, 0.93])

        path3 = output_dir / "diagnostic_spillway_top_gains.png"
        fig3.savefig(path3, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path3}")
        plt.close(fig3)

    print("\nDone!")


if __name__ == "__main__":
    main()
