"""
Edge Extrusion Debug Visualizations

Diagnostic plots for understanding rectangle edge sampling and coordinate transformations.
Helps visualize:
- Original rectangle sampling in DEM pixel space
- Coordinate transformations (original → geographic → reprojected → final mesh)
- Boundary point distributions before/after deduplication/sorting
- Edge distribution analysis (north/south/east/west)
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def plot_rectangle_edge_sampling(
    dem_shape: Tuple[int, int],
    edge_pixels_dense: List[Tuple[float, float]],
    edge_pixels_sparse: List[Tuple[float, float]],
    output_path: Path,
) -> None:
    """
    Plot rectangle edge sampling at different spacings.

    Shows the difference between dense (1px spacing) and sparse (5px spacing)
    edge sampling in the original DEM pixel coordinate space.

    Args:
        dem_shape: (height, width) of DEM
        edge_pixels_dense: Edge pixels sampled at 1px spacing
        edge_pixels_sparse: Edge pixels sampled at 5px spacing
        output_path: Where to save the plot
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    height, width = dem_shape

    # Plot 1: Dense sampling
    if edge_pixels_dense:
        dense_array = np.array(edge_pixels_dense)
        ax1.scatter(dense_array[:, 1], dense_array[:, 0], s=2, alpha=0.5, label="Edge pixels")
        ax1.set_xlim(-5, width + 5)
        ax1.set_ylim(height + 5, -5)  # Flip Y axis to match image coords
        ax1.set_aspect("equal")
        ax1.set_title(f"Dense Edge Sampling\n({len(edge_pixels_dense)} points)")
        ax1.set_xlabel("X (pixels)")
        ax1.set_ylabel("Y (pixels)")
        ax1.grid(True, alpha=0.3)

    # Plot 2: Sparse sampling
    if edge_pixels_sparse:
        sparse_array = np.array(edge_pixels_sparse)
        ax2.scatter(sparse_array[:, 1], sparse_array[:, 0], s=20, alpha=0.7, label="Edge pixels")
        ax2.set_xlim(-5, width + 5)
        ax2.set_ylim(height + 5, -5)
        ax2.set_aspect("equal")
        ax2.set_title(f"Sparse Edge Sampling\n({len(edge_pixels_sparse)} points)")
        ax2.set_xlabel("X (pixels)")
        ax2.set_ylabel("Y (pixels)")
        ax2.grid(True, alpha=0.3)

    # Plot 3: Overlay both
    if edge_pixels_dense:
        dense_array = np.array(edge_pixels_dense)
        ax3.scatter(
            dense_array[:, 1], dense_array[:, 0], s=1, alpha=0.3, label="Dense (1px)"
        )
    if edge_pixels_sparse:
        sparse_array = np.array(edge_pixels_sparse)
        ax3.scatter(
            sparse_array[:, 1], sparse_array[:, 0], s=15, alpha=0.7, label="Sparse (5px)"
        )
    ax3.set_xlim(-5, width + 5)
    ax3.set_ylim(height + 5, -5)
    ax3.set_aspect("equal")
    ax3.set_title("Comparison: Dense vs Sparse")
    ax3.set_xlabel("X (pixels)")
    ax3.set_ylabel("Y (pixels)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.suptitle(f"Rectangle Edge Sampling - DEM Space ({height}×{width})", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved rectangle edge sampling plot to {output_path}")


def plot_edge_distribution(
    boundary_points: List[Tuple[float, float]],
    output_path: Path,
    title: str = "Edge Distribution",
    margin: float = 0.05,
) -> None:
    """
    Plot boundary points colored by which edge they're on.

    Classifies points as North (top), South (bottom), East (right), or West (left)
    based on their position relative to the bounding box.

    Args:
        boundary_points: List of (y, x) boundary coordinates
        output_path: Where to save the plot
        title: Plot title
        margin: Fraction of range to use for edge classification (default 5%)
    """
    import matplotlib.pyplot as plt

    if not boundary_points:
        logger.warning("No boundary points to plot")
        return

    points_array = np.array(boundary_points)
    y_vals = points_array[:, 0]
    x_vals = points_array[:, 1]

    y_min, y_max = y_vals.min(), y_vals.max()
    x_min, x_max = x_vals.min(), x_vals.max()

    y_range = y_max - y_min
    x_range = x_max - x_min

    # Classify each point
    is_north = y_vals <= (y_min + margin * y_range)
    is_south = y_vals >= (y_max - margin * y_range)
    is_west = x_vals <= (x_min + margin * x_range)
    is_east = x_vals >= (x_max - margin * x_range)

    # Create color array
    colors = np.zeros(len(boundary_points))
    colors[is_north] = 1  # Red for North
    colors[is_south] = 2  # Green for South
    colors[is_west] = 3  # Blue for West
    colors[is_east] = 4  # Yellow for East

    # Count points on each edge
    north_count = np.sum(is_north)
    south_count = np.sum(is_south)
    west_count = np.sum(is_west)
    east_count = np.sum(is_east)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot all points colored by edge
    scatter = ax.scatter(
        x_vals, y_vals, c=colors, cmap="tab10", s=10, alpha=0.6, edgecolors="black", linewidth=0.5
    )

    ax.set_xlim(x_min - 5, x_max + 5)
    ax.set_ylim(y_max + 5, y_min - 5)  # Flip Y
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)

    # Add edge counts as text
    info_text = (
        f"North: {north_count:,} ({100*north_count/len(boundary_points):.1f}%)\n"
        f"South: {south_count:,} ({100*south_count/len(boundary_points):.1f}%)\n"
        f"West:  {west_count:,} ({100*west_count/len(boundary_points):.1f}%)\n"
        f"East:  {east_count:,} ({100*east_count/len(boundary_points):.1f}%)\n"
        f"Total: {len(boundary_points):,}"
    )
    ax.text(
        0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
        verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved edge distribution plot to {output_path}")


def plot_deduplication_comparison(
    before: List[Tuple[float, float]],
    after: List[Tuple[float, float]],
    output_path: Path,
) -> None:
    """
    Plot before/after comparison of deduplication.

    Shows how many points were removed as duplicates and where
    the remaining points are distributed.

    Args:
        before: Boundary points before deduplication
        after: Boundary points after deduplication
        output_path: Where to save the plot
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot before
    if before:
        before_array = np.array(before)
        ax1.scatter(before_array[:, 1], before_array[:, 0], s=2, alpha=0.5)
        ax1.set_aspect("equal")
        ax1.set_title(f"Before Deduplication\n({len(before):,} points)")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.grid(True, alpha=0.3)

    # Plot after
    if after:
        after_array = np.array(after)
        ax2.scatter(after_array[:, 1], after_array[:, 0], s=5, alpha=0.6)
        ax2.set_aspect("equal")
        ax2.set_title(f"After Deduplication\n({len(after):,} points)")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.grid(True, alpha=0.3)

    removed = len(before) - len(after)
    reduction_pct = 100 * removed / len(before) if before else 0

    fig.suptitle(
        f"Deduplication Impact: Removed {removed:,} duplicates ({reduction_pct:.1f}% reduction)",
        fontsize=14
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved deduplication comparison plot to {output_path}")


def plot_sorting_effect(
    before: np.ndarray,
    after: np.ndarray,
    output_path: Path,
) -> None:
    """
    Plot the effect of angular sorting on boundary points.

    Shows how points are reordered from shuffled state into a proper
    closed loop around the boundary.

    Args:
        before: Points before sorting (may be in any order)
        after: Points after angular sorting
        output_path: Where to save the plot
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Before (scattered)
    ax1.scatter(before[:, 1], before[:, 0], s=2, alpha=0.5, c=np.arange(len(before)), cmap="tab20")
    ax1.set_aspect("equal")
    ax1.set_title("Before Sorting\n(Scattered points)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.grid(True, alpha=0.3)

    # Plot 2: After (sorted with line)
    ax2.scatter(after[:, 1], after[:, 0], s=5, alpha=0.6, c=np.arange(len(after)), cmap="viridis")
    # Draw line connecting points in order
    if len(after) > 1:
        after_closed = np.vstack([after, after[0]])  # Close the loop
        ax2.plot(after_closed[:, 1], after_closed[:, 0], "k-", linewidth=0.5, alpha=0.3)
    ax2.set_aspect("equal")
    ax2.set_title("After Sorting\n(Closed loop)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Show point ordering sequence
    if len(after) > 0:
        # Show first 20 points and their order
        sample_size = min(20, len(after))
        sample_points = after[:sample_size]
        colors = np.arange(sample_size)

        ax3.scatter(sample_points[:, 1], sample_points[:, 0], s=50, c=colors, cmap="rainbow",
                   edgecolors="black", linewidth=1)

        # Add numbers to show order
        for i, (y, x) in enumerate(sample_points):
            ax3.text(x, y, str(i), fontsize=8, ha="center", va="center")

        if len(sample_points) > 1:
            sample_points_closed = np.vstack([sample_points, sample_points[0]])
            ax3.plot(sample_points_closed[:, 1], sample_points_closed[:, 0], "k--",
                    linewidth=0.5, alpha=0.3)

        ax3.set_aspect("equal")
        ax3.set_title(f"Point Ordering\n(First {sample_size} points)")
        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")
        ax3.grid(True, alpha=0.3)

    fig.suptitle("Angular Sorting Effect", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved sorting effect plot to {output_path}")


def plot_transformation_pipeline(
    terrain,
    output_dir: Path,
    edge_sample_spacing: float = 1.0,
) -> None:
    """
    Create a series of plots showing each stage of coordinate transformation.

    Visualizes:
    1. Original rectangle in DEM pixel space (EPSG:4326)
    2. Geographic coordinates after Affine transform
    3. Reprojected coordinates (if CRS changed)
    4. Final mesh pixel space

    Args:
        terrain: Terrain object with transforms configured
        output_dir: Directory to save stage plots
        edge_sample_spacing: Spacing for edge sampling
    """
    import matplotlib.pyplot as plt
    from src.terrain.mesh_operations import generate_rectangle_edge_pixels

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dem_shape = terrain.dem_shape
    height, width = dem_shape

    # Generate original edge pixels
    edge_pixels = generate_rectangle_edge_pixels(dem_shape, edge_sample_spacing)
    edge_array = np.array(edge_pixels)

    # Stage 1: Original DEM space
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(edge_array[:, 1], edge_array[:, 0], s=2, alpha=0.5, c="blue")
    ax.set_xlim(-5, width + 5)
    ax.set_ylim(height + 5, -5)
    ax.set_aspect("equal")
    ax.set_title("Stage 1: Original Rectangle\nDEM Pixel Space (EPSG:4326)")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "stage_1_original_rectangle.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Created stage 1 plot: original rectangle")

    # Stage 2: Geographic coordinates
    try:
        transform = terrain.dem_transform
        geo_coords = np.array([transform * (x, y) for y, x in edge_pixels])

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(geo_coords[:, 0], geo_coords[:, 1], s=2, alpha=0.5, c="green")
        ax.set_aspect("equal")
        ax.set_title("Stage 2: Geographic Coordinates\nAfter Affine Transform")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "stage_2_geographic_coords.png", dpi=150, bbox_inches="tight")
        plt.close()

        logger.info("Created stage 2 plot: geographic coordinates")
    except Exception as e:
        logger.warning(f"Could not create stage 2 plot: {e}")

    # Stage 3: Transformed rectangle (if transforms applied)
    try:
        dem_layer = terrain.data_layers.get("dem")
        if dem_layer and "transformed_data" in dem_layer:
            transformed_shape = dem_layer["transformed_data"].shape
            t_height, t_width = transformed_shape

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(
                0.5, 0.5, f"Transformed Mesh Space\n{t_height} × {t_width} pixels",
                ha="center", va="center", fontsize=14, transform=ax.transAxes
            )
            ax.set_xlim(-5, t_width + 5)
            ax.set_ylim(t_height + 5, -5)
            ax.set_aspect("equal")
            ax.set_title("Stage 3: Transformed Rectangle\nMesh Pixel Space (EPSG:32617)")
            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "stage_3_transformed_rectangle.png", dpi=150, bbox_inches="tight")
            plt.close()

            logger.info("Created stage 3 plot: transformed rectangle")
    except Exception as e:
        logger.warning(f"Could not create stage 3 plot: {e}")

    # Stage 4: Mesh vertices
    try:
        if hasattr(terrain, "y_valid") and hasattr(terrain, "x_valid"):
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(terrain.x_valid, terrain.y_valid, s=1, alpha=0.3, c="red")
            ax.set_aspect("equal")
            ax.set_title(f"Stage 4: Valid Mesh Vertices\n({len(terrain.y_valid):,} points)")
            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "stage_4_mesh_vertices.png", dpi=150, bbox_inches="tight")
            plt.close()

            logger.info("Created stage 4 plot: mesh vertices")
    except Exception as e:
        logger.warning(f"Could not create stage 4 plot: {e}")


def create_full_pipeline_debug_plot(
    terrain,
    output_path: Path,
    edge_sample_spacing: float = 1.0,
) -> None:
    """
    Create a comprehensive debug visualization of the entire rectangle edge pipeline.

    Single plot showing:
    - Original rectangle sampling
    - Edge distribution before/after transforms
    - Deduplication statistics
    - Final boundary distribution

    Args:
        terrain: Terrain object with configured transforms
        output_path: Where to save the comprehensive plot
        edge_sample_spacing: Spacing for edge sampling
    """
    import matplotlib.pyplot as plt
    from src.terrain.mesh_operations import (
        generate_rectangle_edge_pixels,
        deduplicate_boundary_points,
    )

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    dem_shape = terrain.dem_shape
    height, width = dem_shape

    # Original edge sampling
    ax1 = fig.add_subplot(gs[0, 0])
    edge_pixels = generate_rectangle_edge_pixels(dem_shape, edge_sample_spacing)
    edge_array = np.array(edge_pixels)
    ax1.scatter(edge_array[:, 1], edge_array[:, 0], s=1, alpha=0.5)
    ax1.set_xlim(-5, width + 5)
    ax1.set_ylim(height + 5, -5)
    ax1.set_aspect("equal")
    ax1.set_title(f"Original Sampling\n({len(edge_pixels):,} points)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    # Edge distribution - original
    ax2 = fig.add_subplot(gs[0, 1])
    y_vals = edge_array[:, 0]
    x_vals = edge_array[:, 1]
    y_min, y_max = y_vals.min(), y_vals.max()
    x_min, x_max = x_vals.min(), x_vals.max()
    y_range = y_max - y_min
    x_range = x_max - x_min
    margin = 0.05

    is_north = y_vals <= (y_min + margin * y_range)
    is_south = y_vals >= (y_max - margin * y_range)
    is_west = x_vals <= (x_min + margin * x_range)
    is_east = x_vals >= (x_max - margin * x_range)

    colors = np.zeros(len(edge_pixels))
    colors[is_north] = 1
    colors[is_south] = 2
    colors[is_west] = 3
    colors[is_east] = 4

    ax2.scatter(x_vals, y_vals, c=colors, cmap="tab10", s=2, alpha=0.6)
    ax2.set_aspect("equal")
    ax2.set_title("Distribution\n(N=1, S=2, W=3, E=4)")

    # Statistics
    ax3 = fig.add_subplot(gs[0, 2])
    north_count = np.sum(is_north)
    south_count = np.sum(is_south)
    west_count = np.sum(is_west)
    east_count = np.sum(is_east)

    stats_text = (
        f"Original Sampling Statistics\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"North: {north_count:,} ({100*north_count/len(edge_pixels):.1f}%)\n"
        f"South: {south_count:,} ({100*south_count/len(edge_pixels):.1f}%)\n"
        f"West:  {west_count:,} ({100*west_count/len(edge_pixels):.1f}%)\n"
        f"East:  {east_count:,} ({100*east_count/len(edge_pixels):.1f}%)\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"Total: {len(edge_pixels):,} points"
    )
    ax3.text(0.1, 0.5, stats_text, fontsize=10, family="monospace",
            verticalalignment="center", bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8))
    ax3.axis("off")

    # After deduplication
    ax4 = fig.add_subplot(gs[1, 0])
    edge_pixels_dedup = deduplicate_boundary_points(edge_pixels)
    if edge_pixels_dedup:
        dedup_array = np.array(edge_pixels_dedup)
        ax4.scatter(dedup_array[:, 1], dedup_array[:, 0], s=3, alpha=0.6)
        ax4.set_xlim(-5, width + 5)
        ax4.set_ylim(height + 5, -5)
        ax4.set_aspect("equal")
        removed = len(edge_pixels) - len(edge_pixels_dedup)
        ax4.set_title(f"After Deduplication\n({len(edge_pixels_dedup):,} points, {removed:,} removed)")
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")

    # Deduplication stats
    ax5 = fig.add_subplot(gs[1, 1])
    removed = len(edge_pixels) - len(edge_pixels_dedup)
    removed_pct = 100 * removed / len(edge_pixels) if edge_pixels else 0
    dedup_text = (
        f"Deduplication Statistics\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"Before: {len(edge_pixels):,} points\n"
        f"After:  {len(edge_pixels_dedup):,} points\n"
        f"Removed: {removed:,} ({removed_pct:.1f}%)\n"
        f"\n"
        f"Most duplicates from\n"
        f"coordinate transformation"
    )
    ax5.text(0.1, 0.5, dedup_text, fontsize=10, family="monospace",
            verticalalignment="center", bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax5.axis("off")

    # Info text
    ax6 = fig.add_subplot(gs[1, 2])
    info_text = (
        f"Pipeline Information\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"DEM shape: {height}×{width}\n"
        f"Sampling: {edge_sample_spacing}px spacing\n"
        f"\n"
        f"Transforms:\n"
        f"1. Affine (pixel→geo)\n"
        f"2. CRS reproject\n"
        f"3. Inverse affine\n"
        f"4. Round & filter\n"
        f"\n"
        f"Uses angular sorting\n"
        f"for >100 point boundaries"
    )
    ax6.text(0.1, 0.5, info_text, fontsize=9, family="monospace",
            verticalalignment="center", bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8))
    ax6.axis("off")

    # Valid mesh vertices (if available)
    ax7 = fig.add_subplot(gs[2, :])
    if hasattr(terrain, "y_valid") and hasattr(terrain, "x_valid"):
        ax7.scatter(terrain.x_valid, terrain.y_valid, s=0.5, alpha=0.2, c="red", label="Valid vertices")
        if edge_pixels_dedup:
            dedup_array = np.array(edge_pixels_dedup)
            ax7.scatter(dedup_array[:, 1], dedup_array[:, 0], s=5, alpha=0.8, c="blue",
                       label="Boundary points", edgecolors="darkblue", linewidth=0.5)
        ax7.set_aspect("equal")
        ax7.set_title(f"Final Configuration: {len(terrain.y_valid):,} valid vertices + {len(edge_pixels_dedup):,} boundary points")
        ax7.set_xlabel("X")
        ax7.set_ylabel("Y")
        ax7.legend(loc="upper right")
    else:
        if edge_pixels_dedup:
            dedup_array = np.array(edge_pixels_dedup)
            ax7.scatter(dedup_array[:, 1], dedup_array[:, 0], s=5, alpha=0.8, c="blue")
        ax7.set_title(f"Boundary Points: {len(edge_pixels_dedup):,} points (mesh vertices not yet available)")
        ax7.set_xlabel("X")
        ax7.set_ylabel("Y")

    fig.suptitle(
        f"Rectangle Edge Extrusion - Complete Pipeline Debug\nDEM: {height}×{width} | "
        f"Spacing: {edge_sample_spacing}px",
        fontsize=14, fontweight="bold"
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved comprehensive debug plot to {output_path}")
