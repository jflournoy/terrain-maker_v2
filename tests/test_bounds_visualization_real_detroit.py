"""
Step-by-step visualization of transform-aware edge extrusion with real Detroit bounds.

Shows how edge pixels flow through the transformation pipeline:
1. WGS84 original rectangle
2. Reprojected to UTM (shape distorted)
3. After horizontal flip
4. Downsampled to final mesh (edges warped by downsampling)
5. Compare final edge shape to actual mesh grid

Uses the refactored TransformationPipeline and EdgeTransformer classes
for cleaner, more testable code.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import logging

from src.terrain.visualization.bounds_pipeline import (
    TransformationPipeline,
    EdgeTransformer,
    SimpleAffine,
)

logger = logging.getLogger(__name__)


def sample_rectangle_edges(height, width, spacing=1.0):
    """Sample edge pixels from rectangle."""
    edges = []
    for x in np.arange(0, width, spacing):
        edges.append((0, x))
        edges.append((height - 1, x))
    for y in np.arange(spacing, height - 1, spacing):
        edges.append((y, 0))
        edges.append((y, width - 1))
    return [(int(y), int(x)) for y, x in edges]


def map_pixel_to_world(y, x, affine):
    """Map grid pixel to world coordinates using SimpleAffine.

    This helper function maintains backward compatibility with existing code.
    """
    return affine.map_pixel_to_world(y, x)


class TestBoundsVisualizationRealDetroit:
    """Tests with real Detroit bounds, step-by-step visualization."""

    @pytest.fixture
    def detroit_bounds(self):
        """Real Detroit DEM bounds - 2° × 2° area centered on Detroit.

        bbox_utm computed using pyproj: WGS84 corners → UTM Zone 17N.
        Note: The UTM projection is NON-LINEAR. The rectangular WGS84 boundary
        becomes curved in UTM space due to Transverse Mercator projection.
        """
        return {
            'original_shape': (7_201, 7_201),  # 2° × 2° at 1 arc-second
            'bbox_wgs84': (-84.0, 41.0, -82.0, 43.0),
            # UTM bbox from pyproj transformation of corners:
            'bbox_utm': (247_679, 4_539_238, 418_491, 4_765_182),
            'target_vertices': 50_000,  # Further increased to see curvature more clearly
        }

    @pytest.fixture
    def output_dir(self):
        """Create output directory for plots."""
        output_path = Path(__file__).parent.parent / "docs" / "images" / "bounds_visualization"
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    def test_stage_1_wgs84_original(self, detroit_bounds, output_dir):
        """Visualize Stage 1: Original DEM in WGS84."""
        logger.info("\n" + "="*70)
        logger.info("STAGE 1: ORIGINAL DEM (WGS84)")
        logger.info("="*70)

        original_shape = detroit_bounds['original_shape']
        bbox_wgs84 = detroit_bounds['bbox_wgs84']

        logger.info(f"DEM dimensions: {original_shape[0]:,} × {original_shape[1]:,} pixels")
        logger.info(f"WGS84 bounds: {bbox_wgs84}")

        # Create affine transform for WGS84
        scale_x = (bbox_wgs84[2] - bbox_wgs84[0]) / original_shape[1]
        scale_y = -(bbox_wgs84[3] - bbox_wgs84[1]) / original_shape[0]
        affine_wgs84 = SimpleAffine(
            c=bbox_wgs84[0],
            f=bbox_wgs84[3],
            a=scale_x,
            e=scale_y,
            b=0,
            d=0
        )

        # Sample edges
        edge_pixels = sample_rectangle_edges(
            original_shape[0], original_shape[1], spacing=100
        )
        logger.info(f"Edge pixels sampled: {len(edge_pixels)} (at 100-pixel spacing)")

        # Map to WGS84 coordinates
        edge_coords = [map_pixel_to_world(y, x, affine_wgs84) for y, x in edge_pixels]

        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))

        # Draw boundary rectangle
        rect = patches.Rectangle(
            (bbox_wgs84[0], bbox_wgs84[1]),
            bbox_wgs84[2] - bbox_wgs84[0],
            bbox_wgs84[3] - bbox_wgs84[1],
            linewidth=3, edgecolor='blue', facecolor='none', label='Boundary'
        )
        ax.add_patch(rect)

        # Draw edge samples
        edge_x = [c[0] for c in edge_coords]
        edge_y = [c[1] for c in edge_coords]
        ax.scatter(edge_x, edge_y, c='blue', s=15, alpha=0.6, label=f'Edge samples ({len(edge_pixels)})')

        ax.set_title(
            f'Stage 1: Original DEM (WGS84)\n{original_shape[0]:,}×{original_shape[1]:,} pixels',
            fontsize=14, fontweight='bold'
        )
        ax.set_xlabel('Longitude (°W)', fontsize=11)
        ax.set_ylabel('Latitude (°N)', fontsize=11)
        ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.02, 1))
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        output_path = output_dir / "01_stage1_wgs84_original.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {output_path}")

    def test_stage_2_reprojected_utm(self, detroit_bounds, output_dir):
        """Visualize Stage 2: Reprojected to UTM showing NON-LINEAR curvature.

        The WGS84 → UTM transformation is a Transverse Mercator projection,
        which is NON-LINEAR. A horizontal line (constant latitude) in WGS84
        becomes CURVED in UTM space. This test visualizes that curvature.
        """
        logger.info("\n" + "="*70)
        logger.info("STAGE 2: REPROJECTED TO UTM (EPSG:32617) - NON-LINEAR CURVATURE")
        logger.info("="*70)

        from pyproj import Transformer as PyProjTransformer

        original_shape = detroit_bounds['original_shape']
        bbox_wgs84 = detroit_bounds['bbox_wgs84']
        bbox_utm = detroit_bounds['bbox_utm']

        # Create pyproj transformer for proper non-linear projection
        proj_transformer = PyProjTransformer.from_crs(
            "EPSG:4326", "EPSG:32617", always_xy=True
        )

        # Distortion at Michigan latitude (for shape estimation)
        distortion_factor = 0.74  # cos(42°)
        reprojected_shape = (
            int(original_shape[0] * distortion_factor),
            int(original_shape[1] / distortion_factor)
        )

        logger.info(f"Original shape: {original_shape}")
        logger.info(f"Reprojected shape: {reprojected_shape}")

        # Create affine for WGS84 (pixel → geographic coordinates)
        scale_x_wgs84 = (bbox_wgs84[2] - bbox_wgs84[0]) / original_shape[1]
        scale_y_wgs84 = -(bbox_wgs84[3] - bbox_wgs84[1]) / original_shape[0]
        affine_wgs84 = SimpleAffine(
            c=bbox_wgs84[0], f=bbox_wgs84[3],
            a=scale_x_wgs84, e=scale_y_wgs84,
            b=0, d=0
        )

        # Sample DENSE points along each edge to capture curvature
        # We need many points per edge, not just corners
        n_samples_per_edge = 100

        def sample_edge(start_y, start_x, end_y, end_x, n_samples):
            """Sample n points along an edge."""
            return [
                (int(start_y + (end_y - start_y) * t / (n_samples - 1)),
                 int(start_x + (end_x - start_x) * t / (n_samples - 1)))
                for t in range(n_samples)
            ]

        h, w = original_shape
        # Sample all 4 edges densely
        top_edge = sample_edge(0, 0, 0, w-1, n_samples_per_edge)
        bottom_edge = sample_edge(h-1, 0, h-1, w-1, n_samples_per_edge)
        left_edge = sample_edge(0, 0, h-1, 0, n_samples_per_edge)
        right_edge = sample_edge(0, w-1, h-1, w-1, n_samples_per_edge)

        logger.info(f"Sampled {n_samples_per_edge} points per edge (400 total)")

        # Transform each edge through pyproj (NON-LINEAR!)
        def transform_edge_to_utm(edge_pixels):
            """Transform edge pixels: pixel → WGS84 → UTM (non-linear)."""
            utm_coords = []
            for y, x in edge_pixels:
                # Pixel → WGS84
                lon, lat = affine_wgs84.map_pixel_to_world(y, x)
                # WGS84 → UTM (non-linear Transverse Mercator!)
                easting, northing = proj_transformer.transform(lon, lat)
                utm_coords.append((easting, northing))
            return utm_coords

        top_utm = transform_edge_to_utm(top_edge)
        bottom_utm = transform_edge_to_utm(bottom_edge)
        left_utm = transform_edge_to_utm(left_edge)
        right_utm = transform_edge_to_utm(right_edge)

        # Measure curvature: how much do the edges deviate from straight lines?
        top_northings = [c[1] for c in top_utm]
        top_curvature = max(top_northings) - min(top_northings)
        bottom_northings = [c[1] for c in bottom_utm]
        bottom_curvature = max(bottom_northings) - min(bottom_northings)

        logger.info(f"Top edge curvature: {top_curvature:.1f} meters")
        logger.info(f"Bottom edge curvature: {bottom_curvature:.1f} meters")

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Left: WGS84 rectangle (straight edges)
        ax = axes[0]
        rect1 = patches.Rectangle(
            (bbox_wgs84[0], bbox_wgs84[1]),
            bbox_wgs84[2] - bbox_wgs84[0],
            bbox_wgs84[3] - bbox_wgs84[1],
            linewidth=2, edgecolor='blue', facecolor='none', label='WGS84 boundary'
        )
        ax.add_patch(rect1)

        # Plot edge samples in WGS84
        for edge, color, label in [
            (top_edge, 'red', 'Top edge (lat=42.5°)'),
            (bottom_edge, 'green', 'Bottom edge (lat=41.0°)'),
            (left_edge, 'purple', 'Left edge'),
            (right_edge, 'orange', 'Right edge'),
        ]:
            coords = [affine_wgs84.map_pixel_to_world(y, x) for y, x in edge]
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            ax.plot(xs, ys, '-', color=color, linewidth=2, label=label)

        ax.set_title('WGS84 (Original)\nStraight edges (constant lat/lon)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Longitude (°W)')
        ax.set_ylabel('Latitude (°N)')
        ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1))
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # Right: UTM (CURVED edges due to non-linear projection)
        ax = axes[1]

        # Draw the UTM bounding box for reference
        rect2 = patches.Rectangle(
            (bbox_utm[0], bbox_utm[1]),
            bbox_utm[2] - bbox_utm[0],
            bbox_utm[3] - bbox_utm[1],
            linewidth=1, edgecolor='gray', facecolor='none',
            linestyle='--', label='UTM bbox (reference)'
        )
        ax.add_patch(rect2)

        # Plot the ACTUAL curved edges from pyproj transformation
        for utm_coords, color, label in [
            (top_utm, 'red', f'Top edge (curved {top_curvature:.0f}m)'),
            (bottom_utm, 'green', f'Bottom edge (curved {bottom_curvature:.0f}m)'),
            (left_utm, 'purple', 'Left edge'),
            (right_utm, 'orange', 'Right edge'),
        ]:
            xs = [c[0] for c in utm_coords]
            ys = [c[1] for c in utm_coords]
            ax.plot(xs, ys, '-', color=color, linewidth=2, label=label)

        ax.set_title(f'UTM Zone 17N (Transverse Mercator)\nCURVED edges from non-linear projection',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
        ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1))
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.suptitle('Stage 2: WGS84 → UTM Non-Linear Projection\n'
                    'Horizontal lines (constant latitude) become CURVED in UTM',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        output_path = output_dir / "02_stage2_utm_reprojected.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {output_path}")

    def test_stage_3_horizontal_flip(self, detroit_bounds, output_dir):
        """Visualize Stage 3: After horizontal flip (shape unchanged, edges flipped).

        Uses proper pyproj transformation for WGS84 → UTM, then applies flip.
        """
        logger.info("\n" + "="*70)
        logger.info("STAGE 3: HORIZONTAL FLIP (using pyproj for reprojection)")
        logger.info("="*70)

        original_shape = detroit_bounds['original_shape']
        bbox_wgs84 = detroit_bounds['bbox_wgs84']
        bbox_utm = detroit_bounds['bbox_utm']

        # Create transformation pipeline with bbox_wgs84 to enable pyproj
        pipeline = TransformationPipeline(
            original_shape=original_shape,
            distortion_factor=0.74,  # cos(42°) - used for shape estimation
            target_vertices=detroit_bounds['target_vertices'],
            bbox_wgs84=bbox_wgs84,
            bbox_utm=bbox_utm,
        )
        transformer = EdgeTransformer(pipeline, use_pyproj=True)

        flipped_shape = pipeline.flipped_shape
        logger.info(f"Shape after flip: {flipped_shape} (unchanged from reprojected)")

        # Sample DENSE edge pixels from original for smooth edges
        edge_pixels_original = sample_rectangle_edges(
            original_shape[0], original_shape[1], spacing=100
        )

        # Use proper pyproj transformation: original → reprojected → flipped
        # This captures the NON-LINEAR curvature of the Transverse Mercator projection
        edge_pixels_flipped = transformer.transform_stage(
            edge_pixels_original, 'original', 'flipped'
        )
        logger.info(f"Edge pixels after flip: {len(edge_pixels_flipped)} (via pyproj)")

        # Map to world coordinates using pipeline's affine
        affine_utm = pipeline.get_affine('reprojected')
        edge_coords_flipped = [map_pixel_to_world(y, x, affine_utm) for y, x in edge_pixels_flipped]

        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))

        rect = patches.Rectangle(
            (bbox_utm[0], bbox_utm[1]),
            bbox_utm[2] - bbox_utm[0],
            bbox_utm[3] - bbox_utm[1],
            linewidth=3, edgecolor='green', facecolor='none', label='UTM Boundary'
        )
        ax.add_patch(rect)

        edge_x = [c[0] for c in edge_coords_flipped]
        edge_y = [c[1] for c in edge_coords_flipped]
        ax.scatter(edge_x, edge_y, c='green', s=15, alpha=0.6,
                  label=f'Edge samples ({len(edge_pixels_flipped)}, pyproj+flip)')

        ax.set_title(
            f'Stage 3: Horizontal Flip (UTM via pyproj)\n{flipped_shape[0]:,}×{flipped_shape[1]:,} pixels',
            fontsize=14, fontweight='bold'
        )
        ax.set_xlabel('Easting (m)', fontsize=11)
        ax.set_ylabel('Northing (m)', fontsize=11)
        ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.02, 1))
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        output_path = output_dir / "03_stage3_flipped.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {output_path}")

    def test_stage_4_downsampled_mesh(self, detroit_bounds, output_dir):
        """Visualize Stage 4: Downsampled to final mesh (edges warped by grid reduction).

        Uses proper pyproj transformation for the full pipeline: WGS84 → UTM → flip → downsample.
        """
        logger.info("\n" + "="*70)
        logger.info("STAGE 4: DOWNSAMPLED TO MESH GRID (using pyproj)")
        logger.info("="*70)

        original_shape = detroit_bounds['original_shape']
        bbox_wgs84 = detroit_bounds['bbox_wgs84']
        bbox_utm = detroit_bounds['bbox_utm']
        target_vertices = detroit_bounds['target_vertices']

        # Create transformation pipeline WITH bbox_wgs84 to enable pyproj
        pipeline = TransformationPipeline(
            original_shape=original_shape,
            distortion_factor=0.74,
            target_vertices=target_vertices,
            bbox_wgs84=bbox_wgs84,
            bbox_utm=bbox_utm,
        )

        logger.info(f"Target vertices: {target_vertices:,}")
        logger.info(f"Final grid shape: {pipeline.final_shape[0]}×{pipeline.final_shape[1]} pixels")
        logger.info(f"Actual vertices: {pipeline.final_shape[0] * pipeline.final_shape[1]}")

        downsample_y, downsample_x = pipeline.downsample_factors
        logger.info(f"Downsample Y: {downsample_y:.1f}x")
        logger.info(f"Downsample X: {downsample_x:.1f}x")

        # Sample DENSE original edges for smooth mesh boundaries
        edge_pixels_original = sample_rectangle_edges(
            original_shape[0], original_shape[1], spacing=100
        )

        # Use transformer with pyproj for proper non-linear projection
        transformer = EdgeTransformer(pipeline, use_pyproj=True)

        # Get edges: integer grid (snapped) vs fractional mesh space (smooth)
        edge_pixels_flipped = transformer.transform_stage(edge_pixels_original, 'original', 'flipped')
        edge_pixels_final = transformer.transform_full_pipeline(edge_pixels_original)
        edge_fractional_clamped = transformer.transform_to_mesh_space(edge_pixels_original, 'original', clamp=True)
        edge_fractional_unclamped = transformer.transform_to_mesh_space(edge_pixels_original, 'original', clamp=False)

        final_height, final_width = pipeline.final_shape

        logger.info(f"Integer grid vertices: {len(edge_pixels_final)} (deduplicated)")
        logger.info(f"Fractional mesh vertices: {len(edge_fractional_clamped)} (all preserved)")

        # Log the range of unclamped coordinates to show true curvature extent
        x_unclamped = [x for _, x in edge_fractional_unclamped]
        y_unclamped = [y for y, _ in edge_fractional_unclamped]
        logger.info(f"Unclamped X range: {min(x_unclamped):.2f} - {max(x_unclamped):.2f} (grid: 0-{final_width-1})")
        logger.info(f"Unclamped Y range: {min(y_unclamped):.2f} - {max(y_unclamped):.2f} (grid: 0-{final_height-1})")

        # Create visualization (show both integer and fractional)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left: Integer grid (snapped to pixels)
        ax = axes[0]

        # Draw mesh grid lines
        for i in range(0, final_height + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        for i in range(0, final_width + 1):
            ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)

        # Draw boundary
        rect = patches.Rectangle(
            (-0.5, -0.5), final_width, final_height,
            linewidth=2, edgecolor='gray', facecolor='none', linestyle='--'
        )
        ax.add_patch(rect)

        # Draw integer edge vertices
        if edge_pixels_final:
            edge_y_final = [p[0] for p in edge_pixels_final]
            edge_x_final = [p[1] for p in edge_pixels_final]
            ax.scatter(edge_x_final, edge_y_final, c='green', s=40, alpha=0.8,
                      label=f'Integer vertices ({len(edge_pixels_final)})')

        ax.set_title(f'Integer Grid (Snapped)\n{len(edge_pixels_final)} unique vertices',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('X pixels')
        ax.set_ylabel('Y pixels')
        ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1))
        ax.set_xlim(-1, final_width)
        ax.set_ylim(final_height, -1)

        # Right: Fractional mesh space showing TRUE projection curvature (unclamped)
        ax = axes[1]

        # Expand view to show unclamped coordinates that exceed grid bounds
        x_unclamped_all = [x for _, x in edge_fractional_unclamped]
        y_unclamped_all = [y for y, _ in edge_fractional_unclamped]
        x_max_view = max(final_width, max(x_unclamped_all) + 1)
        y_max_view = max(final_height, max(y_unclamped_all) + 1)

        # Draw mesh grid lines
        for i in range(0, int(y_max_view) + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        for i in range(0, int(x_max_view) + 1):
            ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)

        # Draw rectangular grid boundary (what mesh operations use)
        rect = patches.Rectangle(
            (-0.5, -0.5), final_width, final_height,
            linewidth=2, edgecolor='red', facecolor='none', linestyle='--',
            label=f'Mesh grid (0-{final_width-1})'
        )
        ax.add_patch(rect)

        # Draw UNCLAMPED fractional edge vertices - shows TRUE curved boundary
        edge_y_unclamped = [p[0] for p in edge_fractional_unclamped]
        edge_x_unclamped = [p[1] for p in edge_fractional_unclamped]
        ax.scatter(edge_x_unclamped, edge_y_unclamped, c='blue', s=15, alpha=0.7,
                  label=f'True boundary ({len(edge_fractional_unclamped)})')

        # Find and annotate corners (unclamped to show true positions)
        corners_unclamped = transformer.transform_to_mesh_space([
            (0, 0),  # NW
            (0, original_shape[1] - 1),  # NE
            (original_shape[0] - 1, 0),  # SW
            (original_shape[0] - 1, original_shape[1] - 1),  # SE
        ], 'original', clamp=False)

        # Mark corners with labels
        corner_names = ['NW', 'NE', 'SW', 'SE']
        for (y, x), name in zip(corners_unclamped, corner_names):
            ax.plot(x, y, 'go', markersize=10, markeredgecolor='black', markeredgewidth=1)
            # Offset labels based on corner position
            dx = -3 if 'W' in name else 1
            dy = -1.5 if 'N' in name else 1.5
            ax.annotate(f'{name}\n({y:.1f},{x:.1f})', (x, y),
                       xytext=(x + dx, y + dy), fontsize=8, ha='center',
                       arrowprops=dict(arrowstyle='->', color='green', lw=0.5))

        # Log true curvature extent
        logger.info(f"TRUE boundary extends beyond grid:")
        logger.info(f"  X: {min(edge_x_unclamped):.2f} to {max(edge_x_unclamped):.2f} (grid 0-{final_width-1})")
        logger.info(f"  Y: {min(edge_y_unclamped):.2f} to {max(edge_y_unclamped):.2f} (grid 0-{final_height-1})")

        ax.set_title(f'TRUE Projection Curvature (Unclamped)\nRed dashed = mesh grid, Blue = true curved boundary',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('X (fractional)')
        ax.set_ylabel('Y (fractional)')
        ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1))
        ax.set_xlim(-1, x_max_view)
        ax.set_ylim(y_max_view, -1)

        plt.suptitle(f'Stage 4: Integer Grid vs Fractional Mesh Space\n{final_height}×{final_width} grid',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        output_path = output_dir / "04_stage4_downsampled_mesh.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {output_path}")

    def test_stage_5_edge_shape_comparison(self, detroit_bounds, output_dir):
        """Visualize final edge shape: shows how rectangle becomes curved mesh boundary.

        Uses proper pyproj transformation for all stages to capture non-linear curvature.
        """
        logger.info("\n" + "="*70)
        logger.info("STAGE 5: EDGE SHAPE COMPARISON - RECTANGLE TO WARPED MESH (using pyproj)")
        logger.info("="*70)

        original_shape = detroit_bounds['original_shape']
        bbox_wgs84 = detroit_bounds['bbox_wgs84']
        bbox_utm = detroit_bounds['bbox_utm']
        target_vertices = detroit_bounds['target_vertices']

        # Create pipeline WITH bbox_wgs84 to enable pyproj
        pipeline = TransformationPipeline(
            original_shape=original_shape,
            distortion_factor=0.74,
            target_vertices=target_vertices,
            bbox_wgs84=bbox_wgs84,
            bbox_utm=bbox_utm,
        )
        transformer = EdgeTransformer(pipeline, use_pyproj=True)

        # Get all transformation stages
        edge_pixels_original = sample_rectangle_edges(
            original_shape[0], original_shape[1], spacing=50
        )

        # Transform through pipeline at each stage
        edge_pixels_utm = transformer.transform_stage(edge_pixels_original, 'original', 'reprojected')
        edge_pixels_flipped = transformer.transform_stage(edge_pixels_original, 'original', 'flipped')
        edge_pixels_final = transformer.transform_full_pipeline(edge_pixels_original)
        edge_fractional = transformer.transform_to_mesh_space(edge_pixels_original, 'original', clamp=False)

        final_height, final_width = pipeline.final_shape

        logger.info(f"Original edge: {len(edge_pixels_original)} pixels (rectangle)")
        logger.info(f"Final edge (integer): {len(edge_pixels_final)} pixels (deduplicated)")
        logger.info(f"Final edge (fractional): {len(edge_fractional)} pixels (unclamped)")

        # Create detailed comparison
        fig = plt.figure(figsize=(16, 10))

        # Subplot 1: Original rectangle in original space
        ax1 = plt.subplot(2, 3, 1)
        edge_y_orig = [p[0] for p in edge_pixels_original]
        edge_x_orig = [p[1] for p in edge_pixels_original]
        ax1.scatter(edge_x_orig, edge_y_orig, c='blue', s=20, alpha=0.6)
        rect = patches.Rectangle(
            (0, 0), original_shape[1]-1, original_shape[0]-1,
            linewidth=2, edgecolor='blue', facecolor='none'
        )
        ax1.add_patch(rect)
        ax1.set_title('Stage 1: Original\n(Pure rectangle)', fontweight='bold')
        ax1.set_xlabel('X pixels')
        ax1.set_ylabel('Y pixels')
        ax1.set_xlim(-500, original_shape[1]+500)
        ax1.set_ylim(original_shape[0]+500, -500)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')

        # Subplot 2: After reprojection (distorted rectangle)
        ax2 = plt.subplot(2, 3, 2)
        edge_y_utm = [p[0] for p in edge_pixels_utm]
        edge_x_utm = [p[1] for p in edge_pixels_utm]
        ax2.scatter(edge_x_utm, edge_y_utm, c='red', s=20, alpha=0.6)
        rep_h, rep_w = pipeline.reprojected_shape
        rect = patches.Rectangle(
            (0, 0), rep_w-1, rep_h-1,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax2.add_patch(rect)
        ax2.set_title('Stage 2: Reprojected\n(Rectangle distorted)', fontweight='bold')
        ax2.set_xlabel('X pixels')
        ax2.set_ylabel('Y pixels')
        ax2.set_xlim(-500, rep_w+500)
        ax2.set_ylim(rep_h+500, -500)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')

        # Subplot 3: After flip (still rectangular)
        ax3 = plt.subplot(2, 3, 3)
        edge_y_flip = [p[0] for p in edge_pixels_flipped]
        edge_x_flip = [p[1] for p in edge_pixels_flipped]
        ax3.scatter(edge_x_flip, edge_y_flip, c='orange', s=20, alpha=0.6)
        flip_h, flip_w = pipeline.flipped_shape
        rect = patches.Rectangle(
            (0, 0), flip_w-1, flip_h-1,
            linewidth=2, edgecolor='orange', facecolor='none'
        )
        ax3.add_patch(rect)
        ax3.set_title('Stage 3: Flipped\n(Shape unchanged)', fontweight='bold')
        ax3.set_xlabel('X pixels')
        ax3.set_ylabel('Y pixels')
        ax3.set_xlim(-500, flip_w+500)
        ax3.set_ylim(flip_h+500, -500)
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal')

        # Subplot 4: After downsampling - show TRUE curved boundary (unclamped)
        ax4 = plt.subplot(2, 3, 4)

        final_h, final_w = pipeline.final_shape

        # Compute view bounds to show full unclamped curve
        edge_y_frac = [p[0] for p in edge_fractional]
        edge_x_frac = [p[1] for p in edge_fractional]
        x_max_view = max(final_w, max(edge_x_frac) + 1)
        y_max_view = max(final_h, max(edge_y_frac) + 1)

        # Draw mesh grid for context
        for i in range(0, int(y_max_view) + 1, max(1, final_h // 8)):
            ax4.axhline(i, color='gray', linewidth=0.5, alpha=0.2)
        for i in range(0, int(x_max_view) + 1, max(1, final_w // 8)):
            ax4.axvline(i, color='gray', linewidth=0.5, alpha=0.2)

        # Draw mesh grid boundary (red dashed)
        rect = patches.Rectangle(
            (0, 0), final_w-1, final_h-1,
            linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
        )
        ax4.add_patch(rect)

        # Draw fractional mesh vertices (TRUE curved boundary - unclamped)
        ax4.scatter(edge_x_frac, edge_y_frac, c='blue', s=20, alpha=0.6,
                   label=f'True boundary ({len(edge_fractional)})')

        # Draw integer grid vertices (snapped)
        if edge_pixels_final:
            edge_y_final = [p[0] for p in edge_pixels_final]
            edge_x_final = [p[1] for p in edge_pixels_final]
            ax4.scatter(edge_x_final, edge_y_final, c='green', s=40, alpha=0.8, marker='s',
                       label=f'Integer ({len(edge_pixels_final)})')

        ax4.set_title('Stage 4: True Curved Boundary\n(Blue=unclamped, Green=integer)', fontweight='bold', color='darkgreen')
        ax4.set_xlabel('X pixels')
        ax4.set_ylabel('Y pixels')
        ax4.set_xlim(-1, x_max_view)
        ax4.set_ylim(y_max_view, -1)
        ax4.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1))
        ax4.grid(True, alpha=0.3)

        # Subplot 5: Vertex counts (not coverage % - fractional preserves all)
        ax5 = plt.subplot(2, 3, 5)
        stages = ['Original', 'Reprojected', 'Flipped', 'Final\n(Integer)', 'Final\n(Fractional)']
        edge_counts = [
            len(edge_pixels_original),
            len(edge_pixels_utm),
            len(edge_pixels_flipped),
            len(edge_pixels_final),
            len(edge_fractional),
        ]

        colors_bar = ['blue', 'red', 'orange', 'green', 'dodgerblue']
        ax5.bar(range(len(stages)), edge_counts, color=colors_bar, alpha=0.7)
        ax5.set_ylabel('Vertex Count', fontsize=11)
        ax5.set_title('Edge Vertices at Each Stage', fontweight='bold')
        ax5.set_xticks(range(len(stages)))
        ax5.set_xticklabels(stages, fontsize=9)
        ax5.grid(True, alpha=0.3, axis='y')

        # Add count labels on bars
        for i, count in enumerate(edge_counts):
            ax5.text(i, count + 5, str(count), ha='center', va='bottom', fontsize=8)

        # Subplot 6: Key metrics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')

        distortion_factor = pipeline.distortion_factor
        summary = f"""
TRANSFORM-AWARE EDGE EXTRUSION

KEY OBSERVATION:
The rectangle edges get WARPED by the
transformation pipeline:

1. WGS84 Rectangle
   • Perfect rectangle shape
   • {original_shape[0]:,}×{original_shape[1]:,} pixels

2. Reprojection Distortion
   • Height: {distortion_factor:.1%} (compression)
   • Width: {1/distortion_factor:.1%} (expansion)
   • Still roughly rectangular

3. Horizontal Flip
   • Shape unchanged
   • Edges flipped left↔right

4. Downsampling to Mesh
   • {final_h}×{final_w} final grid
   • Edges WARPED by grid quantization
   • Transform-aware sampling:
     {len(edge_pixels_final)} boundary vertices

RESULT: Rectangle → Warped Mesh Boundary
The final edge shape matches the mesh grid
with perfect alignment due to transform-aware
sampling from the original DEM bounds.
        """

        ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        plt.suptitle('Stage 5: Edge Shape Transformation - Rectangle → Warped Mesh Boundary',
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()

        output_path = output_dir / "05_stage5_edge_shape_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {output_path}")

    def test_transform_verification(self, detroit_bounds, output_dir):
        """Verify pixel transformation matches demo settings exactly.

        This test compares the transformation pipeline against direct pyproj
        calculations to ensure the simulation is correct.
        """
        from pyproj import Transformer as PyProjTransformer

        logger.info("\n" + "="*70)
        logger.info("TRANSFORM VERIFICATION: Comparing Pipeline vs Direct Calculation")
        logger.info("="*70)

        original_shape = detroit_bounds['original_shape']
        bbox_wgs84 = detroit_bounds['bbox_wgs84']
        target_vertices = detroit_bounds['target_vertices']

        # Demo settings
        src_crs = "EPSG:4326"   # WGS84 (same as demo)
        dst_crs = "EPSG:32617"  # UTM Zone 17N (same as demo)
        flip = "horizontal"     # Same as demo

        logger.info(f"Demo settings: {src_crs} → {dst_crs}, flip={flip}")

        # Create pyproj transformer for direct calculation
        proj_transformer = PyProjTransformer.from_crs(src_crs, dst_crs, always_xy=True)

        # Create WGS84 affine for pixel → geographic conversion
        min_lon, min_lat, max_lon, max_lat = bbox_wgs84
        scale_x_wgs84 = (max_lon - min_lon) / original_shape[1]
        scale_y_wgs84 = -(max_lat - min_lat) / original_shape[0]
        affine_wgs84 = SimpleAffine(
            c=min_lon, f=max_lat,
            a=scale_x_wgs84, e=scale_y_wgs84,
            b=0, d=0
        )

        # Transform corners using direct pyproj
        corners_pixel = [
            (0, 0),                                          # NW
            (0, original_shape[1] - 1),                     # NE
            (original_shape[0] - 1, 0),                     # SW
            (original_shape[0] - 1, original_shape[1] - 1), # SE
        ]
        corner_names = ['NW', 'NE', 'SW', 'SE']

        logger.info("\nDirect pyproj corner transformation:")
        corners_utm = []
        for (y, x), name in zip(corners_pixel, corner_names):
            lon, lat = affine_wgs84.map_pixel_to_world(y, x)
            easting, northing = proj_transformer.transform(lon, lat)
            corners_utm.append((easting, northing))
            logger.info(f"  {name}: pixel ({y}, {x}) → WGS84 ({lon:.4f}, {lat:.4f}) → UTM ({easting:.1f}, {northing:.1f})")

        # Compute ACTUAL UTM envelope from transformed corners
        eastings = [c[0] for c in corners_utm]
        northings = [c[1] for c in corners_utm]
        actual_bbox_utm = (min(eastings), min(northings), max(eastings), max(northings))

        logger.info(f"\nActual UTM envelope from corners:")
        logger.info(f"  Min easting:  {actual_bbox_utm[0]:.1f} (from {corner_names[eastings.index(min(eastings))]})")
        logger.info(f"  Max easting:  {actual_bbox_utm[2]:.1f} (from {corner_names[eastings.index(max(eastings))]})")
        logger.info(f"  Min northing: {actual_bbox_utm[1]:.1f} (from {corner_names[northings.index(min(northings))]})")
        logger.info(f"  Max northing: {actual_bbox_utm[3]:.1f} (from {corner_names[northings.index(max(northings))]})")

        # Create pipeline with actual computed UTM bbox
        pipeline = TransformationPipeline(
            original_shape=original_shape,
            distortion_factor=0.74,
            target_vertices=target_vertices,
            bbox_wgs84=bbox_wgs84,
            bbox_utm=actual_bbox_utm,
            src_crs=src_crs,
            dst_crs=dst_crs,
        )
        transformer = EdgeTransformer(pipeline, use_pyproj=True)

        # Sample dense edge pixels
        edge_pixels_original = sample_rectangle_edges(
            original_shape[0], original_shape[1], spacing=100
        )

        # Transform to final mesh space (fractional)
        edge_fractional = transformer.transform_to_mesh_space(edge_pixels_original, 'original', clamp=False)

        final_height, final_width = pipeline.final_shape
        logger.info(f"\nFinal mesh grid: {final_height}×{final_width}")
        logger.info(f"Downsample factors: Y={pipeline.downsample_factors[0]:.1f}, X={pipeline.downsample_factors[1]:.1f}")

        # Create verification visualization
        fig = plt.figure(figsize=(18, 6))

        # Panel 1: UTM space with actual corner positions
        ax1 = plt.subplot(1, 3, 1)

        # Draw UTM envelope
        rect = patches.Rectangle(
            (actual_bbox_utm[0], actual_bbox_utm[1]),
            actual_bbox_utm[2] - actual_bbox_utm[0],
            actual_bbox_utm[3] - actual_bbox_utm[1],
            linewidth=2, edgecolor='red', facecolor='none', linestyle='--',
            label='UTM envelope'
        )
        ax1.add_patch(rect)

        # Draw actual curved boundary
        edge_utm_coords = []
        for y, x in edge_pixels_original:
            lon, lat = affine_wgs84.map_pixel_to_world(y, x)
            easting, northing = proj_transformer.transform(lon, lat)
            edge_utm_coords.append((easting, northing))

        edge_e = [c[0] for c in edge_utm_coords]
        edge_n = [c[1] for c in edge_utm_coords]
        ax1.scatter(edge_e, edge_n, c='blue', s=10, alpha=0.6, label='Boundary')

        # Mark corners
        for (e, n), name in zip(corners_utm, corner_names):
            ax1.plot(e, n, 'go', markersize=10, markeredgecolor='black', markeredgewidth=1)
            ax1.annotate(name, (e, n), fontsize=9, ha='center', va='bottom' if 'N' in name else 'top')

        ax1.set_title('UTM Space (Direct pyproj)\nRed dashed = envelope', fontweight='bold')
        ax1.set_xlabel('Easting (m)')
        ax1.set_ylabel('Northing (m)')
        ax1.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1))
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)

        # Panel 2: Reprojected pixel space
        ax2 = plt.subplot(1, 3, 2)

        # Transform to reprojected stage via pipeline
        edge_reprojected = transformer.transform_stage(edge_pixels_original, 'original', 'reprojected')

        rep_y = [p[0] for p in edge_reprojected]
        rep_x = [p[1] for p in edge_reprojected]
        ax2.scatter(rep_x, rep_y, c='blue', s=10, alpha=0.6, label='Boundary pixels')

        # Draw grid bounds
        rep_h, rep_w = pipeline.reprojected_shape
        rect = patches.Rectangle(
            (0, 0), rep_w, rep_h,
            linewidth=2, edgecolor='gray', facecolor='none', linestyle='--',
            label=f'Grid ({rep_h}×{rep_w})'
        )
        ax2.add_patch(rect)

        # Show corner positions
        corners_reproj = transformer.transform_stage(corners_pixel, 'original', 'reprojected')
        for (y, x), name in zip(corners_reproj, corner_names):
            ax2.plot(x, y, 'go', markersize=10, markeredgecolor='black', markeredgewidth=1)
            ax2.annotate(f'{name}\n({y},{x})', (x, y), fontsize=8, ha='center',
                        va='bottom' if 'N' in name else 'top')

        ax2.set_title('Reprojected Pixel Space\n(Pipeline transform)', fontweight='bold')
        ax2.set_xlabel('X pixels')
        ax2.set_ylabel('Y pixels')
        ax2.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1))
        ax2.set_xlim(-100, rep_w + 100)
        ax2.set_ylim(rep_h + 100, -100)
        ax2.grid(True, alpha=0.3)

        # Panel 3: Final mesh space (fractional vs integer)
        ax3 = plt.subplot(1, 3, 3)

        # Get both fractional and integer
        edge_final_int = transformer.transform_full_pipeline(edge_pixels_original)

        # Compute view bounds
        edge_y_frac = [p[0] for p in edge_fractional]
        edge_x_frac = [p[1] for p in edge_fractional]
        x_max_view = max(final_width, max(edge_x_frac) + 1)
        y_max_view = max(final_height, max(edge_y_frac) + 1)

        # Draw mesh grid
        for i in range(0, int(y_max_view) + 1):
            ax3.axhline(i, color='gray', linewidth=0.5, alpha=0.2)
        for i in range(0, int(x_max_view) + 1):
            ax3.axvline(i, color='gray', linewidth=0.5, alpha=0.2)

        # Draw grid boundary
        rect = patches.Rectangle(
            (0, 0), final_width - 1, final_height - 1,
            linewidth=2, edgecolor='red', facecolor='none', linestyle='--',
            label=f'Mesh ({final_height}×{final_width})'
        )
        ax3.add_patch(rect)

        # Draw fractional vertices (true curved boundary)
        ax3.scatter(edge_x_frac, edge_y_frac, c='blue', s=15, alpha=0.6,
                   label=f'Fractional ({len(edge_fractional)})')

        # Draw integer vertices
        edge_y_int = [p[0] for p in edge_final_int]
        edge_x_int = [p[1] for p in edge_final_int]
        ax3.scatter(edge_x_int, edge_y_int, c='green', s=40, alpha=0.8, marker='s',
                   label=f'Integer ({len(edge_final_int)})')

        # Show corner positions (fractional)
        corners_frac = transformer.transform_to_mesh_space(corners_pixel, 'original', clamp=False)
        for (y, x), name in zip(corners_frac, corner_names):
            ax3.annotate(f'{name}\n({y:.1f},{x:.1f})', (x, y), fontsize=8, ha='center',
                        va='bottom' if 'N' in name else 'top',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))

        ax3.set_title('Final Mesh Space\nBlue=fractional, Green=integer', fontweight='bold')
        ax3.set_xlabel('X (mesh units)')
        ax3.set_ylabel('Y (mesh units)')
        ax3.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1))
        ax3.set_xlim(-1, x_max_view)
        ax3.set_ylim(y_max_view, -1)
        ax3.grid(True, alpha=0.3)

        plt.suptitle('Transform Verification: Demo Settings (EPSG:4326 → EPSG:32617 + horizontal flip)',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        output_path = output_dir / "06_transform_verification.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {output_path}")

        # Verify corner positions are reasonable
        for (y, x), name in zip(corners_frac, corner_names):
            logger.info(f"Corner {name} in mesh space: ({y:.2f}, {x:.2f})")

        # Check that all corners are near expected positions
        # After flip: NW becomes NE, NE becomes NW, etc.
        assert corners_frac[0][0] >= -1 and corners_frac[0][0] <= 1, f"NW corner Y out of range: {corners_frac[0][0]}"
        assert corners_frac[2][0] >= final_height - 5, f"SW corner Y out of range: {corners_frac[2][0]}"

    def test_pixel_vertex_overlap(self, detroit_bounds, output_dir):
        """Visualize overlap between final integer pixels and fractional vertices.

        Shows how fractional vertices relate to integer grid pixels, helping
        verify that edge extrusion will connect properly.
        """
        from pyproj import Transformer as PyProjTransformer

        logger.info("\n" + "="*70)
        logger.info("PIXEL-VERTEX OVERLAP: Integer Grid vs Fractional Boundary")
        logger.info("="*70)

        original_shape = detroit_bounds['original_shape']
        bbox_wgs84 = detroit_bounds['bbox_wgs84']
        target_vertices = detroit_bounds['target_vertices']

        # Compute actual UTM bbox from corners
        src_crs = "EPSG:4326"
        dst_crs = "EPSG:32617"
        proj_transformer = PyProjTransformer.from_crs(src_crs, dst_crs, always_xy=True)

        min_lon, min_lat, max_lon, max_lat = bbox_wgs84
        scale_x_wgs84 = (max_lon - min_lon) / original_shape[1]
        scale_y_wgs84 = -(max_lat - min_lat) / original_shape[0]
        affine_wgs84 = SimpleAffine(
            c=min_lon, f=max_lat,
            a=scale_x_wgs84, e=scale_y_wgs84,
            b=0, d=0
        )

        corners_pixel = [
            (0, 0), (0, original_shape[1] - 1),
            (original_shape[0] - 1, 0), (original_shape[0] - 1, original_shape[1] - 1),
        ]
        corners_utm = []
        for y, x in corners_pixel:
            lon, lat = affine_wgs84.map_pixel_to_world(y, x)
            easting, northing = proj_transformer.transform(lon, lat)
            corners_utm.append((easting, northing))

        eastings = [c[0] for c in corners_utm]
        northings = [c[1] for c in corners_utm]
        actual_bbox_utm = (min(eastings), min(northings), max(eastings), max(northings))

        # Create pipeline
        pipeline = TransformationPipeline(
            original_shape=original_shape,
            distortion_factor=0.74,
            target_vertices=target_vertices,
            bbox_wgs84=bbox_wgs84,
            bbox_utm=actual_bbox_utm,
            src_crs=src_crs,
            dst_crs=dst_crs,
        )
        transformer = EdgeTransformer(pipeline, use_pyproj=True)

        final_height, final_width = pipeline.final_shape
        logger.info(f"Final mesh grid: {final_height}×{final_width}")

        # Sample dense edge pixels
        edge_pixels_original = sample_rectangle_edges(
            original_shape[0], original_shape[1], spacing=50
        )

        # Get fractional vertices (true curved boundary)
        edge_fractional = transformer.transform_to_mesh_space(edge_pixels_original, 'original', clamp=False)

        # Compute integer vertices by rounding fractional coords (NOT using clamped pipeline)
        # This shows which grid cells the curved boundary actually passes through
        edge_integer_from_frac = set()
        for y, x in edge_fractional:
            # Round to nearest integer, then clamp to valid grid range
            y_int = max(0, min(final_height - 1, int(round(y))))
            x_int = max(0, min(final_width - 1, int(round(x))))
            edge_integer_from_frac.add((y_int, x_int))
        edge_integer = list(edge_integer_from_frac)

        # Create detailed overlap visualization
        fig = plt.figure(figsize=(16, 14))

        # Panel 1: Full view showing both
        ax1 = plt.subplot(2, 2, 1)

        # Draw mesh grid
        for i in range(final_height + 1):
            ax1.axhline(i, color='lightgray', linewidth=0.3, alpha=0.5)
        for i in range(final_width + 1):
            ax1.axvline(i, color='lightgray', linewidth=0.3, alpha=0.5)

        # Draw grid boundary
        rect = patches.Rectangle(
            (0, 0), final_width - 1, final_height - 1,
            linewidth=2, edgecolor='red', facecolor='none', linestyle='--',
            label=f'Mesh grid ({final_height}×{final_width})'
        )
        ax1.add_patch(rect)

        # Draw fractional vertices
        frac_y = [p[0] for p in edge_fractional]
        frac_x = [p[1] for p in edge_fractional]
        ax1.scatter(frac_x, frac_y, c='blue', s=8, alpha=0.6,
                   label=f'Fractional ({len(edge_fractional)})')

        # Draw integer vertices
        int_y = [p[0] for p in edge_integer]
        int_x = [p[1] for p in edge_integer]
        ax1.scatter(int_x, int_y, c='green', s=30, alpha=0.8, marker='s',
                   label=f'Integer ({len(edge_integer)})')

        ax1.set_title('Full View: Integer vs Fractional Vertices', fontweight='bold')
        ax1.set_xlabel('X (mesh units)')
        ax1.set_ylabel('Y (mesh units)')
        ax1.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1))
        ax1.set_xlim(-2, max(final_width, max(frac_x)) + 2)
        ax1.set_ylim(max(final_height, max(frac_y)) + 2, -2)

        # Panel 2: Zoomed view of top-right corner (where curvature is most visible)
        ax2 = plt.subplot(2, 2, 2)

        # Zoom region: right edge where curvature extends beyond grid
        zoom_x_min = final_width - 20
        zoom_x_max = max(frac_x) + 2
        zoom_y_min = -2
        zoom_y_max = 25

        # Draw mesh grid in zoom area
        for i in range(int(zoom_y_min), int(zoom_y_max) + 1):
            ax2.axhline(i, color='lightgray', linewidth=0.5, alpha=0.7)
        for i in range(int(zoom_x_min), int(zoom_x_max) + 1):
            ax2.axvline(i, color='lightgray', linewidth=0.5, alpha=0.7)

        # Draw grid boundary
        ax2.axvline(final_width - 1, color='red', linewidth=2, linestyle='--', label='Grid edge')
        ax2.axhline(0, color='red', linewidth=2, linestyle='--')

        # Filter points in zoom region
        zoom_frac = [(y, x) for y, x in edge_fractional
                    if zoom_x_min <= x <= zoom_x_max and zoom_y_min <= y <= zoom_y_max]
        zoom_int = [(y, x) for y, x in edge_integer
                   if zoom_x_min <= x <= zoom_x_max and zoom_y_min <= y <= zoom_y_max]

        # Draw fractional vertices (larger in zoom)
        if zoom_frac:
            zf_y = [p[0] for p in zoom_frac]
            zf_x = [p[1] for p in zoom_frac]
            ax2.scatter(zf_x, zf_y, c='blue', s=40, alpha=0.7,
                       label=f'Fractional ({len(zoom_frac)})')

        # Draw integer vertices
        if zoom_int:
            zi_y = [p[0] for p in zoom_int]
            zi_x = [p[1] for p in zoom_int]
            ax2.scatter(zi_x, zi_y, c='green', s=80, alpha=0.8, marker='s',
                       label=f'Integer ({len(zoom_int)})')

        # Draw pixel centers as small dots
        for y in range(max(0, int(zoom_y_min)), min(final_height, int(zoom_y_max) + 1)):
            for x in range(max(0, int(zoom_x_min)), min(final_width, int(zoom_x_max) + 1)):
                ax2.plot(x, y, 'k.', markersize=2, alpha=0.3)

        ax2.set_title('Zoom: Top-Right Corner\n(Curvature extends beyond grid)', fontweight='bold')
        ax2.set_xlabel('X (mesh units)')
        ax2.set_ylabel('Y (mesh units)')
        ax2.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1))
        ax2.set_xlim(zoom_x_min, zoom_x_max)
        ax2.set_ylim(zoom_y_max, zoom_y_min)

        # Panel 3: Zoomed view of bottom-left corner
        ax3 = plt.subplot(2, 2, 3)

        zoom_x_min2 = -2
        zoom_x_max2 = 20
        zoom_y_min2 = final_height - 20
        zoom_y_max2 = max(frac_y) + 2

        # Draw mesh grid
        for i in range(int(zoom_y_min2), int(zoom_y_max2) + 1):
            ax3.axhline(i, color='lightgray', linewidth=0.5, alpha=0.7)
        for i in range(int(zoom_x_min2), int(zoom_x_max2) + 1):
            ax3.axvline(i, color='lightgray', linewidth=0.5, alpha=0.7)

        # Draw grid boundary
        ax3.axvline(0, color='red', linewidth=2, linestyle='--', label='Grid edge')
        ax3.axhline(final_height - 1, color='red', linewidth=2, linestyle='--')

        # Filter points
        zoom_frac2 = [(y, x) for y, x in edge_fractional
                     if zoom_x_min2 <= x <= zoom_x_max2 and zoom_y_min2 <= y <= zoom_y_max2]
        zoom_int2 = [(y, x) for y, x in edge_integer
                    if zoom_x_min2 <= x <= zoom_x_max2 and zoom_y_min2 <= y <= zoom_y_max2]

        if zoom_frac2:
            zf2_y = [p[0] for p in zoom_frac2]
            zf2_x = [p[1] for p in zoom_frac2]
            ax3.scatter(zf2_x, zf2_y, c='blue', s=40, alpha=0.7,
                       label=f'Fractional ({len(zoom_frac2)})')

        if zoom_int2:
            zi2_y = [p[0] for p in zoom_int2]
            zi2_x = [p[1] for p in zoom_int2]
            ax3.scatter(zi2_x, zi2_y, c='green', s=80, alpha=0.8, marker='s',
                       label=f'Integer ({len(zoom_int2)})')

        # Draw pixel centers
        for y in range(max(0, int(zoom_y_min2)), min(final_height, int(zoom_y_max2) + 1)):
            for x in range(max(0, int(zoom_x_min2)), min(final_width, int(zoom_x_max2) + 1)):
                ax3.plot(x, y, 'k.', markersize=2, alpha=0.3)

        ax3.set_title('Zoom: Bottom-Left Corner', fontweight='bold')
        ax3.set_xlabel('X (mesh units)')
        ax3.set_ylabel('Y (mesh units)')
        ax3.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1))
        ax3.set_xlim(zoom_x_min2, zoom_x_max2)
        ax3.set_ylim(zoom_y_max2, zoom_y_min2)

        # Panel 4: Distance analysis
        ax4 = plt.subplot(2, 2, 4)

        # For each fractional vertex, find distance to nearest integer vertex
        distances = []
        for fy, fx in edge_fractional:
            min_dist = float('inf')
            for iy, ix in edge_integer:
                dist = np.sqrt((fy - iy)**2 + (fx - ix)**2)
                if dist < min_dist:
                    min_dist = dist
            distances.append(min_dist)

        ax4.hist(distances, bins=30, color='purple', alpha=0.7, edgecolor='black')
        ax4.axvline(np.mean(distances), color='red', linestyle='--',
                   label=f'Mean: {np.mean(distances):.2f}')
        ax4.axvline(np.median(distances), color='orange', linestyle='--',
                   label=f'Median: {np.median(distances):.2f}')

        ax4.set_title('Distance: Fractional to Nearest Integer Vertex', fontweight='bold')
        ax4.set_xlabel('Distance (mesh units)')
        ax4.set_ylabel('Count')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)

        logger.info(f"Distance statistics:")
        logger.info(f"  Mean: {np.mean(distances):.3f}")
        logger.info(f"  Median: {np.median(distances):.3f}")
        logger.info(f"  Max: {np.max(distances):.3f}")
        logger.info(f"  Min: {np.min(distances):.3f}")

        plt.suptitle('Pixel-Vertex Overlap Analysis\nHow fractional vertices relate to integer grid',
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()

        output_path = output_dir / "07_pixel_vertex_overlap.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved: {output_path}")

    def test_all_stages_summary(self, detroit_bounds, output_dir):
        """Generate comprehensive summary of all stages (with pyproj)."""
        logger.info("\n" + "="*70)
        logger.info("FINAL SUMMARY: COMPLETE TRANSFORMATION PIPELINE (pyproj)")
        logger.info("="*70)

        original_shape = detroit_bounds['original_shape']
        bbox_wgs84 = detroit_bounds['bbox_wgs84']
        bbox_utm = detroit_bounds['bbox_utm']
        target_vertices = detroit_bounds['target_vertices']

        # Create pipeline with pyproj support for consistent calculations
        pipeline = TransformationPipeline(
            original_shape=original_shape,
            distortion_factor=0.74,
            target_vertices=target_vertices,
            bbox_wgs84=bbox_wgs84,
            bbox_utm=bbox_utm,
        )

        logger.info("")
        logger.info(f"Stage 1 (WGS84):        {original_shape[0]:>8,} × {original_shape[1]:<8,} pixels")
        logger.info(f"Stage 2 (UTM):          {pipeline.reprojected_shape[0]:>8,} × {pipeline.reprojected_shape[1]:<8,} pixels")
        logger.info(f"Stage 3 (Flipped):      {pipeline.flipped_shape[0]:>8,} × {pipeline.flipped_shape[1]:<8,} pixels")
        logger.info(f"Stage 4 (Final Mesh):   {pipeline.final_shape[0]:>8} × {pipeline.final_shape[1]:<8} pixels")
        logger.info("")
        logger.info(f"Total vertices: {pipeline.final_shape[0] * pipeline.final_shape[1]:,}")
        logger.info(f"Projection distortion: {abs(1-pipeline.distortion_factor)*100:.1f}%")
        logger.info("")
        logger.info(f"✓ All plots saved to {output_dir}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
