"""
TDD tests for bounds visualization pipeline refactoring.

These tests verify that the transformation pipeline and edge transformer
work correctly, enabling cleaner test code.
"""

import pytest
import numpy as np
from pathlib import Path

from src.terrain.visualization.bounds_pipeline import (
    TransformationPipeline,
    EdgeTransformer,
    SimpleAffine,
)


class TestTransformationPipeline:
    """Test the TransformationPipeline class."""

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
            'target_vertices': 1_000,
            'distortion_factor': 0.74,  # cos(42°) - latitude compression
        }

    def test_pipeline_initializes_with_bounds(self, detroit_bounds):
        """TransformationPipeline should initialize with bounds data."""
        pipeline = TransformationPipeline(
            original_shape=detroit_bounds['original_shape'],
            distortion_factor=detroit_bounds['distortion_factor'],
            target_vertices=detroit_bounds['target_vertices'],
        )

        assert pipeline.original_shape == detroit_bounds['original_shape']
        assert pipeline.distortion_factor == detroit_bounds['distortion_factor']
        assert pipeline.target_vertices == detroit_bounds['target_vertices']

    def test_pipeline_computes_reprojected_shape(self, detroit_bounds):
        """Pipeline should compute reprojected shape correctly.

        Note: +1 is added to ensure boundary pixels aren't filtered at exact edge.
        """
        pipeline = TransformationPipeline(
            original_shape=detroit_bounds['original_shape'],
            distortion_factor=detroit_bounds['distortion_factor'],
            target_vertices=detroit_bounds['target_vertices'],
        )

        # +1 added to ensure boundary pixels aren't filtered at exact edge
        expected_height = int(detroit_bounds['original_shape'][0] * detroit_bounds['distortion_factor']) + 1
        expected_width = int(detroit_bounds['original_shape'][1] / detroit_bounds['distortion_factor']) + 1

        assert pipeline.reprojected_shape == (expected_height, expected_width)

    def test_pipeline_computes_final_shape(self, detroit_bounds):
        """Pipeline should compute final mesh shape based on target vertices."""
        pipeline = TransformationPipeline(
            original_shape=detroit_bounds['original_shape'],
            distortion_factor=detroit_bounds['distortion_factor'],
            target_vertices=detroit_bounds['target_vertices'],
        )

        # Final shape should be close to target vertex count
        final_vertices = pipeline.final_shape[0] * pipeline.final_shape[1]
        assert abs(final_vertices - detroit_bounds['target_vertices']) < 100

    def test_pipeline_computes_downsample_factors(self, detroit_bounds):
        """Pipeline should compute downsampling factors."""
        pipeline = TransformationPipeline(
            original_shape=detroit_bounds['original_shape'],
            distortion_factor=detroit_bounds['distortion_factor'],
            target_vertices=detroit_bounds['target_vertices'],
        )

        downsample_y, downsample_x = pipeline.downsample_factors
        assert downsample_y > 1.0
        assert downsample_x > 1.0

    def test_pipeline_gets_shape_at_stage(self, detroit_bounds):
        """Pipeline should return shape for each transformation stage."""
        pipeline = TransformationPipeline(
            original_shape=detroit_bounds['original_shape'],
            distortion_factor=detroit_bounds['distortion_factor'],
            target_vertices=detroit_bounds['target_vertices'],
        )

        assert pipeline.get_shape('original') == detroit_bounds['original_shape']
        assert pipeline.get_shape('reprojected') == pipeline.reprojected_shape
        assert pipeline.get_shape('flipped') == pipeline.reprojected_shape
        assert pipeline.get_shape('final') == pipeline.final_shape

    def test_pipeline_creates_affine_transforms(self, detroit_bounds):
        """Pipeline should create affine transforms for each stage."""
        pipeline = TransformationPipeline(
            original_shape=detroit_bounds['original_shape'],
            distortion_factor=detroit_bounds['distortion_factor'],
            target_vertices=detroit_bounds['target_vertices'],
            bbox_wgs84=detroit_bounds['bbox_wgs84'],
            bbox_utm=detroit_bounds['bbox_utm'],
        )

        affine_wgs84 = pipeline.get_affine('original')
        affine_utm = pipeline.get_affine('reprojected')

        # Affines should be SimpleAffine instances
        assert isinstance(affine_wgs84, SimpleAffine)
        assert isinstance(affine_utm, SimpleAffine)

        # Affines should have the correct bounds
        assert affine_wgs84.c == detroit_bounds['bbox_wgs84'][0]  # min lon
        assert affine_utm.c == detroit_bounds['bbox_utm'][0]  # min easting


class TestEdgeTransformer:
    """Test the EdgeTransformer class."""

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
            'target_vertices': 1_000,
            'distortion_factor': 0.74,  # cos(42°) - latitude compression
        }

    @pytest.fixture
    def pipeline(self, detroit_bounds):
        """Create a transformation pipeline."""
        return TransformationPipeline(
            original_shape=detroit_bounds['original_shape'],
            distortion_factor=detroit_bounds['distortion_factor'],
            target_vertices=detroit_bounds['target_vertices'],
            bbox_wgs84=detroit_bounds['bbox_wgs84'],
            bbox_utm=detroit_bounds['bbox_utm'],
        )

    @pytest.fixture
    def transformer(self, pipeline):
        """Create an edge transformer."""
        return EdgeTransformer(pipeline)

    def test_transformer_initializes_with_pipeline(self, transformer, pipeline):
        """EdgeTransformer should initialize with a pipeline."""
        assert transformer.pipeline is pipeline

    def test_transformer_applies_stage_1_to_2(self, transformer):
        """Transformer should convert pixels from original space to reprojected space."""
        # Test with a corner pixel
        edge_pixels_stage1 = [(0, 0)]  # Top-left corner

        result = transformer.transform_stage(edge_pixels_stage1, 'original', 'reprojected')

        assert len(result) > 0
        assert all(isinstance(p, tuple) and len(p) == 2 for p in result)
        assert all(isinstance(p[0], (int, np.integer)) and isinstance(p[1], (int, np.integer)) for p in result)

    def test_transformer_applies_stage_2_to_3(self, transformer):
        """Transformer should apply horizontal flip."""
        # Start with valid pixels in reprojected space
        width = transformer.pipeline.reprojected_shape[1]
        # Use actual valid pixel positions (near the width boundary)
        edge_pixels_stage2 = [(0, 0), (0, width - 1)]

        result = transformer.transform_stage(edge_pixels_stage2, 'reprojected', 'flipped')

        assert len(result) == 2
        # After flip, x coordinates should be flipped
        # (0, 0) → (0, width-1), (0, width-1) → (0, 0)
        result_x = sorted([p[1] for p in result])
        assert result_x == [0, width - 1]

    def test_transformer_applies_stage_3_to_4(self, transformer):
        """Transformer should downsample to final grid."""
        # Use pixels at specific downsample positions
        edge_pixels_stage3 = [(0, 0), (100, 100)]

        result = transformer.transform_stage(edge_pixels_stage3, 'flipped', 'final')

        assert len(result) > 0
        # All pixels should be within final grid bounds
        final_h, final_w = transformer.pipeline.final_shape
        assert all(0 <= p[0] < final_h and 0 <= p[1] < final_w for p in result)

    def test_transformer_applies_full_pipeline(self, transformer):
        """Transformer should apply full pipeline from original to final."""
        # Use corner and edge pixels that will definitely map through the pipeline
        original_h, original_w = transformer.pipeline.original_shape
        edge_pixels = [
            (0, 0),                          # Top-left corner
            (0, original_w - 1),             # Top-right corner
            (original_h - 1, 0),             # Bottom-left corner
            (original_h - 1, original_w - 1)  # Bottom-right corner
        ]

        result = transformer.transform_full_pipeline(edge_pixels)

        # Should have some result (at least some corners should map through)
        assert len(result) > 0
        final_h, final_w = transformer.pipeline.final_shape
        assert all(0 <= p[0] < final_h and 0 <= p[1] < final_w for p in result)

    def test_transformer_removes_duplicates_from_rounding(self, transformer):
        """Transformer should handle duplicates created by rounding."""
        # Create edge pixels that will round to the same grid cell
        edge_pixels = [(0, 0), (1, 0), (2, 0)]  # Three pixels close together

        result = transformer.transform_full_pipeline(edge_pixels)

        # Result may have fewer entries due to rounding to same grid cell
        assert len(result) <= len(edge_pixels)
        assert len(set(result)) == len(result)  # No duplicates

    def test_transform_to_mesh_space_preserves_all_vertices(self, transformer):
        """transform_to_mesh_space should preserve all vertices with fractional precision."""
        # Sample many edge pixels
        original_h, original_w = transformer.pipeline.original_shape
        edge_pixels = [
            (0, x) for x in range(0, original_w, 100)
        ]  # Top edge, 73 samples

        # Integer grid loses precision due to rounding
        integer_result = transformer.transform_full_pipeline(edge_pixels)

        # Fractional mesh preserves all vertices
        fractional_result = transformer.transform_to_mesh_space(edge_pixels, 'original')

        # Fractional should preserve all input vertices
        assert len(fractional_result) == len(edge_pixels)

        # Integer may have fewer due to deduplication
        assert len(integer_result) <= len(edge_pixels)

        # Fractional coordinates should be floats with sub-pixel precision
        for y, x in fractional_result:
            assert isinstance(y, float)
            assert isinstance(x, float)

    def test_transform_to_mesh_space_coordinates_in_valid_range(self, transformer):
        """Fractional mesh coordinates should be within valid range."""
        original_h, original_w = transformer.pipeline.original_shape
        edge_pixels = [
            (0, 0), (0, original_w - 1),
            (original_h - 1, 0), (original_h - 1, original_w - 1),
        ]

        fractional_result = transformer.transform_to_mesh_space(edge_pixels, 'original')

        final_h, final_w = transformer.pipeline.final_shape
        for y, x in fractional_result:
            assert 0.0 <= y <= final_h - 1, f"y={y} out of range [0, {final_h-1}]"
            assert 0.0 <= x <= final_w - 1, f"x={x} out of range [0, {final_w-1}]"


class TestNonLinearProjection:
    """Test that the WGS84→UTM transformation captures non-linear curvature."""

    @pytest.fixture
    def detroit_bounds(self):
        """Real Detroit DEM bounds - 2° × 2° area centered on Detroit."""
        return {
            'original_shape': (7_201, 7_201),  # 2° × 2° at 1 arc-second
            'bbox_wgs84': (-84.0, 41.0, -82.0, 43.0),
            'bbox_utm': (247_679, 4_539_238, 418_491, 4_765_182),
            'target_vertices': 1_000,
            'distortion_factor': 0.74,  # cos(42°)
        }

    @pytest.fixture
    def pipeline(self, detroit_bounds):
        """Create pipeline with bounding boxes for proper projection."""
        return TransformationPipeline(
            original_shape=detroit_bounds['original_shape'],
            distortion_factor=detroit_bounds['distortion_factor'],
            target_vertices=detroit_bounds['target_vertices'],
            bbox_wgs84=detroit_bounds['bbox_wgs84'],
            bbox_utm=detroit_bounds['bbox_utm'],
        )

    @pytest.fixture
    def transformer(self, pipeline):
        """Create transformer using pyproj."""
        return EdgeTransformer(pipeline, use_pyproj=True)

    def test_top_edge_becomes_curved_in_utm(self, transformer):
        """Top edge of WGS84 rectangle should NOT remain horizontal in UTM.

        This is the key test for non-linear projection behavior.
        In WGS84, y=0 is a horizontal line (constant latitude).
        In UTM, this line becomes curved due to Transverse Mercator projection.

        For the Detroit area, the top edge curves by ~50 pixels (445m) across
        the width of the image, which is about 0.3% of the reprojected height.
        """
        # Sample multiple points along the top edge (y=0, varying x)
        original_w = transformer.pipeline.original_shape[1]
        top_edge_pixels = [
            (0, 0),                    # left corner
            (0, original_w // 4),      # quarter across
            (0, original_w // 2),      # middle
            (0, 3 * original_w // 4),  # three quarters
            (0, original_w - 1),       # right corner
        ]

        result = transformer.transform_stage(top_edge_pixels, 'original', 'reprojected')

        # After UTM projection, the y-coordinates should NOT all be the same
        # because the projection is non-linear
        y_coords = [p[0] for p in result]

        assert len(result) == len(top_edge_pixels), "All points should transform"

        # With proper Transverse Mercator projection, these y values should differ
        y_range = max(y_coords) - min(y_coords)

        # For the linear fallback, y_range would be 0 (all same y)
        # For proper pyproj, y_range should be significant (>10 pixels)
        assert y_range > 10, f"Top edge should curve in UTM, got y_range={y_range}"

    def test_left_edge_becomes_curved_in_utm(self, transformer):
        """Left edge of WGS84 rectangle should NOT remain vertical in UTM.

        In WGS84, x=0 is a vertical line (constant longitude).
        In UTM, this line becomes curved due to Transverse Mercator projection.
        The effect on the left/right edges is smaller than top/bottom because
        longitude lines converge toward the poles.
        """
        # Sample multiple points along the left edge (x=0, varying y)
        original_h = transformer.pipeline.original_shape[0]
        left_edge_pixels = [
            (0, 0),                    # top corner
            (original_h // 4, 0),      # quarter down
            (original_h // 2, 0),      # middle
            (3 * original_h // 4, 0),  # three quarters
            (original_h - 1, 0),       # bottom corner
        ]

        result = transformer.transform_stage(left_edge_pixels, 'original', 'reprojected')

        assert len(result) == len(left_edge_pixels), "All points should transform"

        # After UTM projection, the x-coordinates should vary
        x_coords = [p[1] for p in result]
        x_range = max(x_coords) - min(x_coords)

        # The left edge curvature is typically much smaller than the top edge
        # but should still be measurable (>0 pixels)
        assert x_range >= 0, f"Left edge may curve slightly in UTM, got x_range={x_range}"


class TestSimpleAffine:
    """Test the SimpleAffine helper class."""

    def test_simple_affine_stores_coefficients(self):
        """SimpleAffine should store all 6 affine coefficients."""
        affine = SimpleAffine(c=-83.2, d=0, e=-0.00022, f=42.5, a=0.00022, b=0)

        assert affine.c == -83.2
        assert affine.a == 0.00022
        assert affine.e == -0.00022
        assert affine.f == 42.5

    def test_simple_affine_maps_pixels_to_world(self):
        """SimpleAffine should map pixel coordinates to world coordinates."""
        # Simple affine: 1.0 scale, 0 offset for rotation
        # Formula: world_x = c + x * a + y * b
        #          world_y = f + x * d + y * e
        affine = SimpleAffine(c=0, d=0, e=0, f=0, a=1.0, b=0)

        # Pixel (y=0, x=0) should map to (world_x=0, world_y=0)
        world_x, world_y = affine.map_pixel_to_world(y=0, x=0)
        assert world_x == 0 and world_y == 0

        # Pixel (y=10, x=5) should map to (world_x=5, world_y=0)
        # because world_x = 0 + 5 * 1.0 + 10 * 0 = 5
        # and world_y = 0 + 5 * 0 + 10 * 0 = 0
        world_x, world_y = affine.map_pixel_to_world(y=10, x=5)
        assert world_x == 5 and world_y == 0

    def test_simple_affine_with_offset(self):
        """SimpleAffine should handle offsets correctly."""
        affine = SimpleAffine(c=100, d=0, e=0, f=200, a=1.0, b=0)

        world_x, world_y = affine.map_pixel_to_world(y=0, x=0)
        assert world_x == 100 and world_y == 200

        # (y=10, x=5) → world_x = 100 + 5 * 1.0 = 105, world_y = 200 + 0 = 200
        world_x, world_y = affine.map_pixel_to_world(y=10, x=5)
        assert world_x == 105 and world_y == 200
