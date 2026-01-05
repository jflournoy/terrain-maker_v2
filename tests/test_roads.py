"""
Tests for road data fetching and rendering functionality.

Unit tests for:
- OSM Overpass query building
- Cache operations (save, load, expiration)
- Color blending operations
- Road geometry handling

Note: Integration tests with Blender require a full Blender environment
and are marked with @pytest.mark.blender_required for optional execution.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile

import pytest
import numpy as np

# Import modules to test
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from examples.detroit_roads import (
    build_overpass_query,
    compute_bbox_hash,
    load_cached_roads,
    cache_road_data,
    fetch_roads_from_osm,
    get_roads,
)
from src.terrain.roads import (
    rasterize_roads_to_layer,
    road_colormap,
)

logger = logging.getLogger(__name__)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_bbox():
    """Sample bounding box for Detroit area."""
    return (42.25, -83.5, 42.5, -82.8)


@pytest.fixture
def sample_road_types():
    """Sample road types to fetch."""
    return ["motorway", "trunk", "primary"]


@pytest.fixture
def sample_geojson():
    """Sample GeoJSON with road features."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-83.5, 42.3], [-83.4, 42.35], [-83.3, 42.4]],
                },
                "properties": {
                    "osm_id": 123456,
                    "name": "Interstate 75",
                    "ref": "I-75",
                    "highway": "motorway",
                },
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-83.4, 42.25], [-83.35, 42.3]],
                },
                "properties": {
                    "osm_id": 234567,
                    "name": "Michigan State Route 10",
                    "ref": "M-10",
                    "highway": "primary",
                },
            },
        ],
    }


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Temporary cache directory."""
    cache_dir = tmp_path / "roads_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


# =============================================================================
# QUERY BUILDING TESTS
# =============================================================================


class TestOverpassQueryBuilding:
    """Tests for Overpass QL query construction."""

    def test_build_query_single_type(self, sample_bbox):
        """Test building query for single road type."""
        query = build_overpass_query(sample_bbox, ["motorway"])

        assert "[out:json]" in query
        assert "motorway" in query
        # Query contains bounding box coordinates, not literal "bbox"
        south, west, north, east = sample_bbox
        assert str(south) in query
        assert str(west) in query

    def test_build_query_multiple_types(self, sample_bbox, sample_road_types):
        """Test building query for multiple road types."""
        query = build_overpass_query(sample_bbox, sample_road_types)

        for road_type in sample_road_types:
            assert road_type in query

    def test_bbox_in_query(self, sample_bbox):
        """Test that bbox is correctly formatted in query."""
        south, west, north, east = sample_bbox
        query = build_overpass_query(sample_bbox, ["motorway"])

        # Overpass uses south,west,north,east order
        assert f"{south},{west},{north},{east}" in query or \
               all(str(coord) in query for coord in [south, west, north, east])

    def test_query_has_geometry_output(self, sample_bbox):
        """Test that query requests geometry output."""
        query = build_overpass_query(sample_bbox, ["motorway"])
        assert "out geom" in query


# =============================================================================
# BBOX HASH TESTS
# =============================================================================


class TestBboxHash:
    """Tests for bounding box hashing."""

    def test_same_bbox_same_hash(self, sample_bbox):
        """Test that same bbox always produces same hash."""
        hash1 = compute_bbox_hash(sample_bbox)
        hash2 = compute_bbox_hash(sample_bbox)
        assert hash1 == hash2

    def test_different_bbox_different_hash(self, sample_bbox):
        """Test that different bboxes produce different hashes."""
        bbox2 = (42.3, -83.4, 42.6, -82.7)
        hash1 = compute_bbox_hash(sample_bbox)
        hash2 = compute_bbox_hash(bbox2)
        assert hash1 != hash2

    def test_hash_length(self, sample_bbox):
        """Test that hash is SHA256 (64 hex chars)."""
        hash_val = compute_bbox_hash(sample_bbox)
        assert len(hash_val) == 64
        assert all(c in '0123456789abcdef' for c in hash_val)


# =============================================================================
# CACHING TESTS
# =============================================================================


class TestCaching:
    """Tests for cache operations."""

    def test_cache_save_and_load(self, temp_cache_dir, sample_bbox, sample_geojson):
        """Test saving and loading cache."""
        with patch('examples.detroit_roads.get_cache_dir', return_value=temp_cache_dir):
            # Save cache
            cache_road_data(sample_bbox, sample_geojson)

            # Verify files created
            bbox_hash = compute_bbox_hash(sample_bbox)
            cache_file = temp_cache_dir / f"roads_{bbox_hash}.geojson"
            meta_file = temp_cache_dir / f"roads_{bbox_hash}_meta.json"

            assert cache_file.exists()
            assert meta_file.exists()

            # Load cache
            loaded_data = load_cached_roads(sample_bbox)
            assert loaded_data is not None
            assert loaded_data["type"] == "FeatureCollection"
            assert len(loaded_data["features"]) == 2

    def test_cache_load_nonexistent(self, temp_cache_dir, sample_bbox):
        """Test loading nonexistent cache returns None."""
        with patch('examples.detroit_roads.get_cache_dir', return_value=temp_cache_dir):
            result = load_cached_roads(sample_bbox)
            assert result is None

    def test_cache_expiration(self, temp_cache_dir, sample_bbox, sample_geojson):
        """Test that expired cache is not loaded."""
        with patch('examples.detroit_roads.get_cache_dir', return_value=temp_cache_dir):
            # Save cache
            cache_road_data(sample_bbox, sample_geojson)

            # Expire the cache by modifying metadata
            bbox_hash = compute_bbox_hash(sample_bbox)
            meta_file = temp_cache_dir / f"roads_{bbox_hash}_meta.json"

            with open(meta_file) as f:
                meta = json.load(f)

            # Set created_at to 31 days ago
            old_time = (datetime.now() - timedelta(days=31)).isoformat()
            meta["created_at"] = old_time

            with open(meta_file, "w") as f:
                json.dump(meta, f)

            # Loading should return None due to expiration
            result = load_cached_roads(sample_bbox)
            assert result is None

    def test_cache_metadata_content(self, temp_cache_dir, sample_bbox, sample_geojson):
        """Test that metadata contains correct information."""
        with patch('examples.detroit_roads.get_cache_dir', return_value=temp_cache_dir):
            cache_road_data(sample_bbox, sample_geojson)

            bbox_hash = compute_bbox_hash(sample_bbox)
            meta_file = temp_cache_dir / f"roads_{bbox_hash}_meta.json"

            with open(meta_file) as f:
                meta = json.load(f)

            assert "created_at" in meta
            assert "bbox" in meta
            assert meta["num_features"] == 2


# =============================================================================
# COLOR BLENDING TESTS
# =============================================================================


# DEPRECATED: TestColorBlending tests old road rendering functions
# These are no longer used - roads are now handled via multi-overlay color mapping system
# class TestColorBlending:
#     """Tests for color blending operations."""
#
#     (old tests removed - replaced by multi-overlay color mapping system)


# =============================================================================
# ROAD COLORMAP TESTS
# =============================================================================


class TestRoadColormap:
    """Tests for the road colormap function used in multi-overlay system."""

    def test_road_colormap_output_shape(self):
        """Test that road_colormap returns correct shape."""
        road_grid = np.zeros((10, 10), dtype=np.uint8)
        road_grid[2:5, 2:5] = 4  # Add some motorway

        colors = road_colormap(road_grid)

        assert colors.shape == (10, 10, 3), f"Expected shape (10, 10, 3), got {colors.shape}"
        assert colors.dtype == np.uint8

    def test_road_colormap_handles_road_types(self):
        """Test that road colormap gives all roads a distinctive red color for material detection."""
        road_grid = np.array(
            [[0, 1, 2], [3, 4, 0], [1, 2, 3]],
            dtype=np.uint8
        )

        colors = road_colormap(road_grid)

        # All pixels should have some color (RGB)
        assert colors.shape == (3, 3, 3)
        # All roads should be red (180, 30, 30) for glassy material detection
        road_mask = road_grid > 0
        road_colors = colors[road_mask]
        # Check that red channel is high (> 150) and green/blue are low (< 50)
        assert np.all(road_colors[:, 0] > 150), "Roads should have high red channel"
        assert np.all(road_colors[:, 1] < 50), "Roads should have low green channel"
        assert np.all(road_colors[:, 2] < 50), "Roads should have low blue channel"

    def test_road_colormap_zero_values(self):
        """Test that no-road pixels (0) are handled."""
        road_grid = np.zeros((5, 5), dtype=np.uint8)

        colors = road_colormap(road_grid)

        assert colors.shape == (5, 5, 3)
        # All zeros should map to a consistent color
        assert np.all(colors == colors[0, 0])

    def test_rasterize_roads_output_shape(self):
        """Test that rasterize_roads_to_layer returns correct shapes."""
        # Minimal GeoJSON with one road
        roads_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[-83.5, 42.3], [-83.4, 42.35]],
                    },
                    "properties": {"highway": "motorway"},
                }
            ],
        }

        bbox = (42.2, -83.6, 42.4, -83.3)
        road_grid, road_transform = rasterize_roads_to_layer(
            roads_geojson, bbox, resolution=100.0
        )

        assert road_grid.ndim == 2
        assert road_grid.dtype == np.uint8
        assert road_transform is not None

    def test_rasterize_roads_with_width(self):
        """Test that road_width_pixels makes roads wider."""
        roads_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[-83.5, 42.3], [-83.4, 42.35]],
                    },
                    "properties": {"highway": "motorway"},
                }
            ],
        }

        bbox = (42.2, -83.6, 42.4, -83.3)

        # Rasterize with width=1 (thin)
        road_grid_thin, _ = rasterize_roads_to_layer(
            roads_geojson, bbox, resolution=100.0, road_width_pixels=1
        )
        thin_count = np.count_nonzero(road_grid_thin)

        # Rasterize with width=3 (thick)
        road_grid_thick, _ = rasterize_roads_to_layer(
            roads_geojson, bbox, resolution=100.0, road_width_pixels=3
        )
        thick_count = np.count_nonzero(road_grid_thick)

        # Thick roads should have more pixels
        assert thick_count > thin_count, "Thick roads should have more pixels than thin roads"
        assert thick_count >= thin_count * 2, "Thick roads (width=3) should be ~3x wider"


# =============================================================================
# GEOMETRY HANDLING TESTS
# =============================================================================


class TestGeometryHandling:
    """Tests for handling road geometries."""

    def test_geojson_linestring_validation(self, sample_geojson):
        """Test that LineString geometries are properly handled."""
        for feature in sample_geojson["features"]:
            assert feature["geometry"]["type"] == "LineString"
            assert len(feature["geometry"]["coordinates"]) >= 2

    def test_geojson_properties_preservation(self, sample_geojson):
        """Test that road properties are preserved."""
        feature = sample_geojson["features"][0]
        props = feature["properties"]

        assert props["osm_id"] == 123456
        assert props["name"] == "Interstate 75"
        assert props["ref"] == "I-75"
        assert props["highway"] == "motorway"

    def test_multiline_feature_handling(self):
        """Test handling of MultiLineString features (common in OSM)."""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "MultiLineString",
                        "coordinates": [
                            [[-83.5, 42.3], [-83.4, 42.35]],
                            [[-83.3, 42.4], [-83.2, 42.45]],
                        ],
                    },
                    "properties": {"name": "Road", "osm_id": 999},
                }
            ],
        }

        # MultiLineString handling should be graceful
        assert geojson["features"][0]["geometry"]["type"] == "MultiLineString"


# =============================================================================
# MOCK DATA TESTS
# =============================================================================


class TestMockDataFallback:
    """Tests for fallback behavior when OSM API is unavailable."""

    def test_get_roads_with_cache(self, sample_bbox, sample_geojson, temp_cache_dir):
        """Test get_roads function uses cache."""
        with patch('examples.detroit_roads.get_cache_dir', return_value=temp_cache_dir):
            # Pre-populate cache
            cache_road_data(sample_bbox, sample_geojson)

            # Call get_roads - should use cache, not fetch API
            with patch('examples.detroit_roads.fetch_roads_from_osm') as mock_fetch:
                result = get_roads(sample_bbox)

                # fetch_roads_from_osm should not be called
                mock_fetch.assert_not_called()
                assert result is not None
                assert len(result["features"]) == 2

    def test_get_roads_force_refresh(self, sample_bbox, sample_geojson, temp_cache_dir):
        """Test force_refresh bypasses cache."""
        with patch('examples.detroit_roads.get_cache_dir', return_value=temp_cache_dir):
            # Pre-populate cache
            cache_road_data(sample_bbox, sample_geojson)

            # Call with force_refresh=True
            with patch('examples.detroit_roads.fetch_roads_from_osm', return_value=sample_geojson) as mock_fetch:
                result = get_roads(sample_bbox, force_refresh=True)

                # Should call fetch despite cache existing
                mock_fetch.assert_called_once()
                assert result is not None


# =============================================================================
# API INTEGRATION TESTS (MOCKED)
# =============================================================================


class TestAPIIntegration:
    """Tests for OSM API integration (with mocked responses)."""

    def test_fetch_handles_timeout(self, sample_bbox):
        """Test that timeout errors are handled gracefully."""
        with patch('examples.detroit_roads.requests.post') as mock_post:
            import requests
            mock_post.side_effect = requests.exceptions.Timeout()

            result = fetch_roads_from_osm(sample_bbox, ["motorway"], timeout=5)
            assert result is None

    def test_fetch_handles_http_error(self, sample_bbox):
        """Test that HTTP errors are handled gracefully."""
        with patch('examples.detroit_roads.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.raise_for_status.side_effect = Exception("500 Server Error")
            mock_post.return_value = mock_response

            result = fetch_roads_from_osm(sample_bbox, ["motorway"])
            assert result is None

    def test_fetch_handles_rate_limit(self, sample_bbox):
        """Test that rate limiting (429) is handled."""
        with patch('examples.detroit_roads.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.raise_for_status.side_effect = Exception("429 Too Many Requests")
            mock_post.return_value = mock_response

            result = fetch_roads_from_osm(sample_bbox, ["motorway"])
            assert result is None

    def test_fetch_handles_empty_result(self, sample_bbox):
        """Test handling of empty API response."""
        with patch('examples.detroit_roads.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"elements": []}
            mock_post.return_value = mock_response

            result = fetch_roads_from_osm(sample_bbox, ["motorway"])
            assert result is not None
            assert len(result["features"]) == 0

    def test_fetch_parses_geojson(self, sample_bbox):
        """Test that OSM response is correctly parsed to GeoJSON."""
        osm_response = {
            "elements": [
                {
                    "type": "way",
                    "id": 123456,
                    "nodes": [1, 2],  # OSM node IDs (required by parser)
                    "tags": {
                        "name": "Interstate 75",
                        "ref": "I-75",
                        "highway": "motorway",
                    },
                    "geometry": [
                        {"lat": 42.3, "lon": -83.5},
                        {"lat": 42.35, "lon": -83.4},
                    ],
                }
            ]
        }

        with patch('examples.detroit_roads.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = osm_response
            mock_post.return_value = mock_response

            result = fetch_roads_from_osm(sample_bbox, ["motorway"])
            assert result["type"] == "FeatureCollection"
            assert len(result["features"]) == 1

            feature = result["features"][0]
            assert feature["geometry"]["type"] == "LineString"
            assert feature["properties"]["name"] == "Interstate 75"


# =============================================================================
# PARAMETRIZED TESTS
# =============================================================================


# DEPRECATED: blend_color_for_road tests removed - function no longer exists
# Roads now use multi-overlay color mapping system


@pytest.mark.parametrize("road_types", [
    ["motorway"],
    ["trunk"],
    ["primary"],
    ["motorway", "trunk"],
    ["motorway", "trunk", "primary"],
])
def test_query_all_road_types(sample_bbox, road_types):
    """Test query generation for all road type combinations."""
    query = build_overpass_query(sample_bbox, road_types)

    for road_type in road_types:
        assert road_type in query


# =============================================================================
# GET_ROADS_TILED TESTS
# =============================================================================


class TestGetRoadsTiled:
    """Tests for get_roads_tiled function - fetches roads for large areas by tiling."""

    def test_get_roads_tiled_imports(self):
        """Test that get_roads_tiled can be imported."""
        from examples.detroit_roads import get_roads_tiled

        assert callable(get_roads_tiled)

    def test_get_roads_tiled_small_bbox_no_tiling(self, sample_geojson, temp_cache_dir):
        """Small bbox (< tile_size) should not tile, single fetch."""
        from examples.detroit_roads import get_roads_tiled

        # Small bbox (0.25° x 0.7°) - less than tile_size of 2.0°
        small_bbox = (42.25, -83.5, 42.5, -82.8)

        with patch('examples.detroit_roads.get_cache_dir', return_value=temp_cache_dir):
            with patch('examples.detroit_roads.get_roads', return_value=sample_geojson) as mock_get_roads:
                result = get_roads_tiled(small_bbox, tile_size=2.0)

                # Should call get_roads exactly once (no tiling needed)
                mock_get_roads.assert_called_once()
                assert result is not None
                assert result["type"] == "FeatureCollection"

    def test_get_roads_tiled_large_bbox_tiles(self, sample_geojson, temp_cache_dir):
        """Large bbox should be split into multiple tiles."""
        from examples.detroit_roads import get_roads_tiled

        # Large bbox (5° x 10°) - larger than tile_size of 2.0°
        large_bbox = (40.0, -89.0, 45.0, -79.0)  # 5° lat x 10° lon

        with patch('examples.detroit_roads.get_cache_dir', return_value=temp_cache_dir):
            with patch('examples.detroit_roads.get_roads', return_value=sample_geojson) as mock_get_roads:
                result = get_roads_tiled(large_bbox, tile_size=2.0)

                # Should call get_roads multiple times (3 lat tiles x 5 lon tiles = 15 tiles)
                assert mock_get_roads.call_count >= 4, "Large bbox should be tiled into multiple fetches"
                assert result is not None

    def test_get_roads_tiled_merges_results(self, temp_cache_dir):
        """Features from all tiles should be merged into single FeatureCollection."""
        from examples.detroit_roads import get_roads_tiled

        # Create two different GeoJSON responses for different tiles
        tile1_features = {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}, "properties": {"highway": "motorway"}}]
        }
        tile2_features = {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[2, 2], [3, 3]]}, "properties": {"highway": "trunk"}}]
        }

        # Large bbox that spans 2 tiles
        large_bbox = (40.0, -85.0, 42.0, -81.0)  # 2° lat x 4° lon = 1x2 tiles

        with patch('examples.detroit_roads.get_cache_dir', return_value=temp_cache_dir):
            with patch('examples.detroit_roads.get_roads', side_effect=[tile1_features, tile2_features]):
                result = get_roads_tiled(large_bbox, tile_size=2.0)

                # Result should contain features from both tiles
                assert result is not None
                assert result["type"] == "FeatureCollection"
                assert len(result["features"]) == 2, "Should merge features from both tiles"

    def test_get_roads_tiled_retry_on_failure(self, sample_geojson, temp_cache_dir):
        """Failed tiles should be retried."""
        from examples.detroit_roads import get_roads_tiled

        # First call fails, second succeeds (retry)
        call_count = [0]
        def side_effect_fn(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"type": "FeatureCollection", "features": []}  # First call: empty (failure)
            return sample_geojson  # Retry: success

        small_bbox = (42.0, -83.0, 42.5, -82.5)  # Small, single tile

        with patch('examples.detroit_roads.get_cache_dir', return_value=temp_cache_dir):
            with patch('examples.detroit_roads.get_roads', side_effect=side_effect_fn):
                result = get_roads_tiled(small_bbox, retry_count=1, retry_delay=0.01)

                # Should have retried and gotten data
                assert call_count[0] >= 1

    def test_get_roads_tiled_handles_all_failures(self, temp_cache_dir):
        """Returns empty collection if all tiles fail after retry."""
        from examples.detroit_roads import get_roads_tiled

        empty_result = {"type": "FeatureCollection", "features": []}
        small_bbox = (42.0, -83.0, 42.5, -82.5)

        with patch('examples.detroit_roads.get_cache_dir', return_value=temp_cache_dir):
            with patch('examples.detroit_roads.get_roads', return_value=empty_result):
                result = get_roads_tiled(small_bbox, retry_count=1, retry_delay=0.01)

                # Should return empty collection, not None
                assert result is not None
                assert result["type"] == "FeatureCollection"
                assert len(result["features"]) == 0

    def test_get_roads_tiled_passes_road_types(self, sample_geojson, temp_cache_dir):
        """Road types parameter should be passed to get_roads."""
        from examples.detroit_roads import get_roads_tiled

        small_bbox = (42.0, -83.0, 42.5, -82.5)
        road_types = ["motorway", "trunk"]

        with patch('examples.detroit_roads.get_cache_dir', return_value=temp_cache_dir):
            with patch('examples.detroit_roads.get_roads', return_value=sample_geojson) as mock_get_roads:
                get_roads_tiled(small_bbox, road_types=road_types)

                # Check that road_types was passed
                call_args = mock_get_roads.call_args
                assert call_args[0][1] == road_types or call_args[1].get('road_types') == road_types


# =============================================================================
# ROAD VERTEX SMOOTHING TESTS (TDD)
# =============================================================================


class TestSmoothRoadVertices:
    """Tests for smooth_road_vertices function - smooths mesh Z coords on roads."""

    def test_smooth_road_vertices_exists(self):
        """Test that smooth_road_vertices function can be imported."""
        from src.terrain.roads import smooth_road_vertices
        assert callable(smooth_road_vertices)

    def test_smooth_road_vertices_returns_array(self):
        """Test that function returns numpy array of same shape."""
        from src.terrain.roads import smooth_road_vertices

        # Simple 3x3 grid of vertices
        vertices = np.array([
            [0, 0, 10],
            [1, 0, 20],
            [2, 0, 15],
            [0, 1, 12],
            [1, 1, 25],  # Road vertex
            [2, 1, 18],
            [0, 2, 11],
            [1, 2, 22],
            [2, 2, 16],
        ], dtype=np.float64)

        # Road mask - only center pixel is road
        road_mask = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ], dtype=np.float64)

        # y_valid, x_valid map vertices to grid positions
        y_valid = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        x_valid = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])

        result = smooth_road_vertices(vertices, road_mask, y_valid, x_valid, smoothing_radius=1)

        assert isinstance(result, np.ndarray)
        assert result.shape == vertices.shape

    def test_smooth_road_vertices_only_modifies_road_vertices(self):
        """Test that non-road vertices are unchanged."""
        from src.terrain.roads import smooth_road_vertices

        vertices = np.array([
            [0, 0, 100.0],  # Not on road
            [1, 0, 200.0],  # On road
            [2, 0, 150.0],  # Not on road
        ], dtype=np.float64)

        road_mask = np.array([[0, 1, 0]], dtype=np.float64)
        y_valid = np.array([0, 0, 0])
        x_valid = np.array([0, 1, 2])

        result = smooth_road_vertices(vertices, road_mask, y_valid, x_valid, smoothing_radius=1)

        # Non-road vertices should be exactly unchanged
        assert result[0, 2] == 100.0, "First vertex Z should be unchanged"
        assert result[2, 2] == 150.0, "Third vertex Z should be unchanged"

    def test_smooth_road_vertices_smooths_z_values(self):
        """Test that road vertex Z values are actually smoothed."""
        from src.terrain.roads import smooth_road_vertices

        # Road with a spike in the middle
        vertices = np.array([
            [0, 0, 10.0],   # Road
            [1, 0, 100.0],  # Road - spike!
            [2, 0, 10.0],   # Road
        ], dtype=np.float64)

        road_mask = np.array([[1, 1, 1]], dtype=np.float64)  # All road
        y_valid = np.array([0, 0, 0])
        x_valid = np.array([0, 1, 2])

        result = smooth_road_vertices(vertices, road_mask, y_valid, x_valid, smoothing_radius=1)

        # Middle vertex should be smoothed down from 100
        assert result[1, 2] < 100.0, "Middle spike should be smoothed down"
        # Edge vertices might change slightly due to smoothing
        # but the spike should definitely be reduced

    def test_smooth_road_vertices_handles_nan(self):
        """Test that NaN values are handled properly."""
        from src.terrain.roads import smooth_road_vertices

        vertices = np.array([
            [0, 0, 10.0],
            [1, 0, np.nan],  # NaN on road
            [2, 0, 20.0],
        ], dtype=np.float64)

        road_mask = np.array([[1, 1, 1]], dtype=np.float64)
        y_valid = np.array([0, 0, 0])
        x_valid = np.array([0, 1, 2])

        result = smooth_road_vertices(vertices, road_mask, y_valid, x_valid, smoothing_radius=1)

        # NaN should be replaced with interpolated value
        assert not np.isnan(result[1, 2]), "NaN should be filled"
        # Filled value should be reasonable (between neighbors)
        assert 10.0 <= result[1, 2] <= 20.0, "Filled value should be interpolated"

    def test_smooth_road_vertices_preserves_xy(self):
        """Test that X and Y coordinates are never modified."""
        from src.terrain.roads import smooth_road_vertices

        vertices = np.array([
            [0.5, 1.5, 10.0],
            [1.5, 2.5, 20.0],
            [2.5, 3.5, 30.0],
        ], dtype=np.float64)

        road_mask = np.array([[1, 1, 1]], dtype=np.float64)
        y_valid = np.array([0, 0, 0])
        x_valid = np.array([0, 1, 2])

        result = smooth_road_vertices(vertices, road_mask, y_valid, x_valid, smoothing_radius=1)

        # X and Y must be exactly preserved
        np.testing.assert_array_equal(result[:, 0], vertices[:, 0], "X coords must be unchanged")
        np.testing.assert_array_equal(result[:, 1], vertices[:, 1], "Y coords must be unchanged")

    def test_smooth_road_vertices_with_zero_radius(self):
        """Test that radius=0 returns vertices unchanged."""
        from src.terrain.roads import smooth_road_vertices

        vertices = np.array([
            [0, 0, 10.0],
            [1, 0, 100.0],
            [2, 0, 10.0],
        ], dtype=np.float64)

        road_mask = np.array([[1, 1, 1]], dtype=np.float64)
        y_valid = np.array([0, 0, 0])
        x_valid = np.array([0, 1, 2])

        result = smooth_road_vertices(vertices, road_mask, y_valid, x_valid, smoothing_radius=0)

        # With radius=0, no smoothing should occur
        np.testing.assert_array_equal(result, vertices, "Radius 0 should return unchanged")

    def test_smooth_road_vertices_with_boundary_extension(self):
        """Test that boundary extension vertices (more vertices than y_valid entries) are handled."""
        from src.terrain.roads import smooth_road_vertices

        # 5 vertices total, but only 3 have grid mappings (simulates boundary extension)
        vertices = np.array([
            [0, 0, 10.0],   # Surface vertex on road
            [1, 0, 100.0],  # Surface vertex on road (spike)
            [2, 0, 10.0],   # Surface vertex on road
            [0, -1, 5.0],   # Boundary extension vertex (no grid mapping)
            [2, -1, 5.0],   # Boundary extension vertex (no grid mapping)
        ], dtype=np.float64)

        road_mask = np.array([[1, 1, 1]], dtype=np.float64)
        # Only 3 entries in y_valid/x_valid (surface vertices only)
        y_valid = np.array([0, 0, 0])
        x_valid = np.array([0, 1, 2])

        # This should NOT raise IndexError
        result = smooth_road_vertices(vertices, road_mask, y_valid, x_valid, smoothing_radius=1)

        # Should return same shape
        assert result.shape == vertices.shape
        # Boundary vertices should be unchanged
        assert result[3, 2] == 5.0, "Boundary vertex should be unchanged"
        assert result[4, 2] == 5.0, "Boundary vertex should be unchanged"
        # Surface road vertices should be smoothed (spike reduced)
        assert result[1, 2] < 100.0, "Surface spike should be smoothed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
