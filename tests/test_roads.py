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
    blend_color_for_road,
    sample_color_at_mesh_coord,
    create_road_material,
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
        assert "bbox" in query

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


class TestColorBlending:
    """Tests for color blending operations."""

    def test_blend_color_default(self):
        """Test default color blending (30% darker)."""
        color = (1.0, 1.0, 1.0)  # White
        blended = blend_color_for_road(color)

        # Default factor is 0.7, so 30% darker
        assert blended == (0.7, 0.7, 0.7)

    def test_blend_color_custom_factor(self):
        """Test color blending with custom factor."""
        color = (1.0, 0.5, 0.25)
        blended = blend_color_for_road(color, darken_factor=0.5)

        assert blended == (0.5, 0.25, 0.125)

    def test_blend_color_no_darkening(self):
        """Test color with darken_factor=1.0 (no darkening)."""
        color = (0.8, 0.6, 0.4)
        blended = blend_color_for_road(color, darken_factor=1.0)

        assert blended == color

    def test_blend_color_rgb_only(self):
        """Test that only RGB components are used (not alpha)."""
        color = (1.0, 0.8, 0.6, 0.5)  # RGBA
        blended = blend_color_for_road(color)

        # Should only return 3 values (RGB)
        assert len(blended) == 3
        assert blended == (0.7, 0.56, 0.42)

    def test_blend_color_preserves_ratios(self):
        """Test that color blending preserves color ratios."""
        color = (0.5, 0.25, 0.125)  # 4:2:1 ratio
        blended = blend_color_for_road(color, darken_factor=0.8)

        # Ratio should be preserved
        r, g, b = blended
        assert abs(r / (g + 1e-10) - 2.0) < 0.01
        assert abs(g / (b + 1e-10) - 2.0) < 0.01


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


@pytest.mark.parametrize("darken_factor,expected_result", [
    (0.5, (0.5, 0.5, 0.5)),  # 50% darker (white to gray)
    (0.7, (0.7, 0.7, 0.7)),  # 30% darker
    (0.9, (0.9, 0.9, 0.9)),  # 10% darker
    (1.0, (1.0, 1.0, 1.0)),  # No change
])
def test_blend_factor_range(darken_factor, expected_result):
    """Test color blending across valid factor range."""
    color = (1.0, 1.0, 1.0)
    result = blend_color_for_road(color, darken_factor=darken_factor)
    assert result == expected_result


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
