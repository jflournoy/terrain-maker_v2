"""
Tests for ring mask computation around park/skiing areas.

TDD RED phase: These tests define the expected behavior for computing
ring (annulus) masks around geographic points to create visual outlines.
"""

import numpy as np
import pytest


class TestComputeRingMask:
    """Tests for compute_ring_mask_grid function."""

    def test_compute_ring_mask_grid_exists(self):
        """Test that compute_ring_mask_grid function is importable."""
        from src.terrain.core import Terrain

        terrain = Terrain.__new__(Terrain)
        assert hasattr(terrain, "compute_ring_mask_grid")

    def test_compute_ring_mask_grid_returns_boolean_array(self):
        """Test that function returns a boolean numpy array."""
        from src.terrain.core import Terrain
        import logging

        # Create minimal terrain with mock data
        terrain = Terrain.__new__(Terrain)
        terrain.dem = np.ones((100, 100), dtype=np.float32) * 200
        terrain.transform = None  # Will use identity
        terrain.crs = "EPSG:4326"
        terrain._transforms = []
        terrain._transformed_dem = terrain.dem.copy()
        terrain._transformed_transform = None
        terrain._transformed_crs = "EPSG:4326"
        terrain.data_layers = {}  # Empty data_layers triggers fallback to _transformed_dem
        terrain.logger = logging.getLogger("test")

        # Single point in center
        lons = np.array([0.5])
        lats = np.array([0.5])

        result = terrain.compute_ring_mask_grid(
            lons, lats,
            inner_radius_meters=500,
            outer_radius_meters=600,
        )

        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        assert result.shape == terrain._transformed_dem.shape

    def test_compute_ring_mask_grid_ring_shape(self):
        """Test that the mask forms a ring (annulus), not a filled circle."""
        from src.terrain.core import Terrain
        import logging

        terrain = Terrain.__new__(Terrain)
        # Use a larger grid for better ring visualization
        terrain.dem = np.ones((200, 200), dtype=np.float32) * 200
        terrain.transform = None
        terrain.crs = "EPSG:4326"
        terrain._transforms = []
        terrain._transformed_dem = terrain.dem.copy()
        terrain._transformed_transform = None
        terrain._transformed_crs = "EPSG:4326"
        terrain.data_layers = {}
        terrain.logger = logging.getLogger("test")

        # Single point in center
        lons = np.array([0.5])
        lats = np.array([0.5])

        # Inner radius < outer radius creates a ring
        result = terrain.compute_ring_mask_grid(
            lons, lats,
            inner_radius_meters=400,
            outer_radius_meters=600,
        )

        # The center should be False (not in the ring)
        center_y, center_x = 100, 100  # Grid center
        assert result[center_y, center_x] == False, "Center should be outside the ring"

        # Some pixels at the ring distance should be True
        assert result.any(), "Ring should have some True pixels"

    def test_compute_ring_mask_grid_no_points_returns_empty(self):
        """Test that empty points array returns all-False mask."""
        from src.terrain.core import Terrain
        import logging

        terrain = Terrain.__new__(Terrain)
        terrain.dem = np.ones((100, 100), dtype=np.float32) * 200
        terrain.transform = None
        terrain.crs = "EPSG:4326"
        terrain._transforms = []
        terrain._transformed_dem = terrain.dem.copy()
        terrain._transformed_transform = None
        terrain._transformed_crs = "EPSG:4326"
        terrain.data_layers = {}
        terrain.logger = logging.getLogger("test")

        lons = np.array([])
        lats = np.array([])

        result = terrain.compute_ring_mask_grid(
            lons, lats,
            inner_radius_meters=500,
            outer_radius_meters=600,
        )

        assert not result.any(), "Empty points should produce all-False mask"

    def test_compute_ring_mask_grid_multiple_points(self):
        """Test that multiple points each get rings."""
        from src.terrain.core import Terrain
        import logging

        terrain = Terrain.__new__(Terrain)
        terrain.dem = np.ones((200, 200), dtype=np.float32) * 200
        terrain.transform = None
        terrain.crs = "EPSG:4326"
        terrain._transforms = []
        terrain._transformed_dem = terrain.dem.copy()
        terrain._transformed_transform = None
        terrain._transformed_crs = "EPSG:4326"
        terrain.data_layers = {}
        terrain.logger = logging.getLogger("test")

        # Two points far apart
        lons = np.array([0.25, 0.75])
        lats = np.array([0.5, 0.5])

        result = terrain.compute_ring_mask_grid(
            lons, lats,
            inner_radius_meters=200,
            outer_radius_meters=300,
        )

        # Both regions should have ring pixels
        left_region = result[:, :100]
        right_region = result[:, 100:]

        assert left_region.any(), "Left point should have ring pixels"
        assert right_region.any(), "Right point should have ring pixels"

    def test_compute_ring_mask_grid_inner_radius_zero_is_filled(self):
        """Test that inner_radius=0 creates a filled circle, not a ring."""
        from src.terrain.core import Terrain
        import logging

        terrain = Terrain.__new__(Terrain)
        terrain.dem = np.ones((200, 200), dtype=np.float32) * 200
        terrain.transform = None
        terrain.crs = "EPSG:4326"
        terrain._transforms = []
        terrain._transformed_dem = terrain.dem.copy()
        terrain._transformed_transform = None
        terrain._transformed_crs = "EPSG:4326"
        terrain.data_layers = {}
        terrain.logger = logging.getLogger("test")

        lons = np.array([0.5])
        lats = np.array([0.5])

        result = terrain.compute_ring_mask_grid(
            lons, lats,
            inner_radius_meters=0,
            outer_radius_meters=600,
        )

        # Center should be True (filled circle includes center)
        center_y, center_x = 100, 100
        assert result[center_y, center_x] == True, "Filled circle should include center"


class TestApplyRingColor:
    """Tests for applying dark ring colors to vertex colors."""

    def test_apply_ring_color_darkens_ring_vertices(self):
        """Test that ring mask causes vertices to be darkened."""
        from src.terrain.core import Terrain

        terrain = Terrain.__new__(Terrain)
        terrain.dem = np.ones((50, 50), dtype=np.float32) * 200
        terrain.transform = None
        terrain.crs = "EPSG:4326"
        terrain._transforms = []
        terrain._transformed_dem = terrain.dem.copy()
        terrain._transformed_transform = None
        terrain._transformed_crs = "EPSG:4326"

        # Create mock vertex data
        n_vertices = 100
        terrain.y_valid = np.random.randint(0, 50, n_vertices)
        terrain.x_valid = np.random.randint(0, 50, n_vertices)
        terrain.colors = np.ones((n_vertices, 4), dtype=np.float32)  # Start white

        # Create a ring mask with some True values
        ring_mask = np.zeros((50, 50), dtype=bool)
        ring_mask[20:30, 20:30] = True  # Some ring pixels

        # Apply ring color
        ring_color = (0.1, 0.1, 0.1)  # Dark gray
        terrain.apply_ring_color(ring_mask, ring_color)

        # Check that some vertices in the ring region are darkened
        # Vertices where ring_mask[y, x] is True should have dark color
        ring_vertex_indices = []
        for i, (y, x) in enumerate(zip(terrain.y_valid, terrain.x_valid)):
            if ring_mask[y, x]:
                ring_vertex_indices.append(i)

        if ring_vertex_indices:
            # At least one ring vertex should be dark
            for idx in ring_vertex_indices:
                assert terrain.colors[idx, 0] < 0.5, "Ring vertex should be darkened"

    def test_apply_ring_color_preserves_non_ring_vertices(self):
        """Test that non-ring vertices keep their original color."""
        from src.terrain.core import Terrain

        terrain = Terrain.__new__(Terrain)
        terrain.dem = np.ones((50, 50), dtype=np.float32) * 200
        terrain.transform = None
        terrain.crs = "EPSG:4326"
        terrain._transforms = []
        terrain._transformed_dem = terrain.dem.copy()
        terrain._transformed_transform = None
        terrain._transformed_crs = "EPSG:4326"

        n_vertices = 100
        terrain.y_valid = np.arange(n_vertices) % 50
        terrain.x_valid = np.arange(n_vertices) % 50
        original_colors = np.ones((n_vertices, 4), dtype=np.float32) * 0.8
        terrain.colors = original_colors.copy()

        # Empty ring mask
        ring_mask = np.zeros((50, 50), dtype=bool)

        terrain.apply_ring_color(ring_mask, (0.1, 0.1, 0.1))

        # All vertices should be unchanged
        np.testing.assert_array_equal(terrain.colors, original_colors)


class TestRingMaskParameters:
    """Tests for ring mask parameter validation and edge cases."""

    def test_ring_mask_invalid_radii_raises(self):
        """Test that inner_radius > outer_radius raises ValueError."""
        from src.terrain.core import Terrain
        import logging

        terrain = Terrain.__new__(Terrain)
        terrain.dem = np.ones((100, 100), dtype=np.float32) * 200
        terrain.transform = None
        terrain.crs = "EPSG:4326"
        terrain._transforms = []
        terrain._transformed_dem = terrain.dem.copy()
        terrain._transformed_transform = None
        terrain._transformed_crs = "EPSG:4326"
        terrain.data_layers = {}
        terrain.logger = logging.getLogger("test")

        lons = np.array([0.5])
        lats = np.array([0.5])

        with pytest.raises(ValueError) as exc_info:
            terrain.compute_ring_mask_grid(
                lons, lats,
                inner_radius_meters=600,  # Inner > outer is invalid
                outer_radius_meters=500,
            )

        assert "inner_radius" in str(exc_info.value).lower() or "radius" in str(exc_info.value).lower()

    def test_ring_mask_negative_radius_raises(self):
        """Test that negative radius raises ValueError."""
        from src.terrain.core import Terrain
        import logging

        terrain = Terrain.__new__(Terrain)
        terrain.dem = np.ones((100, 100), dtype=np.float32) * 200
        terrain.transform = None
        terrain.crs = "EPSG:4326"
        terrain._transforms = []
        terrain._transformed_dem = terrain.dem.copy()
        terrain._transformed_transform = None
        terrain._transformed_crs = "EPSG:4326"
        terrain.data_layers = {}
        terrain.logger = logging.getLogger("test")

        lons = np.array([0.5])
        lats = np.array([0.5])

        with pytest.raises(ValueError):
            terrain.compute_ring_mask_grid(
                lons, lats,
                inner_radius_meters=-100,
                outer_radius_meters=500,
            )
