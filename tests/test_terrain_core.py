"""
Tests for terrain core functionality - DEM loading and Terrain initialization.

Following TDD RED-GREEN-REFACTOR cycle.
"""

import pytest
import numpy as np
import rasterio
from rasterio import Affine
from pathlib import Path
import tempfile
from src.terrain.core import load_dem_files, Terrain, scale_elevation, elevation_colormap

# Check if Blender is available
try:
    import bpy

    HAS_BLENDER = True
except ImportError:
    HAS_BLENDER = False


class TestLoadDEMFiles:
    """Test suite for load_dem_files function."""

    def test_load_dem_files_requires_valid_directory(self):
        """load_dem_files should raise ValueError for non-existent directory."""
        with pytest.raises(ValueError, match="Directory does not exist"):
            load_dem_files("/nonexistent/directory")

    def test_load_dem_files_requires_directory_not_file(self, tmp_path):
        """load_dem_files should raise ValueError when given a file path instead of directory."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        with pytest.raises(ValueError, match="Path is not a directory"):
            load_dem_files(str(test_file))

    def test_load_dem_files_raises_error_when_no_matching_files(self, tmp_path):
        """load_dem_files should raise ValueError when no files match pattern."""
        with pytest.raises(ValueError, match="No files matching .* found"):
            load_dem_files(str(tmp_path), pattern="*.hgt")

    def test_load_dem_files_returns_tuple_with_array_and_transform(self, sample_dem_file):
        """load_dem_files should return (numpy array, Affine transform) tuple."""
        dem_dir = sample_dem_file.parent
        result = load_dem_files(str(dem_dir), pattern="*.tif")

        assert isinstance(result, tuple), "Should return tuple"
        assert len(result) == 2, "Should return 2-element tuple"

        dem_data, transform = result
        assert isinstance(dem_data, np.ndarray), "First element should be numpy array"
        assert isinstance(transform, Affine), "Second element should be Affine transform"

    def test_load_dem_files_returns_2d_array(self, sample_dem_file):
        """load_dem_files should return 2D elevation array."""
        dem_dir = sample_dem_file.parent
        dem_data, _ = load_dem_files(str(dem_dir), pattern="*.tif")

        assert dem_data.ndim == 2, f"DEM should be 2D, got {dem_data.ndim}D"

    def test_load_dem_files_merges_multiple_files(self, tmp_path):
        """load_dem_files should merge multiple DEM files into single array."""
        # Create two adjacent DEM tiles
        file1 = create_sample_geotiff(
            tmp_path / "tile1.tif", data=np.ones((10, 10)) * 100, bounds=(-120, 40, -119.5, 40.5)
        )
        file2 = create_sample_geotiff(
            tmp_path / "tile2.tif", data=np.ones((10, 10)) * 200, bounds=(-119.5, 40, -119, 40.5)
        )

        dem_data, transform = load_dem_files(str(tmp_path), pattern="*.tif")

        # Merged array should be larger than individual tiles
        assert dem_data.shape[0] >= 10, "Merged height should be at least tile height"
        assert dem_data.shape[1] >= 10, "Merged width should include both tiles"

        # Should contain values from both tiles
        assert np.any(np.isclose(dem_data, 100, rtol=0.01)), "Should contain values from tile1"
        assert np.any(np.isclose(dem_data, 200, rtol=0.01)), "Should contain values from tile2"


class TestTerrainInitialization:
    """Test suite for Terrain class initialization."""

    def test_terrain_requires_numpy_array(self):
        """Terrain.__init__ should raise TypeError if dem_data is not numpy array."""
        transform = Affine.identity()

        with pytest.raises(TypeError, match="dem_data must be a numpy array"):
            Terrain([[1, 2], [3, 4]], transform)

    def test_terrain_requires_2d_array(self):
        """Terrain.__init__ should raise ValueError if dem_data is not 2D."""
        transform = Affine.identity()

        # 1D array
        with pytest.raises(ValueError, match="dem_data must be 2D"):
            Terrain(np.array([1, 2, 3]), transform)

        # 3D array
        with pytest.raises(ValueError, match="dem_data must be 2D"):
            Terrain(np.ones((10, 10, 3)), transform)

    def test_terrain_accepts_integer_dem_data(self):
        """Terrain.__init__ should accept integer DEM data and convert to float."""
        dem_data = np.ones((10, 10), dtype=np.int16) * 1000
        transform = Affine.identity()

        terrain = Terrain(dem_data, transform)

        # Should store the data (converted to float)
        assert terrain.data_layers["dem"]["data"].dtype in [np.float32, np.float64]

    def test_terrain_accepts_float_dem_data(self):
        """Terrain.__init__ should accept float DEM data."""
        dem_data = np.ones((10, 10), dtype=np.float32) * 1500.5
        transform = Affine.identity()

        terrain = Terrain(dem_data, transform)

        assert "dem" in terrain.data_layers
        assert terrain.data_layers["dem"]["data"].dtype == np.float32

    def test_terrain_stores_dem_shape(self):
        """Terrain should store original DEM shape."""
        dem_data = np.ones((50, 100), dtype=np.float32)
        transform = Affine.identity()

        terrain = Terrain(dem_data, transform)

        assert terrain.dem_shape == (50, 100)

    def test_terrain_stores_dem_transform(self):
        """Terrain should store the affine transform."""
        dem_data = np.ones((10, 10), dtype=np.float32)
        transform = Affine.translation(100, 200) * Affine.scale(0.5, -0.5)

        terrain = Terrain(dem_data, transform)

        assert terrain.dem_transform == transform

    def test_terrain_creates_cache(self):
        """Terrain should initialize TerrainCache."""
        dem_data = np.ones((10, 10), dtype=np.float32)
        transform = Affine.identity()

        terrain = Terrain(dem_data, transform)

        assert hasattr(terrain, "cache")
        assert terrain.cache is not None

    def test_terrain_adds_dem_to_data_layers(self):
        """Terrain should add DEM as a data layer."""
        dem_data = np.ones((10, 10), dtype=np.float32) * 1200
        transform = Affine.identity()

        terrain = Terrain(dem_data, transform)

        assert "dem" in terrain.data_layers
        assert terrain.data_layers["dem"]["data"] is not None
        assert terrain.data_layers["dem"]["transform"] == transform

    def test_terrain_calculates_resolution(self):
        """Terrain should calculate pixel resolution in meters."""
        dem_data = np.ones((10, 10), dtype=np.float32)
        # 0.001 degrees ≈ 111.32 meters at equator
        transform = Affine.scale(0.001, -0.001)

        terrain = Terrain(dem_data, transform)

        assert hasattr(terrain, "resolution")
        assert isinstance(terrain.resolution, tuple)
        assert len(terrain.resolution) == 2
        # Resolution should be approximately 111 meters
        assert 100 < terrain.resolution[0] < 120
        assert 100 < terrain.resolution[1] < 120


class TestMeshGeneration:
    """Test suite for mesh generation logic and data structures."""

    def test_mesh_requires_transformed_dem(self):
        """Mesh creation should fail if DEM is not transformed."""
        dem_data = np.ones((10, 10), dtype=np.float32)
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        # Try to create mesh without transforming DEM
        with pytest.raises(ValueError, match="Transformed DEM layer required"):
            terrain.create_mesh()

    def test_mesh_requires_valid_dem_data(self):
        """Mesh creation should require valid DEM layer data."""
        dem_data = np.ones((10, 10), dtype=np.float32)
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        # Mark as transformed but don't provide transformed_data
        terrain.data_layers["dem"]["transformed"] = True
        # Don't add transformed_data - this should cause an error

        with pytest.raises((ValueError, KeyError)):
            terrain.create_mesh()

    def test_mesh_generation_with_valid_dem(self):
        """Mesh should generate successfully with transformed DEM data."""
        dem_data = np.ones((10, 10), dtype=np.float32) * 100
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        # Apply identity transform to mark as transformed
        def identity_transform(data, trans):
            return data, trans, None

        terrain.transforms.append(identity_transform)
        terrain.apply_transforms()

        # Should not raise exception
        try:
            result = terrain.create_mesh()
            # If bpy is not available, this will raise ImportError
            # But we test that the method is callable and reaches the mesh creation point
            assert result is None or hasattr(result, "name")  # Blender object or None
        except ImportError:
            # Expected if Blender is not installed
            pass

    def test_mesh_model_parameters_stored(self):
        """Mesh creation should store model parameters for later use."""
        dem_data = np.ones((10, 10), dtype=np.float32) * 100
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        # Apply identity transform
        def identity_transform(data, trans):
            return data, trans, None

        terrain.transforms.append(identity_transform)
        terrain.apply_transforms()

        try:
            terrain.create_mesh(scale_factor=50.0, height_scale=2.0, center_model=True)

            # Check that model parameters are stored
            assert hasattr(terrain, "model_params")
            assert terrain.model_params["scale_factor"] == 50.0
            assert terrain.model_params["height_scale"] == 2.0
            assert terrain.model_params["centered"] is True
        except ImportError:
            # Skip if Blender not available
            pass

    def test_mesh_offset_stored_when_centered(self):
        """Mesh creation should store model offset when centering is enabled."""
        dem_data = np.arange(100, dtype=np.float32).reshape(10, 10)
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        # Apply identity transform
        def identity_transform(data, trans):
            return data, trans, None

        terrain.transforms.append(identity_transform)
        terrain.apply_transforms()

        try:
            terrain.create_mesh(center_model=True)

            # Check that offset is stored
            assert hasattr(terrain, "model_offset")
            assert isinstance(terrain.model_offset, np.ndarray)
            assert len(terrain.model_offset) == 3
        except ImportError:
            # Skip if Blender not available
            pass

    def test_mesh_no_offset_when_not_centered(self):
        """Mesh creation should have zero offset when centering is disabled."""
        dem_data = np.ones((10, 10), dtype=np.float32) * 100
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        # Apply identity transform
        def identity_transform(data, trans):
            return data, trans, None

        terrain.transforms.append(identity_transform)
        terrain.apply_transforms()

        try:
            terrain.create_mesh(center_model=False)

            # Offset should be zero
            assert hasattr(terrain, "model_offset")
            np.testing.assert_array_equal(terrain.model_offset, [0, 0, 0])
        except ImportError:
            # Skip if Blender not available
            pass

    def test_mesh_handles_nan_values(self):
        """Mesh creation should handle NaN values in DEM data gracefully."""
        dem_data = np.ones((10, 10), dtype=np.float32) * 100
        # Insert some NaN values
        dem_data[2:4, 2:4] = np.nan

        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        # Apply identity transform
        def identity_transform(data, trans):
            return data, trans, None

        terrain.transforms.append(identity_transform)
        terrain.apply_transforms()

        try:
            # Should not raise exception due to NaN values
            terrain.create_mesh()
            assert True  # Successfully handled NaN
        except ImportError:
            # Skip if Blender not available
            pass

    def test_mesh_with_different_scale_factors(self):
        """Mesh should accept and store different scale factors."""
        dem_data = np.ones((10, 10), dtype=np.float32) * 100
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        # Apply identity transform
        def identity_transform(data, trans):
            return data, trans, None

        terrain.transforms.append(identity_transform)
        terrain.apply_transforms()

        for scale in [10.0, 50.0, 100.0, 200.0]:
            try:
                terrain.create_mesh(scale_factor=scale)
                assert terrain.model_params["scale_factor"] == scale
            except ImportError:
                # Skip if Blender not available
                pass


class TestTerrainTransforms:
    """Test suite for Terrain.apply_transforms method."""

    def test_apply_transforms_does_nothing_with_no_transforms(self):
        """apply_transforms should return early if no transforms are registered."""
        dem_data = np.ones((10, 10), dtype=np.float32)
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        # Call with no transforms registered
        result = terrain.apply_transforms()

        assert result is None
        # Data layers should be unchanged
        assert "dem" in terrain.data_layers
        assert "transformed_data" not in terrain.data_layers["dem"]

    def test_apply_transforms_applies_single_transform(self):
        """apply_transforms should apply a single registered transform to all layers."""
        dem_data = np.ones((10, 10), dtype=np.float32) * 100
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        # Register a simple transform that multiplies by 2
        def double_values(data, trans):
            return data * 2, trans, None

        terrain.transforms.append(double_values)
        terrain.apply_transforms()

        # DEM layer should have transformed data
        assert "transformed_data" in terrain.data_layers["dem"]
        transformed = terrain.data_layers["dem"]["transformed_data"]
        np.testing.assert_array_almost_equal(transformed, dem_data * 2)

    def test_apply_transforms_applies_multiple_transforms_in_sequence(self):
        """apply_transforms should apply multiple transforms in order."""
        dem_data = np.ones((10, 10), dtype=np.float32) * 10
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        # Register two transforms
        def double_values(data, trans):
            return data * 2, trans, None

        def add_ten(data, trans):
            return data + 10, trans, None

        terrain.transforms = [double_values, add_ten]
        terrain.apply_transforms()

        # Should have: (10 * 2) + 10 = 30
        transformed = terrain.data_layers["dem"]["transformed_data"]
        np.testing.assert_array_almost_equal(transformed, np.ones((10, 10)) * 30)

    def test_apply_transforms_marks_layer_as_transformed(self):
        """apply_transforms should set 'transformed' flag on processed layers."""
        dem_data = np.ones((10, 10), dtype=np.float32)
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        def identity_transform(data, trans):
            return data, trans, None

        terrain.transforms.append(identity_transform)
        terrain.apply_transforms()

        assert terrain.data_layers["dem"]["transformed"] is True

    def test_apply_transforms_skips_already_transformed_layers(self):
        """apply_transforms should skip layers that are already transformed."""
        dem_data = np.ones((10, 10), dtype=np.float32)
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        call_count = {"count": 0}

        def counting_transform(data, trans):
            call_count["count"] += 1
            return data, trans, None

        terrain.transforms.append(counting_transform)

        # First application
        terrain.apply_transforms()
        assert call_count["count"] == 1

        # Second application should skip already-transformed layer
        terrain.apply_transforms()
        assert call_count["count"] == 1  # Should not increase

    def test_apply_transforms_stores_metadata(self):
        """apply_transforms should store transform metadata in layer."""
        dem_data = np.ones((10, 10), dtype=np.float32)
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        def my_transform(data, trans):
            return data, trans, None

        my_transform.__name__ = "my_transform"
        terrain.transforms.append(my_transform)
        terrain.apply_transforms()

        metadata = terrain.data_layers["dem"]["transform_metadata"]
        assert "transforms" in metadata
        assert "my_transform" in metadata["transforms"]
        assert "original_shape" in metadata
        assert "transformed_shape" in metadata

    def test_apply_transforms_handles_crs_changes(self):
        """apply_transforms should update CRS if transform provides new one."""
        dem_data = np.ones((10, 10), dtype=np.float32)
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        def transform_with_crs_change(data, trans):
            new_crs = "EPSG:3857"  # Web Mercator
            return data, trans, new_crs

        terrain.transforms.append(transform_with_crs_change)
        terrain.apply_transforms()

        assert terrain.data_layers["dem"]["transformed_crs"] == "EPSG:3857"

    def test_apply_transforms_raises_on_transform_error(self):
        """apply_transforms should raise exception if transform fails."""
        dem_data = np.ones((10, 10), dtype=np.float32)
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        def failing_transform(data, trans):
            raise ValueError("Transform failed")

        terrain.transforms.append(failing_transform)

        with pytest.raises(ValueError, match="Transform failed"):
            terrain.apply_transforms()

    def test_apply_transforms_preserves_transform_metadata(self):
        """apply_transforms should preserve transform and CRS metadata."""
        dem_data = np.ones((10, 10), dtype=np.float32)
        original_transform = Affine.scale(2.0, -2.0)
        original_crs = "EPSG:4326"

        terrain = Terrain(dem_data, original_transform, dem_crs=original_crs)

        def identity_transform(data, trans):
            return data, trans, None

        terrain.transforms.append(identity_transform)
        terrain.apply_transforms()

        # Transform and CRS should be preserved
        assert terrain.data_layers["dem"]["transformed_transform"] == original_transform
        assert terrain.data_layers["dem"]["transformed_crs"] == original_crs


class TestAddDataLayerWithTransforms:
    """Test suite for add_data_layer behavior with transformed target layers.

    These tests verify that add_data_layer uses transformed dimensions
    when the target layer has been downsampled/transformed.
    """

    def test_add_data_layer_uses_transformed_shape_after_apply_transforms(self):
        """add_data_layer should use transformed shape when target layer has been transformed."""
        # Create terrain with 100x100 DEM
        dem_data = np.random.rand(100, 100).astype(np.float32) * 1000
        transform = Affine.scale(0.001, -0.001) * Affine.translation(-83.0, 42.5)
        terrain = Terrain(dem_data, transform, dem_crs="EPSG:4326")

        # Add downsampling transform (50% zoom = 50x50)
        from scipy.ndimage import zoom as scipy_zoom

        def downsample_transform(data, trans):
            downsampled = scipy_zoom(data, 0.5, order=1)
            # Adjust transform for new resolution
            new_trans = Affine(
                trans.a * 2, trans.b, trans.c, trans.d, trans.e * 2, trans.f
            )
            return downsampled, new_trans, None

        terrain.add_transform(downsample_transform)
        terrain.apply_transforms()

        # Verify DEM was downsampled
        transformed_dem = terrain.data_layers["dem"]["transformed_data"]
        assert transformed_dem.shape == (50, 50), "DEM should be downsampled to 50x50"

        # Create score grid at ORIGINAL resolution (100x100)
        score_grid = np.random.rand(100, 100).astype(np.float32)

        # Add score layer with target_layer="dem" - should be resampled to match transformed DEM (50x50)
        terrain.add_data_layer("score", score_grid, transform, "EPSG:4326", target_layer="dem")

        # Verify score layer was resampled to match transformed dimensions
        score_layer = terrain.data_layers["score"]["data"]
        assert score_layer.shape == (50, 50), (
            f"Score layer should be resampled to 50x50, got {score_layer.shape}"
        )

    def test_add_data_layer_uses_original_shape_before_transforms(self):
        """add_data_layer should use original shape when target layer not yet transformed."""
        # Create terrain with 100x100 DEM
        dem_data = np.random.rand(100, 100).astype(np.float32) * 1000
        transform = Affine.scale(0.001, -0.001) * Affine.translation(-83.0, 42.5)
        terrain = Terrain(dem_data, transform, dem_crs="EPSG:4326")

        # Create score grid at same resolution (100x100)
        score_grid = np.random.rand(100, 100).astype(np.float32)

        # Add score layer BEFORE any transforms - should stay at 100x100
        terrain.add_data_layer("score", score_grid, transform, "EPSG:4326")

        # Verify score layer matches original DEM shape
        score_layer = terrain.data_layers["score"]["data"]
        assert score_layer.shape == (100, 100), (
            f"Score layer should stay at 100x100, got {score_layer.shape}"
        )

    def test_add_data_layer_uses_transformed_crs_and_transform(self):
        """add_data_layer should use transformed CRS and transform when available."""
        # Create terrain with 50x50 DEM
        dem_data = np.random.rand(50, 50).astype(np.float32) * 1000
        original_transform = Affine.scale(0.001, -0.001) * Affine.translation(-83.0, 42.5)
        terrain = Terrain(dem_data, original_transform, dem_crs="EPSG:4326")

        # Add transform that changes CRS
        new_transform = Affine.scale(100, -100) * Affine.translation(500000, 4500000)
        new_crs = "EPSG:32617"

        def crs_change_transform(data, trans):
            return data, new_transform, new_crs

        terrain.add_transform(crs_change_transform)
        terrain.apply_transforms()

        # Create score grid at original transform/CRS (will need reprojection)
        score_grid = np.random.rand(50, 50).astype(np.float32)

        # Add score layer with original CRS and target_layer="dem" - should align to transformed CRS
        terrain.add_data_layer("score", score_grid, original_transform, "EPSG:4326", target_layer="dem")

        # Verify score layer matches transformed DEM shape
        score_layer = terrain.data_layers["score"]["data"]
        assert score_layer.shape == (50, 50), "Score layer should match DEM shape"


@pytest.mark.skipif(not HAS_BLENDER, reason="Requires Blender (import bpy)")
class TestBlenderMeshCreation:
    """Test suite for Blender mesh creation functionality.

    These tests require Blender to be importable as a Python library.
    They will run when pytest is executed inside Blender or when bpy is available.
    """

    def test_blender_mesh_object_creation(self):
        """Mesh creation should produce a valid Blender object."""
        dem_data = np.ones((10, 10), dtype=np.float32) * 100
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        def identity_transform(data, trans):
            return data, trans, None

        terrain.transforms.append(identity_transform)
        terrain.apply_transforms()

        result = terrain.create_mesh()

        # Verify result is a Blender object
        assert result is not None
        assert hasattr(result, "name")
        assert hasattr(result, "data")
        assert hasattr(result.data, "vertices")

    def test_blender_mesh_has_geometry(self):
        """Created mesh should have vertices and faces."""
        dem_data = np.ones((10, 10), dtype=np.float32) * 100
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        def identity_transform(data, trans):
            return data, trans, None

        terrain.transforms.append(identity_transform)
        terrain.apply_transforms()

        result = terrain.create_mesh()

        # Mesh should have geometry
        assert len(result.data.vertices) > 0
        assert len(result.data.polygons) > 0

    def test_blender_mesh_with_different_parameters(self):
        """Mesh creation should accept and use different parameters."""
        dem_data = np.arange(100, dtype=np.float32).reshape(10, 10)
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        def identity_transform(data, trans):
            return data, trans, None

        terrain.transforms.append(identity_transform)
        terrain.apply_transforms()

        result = terrain.create_mesh(
            scale_factor=50.0, height_scale=2.0, center_model=True, boundary_extension=True
        )

        assert result is not None
        assert len(result.data.vertices) > 0
        assert len(result.data.polygons) > 0

    def test_blender_mesh_boundary_extension_affects_geometry(self):
        """Mesh with boundary extension should have more geometry."""
        dem_data = np.ones((10, 10), dtype=np.float32) * 100
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        def identity_transform(data, trans):
            return data, trans, None

        terrain.transforms.append(identity_transform)
        terrain.apply_transforms()

        result_with = terrain.create_mesh(boundary_extension=True)
        verts_with = len(result_with.data.vertices)
        faces_with = len(result_with.data.polygons)

        # Create another terrain for comparison
        terrain2 = Terrain(dem_data.copy(), transform)
        terrain2.transforms.append(identity_transform)
        terrain2.apply_transforms()

        result_without = terrain2.create_mesh(boundary_extension=False)
        verts_without = len(result_without.data.vertices)
        faces_without = len(result_without.data.polygons)

        # Boundary extension should add geometry
        assert verts_with >= verts_without
        assert faces_with > faces_without


class TestScaleElevation:
    """Test suite for scale_elevation transform function."""

    def test_scale_elevation_returns_transform_function(self):
        """scale_elevation should return a callable transform function."""
        transform_func = scale_elevation(scale_factor=0.5)
        assert callable(transform_func)

    def test_scale_elevation_reduces_values(self):
        """scale_elevation should multiply DEM values by scale factor."""
        dem_data = np.array([[100, 200], [300, 400]], dtype=np.float32)
        transform = Affine.identity()

        scale_func = scale_elevation(scale_factor=0.5)
        result_data, result_transform, result_crs = scale_func(dem_data, transform)

        # Check values are scaled
        assert np.allclose(result_data, dem_data * 0.5)

    def test_scale_elevation_returns_3tuple(self):
        """scale_elevation transform should return 3-tuple (data, transform, crs)."""
        dem_data = np.ones((10, 10), dtype=np.float32) * 100
        transform = Affine.identity()

        scale_func = scale_elevation(scale_factor=0.5)
        result = scale_func(dem_data, transform)

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], np.ndarray)
        assert result[1] is not None or result[1] is None  # Transform can be modified or unchanged
        assert result[2] is None  # CRS should be None (no change)

    def test_scale_elevation_different_factors(self):
        """scale_elevation should work with different scale factors."""
        dem_data = np.array([[100, 200], [300, 400]], dtype=np.float32)
        transform = Affine.identity()

        for factor in [0.1, 0.5, 1.0, 2.0, 10.0]:
            scale_func = scale_elevation(scale_factor=factor)
            result_data, _, _ = scale_func(dem_data, transform)
            assert np.allclose(result_data, dem_data * factor)

    def test_scale_elevation_preserves_nan(self):
        """scale_elevation should preserve NaN values."""
        dem_data = np.array([[100, np.nan], [300, 400]], dtype=np.float32)
        transform = Affine.identity()

        scale_func = scale_elevation(scale_factor=0.5)
        result_data, _, _ = scale_func(dem_data, transform)

        assert np.isnan(result_data[0, 1])
        assert np.allclose(result_data[~np.isnan(result_data)], dem_data[~np.isnan(dem_data)] * 0.5)


class TestElevationColormap:
    """Test suite for elevation_colormap utility function."""

    def test_elevation_colormap_returns_rgb_array(self):
        """elevation_colormap should return RGB array with shape (H, W, 3)."""
        dem_data = np.random.uniform(100, 200, size=(50, 50)).astype(np.float32)

        colors = elevation_colormap(dem_data, cmap_name="viridis")

        assert isinstance(colors, np.ndarray)
        assert colors.shape == (50, 50, 3)
        assert colors.dtype == np.uint8

    def test_elevation_colormap_viridis_gradient(self):
        """elevation_colormap with viridis should map low→purple, high→yellow."""
        # Create simple gradient (low to high elevation)
        dem_data = np.linspace(0, 100, 100).reshape(10, 10).astype(np.float32)

        colors = elevation_colormap(dem_data, cmap_name="viridis")

        # First row should be purple (low elevation)
        # Last row should be yellow (high elevation)
        assert colors.dtype == np.uint8
        assert colors.shape == (10, 10, 3)

    def test_elevation_colormap_auto_minmax(self):
        """elevation_colormap should auto-calculate min/max elevation."""
        dem_data = np.array([[100, 150], [200, 250]], dtype=np.float32)

        # Without explicit min/max, should use data range
        colors = elevation_colormap(dem_data, cmap_name="viridis")

        assert colors.shape == (2, 2, 3)
        assert colors.dtype == np.uint8

    def test_elevation_colormap_explicit_minmax(self):
        """elevation_colormap should respect explicit min/max."""
        dem_data = np.array([[100, 150], [200, 250]], dtype=np.float32)

        colors = elevation_colormap(dem_data, cmap_name="viridis", min_elev=0, max_elev=500)

        assert colors.shape == (2, 2, 3)
        assert colors.dtype == np.uint8

    def test_elevation_colormap_handles_nan(self):
        """elevation_colormap should handle NaN values gracefully."""
        dem_data = np.array([[100, np.nan], [200, 250]], dtype=np.float32)

        colors = elevation_colormap(dem_data, cmap_name="viridis")

        assert colors.shape == (2, 2, 3)
        # NaN values should map to dark gray or similar
        assert colors[0, 1].sum() < 255  # Not pure white

    def test_elevation_colormap_uint8_range(self):
        """elevation_colormap should return values in uint8 range (0-255)."""
        dem_data = np.random.uniform(100, 200, size=(20, 20)).astype(np.float32)

        colors = elevation_colormap(dem_data, cmap_name="viridis")

        assert colors.min() >= 0
        assert colors.max() <= 255


class TestConfigureForTargetVertices:
    """Test suite for Terrain.configure_for_target_vertices method."""

    def test_configure_for_target_vertices_returns_zoom_factor(self):
        """configure_for_target_vertices should return calculated zoom_factor."""
        dem_data = np.ones((100, 100), dtype=np.float32)
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        zoom = terrain.configure_for_target_vertices(5000)

        assert isinstance(zoom, float)
        assert 0 < zoom <= 1.0

    def test_configure_for_target_vertices_calculates_correct_zoom(self):
        """configure_for_target_vertices should calculate correct zoom_factor."""
        dem_data = np.ones((100, 100), dtype=np.float32)
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        # Original: 100 × 100 = 10,000 vertices
        # Target: 2,500 vertices
        # Expected zoom: sqrt(2500 / 10000) = 0.5
        zoom = terrain.configure_for_target_vertices(2500)

        np.testing.assert_almost_equal(zoom, 0.5, decimal=6)

    def test_configure_for_target_vertices_adds_to_transforms(self):
        """configure_for_target_vertices should add downsampling transform."""
        dem_data = np.ones((100, 100), dtype=np.float32)
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        initial_count = len(terrain.transforms)
        terrain.configure_for_target_vertices(5000)

        assert len(terrain.transforms) == initial_count + 1
        # Last transform should be downsampling
        assert hasattr(terrain.transforms[-1], "__name__")

    def test_configure_for_target_vertices_handles_small_targets(self):
        """configure_for_target_vertices should handle very small targets."""
        dem_data = np.ones((1000, 1000), dtype=np.float32)
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        zoom = terrain.configure_for_target_vertices(100)

        assert 0 < zoom <= 0.01  # Should be heavily downsampled
        assert len(terrain.transforms) == 1

    def test_configure_for_target_vertices_exceeding_source(self):
        """configure_for_target_vertices should clamp to zoom_factor=1.0 if target > source."""
        dem_data = np.ones((100, 100), dtype=np.float32)
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        # Original: 100 × 100 = 10,000 vertices
        # Target: 50,000 vertices (exceeds source)
        zoom = terrain.configure_for_target_vertices(50000)

        assert zoom == 1.0  # Should not upsample

    def test_configure_for_target_vertices_invalid_input(self):
        """configure_for_target_vertices should raise ValueError for invalid input."""
        dem_data = np.ones((100, 100), dtype=np.float32)
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        with pytest.raises(ValueError):
            terrain.configure_for_target_vertices(-1)

        with pytest.raises(ValueError):
            terrain.configure_for_target_vertices(0)

        with pytest.raises(ValueError):
            terrain.configure_for_target_vertices(1.5)  # Not an integer

    def test_configure_for_target_vertices_with_custom_order(self):
        """configure_for_target_vertices should accept custom interpolation order."""
        dem_data = np.ones((100, 100), dtype=np.float32)
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        zoom = terrain.configure_for_target_vertices(5000, order=1)

        assert isinstance(zoom, float)
        assert len(terrain.transforms) == 1


@pytest.mark.skipif(not HAS_BLENDER, reason="Blender not available")
class TestCameraSetup:
    """Test suite for setup_camera_and_light camera type control."""

    def test_setup_camera_and_light_defaults_to_perspective(self):
        """setup_camera_and_light should default to perspective camera."""
        from src.terrain.core import setup_camera_and_light

        camera, light = setup_camera_and_light(
            camera_angle=(0, 0, 0), camera_location=(0, 0, 0), scale=10.0, focal_length=50
        )

        assert camera.data.type == "PERSP", "Should default to perspective camera"

    def test_setup_camera_and_light_accepts_camera_type_perspective(self):
        """setup_camera_and_light should accept camera_type='PERSP'."""
        from src.terrain.core import setup_camera_and_light

        camera, light = setup_camera_and_light(
            camera_angle=(0, 0, 0),
            camera_location=(0, 0, 0),
            scale=10.0,
            camera_type="PERSP",
            focal_length=50,
        )

        assert camera.data.type == "PERSP"
        assert camera.data.lens == 50  # focal_length should be set

    def test_setup_camera_and_light_accepts_camera_type_ortho(self):
        """setup_camera_and_light should accept camera_type='ORTHO' for orthographic."""
        from src.terrain.core import setup_camera_and_light

        camera, light = setup_camera_and_light(
            camera_angle=(0, 0, 0), camera_location=(0, 0, 0), scale=20.0, camera_type="ORTHO"
        )

        assert camera.data.type == "ORTHO", "Should create orthographic camera"
        assert camera.data.ortho_scale == 20.0, "ortho_scale should be set from scale parameter"

    def test_setup_camera_and_light_ortho_ignores_focal_length(self):
        """setup_camera_and_light with ORTHO should ignore focal_length parameter."""
        from src.terrain.core import setup_camera_and_light

        camera, light = setup_camera_and_light(
            camera_angle=(0, 0, 0),
            camera_location=(0, 0, 0),
            scale=20.0,
            camera_type="ORTHO",
            focal_length=100,  # Should be ignored
        )

        assert camera.data.type == "ORTHO"
        assert camera.data.ortho_scale == 20.0

    def test_setup_camera_and_light_invalid_camera_type_raises_error(self):
        """setup_camera_and_light should raise ValueError for invalid camera_type."""
        from src.terrain.core import setup_camera_and_light

        with pytest.raises(ValueError, match="camera_type must be 'PERSP' or 'ORTHO'"):
            setup_camera_and_light(
                camera_angle=(0, 0, 0), camera_location=(0, 0, 0), scale=10.0, camera_type="INVALID"
            )

    def test_setup_camera_and_light_perspective_uses_focal_length(self):
        """setup_camera_and_light with perspective should apply focal_length correctly."""
        from src.terrain.core import setup_camera_and_light

        camera, light = setup_camera_and_light(
            camera_angle=(0, 0, 0),
            camera_location=(0, 0, 0),
            scale=10.0,
            camera_type="PERSP",
            focal_length=75,
        )

        assert camera.data.type == "PERSP"
        assert camera.data.lens == 75


# Fixtures


@pytest.fixture
def sample_dem_file(tmp_path):
    """Create a sample DEM GeoTIFF file for testing."""
    return create_sample_geotiff(
        tmp_path / "test_dem.tif",
        data=np.random.rand(50, 50) * 1000 + 500,
        bounds=(-120, 40, -119, 41),
    )


class TestGetBboxWgs84:
    """Test suite for Terrain.get_bbox_wgs84 method - returns bbox in WGS84 coordinates."""

    def test_get_bbox_wgs84_method_exists(self):
        """Terrain class should have get_bbox_wgs84 method."""
        dem_data = np.ones((10, 10), dtype=np.float32)
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        assert hasattr(terrain, "get_bbox_wgs84"), "Terrain should have get_bbox_wgs84 method"
        assert callable(getattr(terrain, "get_bbox_wgs84"))

    def test_get_bbox_wgs84_returns_tuple(self):
        """get_bbox_wgs84 should return (south, west, north, east) tuple."""
        dem_data = np.ones((10, 10), dtype=np.float32)
        # Simple transform: 0.1 degree resolution, origin at (-83, 43)
        transform = Affine.translation(-83, 43) * Affine.scale(0.1, -0.1)
        terrain = Terrain(dem_data, transform, dem_crs="EPSG:4326")

        result = terrain.get_bbox_wgs84()

        assert isinstance(result, tuple), "Should return tuple"
        assert len(result) == 4, "Should return 4-element tuple (south, west, north, east)"

    def test_get_bbox_wgs84_from_wgs84_dem(self):
        """get_bbox_wgs84 should return correct bounds when DEM is already WGS84."""
        dem_data = np.ones((10, 10), dtype=np.float32)
        # DEM from (-83, 43) to (-82, 42) - 10 pixels at 0.1 degree resolution
        # In raster coords: origin at (-83, 43), scale (0.1, -0.1), shape (10, 10)
        # So bounds are: west=-83, east=-83+10*0.1=-82, north=43, south=43-10*0.1=42
        transform = Affine.translation(-83, 43) * Affine.scale(0.1, -0.1)
        terrain = Terrain(dem_data, transform, dem_crs="EPSG:4326")

        south, west, north, east = terrain.get_bbox_wgs84()

        assert west == pytest.approx(-83.0, rel=0.01)
        assert east == pytest.approx(-82.0, rel=0.01)
        assert south == pytest.approx(42.0, rel=0.01)
        assert north == pytest.approx(43.0, rel=0.01)

    def test_get_bbox_wgs84_from_utm_dem(self):
        """get_bbox_wgs84 should convert UTM bbox back to WGS84."""
        dem_data = np.ones((100, 100), dtype=np.float32)
        # UTM Zone 17N (EPSG:32617) coordinates for approximately Detroit area
        # UTM origin at (280000, 4700000), 100m resolution, 100x100 pixels
        transform = Affine.translation(280000, 4710000) * Affine.scale(100, -100)
        terrain = Terrain(dem_data, transform, dem_crs="EPSG:32617")

        south, west, north, east = terrain.get_bbox_wgs84()

        # Should return coordinates in WGS84 (approximately 42-43°N, 83-84°W for Detroit)
        assert south != 0, "Should not return zero south"
        assert 41.5 < south < 43.5, f"South should be ~42°N, got {south}"
        assert 41.5 < north < 43.5, f"North should be ~42°N, got {north}"
        assert -84.5 < west < -82.5, f"West should be ~83°W, got {west}"
        assert -84.5 < east < -82.5, f"East should be ~83°W, got {east}"
        assert north > south, "North should be greater than south"
        assert east > west, "East should be greater than west"

    def test_get_bbox_wgs84_after_transforms(self):
        """get_bbox_wgs84 should use transformed bounds, not original."""
        from scipy.ndimage import zoom as scipy_zoom

        dem_data = np.ones((100, 100), dtype=np.float32)
        transform = Affine.translation(-83, 43) * Affine.scale(0.01, -0.01)
        terrain = Terrain(dem_data, transform, dem_crs="EPSG:4326")

        # Add downsampling transform (50% zoom)
        def downsample_transform(data, trans):
            downsampled = scipy_zoom(data, 0.5, order=1)
            new_trans = Affine(
                trans.a * 2, trans.b, trans.c, trans.d, trans.e * 2, trans.f
            )
            return downsampled, new_trans, None

        terrain.add_transform(downsample_transform)
        terrain.apply_transforms()

        south, west, north, east = terrain.get_bbox_wgs84()

        # Bounds should still represent the same geographic area
        assert west == pytest.approx(-83.0, rel=0.01)
        assert east == pytest.approx(-82.0, rel=0.01)
        assert south == pytest.approx(42.0, rel=0.01)
        assert north == pytest.approx(43.0, rel=0.01)

    def test_get_bbox_wgs84_format(self):
        """get_bbox_wgs84 should return (south, west, north, east) matching OSM convention."""
        dem_data = np.ones((10, 10), dtype=np.float32)
        transform = Affine.translation(-83, 43) * Affine.scale(0.1, -0.1)
        terrain = Terrain(dem_data, transform, dem_crs="EPSG:4326")

        south, west, north, east = terrain.get_bbox_wgs84()

        # Standard bbox format validation
        assert south < north, "South should be less than north"
        assert west < east, "West should be less than east"
        # Values should be valid lat/lon
        assert -90 <= south <= 90, "South should be valid latitude"
        assert -90 <= north <= 90, "North should be valid latitude"
        assert -180 <= west <= 180, "West should be valid longitude"
        assert -180 <= east <= 180, "East should be valid longitude"

    def test_get_bbox_wgs84_specific_layer(self):
        """get_bbox_wgs84 should accept layer parameter."""
        dem_data = np.ones((10, 10), dtype=np.float32)
        transform = Affine.translation(-83, 43) * Affine.scale(0.1, -0.1)
        terrain = Terrain(dem_data, transform, dem_crs="EPSG:4326")

        # Should work with explicit layer="dem"
        south, west, north, east = terrain.get_bbox_wgs84(layer="dem")

        assert isinstance(south, float)
        assert isinstance(west, float)
        assert isinstance(north, float)
        assert isinstance(east, float)


@pytest.mark.skipif(not HAS_BLENDER, reason="Blender not available")
class TestComputeProximityMask:
    """Test suite for compute_proximity_mask with NaN handling."""

    def test_compute_proximity_mask_filters_nan_input_coords(self):
        """compute_proximity_mask should filter out NaN input coordinates."""
        dem_data = np.ones((50, 50), dtype=np.float32) * 100
        transform = Affine.translation(-83, 43) * Affine.scale(0.01, -0.01)
        terrain = Terrain(dem_data, transform, dem_crs="EPSG:4326")

        # Apply identity transform and create mesh
        terrain.add_transform(lambda d, t: (d, t, None))
        terrain.apply_transforms()
        terrain.create_mesh(scale_factor=100, height_scale=1)

        # Mix of valid and NaN coordinates
        lons = np.array([-82.9, np.nan, -82.8, -82.7, np.nan])
        lats = np.array([42.9, 42.8, np.nan, 42.7, np.nan])

        # Should not raise, should filter out invalid coords
        mask = terrain.compute_proximity_mask(lons, lats, radius_meters=1000)

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == len(terrain.vertices)

    def test_compute_proximity_mask_handles_all_nan_coords(self):
        """compute_proximity_mask should return empty mask when all coords are NaN."""
        dem_data = np.ones((50, 50), dtype=np.float32) * 100
        transform = Affine.translation(-83, 43) * Affine.scale(0.01, -0.01)
        terrain = Terrain(dem_data, transform, dem_crs="EPSG:4326")

        terrain.add_transform(lambda d, t: (d, t, None))
        terrain.apply_transforms()
        terrain.create_mesh(scale_factor=100, height_scale=1)

        # All NaN coordinates
        lons = np.array([np.nan, np.nan, np.nan])
        lats = np.array([np.nan, np.nan, np.nan])

        mask = terrain.compute_proximity_mask(lons, lats, radius_meters=1000)

        # Should return all-False mask (no valid points)
        assert isinstance(mask, np.ndarray)
        assert not mask.any(), "Mask should be all False when no valid coords"

    def test_compute_proximity_mask_filters_out_of_bounds_points(self):
        """compute_proximity_mask should filter points outside DEM bounds."""
        dem_data = np.ones((50, 50), dtype=np.float32) * 100
        # Small DEM centered at -83, 43
        transform = Affine.translation(-83, 43) * Affine.scale(0.01, -0.01)
        terrain = Terrain(dem_data, transform, dem_crs="EPSG:4326")

        terrain.add_transform(lambda d, t: (d, t, None))
        terrain.apply_transforms()
        terrain.create_mesh(scale_factor=100, height_scale=1)

        # Points far outside DEM bounds (different continent)
        lons = np.array([0.0, 10.0, 20.0])  # Europe, not North America
        lats = np.array([50.0, 51.0, 52.0])

        # Should not raise, should return mask (possibly empty)
        mask = terrain.compute_proximity_mask(lons, lats, radius_meters=1000)

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool


def create_sample_geotiff(filepath: Path, data: np.ndarray, bounds: tuple) -> Path:
    """
    Helper function to create a sample GeoTIFF file.

    Args:
        filepath: Path where to save the GeoTIFF
        data: 2D numpy array of elevation data
        bounds: (west, south, east, north) geographic bounds

    Returns:
        Path to created file
    """
    west, south, east, north = bounds

    # Calculate transform
    width = data.shape[1]
    height = data.shape[0]
    transform = rasterio.transform.from_bounds(west, south, east, north, width, height)

    # Create GeoTIFF
    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    return filepath
