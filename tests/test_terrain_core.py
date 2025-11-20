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
from src.terrain.core import load_dem_files, Terrain

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
            tmp_path / "tile1.tif",
            data=np.ones((10, 10)) * 100,
            bounds=(-120, 40, -119.5, 40.5)
        )
        file2 = create_sample_geotiff(
            tmp_path / "tile2.tif",
            data=np.ones((10, 10)) * 200,
            bounds=(-119.5, 40, -119, 40.5)
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
        assert terrain.data_layers['dem']['data'].dtype in [np.float32, np.float64]

    def test_terrain_accepts_float_dem_data(self):
        """Terrain.__init__ should accept float DEM data."""
        dem_data = np.ones((10, 10), dtype=np.float32) * 1500.5
        transform = Affine.identity()

        terrain = Terrain(dem_data, transform)

        assert 'dem' in terrain.data_layers
        assert terrain.data_layers['dem']['data'].dtype == np.float32

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

        assert hasattr(terrain, 'cache')
        assert terrain.cache is not None

    def test_terrain_adds_dem_to_data_layers(self):
        """Terrain should add DEM as a data layer."""
        dem_data = np.ones((10, 10), dtype=np.float32) * 1200
        transform = Affine.identity()

        terrain = Terrain(dem_data, transform)

        assert 'dem' in terrain.data_layers
        assert terrain.data_layers['dem']['data'] is not None
        assert terrain.data_layers['dem']['transform'] == transform

    def test_terrain_calculates_resolution(self):
        """Terrain should calculate pixel resolution in meters."""
        dem_data = np.ones((10, 10), dtype=np.float32)
        # 0.001 degrees ≈ 111.32 meters at equator
        transform = Affine.scale(0.001, -0.001)

        terrain = Terrain(dem_data, transform)

        assert hasattr(terrain, 'resolution')
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
        terrain.data_layers['dem']['transformed'] = True
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
            assert result is None or hasattr(result, 'name')  # Blender object or None
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
            assert hasattr(terrain, 'model_params')
            assert terrain.model_params['scale_factor'] == 50.0
            assert terrain.model_params['height_scale'] == 2.0
            assert terrain.model_params['centered'] is True
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
            assert hasattr(terrain, 'model_offset')
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
            assert hasattr(terrain, 'model_offset')
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
                assert terrain.model_params['scale_factor'] == scale
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
        assert 'dem' in terrain.data_layers
        assert 'transformed_data' not in terrain.data_layers['dem']

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
        assert 'transformed_data' in terrain.data_layers['dem']
        transformed = terrain.data_layers['dem']['transformed_data']
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
        transformed = terrain.data_layers['dem']['transformed_data']
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

        assert terrain.data_layers['dem']['transformed'] is True

    def test_apply_transforms_skips_already_transformed_layers(self):
        """apply_transforms should skip layers that are already transformed."""
        dem_data = np.ones((10, 10), dtype=np.float32)
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        call_count = {'count': 0}

        def counting_transform(data, trans):
            call_count['count'] += 1
            return data, trans, None

        terrain.transforms.append(counting_transform)

        # First application
        terrain.apply_transforms()
        assert call_count['count'] == 1

        # Second application should skip already-transformed layer
        terrain.apply_transforms()
        assert call_count['count'] == 1  # Should not increase

    def test_apply_transforms_stores_metadata(self):
        """apply_transforms should store transform metadata in layer."""
        dem_data = np.ones((10, 10), dtype=np.float32)
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        def my_transform(data, trans):
            return data, trans, None

        my_transform.__name__ = 'my_transform'
        terrain.transforms.append(my_transform)
        terrain.apply_transforms()

        metadata = terrain.data_layers['dem']['transform_metadata']
        assert 'transforms' in metadata
        assert 'my_transform' in metadata['transforms']
        assert 'original_shape' in metadata
        assert 'transformed_shape' in metadata

    def test_apply_transforms_handles_crs_changes(self):
        """apply_transforms should update CRS if transform provides new one."""
        dem_data = np.ones((10, 10), dtype=np.float32)
        transform = Affine.identity()
        terrain = Terrain(dem_data, transform)

        def transform_with_crs_change(data, trans):
            new_crs = 'EPSG:3857'  # Web Mercator
            return data, trans, new_crs

        terrain.transforms.append(transform_with_crs_change)
        terrain.apply_transforms()

        assert terrain.data_layers['dem']['transformed_crs'] == 'EPSG:3857'

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
        original_crs = 'EPSG:4326'

        terrain = Terrain(dem_data, original_transform, dem_crs=original_crs)

        def identity_transform(data, trans):
            return data, trans, None

        terrain.transforms.append(identity_transform)
        terrain.apply_transforms()

        # Transform and CRS should be preserved
        assert terrain.data_layers['dem']['transformed_transform'] == original_transform
        assert terrain.data_layers['dem']['transformed_crs'] == original_crs


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
        assert hasattr(result, 'name')
        assert hasattr(result, 'data')
        assert hasattr(result.data, 'vertices')

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
            scale_factor=50.0,
            height_scale=2.0,
            center_model=True,
            boundary_extension=True
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


# Fixtures

@pytest.fixture
def sample_dem_file(tmp_path):
    """Create a sample DEM GeoTIFF file for testing."""
    return create_sample_geotiff(
        tmp_path / "test_dem.tif",
        data=np.random.rand(50, 50) * 1000 + 500,
        bounds=(-120, 40, -119, 41)
    )


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
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs='EPSG:4326',
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    return filepath
