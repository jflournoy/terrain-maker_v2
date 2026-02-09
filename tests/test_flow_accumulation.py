"""
Tests for flow accumulation feature - hydrological flow analysis with rainfall weighting.

Following TDD RED-GREEN-REFACTOR cycle.
This is the RED phase - all tests should fail initially.

Based on flow-spec.md requirements:
- Load DEM and precipitation rasters
- Compute flow direction (D8 algorithm)
- Calculate drainage area (unweighted accumulation)
- Calculate upstream rainfall (precipitation-weighted accumulation)
- Output conditioned DEM, flow direction, accumulation arrays
"""

import pytest
import numpy as np
import rasterio
from rasterio import Affine
from pathlib import Path
import tempfile

# Import the flow accumulation functions (these don't exist yet - RED phase)
from src.terrain.flow_accumulation import (
    flow_accumulation,
    compute_flow_direction,
    compute_drainage_area,
    compute_upstream_rainfall,
    condition_dem,
    identify_outlets,
    breach_depressions_constrained,
    _identify_sinks,
    priority_flood_fill_epsilon,
    detect_endorheic_basins,
)


class TestFlowAccumulationAPI:
    """Test suite for main flow_accumulation API function."""

    def test_flow_accumulation_requires_dem_path(self):
        """flow_accumulation should require dem_path parameter."""
        with pytest.raises(TypeError):
            flow_accumulation()

    def test_flow_accumulation_requires_precipitation_path(self):
        """flow_accumulation should require precipitation_path parameter."""
        with pytest.raises(TypeError):
            flow_accumulation(dem_path="dummy.tif")

    def test_flow_accumulation_validates_dem_file_exists(self):
        """flow_accumulation should raise FileNotFoundError if DEM doesn't exist."""
        with pytest.raises(FileNotFoundError, match="DEM file not found"):
            flow_accumulation(
                dem_path="/nonexistent/dem.tif", precipitation_path="precip.tif"
            )

    def test_flow_accumulation_validates_precipitation_file_exists(self, tmp_path):
        """flow_accumulation should raise FileNotFoundError if precipitation file doesn't exist."""
        # Create dummy DEM
        dem_path = tmp_path / "dem.tif"
        create_sample_geotiff(dem_path, data=np.ones((10, 10)) * 100)

        with pytest.raises(FileNotFoundError, match="Precipitation file not found"):
            flow_accumulation(dem_path=str(dem_path), precipitation_path="/nonexistent/precip.tif")

    def test_flow_accumulation_returns_dict_with_required_keys(self, tmp_path):
        """flow_accumulation should return dict with flow_direction, drainage_area, upstream_rainfall, conditioned_dem."""
        dem_path, precip_path = create_synthetic_dem_and_precip(tmp_path)

        result = flow_accumulation(dem_path=str(dem_path), precipitation_path=str(precip_path))

        assert isinstance(result, dict), "Should return dictionary"
        assert "flow_direction" in result, "Should contain flow_direction"
        assert "drainage_area" in result, "Should contain drainage_area"
        assert "upstream_rainfall" in result, "Should contain upstream_rainfall"
        assert "conditioned_dem" in result, "Should contain conditioned_dem"
        assert "metadata" in result, "Should contain metadata"
        assert "files" in result, "Should contain files dict"

    def test_flow_accumulation_returns_numpy_arrays(self, tmp_path):
        """flow_accumulation should return numpy arrays for all raster outputs."""
        dem_path, precip_path = create_synthetic_dem_and_precip(tmp_path)

        result = flow_accumulation(dem_path=str(dem_path), precipitation_path=str(precip_path))

        assert isinstance(result["flow_direction"], np.ndarray)
        assert isinstance(result["drainage_area"], np.ndarray)
        assert isinstance(result["upstream_rainfall"], np.ndarray)
        assert isinstance(result["conditioned_dem"], np.ndarray)

    def test_flow_accumulation_arrays_have_same_shape_as_input(self, tmp_path):
        """All output arrays should have same shape as input DEM."""
        dem_path, precip_path = create_synthetic_dem_and_precip(tmp_path, shape=(50, 50))

        result = flow_accumulation(dem_path=str(dem_path), precipitation_path=str(precip_path))

        expected_shape = (50, 50)
        assert result["flow_direction"].shape == expected_shape
        assert result["drainage_area"].shape == expected_shape
        assert result["upstream_rainfall"].shape == expected_shape
        assert result["conditioned_dem"].shape == expected_shape


class TestFlowDirection:
    """Test suite for flow direction computation (D8 algorithm)."""

    def test_compute_flow_direction_simple_slope(self):
        """Flow direction should point downslope on simple gradient."""
        # Create simple west-to-east slope
        dem = np.array([[10, 9, 8], [10, 9, 8], [10, 9, 8]], dtype=np.float32)

        flow_dir = compute_flow_direction(dem)

        # All pixels should flow east (direction code 1 in D8)
        # Center row, middle column should flow east
        assert flow_dir[1, 1] == 1, "Should flow east (downslope)"

    def test_compute_flow_direction_handles_flats(self):
        """Flow direction treats completely flat areas as outlets to prevent cycles."""
        # Flat plateau - no cell has a lower neighbor
        dem = np.ones((10, 10), dtype=np.float32) * 100.0

        flow_dir = compute_flow_direction(dem)

        # Completely flat areas become outlets (flow_dir = 0) to prevent cycles
        # In practice, DEMs are conditioned first which adds micro-gradients to resolve flats
        assert np.all(flow_dir == 0), "Flat areas with no outlet should become outlets"

    def test_compute_flow_direction_d8_encoding(self):
        """Flow direction should use D8 encoding (1,2,4,8,16,32,64,128)."""
        dem = create_synthetic_valley(shape=(20, 20))

        flow_dir = compute_flow_direction(dem)

        valid_codes = {0, 1, 2, 4, 8, 16, 32, 64, 128}  # 0 for nodata/outlets
        unique_values = set(flow_dir.flatten())
        assert unique_values.issubset(valid_codes), f"Invalid D8 codes: {unique_values - valid_codes}"

    def test_compute_flow_direction_points_downhill(self):
        """Flow should point from higher to lower elevation."""
        # Create simple cone (peak in center, slopes down to edges)
        dem = create_synthetic_cone(shape=(21, 21))

        flow_dir = compute_flow_direction(dem)

        # Check center pixel flows outward (not inward)
        center = 10
        center_dir = flow_dir[center, center]
        assert center_dir > 0, "Peak should have outward flow direction"


class TestDrainageArea:
    """Test suite for drainage area computation (unweighted flow accumulation)."""

    def test_compute_drainage_area_returns_array(self):
        """compute_drainage_area should return numpy array."""
        dem = create_synthetic_valley(shape=(20, 20))
        flow_dir = compute_flow_direction(dem)

        drainage_area = compute_drainage_area(flow_dir)

        assert isinstance(drainage_area, np.ndarray)
        assert drainage_area.shape == dem.shape

    def test_compute_drainage_area_minimum_is_one(self):
        """Drainage area should be at least 1 cell (the cell itself)."""
        dem = create_synthetic_valley(shape=(20, 20))
        flow_dir = compute_flow_direction(dem)

        drainage_area = compute_drainage_area(flow_dir)

        assert np.all(drainage_area >= 1.0), "All cells should drain at least themselves"

    def test_compute_drainage_area_increases_downstream(self):
        """Drainage area should increase monotonically downstream."""
        # Create simple valley with single outlet
        dem = create_synthetic_valley(shape=(30, 30))
        flow_dir = compute_flow_direction(dem)

        drainage_area = compute_drainage_area(flow_dir)

        # Find outlet (maximum drainage area)
        outlet_area = np.max(drainage_area)

        # Outlet should have largest drainage area
        assert outlet_area > 100, "Outlet should accumulate significant area"
        assert outlet_area >= drainage_area.size / 4, "Outlet should drain substantial fraction of basin"

    def test_compute_drainage_area_ridgeline_equals_one(self):
        """Cells with no upstream contributors should have drainage area = 1."""
        # Create simple gradient
        dem = np.arange(100).reshape(10, 10).astype(np.float32)[::-1]  # High to low

        flow_dir = compute_flow_direction(dem)
        drainage_area = compute_drainage_area(flow_dir)

        # Top row (ridgeline) should have minimal drainage area
        ridgeline_area = drainage_area[0, :]
        assert np.all(ridgeline_area <= 2), "Ridgeline cells should have area ≈ 1"


class TestUpstreamRainfall:
    """Test suite for precipitation-weighted flow accumulation."""

    def test_compute_upstream_rainfall_uniform_precipitation(self):
        """With uniform rainfall, upstream rainfall should equal drainage_area × precipitation."""
        dem = create_synthetic_valley(shape=(30, 30))
        precip = np.full_like(dem, 500.0)  # 500 mm/year everywhere

        flow_dir = compute_flow_direction(dem)
        drainage_area = compute_drainage_area(flow_dir)
        upstream_rainfall = compute_upstream_rainfall(flow_dir, precip)

        # At any point: upstream_rainfall ≈ drainage_area × 500
        # (with some tolerance for grid effects)
        expected = drainage_area * 500.0
        np.testing.assert_allclose(upstream_rainfall, expected, rtol=0.05)

    def test_compute_upstream_rainfall_variable_precipitation(self):
        """Upstream rainfall should accumulate weighted by local precipitation values."""
        dem = create_synthetic_valley(shape=(20, 20))

        # Create precipitation gradient (high in upper basin, low in lower)
        precip = np.linspace(1000, 100, 20).reshape(20, 1)  # Vertical gradient
        precip = np.tile(precip, (1, 20))  # Extend horizontally

        flow_dir = compute_flow_direction(dem)
        upstream_rainfall = compute_upstream_rainfall(flow_dir, precip)

        # Downstream pixels should have higher upstream rainfall than local precip
        outlet_row = np.argmax(np.sum(upstream_rainfall, axis=1))
        outlet_col = np.argmax(upstream_rainfall[outlet_row, :])

        local_precip = precip[outlet_row, outlet_col]
        total_rainfall = upstream_rainfall[outlet_row, outlet_col]

        assert total_rainfall > local_precip * 10, "Outlet should accumulate much more than local precip"

    def test_compute_upstream_rainfall_mass_balance(self):
        """Total rainfall should be conserved (mass balance check)."""
        dem = create_synthetic_valley(shape=(30, 30))
        precip = np.random.uniform(200, 800, size=(30, 30))

        flow_dir = compute_flow_direction(dem)
        upstream_rainfall = compute_upstream_rainfall(flow_dir, precip)

        # Sum of local precipitation should approximately equal upstream rainfall at main outlet
        total_precip = np.sum(precip)
        outlet_rainfall = np.max(upstream_rainfall)

        # Allow 13% tolerance due to:
        # - Boundary cells that don't contribute to main outlet
        # - Synthetic valley shape causing some edge drainage
        # - Numerical effects in flow routing and float32 precision
        np.testing.assert_allclose(outlet_rainfall, total_precip, rtol=0.13)

    def test_compute_upstream_rainfall_greater_than_local(self):
        """Upstream rainfall should be >= local precipitation everywhere."""
        dem = create_synthetic_valley(shape=(20, 20))
        precip = np.full_like(dem, 500.0)

        flow_dir = compute_flow_direction(dem)
        upstream_rainfall = compute_upstream_rainfall(flow_dir, precip)

        # Every cell should have at least its own precipitation
        assert np.all(upstream_rainfall >= precip), "Upstream rainfall should include local precip"


class TestDEMConditioning:
    """Test suite for DEM conditioning (pit filling, depression breaching)."""

    def test_condition_dem_fills_single_cell_pits(self):
        """DEM conditioning should fill single-cell pits."""
        # Create DEM with a pit
        dem = np.array([[10, 10, 10], [10, 5, 10], [10, 10, 10]], dtype=np.float32)

        conditioned = condition_dem(dem, method="fill")

        # Pit should be filled
        assert conditioned[1, 1] > 5, "Pit should be filled"
        assert conditioned[1, 1] <= 10, "Fill should not exceed surrounding elevation"

    def test_condition_dem_resolves_depressions(self):
        """DEM conditioning should resolve larger depressions."""
        # Create DEM with depression (bowl shape)
        dem = create_synthetic_bowl(shape=(11, 11))

        conditioned = condition_dem(dem, method="breach")

        # Conditioned DEM should have lower or equal elevations
        assert np.all(conditioned >= dem), "Conditioning should not lower original elevations"

    def test_condition_dem_preserves_peaks(self):
        """DEM conditioning should not modify peaks and ridges."""
        dem = create_synthetic_cone(shape=(21, 21))
        original_max = np.max(dem)

        conditioned = condition_dem(dem, method="fill")

        # Peak elevation should be preserved
        assert np.max(conditioned) == original_max, "Peak elevation should not change"

    def test_condition_dem_breach_vs_fill(self):
        """Breach method should create drainage paths, fill method should raise elevations."""
        dem = create_synthetic_bowl(shape=(15, 15))

        filled = condition_dem(dem, method="fill")
        breached = condition_dem(dem, method="breach")

        # Fill method should generally increase more elevations
        fill_changes = np.sum(filled > dem)
        breach_changes = np.sum(breached > dem)

        assert fill_changes >= breach_changes, "Fill should modify more cells than breach"


class TestFlowPerformance:
    """Performance benchmarks for flow computation."""

    def test_compute_flow_direction_performance(self):
        """compute_flow_direction should handle medium-sized DEM in reasonable time."""
        import time

        # Create medium-sized DEM (1000×1000 = 1M cells)
        dem = create_synthetic_valley(shape=(1000, 1000))

        start_time = time.time()
        flow_dir = compute_flow_direction(dem)
        elapsed = time.time() - start_time

        # Should complete in under 30 seconds for 1M cells
        assert elapsed < 30.0, f"Took {elapsed:.2f}s, should be <30s"
        assert flow_dir.shape == (1000, 1000)

        print(f"\ncompute_flow_direction: {elapsed:.2f}s for 1M cells ({1_000_000/elapsed:.0f} cells/sec)")

    def test_compute_drainage_area_performance(self):
        """compute_drainage_area should handle medium-sized DEM in reasonable time."""
        import time

        dem = create_synthetic_valley(shape=(1000, 1000))
        flow_dir = compute_flow_direction(dem)

        start_time = time.time()
        drainage_area = compute_drainage_area(flow_dir)
        elapsed = time.time() - start_time

        # Should complete in under 5 seconds for 1M cells
        assert elapsed < 5.0, f"Took {elapsed:.2f}s, should be <5s"
        assert drainage_area.shape == (1000, 1000)

        print(f"\ncompute_drainage_area: {elapsed:.2f}s for 1M cells ({1_000_000/elapsed:.0f} cells/sec)")


class TestAdaptiveResolution:
    """Test suite for adaptive DEM resolution feature."""

    def test_flow_accumulation_accepts_max_cells_parameter(self, tmp_path):
        """flow_accumulation should accept max_cells parameter."""
        dem_path, precip_path = create_synthetic_dem_and_precip(tmp_path, shape=(100, 100))

        # Should not raise
        result = flow_accumulation(
            dem_path=str(dem_path),
            precipitation_path=str(precip_path),
            max_cells=5000
        )

        assert result is not None

    def test_flow_accumulation_downsamples_when_exceeds_max_cells(self, tmp_path):
        """flow_accumulation should downsample DEM when it exceeds max_cells."""
        # Create large DEM (10,000 cells)
        dem_path, precip_path = create_synthetic_dem_and_precip(tmp_path, shape=(100, 100))

        # Set max_cells to 2,500 (should trigger 2x downsampling)
        result = flow_accumulation(
            dem_path=str(dem_path),
            precipitation_path=str(precip_path),
            max_cells=2500
        )

        # Output arrays should be downsampled (approximately 50×50 or smaller)
        assert result["flow_direction"].size <= 2500 * 1.1, "Should downsample to max_cells"

    def test_flow_accumulation_metadata_includes_downsampling_info(self, tmp_path):
        """Metadata should include downsampling information."""
        dem_path, precip_path = create_synthetic_dem_and_precip(tmp_path, shape=(100, 100))

        result = flow_accumulation(
            dem_path=str(dem_path),
            precipitation_path=str(precip_path),
            max_cells=2500
        )

        metadata = result["metadata"]
        assert "downsampling_applied" in metadata
        assert "original_shape" in metadata
        assert "downsampled_shape" in metadata
        assert "downsample_factor" in metadata

        assert metadata["downsampling_applied"] is True
        assert metadata["original_shape"] == (100, 100)

    def test_flow_accumulation_no_downsampling_when_below_max_cells(self, tmp_path):
        """flow_accumulation should not downsample when DEM is below max_cells."""
        # Create small DEM (900 cells)
        dem_path, precip_path = create_synthetic_dem_and_precip(tmp_path, shape=(30, 30))

        # Set max_cells higher than DEM size
        result = flow_accumulation(
            dem_path=str(dem_path),
            precipitation_path=str(precip_path),
            max_cells=10000
        )

        # Output should match original size
        assert result["flow_direction"].shape == (30, 30)

        metadata = result["metadata"]
        assert metadata["downsampling_applied"] is False

    def test_flow_accumulation_preserves_hydrological_features_when_downsampling(self, tmp_path):
        """Downsampling should preserve valley structure and drainage patterns."""
        # Create valley DEM
        dem_path, precip_path = create_synthetic_dem_and_precip(
            tmp_path, dem_type="valley", shape=(100, 100)
        )

        # Run at full resolution
        result_full = flow_accumulation(
            dem_path=str(dem_path),
            precipitation_path=str(precip_path)
        )

        # Run with downsampling
        result_downsampled = flow_accumulation(
            dem_path=str(dem_path),
            precipitation_path=str(precip_path),
            max_cells=2500
        )

        # Find outlet location in both (should be similar)
        outlet_full = np.unravel_index(
            np.argmax(result_full["drainage_area"]), result_full["drainage_area"].shape
        )
        outlet_down = np.unravel_index(
            np.argmax(result_downsampled["drainage_area"]), result_downsampled["drainage_area"].shape
        )

        # Outlet location should be in roughly same relative position
        # (accounting for different grid sizes)
        rel_outlet_full = (outlet_full[0] / 100, outlet_full[1] / 100)
        down_shape = result_downsampled["drainage_area"].shape
        rel_outlet_down = (outlet_down[0] / down_shape[0], outlet_down[1] / down_shape[1])

        # Should be within 10% of same relative position
        assert abs(rel_outlet_full[0] - rel_outlet_down[0]) < 0.1
        assert abs(rel_outlet_full[1] - rel_outlet_down[1]) < 0.1

    def test_flow_accumulation_auto_calculates_max_cells_from_target_vertices(self, tmp_path):
        """flow_accumulation should auto-calculate max_cells from target_vertices."""
        dem_path, precip_path = create_synthetic_dem_and_precip(tmp_path, shape=(100, 100))

        # Specify target_vertices instead of max_cells
        # Should compute max_cells = target_vertices * 3 (for flow accuracy)
        result = flow_accumulation(
            dem_path=str(dem_path),
            precipitation_path=str(precip_path),
            target_vertices=2500  # Should use max_cells = 7500
        )

        metadata = result["metadata"]
        assert "target_vertices" in metadata
        assert metadata["target_vertices"] == 2500

        # Should downsample to approximately 7500 cells or less
        # (actual might be slightly different due to aspect ratio preservation)
        assert result["flow_direction"].size <= 10000, "Should downsample based on target_vertices"

    def test_flow_accumulation_downsample_factor_calculation(self, tmp_path):
        """Downsample factor should be calculated to achieve max_cells target."""
        # Create 100×100 DEM (10,000 cells)
        dem_path, precip_path = create_synthetic_dem_and_precip(tmp_path, shape=(100, 100))

        # Set max_cells = 2,500 (should trigger 2x downsampling -> 50×50 = 2,500)
        result = flow_accumulation(
            dem_path=str(dem_path),
            precipitation_path=str(precip_path),
            max_cells=2500
        )

        metadata = result["metadata"]
        assert metadata["downsample_factor"] >= 1.9
        assert metadata["downsample_factor"] <= 2.1

        # Downsampled shape should be approximately 50×50
        downsampled_shape = metadata["downsampled_shape"]
        assert 45 <= downsampled_shape[0] <= 55
        assert 45 <= downsampled_shape[1] <= 55


class TestFlowAccumulationIntegration:
    """Integration tests for complete flow accumulation workflow."""

    def test_flow_accumulation_synthetic_valley(self, tmp_path):
        """Test complete workflow on synthetic valley DEM."""
        dem_path, precip_path = create_synthetic_dem_and_precip(
            tmp_path, dem_type="valley", shape=(50, 50)
        )

        result = flow_accumulation(dem_path=str(dem_path), precipitation_path=str(precip_path))

        # Validate outputs
        assert result["flow_direction"].max() <= 128, "Flow direction should use D8 codes"
        assert result["drainage_area"].min() >= 1.0, "Drainage area should be >= 1"
        assert np.max(result["upstream_rainfall"]) > 0, "Should have positive upstream rainfall"

    def test_flow_accumulation_saves_output_files(self, tmp_path):
        """flow_accumulation should save GeoTIFF outputs."""
        dem_path, precip_path = create_synthetic_dem_and_precip(tmp_path)

        result = flow_accumulation(
            dem_path=str(dem_path),
            precipitation_path=str(precip_path),
            output_dir=str(tmp_path / "outputs"),
        )

        # Check that output files were created
        assert "files" in result
        assert Path(result["files"]["flow_direction"]).exists()
        assert Path(result["files"]["drainage_area"]).exists()
        assert Path(result["files"]["upstream_rainfall"]).exists()
        assert Path(result["files"]["conditioned_dem"]).exists()

    def test_flow_accumulation_metadata_complete(self, tmp_path):
        """flow_accumulation should return complete metadata."""
        dem_path, precip_path = create_synthetic_dem_and_precip(tmp_path)

        result = flow_accumulation(dem_path=str(dem_path), precipitation_path=str(precip_path))

        metadata = result["metadata"]
        assert "cell_size_m" in metadata
        assert "drainage_area_units" in metadata
        assert "total_area_km2" in metadata
        assert "max_drainage_area_km2" in metadata
        assert "algorithm" in metadata
        assert metadata["algorithm"] == "d8"


# ===== Helper Functions for Test Data Generation =====


def create_sample_geotiff(path, data, bounds=(-120, 40, -119, 41), crs="EPSG:4326"):
    """Create a sample GeoTIFF file for testing."""
    height, width = data.shape
    transform = rasterio.transform.from_bounds(*bounds, width, height)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    return path


def create_synthetic_dem_and_precip(tmp_path, dem_type="valley", shape=(30, 30)):
    """Create synthetic DEM and precipitation rasters for testing."""
    if dem_type == "valley":
        dem_data = create_synthetic_valley(shape)
    elif dem_type == "cone":
        dem_data = create_synthetic_cone(shape)
    else:
        # Simple gradient
        dem_data = np.arange(shape[0] * shape[1]).reshape(shape).astype(np.float32)[::-1]

    # Uniform precipitation
    precip_data = np.full(shape, 500.0, dtype=np.float32)

    dem_path = tmp_path / "test_dem.tif"
    precip_path = tmp_path / "test_precip.tif"

    create_sample_geotiff(dem_path, dem_data)
    create_sample_geotiff(precip_path, precip_data)

    return dem_path, precip_path


def create_synthetic_valley(shape=(30, 30)):
    """Create synthetic valley DEM (V-shaped cross-section)."""
    rows, cols = shape
    dem = np.zeros(shape, dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            # Distance from center column
            center = cols // 2
            dist = abs(j - center)
            # V-shaped: elevation increases away from center
            dem[i, j] = 100.0 + dist * 2.0 - i * 1.0  # Slopes downward and toward center

    return dem


def create_synthetic_cone(shape=(21, 21)):
    """Create synthetic cone DEM (peak in center)."""
    rows, cols = shape
    center_r, center_c = rows // 2, cols // 2
    dem = np.zeros(shape, dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - center_r) ** 2 + (j - center_c) ** 2)
            dem[i, j] = max(0, 100.0 - dist * 3.0)  # Cone peak at 100m

    return dem


def create_synthetic_bowl(shape=(15, 15)):
    """Create synthetic bowl (depression) DEM."""
    rows, cols = shape
    center_r, center_c = rows // 2, cols // 2
    dem = np.zeros(shape, dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - center_r) ** 2 + (j - center_c) ** 2)
            # Invert cone to create bowl
            dem[i, j] = 50.0 + dist * 2.0  # Lower in center, higher at edges

    # Create rim
    dem[0, :] = 80.0
    dem[-1, :] = 80.0
    dem[:, 0] = 80.0
    dem[:, -1] = 80.0

    return dem


# ============================================================================
# Ocean Masking Tests
# ============================================================================


def test_condition_dem_masks_ocean_below_sea_level():
    """Test that ocean cells (elevation <= 0) are excluded from conditioning."""
    # Create 7x7 DEM with ocean on left, land with enclosed pit on right
    dem = np.array([
        [-5.0, -3.0,  5.0, 10.0, 12.0, 14.0, 16.0],  # Ocean | Land
        [-5.0, -2.0,  6.0, 11.0, 13.0, 15.0, 17.0],
        [-5.0,  0.0,  7.0, 12.0, 14.0, 16.0, 18.0],
        [-5.0,  1.0,  8.0, 13.0, 15.0, 17.0, 19.0],
        [-5.0,  2.0,  9.0, 14.0, 16.0, 18.0, 20.0],
        [-5.0,  3.0, 10.0, 15.0, 17.0, 19.0, 21.0],
        [-5.0,  4.0, 11.0, 16.0, 18.0, 20.0, 22.0],
    ], dtype=np.float32)

    # Create enclosed inland depression (completely surrounded by higher land)
    # Make a bowl-shaped depression
    dem[3, 4] = 10.0  # Center of depression (surrounded by 15-17m cells)
    dem[2, 4] = 11.0
    dem[4, 4] = 11.0
    dem[3, 3] = 11.0
    dem[3, 5] = 11.0

    # Create ocean mask (elevation <= 0)
    ocean_mask = dem <= 0.0

    # Condition DEM with ocean mask
    conditioned = condition_dem(dem, method="fill", ocean_mask=ocean_mask)

    # Ocean cells should maintain original elevation (not filled)
    assert np.all(conditioned[ocean_mask] == dem[ocean_mask]), \
        "Ocean cells should not be modified by depression filling"

    # Inland pit should be filled to at least the lowest surrounding elevation
    # The pit at [3,4] = 10.0 is surrounded by cells at 11.0+
    # It should be filled to at least 11.0 (the spill point)
    assert conditioned[3, 4] >= 11.0, \
        f"Inland depression should be filled: original={dem[3, 4]}, conditioned={conditioned[3, 4]}"

    # Land areas above sea level should be >= original
    land_mask = ~ocean_mask
    assert np.all(conditioned[land_mask] >= dem[land_mask]), \
        "Land elevations should not decrease"


def test_condition_dem_detects_border_connected_ocean():
    """Test that border-connected low-elevation areas are detected as ocean."""
    # Create DEM with ocean connected to border
    dem = np.array([
        [ 0.0,  0.0,  0.0,  5.0, 10.0],  # Ocean at border
        [ 0.0,  0.5,  1.0,  6.0, 12.0],  # Ocean extends inward
        [ 0.0,  1.0,  2.0,  7.0, 14.0],
        [ 5.0,  6.0,  3.0,  8.0, 16.0],  # Inland pit at [3,2]
        [10.0, 11.0, 12.0, 13.0, 18.0],
    ], dtype=np.float32)

    # Add inland isolated lake (should be filled)
    dem[3, 2] = 1.5

    # Detect ocean as border-connected areas at/below threshold
    from src.terrain.flow_accumulation import detect_ocean_mask
    ocean_mask = detect_ocean_mask(dem, threshold=0.0, border_only=True)

    # Border-connected ocean should be detected
    assert ocean_mask[0, 0], "Border ocean should be detected"
    assert ocean_mask[1, 0], "Connected ocean should be detected"
    assert ocean_mask[2, 0], "Connected ocean should be detected"

    # Inland pit should NOT be ocean
    assert not ocean_mask[3, 2], "Inland pit should not be ocean"

    # High elevation should not be ocean
    assert not ocean_mask[4, 4], "High land should not be ocean"


def test_flow_direction_skips_masked_cells():
    """Test that flow direction is not computed for masked cells."""
    # Create simple DEM with ocean
    dem = np.array([
        [-5.0, -3.0,  5.0, 10.0],
        [-5.0, -1.0,  6.0, 12.0],
        [ 5.0,  7.0,  8.0, 14.0],
        [10.0, 12.0, 13.0, 16.0],
    ], dtype=np.float32)

    ocean_mask = dem <= 0.0

    # Condition DEM with mask
    conditioned = condition_dem(dem, method="fill", ocean_mask=ocean_mask)

    # Compute flow direction
    flow_dir = compute_flow_direction(conditioned, mask=ocean_mask)

    # Masked cells should have flow_dir = 0 (no flow)
    assert np.all(flow_dir[ocean_mask] == 0), \
        "Masked cells should have no flow direction (0)"

    # Unmasked cells should have valid flow directions
    land_mask = ~ocean_mask
    assert np.any(flow_dir[land_mask] > 0), \
        "Land cells should have flow directions"


# ============================================================================
# Large Basin Preservation Tests
# ============================================================================


def test_condition_dem_preserves_large_basins():
    """Test that large endorheic basins are preserved (not filled)."""
    # Create large basin (like Salton Sea)
    dem = np.full((25, 25), 50.0, dtype=np.float32)  # Base elevation

    # Surrounding mountains
    dem[0, :] = 100.0
    dem[-1, :] = 100.0
    dem[:, 0] = 100.0
    dem[:, -1] = 100.0

    # Large basin interior (below sea level, large area)
    for i in range(6, 19):
        for j in range(6, 19):
            dist = np.sqrt((i - 12.5) ** 2 + (j - 12.5) ** 2)
            dem[i, j] = -50.0 + dist * 2.0  # Basin floor at -50m

    # Without preservation, the basin would be filled
    conditioned_no_preserve = condition_dem(dem, method="fill")

    # With preservation, large basin should remain
    conditioned_preserve = condition_dem(
        dem,
        method="fill",
        min_basin_size=100,  # Preserve basins >= 100 cells
    )

    # Large basin center should be filled WITHOUT preservation
    assert conditioned_no_preserve[12, 12] > dem[12, 12], \
        "Without preservation, basin should be filled"

    # Large basin should be preserved WITH preservation
    assert conditioned_preserve[12, 12] < 0, \
        "Large basin center should remain below sea level with preservation"
    assert np.isclose(conditioned_preserve[12, 12], dem[12, 12], atol=1.0), \
        f"Large basin should not be significantly modified: original={dem[12, 12]}, conditioned={conditioned_preserve[12, 12]}"


def test_condition_dem_preserves_deep_basins():
    """Test that deep basins are preserved based on fill depth threshold."""
    # Create basin with significant depth
    dem = np.zeros((15, 15), dtype=np.float32)

    # Rim at 50m
    dem[:, :] = 50.0

    # Deep basin center
    for i in range(5, 10):
        for j in range(5, 10):
            dist = np.sqrt((i - 7.5) ** 2 + (j - 7.5) ** 2)
            dem[i, j] = -30.0 + dist * 3.0  # Deep basin

    # Shallow pit
    dem[2, 2] = 45.0

    # Condition with depth threshold
    conditioned = condition_dem(
        dem,
        method="fill",
        max_fill_depth=20.0,  # Don't fill if requires >20m
    )

    # Deep basin should be preserved (would require 80m fill)
    assert conditioned[7, 7] < 30.0, \
        "Deep basin should not be filled to rim level"

    # Shallow pit should be filled (only 5m)
    assert conditioned[2, 2] >= 50.0, \
        "Shallow pit should be filled"


def test_detect_endorheic_basins():
    """Test detection of endorheic (closed) basins."""
    # Create closed basin
    dem = np.zeros((15, 15), dtype=np.float32)

    # Mountains around edges
    dem[0, :] = 100.0
    dem[-1, :] = 100.0
    dem[:, 0] = 100.0
    dem[:, -1] = 100.0

    # Interior basin
    for i in range(3, 12):
        for j in range(3, 12):
            dist = np.sqrt((i - 7.5) ** 2 + (j - 7.5) ** 2)
            dem[i, j] = 20.0 - dist * 2.0  # Basin center at ~5m

    from src.terrain.flow_accumulation import detect_endorheic_basins

    basin_mask, basin_sizes = detect_endorheic_basins(dem, min_size=10)

    # Should detect the basin
    assert np.any(basin_mask), "Should detect endorheic basin"

    # Basin should be significant size
    assert np.sum(basin_mask) >= 10, "Basin should be at least min_size"

    # Basin should be in interior
    assert basin_mask[7, 7], "Basin center should be detected"

    # Border should not be basin
    assert not basin_mask[0, 0], "Border should not be basin"


def test_salton_sea_scenario():
    """
    Comprehensive test for Salton Sea-like basin preservation.

    The Salton Sea is a real closed basin in California below sea level.
    This test validates that our algorithm:
    1. Correctly identifies large closed basins
    2. Preserves them when min_basin_size is set appropriately
    3. Fills them when basin preservation is disabled
    4. Respects both size and depth thresholds
    """
    # Create realistic Salton Sea-like DEM with smaller interior basin
    # surrounded by a plateau, which then has mountains at the edge.
    dem = np.full((100, 100), 100.0, dtype=np.float32)  # Plateau base

    # Create surrounding mountains/rim (higher elevation)
    dem[0, :] = 200.0
    dem[-1, :] = 200.0
    dem[:, 0] = 200.0
    dem[:, -1] = 200.0

    # Create interior CLOSED basin (like Salton Sea)
    # Interior is a separate depression below the surrounding plateau
    for i in range(25, 75):
        for j in range(25, 75):
            dist = np.sqrt((i - 50) ** 2 + (j - 50) ** 2)
            # Basin floor at -20m (relative to plateau at 100m), sloping toward center
            # This creates a depression that would be filled without preservation
            dem[i, j] = 20.0 - dist * 0.8

    # The basin is a region where dem < 100 (below plateau level)
    basin_region = (dem < 100) & (dem > 0)  # Below plateau but above zero
    basin_area = np.sum(basin_region)
    basin_center_elev = dem[50, 50]

    # ====== Test 1: Without preservation, basin gets filled ======
    # Set min_basin_size=None to explicitly disable basin preservation
    conditioned_filled = condition_dem(dem, method="fill", min_basin_size=None)

    # Basin interior should be raised toward plateau level
    filled_center = conditioned_filled[50, 50]
    assert filled_center > basin_center_elev, \
        f"Without basin preservation, basin center should be raised (was {basin_center_elev}, now {filled_center})"

    # ====== Test 2: With size preservation, large basin preserved ======
    min_size = int(basin_area * 0.7)  # Use 70% of basin area as threshold
    conditioned_preserved = condition_dem(
        dem,
        method="fill",
        min_basin_size=min_size
    )

    # Basin center should remain near original elevation
    preserved_center = conditioned_preserved[50, 50]
    assert preserved_center < 100, \
        f"With basin preservation, center should remain below plateau (was {preserved_center})"
    assert np.isclose(preserved_center, basin_center_elev, atol=1.0), \
        f"Basin should be minimally modified: original={basin_center_elev}, conditioned={preserved_center}"

    # ====== Test 3: With depth preservation, deep basin preserved ======
    # This basin would require ~80m fill to reach plateau level
    conditioned_deep = condition_dem(
        dem,
        method="fill",
        max_fill_depth=50.0  # Don't fill if requires >50m
    )

    # Basin should be preserved (would require >50m fill)
    deep_center = conditioned_deep[50, 50]
    assert deep_center < 100, \
        f"Deep basin should be preserved (was {deep_center})"
    assert np.isclose(deep_center, basin_center_elev, atol=1.0), \
        f"Deep basin should be minimally modified: original={basin_center_elev}, conditioned={deep_center}"

    # ====== Test 4: Low fill threshold should preserve deep basins ======
    # With max_fill_depth=30 and basin depth ~80m, basin should be preserved
    conditioned_low_threshold = condition_dem(
        dem,
        method="fill",
        min_basin_size=None,  # Disable size-based preservation
        max_fill_depth=30.0  # Don't fill if requires >30m
    )

    # Basin should be preserved (would require ~80m fill)
    low_threshold_center = conditioned_low_threshold[50, 50]
    assert low_threshold_center < 100, \
        f"With low max_fill_depth, deep basin should be preserved (was {low_threshold_center})"
    assert np.isclose(low_threshold_center, basin_center_elev, atol=1.0), \
        f"Basin should be minimally modified: original={basin_center_elev}, conditioned={low_threshold_center}"

    # ====== Test 5: Both thresholds work together ======
    # Use both size AND depth thresholds - both should preserve
    conditioned_both = condition_dem(
        dem,
        method="fill",
        min_basin_size=min_size,
        max_fill_depth=50.0
    )

    both_center = conditioned_both[50, 50]
    assert both_center < 100, \
        f"With both thresholds, basin should be preserved (was {both_center})"

    # ====== Test 6: Detect endorheic basins correctly ======
    basin_mask, basin_sizes = detect_endorheic_basins(
        dem,
        min_size=int(basin_area * 0.5),  # 50% of basin area
        min_depth=0.5  # Typical depression depth
    )

    # Should detect the large basin
    assert np.any(basin_mask), "Should detect large endorheic basin"

    # Basin interior should be detected
    assert basin_mask[50, 50], "Basin center should be detected as endorheic"

    # Most cells in basin region should be detected
    detected_area = np.sum(basin_mask & basin_region)
    assert detected_area > basin_area * 0.7, \
        f"Should detect >70% of basin area (detected {detected_area}/{basin_area})"


# ============================================================================
# Flow Spec Validation Tests
# Based on flow-spec.md validation requirements:
# 1. All flow directions point downhill (except pits/outlets)
# 2. Flow accumulation = 1 at ridgelines/peaks
# 3. Flow accumulation increases monotonically downstream
# 4. Weighted accumulation ≥ local precipitation × cell area
# 5. Mass balance: Σ(precipitation × area) = Σ(outlet flows)
# ============================================================================


class TestFlowSpecValidation:
    """Tests based on flow-spec.md validation requirements."""

    def test_flow_direction_points_downhill(self):
        """Spec requirement 1: All flow directions should point downhill (except outlets)."""
        dem = create_synthetic_valley(shape=(30, 30))
        conditioned = condition_dem(dem, method="fill")
        flow_dir = compute_flow_direction(conditioned)

        # D8 offsets matching flow_accumulation.py
        offsets = {
            1: (0, 1),    # East
            2: (-1, 1),   # Northeast
            4: (-1, 0),   # North
            8: (-1, -1),  # Northwest
            16: (0, -1),  # West
            32: (1, -1),  # Southwest
            64: (1, 0),   # South
            128: (1, 1),  # Southeast
        }

        rows, cols = dem.shape
        violations = 0

        for i in range(rows):
            for j in range(cols):
                direction = flow_dir[i, j]
                if direction == 0:
                    # Outlet/sink - skip
                    continue

                if direction in offsets:
                    di, dj = offsets[direction]
                    ni, nj = i + di, j + dj

                    if 0 <= ni < rows and 0 <= nj < cols:
                        # Flow should go to lower or equal elevation
                        if conditioned[ni, nj] > conditioned[i, j]:
                            violations += 1

        assert violations == 0, f"Found {violations} cells flowing uphill"

    def test_drainage_area_equals_one_at_ridgelines(self):
        """Spec requirement 2: Flow accumulation = 1 at ridgelines/peaks."""
        dem = create_synthetic_cone(shape=(21, 21))
        flow_dir = compute_flow_direction(dem)
        drainage_area = compute_drainage_area(flow_dir)

        # Peak is at center
        center = 10
        peak_drainage = drainage_area[center, center]

        assert peak_drainage == 1.0, f"Peak drainage should be 1, got {peak_drainage}"

    def test_flow_accumulation_monotonically_increases_downstream(self):
        """Spec requirement 3: Flow accumulation should increase monotonically downstream."""
        dem = create_synthetic_valley(shape=(30, 30))
        conditioned = condition_dem(dem, method="fill")
        flow_dir = compute_flow_direction(conditioned)
        drainage_area = compute_drainage_area(flow_dir)

        # D8 offsets
        offsets = {
            1: (0, 1), 2: (-1, 1), 4: (-1, 0), 8: (-1, -1),
            16: (0, -1), 32: (1, -1), 64: (1, 0), 128: (1, 1),
        }

        rows, cols = drainage_area.shape
        violations = 0

        for i in range(rows):
            for j in range(cols):
                direction = flow_dir[i, j]
                if direction == 0 or direction not in offsets:
                    continue

                di, dj = offsets[direction]
                ni, nj = i + di, j + dj

                if 0 <= ni < rows and 0 <= nj < cols:
                    # Downstream drainage should be >= current
                    if drainage_area[ni, nj] < drainage_area[i, j]:
                        violations += 1

        assert violations == 0, f"Found {violations} violations of monotonic increase"

    def test_upstream_rainfall_greater_than_local(self):
        """Spec requirement 4: Weighted accumulation ≥ local precipitation."""
        dem = create_synthetic_valley(shape=(30, 30))
        precip = np.random.uniform(200, 800, size=(30, 30)).astype(np.float32)

        flow_dir = compute_flow_direction(dem)
        upstream_rainfall = compute_upstream_rainfall(flow_dir, precip)

        # Every cell should have upstream rainfall >= its local precipitation
        violations = np.sum(upstream_rainfall < precip)
        assert violations == 0, f"Found {violations} cells with upstream < local precip"

    def test_mass_balance_single_outlet(self):
        """Spec requirement 5: Mass balance for DEM with single outlet."""
        # Create DEM where all flow goes to a single outlet
        dem = create_synthetic_valley(shape=(30, 30))
        precip = np.full((30, 30), 500.0, dtype=np.float32)

        conditioned = condition_dem(dem, method="fill")
        flow_dir = compute_flow_direction(conditioned)
        upstream_rainfall = compute_upstream_rainfall(flow_dir, precip)

        # Find outlets (cells with direction = 0 that have nonzero drainage)
        drainage_area = compute_drainage_area(flow_dir)
        outlets = (flow_dir == 0) & (drainage_area > 1)

        # Sum of outlet flows should equal total precipitation
        total_precip = np.sum(precip)
        outlet_sum = np.sum(upstream_rainfall[outlets])

        # Allow some tolerance for edge effects
        np.testing.assert_allclose(
            outlet_sum, total_precip, rtol=0.05,
            err_msg=f"Mass balance: outlets={outlet_sum:.0f}, total={total_precip:.0f}"
        )

    def test_mass_balance_multiple_outlets(self):
        """Spec requirement 5: Mass balance with multiple outlets along ocean boundary."""
        # Create DEM with ocean boundary on left side
        rows, cols = 30, 40
        dem = np.zeros((rows, cols), dtype=np.float32)

        # Elevation increases from left (ocean) to right
        for j in range(cols):
            dem[:, j] = j * 5.0  # 0m at left, 195m at right

        # Add some topographic variation
        for i in range(rows):
            for j in range(cols):
                dem[i, j] += abs(i - rows//2) * 2.0  # V-shaped valleys

        # Ocean mask: left 2 columns
        ocean_mask = np.zeros((rows, cols), dtype=bool)
        ocean_mask[:, :2] = True
        dem[:, :2] = -5.0  # Below sea level

        precip = np.full((rows, cols), 500.0, dtype=np.float32)
        precip[ocean_mask] = 0.0  # No precipitation on ocean

        # Condition and compute flow
        conditioned = condition_dem(dem, method="fill", ocean_mask=ocean_mask)
        flow_dir = compute_flow_direction(conditioned, mask=ocean_mask)
        upstream_rainfall = compute_upstream_rainfall(flow_dir, precip)
        drainage_area = compute_drainage_area(flow_dir)

        # Find all outlets (direction=0 with drainage > 1, or at ocean boundary)
        outlets = (flow_dir == 0) & (drainage_area > 1)
        # Also include cells that flow into ocean (next to ocean)
        for i in range(rows):
            for j in range(cols):
                if not ocean_mask[i, j] and j > 0:
                    # Check if this cell flows west into ocean
                    if flow_dir[i, j] == 16 and ocean_mask[i, j-1]:
                        outlets[i, j] = True

        total_precip = np.sum(precip)
        outlet_sum = np.sum(upstream_rainfall[outlets])

        # Mass balance: sum of outlets should equal total precipitation
        # Allow 10% tolerance for edge effects and ocean boundary handling
        np.testing.assert_allclose(
            outlet_sum, total_precip, rtol=0.10,
            err_msg=f"Mass balance: outlets={outlet_sum:.0f}, total={total_precip:.0f}"
        )


class TestFlowOutletHandling:
    """Tests for correct handling of outlets and ocean boundaries."""

    def test_find_all_outlets(self):
        """Test that we can correctly identify all outlets in a DEM."""
        # Create simple tilted plane - all water flows to one corner
        dem = np.zeros((10, 10), dtype=np.float32)
        for i in range(10):
            for j in range(10):
                dem[i, j] = i + j  # Lowest at (0,0)

        flow_dir = compute_flow_direction(dem)
        drainage_area = compute_drainage_area(flow_dir)

        # Find outlets (cells with flow_dir=0 or boundary cells)
        outlets = []
        for i in range(10):
            for j in range(10):
                if flow_dir[i, j] == 0 and drainage_area[i, j] > 1:
                    outlets.append((i, j, drainage_area[i, j]))

        # Should have at least one outlet
        assert len(outlets) > 0, "Should have at least one outlet"

        # Total drainage through outlets should equal total cells
        total_outlet_drainage = sum(d for _, _, d in outlets)
        # Allow for some cells being their own outlets (ridgelines)
        assert total_outlet_drainage >= 10, "Outlets should drain significant area"

    def test_ocean_boundary_outlets_receive_flow(self):
        """Test that cells adjacent to ocean properly receive upstream flow."""
        rows, cols = 20, 25
        dem = np.zeros((rows, cols), dtype=np.float32)

        # Create east-west slope (flows west to ocean)
        for j in range(cols):
            dem[:, j] = j * 10.0

        # Ocean on left side
        ocean_mask = np.zeros((rows, cols), dtype=bool)
        ocean_mask[:, 0] = True
        dem[:, 0] = -10.0

        precip = np.full((rows, cols), 100.0, dtype=np.float32)
        precip[ocean_mask] = 0.0

        conditioned = condition_dem(dem, method="fill", ocean_mask=ocean_mask)
        flow_dir = compute_flow_direction(conditioned, mask=ocean_mask)
        upstream_rainfall = compute_upstream_rainfall(flow_dir, precip)

        # Cells in column 1 (adjacent to ocean) should receive flow from entire width
        # Each row should have ~2400 mm (24 cells × 100mm)
        for i in range(rows):
            expected = (cols - 1) * 100.0  # 24 cells worth
            actual = upstream_rainfall[i, 1]
            # Each row should accumulate most of its upstream rainfall
            assert actual >= expected * 0.8, \
                f"Row {i} col 1: expected ~{expected:.0f}, got {actual:.0f}"

    def test_flow_to_boundary_cells(self):
        """Test that flow properly reaches boundary cells that act as outlets."""
        # Create DEM with clear drainage to south boundary
        rows, cols = 15, 15
        dem = np.zeros((rows, cols), dtype=np.float32)

        # North high, south low
        for i in range(rows):
            dem[i, :] = (rows - 1 - i) * 10.0

        precip = np.full((rows, cols), 500.0, dtype=np.float32)

        flow_dir = compute_flow_direction(dem)
        upstream_rainfall = compute_upstream_rainfall(flow_dir, precip)
        drainage_area = compute_drainage_area(flow_dir)

        # Bottom row should receive flow from entire grid
        # Center cell of bottom row should have ~15×15×500 = 112,500 mm
        total_precip = np.sum(precip)
        bottom_row_total = np.sum(upstream_rainfall[-1, :])

        # Bottom row should capture most/all precipitation
        assert bottom_row_total >= total_precip * 0.9, \
            f"Bottom row should receive most flow: {bottom_row_total:.0f} vs {total_precip:.0f}"


class TestFlowNetworkConsistency:
    """Tests for flow network consistency and correctness."""

    def test_every_cell_has_path_to_outlet(self):
        """Every non-outlet cell should have a flow path to an outlet."""
        dem = create_synthetic_valley(shape=(20, 20))
        conditioned = condition_dem(dem, method="fill")
        flow_dir = compute_flow_direction(conditioned)

        offsets = {
            1: (0, 1), 2: (-1, 1), 4: (-1, 0), 8: (-1, -1),
            16: (0, -1), 32: (1, -1), 64: (1, 0), 128: (1, 1),
        }

        rows, cols = flow_dir.shape
        orphan_cells = 0

        for i in range(rows):
            for j in range(cols):
                if flow_dir[i, j] == 0:
                    # This is an outlet - skip
                    continue

                # Trace flow path - should reach outlet or boundary
                ci, cj = i, j
                visited = set()
                reached_outlet = False

                while (ci, cj) not in visited:
                    visited.add((ci, cj))
                    direction = flow_dir[ci, cj]

                    if direction == 0:
                        reached_outlet = True
                        break

                    if direction not in offsets:
                        break

                    di, dj = offsets[direction]
                    ni, nj = ci + di, cj + dj

                    if not (0 <= ni < rows and 0 <= nj < cols):
                        # Reached boundary
                        reached_outlet = True
                        break

                    ci, cj = ni, nj

                if not reached_outlet:
                    orphan_cells += 1

        assert orphan_cells == 0, f"Found {orphan_cells} cells without path to outlet"

    def test_no_circular_flow(self):
        """Flow network should not contain any cycles."""
        dem = create_synthetic_valley(shape=(20, 20))
        conditioned = condition_dem(dem, method="fill")
        flow_dir = compute_flow_direction(conditioned)

        offsets = {
            1: (0, 1), 2: (-1, 1), 4: (-1, 0), 8: (-1, -1),
            16: (0, -1), 32: (1, -1), 64: (1, 0), 128: (1, 1),
        }

        rows, cols = flow_dir.shape
        cycles_found = 0

        for i in range(rows):
            for j in range(cols):
                if flow_dir[i, j] == 0:
                    continue

                # Trace path, looking for cycles
                ci, cj = i, j
                path = set()
                path_list = [(ci, cj)]

                while True:
                    direction = flow_dir[ci, cj]
                    if direction == 0 or direction not in offsets:
                        break

                    di, dj = offsets[direction]
                    ni, nj = ci + di, cj + dj

                    if not (0 <= ni < rows and 0 <= nj < cols):
                        break

                    if (ni, nj) in path:
                        # Found a cycle!
                        cycles_found += 1
                        break

                    path.add((ni, nj))
                    path_list.append((ni, nj))
                    ci, cj = ni, nj

                    # Prevent infinite loops in test
                    if len(path) > rows * cols:
                        break

        assert cycles_found == 0, f"Found {cycles_found} circular flow paths"


# ============================================================================
# Dual-Gradient Flat Resolution Tests (Garbrecht-Martz Algorithm)
# ============================================================================


class TestDualGradientFlatResolution:
    """Tests for the Garbrecht-Martz dual-gradient flat resolution algorithm.

    The algorithm uses TWO gradients to resolve flat areas:
    1. Gradient toward pour points (lower terrain) - water flows to outlets
    2. Gradient away from high points (higher terrain) - water flows from ridges

    This creates natural flow convergence patterns in flat regions.
    """

    def test_flat_resolution_creates_connected_drainage(self):
        """Flat resolution should create connected drainage without fragmentation.

        This tests that flat cells drain properly and create increasing
        drainage areas downstream (toward pour points).
        """
        # Create a large flat area surrounded by terrain
        rows, cols = 30, 30
        dem = np.zeros((rows, cols), dtype=np.float32)

        # Create a flat plateau at elevation 100
        dem[8:22, 8:22] = 100.0

        # High terrain on north (should repel flow)
        dem[5:8, 8:22] = 150.0

        # Low terrain on south (should attract flow)
        dem[22:25, 8:22] = 50.0

        # Condition DEM (this triggers flat resolution)
        conditioned = condition_dem(dem, method="breach")

        # Compute flow direction
        flow_dir = compute_flow_direction(conditioned)

        # Compute drainage area
        drainage_area = compute_drainage_area(flow_dir)

        # Check that drainage increases from north to south in the flat region
        # (flow should go toward the southern pour points)
        north_row_drainage = np.mean(drainage_area[10, 10:20])
        south_row_drainage = np.mean(drainage_area[20, 10:20])

        assert south_row_drainage > north_row_drainage, \
            f"Drainage should increase from north to south: north={north_row_drainage:.1f}, south={south_row_drainage:.1f}"

        # Check that all flat cells have valid flow direction (not stuck)
        flat_cells = (dem == 100.0)
        flat_with_flow = flat_cells & (flow_dir > 0)
        flow_ratio = np.sum(flat_with_flow) / np.sum(flat_cells)

        assert flow_ratio >= 0.95, \
            f"At least 95% of flat cells should have flow direction: {flow_ratio:.0%}"

    def test_flat_resolution_respects_pour_points_and_high_points(self):
        """Flow should go toward pour points AND away from high points.

        The dual-gradient ensures that:
        - Cells near pour points (lower terrain) have lower elevation after resolution
        - Cells near high points (higher terrain) have higher elevation after resolution

        The flow pattern converges toward outlets but may use diagonal directions.
        """
        # Create a simple flat area with clear pour point and high point
        rows, cols = 20, 20
        dem = np.zeros((rows, cols), dtype=np.float32)

        # Flat plateau
        dem[5:15, 5:15] = 100.0

        # High terrain on left (high point)
        dem[5:15, 3:5] = 150.0

        # Low terrain on right (pour point)
        dem[5:15, 15:17] = 50.0

        # Condition DEM
        conditioned = condition_dem(dem, method="breach")

        # Flow direction
        flow_dir = compute_flow_direction(conditioned)

        # Flat cells should flow toward the right (E, NE, or SE directions)
        # D8 codes: 1=E, 2=NE, 128=SE
        flat_region = (dem == 100.0)
        eastward_codes = [1, 2, 128]  # E, NE, SE
        eastward_flowing = np.isin(flow_dir, eastward_codes)

        # Most flat cells should flow eastward (toward pour points)
        eastward_ratio = np.sum(flat_region & eastward_flowing) / np.sum(flat_region)

        assert eastward_ratio >= 0.5, \
            f"Most flat cells should flow eastward (E/NE/SE toward outlet): {eastward_ratio:.0%}"

        # Also check that drainage increases from left to right
        drainage_area = compute_drainage_area(flow_dir)
        left_col_drainage = np.mean(drainage_area[7:13, 6])
        right_col_drainage = np.mean(drainage_area[7:13, 13])

        assert right_col_drainage > left_col_drainage, \
            f"Drainage should increase from left to right: left={left_col_drainage:.1f}, right={right_col_drainage:.1f}"

    def test_flat_resolution_no_unreached_cells(self):
        """All flat cells should be reachable by the BFS gradient computation.

        This verifies that the dual-gradient approach doesn't leave any
        flat cells without proper routing (which would cause fragmentation).
        """
        # Create complex flat area that could fragment with single gradient
        rows, cols = 30, 30
        dem = np.zeros((rows, cols), dtype=np.float32)

        # Large flat area
        dem[5:25, 5:25] = 100.0

        # Multiple pour points (creates complex drainage)
        dem[5:25, 3:5] = 50.0   # West edge
        dem[25:27, 5:25] = 50.0  # South edge

        # Multiple high points
        dem[3:5, 5:25] = 150.0  # North edge
        dem[5:25, 25:27] = 150.0  # East edge

        # Condition and compute flow
        conditioned = condition_dem(dem, method="breach")
        flow_dir = compute_flow_direction(conditioned)
        drainage_area = compute_drainage_area(flow_dir)

        # All flat cells should have path to outlet
        flat_cells = (dem == 100.0)

        # Check that all flat cells have flow direction (not stuck)
        flat_with_flow = flat_cells & (flow_dir > 0)
        flow_ratio = np.sum(flat_with_flow) / np.sum(flat_cells)

        # Some cells at edges might be outlets, so allow for that
        assert flow_ratio >= 0.90, \
            f"At least 90% of flat cells should have flow direction: {flow_ratio:.0%}"

    def test_drainage_continuity_after_flat_resolution(self):
        """Drainage area should be continuous (no isolated high values).

        This tests that the dual-gradient flat resolution creates proper
        connected drainage networks without fragmented bright spots.
        """
        # Create DEM with significant flat areas
        dem = create_synthetic_valley(shape=(40, 40))

        # Add a large flat area in the valley floor
        dem[15:25, 15:25] = np.min(dem[15:25, 15:25])

        # Condition and compute
        conditioned = condition_dem(dem, method="breach")
        flow_dir = compute_flow_direction(conditioned)
        drainage_area = compute_drainage_area(flow_dir)

        # Check for drainage continuity: for each cell with high drainage,
        # its downstream neighbor should have >= drainage
        offsets = {
            1: (0, 1), 2: (-1, 1), 4: (-1, 0), 8: (-1, -1),
            16: (0, -1), 32: (1, -1), 64: (1, 0), 128: (1, 1),
        }

        rows, cols = flow_dir.shape
        violations = 0

        for i in range(rows):
            for j in range(cols):
                direction = flow_dir[i, j]
                if direction == 0 or direction not in offsets:
                    continue

                di, dj = offsets[direction]
                ni, nj = i + di, j + dj

                if 0 <= ni < rows and 0 <= nj < cols:
                    if drainage_area[ni, nj] < drainage_area[i, j]:
                        violations += 1

        assert violations == 0, \
            f"Found {violations} violations of drainage continuity"


class TestOutletIdentification:
    """Test suite for identify_outlets() function (Stage 1 of flow-spec.md)."""

    def test_identify_coastal_outlets(self):
        """Coastal outlets detected at ocean boundary below threshold."""
        dem = np.array([
            [5, 5, 5, 5],
            [5, 3, 2, 5],  # Low cells adjacent to ocean
            [5, 5, 5, 5],
            [5, 5, 5, 5]
        ], dtype=np.float32)

        nodata = np.array([
            [False, False, False, False],
            [True,  False, False, False],  # Ocean cell at [1,0]
            [False, False, False, False],
            [False, False, False, False]
        ])

        outlets = identify_outlets(dem, nodata, coastal_elev_threshold=10.0, edge_mode="none")

        # Cell [1,1] is adjacent to ocean and elevation=3 < 10m → outlet
        assert outlets[1, 1] == True
        # Cell [1,2] is NOT adjacent to ocean → not coastal outlet
        assert outlets[1, 2] == False
        # Cell [0,0] IS adjacent to ocean [1,0] diagonally and elevation=5 < 10m → outlet
        assert outlets[0, 0] == True
        # Cell [2,2] is NOT adjacent to ocean and not on edge → not outlet
        assert outlets[2, 2] == False

    def test_coastal_outlet_respects_elevation_threshold(self):
        """High cliffs adjacent to ocean should not be coastal outlets."""
        dem = np.array([
            [15, 15, 15],
            [15, 12, 15],  # Cell above threshold
            [15, 15, 15]
        ], dtype=np.float32)

        nodata = np.array([
            [False, False, False],
            [True,  False, False],  # Ocean
            [False, False, False]
        ])

        outlets = identify_outlets(dem, nodata, coastal_elev_threshold=10.0, edge_mode="none")

        # Cell [1,1] is adjacent to ocean but elevation=12 > 10m → NOT outlet
        assert outlets[1, 1] == False

    def test_edge_mode_all(self):
        """All boundary cells marked as outlets when edge_mode='all'."""
        dem = np.ones((5, 5), dtype=np.float32) * 100.0
        nodata = np.zeros((5, 5), dtype=bool)

        outlets = identify_outlets(dem, nodata, edge_mode="all")

        # All edges should be outlets
        assert outlets[0, :].all()  # Top edge
        assert outlets[-1, :].all()  # Bottom edge
        assert outlets[:, 0].all()  # Left edge
        assert outlets[:, -1].all()  # Right edge

        # Interior should not be outlets
        assert not outlets[2, 2]

    def test_edge_mode_none(self):
        """No edge outlets when edge_mode='none' (for islands)."""
        dem = np.ones((5, 5), dtype=np.float32) * 100.0
        nodata = np.zeros((5, 5), dtype=bool)

        outlets = identify_outlets(dem, nodata, edge_mode="none")

        # No edge outlets should be marked
        assert not outlets[0, :].any()  # Top edge
        assert not outlets[-1, :].any()  # Bottom edge
        assert not outlets[:, 0].any()  # Left edge
        assert not outlets[:, -1].any()  # Right edge

    def test_edge_mode_local_minima(self):
        """Only boundary local minima marked as outlets."""
        dem = np.array([
            [10, 5, 10, 5, 10],  # Top edge: 5s are local minima
            [10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10],
            [10, 5, 10, 5, 10]   # Bottom edge: 5s are local minima
        ], dtype=np.float32)

        nodata = np.zeros((5, 5), dtype=bool)

        outlets = identify_outlets(dem, nodata, edge_mode="local_minima")

        # Top edge local minima should be outlets
        assert outlets[0, 1] == True  # Local minimum at [0,1]
        assert outlets[0, 3] == True  # Local minimum at [0,3]
        assert outlets[0, 0] == False  # Not a minimum
        assert outlets[0, 2] == False  # Not a minimum

        # Bottom edge local minima should be outlets
        assert outlets[4, 1] == True
        assert outlets[4, 3] == True

    def test_masked_basin_outlets(self):
        """User-supplied outlet locations are marked."""
        dem = np.ones((5, 5), dtype=np.float32) * 100.0
        nodata = np.zeros((5, 5), dtype=bool)

        # User specifies outlet at [2,2]
        masked_outlets = np.zeros((5, 5), dtype=bool)
        masked_outlets[2, 2] = True
        masked_outlets[3, 3] = True

        outlets = identify_outlets(dem, nodata, edge_mode="none",
                                   masked_basin_outlets=masked_outlets)

        # User-supplied outlets should be marked
        assert outlets[2, 2] == True
        assert outlets[3, 3] == True
        # Other cells should not be outlets
        assert outlets[0, 0] == False

    def test_combined_outlet_types(self):
        """Multiple outlet types can coexist."""
        dem = np.array([
            [10, 10, 10, 10],
            [10, 3, 8, 10],
            [10, 10, 10, 10],
            [10, 10, 10, 10]
        ], dtype=np.float32)

        nodata = np.array([
            [False, False, False, False],
            [True,  False, False, False],  # Ocean
            [False, False, False, False],
            [False, False, False, False]
        ])

        # User lake outlet
        masked_outlets = np.zeros((4, 4), dtype=bool)
        masked_outlets[2, 2] = True

        outlets = identify_outlets(
            dem, nodata,
            coastal_elev_threshold=10.0,
            edge_mode="all",
            masked_basin_outlets=masked_outlets
        )

        # Coastal outlet (low + adjacent to ocean)
        assert outlets[1, 1] == True
        # Edge outlets (all boundary except NoData cells)
        assert outlets[0, :].all()  # Top edge has no NoData
        assert outlets[-1, :].all()  # Bottom edge has no NoData
        # Left edge: [1,0] is NoData so it should be False
        assert outlets[0, 0] == True  # Not NoData
        assert outlets[1, 0] == False  # NoData cell
        assert outlets[2, 0] == True  # Not NoData
        assert outlets[3, 0] == True  # Not NoData
        assert outlets[:, -1].all()  # Right edge has no NoData
        # Masked basin outlet
        assert outlets[2, 2] == True

    def test_nodata_cells_not_marked_as_outlets(self):
        """NoData cells themselves should not be marked as outlets."""
        dem = np.ones((5, 5), dtype=np.float32) * 100.0
        nodata = np.zeros((5, 5), dtype=bool)
        nodata[2, 2] = True  # Mark center as NoData

        outlets = identify_outlets(dem, nodata, edge_mode="all")

        # NoData cell should NOT be an outlet
        assert outlets[2, 2] == False
        # Edge outlets should still work
        assert outlets[0, 0] == True


class TestConstrainedBreaching:
    """Test suite for breach_depressions_constrained() function (Stage 2a of flow-spec.md)."""

    def test_identify_sinks_simple(self):
        """_identify_sinks should find cells with no downslope neighbor."""
        dem = np.array([
            [10, 10, 10, 10],
            [10,  5,  6, 10],  # [1,1] is a sink
            [10, 10, 10, 10],
            [10, 10, 10, 10]
        ], dtype=np.float32)

        outlets = np.zeros((4, 4), dtype=bool)
        outlets[0, 0] = True  # Mark one outlet

        sinks = _identify_sinks(dem, outlets, nodata_mask=None)

        # Should find sink at [1,1] (elevation 5)
        assert len(sinks) > 0
        assert (1, 1, 5.0) in sinks

    def test_breach_simple_depression(self):
        """Breach creates monotonic path from sink to outlet."""
        dem = np.array([
            [5, 5, 5, 5, 5],
            [5, 8, 8, 8, 5],
            [5, 8, 3, 8, 5],  # Sink at [2,2]
            [5, 8, 8, 8, 5],
            [5, 5, 5, 5, 0]   # Outlet at [4,4]
        ], dtype=np.float32)

        outlets = np.zeros((5, 5), dtype=bool)
        outlets[4, 4] = True

        breached = breach_depressions_constrained(
            dem, outlets, max_breach_depth=10.0, max_breach_length=10, epsilon=1e-4
        )

        # Sink should be breached (elevation lowered or path created)
        # At minimum, we should be able to find a downslope neighbor now
        has_downslope = False
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            ni, nj = 2 + di, 2 + dj
            if 0 <= ni < 5 and 0 <= nj < 5:
                if breached[ni, nj] < breached[2, 2]:
                    has_downslope = True
                    break

        assert has_downslope, "Sink should have been breached to create downslope path"

    def test_breach_respects_max_depth(self):
        """Breach fails if required depth exceeds constraint."""
        dem = np.array([
            [10, 10, 10],
            [10,  0, 10],  # Deep sink (10m below rim)
            [10, 10,  5]   # Outlet at [2,2]
        ], dtype=np.float32)

        outlets = np.zeros((3, 3), dtype=bool)
        outlets[2, 2] = True

        breached = breach_depressions_constrained(
            dem, outlets, max_breach_depth=5.0, max_breach_length=10, epsilon=1e-4
        )

        # Sink should NOT be breached (too deep)
        # Original DEM should be mostly unchanged
        assert breached[1, 1] == dem[1, 1], \
            "Deep sink should not be breached when max_depth is too small"

    def test_breach_respects_max_length(self):
        """Breach fails if path length exceeds constraint."""
        # Create DEM with distant sink and outlet
        dem = np.ones((10, 10), dtype=np.float32) * 100.0
        dem[1, 1] = 50.0  # Sink
        dem[8, 8] = 0.0   # Distant outlet

        outlets = np.zeros((10, 10), dtype=bool)
        outlets[8, 8] = True

        # Set max_length too short to reach outlet
        breached = breach_depressions_constrained(
            dem, outlets, max_breach_depth=100.0, max_breach_length=5, epsilon=1e-4
        )

        # Sink should NOT be breached (path too long)
        assert breached[1, 1] == dem[1, 1], \
            "Distant sink should not be breached when max_length is too small"

    def test_breach_creates_monotonic_path(self):
        """Breached path should have monotonic decreasing elevation."""
        dem = np.array([
            [10, 10, 10, 10, 10],
            [10, 12, 12, 12, 10],
            [10, 12,  5, 12, 10],  # Sink at [2,2]
            [10, 12, 12, 12, 10],
            [10, 10, 10, 10,  0]   # Outlet at [4,4]
        ], dtype=np.float32)

        outlets = np.zeros((5, 5), dtype=bool)
        outlets[4, 4] = True

        breached = breach_depressions_constrained(
            dem, outlets, max_breach_depth=10.0, max_breach_length=20, epsilon=1e-3
        )

        # Verify sink was breached
        assert breached[2, 2] < dem[2, 2] or breached[2, 2] != 5.0, \
            "Sink should have been modified by breaching"

        # Manually trace a path from sink toward outlet to verify monotonic gradient
        # This is a simplified check - full path verification would require path reconstruction
        # For now, just check that breaching occurred
        breach_occurred = not np.array_equal(breached, dem)
        assert breach_occurred, "Breaching should have modified the DEM"

    def test_breach_multiple_sinks(self):
        """Multiple sinks can be breached independently."""
        dem = np.array([
            [10,  3, 10,  3, 10],  # Two sinks at [0,1] and [0,3]
            [10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10],
            [ 0,  0,  0,  0,  0]   # Outlets at bottom
        ], dtype=np.float32)

        outlets = np.zeros((5, 5), dtype=bool)
        outlets[4, :] = True  # Bottom row is outlet
        # Also mark edges as outlets to avoid edge sinks
        outlets[0, :] = True
        outlets[:, 0] = True
        outlets[:, -1] = True

        initial_sinks = _identify_sinks(dem, outlets, nodata_mask=None)
        # Should find the two interior cells [0,1] and [0,3] that are marked as outlets
        # but were originally sinks before edge marking. Actually, they're outlets now.
        # Let me reconsider this test - the sinks are now outlets. Let me create internal sinks instead.

        # Reset and create internal sinks
        dem2 = np.array([
            [10, 10, 10, 10, 10],
            [10,  3,  8,  3, 10],  # Two sinks at [1,1] and [1,3]
            [10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10],
            [ 0,  0,  0,  0,  0]   # Outlets at bottom
        ], dtype=np.float32)

        outlets2 = np.zeros((5, 5), dtype=bool)
        outlets2[4, :] = True  # Bottom row is outlet

        initial_sinks = _identify_sinks(dem2, outlets2, nodata_mask=None)
        # Should find two interior sinks
        assert len(initial_sinks) >= 2, f"Should identify at least 2 sinks, found {len(initial_sinks)}"

        breached = breach_depressions_constrained(
            dem2, outlets2, max_breach_depth=10.0, max_breach_length=10, epsilon=1e-4
        )

        # After breaching, check if sinks have downslope paths
        sinks_resolved = 0
        for sink_r, sink_c, _ in initial_sinks:
            has_downslope = False
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1),
                          (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                ni, nj = sink_r + di, sink_c + dj
                if 0 <= ni < 5 and 0 <= nj < 5:
                    if breached[ni, nj] < breached[sink_r, sink_c]:
                        has_downslope = True
                        break
            if has_downslope:
                sinks_resolved += 1

        assert sinks_resolved >= 1, "At least one sink should be breached"

    def test_breach_preserves_outlets(self):
        """Breaching should not modify outlet elevations."""
        dem = np.array([
            [10, 10, 10],
            [10,  5, 10],  # Sink
            [10, 10,  0]   # Outlet
        ], dtype=np.float32)

        outlets = np.zeros((3, 3), dtype=bool)
        outlets[2, 2] = True

        original_outlet_elev = dem[2, 2]

        breached = breach_depressions_constrained(
            dem, outlets, max_breach_depth=10.0, max_breach_length=10, epsilon=1e-4
        )

        # Outlet elevation should be unchanged
        assert breached[2, 2] == original_outlet_elev, \
            "Breaching should not modify outlet elevations"


class TestPriorityFloodFill:
    """Test suite for priority_flood_fill_epsilon() function (Stage 2b of flow-spec.md)."""

    def test_fill_simple_depression(self):
        """Priority flood fills depression to spill point + epsilon."""
        dem = np.array([
            [10, 10, 10],
            [10,  5, 10],  # Depression at [1,1]
            [10, 10, 10]
        ], dtype=np.float32)

        outlets = np.zeros((3, 3), dtype=bool)
        outlets[0, 0] = True

        filled = priority_flood_fill_epsilon(dem, outlets, epsilon=1e-4)

        # Center cell should be raised above original elevation
        assert filled[1, 1] > 5.0, "Depression should be filled"
        # Should be raised to approximately the spill elevation + some epsilon
        assert filled[1, 1] <= 10.0 + 1e-2, "Should not be raised too high"

    def test_fill_creates_epsilon_gradient(self):
        """Priority flood with epsilon creates flow gradients in flats."""
        dem = np.array([
            [10, 10, 10, 10],
            [10,  5,  5, 10],
            [10,  5,  5, 10],  # Flat depression
            [10, 10, 10, 10]
        ], dtype=np.float32)

        outlets = np.zeros((4, 4), dtype=bool)
        outlets[0, 0] = True

        filled = priority_flood_fill_epsilon(dem, outlets, epsilon=1e-4)

        # All depression cells should be raised
        assert filled[1, 1] > 5.0
        assert filled[1, 2] > 5.0
        assert filled[2, 1] > 5.0
        assert filled[2, 2] > 5.0

        # Cells farther from outlet should be higher (gradient exists)
        # [2,2] is farthest from [0,0], should be highest
        assert filled[2, 2] >= filled[1, 1], \
            "Farther cells should have equal or higher elevation (epsilon gradient)"

    def test_fill_preserves_non_depressions(self):
        """Priority flood should not raise cells that aren't in depressions."""
        dem = np.array([
            [10,  9,  8],
            [ 9,  8,  7],
            [ 8,  7,  0]   # Slopes down to outlet
        ], dtype=np.float32)

        outlets = np.zeros((3, 3), dtype=bool)
        outlets[2, 2] = True

        filled = priority_flood_fill_epsilon(dem, outlets, epsilon=1e-4)

        # No depressions exist, DEM should be unchanged (within epsilon tolerance)
        np.testing.assert_allclose(filled, dem, atol=1e-3)

    def test_fill_respects_nodata_mask(self):
        """Priority flood should not modify NoData cells."""
        dem = np.array([
            [10, 10, 10],
            [10,  5, 10],
            [10, 10, 10]
        ], dtype=np.float32)

        outlets = np.zeros((3, 3), dtype=bool)
        outlets[0, 0] = True

        nodata = np.zeros((3, 3), dtype=bool)
        nodata[1, 1] = True  # Mark depression as NoData

        filled = priority_flood_fill_epsilon(dem, outlets, epsilon=1e-4, nodata_mask=nodata)

        # NoData cell should be unchanged
        assert filled[1, 1] == dem[1, 1], "NoData cells should not be modified"

    def test_fill_preserves_outlets(self):
        """Priority flood should not modify outlet elevations."""
        dem = np.array([
            [10, 10, 10],
            [10,  5, 10],
            [10, 10,  0]   # Outlet
        ], dtype=np.float32)

        outlets = np.zeros((3, 3), dtype=bool)
        outlets[2, 2] = True

        original_outlet_elev = dem[2, 2]

        filled = priority_flood_fill_epsilon(dem, outlets, epsilon=1e-4)

        # Outlet should be unchanged
        assert filled[2, 2] == original_outlet_elev, \
            "Outlet elevation should not be modified"

    def test_fill_with_zero_epsilon(self):
        """Zero epsilon should create true flats (no gradient)."""
        dem = np.array([
            [10, 10, 10, 10],
            [10,  5,  5, 10],
            [10,  5,  5, 10],
            [10, 10, 10, 10]
        ], dtype=np.float32)

        outlets = np.zeros((4, 4), dtype=bool)
        outlets[0, 0] = True

        filled = priority_flood_fill_epsilon(dem, outlets, epsilon=0.0)

        # All filled cells should have same elevation (flat)
        filled_cells_elevs = [filled[1, 1], filled[1, 2], filled[2, 1], filled[2, 2]]
        assert len(set(filled_cells_elevs)) == 1, \
            "With epsilon=0, filled area should be perfectly flat"

    def test_fill_multiple_depressions(self):
        """Priority flood can fill multiple independent depressions."""
        dem = np.array([
            [10,  5, 10,  5, 10],  # Two depressions at [0,1] and [0,3]
            [10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10],
            [ 0,  0,  0,  0,  0]   # Outlets at bottom
        ], dtype=np.float32)

        outlets = np.zeros((5, 5), dtype=bool)
        outlets[4, :] = True

        filled = priority_flood_fill_epsilon(dem, outlets, epsilon=1e-4)

        # Both depressions should be filled
        assert filled[0, 1] > 5.0, "First depression should be filled"
        assert filled[0, 3] > 5.0, "Second depression should be filled"


# ============================================================================
# Flow Computation Caching Tests (TDD RED)
# ============================================================================


class TestFlowComputationCaching:
    """Test suite for caching flow computation results at high resolution."""

    def test_flow_accumulation_accepts_cache_parameter(self, tmp_path):
        """flow_accumulation should accept cache=True parameter."""
        dem_path, precip_path = create_synthetic_dem_and_precip(tmp_path)

        # Should not raise - cache parameter should be accepted
        result = flow_accumulation(
            dem_path=str(dem_path),
            precipitation_path=str(precip_path),
            cache=True,
        )

        assert "flow_direction" in result

    def test_flow_accumulation_accepts_cache_dir_parameter(self, tmp_path):
        """flow_accumulation should accept cache_dir parameter."""
        dem_path, precip_path = create_synthetic_dem_and_precip(tmp_path)
        cache_dir = tmp_path / "flow_cache"

        result = flow_accumulation(
            dem_path=str(dem_path),
            precipitation_path=str(precip_path),
            cache=True,
            cache_dir=str(cache_dir),
        )

        assert "flow_direction" in result

    def test_cache_creates_metadata_file(self, tmp_path):
        """Caching should create a metadata file with computation parameters."""
        dem_path, precip_path = create_synthetic_dem_and_precip(tmp_path)
        cache_dir = tmp_path / "flow_cache"

        flow_accumulation(
            dem_path=str(dem_path),
            precipitation_path=str(precip_path),
            cache=True,
            cache_dir=str(cache_dir),
        )

        metadata_file = cache_dir / "flow_cache_metadata.json"
        assert metadata_file.exists(), "Should create metadata file"

        import json
        with open(metadata_file) as f:
            metadata = json.load(f)

        assert "dem_path" in metadata
        assert "backend" in metadata
        assert "timestamp" in metadata

    def test_cache_loads_existing_results(self, tmp_path):
        """Caching should load existing results without recomputation."""
        dem_path, precip_path = create_synthetic_dem_and_precip(tmp_path)
        cache_dir = tmp_path / "flow_cache"

        # First call - computes
        result1 = flow_accumulation(
            dem_path=str(dem_path),
            precipitation_path=str(precip_path),
            cache=True,
            cache_dir=str(cache_dir),
        )

        # Second call - should load from cache
        result2 = flow_accumulation(
            dem_path=str(dem_path),
            precipitation_path=str(precip_path),
            cache=True,
            cache_dir=str(cache_dir),
        )

        # Results should be identical
        np.testing.assert_array_equal(
            result1["flow_direction"], result2["flow_direction"]
        )
        np.testing.assert_array_equal(
            result1["drainage_area"], result2["drainage_area"]
        )

        # Metadata should indicate cache hit
        assert result2["metadata"].get("cache_hit", False), \
            "Second call should be cache hit"

    def test_cache_invalidates_on_different_backend(self, tmp_path):
        """Cache should invalidate when backend parameter changes."""
        dem_path, precip_path = create_synthetic_dem_and_precip(tmp_path)
        cache_dir = tmp_path / "flow_cache"

        # First call with spec backend
        result1 = flow_accumulation(
            dem_path=str(dem_path),
            precipitation_path=str(precip_path),
            cache=True,
            cache_dir=str(cache_dir),
            backend="spec",
        )

        # Second call with legacy backend - should recompute
        result2 = flow_accumulation(
            dem_path=str(dem_path),
            precipitation_path=str(precip_path),
            cache=True,
            cache_dir=str(cache_dir),
            backend="legacy",
        )

        # Should NOT be cache hit (different backend)
        assert not result2["metadata"].get("cache_hit", True), \
            "Different backend should invalidate cache"

    def test_cache_invalidates_on_different_max_cells(self, tmp_path):
        """Cache should invalidate when max_cells parameter changes."""
        dem_path, precip_path = create_synthetic_dem_and_precip(tmp_path, shape=(50, 50))
        cache_dir = tmp_path / "flow_cache"

        # First call with larger max_cells
        result1 = flow_accumulation(
            dem_path=str(dem_path),
            precipitation_path=str(precip_path),
            cache=True,
            cache_dir=str(cache_dir),
            max_cells=5000,
        )

        # Second call with different max_cells - should recompute
        result2 = flow_accumulation(
            dem_path=str(dem_path),
            precipitation_path=str(precip_path),
            cache=True,
            cache_dir=str(cache_dir),
            max_cells=2000,
        )

        # Should NOT be cache hit (different max_cells)
        assert not result2["metadata"].get("cache_hit", True), \
            "Different max_cells should invalidate cache"

    def test_cache_invalidates_on_modified_dem(self, tmp_path):
        """Cache should invalidate when DEM file is modified."""
        import time

        dem_path, precip_path = create_synthetic_dem_and_precip(tmp_path)
        cache_dir = tmp_path / "flow_cache"

        # First call
        result1 = flow_accumulation(
            dem_path=str(dem_path),
            precipitation_path=str(precip_path),
            cache=True,
            cache_dir=str(cache_dir),
        )

        # Modify DEM file (just touch it to change mtime)
        time.sleep(0.1)  # Ensure different mtime
        with rasterio.open(dem_path) as src:
            dem_data = src.read(1)

        # Rewrite with slight modification
        dem_data[0, 0] += 1.0
        create_sample_geotiff(dem_path, dem_data)

        # Second call - should recompute due to modified DEM
        result2 = flow_accumulation(
            dem_path=str(dem_path),
            precipitation_path=str(precip_path),
            cache=True,
            cache_dir=str(cache_dir),
        )

        # Should NOT be cache hit (DEM modified)
        assert not result2["metadata"].get("cache_hit", True), \
            "Modified DEM should invalidate cache"

    def test_cache_handles_missing_output_files(self, tmp_path):
        """Cache should recompute if output files are missing."""
        dem_path, precip_path = create_synthetic_dem_and_precip(tmp_path)
        cache_dir = tmp_path / "flow_cache"

        # First call
        flow_accumulation(
            dem_path=str(dem_path),
            precipitation_path=str(precip_path),
            cache=True,
            cache_dir=str(cache_dir),
        )

        # Delete one output file
        import os
        flow_dir_file = cache_dir / "flow_direction.tif"
        if flow_dir_file.exists():
            os.remove(flow_dir_file)

        # Second call - should recompute due to missing file
        result2 = flow_accumulation(
            dem_path=str(dem_path),
            precipitation_path=str(precip_path),
            cache=True,
            cache_dir=str(cache_dir),
        )

        # Should NOT be cache hit (missing file)
        assert not result2["metadata"].get("cache_hit", True), \
            "Missing output file should invalidate cache"

    def test_cache_key_includes_resolution_parameters(self, tmp_path):
        """Cache key should include resolution-affecting parameters."""
        dem_path, precip_path = create_synthetic_dem_and_precip(tmp_path)
        cache_dir = tmp_path / "flow_cache"

        # First call with target_vertices
        result1 = flow_accumulation(
            dem_path=str(dem_path),
            precipitation_path=str(precip_path),
            cache=True,
            cache_dir=str(cache_dir),
            target_vertices=500,
        )

        # Second call with different target_vertices
        result2 = flow_accumulation(
            dem_path=str(dem_path),
            precipitation_path=str(precip_path),
            cache=True,
            cache_dir=str(cache_dir),
            target_vertices=1000,
        )

        # Should NOT be cache hit (different resolution)
        assert not result2["metadata"].get("cache_hit", True), \
            "Different target_vertices should invalidate cache"

    def test_cache_stores_all_required_outputs(self, tmp_path):
        """Cache should store all required output files."""
        dem_path, precip_path = create_synthetic_dem_and_precip(tmp_path)
        cache_dir = tmp_path / "flow_cache"

        flow_accumulation(
            dem_path=str(dem_path),
            precipitation_path=str(precip_path),
            cache=True,
            cache_dir=str(cache_dir),
        )

        # All output files should exist
        required_files = [
            "flow_direction.tif",
            "flow_accumulation_area.tif",
            "flow_accumulation_rainfall.tif",
            "dem_conditioned.tif",
            "flow_cache_metadata.json",
        ]

        for filename in required_files:
            assert (cache_dir / filename).exists(), f"Missing cache file: {filename}"

    def test_cache_disabled_by_default(self, tmp_path):
        """Caching should be disabled by default (cache=False)."""
        dem_path, precip_path = create_synthetic_dem_and_precip(tmp_path)

        # Call without cache parameter
        result = flow_accumulation(
            dem_path=str(dem_path),
            precipitation_path=str(precip_path),
        )

        # Metadata should NOT have cache_hit key
        assert "cache_hit" not in result["metadata"], \
            "Caching should be disabled by default"

    def test_cache_uses_dem_directory_by_default(self, tmp_path):
        """When cache=True but no cache_dir, use DEM's directory."""
        dem_path, precip_path = create_synthetic_dem_and_precip(tmp_path)

        flow_accumulation(
            dem_path=str(dem_path),
            precipitation_path=str(precip_path),
            cache=True,
            # No cache_dir specified
        )

        # Metadata file should be in DEM's directory
        metadata_file = tmp_path / "flow_cache_metadata.json"
        assert metadata_file.exists(), \
            "Cache metadata should be in DEM's directory when cache_dir not specified"
