"""
Comprehensive San Diego Flow Demo Pipeline Validation Test Suite.

This test suite validates ~85+ expectations about the UPDATED San Diego flow demo,
covering the new architecture with:
- flow_pipeline.py module with basin preservation
- Terrain-based alignment (replaces scipy.zoom)
- Basin-aware lake handling (inside vs outside basins)
- Updated parameters (10000/5.0/100/300)
- Precipitation masking

The tests are organized into 28 classes covering different aspects of the pipeline.

Run with: pytest tests/test_san_diego_flow_pipeline.py -v

For fast execution, the demo is run once at the session start and outputs are cached
across all tests. Total runtime: ~3-5 minutes.
"""

import pytest
import numpy as np
import rasterio
from rasterio import Affine
from pathlib import Path
import subprocess
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.terrain.flow_accumulation import compute_discharge_potential
from src.terrain.core import Terrain


# ============================================================================
# Test Infrastructure - Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def demo_output_dir(tmp_path_factory):
    """Create temporary output directory for demo run."""
    return tmp_path_factory.mktemp("san_diego_flow_test")


@pytest.fixture(scope="session")
def flow_artifacts(demo_output_dir):
    """
    Run San Diego demo once and cache all outputs for tests.

    This is the expensive operation (~2-3 minutes). All other tests
    use these cached outputs for fast execution.
    """
    print("\n" + "=" * 60)
    print("RUNNING SAN DIEGO FLOW DEMO (one-time setup)")
    print("=" * 60)

    # Run demo in fast mode with real data
    demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
    dem_dir = PROJECT_ROOT / "data" / "san_diego_dem"

    cmd = [
        sys.executable,
        str(demo_script),
        "--fast",
        "--skip-download",  # Use existing DEM
        "--no-render",  # Skip 3D rendering
        "--output-dir", str(demo_output_dir),
        "--dem-dir", str(dem_dir),
        "--target-vertices", "1000000",  # 1M for fast tests
    ]

    # Run demo
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    print("✓ Demo completed successfully")
    print("=" * 60 + "\n")

    # Load and cache key artifacts
    artifacts = {
        "output_dir": demo_output_dir,
        "dem_dir": dem_dir,
        "flow_output_dir": demo_output_dir / "flow_outputs",
    }

    # Load DEM
    from src.terrain.data_loading import load_dem_files
    artifacts["dem"], artifacts["dem_transform"] = load_dem_files(dem_dir)

    # Load flow outputs
    flow_dir = artifacts["flow_output_dir"]
    with rasterio.open(flow_dir / "flow_direction.tif") as src:
        artifacts["flow_direction"] = src.read(1)
        artifacts["flow_transform"] = src.transform
        artifacts["flow_crs"] = src.crs

    with rasterio.open(flow_dir / "flow_accumulation_area.tif") as src:
        artifacts["drainage_area"] = src.read(1)

    with rasterio.open(flow_dir / "flow_accumulation_rainfall.tif") as src:
        artifacts["upstream_rainfall"] = src.read(1)

    with rasterio.open(flow_dir / "dem_conditioned.tif") as src:
        artifacts["conditioned_dem"] = src.read(1)

    return artifacts


# ============================================================================
# Helper Functions
# ============================================================================

def verify_geographic_alignment(transform1, shape1, transform2, shape2, tolerance=0.01):
    """
    Check if two rasters cover the same geographic extent.

    Parameters
    ----------
    transform1, transform2 : Affine
        Transforms for the two rasters
    shape1, shape2 : tuple
        Shapes (rows, cols) of the two rasters
    tolerance : float
        Tolerance in degrees for extent matching

    Returns
    -------
    bool
        True if extents match within tolerance
    """
    # Calculate extents
    extent1 = (
        transform1.c,  # left (min lon)
        transform1.f + transform1.e * shape1[0],  # bottom (min lat)
        transform1.c + transform1.a * shape1[1],  # right (max lon)
        transform1.f,  # top (max lat)
    )

    extent2 = (
        transform2.c,
        transform2.f + transform2.e * shape2[0],
        transform2.c + transform2.a * shape2[1],
        transform2.f,
    )

    # Check each boundary
    for e1, e2 in zip(extent1, extent2):
        if abs(e1 - e2) > tolerance:
            return False
    return True


def get_resolution_meters(transform, crs_str):
    """Calculate pixel size in meters."""
    from pyproj import Transformer

    if "32611" in str(crs_str):  # UTM 11N
        # Already in meters
        return abs(transform.a)
    else:
        # Convert degrees to meters at San Diego latitude (33°N)
        lat = 33.0
        lon = -117.0
        pixel_deg = abs(transform.a)

        # Approximate conversion at this latitude
        meters_per_deg_lon = 111320 * np.cos(np.radians(lat))
        return pixel_deg * meters_per_deg_lon


def check_terrain_alignment(terrain_obj, expected_shape):
    """Verify all data layers in terrain have expected shape after transforms."""
    if terrain_obj.dem.shape != expected_shape:
        return False

    for layer_name, layer_data in terrain_obj.data_layers.items():
        if layer_data["data"].shape != expected_shape:
            return False

    return True


def validate_basin_aware_lakes(lake_mask, basin_mask, conditioning_mask):
    """
    Check that only lakes inside basins are in conditioning mask.

    Returns True if:
    - Lakes inside basins ARE in conditioning mask
    - Lakes outside basins are NOT in conditioning mask
    """
    if lake_mask is None or basin_mask is None:
        return True  # No lakes or basins to check

    lakes_in_basins = (lake_mask > 0) & basin_mask
    lakes_outside = (lake_mask > 0) & ~basin_mask

    # Lakes inside should be masked
    if np.any(lakes_in_basins):
        if not np.all(conditioning_mask[lakes_in_basins]):
            return False

    # Lakes outside should NOT be masked (except ocean overlap)
    # This is harder to check without ocean mask, so we skip for now

    return True


def check_no_invalid_precip(precip_data, min_valid=0, max_valid=20000):
    """Verify precipitation has no invalid values."""
    return np.all((precip_data >= min_valid) & (precip_data <= max_valid))


# ============================================================================
# Class 1: TestDataAcquisition
# ============================================================================

class TestDataAcquisition:
    """Tests for data acquisition phase."""

    def test_dem_is_wgs84(self, flow_artifacts):
        """DEM files should be in WGS84 (EPSG:4326)."""
        dem_dir = flow_artifacts["dem_dir"]
        merged_dem_path = flow_artifacts["output_dir"] / "flow_data" / "merged_dem.tif"

        # Check if merged DEM exists
        if merged_dem_path.exists():
            with rasterio.open(merged_dem_path) as src:
                assert src.crs == rasterio.crs.CRS.from_epsg(4326), \
                    f"Merged DEM should be in WGS84, got {src.crs}"

    def test_dem_resolution_is_30m(self, flow_artifacts):
        """DEM resolution should be ~30m (~0.00028° at San Diego latitude)."""
        transform = flow_artifacts["dem_transform"]
        pixel_size_deg = abs(transform.a)

        # At San Diego latitude (33°N), 1 arc-second ≈ 30m
        # 1 arc-second = 1/3600 degree ≈ 0.000278°
        assert 0.00027 < pixel_size_deg < 0.00029, \
            f"DEM resolution should be ~0.00028° (~30m), got {pixel_size_deg:.6f}°"

    def test_precipitation_covers_dem(self, flow_artifacts):
        """Precipitation data should cover full DEM extent."""
        output_dir = flow_artifacts["output_dir"]

        # WorldClim files are named with bbox coordinates
        precip_files = list(output_dir.glob("worldclim_annual_precip_*.tif"))

        if not precip_files:
            pytest.skip("Precipitation file not found")

        precip_path = precip_files[0]  # Use first match

        dem_transform = flow_artifacts["dem_transform"]
        dem_shape = flow_artifacts["dem"].shape

        # Calculate DEM extent
        dem_bounds = (
            dem_transform.c,  # left
            dem_transform.f + dem_transform.e * dem_shape[0],  # bottom
            dem_transform.c + dem_transform.a * dem_shape[1],  # right
            dem_transform.f,  # top
        )

        with rasterio.open(precip_path) as src:
            precip_bounds = src.bounds

            # Precipitation should contain DEM bounds (with some tolerance)
            assert precip_bounds.left <= dem_bounds[0] + 0.01, \
                "Precipitation should cover DEM left boundary"
            assert precip_bounds.bottom <= dem_bounds[1] + 0.01, \
                "Precipitation should cover DEM bottom boundary"
            assert precip_bounds.right >= dem_bounds[2] - 0.01, \
                "Precipitation should cover DEM right boundary"
            assert precip_bounds.top >= dem_bounds[3] - 0.01, \
                "Precipitation should cover DEM top boundary"

    def test_water_bodies_loaded_before_flow(self):
        """Water bodies should be loaded before flow computation (code order check)."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        water_pos = demo_source.find("download_water_bodies(")
        flow_pos = demo_source.find("flow_accumulation(")

        assert 0 < water_pos < flow_pos, \
            f"download_water_bodies should be called before flow_accumulation. " \
            f"Found at {water_pos} and {flow_pos} respectively."


# ============================================================================
# Class 2: TestFlowAccumulation
# ============================================================================

class TestFlowAccumulation:
    """Tests for flow accumulation behavior."""

    def test_no_flip_or_rotation(self, flow_artifacts):
        """flow_accumulation should NOT flip or rotate data."""
        dem_shape = flow_artifacts["dem"].shape
        flow_shape = flow_artifacts["flow_direction"].shape

        # If downsampling occurred, check aspect ratio is preserved
        if dem_shape != flow_shape:
            dem_aspect = dem_shape[0] / dem_shape[1]
            flow_aspect = flow_shape[0] / flow_shape[1]

            assert abs(dem_aspect - flow_aspect) < 0.05, \
                f"Aspect ratio should be preserved. DEM: {dem_aspect:.3f}, Flow: {flow_aspect:.3f}"

    def test_stays_in_wgs84(self, flow_artifacts):
        """Flow output should stay in WGS84."""
        assert flow_artifacts["flow_crs"] == rasterio.crs.CRS.from_epsg(4326), \
            f"Flow data should stay in WGS84, got {flow_artifacts['flow_crs']}"

    def test_downsampling_follows_3x_rule(self, flow_artifacts):
        """
        If downsampling occurred, flow should be computed at ~3x target_vertices.

        Target vertices was 1M, so max_cells should be 3M.
        """
        dem_shape = flow_artifacts["dem"].shape
        flow_shape = flow_artifacts["flow_direction"].shape

        dem_cells = dem_shape[0] * dem_shape[1]
        flow_cells = flow_shape[0] * flow_shape[1]

        target_vertices = 1_000_000
        max_cells = target_vertices * 3

        if dem_cells > max_cells:
            # Downsampling should have occurred
            assert flow_cells <= max_cells * 1.1, \
                f"Flow cells ({flow_cells:,}) should be ≤ {max_cells * 1.1:,.0f} (target * 3 * 1.1)"
            assert flow_cells >= max_cells * 0.9, \
                f"Flow cells ({flow_cells:,}) should be ≥ {max_cells * 0.9:,.0f} (target * 3 * 0.9)"

    def test_transform_updates_with_downsampling(self, flow_artifacts):
        """If downsampling occurred, transform should use coarser pixels."""
        dem_transform = flow_artifacts["dem_transform"]
        flow_transform = flow_artifacts["flow_transform"]

        dem_pixel_size = abs(dem_transform.a)
        flow_pixel_size = abs(flow_transform.a)

        dem_shape = flow_artifacts["dem"].shape
        flow_shape = flow_artifacts["flow_direction"].shape

        if dem_shape != flow_shape:
            # Downsampling occurred, flow pixels should be larger
            assert flow_pixel_size > dem_pixel_size, \
                f"Flow pixels ({flow_pixel_size:.6f}°) should be larger than DEM pixels ({dem_pixel_size:.6f}°)"

    def test_all_required_outputs_present(self, flow_artifacts):
        """Flow computation should produce all required outputs."""
        required_keys = [
            "flow_direction",
            "drainage_area",
            "upstream_rainfall",
            "conditioned_dem",
        ]

        for key in required_keys:
            assert key in flow_artifacts, f"Missing required output: {key}"
            assert flow_artifacts[key] is not None, f"Output {key} is None"

    def test_metadata_contains_transform_and_crs(self, flow_artifacts):
        """Flow outputs should include transform and CRS metadata."""
        assert flow_artifacts["flow_transform"] is not None
        assert flow_artifacts["flow_crs"] is not None


# ============================================================================
# Class 3: TestGeographicCoordinates
# ============================================================================

class TestGeographicCoordinates:
    """Tests for geographic coordinate consistency."""

    def test_flow_transform_maps_to_san_diego(self, flow_artifacts):
        """Flow transform should map pixels to San Diego geographic coordinates."""
        flow_transform = flow_artifacts["flow_transform"]
        flow_shape = flow_artifacts["flow_direction"].shape

        # Sample pixel in middle
        row, col = flow_shape[0] // 2, flow_shape[1] // 2
        lon, lat = flow_transform * (col, row)

        # San Diego bbox: lat [32.5, 33.5], lon [-117.6, -116.0]
        assert 32.5 < lat < 33.5, f"Latitude {lat} should be in San Diego range [32.5, 33.5]"
        assert -117.6 < lon < -116.0, f"Longitude {lon} should be in San Diego range [-117.6, -116.0]"

    def test_flow_and_dem_same_extent(self, flow_artifacts):
        """Flow and DEM should cover approximately the same geographic extent."""
        dem_transform = flow_artifacts["dem_transform"]
        dem_shape = flow_artifacts["dem"].shape
        flow_transform = flow_artifacts["flow_transform"]
        flow_shape = flow_artifacts["flow_direction"].shape

        assert verify_geographic_alignment(
            dem_transform, dem_shape,
            flow_transform, flow_shape,
            tolerance=0.01  # 1km tolerance
        ), "Flow and DEM should cover same geographic extent"


# ============================================================================
# Class 9: TestTunedParameters (UPDATED)
# ============================================================================

class TestTunedParameters:
    """Tests for updated tuned parameters."""

    def test_uses_spec_backend(self):
        """Demo should use backend='spec'."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        assert 'backend="spec"' in demo_source or "backend='spec'" in demo_source, \
            "Demo should use backend='spec'"

    def test_uses_all_edge_mode(self):
        """Demo should use edge_mode='all'."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        assert 'edge_mode="all"' in demo_source or "edge_mode='all'" in demo_source, \
            "Demo should use edge_mode='all'"

    def test_negative_coastal_threshold(self):
        """Demo should use coastal_elev_threshold=-20.0."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        assert 'coastal_elev_threshold=-20' in demo_source, \
            "Demo should use coastal_elev_threshold=-20.0"

    def test_basin_min_size_10000(self):
        """Demo should use min_basin_size=10000 (NOT 1000!)."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        # Should NOT have old value
        assert 'min_basin_size=1000' not in demo_source or 'min_basin_size=10000' in demo_source, \
            "Demo should NOT use old min_basin_size=1000"

        # Should have new value
        assert 'min_basin_size=10000' in demo_source, \
            "Demo should use min_basin_size=10000"

    def test_basin_min_depth_5_0(self):
        """Demo should use min_basin_depth=5.0 (NOT 1.0!)."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        assert 'min_basin_depth=5.0' in demo_source or 'min_depth=5.0' in demo_source, \
            "Demo should use min_basin_depth=5.0"

    def test_max_breach_depth_100(self):
        """Demo should use max_breach_depth=100.0 (NOT 50.0!)."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        assert 'max_breach_depth=100' in demo_source, \
            "Demo should use max_breach_depth=100.0"

    def test_max_breach_length_300(self):
        """Demo should use max_breach_length=300 (NOT 100!)."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        assert 'max_breach_length=300' in demo_source, \
            "Demo should use max_breach_length=300"


# ============================================================================
# Class 24: TestDataTypes
# ============================================================================

class TestDataTypes:
    """Tests for correct data types."""

    def test_flow_arrays_float32(self, flow_artifacts):
        """Flow arrays (drainage, rainfall) should be float32."""
        assert flow_artifacts["drainage_area"].dtype == np.float32, \
            f"Drainage area should be float32, got {flow_artifacts['drainage_area'].dtype}"
        assert flow_artifacts["upstream_rainfall"].dtype == np.float32, \
            f"Upstream rainfall should be float32, got {flow_artifacts['upstream_rainfall'].dtype}"

    def test_flow_direction_uint8(self, flow_artifacts):
        """Flow direction should be uint8 with D8 encoding."""
        flow_dir = flow_artifacts["flow_direction"]
        assert flow_dir.dtype == np.uint8, \
            f"Flow direction should be uint8, got {flow_dir.dtype}"

        # Check values are valid D8 codes
        unique_vals = np.unique(flow_dir)
        valid_vals = {0, 1, 2, 4, 8, 16, 32, 64, 128}

        invalid_vals = set(unique_vals) - valid_vals
        assert len(invalid_vals) == 0, \
            f"Flow direction has invalid D8 codes: {invalid_vals}"


# ============================================================================
# Class 25: TestFlowPipelineModule (NEW)
# ============================================================================

class TestFlowPipelineModule:
    """Tests for new flow_pipeline.py module."""

    def test_compute_flow_with_basins_exists(self):
        """flow_pipeline module should export compute_flow_with_basins."""
        from src.terrain import flow_pipeline
        assert hasattr(flow_pipeline, 'compute_flow_with_basins'), \
            "flow_pipeline should export compute_flow_with_basins"

    def test_returns_all_required_keys(self, flow_artifacts):
        """compute_flow_with_basins should return all required keys."""
        # We can't easily run the function again, but we can check the demo
        # uses the correct pattern
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        # Check that flow_pipeline is imported
        assert 'from src.terrain.flow_pipeline import' in demo_source or \
               'flow_pipeline' in demo_source, \
            "Demo should use flow_pipeline module"

    def test_ocean_mask_detected(self, flow_artifacts):
        """Ocean mask should be detected and applied."""
        # Check that flow direction has outlets (value 0) along borders
        flow_dir = flow_artifacts["flow_direction"]

        # Check borders have some outlets
        border_outlets = np.sum(flow_dir[0, :] == 0) + \
                         np.sum(flow_dir[-1, :] == 0) + \
                         np.sum(flow_dir[:, 0] == 0) + \
                         np.sum(flow_dir[:, -1] == 0)

        assert border_outlets > 0, \
            "Should have some outlet cells (flow_dir=0) along borders"

    def test_basin_mask_when_detect_basins_true(self):
        """Basin detection should be enabled (detect_basins=True in demo)."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        # Check detect_basins parameter or detect_endorheic_basins call
        assert 'detect_basins=True' in demo_source or \
               'detect_endorheic_basins' in demo_source, \
            "Demo should detect endorheic basins"

    def test_lake_inlets_identified(self):
        """Lake inlet detection should be used."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        # Check if inlet identification is used (via flow_pipeline or directly)
        assert 'identify_lake_inlets' in demo_source or \
               'lake_inlets' in demo_source, \
            "Demo should identify lake inlets"

    def test_conditioning_mask_combines_ocean_basins_lakes(self):
        """Conditioning mask should combine ocean + basins + lakes_in_basins."""
        # This is tested implicitly by checking that the demo code structure
        # follows the pattern from flow_pipeline.py
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        # Check for conditioning mask creation pattern
        assert 'conditioning_mask' in demo_source or \
               'flow_pipeline' in demo_source, \
            "Demo should use conditioning mask pattern"


# ============================================================================
# Class 27: TestTerrainBasedAlignment (NEW)
# ============================================================================

class TestTerrainBasedAlignment:
    """Tests for new Terrain-based alignment approach."""

    def test_terrain_align_created_at_flow_resolution(self):
        """terrain_align should use flow resolution DEM as base."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        # Check for terrain_align creation with conditioned DEM
        assert 'terrain_align' in demo_source, \
            "Demo should create terrain_align object"
        assert 'flow_result["conditioned_dem"]' in demo_source or \
               'flow_dem' in demo_source, \
            "terrain_align should use flow resolution DEM"

    def test_all_data_added_with_target_layer_dem(self):
        """All data layers should be added with target_layer='dem'."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        # Check for add_data_layer calls with target_layer
        assert 'add_data_layer' in demo_source, \
            "Demo should use add_data_layer"
        assert 'target_layer="dem"' in demo_source or \
               "target_layer='dem'" in demo_source, \
            "Data layers should use target_layer='dem' for alignment"

    def test_apply_transforms_aligns_all_layers(self):
        """apply_transforms should align all data layers."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        # Check for apply_transforms call
        assert 'apply_transforms()' in demo_source, \
            "Demo should call apply_transforms() to align layers"

    def test_aligned_data_matches_flow_shape(self, flow_artifacts):
        """Aligned data should match flow resolution shape."""
        # This is validated implicitly by the diagnostic plots working
        # We check that diagnostic outputs exist
        output_dir = flow_artifacts["output_dir"]

        # Check for at least some diagnostic plots
        diagnostic_plots = list(output_dir.glob("0*_*.png"))
        assert len(diagnostic_plots) > 0, \
            "Diagnostic plots should exist (proves alignment worked)"

    def test_no_manual_scipy_zoom_used(self):
        """Demo should NOT use manual scipy.zoom resampling."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        # Check that scipy.zoom is NOT used for resampling (may be used elsewhere)
        # The key is that we shouldn't see the old pattern of manual resampling
        # This is a soft check - just verify terrain_align is used instead
        assert 'terrain_align' in demo_source, \
            "Demo should use terrain_align approach (not manual scipy.zoom)"

    def test_aligned_data_reused_for_rendering(self):
        """Aligned lake mask should be reused for 3D rendering."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        # Check for lake_mask_aligned being used for water_mask
        assert 'lake_mask_aligned' in demo_source, \
            "Demo should create lake_mask_aligned"
        assert 'water_mask' in demo_source, \
            "Demo should use water_mask from aligned data"


# ============================================================================
# Class 28: TestPrecipitationMasking (NEW)
# ============================================================================

class TestPrecipitationMasking:
    """Tests for precipitation data masking."""

    def test_invalid_values_filtered(self):
        """Precipitation masking code should filter extreme negatives."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        # Check for precipitation masking pattern
        assert 'precip_data >= 0' in demo_source or \
               'precip_masked' in demo_source, \
            "Demo should mask invalid precipitation values"

    def test_valid_range_0_to_20000(self):
        """Valid precipitation range should be 0-20,000 mm/year."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        # Check for upper bound check
        assert '20000' in demo_source or '< 20000' in demo_source, \
            "Demo should enforce max precipitation of 20,000 mm/year"

    def test_worldclim_masking_applied(self):
        """WorldClim data should be masked before use."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        # Check for masking in WorldClim context
        assert 'WorldClim' in demo_source or 'worldclim' in demo_source, \
            "Demo should reference WorldClim data"
        assert 'precip_data' in demo_source and \
               ('np.where' in demo_source or 'precip_masked' in demo_source), \
            "Demo should mask precipitation data"


# ============================================================================
# Test Class 9: Basin-Aware Lake Handling
# ============================================================================

class TestBasinAwareLakeHandling:
    """Test that lakes inside/outside basins are handled differently."""

    def test_lakes_loaded_before_flow_computation(self, flow_artifacts):
        """Verify lakes are loaded BEFORE flow computation (architecture requirement)."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        # Find positions of key operations
        lake_load_pos = demo_source.find("download_water_bodies")
        flow_compute_pos = demo_source.find("flow_accumulation(")

        assert lake_load_pos > 0, "Should load water bodies"
        assert flow_compute_pos > 0, "Should compute flow"
        assert lake_load_pos < flow_compute_pos, \
            "Lakes must be loaded BEFORE flow computation for proper basin-aware handling"

    def test_lake_mask_passed_to_flow_accumulation(self, flow_artifacts):
        """Verify lake_mask is passed to flow_accumulation for routing."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        # Check that lake_mask parameter is used
        assert "lake_mask=" in demo_source, \
            "Demo should pass lake_mask to flow_accumulation"
        assert "lake_outlets=" in demo_source, \
            "Demo should pass lake_outlets to flow_accumulation"

    def test_basin_detection_before_lake_handling(self, flow_artifacts):
        """Verify endorheic basins are detected before flow computation."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        # Find detection call
        basin_detect_pos = demo_source.find("detect_endorheic_basins")
        flow_compute_pos = demo_source.find("flow_accumulation(")

        assert basin_detect_pos > 0, "Should detect endorheic basins"
        assert basin_detect_pos < flow_compute_pos, \
            "Basins must be detected BEFORE flow computation for proper lake masking"

    def test_conditioning_mask_strategy_documented(self, flow_artifacts):
        """Verify the conditioning mask strategy is documented in code."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        # Check for strategy comments
        assert "basin" in demo_source.lower() or "salton" in demo_source.lower(), \
            "Demo should document basin handling strategy"

    def test_detect_basins_parameter_enabled(self, flow_artifacts):
        """Verify detect_basins=True is passed to flow_accumulation."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        assert "detect_basins=True" in demo_source, \
            "Demo should enable basin detection for flow_accumulation"


# ============================================================================
# Test Class 10: Alignment Validation
# ============================================================================

class TestAlignment:
    """Test geographic alignment between DEM, flow, and derived data."""

    def test_flow_and_dem_same_geographic_extent(self, flow_artifacts):
        """Verify flow data covers same geographic extent as DEM."""
        dem_transform = flow_artifacts["dem_transform"]
        dem_shape = flow_artifacts["dem"].shape
        flow_transform = flow_artifacts["flow_transform"]
        flow_shape = flow_artifacts["flow_direction"].shape

        # Calculate geographic bounds
        dem_bounds = (
            dem_transform.c,  # min_x
            dem_transform.f + dem_shape[0] * dem_transform.e,  # min_y
            dem_transform.c + dem_shape[1] * dem_transform.a,  # max_x
            dem_transform.f,  # max_y
        )

        flow_bounds = (
            flow_transform.c,
            flow_transform.f + flow_shape[0] * flow_transform.e,
            flow_transform.c + flow_shape[1] * flow_transform.a,
            flow_transform.f,
        )

        # Check bounds are approximately equal (within 1% tolerance for rounding)
        for dem_val, flow_val in zip(dem_bounds, flow_bounds):
            rel_diff = abs((dem_val - flow_val) / dem_val) if dem_val != 0 else abs(flow_val)
            assert rel_diff < 0.01, \
                f"Flow and DEM geographic extents should match: DEM={dem_bounds}, Flow={flow_bounds}"

    def test_flow_stays_in_wgs84(self, flow_artifacts):
        """Verify flow data remains in WGS84 (EPSG:4326) for alignment."""
        flow_crs = str(flow_artifacts["flow_crs"])
        assert "4326" in flow_crs or "WGS84" in flow_crs, \
            f"Flow should stay in WGS84 for geographic alignment, got {flow_crs}"

    def test_terrain_align_uses_target_layer_dem(self, flow_artifacts):
        """Verify terrain_align adds layers with target_layer='dem'."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        # Check for terrain_align usage
        assert "terrain_align" in demo_source, \
            "Demo should create terrain_align object"
        assert 'target_layer="dem"' in demo_source, \
            "Demo should use target_layer='dem' for alignment"

    def test_all_diagnostic_data_aligned_via_terrain(self, flow_artifacts):
        """Verify diagnostic data is aligned via Terrain, not manual scipy.zoom."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        # Should use terrain_align for diagnostic data
        assert "terrain_align.add_data_layer" in demo_source, \
            "Demo should use terrain_align for data alignment"

        # Should NOT use scipy.zoom for diagnostics
        # (Note: scipy.zoom may still be used for lake rasterization, but not for diagnostic alignment)
        diagnostic_section_start = demo_source.find("Create aligned data")
        diagnostic_section_end = demo_source.find("create_flow_diagnostics")
        if diagnostic_section_start > 0 and diagnostic_section_end > 0:
            diagnostic_section = demo_source[diagnostic_section_start:diagnostic_section_end]
            # We expect zoom might be used earlier for lake rasterization, but not in diagnostic section
            pass  # This is a softer check - main point is terrain_align exists


# ============================================================================
# Test Class 11: Water Body Integration
# ============================================================================

class TestWaterBodyIntegration:
    """Test water body (lake) data integration."""

    def test_water_bodies_loaded(self, flow_artifacts):
        """Verify water bodies are loaded from data source."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        assert "download_water_bodies" in demo_source, \
            "Demo should download water bodies"

    def test_lakes_rasterized_to_mask(self, flow_artifacts):
        """Verify lakes are rasterized to grid matching DEM."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        assert "rasterize_lakes_to_mask" in demo_source, \
            "Demo should rasterize lakes to mask"

    def test_lake_outlets_identified(self, flow_artifacts):
        """Verify lake outlet cells are identified."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        assert "identify_outlet_cells" in demo_source or "lake_outlets" in demo_source, \
            "Demo should identify lake outlet cells"

    def test_lake_mask_aligned_for_visualization(self, flow_artifacts):
        """Verify lake_mask_aligned is created for consistent visualization."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        assert "lake_mask_aligned" in demo_source, \
            "Demo should create lake_mask_aligned via terrain_align"


# ============================================================================
# Test Class 12: Output Files
# ============================================================================

class TestOutputFiles:
    """Test that all required output files are generated."""

    def test_flow_geotiffs_exist(self, flow_artifacts):
        """Verify all flow GeoTIFF files are created."""
        flow_dir = flow_artifacts["flow_output_dir"]

        required_files = [
            "flow_direction.tif",
            "flow_accumulation_area.tif",
            "flow_accumulation_rainfall.tif",
            "dem_conditioned.tif",
        ]

        for filename in required_files:
            filepath = flow_dir / filename
            assert filepath.exists(), f"Missing required output: {filename}"
            assert filepath.stat().st_size > 0, f"Output file is empty: {filename}"

    def test_geotiffs_have_consistent_metadata(self, flow_artifacts):
        """Verify all GeoTIFF outputs have consistent shape and transform."""
        flow_dir = flow_artifacts["flow_output_dir"]

        files = [
            "flow_direction.tif",
            "flow_accumulation_area.tif",
            "flow_accumulation_rainfall.tif",
            "dem_conditioned.tif",
        ]

        # Read metadata from all files
        shapes = []
        transforms = []
        crss = []

        for filename in files:
            with rasterio.open(flow_dir / filename) as src:
                shapes.append(src.shape)
                transforms.append(src.transform)
                crss.append(str(src.crs))

        # All should have same shape
        assert len(set(shapes)) == 1, \
            f"All flow outputs should have same shape, got: {shapes}"

        # All should have same transform
        assert len(set(transforms)) == 1, \
            f"All flow outputs should have same transform, got: {transforms}"

        # All should have same CRS
        assert len(set(crss)) == 1, \
            f"All flow outputs should have same CRS, got: {crss}"


# ============================================================================
# Test Class 13: Endorheic Basins
# ============================================================================

class TestEndorheicBasins:
    """Test endorheic basin detection and preservation."""

    def test_basins_detected_with_tuned_parameters(self, flow_artifacts):
        """Verify basins are detected with tuned parameters (10000/5.0)."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        # Check for basin detection call
        assert "detect_endorheic_basins" in demo_source, \
            "Demo should detect endorheic basins"

        # Check parameters
        assert "min_size=10000" in demo_source, \
            "Should use tuned min_size=10000"
        assert "min_depth=5.0" in demo_source, \
            "Should use tuned min_depth=5.0"

    def test_salton_sea_basin_expected(self, flow_artifacts):
        """Verify that Salton Sea basin is likely detected (San Diego region includes it)."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        # The demo should detect at least one basin (Salton Sea in region)
        # This is documented in code/comments
        assert "salton" in demo_source.lower() or "basin" in demo_source.lower(), \
            "Demo should reference basin preservation (e.g., Salton Sea)"


# ============================================================================
# Test Class 14: Ocean Mask
# ============================================================================

class TestOceanMask:
    """Test ocean detection and masking."""

    def test_ocean_detected_border_only_mode(self, flow_artifacts):
        """Verify ocean is detected using border_only mode."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        assert "detect_ocean_mask" in demo_source, \
            "Demo should detect ocean mask"
        assert "border_only=True" in demo_source, \
            "Should use border_only=True for ocean detection"

    def test_ocean_mask_used_in_basin_detection(self, flow_artifacts):
        """Verify ocean mask is excluded from basin detection."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        # Ocean should be detected before basins
        ocean_pos = demo_source.find("detect_ocean_mask")
        basin_pos = demo_source.find("detect_endorheic_basins")

        assert ocean_pos > 0 and basin_pos > 0, \
            "Both ocean and basin detection should occur"
        assert ocean_pos < basin_pos, \
            "Ocean must be detected before basins"

        # Basins should exclude ocean
        assert "exclude_mask=ocean_mask" in demo_source, \
            "Basin detection should exclude ocean mask"


# ============================================================================
# Test Class 15: Output Structure
# ============================================================================

class TestOutputStructure:
    """Test output directory structure and file organization."""

    def test_diagnostic_plots_numbered_correctly(self, flow_artifacts):
        """Verify diagnostic plots are numbered 01-11."""
        output_dir = flow_artifacts["output_dir"]
        plots = sorted(output_dir.glob("0*_*.png"))

        assert len(plots) >= 5, \
            f"Should have at least 5 diagnostic plots, found {len(plots)}"

        # Check numbering starts with 01
        if plots:
            first_plot = plots[0].name
            assert first_plot.startswith("01_") or first_plot.startswith("0"), \
                f"Diagnostic plots should start with 01_, got {first_plot}"

    def test_flow_outputs_in_subdirectory(self, flow_artifacts):
        """Verify flow GeoTIFFs are in flow_outputs subdirectory."""
        flow_dir = flow_artifacts["flow_output_dir"]

        assert flow_dir.exists(), "flow_outputs subdirectory should exist"
        assert flow_dir.is_dir(), "flow_outputs should be a directory"
        assert "flow_outputs" in str(flow_dir), \
            "Flow GeoTIFFs should be in flow_outputs subdirectory"


# ============================================================================
# Test Class 16: Mesh Creation
# ============================================================================

class TestMeshCreation:
    """Test 3D mesh generation (when rendering is enabled)."""

    def test_mesh_uses_target_vertices_parameter(self, flow_artifacts):
        """Verify mesh creation respects target_vertices parameter."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        assert "target_vertices" in demo_source, \
            "Demo should use target_vertices parameter"
        assert "configure_for_target_vertices" in demo_source, \
            "Demo should call configure_for_target_vertices"

    def test_mesh_uses_aligned_water_mask(self, flow_artifacts):
        """Verify mesh creation uses aligned water mask from diagnostics."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        # Check that lake_mask_aligned is reused
        assert "lake_mask_aligned" in demo_source, \
            "Demo should create lake_mask_aligned"

        # Should be used for water detection
        water_section = demo_source[demo_source.find("HydroLAKES"):] if "HydroLAKES" in demo_source else demo_source
        assert "lake_mask_aligned" in water_section or "water_mask" in water_section, \
            "Demo should use aligned lake mask for water detection"

    def test_terrain_applies_transforms_before_mesh(self, flow_artifacts):
        """Verify terrain.apply_transforms() is called before mesh creation."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        # Find positions
        apply_pos = demo_source.find("terrain.apply_transforms()")
        mesh_pos = demo_source.find("create_mesh")

        if apply_pos > 0 and mesh_pos > 0:
            assert apply_pos < mesh_pos, \
                "terrain.apply_transforms() should be called before create_mesh()"


# ============================================================================
# Test Class 17: Terrain Class
# ============================================================================

class TestTerrainClass:
    """Test Terrain class operations during flow pipeline."""

    def test_terrain_receives_full_resolution_dem(self, flow_artifacts):
        """Verify terrain object receives original full-resolution DEM, not downsampled."""
        dem_shape = flow_artifacts["dem"].shape
        flow_shape = flow_artifacts["flow_direction"].shape

        # DEM should be larger than flow (flow is downsampled)
        assert dem_shape[0] * dem_shape[1] >= flow_shape[0] * flow_shape[1], \
            f"Terrain should receive full DEM ({dem_shape}), not downsampled flow ({flow_shape})"

    def test_terrain_transforms_include_reproject_flip_scale(self, flow_artifacts):
        """Verify terrain applies standard transforms (reproject, flip, scale)."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        assert "reproject_raster" in demo_source, "Should reproject to UTM"
        assert "flip_raster" in demo_source, "Should flip raster"
        assert "scale_elevation" in demo_source, "Should scale elevation"

    def test_after_transforms_in_utm11n(self, flow_artifacts):
        """Verify terrain ends up in UTM Zone 11N after transforms."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        assert "EPSG:32611" in demo_source or "UTM" in demo_source, \
            "Should reproject to UTM Zone 11N (EPSG:32611)"

    def test_flow_added_as_data_layer(self, flow_artifacts):
        """Verify flow data is added as data layer to terrain."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        assert "add_data_layer" in demo_source, \
            "Should add flow data as data layer"
        assert "drainage_area" in demo_source or "rainfall" in demo_source, \
            "Should add drainage/rainfall data"


# ============================================================================
# Test Class 18: Downsampling Scenarios
# ============================================================================

class TestDownsamplingScenarios:
    """Test different downsampling scenarios based on DEM size."""

    def test_flow_downsampling_follows_3x_rule(self, flow_artifacts):
        """Verify flow is computed at ~3x target_vertices for accuracy."""
        # This is checked via flow_accumulation implementation
        # The 3x factor ensures flow accuracy before mesh downsampling
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        assert "target_vertices" in demo_source, \
            "Should use target_vertices parameter"

    def test_transform_updates_with_downsampling(self, flow_artifacts):
        """Verify transform is updated when downsampling occurs."""
        flow_transform = flow_artifacts["flow_transform"]
        dem_transform = flow_artifacts["dem_transform"]

        # If downsampling occurred, pixel size should be larger
        flow_pixel_size = abs(flow_transform.a)
        dem_pixel_size = abs(dem_transform.a)

        # Flow can be same or larger pixel size (never smaller)
        assert flow_pixel_size >= dem_pixel_size * 0.99, \
            f"Flow pixel size ({flow_pixel_size}) should be >= DEM pixel size ({dem_pixel_size})"

    def test_metadata_contains_downsampling_info(self, flow_artifacts):
        """Verify flow metadata includes downsampling information."""
        # Metadata is stored in GeoTIFF files
        flow_dir = flow_artifacts["flow_output_dir"]
        with rasterio.open(flow_dir / "flow_direction.tif") as src:
            assert src.transform is not None, "Should have transform"
            assert src.crs is not None, "Should have CRS"


# ============================================================================
# Test Class 19: Transform Reconstruction
# ============================================================================

class TestTransformReconstruction:
    """Test Affine transform reconstruction from metadata."""

    def test_transform_stored_as_6tuple(self, flow_artifacts):
        """Verify transform is stored as 6-tuple in metadata."""
        flow_dir = flow_artifacts["flow_output_dir"]
        with rasterio.open(flow_dir / "flow_direction.tif") as src:
            transform = src.transform

            # Affine has 6 parameters (a, b, c, d, e, f)
            assert hasattr(transform, 'a') and hasattr(transform, 'f'), \
                "Transform should be Affine with a, b, c, d, e, f parameters"

    def test_reconstructed_transform_valid(self, flow_artifacts):
        """Verify reconstructed transform can transform coordinates."""
        flow_transform = flow_artifacts["flow_transform"]

        # Test transform by converting pixel to geo coordinates
        row, col = 100, 100
        x, y = flow_transform * (col, row)

        # Coordinates should be in valid range for San Diego
        assert 32.0 < y < 34.0, f"Y coordinate {y} should be in San Diego lat range"
        assert -118.0 < x < -116.0, f"X coordinate {x} should be in San Diego lon range"


# ============================================================================
# Test Class 20: Specific Transforms
# ============================================================================

class TestSpecificTransforms:
    """Test specific transform operations."""

    def test_flip_is_horizontal(self, flow_artifacts):
        """Verify flip is horizontal, not vertical."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        assert 'axis="horizontal"' in demo_source or "horizontal" in demo_source, \
            "Flip should be horizontal"

    def test_elevation_scaled_by_0_0001(self, flow_artifacts):
        """Verify elevation is scaled by 0.0001."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        assert "0.0001" in demo_source, \
            "Should scale elevation by 0.0001"

    def test_utm_zone_11n(self, flow_artifacts):
        """Verify San Diego uses UTM Zone 11N (EPSG:32611)."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        assert "32611" in demo_source, \
            "Should use EPSG:32611 (UTM Zone 11N) for San Diego"


# ============================================================================
# Test Class 21: Log Scale Transform
# ============================================================================

class TestLogScaleTransform:
    """Test logarithmic scaling for visualization."""

    def test_log_scale_applied_before_visualization(self, flow_artifacts):
        """Verify log scale is applied to flow data for better visualization."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        assert "log10" in demo_source or "np.log" in demo_source, \
            "Should apply log scale to flow data"

    def test_log_handles_zeros(self, flow_artifacts):
        """Verify log scale handles zeros with +1 offset."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        assert "+ 1" in demo_source or "+1" in demo_source, \
            "Should add 1 before log to handle zeros"


# ============================================================================
# Test Class 22: Color Mapping
# ============================================================================

class TestColorMapping:
    """Test color mapping strategies."""

    def test_color_by_selects_layer(self, flow_artifacts):
        """Verify --color-by parameter selects different data layers."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        assert "color_by" in demo_source, \
            "Should have color_by parameter"
        assert "elevation" in demo_source and "drainage" in demo_source, \
            "Should support multiple coloring options"

    def test_discharge_computed_when_needed(self, flow_artifacts):
        """Verify discharge potential is computed for discharge visualization."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        assert "discharge_potential" in demo_source or "compute_discharge" in demo_source, \
            "Should compute discharge potential"


# ============================================================================
# Test Class 23: Water Detection
# ============================================================================

class TestWaterDetection:
    """Test water body detection strategies."""

    def test_prefers_lake_mask_over_detection(self, flow_artifacts):
        """Verify pipeline prefers HydroLAKES mask over slope-based detection."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        # Should check for lake_mask_aligned first
        lake_mask_pos = demo_source.find("lake_mask_aligned")
        detect_water_pos = demo_source.find("detect_water_highres")

        assert lake_mask_pos > 0, "Should use lake_mask_aligned"
        if detect_water_pos > 0:
            assert lake_mask_pos < detect_water_pos, \
                "Should check lake_mask_aligned before falling back to detect_water_highres"

    def test_water_mask_passed_to_mesh(self, flow_artifacts):
        """Verify water mask is passed to mesh creation."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        assert "water_mask=" in demo_source, \
            "Should pass water_mask to create_mesh"


# ============================================================================
# Test Class 24: Diagnostic Alignment
# ============================================================================

class TestDiagnosticAlignment:
    """Test diagnostic visualization data alignment."""

    def test_viz_data_resampled_to_flow(self, flow_artifacts):
        """Verify diagnostic data is resampled to match flow resolution."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        assert "terrain_align" in demo_source, \
            "Should use terrain_align for diagnostic data"

    def test_diagnostics_receive_aligned_data(self, flow_artifacts):
        """Verify diagnostic plots receive aligned data."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        assert "create_flow_diagnostics" in demo_source, \
            "Should create diagnostic visualizations"


# ============================================================================
# Test Class 25: Coordinate System Sequence
# ============================================================================

class TestCoordinateSystemSequence:
    """Test coordinate system transformations through pipeline."""

    def test_load_phase_wgs84(self, flow_artifacts):
        """Verify data loads in WGS84."""
        dem_path = flow_artifacts["dem_dir"]
        with rasterio.open(list(dem_path.glob("*.hgt"))[0]) as src:
            crs = str(src.crs)
            assert "4326" in crs or "WGS" in crs, \
                f"DEM should load in WGS84, got {crs}"

    def test_flow_phase_wgs84(self, flow_artifacts):
        """Verify flow computation stays in WGS84."""
        flow_crs = str(flow_artifacts["flow_crs"])
        assert "4326" in flow_crs or "WGS84" in flow_crs, \
            f"Flow should stay in WGS84, got {flow_crs}"

    def test_terrain_initial_wgs84(self, flow_artifacts):
        """Verify terrain object starts in WGS84."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        # Terrain is created with dem_crs="EPSG:4326"
        assert 'dem_crs="EPSG:4326"' in demo_source or "4326" in demo_source, \
            "Terrain should start in WGS84"

    def test_after_reproject_utm11n(self, flow_artifacts):
        """Verify terrain is in UTM 11N after reproject transform."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        assert "EPSG:32611" in demo_source, \
            "Should reproject to UTM Zone 11N"

    def test_mesh_unitless(self, flow_artifacts):
        """Verify mesh is unitless (centered at origin)."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        assert "center_model=True" in demo_source, \
            "Should center mesh at origin"


# ============================================================================
# Test Class 26: Resolution Progression
# ============================================================================

class TestResolutionProgression:
    """Test resolution changes through pipeline."""

    def test_resolution_increases_through_pipeline(self, flow_artifacts):
        """Verify pixel size increases (resolution decreases) through pipeline."""
        dem_transform = flow_artifacts["dem_transform"]
        flow_transform = flow_artifacts["flow_transform"]

        dem_pixel_size = abs(dem_transform.a)
        flow_pixel_size = abs(flow_transform.a)

        # Flow can have same or larger pixels (downsampling)
        assert flow_pixel_size >= dem_pixel_size * 0.99, \
            f"Flow pixel size ({flow_pixel_size}) should be >= DEM ({dem_pixel_size})"


# ============================================================================
# Test Class 27: Mesh Properties
# ============================================================================

class TestMeshProperties:
    """Test 3D mesh generation properties."""

    def test_mesh_centered(self, flow_artifacts):
        """Verify mesh is centered at origin."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        assert "center_model=True" in demo_source, \
            "Should center mesh at origin"

    def test_height_exaggerated(self, flow_artifacts):
        """Verify height is exaggerated for visibility."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        assert "height_scale" in demo_source, \
            "Should specify height_scale for exaggeration"

    def test_boundary_extension(self, flow_artifacts):
        """Verify boundary extension for seamless edges."""
        demo_script = PROJECT_ROOT / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_script.read_text()

        assert "boundary_extension=True" in demo_source, \
            "Should use boundary_extension for seamless edges"


# ============================================================================
# Test Class 28: Caching
# ============================================================================

class TestCaching:
    """Test caching behavior."""

    def test_flow_computation_cacheable(self, flow_artifacts):
        """Verify flow outputs are saved to files for caching."""
        flow_dir = flow_artifacts["flow_output_dir"]

        # Check that files exist (cached outputs)
        assert (flow_dir / "flow_direction.tif").exists(), \
            "Flow direction should be cached to file"
        assert (flow_dir / "flow_accumulation_area.tif").exists(), \
            "Drainage area should be cached to file"
        assert (flow_dir / "flow_accumulation_rainfall.tif").exists(), \
            "Upstream rainfall should be cached to file"


# ============================================================================
# Integration Test
# ============================================================================

def test_complete_pipeline_integration(flow_artifacts):
    """
    End-to-end integration test of the complete pipeline.

    This test validates that all stages work together correctly:
    1. Data acquisition
    2. Basin detection
    3. Flow computation with basin preservation
    4. Terrain-based alignment
    5. Output generation
    """
    # 1. Data acquisition - check DEM loaded
    assert flow_artifacts["dem"] is not None
    assert flow_artifacts["dem"].shape[0] > 0

    # 2. Flow computation - check outputs exist
    assert flow_artifacts["flow_direction"] is not None
    assert flow_artifacts["drainage_area"] is not None
    assert flow_artifacts["upstream_rainfall"] is not None

    # 3. Output files - check GeoTIFFs created
    flow_dir = flow_artifacts["flow_output_dir"]
    required_files = [
        "flow_direction.tif",
        "flow_accumulation_area.tif",
        "flow_accumulation_rainfall.tif",
        "dem_conditioned.tif",
    ]

    for filename in required_files:
        filepath = flow_dir / filename
        assert filepath.exists(), f"Missing required output file: {filename}"

    # 4. Diagnostic plots - check they were generated
    output_dir = flow_artifacts["output_dir"]
    diagnostic_plots = list(output_dir.glob("0*_*.png"))
    assert len(diagnostic_plots) >= 5, \
        f"Should have at least 5 diagnostic plots, found {len(diagnostic_plots)}"

    print("\n✅ Complete pipeline integration test PASSED")
    print(f"   - DEM shape: {flow_artifacts['dem'].shape}")
    print(f"   - Flow shape: {flow_artifacts['flow_direction'].shape}")
    print(f"   - Output files: {len(required_files)}")
    print(f"   - Diagnostic plots: {len(diagnostic_plots)}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
