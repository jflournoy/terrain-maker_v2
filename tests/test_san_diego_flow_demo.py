"""
🔴 TDD RED: Failing tests for San Diego Flow Demo refactoring.

These tests define the NEW behavior for the refactored san_diego_flow_demo.py:
1. Support for `--color-by discharge` (discharge potential = drainage × rainfall weight)
2. Integration with HydroLAKES water body data
3. Modular flow computation using new library functions

All tests should FAIL initially - this is the RED phase of TDD.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys


class TestDischargeColorOption:
    """Tests for new --color-by discharge option."""

    def test_color_by_choices_includes_discharge(self):
        """argparse choices should include 'discharge' option."""
        # Import the demo module
        sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

        # We need to test that the argparse choices include 'discharge'
        # This requires inspecting the parser configuration
        from san_diego_flow_demo import main
        import argparse

        # Create a mock parser to capture the choices
        with patch('argparse.ArgumentParser.parse_args') as mock_parse:
            with patch('argparse.ArgumentParser.add_argument') as mock_add:
                # Capture all add_argument calls
                calls = []

                def capture_add_arg(*args, **kwargs):
                    calls.append((args, kwargs))
                    return MagicMock()

                mock_add.side_effect = capture_add_arg
                mock_parse.side_effect = SystemExit  # Stop execution

                try:
                    main()
                except (SystemExit, Exception):
                    pass

        # Find the --color-by argument
        color_by_call = None
        for args, kwargs in calls:
            if '--color-by' in args:
                color_by_call = kwargs
                break

        assert color_by_call is not None, "--color-by argument not found"
        assert 'choices' in color_by_call, "--color-by should have choices"
        assert 'discharge' in color_by_call['choices'], \
            f"'discharge' not in choices: {color_by_call['choices']}"


class TestDischargeCalculation:
    """Tests for discharge potential calculation."""

    def test_compute_discharge_potential_returns_array(self):
        """compute_discharge_potential should return numpy array."""
        from src.terrain.flow_accumulation import compute_discharge_potential

        drainage_area = np.array([[1, 2, 4], [2, 8, 16], [4, 16, 64]], dtype=np.float32)
        upstream_rainfall = np.array([[100, 200, 400], [200, 800, 1600], [400, 1600, 6400]], dtype=np.float32)

        discharge = compute_discharge_potential(drainage_area, upstream_rainfall)

        assert isinstance(discharge, np.ndarray)
        assert discharge.shape == drainage_area.shape

    def test_compute_discharge_potential_formula(self):
        """Discharge potential = drainage_area × (upstream_rainfall / mean_rainfall)."""
        from src.terrain.flow_accumulation import compute_discharge_potential

        drainage_area = np.array([[1, 2], [4, 8]], dtype=np.float32)
        upstream_rainfall = np.array([[100, 200], [400, 800]], dtype=np.float32)

        discharge = compute_discharge_potential(drainage_area, upstream_rainfall)

        # Mean rainfall = (100 + 200 + 400 + 800) / 4 = 375
        mean_rainfall = np.mean(upstream_rainfall)

        # Expected discharge = drainage_area * (upstream_rainfall / mean_rainfall)
        expected = drainage_area * (upstream_rainfall / mean_rainfall)

        np.testing.assert_allclose(discharge, expected, rtol=1e-5)

    def test_compute_discharge_potential_handles_zeros(self):
        """Should handle zero rainfall cells without division errors."""
        from src.terrain.flow_accumulation import compute_discharge_potential

        drainage_area = np.array([[1, 2], [4, 8]], dtype=np.float32)
        upstream_rainfall = np.array([[0, 200], [400, 0]], dtype=np.float32)

        # Should not raise an error
        discharge = compute_discharge_potential(drainage_area, upstream_rainfall)

        # Zero rainfall cells should have zero discharge
        assert discharge[0, 0] == 0
        assert discharge[1, 1] == 0


class TestWaterBodyIntegration:
    """Tests for HydroLAKES integration in the demo."""

    def test_demo_supports_water_bodies_flag(self):
        """Demo should support water bodies control flag (--water-bodies or --no-water-bodies)."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

        from san_diego_flow_demo import main
        import argparse

        with patch('argparse.ArgumentParser.parse_args') as mock_parse:
            with patch('argparse.ArgumentParser.add_argument') as mock_add:
                calls = []

                def capture_add_arg(*args, **kwargs):
                    calls.append((args, kwargs))
                    return MagicMock()

                mock_add.side_effect = capture_add_arg
                mock_parse.side_effect = SystemExit

                try:
                    main()
                except (SystemExit, Exception):
                    pass

        # Find water bodies argument (either --water-bodies or --no-water-bodies)
        water_bodies_found = any(
            '--water-bodies' in args or '--lakes' in args or '--no-water-bodies' in args
            for args, kwargs in calls
        )

        assert water_bodies_found, "Demo should support --water-bodies, --no-water-bodies, or --lakes flag"

    def test_demo_supports_data_source_option(self):
        """Demo should support --data-source for selecting lake data provider."""
        sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

        from san_diego_flow_demo import main

        with patch('argparse.ArgumentParser.parse_args') as mock_parse:
            with patch('argparse.ArgumentParser.add_argument') as mock_add:
                calls = []

                def capture_add_arg(*args, **kwargs):
                    calls.append((args, kwargs))
                    return MagicMock()

                mock_add.side_effect = capture_add_arg
                mock_parse.side_effect = SystemExit

                try:
                    main()
                except (SystemExit, Exception):
                    pass

        # Find data-source argument
        data_source_call = None
        for args, kwargs in calls:
            if '--data-source' in args:
                data_source_call = kwargs
                break

        assert data_source_call is not None, "--data-source argument not found"
        if 'choices' in data_source_call:
            assert 'hydrolakes' in data_source_call['choices'], \
                "hydrolakes should be a valid data source"


class TestModularFlowComputation:
    """Tests for modular flow computation in demo."""

    def test_demo_uses_modular_functions(self):
        """Demo should use modular flow functions instead of monolithic flow_accumulation."""
        demo_path = Path(__file__).parent.parent / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_path.read_text()

        # Check for imports of modular functions
        modular_imports = [
            "compute_flow_direction",
            "compute_drainage_area",
            "condition_dem",
        ]

        missing_imports = [
            name for name in modular_imports
            if name not in demo_source
        ]

        assert len(missing_imports) == 0, \
            f"Demo should import modular functions: {missing_imports}"

    def test_demo_computes_discharge_potential(self):
        """Demo should compute discharge potential when --color-by discharge."""
        demo_path = Path(__file__).parent.parent / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_path.read_text()

        # Check for discharge potential computation
        assert "discharge_potential" in demo_source or "discharge" in demo_source.lower(), \
            "Demo should compute discharge potential for --color-by discharge"


class TestDemoOutputs:
    """Tests for demo output behavior."""

    def test_demo_adds_discharge_layer(self):
        """Demo should add discharge_potential as a data layer."""
        demo_path = Path(__file__).parent.parent / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_path.read_text()

        # Check for adding discharge as data layer
        assert 'add_data_layer' in demo_source, "Demo should use add_data_layer"

        # Should add discharge layer
        discharge_patterns = [
            'discharge_potential',
            '"discharge"',
            "'discharge'",
        ]
        found_discharge = any(p in demo_source for p in discharge_patterns)
        assert found_discharge, "Demo should add discharge_potential as a data layer"

    def test_demo_supports_lake_overlay(self):
        """Demo should support overlaying lakes on visualization."""
        demo_path = Path(__file__).parent.parent / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_path.read_text()

        # Check for lake-related visualization
        lake_viz_patterns = [
            'lake_mask',
            'water_bodies',
            'lakes',
        ]
        found_lake_viz = any(p in demo_source for p in lake_viz_patterns)
        assert found_lake_viz, "Demo should support lake visualization overlay"


class TestFlowAccumulationModule:
    """Tests for new functions in flow_accumulation module."""

    def test_module_exports_compute_discharge_potential(self):
        """flow_accumulation module should export compute_discharge_potential."""
        from src.terrain import flow_accumulation

        assert hasattr(flow_accumulation, 'compute_discharge_potential'), \
            "flow_accumulation module should export compute_discharge_potential"

    def test_compute_discharge_potential_signature(self):
        """compute_discharge_potential should accept drainage_area and upstream_rainfall."""
        from src.terrain.flow_accumulation import compute_discharge_potential
        import inspect

        sig = inspect.signature(compute_discharge_potential)
        params = list(sig.parameters.keys())

        assert 'drainage_area' in params, "Should accept drainage_area parameter"
        assert 'upstream_rainfall' in params, "Should accept upstream_rainfall parameter"


# Fixtures for integration tests

@pytest.fixture
def sample_dem():
    """Create a simple synthetic DEM for testing."""
    # Simple valley shape
    rows, cols = 50, 50
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    X, Y = np.meshgrid(x, y)

    # V-shaped valley draining south
    dem = 100 + np.abs(X) * 50 - Y * 30
    return dem.astype(np.float32)


@pytest.fixture
def sample_precip():
    """Create synthetic precipitation for testing."""
    rows, cols = 50, 50
    # Higher precipitation in mountains
    precip = np.ones((rows, cols), dtype=np.float32) * 500
    precip[:25, :] += 300  # More rain in north (higher elevation)
    return precip


class TestIntegration:
    """Integration tests for the refactored demo."""

    def test_discharge_potential_computation_end_to_end(self, sample_dem, sample_precip):
        """Full computation of discharge potential from DEM and precipitation."""
        from src.terrain.flow_accumulation import (
            compute_flow_direction,
            compute_drainage_area,
            compute_upstream_rainfall,
            compute_discharge_potential,
            condition_dem,
        )

        # Condition DEM
        dem_conditioned = condition_dem(sample_dem)

        # Compute flow
        flow_dir = compute_flow_direction(dem_conditioned)
        drainage_area = compute_drainage_area(flow_dir)
        upstream_rainfall = compute_upstream_rainfall(flow_dir, sample_precip)

        # Compute discharge potential
        discharge = compute_discharge_potential(drainage_area, upstream_rainfall)

        # Validate output
        assert discharge.shape == sample_dem.shape
        assert np.all(discharge >= 0), "Discharge should be non-negative"
        assert np.max(discharge) > np.mean(discharge), "Should have concentration at outlets"

    def test_discharge_correlates_with_drainage(self, sample_dem, sample_precip):
        """Discharge potential should correlate with drainage area."""
        from src.terrain.flow_accumulation import (
            compute_flow_direction,
            compute_drainage_area,
            compute_upstream_rainfall,
            compute_discharge_potential,
            condition_dem,
        )

        dem_conditioned = condition_dem(sample_dem)
        flow_dir = compute_flow_direction(dem_conditioned)
        drainage_area = compute_drainage_area(flow_dir)
        upstream_rainfall = compute_upstream_rainfall(flow_dir, sample_precip)
        discharge = compute_discharge_potential(drainage_area, upstream_rainfall)

        # Find max discharge and max drainage locations
        max_discharge_idx = np.unravel_index(np.argmax(discharge), discharge.shape)
        max_drainage_idx = np.unravel_index(np.argmax(drainage_area), drainage_area.shape)

        # They should be at or near the same location (outlet)
        distance = np.sqrt(
            (max_discharge_idx[0] - max_drainage_idx[0])**2 +
            (max_discharge_idx[1] - max_drainage_idx[1])**2
        )

        assert distance < 5, \
            f"Max discharge and drainage should be near same location (distance={distance})"


# ============================================================================
# TDD RED: Tests for San Diego Demo Parameter Alignment
# ============================================================================
# These tests ensure san_diego_flow_demo.py uses the same tuned parameters
# as validate_flow_with_water_bodies.py:
#
#   --backend spec
#   --edge-mode all
#   --coastal-elev-threshold -20
#   --detect-basins
#   --basin-min-size 1000
#   --basin-min-depth 1
#   --data-source hydrolakes
# ============================================================================


class TestFlowParameterAlignment:
    """Tests ensuring demo uses tuned validation parameters."""

    def test_demo_uses_spec_backend(self):
        """Demo should use backend='spec' for flow accumulation."""
        demo_path = Path(__file__).parent.parent / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_path.read_text()

        # Should explicitly pass backend="spec" to flow_accumulation
        assert 'backend="spec"' in demo_source or "backend='spec'" in demo_source, \
            "Demo should explicitly use backend='spec' for flow_accumulation"

    def test_demo_uses_negative_coastal_threshold(self):
        """Demo should use coastal_elev_threshold=-20 for below-sea-level outlets."""
        demo_path = Path(__file__).parent.parent / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_path.read_text()

        # Should set coastal_elev_threshold=-20 (or lower)
        assert 'coastal_elev_threshold=-20' in demo_source or \
               'coastal_elev_threshold = -20' in demo_source, \
            "Demo should use coastal_elev_threshold=-20 for below-sea-level outlets"

    def test_demo_uses_correct_basin_min_size(self):
        """Demo should use min_basin_size=1000 (not 2500 or 5000)."""
        demo_path = Path(__file__).parent.parent / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_path.read_text()

        # Should NOT have min_basin_size=2500 or 5000
        assert 'min_basin_size=2500' not in demo_source, \
            "Demo should not use min_basin_size=2500 (old value)"
        assert 'min_basin_size=5000' not in demo_source, \
            "Demo should not use min_basin_size=5000 (old value)"

        # Should have min_basin_size=1000
        assert 'min_basin_size=1000' in demo_source or \
               'min_basin_size = 1000' in demo_source, \
            "Demo should use min_basin_size=1000"

    def test_demo_detects_endorheic_basins(self):
        """Demo should detect and preserve endorheic basins."""
        demo_path = Path(__file__).parent.parent / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_path.read_text()

        # Should import or use detect_endorheic_basins
        assert 'detect_endorheic_basins' in demo_source, \
            "Demo should use detect_endorheic_basins for basin preservation"

    def test_demo_uses_basin_min_depth(self):
        """Demo should use basin_min_depth parameter."""
        demo_path = Path(__file__).parent.parent / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_path.read_text()

        # Should set min_depth or basin_min_depth
        min_depth_patterns = [
            'min_depth=1',
            'min_depth = 1',
            'basin_min_depth=1',
            'basin_min_depth = 1',
        ]
        found = any(p in demo_source for p in min_depth_patterns)
        assert found, \
            "Demo should use min_depth=1 for basin detection"


class TestWaterBodyIntegrationOrder:
    """Tests ensuring water bodies are integrated BEFORE flow computation."""

    def test_demo_loads_water_bodies_before_flow(self):
        """Water bodies should be loaded before calling flow_accumulation."""
        demo_path = Path(__file__).parent.parent / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_path.read_text()

        # Find positions of key operations
        flow_acc_pos = demo_source.find('flow_accumulation(')
        water_bodies_pos = demo_source.find('download_water_bodies(')

        assert water_bodies_pos != -1, \
            "Demo should call download_water_bodies"
        assert flow_acc_pos != -1, \
            "Demo should call flow_accumulation"

        # Water bodies download should come BEFORE flow_accumulation
        assert water_bodies_pos < flow_acc_pos, \
            f"download_water_bodies should be called BEFORE flow_accumulation " \
            f"(water_bodies at {water_bodies_pos}, flow_acc at {flow_acc_pos})"

    def test_demo_passes_lake_mask_to_flow(self):
        """Demo should pass lake_mask parameter to flow_accumulation."""
        demo_path = Path(__file__).parent.parent / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_path.read_text()

        # Should pass lake_mask to flow_accumulation
        assert 'lake_mask=' in demo_source, \
            "Demo should pass lake_mask to flow_accumulation"

    def test_demo_passes_lake_outlets_to_flow(self):
        """Demo should pass lake_outlets parameter to flow_accumulation."""
        demo_path = Path(__file__).parent.parent / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_path.read_text()

        # Should pass lake_outlets to flow_accumulation
        assert 'lake_outlets=' in demo_source, \
            "Demo should pass lake_outlets to flow_accumulation"


class TestHydroLAKESIntegration:
    """Tests for HydroLAKES water body integration."""

    def test_demo_defaults_to_hydrolakes(self):
        """Demo should default to using HydroLAKES data source."""
        demo_path = Path(__file__).parent.parent / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_path.read_text()

        # Check --data-source default is hydrolakes
        # Looking for pattern: default="hydrolakes" or default='hydrolakes'
        hydrolakes_default_patterns = [
            'default="hydrolakes"',
            "default='hydrolakes'",
        ]
        found = any(p in demo_source for p in hydrolakes_default_patterns)
        assert found, \
            "Demo should default to data_source='hydrolakes'"

    def test_demo_enables_water_bodies_by_default(self):
        """Demo should enable water bodies by default (not require --water-bodies flag)."""
        demo_path = Path(__file__).parent.parent / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_path.read_text()

        # Water bodies should be enabled by default
        # This could be: default=True for --water-bodies, or always loading water bodies
        water_bodies_default_patterns = [
            'water_bodies=True',  # hardcoded
            'default=True',       # --water-bodies default
            'action="store_false"',  # --no-water-bodies pattern
        ]

        # Check if water bodies are loaded unconditionally
        unconditional_load = 'download_water_bodies(' in demo_source and \
                            'if args.water_bodies' not in demo_source

        found = any(p in demo_source for p in water_bodies_default_patterns) or unconditional_load

        # This test documents the DESIRED behavior - water bodies should be default
        assert found or unconditional_load, \
            "Demo should enable water bodies by default"


class TestDemoFlowAccumulationCall:
    """Tests for the actual flow_accumulation call parameters."""

    def test_flow_accumulation_receives_all_tuned_params(self):
        """flow_accumulation should receive all tuned parameters."""
        demo_path = Path(__file__).parent.parent / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_path.read_text()

        # Find the flow_accumulation call
        # Required parameters from tuned settings:
        required_params = [
            'backend=',
            'coastal_elev_threshold=',
            'edge_mode=',
        ]

        for param in required_params:
            assert param in demo_source, \
                f"flow_accumulation should include {param}"

    def test_demo_uses_edge_mode_all(self):
        """Demo should use edge_mode='all' for boundary outlets."""
        demo_path = Path(__file__).parent.parent / "examples" / "san_diego_flow_demo.py"
        demo_source = demo_path.read_text()

        # Should set edge_mode="all"
        assert 'edge_mode="all"' in demo_source or \
               "edge_mode='all'" in demo_source, \
            "Demo should use edge_mode='all'"
