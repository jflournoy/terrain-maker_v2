"""
Tests for Detroit snow integration example.

This test suite validates the snow integration example that demonstrates
combining terrain elevation data with snow analysis to produce visual outputs
at multiple stages of the pipeline.
"""

import pytest
import tempfile
import shutil
from pathlib import Path


class TestDetroitSnowExample:
    """Test the Detroit snow integration example script."""

    def setup_method(self):
        """Set up test fixtures."""
        self.output_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

    def test_example_script_exists(self):
        """Test that the detroit_snow_sledding.py example exists."""
        example_path = Path(__file__).parent.parent / "examples" / "detroit_snow_sledding.py"
        assert example_path.exists(), f"Example script not found at {example_path}"

    def test_example_imports(self):
        """Test that example script can be imported without errors."""
        import sys
        from pathlib import Path

        # Add examples directory to path
        examples_dir = Path(__file__).parent.parent / "examples"
        sys.path.insert(0, str(examples_dir))

        try:
            # Import should not raise
            import detroit_snow_sledding
            assert hasattr(detroit_snow_sledding, "main")
        finally:
            sys.path.remove(str(examples_dir))

    def test_example_produces_dem_visualization(self):
        """Test that example produces DEM visualization output."""
        from pathlib import Path
        import subprocess
        import sys

        example_path = Path(__file__).parent.parent / "examples" / "detroit_snow_sledding.py"

        # Run example with output directory
        result = subprocess.run(
            [sys.executable, str(example_path), "--output-dir", str(self.output_dir), "--step", "dem"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Check success
        assert result.returncode == 0, f"Example failed: {result.stderr}"

        # Check DEM visualization was created
        dem_viz = self.output_dir / "detroit_dem.png"
        assert dem_viz.exists(), "DEM visualization not created"
        assert dem_viz.stat().st_size > 0, "DEM visualization is empty"

    def test_example_produces_snow_depth_visualization(self):
        """Test that example produces snow depth visualization output."""
        from pathlib import Path
        import subprocess
        import sys

        example_path = Path(__file__).parent.parent / "examples" / "detroit_snow_sledding.py"

        # Run example with output directory
        result = subprocess.run(
            [
                sys.executable,
                str(example_path),
                "--output-dir",
                str(self.output_dir),
                "--step",
                "snow",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Check success
        assert result.returncode == 0, f"Example failed: {result.stderr}"

        # Check snow depth visualization was created
        snow_viz = self.output_dir / "detroit_snow_depth.png"
        assert snow_viz.exists(), "Snow depth visualization not created"
        assert snow_viz.stat().st_size > 0, "Snow depth visualization is empty"

    def test_example_produces_sledding_score_visualization(self):
        """Test that example produces sledding score visualization output."""
        from pathlib import Path
        import subprocess
        import sys

        example_path = Path(__file__).parent.parent / "examples" / "detroit_snow_sledding.py"

        # Run example with output directory
        result = subprocess.run(
            [
                sys.executable,
                str(example_path),
                "--output-dir",
                str(self.output_dir),
                "--step",
                "score",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Check success
        assert result.returncode == 0, f"Example failed: {result.stderr}"

        # Check sledding score visualization was created
        score_viz = self.output_dir / "detroit_sledding_score.png"
        assert score_viz.exists(), "Sledding score visualization not created"
        assert score_viz.stat().st_size > 0, "Sledding score visualization is empty"

    @pytest.mark.skipif(
        "bpy" not in __import__("sys").modules, reason="Blender (bpy) not available"
    )
    def test_example_produces_3d_render(self):
        """Test that example produces 3D render with snow overlay."""
        from pathlib import Path
        import subprocess
        import sys

        example_path = Path(__file__).parent.parent / "examples" / "detroit_snow_sledding.py"

        # Run example with output directory
        result = subprocess.run(
            [
                sys.executable,
                str(example_path),
                "--output-dir",
                str(self.output_dir),
                "--step",
                "render",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Check success
        assert result.returncode == 0, f"Example failed: {result.stderr}"

        # Check 3D render was created
        render_output = self.output_dir / "detroit_snow_3d.png"
        assert render_output.exists(), "3D render not created"
        assert render_output.stat().st_size > 0, "3D render is empty"

    def test_example_produces_all_outputs(self):
        """Test that example can produce all outputs in one run."""
        from pathlib import Path
        import subprocess
        import sys

        example_path = Path(__file__).parent.parent / "examples" / "detroit_snow_sledding.py"

        # Run example with output directory and all steps
        result = subprocess.run(
            [
                sys.executable,
                str(example_path),
                "--output-dir",
                str(self.output_dir),
                "--all-steps",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )

        # Check success
        assert result.returncode == 0, f"Example failed: {result.stderr}"

        # Check all visualizations were created
        expected_outputs = [
            "detroit_dem.png",
            "detroit_snow_depth.png",
            "detroit_sledding_score.png",
        ]

        for output_file in expected_outputs:
            output_path = self.output_dir / output_file
            assert output_path.exists(), f"Output not created: {output_file}"
            assert output_path.stat().st_size > 0, f"Output is empty: {output_file}"

    def test_example_accepts_mock_data_flag(self):
        """Test that example can run with mock data for testing."""
        from pathlib import Path
        import subprocess
        import sys

        example_path = Path(__file__).parent.parent / "examples" / "detroit_snow_sledding.py"

        # Run example with mock data flag
        result = subprocess.run(
            [
                sys.executable,
                str(example_path),
                "--output-dir",
                str(self.output_dir),
                "--mock-data",
                "--step",
                "dem",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Check success
        assert result.returncode == 0, f"Example with mock data failed: {result.stderr}"

        # Check output was created
        dem_viz = self.output_dir / "detroit_dem.png"
        assert dem_viz.exists(), "DEM visualization not created with mock data"

    def test_example_help_message(self):
        """Test that example provides helpful usage information."""
        from pathlib import Path
        import subprocess
        import sys

        example_path = Path(__file__).parent.parent / "examples" / "detroit_snow_sledding.py"

        # Run example with --help
        result = subprocess.run(
            [sys.executable, str(example_path), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Check help message
        assert result.returncode == 0
        assert "--output-dir" in result.stdout
        assert "--step" in result.stdout
        assert "--mock-data" in result.stdout
        assert "detroit" in result.stdout.lower() or "snow" in result.stdout.lower()
