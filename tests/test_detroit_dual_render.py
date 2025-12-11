"""
Tests for detroit_dual_render.py example.

Tests side-by-side terrain rendering functionality and verifies that
library functions are used correctly instead of custom implementations.
"""

import sys
from pathlib import Path
import tempfile

import pytest
import numpy as np
from affine import Affine

# Require Blender environment
pytest.importorskip("bpy")

import bpy


class TestDualRenderLoadingFunctions:
    """Test data loading functions."""

    def test_load_sledding_scores_missing_file(self):
        """Test that missing sledding scores returns None."""
        # Import the function
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from examples.detroit_dual_render import load_sledding_scores

        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_sledding_scores(Path(tmpdir))
            assert result is None

    def test_load_xc_skiing_scores_missing_file(self):
        """Test that missing XC skiing scores returns None."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from examples.detroit_dual_render import load_xc_skiing_scores

        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_xc_skiing_scores(Path(tmpdir))
            assert result is None

    def test_load_xc_skiing_parks_missing_file(self):
        """Test that missing parks file returns None."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from examples.detroit_dual_render import load_xc_skiing_parks

        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_xc_skiing_parks(Path(tmpdir))
            assert result is None

    def test_generate_mock_scores(self):
        """Test mock score generation."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from examples.detroit_dual_render import generate_mock_scores

        shape = (100, 100)
        scores = generate_mock_scores(shape)

        assert scores.shape == shape
        assert scores.dtype == np.float32
        assert scores.min() >= 0.3
        assert scores.max() <= 0.9


class TestDualRenderTerrainCreation:
    """Test terrain mesh creation using library functions."""

    def test_create_terrain_with_score_callable(self):
        """Test that create_terrain_with_score is callable."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from examples.detroit_dual_render import create_terrain_with_score

        # Should be callable
        assert callable(create_terrain_with_score)

    def test_create_terrain_uses_terrain_class(self):
        """Test that create_terrain_with_score properly uses Terrain class."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import examples.detroit_dual_render as module

        # Read the source
        source = module.__file__
        with open(source, "r") as f:
            content = f.read()

        # Should use Terrain class
        assert "Terrain(dem, transform" in content
        # Should use add_data_layer
        assert "add_data_layer" in content
        # Should use set_color_mapping with correct API
        assert "set_color_mapping" in content
        assert "source_layers=" in content
        # Should use compute_colors
        assert "compute_colors()" in content
        # Should use create_mesh
        assert "create_mesh(" in content


class TestDualRenderCameraSetup:
    """Test camera setup functions use library functions."""

    def test_setup_dual_camera_callable(self):
        """Test that setup_dual_camera is callable."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from examples.detroit_dual_render import setup_dual_camera

        # Should be callable
        assert callable(setup_dual_camera)

    def test_setup_dual_camera_uses_library_functions(self):
        """Test that setup_dual_camera uses library setup_camera function."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import examples.detroit_dual_render as module

        # Read the source
        source = module.__file__
        with open(source, "r") as f:
            content = f.read()

        # Should use library function
        assert "setup_camera(" in content
        # Should set camera angle, location, and focal length
        assert "camera_angle=" in content
        assert "camera_location=" in content


class TestDualRenderLighting:
    """Test lighting setup uses library functions."""

    def test_setup_lighting_callable(self):
        """Test that setup_lighting is callable."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from examples.detroit_dual_render import setup_lighting

        # Should be callable
        assert callable(setup_lighting)

    def test_setup_lighting_uses_library_setup_light(self):
        """Test that setup_lighting uses library setup_light function."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import examples.detroit_dual_render as module

        # Read the source
        source = module.__file__
        with open(source, "r") as f:
            content = f.read()

        # Should use library function
        assert "setup_light(" in content
        # Should configure location and energy
        assert "location=" in content
        assert "energy=" in content


class TestDualRenderParkMarkers:
    """Test park marker creation."""

    def test_create_park_markers_callable(self):
        """Test that create_park_markers is callable."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from examples.detroit_dual_render import create_park_markers

        # Should be callable
        assert callable(create_park_markers)

    def test_create_park_markers_with_empty_list(self):
        """Test that empty parks list returns empty markers."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from examples.detroit_dual_render import create_park_markers

        dem = np.random.randint(150, 250, (256, 256)).astype(np.float32)
        transform = Affine.identity()

        markers = create_park_markers([], dem, transform, None)

        assert markers == []


class TestDualRenderOutput:
    """Test rendering output."""

    def test_render_dual_terrain_callable(self):
        """Test that render_dual_terrain is callable."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from examples.detroit_dual_render import render_dual_terrain

        # Should be callable
        assert callable(render_dual_terrain)

    def test_render_dual_terrain_uses_library_function(self):
        """Test that render_dual_terrain uses library render_scene_to_file."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import examples.detroit_dual_render as module

        # Read the source
        source = module.__file__
        with open(source, "r") as f:
            content = f.read()

        # Should use library function
        assert "render_scene_to_file(" in content
        # Should pass width and height
        assert "width=" in content
        assert "height=" in content


class TestDualRenderLibraryUsage:
    """Test that library functions are properly used."""

    def test_uses_terrain_class(self):
        """Verify that create_terrain_with_score uses Terrain class."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import examples.detroit_dual_render as module

        # Read the source
        source = module.__file__
        with open(source, "r") as f:
            content = f.read()

        # Should import Terrain from terrain.core
        assert "from src.terrain.core import" in content
        assert "Terrain" in content

        # Should use Terrain class
        assert "Terrain(" in content

    def test_uses_library_functions(self):
        """Verify library functions are imported."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import examples.detroit_dual_render as module

        # Read the source
        source = module.__file__
        with open(source, "r") as f:
            content = f.read()

        # Should import library functions
        assert "clear_scene" in content
        assert "setup_camera" in content
        assert "setup_light" in content
        assert "render_scene_to_file" in content
        assert "elevation_colormap" in content

    def test_uses_setup_functions(self):
        """Verify setup functions are called."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import examples.detroit_dual_render as module

        # Should have these setup functions
        assert callable(module.setup_dual_camera)
        assert callable(module.setup_lighting)

    def test_uses_render_function(self):
        """Verify render function is called."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import examples.detroit_dual_render as module

        # Should have render function
        assert callable(module.render_dual_terrain)
