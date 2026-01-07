"""
Tests for rendering operations.

Tests Blender render configuration and scene rendering functions.
"""

import pytest
from pathlib import Path

# These tests require Blender environment
pytest.importorskip("bpy")


class TestSetupRenderSettings:
    """Tests for setup_render_settings function."""

    def test_setup_render_settings_imports(self):
        """Test that setup_render_settings can be imported."""
        from src.terrain.rendering import setup_render_settings

        assert callable(setup_render_settings)

    def test_setup_render_settings_sets_cycles_engine(self):
        """Test that Cycles render engine is configured."""
        import bpy
        from src.terrain.rendering import setup_render_settings

        setup_render_settings()

        # Should set Cycles as render engine
        assert bpy.context.scene.render.engine == "CYCLES"

    def test_setup_render_settings_configures_samples(self):
        """Test that render samples are configured."""
        import bpy
        from src.terrain.rendering import setup_render_settings

        custom_samples = 256
        setup_render_settings(samples=custom_samples)

        assert bpy.context.scene.cycles.samples == custom_samples

    def test_setup_render_settings_configures_denoising(self):
        """Test that denoising is configured."""
        import bpy
        from src.terrain.rendering import setup_render_settings

        setup_render_settings(use_denoising=True)

        assert bpy.context.scene.cycles.use_denoising is True

    def test_setup_render_settings_configures_preview_samples(self):
        """Test that preview samples are configured."""
        import bpy
        from src.terrain.rendering import setup_render_settings

        preview_samples = 64
        setup_render_settings(preview_samples=preview_samples)

        assert bpy.context.scene.cycles.preview_samples == preview_samples

    def test_setup_render_settings_sets_color_management(self):
        """Test that color management is configured for sRGB."""
        import bpy
        from src.terrain.rendering import setup_render_settings

        setup_render_settings()

        scene = bpy.context.scene
        assert scene.view_settings.view_transform == "Standard"
        assert scene.display_settings.display_device == "sRGB"

    def test_setup_render_settings_configures_bounces(self):
        """Test that light bounces are configured."""
        import bpy
        from src.terrain.rendering import setup_render_settings

        setup_render_settings()

        cycles = bpy.context.scene.cycles
        assert cycles.max_bounces == 32
        assert cycles.transparent_max_bounces == 32

    def test_setup_render_settings_cpu_fallback(self):
        """Test that GPU configuration is skipped when use_gpu=False."""
        import bpy
        from src.terrain.rendering import setup_render_settings

        # Should not raise error when use_gpu=False
        # This just skips GPU configuration, doesn't force CPU
        try:
            setup_render_settings(use_gpu=False)
            # Test passes if no exception raised
            assert True
        except Exception:
            assert False, "Should not raise exception with use_gpu=False"


class TestRenderSceneToFile:
    """Tests for render_scene_to_file function."""

    def test_render_scene_to_file_imports(self):
        """Test that render_scene_to_file can be imported."""
        from src.terrain.rendering import render_scene_to_file

        assert callable(render_scene_to_file)

    def test_render_scene_to_file_configures_output_path(self):
        """Test that output path is configured (without actual render)."""
        import bpy
        from src.terrain.rendering import render_scene_to_file
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_render.png"

            # We can't test actual rendering without a scene, but we can test
            # that the function configures the render settings
            # Note: This will fail to render but should set up the path
            try:
                render_scene_to_file(output_path, width=800, height=600, save_blend_file=False)
            except Exception:
                pass  # Expected to fail without proper scene

            # Check that render path was set
            assert str(output_path) in bpy.context.scene.render.filepath

    def test_render_scene_to_file_configures_resolution(self):
        """Test that resolution is configured."""
        import bpy
        from src.terrain.rendering import render_scene_to_file
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_render.png"
            width, height = 1280, 720

            try:
                render_scene_to_file(
                    output_path, width=width, height=height, save_blend_file=False
                )
            except Exception:
                pass  # Expected to fail without proper scene

            # Check resolution was set
            assert bpy.context.scene.render.resolution_x == width
            assert bpy.context.scene.render.resolution_y == height

    def test_render_scene_to_file_configures_format(self):
        """Test that file format is configured."""
        import bpy
        from src.terrain.rendering import render_scene_to_file
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_render.jpg"

            try:
                render_scene_to_file(
                    output_path, file_format="JPEG", color_mode="RGB", save_blend_file=False
                )
            except Exception:
                pass  # Expected to fail without proper scene

            # Check format settings
            assert bpy.context.scene.render.image_settings.file_format == "JPEG"
            assert bpy.context.scene.render.image_settings.color_mode == "RGB"

    def test_render_scene_to_file_png_compression(self):
        """Test that PNG compression is configured."""
        import bpy
        from src.terrain.rendering import render_scene_to_file
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_render.png"
            compression = 50

            try:
                render_scene_to_file(
                    output_path, file_format="PNG", compression=compression, save_blend_file=False
                )
            except Exception:
                pass  # Expected to fail without proper scene

            # Check compression was set
            assert bpy.context.scene.render.image_settings.compression == compression

    def test_render_scene_to_file_returns_path_or_none(self):
        """Test that function returns Path or None."""
        from src.terrain.rendering import render_scene_to_file
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_render.png"

            result = render_scene_to_file(output_path, save_blend_file=False)

            # Should return None or Path
            assert result is None or isinstance(result, Path)


class TestSetupRenderSettingsMemory:
    """Tests for GPU memory-saving options in setup_render_settings (TDD).

    These options help render large images without running out of GPU VRAM.
    """

    def test_setup_render_settings_accepts_persistent_data(self):
        """Test that use_persistent_data parameter is accepted."""
        from src.terrain.rendering import setup_render_settings

        # Should not raise error when persistent_data is passed
        setup_render_settings(use_persistent_data=True)

    def test_setup_render_settings_persistent_data_enabled(self):
        """Test that persistent data is enabled when requested."""
        import bpy
        from src.terrain.rendering import setup_render_settings

        setup_render_settings(use_persistent_data=True)

        # scene.render.use_persistent_data should be True
        assert bpy.context.scene.render.use_persistent_data is True

    def test_setup_render_settings_persistent_data_disabled_by_default(self):
        """Test that persistent data is not enabled by default."""
        import bpy
        from src.terrain.rendering import setup_render_settings

        # Reset to known state
        bpy.context.scene.render.use_persistent_data = False
        setup_render_settings()

        # Default behavior should not change persistent data (keep disabled)
        assert bpy.context.scene.render.use_persistent_data is False

    def test_setup_render_settings_accepts_auto_tile(self):
        """Test that use_auto_tile parameter is accepted."""
        from src.terrain.rendering import setup_render_settings

        # Should not raise error when auto_tile is passed
        setup_render_settings(use_auto_tile=True)

    def test_setup_render_settings_auto_tile_enabled(self):
        """Test that auto tiling is enabled when requested.

        Auto-tiling renders the image in smaller tiles to reduce VRAM usage.
        This is essential for large print-quality renders (3000x2400+).
        """
        import bpy
        from src.terrain.rendering import setup_render_settings

        setup_render_settings(use_auto_tile=True)

        # In Blender 4.0+, this is scene.cycles.use_auto_tile
        assert bpy.context.scene.cycles.use_auto_tile is True

    def test_setup_render_settings_accepts_tile_size(self):
        """Test that tile_size parameter is accepted."""
        from src.terrain.rendering import setup_render_settings

        # Should not raise error when tile_size is passed
        setup_render_settings(use_auto_tile=True, tile_size=1024)

    def test_setup_render_settings_tile_size_applied(self):
        """Test that tile size is applied when auto_tile is enabled.

        Smaller tiles = less VRAM but slower rendering.
        Typical values: 512, 1024, 2048 pixels.
        """
        import bpy
        from src.terrain.rendering import setup_render_settings

        tile_size = 1024
        setup_render_settings(use_auto_tile=True, tile_size=tile_size)

        # Tile size should be applied
        assert bpy.context.scene.cycles.tile_size == tile_size

    def test_setup_render_settings_auto_tile_disabled_by_default(self):
        """Test that auto-tile is not enabled by default.

        Auto-tiling adds overhead for small renders, so it should only
        be enabled when explicitly requested for large images.
        """
        import bpy
        from src.terrain.rendering import setup_render_settings

        # Reset to known state
        bpy.context.scene.cycles.use_auto_tile = False
        setup_render_settings()

        # Default behavior should not enable auto-tiling
        assert bpy.context.scene.cycles.use_auto_tile is False

    def test_setup_render_settings_combined_memory_options(self):
        """Test that all memory-saving options can be combined.

        For large renders, users typically want all memory-saving options:
        - persistent_data: Reuses scene data between frames
        - auto_tile: Splits large image into smaller GPU-friendly tiles
        """
        import bpy
        from src.terrain.rendering import setup_render_settings

        setup_render_settings(
            use_persistent_data=True,
            use_auto_tile=True,
            tile_size=512,  # Smaller tiles for limited VRAM
        )

        # All settings should be applied
        assert bpy.context.scene.render.use_persistent_data is True
        assert bpy.context.scene.cycles.use_auto_tile is True
        assert bpy.context.scene.cycles.tile_size == 512
