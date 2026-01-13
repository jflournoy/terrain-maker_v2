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


class TestGPUMemoryErrorDetection:
    """Tests for GPU memory error detection in render retry logic."""

    def test_is_gpu_memory_error_detects_cuda_out_of_memory(self):
        """Test detection of CUDA out of memory errors."""
        from src.terrain.rendering import _is_gpu_memory_error

        error = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        assert _is_gpu_memory_error(error) is True

    def test_is_gpu_memory_error_detects_generic_out_of_memory(self):
        """Test detection of generic out of memory errors."""
        from src.terrain.rendering import _is_gpu_memory_error

        error = RuntimeError("Out of memory allocating render buffer")
        assert _is_gpu_memory_error(error) is True

    def test_is_gpu_memory_error_detects_gpu_memory(self):
        """Test detection of GPU memory errors."""
        from src.terrain.rendering import _is_gpu_memory_error

        error = RuntimeError("GPU memory exhausted during rendering")
        assert _is_gpu_memory_error(error) is True

    def test_is_gpu_memory_error_detects_vram(self):
        """Test detection of VRAM errors."""
        from src.terrain.rendering import _is_gpu_memory_error

        error = RuntimeError("Insufficient VRAM for tile size")
        assert _is_gpu_memory_error(error) is True

    def test_is_gpu_memory_error_detects_cuda_error(self):
        """Test detection of general CUDA errors."""
        from src.terrain.rendering import _is_gpu_memory_error

        error = RuntimeError("CUDA error: device-side assert triggered")
        assert _is_gpu_memory_error(error) is True

    def test_is_gpu_memory_error_detects_optix(self):
        """Test detection of OptiX denoiser errors."""
        from src.terrain.rendering import _is_gpu_memory_error

        error = RuntimeError("OptiX denoiser failed to allocate memory")
        assert _is_gpu_memory_error(error) is True

    def test_is_gpu_memory_error_ignores_other_errors(self):
        """Test that non-GPU errors are not flagged."""
        from src.terrain.rendering import _is_gpu_memory_error

        error = RuntimeError("File not found: scene.blend")
        assert _is_gpu_memory_error(error) is False

    def test_is_gpu_memory_error_ignores_network_errors(self):
        """Test that network errors are not flagged as GPU errors."""
        from src.terrain.rendering import _is_gpu_memory_error

        error = RuntimeError("Connection timeout reading asset")
        assert _is_gpu_memory_error(error) is False

    def test_is_gpu_memory_error_case_insensitive(self):
        """Test that error detection is case insensitive."""
        from src.terrain.rendering import _is_gpu_memory_error

        error = RuntimeError("cuda OUT OF MEMORY error occurred")
        assert _is_gpu_memory_error(error) is True


class TestRenderRetryParameters:
    """Tests for render retry parameters."""

    def test_render_scene_to_file_accepts_max_retries(self):
        """Test that max_retries parameter is accepted."""
        from src.terrain.rendering import render_scene_to_file
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.png"
            # Should not raise error when max_retries is passed
            result = render_scene_to_file(output_path, max_retries=0, save_blend_file=False)
            assert result is None or isinstance(result, Path)

    def test_render_scene_to_file_accepts_retry_delay(self):
        """Test that retry_delay parameter is accepted."""
        from src.terrain.rendering import render_scene_to_file
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.png"
            # Should not raise error when retry_delay is passed
            result = render_scene_to_file(
                output_path, retry_delay=1.0, max_retries=0, save_blend_file=False
            )
            assert result is None or isinstance(result, Path)

    def test_render_scene_to_file_default_max_retries(self):
        """Test that default max_retries is 3."""
        import inspect
        from src.terrain.rendering import render_scene_to_file

        sig = inspect.signature(render_scene_to_file)
        assert sig.parameters["max_retries"].default == 3

    def test_render_scene_to_file_default_retry_delay(self):
        """Test that default retry_delay is 5.0 seconds."""
        import inspect
        from src.terrain.rendering import render_scene_to_file

        sig = inspect.signature(render_scene_to_file)
        assert sig.parameters["retry_delay"].default == 5.0
