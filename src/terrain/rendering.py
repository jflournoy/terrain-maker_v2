"""
Rendering operations for Blender terrain visualization.

This module contains functions for configuring Blender render settings
and executing scene rendering.
"""

import logging
from pathlib import Path

import bpy

logger = logging.getLogger(__name__)


def setup_render_settings(
    use_gpu: bool = True,
    samples: int = 128,
    preview_samples: int = 32,
    use_denoising: bool = True,
    denoiser: str = "OPTIX",
    compute_device: str = "OPTIX",
) -> None:
    """
    Configure Blender render settings for high-quality terrain visualization.

    Args:
        use_gpu: Whether to use GPU acceleration
        samples: Number of render samples
        preview_samples: Number of viewport preview samples
        use_denoising: Whether to enable denoising
        denoiser: Type of denoiser to use ('OPTIX', 'OPENIMAGEDENOISE', 'NLM')
        compute_device: Compute device type ('OPTIX', 'CUDA', 'HIP', 'METAL')
    """
    logger.info("Configuring render settings...")

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"

    # Configure color management for sRGB
    logger.info("Setting up color management...")
    scene.view_settings.view_transform = "Standard"  # sRGB in Blender 4.3
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    scene.display_settings.display_device = "sRGB"

    # Configure render output settings
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = "16"
    scene.render.film_transparent = False

    logger.info("Configuring Cycles settings...")
    # Configure Cycles render settings
    cycles = scene.cycles
    cycles.max_bounces = 32
    cycles.transparent_max_bounces = 32
    cycles.transmission_bounces = 32
    cycles.volume_bounces = 2
    cycles.volume_step_rate = 5.0
    cycles.volume_max_steps = 32
    cycles.caustics_reflective = False
    cycles.caustics_refractive = False
    cycles.sample_clamp_indirect = 0.0

    # Configure sampling settings
    cycles.samples = samples
    cycles.preview_samples = preview_samples
    cycles.use_denoising = use_denoising
    cycles.denoiser = denoiser
    cycles.use_adaptive_sampling = True

    # Configure GPU settings if requested
    if use_gpu:
        logger.info(f"Setting up GPU rendering with {compute_device}...")
        try:
            cycles.device = "GPU"
            prefs = bpy.context.preferences
            cprefs = prefs.addons["cycles"].preferences
            cprefs.compute_device_type = compute_device

            # Enable all available devices
            device_count = 0
            for device in cprefs.devices:
                device.use = True
                device_count += 1
            logger.info(f"Enabled {device_count} compute devices")

        except Exception as e:
            logger.error(f"Failed to configure GPU rendering: {str(e)}")
            logger.warning("Falling back to CPU rendering")
            cycles.device = "CPU"

    logger.info("Render settings configured successfully")


def render_scene_to_file(
    output_path,
    width=1920,
    height=1440,
    file_format="PNG",
    color_mode="RGBA",
    compression=90,
    save_blend_file=True,
):
    """
    Render the current Blender scene to file.

    Args:
        output_path (str or Path): Path where output file will be saved
        width (int): Render width in pixels (default: 1920)
        height (int): Render height in pixels (default: 1440)
        file_format (str): Output format 'PNG', 'JPEG', etc. (default: 'PNG')
        color_mode (str): 'RGBA' or 'RGB' (default: 'RGBA')
        compression (int): PNG compression level 0-100 (default: 90)
        save_blend_file (bool): Also save .blend project file (default: True)

    Returns:
        Path: Path to rendered file if successful, None otherwise
    """
    try:
        import bpy
    except ImportError:
        logger.warning("Blender/bpy not available - skipping render")
        return None

    output_path = Path(output_path).resolve()
    logger.info(f"Rendering scene to {output_path}")

    try:
        # Configure render output
        bpy.context.scene.render.filepath = str(output_path)
        bpy.context.scene.render.image_settings.file_format = file_format
        bpy.context.scene.render.image_settings.color_mode = color_mode

        if file_format == "PNG":
            bpy.context.scene.render.image_settings.compression = compression

        bpy.context.scene.render.resolution_x = width
        bpy.context.scene.render.resolution_y = height
        bpy.context.scene.render.resolution_percentage = 100

        logger.info(f"Render: {width}×{height} {file_format} ({color_mode})")

        # Execute render
        bpy.ops.render.render(write_still=True)

        # Verify output
        if output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"Rendered successfully: {file_size_mb:.1f} MB")

            # Save Blender file if requested
            if save_blend_file:
                blend_path = output_path.parent / output_path.stem
                blend_path = blend_path.with_suffix(".blend")
                try:
                    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
                    logger.info(f"Saved Blender file: {blend_path.name}")
                except Exception as e:
                    logger.warning(f"Could not save Blender file: {e}")

            return output_path
        else:
            logger.error("Render file was not created")
            return None

    except Exception as e:
        logger.error(f"Render failed: {str(e)}")
        return None
