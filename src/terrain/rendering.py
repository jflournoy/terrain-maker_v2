"""
Rendering operations for Blender terrain visualization.

This module contains functions for configuring Blender render settings
and executing scene rendering.
"""

import logging
import time
from pathlib import Path

import bpy

logger = logging.getLogger(__name__)


# =============================================================================
# RENDER PROGRESS TRACKING
# =============================================================================

class RenderProgressTracker:
    """Track and report render progress for tiled rendering.

    Uses Blender's render handlers to provide progress updates during
    long-running renders. Particularly useful with auto-tiling enabled.
    """

    def __init__(self, log: logging.Logger = None):
        self.log = log or logger
        self.start_time = None
        self.last_update = 0
        self.update_interval = 5.0  # Seconds between progress updates
        self._handlers_registered = False

    def _on_render_init(self, scene, depsgraph=None):
        """Called when render initializes.

        Note: Blender 4.0+ passes depsgraph as second argument.
        """
        self.start_time = time.time()
        self.last_update = 0
        self.log.info("Render starting...")

    def _on_render_pre(self, scene, depsgraph=None):
        """Called before each render pass."""
        pass

    def _on_render_post(self, scene, depsgraph=None):
        """Called after each render pass."""
        pass

    def _on_render_stats(self, scene, depsgraph=None):
        """Called periodically with render statistics.

        Note: This is called frequently during render. We throttle updates
        to avoid log spam. Blender 4.0+ passes depsgraph as second argument.
        """
        if self.start_time is None:
            return

        elapsed = time.time() - self.start_time

        # Only log every update_interval seconds
        if elapsed - self.last_update >= self.update_interval:
            self.last_update = elapsed

            # Get render progress info from Blender
            try:
                # Try to get progress from render result
                cycles = scene.cycles

                # Build progress message
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)

                if cycles.use_auto_tile:
                    self.log.info(f"  Rendering... [{minutes:02d}:{seconds:02d}] (tiled mode)")
                else:
                    self.log.info(f"  Rendering... [{minutes:02d}:{seconds:02d}]")
            except Exception:
                pass  # Silently ignore errors in progress reporting

    def _on_render_complete(self, scene, depsgraph=None):
        """Called when render completes successfully."""
        if self.start_time:
            elapsed = time.time() - self.start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.log.info(f"  Render completed in {minutes:02d}:{seconds:02d}")

    def _on_render_cancel(self, scene, depsgraph=None):
        """Called when render is cancelled."""
        self.log.warning("  Render cancelled")

    def register(self):
        """Register render progress handlers."""
        if self._handlers_registered:
            return

        bpy.app.handlers.render_init.append(self._on_render_init)
        bpy.app.handlers.render_pre.append(self._on_render_pre)
        bpy.app.handlers.render_post.append(self._on_render_post)
        bpy.app.handlers.render_complete.append(self._on_render_complete)
        bpy.app.handlers.render_cancel.append(self._on_render_cancel)

        # render_stats may not exist in all Blender versions
        if hasattr(bpy.app.handlers, 'render_stats'):
            bpy.app.handlers.render_stats.append(self._on_render_stats)

        self._handlers_registered = True

    def unregister(self):
        """Unregister render progress handlers."""
        if not self._handlers_registered:
            return

        handlers_to_check = [
            (bpy.app.handlers.render_init, self._on_render_init),
            (bpy.app.handlers.render_pre, self._on_render_pre),
            (bpy.app.handlers.render_post, self._on_render_post),
            (bpy.app.handlers.render_complete, self._on_render_complete),
            (bpy.app.handlers.render_cancel, self._on_render_cancel),
        ]

        if hasattr(bpy.app.handlers, 'render_stats'):
            handlers_to_check.append(
                (bpy.app.handlers.render_stats, self._on_render_stats)
            )

        for handler_list, handler_func in handlers_to_check:
            if handler_func in handler_list:
                handler_list.remove(handler_func)

        self._handlers_registered = False


# Global progress tracker instance
_progress_tracker = None


def setup_render_settings(
    use_gpu: bool = True,
    samples: int = 128,
    preview_samples: int = 32,
    use_denoising: bool = True,
    denoiser: str = "OPTIX",
    compute_device: str = "OPTIX",
    use_ambient_occlusion: bool = True,
    ao_distance: float = 1.0,
    ao_factor: float = 1.0,
    use_persistent_data: bool = False,
    use_auto_tile: bool = False,
    tile_size: int = 2048,
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
        use_ambient_occlusion: Enable ambient occlusion (darkens crevices)
        ao_distance: AO sampling distance (default: 1.0 Blender units)
        ao_factor: AO strength multiplier (default: 1.0)
        use_persistent_data: Keep scene data in memory between frames (default: False)
        use_auto_tile: Enable automatic tiling for large renders (default: False).
            Splits large images into smaller GPU-friendly tiles to reduce VRAM usage.
            Essential for print-quality renders (3000x2400+ pixels).
        tile_size: Tile size in pixels when auto_tile is enabled (default: 2048).
            Smaller tiles = less VRAM but slower rendering. Try 512-1024 for limited VRAM.
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

    # Configure memory-saving options for large renders
    if use_persistent_data:
        logger.info("Enabling persistent data (keeps scene in memory)")
        scene.render.use_persistent_data = True

    if use_auto_tile:
        logger.info(f"Enabling auto-tiling (tile size: {tile_size}px)")
        cycles.use_auto_tile = True
        cycles.tile_size = tile_size

    # Configure ambient occlusion
    # In Cycles, AO is achieved through path tracing. Fast GI can add AO-like effects.
    if use_ambient_occlusion:
        try:
            # Enable fast GI (includes AO-like effects via approximation)
            cycles.use_fast_gi = True
            cycles.fast_gi_method = "REPLACE"  # Replace bounces with fast approximation
            cycles.ao_bounces = 2  # Number of bounces to approximate with AO
            cycles.ao_bounces_render = 2
            logger.info("Fast GI with AO approximation enabled")
        except AttributeError:
            # Fallback for older Blender versions - AO is natural in Cycles path tracing
            logger.info("Ambient occlusion: using natural Cycles path tracing")

    logger.info("Render settings configured successfully")


def render_scene_to_file(
    output_path,
    width=1920,
    height=1440,
    file_format="PNG",
    color_mode="RGBA",
    compression=90,
    save_blend_file=True,
    show_progress=True,
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
        show_progress (bool): Show render progress updates (default: True).
            Logs elapsed time every 5 seconds during rendering.

    Returns:
        Path: Path to rendered file if successful, None otherwise
    """
    global _progress_tracker

    try:
        import bpy
    except ImportError:
        logger.warning("Blender/bpy not available - skipping render")
        return None

    output_path = Path(output_path).resolve()
    logger.info(f"Rendering scene to {output_path}")

    # Set up progress tracking if requested
    if show_progress:
        if _progress_tracker is None:
            _progress_tracker = RenderProgressTracker(logger)
        _progress_tracker.register()

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

        # Log tile info if auto-tiling is enabled
        cycles = bpy.context.scene.cycles
        if cycles.use_auto_tile:
            tile_size = cycles.tile_size
            tiles_x = (width + tile_size - 1) // tile_size
            tiles_y = (height + tile_size - 1) // tile_size
            total_tiles = tiles_x * tiles_y
            logger.info(f"Render: {width}×{height} {file_format} ({color_mode})")
            logger.info(f"  Tiles: {tiles_x}×{tiles_y} = {total_tiles} tiles @ {tile_size}px")
        else:
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

    finally:
        # Clean up progress tracker
        if show_progress and _progress_tracker is not None:
            _progress_tracker.unregister()


def get_render_settings_report() -> dict:
    """
    Query Blender for the actual render settings used.

    Returns a dictionary of all render-relevant settings, useful for
    debugging, reproducibility, and verification.

    Returns:
        dict: Dictionary containing all render settings from Blender
    """
    scene = bpy.context.scene
    render = scene.render
    cycles = scene.cycles
    view = scene.view_settings

    # Get compute device info
    device_info = "CPU"
    device_list = []
    try:
        prefs = bpy.context.preferences
        cprefs = prefs.addons["cycles"].preferences
        device_info = f"{cycles.device} ({cprefs.compute_device_type})"
        device_list = [d.name for d in cprefs.devices if d.use]
    except Exception:
        pass

    # Get world settings
    world_info = {}
    if scene.world and scene.world.use_nodes:
        nodes = scene.world.node_tree.nodes
        world_info["has_nodes"] = True
        world_info["node_types"] = [n.type for n in nodes]

        # Check for sky texture
        for node in nodes:
            if node.type == "TEX_SKY":
                world_info["sky_type"] = node.sky_type
                if hasattr(node, "sun_elevation"):
                    from math import degrees
                    world_info["sun_elevation"] = f"{degrees(node.sun_elevation):.1f}°"
                    world_info["sun_rotation"] = f"{degrees(node.sun_rotation):.1f}°"
                if hasattr(node, "sun_size"):
                    world_info["sun_size"] = f"{degrees(node.sun_size):.2f}°"
            if node.type == "VOLUME_PRINCIPLED":
                world_info["atmosphere_density"] = node.inputs["Density"].default_value

    # Get Fast GI / AO settings
    ao_info = {}
    try:
        ao_info["fast_gi"] = cycles.use_fast_gi
        if cycles.use_fast_gi:
            ao_info["fast_gi_method"] = cycles.fast_gi_method
            ao_info["ao_bounces"] = cycles.ao_bounces
    except AttributeError:
        ao_info["fast_gi"] = "Not available"

    # Get light info
    lights = []
    for obj in bpy.data.objects:
        if obj.type == "LIGHT":
            light_data = obj.data
            lights.append({
                "name": obj.name,
                "type": light_data.type,
                "energy": light_data.energy,
                "angle": f"{light_data.angle:.2f} rad" if hasattr(light_data, "angle") else "N/A",
            })

    # Get camera info
    camera_info = {}
    if scene.camera:
        cam = scene.camera
        cam_data = cam.data
        camera_info = {
            "name": cam.name,
            "type": cam_data.type,
            "location": f"({cam.location.x:.2f}, {cam.location.y:.2f}, {cam.location.z:.2f})",
        }
        if cam_data.type == "ORTHO":
            camera_info["ortho_scale"] = cam_data.ortho_scale
        else:
            camera_info["focal_length"] = f"{cam_data.lens}mm"

    return {
        "engine": render.engine,
        "resolution": f"{render.resolution_x}×{render.resolution_y}",
        "resolution_percentage": f"{render.resolution_percentage}%",
        "samples": cycles.samples,
        "preview_samples": cycles.preview_samples,
        "device": device_info,
        "active_devices": device_list,
        "denoising": {
            "enabled": cycles.use_denoising,
            "denoiser": cycles.denoiser if cycles.use_denoising else "N/A",
        },
        "adaptive_sampling": cycles.use_adaptive_sampling,
        "bounces": {
            "max": cycles.max_bounces,
            "transparent": cycles.transparent_max_bounces,
            "transmission": cycles.transmission_bounces,
            "volume": cycles.volume_bounces,
        },
        "ambient_occlusion": ao_info,
        "color_management": {
            "display_device": scene.display_settings.display_device,
            "view_transform": view.view_transform,
            "look": view.look,
            "exposure": view.exposure,
            "gamma": view.gamma,
        },
        "output": {
            "format": render.image_settings.file_format,
            "color_mode": render.image_settings.color_mode,
            "color_depth": render.image_settings.color_depth,
            "film_transparent": render.film_transparent,
        },
        "world": world_info,
        "camera": camera_info,
        "lights": lights,
    }


def print_render_settings_report(log: logging.Logger = None) -> None:
    """
    Print a formatted report of all Blender render settings.

    Queries Blender for actual settings and prints them in a readable format.
    Useful for debugging and ensuring settings are correctly applied.

    Args:
        log: Logger to use (defaults to module logger)
    """
    if log is None:
        log = logger

    settings = get_render_settings_report()

    log.info("")
    log.info("=" * 70)
    log.info("BLENDER RENDER SETTINGS REPORT")
    log.info("=" * 70)

    # Core render settings
    log.info(f"Engine: {settings['engine']}")
    log.info(f"Resolution: {settings['resolution']} @ {settings['resolution_percentage']}")
    log.info(f"Samples: {settings['samples']} (preview: {settings['preview_samples']})")
    log.info(f"Device: {settings['device']}")
    if settings['active_devices']:
        log.info(f"  Active: {', '.join(settings['active_devices'])}")

    # Denoising
    denoise = settings['denoising']
    if denoise['enabled']:
        log.info(f"Denoising: {denoise['denoiser']}")
    else:
        log.info("Denoising: Disabled")

    # Adaptive sampling
    log.info(f"Adaptive Sampling: {'Enabled' if settings['adaptive_sampling'] else 'Disabled'}")

    # Bounces
    bounces = settings['bounces']
    log.info(f"Bounces: max={bounces['max']}, transparent={bounces['transparent']}, "
             f"transmission={bounces['transmission']}, volume={bounces['volume']}")

    # AO
    ao = settings['ambient_occlusion']
    if ao.get('fast_gi'):
        log.info(f"Fast GI: {ao.get('fast_gi_method', 'Unknown')} (AO bounces: {ao.get('ao_bounces', '?')})")
    else:
        log.info("Fast GI: Disabled")

    # Color management
    cm = settings['color_management']
    log.info(f"Color Management: {cm['display_device']} / {cm['view_transform']} "
             f"(exposure={cm['exposure']}, gamma={cm['gamma']})")

    # Output
    out = settings['output']
    log.info(f"Output: {out['format']} {out['color_mode']} {out['color_depth']}-bit "
             f"(transparent={out['film_transparent']})")

    # World
    world = settings['world']
    if world:
        log.info("World:")
        if 'sky_type' in world:
            log.info(f"  Sky: {world['sky_type']}")
            if 'sun_elevation' in world:
                sun_info = f"elevation={world['sun_elevation']}, rotation={world['sun_rotation']}"
                if 'sun_size' in world:
                    sun_info += f", size={world['sun_size']}"
                log.info(f"  Sun: {sun_info}")
        if 'atmosphere_density' in world:
            log.info(f"  Atmosphere: density={world['atmosphere_density']:.4f}")
        if 'node_types' in world:
            log.info(f"  Nodes: {', '.join(world['node_types'])}")

    # Camera
    cam = settings['camera']
    if cam:
        log.info(f"Camera: {cam['name']} ({cam['type']}) at {cam['location']}")
        if 'ortho_scale' in cam:
            log.info(f"  Ortho scale: {cam['ortho_scale']}")
        elif 'focal_length' in cam:
            log.info(f"  Focal length: {cam['focal_length']}")

    # Lights
    lights = settings['lights']
    if lights:
        log.info(f"Lights: {len(lights)}")
        for light in lights:
            log.info(f"  {light['name']}: {light['type']} energy={light['energy']} angle={light['angle']}")

    log.info("=" * 70)
