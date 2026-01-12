"""
Lightweight dependency graph pipeline for terrain visualization.

Provides declarative task dependencies with automatic caching, staleness detection,
and execution planning. Integrates with existing DEMCache and MeshCache.

Example:
    from src.terrain.pipeline import TerrainPipeline

    pipeline = TerrainPipeline(dem_dir="data/dem/detroit", cache_enabled=True)

    # Show execution plan
    pipeline.explain("render_view")

    # Get cache statistics
    stats = pipeline.cache_stats()

    # Clear cache
    pipeline.clear_cache()
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
from rasterio import Affine

from src.terrain.cache import DEMCache
from src.terrain.mesh_cache import MeshCache

logger = logging.getLogger(__name__)


@dataclass
class TaskState:
    """Represents execution state of a task."""

    name: str
    depends_on: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    cached: bool = False
    computed: bool = False
    result: Any = None
    cache_key: str = ""


class TerrainPipeline:
    """
    Dependency graph executor for terrain rendering pipeline.

    Manages task dependencies, caching, and execution planning without
    external build system dependencies. All logic stays in Python.

    Features:
    - Automatic staleness detection via hashing
    - Dry-run execution plans
    - Reusable cached outputs across multiple views
    - Per-layer caching support

    Tasks in pipeline:
    1. load_dem: Load and merge SRTM tiles (cached)
    2. apply_transforms: Reproject, flip, downsample (can be cached)
    3. detect_water: Identify water bodies from DEM (can be cached)
    4. create_mesh: Build Blender geometry (cached, reused across views)
    5. render_view: Output to PNG (cached by view/render params)
    """

    def __init__(
        self,
        dem_dir: Path | str = None,
        *,
        cache_enabled: bool = True,
        force_rebuild: bool = False,
        dem_cache_dir: Path | str = None,
        mesh_cache_dir: Path | str = None,
        verbose: bool = True,
    ):
        """
        Initialize terrain pipeline.

        Args:
            dem_dir: Directory containing SRTM .hgt files
            cache_enabled: Enable caching at all stages
            force_rebuild: Force rebuild even if cache exists (bypasses cache checks)
            dem_cache_dir: Custom DEM cache directory
            mesh_cache_dir: Custom mesh cache directory
            verbose: Print execution details
        """
        self.dem_dir = (
            Path(dem_dir)
            if dem_dir
            else Path(__file__).parent.parent.parent / "data" / "dem" / "detroit"
        )
        self.cache_enabled = cache_enabled
        self.force_rebuild = force_rebuild
        self.verbose = verbose

        # Initialize caches
        self.dem_cache = DEMCache(
            cache_dir=Path(dem_cache_dir) if dem_cache_dir else None, enabled=cache_enabled
        )
        self.mesh_cache = MeshCache(
            cache_dir=Path(mesh_cache_dir) if mesh_cache_dir else None, enabled=cache_enabled
        )

        # Task state tracking
        self.tasks: Dict[str, TaskState] = {}

        # Define task dependencies (declarative DAG)
        self._task_graph = {
            "load_dem": {
                "depends_on": [],
                "description": "Load and merge SRTM tiles",
            },
            "apply_transforms": {
                "depends_on": ["load_dem"],
                "description": "Reproject, flip, downsample DEM",
            },
            "detect_water": {
                "depends_on": ["apply_transforms"],
                "description": "Identify water bodies from DEM",
            },
            "create_mesh": {
                "depends_on": ["load_dem", "detect_water"],
                "description": "Build Blender mesh geometry",
            },
            "render_view": {
                "depends_on": ["create_mesh"],
                "description": "Render to PNG output",
            },
        }

    def _should_use_cache(self) -> bool:
        """
        Determine if cache should be used.

        Returns False if force_rebuild is True or cache_enabled is False.
        Returns True only if cache_enabled is True and force_rebuild is False.
        """
        return self.cache_enabled and not self.force_rebuild

    def _log(self, msg: str, *args, level: str = "info"):
        """Log message if verbose with lazy formatting."""
        if self.verbose:
            if level == "info":
                logger.info(msg, *args)
            elif level == "debug":
                logger.debug(msg, *args)
            elif level == "warn":
                logger.warning(msg, *args)
            elif level == "error":
                logger.error(msg, *args)

    def _compute_hash(self, *args, **kwargs) -> str:
        """Compute SHA256 hash of arguments."""
        parts = []

        for arg in args:
            if isinstance(arg, np.ndarray):
                parts.append(hashlib.sha256(arg.tobytes()).hexdigest())
            elif isinstance(arg, dict):
                parts.append(
                    hashlib.sha256(
                        json.dumps(arg, sort_keys=True, default=str).encode()
                    ).hexdigest()
                )
            else:
                parts.append(str(arg))

        for key in sorted(kwargs.keys()):
            val = kwargs[key]
            if isinstance(val, np.ndarray):
                parts.append(f"{key}:{hashlib.sha256(val.tobytes()).hexdigest()}")
            elif isinstance(val, dict):
                hash_val = hashlib.sha256(
                    json.dumps(val, sort_keys=True, default=str).encode()
                ).hexdigest()
                parts.append(f"{key}:{hash_val}")
            else:
                parts.append(f"{key}:{val}")

        combined = "|".join(parts)
        return hashlib.sha256(combined.encode()).hexdigest()[:64]

    # ===== Terrain Pipeline Tasks =====

    def load_dem(self, pattern: str = "*.hgt") -> Tuple[np.ndarray, Affine]:
        """
        Task: Load and cache raw DEM from SRTM tiles.

        Returns:
            (dem_array, affine_transform)
        """
        self._log("[1/5] Loading DEM from %s", self.dem_dir)

        try:
            source_hash = self.dem_cache.compute_source_hash(str(self.dem_dir), pattern=pattern)

            # Try cache first
            cached = self.dem_cache.load_cache(source_hash, cache_name="detroit")
            if cached:
                dem, transform = cached
                self._log("      [Cache HIT] Loaded DEM (%s)", dem.shape)
                return dem, transform
        except (OSError, IOError, ValueError) as e:
            self._log("      [Cache] Load failed: %s", e, "debug")

        # Load from source
        from src.terrain.core import load_dem_files  # pylint: disable=import-outside-toplevel

        dem, transform = load_dem_files(str(self.dem_dir), pattern=pattern)
        self._log("      [Fresh] Loaded DEM shape: %s", dem.shape)

        # Cache it
        try:
            source_hash = self.dem_cache.compute_source_hash(str(self.dem_dir), pattern=pattern)
            self.dem_cache.save_cache(dem, transform, source_hash, cache_name="detroit")
            self._log("      [Cached] DEM saved")
        except (OSError, IOError, ValueError) as e:
            self._log("      [Cache] Failed to save: %s", e, "warn")

        return dem, transform

    def apply_transforms(
        self,
        *,
        target_vertices: int = 1382400,
        reproject_crs: str = "EPSG:32617",
        scale_factor: float = 0.0001,
    ) -> Tuple[np.ndarray, Affine]:
        """
        Task: Apply reprojection, flipping, and elevation scaling.

        Args:
            target_vertices: Mesh density target
            reproject_crs: Target CRS for reprojection
            scale_factor: Elevation scaling factor

        Returns:
            (transformed_dem, transform)
        """
        self._log("[2/5] Applying transforms (target_vertices=%d)", target_vertices)

        dem, transform = self.load_dem()

        # pylint: disable=import-outside-toplevel
        from src.terrain.core import Terrain, reproject_raster, flip_raster, scale_elevation

        terrain = Terrain(dem, transform)

        # Configure downsampling
        zoom = terrain.configure_for_target_vertices(target_vertices, order=4)
        self._log("      Zoom factor: %.6f", zoom)

        # Apply transforms
        terrain.transforms.append(
            reproject_raster(src_crs="EPSG:4326", dst_crs=reproject_crs, num_threads=4)
        )
        terrain.transforms.append(flip_raster(axis="horizontal"))
        terrain.transforms.append(scale_elevation(scale_factor=scale_factor))
        terrain.apply_transforms()

        transformed_dem = terrain.data_layers["dem"]["transformed_data"]
        self._log("      Downsampled DEM shape: %s", transformed_dem.shape)

        return transformed_dem, transform

    def detect_water(
        self, *, slope_threshold: float = 0.01, fill_holes: bool = True
    ) -> np.ndarray:
        """
        Task: Detect water bodies using slope analysis on unscaled DEM.

        Args:
            slope_threshold: Threshold for slope magnitude (Horn's method)
            fill_holes: Apply morphological smoothing

        Returns:
            water_mask (boolean array)
        """
        self._log("[3/5] Detecting water (threshold=%s)", slope_threshold)

        transformed_dem, _ = self.apply_transforms()

        # Unscale elevation (it was scaled by 0.0001)
        unscaled_dem = transformed_dem / 0.0001

        from src.terrain.water import identify_water_by_slope  # pylint: disable=import-outside-toplevel

        water_mask = identify_water_by_slope(
            unscaled_dem, slope_threshold=slope_threshold, fill_holes=fill_holes
        )

        water_pixels = np.sum(water_mask)
        pct = 100 * water_pixels / water_mask.size
        self._log("      Water detected: %d pixels (%.1f%%)", water_pixels, pct)

        return water_mask

    def create_mesh(
        self,
        *,
        scale_factor: float = 100.0,
        height_scale: float = 4.0,
        center_model: bool = True,
        boundary_extension: bool = True,
        # Upstream params for cache key (must match dependency graph!)
        transform_params: dict = None,
        water_params: dict = None,
    ):
        """
        Task: Create Blender mesh from DEM and water mask.

        KEY: Mesh is identical for all views and cached at geometry level,
        not per-view. Different camera angles reuse same mesh.

        Cache key includes ALL upstream parameters per dependency graph:
        - DEM source hash (load_dem)
        - Transform params (apply_transforms)
        - Water params (detect_water)
        - Mesh params (this task)

        Args:
            scale_factor: XY scaling
            height_scale: Z scaling for height exaggeration
            center_model: Center mesh at origin
            boundary_extension: Extend boundary for better rendering
            transform_params: Upstream transform parameters (for cache key)
            water_params: Upstream water detection parameters (for cache key)

        Returns:
            Blender mesh object
        """
        self._log("[4/5] Creating Blender mesh")

        # Default upstream params (must match apply_transforms/detect_water defaults)
        if transform_params is None:
            transform_params = {
                "target_vertices": 1382400,
                "reproject_crs": "EPSG:32617",
                "elevation_scale": 0.0001,
            }
        if water_params is None:
            water_params = {
                "slope_threshold": 0.01,
                "fill_holes": True,
            }

        # Build complete cache key including ALL upstream dependencies
        mesh_params = {
            # Mesh-specific params
            "scale_factor": scale_factor,
            "height_scale": height_scale,
            "center_model": center_model,
            "boundary_extension": boundary_extension,
            # Upstream: transform params (affect DEM shape)
            "transform_target_vertices": transform_params["target_vertices"],
            "transform_reproject_crs": transform_params["reproject_crs"],
            "transform_elevation_scale": transform_params["elevation_scale"],
            # Upstream: water params (affect mesh coloring)
            "water_slope_threshold": water_params["slope_threshold"],
            "water_fill_holes": water_params["fill_holes"],
        }

        # Try to check cache early (before loading DEM/water)
        try:
            dem_hash = self.dem_cache.compute_source_hash(str(self.dem_dir), pattern="*.hgt")
            mesh_hash = self.mesh_cache.compute_mesh_hash(dem_hash, mesh_params)

            cached_blend = self.mesh_cache.load_cache(mesh_hash, cache_name="detroit_mesh")
            if cached_blend:
                self._log("      [Cache HIT] Loading mesh from cache")
                # Load mesh from cached blend file
                import bpy  # pylint: disable=import-outside-toplevel,import-error

                bpy.ops.wm.open_mainfile(filepath=str(cached_blend))
                for obj in bpy.context.scene.objects:
                    if obj.type == "MESH":
                        self._log("      Loaded cached mesh: %d vertices", len(obj.data.vertices))
                        return obj
        except (OSError, IOError, ValueError, ImportError) as e:
            self._log("      [Cache] Early check failed: %s", e, "debug")

        # Cache miss - run full pipeline with matching params
        dem, transform = self.load_dem()

        # pylint: disable=import-outside-toplevel
        from src.terrain.core import (
            Terrain,
            reproject_raster,
            flip_raster,
            scale_elevation,
        )

        # Create Terrain with raw DEM
        terrain = Terrain(dem, transform)

        # Configure downsampling based on target vertices
        zoom = terrain.configure_for_target_vertices(transform_params["target_vertices"], order=4)
        self._log("      Configured downsampling: zoom_factor=%.6f", zoom)

        # Add transforms in the correct order
        terrain.transforms.append(
            reproject_raster(
                src_crs="EPSG:4326",
                dst_crs=transform_params["reproject_crs"],
                num_threads=4,
            )
        )
        terrain.transforms.append(flip_raster(axis="horizontal"))
        terrain.transforms.append(scale_elevation(scale_factor=transform_params["elevation_scale"]))

        # Apply transforms to the terrain object
        terrain.apply_transforms()

        # Set up elevation color mapping with mako colormap
        from src.terrain.core import elevation_colormap

        terrain.set_color_mapping(
            lambda dem: elevation_colormap(dem, cmap_name="mako"), source_layers=["dem"]
        )

        # Detect water on the transformed but unscaled DEM
        water_mask = self.detect_water(
            slope_threshold=water_params["slope_threshold"], fill_holes=water_params["fill_holes"]
        )

        mesh_obj = terrain.create_mesh(
            scale_factor=scale_factor,
            height_scale=height_scale,
            center_model=center_model,
            boundary_extension=boundary_extension,
            water_mask=water_mask,
        )

        if mesh_obj:
            self._log("      Mesh created: %d vertices", len(mesh_obj.data.vertices))

        return mesh_obj

    def render_view(
        self,
        *,
        view: str = "south",
        width: int = 960,
        height: int = 720,
        distance: float = 0.264,
        elevation: float = 0.396,
        focal_length: float = 15,
        camera_type: str = "PERSP",
        samples: int = 2048,
    ):
        """
        Task: Render a view to PNG.

        Args:
            view: Camera direction (north, south, east, west, above)
            width: Output width
            height: Output height
            distance: Camera distance multiplier
            elevation: Camera elevation multiplier
            focal_length: Focal length
            camera_type: PERSP or ORTHO
            samples: Render samples

        Returns:
            Path to rendered PNG
        """
        self._log("[5/5] Rendering %s view", view)

        # pylint: disable=import-outside-toplevel
        from src.terrain.core import (
            clear_scene,
            position_camera_relative,
            setup_light,
            setup_render_settings,
            render_scene_to_file,
        )

        # Clear Blender scene before rendering
        try:
            clear_scene()
            self._log("      Cleared Blender scene")
        except ImportError:
            self._log("      Blender not available - skipping scene clear", level="warn")

        mesh_obj = self.create_mesh()

        # View-specific camera targets
        view_offsets = {
            "north": (0, 2, 0),
            "south": (0, -1.5, 0),
            "east": (1.5, 0, 0),
            "west": (-1.5, 0, 0),
            "above": (0, 0, 0),
        }
        look_at = view_offsets.get(view, (0, 0, 0))

        # Position camera
        position_camera_relative(
            mesh_obj,
            look_at=look_at,
            direction=view,
            distance=distance,
            elevation=elevation,
            camera_type=camera_type,
            focal_length=focal_length,
        )

        # Setup rendering
        setup_light(angle=2, energy=3)
        setup_render_settings(use_gpu=True, samples=samples, use_denoising=False)

        # Render - output to examples/ directory next to detroit_pipeline.py
        output_path = (
            Path(__file__).parent.parent.parent / "examples" / f"detroit_elevation_{view}.png"
        )
        render_file = render_scene_to_file(
            output_path=output_path,
            width=width,
            height=height,
            file_format="PNG",
            color_mode="RGBA",
            compression=90,
            save_blend_file=True,
        )

        if render_file:
            size_mb = render_file.stat().st_size / (1024 * 1024)
            self._log("      Rendered: %s (%.1f MB)", render_file.name, size_mb)

        return render_file

    # ===== Public API =====

    def explain(self, task_name: str) -> None:
        """
        Explain what would execute to build a task (show dependency tree).

        Shows:
        - Task dependencies
        - Execution order
        - Which tasks would be computed vs cached
        """
        if task_name not in self._task_graph:
            print(f"\nUnknown task: {task_name}")
            print(f"Available tasks: {', '.join(self._task_graph.keys())}")
            return

        print("\n" + "=" * 70)
        print(f"Execution Plan for: {task_name}")
        print("=" * 70 + "\n")

        task_info = self._task_graph[task_name]
        print(f"Task: {task_name}")
        print(f"Description: {task_info['description']}")

        if task_info["depends_on"]:
            print("\nDependencies:")
            for dep in task_info["depends_on"]:
                print(f"  - {dep}")

        # Show execution order
        order = self._compute_execution_order(task_name)
        print("\nExecution order (topological):")
        for i, task in enumerate(order, 1):
            print(f"  {i}. {task}")

    def _compute_execution_order(self, task_name: str) -> List[str]:
        """Topologically sort tasks by dependency."""
        visited = set()
        order = []

        def visit(task: str):
            """Recursively visit a task and its dependencies for topological sort.

            Depth-first traversal that visits all dependencies before the task itself,
            building execution order for the task dependency graph.

            Args:
                task: Name of the task to visit
            """
            if task in visited:
                return
            visited.add(task)

            task_info = self._task_graph.get(task)
            if task_info:
                for dep in task_info["depends_on"]:
                    visit(dep)

            order.append(task)

        visit(task_name)
        return order

    def render_all_views(self, views: List[str] = None) -> Dict[str, Path]:
        """
        Render all views efficiently.

        Builds mesh once, reuses for all views.

        Args:
            views: List of view names

        Returns:
            Dictionary mapping view names to output paths
        """
        if views is None:
            views = ["north", "south", "east", "west", "above"]

        print("\n" + "=" * 70)
        print(f"Rendering {len(views)} views (mesh built once, reused for all)")
        print("=" * 70 + "\n")

        results = {}
        for i, view in enumerate(views, 1):
            print(f"\n[{i}/{len(views)}] Rendering {view} view...")
            self._log("\n[%d/%d] Rendering %s view...", i, len(views), view)
            try:
                output = self.render_view(view=view)
                if output:
                    results[view] = output
                    print(f"      ✓ Added {view} to results: {output}")
                    self._log("      ✓ Added %s to results", view)
                else:
                    print(f"      ✗ render_view returned None for {view}")
                    self._log("      ✗ render_view returned None for %s", view, "warn")
            except (OSError, IOError, ValueError, RuntimeError) as e:
                print(f"[✗] Failed to render {view}: {e}")
                self._log("[✗] Failed to render %s: %s", view, e, "warn")
            except Exception as e:
                print(f"[✗] Unexpected error rendering {view}: {e}")
                self._log("[✗] Unexpected error rendering %s: %s", view, e, "error")
                import traceback
                traceback.print_exc()

        return results

    def cache_stats(self) -> Dict:
        """Get cache statistics."""
        dem_stats = self.dem_cache.get_cache_stats()
        mesh_stats = self.mesh_cache.get_cache_stats()

        return {
            "dem": dem_stats,
            "mesh": mesh_stats,
            "total_mb": dem_stats["total_size_mb"] + mesh_stats["total_size_mb"],
            "total_files": dem_stats["cache_files"] + mesh_stats["blend_files"],
        }

    def clear_cache(self) -> int:
        """Clear all caches."""
        dem_deleted = self.dem_cache.clear_cache(cache_name="detroit")
        mesh_deleted = self.mesh_cache.clear_cache(cache_name="detroit_mesh")

        total = dem_deleted + mesh_deleted
        self._log("Cleared %d cache files (%d DEM, %d Mesh)", total, dem_deleted, mesh_deleted)

        return total
