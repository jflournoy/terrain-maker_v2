"""Mesh creation and mesh-space coordinate conversion."""

from __future__ import annotations

import time

try:
    import bpy
except ImportError:
    bpy = None
from scipy.ndimage import zoom
import numpy as np
import logging

# Output handling is configured once for the whole package in _logging.py
logger = logging.getLogger(__name__)


class TerrainMeshMixin:
    """Terrain methods: mesh creation and mesh-space coordinate conversion."""

    def create_mesh(
        self,
        base_depth=0.2,
        boundary_extension=True,
        scale_factor=100.0,
        height_scale=1.0,
        center_model=True,
        verbose=True,
        detect_water=False,
        water_slope_threshold=0.5,
        water_mask=None,
        two_tier_edge=True,
        edge_mid_depth=None,
        edge_base_material="clay",
        edge_blend_colors=False,
        smooth_boundary=False,
        smooth_boundary_window=5,
        use_catmull_rom=False,
        catmull_rom_subdivisions=10,
        use_rectangle_edges=True,
        use_fractional_edges=True,
        edge_sample_spacing=0.33,
    ):
        """
        Create a Blender mesh from transformed DEM data with both performance and control.

        Generates vertices from DEM elevation values and faces for connectivity. Optionally
        creates boundary faces to close the mesh into a solid. Supports coordinate scaling
        and elevation scaling for visualization. Can optionally detect and apply water bodies
        to vertex alpha channel for water rendering.

        Args:
            base_depth (float): Positive depth offset below minimum surface elevation (default: 0.2).
                Creates a flat base plane at: min_surface_z - base_depth.
                Used when boundary_extension=True to create side faces.
                Positive values extend below surface, negative extend above.
            boundary_extension (bool): Whether to create side faces around the terrain boundary
                to close the mesh (default: True). If False, creates open terrain surface.
            scale_factor (float): Horizontal scale divisor for x/y coordinates (default: 100.0).
                Higher values produce smaller meshes. E.g., 100 means 100 DEM units = 1 Blender unit.
            height_scale (float): Multiplier for elevation values (default: 1.0). Vertically
                exaggerates or reduces terrain features. Values > 1 exaggerate, < 1 flatten.
            center_model (bool): Whether to center the model at origin (default: True).
                Centers XY coordinates but preserves absolute Z elevation values.
            verbose (bool): Whether to log detailed progress information (default: True).
            detect_water (bool): Whether to detect water bodies and apply to alpha channel
                (default: False). Uses slope-based detection on transformed DEM.
            water_slope_threshold (float): Maximum slope magnitude to classify as water
                (default: 0.5). Only used if detect_water=True and water_mask is None.
            water_mask (np.ndarray): Pre-computed boolean water mask (True=water, False=land).
                If provided, this mask is used instead of computing water detection.
                Allows water detection on unscaled DEM before elevation scaling transforms.
            two_tier_edge (bool): Enable two-tier edge extrusion (default: True).
                Creates a small colored edge near the surface with a larger uniform base below.
            edge_mid_depth (float): Positive depth offset below surface for middle tier (default: auto-calculated).
                If None, automatically set to base_depth * 0.25 (typically 0.05).
                Positive values extend below surface, negative extend above surface.
            edge_base_material (str | tuple): Material for base layer (default: "clay").
                Either a preset name ("clay", "obsidian", "chrome", "plastic", "gold", "ivory")
                or an RGB tuple (0-1 range).
            edge_blend_colors (bool): Blend surface colors to mid tier in two-tier mode (default: False).
                If False, mid tier uses base_material color for sharp transition between mesh and edge.
            smooth_boundary (bool): Apply smoothing to boundary points to eliminate stair-step edges
                (default: False). Useful for smoother mesh transitions when using two-tier edge.
            smooth_boundary_window (int): Window size for boundary smoothing (default: 5).
                Larger values produce more smoothing. Only used when smooth_boundary=True.
            use_catmull_rom (bool): Use Catmull-Rom curve fitting for smooth boundary geometry
                (default: False). When enabled, eliminates pixel-grid staircase pattern entirely
                by fitting smooth parametric curve through boundary points.
            catmull_rom_subdivisions (int): Number of interpolated points per boundary segment
                when using Catmull-Rom curves (default: 10). Higher values = smoother curve
                but more vertices. Only used when use_catmull_rom=True.
            use_rectangle_edges (bool): Use rectangle-edge sampling instead of morphological
                boundary detection (default: True). ~150x faster than morphological detection.
                Ideal for rectangular DEMs from raster sources. Uses original DEM shape to
                generate clean, regularly-sampled edge vertices.
            use_fractional_edges (bool): Use fractional edge coordinates that preserve projection
                curvature (default: True). When enabled with use_rectangle_edges, surface tier aligns
                with mesh boundary (no gap) while mid/base tiers use fractional X,Y positions to follow
                smooth WGS84→UTM projection curves. Creates geographically accurate curved edges that
                connect seamlessly to terrain.
            edge_sample_spacing (float): Pixel spacing for edge/skirt vertices (default: 0.33).
                Lower values = denser skirt (smoother but more memory). 0.33 = 3x denser than
                mesh edge. 1.0 = same density as mesh edge. 2.0 = half density (good for large
                meshes to avoid OOM).

        Returns:
            bpy.types.Object | None: The created terrain mesh object, or None if creation failed.

        Raises:
            ValueError: If transformed DEM layer is not available (apply_transforms() not called).

        Examples:
            # Default two-tier edge with smooth fractional edges and red clay base
            mesh = terrain.create_mesh(boundary_extension=True)

            # Single-tier edge (backwards compatible)
            mesh = terrain.create_mesh(two_tier_edge=False)

            # Two-tier with gold base material
            mesh = terrain.create_mesh(edge_base_material="gold")

            # Two-tier with deeper edge (1 unit below min surface, 0.2 units below surface for mid)
            mesh = terrain.create_mesh(
                base_depth=1.0,       # Deep base (1 unit below min surface)
                edge_mid_depth=0.2    # Deeper colored edge (0.2 units below surface)
            )

            # Two-tier with custom RGB color
            mesh = terrain.create_mesh(edge_base_material=(0.6, 0.55, 0.5))

            # Disable fractional edges for simple rectangular boundary
            mesh = terrain.create_mesh(use_fractional_edges=False)
        """
        start_time = time.time()
        self.logger.info("Creating terrain mesh...")

        # Get transformed DEM data
        if "dem" not in self.data_layers or not self.data_layers["dem"].get("transformed", False):
            raise ValueError("Transformed DEM layer required for mesh creation")

        # Note: Color computation moved to after vertex position generation
        # (multi-overlay mode needs y_valid and x_valid to exist)

        # Water detection - store mask for later (coloring happens after compute_colors)
        _water_mask_for_coloring = None
        if detect_water or water_mask is not None:
            if water_mask is None:
                # Compute water mask from transformed DEM
                dem_data = self.data_layers["dem"]["transformed_data"]
                self.logger.info(
                    f"Detecting water bodies (slope threshold: {water_slope_threshold})..."
                )
                from terrain_maker.terrain.water import identify_water_by_slope

                water_mask = identify_water_by_slope(
                    dem_data, slope_threshold=water_slope_threshold, fill_holes=True
                )
            else:
                self.logger.info(
                    f"Using pre-computed water mask ({np.sum(water_mask)} water pixels)"
                )
            _water_mask_for_coloring = water_mask

        dem_data = self.data_layers["dem"]["transformed_data"]
        height, width = dem_data.shape

        # Create valid points mask (non-NaN values)
        valid_mask = ~np.isnan(dem_data)

        # Generate vertex positions with scaling
        self.logger.info("Generating vertex positions...")
        from terrain_maker.terrain.mesh_operations import generate_vertex_positions

        positions, y_valid, x_valid = generate_vertex_positions(
            dem_data, valid_mask, scale_factor, height_scale
        )

        # Store y_valid and x_valid as instance attributes for color computation
        self.y_valid = y_valid
        self.x_valid = x_valid

        # Compute colors if color mapping is set and colors haven't been computed yet
        # Check for any color mapping mode (standard, blended, or multi-overlay)
        has_color_mapping = (
            hasattr(self, "color_mapping") or
            hasattr(self, "base_colormap") or
            hasattr(self, "color_mapping_mode")
        )
        if has_color_mapping and not hasattr(self, "colors"):
            self.compute_colors()

        # Apply water coloring AFTER color computation (so we modify the colormap colors)
        if _water_mask_for_coloring is not None and hasattr(self, "colors") and self.colors is not None:
            from scipy.ndimage import distance_transform_edt

            water_mask = _water_mask_for_coloring

            # Validate and resample water mask to match colors shape if needed
            expected_shape = self.colors.shape[:2] if self.colors.ndim == 3 else dem_data.shape
            if water_mask.shape != expected_shape:
                self.logger.warning(
                    f"Water mask shape {water_mask.shape} does not match colors shape {expected_shape}. "
                    f"Resampling water mask to match colors. This can happen when colors are computed from "
                    f"a different layer than DEM (e.g., score layers)."
                )
                # Resample water mask to match colors shape
                zoom_y = expected_shape[0] / water_mask.shape[0]
                zoom_x = expected_shape[1] / water_mask.shape[1]
                water_mask = zoom(
                    water_mask.astype(np.float32),
                    zoom=(zoom_y, zoom_x),
                    order=0,  # Nearest neighbor for boolean
                    prefilter=False,
                ).astype(np.bool_)

            water_distances = distance_transform_edt(water_mask)

            # Define gradient colors: blue to dark blue (Great Lakes depth)
            edge_color = np.array([25, 85, 125], dtype=np.float32)  # Medium blue (shore)
            center_color = np.array([15, 50, 85], dtype=np.float32)  # Deep blue (center)

            # Map water mask (2D grid) to vertex indices
            # self.colors is a 1D vertex array, not a 2D grid
            water_at_vertices = water_mask[self.y_valid, self.x_valid]
            water_vertex_indices = np.where(water_at_vertices)[0]

            # Get raw pixel distances for water vertices using grid coordinates
            water_y = self.y_valid[water_vertex_indices]
            water_x = self.x_valid[water_vertex_indices]
            water_pixel_distances = water_distances[water_y, water_x]

            # Cartographic shoreline vignette style (vintage map aesthetic)
            # Gradient only in shoreline band; interior water is uniform dark
            shoreline_width_pixels = 12

            # t=1 means dark (interior), t=0 means light (at shore edge)
            # Start with all water as interior (dark)
            t = np.ones_like(water_pixel_distances)

            # Apply gradient only within shoreline band
            in_shoreline_band = water_pixel_distances < shoreline_width_pixels
            t[in_shoreline_band] = water_pixel_distances[in_shoreline_band] / shoreline_width_pixels

            # Power curve for smoother transition
            t = np.power(t, 0.5)[:, np.newaxis]
            water_colors = edge_color * (1 - t) + center_color * t

            # Apply water colors - handle both grid-space (H, W, 4) and vertex-space (N, 4) colors
            if self.colors.ndim == 3:
                # Grid-space colors - index using y, x coordinates directly
                self.colors[water_y, water_x, :3] = water_colors.astype(np.uint8)
            else:
                # Vertex-space colors - index using vertex indices
                self.colors[water_vertex_indices, :3] = water_colors.astype(np.uint8)

            self.logger.info(f"Water colored with depth gradient ({np.sum(water_mask)} water pixels)")

        # Center the model if requested
        if center_model:
            self.logger.info("Centering model at origin...")
            # Calculate centroid
            centroid = np.mean(positions, axis=0)
            # Center horizontally (x, y) but preserve elevation (z)
            positions[:, 0] -= centroid[0]
            positions[:, 1] -= centroid[1]

            # Store offset for later reference (camera positioning)
            self.model_offset = centroid
        else:
            self.model_offset = np.array([0, 0, 0])

        # Store model parameters for reference
        self.model_params = {
            "scale_factor": scale_factor,
            "height_scale": height_scale,
            "centered": center_model,
            "offset": self.model_offset.tolist(),
            "base_depth": base_depth,
            "two_tier_edge": two_tier_edge,
            "edge_mid_depth": edge_mid_depth,
            "edge_base_material": edge_base_material,
            "edge_blend_colors": edge_blend_colors,
        }

        # Create mapping from (y,x) coords to vertex indices - using dictionaries for O(1) lookups
        self.logger.info("Creating coordinate to index mapping...")
        coord_to_index = {(y, x): i for i, (y, x) in enumerate(zip(y_valid, x_valid))}

        # OPTIMIZATION: Find boundary points using morphological operations
        self.logger.info("Finding boundary points with optimized algorithm...")
        from terrain_maker.terrain.mesh_operations import find_boundary_points

        boundary_coords = find_boundary_points(valid_mask)

        # Only sort boundary points if needed (they're used for side faces)
        boundary_winding = "counter-clockwise"  # Default winding direction
        if boundary_extension:
            boundary_points = self._sort_boundary_points_optimized(boundary_coords)

            # Determine winding direction based on boundary type
            if use_rectangle_edges:
                # Rectangle edge sampling always traces clockwise in image coordinates
                # (top→right→bottom→left with y increasing downward)
                boundary_winding = "clockwise"
                self.logger.info(f"Boundary winding direction: {boundary_winding} (rectangle edges always clockwise)")
            elif len(boundary_points) >= 3:
                # Compute signed area to determine winding for morphological boundary.
                # In image coordinates (y down), sum((x2-x1)*(y2+y1)) is negative for
                # a clockwise loop, the same convention the rectangle branch uses.
                signed_area = 0
                for i in range(len(boundary_points)):
                    y1, x1 = boundary_points[i]
                    y2, x2 = boundary_points[(i + 1) % len(boundary_points)]
                    signed_area += (x2 - x1) * (y2 + y1)
                boundary_winding = "clockwise" if signed_area < 0 else "counter-clockwise"
                self.logger.info(f"Boundary winding direction: {boundary_winding} (signed area: {signed_area:.2f})")
        else:
            boundary_points = boundary_coords

        # OPTIMIZATION: Vectorized face generation using NumPy operations
        self.logger.info("Generating faces with vectorized operations...")
        from terrain_maker.terrain.mesh_operations import generate_faces

        faces = generate_faces(height, width, coord_to_index)

        # Handle boundary extension if needed
        if boundary_extension:
            self.logger.info("Creating optimized boundary extension...")
            from terrain_maker.terrain.mesh_operations import create_boundary_extension

            # Get surface colors for two-tier edge (if available and two_tier_edge enabled)
            surface_colors = None
            if two_tier_edge and hasattr(self, "colors") and self.colors is not None:
                # Flatten colors if they're 2D grid (H, W, 3) to 1D (N, 3)
                if self.colors.ndim == 3:
                    surface_colors = self.colors[y_valid, x_valid, :]
                else:
                    surface_colors = self.colors

            # Call create_boundary_extension with two-tier and smoothing parameters
            result = create_boundary_extension(
                positions,
                boundary_points,
                coord_to_index,
                base_depth,
                two_tier=two_tier_edge,
                mid_depth=edge_mid_depth,
                base_material=edge_base_material,
                blend_edge_colors=edge_blend_colors,
                surface_colors=surface_colors,
                smooth_boundary=smooth_boundary,
                smooth_window_size=smooth_boundary_window,
                use_catmull_rom=use_catmull_rom,
                catmull_rom_subdivisions=catmull_rom_subdivisions,
                use_rectangle_edges=use_rectangle_edges,
                terrain=self if use_rectangle_edges else None,  # NEW: Pass terrain for transform-aware edges
                edge_sample_spacing=edge_sample_spacing,  # User-configurable density
                boundary_winding=boundary_winding,  # Pass winding direction for correct face normals
                use_fractional_edges=use_fractional_edges,  # NEW: Use fractional coords for projection curvature
                scale_factor=scale_factor,  # For fractional edge X,Y computation
                model_offset=self.model_offset,  # For fractional edge X,Y computation
            )

            # Handle return value (2-tuple for single-tier, 3-tuple for two-tier)
            if two_tier_edge:
                boundary_vertices, boundary_faces, boundary_colors = result
                # Store boundary_colors separately for Blender vertex coloring
                # Don't extend self.colors - it stays as 2D grid for surface vertices
                # Boundary colors will be applied directly to mesh vertices after creation
                self.boundary_colors = boundary_colors
            else:
                boundary_vertices, boundary_faces = result
                self.boundary_colors = None

            # Extend vertices with boundary vertices
            vertices = np.vstack([positions, boundary_vertices])
            # Add boundary faces to complete the mesh
            faces.extend(boundary_faces)
        else:
            vertices = positions

        # Store vertices and faces for later use (e.g., proximity calculations)
        self.vertices = vertices
        self.faces = faces
        self.y_valid = y_valid
        self.x_valid = x_valid

        # Create the Blender mesh
        try:
            from terrain_maker.terrain.blender_integration import create_blender_mesh

            # Prepare colors if available
            colors = self.colors if hasattr(self, "colors") else None
            boundary_colors = self.boundary_colors if hasattr(self, "boundary_colors") else None

            obj = create_blender_mesh(
                vertices,
                faces,
                colors=colors,
                y_valid=y_valid,
                x_valid=x_valid,
                boundary_colors=boundary_colors,
                name="TerrainMesh",
                logger=self.logger,
            )

            elapsed = time.time() - start_time
            self.logger.info(f"Terrain mesh created successfully in {elapsed:.2f} seconds")
            self.terrain_obj = obj
            return obj

        except Exception as e:
            self.logger.error(f"Error creating terrain mesh: {str(e)}")
            raise

    def _sort_boundary_points_optimized(self, boundary_coords):
        """
        Sort boundary points efficiently using spatial relationships.

        Args:
            boundary_coords: List of (y, x) coordinate tuples

        Returns:
            list: Sorted boundary points forming a continuous path
        """
        from terrain_maker.terrain.mesh_operations import sort_boundary_points

        return sort_boundary_points(boundary_coords)

    def geo_to_mesh_coords(
        self,
        lon: np.ndarray | float,
        lat: np.ndarray | float,
        elevation_offset: float = 0.0,
        input_crs: str = "EPSG:4326",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert geographic coordinates to Blender mesh coordinates.

        Transforms lon/lat points through the same pipeline used for the terrain mesh,
        producing coordinates that align with the rendered terrain. Useful for placing
        markers, labels, or other objects at geographic locations on the terrain.

        Args:
            lon: Longitude(s) or X coordinate(s). Can be a single float or array.
            lat: Latitude(s) or Y coordinate(s). Can be a single float or array
                (must match lon shape).
            elevation_offset: Height above terrain surface in mesh units (default: 0.0).
                Positive values place points above the terrain.
            input_crs: CRS of input coordinates (default: "EPSG:4326" for WGS84 lon/lat).
                Will be reprojected to match the transformed DEM's CRS if different.

        Returns:
            Tuple of (x, y, z) arrays in Blender mesh coordinates. If single values
            were passed for lon/lat, returns single float values.

        Raises:
            RuntimeError: If create_mesh() has not been called yet (model_params not set).
            ValueError: If the DEM layer has not been transformed yet.

        Example:
            >>> terrain = Terrain(dem, transform)
            >>> terrain.add_transform(reproject_to_utm("EPSG:4326", "EPSG:32617"))
            >>> terrain.apply_transforms()
            >>> mesh = terrain.create_mesh()
            >>> # Get mesh coords for a park at (-83.1, 42.4) in WGS84
            >>> x, y, z = terrain.geo_to_mesh_coords(-83.1, 42.4, elevation_offset=0.1)
            >>> # Create marker at (x, y, z) in Blender
        """
        from pyproj import Transformer

        # Check prerequisites
        if not hasattr(self, "model_params") or self.model_params is None:
            raise RuntimeError(
                "create_mesh() must be called before geo_to_mesh_coords(). "
                "The mesh parameters are needed for coordinate conversion."
            )

        dem_info = self.data_layers.get("dem", {})
        if not dem_info.get("transformed", False):
            raise ValueError(
                "DEM layer must be transformed before geo_to_mesh_coords(). "
                "Call apply_transforms() first."
            )

        # Get transformed DEM and its transform
        dem_data = dem_info["transformed_data"]
        transform = dem_info.get("transformed_transform")
        dem_crs = dem_info.get("transformed_crs", "EPSG:4326")

        if transform is None:
            raise ValueError("No transform found for transformed DEM layer.")

        # Convert to arrays for uniform handling
        lon_arr = np.atleast_1d(np.asarray(lon, dtype=np.float64))
        lat_arr = np.atleast_1d(np.asarray(lat, dtype=np.float64))
        scalar_input = lon_arr.shape == (1,) and not hasattr(lon, "__len__")

        if lon_arr.shape != lat_arr.shape:
            raise ValueError(
                f"lon and lat must have the same shape. Got {lon_arr.shape} and {lat_arr.shape}"
            )

        # Reproject coordinates if input CRS differs from DEM CRS
        if input_crs != dem_crs:
            self.logger.debug(f"  Reprojecting {len(lon_arr)} points from {input_crs} to {dem_crs}")
            transformer = Transformer.from_crs(input_crs, dem_crs, always_xy=True)
            lon_arr, lat_arr = transformer.transform(lon_arr, lat_arr)

        # Convert geographic coords to pixel coords using inverse transform
        # Affine: (col, row) = ~transform * (x, y) where x=easting, y=northing
        inv_transform = ~transform
        cols, rows = inv_transform * (lon_arr, lat_arr)

        # Round to nearest pixel
        rows = np.round(rows).astype(int)
        cols = np.round(cols).astype(int)

        # Get DEM shape
        height, width = dem_data.shape

        # Clamp to valid range and track out-of-bounds points
        valid_mask = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        rows_clamped = np.clip(rows, 0, height - 1)
        cols_clamped = np.clip(cols, 0, width - 1)

        # Get elevation values from DEM
        elevations = dem_data[rows_clamped, cols_clamped]

        # Handle out-of-bounds or NaN elevations
        invalid_mask = ~valid_mask | np.isnan(elevations)
        if np.any(invalid_mask):
            # Use mean elevation for invalid points
            valid_elevations = dem_data[~np.isnan(dem_data)]
            mean_elev = np.mean(valid_elevations) if len(valid_elevations) > 0 else 0.0
            elevations = np.where(invalid_mask, mean_elev, elevations)
            self.logger.debug(
                f"  {np.sum(invalid_mask)} points outside DEM bounds, using mean elevation"
            )

        # Apply mesh scaling (same as generate_vertex_positions)
        scale_factor = self.model_params["scale_factor"]
        height_scale = self.model_params["height_scale"]

        x = cols.astype(np.float64) / scale_factor
        y = rows.astype(np.float64) / scale_factor
        z = elevations * height_scale + elevation_offset

        # Apply centering offset if model was centered
        if self.model_params.get("centered", False):
            offset = self.model_params["offset"]
            x = x - offset[0]
            y = y - offset[1]
            # Note: z offset not applied - elevation_offset handles vertical positioning

        # Return scalars if input was scalar
        if scalar_input:
            return float(x[0]), float(y[0]), float(z[0])

        return x, y, z
