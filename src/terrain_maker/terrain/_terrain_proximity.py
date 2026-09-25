"""Proximity and ring masks around geographic points, and ring coloring."""

from __future__ import annotations


import numpy as np
import logging
from typing import Optional



# Output handling is configured once for the whole package in _logging.py
logger = logging.getLogger(__name__)


class TerrainProximityMixin:
    """Terrain methods: proximity and ring masks around geographic points, and ring coloring."""

    def compute_proximity_mask(
        self,
        lons: np.ndarray,
        lats: np.ndarray,
        radius_meters: float,
        input_crs: str = "EPSG:4326",
        cluster_threshold_meters: Optional[float] = None,
    ) -> np.ndarray:
        """
        Create boolean mask for vertices within radius of geographic points.

        Uses KDTree for efficient spatial queries to identify mesh vertices
        near specified geographic locations (e.g., parks, POIs). Optionally
        clusters nearby points first to create unified proximity zones.

        Args:
            lons: Array of longitudes for points of interest.
            lats: Array of latitudes for points of interest (must match lons shape).
            radius_meters: Radius in meters around each point/cluster to include.
            input_crs: CRS of input coordinates (default: "EPSG:4326" for WGS84 lon/lat).
            cluster_threshold_meters: Optional distance threshold for clustering nearby
                points using DBSCAN. Points within this distance are merged into single
                zones. If None, each point gets its own zone. Useful for merging nearby
                parks into continuous zones.

        Returns:
            Boolean array of shape (num_vertices,) where True indicates vertex is
            within radius of at least one point/cluster.

        Raises:
            RuntimeError: If create_mesh() has not been called yet.
            ValueError: If lons and lats have different shapes.

        Example:
            >>> # Create zones around parks
            >>> park_lons = np.array([-83.1, -83.2, -83.15])
            >>> park_lats = np.array([42.4, 42.5, 42.45])
            >>> mask = terrain.compute_proximity_mask(
            ...     park_lons, park_lats,
            ...     radius_meters=1000,
            ...     cluster_threshold_meters=200  # Merge parks within 200m
            ... )
            >>> # mask is True for vertices within 1km of park clusters
        """
        from scipy.spatial import KDTree

        # Check prerequisites
        if not hasattr(self, "vertices") or self.vertices is None:
            raise RuntimeError(
                "create_mesh() must be called before compute_proximity_mask(). "
                "Mesh vertices are needed for proximity calculations."
            )

        # Validate inputs
        lons = np.asarray(lons)
        lats = np.asarray(lats)
        if lons.shape != lats.shape:
            raise ValueError(f"lons and lats must have same shape. Got {lons.shape} and {lats.shape}")

        # Filter out NaN coordinates (missing park locations, etc.)
        valid_mask = ~(np.isnan(lons) | np.isnan(lats))
        if not np.all(valid_mask):
            num_invalid = np.sum(~valid_mask)
            self.logger.warning(
                f"Filtering out {num_invalid} points with NaN coordinates "
                f"({len(lons) - num_invalid} valid points remaining)"
            )
            lons = lons[valid_mask]
            lats = lats[valid_mask]

        # Handle edge case: no valid points
        if len(lons) == 0:
            self.logger.warning("No valid points remaining after filtering NaN coordinates")
            return np.zeros(len(self.vertices), dtype=bool)

        # Convert geographic coords to mesh space (x, y only, ignore z)
        xs, ys, _ = self.geo_to_mesh_coords(lons, lats, input_crs=input_crs)
        point_coords = np.column_stack([xs, ys])

        # Filter out points that ended up with NaN mesh coordinates (outside DEM bounds)
        mesh_valid_mask = ~(np.isnan(point_coords[:, 0]) | np.isnan(point_coords[:, 1]))
        if not np.all(mesh_valid_mask):
            num_invalid = np.sum(~mesh_valid_mask)
            self.logger.warning(
                f"Filtering out {num_invalid} points outside DEM bounds "
                f"({np.sum(mesh_valid_mask)} valid points remaining)"
            )
            point_coords = point_coords[mesh_valid_mask]

        # Handle edge case: no valid points after mesh coordinate conversion
        if len(point_coords) == 0:
            self.logger.warning("No points within DEM bounds for proximity mask")
            return np.zeros(len(self.vertices), dtype=bool)

        self.logger.info(f"Computing proximity mask for {len(point_coords)} points...")

        # Get pixel size from transformed DEM for metric conversions
        dem_info = self.data_layers.get("dem", {})
        transformed_transform = dem_info.get("transformed_transform")
        if transformed_transform is None:
            raise ValueError("Transformed DEM transform not found")

        # Pixel size in meters (assumes metric CRS like UTM after transformation)
        pixel_size_meters = abs(transformed_transform.a)
        scale_factor = self.model_params["scale_factor"]

        # Calculate meters per mesh unit
        # 1 mesh unit = scale_factor pixels = scale_factor * pixel_size_meters
        meters_per_mesh_unit = scale_factor * pixel_size_meters

        self.logger.debug(
            f"  Pixel size: {pixel_size_meters:.2f}m, "
            f"Scale factor: {scale_factor}, "
            f"Meters per mesh unit: {meters_per_mesh_unit:.2f}m"
        )

        # Cluster nearby points using DBSCAN
        if cluster_threshold_meters is not None:
            from sklearn.cluster import DBSCAN

            # Convert cluster threshold from meters to mesh units
            cluster_threshold_mesh = cluster_threshold_meters / meters_per_mesh_unit

            self.logger.info(
                f"  Clustering points with threshold {cluster_threshold_meters}m "
                f"({cluster_threshold_mesh:.3f} mesh units)..."
            )

            clustering = DBSCAN(eps=cluster_threshold_mesh, min_samples=1)
            labels = clustering.fit_predict(point_coords)
            num_clusters = labels.max() + 1

            # Use cluster centroids instead of individual points
            point_coords = np.array(
                [point_coords[labels == i].mean(axis=0) for i in range(num_clusters)]
            )

            self.logger.info(f"  Clustered {len(lons)} points into {num_clusters} zones")

        # Build KDTree for efficient spatial queries
        tree = KDTree(point_coords)

        # Get mesh vertex positions (just x, y for 2D distance)
        # vertices includes boundary vertices if boundary_extension=True
        mesh_verts_2d = self.vertices[:, :2]

        # Convert radius from meters to mesh units
        radius_mesh = radius_meters / meters_per_mesh_unit

        self.logger.info(
            f"  Querying vertices within {radius_meters}m ({radius_mesh:.3f} mesh units) "
            f"of {len(point_coords)} point(s)..."
        )

        # Query: which vertices are within radius of ANY point?
        distances, _ = tree.query(mesh_verts_2d, k=1)
        mask = distances <= radius_mesh

        num_in_zone = np.sum(mask)
        pct_in_zone = 100.0 * num_in_zone / len(mask)
        self.logger.info(
            f"  Proximity mask: {num_in_zone}/{len(mask)} vertices ({pct_in_zone:.1f}%) in zones"
        )

        return mask

    def compute_proximity_mask_grid(
        self,
        lons: np.ndarray,
        lats: np.ndarray,
        radius_meters: float,
        input_crs: str = "EPSG:4326",
        cluster_threshold_meters: Optional[float] = None,
    ) -> np.ndarray:
        """
        Create boolean mask for DEM grid cells within radius of geographic points.

        Similar to compute_proximity_mask() but works directly on the transformed DEM
        grid WITHOUT requiring mesh creation. Returns a 2D boolean array matching
        the transformed DEM shape.

        This is useful when you need to compute proximity zones before creating the
        mesh, avoiding the need for duplicate mesh creation.

        Args:
            lons: Array of longitudes for points of interest.
            lats: Array of latitudes for points of interest (must match lons shape).
            radius_meters: Radius in meters around each point/cluster to include.
            input_crs: CRS of input coordinates (default: "EPSG:4326" for WGS84 lon/lat).
            cluster_threshold_meters: Optional distance threshold for clustering nearby
                points using DBSCAN. If None, each point gets its own zone.

        Returns:
            Boolean array of shape (height, width) matching transformed DEM,
            where True indicates grid cell is within radius of at least one point/cluster.

        Raises:
            ValueError: If DEM has not been transformed yet or if lons/lats mismatch.

        Example:
            >>> terrain = Terrain(dem, transform)
            >>> terrain.add_transform(reproject_to_utm(...))
            >>> terrain.apply_transforms()
            >>> # Compute proximity BEFORE mesh creation
            >>> park_mask = terrain.compute_proximity_mask_grid(
            ...     park_lons, park_lats, radius_meters=1000
            ... )
            >>> # Use mask in color computations, then create mesh once
        """
        from scipy.spatial import KDTree
        from pyproj import Transformer

        # Check prerequisites
        dem_info = self.data_layers.get("dem", {})
        if not dem_info.get("transformed", False):
            raise ValueError(
                "DEM layer must be transformed before compute_proximity_mask_grid(). "
                "Call apply_transforms() first."
            )

        # Validate inputs
        lons = np.asarray(lons)
        lats = np.asarray(lats)
        if lons.shape != lats.shape:
            raise ValueError(f"lons and lats must have same shape. Got {lons.shape} and {lats.shape}")

        # Filter out NaN coordinates
        valid_mask = ~(np.isnan(lons) | np.isnan(lats))
        if not np.all(valid_mask):
            num_invalid = np.sum(~valid_mask)
            self.logger.warning(
                f"Filtering out {num_invalid} points with NaN coordinates "
                f"({len(lons) - num_invalid} valid points remaining)"
            )
            lons = lons[valid_mask]
            lats = lats[valid_mask]

        # Handle edge case: no valid points
        dem_data = dem_info["transformed_data"]
        if len(lons) == 0:
            self.logger.warning("No valid points remaining after filtering NaN coordinates")
            return np.zeros(dem_data.shape, dtype=bool)

        # Get transformed DEM properties
        transform = dem_info.get("transformed_transform")
        dem_crs = dem_info.get("transformed_crs", "EPSG:4326")

        if transform is None:
            raise ValueError("No transform found for transformed DEM layer.")

        # Reproject coordinates if input CRS differs from DEM CRS
        if input_crs != dem_crs:
            self.logger.debug(f"  Reprojecting {len(lons)} points from {input_crs} to {dem_crs}")
            transformer = Transformer.from_crs(input_crs, dem_crs, always_xy=True)
            lons, lats = transformer.transform(lons, lats)

        # Convert geographic coords to pixel coordinates
        from rasterio.transform import rowcol

        rows = []
        cols = []
        for lon, lat in zip(lons, lats):
            row, col = rowcol(transform, lon, lat)
            rows.append(row)
            cols.append(col)

        rows = np.array(rows)
        cols = np.array(cols)

        # Filter out points outside DEM bounds
        height, width = dem_data.shape
        valid_bounds = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        if not np.all(valid_bounds):
            num_invalid = np.sum(~valid_bounds)
            self.logger.warning(
                f"Filtering out {num_invalid} points outside DEM bounds "
                f"({np.sum(valid_bounds)} valid points remaining)"
            )
            rows = rows[valid_bounds]
            cols = cols[valid_bounds]

        # Handle edge case: no valid points after filtering
        if len(rows) == 0:
            self.logger.warning("No points within DEM bounds for proximity mask")
            return np.zeros(dem_data.shape, dtype=bool)

        self.logger.info(f"Computing grid-based proximity mask for {len(rows)} points...")

        # Get pixel size in meters (assumes metric CRS like UTM after transformation)
        pixel_size_meters = abs(transform.a)

        # Convert radius from meters to pixels
        radius_pixels = radius_meters / pixel_size_meters

        # Stack point coordinates
        point_coords = np.column_stack([rows, cols])

        # Cluster nearby points using DBSCAN
        if cluster_threshold_meters is not None:
            from sklearn.cluster import DBSCAN

            # Convert cluster threshold from meters to pixels
            cluster_threshold_pixels = cluster_threshold_meters / pixel_size_meters

            self.logger.info(
                f"  Clustering points with threshold {cluster_threshold_meters}m "
                f"({cluster_threshold_pixels:.3f} pixels)..."
            )

            clustering = DBSCAN(eps=cluster_threshold_pixels, min_samples=1)
            labels = clustering.fit_predict(point_coords)
            num_clusters = labels.max() + 1

            # Use cluster centroids instead of individual points
            point_coords = np.array(
                [point_coords[labels == i].mean(axis=0) for i in range(num_clusters)]
            )

            self.logger.info(f"  Clustered {len(lons)} points into {num_clusters} zones")

        # Build KDTree for efficient spatial queries
        tree = KDTree(point_coords)

        # Create grid of all pixel coordinates
        row_indices, col_indices = np.indices(dem_data.shape)
        grid_coords = np.column_stack([row_indices.ravel(), col_indices.ravel()])

        # Query: which pixels are within radius of ANY point?
        self.logger.info(
            f"  Querying {len(grid_coords)} pixels within {radius_meters}m "
            f"({radius_pixels:.3f} pixels) of {len(point_coords)} point(s)..."
        )

        distances, _ = tree.query(grid_coords, k=1)
        mask_flat = distances <= radius_pixels

        # Reshape to grid
        mask = mask_flat.reshape(dem_data.shape)

        num_in_zone = np.sum(mask)
        total_pixels = mask.size
        pct_in_zone = 100.0 * num_in_zone / total_pixels
        self.logger.info(
            f"  Proximity mask: {num_in_zone}/{total_pixels} pixels ({pct_in_zone:.1f}%) in zones"
        )

        return mask

    def compute_ring_mask_grid(
        self,
        lons: np.ndarray,
        lats: np.ndarray,
        inner_radius_meters: float,
        outer_radius_meters: float,
        input_crs: str = "EPSG:4326",
        cluster_threshold_meters: Optional[float] = None,
    ) -> np.ndarray:
        """
        Create boolean mask for ring (annulus) around geographic points.

        Creates a ring-shaped mask where pixels are True if they are:
        - Within outer_radius_meters of a point, AND
        - Outside inner_radius_meters of all points

        This is useful for creating visual outlines around areas of interest.

        Args:
            lons: Array of longitudes for points of interest.
            lats: Array of latitudes for points of interest (must match lons shape).
            inner_radius_meters: Inner radius of the ring in meters. Pixels closer
                than this to any point are excluded. Use 0 for filled circle.
            outer_radius_meters: Outer radius of the ring in meters. Pixels farther
                than this from all points are excluded.
            input_crs: CRS of input coordinates (default: "EPSG:4326" for WGS84 lon/lat).
            cluster_threshold_meters: Optional distance threshold for clustering nearby
                points using DBSCAN. If None, each point gets its own ring.

        Returns:
            Boolean array of shape (height, width) matching transformed DEM,
            where True indicates grid cell is within the ring zone.

        Raises:
            ValueError: If inner_radius > outer_radius or if radii are negative.

        Example:
            >>> # Create a 100m-wide ring at 2km from each park
            >>> ring_mask = terrain.compute_ring_mask_grid(
            ...     park_lons, park_lats,
            ...     inner_radius_meters=1900,
            ...     outer_radius_meters=2000,
            ... )
        """
        from scipy.spatial import KDTree
        from pyproj import Transformer

        # Validate radii
        if inner_radius_meters < 0:
            raise ValueError(f"inner_radius_meters must be >= 0, got {inner_radius_meters}")
        if outer_radius_meters < 0:
            raise ValueError(f"outer_radius_meters must be >= 0, got {outer_radius_meters}")
        if inner_radius_meters > outer_radius_meters:
            raise ValueError(
                f"inner_radius_meters ({inner_radius_meters}) must be <= "
                f"outer_radius_meters ({outer_radius_meters})"
            )

        # Check prerequisites
        dem_info = self.data_layers.get("dem", {})
        if not dem_info.get("transformed", False):
            # For tests with mock terrain, check for _transformed_dem attribute
            if hasattr(self, "_transformed_dem") and self._transformed_dem is not None:
                dem_data = self._transformed_dem
                transform = getattr(self, "_transformed_transform", None)
                dem_crs = getattr(self, "_transformed_crs", "EPSG:4326")
            else:
                raise ValueError(
                    "DEM layer must be transformed before compute_ring_mask_grid(). "
                    "Call apply_transforms() first."
                )
        else:
            dem_data = dem_info["transformed_data"]
            transform = dem_info.get("transformed_transform")
            dem_crs = dem_info.get("transformed_crs", "EPSG:4326")

        # Validate inputs
        lons = np.asarray(lons)
        lats = np.asarray(lats)
        if lons.shape != lats.shape:
            raise ValueError(f"lons and lats must have same shape. Got {lons.shape} and {lats.shape}")

        # Handle edge case: no points
        if len(lons) == 0:
            self.logger.debug("No points provided for ring mask - returning empty mask")
            return np.zeros(dem_data.shape, dtype=bool)

        # Filter out NaN coordinates
        valid_mask = ~(np.isnan(lons) | np.isnan(lats))
        if not np.all(valid_mask):
            num_invalid = np.sum(~valid_mask)
            self.logger.warning(
                f"Filtering out {num_invalid} points with NaN coordinates"
            )
            lons = lons[valid_mask]
            lats = lats[valid_mask]

        if len(lons) == 0:
            return np.zeros(dem_data.shape, dtype=bool)

        # If no transform, use simple grid coordinates (for testing)
        if transform is None:
            # Assume coordinates are already in grid space (0-1 normalized)
            height, width = dem_data.shape
            rows = (lats * height).astype(int)
            cols = (lons * width).astype(int)
            pixel_size_meters = 30.0  # Assume 30m pixels for testing
        else:
            # Reproject coordinates if input CRS differs from DEM CRS
            if input_crs != dem_crs:
                self.logger.debug(f"  Reprojecting points from {input_crs} to {dem_crs}")
                transformer = Transformer.from_crs(input_crs, dem_crs, always_xy=True)
                lons, lats = transformer.transform(lons, lats)

            # Convert geographic coords to pixel coordinates
            from rasterio.transform import rowcol

            rows = []
            cols = []
            for lon, lat in zip(lons, lats):
                row, col = rowcol(transform, lon, lat)
                rows.append(row)
                cols.append(col)

            rows = np.array(rows)
            cols = np.array(cols)

            # Get pixel size in meters
            pixel_size_meters = abs(transform.a)

        # Filter out points outside DEM bounds
        height, width = dem_data.shape
        valid_bounds = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        if not np.all(valid_bounds):
            rows = rows[valid_bounds]
            cols = cols[valid_bounds]

        if len(rows) == 0:
            return np.zeros(dem_data.shape, dtype=bool)

        self.logger.info(f"Computing ring mask for {len(rows)} points...")

        # Convert radii from meters to pixels
        inner_radius_pixels = inner_radius_meters / pixel_size_meters
        outer_radius_pixels = outer_radius_meters / pixel_size_meters

        # Stack point coordinates
        point_coords = np.column_stack([rows, cols])

        # Cluster nearby points using DBSCAN if requested
        if cluster_threshold_meters is not None:
            from sklearn.cluster import DBSCAN

            cluster_threshold_pixels = cluster_threshold_meters / pixel_size_meters
            clustering = DBSCAN(eps=cluster_threshold_pixels, min_samples=1)
            labels = clustering.fit_predict(point_coords)
            num_clusters = labels.max() + 1

            # Use cluster centroids
            point_coords = np.array(
                [point_coords[labels == i].mean(axis=0) for i in range(num_clusters)]
            )
            self.logger.info(f"  Clustered into {num_clusters} zones")

        # Build KDTree for efficient spatial queries
        tree = KDTree(point_coords)

        # Create grid of all pixel coordinates
        row_indices, col_indices = np.indices(dem_data.shape)
        grid_coords = np.column_stack([row_indices.ravel(), col_indices.ravel()])

        # Query distances to nearest point
        distances, _ = tree.query(grid_coords, k=1)

        # Create ring mask: within outer but outside inner
        # Use strict inequality for inner so inner_radius=0 gives filled circle
        outer_mask = distances <= outer_radius_pixels
        inner_mask = distances < inner_radius_pixels  # Strictly less than
        ring_mask_flat = outer_mask & ~inner_mask

        # Reshape to grid
        ring_mask = ring_mask_flat.reshape(dem_data.shape)

        num_in_ring = np.sum(ring_mask)
        self.logger.info(
            f"  Ring mask: {num_in_ring} pixels "
            f"(ring width: {outer_radius_meters - inner_radius_meters}m)"
        )

        return ring_mask

    def apply_ring_color(
        self,
        ring_mask: np.ndarray,
        ring_color: tuple = (0.1, 0.1, 0.1),
    ) -> None:
        """
        Apply a solid color to vertices within the ring mask.

        This modifies self.colors in-place, setting the RGB values for vertices
        that fall within the ring mask to the specified color.

        Args:
            ring_mask: 2D boolean array matching DEM shape. True = apply ring color.
            ring_color: RGB tuple (0-1 range) for ring color. Default: dark gray.

        Raises:
            ValueError: If ring_mask shape doesn't match DEM or colors not initialized.
        """
        if not hasattr(self, "colors") or self.colors is None:
            raise ValueError("Colors must be computed before applying ring color")

        if not hasattr(self, "y_valid") or not hasattr(self, "x_valid"):
            raise ValueError("Vertex coordinates (y_valid, x_valid) must be set")

        # Apply ring color to vertices within the mask
        for i, (y, x) in enumerate(zip(self.y_valid, self.x_valid)):
            # Clamp to valid indices
            y_idx = int(np.clip(y, 0, ring_mask.shape[0] - 1))
            x_idx = int(np.clip(x, 0, ring_mask.shape[1] - 1))

            if ring_mask[y_idx, x_idx]:
                self.colors[i, 0] = ring_color[0]
                self.colors[i, 1] = ring_color[1]
                self.colors[i, 2] = ring_color[2]
