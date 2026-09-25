"""Proximity and ring masks around geographic points, and ring coloring."""

from __future__ import annotations


import numpy as np
import logging

from pyproj import Transformer
from rasterio.transform import rowcol
from scipy.spatial import KDTree
from typing import Optional

# Output handling is configured once for the whole package in _logging.py
logger = logging.getLogger(__name__)


def _drop_nan_points(lons, lats, logger):
    """Validate matching shapes and drop points with a NaN coordinate."""
    lons, lats = np.asarray(lons), np.asarray(lats)
    if lons.shape != lats.shape:
        raise ValueError(f"lons and lats must have same shape. Got {lons.shape} and {lats.shape}")
    valid = ~(np.isnan(lons) | np.isnan(lats))
    if not np.all(valid):
        logger.warning(
            f"Filtering out {np.sum(~valid)} points with NaN coordinates "
            f"({np.sum(valid)} valid points remaining)"
        )
        lons, lats = lons[valid], lats[valid]
    return lons, lats


def _keep_points(points, inside, logger):
    """Rows of points where inside is True, warning about the ones dropped."""
    if not np.all(inside):
        logger.warning(
            f"Filtering out {np.sum(~inside)} points outside DEM bounds "
            f"({np.sum(inside)} valid points remaining)"
        )
    return points[inside]


def _pixel_points(
    lons, lats, shape, transform, dem_crs, input_crs, logger, require_transform=False
):
    """(row, col) of the valid points inside a grid, and the pixel size in meters.

    Without a transform (only allowed if not require_transform), points are taken as
    normalized [0, 1] grid coordinates with 30 m pixels, a mode used by tests.
    """
    lons, lats = _drop_nan_points(lons, lats, logger)
    if len(lons) == 0:
        logger.warning("No valid points remaining after filtering NaN coordinates")
        return np.empty((0, 2), dtype=int), None

    if transform is None:
        if require_transform:
            raise ValueError("No transform found for transformed DEM layer.")
        rows = (lats * shape[0]).astype(int)
        cols = (lons * shape[1]).astype(int)
        pixel_size_meters = 30.0
    else:
        if input_crs != dem_crs:
            logger.debug(f"  Reprojecting {len(lons)} points from {input_crs} to {dem_crs}")
            transformer = Transformer.from_crs(input_crs, dem_crs, always_xy=True)
            lons, lats = transformer.transform(lons, lats)
        rows, cols = (np.asarray(v) for v in rowcol(transform, lons, lats))
        # Assumes a metric CRS (e.g. UTM) after transformation
        pixel_size_meters = abs(transform.a)

    inside = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    return _keep_points(np.column_stack([rows, cols]), inside, logger), pixel_size_meters


def _cluster_centroids(points, eps, logger):
    """Merge points within eps of each other (DBSCAN chains) into their centroids."""
    from sklearn.cluster import DBSCAN

    labels = DBSCAN(eps=eps, min_samples=1).fit_predict(points)
    n_clusters = labels.max() + 1
    logger.info(f"  Clustered {len(points)} points into {n_clusters} zones")
    return np.array([points[labels == i].mean(axis=0) for i in range(n_clusters)])


def _nearest_point_distance_grid(points, shape):
    """Distance in pixels from every grid cell to the nearest of points (row, col)."""
    rows, cols = np.indices(shape)
    grid = np.column_stack([rows.ravel(), cols.ravel()])
    distances, _ = KDTree(points).query(grid, k=1)
    return distances.reshape(shape)


def _log_coverage(mask, unit, logger):
    logger.info(
        f"  Proximity mask: {np.sum(mask)}/{mask.size} {unit} "
        f"({100.0 * np.sum(mask) / mask.size:.1f}%) in zones"
    )


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
        if not hasattr(self, "vertices") or self.vertices is None:
            raise RuntimeError(
                "create_mesh() must be called before compute_proximity_mask(). "
                "Mesh vertices are needed for proximity calculations."
            )
        no_points = np.zeros(len(self.vertices), dtype=bool)

        lons, lats = _drop_nan_points(lons, lats, self.logger)
        if len(lons) == 0:
            self.logger.warning("No valid points remaining after filtering NaN coordinates")
            return no_points

        xs, ys, _ = self.geo_to_mesh_coords(lons, lats, input_crs=input_crs)
        points = np.column_stack([xs, ys])
        inside = ~(np.isnan(points[:, 0]) | np.isnan(points[:, 1]))  # NaN = outside the DEM
        points = _keep_points(points, inside, self.logger)
        if len(points) == 0:
            self.logger.warning("No points within DEM bounds for proximity mask")
            return no_points
        self.logger.info(f"Computing proximity mask for {len(points)} points...")

        transform = self.data_layers.get("dem", {}).get("transformed_transform")
        if transform is None:
            raise ValueError("Transformed DEM transform not found")
        # 1 mesh unit = scale_factor pixels (assumes a metric CRS after transformation)
        pixel_size_meters = abs(transform.a)
        meters_per_mesh_unit = self.model_params["scale_factor"] * pixel_size_meters
        self.logger.debug(
            f"  Pixel size: {pixel_size_meters:.2f}m, "
            f"Meters per mesh unit: {meters_per_mesh_unit:.2f}m"
        )

        if cluster_threshold_meters is not None:
            points = _cluster_centroids(
                points, cluster_threshold_meters / meters_per_mesh_unit, self.logger
            )

        # Includes boundary (skirt) vertices when boundary_extension=True
        distances, _ = KDTree(points).query(self.vertices[:, :2], k=1)
        mask = distances <= radius_meters / meters_per_mesh_unit
        _log_coverage(mask, "vertices", self.logger)
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
        dem_info = self.data_layers.get("dem", {})
        if not dem_info.get("transformed", False):
            raise ValueError(
                "DEM layer must be transformed before compute_proximity_mask_grid(). "
                "Call apply_transforms() first."
            )
        dem_data = dem_info["transformed_data"]

        points, pixel_size_meters = _pixel_points(
            lons,
            lats,
            dem_data.shape,
            dem_info.get("transformed_transform"),
            dem_info.get("transformed_crs", "EPSG:4326"),
            input_crs,
            self.logger,
            require_transform=True,
        )
        if len(points) == 0:
            self.logger.warning("No points within DEM bounds for proximity mask")
            return np.zeros(dem_data.shape, dtype=bool)
        self.logger.info(f"Computing grid-based proximity mask for {len(points)} points...")

        if cluster_threshold_meters is not None:
            points = _cluster_centroids(
                points, cluster_threshold_meters / pixel_size_meters, self.logger
            )
        distances = _nearest_point_distance_grid(points, dem_data.shape)
        mask = distances <= radius_meters / pixel_size_meters
        _log_coverage(mask, "pixels", self.logger)
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
        if inner_radius_meters < 0:
            raise ValueError(f"inner_radius_meters must be >= 0, got {inner_radius_meters}")
        if outer_radius_meters < 0:
            raise ValueError(f"outer_radius_meters must be >= 0, got {outer_radius_meters}")
        if inner_radius_meters > outer_radius_meters:
            raise ValueError(
                f"inner_radius_meters ({inner_radius_meters}) must be <= "
                f"outer_radius_meters ({outer_radius_meters})"
            )

        dem_info = self.data_layers.get("dem", {})
        if dem_info.get("transformed", False):
            dem_data = dem_info["transformed_data"]
            transform = dem_info.get("transformed_transform")
            dem_crs = dem_info.get("transformed_crs", "EPSG:4326")
        elif getattr(self, "_transformed_dem", None) is not None:
            # Lightweight stand-in used by tests
            dem_data = self._transformed_dem
            transform = getattr(self, "_transformed_transform", None)
            dem_crs = getattr(self, "_transformed_crs", "EPSG:4326")
        else:
            raise ValueError(
                "DEM layer must be transformed before compute_ring_mask_grid(). "
                "Call apply_transforms() first."
            )

        points, pixel_size_meters = _pixel_points(
            lons, lats, dem_data.shape, transform, dem_crs, input_crs, self.logger
        )
        if len(points) == 0:
            return np.zeros(dem_data.shape, dtype=bool)
        self.logger.info(f"Computing ring mask for {len(points)} points...")

        if cluster_threshold_meters is not None:
            points = _cluster_centroids(
                points, cluster_threshold_meters / pixel_size_meters, self.logger
            )
        distances = _nearest_point_distance_grid(points, dem_data.shape)
        # Strict inequality on the inner edge so inner_radius=0 gives a filled circle
        ring_mask = (distances <= outer_radius_meters / pixel_size_meters) & ~(
            distances < inner_radius_meters / pixel_size_meters
        )
        self.logger.info(
            f"  Ring mask: {np.sum(ring_mask)} pixels "
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

        ys = np.clip(self.y_valid, 0, ring_mask.shape[0] - 1).astype(int)
        xs = np.clip(self.x_valid, 0, ring_mask.shape[1] - 1).astype(int)
        in_ring = np.nonzero(ring_mask[ys, xs])[0]
        for channel in range(3):
            self.colors[in_ring, channel] = ring_color[channel]
