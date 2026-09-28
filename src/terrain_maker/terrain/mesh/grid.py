"""Surface mesh from a DEM grid: vertex positions, faces, vertex colors and MeshData."""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage

from terrain_maker.terrain._numba_compat import NUMBA_AVAILABLE, jit, prange

logger = logging.getLogger(__name__)


def find_boundary_points(valid_mask):
    """
    Find boundary points using morphological operations.

    Identifies points on the edge of valid regions using binary erosion.
    A point is considered a boundary point if it is valid but has at least
    one invalid neighbor in a 4-connected neighborhood.

    Args:
        valid_mask (np.ndarray): Boolean mask indicating valid points (True for valid)

    Returns:
        list: List of (y, x) coordinate tuples representing boundary points
    """
    # Interior points have 4 neighbors in a 4-connected neighborhood
    struct = ndimage.generate_binary_structure(2, 1)  # 4-connected structure
    eroded = ndimage.binary_erosion(valid_mask, struct)
    boundary_mask = valid_mask & ~eroded

    # Get boundary coords as (y,x) tuples
    boundary_indices = np.where(boundary_mask)
    boundary_coords = list(zip(boundary_indices[0], boundary_indices[1]))

    return boundary_coords


def generate_vertex_positions(dem_data, valid_mask, scale_factor=100.0, height_scale=1.0):
    """
    Generate 3D vertex positions from DEM data.

    Converts 2D elevation grid into 3D positions for mesh vertices, applying
    scaling factors for visualization. Only generates vertices for valid (non-NaN)
    DEM values.

    Args:
        dem_data (np.ndarray): 2D array of elevation values (height x width)
        valid_mask (np.ndarray): Boolean mask indicating valid points (True for non-NaN)
        scale_factor (float): Horizontal scale divisor for x/y coordinates (default: 100.0).
            Higher values produce smaller meshes. E.g., 100 means 100 DEM units = 1 unit.
        height_scale (float): Multiplier for elevation values (default: 1.0).
            Values > 1 exaggerate terrain, < 1 flatten it.

    Returns:
        tuple: (positions, y_valid, x_valid) where:
            - positions: np.ndarray of shape (n_valid, 3) with (x, y, z) coordinates
            - y_valid: np.ndarray of y indices for valid points
            - x_valid: np.ndarray of x indices for valid points
    """
    height, width = dem_data.shape

    # Use NumPy for coordinate generation
    y_indices, x_indices = np.mgrid[0:height, 0:width]
    y_valid = y_indices[valid_mask]
    x_valid = x_indices[valid_mask]

    # Generate vertex positions with scaling
    positions = np.column_stack(
        [
            x_valid / scale_factor,  # x position
            y_valid / scale_factor,  # y position
            dem_data[valid_mask] * height_scale,  # z position with height scaling
        ]
    )

    return positions, y_valid, x_valid


@jit(nopython=True, parallel=True, cache=True)
def _generate_faces_numba(height, width, index_grid):
    """
    Numba-accelerated face generation.

    Args:
        height: Grid height
        width: Grid width
        index_grid: 2D array where index_grid[y,x] = vertex index, or -1 if invalid

    Returns:
        faces: Array of face vertex indices (n_faces, 4), padded with -1 for triangles
        n_faces: Number of valid faces
    """
    # Pre-allocate maximum possible faces (each cell can have 1 quad)
    max_faces = (height - 1) * (width - 1)
    faces = np.full((max_faces, 4), -1, dtype=np.int64)
    face_count = 0

    # Process each potential quad
    for y in prange(height - 1):
        for x in range(width - 1):
            # Get indices for quad corners
            i0 = index_grid[y, x]
            i1 = index_grid[y, x + 1]
            i2 = index_grid[y + 1, x + 1]
            i3 = index_grid[y + 1, x]

            # Count valid corners
            valid_count = (i0 >= 0) + (i1 >= 0) + (i2 >= 0) + (i3 >= 0)

            if valid_count >= 3:
                # Store face (quads have 4 indices, triangles have 3 with -1 padding)
                idx = y * (width - 1) + x  # Unique index for this quad position
                if valid_count == 4:
                    faces[idx, 0] = i0
                    faces[idx, 1] = i1
                    faces[idx, 2] = i2
                    faces[idx, 3] = i3
                else:
                    # Triangle - collect valid indices
                    j = 0
                    if i0 >= 0:
                        faces[idx, j] = i0
                        j += 1
                    if i1 >= 0:
                        faces[idx, j] = i1
                        j += 1
                    if i2 >= 0:
                        faces[idx, j] = i2
                        j += 1
                    if i3 >= 0:
                        faces[idx, j] = i3
                        j += 1

    return faces


def generate_faces(height, width, coord_to_index, batch_size=10000):
    """
    Generate mesh faces from a grid of valid points.

    Creates quad faces for the mesh by checking each potential quad position
    and verifying that its corners exist in the coordinate-to-index mapping.
    If a quad has all 4 corners, creates a quad face. If it has 3 corners,
    creates a triangle face. Skips quads with fewer than 3 corners.

    Args:
        height (int): Height of the DEM grid
        width (int): Width of the DEM grid
        coord_to_index (dict): Mapping from (y, x) coordinates to vertex indices
        batch_size (int): Number of quads to process in each batch (default: 10000)

    Returns:
        list: List of face tuples, where each tuple contains vertex indices
    """
    if NUMBA_AVAILABLE:
        # Convert dict to 2D index grid for Numba
        index_grid = np.full((height, width), -1, dtype=np.int64)
        for (y, x), idx in coord_to_index.items():
            index_grid[y, x] = idx

        # Run Numba-accelerated version
        faces_array = _generate_faces_numba(height, width, index_grid)

        # Filter out unused slots and convert to list of tuples
        faces = []
        for i in range(faces_array.shape[0]):
            if faces_array[i, 0] >= 0:  # Valid face
                if faces_array[i, 3] >= 0:  # Quad
                    faces.append(tuple(faces_array[i]))
                else:  # Triangle
                    faces.append(tuple(faces_array[i, :3]))
        return faces

    # Fallback: Original Python implementation
    y_quads, x_quads = np.mgrid[0 : height - 1, 0 : width - 1]
    y_quads = y_quads.flatten()
    x_quads = x_quads.flatten()

    faces = []
    n_quads = len(y_quads)

    for batch_start in range(0, n_quads, batch_size):
        batch_end = min(batch_start + batch_size, n_quads)
        batch_y = y_quads[batch_start:batch_end]
        batch_x = x_quads[batch_start:batch_end]

        for i in range(batch_end - batch_start):
            y, x = batch_y[i], batch_x[i]
            quad_points = [(y, x), (y, x + 1), (y + 1, x + 1), (y + 1, x)]

            valid_indices = []
            for point in quad_points:
                if point in coord_to_index:
                    valid_indices.append(coord_to_index[point])

            if len(valid_indices) >= 3:
                faces.append(tuple(valid_indices))

    return faces


def _to_unit_rgba(colors):
    """Colors as float32 RGBA in 0-1. Integer dtypes are 0-255; floats above 1 are too."""
    colors = np.asarray(colors)
    unit = colors.astype(np.float32)
    if np.issubdtype(colors.dtype, np.integer) or (unit.size and unit.max() > 1.0):
        unit /= 255.0
    if unit.shape[-1] == 3:
        unit = np.concatenate([unit, np.ones(unit.shape[:-1] + (1,), dtype=np.float32)], axis=-1)
    return unit


def vertex_colors_rgba(n_vertices, colors=None, y_valid=None, x_valid=None, boundary_colors=None):
    """Per-vertex RGBA (0-1 float32) for a terrain mesh; uncolored vertices are white.

    Surface vertices (the first len(y_valid)) take colors[y, x] from an (H, W, C) grid, or
    colors[i] from an (N, C) per-vertex array. Skirt vertices after them take boundary_colors.
    """
    result = np.ones((n_vertices, 4), dtype=np.float32)
    n_surface = 0
    if colors is not None and y_valid is not None and x_valid is not None:
        unit = _to_unit_rgba(colors)
        n_surface = min(len(y_valid), n_vertices)
        ys, xs = np.asarray(y_valid[:n_surface]), np.asarray(x_valid[:n_surface])
        if unit.ndim == 3:
            inside = (ys >= 0) & (ys < unit.shape[0]) & (xs >= 0) & (xs < unit.shape[1])
            if not np.all(inside):
                # The color grid must be the DEM grid; a smaller one means misaligned colors
                raise ValueError(
                    f"{np.sum(~inside)} surface vertices lie outside the {unit.shape[:2]} "
                    "color grid; colors must be computed on the DEM grid"
                )
            result[:n_surface] = unit[ys, xs]
        else:
            n = min(n_surface, len(unit))
            result[:n] = unit[:n]
    elif y_valid is not None:
        n_surface = len(y_valid)
    if boundary_colors is not None:
        unit = _to_unit_rgba(boundary_colors)
        n = max(0, min(len(unit), n_vertices - n_surface))
        result[n_surface : n_surface + n] = unit[:n]
    return result


@dataclass
class MeshData:
    """A terrain mesh independent of Blender.

    vertices are (V, 3); the first len(y_valid) are surface vertices at grid cells
    (y_valid, x_valid), the rest are skirt vertices. colors is the (H, W, 4) color grid
    and boundary_colors the per-skirt-vertex colors, either may be None.
    """

    vertices: np.ndarray
    faces: list
    y_valid: np.ndarray
    x_valid: np.ndarray
    colors: Optional[np.ndarray] = None
    boundary_colors: Optional[np.ndarray] = None

    def vertex_colors(self):
        """Per-vertex RGBA in 0-1 (white where uncolored)."""
        return vertex_colors_rgba(
            len(self.vertices), self.colors, self.y_valid, self.x_valid, self.boundary_colors
        )
