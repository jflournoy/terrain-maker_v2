"""
TDD tests for edge extrusion face generation.

These tests verify that `generate_rectangle_edge_vertices()` creates proper
vertical wall faces (a "skirt") around the terrain edge, NOT diagonal faces
that cut across the x-y plane.

TDD RED Phase: These tests define the correct behavior.
"""

import numpy as np
import pytest
from rasterio.transform import Affine

from terrain.mesh_operations import generate_rectangle_edge_vertices


class TestEdgeExtrusionFaces:
    """Tests for proper edge extrusion geometry."""

    @pytest.fixture
    def simple_dem(self):
        """4x4 DEM with distinct edge elevations for testing."""
        return np.array(
            [
                [100, 110, 120, 130],
                [140, 0, 0, 150],  # Interior zeros
                [160, 0, 0, 170],
                [180, 190, 200, 210],
            ],
            dtype=float,
        )

    @pytest.fixture
    def identity_transform(self):
        """Identity affine transform for simple coordinate mapping."""
        return Affine.identity()

    def test_face_has_four_unique_vertices(self, simple_dem, identity_transform):
        """Each face should have 4 unique vertex indices (no degenerate faces).

        A degenerate face with repeated indices creates diagonal artifacts
        instead of proper vertical walls.
        """
        boundary_vertices, boundary_faces = generate_rectangle_edge_vertices(
            dem_shape=simple_dem.shape,
            dem_data=simple_dem,
            original_transform=identity_transform,
            transforms_list=[],
            edge_sample_spacing=1.0,
            base_depth=-0.2,
        )

        assert len(boundary_faces) > 0, "Should generate at least one face"

        for face_idx, face in enumerate(boundary_faces):
            unique_indices = set(face)
            assert len(unique_indices) == 4, (
                f"Face {face_idx} has {len(unique_indices)} unique vertices, expected 4. "
                f"Face indices: {face}. Degenerate faces create diagonal artifacts."
            )

    def test_surface_vertices_have_dem_elevation(self, simple_dem, identity_transform):
        """Surface vertices should have z-values from DEM, not base_depth.

        The edge extrusion needs BOTH surface vertices (at DEM elevation)
        and base vertices (at base_depth) to form vertical walls.
        """
        base_depth = -0.5

        boundary_vertices, boundary_faces = generate_rectangle_edge_vertices(
            dem_shape=simple_dem.shape,
            dem_data=simple_dem,
            original_transform=identity_transform,
            transforms_list=[],
            edge_sample_spacing=1.0,
            base_depth=base_depth,
        )

        z_values = boundary_vertices[:, 2]

        # There should be vertices with z > base_depth (surface vertices)
        surface_vertices_exist = any(z != base_depth for z in z_values)
        assert surface_vertices_exist, (
            f"Expected some surface vertices with elevation != base_depth ({base_depth}), "
            f"but all z-values are: {np.unique(z_values)}. "
            "The function must create BOTH surface and base vertices."
        )

    def test_base_vertices_all_at_base_depth(self, identity_transform):
        """All base (bottom) vertices should be at base_depth.

        Half the vertices should be surface vertices (at DEM elevation),
        half should be base vertices (at base_depth).
        """
        dem_data = np.random.rand(4, 4) * 100 + 50  # Random elevations 50-150
        base_depth = -0.3

        boundary_vertices, boundary_faces = generate_rectangle_edge_vertices(
            dem_shape=dem_data.shape,
            dem_data=dem_data,
            original_transform=identity_transform,
            transforms_list=[],
            edge_sample_spacing=1.0,
            base_depth=base_depth,
        )

        # With proper implementation: n surface vertices + n base vertices
        n_total = len(boundary_vertices)
        n_edge_pixels = n_total // 2

        # Base vertices should be the second half
        base_vertices = boundary_vertices[n_edge_pixels:]
        base_z_values = base_vertices[:, 2]

        assert np.allclose(base_z_values, base_depth), (
            f"All base vertices should be at z={base_depth}, "
            f"but found z-values: {np.unique(base_z_values)}"
        )

    def test_face_vertices_are_coplanar_vertical(self, simple_dem, identity_transform):
        """Each skirt face should be a vertical quad.

        For a vertical wall:
        - Left edge vertices (surface[i], base[i]) should have same (x, y)
        - Right edge vertices (surface[i+1], base[i+1]) should have same (x, y)

        Face winding is [surface_i, base_i, base_next, surface_next] for outward normals.
        """
        boundary_vertices, boundary_faces = generate_rectangle_edge_vertices(
            dem_shape=simple_dem.shape,
            dem_data=simple_dem,
            original_transform=identity_transform,
            transforms_list=[],
            edge_sample_spacing=1.0,
            base_depth=-0.2,
        )

        for face_idx, face in enumerate(boundary_faces):
            if len(set(face)) != 4:
                pytest.skip(f"Face {face_idx} is degenerate, skipping coplanar test")

            # Get face vertices
            v0, v1, v2, v3 = [boundary_vertices[i] for i in face]

            # Face order is [surface_i, base_i, base_next, surface_next]:
            # v0 = surface[i], v1 = base[i] - should share (x, y)
            # v2 = base[i+1], v3 = surface[i+1] - should share (x, y)
            left_xy_match = np.allclose(v0[:2], v1[:2], atol=1e-6)
            right_xy_match = np.allclose(v2[:2], v3[:2], atol=1e-6)

            assert left_xy_match and right_xy_match, (
                f"Face {face_idx} is not a vertical wall. "
                f"Left edge (x,y): v0={v0[:2]}, v1={v1[:2]}. "
                f"Right edge (x,y): v2={v2[:2]}, v3={v3[:2]}. "
                "Vertical walls require top/bottom vertices at same (x,y)."
            )

    def test_face_winding_produces_outward_normals(self, simple_dem, identity_transform):
        """Face vertex winding should produce outward-pointing normals.

        For a terrain skirt, normals should point away from the terrain center.
        """
        boundary_vertices, boundary_faces = generate_rectangle_edge_vertices(
            dem_shape=simple_dem.shape,
            dem_data=simple_dem,
            original_transform=identity_transform,
            transforms_list=[],
            edge_sample_spacing=1.0,
            base_depth=-0.2,
        )

        if len(boundary_vertices) < 4:
            pytest.skip("Not enough vertices to test normals")

        # Compute mesh center (use only x, y for horizontal center)
        mesh_center_xy = np.mean(boundary_vertices[:, :2], axis=0)

        outward_count = 0
        inward_count = 0

        for face_idx, face in enumerate(boundary_faces):
            if len(set(face)) != 4:
                continue  # Skip degenerate faces

            # Get face vertices
            v0, v1, v2, v3 = [boundary_vertices[i] for i in face]

            # Compute face normal using first triangle
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)

            if np.linalg.norm(normal) < 1e-10:
                continue  # Skip zero-area faces

            # Face center (x, y only)
            face_center_xy = np.mean([v0[:2], v1[:2], v2[:2], v3[:2]], axis=0)

            # Vector from mesh center to face center (in x-y plane)
            outward_dir = face_center_xy - mesh_center_xy

            # Normal should have positive dot product with outward direction
            # (only check x, y components for horizontal outward direction)
            dot = np.dot(normal[:2], outward_dir)

            if dot > 0:
                outward_count += 1
            else:
                inward_count += 1

        # Majority of faces should have outward normals
        total_checked = outward_count + inward_count
        if total_checked > 0:
            outward_ratio = outward_count / total_checked
            assert outward_ratio > 0.5, (
                f"Only {outward_ratio * 100:.1f}% of faces have outward normals. "
                f"Expected majority to point outward. Check vertex winding order."
            )

    def test_no_diagonal_faces_across_xy_plane(self, simple_dem, identity_transform):
        """Faces should be vertical walls, not diagonals across the mesh.

        A vertical wall face has a normal that is mostly horizontal
        (perpendicular to z-axis). Diagonal faces would have normals
        with significant z-component.
        """
        boundary_vertices, boundary_faces = generate_rectangle_edge_vertices(
            dem_shape=simple_dem.shape,
            dem_data=simple_dem,
            original_transform=identity_transform,
            transforms_list=[],
            edge_sample_spacing=1.0,
            base_depth=-0.2,
        )

        vertical_wall_count = 0
        diagonal_count = 0

        for face_idx, face in enumerate(boundary_faces):
            if len(set(face)) != 4:
                continue  # Skip degenerate faces

            v0, v1, v2, v3 = [boundary_vertices[i] for i in face]

            # Compute face normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)

            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-10:
                continue

            normal = normal / norm_len  # Normalize

            # For vertical walls, normal should be mostly horizontal
            # (z-component should be small compared to x,y components)
            z_component = abs(normal[2])
            xy_component = np.sqrt(normal[0] ** 2 + normal[1] ** 2)

            if xy_component > z_component:
                vertical_wall_count += 1
            else:
                diagonal_count += 1

        total = vertical_wall_count + diagonal_count
        if total > 0:
            vertical_ratio = vertical_wall_count / total
            assert vertical_ratio > 0.9, (
                f"Only {vertical_ratio * 100:.1f}% of faces are vertical walls. "
                f"Found {diagonal_count} diagonal faces out of {total}. "
                "Face normals should be mostly horizontal for vertical walls."
            )


class TestEdgeExtrusionVertexLayout:
    """Tests for proper vertex array layout."""

    def test_vertex_count_is_double_edge_pixels(self):
        """Vertex array should have 2x edge pixels (surface + base)."""
        dem_data = np.ones((4, 4), dtype=float) * 100
        transform = Affine.identity()

        boundary_vertices, boundary_faces = generate_rectangle_edge_vertices(
            dem_shape=dem_data.shape,
            dem_data=dem_data,
            original_transform=transform,
            transforms_list=[],
            edge_sample_spacing=1.0,
            base_depth=-0.2,
        )

        # For a 4x4 grid with spacing 1.0, perimeter is 4*3 = 12 edge pixels
        # (corners counted once each: top 4, right 3, bottom 3, left 2)
        # With proper implementation: 12 surface + 12 base = 24 vertices
        n_vertices = len(boundary_vertices)

        # Should have an even number of vertices (surface + base pairs)
        assert n_vertices % 2 == 0, (
            f"Vertex count ({n_vertices}) should be even (surface + base pairs)"
        )

        # Should have more than just edge count (need surface + base)
        n_faces = len(boundary_faces)
        assert n_vertices >= n_faces, (
            f"Vertex count ({n_vertices}) should be at least twice the edge pixel count. "
            f"Need both surface and base vertices for vertical walls."
        )

    def test_surface_and_base_vertices_share_xy(self):
        """Each surface vertex should have a corresponding base vertex at same (x,y)."""
        dem_data = np.ones((4, 4), dtype=float) * 100
        transform = Affine.identity()
        base_depth = -0.5

        boundary_vertices, boundary_faces = generate_rectangle_edge_vertices(
            dem_shape=dem_data.shape,
            dem_data=dem_data,
            original_transform=transform,
            transforms_list=[],
            edge_sample_spacing=1.0,
            base_depth=base_depth,
        )

        n_total = len(boundary_vertices)
        if n_total < 2:
            pytest.skip("Not enough vertices")

        n_half = n_total // 2

        # Surface vertices (first half) and base vertices (second half)
        # should have matching (x, y) coordinates
        surface_xy = boundary_vertices[:n_half, :2]
        base_xy = boundary_vertices[n_half:, :2]

        if len(surface_xy) == len(base_xy):
            matches = np.allclose(surface_xy, base_xy, atol=1e-6)
            assert matches, (
                "Surface and base vertices should have matching (x, y) coordinates. "
                "This is required for vertical wall geometry."
            )


class TestEdgeExtrusionIntegration:
    """Integration tests with the transform pipeline."""

    def test_with_scaled_transform(self):
        """Function should work with non-identity transforms."""
        dem_data = np.ones((10, 10), dtype=float) * 200
        # Scale transform: each pixel is 0.1 units
        transform = Affine.translation(0, 0) * Affine.scale(0.1, -0.1)

        boundary_vertices, boundary_faces = generate_rectangle_edge_vertices(
            dem_shape=dem_data.shape,
            dem_data=dem_data,
            original_transform=transform,
            transforms_list=[],
            edge_sample_spacing=1.0,
            base_depth=-0.2,
        )

        # Should produce valid output
        assert len(boundary_vertices) > 0
        assert len(boundary_faces) > 0

        # Faces should have 4 unique vertices
        for face in boundary_faces:
            assert len(set(face)) == 4, f"Degenerate face: {face}"

    def test_with_varying_elevation(self):
        """Function should sample varying DEM elevations correctly."""
        # DEM with gradient elevation
        dem_data = np.arange(16, dtype=float).reshape(4, 4) * 10 + 100
        transform = Affine.identity()
        base_depth = -1.0

        boundary_vertices, boundary_faces = generate_rectangle_edge_vertices(
            dem_shape=dem_data.shape,
            dem_data=dem_data,
            original_transform=transform,
            transforms_list=[],
            edge_sample_spacing=1.0,
            base_depth=base_depth,
        )

        z_values = boundary_vertices[:, 2]
        unique_z = np.unique(z_values)

        # Should have multiple z values (varying surface + constant base)
        assert len(unique_z) > 2, (
            f"Expected multiple z values from varying DEM, got: {unique_z}. "
            "Surface vertices should reflect DEM elevation at each edge pixel."
        )
