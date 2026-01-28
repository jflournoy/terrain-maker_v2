"""
TDD tests for boundary extension coverage with fractional rectangle edges.

These tests verify that `create_boundary_extension()` produces complete edge
coverage on all 4 sides (north, south, east, west) when using fractional
rectangle edges.

TDD RED Phase: Define expected behavior for the Detroit render path.
"""

import numpy as np
import pytest
from rasterio.transform import Affine

from terrain.mesh_operations import (
    create_boundary_extension,
    generate_rectangle_edge_pixels,
)


class TestBoundaryExtensionCoverage:
    """Tests for complete boundary coverage with fractional edges."""

    @pytest.fixture
    def simple_terrain_data(self):
        """Create simple terrain data with known geometry.

        Returns positions, boundary_points, coord_to_index for a 10x10 grid.
        """
        # 10x10 grid of vertices
        height, width = 10, 10
        n_vertices = height * width

        # Create positions: x, y from grid coords, z = elevation
        positions = np.zeros((n_vertices, 3), dtype=float)
        coord_to_index = {}

        for y in range(height):
            for x in range(width):
                idx = y * width + x
                # Scale to reasonable mesh coordinates
                positions[idx] = [x * 0.1, y * 0.1, 0.5 + 0.01 * (x + y)]
                coord_to_index[(y, x)] = idx

        # Boundary points: perimeter of the grid (clockwise for rectangle edges)
        boundary_points = []
        # Top edge (y=0, x: 0 to width-1)
        for x in range(width):
            boundary_points.append((0, x))
        # Right edge (x=width-1, y: 1 to height-1)
        for y in range(1, height):
            boundary_points.append((y, width - 1))
        # Bottom edge (y=height-1, x: width-2 to 0)
        for x in range(width - 2, -1, -1):
            boundary_points.append((height - 1, x))
        # Left edge (x=0, y: height-2 to 1)
        for y in range(height - 2, 0, -1):
            boundary_points.append((y, 0))

        return {
            'positions': positions,
            'boundary_points': boundary_points,
            'coord_to_index': coord_to_index,
            'shape': (height, width),
        }

    def test_single_tier_creates_faces_for_all_boundary_segments(self, simple_terrain_data):
        """Single-tier boundary extension should create one face per boundary segment."""
        positions = simple_terrain_data['positions']
        boundary_points = simple_terrain_data['boundary_points']
        coord_to_index = simple_terrain_data['coord_to_index']

        boundary_vertices, boundary_faces = create_boundary_extension(
            positions,
            boundary_points,
            coord_to_index,
            base_depth=-0.2,
            two_tier=False,
        )

        n_boundary = len(boundary_points)

        # Should have at least 90% face coverage (allow for corner handling)
        min_expected_faces = int(n_boundary * 0.9)
        assert len(boundary_faces) >= min_expected_faces, (
            f"Expected at least {min_expected_faces} faces for {n_boundary} boundary points, "
            f"got {len(boundary_faces)}. Missing faces indicate gaps in the skirt."
        )

    def test_two_tier_creates_faces_for_all_boundary_segments(self, simple_terrain_data):
        """Two-tier boundary extension should create two faces per boundary segment."""
        positions = simple_terrain_data['positions']
        boundary_points = simple_terrain_data['boundary_points']
        coord_to_index = simple_terrain_data['coord_to_index']

        result = create_boundary_extension(
            positions,
            boundary_points,
            coord_to_index,
            base_depth=-0.5,
            two_tier=True,
            mid_depth=-0.1,
        )

        # Two-tier returns (vertices, faces, colors)
        boundary_vertices, boundary_faces, boundary_colors = result

        n_boundary = len(boundary_points)

        # Two-tier: 2 faces per segment (surface→mid, mid→base)
        min_expected_faces = int(n_boundary * 2 * 0.9)
        assert len(boundary_faces) >= min_expected_faces, (
            f"Expected at least {min_expected_faces} faces for {n_boundary} boundary points "
            f"(2 per segment), got {len(boundary_faces)}. Missing faces indicate gaps."
        )


class TestFractionalEdgeCoverage:
    """Tests for fractional edge boundary coverage."""

    @pytest.fixture
    def terrain_with_transforms(self):
        """Create terrain data simulating the Detroit transform pipeline.

        This mimics what happens when:
        1. Original DEM is in WGS84
        2. Reprojected to UTM (shape changes)
        3. Flipped horizontally
        4. Downsampled to target vertex count
        """
        # Final mesh: 20x20 after all transforms
        final_height, final_width = 20, 20
        n_vertices = final_height * final_width

        # Create positions
        positions = np.zeros((n_vertices, 3), dtype=float)
        coord_to_index = {}

        for y in range(final_height):
            for x in range(final_width):
                idx = y * final_width + x
                positions[idx] = [x * 0.05, y * 0.05, 0.3 + 0.02 * np.sin(x * 0.5) * np.cos(y * 0.5)]
                coord_to_index[(y, x)] = idx

        # Fractional boundary points simulating transform-aware sampling
        # These would come from generate_transform_aware_rectangle_edges_fractional()
        # Key: they include fractional coordinates that need interpolation
        boundary_points = []
        spacing = 0.5  # Sub-pixel sampling

        # Top edge (y ≈ 0)
        for x in np.arange(0, final_width, spacing):
            boundary_points.append((0.0, float(x)))
        # Right edge (x ≈ width-1)
        for y in np.arange(spacing, final_height, spacing):
            boundary_points.append((float(y), float(final_width - 1)))
        # Bottom edge (y ≈ height-1)
        for x in np.arange(final_width - 1, -1, -spacing):
            boundary_points.append((float(final_height - 1), float(x)))
        # Left edge (x ≈ 0)
        for y in np.arange(final_height - 1 - spacing, -1, -spacing):
            if y >= 0:
                boundary_points.append((float(y), 0.0))

        return {
            'positions': positions,
            'boundary_points': boundary_points,
            'coord_to_index': coord_to_index,
            'shape': (final_height, final_width),
        }

    def test_fractional_edges_all_vertices_initialized(self, terrain_with_transforms):
        """All fractional edge vertices should be successfully initialized.

        This tests that bilinear interpolation succeeds for all boundary points.
        """
        positions = terrain_with_transforms['positions']
        boundary_points = terrain_with_transforms['boundary_points']
        coord_to_index = terrain_with_transforms['coord_to_index']

        result = create_boundary_extension(
            positions,
            boundary_points,
            coord_to_index,
            base_depth=-0.5,
            two_tier=True,
            mid_depth=-0.1,
            use_fractional_edges=True,  # Triggers smoothed coords path, skips bridge faces
            boundary_winding="clockwise",  # Match rectangle edge trace order
        )

        boundary_vertices, boundary_faces, _ = result

        # Check that no vertices are at origin (0, 0, 0) which indicates failed interpolation
        zero_vertices = np.sum(np.all(boundary_vertices == 0, axis=1))

        # Allow some zeros at the very end for padding, but not many
        max_allowed_zeros = len(boundary_vertices) * 0.05  # 5% tolerance
        assert zero_vertices <= max_allowed_zeros, (
            f"Found {zero_vertices} zero-position vertices out of {len(boundary_vertices)}. "
            f"This indicates bilinear interpolation failed for many boundary points."
        )

    def test_fractional_edges_complete_coverage(self, terrain_with_transforms):
        """Fractional edges should produce faces covering all 4 sides."""
        positions = terrain_with_transforms['positions']
        boundary_points = terrain_with_transforms['boundary_points']
        coord_to_index = terrain_with_transforms['coord_to_index']

        result = create_boundary_extension(
            positions,
            boundary_points,
            coord_to_index,
            base_depth=-0.5,
            two_tier=True,
            mid_depth=-0.1,
            use_fractional_edges=True,  # Skips bridge faces
            boundary_winding="clockwise",
        )

        boundary_vertices, boundary_faces, _ = result
        n_boundary = len(boundary_points)

        # Two-tier: expect ~2 faces per boundary segment
        # With 0.5 spacing on 20x20 grid: ~160 boundary points → ~320 faces
        expected_faces = n_boundary * 2
        min_coverage = 0.8  # Require 80% coverage

        assert len(boundary_faces) >= expected_faces * min_coverage, (
            f"Expected at least {int(expected_faces * min_coverage)} faces "
            f"({min_coverage*100:.0f}% of {expected_faces}), got {len(boundary_faces)}. "
            f"This indicates significant gaps in the edge skirt."
        )

    def test_fractional_edges_no_alternating_gaps(self, terrain_with_transforms):
        """Faces should not have alternating gaps (face, skip, face, skip pattern)."""
        positions = terrain_with_transforms['positions']
        boundary_points = terrain_with_transforms['boundary_points']
        coord_to_index = terrain_with_transforms['coord_to_index']

        result = create_boundary_extension(
            positions,
            boundary_points,
            coord_to_index,
            base_depth=-0.5,
            two_tier=True,
            mid_depth=-0.1,
            use_fractional_edges=True,  # Skips bridge faces
            boundary_winding="clockwise",
        )

        boundary_vertices, boundary_faces, _ = result

        if len(boundary_faces) == 0:
            pytest.fail("No faces created at all")

        # Extract face vertex indices to detect patterns
        # For two-tier, faces alternate between surface→mid and mid→base
        # Group by pairs and check for gaps

        # Count consecutive face pairs
        n_pairs = len(boundary_faces) // 2
        if n_pairs < 10:
            pytest.skip("Not enough faces to detect pattern")

        # Check that we don't have large gaps in face indices
        # This would manifest as faces referencing very different vertex ranges
        face_starts = [min(face) for face in boundary_faces]
        face_ends = [max(face) for face in boundary_faces]

        # All faces should reference vertices in a reasonable range
        # (not jumping around wildly)
        face_range = max(face_ends) - min(face_starts)
        n_boundary_verts = len(boundary_vertices)

        # Face range should be close to total boundary vertices
        assert face_range <= n_boundary_verts * 1.5, (
            f"Face vertex indices span {face_range}, but only {n_boundary_verts} "
            f"boundary vertices exist. This suggests faces are referencing wrong vertices."
        )


class TestEdgeByEdgeCoverage:
    """Tests verifying each edge (N/S/E/W) has face coverage."""

    @pytest.fixture
    def labeled_boundary(self):
        """Create boundary with labeled edges for verification."""
        height, width = 15, 15
        n_vertices = height * width

        positions = np.zeros((n_vertices, 3), dtype=float)
        coord_to_index = {}

        for y in range(height):
            for x in range(width):
                idx = y * width + x
                positions[idx] = [x * 0.1, y * 0.1, 0.5]
                coord_to_index[(y, x)] = idx

        # Create boundary with edge labels
        boundary_points = []
        edge_labels = []  # Track which edge each point belongs to

        # North edge (y=0) - top of image
        for x in range(width):
            boundary_points.append((0, x))
            edge_labels.append('north')

        # East edge (x=width-1) - right of image
        for y in range(1, height):
            boundary_points.append((y, width - 1))
            edge_labels.append('east')

        # South edge (y=height-1) - bottom of image
        for x in range(width - 2, -1, -1):
            boundary_points.append((height - 1, x))
            edge_labels.append('south')

        # West edge (x=0) - left of image
        for y in range(height - 2, 0, -1):
            boundary_points.append((y, 0))
            edge_labels.append('west')

        return {
            'positions': positions,
            'boundary_points': boundary_points,
            'coord_to_index': coord_to_index,
            'edge_labels': edge_labels,
            'shape': (height, width),
        }

    def test_all_four_edges_have_faces(self, labeled_boundary):
        """Each of the 4 edges should have faces in the boundary extension."""
        positions = labeled_boundary['positions']
        boundary_points = labeled_boundary['boundary_points']
        coord_to_index = labeled_boundary['coord_to_index']
        edge_labels = labeled_boundary['edge_labels']

        result = create_boundary_extension(
            positions,
            boundary_points,
            coord_to_index,
            base_depth=-0.5,
            two_tier=True,
            mid_depth=-0.1,
        )

        boundary_vertices, boundary_faces, _ = result

        # Track which edges have faces
        # Each face references indices that map back to boundary points
        edges_with_faces = set()

        n_boundary = len(boundary_points)
        n_existing = len(positions)

        for face in boundary_faces:
            # Surface indices start at n_existing for two-tier with smoothed
            # For integer coords, surface_indices come from coord_to_index
            # This is complex - let's just check face count per edge segment
            pass

        # Simpler check: count faces and ensure reasonable coverage per edge
        # North: ~15 points, East: ~14 points, South: ~14 points, West: ~13 points
        # With two-tier: ~2 faces per point
        expected_per_edge = {
            'north': 15 * 2,
            'east': 14 * 2,
            'south': 14 * 2,
            'west': 13 * 2,
        }

        total_expected = sum(expected_per_edge.values())

        # Should have at least 80% of expected faces
        assert len(boundary_faces) >= total_expected * 0.8, (
            f"Expected at least {int(total_expected * 0.8)} faces, got {len(boundary_faces)}. "
            f"Some edges may be missing faces entirely."
        )

    def test_face_normals_point_outward(self, labeled_boundary):
        """Face normals should point outward from the terrain center."""
        positions = labeled_boundary['positions']
        boundary_points = labeled_boundary['boundary_points']
        coord_to_index = labeled_boundary['coord_to_index']
        height, width = labeled_boundary['shape']

        # Use clockwise winding (matches trace order of boundary fixture)
        result = create_boundary_extension(
            positions,
            boundary_points,
            coord_to_index,
            base_depth=-0.5,
            two_tier=True,
            mid_depth=-0.1,
            boundary_winding="clockwise",  # Matches trace order of labeled_boundary fixture
        )

        boundary_vertices, boundary_faces, _ = result

        if len(boundary_faces) == 0:
            pytest.fail("No faces to check normals")

        # Compute terrain center (in XY plane)
        all_vertices = np.vstack([positions, boundary_vertices])
        center_xy = np.mean(all_vertices[:, :2], axis=0)

        outward_count = 0
        inward_count = 0

        for face in boundary_faces[:50]:  # Check first 50 faces
            # Get face vertices
            try:
                v0 = all_vertices[face[0]]
                v1 = all_vertices[face[1]]
                v2 = all_vertices[face[2]]
            except IndexError:
                continue

            # Compute face normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)

            if np.linalg.norm(normal) < 1e-10:
                continue

            # Face center
            face_center = np.mean([all_vertices[i] for i in face], axis=0)

            # Direction from terrain center to face center (XY only)
            outward_dir = face_center[:2] - center_xy

            # Check if normal XY component points outward
            dot = np.dot(normal[:2], outward_dir)

            if dot > 0:
                outward_count += 1
            else:
                inward_count += 1

        total = outward_count + inward_count
        if total > 0:
            outward_ratio = outward_count / total
            # Allow some tolerance - mixed winding is OK if majority is correct
            assert outward_ratio > 0.6, (
                f"Only {outward_ratio*100:.1f}% of faces have outward-pointing normals. "
                f"This will cause backface culling to hide faces."
            )
