"""
Tests for blender_integration module.

Tests Blender-specific mesh creation and material application functions.
"""

import pytest

# These tests require Blender environment
pytest.importorskip("bpy")

import numpy as np


class TestBlenderMeshCreation:
    """Tests for create_blender_mesh function."""

    def test_create_blender_mesh_imports(self):
        """Test that create_blender_mesh can be imported."""
        from src.terrain.blender_integration import create_blender_mesh

        assert callable(create_blender_mesh)

    def test_create_blender_mesh_basic(self):
        """Test basic Blender mesh creation from vertices and faces."""
        import bpy
        from src.terrain.blender_integration import create_blender_mesh

        # Simple triangle
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = [(0, 1, 2)]

        obj = create_blender_mesh(vertices, faces, name="TestMesh")

        # Should create object
        assert obj is not None
        assert obj.name == "TestMesh"

        # Should have correct geometry
        assert len(obj.data.vertices) == 3
        assert len(obj.data.polygons) == 1

        # Cleanup (save mesh reference before removing object)
        mesh_data = obj.data
        bpy.data.objects.remove(obj)
        bpy.data.meshes.remove(mesh_data)

    def test_create_blender_mesh_with_colors(self):
        """Test mesh creation with vertex colors."""
        import bpy
        from src.terrain.blender_integration import create_blender_mesh

        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = [(0, 1, 2)]

        # RGB colors (uint8)
        colors = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
        y_valid = np.array([0, 0, 0])
        x_valid = np.array([0, 1, 2])

        obj = create_blender_mesh(
            vertices, faces, colors=colors, y_valid=y_valid, x_valid=x_valid, name="ColoredMesh"
        )

        # Should have vertex color layer
        assert len(obj.data.vertex_colors) > 0

        # Cleanup (save mesh reference before removing object)
        mesh_data = obj.data
        bpy.data.objects.remove(obj)
        bpy.data.meshes.remove(mesh_data)

    def test_create_blender_mesh_returns_object(self):
        """Test that function returns bpy Object."""
        import bpy
        from src.terrain.blender_integration import create_blender_mesh

        vertices = np.array([[0, 0, 0]], dtype=float)
        faces = []

        obj = create_blender_mesh(vertices, faces)

        assert isinstance(obj, bpy.types.Object)

        # Cleanup (save mesh reference before removing object)
        mesh_data = obj.data
        bpy.data.objects.remove(obj)
        bpy.data.meshes.remove(mesh_data)
