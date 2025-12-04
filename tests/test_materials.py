"""
Tests for material and shader operations.

Tests Blender material setup, shader nodes, and background plane creation.
"""

import pytest

# These tests require Blender environment
pytest.importorskip("bpy")

import numpy as np


class TestApplyColormapMaterial:
    """Tests for apply_colormap_material function."""

    def test_apply_colormap_material_imports(self):
        """Test that apply_colormap_material can be imported."""
        from src.terrain.materials import apply_colormap_material

        assert callable(apply_colormap_material)

    def test_apply_colormap_material_creates_nodes(self):
        """Test that material nodes are created."""
        import bpy
        from src.terrain.materials import apply_colormap_material

        # Create material
        mat = bpy.data.materials.new("TestMaterial")
        mat.use_nodes = True

        apply_colormap_material(mat)

        # Should have nodes in the tree
        assert len(mat.node_tree.nodes) > 0

        # Should have essential nodes
        node_types = [node.bl_idname for node in mat.node_tree.nodes]
        assert "ShaderNodeOutputMaterial" in node_types
        assert "ShaderNodeBsdfPrincipled" in node_types
        assert "ShaderNodeEmission" in node_types
        assert "ShaderNodeVertexColor" in node_types

        # Cleanup
        bpy.data.materials.remove(mat)

    def test_apply_colormap_material_sets_vertex_color_layer(self):
        """Test that vertex color layer is set."""
        import bpy
        from src.terrain.materials import apply_colormap_material

        mat = bpy.data.materials.new("TestMaterial")
        mat.use_nodes = True

        apply_colormap_material(mat)

        # Find vertex color node
        vertex_color_nodes = [n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeVertexColor"]
        assert len(vertex_color_nodes) == 1
        assert vertex_color_nodes[0].layer_name == "TerrainColors"

        # Cleanup
        bpy.data.materials.remove(mat)

    def test_apply_colormap_material_creates_connections(self):
        """Test that shader nodes are properly connected."""
        import bpy
        from src.terrain.materials import apply_colormap_material

        mat = bpy.data.materials.new("TestMaterial")
        mat.use_nodes = True

        apply_colormap_material(mat)

        # Should have connections
        assert len(mat.node_tree.links) > 0

        # Cleanup
        bpy.data.materials.remove(mat)


class TestApplyWaterShader:
    """Tests for apply_water_shader function."""

    def test_apply_water_shader_imports(self):
        """Test that apply_water_shader can be imported."""
        from src.terrain.materials import apply_water_shader

        assert callable(apply_water_shader)

    def test_apply_water_shader_creates_nodes(self):
        """Test that water shader nodes are created."""
        import bpy
        from src.terrain.materials import apply_water_shader

        mat = bpy.data.materials.new("WaterMaterial")
        mat.use_nodes = True

        apply_water_shader(mat)

        # Should have nodes in the tree
        assert len(mat.node_tree.nodes) > 0

        # Should have essential nodes for water shader
        node_types = [node.bl_idname for node in mat.node_tree.nodes]
        assert "ShaderNodeOutputMaterial" in node_types
        assert "ShaderNodeBsdfPrincipled" in node_types
        assert "ShaderNodeEmission" in node_types
        assert "ShaderNodeVertexColor" in node_types
        assert "ShaderNodeMixRGB" in node_types

        # Cleanup
        bpy.data.materials.remove(mat)

    def test_apply_water_shader_custom_color(self):
        """Test water shader with custom water color."""
        import bpy
        from src.terrain.materials import apply_water_shader

        mat = bpy.data.materials.new("WaterMaterial")
        mat.use_nodes = True

        custom_color = (0.0, 1.0, 0.0)  # Green water
        apply_water_shader(mat, water_color=custom_color)

        # Find MixRGB node and check color
        mix_rgb_nodes = [n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeMixRGB"]
        assert len(mix_rgb_nodes) == 1

        # Water color should be in Color1 input
        water_rgba = mix_rgb_nodes[0].inputs["Color1"].default_value
        assert water_rgba[0] == custom_color[0]
        assert water_rgba[1] == custom_color[1]
        assert water_rgba[2] == custom_color[2]

        # Cleanup
        bpy.data.materials.remove(mat)

    def test_apply_water_shader_default_color(self):
        """Test water shader with default blue color."""
        import bpy
        from src.terrain.materials import apply_water_shader

        mat = bpy.data.materials.new("WaterMaterial")
        mat.use_nodes = True

        apply_water_shader(mat)

        # Find MixRGB node
        mix_rgb_nodes = [n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeMixRGB"]
        water_rgba = mix_rgb_nodes[0].inputs["Color1"].default_value

        # Should be blue-ish (default)
        assert water_rgba[2] > water_rgba[0]  # Blue > Red
        assert water_rgba[2] > water_rgba[1]  # Blue > Green

        # Cleanup
        bpy.data.materials.remove(mat)


class TestCreateBackgroundPlane:
    """Tests for create_background_plane function."""

    def test_create_background_plane_imports(self):
        """Test that create_background_plane can be imported."""
        from src.terrain.materials import create_background_plane

        assert callable(create_background_plane)

    def test_create_background_plane_basic(self):
        """Test basic background plane creation."""
        import bpy
        from src.terrain.materials import create_background_plane

        # Clear scene
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj)

        # Create simple terrain object
        bpy.ops.mesh.primitive_cube_add()
        terrain_obj = bpy.context.active_object

        plane = create_background_plane(terrain_obj)

        # Should create object
        assert plane is not None
        assert "BackgroundPlane" in plane.name

        # Should have mesh data
        assert plane.data is not None
        assert len(plane.data.vertices) == 4
        assert len(plane.data.polygons) == 1

        # Cleanup
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj)
        for mesh in list(bpy.data.meshes):
            bpy.data.meshes.remove(mesh)
        for mat in list(bpy.data.materials):
            bpy.data.materials.remove(mat)

    def test_create_background_plane_with_custom_depth(self):
        """Test background plane with custom depth."""
        import bpy
        from src.terrain.materials import create_background_plane

        # Clear scene
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj)

        # Create terrain
        bpy.ops.mesh.primitive_cube_add()
        terrain_obj = bpy.context.active_object

        custom_depth = -5.0
        plane = create_background_plane(terrain_obj, depth=custom_depth)

        # Check depth (Z coordinate of first vertex)
        assert plane.data.vertices[0].co[2] == custom_depth

        # Cleanup
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj)
        for mesh in list(bpy.data.meshes):
            bpy.data.meshes.remove(mesh)
        for mat in list(bpy.data.materials):
            bpy.data.materials.remove(mat)

    def test_create_background_plane_with_scale_factor(self):
        """Test background plane with custom scale factor."""
        import bpy
        from src.terrain.materials import create_background_plane

        # Clear scene
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj)

        # Create terrain (default cube is 2x2)
        bpy.ops.mesh.primitive_cube_add()
        terrain_obj = bpy.context.active_object

        scale_factor = 3.0
        plane = create_background_plane(terrain_obj, scale_factor=scale_factor)

        # Plane should be larger than terrain
        # Check that plane vertices span more than the terrain
        plane_verts = [v.co for v in plane.data.vertices]
        x_coords = [v[0] for v in plane_verts]
        y_coords = [v[1] for v in plane_verts]

        plane_x_size = max(x_coords) - min(x_coords)
        plane_y_size = max(y_coords) - min(y_coords)

        # Default cube terrain is 2x2, plane should be roughly 2 * scale_factor
        assert plane_x_size > 2.0
        assert plane_y_size > 2.0

        # Cleanup
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj)
        for mesh in list(bpy.data.meshes):
            bpy.data.meshes.remove(mesh)
        for mat in list(bpy.data.materials):
            bpy.data.materials.remove(mat)

    def test_create_background_plane_has_material(self):
        """Test that background plane has material."""
        import bpy
        from src.terrain.materials import create_background_plane

        # Clear scene
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj)

        # Create terrain
        bpy.ops.mesh.primitive_cube_add()
        terrain_obj = bpy.context.active_object

        plane = create_background_plane(terrain_obj)

        # Should have material
        assert len(plane.data.materials) > 0
        mat = plane.data.materials[0]
        assert mat.use_nodes is True

        # Should have shader nodes
        assert len(mat.node_tree.nodes) > 0

        # Cleanup
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj)
        for mesh in list(bpy.data.meshes):
            bpy.data.meshes.remove(mesh)
        for mat in list(bpy.data.materials):
            bpy.data.materials.remove(mat)

    def test_create_background_plane_custom_material_params(self):
        """Test background plane with custom material parameters."""
        import bpy
        from src.terrain.materials import create_background_plane

        # Clear scene
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj)

        # Create terrain
        bpy.ops.mesh.primitive_cube_add()
        terrain_obj = bpy.context.active_object

        custom_params = {"emission_strength": 2.0, "roughness": 0.5}
        plane = create_background_plane(terrain_obj, material_params=custom_params)

        # Should have material
        mat = plane.data.materials[0]

        # Find principled shader
        principled_nodes = [n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"]
        assert len(principled_nodes) == 1

        principled = principled_nodes[0]
        assert principled.inputs["Emission Strength"].default_value == 2.0
        assert principled.inputs["Roughness"].default_value == 0.5

        # Cleanup
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj)
        for mesh in list(bpy.data.meshes):
            bpy.data.meshes.remove(mesh)
        for mat in list(bpy.data.materials):
            bpy.data.materials.remove(mat)

    def test_create_background_plane_none_terrain_raises(self):
        """Test that None terrain object raises ValueError."""
        from src.terrain.materials import create_background_plane

        with pytest.raises(ValueError):
            create_background_plane(None)
