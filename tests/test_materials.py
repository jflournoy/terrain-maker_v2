"""
Tests for material and shader operations.

Tests Blender material setup, shader nodes, and background plane creation.
"""

import pytest


# =============================================================================
# Tests for BASE_MATERIALS and get_base_material_color (no Blender required)
# =============================================================================


class TestBaseMaterials:
    """Tests for BASE_MATERIALS dictionary and get_base_material_color function."""

    def test_base_materials_exists(self):
        """Test that BASE_MATERIALS dictionary exists."""
        from src.terrain.materials import BASE_MATERIALS

        assert isinstance(BASE_MATERIALS, dict)
        assert len(BASE_MATERIALS) > 0

    def test_base_materials_all_presets(self):
        """Test that all expected material presets are present."""
        from src.terrain.materials import BASE_MATERIALS

        expected_materials = ["clay", "obsidian", "chrome", "plastic", "gold", "ivory"]
        for material in expected_materials:
            assert material in BASE_MATERIALS, f"Missing material: {material}"

    def test_base_materials_values_are_rgb_tuples(self):
        """Test that all material values are RGB tuples."""
        from src.terrain.materials import BASE_MATERIALS

        for name, rgb in BASE_MATERIALS.items():
            assert isinstance(rgb, tuple), f"{name} value is not a tuple"
            assert len(rgb) == 3, f"{name} RGB tuple should have 3 values"
            for component in rgb:
                assert isinstance(component, (int, float)), f"{name} has non-numeric component"
                assert 0.0 <= component <= 1.0, f"{name} has out-of-range component"

    def test_base_materials_expected_colors(self):
        """Test that material presets have expected RGB values."""
        from src.terrain.materials import BASE_MATERIALS

        # Test a few known values
        assert BASE_MATERIALS["clay"] == (0.5, 0.48, 0.45)
        assert BASE_MATERIALS["obsidian"] == (0.02, 0.02, 0.02)
        assert BASE_MATERIALS["chrome"] == (0.9, 0.9, 0.92)
        assert BASE_MATERIALS["plastic"] == (0.95, 0.95, 0.95)
        assert BASE_MATERIALS["gold"] == (1.0, 0.766, 0.336)
        assert BASE_MATERIALS["ivory"] == (0.95, 0.93, 0.88)


class TestGetBaseMaterialColor:
    """Tests for get_base_material_color function."""

    def test_get_base_material_color_exists(self):
        """Test that get_base_material_color function exists."""
        from src.terrain.materials import get_base_material_color

        assert callable(get_base_material_color)

    def test_get_base_material_color_preset_lowercase(self):
        """Test resolving preset material by lowercase name."""
        from src.terrain.materials import get_base_material_color

        result = get_base_material_color("clay")
        assert result == (0.5, 0.48, 0.45)

        result = get_base_material_color("gold")
        assert result == (1.0, 0.766, 0.336)

    def test_get_base_material_color_case_insensitive(self):
        """Test that material lookup is case-insensitive."""
        from src.terrain.materials import get_base_material_color

        # Test various cases
        assert get_base_material_color("Clay") == (0.5, 0.48, 0.45)
        assert get_base_material_color("CLAY") == (0.5, 0.48, 0.45)
        assert get_base_material_color("cLaY") == (0.5, 0.48, 0.45)

        assert get_base_material_color("Gold") == (1.0, 0.766, 0.336)
        assert get_base_material_color("GOLD") == (1.0, 0.766, 0.336)

    def test_get_base_material_color_rgb_tuple_passthrough(self):
        """Test that RGB tuples are passed through unchanged."""
        from src.terrain.materials import get_base_material_color

        custom_rgb = (0.6, 0.55, 0.5)
        result = get_base_material_color(custom_rgb)
        assert result == custom_rgb

        another_rgb = (0.1, 0.2, 0.3)
        result = get_base_material_color(another_rgb)
        assert result == another_rgb

    def test_get_base_material_color_invalid_name_raises(self):
        """Test that invalid material name raises ValueError."""
        from src.terrain.materials import get_base_material_color

        with pytest.raises(ValueError) as exc_info:
            get_base_material_color("invalid_material")

        # Check error message is helpful
        error_msg = str(exc_info.value)
        assert "Unknown color preset" in error_msg
        assert "invalid_material" in error_msg
        assert "Valid options" in error_msg

    def test_get_base_material_color_all_presets(self):
        """Test that all preset materials resolve correctly."""
        from src.terrain.materials import get_base_material_color, BASE_MATERIALS

        for material_name, expected_rgb in BASE_MATERIALS.items():
            result = get_base_material_color(material_name)
            assert result == expected_rgb, f"Material {material_name} did not resolve correctly"


# =============================================================================
# Tests for TERRAIN_MATERIALS and get_terrain_material_params (no Blender required)
# =============================================================================


class TestTerrainMaterials:
    """Tests for TERRAIN_MATERIALS dictionary and related functions."""

    def test_terrain_materials_exists(self):
        """Test that TERRAIN_MATERIALS dictionary exists."""
        from src.terrain.materials import TERRAIN_MATERIALS

        assert isinstance(TERRAIN_MATERIALS, dict)
        assert len(TERRAIN_MATERIALS) > 0

    def test_terrain_materials_all_presets(self):
        """Test that all expected terrain material presets are present."""
        from src.terrain.materials import TERRAIN_MATERIALS

        # Dielectric presets (preserve color accuracy)
        dielectric_presets = ["matte", "eggshell", "satin", "ceramic", "lacquered", "clearcoat", "velvet"]
        # Material-style presets (from color materials)
        material_style_presets = ["clay", "plastic", "ivory", "obsidian", "chrome", "gold", "mineral"]
        expected_materials = dielectric_presets + material_style_presets
        for material in expected_materials:
            assert material in TERRAIN_MATERIALS, f"Missing terrain material: {material}"

    def test_terrain_materials_have_required_keys(self):
        """Test that all terrain materials have required parameter keys."""
        from src.terrain.materials import TERRAIN_MATERIALS

        required_keys = ["roughness", "metallic", "specular_ior_level"]
        for name, params in TERRAIN_MATERIALS.items():
            for key in required_keys:
                assert key in params, f"Terrain material '{name}' missing key: {key}"

    def test_terrain_materials_values_in_range(self):
        """Test that terrain material values are in valid ranges."""
        from src.terrain.materials import TERRAIN_MATERIALS

        for name, params in TERRAIN_MATERIALS.items():
            # Roughness should be 0-1
            assert 0.0 <= params["roughness"] <= 1.0, f"{name} roughness out of range"
            # Metallic should be 0-1 (some materials like chrome/gold are metallic)
            assert 0.0 <= params["metallic"] <= 1.0, f"{name} metallic out of range"
            # Specular IOR level should be 0-1
            assert 0.0 <= params["specular_ior_level"] <= 1.0, f"{name} specular_ior_level out of range"

    def test_default_terrain_material_exists(self):
        """Test that DEFAULT_TERRAIN_MATERIAL is defined and valid."""
        from src.terrain.materials import DEFAULT_TERRAIN_MATERIAL, TERRAIN_MATERIALS

        assert DEFAULT_TERRAIN_MATERIAL in TERRAIN_MATERIALS
        assert DEFAULT_TERRAIN_MATERIAL == "satin"  # We set satin as default


class TestGetTerrainMaterialParams:
    """Tests for get_terrain_material_params function."""

    def test_get_terrain_material_params_exists(self):
        """Test that get_terrain_material_params function exists."""
        from src.terrain.materials import get_terrain_material_params

        assert callable(get_terrain_material_params)

    def test_get_terrain_material_params_preset_lowercase(self):
        """Test getting terrain material parameters by lowercase name."""
        from src.terrain.materials import get_terrain_material_params

        result = get_terrain_material_params("satin")
        assert isinstance(result, dict)
        assert "roughness" in result
        assert result["roughness"] == 0.7

    def test_get_terrain_material_params_case_insensitive(self):
        """Test that terrain material lookup is case-insensitive."""
        from src.terrain.materials import get_terrain_material_params

        lower = get_terrain_material_params("satin")
        upper = get_terrain_material_params("SATIN")
        mixed = get_terrain_material_params("SaTiN")

        assert lower == upper == mixed

    def test_get_terrain_material_params_returns_copy(self):
        """Test that returned dict is a copy (modifying doesn't affect original)."""
        from src.terrain.materials import get_terrain_material_params, TERRAIN_MATERIALS

        result = get_terrain_material_params("satin")
        original_roughness = TERRAIN_MATERIALS["satin"]["roughness"]

        # Modify the result
        result["roughness"] = 999.0

        # Original should be unchanged
        assert TERRAIN_MATERIALS["satin"]["roughness"] == original_roughness

    def test_get_terrain_material_params_invalid_name_raises(self):
        """Test that invalid terrain material name raises ValueError."""
        from src.terrain.materials import get_terrain_material_params

        with pytest.raises(ValueError) as exc_info:
            get_terrain_material_params("invalid_material")

        error_msg = str(exc_info.value)
        assert "Unknown terrain material" in error_msg
        assert "invalid_material" in error_msg

    def test_get_terrain_material_params_all_presets(self):
        """Test that all presets can be retrieved."""
        from src.terrain.materials import get_terrain_material_params, TERRAIN_MATERIALS

        for material_name in TERRAIN_MATERIALS:
            result = get_terrain_material_params(material_name)
            assert result is not None


# =============================================================================
# Tests for unified color system (ALL_COLORS and get_color)
# =============================================================================


class TestUnifiedColors:
    """Tests for the unified color system."""

    def test_all_colors_exists(self):
        """Test that ALL_COLORS dictionary exists and combines both color sets."""
        from src.terrain.materials import ALL_COLORS, ROAD_COLORS, BASE_MATERIALS

        assert isinstance(ALL_COLORS, dict)
        assert len(ALL_COLORS) > 0
        # Should contain all road colors and all base materials
        for color in ROAD_COLORS:
            assert color in ALL_COLORS
        for color in BASE_MATERIALS:
            assert color in ALL_COLORS

    def test_get_all_colors_help_returns_string(self):
        """Test that get_all_colors_help returns a string with all colors."""
        from src.terrain.materials import get_all_colors_help, ALL_COLORS

        result = get_all_colors_help()
        assert isinstance(result, str)
        for color in ALL_COLORS:
            assert color in result

    def test_get_all_colors_choices_returns_list(self):
        """Test that get_all_colors_choices returns a list."""
        from src.terrain.materials import get_all_colors_choices, ALL_COLORS

        result = get_all_colors_choices()
        assert isinstance(result, list)
        assert set(result) == set(ALL_COLORS.keys())

    def test_get_color_exists(self):
        """Test that get_color function exists."""
        from src.terrain.materials import get_color

        assert callable(get_color)

    def test_get_color_resolves_road_colors(self):
        """Test that get_color resolves road color presets."""
        from src.terrain.materials import get_color, ROAD_COLORS

        for color_name, expected_rgb in ROAD_COLORS.items():
            result = get_color(color_name)
            assert result == expected_rgb

    def test_get_color_resolves_base_materials(self):
        """Test that get_color resolves base material presets."""
        from src.terrain.materials import get_color, BASE_MATERIALS

        for color_name, expected_rgb in BASE_MATERIALS.items():
            result = get_color(color_name)
            assert result == expected_rgb

    def test_get_color_case_insensitive(self):
        """Test that get_color is case-insensitive."""
        from src.terrain.materials import get_color

        assert get_color("azurite") == get_color("AZURITE") == get_color("Azurite")
        assert get_color("clay") == get_color("CLAY") == get_color("Clay")

    def test_get_color_rgb_tuple_passthrough(self):
        """Test that RGB tuples are passed through unchanged."""
        from src.terrain.materials import get_color

        custom_rgb = (0.6, 0.55, 0.5)
        result = get_color(custom_rgb)
        assert result == custom_rgb

    def test_get_color_invalid_name_raises(self):
        """Test that invalid color name raises ValueError."""
        from src.terrain.materials import get_color

        with pytest.raises(ValueError) as exc_info:
            get_color("invalid_color")

        error_msg = str(exc_info.value)
        assert "Unknown color preset" in error_msg
        assert "invalid_color" in error_msg

    def test_get_color_terrain_materials_fallback(self):
        """Test that terrain material names without colors return clay gray."""
        from src.terrain.materials import get_color, ALL_COLORS, TERRAIN_MATERIALS

        # These terrain materials don't have associated colors in ALL_COLORS
        shader_only_materials = ["satin", "matte", "eggshell", "ceramic", "lacquered", "clearcoat", "velvet"]
        clay_rgb = ALL_COLORS["clay"]

        for material in shader_only_materials:
            # Verify it's a terrain material but not a color
            assert material in TERRAIN_MATERIALS
            assert material not in ALL_COLORS
            # get_color should return clay gray as fallback
            result = get_color(material)
            assert result == clay_rgb, f"Expected clay gray for {material}, got {result}"


# =============================================================================
# Tests for helper functions (help text and choices for CLI arguments)
# =============================================================================


class TestMaterialHelperFunctions:
    """Tests for helper functions that generate help text and choices for CLI."""

    def test_get_road_colors_help_returns_string(self):
        """Test that get_road_colors_help returns a string."""
        from src.terrain.materials import get_road_colors_help

        result = get_road_colors_help()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_road_colors_help_contains_all_presets(self):
        """Test that get_road_colors_help contains all ROAD_COLORS keys."""
        from src.terrain.materials import get_road_colors_help, ROAD_COLORS

        result = get_road_colors_help()
        for color_name in ROAD_COLORS:
            assert color_name in result, f"Missing road color in help: {color_name}"

    def test_get_road_colors_choices_returns_list(self):
        """Test that get_road_colors_choices returns a list."""
        from src.terrain.materials import get_road_colors_choices

        result = get_road_colors_choices()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_get_road_colors_choices_matches_dict(self):
        """Test that get_road_colors_choices returns ALL_COLORS (unified colors)."""
        from src.terrain.materials import get_road_colors_choices, ALL_COLORS

        result = get_road_colors_choices()
        assert set(result) == set(ALL_COLORS.keys())

    def test_get_base_materials_help_returns_string(self):
        """Test that get_base_materials_help returns a string."""
        from src.terrain.materials import get_base_materials_help

        result = get_base_materials_help()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_base_materials_help_contains_all_presets(self):
        """Test that get_base_materials_help contains all BASE_MATERIALS keys."""
        from src.terrain.materials import get_base_materials_help, BASE_MATERIALS

        result = get_base_materials_help()
        for material_name in BASE_MATERIALS:
            assert material_name in result, f"Missing base material in help: {material_name}"

    def test_get_base_materials_choices_returns_list(self):
        """Test that get_base_materials_choices returns a list."""
        from src.terrain.materials import get_base_materials_choices

        result = get_base_materials_choices()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_get_base_materials_choices_matches_dict(self):
        """Test that get_base_materials_choices returns ALL_COLORS (unified colors)."""
        from src.terrain.materials import get_base_materials_choices, ALL_COLORS

        result = get_base_materials_choices()
        assert set(result) == set(ALL_COLORS.keys())

    def test_get_terrain_materials_help_returns_string(self):
        """Test that get_terrain_materials_help returns a string."""
        from src.terrain.materials import get_terrain_materials_help

        result = get_terrain_materials_help()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_terrain_materials_help_contains_all_presets(self):
        """Test that get_terrain_materials_help contains all TERRAIN_MATERIALS keys."""
        from src.terrain.materials import get_terrain_materials_help, TERRAIN_MATERIALS

        result = get_terrain_materials_help()
        for material_name in TERRAIN_MATERIALS:
            assert material_name in result, f"Missing terrain material in help: {material_name}"

    def test_get_terrain_materials_choices_returns_list(self):
        """Test that get_terrain_materials_choices returns a list."""
        from src.terrain.materials import get_terrain_materials_choices

        result = get_terrain_materials_choices()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_get_terrain_materials_choices_matches_dict(self):
        """Test that get_terrain_materials_choices matches TERRAIN_MATERIALS keys."""
        from src.terrain.materials import get_terrain_materials_choices, TERRAIN_MATERIALS

        result = get_terrain_materials_choices()
        assert set(result) == set(TERRAIN_MATERIALS.keys())


# =============================================================================
# Blender-dependent tests below (require bpy)
# =============================================================================

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

        # Should have essential nodes (pure Principled BSDF, no emission)
        node_types = [node.bl_idname for node in mat.node_tree.nodes]
        assert "ShaderNodeOutputMaterial" in node_types
        assert "ShaderNodeBsdfPrincipled" in node_types
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

        # Should have essential nodes for water shader (pure Principled BSDF, no emission)
        node_types = [node.bl_idname for node in mat.node_tree.nodes]
        assert "ShaderNodeOutputMaterial" in node_types
        assert "ShaderNodeBsdfPrincipled" in node_types
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
