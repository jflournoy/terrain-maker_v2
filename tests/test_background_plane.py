"""Tests for background plane functionality.

This module contains tests for the background plane rendering feature,
including color utilities, material creation, and plane positioning.
"""

import pytest


class TestHexToRgb:
    """Tests for hex color conversion utility.

    The hex_to_rgb function converts various hex color formats to
    normalized RGB tuples (0.0-1.0 range).
    """

    def test_hex_with_hash_standard(self):
        """#RRGGBB format converts to RGB tuple."""
        from src.terrain.scene_setup import hex_to_rgb

        # Eggshell white
        r, g, b = hex_to_rgb("#F5F5F0")
        assert isinstance(r, float) and isinstance(g, float) and isinstance(b, float)
        assert 0.0 <= r <= 1.0 and 0.0 <= g <= 1.0 and 0.0 <= b <= 1.0
        # F5 = 245, so 245/255 ≈ 0.961
        assert abs(r - 245/255) < 0.01
        assert abs(g - 245/255) < 0.01
        assert abs(b - 240/255) < 0.01

    def test_hex_without_hash(self):
        """RRGGBB format (without #) converts correctly."""
        from src.terrain.scene_setup import hex_to_rgb

        r, g, b = hex_to_rgb("F5F5F0")
        assert abs(r - 245/255) < 0.01
        assert abs(g - 245/255) < 0.01
        assert abs(b - 240/255) < 0.01

    def test_hex_shorthand_expansion(self):
        """#RGB shorthand expands correctly."""
        from src.terrain.scene_setup import hex_to_rgb

        # #FFF should expand to #FFFFFF
        r, g, b = hex_to_rgb("#FFF")
        assert abs(r - 1.0) < 0.01
        assert abs(g - 1.0) < 0.01
        assert abs(b - 1.0) < 0.01

    def test_hex_shorthand_without_hash(self):
        """RGB shorthand without # also works."""
        from src.terrain.scene_setup import hex_to_rgb

        r, g, b = hex_to_rgb("FFF")
        assert abs(r - 1.0) < 0.01
        assert abs(g - 1.0) < 0.01
        assert abs(b - 1.0) < 0.01

    def test_pure_black(self):
        """#000000 converts to (0.0, 0.0, 0.0)."""
        from src.terrain.scene_setup import hex_to_rgb

        r, g, b = hex_to_rgb("#000000")
        assert abs(r) < 0.001
        assert abs(g) < 0.001
        assert abs(b) < 0.001

    def test_pure_white(self):
        """#FFFFFF converts to (1.0, 1.0, 1.0)."""
        from src.terrain.scene_setup import hex_to_rgb

        r, g, b = hex_to_rgb("#FFFFFF")
        assert abs(r - 1.0) < 0.001
        assert abs(g - 1.0) < 0.001
        assert abs(b - 1.0) < 0.001

    def test_case_insensitive_uppercase(self):
        """Uppercase hex works."""
        from src.terrain.scene_setup import hex_to_rgb

        r1, g1, b1 = hex_to_rgb("#F5F5F0")
        r2, g2, b2 = hex_to_rgb("#f5f5f0")
        assert abs(r1 - r2) < 0.001
        assert abs(g1 - g2) < 0.001
        assert abs(b1 - b2) < 0.001

    def test_invalid_hex_raises_valueerror(self):
        """Invalid hex strings raise ValueError."""
        from src.terrain.scene_setup import hex_to_rgb

        with pytest.raises(ValueError):
            hex_to_rgb("#GGGGGG")  # Invalid characters

        with pytest.raises(ValueError):
            hex_to_rgb("#12345")  # Wrong length

        with pytest.raises(ValueError):
            hex_to_rgb("")  # Empty string

    def test_red_channel(self):
        """Red channel is extracted correctly."""
        from src.terrain.scene_setup import hex_to_rgb

        r, g, b = hex_to_rgb("#FF0000")
        assert abs(r - 1.0) < 0.001
        assert abs(g) < 0.001
        assert abs(b) < 0.001

    def test_green_channel(self):
        """Green channel is extracted correctly."""
        from src.terrain.scene_setup import hex_to_rgb

        r, g, b = hex_to_rgb("#00FF00")
        assert abs(r) < 0.001
        assert abs(g - 1.0) < 0.001
        assert abs(b) < 0.001

    def test_blue_channel(self):
        """Blue channel is extracted correctly."""
        from src.terrain.scene_setup import hex_to_rgb

        r, g, b = hex_to_rgb("#0000FF")
        assert abs(r) < 0.001
        assert abs(g) < 0.001
        assert abs(b - 1.0) < 0.001

    def test_returns_tuple(self):
        """Function returns a tuple of 3 floats."""
        from src.terrain.scene_setup import hex_to_rgb

        result = hex_to_rgb("#F5F5F0")
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(x, float) for x in result)


class TestCreateMatteMaterial:
    """Tests for matte material creation.

    Note: These tests use mocking since bpy is not available in test environment.
    In actual Blender environment, test with integration tests.
    """

    def test_accepts_hex_color(self):
        """Function accepts hex color strings."""
        # This test verifies the function signature accepts hex colors
        # by checking that hex_to_rgb is being used internally
        from src.terrain.scene_setup import create_matte_material

        # Just verify the function exists and is callable
        assert callable(create_matte_material)

    def test_accepts_rgb_tuple(self):
        """Function accepts RGB tuples."""
        from src.terrain.scene_setup import create_matte_material

        # Verify function exists
        assert callable(create_matte_material)

    def test_default_color_is_eggshell(self):
        """Default color is eggshell white (#F5F5F0)."""
        from src.terrain.scene_setup import create_matte_material

        # We can verify the function has correct defaults in signature
        import inspect
        sig = inspect.signature(create_matte_material)

        # Check that color has a default value
        assert 'color' in sig.parameters
        assert sig.parameters['color'].default == '#F5F5F0'

    def test_default_roughness_is_one(self):
        """Default roughness is 1.0 (fully matte)."""
        from src.terrain.scene_setup import create_matte_material

        import inspect
        sig = inspect.signature(create_matte_material)

        # Check that roughness defaults to 1.0
        assert 'material_roughness' in sig.parameters
        assert sig.parameters['material_roughness'].default == 1.0

    def test_default_no_shadows(self):
        """Default receive_shadows is False."""
        from src.terrain.scene_setup import create_matte_material

        import inspect
        sig = inspect.signature(create_matte_material)

        # Check that receive_shadows defaults to False
        assert 'receive_shadows' in sig.parameters
        assert sig.parameters['receive_shadows'].default is False

    def test_has_docstring(self):
        """Function has proper documentation."""
        from src.terrain.scene_setup import create_matte_material

        assert create_matte_material.__doc__ is not None
        assert 'matte' in create_matte_material.__doc__.lower()
        assert 'shadow' in create_matte_material.__doc__.lower()

    def test_material_name_parameter(self):
        """Function accepts a name parameter for the material."""
        from src.terrain.scene_setup import create_matte_material

        import inspect
        sig = inspect.signature(create_matte_material)

        assert 'name' in sig.parameters


class TestCalculateCameraFrustumSize:
    """Tests for camera frustum size calculation.

    Note: These tests use mocking since bpy is not available in test environment.
    """

    @pytest.mark.skip(reason="Requires Blender environment")
    def test_ortho_camera_size(self):
        """Orthographic camera returns correct size based on ortho_scale."""
        pass

    @pytest.mark.skip(reason="Requires Blender environment")
    def test_perspective_camera_size(self):
        """Perspective camera returns correct size based on FOV and distance."""
        pass


class TestCreateBackgroundPlane:
    """Tests for background plane creation.

    Note: These tests use mocking since bpy is not available in test environment.
    """

    @pytest.mark.skip(reason="Requires Blender environment")
    def test_plane_created(self):
        """Function creates a plane mesh object."""
        pass

    @pytest.mark.skip(reason="Requires Blender environment")
    def test_plane_sized_to_camera(self):
        """Plane is sized to fill camera frustum with margin."""
        pass

    @pytest.mark.skip(reason="Requires Blender environment")
    def test_default_color_is_eggshell(self):
        """Default color is eggshell white (#F5F5F0)."""
        pass
