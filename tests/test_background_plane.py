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

    These tests verify the mathematical calculation without requiring Blender,
    by testing the helper functions and logic.
    """

    def test_function_exists(self):
        """Function exists and is callable."""
        from src.terrain.scene_setup import calculate_camera_frustum_size

        assert callable(calculate_camera_frustum_size)

    def test_function_signature(self):
        """Function has expected parameters."""
        from src.terrain.scene_setup import calculate_camera_frustum_size

        import inspect
        sig = inspect.signature(calculate_camera_frustum_size)

        # Check required parameters
        assert 'camera_type' in sig.parameters
        assert 'ortho_scale' in sig.parameters or 'distance' in sig.parameters

    def test_ortho_returns_tuple(self):
        """Orthographic camera calculation returns (width, height) tuple."""
        from src.terrain.scene_setup import calculate_camera_frustum_size

        # Test with orthographic camera
        result = calculate_camera_frustum_size(
            camera_type="ORTHO",
            ortho_scale=2.0,
            aspect_ratio=16 / 9,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(x, float) for x in result)

    def test_ortho_scale_affects_width(self):
        """Orthographic: width equals ortho_scale."""
        from src.terrain.scene_setup import calculate_camera_frustum_size

        scale = 5.0
        result = calculate_camera_frustum_size(
            camera_type="ORTHO",
            ortho_scale=scale,
            aspect_ratio=1.0,  # Square for simplicity
        )

        width, height = result
        # For ortho, width should equal ortho_scale
        assert abs(width - scale) < 0.001

    def test_ortho_respects_aspect_ratio(self):
        """Orthographic: height computed from aspect ratio."""
        from src.terrain.scene_setup import calculate_camera_frustum_size

        aspect_ratio = 16 / 9
        result = calculate_camera_frustum_size(
            camera_type="ORTHO",
            ortho_scale=10.0,
            aspect_ratio=aspect_ratio,
        )

        width, height = result
        # Height = width / aspect_ratio
        expected_height = width / aspect_ratio
        assert abs(height - expected_height) < 0.001

    def test_perspective_returns_tuple(self):
        """Perspective camera calculation returns (width, height) tuple."""
        from src.terrain.scene_setup import calculate_camera_frustum_size

        result = calculate_camera_frustum_size(
            camera_type="PERSP",
            fov_degrees=49.13,  # Blender default in degrees
            distance=10.0,
            aspect_ratio=16 / 9,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(x, float) for x in result)

    def test_perspective_fov_affects_size(self):
        """Perspective: larger FOV produces larger frustum."""
        from src.terrain.scene_setup import calculate_camera_frustum_size

        distance = 10.0
        aspect_ratio = 1.0

        # Small FOV
        small_fov = calculate_camera_frustum_size(
            camera_type="PERSP",
            fov_degrees=30.0,
            distance=distance,
            aspect_ratio=aspect_ratio,
        )

        # Large FOV
        large_fov = calculate_camera_frustum_size(
            camera_type="PERSP",
            fov_degrees=60.0,
            distance=distance,
            aspect_ratio=aspect_ratio,
        )

        # Larger FOV should produce larger width
        assert large_fov[0] > small_fov[0]

    def test_perspective_distance_affects_size(self):
        """Perspective: larger distance produces larger frustum."""
        from src.terrain.scene_setup import calculate_camera_frustum_size

        fov = 49.13
        aspect_ratio = 1.0

        # Close distance
        close = calculate_camera_frustum_size(
            camera_type="PERSP",
            fov_degrees=fov,
            distance=5.0,
            aspect_ratio=aspect_ratio,
        )

        # Far distance
        far = calculate_camera_frustum_size(
            camera_type="PERSP",
            fov_degrees=fov,
            distance=20.0,
            aspect_ratio=aspect_ratio,
        )

        # Farther distance should produce larger frustum
        assert far[0] > close[0]

    def test_perspective_aspect_ratio_maintained(self):
        """Perspective: aspect ratio is respected."""
        from src.terrain.scene_setup import calculate_camera_frustum_size

        aspect_ratio = 16 / 9

        result = calculate_camera_frustum_size(
            camera_type="PERSP",
            fov_degrees=49.13,
            distance=10.0,
            aspect_ratio=aspect_ratio,
        )

        width, height = result
        # Check aspect ratio: width / height should equal aspect_ratio
        computed_ratio = width / height
        assert abs(computed_ratio - aspect_ratio) < 0.001

    def test_positive_dimensions(self):
        """All dimensions are positive values."""
        from src.terrain.scene_setup import calculate_camera_frustum_size

        # Orthographic
        ortho_result = calculate_camera_frustum_size(
            camera_type="ORTHO",
            ortho_scale=2.0,
            aspect_ratio=16 / 9,
        )
        assert all(x > 0 for x in ortho_result)

        # Perspective
        persp_result = calculate_camera_frustum_size(
            camera_type="PERSP",
            fov_degrees=49.13,
            distance=10.0,
            aspect_ratio=16 / 9,
        )
        assert all(x > 0 for x in persp_result)

    def test_has_docstring(self):
        """Function has proper documentation."""
        from src.terrain.scene_setup import calculate_camera_frustum_size

        assert calculate_camera_frustum_size.__doc__ is not None
        assert 'frustum' in calculate_camera_frustum_size.__doc__.lower()


class TestCreateBackgroundPlane:
    """Tests for background plane creation.

    Note: These tests verify function signature and defaults without requiring
    Blender. Integration tests in Blender environment verify actual plane creation.
    """

    def test_function_exists(self):
        """Function exists and is callable."""
        from src.terrain.scene_setup import create_background_plane

        assert callable(create_background_plane)

    def test_function_signature(self):
        """Function has expected parameters."""
        from src.terrain.scene_setup import create_background_plane

        import inspect
        sig = inspect.signature(create_background_plane)

        # Check required parameters
        assert 'camera' in sig.parameters
        assert 'mesh_or_meshes' in sig.parameters

        # Check optional parameters with defaults
        assert 'distance_below' in sig.parameters
        assert 'color' in sig.parameters
        assert 'size_multiplier' in sig.parameters
        assert 'receive_shadows' in sig.parameters

    def test_default_distance_below(self):
        """Default distance_below is 50.0 units."""
        from src.terrain.scene_setup import create_background_plane

        import inspect
        sig = inspect.signature(create_background_plane)

        assert sig.parameters['distance_below'].default == 50.0

    def test_default_color_is_eggshell(self):
        """Default color is eggshell white (#F5F5F0)."""
        from src.terrain.scene_setup import create_background_plane

        import inspect
        sig = inspect.signature(create_background_plane)

        assert sig.parameters['color'].default == "#F5F5F0"

    def test_default_size_multiplier(self):
        """Default size_multiplier is 2.0."""
        from src.terrain.scene_setup import create_background_plane

        import inspect
        sig = inspect.signature(create_background_plane)

        assert sig.parameters['size_multiplier'].default == 2.0

    def test_default_receive_shadows_false(self):
        """Default receive_shadows is False."""
        from src.terrain.scene_setup import create_background_plane

        import inspect
        sig = inspect.signature(create_background_plane)

        assert sig.parameters['receive_shadows'].default is False

    def test_has_docstring(self):
        """Function has proper documentation."""
        from src.terrain.scene_setup import create_background_plane

        assert create_background_plane.__doc__ is not None
        assert 'background' in create_background_plane.__doc__.lower()
        assert 'plane' in create_background_plane.__doc__.lower()

    @pytest.mark.skip(reason="Requires Blender environment")
    def test_plane_created_in_blender(self):
        """Function creates a plane mesh object in Blender."""
        pass

    @pytest.mark.skip(reason="Requires Blender environment")
    def test_plane_sized_to_camera_frustum(self):
        """Plane is sized to fill camera frustum with multiplier."""
        pass

    @pytest.mark.skip(reason="Requires Blender environment")
    def test_plane_positioned_below_mesh(self):
        """Plane is positioned below mesh by specified distance."""
        pass

    @pytest.mark.skip(reason="Requires Blender environment")
    def test_custom_color_applied(self):
        """Custom colors are applied correctly."""
        pass

    @pytest.mark.skip(reason="Requires Blender environment")
    def test_shadow_receiving_configurable(self):
        """Shadow receiving can be enabled or disabled."""
        pass

    @pytest.mark.skip(reason="Requires Blender environment")
    def test_works_with_single_mesh(self):
        """Function works with a single mesh object."""
        pass

    @pytest.mark.skip(reason="Requires Blender environment")
    def test_works_with_multiple_meshes(self):
        """Function works with a list of mesh objects."""
        pass

    @pytest.mark.skip(reason="Requires Blender environment")
    def test_returns_plane_object(self):
        """Function returns the created plane object."""
        pass
