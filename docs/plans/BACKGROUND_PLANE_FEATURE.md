# Background Plane Feature - TDD Implementation Plan

## Overview

Add a configurable background plane for Blender renders that sits below the terrain mesh. The plane should fill the camera view, use a matte material, and optionally receive shadows.

## User Stories

1. **As a renderer**, I want a clean background behind my terrain so the render looks professional
2. **As a designer**, I want to specify the background color to match my design aesthetic
3. **As an artist**, I want the option to show or hide drop shadows for different visual effects

## Technical Requirements

### Core Library Function: `create_background_plane()`

**Location**: `src/terrain/scene_setup.py`

**Parameters**:
- `camera`: Blender camera object (required to calculate plane size)
- `mesh_or_meshes`: Mesh object(s) to position plane relative to (required)
- `distance_below`: Distance below the lowest point of mesh (default: 50.0 units)
- `color`: RGB tuple or hex string (default: "#F5F5F0" eggshell white)
- `size_multiplier`: How much bigger than camera frustum (default: 2.0 for safety margin)
- `receive_shadows`: Whether plane receives shadows (default: False)
- `material_roughness`: Matte roughness 0-1 (default: 1.0 for fully matte)

**Returns**: Blender plane object

### Helper Functions

1. `calculate_camera_frustum_size(camera, distance)` - Calculate visible area at given distance
2. `create_matte_material(name, color, roughness, receive_shadows)` - Create principled BSDF material
3. `hex_to_rgb(hex_color)` - Convert hex color to RGB tuple

### Integration Points

1. Add `--background` flag to `detroit_combined_render.py`
2. Add `--background-color` option (default: eggshell white)
3. Add `--background-shadow` flag to enable shadow receiving

---

## GitHub Issues

### Issue #1: Create camera frustum size calculation function

**Title**: feat(scene): add function to calculate camera frustum size at distance

**Description**:
Create a function that calculates the visible area of an orthographic or perspective camera at a given distance. This is needed to size the background plane correctly.

**Acceptance Criteria**:
- [ ] Handles orthographic cameras (uses ortho_scale)
- [ ] Handles perspective cameras (uses FOV + distance)
- [ ] Returns (width, height) tuple in Blender units
- [ ] Works with any camera angle/rotation
- [ ] Unit tests cover both camera types

**Labels**: enhancement, scene-setup

---

### Issue #2: Create matte material helper function

**Title**: feat(scene): add function to create matte materials with configurable shadow receiving

**Description**:
Create a helper function that generates a Principled BSDF material with matte (non-reflective) properties and configurable shadow visibility.

**Acceptance Criteria**:
- [ ] Creates Principled BSDF material
- [ ] Supports RGB tuple or hex color input
- [ ] Roughness parameter (default 1.0 = fully matte)
- [ ] Shadow receiving toggle via shader nodes
- [ ] Returns material object for assignment
- [ ] Unit tests verify material properties

**Labels**: enhancement, scene-setup

---

### Issue #3: Create background plane function

**Title**: feat(scene): add create_background_plane() for render backgrounds

**Description**:
Create the main function that generates a background plane sized to fill the camera view, positioned below the terrain mesh.

**Acceptance Criteria**:
- [ ] Creates plane mesh in Blender
- [ ] Sizes plane to fill camera frustum with safety margin
- [ ] Positions plane below lowest mesh point by specified distance
- [ ] Applies matte material with specified color
- [ ] Configurable shadow receiving (default: off for no drop shadow)
- [ ] Works with single mesh or list of meshes
- [ ] Returns plane object for further manipulation
- [ ] Unit tests verify positioning and sizing

**Labels**: enhancement, scene-setup

---

### Issue #4: Add hex color conversion utility

**Title**: feat(utils): add hex_to_rgb color conversion function

**Description**:
Create a utility function to convert hex color strings to RGB tuples for Blender materials.

**Acceptance Criteria**:
- [ ] Handles "#RRGGBB" format
- [ ] Handles "#RGB" shorthand
- [ ] Handles "RRGGBB" without hash
- [ ] Returns (r, g, b) tuple with values 0.0-1.0
- [ ] Raises ValueError for invalid input
- [ ] Unit tests cover all formats and edge cases

**Labels**: enhancement, utilities

---

### Issue #5: Integrate background plane into detroit_combined_render.py

**Title**: feat(examples): add background plane options to combined render

**Description**:
Add CLI options to enable and configure the background plane in the Detroit combined render example.

**Acceptance Criteria**:
- [ ] `--background` flag enables background plane
- [ ] `--background-color` option (default: #F5F5F0 eggshell)
- [ ] `--background-shadow` flag enables shadow receiving
- [ ] `--background-distance` option (default: 50.0)
- [ ] Background plane appears in rendered output
- [ ] Integration test verifies plane creation

**Labels**: enhancement, examples

---

## TDD Test Cases

### Test File: `tests/test_background_plane.py`

```python
"""Tests for background plane functionality."""

import pytest
import numpy as np

# Will need to mock bpy for unit tests


class TestHexToRgb:
    """Tests for hex color conversion."""

    def test_hex_with_hash(self):
        """#RRGGBB format converts correctly."""
        # #F5F5F0 (eggshell) -> (0.96, 0.96, 0.94)
        pass

    def test_hex_without_hash(self):
        """RRGGBB format converts correctly."""
        pass

    def test_hex_shorthand(self):
        """#RGB shorthand expands correctly."""
        # #FFF -> (1.0, 1.0, 1.0)
        pass

    def test_invalid_hex_raises(self):
        """Invalid hex strings raise ValueError."""
        pass

    def test_case_insensitive(self):
        """Hex parsing is case insensitive."""
        pass


class TestCameraFrustumSize:
    """Tests for camera frustum calculation."""

    def test_ortho_camera_size(self):
        """Orthographic camera returns correct size based on ortho_scale."""
        pass

    def test_perspective_camera_size(self):
        """Perspective camera returns correct size based on FOV and distance."""
        pass

    def test_camera_aspect_ratio(self):
        """Frustum size respects render aspect ratio."""
        pass


class TestCreateMatteMaterial:
    """Tests for matte material creation."""

    def test_creates_principled_bsdf(self):
        """Material uses Principled BSDF shader."""
        pass

    def test_default_roughness_is_matte(self):
        """Default roughness is 1.0 (fully matte)."""
        pass

    def test_color_applied_correctly(self):
        """Base color is set correctly from input."""
        pass

    def test_shadow_receiving_disabled(self):
        """Shadow receiving can be disabled via shader mix."""
        pass

    def test_shadow_receiving_enabled(self):
        """Shadow receiving can be enabled."""
        pass


class TestCreateBackgroundPlane:
    """Tests for background plane creation."""

    def test_plane_created(self):
        """Function creates a plane mesh object."""
        pass

    def test_plane_sized_to_camera(self):
        """Plane is sized to fill camera frustum with margin."""
        pass

    def test_plane_positioned_below_mesh(self):
        """Plane is positioned below mesh by specified distance."""
        pass

    def test_plane_positioned_below_multiple_meshes(self):
        """Plane is positioned below lowest of multiple meshes."""
        pass

    def test_default_color_is_eggshell(self):
        """Default color is eggshell white (#F5F5F0)."""
        pass

    def test_custom_color_applied(self):
        """Custom colors are applied correctly."""
        pass

    def test_default_no_shadows(self):
        """By default, plane does not receive shadows."""
        pass

    def test_shadows_can_be_enabled(self):
        """Shadow receiving can be enabled."""
        pass

    def test_returns_plane_object(self):
        """Function returns the created plane object."""
        pass
```

---

## Implementation Order (TDD Red-Green-Refactor)

### Phase 1: Color Utilities (Issue #4)
1. RED: Write failing tests for `hex_to_rgb()`
2. GREEN: Implement minimal `hex_to_rgb()` to pass tests
3. REFACTOR: Clean up, add docstrings

### Phase 2: Material Creation (Issue #2)
1. RED: Write failing tests for `create_matte_material()`
2. GREEN: Implement material creation (requires bpy mocking)
3. REFACTOR: Extract common patterns

### Phase 3: Camera Frustum (Issue #1)
1. RED: Write failing tests for `calculate_camera_frustum_size()`
2. GREEN: Implement frustum calculation
3. REFACTOR: Handle edge cases

### Phase 4: Background Plane (Issue #3)
1. RED: Write failing tests for `create_background_plane()`
2. GREEN: Implement plane creation using helper functions
3. REFACTOR: Ensure clean API

### Phase 5: Integration (Issue #5)
1. RED: Write integration test for CLI options
2. GREEN: Add CLI options to detroit_combined_render.py
3. REFACTOR: Update documentation

---

## Default Values

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `color` | `#F5F5F0` | Eggshell white - warm, neutral, professional |
| `distance_below` | `50.0` | Far enough to avoid visible shadows at default light angles |
| `size_multiplier` | `2.0` | 2x camera frustum ensures full coverage with margin |
| `receive_shadows` | `False` | Clean background by default |
| `material_roughness` | `1.0` | Fully matte, no reflections |

---

## Color Presets (Future Enhancement)

Consider adding named presets:
- `"eggshell"` - #F5F5F0 (default)
- `"white"` - #FFFFFF
- `"paper"` - #FFFEF5 (warm white)
- `"gray"` - #E0E0E0
- `"charcoal"` - #36454F
- `"black"` - #000000

---

## API Example

```python
from src.terrain.scene_setup import create_background_plane

# Basic usage (eggshell white, no shadows)
background = create_background_plane(
    camera=camera,
    mesh_or_meshes=terrain_mesh,
)

# Custom color with shadows
background = create_background_plane(
    camera=camera,
    mesh_or_meshes=[mesh_left, mesh_right],
    color="#E8E4D9",  # Warm cream
    distance_below=30.0,  # Closer for visible shadow
    receive_shadows=True,
)

# CLI usage
python examples/detroit_combined_render.py --background --background-color "#F5F5F0"
python examples/detroit_combined_render.py --background --background-shadow  # Enable shadows
```

---

## Files to Create/Modify

### New Files
- `tests/test_background_plane.py` - Test suite

### Modified Files
- `src/terrain/scene_setup.py` - Add new functions
- `examples/detroit_combined_render.py` - Add CLI options
- `src/terrain/__init__.py` - Export new functions (if needed)

---

## Estimated Complexity

| Component | Complexity | Notes |
|-----------|------------|-------|
| hex_to_rgb | Low | Simple string parsing |
| create_matte_material | Medium | Blender shader nodes |
| calculate_camera_frustum_size | Medium | Math for ortho/perspective |
| create_background_plane | Medium | Combines all components |
| CLI integration | Low | argparse additions |

**Total estimate**: ~50-80 Claude interactions following TDD methodology
