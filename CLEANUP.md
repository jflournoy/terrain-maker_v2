# Library Cleanup Tracker

**Last Updated:** 2026-01-29

This document tracks code cleanup opportunities for more aggressive refactoring.

## Conservative Cleanup (COMPLETED 2026-01-29)

- ✅ **Fixed bug**: `configure_for_target_vertices()` - examples were passing `order=4` but function expects `method="average"`
- ✅ **Removed dead code**: `sort_boundary_points_perimeter()` - never called, redundant with `sort_boundary_points_angular()`
- ✅ **Removed unused**: Drive-time visualization functions (`load_drive_time_data`, `create_drive_time_curves`)

## Aggressive Cleanup Candidates

### Experimental Features (Unused but Potentially Useful)

**Priority: MEDIUM**

#### 1. `horn_slope()` in advanced_viz.py
- **Status**: Exported in `__init__.py` but never used in examples
- **Purpose**: GPU-accelerated slope calculation (7x speedup with CUDA)
- **Decision needed**:
  - [ ] Create example demonstrating GPU acceleration
  - [ ] Deprecate if no plans to promote
- **Location**: [src/terrain/advanced_viz.py:26](src/terrain/advanced_viz.py#L26)

#### 2. `visualize_dem()` in core.py
- **Status**: Terrain class method, never used in examples
- **Purpose**: Advanced matplotlib-based layer visualization with histograms
- **Value**: Useful debugging tool but needs documentation/example
- **Decision needed**:
  - [ ] Create example or document as debugging utility
  - [ ] Remove if not part of public API
- **Location**: [src/terrain/core.py:1094](src/terrain/core.py#L1094)

### Debug-Only Utilities

**Priority: LOW**

#### 3. Visualization Debug Modules
- **`bounds_pipeline.py`**: Visualizes transformation pipeline bounds
  - Only used in tests and archived debug scripts
  - **Action**: Mark as debugging utility in docstring
- **`edge_debug.py`**: Edge extrusion debugging visualizations
  - Only used in tests and archived debug scripts
  - **Action**: Mark as debugging utility in docstring

### Pre-Existing Documentation Issues

**Priority: HIGH (User-facing)**

#### 4. Stale Image References in Documentation
- **Files affected**: `docs/EXAMPLES.md`, `docs/SNOW_SLEDDING.md`
- **Issue**: 14 image references to `04_score_components/` files that were removed in earlier commits
- **Missing images**:
  - `additive_sum.png`, `aspect_score.png`, `cliff_penalty.png`
  - `consistency_score.png`, `depth_score.png`, `multiplicative.png`
  - `raw_interseason_cv.png`, `raw_intraseason_cv.png`, `raw_slope_mean.png`
  - `raw_snow_depth.png`, `runout_bonus.png`, `score_reduction.png`
  - `slope_score.png`, `terrain_consistency.png`
- **Action**: Either regenerate images or update docs to reference existing images

## API Simplification Opportunities

### Function Parameters

Review function signatures for parameters that:
- Are never overridden from defaults
- Accept multiple values but only one is used in practice
- Could be removed to simplify the API

**Next step**: Audit all public API functions for parameter usage patterns across examples.

## Notes

- This tracker follows the "conservative cleanup" completed on 2026-01-29
- Items here require more investigation or breaking changes
- Track completed items with ✅ checkbox and date
