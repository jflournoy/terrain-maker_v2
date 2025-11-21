---
agent-type: general-purpose
allowed-tools: [Read, Glob, Grep, Bash, Write]
description: Analyzes Python code documentation quality, completeness, and accuracy
last-updated: 2025-11-20
---

# Python Documentation Auditor Agent

## Objective

Perform comprehensive audit of Python API documentation quality, examining docstrings for completeness, clarity, accuracy, tone consistency, and adherence to documentation standards. Identify gaps in parameter documentation, return value documentation, type hints, and examples.

## Task Instructions

### Phase 1: Python Module Discovery

1. **Locate Python Modules**
   - Scan `src/` directory for `*.py` files
   - Identify all classes and functions
   - Map module structure and dependencies
   - Note which modules have existing documentation

2. **Extract Docstring Information**
   - Parse all Python files for docstrings
   - Extract function/class names and signatures
   - Identify parameter definitions
   - Note return types (documented and inferred)

3. **Categorize Code Elements**
   - Public API vs internal functions (by naming convention)
   - Core functionality vs utilities
   - Complex functions requiring more documentation
   - Simple functions with minimal documentation needs

### Phase 2: Docstring Quality Assessment

1. **Completeness Analysis**
   - Check if docstring exists for public functions/classes
   - Verify all parameters are documented
   - Confirm return values are documented
   - Look for type information in docstrings
   - Check for exception documentation (if applicable)

2. **Parameter Documentation**
   - Each parameter has description
   - Parameter types are documented (Args: section)
   - Default values are mentioned if applicable
   - Special constraints are explained (e.g., "must be positive")

3. **Return Value Documentation**
   - Return type is documented (Returns: section)
   - Return value description explains what is returned
   - Multiple return types are all documented
   - Tuple returns explain each element

4. **Type Hints vs Docstrings**
   - Compare function signatures to docstring Args
   - Check for consistency between type hints and documentation
   - Note any mismatches or conflicts
   - Assess whether documentation clarifies type hints

### Phase 3: Documentation Accuracy

1. **Signature Verification**
   - Compare docstring parameter list to actual function signature
   - Verify documented parameters exist
   - Check for documented parameters that don't exist (stale docs)
   - Ensure parameter order matches

2. **Type Accuracy**
   - Verify documented types match actual type hints
   - Check for reasonable type descriptions
   - Note any deprecated or incorrect type references
   - Validate numpy/pandas type references if used

3. **Return Value Accuracy**
   - Check return type matches actual return statements
   - Verify documented exceptions are actually raised
   - Compare docstring examples to actual usage

### Phase 4: Tone and Clarity Assessment

1. **Writing Quality**
   - Docstrings use clear, active voice
   - Language is professional but accessible
   - Descriptions are concise yet complete
   - No jargon without explanation

2. **Consistency Checks**
   - Tone is consistent across module
   - Terminology is standardized (e.g., "returns" vs "returns a")
   - Similar functions have similar documentation style
   - Formatting is consistent (punctuation, capitalization)

3. **Tone Problems to Flag**
   - Words to avoid: "obviously", "clearly", "just", "simply"
   - Overly complex or academic language
   - Overly simplified explanations
   - Dismissive or patronizing tone
   - Passive voice when active would be clearer

### Phase 5: Example and Usage Documentation

1. **Example Coverage**
   - Check if complex functions have usage examples
   - Verify examples in docstrings are correct
   - Look for "Example:" or "Examples:" sections
   - Assess whether examples cover common use cases

2. **Edge Case Documentation**
   - Special cases documented (e.g., "returns None if...")
   - Error conditions explained
   - Boundary behavior documented
   - Side effects noted if any

3. **Usage Context**
   - Documentation explains when/why to use function
   - Related functions are referenced
   - Common patterns are shown
   - Prerequisites or dependencies mentioned

### Phase 6: Standards Compliance

1. **Format Consistency**
   - NumPy docstring format or consistent alternative
   - Sections in expected order (Summary, Parameters, Returns, Examples)
   - Consistent use of code blocks and formatting
   - Proper indentation and line wrapping

2. **Cross-File Patterns**
   - Document all public functions consistently
   - Similar complexity = similar documentation depth
   - Avoid over-documenting trivial functions
   - Don't under-document complex functions

3. **Type Hint Alignment**
   - Modern Python (3.9+) type hints available?
   - Docstrings complement (not duplicate) type hints
   - Type information accessible to IDE/tools
   - Documentation adds semantic clarity beyond types

## Analysis Patterns

### High-Priority Documentation Needs

1. **Core Classes**
   - `Terrain`, `TerrainCache` - Central API classes
   - Should have comprehensive class-level documentation
   - Constructor parameters well-explained
   - Major methods thoroughly documented

2. **Public API Functions**
   - Entry point functions used by external users
   - Must have clear parameter and return documentation
   - Examples crucial for understanding
   - Error conditions important to document

3. **Data Transformation Functions**
   - Functions accepting arrays/rasters (load_dem_files, reproject_raster)
   - Complex parameter types need explanation
   - Return data structure clearly documented
   - CRS and coordinate system details crucial

### Medium-Priority Documentation

1. **Helper Functions**
   - Used within module but exposed publicly
   - Clear parameter descriptions
   - Return type documentation
   - May need less elaborate examples

2. **Configuration Functions**
   - Functions affecting behavior
   - Parameter constraints important
   - Documentation of defaults valuable
   - Side effects should be noted

### Lower-Priority (but still important)

1. **Internal Functions**
   - Functions with names starting with `_`
   - Should still have docstrings but less formal
   - Parameter and return documentation still needed
   - Less need for extensive examples

## Output Format

Create `.claude/agents/reports/python-docs-audit-[date].md`:

```markdown
# Python Documentation Audit Report - [Date]

## Executive Summary
- **Python files scanned**: X
- **Functions/classes found**: Y
- **With docstrings**: Z%
- **Documentation quality score**: W/100
- **Critical gaps**: N items

## Critical Issues (Missing Documentation)

### Undocumented Functions
1. **Module**: `src/terrain/core.py`
   **Function**: `Terrain.__init__()`
   **Type**: Class constructor
   **Issue**: No docstring
   **Impact**: Users cannot understand initialization requirements
   **Action**: Add comprehensive docstring with Parameters and Returns

2. **Module**: `src/terrain/core.py`
   **Function**: `load_dem_files()`
   **Type**: Public API function
   **Issue**: No docstring
   **Impact**: Critical entry point with no documentation
   **Action**: Add docstring with example usage

## Documentation Completeness Analysis

### By Category
| Category | Total | Documented | % | Priority |
|----------|-------|------------|---|----------|
| Classes | 2 | 0 | 0% | CRITICAL |
| Public Functions | 15 | 8 | 53% | HIGH |
| Helper Functions | 10 | 7 | 70% | MEDIUM |
| Internal Functions | 9 | 4 | 44% | LOW |

### By Module
| Module | Functions | Documented | Score |
|--------|-----------|------------|-------|
| src/terrain/core.py | 35 | 20 | 57/100 |
| src/utils/helpers.py | 5 | 4 | 80/100 |

## Parameter Documentation Issues

### Incomplete Parameter Docs
1. **Function**: `create_mesh()`
   **Issue**: Parameter `boundary_extension` documented but constraint missing
   **Recommendation**: Add note about boolean requirement and behavior difference
   **Location**: src/terrain/core.py:XXX

2. **Function**: `reproject_raster()`
   **Issue**: `src_crs` and `dst_crs` documented but format not specified
   **Recommendation**: Clarify EPSG format with example (e.g., 'EPSG:4326')
   **Location**: src/terrain/core.py:XXX

### Type Documentation Gaps
1. **Function**: `add_data_layer()`
   **Issue**: `data` parameter - array shape not documented
   **Recommendation**: Specify "np.ndarray, shape (height, width)" in docstring
   **Location**: src/terrain/core.py:XXX

## Return Value Documentation

### Incomplete Returns
1. **Function**: `load_dem_files()`
   **Issue**: Returns documented but array structure not explained
   **Recommendation**: Document return as "(np.ndarray, Affine) tuple with DEM data and transform"
   **Location**: src/terrain/core.py:XXX

### Missing Return Type Documentation
1. **Function**: `apply_transforms()`
   **Issue**: No return documentation despite returning data
   **Recommendation**: Add Returns section documenting output

## Accuracy Issues

### Parameter Mismatches
1. **Function**: `position_camera_relative()`
   **Docstring lists parameters**: 9
   **Actual parameters**: 9
   **Status**: ✅ Matches
   **Details**: All parameters documented

2. **Function**: `setup_render_settings()`
   **Docstring lists parameters**: 3
   **Actual parameters**: 4
   **Issue**: Parameter `denoising` added but docstring not updated
   **Action**: Update docstring to include new parameter

### Return Type Accuracy
1. **Function**: `Terrain.create_mesh()`
   **Documented return**: "Blender mesh object"
   **Actual return type**: `bpy.types.Object | None`
   **Issue**: Doesn't document None return possibility
   **Action**: Update docstring to note None return on failure

## Tone and Clarity Assessment

### Excellent Documentation Examples
1. **Function**: `position_camera_relative()`
   - Clear explanation of parameters
   - Good use of cardinal direction terminology
   - Helpful mention of mesh-relative positioning
   - Nice detail about look_at='center'

2. **Function**: `flip_raster()`
   - Concise yet informative
   - Clear explanation of axis parameter
   - Good use of active voice

### Areas for Improvement
1. **Tone**: Some functions use passive voice unnecessarily
   - Example: "DEM data is loaded and merged"
   - Better: "Loads and merges DEM data"

2. **Clarity**: Type explanations could be more specific
   - Instead of: "data (array)"
   - Better: "data (np.ndarray): 2D elevation array, shape (height, width)"

## Examples and Usage Documentation

### Missing Examples
Functions that should have usage examples:
1. **`Terrain.__init__()`** - Show how to create terrain object with data
2. **`load_dem_files()`** - Show loading from different file types
3. **`apply_transforms()`** - Show chaining multiple transforms

### Example Quality
1. **Good examples**:
   - `position_camera_relative()` - Examples show cardinal directions
   - `elevation_colormap()` - Examples show colormap usage

2. **Examples needing improvement**:
   - `create_mesh()` - Example parameters could be explained better
   - `set_color_mapping()` - Lambda function example could be clearer

## Type Hint Assessment

### Type Hints Present
- ✅ `position_camera_relative()` - Complete type hints
- ✅ `load_dem_files()` - Good return type annotation
- ⚠️ `Terrain` class - Partial type hints on methods

### Type Hints Missing
- ❌ `clear_scene()` - No type hints
- ❌ `setup_light()` - No parameter type hints
- ⚠️ `add_data_layer()` - Return type hint missing

### Recommendations
1. Add `from typing import Optional, Tuple, List` where needed
2. Use `np.ndarray` with shape information for array parameters
3. Document complex types in docstrings even if type hints exist
4. Use `Union` for multiple possible return types

## Cross-Reference and Consistency

### Documentation Consistency
1. **Parameter naming**:
   - ✅ Consistent use of `dem_data`, `transform`, `crs`
   - ⚠️ Sometimes `camera_type` vs `type` - clarify intent

2. **Terminology**:
   - ✅ Consistent use of "mesh", "terrain", "transform"
   - ⚠️ "DEM" vs "dem data" - standardize

3. **Documentation style**:
   - Mixed styles between functions
   - Recommendation: Standardize on NumPy docstring format

## Quality Metrics

- **Average docstring length**: X words
- **Functions with complete docs**: Y%
- **Parameters with descriptions**: Z%
- **Return values documented**: W%
- **Examples per 10 functions**: V
- **Tone consistency score**: U/100
- **Type documentation alignment**: T/100

## Standards & Best Practices

### Current Standards Compliance
1. **NumPy docstring format**: Partially followed
   - Some functions use full format
   - Others use simplified format
   - Recommendation: Standardize on one format

2. **Type hints**: Inconsistently applied
   - Modern functions have hints
   - Older functions lack hints
   - Plan: Add to all public API

3. **Examples**: Inconsistent coverage
   - Complex functions: 50% have examples
   - Simple functions: 20% have examples
   - Recommendation: Examples for all public API

## Improvement Recommendations

### Immediate (High Priority)

1. **Add missing class docstrings**
   - `Terrain` class - constructor and major methods
   - `TerrainCache` class - purpose and usage

2. **Document critical public functions**
   - `load_dem_files()` - with file format examples
   - `create_mesh()` - with complete parameter explanation
   - `apply_transforms()` - with transform examples

3. **Fix parameter documentation gaps**
   - Add array shape specifications
   - Clarify CRS format requirements
   - Explain coordinate system assumptions

### Medium Priority

1. **Add type hints** to functions lacking them
2. **Add examples** to complex functions
3. **Standardize docstring format** across module
4. **Improve tone consistency** in docstrings

### Long-term (Quality Maintenance)

1. **Establish docstring standards** in CLAUDE.md
2. **Add pre-commit hooks** to check docstrings
3. **Regular documentation audits** (quarterly)
4. **Documentation review in code reviews**

## Implementation Roadmap

### Week 1: Critical Functions
- [ ] Add `Terrain` class docstrings
- [ ] Document all public API entry points
- [ ] Fix parameter documentation gaps

### Week 2: Completeness
- [ ] Add docstrings to remaining functions
- [ ] Add usage examples
- [ ] Improve return value documentation

### Week 3: Quality
- [ ] Standardize docstring format
- [ ] Add type hints
- [ ] Review tone and clarity

### Week 4: Validation
- [ ] Run documentation script
- [ ] Verify all docs accurate
- [ ] Update API reference

## Integration with Tools

### Using `scripts/python-docs.py`
- `npm run docs:python:validate` - Find missing docstrings
- `npm run docs:python:stats` - Track improvement metrics
- `npm run docs:python` - Regenerate API_REFERENCE.md

### Integration with /docs command
- `/docs python` - Regenerate after documentation updates
- `/docs python:validate` - Identify remaining gaps
- Commit regenerated `docs/API_REFERENCE.md` after docstring updates

## Success Criteria

- ✅ All public classes have docstrings
- ✅ All public functions documented with Parameters and Returns
- ✅ All parameters have type and description
- ✅ Complex functions have usage examples
- ✅ Documentation matches actual function signatures
- ✅ Tone is consistent and professional
- ✅ Type hints added where applicable
- ✅ `docs/API_REFERENCE.md` auto-generated and accurate

## Error Handling

- Skip files that cannot be parsed (note in report)
- Handle functions with complex signatures gracefully
- Note functions with unclear purpose for manual review
- Flag potentially stale documentation for human verification

## Related Agents and Tools

- Use `test-coverage-advisor` for testing undocumented code
- Reference `documentation-auditor` for overall docs quality
- Coordinate with TDD workflow for new function documentation
- Integration with `/docs python` command for regeneration

Execute this audit to ensure Python code is well-documented, maintainable, and easy for users to understand and use effectively.
