Here's the current design specification for the terrain visualization project:

# Terrain Visualization Design Specification

## Core Class: Terrain

The core functionality is encapsulated in a Terrain class that handles geographic data, transforms, and visualization.

### Initialization
- Create with either 3D coordinates (x,y,z) or 2D coordinates (x,y) with constant z
- Set up caching using joblib.Memory
- Initialize empty lists/dicts for transforms, data layers, and color mapping

### Key Components

1. Transforms
- List of transformation functions to be applied to coordinates and data
- Applied in sequence to all data
- Should handle case of no transforms gracefully
- Transforms are cached using joblib

2. Data Layers
- Geographic data sources (slopes, snow coverage, etc.)
- Each layer includes:
  - Original data
  - Original transform
  - Transformed state flag
- Must be aligned to same coordinate space

3. Color Mapping
- Function to convert data values to colors
- Specifies which data layers to use
- Color computation is cached

### Method Dependencies
1. First: Add transforms and data layers
2. Then: Apply transforms
3. Then: Compute colors
4. Finally: Create mesh

### Key Methods

1. `add_transform(transform_func)`
- Add transform to pipeline
- Transforms are applied in order added

2. `add_data_layer(name, data, transform)`
- Add new geographic data source
- Store original transform for alignment

3. `set_color_mapping(mapping_func, source_layers)`
- Define how to convert data to colors
- Specify which layers to use

4. `apply_transforms()`
- Apply all transforms to coordinates and data
- Uses joblib caching
- Handles case of no transforms

5. `compute_colors()`
- Uses specified mapping to generate colors
- Requires transforms to be applied first
- Uses joblib caching

6. `create_mesh()`
- Creates Blender mesh from transformed data
- Requires colors to be computed first

### Caching Strategy
- Use joblib.Memory for persistent caching
- Cache both transforms and color computation
- Cache keys based on input data and parameters

### Future Components To Be Implemented
1. Mesh creation functionality
2. Scene setup and rendering
3. Additional visualization elements (legends, boundaries)

This design allows for:
- Flexible data sources
- Pluggable transform pipeline
- Customizable color mapping
- Efficient caching of expensive operations
- Clear dependency chain of operations

The class maintains all state and caching while keeping the actual operations (transforms, color mapping) as pure functions that can be passed in.