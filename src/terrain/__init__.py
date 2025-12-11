"""
Terrain visualization and analysis package.

Core functionality:
- Terrain class for DEM processing and rendering
- Pipeline for transform-based data processing
- GriddedDataLoader for memory-safe tiled processing
- Advanced visualization features (slopes, drive-time curves, legends)
"""

from .core import Terrain
from .pipeline import TerrainPipeline
from .gridded_data import (
    GriddedDataLoader,
    TiledDataConfig,
    downsample_for_viz,
    create_mock_snow_data,
)
from .advanced_viz import (
    horn_slope,
    load_drive_time_data,
    create_drive_time_curves,
    create_values_legend,
)

__all__ = [
    "Terrain",
    "TerrainPipeline",
    "GriddedDataLoader",
    "TiledDataConfig",
    "downsample_for_viz",
    "create_mock_snow_data",
    "horn_slope",
    "load_drive_time_data",
    "create_drive_time_curves",
    "create_values_legend",
]
