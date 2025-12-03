"""
Terrain visualization and analysis package.

Core functionality:
- Terrain class for DEM processing and rendering
- Pipeline for transform-based data processing
- Advanced visualization features (slopes, drive-time curves, legends)
"""

from .core import Terrain
from .pipeline import TerrainPipeline
from .advanced_viz import (
    horn_slope,
    load_drive_time_data,
    create_drive_time_curves,
    create_values_legend,
)

__all__ = [
    "Terrain",
    "TerrainPipeline",
    "horn_slope",
    "load_drive_time_data",
    "create_drive_time_curves",
    "create_values_legend",
]
