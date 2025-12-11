"""
Snow data processing and analysis.

This module provides utilities for working with snow data:
- SNODAS data loading and processing
- Snow statistics computation
- Slope statistics for snow analysis
"""

from .snodas import (
    batch_process_snodas_data,
    calculate_snow_statistics,
)

__all__ = [
    "batch_process_snodas_data",
    "calculate_snow_statistics",
]
