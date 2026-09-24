"""
Flow computation pipeline with basin preservation.

This module provides a validated flow computation pattern that properly handles:
1. Ocean detection
2. Endorheic basin detection and preservation
3. Water body (lake) integration with basin awareness
4. DEM conditioning with combined masks
5. Flow direction and drainage area computation

The implementation lives in :mod:`src.terrain.flow_accumulation`; this module
re-exports it so ``from src.terrain.flow_pipeline import compute_flow_with_basins``
keeps working.
"""

from src.terrain.flow_accumulation import compute_flow_with_basins

__all__ = ["compute_flow_with_basins"]
