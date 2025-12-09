"""
Scoring module for terrain suitability analysis.

Provides transformation functions and combination logic for computing
multi-factor suitability scores (e.g., sledding suitability).

Transformation types:
- trapezoidal: Sweet spot with ramp-up/down (e.g., slope angle, snow depth)
- dealbreaker: Step function with optional falloff (e.g., cliff detection)
- linear: Simple normalization with optional power/invert (e.g., coverage ratio)
- terrain_consistency: Combined roughness + slope_std metric

Combination:
- ScoreComponent: Defines a single scoring factor
- ScoreCombiner: Combines components into final score using:
  - Additive components (weighted sum)
  - Multiplicative components (penalties)
"""

from src.scoring.transforms import (
    trapezoidal,
    dealbreaker,
    linear,
    terrain_consistency,
)
from src.scoring.combiner import ScoreComponent, ScoreCombiner

__all__ = [
    # Transforms
    "trapezoidal",
    "dealbreaker",
    "linear",
    "terrain_consistency",
    # Combiner
    "ScoreComponent",
    "ScoreCombiner",
]
