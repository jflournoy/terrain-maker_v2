"""
Scoring configurations for different use cases.

Available configs:
- sledding: Sledding suitability scoring
"""

from src.scoring.configs.sledding import (
    DEFAULT_SLEDDING_SCORER,
    DEFAULT_SLEDDING_CONFIG,
    create_default_sledding_scorer,
    compute_derived_inputs,
    get_required_inputs,
)

__all__ = [
    "DEFAULT_SLEDDING_SCORER",
    "DEFAULT_SLEDDING_CONFIG",
    "create_default_sledding_scorer",
    "compute_derived_inputs",
    "get_required_inputs",
]
