"""
Scoring configurations for different use cases.

Available configs:
- sledding: Sledding suitability scoring
- xc_skiing: Cross-country skiing suitability scoring
"""

from src.scoring.configs.sledding import (
    DEFAULT_SLEDDING_SCORER,
    DEFAULT_SLEDDING_CONFIG,
    create_default_sledding_scorer,
    compute_derived_inputs as sledding_compute_derived_inputs,
    get_required_inputs as sledding_get_required_inputs,
)

from src.scoring.configs.xc_skiing import (
    DEFAULT_XC_SKIING_SCORER,
    DEFAULT_XC_SKIING_CONFIG,
    create_xc_skiing_scorer,
    compute_derived_inputs as xc_skiing_compute_derived_inputs,
    get_required_inputs as xc_skiing_get_required_inputs,
)

__all__ = [
    "DEFAULT_SLEDDING_SCORER",
    "DEFAULT_SLEDDING_CONFIG",
    "create_default_sledding_scorer",
    "sledding_compute_derived_inputs",
    "sledding_get_required_inputs",
    "DEFAULT_XC_SKIING_SCORER",
    "DEFAULT_XC_SKIING_CONFIG",
    "create_xc_skiing_scorer",
    "xc_skiing_compute_derived_inputs",
    "xc_skiing_get_required_inputs",
]
