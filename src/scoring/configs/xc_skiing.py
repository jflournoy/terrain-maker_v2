"""
Cross-country skiing suitability scoring configuration.

This config defines how snow statistics are combined into a cross-country
skiing suitability score. XC skiing depends primarily on snow conditions
(depth, coverage, consistency) with no penalty for slope.

Score formula:
    final = weighted sum of snow metrics (no multiplicative penalties)

Components:
- snow_depth: Adequate snow base (trapezoidal, 100-400mm sweet spot)
- snow_coverage: Reliability of snow days (linear)
- snow_consistency: Snow reliability (RMS of inter/intra-season CVs)

Unlike sledding, no penalties for slope steepness - XC skiing can adapt
to various terrain grades. Only snow quality matters.
"""

import numpy as np

from src.scoring.combiner import ScoreComponent, ScoreCombiner


def create_xc_skiing_scorer() -> ScoreCombiner:
    """
    Create the XC skiing suitability scorer.

    Returns:
        ScoreCombiner configured for XC skiing suitability analysis.

    Example:
        >>> scorer = create_xc_skiing_scorer()
        >>> score = scorer.compute({
        ...     "snow_depth": 250.0,     # mm
        ...     "snow_coverage": 0.75,   # ratio 0-1
        ...     "snow_consistency": 0.25,# CV (lower is better)
        ... })
    """
    return ScoreCombiner(
        name="xc_skiing_suitability",
        components=[
            # Snow depth: adequate base for XC skiing
            # Units: millimeters (SNODAS native)
            # WEIGHT: 40% - deep snow base is essential for XC
            ScoreComponent(
                name="snow_depth",
                transform="trapezoidal",
                transform_params={
                    "sweet_range": (100, 400),  # ~4-16 inches ideal (10-40cm)
                    "ramp_range": (50, 800),    # 2-32 inches usable (5-80cm)
                },
                role="additive",
                weight=0.30,
            ),

            # Snow coverage: fraction of winter days with snow
            # Higher is better (more XC skiing opportunities)
            # WEIGHT: 35% - reliability of snow days is very important for XC
            ScoreComponent(
                name="snow_coverage",
                transform="linear",
                transform_params={
                    "value_range": (0, 1),
                    "power": 0.5,  # sqrt for diminishing returns
                },
                role="additive",
                weight=0.60,
            ),

            # Snow consistency: year-to-year reliability (CV)
            # Lower CV = more reliable = better (inverted)
            # WEIGHT: 25% - consistent snow is important for planning XC routes
            ScoreComponent(
                name="snow_consistency",
                transform="linear",
                transform_params={
                    "value_range": (0, 1.5),  # CV range
                    "invert": True,           # Low CV = high score
                },
                role="additive",
                weight=0.10,
            ),
        ],
    )


# Default scorer instance
DEFAULT_XC_SKIING_SCORER = create_xc_skiing_scorer()


# Export as dict for JSON serialization
DEFAULT_XC_SKIING_CONFIG = DEFAULT_XC_SKIING_SCORER.to_dict()


def get_required_inputs() -> dict[str, str]:
    """
    Get documentation of required inputs for the XC skiing scorer.

    Returns:
        Dictionary mapping input names to descriptions.
    """
    return {
        "snow_depth": "Median max snow depth in mm (from SNODAS stats)",
        "snow_coverage": "Mean snow day ratio 0-1 (from SNODAS stats)",
        "snow_consistency": "RMS of inter/intra-season CVs via snow_consistency()",
    }


def compute_derived_inputs(snow_stats: dict) -> dict:
    """
    Compute derived inputs from snow statistics.

    This helper prepares snow metrics for the XC skiing scorer.

    Args:
        snow_stats: Dictionary with SNODAS statistics

    Returns:
        Dictionary ready to pass to scorer.compute()
    """
    from src.scoring.transforms import snow_consistency

    # Snow consistency: RMS of inter-season and intra-season CVs
    snow_cons = snow_consistency(
        snow_stats["interseason_cv"],
        snow_stats["mean_intraseason_cv"],
    )

    return {
        "snow_depth": snow_stats["median_max_depth"],
        "snow_coverage": snow_stats["mean_snow_day_ratio"],
        "snow_consistency": snow_cons,
    }


def compute_improved_xc_skiing_score(
    snow_stats: dict,
) -> np.ndarray:
    """
    Compute improved XC skiing score using deal breakers and linear coverage.

    This is the new scoring system that uses:
    - Trapezoid function for snow depth (100-400mm optimal)
    - Linear coverage (proportional to snow days, no diminishing returns)
    - Inverted consistency (lower CV = better)
    - Hard deal breaker: Coverage < 15% (< ~18 days per season)
    - Weighted sum (depth 30%, coverage 60%, consistency 10%)

    Parks handle terrain safety, so only snow conditions matter.

    Args:
        snow_stats: Dictionary with SNODAS statistics

    Returns:
        XC skiing suitability score array (0-1.0)
    """
    from src.terrain.scoring import compute_xc_skiing_score
    from src.scoring.transforms import snow_consistency

    # Get snow metrics
    snow_depth_mm = snow_stats["median_max_depth"]
    snow_coverage = snow_stats["mean_snow_day_ratio"]

    # Compute consistency (RMS of inter/intra-season CVs)
    snow_cons = snow_consistency(
        snow_stats["interseason_cv"],
        snow_stats["mean_intraseason_cv"],
    )

    # Compute improved score
    score = compute_xc_skiing_score(
        snow_depth=snow_depth_mm,
        snow_coverage=snow_coverage,
        snow_consistency=snow_cons,
    )

    return score
