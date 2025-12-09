"""
Default sledding suitability scoring configuration.

This config defines how terrain and snow statistics are combined into
a sledding suitability score. Users can modify this config or create
their own based on local conditions.

Score formula:
    final = (weighted sum of additive) × (product of multiplicative)

Components:
- Additive (weighted sum to 1.0):
  - slope_mean: Ideal slope angle (trapezoidal, 5-15° sweet spot)
  - snow_depth: Adequate snow coverage (trapezoidal, 150-500mm sweet spot)
  - snow_coverage: Reliability of snow days (linear)
  - snow_consistency: Snow reliability (RMS of inter/intra-season CVs)
  - aspect_bonus: North-facing snow retention bonus (linear)
  - runout_bonus: Safe stopping area available (linear)

- Multiplicative (penalties, only extreme values):
  - cliff_penalty: Dangerous steep sections (dealbreaker on p95)
  - terrain_consistency: Extreme roughness only (soft threshold at 50%)
"""

from src.scoring.combiner import ScoreComponent, ScoreCombiner


def create_default_sledding_scorer() -> ScoreCombiner:
    """
    Create the default sledding suitability scorer.

    Returns:
        ScoreCombiner configured for sledding suitability analysis.

    Example:
        >>> scorer = create_default_sledding_scorer()
        >>> score = scorer.compute({
        ...     "slope_mean": 10.0,      # degrees
        ...     "snow_depth": 300.0,     # mm
        ...     "snow_coverage": 0.7,    # ratio 0-1
        ...     "snow_consistency": 0.3, # CV (lower is better)
        ...     "aspect_bonus": 0.03,    # northness × strength
        ...     "runout_bonus": 1.0,     # 1.0 if slope_min < 5°
        ...     "slope_p95": 20.0,       # degrees (for cliff detection)
        ...     "roughness": 10.0,       # meters
        ...     "slope_std": 3.0,        # degrees
        ... })
    """
    return ScoreCombiner(
        name="sledding_suitability",
        components=[
            # =================================================================
            # ADDITIVE COMPONENTS (weighted sum = 1.0)
            # =================================================================

            # Slope angle: 5-15° is ideal for sledding
            # Too flat = boring, too steep = dangerous
            # WEIGHT: 30% - most important factor for sledding experience
            ScoreComponent(
                name="slope_mean",
                transform="trapezoidal",
                transform_params={
                    "sweet_range": (5, 15),   # Ideal: 5-15 degrees
                    "ramp_range": (3, 25),    # Usable: 3-25 degrees
                },
                role="additive",
                weight=0.30,
            ),

            # Snow depth: need enough cushion but not too deep
            # Units: millimeters (SNODAS native)
            # WEIGHT: 15% - enough snow matters but less critical than other factors
            ScoreComponent(
                name="snow_depth",
                transform="trapezoidal",
                transform_params={
                    "sweet_range": (150, 500),  # ~6-20 inches ideal
                    "ramp_range": (50, 1000),   # 2-40 inches usable
                },
                role="additive",
                weight=0.15,
            ),

            # Snow coverage: fraction of winter days with snow
            # Higher is better (more sledding opportunities)
            # WEIGHT: 25% - reliability of snow days is very important
            ScoreComponent(
                name="snow_coverage",
                transform="linear",
                transform_params={
                    "value_range": (0, 1),
                    "power": 0.5,  # sqrt for diminishing returns
                },
                role="additive",
                weight=0.25,
            ),

            # Snow consistency: year-to-year reliability (CV)
            # Lower CV = more reliable = better (inverted)
            # WEIGHT: 20% - predictable snow year-over-year is important
            ScoreComponent(
                name="snow_consistency",
                transform="linear",
                transform_params={
                    "value_range": (0, 1.5),  # CV range
                    "invert": True,           # Low CV = high score
                },
                role="additive",
                weight=0.20,
            ),

            # Aspect bonus: north-facing slopes retain snow better
            # Pre-computed as: cos(aspect_radians) × aspect_strength × 0.05
            # Range: -0.05 (south) to +0.05 (north)
            # WEIGHT: 5% - minor bonus
            ScoreComponent(
                name="aspect_bonus",
                transform="linear",
                transform_params={
                    "value_range": (-0.05, 0.05),
                },
                role="additive",
                weight=0.05,
            ),

            # Runout bonus: safe stopping area
            # Pre-computed as: 1.0 if slope_min < 5° else 0.0
            # WEIGHT: 5% - safety bonus
            ScoreComponent(
                name="runout_bonus",
                transform="linear",
                transform_params={
                    "value_range": (0, 1),
                },
                role="additive",
                weight=0.05,
            ),

            # =================================================================
            # MULTIPLICATIVE COMPONENTS (penalties)
            # =================================================================

            # Cliff penalty: dangerous steep sections
            # Based on slope_p95 (95th percentile slope in tile)
            # Soft falloff from 25° to 35°
            ScoreComponent(
                name="slope_p95",
                transform="dealbreaker",
                transform_params={
                    "threshold": 25,   # Start penalizing at 25°
                    "falloff": 10,     # Full penalty at 35°
                },
                role="multiplicative",
            ),

            # Terrain consistency: smooth vs undulating
            # Pre-computed via terrain_consistency(roughness, slope_std)
            # Already in 0-1 range (1 = consistent, 0 = rough)
            ScoreComponent(
                name="terrain_consistency",
                transform="linear",
                transform_params={
                    "value_range": (0, 1),
                },
                role="multiplicative",
            ),
        ],
    )


# Default scorer instance
DEFAULT_SLEDDING_SCORER = create_default_sledding_scorer()


# Export as dict for JSON serialization
DEFAULT_SLEDDING_CONFIG = DEFAULT_SLEDDING_SCORER.to_dict()


def get_required_inputs() -> dict[str, str]:
    """
    Get documentation of required inputs for the sledding scorer.

    Returns:
        Dictionary mapping input names to descriptions.
    """
    return {
        "slope_mean": "Mean slope angle in degrees (from SlopeStatistics)",
        "snow_depth": "Median max snow depth in mm (from SNODAS stats)",
        "snow_coverage": "Mean snow day ratio 0-1 (from SNODAS stats)",
        "snow_consistency": "RMS of inter/intra-season CVs via snow_consistency()",
        "aspect_bonus": "Pre-computed: cos(aspect) × strength × 0.05",
        "runout_bonus": "Pre-computed: 1.0 if slope_min < 5° else 0.0",
        "slope_p95": "95th percentile slope in degrees (from SlopeStatistics)",
        "terrain_consistency": "Pre-computed via terrain_consistency() - extreme only",
    }


def compute_derived_inputs(
    slope_stats,
    snow_stats: dict,
) -> dict:
    """
    Compute derived inputs from raw statistics.

    This helper computes the pre-processed inputs that the scorer expects.

    Args:
        slope_stats: SlopeStatistics object from compute_tiled_slope_statistics()
        snow_stats: Dictionary with SNODAS statistics

    Returns:
        Dictionary ready to pass to scorer.compute()
    """
    import numpy as np
    from src.scoring.transforms import snow_consistency, terrain_consistency

    # Aspect bonus: northness × strength
    # cos(0°) = 1 (north), cos(180°) = -1 (south)
    northness = np.cos(np.radians(slope_stats.dominant_aspect))
    aspect_bonus = northness * slope_stats.aspect_strength * 0.05

    # Runout bonus: 1.0 if there's a flat area (slope_min < 5°)
    runout_bonus = np.where(slope_stats.slope_min < 5, 1.0, 0.0)

    # Snow consistency: RMS of inter-season and intra-season CVs
    snow_cons = snow_consistency(
        snow_stats["interseason_cv"],
        snow_stats["mean_intraseason_cv"],
    )

    # Terrain consistency: combined roughness + slope_std (only extreme penalty)
    terrain_cons = terrain_consistency(
        slope_stats.roughness,
        slope_stats.slope_std,
    )

    return {
        # Direct from slope stats
        "slope_mean": slope_stats.slope_mean,
        "slope_p95": slope_stats.slope_p95,
        # Direct from snow stats
        "snow_depth": snow_stats["median_max_depth"],
        "snow_coverage": snow_stats["mean_snow_day_ratio"],
        # Derived (combined metrics)
        "snow_consistency": snow_cons,
        "aspect_bonus": aspect_bonus,
        "runout_bonus": runout_bonus,
        "terrain_consistency": terrain_cons,
    }
