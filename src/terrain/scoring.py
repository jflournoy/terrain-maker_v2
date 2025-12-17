"""
Scoring functions for terrain suitability analysis.

This module contains scoring functions for evaluating terrain suitability
for various activities like sledding and cross-country skiing.
"""

import numpy as np


def trapezoid_score(
    value,
    min_value: float,
    optimal_min: float,
    optimal_max: float,
    max_value: float,
) -> np.ndarray:
    """
    Compute trapezoid (sweet spot) scoring for a parameter.

    The trapezoid function creates a sweet spot scoring pattern:
    - Below min_value: score = 0
    - Between min_value and optimal_min: linear ramp from 0 to 1
    - Between optimal_min and optimal_max: score = 1 (optimal range)
    - Between optimal_max and max_value: linear ramp from 1 to 0
    - Above max_value: score = 0

    This is useful for parameters where there's a "just right" range and
    both too little and too much are bad.

    Args:
        value: The value(s) to score (scalar or array)
        min_value: Minimum acceptable value (below this scores 0)
        optimal_min: Start of optimal range (scores 1.0)
        optimal_max: End of optimal range (scores 1.0)
        max_value: Maximum acceptable value (above this scores 0)

    Returns:
        Score(s) in range [0, 1], same shape as value

    Example:
        >>> # Snow depth scoring: 4-8-16-24 inches
        >>> trapezoid_score(12.0, 4.0, 8.0, 16.0, 24.0)
        1.0  # In optimal range
        >>> trapezoid_score(6.0, 4.0, 8.0, 16.0, 24.0)
        0.5  # Halfway up ramp
    """
    # Convert to numpy array for consistent handling
    value = np.atleast_1d(value)
    scores = np.zeros_like(value, dtype=np.float64)

    # Below min_value: score = 0 (already initialized)

    # Ramp up: min_value to optimal_min
    mask = (value >= min_value) & (value < optimal_min)
    if np.any(mask):
        scores[mask] = (value[mask] - min_value) / (optimal_min - min_value)

    # Optimal range: optimal_min to optimal_max
    mask = (value >= optimal_min) & (value <= optimal_max)
    scores[mask] = 1.0

    # Ramp down: optimal_max to max_value
    mask = (value > optimal_max) & (value <= max_value)
    if np.any(mask):
        scores[mask] = 1.0 - (value[mask] - optimal_max) / (max_value - optimal_max)

    # Above max_value: score = 0 (already initialized)

    # Return scalar if input was scalar
    if scores.shape == (1,):
        return float(scores[0])
    return scores


def sledding_deal_breakers(
    slope,
    roughness,
    coverage_months,
    max_slope: float = 40.0,
    max_roughness: float = 6.0,
    min_coverage: float = 0.5,
) -> np.ndarray:
    """
    Identify deal breaker conditions for sledding.

    Some terrain features are absolute deal breakers for sledding:
    - Slope > 40°: Extreme cliffs (double-black-diamond terrain)
    - Roughness > 6m: Very rough terrain (cliff faces, boulders)
    - Coverage < 0.5 months: Not enough snow season

    Args:
        slope: Terrain slope in degrees (scalar or array)
        roughness: Elevation std dev in meters (scalar or array) - physical terrain roughness
        coverage_months: Months of snow coverage (scalar or array)
        max_slope: Maximum acceptable slope (default: 40°)
        max_roughness: Maximum acceptable roughness in meters (default: 6.0m)
        min_coverage: Minimum acceptable coverage months (default: 0.5)

    Returns:
        Boolean array indicating deal breaker locations (True = deal breaker)

    Example:
        >>> sledding_deal_breakers(45.0, 3.0, 3.0)
        True  # Too steep (>40°)
        >>> sledding_deal_breakers(10.0, 2.0, 3.0)
        False  # Acceptable
        >>> sledding_deal_breakers(15.0, 8.0, 3.0)
        True  # Too rough (>6m)
    """
    # Convert to numpy arrays
    slope = np.atleast_1d(slope)
    roughness = np.atleast_1d(roughness)
    coverage_months = np.atleast_1d(coverage_months)

    # Check each deal breaker condition
    too_steep = slope > max_slope
    too_rough = roughness > max_roughness
    insufficient_coverage = coverage_months < min_coverage

    # Combine with logical OR
    is_deal_breaker = too_steep | too_rough | insufficient_coverage

    # Return scalar if inputs were scalar
    if is_deal_breaker.shape == (1,):
        return bool(is_deal_breaker[0])
    return is_deal_breaker


def coverage_diminishing_returns(coverage_months, tau: float = 2.0) -> np.ndarray:
    """
    Score snow coverage with diminishing returns.

    Some coverage is critical, but beyond a certain point more coverage
    doesn't add much value. Uses exponential saturation:
        score = 1 - exp(-coverage_months / tau)

    This gives:
    - 0 months: 0.0
    - 1 month: ~0.39
    - 2 months: ~0.63
    - 4 months: ~0.86
    - 6 months: ~0.95

    Args:
        coverage_months: Months of snow coverage (scalar or array)
        tau: Time constant for saturation (default: 2.0 months)

    Returns:
        Score(s) in range [0, 1), same shape as coverage_months

    Example:
        >>> coverage_diminishing_returns(2.0)
        0.632  # Two months gives ~63% score
        >>> coverage_diminishing_returns(6.0)
        0.950  # Six months gives ~95% score
    """
    coverage_months = np.atleast_1d(coverage_months)

    # Exponential saturation: 1 - e^(-x/tau)
    scores = 1.0 - np.exp(-coverage_months / tau)

    # Return scalar if input was scalar
    if scores.shape == (1,):
        return float(scores[0])
    return scores


def sledding_synergy_bonus(
    slope,
    snow_depth,
    coverage_months,
    roughness,
) -> np.ndarray:
    """
    Compute synergy bonuses for exceptional sledding combinations.

    Some combinations of factors create exceptional sledding that's more
    than the sum of its parts. This function identifies these synergies
    and applies multiplicative bonuses.

    Synergies (applied in hierarchical priority order):
    1. Perfect combo (slope + snow + coverage) = +30% (highest priority)
    2. Consistent coverage + good slope + smooth terrain = +15%
    3. Moderate slope + very smooth terrain = +20%
    4. Smooth terrain + perfect slope = +10% (lowest priority)

    Higher priority bonuses prevent lower priority bonuses from applying
    to avoid over-rewarding.

    Args:
        slope: Terrain slope in degrees (scalar or array)
        snow_depth: Snow depth in inches (scalar or array)
        coverage_months: Months of snow coverage (scalar or array)
        roughness: Elevation std dev in meters (scalar or array) - physical terrain roughness

    Returns:
        Bonus multiplier(s) >= 1.0, same shape as inputs

    Example:
        >>> sledding_synergy_bonus(9.0, 12.0, 3.5, 2.0)
        1.30  # Perfect combo bonus only
        >>> sledding_synergy_bonus(9.0, 12.0, 4.0, 1.5)
        1.65  # Perfect combo + smooth terrain bonuses stack
        >>> sledding_synergy_bonus(15.0, 5.0, 1.0, 5.0)
        1.0  # No synergies
    """
    # Convert to numpy arrays
    slope = np.atleast_1d(slope)
    snow_depth = np.atleast_1d(snow_depth)
    coverage_months = np.atleast_1d(coverage_months)
    roughness = np.atleast_1d(roughness)

    # Start with no bonus
    bonus = np.ones_like(slope, dtype=np.float64)

    # Define optimal ranges
    perfect_slope = (slope >= 6.0) & (slope <= 12.0)
    optimal_snow = (snow_depth >= 2.0) & (snow_depth <= 12.0)  # Match trapezoid sweet spot
    long_coverage = coverage_months >= 3.0
    very_long_coverage = coverage_months >= 4.0
    smooth_terrain = roughness < 3.0  # < 3m elevation variation (gentle rolling hills)
    very_smooth_terrain = roughness < 1.5  # < 1.5m variation (golf course smooth)

    # Track which locations have received bonuses
    has_bonus = np.zeros_like(slope, dtype=bool)

    # Synergy 1: Perfect combo (slope + snow + coverage) = +30%
    # This is the highest tier - areas with this don't get other bonuses
    synergy1 = perfect_slope & optimal_snow & long_coverage
    bonus[synergy1] *= 1.30
    has_bonus[synergy1] = True

    # Synergy 2: Consistent coverage + good slope + smooth terrain = +15%
    # Only apply if no higher bonus already given
    synergy2 = very_long_coverage & perfect_slope & smooth_terrain & ~has_bonus
    bonus[synergy2] *= 1.15
    has_bonus[synergy2] = True

    # Synergy 3: Moderate slope + very smooth terrain = +20%
    # Can stack with synergy 1 (perfect combo), but not others
    synergy3 = perfect_slope & very_smooth_terrain & ~synergy2
    bonus[synergy3] *= 1.20

    # Synergy 4: Smooth terrain + perfect slope = +10%
    # Only apply if no other bonus given (lowest priority)
    synergy4 = smooth_terrain & perfect_slope & ~has_bonus & ~synergy3
    bonus[synergy4] *= 1.10

    # Return scalar if inputs were scalar
    if bonus.shape == (1,):
        return float(bonus[0])
    return bonus


def compute_sledding_score(
    snow_depth,
    slope,
    coverage_months,
    roughness,
) -> np.ndarray:
    """
    Compute overall sledding suitability score.

    Combines multiple factors using a multiplicative model where:
    1. Deal breakers → immediate zero score
    2. Base score = snow_score × slope_score × coverage_score
    3. Final score = base_score × synergy_bonus

    The multiplicative approach ensures that poor performance in any
    one factor significantly reduces the overall score, while synergies
    can boost exceptional combinations.

    Args:
        snow_depth: Snow depth in inches (scalar or array)
        slope: Terrain slope in degrees (scalar or array)
        coverage_months: Months of snow coverage (scalar or array)
        roughness: Elevation std dev in meters (scalar or array) - physical terrain roughness

    Returns:
        Score(s) in range [0, ~1.5], same shape as inputs
        (Can exceed 1.0 due to synergy bonuses)

    Example:
        >>> # Perfect conditions
        >>> compute_sledding_score(12.0, 9.0, 4.0, 2.0)
        1.12  # High score with bonuses
        >>> # Deal breaker (too steep)
        >>> compute_sledding_score(12.0, 45.0, 4.0, 2.0)
        0.0  # Too steep (>40°)
        >>> # Deal breaker (too rough)
        >>> compute_sledding_score(12.0, 10.0, 4.0, 8.0)
        0.0  # Too rough (>6m)
    """
    # Convert to numpy arrays
    snow_depth = np.atleast_1d(snow_depth)
    slope = np.atleast_1d(slope)
    coverage_months = np.atleast_1d(coverage_months)
    roughness = np.atleast_1d(roughness)

    # Check for deal breakers (this returns boolean or array)
    is_deal_breaker = sledding_deal_breakers(
        slope=slope,
        roughness=roughness,
        coverage_months=coverage_months,
    )

    # Ensure is_deal_breaker is array for consistent indexing
    is_deal_breaker = np.atleast_1d(is_deal_breaker)

    # Initialize scores
    scores = np.zeros_like(slope, dtype=np.float64)

    # Only compute scores for non-deal-breaker locations
    valid = ~is_deal_breaker

    if not np.any(valid):
        # All are deal breakers
        if scores.shape == (1,):
            return 0.0
        return scores

    # Score snow depth: 1-4-12-20 inches (25-100-300-500mm)
    snow_score = trapezoid_score(
        value=snow_depth[valid],
        min_value=1.0,  # Marginal on grass (~25mm)
        optimal_min=4.0,  # Good coverage (~100mm)
        optimal_max=12.0,  # Excellent range (~300mm)
        max_value=20.0,  # Too much for little kids (~500mm)
    )

    # Score slope: 1-6-12-20 degrees (lowered min from 3 to 1, 35° is deal breaker)
    slope_score = trapezoid_score(
        value=slope[valid],
        min_value=1.0,  # Allow gentler slopes
        optimal_min=6.0,
        optimal_max=12.0,
        max_value=20.0,
    )

    # Score coverage with diminishing returns
    coverage_score = coverage_diminishing_returns(coverage_months[valid])

    # Multiplicative base score
    base_score = snow_score * slope_score * coverage_score

    # Apply synergy bonuses
    synergy = sledding_synergy_bonus(
        slope=slope[valid],
        snow_depth=snow_depth[valid],
        coverage_months=coverage_months[valid],
        roughness=roughness[valid],
    )

    # Final score with bonuses
    scores[valid] = base_score * synergy

    # Return scalar if inputs were scalar
    if scores.shape == (1,):
        return float(scores[0])
    return scores


# =============================================================================
# Cross-Country Skiing Scoring
# =============================================================================


def xc_skiing_deal_breakers(
    snow_coverage: float | np.ndarray,
    min_coverage: float = 0.15,
) -> bool | np.ndarray:
    """
    Identify deal breaker conditions for cross-country skiing.

    XC skiing depends primarily on snow reliability. Parks handle terrain safety,
    so only snow coverage matters as a deal breaker.

    Deal breaker:
    - Snow coverage < 15% of days (< ~18 days per winter season)

    Args:
        snow_coverage: Fraction of days with snow (0-1)
        min_coverage: Minimum snow coverage threshold (default 0.15)

    Returns:
        Boolean or array of booleans indicating deal breaker conditions

    Examples:
        >>> xc_skiing_deal_breakers(0.5)
        False  # 50% coverage is good

        >>> xc_skiing_deal_breakers(0.1)
        True  # Only 10% coverage - too unreliable

        >>> xc_skiing_deal_breakers(np.array([0.1, 0.3, 0.8]))
        array([True, False, False])
    """
    # Convert to numpy arrays
    snow_coverage = np.atleast_1d(snow_coverage)

    # Check deal breaker condition
    insufficient_coverage = snow_coverage < min_coverage

    # Return scalar if input was scalar
    if insufficient_coverage.shape == (1,):
        return bool(insufficient_coverage[0])
    return insufficient_coverage


def compute_xc_skiing_score(
    snow_depth: float | np.ndarray,
    snow_coverage: float | np.ndarray,
    snow_consistency: float | np.ndarray,
    min_depth: float = 50.0,
    optimal_depth_min: float = 100.0,
    optimal_depth_max: float = 400.0,
    max_depth: float = 800.0,
    min_coverage: float = 0.15,
) -> float | np.ndarray:
    """
    Compute overall cross-country skiing suitability score.

    XC skiing scoring focuses on snow conditions (parks handle terrain safety):
    - Snow depth trapezoid (30% weight): optimal 100-400mm, usable 50-800mm
    - Snow coverage linear (60% weight): proportional to days with snow
    - Snow consistency inverted (10% weight): low CV = reliable

    Deal breaker:
    - Snow coverage < 15% (< ~18 days per season) → Score = 0

    Final score combines:
    - Base score: Weighted sum of depth, coverage, consistency
    - Range: 0 (poor) to 1.0 (excellent)

    Args:
        snow_depth: Snow depth in mm (SNODAS native units)
        snow_coverage: Fraction of days with snow (0-1)
        snow_consistency: Coefficient of variation (lower is better, 0-1.5)
        min_depth: Minimum usable snow depth (mm)
        optimal_depth_min: Lower bound of optimal depth range (mm)
        optimal_depth_max: Upper bound of optimal depth range (mm)
        max_depth: Maximum usable snow depth (mm)
        min_coverage: Minimum snow coverage threshold (15% = 0.15)

    Returns:
        XC skiing suitability score (0-1)

    Examples:
        >>> compute_xc_skiing_score(250.0, 0.75, 0.3)
        0.78  # Excellent conditions

        >>> compute_xc_skiing_score(150.0, 0.1, 0.5)
        0.0  # Deal breaker - coverage too low

        >>> compute_xc_skiing_score(100.0, 0.5, 0.8)
        0.53  # Good enough conditions
    """
    # Convert to numpy arrays
    snow_depth = np.atleast_1d(snow_depth)
    snow_coverage = np.atleast_1d(snow_coverage)
    snow_consistency = np.atleast_1d(snow_consistency)

    # Initialize scores to zero
    scores = np.zeros_like(snow_depth, dtype=float)

    # Check deal breakers (ensure array for indexing)
    is_deal_breaker = np.atleast_1d(xc_skiing_deal_breakers(
        snow_coverage=snow_coverage,
        min_coverage=min_coverage,
    ))

    # Only score valid terrain (non-deal breaker)
    valid = ~is_deal_breaker

    if not np.any(valid):
        # All deal breakers
        if scores.shape == (1,):
            return 0.0
        return scores

    # Score snow depth with trapezoid function
    depth_score = trapezoid_score(
        value=snow_depth[valid],
        min_value=min_depth,
        optimal_min=optimal_depth_min,
        optimal_max=optimal_depth_max,
        max_value=max_depth,
    )

    # Score coverage with linear transform (proportional to days with snow)
    # No diminishing returns - 50% coverage = 0.5 score
    coverage_score = np.clip(snow_coverage[valid], 0.0, 1.0)

    # Score consistency (inverted - lower CV is better)
    # Normalize to 0-1 range where 0 CV = 1.0 score, 1.5 CV = 0.0 score
    consistency_score = np.clip(1.0 - (snow_consistency[valid] / 1.5), 0.0, 1.0)

    # Weighted sum (no multiplicative factors - additive only)
    # Weights: depth 30%, coverage 60%, consistency 10%
    scores[valid] = (
        0.30 * depth_score +
        0.60 * coverage_score +
        0.10 * consistency_score
    )

    # Return scalar if inputs were scalar
    if scores.shape == (1,):
        return float(scores[0])
    return scores
