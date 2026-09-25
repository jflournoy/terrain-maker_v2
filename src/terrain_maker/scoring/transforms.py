"""
Scoring transformation functions.

All transformations convert raw values into scores in the range [0, 1].

Transformation types:
1. trapezoidal - sweet spot with ramp-up/down (e.g., slope_mean, snow_depth)
2. dealbreaker - step function with optional soft falloff (e.g., cliff detection)
3. linear - simple normalization with optional power scaling
4. terrain_consistency - combined roughness + slope_std metric

These are designed to be composable and user-configurable.
"""

from typing import Union
import numpy as np

# Type alias for values that can be scalar or array
NumericType = Union[float, np.ndarray]


def trapezoidal(
    value: NumericType,
    sweet_range: tuple[float, float],
    ramp_range: tuple[float, float],
) -> NumericType:
    """
    Trapezoidal transformation with a sweet spot.

    Returns 1.0 for values in the sweet spot, ramps up/down at edges,
    and returns 0.0 outside the ramp range.

    Shape:
              ___________
             /           \\
            /             \\
    _______/               \\_______
           |   |       |   |
         ramp sweet  sweet ramp
         start start   end  end

    Args:
        value: Input value(s) to transform
        sweet_range: (start, end) of the sweet spot where score = 1.0
        ramp_range: (start, end) of the full ramp range where score transitions

    Returns:
        Score in [0, 1]

    Example:
        >>> trapezoidal(10.0, sweet_range=(5, 15), ramp_range=(3, 25))
        1.0
        >>> trapezoidal(4.0, sweet_range=(5, 15), ramp_range=(3, 25))
        0.5
    """
    value = np.asarray(value)
    sweet_start, sweet_end = sweet_range
    ramp_start, ramp_end = ramp_range

    result = np.zeros_like(value, dtype=float)

    # In sweet spot: score = 1.0
    in_sweet = (value >= sweet_start) & (value <= sweet_end)
    result[in_sweet] = 1.0

    # Ramp up: from ramp_start to sweet_start
    in_ramp_up = (value >= ramp_start) & (value < sweet_start)
    if np.any(in_ramp_up):
        ramp_up_width = sweet_start - ramp_start
        if ramp_up_width > 0:
            result[in_ramp_up] = (value[in_ramp_up] - ramp_start) / ramp_up_width

    # Ramp down: from sweet_end to ramp_end
    in_ramp_down = (value > sweet_end) & (value <= ramp_end)
    if np.any(in_ramp_down):
        ramp_down_width = ramp_end - sweet_end
        if ramp_down_width > 0:
            result[in_ramp_down] = 1.0 - (value[in_ramp_down] - sweet_end) / ramp_down_width

    # Outside ramp range: score = 0.0 (already initialized)

    # Return scalar if input was scalar
    if result.ndim == 0:
        return float(result)
    return result


def dealbreaker(
    value: NumericType,
    threshold: float,
    falloff: float = 0,
    below_is_good: bool = True,
) -> NumericType:
    """
    Dealbreaker (step function) transformation.

    Returns 1.0 (no penalty) for safe values, 0.0 (full penalty) for dangerous values.
    Optional soft falloff for gradual transition.

    Shape (hard cutoff, falloff=0):
        _______
               |
               |________
              threshold

    Shape (soft falloff):
        _______
               \\
                \\______
              threshold  +falloff

    Args:
        value: Input value(s) to transform
        threshold: The cutoff point
        falloff: Width of soft transition after threshold (0 = hard cutoff)
        below_is_good: If True, values below threshold are good (1.0).
                       If False, values above threshold are good (1.0).

    Returns:
        Score in [0, 1] where 1.0 = safe, 0.0 = dealbreaker

    Example:
        >>> dealbreaker(10.0, threshold=25)  # Below threshold
        1.0
        >>> dealbreaker(30.0, threshold=25, falloff=0)  # Above, hard cutoff
        0.0
        >>> dealbreaker(30.0, threshold=25, falloff=10)  # Midway in falloff
        0.5
    """
    value = np.asarray(value)
    result = np.ones_like(value, dtype=float)

    if below_is_good:
        # Values above threshold get penalized
        if falloff == 0:
            # Hard cutoff
            result[value >= threshold] = 0.0
        else:
            # Soft falloff
            in_falloff = (value > threshold) & (value < threshold + falloff)
            past_falloff = value >= threshold + falloff

            result[in_falloff] = 1.0 - (value[in_falloff] - threshold) / falloff
            result[past_falloff] = 0.0
    else:
        # Values below threshold get penalized (inverted)
        if falloff == 0:
            # Hard cutoff
            result[value < threshold] = 0.0
        else:
            # Soft falloff below threshold
            in_falloff = (value < threshold) & (value > threshold - falloff)
            past_falloff = value <= threshold - falloff

            result[in_falloff] = (value[in_falloff] - (threshold - falloff)) / falloff
            result[past_falloff] = 0.0

    # Return scalar if input was scalar
    if result.ndim == 0:
        return float(result)
    return result


def linear(
    value: NumericType,
    value_range: tuple[float, float],
    invert: bool = False,
    power: float = 1.0,
) -> NumericType:
    """
    Linear normalization transformation.

    Maps value_range to [0, 1], clamping values outside the range.
    Optional power scaling for non-linear relationships.

    Args:
        value: Input value(s) to transform
        value_range: (min, max) range to normalize
        invert: If True, high values map to low scores
        power: Power to apply (0.5 = sqrt for diminishing returns, 2.0 = squared)

    Returns:
        Score in [0, 1]

    Example:
        >>> linear(50.0, value_range=(0, 100))
        0.5
        >>> linear(0.5, value_range=(0, 1), invert=True)  # High CV = bad
        0.5
        >>> linear(0.5, value_range=(0, 1), power=0.5)  # sqrt scaling
        0.707...
    """
    value = np.asarray(value)
    vmin, vmax = value_range

    # Normalize to [0, 1]
    normalized = (value - vmin) / (vmax - vmin)

    # Clamp to [0, 1]
    result = np.clip(normalized, 0.0, 1.0)

    # Apply power scaling
    if power != 1.0:
        result = np.power(result, power)

    # Invert if requested
    if invert:
        result = 1.0 - result

    # Return scalar if input was scalar
    if result.ndim == 0:
        return float(result)
    return result


def snow_consistency(
    interseason_cv: NumericType,
    intraseason_cv: NumericType,
    interseason_threshold: float = 1.5,
    intraseason_threshold: float = 1.0,
) -> NumericType:
    """
    Combined snow consistency metric from year-to-year and within-winter variability.

    Uses RMS (root mean square) to combine normalized CVs.
    Returns 1.0 for consistent snow, with gradual falloff for high variability.

    This is used as an ADDITIVE component (weighted contribution to score),
    not a multiplicative penalty. The score represents how reliable the snow is.

    Args:
        interseason_cv: Year-to-year coefficient of variation
        intraseason_cv: Within-winter coefficient of variation
        interseason_threshold: CV value that maps to full inconsistency (default: 1.5)
        intraseason_threshold: CV value that maps to full inconsistency (default: 1.0)

    Returns:
        Consistency score in [0, 1] where 1.0 = reliable, 0.0 = unreliable

    Example:
        >>> snow_consistency(interseason_cv=0.0, intraseason_cv=0.0)
        1.0
        >>> snow_consistency(interseason_cv=1.5, intraseason_cv=1.0)
        0.0
        >>> snow_consistency(interseason_cv=0.75, intraseason_cv=0.5)  # Both at 50%
        0.5
    """
    interseason_cv = np.asarray(interseason_cv)
    intraseason_cv = np.asarray(intraseason_cv)

    # Normalize each to [0, 1] where 1 = fully inconsistent
    inter_norm = np.clip(interseason_cv / interseason_threshold, 0.0, 1.0)
    intra_norm = np.clip(intraseason_cv / intraseason_threshold, 0.0, 1.0)

    # Combine using RMS (natural for variance-like measures)
    inconsistency = np.sqrt((inter_norm**2 + intra_norm**2) / 2)

    # Invert to get consistency (1 = good, 0 = bad)
    result = 1.0 - inconsistency

    # Return scalar if input was scalar
    if result.ndim == 0:
        return float(result)
    return result


def terrain_consistency(
    roughness: NumericType,
    slope_std: NumericType,
    roughness_threshold: float = 30.0,
    slope_std_threshold: float = 10.0,
    soft_start: float = 0.5,
) -> NumericType:
    """
    Combined terrain consistency metric from roughness and slope variability.

    Uses RMS to combine normalized roughness and slope_std, but only penalizes
    EXTREME inconsistency. Most terrain gets a score of 1.0 (no penalty).

    The penalty only kicks in when the combined inconsistency exceeds soft_start,
    providing a gradual falloff for very rough terrain.

    Args:
        roughness: Elevation standard deviation in meters
        slope_std: Slope standard deviation in degrees
        roughness_threshold: Roughness value that maps to full inconsistency
        slope_std_threshold: Slope std value that maps to full inconsistency
        soft_start: Normalized inconsistency level where penalty begins (default 0.5)

    Returns:
        Consistency score in [0, 1] where 1.0 = acceptable, 0.0 = extremely rough

    Example:
        >>> terrain_consistency(roughness=0.0, slope_std=0.0)
        1.0
        >>> terrain_consistency(roughness=15.0, slope_std=5.0)  # Both at 50%, at threshold
        1.0
        >>> terrain_consistency(roughness=30.0, slope_std=10.0)  # Extreme
        0.0
    """
    roughness = np.asarray(roughness)
    slope_std = np.asarray(slope_std)

    # Normalize each to [0, 1] where 1 = fully inconsistent
    roughness_norm = np.clip(roughness / roughness_threshold, 0.0, 1.0)
    slope_std_norm = np.clip(slope_std / slope_std_threshold, 0.0, 1.0)

    # Combine using RMS (natural for variance-like measures)
    inconsistency = np.sqrt((roughness_norm**2 + slope_std_norm**2) / 2)

    # Only penalize EXTREME inconsistency (above soft_start threshold)
    # Below soft_start: score = 1.0 (no penalty)
    # Above soft_start: gradual falloff to 0.0
    result = np.ones_like(inconsistency, dtype=float)
    extreme = inconsistency > soft_start
    if np.any(extreme):
        # Scale from soft_start->1.0 to 1.0->0.0
        falloff_range = 1.0 - soft_start
        result[extreme] = 1.0 - (inconsistency[extreme] - soft_start) / falloff_range

    result = np.clip(result, 0.0, 1.0)

    # Return scalar if input was scalar
    if result.ndim == 0:
        return float(result)
    return result
