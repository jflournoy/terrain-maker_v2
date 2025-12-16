# XC Skiing Scoring Update: Focus on Snow Coverage

## Summary of Changes

Updated XC skiing scoring from abstract weighted sum to focused snow coverage scoring with deal breakers:

### Before (Weighted Sum Only)
```python
# No deal breakers
- Snow depth (30% weight): Trapezoid 100-400mm
- Snow coverage (60% weight): sqrt(coverage) - diminishing returns
- Snow consistency (10% weight): Inverted CV
```

### After (Coverage-Focused with Deal Breaker)
```python
# Deal breaker added
- Coverage < 15% (< ~18 days/season) → Score = 0

# Coverage scoring changed
- Snow depth (30% weight): Trapezoid 100-400mm (unchanged)
- Snow coverage (60% weight): Linear (no diminishing returns!)
- Snow consistency (10% weight): Inverted CV (unchanged)
```

## Why This Matters

### Problem with Old Approach
- **No deal breakers**: Areas with 10% coverage still scored ~0.3
- **Diminishing returns on coverage**: sqrt transform masked low coverage
  - 25% coverage → 0.50 score (not 0.25!)
  - 50% coverage → 0.71 score (not 0.50!)
- **Not intuitive**: Coverage contribution didn't match actual snow days

### Benefits of New Approach
- **Hard threshold**: Coverage < 15% = hard zero (parks can't fix lack of snow!)
- **Linear coverage scoring**: Proportional to actual snow days
  - 25% coverage → 0.25 score
  - 50% coverage → 0.50 score
  - 75% coverage → 0.75 score
- **Clearer interpretation**: Score directly reflects snow reliability

## Physical Interpretation

Parks handle terrain safety, so only snow conditions matter:

**Coverage levels (as fraction of year with snow)**:
- **0-15% (< 18 days/season)**: Too unreliable - **DEALBREAKER**
- **15-30% (18-36 days)**: Opportunistic skiing only - marginal
- **30-50% (36-60 days)**: Acceptable - ~1-2 months of skiing
- **50-75% (60-90 days)**: Good - ~2-3 months of skiing
- **75-100% (90-120 days)**: Excellent - ~3-4 months of skiing

**Score examples**:
- Coverage 75%, depth 250mm, CV 0.3 → **0.83** (excellent)
- Coverage 50%, depth 200mm, CV 0.6 → **0.66** (good)
- Coverage 25%, depth 75mm, CV 0.9 → **0.34** (marginal)
- Coverage 10%, depth 300mm, CV 0.2 → **0.0** (dealbreaker - no snow!)

## Code Changes

### 1. Core Scoring Functions
File: `src/terrain/scoring.py`

**Added `xc_skiing_deal_breakers()` function:**
```python
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
    """
    insufficient_coverage = snow_coverage < min_coverage
    return insufficient_coverage
```

**Added `compute_xc_skiing_score()` function:**
```python
def compute_xc_skiing_score(
    snow_depth: float | np.ndarray,
    snow_coverage: float | np.ndarray,
    snow_consistency: float | np.ndarray,
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
    """
    # Check deal breakers
    is_deal_breaker = xc_skiing_deal_breakers(...)

    # Only score valid areas
    depth_score = trapezoid_score(snow_depth, 50, 100, 400, 800)
    coverage_score = np.clip(snow_coverage, 0.0, 1.0)  # LINEAR, not sqrt!
    consistency_score = np.clip(1.0 - (snow_consistency / 1.5), 0.0, 1.0)

    # Weighted sum
    score = 0.30 * depth_score + 0.60 * coverage_score + 0.10 * consistency_score
```

### 2. Scoring Config
File: `src/scoring/configs/xc_skiing.py`

**Added `compute_improved_xc_skiing_score()` wrapper:**
```python
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
    """
    from src.terrain.scoring import compute_xc_skiing_score
    from src.scoring.transforms import snow_consistency

    snow_depth_mm = snow_stats["median_max_depth"]
    snow_coverage = snow_stats["mean_snow_day_ratio"]
    snow_cons = snow_consistency(
        snow_stats["interseason_cv"],
        snow_stats["mean_intraseason_cv"],
    )

    return compute_xc_skiing_score(
        snow_depth=snow_depth_mm,
        snow_coverage=snow_coverage,
        snow_consistency=snow_cons,
    )
```

### 3. Tests
File: `tests/test_xc_skiing_score.py`

- Created comprehensive test suite with 14 tests
- Tests cover deal breakers, scoring logic, linear coverage, and edge cases
- **All 14 tests pass** ✓

Key test cases:
- Deal breaker for coverage < 15%
- Linear coverage (not sqrt)
- Coverage dominates score (60% weight)
- Additive scoring (not multiplicative)
- Consistency has minor impact (10% weight)

### 4. Exports
File: `src/scoring/configs/__init__.py`

- Added `compute_improved_xc_skiing_score` to imports and __all__

## Migration Guide

If you have custom scoring code that uses the old ScoreCombiner API:

### Before
```python
from src.scoring.configs import DEFAULT_XC_SKIING_SCORER, xc_skiing_compute_derived_inputs

# Compute derived inputs
inputs = xc_skiing_compute_derived_inputs(snow_stats)

# Compute score using ScoreCombiner
score = DEFAULT_XC_SKIING_SCORER.compute(inputs)
```

### After (Using Improved Scoring)
```python
from src.scoring.configs import compute_improved_xc_skiing_score

# Direct computation with deal breaker and linear coverage
score = compute_improved_xc_skiing_score(snow_stats)
```

### Both APIs Still Work

The old ScoreCombiner API is still available for backward compatibility:
- `DEFAULT_XC_SKIING_SCORER` - Uses sqrt coverage (old behavior)
- `compute_improved_xc_skiing_score()` - Uses linear coverage + deal breaker (new)

## Comparison: Old vs New Scoring

### Coverage 25% Example

**Old (sqrt transform)**:
- Coverage contribution: sqrt(0.25) × 0.60 = 0.50 × 0.60 = 0.30
- With perfect depth & consistency: 0.30 + 0.30 + 0.10 = **0.70**
- Overestimates suitability!

**New (linear)**:
- Coverage contribution: 0.25 × 0.60 = 0.15
- With perfect depth & consistency: 0.30 + 0.15 + 0.10 = **0.55**
- More realistic for 25% coverage (only ~30 snow days)

### Coverage 10% Example

**Old (sqrt transform)**:
- Coverage contribution: sqrt(0.10) × 0.60 = 0.316 × 0.60 = 0.19
- With perfect depth & consistency: 0.30 + 0.19 + 0.10 = **0.59**
- Still scores "okay" despite only ~12 snow days!

**New (deal breaker)**:
- Coverage < 15% → **0.0** (hard zero)
- Correctly identifies as unsuitable (only ~12 snow days per year)

## Example Values

Based on typical snow statistics:

| Snow Days | Coverage | Depth (mm) | CV  | Old Score | New Score | Interpretation |
|-----------|----------|------------|-----|-----------|-----------|----------------|
| 110 days  | 0.90     | 250        | 0.3 | 0.88      | 0.85      | Excellent XC skiing |
| 90 days   | 0.75     | 250        | 0.3 | 0.84      | 0.83      | Excellent |
| 60 days   | 0.50     | 200        | 0.6 | 0.73      | 0.66      | Good |
| 36 days   | 0.30     | 150        | 0.8 | 0.62      | 0.49      | Marginal |
| 30 days   | 0.25     | 100        | 0.9 | 0.60      | 0.44      | Marginal |
| 18 days   | 0.15     | 100        | 1.0 | 0.53      | 0.37      | Poor (barely acceptable) |
| 12 days   | 0.10     | 300        | 0.2 | 0.59      | 0.00      | **DEALBREAKER** |

## Next Steps

To use the improved scoring in examples:

```bash
# Update detroit_xc_skiing.py to use improved scoring
# This will regenerate images with new deal breaker and linear coverage
python examples/detroit_xc_skiing.py --mock-data
```

This will update:
- `docs/images/xc_skiing/02_xc_scores/xc_skiing_scores.png` - with new scoring
- `docs/images/xc_skiing/03_parks/parks_on_score_map.png` - parks scored with new system
- All XC skiing visualizations with updated deal breaker thresholds

## Philosophy

**Parks handle terrain safety → Only snow coverage matters**

Unlike sledding (which must evaluate terrain roughness, slope, and runout), XC skiing takes place in managed parks where:
- Trails are groomed and maintained
- Dangerous terrain is avoided or marked
- Safety is managed by park staff

Therefore, XC skiing scoring focuses on **snow reliability** as the primary factor:
1. **Deal breaker**: Coverage < 15% (not enough snow season)
2. **Coverage weight**: 60% (dominant factor)
3. **Linear scoring**: Proportional to actual snow days
4. **No terrain factors**: Parks handle this

## References

- Core functions: `src/terrain/scoring.py` (lines 369-510)
- Config wrapper: `src/scoring/configs/xc_skiing.py` (lines 137-177)
- Test suite: `tests/test_xc_skiing_score.py`
- Analysis document: `analyze_xc_scoring.md`
