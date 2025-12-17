# Improved Sledding Score Formula

**Method**: Trapezoid Functions + Deal Breakers + Synergy Bonuses

## Sweet Spot Scoring (Trapezoid Functions)

Each factor is scored using trapezoid functions with optimal ranges:

```
Snow Depth:   1" ━━━ [4-12"] ━━━ 20"  (inches, 25-100-300-500mm)
Slope:        1° ━━━ [6-12°] ━━━ 20°  (degrees)
Coverage:     Diminishing returns: 1 - exp(-months/2)
```

## Deal Breakers (Hard Zeros)

```
Slope > 40°  OR  Roughness > 6m  OR  Coverage < 0.5 months  →  Score = 0
```

## Base Score (Multiplicative)

```
BASE = Snow_Trapezoid × Slope_Trapezoid × Coverage_Score
```

## Synergy Bonuses (Hierarchical)

```
1. Perfect Combo (slope + snow + 3+ mo):      +30%
2. Consistent (4+ mo + good slope + low var): +15%
3. Moderate Slope + Flat Runout:              +20%
4. Low Variability + Perfect Slope:           +10%
```

## Final Score

```
FINAL = BASE × SYNERGY_BONUS  (range: 0 to ~1.5)
```
