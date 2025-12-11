# Sledding Score Formula

**Scorer**: sledding_suitability

## Additive Components (weighted sum = base score)

```
0.30 × Slope Mean  +  0.15 × Snow Depth  +  0.25 × Snow Coverage  +  0.20 × Snow Consistency  +  0.05 × Aspect Bonus  +  0.05 × Runout Bonus
```

## Multiplicative Penalties

```
× Slope P95  ×  Terrain Consistency
```

## Final Equation

```
FINAL = (0.30×Slopem + 0.15×Snowde + 0.25×Snowco + 0.20×Snowco + 0.05×Aspect + 0.05×Runout) × Slopep95 × Terrainc
```
