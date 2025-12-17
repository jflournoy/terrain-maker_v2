# Sledding Deal Breaker Analysis

## Current Thresholds

```python
max_slope = 35.0°           # Too steep
max_variability = 5.0°      # Too rough/inconsistent
min_coverage = 0.5 months   # Not enough snow
```

## What These Mean

### 1. Slope > 35° (Cliff/Danger Threshold)

**What is 35°?**
- Expert/extreme ski terrain (double black diamond)
- Most ski resorts don't have sustained pitches this steep
- Walking up this requires hands for balance
- Sledding at this angle = dangerous/uncontrolled speed

**Real-world comparison:**
- Beginner ski slope: 5-10°
- Intermediate ski slope: 10-20°
- Advanced ski slope: 20-30°
- **Extreme terrain: 35°+**

**For family sledding on grass:**
- You'll likely NEVER see 35° in a public park
- The trapezoid already penalizes slopes above 20° (max acceptable)
- This dealbreaker may be redundant

**Recommendation:**
- ✅ **KEEP at 35°** (safety backstop for extreme terrain)
- Or **RAISE to 40°** (only catch truly dangerous cliffs)
- Or **REMOVE entirely** (already heavily penalized by trapezoid scoring)

---

### 2. Variability > 5° (Roughness/Consistency Threshold)

**What is 5° variability?**
- Standard deviation of slope within a local area
- Measures terrain roughness and consistency
- 5° = moderate bumpiness (slope changes 5° frequently)

**Real-world comparison:**
- Smooth grass: 0-2° variability
- Natural rolling hills: 2-4° variability
- Bumpy/mogul terrain: 4-6° variability
- **Very rough/rocky: 6°+ variability**

**For family sledding on grass:**
- Some bumpiness can be FUN (natural moguls!)
- Grass cushions the bumps
- Kids often prefer varied terrain (more exciting)
- 5° might be too strict for natural park hills

**Recommendation:**
- ❓ **RAISE to 7-8°** (allow more natural terrain variation)
- ❓ **REMOVE entirely** (let trapezoid scoring handle this)
- ✅ **KEEP at 5°** (maintain consistency requirement)

---

### 3. Coverage < 0.5 months (Snow Reliability Threshold)

**What is 0.5 months?**
- Half a month = ~15 days of snow coverage
- Opportunistic sledding (wait for a good snowfall)
- Not a reliable winter activity

**Real-world comparison:**
- 0.5 months: Occasional sledding (1-2 good weeks)
- 1 month: Decent season (multiple trips)
- 2+ months: Reliable winter activity
- 4+ months: Snow country

**For family sledding on grass:**
- 0.5 months seems reasonable for Midwest parks
- Allows for opportunistic sledding after snowstorms
- Grass makes snow depth less critical (cushioning)

**Recommendation:**
- ✅ **KEEP at 0.5 months** (reasonable minimum)
- Or **LOWER to 0.25 months** (even more opportunistic)
- Or **RAISE to 1.0 month** (only reliable locations)

---

## Impact on Scoring

**Current deal breakers are VERY lenient:**
- Only filter truly extreme/dangerous terrain
- Most park hills will pass all thresholds
- Scoring is mainly driven by trapezoid sweet spots and synergies

**If deal breakers filter nothing:**
- They're essentially safety backstops
- Real scoring happens via trapezoid functions
- This is probably the RIGHT approach for family sledding

---

## Recommended Changes

Based on the use case (family sledding on grassy park hills):

### Option 1: Keep Current (Conservative Safety Approach)
```python
max_slope = 35.0°           # Safety backstop
max_variability = 5.0°      # Consistency requirement
min_coverage = 0.5 months   # Opportunistic sledding
```
**Pro:** Safe, conservative
**Con:** Variability might be too strict

### Option 2: Relax Variability (Natural Terrain Approach)
```python
max_slope = 35.0°           # Safety backstop
max_variability = 7.0°      # Allow bumpier terrain (more fun!)
min_coverage = 0.5 months   # Opportunistic sledding
```
**Pro:** Allows natural rolling hills
**Con:** May allow some rough terrain

### Option 3: Remove Slope Dealbreaker (Trust Trapezoid)
```python
max_slope = None            # No hard cutoff (trapezoid handles it)
max_variability = 7.0°      # Allow bumpy terrain
min_coverage = 0.5 months   # Opportunistic sledding
```
**Pro:** Simpler, more gradual scoring
**Con:** No safety backstop for extreme terrain

### Option 4: Aggressive Filtering (High Standards)
```python
max_slope = 30.0°           # Stricter safety
max_variability = 4.0°      # Smoother terrain only
min_coverage = 1.0 months   # Reliable snow
```
**Pro:** Only excellent locations score well
**Con:** May filter too much natural terrain

---

## My Recommendation

**For family sledding on grassy park hills:**

```python
max_slope = 40.0°           # Raised (extreme cliffs only)
max_variability = 7.0°      # Raised (allow natural bumps)
min_coverage = 0.5 months   # Keep (opportunistic is OK)
```

**Rationale:**
1. **Slope 40°**: You'll never see this in parks, but it's a good safety backstop
2. **Variability 7°**: Natural hills have bumps - that's part of the fun!
3. **Coverage 0.5 months**: Reasonable for Midwest opportunistic sledding

This keeps the deal breakers as safety backstops without being overly restrictive, while the trapezoid scoring does the real work of identifying optimal terrain.
