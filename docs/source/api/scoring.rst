Scoring Module
==============

Terrain suitability scoring functions for outdoor activities.

This module provides scoring functions for evaluating terrain suitability
for activities like sledding and cross-country skiing. Scores are normalized
to [0, 1] range where 1.0 = optimal conditions.

Core Scoring Functions
----------------------

.. autofunction:: src.terrain.scoring.trapezoid_score

   Compute trapezoid (sweet spot) scoring pattern.

   **Trapezoid scoring pattern:**

   - Below ``min_value``: score = 0 (too low)
   - ``min_value`` → ``optimal_min``: linear ramp 0→1
   - ``optimal_min`` → ``optimal_max``: score = 1 (sweet spot!)
   - ``optimal_max`` → ``max_value``: linear ramp 1→0
   - Above ``max_value``: score = 0 (too high)

   Example::

       from src.terrain.scoring import trapezoid_score

       # Snow depth scoring: 4-8-16-24 inches
       # (too shallow, ramp up, optimal, ramp down, too deep)
       scores = trapezoid_score(
           snow_depth,
           min_value=4.0,
           optimal_min=8.0,
           optimal_max=16.0,
           max_value=24.0
       )

       # 12" snow → score = 1.0 (in sweet spot)
       # 6" snow → score = 0.5 (halfway up ramp)
       # 2" snow → score = 0.0 (too shallow)

Sledding Scoring
----------------

.. autofunction:: src.terrain.scoring.compute_sledding_score

   Compute sledding suitability score from terrain and snow data.

   **Factors evaluated:**

   - Slope (5-15° optimal, steeper = faster but harder)
   - Snow depth (8-16" optimal)
   - Snow coverage duration (more months = better)
   - Terrain roughness (smoother = safer)

   **Deal breakers (automatic score = 0):**

   - Slope > 40° (extreme terrain)
   - Roughness > 6m (cliff faces)
   - Coverage < 0.5 months (too little snow)

   Example::

       from src.terrain.scoring import compute_sledding_score

       sledding_scores = compute_sledding_score(
           slope=slope_data,           # degrees
           snow_depth=swe_data * 10,   # inches
           coverage_months=coverage,   # months
           roughness=roughness_data    # meters
       )

       # Returns array of scores [0-1]
       print(f"Mean score: {sledding_scores.mean():.2f}")

   See :doc:`../examples/sledding` for complete usage.

.. autofunction:: src.terrain.scoring.sledding_deal_breakers

   Identify deal-breaker conditions for sledding.

   Returns boolean mask where True = deal breaker (unsuitable).

.. autofunction:: src.terrain.scoring.sledding_synergy_bonus

   Apply bonus for favorable slope + snow depth combinations.

   **Synergy:** Steeper slopes need deeper snow for safe landings.

.. autofunction:: src.terrain.scoring.coverage_diminishing_returns

   Apply diminishing returns to snow coverage duration.

   **Logic:** 3 months vs 2 months is significant; 8 months vs 7 months less so.

Cross-Country Skiing Scoring
-----------------------------

.. autofunction:: src.terrain.scoring.compute_xc_skiing_score

   Compute cross-country skiing suitability score.

   **Factors evaluated:**

   - Slope (0-10° optimal, flatter = easier)
   - Snow depth (6-18" optimal)
   - Snow coverage duration (longer = better season)
   - Terrain roughness (smoother = better)

   **Deal breakers (automatic score = 0):**

   - Slope > 25° (too steep for XC)
   - Coverage < 1.0 months (too little snow)

   Example::

       from src.terrain.scoring import compute_xc_skiing_score

       xc_scores = compute_xc_skiing_score(
           slope=slope_data,
           snow_depth=swe_data * 10,
           coverage_months=coverage,
           roughness=roughness_data
       )

   See examples/detroit_xc_skiing.py for complete usage.

.. autofunction:: src.terrain.scoring.xc_skiing_deal_breakers

   Identify deal-breaker conditions for cross-country skiing.

Score Combination Patterns
---------------------------

**Pattern 1: Multiply individual factor scores**

::

    slope_score = trapezoid_score(slope, 5, 10, 15, 25)
    depth_score = trapezoid_score(depth, 4, 8, 16, 24)
    coverage_score = coverage_diminishing_returns(coverage)

    # Multiplicative combination (all factors matter)
    final_score = slope_score * depth_score * coverage_score

**Pattern 2: Apply deal breakers**

::

    # Compute base score
    base_score = slope_score * depth_score

    # Zero out deal breakers
    deal_breaker_mask = sledding_deal_breakers(slope, roughness, coverage)
    final_score = np.where(deal_breaker_mask, 0.0, base_score)

**Pattern 3: Add synergy bonuses**

::

    base_score = slope_score * depth_score * coverage_score

    # Bonus for favorable combinations
    synergy = sledding_synergy_bonus(slope, depth)
    final_score = base_score * (1.0 + 0.2 * synergy)  # Up to 20% bonus

Score Interpretation
--------------------

**Score ranges:**

- ``0.8-1.0``: Excellent conditions
- ``0.6-0.8``: Good conditions
- ``0.4-0.6``: Fair conditions
- ``0.2-0.4``: Marginal conditions
- ``0.0-0.2``: Poor conditions

**Visualization:**

Use perceptually uniform colormaps for score visualization::

    from src.terrain.color_mapping import elevation_colormap

    # Visualize scores
    colors = elevation_colormap(
        sledding_scores,
        cmap_name='boreal_mako',  # Custom sledding colormap
        min_elev=0.0,
        max_elev=1.0
    )

See Also
--------

- :doc:`../examples/sledding` - Sledding score computation and visualization
- :doc:`../examples/combined_render` - Dual score visualization (sledding + XC)
- :doc:`color_mapping` - Colormaps for score visualization
