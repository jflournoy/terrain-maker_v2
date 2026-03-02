Score Combiner Module
=====================

Multi-factor suitability scoring framework.

This module provides a composable scoring system for evaluating terrain
suitability for outdoor activities. It separates scoring into three concerns:

1. **Transforms** - Convert raw values to [0, 1] scores
2. **Components** - Define individual scoring factors with roles
3. **Combiner** - Combine components into a final score

Formula: ``final = (weighted sum of additive) * (product of multiplicative)``

Scoring Transforms
------------------

All transforms convert raw values into scores in [0, 1].

.. automodule:: src.scoring.transforms
   :members:
   :undoc-members:
   :show-inheritance:

Score Components and Combiner
-----------------------------

.. automodule:: src.scoring.combiner
   :members:
   :undoc-members:
   :show-inheritance:

Pre-built Scoring Configurations
---------------------------------

Sledding Scorer
~~~~~~~~~~~~~~~

.. automodule:: src.scoring.configs.sledding
   :members:
   :undoc-members:
   :show-inheritance:

XC Skiing Scorer
~~~~~~~~~~~~~~~~

.. automodule:: src.scoring.configs.xc_skiing
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Using the combiner framework::

    from src.scoring.combiner import ScoreCombiner, ScoreComponent

    scorer = ScoreCombiner(
        components=[
            ScoreComponent(
                name="slope",
                transform="trapezoidal",
                params={"sweet_range": (5, 15), "ramp_range": (3, 25)},
                role="additive",
                weight=1.0,
            ),
            ScoreComponent(
                name="cliff",
                transform="dealbreaker",
                params={"threshold": 25, "falloff": 10},
                role="multiplicative",
            ),
        ]
    )

    scores = scorer.combine(slope=slope_data, cliff=p95_data)

Using pre-built scorers::

    from src.scoring.configs.sledding import DEFAULT_SLEDDING_SCORER

    scores = DEFAULT_SLEDDING_SCORER.combine(
        slope_mean=slope_stats.slope_mean,
        slope_p95=slope_stats.slope_p95,
        roughness=slope_stats.roughness,
        slope_std=slope_stats.slope_std,
        snow_depth=snow_stats["median_max_depth"],
        snow_coverage=snow_stats["mean_snow_day_ratio"],
        snow_consistency=consistency,
    )

See Also
--------

- :doc:`scoring` - Legacy scoring functions (trapezoid_score, compute_sledding_score)
- :doc:`../examples/sledding` - Sledding score example
