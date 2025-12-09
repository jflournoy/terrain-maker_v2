"""
TDD RED Phase: Tests for scoring combination function.

The combiner takes multiple score components and combines them:
1. Additive components - weighted sum (e.g., slope_score, depth_score)
2. Multiplicative components - penalties that reduce the score (e.g., cliff_penalty)

Formula: final_score = (sum of weighted additive) * (product of multiplicative)
"""

import numpy as np
import pytest


# =============================================================================
# SCORE COMPONENT DEFINITION TESTS
# =============================================================================


class TestScoreComponent:
    """Test ScoreComponent definition."""

    def test_create_additive_component(self):
        """Can create an additive score component."""
        from src.scoring.combiner import ScoreComponent

        component = ScoreComponent(
            name="slope_score",
            transform="trapezoidal",
            transform_params={"sweet_range": (5, 15), "ramp_range": (3, 25)},
            role="additive",
            weight=0.25,
        )

        assert component.name == "slope_score"
        assert component.transform == "trapezoidal"
        assert component.role == "additive"
        assert component.weight == 0.25

    def test_create_multiplicative_component(self):
        """Can create a multiplicative (penalty) component."""
        from src.scoring.combiner import ScoreComponent

        component = ScoreComponent(
            name="cliff_penalty",
            transform="dealbreaker",
            transform_params={"threshold": 25, "falloff": 10},
            role="multiplicative",
        )

        assert component.name == "cliff_penalty"
        assert component.role == "multiplicative"
        assert component.weight is None  # Multiplicative components don't have weights

    def test_additive_requires_weight(self):
        """Additive components must have a weight."""
        from src.scoring.combiner import ScoreComponent

        with pytest.raises(ValueError, match="weight"):
            ScoreComponent(
                name="test",
                transform="linear",
                transform_params={"value_range": (0, 1)},
                role="additive",
                weight=None,  # Should raise
            )

    def test_apply_transform(self):
        """Component can apply its transform to a value."""
        from src.scoring.combiner import ScoreComponent

        component = ScoreComponent(
            name="slope_score",
            transform="trapezoidal",
            transform_params={"sweet_range": (5, 15), "ramp_range": (3, 25)},
            role="additive",
            weight=0.25,
        )

        # In sweet spot
        result = component.apply(10.0)
        assert result == 1.0

        # At extreme
        result = component.apply(30.0)
        assert result == 0.0


# =============================================================================
# SCORE COMBINER TESTS
# =============================================================================


class TestScoreCombiner:
    """Test ScoreCombiner for combining multiple components."""

    def test_create_combiner_with_components(self):
        """Can create a combiner with multiple components."""
        from src.scoring.combiner import ScoreCombiner, ScoreComponent

        combiner = ScoreCombiner(
            name="sledding_score",
            components=[
                ScoreComponent(
                    name="slope",
                    transform="trapezoidal",
                    transform_params={"sweet_range": (5, 15), "ramp_range": (3, 25)},
                    role="additive",
                    weight=0.5,
                ),
                ScoreComponent(
                    name="depth",
                    transform="linear",
                    transform_params={"value_range": (0, 500)},
                    role="additive",
                    weight=0.5,
                ),
            ],
        )

        assert combiner.name == "sledding_score"
        assert len(combiner.components) == 2

    def test_additive_weights_must_sum_to_one(self):
        """Additive component weights should sum to 1.0."""
        from src.scoring.combiner import ScoreCombiner, ScoreComponent

        with pytest.raises(ValueError, match="sum to 1"):
            ScoreCombiner(
                name="test",
                components=[
                    ScoreComponent(
                        name="a",
                        transform="linear",
                        transform_params={"value_range": (0, 1)},
                        role="additive",
                        weight=0.3,
                    ),
                    ScoreComponent(
                        name="b",
                        transform="linear",
                        transform_params={"value_range": (0, 1)},
                        role="additive",
                        weight=0.3,  # Total = 0.6, not 1.0
                    ),
                ],
            )

    def test_compute_additive_only(self):
        """Compute score with only additive components."""
        from src.scoring.combiner import ScoreCombiner, ScoreComponent

        combiner = ScoreCombiner(
            name="test",
            components=[
                ScoreComponent(
                    name="a",
                    transform="linear",
                    transform_params={"value_range": (0, 100)},
                    role="additive",
                    weight=0.6,
                ),
                ScoreComponent(
                    name="b",
                    transform="linear",
                    transform_params={"value_range": (0, 100)},
                    role="additive",
                    weight=0.4,
                ),
            ],
        )

        # a=50 -> 0.5, b=100 -> 1.0
        # score = 0.6 * 0.5 + 0.4 * 1.0 = 0.3 + 0.4 = 0.7
        result = combiner.compute({"a": 50.0, "b": 100.0})
        assert result == pytest.approx(0.7, rel=1e-5)

    def test_compute_with_multiplicative(self):
        """Compute score with both additive and multiplicative components."""
        from src.scoring.combiner import ScoreCombiner, ScoreComponent

        combiner = ScoreCombiner(
            name="test",
            components=[
                ScoreComponent(
                    name="score",
                    transform="linear",
                    transform_params={"value_range": (0, 100)},
                    role="additive",
                    weight=1.0,
                ),
                ScoreComponent(
                    name="penalty",
                    transform="dealbreaker",
                    transform_params={"threshold": 50, "falloff": 0},
                    role="multiplicative",
                ),
            ],
        )

        # score=100 -> 1.0, penalty value=40 (below threshold) -> 1.0
        # final = 1.0 * 1.0 = 1.0
        result = combiner.compute({"score": 100.0, "penalty": 40.0})
        assert result == 1.0

        # score=100 -> 1.0, penalty value=60 (above threshold) -> 0.0
        # final = 1.0 * 0.0 = 0.0
        result = combiner.compute({"score": 100.0, "penalty": 60.0})
        assert result == 0.0

    def test_compute_with_numpy_arrays(self):
        """Should work with numpy arrays element-wise."""
        from src.scoring.combiner import ScoreCombiner, ScoreComponent

        combiner = ScoreCombiner(
            name="test",
            components=[
                ScoreComponent(
                    name="score",
                    transform="linear",
                    transform_params={"value_range": (0, 100)},
                    role="additive",
                    weight=1.0,
                ),
                ScoreComponent(
                    name="penalty",
                    transform="dealbreaker",
                    transform_params={"threshold": 50, "falloff": 0},
                    role="multiplicative",
                ),
            ],
        )

        result = combiner.compute({
            "score": np.array([50.0, 100.0, 100.0]),
            "penalty": np.array([0.0, 40.0, 60.0]),  # safe, safe, dangerous
        })

        assert isinstance(result, np.ndarray)
        assert result[0] == pytest.approx(0.5, rel=1e-5)  # 0.5 * 1.0
        assert result[1] == pytest.approx(1.0, rel=1e-5)  # 1.0 * 1.0
        assert result[2] == pytest.approx(0.0, rel=1e-5)  # 1.0 * 0.0

    def test_multiple_multiplicative_components(self):
        """Multiple multiplicative components should all multiply."""
        from src.scoring.combiner import ScoreCombiner, ScoreComponent

        combiner = ScoreCombiner(
            name="test",
            components=[
                ScoreComponent(
                    name="score",
                    transform="linear",
                    transform_params={"value_range": (0, 100)},
                    role="additive",
                    weight=1.0,
                ),
                ScoreComponent(
                    name="penalty1",
                    transform="linear",
                    transform_params={"value_range": (0, 1)},  # 0-1 direct
                    role="multiplicative",
                ),
                ScoreComponent(
                    name="penalty2",
                    transform="linear",
                    transform_params={"value_range": (0, 1)},
                    role="multiplicative",
                ),
            ],
        )

        # score=100 -> 1.0, penalty1=0.5 -> 0.5, penalty2=0.5 -> 0.5
        # final = 1.0 * 0.5 * 0.5 = 0.25
        result = combiner.compute({"score": 100.0, "penalty1": 0.5, "penalty2": 0.5})
        assert result == pytest.approx(0.25, rel=1e-5)

    def test_get_component_scores(self):
        """Can retrieve individual component scores for debugging."""
        from src.scoring.combiner import ScoreCombiner, ScoreComponent

        combiner = ScoreCombiner(
            name="test",
            components=[
                ScoreComponent(
                    name="slope",
                    transform="trapezoidal",
                    transform_params={"sweet_range": (5, 15), "ramp_range": (3, 25)},
                    role="additive",
                    weight=0.5,
                ),
                ScoreComponent(
                    name="depth",
                    transform="linear",
                    transform_params={"value_range": (0, 500)},
                    role="additive",
                    weight=0.5,
                ),
                ScoreComponent(
                    name="cliff",
                    transform="dealbreaker",
                    transform_params={"threshold": 25, "falloff": 10},
                    role="multiplicative",
                ),
            ],
        )

        inputs = {"slope": 10.0, "depth": 250.0, "cliff": 20.0}
        component_scores = combiner.get_component_scores(inputs)

        assert "slope" in component_scores
        assert "depth" in component_scores
        assert "cliff" in component_scores
        assert component_scores["slope"] == 1.0  # In sweet spot
        assert component_scores["depth"] == 0.5  # 250/500
        assert component_scores["cliff"] == 1.0  # Below threshold


# =============================================================================
# SLEDDING SCORE CONFIG TESTS
# =============================================================================


class TestSleddingScoreConfig:
    """Test creating a full sledding score configuration."""

    def test_create_sledding_scorer(self):
        """Can create a complete sledding score combiner."""
        from src.scoring.combiner import ScoreCombiner, ScoreComponent

        sledding = ScoreCombiner(
            name="sledding_suitability",
            components=[
                # Additive components (weighted sum)
                ScoreComponent(
                    name="slope_mean",
                    transform="trapezoidal",
                    transform_params={"sweet_range": (5, 15), "ramp_range": (3, 25)},
                    role="additive",
                    weight=0.25,
                ),
                ScoreComponent(
                    name="snow_depth",
                    transform="trapezoidal",
                    transform_params={"sweet_range": (150, 500), "ramp_range": (50, 1000)},
                    role="additive",
                    weight=0.25,
                ),
                ScoreComponent(
                    name="snow_coverage",
                    transform="linear",
                    transform_params={"value_range": (0, 1), "power": 0.5},
                    role="additive",
                    weight=0.20,
                ),
                ScoreComponent(
                    name="snow_consistency",
                    transform="linear",
                    transform_params={"value_range": (0, 1), "invert": True},
                    role="additive",
                    weight=0.15,
                ),
                ScoreComponent(
                    name="aspect_bonus",
                    transform="linear",
                    transform_params={"value_range": (-0.05, 0.05)},
                    role="additive",
                    weight=0.10,
                ),
                ScoreComponent(
                    name="runout_bonus",
                    transform="linear",
                    transform_params={"value_range": (0, 1)},
                    role="additive",
                    weight=0.05,
                ),
                # Multiplicative components (penalties)
                ScoreComponent(
                    name="cliff_penalty",
                    transform="dealbreaker",
                    transform_params={"threshold": 25, "falloff": 10},
                    role="multiplicative",
                ),
                ScoreComponent(
                    name="terrain_consistency",
                    transform="linear",
                    transform_params={"value_range": (0, 1)},
                    role="multiplicative",
                ),
            ],
        )

        # Test with ideal values
        result = sledding.compute({
            "slope_mean": 10.0,         # In sweet spot -> 1.0
            "snow_depth": 300.0,        # In sweet spot -> 1.0
            "snow_coverage": 1.0,       # Max coverage -> 1.0
            "snow_consistency": 0.0,    # Zero CV (inverted) -> 1.0
            "aspect_bonus": 0.05,       # Max north bonus -> 1.0
            "runout_bonus": 1.0,        # Has runout -> 1.0
            "cliff_penalty": 10.0,      # Safe -> 1.0
            "terrain_consistency": 1.0, # Consistent -> 1.0
        })

        # All components at 1.0, so final score should be 1.0
        assert result == pytest.approx(1.0, rel=1e-3)

    def test_to_dict_and_from_dict(self):
        """Can serialize/deserialize a combiner config."""
        from src.scoring.combiner import ScoreCombiner, ScoreComponent

        original = ScoreCombiner(
            name="test",
            components=[
                ScoreComponent(
                    name="slope",
                    transform="trapezoidal",
                    transform_params={"sweet_range": (5, 15), "ramp_range": (3, 25)},
                    role="additive",
                    weight=1.0,
                ),
            ],
        )

        # Serialize to dict
        config_dict = original.to_dict()
        assert config_dict["name"] == "test"
        assert len(config_dict["components"]) == 1

        # Deserialize back
        restored = ScoreCombiner.from_dict(config_dict)
        assert restored.name == original.name
        assert len(restored.components) == len(original.components)

        # Should produce same results
        assert restored.compute({"slope": 10.0}) == original.compute({"slope": 10.0})
