"""
Score combination system for multi-factor suitability scoring.

Provides:
- ScoreComponent: Defines a single scoring factor with transform and role
- ScoreCombiner: Combines multiple components into a final score

Components have two roles:
- additive: Weighted sum (e.g., slope_score, depth_score)
- multiplicative: Penalties that reduce the score (e.g., cliff_penalty)

Formula: final_score = (sum of weighted additive) * (product of multiplicative)
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

import numpy as np

from src.scoring.transforms import trapezoidal, dealbreaker, linear, terrain_consistency

# Type alias
NumericType = Union[float, np.ndarray]

# Map transform names to functions
TRANSFORM_FUNCTIONS = {
    "trapezoidal": trapezoidal,
    "dealbreaker": dealbreaker,
    "linear": linear,
    "terrain_consistency": terrain_consistency,
}


@dataclass
class ScoreComponent:
    """
    A single scoring component with transform and role.

    Attributes:
        name: Identifier for this component (used as key in input dict)
        transform: Name of transform function ("trapezoidal", "dealbreaker", "linear")
        transform_params: Parameters to pass to the transform function
        role: "additive" (weighted sum) or "multiplicative" (penalty)
        weight: Weight for additive components (must be provided if role="additive")
    """

    name: str
    transform: str
    transform_params: dict[str, Any]
    role: Literal["additive", "multiplicative"]
    weight: Optional[float] = None

    def __post_init__(self):
        """Validate the component configuration."""
        if self.role == "additive" and self.weight is None:
            raise ValueError(
                f"Component '{self.name}' has role='additive' but no weight. "
                "Additive components must have a weight."
            )

        if self.transform not in TRANSFORM_FUNCTIONS:
            raise ValueError(
                f"Unknown transform '{self.transform}'. "
                f"Available: {list(TRANSFORM_FUNCTIONS.keys())}"
            )

    def apply(self, value: NumericType) -> NumericType:
        """
        Apply this component's transform to a value.

        Args:
            value: Raw input value(s)

        Returns:
            Transformed score in [0, 1]
        """
        transform_fn = TRANSFORM_FUNCTIONS[self.transform]
        return transform_fn(value, **self.transform_params)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "transform": self.transform,
            "transform_params": self.transform_params,
            "role": self.role,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScoreComponent":
        """Deserialize from dictionary."""
        return cls(
            name=data["name"],
            transform=data["transform"],
            transform_params=data["transform_params"],
            role=data["role"],
            weight=data.get("weight"),
        )


@dataclass
class ScoreCombiner:
    """
    Combines multiple ScoreComponents into a final score.

    Formula: final_score = (weighted sum of additive) * (product of multiplicative)

    Attributes:
        name: Identifier for this combiner
        components: List of ScoreComponent instances
    """

    name: str
    components: list[ScoreComponent] = field(default_factory=list)

    def __post_init__(self):
        """Validate the combiner configuration."""
        # Check that additive weights sum to 1.0
        additive_weights = [
            c.weight for c in self.components if c.role == "additive"
        ]

        if additive_weights:
            total = sum(additive_weights)
            if not np.isclose(total, 1.0, rtol=1e-5):
                raise ValueError(
                    f"Additive component weights must sum to 1.0, got {total:.4f}. "
                    f"Weights: {additive_weights}"
                )

    def compute(self, inputs: dict[str, NumericType]) -> NumericType:
        """
        Compute the combined score from input values.

        Args:
            inputs: Dictionary mapping component names to their raw values

        Returns:
            Final combined score in [0, 1]
        """
        # Compute all component scores
        component_scores = self.get_component_scores(inputs)

        # Separate additive and multiplicative
        additive_components = [c for c in self.components if c.role == "additive"]
        multiplicative_components = [c for c in self.components if c.role == "multiplicative"]

        # Compute weighted sum of additive components
        if additive_components:
            # Get first score to determine shape/type
            first_score = component_scores[additive_components[0].name]
            if isinstance(first_score, np.ndarray):
                additive_sum = np.zeros_like(first_score, dtype=float)
            else:
                additive_sum = 0.0

            for component in additive_components:
                score = component_scores[component.name]
                additive_sum = additive_sum + component.weight * score
        else:
            additive_sum = 1.0  # No additive components = start at 1.0

        # Multiply by all multiplicative components
        result = additive_sum
        for component in multiplicative_components:
            score = component_scores[component.name]
            result = result * score

        return result

    def get_component_scores(self, inputs: dict[str, NumericType]) -> dict[str, NumericType]:
        """
        Get individual transformed scores for each component.

        Useful for debugging and visualization.

        Args:
            inputs: Dictionary mapping component names to their raw values

        Returns:
            Dictionary mapping component names to their transformed scores
        """
        scores = {}
        for component in self.components:
            if component.name not in inputs:
                raise KeyError(
                    f"Missing input for component '{component.name}'. "
                    f"Available inputs: {list(inputs.keys())}"
                )
            scores[component.name] = component.apply(inputs[component.name])
        return scores

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "components": [c.to_dict() for c in self.components],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScoreCombiner":
        """Deserialize from dictionary."""
        components = [ScoreComponent.from_dict(c) for c in data["components"]]
        return cls(name=data["name"], components=components)
