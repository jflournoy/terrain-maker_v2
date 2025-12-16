#!/usr/bin/env python3
"""
Tests for improved XC skiing score calculation.

Tests the trapezoid functions, deal breakers, and overall XC skiing score computation.
XC skiing focuses on snow conditions only (parks handle terrain safety).
"""

import numpy as np
import pytest

from src.terrain.scoring import (
    trapezoid_score,
    xc_skiing_deal_breakers,
    compute_xc_skiing_score,
)


class TestXCSkiingDealBreakers:
    """Test deal breaker conditions for XC skiing."""

    def test_insufficient_coverage(self):
        """Coverage < 15% should be deal breaker."""
        is_deal_breaker = xc_skiing_deal_breakers(
            snow_coverage=0.10,
        )
        assert is_deal_breaker is True

    def test_acceptable_coverage(self):
        """Coverage >= 15% should not be deal breaker."""
        is_deal_breaker = xc_skiing_deal_breakers(
            snow_coverage=0.20,
        )
        assert is_deal_breaker is False

    def test_excellent_coverage(self):
        """High coverage should not be deal breaker."""
        is_deal_breaker = xc_skiing_deal_breakers(
            snow_coverage=0.75,
        )
        assert is_deal_breaker is False

    def test_array_input(self):
        """Should work with numpy arrays."""
        coverage = np.array([0.10, 0.20, 0.50, 0.75])
        is_deal_breaker = xc_skiing_deal_breakers(
            snow_coverage=coverage,
        )
        expected = np.array([True, False, False, False])
        np.testing.assert_array_equal(is_deal_breaker, expected)


class TestComputeXCSkiingScore:
    """Test overall XC skiing score computation."""

    def test_deal_breaker_gives_zero(self):
        """Deal breakers should result in zero score."""
        score = compute_xc_skiing_score(
            snow_depth=250.0,  # Good depth
            snow_coverage=0.10,  # Too low - deal breaker
            snow_consistency=0.3,  # Good consistency
        )
        assert score == 0.0

    def test_excellent_conditions(self):
        """Excellent conditions should give high score."""
        score = compute_xc_skiing_score(
            snow_depth=250.0,  # Optimal (100-400mm)
            snow_coverage=0.75,  # 75% of days
            snow_consistency=0.3,  # Low CV (reliable)
        )
        # depth: 1.0 (in optimal range)
        # coverage: 0.75 (linear, no sqrt)
        # consistency: 1 - 0.3/1.5 = 0.8
        # score = 0.30*1.0 + 0.60*0.75 + 0.10*0.8 = 0.30 + 0.45 + 0.08 = 0.83
        assert score == pytest.approx(0.83, abs=0.01)

    def test_good_enough_conditions(self):
        """Good enough conditions should give moderate score."""
        score = compute_xc_skiing_score(
            snow_depth=200.0,  # In optimal range
            snow_coverage=0.50,  # 50% of days
            snow_consistency=0.6,  # Moderate CV
        )
        # depth: 1.0 (in optimal range)
        # coverage: 0.50 (linear)
        # consistency: 1 - 0.6/1.5 = 0.6
        # score = 0.30*1.0 + 0.60*0.50 + 0.10*0.6 = 0.30 + 0.30 + 0.06 = 0.66
        assert score == pytest.approx(0.66, abs=0.01)

    def test_marginal_conditions(self):
        """Marginal conditions should give lower score."""
        score = compute_xc_skiing_score(
            snow_depth=75.0,  # Below optimal, ramping down
            snow_coverage=0.25,  # 25% of days
            snow_consistency=0.9,  # Higher CV
        )
        # depth: trapezoid at 75mm (between 50 and 100) = (75-50)/(100-50) = 0.5
        # coverage: 0.25 (linear)
        # consistency: 1 - 0.9/1.5 = 0.4
        # score = 0.30*0.5 + 0.60*0.25 + 0.10*0.4 = 0.15 + 0.15 + 0.04 = 0.34
        assert score == pytest.approx(0.34, abs=0.01)

    def test_poor_snow_depth(self):
        """Poor snow depth should reduce score significantly."""
        score = compute_xc_skiing_score(
            snow_depth=30.0,  # Below minimum (50mm)
            snow_coverage=0.75,  # Good coverage
            snow_consistency=0.3,  # Good consistency
        )
        # depth: 0.0 (below 50mm minimum)
        # coverage: 0.75
        # consistency: 0.8
        # score = 0.30*0.0 + 0.60*0.75 + 0.10*0.8 = 0.0 + 0.45 + 0.08 = 0.53
        assert score == pytest.approx(0.53, abs=0.01)

    def test_coverage_dominates(self):
        """Coverage should dominate score (60% weight)."""
        # Test with good depth but poor coverage
        score_poor_coverage = compute_xc_skiing_score(
            snow_depth=250.0,  # Perfect
            snow_coverage=0.20,  # Just above deal breaker
            snow_consistency=0.3,  # Good
        )

        # Test with poor depth but good coverage
        score_good_coverage = compute_xc_skiing_score(
            snow_depth=60.0,  # Ramping up from minimum
            snow_coverage=0.70,  # Good
            snow_consistency=0.3,  # Good
        )

        # Good coverage should score better despite poor depth
        assert score_good_coverage > score_poor_coverage

    def test_linear_coverage_no_diminishing_returns(self):
        """Coverage should be linear (no sqrt, no diminishing returns)."""
        # 50% coverage should score 0.5 contribution from coverage
        score = compute_xc_skiing_score(
            snow_depth=250.0,  # Perfect
            snow_coverage=0.50,  # 50%
            snow_consistency=0.0,  # Perfect (CV=0)
        )
        # depth: 1.0, coverage: 0.50, consistency: 1.0
        # score = 0.30*1.0 + 0.60*0.50 + 0.10*1.0 = 0.30 + 0.30 + 0.10 = 0.70
        assert score == pytest.approx(0.70, abs=0.01)

    def test_additive_nature(self):
        """Score should be additive - weighted sum, not multiplicative."""
        # Poor coverage doesn't zero out score (unlike sledding)
        score = compute_xc_skiing_score(
            snow_depth=250.0,  # Perfect
            snow_coverage=0.20,  # Just above deal breaker
            snow_consistency=0.3,  # Good
        )
        # With additive scoring, depth and consistency still contribute
        # depth: 1.0, coverage: 0.20, consistency: 0.8
        # score = 0.30*1.0 + 0.60*0.20 + 0.10*0.8 = 0.30 + 0.12 + 0.08 = 0.50
        assert score == pytest.approx(0.50, abs=0.01)

    def test_array_input(self):
        """Should work with numpy arrays."""
        snow_depths = np.array([250.0, 30.0, 200.0])
        coverages = np.array([0.75, 0.75, 0.10])
        consistency = np.array([0.3, 0.3, 0.3])

        scores = compute_xc_skiing_score(
            snow_depth=snow_depths,
            snow_coverage=coverages,
            snow_consistency=consistency,
        )

        assert len(scores) == 3
        assert scores[0] > 0.8  # Excellent conditions
        assert scores[1] > 0.5  # Good coverage, poor depth
        assert scores[2] == 0.0  # Deal breaker (coverage < 15%)

    def test_consistency_minor_impact(self):
        """Consistency should have minor impact (10% weight)."""
        # Perfect consistency
        score_perfect_cons = compute_xc_skiing_score(
            snow_depth=250.0,
            snow_coverage=0.50,
            snow_consistency=0.0,  # CV = 0 (perfect)
        )

        # Poor consistency
        score_poor_cons = compute_xc_skiing_score(
            snow_depth=250.0,
            snow_coverage=0.50,
            snow_consistency=1.5,  # CV = 1.5 (max)
        )

        # Difference should be small (10% weight)
        # Perfect: 0.30 + 0.30 + 0.10 = 0.70
        # Poor: 0.30 + 0.30 + 0.00 = 0.60
        # Difference = 0.10
        assert score_perfect_cons == pytest.approx(0.70, abs=0.01)
        assert score_poor_cons == pytest.approx(0.60, abs=0.01)
        assert score_perfect_cons - score_poor_cons == pytest.approx(0.10, abs=0.01)
