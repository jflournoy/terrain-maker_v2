#!/usr/bin/env python3
"""
Tests for improved sledding score calculation.

Tests the trapezoid functions, deal breakers, synergy bonuses, and overall
sledding score computation.
"""

import numpy as np
import pytest

from src.terrain.scoring import (
    trapezoid_score,
    sledding_deal_breakers,
    coverage_diminishing_returns,
    sledding_synergy_bonus,
    compute_sledding_score,
)


class TestTrapezoidScore:
    """Test trapezoid function for sweet spot scoring."""

    def test_trapezoid_below_min(self):
        """Values below min_value should score 0."""
        score = trapezoid_score(
            value=2.0,
            min_value=4.0,
            optimal_min=8.0,
            optimal_max=16.0,
            max_value=24.0,
        )
        assert score == 0.0

    def test_trapezoid_above_max(self):
        """Values above max_value should score 0."""
        score = trapezoid_score(
            value=30.0,
            min_value=4.0,
            optimal_min=8.0,
            optimal_max=16.0,
            max_value=24.0,
        )
        assert score == 0.0

    def test_trapezoid_in_optimal_range(self):
        """Values in optimal range should score 1.0."""
        score = trapezoid_score(
            value=12.0,
            min_value=4.0,
            optimal_min=8.0,
            optimal_max=16.0,
            max_value=24.0,
        )
        assert score == 1.0

    def test_trapezoid_ramp_up(self):
        """Values between min and optimal_min should ramp up linearly."""
        # Midpoint between 4 and 8 should be 0.5
        score = trapezoid_score(
            value=6.0,
            min_value=4.0,
            optimal_min=8.0,
            optimal_max=16.0,
            max_value=24.0,
        )
        assert score == pytest.approx(0.5, abs=0.01)

    def test_trapezoid_ramp_down(self):
        """Values between optimal_max and max should ramp down linearly."""
        # Midpoint between 16 and 24 should be 0.5
        score = trapezoid_score(
            value=20.0,
            min_value=4.0,
            optimal_min=8.0,
            optimal_max=16.0,
            max_value=24.0,
        )
        assert score == pytest.approx(0.5, abs=0.01)

    def test_trapezoid_array_input(self):
        """Should work with numpy arrays."""
        values = np.array([2.0, 6.0, 12.0, 20.0, 30.0])
        scores = trapezoid_score(
            value=values,
            min_value=4.0,
            optimal_min=8.0,
            optimal_max=16.0,
            max_value=24.0,
        )
        expected = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
        np.testing.assert_allclose(scores, expected, atol=0.01)


class TestSleddingDealBreakers:
    """Test deal breaker conditions for sledding."""

    def test_slope_too_steep(self):
        """Slopes > 40° should be deal breaker."""
        is_deal_breaker = sledding_deal_breakers(
            slope=45.0,
            roughness=2.0,
            coverage_months=3.0,
        )
        assert is_deal_breaker is True

    def test_high_roughness(self):
        """High terrain roughness should be deal breaker."""
        is_deal_breaker = sledding_deal_breakers(
            slope=10.0,
            roughness=8.0,  # High roughness (>6m)
            coverage_months=3.0,
        )
        assert is_deal_breaker is True

    def test_insufficient_coverage(self):
        """Coverage < 0.5 months should be deal breaker."""
        is_deal_breaker = sledding_deal_breakers(
            slope=10.0,
            roughness=2.0,
            coverage_months=0.3,
        )
        assert is_deal_breaker is True

    def test_acceptable_conditions(self):
        """Good conditions should not be deal breaker."""
        is_deal_breaker = sledding_deal_breakers(
            slope=10.0,
            roughness=3.0,
            coverage_months=3.0,
        )
        assert is_deal_breaker is False

    def test_array_input(self):
        """Should work with numpy arrays."""
        slopes = np.array([10.0, 45.0, 10.0, 10.0])
        roughness_vals = np.array([3.0, 3.0, 8.0, 3.0])
        coverage = np.array([3.0, 3.0, 3.0, 0.3])

        is_deal_breaker = sledding_deal_breakers(
            slope=slopes,
            roughness=roughness_vals,
            coverage_months=coverage,
        )
        expected = np.array([False, True, True, True])
        np.testing.assert_array_equal(is_deal_breaker, expected)


class TestCoverageDiminishingReturns:
    """Test coverage scoring with diminishing returns."""

    def test_zero_coverage(self):
        """Zero coverage should score 0."""
        score = coverage_diminishing_returns(0.0)
        assert score == 0.0

    def test_one_month_coverage(self):
        """One month should give reasonable score."""
        score = coverage_diminishing_returns(1.0)
        # 1 - e^(-1/2) ≈ 0.393
        assert score == pytest.approx(0.393, abs=0.01)

    def test_two_months_coverage(self):
        """Two months should give better score with diminishing returns."""
        score = coverage_diminishing_returns(2.0)
        # 1 - e^(-2/2) = 1 - e^(-1) ≈ 0.632
        assert score == pytest.approx(0.632, abs=0.01)

    def test_six_months_coverage(self):
        """Six months should approach 1.0."""
        score = coverage_diminishing_returns(6.0)
        # 1 - e^(-6/2) = 1 - e^(-3) ≈ 0.950
        assert score == pytest.approx(0.950, abs=0.01)

    def test_array_input(self):
        """Should work with numpy arrays."""
        coverage = np.array([0.0, 1.0, 2.0, 6.0])
        scores = coverage_diminishing_returns(coverage)
        expected = np.array([0.0, 0.393, 0.632, 0.950])
        np.testing.assert_allclose(scores, expected, atol=0.01)


class TestSleddingSynergyBonus:
    """Test synergy bonus computation."""

    def test_perfect_combo_bonus(self):
        """Perfect slope + optimal snow + 3+ months should give 30% bonus."""
        bonus = sledding_synergy_bonus(
            slope=9.0,  # In 6-12° optimal range
            snow_depth=8.0,  # In 2-10" optimal range (new trapezoid)
            coverage_months=3.5,
            roughness=2.0,  # Moderate roughness
        )
        assert bonus == pytest.approx(1.30, abs=0.01)  # 30% bonus

    def test_consistent_coverage_bonus(self):
        """Consistent coverage + good slope + smooth terrain should give 15% bonus."""
        bonus = sledding_synergy_bonus(
            slope=10.0,  # In 6-12° optimal range
            snow_depth=20.0,  # Not optimal
            coverage_months=4.0,  # 4+ months
            roughness=2.5,  # Smooth terrain (<3m)
        )
        # Should get consistent coverage bonus (15%)
        assert bonus >= 1.10  # At least some bonus

    def test_moderate_slope_smooth_terrain_bonus(self):
        """Moderate slope + very smooth terrain should give 20% bonus."""
        bonus = sledding_synergy_bonus(
            slope=10.0,  # In 6-12° optimal range
            snow_depth=5.0,  # Not optimal
            coverage_months=2.0,
            roughness=1.0,  # Very smooth terrain (<1.5m)
        )
        # Should get moderate slope + smooth terrain bonus
        assert bonus >= 1.15  # At least some bonus

    def test_smooth_terrain_perfect_slope_bonus(self):
        """Smooth terrain + perfect slope should give 10% bonus."""
        bonus = sledding_synergy_bonus(
            slope=9.0,  # In 6-12° optimal range
            snow_depth=5.0,  # Not optimal
            coverage_months=1.0,
            roughness=2.5,  # Smooth terrain (<3m)
        )
        # Should get smooth terrain + perfect slope bonus
        assert bonus >= 1.05  # At least some bonus

    def test_no_synergy(self):
        """No special combinations should give no bonus."""
        bonus = sledding_synergy_bonus(
            slope=15.0,  # Not in optimal range
            snow_depth=5.0,  # Not in optimal range
            coverage_months=1.0,  # Not 3+
            roughness=5.0,  # Not smooth (<3m)
        )
        assert bonus == 1.0

    def test_multiple_bonuses_stack(self):
        """Multiple synergies should stack."""
        # Perfect combo (30%) + other bonuses should stack
        bonus = sledding_synergy_bonus(
            slope=9.0,  # Perfect
            snow_depth=8.0,  # Optimal (new trapezoid: 2-10")
            coverage_months=5.0,  # 4+ months
            roughness=1.0,  # Very smooth
        )
        # Should have multiple bonuses
        assert bonus > 1.30  # More than just perfect combo


class TestComputeSleddingScore:
    """Test overall sledding score computation."""

    def test_deal_breaker_gives_zero(self):
        """Deal breakers should result in zero score."""
        score = compute_sledding_score(
            snow_depth=12.0,
            slope=45.0,  # Too steep (>40°) - deal breaker
            coverage_months=3.0,
            roughness=2.0,
        )
        assert score == 0.0

    def test_perfect_conditions(self):
        """Perfect conditions should give high score."""
        score = compute_sledding_score(
            snow_depth=8.0,  # Optimal (new trapezoid: 2-10")
            slope=9.0,  # Optimal
            coverage_months=4.0,  # Good
            roughness=2.0,  # Smooth
        )
        # Should be high score (base scores all 1.0, coverage ~0.86, synergy bonus)
        assert score > 0.9

    def test_marginal_conditions(self):
        """Marginal conditions should give moderate score."""
        score = compute_sledding_score(
            snow_depth=6.0,  # Between min and optimal
            slope=5.0,  # Between min and optimal
            coverage_months=1.0,  # Just above deal breaker
            roughness=4.0,  # Moderate roughness
        )
        # Should be moderate score
        assert 0.1 < score < 0.5

    def test_poor_snow_depth(self):
        """Poor snow depth should reduce score significantly."""
        score = compute_sledding_score(
            snow_depth=22.0,  # Near max (ramping down)
            slope=9.0,  # Perfect
            coverage_months=3.0,
            roughness=2.0,
        )
        # Should be reduced due to poor snow depth
        assert score < 0.6

    def test_multiplicative_nature(self):
        """Score should be multiplicative - one poor factor affects total."""
        # All perfect except slope
        score1 = compute_sledding_score(
            snow_depth=12.0,  # Perfect
            slope=18.0,  # Poor (between optimal and max acceptable)
            coverage_months=4.0,  # Good
            roughness=2.0,  # Good
        )

        # All perfect
        score2 = compute_sledding_score(
            snow_depth=12.0,  # Perfect
            slope=9.0,  # Perfect
            coverage_months=4.0,  # Good
            roughness=2.0,  # Good
        )

        # score1 should be significantly lower due to multiplicative effect
        assert score1 < score2 * 0.7

    def test_array_input(self):
        """Should work with numpy arrays."""
        snow_depths = np.array([8.0, 40.0, 6.0])  # Updated to 8.0" (new optimal range)
        slopes = np.array([9.0, 9.0, 5.0])
        coverage = np.array([4.0, 4.0, 1.0])
        roughness_vals = np.array([2.0, 2.0, 4.0])

        scores = compute_sledding_score(
            snow_depth=snow_depths,
            slope=slopes,
            coverage_months=coverage,
            roughness=roughness_vals,
        )

        assert len(scores) == 3
        assert scores[0] > 0.9  # Perfect conditions (8" snow, 9° slope)
        assert 0.1 < scores[2] < 0.5  # Marginal conditions
